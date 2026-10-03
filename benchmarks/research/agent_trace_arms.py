# SPDX-License-Identifier: Apache-2.0

"""Replay the Mooncake tool/agent trace against matched KV-cache arms.

Each arm starts its own vLLM (and DaseR) services with the same GPU KV budget,
block size, and scheduler limits, runs ``benchmarks.mooncake.benchmark`` once,
and stores the summary, DaseR runtime config, and vLLM/DaseR metric snapshots.
After an arm's services stop, its store and scratch directory are deleted
immediately so only the small artifacts under ``--artifact`` remain.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import httpx

from benchmarks.mooncake.benchmark import ROUTES
from benchmarks.utils.servers import (
    LMCACHE_HTTP_PORT,
    ServerManager,
    check_runtime_environment,
    stop_from_pid_file,
)
from benchmarks.utils.sizing import parse_size_bytes
from daser.connector.ipc_client import IPCClientSync

ARMS = (
    "native-apc",
    "noop-connector",
    "vllm-offload",
    "daser-raw",
    "daser-compressed",
    # Compressed bytes on SSD and in L1, but L1 capacity and prefetch leases
    # charged in raw bytes: a codec bolted onto a raw-sized cache.
    "daser-compressed-raw-l1",
    # Concurrent requests for a prefix that is already loading wait for that
    # load instead of loading their own copy.
    "daser-raw-coalesce",
    "daser-compressed-coalesce",
    # L1 split residency (--daser-l1-split-fraction) on raw / compressed stores.
    "daser-raw-split",
    "daser-compressed-split",
    # Fully leased hits restored layer by layer inside the forward pass, and
    # their PIECEWISE-graph baselines (need prefetch for leases), on packed
    # and raw stores.
    "daser-compressed-layerwise",
    "daser-compressed-piecewise",
    "daser-raw-layerwise",
    "daser-raw-piecewise",
    # LMCache MP server (L1 + O_DIRECT fs L2) through its vLLM connector.
    "lmcache",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
GATEWAY_MODES = ("none", "pass", "split")


def _pid_running(pid: int) -> bool:
    """Return whether ``pid`` is a live, non-zombie process.

    Zombies still answer ``kill(pid, 0)`` until their parent reaps them, but
    they hold no files or GPU memory, so they count as exited here.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        # Field 3 of /proc/<pid>/stat is the state; the comm field may contain
        # spaces, so split after its closing parenthesis.
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _wait_pids_exit(pid_file: Path, timeout_s: float = 60.0) -> bool:
    """Wait until every PID recorded in ``pid_file`` has exited.

    Args:
        pid_file: ServerManager PID JSON path.
        timeout_s: Maximum wait in seconds.

    Returns:
        True when all processes are gone (or the file is missing).
    """
    if not pid_file.is_file():
        return True
    pids = [int(item["pid"]) for item in json.loads(pid_file.read_text())]
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        alive = [pid for pid in pids if _pid_running(pid)]
        if not alive:
            return True
        time.sleep(0.5)
    return False


def _json_default(value: Any) -> Any:
    """Serialize bytes fields found in DaseR runtime config."""
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class MatchedServer(ServerManager):
    """ServerManager with prefix caching on and a fixed GPU KV byte budget."""

    def __init__(self, *args: Any, gpu_kv_bytes: int, **kwargs: Any) -> None:
        """Store the GPU KV budget shared by every arm.

        Args:
            *args: Positional ``ServerManager`` arguments.
            gpu_kv_bytes: Value passed to ``--kv-cache-memory-bytes``.
            **kwargs: Keyword ``ServerManager`` arguments.
        """
        super().__init__(*args, **kwargs)
        self.gpu_kv_bytes = gpu_kv_bytes

    def vllm_command(
        self,
        kv_transfer_config: dict[str, Any] | None,
        port: int | None = None,
    ) -> list[str]:
        """Return the vLLM command with APC enabled and a fixed KV budget.

        Args:
            kv_transfer_config: Optional connector configuration.
            port: HTTP port; defaults to the primary ``vllm_port``.

        Returns:
            vLLM argv.
        """
        command = super().vllm_command(kv_transfer_config, port=port)
        command[command.index("--no-enable-prefix-caching")] = "--enable-prefix-caching"
        command.extend(
            [
                "--kv-cache-memory-bytes",
                str(self.gpu_kv_bytes),
                "--enable-prompt-tokens-details",
            ]
        )
        return command


class CPUOffloadServer(MatchedServer):
    """Start vLLM with its built-in CPU KV offload connector."""

    def offload_config(self) -> dict[str, Any]:
        """Return the SimpleCPUOffloadConnector config sized to L1.

        Returns:
            kv-transfer-config mapping.
        """
        return {
            "kv_connector": "SimpleCPUOffloadConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"cpu_bytes_to_use": self.l1_size_bytes},
        }

    async def start_vllm_only(self) -> None:
        """Start vLLM with CPU KV offload enabled."""
        await self._start_vllm("vllm_offload.log", self.offload_config())


class NoopConnectorServer(MatchedServer):
    """Start vLLM with a KV connector that never loads or saves."""

    def offload_config(self) -> dict[str, Any]:
        """Return the no-op connector config.

        Returns:
            kv-transfer-config mapping.
        """
        return {
            "kv_connector": "NoopConnector",
            "kv_connector_module_path": "benchmarks.research.noop_connector",
            "kv_role": "kv_both",
        }

    async def start_vllm_only(self) -> None:
        """Start vLLM with the no-op connector."""
        await self._start_vllm("vllm_noop.log", self.offload_config())


# Arms whose vLLM runs a non-DaseR connector given by ``offload_config()``.
_SERVER_TYPES: dict[str, type[MatchedServer]] = {
    "vllm-offload": CPUOffloadServer,
    "noop-connector": NoopConnectorServer,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse runner arguments.

    Args:
        argv: Optional argv override.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--gpu-id", default="3")
    parser.add_argument(
        "--extra-gpu-ids",
        nargs="*",
        default=[],
        help=(
            "GPUs for additional vLLM instances sharing one DaseR server "
            "(DaseR arms only); requests are routed with --route."
        ),
    )
    parser.add_argument("--route", choices=ROUTES, default="sticky")
    parser.add_argument("--gpu-kv-size", type=parse_size_bytes, default="24gib")
    parser.add_argument(
        "--gpu-util",
        type=float,
        default=None,
        help=(
            "vLLM --gpu-memory-utilization; lower it when several instances "
            "share one GPU (vLLM's default is used when omitted)"
        ),
    )
    parser.add_argument("--l1-size", type=parse_size_bytes, default="64gib")
    parser.add_argument("--l2-size", type=parse_size_bytes, default="256gib")
    parser.add_argument(
        "--skip-l2",
        action="store_true",
        help="Run DaseR arms memory-only: the store lives in L1 (--l1-size).",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-requests", type=int, default=3000)
    parser.add_argument("--start-request", type=int, default=0)
    parser.add_argument("--max-inflight", type=int, default=16)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument(
        "--prefix-dependencies",
        action="store_true",
        help="Closed-loop replay: hold requests until their prefix producer finishes.",
    )
    parser.add_argument(
        "--strict-order",
        action="store_true",
        help="Closed-loop replay in exact trace order (one request after another).",
    )
    parser.add_argument(
        "--ordered-admission",
        action="store_true",
        help="Admit requests in trace order while up to --max-inflight overlap.",
    )
    parser.add_argument(
        "--dependency-gap-s",
        type=float,
        default=0.0,
        help="Idle seconds after each dependency completes (tool / think time).",
    )
    parser.add_argument(
        "--corpus-tokens",
        type=Path,
        default=None,
        help="Real-text .npy token corpus passed to the replay (see longdoc_trace).",
    )
    parser.add_argument(
        "--daser-prefetch-max-requests",
        type=int,
        default=0,
        help="DaseR scheduler prefetch workers (0 disables; DaseR arms only).",
    )
    parser.add_argument(
        "--daser-prefetch-lease-budget",
        type=parse_size_bytes,
        default="0",
        help="DaseR --prefetch-lease-budget (0 = no cap; DaseR arms only).",
    )
    parser.add_argument(
        "--daser-l2-read-stripe",
        type=parse_size_bytes,
        default="0",
        help="DaseR --l2-read-stripe (0 = serial reads; DaseR arms only).",
    )
    parser.add_argument(
        "--daser-l1-split-fraction",
        type=float,
        default=0.6,
        help="DaseR --l1-split-fraction for the *-split arms only.",
    )
    parser.add_argument(
        "--daser-l1-lru",
        action="store_true",
        help="Run DaseR L1 as plain LRU (--no-bip-enabled) instead of BIP.",
    )
    parser.add_argument(
        "--lmcache-eviction-watermark",
        type=float,
        default=None,
        help="LMCache L1 eviction trigger watermark (lmcache arm only).",
    )
    parser.add_argument(
        "--lmcache-l2-native-workers",
        type=int,
        default=0,
        help="Use LMCache's C++ fs_native L2 adapter with N IO threads (0 = fs).",
    )
    parser.add_argument(
        "--lmcache-l2-bounded",
        action="store_true",
        help="Cap LMCache fs_native L2 at --l2-size with LRU eviction.",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8001,
        help="vLLM API port; give concurrent runs distinct ports.",
    )
    parser.add_argument(
        "--daser-port",
        type=int,
        default=2026,
        help="DaseR server port; give concurrent runs distinct ports.",
    )
    parser.add_argument(
        "--abort-after-failures",
        type=int,
        default=5,
        help="Abort an arm's replay after this many failed requests (0 disables).",
    )
    parser.add_argument(
        "--gateway",
        choices=GATEWAY_MODES,
        default="none",
        help=(
            "Replay DaseR arms through daser.gateway: 'pass' forwards unchanged, "
            "'split' plans bidirectional prefill."
        ),
    )
    parser.add_argument("--gateway-compute-tokens-per-s", type=float, default=30000.0)
    parser.add_argument("--gateway-load-bytes-per-s", type=float, default=15e9)
    parser.add_argument("--gateway-min-compute-tokens", type=int, default=4096)
    return parser.parse_args(argv)


def _manager(arm: str, args: argparse.Namespace) -> MatchedServer:
    """Build the service manager for one arm."""
    manager_type = _SERVER_TYPES.get(arm, MatchedServer)
    return manager_type(
        run_id=f"agent-{arm}",
        backend=(
            "daser"
            if arm.startswith("daser")
            else ("lmcache" if arm == "lmcache" else "vllm")
        ),
        model=args.model,
        store_dir=args.scratch / arm,
        gpu_id=args.gpu_id,
        gpu_util=args.gpu_util,
        max_num_seqs=args.max_num_seqs,
        l1_size_bytes=args.l1_size,
        l2_size_bytes=args.l2_size,
        block_size=args.block_size,
        reuse_mode="prefix",
        transfer_mode="iouring",
        startup_timeout=900.0,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        storage_format=(
            "compressed-online" if arm.startswith("daser-compressed") else "raw"
        ),
        daser_l1_accounting=("raw" if arm == "daser-compressed-raw-l1" else "stored"),
        daser_coalesce_loads=arm.endswith("-coalesce"),
        daser_layerwise_load=arm.endswith("-layerwise"),
        daser_piecewise_cudagraph=arm.endswith("-piecewise"),
        skip_l2=args.skip_l2 and arm.startswith("daser"),
        daser_prefetch_max_requests=(
            args.daser_prefetch_max_requests if arm.startswith("daser") else 0
        ),
        daser_prefetch_lease_budget_bytes=(
            args.daser_prefetch_lease_budget if arm.startswith("daser") else 0
        ),
        daser_l2_read_stripe_bytes=(
            args.daser_l2_read_stripe if arm.startswith("daser") else 0
        ),
        daser_l1_split_fraction=(
            args.daser_l1_split_fraction if arm.endswith("-split") else 0.0
        ),
        daser_l1_lru=args.daser_l1_lru,
        gpu_kv_bytes=args.gpu_kv_size,
        vllm_port=args.vllm_port,
        daser_port=args.daser_port,
        extra_vllm_gpu_ids=tuple(args.extra_gpu_ids) if arm.startswith("daser") else (),
        lmcache_l2_odirect=True,
        lmcache_l2_native_workers=args.lmcache_l2_native_workers,
        lmcache_l2_bounded=args.lmcache_l2_bounded,
        **(
            {"lmcache_eviction_watermark": args.lmcache_eviction_watermark}
            if args.lmcache_eviction_watermark is not None
            else {}
        ),
    )


async def _text(url: str) -> str:
    """Fetch a text endpoint, returning an empty string on failure."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.HTTPError:
        return ""


async def _daser_metrics(arm: str, manager: MatchedServer) -> str:
    """Return DaseR Prometheus text for DaseR arms, else an empty string."""
    if not arm.startswith("daser"):
        return ""
    return await _text(f"{manager.daser_url}/metrics")


async def _vllm_metrics(manager: MatchedServer) -> str:
    """Return the Prometheus text of every vLLM instance, one block each."""
    blocks = []
    for url in manager.vllm_urls:
        blocks.append(f"# instance {url}\n" + await _text(f"{url}/metrics"))
    return "\n".join(blocks)


async def _warmup(vllm_url: str, model_name: str) -> float:
    """Send one untimed request outside the trace token space.

    Returns:
        Warmup latency in seconds.
    """
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            f"{vllm_url}/v1/completions",
            json={
                "model": model_name,
                "prompt": [7] * 2048,
                "max_tokens": 4,
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
    return time.perf_counter() - started


async def _start_gateway(
    args: argparse.Namespace, manager: MatchedServer, out_dir: Path
) -> tuple[subprocess.Popen[bytes], str]:
    """Start ``daser.gateway`` in front of the single vLLM instance.

    Returns:
        The gateway process and its base URL once it answers ``/metrics``.
    """
    if len(manager.vllm_urls) != 1:
        raise ValueError("--gateway supports one vLLM instance")
    port = args.vllm_port + 100
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "daser.gateway",
            "--vllm-url",
            manager.vllm_urls[0],
            "--socket-path",
            str(manager.socket_path),
            "--port",
            str(port),
            "--compute-tokens-per-s",
            str(args.gateway_compute_tokens_per_s),
            "--load-bytes-per-s",
            str(args.gateway_load_bytes_per_s),
            "--min-compute-tokens",
            str(args.gateway_min_compute_tokens),
            *(["--split"] if args.gateway == "split" else []),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        stdout=(out_dir / "gateway.log").open("wb"),
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"gateway exited with {proc.returncode}")
        if await _text(f"{url}/metrics"):
            return proc, url
        await asyncio.sleep(0.5)
    proc.terminate()
    raise RuntimeError("gateway did not become ready")


async def run_arm(arm: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run one arm end to end and persist its artifacts.

    Args:
        arm: Arm name.
        args: Parsed runner arguments.

    Returns:
        Result record written to ``<artifact>/<arm>.json``.
    """
    manager = _manager(arm, args)
    out_dir = args.artifact / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model).name
    gateway: subprocess.Popen[bytes] | None = None
    try:
        manifest = await manager.start()
        runtime_config = None
        if arm.startswith("daser"):
            ipc = IPCClientSync(str(manager.socket_path))
            try:
                runtime_config = ipc.get_runtime_config()
            finally:
                ipc.close()
        warmup_s = 0.0
        for url in manager.vllm_urls:
            warmup_s += await _warmup(url, model_name)
        daser_before = await _daser_metrics(arm, manager)
        vllm_before = await _vllm_metrics(manager)
        server_urls = list(manager.vllm_urls)
        if args.gateway != "none" and arm.startswith("daser"):
            gateway, gateway_url = await _start_gateway(args, manager, out_dir)
            server_urls = [gateway_url]
        replay = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.mooncake.benchmark",
                "--trace",
                args.trace,
                "--tokenizer",
                args.model,
                "--trust-remote-code",
                "--server-url",
                *server_urls,
                "--route",
                args.route,
                "--served-model-name",
                model_name,
                "--output-dir",
                str(out_dir),
                "--max-model-len",
                str(args.max_model_len),
                "--start-request",
                str(args.start_request),
                "--max-requests",
                str(args.max_requests),
                "--max-inflight",
                str(args.max_inflight),
                "--time-scale",
                str(args.time_scale),
                "--timeout",
                str(args.timeout),
                "--abort-after-failures",
                str(args.abort_after_failures),
                *(["--prefix-dependencies"] if args.prefix_dependencies else []),
                *(["--strict-order"] if args.strict_order else []),
                *(["--ordered-admission"] if args.ordered_admission else []),
                "--dependency-gap-s",
                str(args.dependency_gap_s),
                *(
                    ["--corpus-tokens", str(args.corpus_tokens)]
                    if args.corpus_tokens is not None
                    else []
                ),
            ],
            cwd=REPO_ROOT,
            stdout=(out_dir / "replay.log").open("wb"),
            stderr=subprocess.STDOUT,
            check=False,
        )
        daser_after = await _daser_metrics(arm, manager)
        vllm_after = await _vllm_metrics(manager)
        if arm == "lmcache":
            # LMCache L1/L2 occupancy and tier state after the replay.
            (out_dir / "lmcache_status_after.json").write_text(
                await _text(f"http://127.0.0.1:{LMCACHE_HTTP_PORT}/status")
            )
        (out_dir / "daser_metrics_before.prom").write_text(daser_before)
        (out_dir / "daser_metrics_after.prom").write_text(daser_after)
        (out_dir / "vllm_metrics_before.prom").write_text(vllm_before)
        (out_dir / "vllm_metrics_after.prom").write_text(vllm_after)
        summary_path = out_dir / "summary.json"
        result = {
            "arm": arm,
            "replay_returncode": replay.returncode,
            "manifest": asdict(manifest),
            "vllm_command": manager.vllm_command(
                manager.daser_kv_transfer_config()
                if arm.startswith("daser")
                else (
                    manager.lmcache_kv_transfer_config()
                    if arm == "lmcache"
                    else (manager.offload_config() if arm in _SERVER_TYPES else None)
                )
            ),
            "runtime_config": runtime_config,
            "gateway": args.gateway if gateway is not None else "none",
            "warmup_seconds": warmup_s,
            "summary": json.loads(summary_path.read_text())
            if summary_path.exists()
            else None,
        }
        (args.artifact / f"{arm}.json").write_text(
            json.dumps(result, indent=2, default=_json_default)
        )
        return result
    finally:
        if gateway is not None:
            gateway.terminate()
            gateway.wait(timeout=30)
        # Keep the service logs with the artifact; stores are scratch.
        for log in (manager.log_dir).glob("*.log"):
            (out_dir / log.name).write_bytes(log.read_bytes())
        stop_from_pid_file(manager.pid_file)
        if _wait_pids_exit(manager.pid_file):
            shutil.rmtree(manager.store_dir, ignore_errors=True)
        else:
            print(f"{arm}: services still alive; kept {manager.store_dir}", flush=True)


async def main_async(args: argparse.Namespace) -> None:
    """Run the selected arms sequentially on one GPU."""
    args.artifact.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        result = await run_arm(arm, args)
        summary = result["summary"] or {}
        ttft = summary.get("ttft_ms", {})
        print(
            f"{arm}: rc={result['replay_returncode']} "
            f"done={summary.get('completed_requests')} "
            f"failed={summary.get('failed_requests')} "
            f"aborted={summary.get('aborted')} "
            f"wall={summary.get('wall_seconds', 0):.0f}s "
            f"ttft_p50={ttft.get('p50', 0):.0f} p95={ttft.get('p95', 0):.0f} "
            f"prefix={summary.get('prefix_cache')}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    check_runtime_environment()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
