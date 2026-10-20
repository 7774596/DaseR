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
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

import httpx

from benchmarks.mooncake.benchmark import ROUTES
from benchmarks.research.counter_trace import CounterTrace
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
    # Fully leased hits restored layer by layer inside the forward pass (need
    # prefetch for leases), on packed and raw stores.
    "daser-compressed-layerwise",
    "daser-raw-layerwise",
    # FragPack attention (--attention-backend CUSTOM, vLLM extra args) holds KV
    # compressed in HBM; DaseR stores its records as block blobs as-is.
    "daser-fragpack",
    # LMCache MP server (L1 + O_DIRECT fs L2) through its vLLM connector,
    # raw and with the DaseR lossless chunk codec (codec-patched checkout).
    "lmcache",
    "lmcache-codec",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


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


def store_disk_usage(store_dir: Path) -> dict[str, int]:
    """Return the allocated disk bytes under each top-level entry of a store.

    Allocated blocks (not apparent size) show how much of a preallocated or
    sparse L2 file was actually written, so arms can compare the L2 capacity
    they really used.

    Args:
        store_dir: Arm scratch directory holding the service stores.

    Returns:
        dict[str, int]: Entry name to allocated bytes, plus ``"total"``.
    """
    usage: dict[str, int] = {}
    if store_dir.is_dir():
        for entry in sorted(store_dir.iterdir()):
            paths = [entry] if entry.is_file() else entry.rglob("*")
            usage[entry.name] = sum(
                p.lstat().st_blocks * 512 for p in paths if p.is_file()
            )
    usage["total"] = sum(usage.values())
    return usage


class ArmTerminated(Exception):
    """Raised when SIGTERM/SIGINT arrives while no replay is running."""


class StopSignals:
    """Turn SIGTERM/SIGINT into an orderly stop of the current arm.

    During a replay the signal is forwarded to the replay subprocess, which
    writes its summary and exits; the arm then stops its services and deletes
    its store as usual, and no further arm starts. Outside a replay the first
    signal raises :class:`ArmTerminated` so ``run_arm``'s ``finally`` still
    stops the services; later signals are ignored so teardown can finish.

    Thread-safety:
        Install and use from the main thread only; Python runs signal handlers
        there.
    """

    def __init__(self) -> None:
        """Create a handler with no replay attached."""
        self.requested = False
        self.child: subprocess.Popen[bytes] | None = None

    def install(self) -> None:
        """Route SIGTERM and SIGINT to :meth:`handle`."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.handle)

    def handle(self, signum: int, frame: Any) -> None:
        """Forward a stop signal to the replay, or unwind into teardown.

        Args:
            signum: Received signal number.
            frame: Interrupted frame (unused).

        Raises:
            ArmTerminated: On the first signal while no replay is running.
        """
        del frame
        first = not self.requested
        self.requested = True
        child = self.child
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            return
        if first:
            raise ArmTerminated(f"signal {signum}")


def _json_default(value: Any) -> Any:
    """Serialize bytes fields found in DaseR runtime config."""
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_object(text: str) -> dict[str, Any]:
    """Parse a CLI JSON object argument.

    Args:
        text: JSON text.

    Returns:
        The parsed object.

    Raises:
        argparse.ArgumentTypeError: if ``text`` is not a JSON object.
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return value


def _speculative_args(config: dict[str, Any] | None) -> tuple[str, ...]:
    """Return ``vllm serve`` arguments enabling ``config`` speculation.

    Args:
        config: ``--speculative-config`` object, or None for no speculation.

    Returns:
        ``("--speculative-config", <compact JSON>)`` or an empty tuple. The JSON
        is passed as one argv item, so it needs no shell quoting.
    """
    if config is None:
        return ()
    return ("--speculative-config", json.dumps(config, separators=(",", ":")))


def _drafter_kv_layers(config: dict[str, Any] | None) -> int:
    """Return the KV layers an EAGLE drafter adds to the worker's KV cache.

    Args:
        config: ``--speculative-config`` object, or None for no speculation.

    Returns:
        The drafter's ``num_hidden_layers`` for ``eagle``/``eagle3`` with a
        draft model, else 0 (n-gram and suffix drafting keep no KV).
    """
    if not config or config.get("method") not in ("eagle", "eagle3"):
        return 0
    draft = json.loads((Path(config["model"]) / "config.json").read_text())
    return int(draft["num_hidden_layers"])


def _storage_format(arm: str) -> str:
    """Return the DaseR storage format of one arm."""
    if arm == "daser-fragpack":
        return "fragpack"
    return "compressed-online" if arm.startswith("daser-compressed") else "raw"


def _probe_command(
    args: argparse.Namespace, server_url: str, model_name: str, out_dir: Path
) -> list[str]:
    """Return the ``cache_state_probe`` command replacing an arm's replay.

    Args:
        args: Parsed runner arguments (``--cache-state-probe`` set).
        server_url: vLLM URL to probe.
        model_name: Served model name.
        out_dir: Arm artifact directory receiving ``summary.json``.

    Returns:
        argv for ``subprocess.run``.

    Raises:
        ValueError: When the trace has no ``--session-tokens`` stream.
    """
    if args.session_tokens is None:
        raise ValueError("--cache-state-probe needs --session-tokens")
    return [
        sys.executable,
        "-m",
        "benchmarks.research.cache_state_probe",
        "--trace",
        str(args.trace),
        "--session-tokens",
        str(args.session_tokens),
        "--server-url",
        server_url,
        "--served-model-name",
        model_name,
        "--output-dir",
        str(out_dir),
        "--requests",
        str(args.cache_state_probe),
        "--max-tokens",
        str(args.cache_state_probe_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--timeout",
        str(args.timeout),
    ]


def _agentx_command(
    args: argparse.Namespace, server_url: str, model_name: str, out_dir: Path
) -> list[str]:
    """Return the official AgentX harness command replacing an arm's replay.

    Args:
        args: Parsed runner arguments (``--agentx-args`` set); the harness
            flags (dataset, tokenizer, concurrency, duration) come from there.
        server_url: vLLM URL the harness drives.
        model_name: Served model name.
        out_dir: Arm artifact directory; the harness writes under ``agentx/``.

    Returns:
        argv for ``subprocess.Popen``.
    """
    return [
        args.agentx_bin,
        "profile",
        "--url",
        server_url,
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        model_name,
        "--output-artifact-dir",
        str(out_dir / "agentx"),
        "--ui-type",
        "simple",
        *shlex.split(args.agentx_args),
    ]


def _restore_probe_command(
    args: argparse.Namespace,
    server_url: str,
    daser_url: str | None,
    model_name: str,
    out_dir: Path,
) -> list[str]:
    """Return the ``prefix_restore_probe`` command replacing an arm's replay.

    Args:
        args: Parsed runner arguments (``--prefix-restore-probe`` set).
        server_url: vLLM URL to probe.
        daser_url: DaseR metrics URL for DaseR arms, else ``None``.
        model_name: Served model name.
        out_dir: Arm artifact directory receiving ``summary.json``.

    Returns:
        argv for ``subprocess.run``.

    Raises:
        ValueError: When the trace has no ``--session-tokens`` stream.
    """
    if args.session_tokens is None:
        raise ValueError("--prefix-restore-probe needs --session-tokens")
    return [
        sys.executable,
        "-m",
        "benchmarks.research.prefix_restore_probe",
        "--session-tokens",
        str(args.session_tokens),
        "--server-url",
        server_url,
        *(["--daser-url", daser_url] if daser_url else []),
        "--served-model-name",
        model_name,
        "--output-dir",
        str(out_dir),
        "--max-model-len",
        str(args.max_model_len),
        "--block-tokens",
        str(args.block_size),
        "--timeout",
        str(args.timeout),
        *shlex.split(args.prefix_restore_probe),
    ]


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
    parser.add_argument(
        "--trace",
        default=None,
        help="Mooncake-format trace to replay (unless --agentx-args is set).",
    )
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
    parser.add_argument("--route-local-blocks", type=int, default=2048)
    parser.add_argument("--route-load-tokens-per-s", type=float, default=360e3)
    parser.add_argument("--route-saturation-s", type=float, default=10.0)
    parser.add_argument("--rebind-after-s", type=float, default=None)
    parser.add_argument("--prewarm-stripped", action="store_true")
    parser.add_argument("--prewarm-chunk-tokens", type=int, default=0)
    parser.add_argument("--prewarm-max-waiting", type=int, default=None)
    parser.add_argument("--session-population", type=int, default=0)
    parser.add_argument("--session-ramp-s", type=float, default=0.0)
    parser.add_argument("--session-rate", type=float, default=0.0)
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
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM and DaseR tensor-parallel ranks; --gpu-id lists one GPU per "
        'rank, comma-separated (e.g. "3,4").',
    )
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="vLLM max_num_seqs; the engine default when omitted.",
    )
    parser.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="vLLM async scheduling (CPU scheduling overlaps GPU steps) for "
        "every arm; on as in vLLM, --no-async-scheduling is an ablation.",
    )
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
        "--chain-outputs",
        action="store_true",
        help="Research: splice each turn's real generated tokens into the next.",
    )
    parser.add_argument(
        "--ordered-admission",
        action="store_true",
        help="Admit requests in trace order while up to --max-inflight overlap.",
    )
    parser.add_argument(
        "--admission",
        choices=("fifo", "random"),
        default="fifo",
        help="Freed --max-inflight slot goes to the oldest or a random waiter.",
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
        "--session-tokens",
        type=Path,
        default=None,
        help="Real session token stream passed to the replay (see lm_trace_export).",
    )
    parser.add_argument(
        "--daser-queue-hint",
        action="store_true",
        help="Send client-queue hints to the DaseR server (DaseR arms only).",
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
    parser.add_argument(
        "--lmcache-eager-prefetch",
        action="store_true",
        help="Submit LMCache lookups when requests enter vLLM's waiting queue.",
    )
    parser.add_argument(
        "--lmcache-pythonpath",
        nargs="+",
        type=Path,
        default=None,
        help="LMCache checkout plus extra dependency dirs to put first on "
        "PYTHONPATH (lmcache arms; default: the sibling LMCache checkout).",
    )
    parser.add_argument(
        "--lmcache-codec-capacity",
        type=float,
        default=None,
        help="Fixed stored capacity fraction of a raw chunk (lmcache-codec arm "
        "only); unset stores variable-length records.",
    )
    parser.add_argument(
        "--codec-codebooks",
        type=Path,
        default=None,
        help="Calibrated codebook directory shared by the daser-compressed* and "
        "lmcache-codec arms, so both sides run the same codec.",
    )
    parser.add_argument(
        "--vllm-extra-args",
        default="",
        help="Extra vllm serve arguments for every arm (shell-split).",
    )
    parser.add_argument(
        "--speculative-config",
        type=_json_object,
        default=None,
        help="vllm serve --speculative-config JSON for every arm, so arms "
        "compare under the same speculative-decoding baseline.",
    )
    parser.add_argument(
        "--cache-state-probe",
        type=int,
        default=0,
        help="Instead of replaying the trace, send this many requests (one "
        "per session) cold, GPU-cached and externally cached with "
        "benchmarks.research.cache_state_probe; needs --session-tokens and "
        "VLLM_SERVER_DEV_MODE=1 in the environment.",
    )
    parser.add_argument(
        "--cache-state-probe-tokens",
        type=int,
        default=256,
        help="Decode length of every --cache-state-probe request.",
    )
    parser.add_argument(
        "--prefix-restore-probe",
        default="",
        help="Instead of replaying the trace, run "
        "benchmarks.research.prefix_restore_probe with these extra arguments "
        '(one quoted string, e.g. "--total-tokens 32768 --tiers ssd"); '
        "needs --session-tokens and VLLM_SERVER_DEV_MODE=1.",
    )
    parser.add_argument(
        "--agentx-args",
        default="",
        help="Instead of replaying --trace, run the official AgentX harness "
        "(agentx profile) against each arm with these extra arguments (one "
        'quoted string, e.g. "--concurrency 8 --public-dataset ...").',
    )
    parser.add_argument(
        "--agentx-bin",
        default="agentx",
        help="AgentX harness executable (its own environment).",
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
        "--counter-trace-interval-s",
        type=float,
        default=0.0,
        help="Poll DaseR tier counters this often during the replay into "
        "counter_trace.jsonl for per-request tier attribution (0 = off).",
    )
    args = parser.parse_args(argv)
    if args.trace is None and not args.agentx_args:
        parser.error("--trace is required unless --agentx-args is set")
    return args


def _manager(arm: str, args: argparse.Namespace) -> MatchedServer:
    """Build the service manager for one arm."""
    manager_type = _SERVER_TYPES.get(arm, MatchedServer)
    return manager_type(
        run_id=f"agent-{arm}",
        backend=(
            "daser"
            if arm.startswith("daser")
            else ("lmcache" if arm.startswith("lmcache") else "vllm")
        ),
        model=args.model,
        store_dir=args.scratch / arm,
        gpu_id=args.gpu_id,
        gpu_util=args.gpu_util,
        max_num_seqs=args.max_num_seqs,
        async_scheduling=args.async_scheduling,
        l1_size_bytes=args.l1_size,
        l2_size_bytes=args.l2_size,
        block_size=args.block_size,
        reuse_mode="prefix",
        transfer_mode="iouring",
        startup_timeout=900.0,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        storage_format=_storage_format(arm),
        daser_l1_accounting=("raw" if arm == "daser-compressed-raw-l1" else "stored"),
        daser_codebooks=(
            args.codec_codebooks if arm.startswith("daser-compressed") else None
        ),
        daser_coalesce_loads=arm.endswith("-coalesce"),
        daser_layerwise_load=arm.endswith("-layerwise"),
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
        daser_extra_kv_layers=_drafter_kv_layers(args.speculative_config),
        gpu_kv_bytes=args.gpu_kv_size,
        vllm_port=args.vllm_port,
        daser_port=args.daser_port,
        extra_vllm_gpu_ids=tuple(args.extra_gpu_ids) if arm.startswith("daser") else (),
        lmcache_l2_odirect=True,
        lmcache_l2_native_workers=args.lmcache_l2_native_workers,
        lmcache_l2_bounded=args.lmcache_l2_bounded,
        lmcache_eager_prefetch=args.lmcache_eager_prefetch,
        lmcache_daser_codec=arm == "lmcache-codec",
        lmcache_codec_capacity=(
            args.lmcache_codec_capacity if arm == "lmcache-codec" else None
        ),
        lmcache_codec_codebooks=(
            args.codec_codebooks if arm == "lmcache-codec" else None
        ),
        extra_vllm_args=tuple(shlex.split(args.vllm_extra_args))
        + _speculative_args(args.speculative_config),
        **(
            {"lmcache_pythonpath": tuple(args.lmcache_pythonpath)}
            if args.lmcache_pythonpath
            else {}
        ),
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


async def run_arm(
    arm: str, args: argparse.Namespace, stop: StopSignals
) -> dict[str, Any]:
    """Run one arm end to end and persist its artifacts.

    Services are stopped and the scratch store deleted however the arm ends,
    including a stop signal (see :class:`StopSignals`).

    Args:
        arm: Arm name.
        args: Parsed runner arguments.
        stop: Installed stop-signal handler; the replay is attached to it.

    Returns:
        Result record written to ``<artifact>/<arm>.json``.
    """
    manager = _manager(arm, args)
    out_dir = args.artifact / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model).name
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
        if args.agentx_args:
            replay_command = _agentx_command(args, server_urls[0], model_name, out_dir)
        elif args.prefix_restore_probe:
            replay_command = _restore_probe_command(
                args,
                server_urls[0],
                manager.daser_url if arm.startswith("daser") else None,
                model_name,
                out_dir,
            )
        else:
            replay_command = (
                _probe_command(args, server_urls[0], model_name, out_dir)
                if args.cache_state_probe
                else [
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
                    "--route-local-blocks",
                    str(args.route_local_blocks),
                    "--route-load-tokens-per-s",
                    str(args.route_load_tokens_per_s),
                    "--route-saturation-s",
                    str(args.route_saturation_s),
                    *(
                        ["--rebind-after-s", str(args.rebind_after_s)]
                        if args.rebind_after_s is not None
                        else []
                    ),
                    *(["--prewarm-stripped"] if args.prewarm_stripped else []),
                    *(
                        ["--prewarm-chunk-tokens", str(args.prewarm_chunk_tokens)]
                        if args.prewarm_chunk_tokens
                        else []
                    ),
                    *(
                        ["--prewarm-max-waiting", str(args.prewarm_max_waiting)]
                        if args.prewarm_max_waiting is not None
                        else []
                    ),
                    *(
                        ["--session-population", str(args.session_population)]
                        if args.session_population
                        else []
                    ),
                    *(
                        ["--session-ramp-s", str(args.session_ramp_s)]
                        if args.session_ramp_s
                        else []
                    ),
                    *(
                        ["--session-rate", str(args.session_rate)]
                        if args.session_rate
                        else []
                    ),
                    *(
                        ["--queue-hint-url", manager.daser_url]
                        if args.daser_queue_hint and arm.startswith("daser")
                        else []
                    ),
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
                    *(["--chain-outputs"] if args.chain_outputs else []),
                    *(["--ordered-admission"] if args.ordered_admission else []),
                    "--admission",
                    args.admission,
                    "--dependency-gap-s",
                    str(args.dependency_gap_s),
                    *(
                        ["--corpus-tokens", str(args.corpus_tokens)]
                        if args.corpus_tokens is not None
                        else []
                    ),
                    *(
                        ["--session-tokens", str(args.session_tokens)]
                        if args.session_tokens is not None
                        else []
                    ),
                ]
            )
        counter_trace = (
            CounterTrace(
                manager.daser_url,
                out_dir / "counter_trace.jsonl",
                args.counter_trace_interval_s,
            )
            if args.counter_trace_interval_s > 0 and arm.startswith("daser")
            else None
        )
        if counter_trace is not None:
            counter_trace.start()
        with (out_dir / "replay.log").open("wb") as replay_log:
            replay = subprocess.Popen(
                replay_command,
                cwd=REPO_ROOT,
                stdout=replay_log,
                stderr=subprocess.STDOUT,
            )
            try:
                stop.child = replay
                replay.wait()
            except ArmTerminated:
                # The signal landed before the replay was attached.
                replay.terminate()
                replay.wait()
                raise
            finally:
                stop.child = None
                if counter_trace is not None:
                    counter_trace.stop()
        daser_after = await _daser_metrics(arm, manager)
        vllm_after = await _vllm_metrics(manager)
        if arm.startswith("lmcache"):
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
                    if arm.startswith("lmcache")
                    else (manager.offload_config() if arm in _SERVER_TYPES else None)
                )
            ),
            "runtime_config": runtime_config,
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
        # Keep the service logs with the artifact; stores are scratch.
        for log in (manager.log_dir).glob("*.log"):
            (out_dir / log.name).write_bytes(log.read_bytes())
        stop_from_pid_file(manager.pid_file)
        if _wait_pids_exit(manager.pid_file):
            # Measured before deletion: the store is gone afterwards.
            (out_dir / "store_disk_usage.json").write_text(
                json.dumps(store_disk_usage(manager.store_dir))
            )
            shutil.rmtree(manager.store_dir, ignore_errors=True)
        else:
            print(f"{arm}: services still alive; kept {manager.store_dir}", flush=True)


def _arm_line(arm: str, result: dict[str, Any]) -> str:
    """Return the one-line console summary of a finished arm."""
    summary = result["summary"] or {}
    if isinstance(summary, list):
        # Restore-probe arms summarise per cell.
        return f"{arm}: rc={result['replay_returncode']} cells={len(summary)}"
    ttft = summary.get("ttft_ms", {})
    return (
        f"{arm}: rc={result['replay_returncode']} "
        f"done={summary.get('completed_requests')} "
        f"failed={summary.get('failed_requests')} "
        f"aborted={summary.get('aborted')} "
        f"wall={summary.get('wall_seconds', 0):.0f}s "
        f"ttft_p50={ttft.get('p50', 0):.0f} p95={ttft.get('p95', 0):.0f} "
        f"prefix={summary.get('prefix_cache')}"
    )


async def main_async(args: argparse.Namespace) -> None:
    """Run the selected arms sequentially on one GPU."""
    args.artifact.mkdir(parents=True, exist_ok=True)
    stop = StopSignals()
    stop.install()
    for arm in args.arms:
        result = await run_arm(arm, args, stop)
        print(_arm_line(arm, result), flush=True)
        if stop.requested:
            print(f"{arm}: stopped by signal; skipping remaining arms", flush=True)
            break


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    check_runtime_environment()
    try:
        asyncio.run(main_async(args))
    except ArmTerminated:
        sys.exit(128 + signal.SIGTERM)


if __name__ == "__main__":
    main()
