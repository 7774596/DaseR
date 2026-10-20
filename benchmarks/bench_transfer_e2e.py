# SPDX-License-Identifier: Apache-2.0
"""Run serial ABBA trials of raw or compressed DaseR transfers on real GPUs.

Without --execute this writes the manifest and experiment plan only. Every
executed arm owns a separate process group and a fresh scratch directory;
results and service logs survive, but stores are removed after service shutdown.
Use an isolated vLLM environment and reserve the selected GPU before execution.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator
import csv
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from types import FrameType
from typing import Any

COVERAGE_CONTRACT_REVISION = "unique-source-v2"


def parse_args() -> argparse.Namespace:
    """Return validated CLI arguments (Namespace); main-thread operation only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--candidate-repo", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, default=None)
    parser.add_argument("--gpu-id", required=True, help="Reserved GPU UUID or ordinal")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--rounds", type=int, default=2, help="Number of ABBA cycles")
    parser.add_argument("--families", type=int, default=8)
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--block-tokens", type=int, default=128)
    parser.add_argument("--concurrency", default="4,16")
    parser.add_argument("--warm-repeats", type=int, default=1)
    parser.add_argument(
        "--storage-format", choices=("raw", "compressed-online"), default="raw"
    )
    parser.add_argument("--prefetch-max-requests", type=int, choices=(0, 2), default=0)
    parser.add_argument("--gen-max-tokens", type=int, choices=(1, 128), default=1)
    parser.add_argument(
        "--min-cycle-ssd-bytes",
        type=int,
        default=1024**3,
        help="Minimum server read_bytes per complete warm cycle (default: 1 GiB)",
    )
    parser.add_argument("--l1-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--l2-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--gpu-util", type=float, default=0.75)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--request-timeout", type=float, default=360.0)
    parser.add_argument("--arm-timeout", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--arm-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--variant",
        choices=("baseline", "candidate"),
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    for name in ("baseline_repo", "candidate_repo", "model", "scratch_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    # Preserve a virtual environment's bin directory even when its Python is a
    # symlink to a base interpreter; vllm must come from this environment's PATH.
    args.python = args.python.absolute()
    args.concurrencies = [int(value) for value in args.concurrency.split(",")]
    if min(args.rounds, args.families, args.warm_repeats, *args.concurrencies) <= 0:
        parser.error("rounds, families, repeats and concurrency must be positive")
    if args.prefix_tokens <= 0 or args.suffix_tokens < 0 or args.block_tokens <= 0:
        parser.error("require positive prefix/block tokens and non-negative suffix")
    if args.prefix_tokens % args.block_tokens or args.suffix_tokens % args.block_tokens:
        parser.error("prefix and suffix lengths must be multiples of block tokens")
    if args.l1_bytes <= 0 or args.l2_bytes <= args.l1_bytes:
        parser.error("require 0 < L1 < L2")
    if args.min_cycle_ssd_bytes <= 0:
        parser.error("--min-cycle-ssd-bytes must be positive")
    if args.scratch_root.resolve().is_relative_to("/tmp"):
        parser.error("large benchmark stores must not use /tmp")
    args.request_manifest = (
        args.request_manifest or args.output / "requests.json"
    ).resolve()
    return args


def write_json(path: Path, payload: Any) -> None:
    """Write JSON payload (Any) atomically to path (Path); one writer required."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pending.replace(path)


def source_details(repo: Path, expected: str | None) -> dict[str, Any]:
    """Verify selected source (Path, optional commit prefix) and return evidence.

    Alters sys.path before imports; call once per fresh interpreter.
    """
    repo = repo.resolve()
    sys.path.insert(0, str(repo))
    modules = (
        "benchmarks.utils.servers",
        "benchmarks.utils.loadgen",
        "benchmarks.multiprefix_bench",
        "daser.config",
        "daser.transfer.iouring.layer",
    )
    files = {}
    for name in modules:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(repo):
            raise RuntimeError(f"wrong source for {name}: {path}, expected {repo}")
        files[name] = str(path)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if expected is not None and not commit.startswith(expected):
        raise RuntimeError(f"commit mismatch: expected {expected}, got {commit}")
    layer = repo / "daser/transfer/iouring/layer.py"
    diff = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "HEAD",
            "--",
            "daser",
            "benchmarks/utils",
            "benchmarks/multiprefix_bench.py",
        ]
    )
    if diff:
        raise RuntimeError("benchmark sources must match the selected commit")
    return {
        "repo": str(repo),
        "commit": commit,
        "module_files": files,
        "layer_sha256": hashlib.sha256(layer.read_bytes()).hexdigest(),
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.executable,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "httpx", "psutil")
        },
    }


def prepare_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Create or validate shared token inputs from args; no services are started."""
    from benchmarks.multiprefix_bench import build_request_manifest
    from benchmarks.utils.constants import slot_size_for_block_tokens
    from daser.compression.format import ONLINE_TILE_SCALARS, CompressedStoreGeometry
    from daser.config import model_geometry_from_path

    if args.request_manifest.exists():
        manifest = json.loads(args.request_manifest.read_text(encoding="utf-8"))
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        manifest = build_request_manifest(
            tokenizer=tokenizer,
            families=args.families,
            requests_per_family=1,
            prefix_tokens=args.prefix_tokens,
            suffix_tokens=args.suffix_tokens,
            block_tokens=args.block_tokens,
            seed=args.seed,
        )
        write_json(args.request_manifest, manifest)
    expected = {
        "families": args.families,
        "requests_per_family": 1,
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "block_tokens": args.block_tokens,
        "seed": args.seed,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("existing request manifest differs from CLI workload")
    slot_bytes = slot_size_for_block_tokens(str(args.model), args.block_tokens)
    family_bytes = (
        (args.prefix_tokens + args.suffix_tokens) // args.block_tokens * slot_bytes
    )
    working_set = family_bytes * args.families
    minimum_slot_bytes = slot_bytes
    if args.storage_format == "compressed-online":
        geometry = model_geometry_from_path(str(args.model))
        minimum_slot_bytes = CompressedStoreGeometry(
            num_slots=1,
            slot_size=slot_bytes,
            block_tokens=args.block_tokens,
            num_layers=geometry.num_layers,
            num_kv_heads=geometry.num_kv_heads,
            head_dim=geometry.head_dim,
            dtype_bytes=geometry.dtype_bytes,
            tile_scalars=ONLINE_TILE_SCALARS,
        ).online_min_stored_length
    # The packed format transfers physical bytes, not raw KV bytes. Use its
    # format-defined minimum to enforce coverage without assuming a ratio.
    minimum_prefix_bytes = args.prefix_tokens // args.block_tokens * minimum_slot_bytes
    minimum_working_set = args.families * minimum_prefix_bytes
    if args.l1_bytes >= minimum_working_set / 2:
        raise ValueError("L1 must be less than half the minimum physical prefix set")
    if not family_bytes <= args.l1_bytes < working_set / 2:
        raise ValueError("L1 must fit one family and be less than half the working set")
    if args.l2_bytes < 2 * working_set:
        raise ValueError("L2 must fit two full working sets to exclude ring eviction")
    return {
        "coverage_contract_revision": COVERAGE_CONTRACT_REVISION,
        "load_coverage_basis": "one unique prefix source per family wave",
        "external_hit_semantics": (
            "lookup-covered tokens, not per-target GPU restore completion"
        ),
        "storage_format": args.storage_format,
        "prefetch_max_requests": args.prefetch_max_requests,
        "gen_max_tokens": args.gen_max_tokens,
        "slot_bytes": slot_bytes,
        "minimum_slot_bytes": minimum_slot_bytes,
        "minimum_prefix_bytes": minimum_prefix_bytes,
        "prefix_bytes": args.prefix_tokens // args.block_tokens * slot_bytes,
        "family_bytes": family_bytes,
        "working_set_bytes": working_set,
        "min_cycle_ssd_bytes": args.min_cycle_ssd_bytes,
        "capacity_ssd_lower_bound_bytes": max(
            0,
            minimum_working_set - args.l1_bytes,
        ),
        "manifest_sha256": hashlib.sha256(
            args.request_manifest.read_bytes()
        ).hexdigest(),
    }


def free_ports() -> tuple[int, int]:
    """Return two distinct free localhost ports; callers must bind promptly."""
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        return first.getsockname()[1], second.getsockname()[1]


def process_read_bytes(pid: int) -> int:
    """Return kernel read_bytes for owned server PID (int); read-only OS probe."""
    rows = Path(f"/proc/{pid}/io").read_text(encoding="utf-8").splitlines()
    return int(
        next(row.split(":", 1)[1] for row in rows if row.startswith("read_bytes:"))
    )


def percentile(values: list[float], fraction: float) -> float | None:
    """Return nearest-rank quantile from values and fraction; pure operation."""
    return (
        sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]
        if values
        else None
    )


def phase_summary(waves: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all waves' requests without filtering by observed storage tier.

    Args: waves contains completed request waves. Returns metrics and validity.
    Pure calculation; failed requests are counted but excluded from latency stats.
    """
    rows = [row for wave in waves for row in wave["requests"]]
    valid = [row for row in rows if not row["validation_errors"]]
    ttfts = [row["ttft_ms"] for row in valid]
    latencies = [row["latency_ms"] for row in valid]
    elapsed_s = sum(wave["elapsed_ms"] for wave in waves) / 1000
    output_tokens = sum(row["completion_tokens"] for row in valid)
    return {
        "request_count": len(rows),
        "successful_requests": len(valid),
        "failed_requests": len(rows) - len(valid),
        "ttft_p50_ms": percentile(ttfts, 0.5),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "completion_tokens": output_tokens,
        "requests_per_second": len(valid) / elapsed_s if elapsed_s else None,
        "output_tokens_per_second": output_tokens / elapsed_s if elapsed_s else None,
        "all_waves_valid": bool(waves) and all(wave["valid"] for wave in waves),
        "l2_reads": sum(wave["counters"]["l2_reads"] for wave in waves),
        "process_read_bytes": sum(
            wave["counters"]["process_read_bytes"] for wave in waves
        ),
        "prefetch_operations": sum(
            wave["counters"]["prefetch_operations"] for wave in waves
        ),
        "server_load_operations": sum(
            wave["counters"]["load_operations"] for wave in waves
        ),
        "server_load_bytes": sum(wave["counters"]["load_bytes"] for wave in waves),
        "external_hit_tokens": sum(
            wave["counters"]["external_hit_tokens"] for wave in waves
        ),
        "external_query_tokens": sum(
            wave["counters"]["external_query_tokens"] for wave in waves
        ),
        "ssd_touched_waves": sum(wave["ssd_touched"] for wave in waves),
        "l1_only_waves": sum(wave["l1_only"] for wave in waves),
    }


def cycle_summary(
    args: argparse.Namespace,
    contract: dict[str, Any],
    waves: list[dict[str, Any]],
    concurrency: int,
    repeat: int,
) -> dict[str, Any]:
    """Validate physical source coverage and logical lookup coverage separately.

    Args:
        args: Namespace containing the fixed workload and SSD limits.
        contract: Dict containing the minimum physical bytes per prefix source.
        waves: Completed request-wave dicts for exactly one pass over all families.
        concurrency: Number of requests required in each family wave.
        repeat: Zero-based warm-cycle index for reporting.
    Returns:
        Dict containing measurements, explicit coverage expectations and failures.
    Async/thread-safety:
        Pure calculation outside timed regions; inputs are never modified.
    """
    summary = phase_summary(waves)
    reasons = []
    expected_requests = args.families * concurrency
    minimum_wave_load_bytes = contract["minimum_prefix_bytes"]
    expected_load_bytes = args.families * minimum_wave_load_bytes
    prompt_tokens = args.prefix_tokens + args.suffix_tokens
    expected_wave_query_tokens = concurrency * prompt_tokens
    expected_wave_hit_tokens = concurrency * (prompt_tokens - 1)
    if len(waves) != args.families or {wave["family_id"] for wave in waves} != set(
        range(args.families)
    ):
        reasons.append("cycle must contain exactly one wave per planned family")
    if summary["request_count"] != expected_requests:
        reasons.append("cycle request count differs from the manifest")
    if not summary["all_waves_valid"]:
        reasons.append("one or more request waves failed validation")
    # One family wave repeats an identical immutable source. Worker planning
    # can read it once and fan out its staging bytes to several destinations.
    # Requiring a separate physical copy per request would reject valid sharing.
    # Check every wave independently so excess traffic from another family
    # cannot conceal a missing source. Raw and packed use the same invariant.
    for wave in waves:
        label = f"wave {wave['ordinal']} (family {wave['family_id']})"
        if len(wave["requests"]) != concurrency:
            reasons.append(f"{label}: request count differs from concurrency")
        counters = wave["counters"]
        if counters["load_bytes"] < minimum_wave_load_bytes:
            reasons.append(
                f"{label}: external load bytes do not cover its unique prefix"
            )
        # All warm requests repeat a fully seeded prompt. The connector leaves
        # its final token for model execution, hence prompt_tokens - 1 hits.
        # These are lookup-coverage counters, not evidence of each target DMA;
        # request/output checks and the normal restore completion contract remain
        # independent requirements. Exact per-wave counts also reject retries
        # or double-counting that could hide under a cycle-wide lower bound.
        if counters["external_query_tokens"] != expected_wave_query_tokens:
            reasons.append(
                f"{label}: external query tokens differ from exact expectation"
            )
        if counters["external_hit_tokens"] != expected_wave_hit_tokens:
            reasons.append(
                f"{label}: external hit tokens differ from exact expectation"
            )
    if summary["l2_reads"] <= 0:
        reasons.append("cycle observed no SSD reads")
    if summary["process_read_bytes"] < args.min_cycle_ssd_bytes:
        reasons.append(
            f"cycle read_bytes={summary['process_read_bytes']} below prespecified "
            f"minimum={args.min_cycle_ssd_bytes}"
        )
    if args.prefetch_max_requests and summary["prefetch_operations"] == 0:
        reasons.append("prefetch-enabled cycle observed no prefetch operations")
    return {
        "coverage_contract_revision": COVERAGE_CONTRACT_REVISION,
        "concurrency": concurrency,
        "repeat": repeat,
        "summary": summary,
        "minimum_ssd_bytes": args.min_cycle_ssd_bytes,
        "minimum_external_load_bytes": expected_load_bytes,
        "minimum_external_load_bytes_per_wave": minimum_wave_load_bytes,
        "expected_external_query_tokens": args.families * expected_wave_query_tokens,
        "expected_external_hit_tokens": args.families * expected_wave_hit_tokens,
        "expected_external_query_tokens_per_wave": expected_wave_query_tokens,
        "expected_external_hit_tokens_per_wave": expected_wave_hit_tokens,
        "wave_ordinals": [wave["ordinal"] for wave in waves],
        "valid": not reasons,
        "invalid_reasons": reasons,
    }


async def counter_snapshot(client: Any, service: Any, pid: int) -> dict[str, float]:
    """Read public HTTP and owned-process counters; shared async client is safe."""
    response = await client.get(f"{service.endpoints['daser'].url}/metrics")
    response.raise_for_status()
    values = {
        "l2_reads": 0.0,
        "load_operations": 0.0,
        "load_bytes": 0.0,
        "l1_hits": 0.0,
        "l1_misses": 0.0,
        "prefetch_operations": 0.0,
        "external_hit_tokens": 0.0,
        "external_query_tokens": 0.0,
    }
    direct = {
        "daser_l2_reads_total": "l2_reads",
        "daser_l1_hits_total": "l1_hits",
        "daser_l1_misses_total": "l1_misses",
        "daser_external_prefix_cache_hits_total": "external_hit_tokens",
        "daser_external_prefix_cache_queries_total": "external_query_tokens",
    }
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric, value = line.rsplit(" ", 1)
        name = metric.split("{", 1)[0]
        if name in direct:
            values[direct[name]] += float(value)
        elif name == "daser_transfer_operations_total" and 'op="load"' in metric:
            values["load_operations"] += float(value)
        elif name == "daser_transfer_bytes_total" and 'op="load"' in metric:
            values["load_bytes"] += float(value)
        elif name == "daser_prefetch_operations_total":
            values["prefetch_operations"] += float(value)
    values["process_read_bytes"] = process_read_bytes(pid)
    return values


async def drain(client: Any, service: Any) -> None:
    """Await public DaseR drain endpoint for service; stores finish before return."""
    response = await client.post(
        f"{service.endpoints['daser'].url}/drain", timeout=360.0
    )
    response.raise_for_status()
    if response.json().get("ok") is not True:
        raise RuntimeError("DaseR did not confirm drain")


async def send_wave(
    args: argparse.Namespace,
    service: Any,
    pid: int,
    family: dict[str, Any],
    phase: str,
    concurrency: int,
    ordinal: int,
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    """Measure one family wave using the public streaming helper and real SSE.

    Args are immutable configuration, service manifest, owned server PID,
    token-family row, phase name, concurrency, unique wave ordinal and optional
    seed output. Returns all requests and validity/counters. One wave at a time.
    """
    import httpx

    from benchmarks.utils.datasets import BenchmarkSample
    from benchmarks.utils.loadgen import vllm_completion_stream

    captures: dict[str, list[bytes]] = {}

    class CaptureStream(httpx.AsyncByteStream):
        """Tee response bytes without changing the public helper's TTFT clock."""

        def __init__(self, stream: httpx.AsyncByteStream, chunks: list[bytes]) -> None:
            self.stream = stream
            self.chunks = chunks

        async def __aiter__(self) -> AsyncIterator[bytes]:
            async for chunk in self.stream:
                self.chunks.append(chunk)
                yield chunk

        async def aclose(self) -> None:
            """Close the wrapped async HTTP stream; awaited by its sole owner."""
            await self.stream.aclose()

    async def capture(response: httpx.Response) -> None:
        if response.request.url.path != "/v1/completions":
            return
        body = json.loads(response.request.content)
        chunks: list[bytes] = []
        captures[body["request_id"]] = chunks
        response.stream = CaptureStream(response.stream, chunks)

    async with httpx.AsyncClient(
        timeout=args.request_timeout,
        event_hooks={"response": [capture]},
        limits=httpx.Limits(max_connections=max(concurrency + 2, 20)),
    ) as client:
        before = await counter_snapshot(client, service, pid)
        semaphore = asyncio.Semaphore(concurrency)
        params = {
            "model": str(args.model),
            "max_tokens": args.gen_max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": args.seed,
            "ignore_eos": True,
            "skip_special_tokens": False,
        }
        if phase == "warm":
            params["kv_transfer_params"] = {"daser_skip_save": True}
        samples = [
            BenchmarkSample(ordinal * 100000 + index, "demand-e2e", "", "", [])
            for index in range(concurrency)
        ]
        start = time.perf_counter()
        requests = await asyncio.gather(
            *[
                vllm_completion_stream(
                    client,
                    service.endpoints["vllm"].url,
                    sample,
                    family["prompt"],
                    params,
                    semaphore,
                    args.request_timeout,
                )
                for sample in samples
            ]
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        await drain(client, service)
        after = await counter_snapshot(client, service, pid)
    rows = []
    for result in requests:
        row = asdict(result)
        token_ids = []
        for line in b"".join(
            captures.get(f"trace-{result.sample_id}", [])
        ).splitlines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            for choice in event.get("choices", []):
                token_ids.extend(choice.get("token_ids") or [])
        errors = []
        if result.error is not None:
            errors.append(result.error)
        if (
            not result.first_token_observed
            or result.completion_tokens != args.gen_max_tokens
        ):
            errors.append("missing token event or unexpected completion token count")
        if len(token_ids) != args.gen_max_tokens:
            errors.append(
                f"expected {args.gen_max_tokens} returned token IDs, "
                f"got {len(token_ids)}"
            )
        if result.prompt_tokens != len(family["prompt"]):
            errors.append("prompt usage differs from token manifest")
        output = {"generated_text": result.generated_text, "token_ids": token_ids}
        if reference is not None and output != reference:
            errors.append(f"output differs from seed: {reference}")
        row.update(token_ids=token_ids, validation_errors=errors)
        rows.append(row)
    counters = {key: after[key] - before[key] for key in before}
    reasons = []
    if any(row["validation_errors"] for row in rows):
        reasons.append("one or more requests failed validation")
    # BIP can protect early or reused allocations even when the total working
    # set exceeds L1. Tier labels are diagnostics only: validating SSD traffic
    # over the complete predetermined family cycle avoids selecting results
    # based on cache state changed by the optimization itself.
    if phase == "cold" and counters["l2_reads"] != 0:
        reasons.append("fresh family unexpectedly loaded cached data during seed")
    if args.prefetch_max_requests == 0 and counters["prefetch_operations"] != 0:
        reasons.append("prefetch observed despite disabled configuration")
    return {
        "phase": phase,
        "family_id": family["family_id"],
        "concurrency": concurrency,
        "ordinal": ordinal,
        "l1_bytes": args.l1_bytes,
        "elapsed_ms": elapsed_ms,
        "counters": counters,
        "ssd_touched": counters["l2_reads"] > 0,
        "l1_only": phase == "warm"
        and counters["l2_reads"] == 0
        and counters["load_bytes"] > 0,
        "valid": not reasons,
        "invalid_reasons": reasons,
        "requests": rows,
    }


def stop_owned_services(pid_file: Path) -> dict[str, Any]:
    """Stop only this arm's descendants, including partial startup; return audit.

    Uses live process-tree ownership and psutil identity checks, including
    startup failures before a PID file exists. Stored PIDs are never signalled.
    Blocking cleanup runs after all requests and outside measured phases.
    """
    import psutil

    del pid_file

    owned = psutil.Process().children(recursive=True)
    identities = [{"pid": proc.pid, "created": proc.create_time()} for proc in owned]
    for proc in reversed(owned):
        try:
            if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=15)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    _, survivors = psutil.wait_procs(alive, timeout=10)
    live = [
        proc.pid
        for proc in survivors
        if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    ]
    if live:
        raise RuntimeError(f"owned services survived cleanup: {live}")
    return {"owned_processes": identities, "surviving_live_processes": live}


async def run_arm(args: argparse.Namespace) -> None:
    """Run one fresh-store arm and always stop owned services before cleanup."""
    repo = getattr(args, f"{args.variant}_repo")
    if Path.cwd().resolve() != repo.resolve():
        raise RuntimeError("arm working directory differs from selected repository")
    source = source_details(repo, getattr(args, f"{args.variant}_commit"))
    from benchmarks.utils.servers import ServerManager

    contract = prepare_manifest(args)
    families = json.loads(args.request_manifest.read_text())["requests"]
    arm_dir = args.output / f"{args.arm_index:02d}-{args.variant}"
    arm_dir.mkdir(parents=True, exist_ok=False)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    store = Path(
        tempfile.mkdtemp(prefix=f"e2e-{args.arm_index:02d}-", dir=args.scratch_root)
    )
    ports = free_ports()
    manager = ServerManager(
        run_id=arm_dir.name,
        backend="daser",
        model=str(args.model),
        store_dir=store,
        gpu_id=args.gpu_id,
        gpu_util=args.gpu_util,
        max_num_seqs=max(args.concurrencies),
        l1_size_bytes=args.l1_bytes,
        l2_size_bytes=args.l2_bytes,
        max_num_batched_tokens=args.max_num_batched_tokens,
        block_size=args.block_tokens,
        reuse_mode="prefix",
        transfer_mode="iouring",
        vllm_port=ports[0],
        daser_port=ports[1],
        startup_timeout=args.startup_timeout,
        max_model_len=(
            args.prefix_tokens
            + args.suffix_tokens
            + max(args.gen_max_tokens, args.block_tokens)
        ),
        skip_l2=False,
        daser_prefetch_max_requests=args.prefetch_max_requests,
        storage_format=args.storage_format,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "variant": args.variant,
        "source": source,
        "contract": contract,
        "status": "starting",
        "scratch_store": str(store),
        "waves": [],
        "cycles": [],
    }
    write_json(arm_dir / "result.json", payload)
    cleanup_ok = False
    try:
        service = await manager.start()
        payload["service"] = asdict(service)
        processes = json.loads(manager.pid_file.read_text())
        payload["processes"] = processes
        server = next(item for item in processes if "daser.server" in item["cmd"])
        for proc in processes:
            if Path(f"/proc/{proc['pid']}/cwd").resolve() != repo.resolve():
                raise RuntimeError(
                    "service working directory differs from selected source"
                )
            environ = Path(f"/proc/{proc['pid']}/environ").read_bytes().split(b"\0")
            pythonpath = next(
                value for value in environ if value.startswith(b"PYTHONPATH=")
            )
            if (
                pythonpath.split(b"=", 1)[1].split(os.pathsep.encode())[0]
                != str(repo.resolve()).encode()
            ):
                raise RuntimeError(
                    "service source PYTHONPATH differs from selected repository"
                )
        payload["status"] = "running"
        references = {}
        for family in families:
            wave = await send_wave(
                args,
                service,
                server["pid"],
                family,
                "cold",
                1,
                len(payload["waves"]),
                None,
            )
            payload["waves"].append(wave)
            write_json(arm_dir / "result.json", payload)
            if not wave["valid"]:
                raise RuntimeError(
                    f"cold family {family['family_id']} failed: "
                    f"{wave['invalid_reasons']}"
                )
            row = wave["requests"][0]
            references[family["family_id"]] = {
                "generated_text": row["generated_text"],
                "token_ids": row["token_ids"],
            }
        payload["seed_outputs"] = references
        for concurrency in args.concurrencies:
            for repeat in range(args.warm_repeats):
                cycle_waves = []
                for family in families:
                    wave = await send_wave(
                        args,
                        service,
                        server["pid"],
                        family,
                        "warm",
                        concurrency,
                        len(payload["waves"]),
                        references[family["family_id"]],
                    )
                    wave["repeat"] = repeat
                    payload["waves"].append(wave)
                    cycle_waves.append(wave)
                    write_json(arm_dir / "result.json", payload)
                    if not wave["valid"]:
                        raise RuntimeError(
                            f"warm family {family['family_id']} failed: "
                            f"{wave['invalid_reasons']}"
                        )
                cycle = cycle_summary(args, contract, cycle_waves, concurrency, repeat)
                payload["cycles"].append(cycle)
                write_json(arm_dir / "result.json", payload)
                if not cycle["valid"]:
                    raise RuntimeError(
                        f"warm cycle concurrency={concurrency}, repeat={repeat} "
                        f"failed: {cycle['invalid_reasons']}"
                    )
        payload["status"] = "ok"
    except BaseException as exc:
        payload["status"] = "error"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload["cold"] = phase_summary(
            [wave for wave in payload["waves"] if wave["phase"] == "cold"]
        )
        payload["warm"] = {
            str(concurrency): phase_summary(
                [
                    wave
                    for wave in payload["waves"]
                    if wave["phase"] == "warm" and wave["concurrency"] == concurrency
                ]
            )
            for concurrency in args.concurrencies
        }
        try:
            payload["cleanup"] = stop_owned_services(manager.pid_file)
            cleanup_ok = True
        except BaseException as exc:
            payload["status"] = "error"
            payload["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if manager.log_dir.exists():
                shutil.copytree(manager.log_dir, arm_dir / "logs", dirs_exist_ok=True)
            if manager.pid_file.exists():
                shutil.copy2(manager.pid_file, arm_dir / "pids.json")
            payload["stores_removed"] = cleanup_ok
            if cleanup_ok:
                shutil.rmtree(store)
            else:
                payload["retained_store"] = str(store)
            write_json(arm_dir / "result.json", payload)


def reap_arm_group(proc: subprocess.Popen[bytes]) -> None:
    """Terminate an isolated owned arm group and verify exit, including orphans.

    Args: proc is a child launched with start_new_session=True. Returns None.
    Called serially by the outer controller, never while timing requests.
    """
    import psutil

    members = []
    for candidate in psutil.process_iter():
        try:
            if os.getpgid(candidate.pid) == proc.pid:
                members.append(candidate)
        except (ProcessLookupError, PermissionError, psutil.NoSuchProcess):
            continue
    if members:
        os.killpg(proc.pid, signal.SIGTERM)
        _, alive = psutil.wait_procs(members, timeout=45)
        if any(p.status() != psutil.STATUS_ZOMBIE for p in alive):
            os.killpg(proc.pid, signal.SIGKILL)
            _, alive = psutil.wait_procs(alive, timeout=10)
        if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in alive):
            raise RuntimeError(f"arm group {proc.pid} did not stop")
    proc.wait()


def clean_interrupted_store(args: argparse.Namespace, result_path: Path) -> None:
    """Remove only an interrupted arm's recorded store after group exit.

    Args: args holds scratch root; result_path is the arm-owned result file.
    Returns None. Caller must first prove all processes in that arm have stopped.
    """
    if not result_path.exists():
        return
    result = json.loads(result_path.read_text())
    store = Path(result["scratch_store"]).resolve()
    if store.parent != args.scratch_root.resolve() or not store.name.startswith("e2e-"):
        raise RuntimeError(f"refusing cleanup of unrecognized store: {store}")
    if store.exists():
        if (store / "logs").exists():
            shutil.copytree(
                store / "logs", result_path.parent / "logs", dirs_exist_ok=True
            )
        shutil.rmtree(store)
    result["stores_removed"] = True
    result["outer_group_cleanup"] = True
    if result["status"] in ("starting", "running"):
        result["status"] = "interrupted"
    write_json(result_path, result)


def ensure_gpu_idle(gpu_id: str) -> dict[str, Any]:
    """Check the selected GPU is idle and permits the two CUDA service processes.

    Args: gpu_id (str) is an NVIDIA UUID or ordinal. Returns the selected UUID
    and two read-only NVIDIA CLI samples. Raises RuntimeError when the selected
    device is busy or incompatible. Blocks outside request timing; does not
    reserve the GPU, change compute mode, or terminate other processes.
    """
    samples = []
    for index in range(2):
        if index:
            time.sleep(3.0)
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu_id,
                "--query-gpu=uuid,compute_mode,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
        rows = list(csv.reader(output.strip().splitlines()))
        if len(rows) != 1 or len(rows[0]) != 4:
            raise RuntimeError("expected one selected GPU from nvidia-smi")
        uuid, mode, memory, utilization = [value.strip() for value in rows[0]]
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                uuid,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        ).strip()
        sample = dict(
            uuid=uuid,
            compute_mode=mode,
            used_mib=int(memory),
            utilization_pct=int(utilization),
            compute_pids=apps.splitlines(),
        )
        if mode != "Default" or int(memory) > 1024 or int(utilization) > 5 or apps:
            raise RuntimeError(f"selected GPU is not idle/compatible: {sample}")
        if samples and uuid != samples[0]["uuid"]:
            raise RuntimeError("selected GPU identity changed between checks")
        samples.append(sample)
    return {"uuid": samples[0]["uuid"], "samples": samples}


def run_suite(args: argparse.Namespace) -> None:
    """Prepare and optionally execute serial ABBA arms from args; no parallel IO."""
    source = source_details(args.candidate_repo, args.candidate_commit)
    contract = prepare_manifest(args)
    order = [
        variant
        for _ in range(args.rounds)
        for variant in ("baseline", "candidate", "candidate", "baseline")
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "planned",
        "order": order,
        "contract": contract,
        "candidate_source": source,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "arms": [],
    }
    write_json(args.output / "suite.json", payload)
    if not args.execute:
        return
    payload["status"] = "running"
    reference = None
    child_env = os.environ.copy()
    child_env["PATH"] = str(args.python.parent) + os.pathsep + child_env["PATH"]
    child_env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    child_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    child_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    shared_argv = []
    for key, value in vars(args).items():
        if key in ("concurrencies", "arm_index", "variant") or value is None:
            continue
        shared_argv.append("--" + key.replace("_", "-"))
        if not isinstance(value, bool):
            shared_argv.append(str(value))
    try:
        for index, variant in enumerate(order):
            state = ensure_gpu_idle(args.gpu_id)
            payload.setdefault("gpu_preflights", []).append(state)
            write_json(args.output / "suite.json", payload)
            result_path = args.output / f"{index:02d}-{variant}" / "result.json"
            command = [
                str(args.python),
                str(Path(__file__).resolve()),
                *shared_argv,
                "--arm-index",
                str(index),
                "--variant",
                variant,
            ]
            log_path = args.output / f"{index:02d}-{variant}-orchestrator.log"
            with log_path.open("wb") as log:
                proc = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                    cwd=getattr(args, f"{variant}_repo"),
                    start_new_session=True,
                )
                try:
                    code = proc.wait(timeout=args.arm_timeout)
                except BaseException:
                    # The isolated process group contains only this arm and its
                    # services, including startup failures without a PID file.
                    reap_arm_group(proc)
                    clean_interrupted_store(args, result_path)
                    raise
            reap_arm_group(proc)
            clean_interrupted_store(args, result_path)
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            payload["arms"].append(
                {
                    "index": index,
                    "variant": variant,
                    "returncode": code,
                    "result": str(result_path),
                }
            )
            write_json(args.output / "suite.json", payload)
            if (
                code != 0
                or result.get("status") != "ok"
                or not result.get("stores_removed")
            ):
                raise RuntimeError(
                    f"arm {index} ({variant}) failed; inspect {log_path}"
                )
            if reference is None:
                reference = result["seed_outputs"]
            elif result["seed_outputs"] != reference:
                raise RuntimeError(
                    f"arm {index}: generated token IDs/text differ across variants"
                )
        payload["status"] = "ok"
        payload["cross_arm_outputs_match"] = True
    except BaseException as exc:
        payload["status"] = "error"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(args.output / "suite.json", payload)


def main() -> None:
    """Dispatch plan/suite or isolated arm; nonzero exit means invalid evidence."""
    args = parse_args()
    signal.signal(signal.SIGTERM, interrupt_arm)
    if args.arm_index is None:
        run_suite(args)
    else:
        if not args.execute or args.variant is None:
            raise ValueError("internal arm mode requires --execute and --variant")
        signal.signal(signal.SIGTERM, interrupt_arm)
        asyncio.run(run_arm(args))


def interrupt_arm(signum: int, frame: FrameType | None) -> None:
    """Turn termination signal (int, frame) into cleanup-capable interruption.

    Returns no value; raises KeyboardInterrupt in the arm's main thread.
    """
    del signum, frame
    raise KeyboardInterrupt("arm termination requested")


if __name__ == "__main__":
    main()
