# SPDX-License-Identifier: Apache-2.0
"""Measure real grouped stores from CUDA/pinned memory through io_uring to SSD.

Acceptance ends when store_bytes_grouped returns; completion ends after drain.
Preparation and verification are untimed. Reopening the transfer before reading
forces verification through SSD instead of accepting an L1 copy as evidence.
Packed inputs are opaque byte records with gaps between allocation envelopes;
this measures transfer behavior, not compression quality or codec throughput.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any

# Keep the CLI directly executable while sharing source identity and atomic
# reporting with the demand benchmark. That module imports no production code;
# source_info subsequently puts the explicitly requested --repo first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.bench_demand_load import (  # noqa: E402
    ALIGNMENT,
    counter_delta,
    parse_size,
    source_info,
    write_result,
)

CASES = ("same", "overlap", "disjoint")
FORMATS = ("raw", "packed")
SOURCES = ("cuda", "pinned")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse optional CLI strings into a validated Namespace, without IO.

    Args:
        argv (Sequence[str] | None): argument strings, or process arguments.
    Returns:
        argparse.Namespace: validated matrix and resource limits.
    Async/thread-safety:
        Pure main-thread setup; no CUDA or storage allocation occurs here.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--variant", required=True)
    parser.add_argument("--expect-commit", default=None)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0, help="CUDA-visible ordinal")
    parser.add_argument("--sizes", default="4KiB,256KiB,4MiB,16MiB")
    parser.add_argument("--concurrency", default="1,4,16")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument("--sources", default="cuda")
    parser.add_argument(
        "--spans", type=int, default=4, help="Maximum spans per request"
    )
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l1-bytes", type=parse_size, default=None)
    parser.add_argument("--max-source-bytes", type=parse_size, default=512 << 20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--bip-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use production-default BIP replacement (enabled)",
    )
    parser.add_argument(
        "--native-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Observe completed real NativeIOUring.write calls",
    )
    args = parser.parse_args(argv)
    try:
        args.sizes = [parse_size(item) for item in args.sizes.split(",")]
        args.concurrency = [int(item) for item in args.concurrency.split(",")]
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    for name, choices in (("cases", CASES), ("formats", FORMATS), ("sources", SOURCES)):
        selected = getattr(args, name).split(",")
        if not selected or any(item not in choices for item in selected):
            parser.error(f"--{name} must select from {','.join(choices)}")
        setattr(args, name, selected)
    if min(args.concurrency) <= 0 or args.repeat <= 0 or args.warmup < 0:
        parser.error("concurrency/repeat must be positive and warmup nonnegative")
    if args.spans <= 0 or args.io_workers <= 0 or args.timeout <= 0:
        parser.error("spans, io-workers and timeout must be positive")
    if max(args.sizes) * max(args.concurrency) > args.max_source_bytes:
        parser.error("largest wave exceeds --max-source-bytes")
    if args.l1_bytes is not None and args.l1_bytes < max(args.sizes):
        parser.error("--l1-bytes must accommodate one largest request")
    return args


def build_requests(
    size: int, concurrency: int, case: str, storage_format: str, max_spans: int
) -> list[list[dict[str, Any]]]:
    """Build aligned grouped store spans with deterministic overlapping keys.

    Args:
        size/concurrency/max_spans (int): request bytes, request count, span limit.
        case (str): same, overlap (half the spans), or disjoint.
        storage_format (str): raw contiguous extents or packed gapped extents.
    Returns:
        list[list[dict[str, Any]]]: source/file offsets and byte counts per request.
    Async/thread-safety:
        Pure geometry calculation, safe for concurrent callers. Raises ValueError
        for half-overlap smaller than two aligned spans rather than rounding it.
    """
    if case not in CASES or storage_format not in FORMATS:
        raise ValueError("unknown case or storage format")
    count = min(max_spans, size // ALIGNMENT)
    while count > 1 and size % (count * ALIGNMENT):
        count -= 1
    if case == "overlap" and (count < 2 or count % 2):
        raise ValueError("half-overlap requires an even number of aligned spans")
    span_bytes = size // count
    stride = span_bytes * (2 if storage_format == "packed" else 1)
    requests = []
    for index in range(concurrency):
        first = (
            0
            if case == "same"
            else index * (count // 2 if case == "overlap" else count)
        )
        requests.append(
            [
                {
                    "source_offset": part * span_bytes,
                    "file_offset": (first + part) * stride,
                    "nbytes": span_bytes,
                    "packed": storage_format == "packed",
                }
                for part in range(count)
            ]
        )
    return requests


def process_write_bytes() -> int | None:
    """Return process storage write bytes (int | None), with synchronous observation.

    No arguments; reads Linux process counters without altering kernel state.
    """
    try:
        fields = dict(
            line.split(":", 1)
            for line in Path("/proc/self/io").read_text().splitlines()
        )
        return int(fields["write_bytes"])
    except (OSError, KeyError, ValueError):
        return None


def summarize(samples: list[dict[str, Any]], numpy: Any) -> dict[str, Any]:
    """Summarize raw samples (list[dict]) using numpy (Any); return metrics (dict).

    Pure CPU work outside timed intervals, safe for synchronous callers.
    """
    requests = [
        value for sample in samples for value in sample["acceptance_latency_ms"]
    ]
    metrics: dict[str, Any] = {"waves": len(samples), "requests": len(requests)}
    for label, values in (
        ("acceptance", requests),
        ("wave_acceptance", [sample["acceptance_ms"] for sample in samples]),
        ("drain", [sample["drain_ms"] for sample in samples]),
        ("completion", [sample["completion_ms"] for sample in samples]),
    ):
        for percentile in (50, 95):
            metrics[f"{label}_p{percentile}_ms"] = float(
                numpy.percentile(values, percentile)
            )
    metrics.update(
        completion_throughput_gib_s=sum(sample["requested_bytes"] for sample in samples)
        / (1 << 30)
        / (sum(sample["completion_ms"] for sample in samples) / 1000),
        l2_writes_total=sum(sample["l2_writes"] for sample in samples),
        native_write_calls_total=(
            sum(sample["native_write_calls"] for sample in samples)
            if samples[0]["native_write_calls"] is not None
            else None
        ),
        native_write_bytes_total=(
            sum(sample["native_write_bytes"] for sample in samples)
            if samples[0]["native_write_bytes"] is not None
            else None
        ),
        correctness=all(sample["correctness"] for sample in samples),
    )
    return metrics


class NativeWriteCounter:
    """Observe real public writes after completion, preserving kernel behavior.

    Args:
        enabled (bool): whether to install the observer.
    Async/thread-safety:
        Install/restore only with no active IO; snapshots and updates use a lock.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.calls = 0
        self.nbytes = 0
        self.lock = threading.Lock()
        self.owner: Any = None
        self.original: Any = None

    def install(self) -> None:
        """Install the observer; no arguments/return, requires quiescent IO."""
        if not self.enabled:
            return
        from daser.transfer.iouring.native import NativeIOUring

        original = NativeIOUring.write
        counter = self

        def counted(
            ring: Any, fd: int, file_offset: int, data: bytes | memoryview
        ) -> int:
            result = original(ring, fd, file_offset, data)
            with counter.lock:
                counter.calls += 1
                counter.nbytes += result
            return result

        self.owner, self.original = NativeIOUring, original
        NativeIOUring.write = counted

    def snapshot(self) -> tuple[int, int]:
        """Return completed calls/bytes (tuple[int, int]) under the counter lock."""
        with self.lock:
            return self.calls, self.nbytes

    def restore(self) -> None:
        """Restore the original method; no arguments/return, requires drained IO."""
        if self.owner is not None:
            self.owner.write = self.original
            self.owner = None


async def run_wave(
    layer_type: Any,
    cupy: Any,
    numpy: Any,
    path: Path,
    requests: list[list[dict[str, Any]]],
    size: int,
    source_kind: str,
    args: argparse.Namespace,
    counter: NativeWriteCounter,
    iteration: int,
) -> dict[str, Any]:
    """Measure one store wave and validate actual SSD contents after reopening.

    Args:
        layer_type/cupy/numpy (Any): production transfer and actual array modules.
        path (Path): ephemeral SSD store, replaced before the wave.
        requests (list[list[dict]]): aligned span geometry; size (int): source bytes.
        source_kind (str): cuda or pinned; args (Namespace): runtime configuration.
        counter (NativeWriteCounter): optional completed-write observer.
        iteration (int): seed offset so consecutive waves use different contents.
    Returns:
        dict[str, Any]: acceptance/completion timing, IO counts, byte-exact result.
    Async/thread-safety:
        One event loop owns all layers; sources stay alive until tasks and native
        IO drain. Verification uses a newly opened layer, forcing L2 reads.
    """
    extents = sorted(
        {(span["file_offset"], span["nbytes"]) for spans in requests for span in spans}
    )
    l1_bytes = args.l1_bytes or size * len(requests)
    l2_bytes = max(l1_bytes, max(offset + length for offset, length in extents))
    rng = numpy.random.default_rng(args.seed + iteration + args.warmup)
    # Every writer agrees on overlapping bytes, making final SSD contents
    # independent of executor scheduling. This is an IO/performance workload;
    # ordered overwrite behavior is verified by transfer regression tests.
    payloads = {
        extent: rng.integers(0, 256, size=extent[1], dtype=numpy.uint8)
        for extent in extents
    }
    owners: list[Any] = []
    sources: list[Any] = []
    for spans in requests:
        payload = numpy.concatenate(
            [payloads[(span["file_offset"], span["nbytes"])] for span in spans]
        )
        if source_kind == "cuda":
            sources.append(cupy.asarray(payload))
        else:
            owner = cupy.cuda.alloc_pinned_memory(size)
            owners.append(owner)
            array = numpy.frombuffer(owner, dtype=numpy.uint8, count=size)
            array[:] = payload
            sources.append(memoryview(array))
    cupy.cuda.get_current_stream().synchronize()
    path.unlink(missing_ok=True)
    layer = layer_type(
        str(path),
        l1_bytes=l1_bytes,
        l2_bytes=l2_bytes,
        io_workers=args.io_workers,
        bip_enabled=args.bip_enabled,
    )
    tasks: list[asyncio.Task[float]] = []
    gate = asyncio.Event()

    async def store_one(source: Any, spans: list[dict[str, Any]]) -> float:
        await gate.wait()
        started = time.perf_counter_ns()
        stored = await layer.store_bytes_grouped(source, spans)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if stored != size:
            raise AssertionError(f"short store: {stored} != {size}")
        return elapsed

    try:
        tasks = [
            asyncio.create_task(store_one(source, spans))
            for source, spans in zip(sources, requests, strict=True)
        ]
        await asyncio.sleep(0)
        native_before = counter.snapshot()
        stats_before = asdict(layer.stats)
        process_before = process_write_bytes()
        started = time.perf_counter_ns()
        gate.set()
        latencies = await asyncio.wait_for(asyncio.gather(*tasks), args.timeout)
        accepted = time.perf_counter_ns()
        l1_at_acceptance = layer.l1_bytes_used
        await asyncio.wait_for(layer.drain(), args.timeout)
        completed = time.perf_counter_ns()
        process_after = process_write_bytes()
        native_after = counter.snapshot()
        native_calls, native_bytes = (
            after - before
            for after, before in zip(native_after, native_before, strict=True)
        )
        stats = {
            key: value - stats_before[key] for key, value in asdict(layer.stats).items()
        }
        expected_bytes = size * len(requests)
        expected_writes = sum(len(spans) for spans in requests)
        if counter.enabled and (
            native_bytes != expected_bytes or native_calls != expected_writes
        ):
            raise AssertionError(
                f"write count mismatch: {(native_calls, native_bytes)} "
                f"!= {(expected_writes, expected_bytes)}"
            )
        result = {
            "acceptance_latency_ms": latencies,
            "acceptance_ms": (accepted - started) / 1e6,
            "drain_ms": (completed - accepted) / 1e6,
            "completion_ms": (completed - started) / 1e6,
            "requested_bytes": expected_bytes,
            "unique_stored_bytes": sum(length for _, length in extents),
            "native_write_calls": native_calls if counter.enabled else None,
            "native_write_bytes": native_bytes if counter.enabled else None,
            "process_write_bytes": counter_delta(process_after, process_before),
            "l1_bytes_at_acceptance": l1_at_acceptance,
            "l1_bytes_after_drain": layer.l1_bytes_used,
            **stats,
        }
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await asyncio.wait_for(layer.drain(), args.timeout)
        finally:
            layer.close()
    sources.clear()
    owners.clear()
    verifier = layer_type(
        str(path),
        l1_bytes=l1_bytes,
        l2_bytes=l2_bytes,
        io_workers=args.io_workers,
        bip_enabled=args.bip_enabled,
    )
    try:
        verified = 0
        for offset, length in extents:
            actual = bytearray(length)
            loaded = await asyncio.wait_for(
                verifier.load_bytes(actual, offset, length), args.timeout
            )
            if loaded != length or not numpy.array_equal(
                numpy.frombuffer(actual, dtype=numpy.uint8), payloads[(offset, length)]
            ):
                raise AssertionError(f"SSD readback mismatch at offset {offset}")
            verified += length
        if verifier.stats.l2_reads == 0:
            raise AssertionError("readback did not reach SSD")
        result.update(
            correctness=True,
            verified_bytes=verified,
            verification_l2_reads=verifier.stats.l2_reads,
        )
    finally:
        try:
            await asyncio.wait_for(verifier.drain(), args.timeout)
        finally:
            verifier.close()
    return result


async def benchmark(args: argparse.Namespace, result: dict[str, Any]) -> None:
    """Execute the matrix from args (Namespace), appending JSON rows to result (dict).

    Returns None. One event loop runs waves serially; each layer drains and closes
    before its ephemeral directory is deleted. Reports remain outside that directory.
    """
    import cupy
    import numpy

    layer_type, result["source"] = source_info(args.repo, args.expect_commit)
    cupy.cuda.Device(args.device).use()
    properties = cupy.cuda.runtime.getDeviceProperties(args.device)
    result["device"] = {
        "visible_ordinal": args.device,
        "name": properties["name"].decode(),
        "pci_bus_id": cupy.cuda.runtime.deviceGetPCIBusId(args.device),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_version": cupy.__version__,
    }
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    counter = NativeWriteCounter(args.native_counts)
    counter.install()
    try:
        with tempfile.TemporaryDirectory(
            prefix="transfer-store-", dir=args.scratch_root
        ) as scratch:
            path = Path(scratch) / "daser.store"
            for size in args.sizes:
                for concurrency in args.concurrency:
                    for storage_format in args.formats:
                        for case in args.cases:
                            for source_kind in args.sources:
                                row: dict[str, Any] = {
                                    "size_bytes": size,
                                    "concurrency": concurrency,
                                    "format": storage_format,
                                    "case": case,
                                    "source_kind": source_kind,
                                    "status": "running",
                                    "samples": [],
                                }
                                result["results"].append(row)
                                try:
                                    requests = build_requests(
                                        size,
                                        concurrency,
                                        case,
                                        storage_format,
                                        args.spans,
                                    )
                                except ValueError as error:
                                    row.update(status="skipped", reason=str(error))
                                    write_result(args.output, result)
                                    continue
                                row.update(
                                    spans_per_request=len(requests[0]),
                                    l1_bytes=args.l1_bytes or size * concurrency,
                                )
                                write_result(args.output, result)
                                try:
                                    for iteration in range(-args.warmup, args.repeat):
                                        sample = await run_wave(
                                            layer_type,
                                            cupy,
                                            numpy,
                                            path,
                                            requests,
                                            size,
                                            source_kind,
                                            args,
                                            counter,
                                            iteration,
                                        )
                                        if iteration >= 0:
                                            sample["iteration"] = iteration
                                            row["samples"].append(sample)
                                    row.update(
                                        status="ok",
                                        summary=summarize(row["samples"], numpy),
                                    )
                                except Exception as error:
                                    row.update(
                                        status="error",
                                        error=f"{type(error).__name__}: {error}",
                                    )
                                    raise
                                finally:
                                    write_result(args.output, result)
                                print(
                                    json.dumps(
                                        {
                                            key: row[key]
                                            for key in (
                                                "size_bytes",
                                                "concurrency",
                                                "format",
                                                "case",
                                                "source_kind",
                                                "status",
                                            )
                                        }
                                    ),
                                    flush=True,
                                )
    finally:
        counter.restore()
    result["complete"] = all(
        row["status"] in ("ok", "skipped") for row in result["results"]
    )


def main() -> int:
    """Run the CLI with no arguments; return exit code (int) after draining IO.

    Owns the main event loop and atomic output reporting, including failure state.
    """
    args = parse_args()
    result: dict[str, Any] = {
        "schema_version": 1,
        "variant": args.variant,
        "complete": False,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "results": [],
        "measurement": (
            "real CUDA/pinned source -> pinned L1 -> O_DIRECT io_uring; "
            "fresh-layer SSD readback untimed"
        ),
    }
    started = time.time()
    try:
        asyncio.run(benchmark(args, result))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        print(result["error"], file=sys.stderr, flush=True)
    finally:
        result["total_elapsed_s"] = time.time() - started
        write_result(args.output, result)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
