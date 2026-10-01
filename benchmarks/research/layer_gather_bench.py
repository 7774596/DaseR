# SPDX-License-Identifier: Apache-2.0

"""Layer-group restore copy microbenchmark through ``copy_ops``.

Lays out cross-layer records in one pinned L1 pool and copies one layer group
(``group`` consecutive layers, K and V planes contiguous inside each record)
from every record into a packed GPU staging buffer, the access pattern of a
per-layer-group restore. Prints memcpyAsync-per-run versus zero-copy gather
throughput per group size.
"""

from __future__ import annotations

import argparse
import time

import cupy

from daser.transfer.iouring import copy_ops
from daser.transfer.iouring.pinned_pool import PinnedMemoryPool


def _best(dst: cupy.ndarray, chunks: list, gather: int) -> tuple[float, float]:
    """Return the best of three (host enqueue ms, enqueue-to-done ms)."""
    best = (float("inf"), float("inf"))
    for _ in range(3):
        cupy.cuda.Device().synchronize()
        start = time.perf_counter()
        copy_ops.copy_grouped_to_cuda_dst(dst, chunks, gather)
        enqueued = time.perf_counter()
        cupy.cuda.Device().synchronize()
        done = time.perf_counter()
        best = min(best, ((enqueued - start) * 1e3, (done - start) * 1e3))
    return best


def _plan_rows(source: object, records: int, args: argparse.Namespace) -> None:
    """Time ``LayerGatherPlan``: setup, per-launch host cost, pipelined GB/s."""
    block = args.layers * args.layer_bytes
    dst = cupy.empty(records * block, dtype=cupy.uint8)
    stream = cupy.cuda.Stream(non_blocking=True)
    start = time.perf_counter()
    plan = copy_ops.LayerGatherPlan(
        [source.ptr_at(i * args.record_bytes) for i in range(records)],
        [int(dst.data.ptr) + i * block for i in range(records)],
        int(stream.ptr),
        int(dst.device.id),
    )
    stream.synchronize()
    setup_ms = (time.perf_counter() - start) * 1e3
    print(f"plan records={records} setup_ms={setup_ms:.2f}")
    print("group launches host_us_per_launch pipelined_GBps")
    for group in (1, 2, 4):
        piece = group * args.layer_bytes
        launches = args.layers // group
        best_host, best_total = float("inf"), float("inf")
        for _ in range(3):
            stream.synchronize()
            start = time.perf_counter()
            for layer in range(launches):
                plan.copy(4096 + layer * piece, layer * piece, piece)
            enqueued = time.perf_counter()
            stream.synchronize()
            done = time.perf_counter()
            best_host = min(best_host, enqueued - start)
            best_total = min(best_total, done - start)
        nbytes = records * piece * launches
        print(
            f"{group:5d} {launches:8d} {best_host / launches * 1e6:18.1f} "
            f"{nbytes / best_total / 1e9:14.1f}"
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-gib", type=float, default=4.0)
    parser.add_argument("--record-bytes", type=int, default=1_777_664)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--layer-bytes", type=int, default=48 * 1024)
    args = parser.parse_args()
    pool_bytes = int(args.pool_gib * 2**30)
    records = pool_bytes // args.record_bytes
    pool = PinnedMemoryPool(pool_bytes)
    source = pool.allocate(records * args.record_bytes)
    assert source is not None
    print(f"records={records} record_bytes={args.record_bytes}")
    print(
        "group piece_kib MB_per_copy memcpy_host_ms memcpy_GBps "
        "gather_host_ms gather_GBps"
    )
    for group in (1, 2, 4, 12, args.layers):
        piece = group * args.layer_bytes
        chunks = [
            (i * piece, source, i * args.record_bytes + 4096, piece)
            for i in range(records)
        ]
        nbytes = records * piece
        dst = cupy.empty(nbytes, dtype=cupy.uint8)
        cells = [f"{group:5d} {piece // 1024:9d} {nbytes / 1e6:11.0f}"]
        for gather in (0, 2):
            host_ms, total_ms = _best(dst, chunks, gather)
            cells.append(f"{host_ms:14.2f} {nbytes / total_ms / 1e6:11.1f}")
        print(" ".join(cells))
        del dst
    _plan_rows(source, records, args)
    source.close()
    pool.close()


if __name__ == "__main__":
    main()
