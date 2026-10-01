# SPDX-License-Identifier: Apache-2.0

"""Server L1 load cost of record-major versus layer-major span lists.

Stores ``records`` packed-size records in an L1-only io_uring transfer layer
and loads them into one CUDA buffer either record-major (one span per record,
split into staging-sized calls) or layer-major (one call per layer group with
one span per record). Prints wall time, host span-resolution time and GB/s,
answering whether the existing span path can drive per-layer delivery.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

import cupy
import numpy as np

from daser.transfer.iouring.layer import TieredIOUringTransferLayer

_PAGE = 4096


async def _timed(
    layer: TieredIOUringTransferLayer, dst: cupy.ndarray, calls: list
) -> float:
    """Return wall ms to issue every grouped call and finish its copies."""
    cupy.cuda.Device().synchronize()
    start = time.perf_counter()
    for spans in calls:
        await layer.load_bytes_grouped(dst, spans)
    cupy.cuda.Device().synchronize()
    return (time.perf_counter() - start) * 1e3


async def _run(args: argparse.Namespace) -> None:
    """Fill L1, then time both span orders."""
    record = args.record_bytes
    plane = (record - _PAGE) // (2 * args.layers) // 16 * 16
    group_bytes = 2 * plane * args.group_layers
    groups = args.layers // args.group_layers
    store = os.path.join(args.scratch, "layer_span.store")
    layer = TieredIOUringTransferLayer(
        path=store,
        l1_bytes=args.records * record + (1 << 30),
        l2_bytes=0,
        skip_l2=True,
        gather_min_runs=args.gather_min_runs,
    )
    try:
        payload = np.random.default_rng(0).integers(0, 256, record, dtype=np.uint8)
        for index in range(args.records):
            await layer.store_bytes(payload, file_offset=index * record, nbytes=record)
        staging = cupy.empty(args.staging_bytes, dtype=cupy.uint8)
        per_call = args.staging_bytes // record
        record_major = [
            [
                {
                    "target_offset": (i - first) * record,
                    "file_offset": i * record,
                    "nbytes": record,
                }
                for i in range(first, min(first + per_call, args.records))
            ]
            for first in range(0, args.records, per_call)
        ]
        # Each record's group lands in its own slot behind a header page, as
        # a shared layer-major layout; records are adjacent in the file, so
        # the span coalescer cannot merge two records' groups.
        slot = _PAGE + group_bytes + _PAGE
        layer_major = [
            [
                {
                    "target_offset": i * slot + _PAGE,
                    "file_offset": i * record + _PAGE + g * group_bytes,
                    "nbytes": group_bytes,
                }
                for i in range(args.records)
            ]
            for g in range(groups)
        ]
        if args.records * slot > args.staging_bytes:
            raise SystemExit("layer-major group does not fit the staging buffer")
        total = args.records * record
        for name, calls, moved in (
            ("record-major", record_major, total),
            ("layer-major", layer_major, args.records * group_bytes * groups),
        ):
            await _timed(layer, staging, calls)
            best = min([await _timed(layer, staging, calls) for _ in range(3)])
            print(
                f"{name:12s} calls={len(calls):4d} spans/call={len(calls[0]):5d} "
                f"wall={best:8.1f} ms per_call={best / len(calls):7.2f} ms "
                f"{moved / best / 1e6:6.1f} GB/s"
            )
        lease = "bench"
        tier = await layer.classify_and_acquire_lease(
            lease,
            [
                {"file_offset": i * record, "nbytes": record}
                for i in range(args.records)
            ],
        )
        if tier != "l1":
            raise SystemExit(f"bench lease is not all-L1: {tier}")
        index = np.arange(args.records, dtype=np.int64)
        groups_runs = [
            (
                index * record + _PAGE + g * group_bytes,
                index * slot + _PAGE,
                np.full(args.records, group_bytes, dtype=np.int64),
            )
            for g in range(groups)
        ]

        async def leased() -> float:
            cupy.cuda.Device().synchronize()
            start = time.perf_counter()
            for files, targets, sizes in groups_runs:
                await layer.load_leased_runs(staging, files, targets, sizes, lease)
            cupy.cuda.Device().synchronize()
            return (time.perf_counter() - start) * 1e3

        await leased()
        best = min([await leased() for _ in range(3)])
        moved = args.records * group_bytes * groups
        print(
            f"{'leased-runs':12s} calls={groups:4d} spans/call={args.records:5d} "
            f"wall={best:8.1f} ms per_call={best / groups:7.2f} ms "
            f"{moved / best / 1e6:6.1f} GB/s"
        )
        await layer.release_lease(lease)
    finally:
        layer.close()
        if os.path.exists(store):
            os.remove(store)


def main() -> None:
    """Parse arguments and run the span-order comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--records", type=int, default=1024)
    parser.add_argument("--record-bytes", type=int, default=13910016)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--group-layers", type=int, default=1)
    parser.add_argument("--staging-bytes", type=int, default=1610612736)
    parser.add_argument("--gather-min-runs", type=int, default=0)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
