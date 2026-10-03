# SPDX-License-Identifier: Apache-2.0

"""L2 promotion cost of filling records in byte phases instead of whole.

Writes ``records`` records to an L2 store, then for each phase count K
promotes them into an empty L1 tier phase by phase: phase k reads bytes
``[k/K, (k+1)/K)`` of every record (4 KiB aligned). A layer-major record
keeps its layer groups in file order, so phase k holds roughly layer group
k/K of every record and a streaming admission could start layer 0 after
phase 0. Prints the time to the first phase, the total, and GB/s, answering
whether phase-ordered SSD reads keep the bandwidth of whole-record reads.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

import numpy as np

from daser.transfer.iouring.layer import TieredIOUringTransferLayer

_PAGE = 4096
_WRITE_CHUNK = 64 << 20


def _phase_bounds(record: int, phases: int) -> list[tuple[int, int]]:
    """Return 4 KiB aligned ``[lo, hi)`` byte ranges splitting one record."""
    cuts = [record * k // phases // _PAGE * _PAGE for k in range(phases)]
    cuts.append(record)
    return [(cuts[k], cuts[k + 1]) for k in range(phases) if cuts[k + 1] > cuts[k]]


def _write_store(path: str, total: int) -> None:
    """Write ``total`` incompressible bytes so reads are not sparse holes."""
    block = np.random.default_rng(0).integers(0, 256, _WRITE_CHUNK, dtype=np.uint8)
    with open(path, "wb") as f:
        written = 0
        while written < total:
            step = min(_WRITE_CHUNK, total - written)
            f.write(block[:step].tobytes())
            written += step
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


async def _fill(args: argparse.Namespace, store: str, phases: int) -> str:
    """Promote every record into a fresh L1 in ``phases`` phases."""
    record = args.record_bytes
    total = args.records * record
    l2_bytes = (total + (1 << 30)) // _PAGE * _PAGE
    layer = TieredIOUringTransferLayer(
        path=store,
        l1_bytes=total + (1 << 30),
        l2_bytes=l2_bytes,
        io_workers=args.io_workers,
        l2_read_stripe_bytes=args.l2_read_stripe,
    )
    try:
        marks: list[float] = []
        start = time.perf_counter()
        for lo, hi in _phase_bounds(record, phases):
            await layer.prefetch_bytes_grouped(
                [
                    {"file_offset": i * record + lo, "nbytes": hi - lo}
                    for i in range(args.records)
                ]
            )
            marks.append((time.perf_counter() - start) * 1e3)
    finally:
        layer.close()
    wall = marks[-1]
    return (
        f"K={phases:2d} piece={record // phases / 1e6:6.2f} MB "
        f"first={marks[0]:8.1f} ms total={wall:8.1f} ms "
        f"{total / wall / 1e6:6.1f} GB/s"
    )


def main() -> None:
    """Run the phase-fill benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--records", type=int, default=1024)
    parser.add_argument("--record-bytes", type=int, default=13910016)
    parser.add_argument("--phases", default="1,4,8")
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--l2-read-stripe", type=int, default=16 << 20)
    args = parser.parse_args()
    if args.record_bytes % _PAGE:
        raise SystemExit("--record-bytes must be 4 KiB aligned")
    os.makedirs(args.scratch, exist_ok=True)
    store = os.path.join(args.scratch, "phase_fill.store")
    try:
        _write_store(store, args.records * args.record_bytes)
        for phases in (int(k) for k in args.phases.split(",")):
            print(asyncio.run(_fill(args, store, phases)), flush=True)
    finally:
        if os.path.exists(store):
            os.remove(store)


if __name__ == "__main__":
    main()
