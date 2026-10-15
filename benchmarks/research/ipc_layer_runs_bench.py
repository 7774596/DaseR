# SPDX-License-Identifier: Apache-2.0

"""Two-process cost of layer-wise leased-run loads through the DaseR IPC.

A spawned server process fills an L1-only io_uring transfer layer with
``records`` packed-size records, leases them and serves IPC. The worker (this
process) registers two fixed CUDA staging buffers, then times record-major
``transfer_load`` batches against per-layer-group ``transfer_load_runs`` calls,
each issued serially (one client) and on a two-buffer ring (one client per
buffer, one call in flight per buffer). Prints one row per arm: wall time,
per-call ms, server-side copy ms and GB/s.
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
import os
import time
from typing import Any

import cupy
import numpy as np

_PAGE = 4096
_LEASE = "bench"


def _make_core() -> Any:
    """Return a minimal ServerCore; the bench never touches the index."""
    from daser.position.fixed_offset import FixedOffsetEncoder
    from daser.retrieval.prefix import PrefixHashIndex
    from daser.server.chunk_manager import ChunkManager
    from daser.server.core import ServerCore
    from daser.server.doc_registry import DocRegistry
    from daser.server.metadata_store import MetadataStore

    store = MetadataStore()
    manager = ChunkManager(
        total_slots=64, metadata_store=store, doc_registry=DocRegistry()
    )
    return ServerCore(
        chunk_manager=manager,
        retrieval_index=PrefixHashIndex(block_tokens=4),
        position_encoder=FixedOffsetEncoder(fixed_offset=0),
        slot_size=1024,
        block_tokens=4,
    )


async def _serve(args: argparse.Namespace, ready: Any, stop: Any) -> None:
    """Fill and lease L1, serve IPC until ``stop`` is set, then clean up."""
    from daser.server.ipc import IPCServer
    from daser.transfer.iouring.layer import TieredIOUringTransferLayer

    store = os.path.join(args.scratch, "ipc_runs.store")
    layer = TieredIOUringTransferLayer(
        path=store,
        l1_bytes=args.records * args.record_bytes + (1 << 30),
        l2_bytes=0,
        skip_l2=True,
        gather_min_runs=args.gather_min_runs,
    )
    server = IPCServer(
        os.path.join(args.scratch, "ipc_runs.sock"),
        _make_core(),
        {"transfer_mode": "iouring"},
    )
    # Bench harness: hand the server a pre-filled transfer layer instead of
    # populating it through lookups.
    server._transfer = layer  # noqa: SLF001
    try:
        payload = np.random.default_rng(0).integers(
            0, 256, args.record_bytes, dtype=np.uint8
        )
        spans = []
        for index in range(args.records):
            offset = index * args.record_bytes
            await layer.store_bytes(payload, offset, args.record_bytes)
            spans.append({"file_offset": offset, "nbytes": args.record_bytes})
        tier = await layer.classify_and_acquire_lease(_LEASE, spans)
        await server.start()
        ready.put(tier)
        while not stop.is_set():
            await asyncio.sleep(0.1)
    finally:
        await server.stop()
        await layer.release_lease(_LEASE)
        layer.close()
        if os.path.exists(store):
            os.remove(store)


def _server_main(args: argparse.Namespace, ready: Any, stop: Any) -> None:
    """Spawned server process entry point."""
    asyncio.run(_serve(args, ready, stop))


async def _register(client: Any, index: int, buffer: cupy.ndarray) -> None:
    """Register one CUDA staging buffer with the server."""
    from daser.transfer.cuda_ipc import (
        cuda_allocation_base_and_offset,
        cuda_array_pci_bus_id,
        cuda_array_pointer,
        export_cuda_ipc_handle,
    )

    ptr = cuda_array_pointer(buffer)
    base, offset = cuda_allocation_base_and_offset(ptr)
    await client.register_staging_cuda(
        direction="load",
        buffer_index=index,
        cuda_ipc_handle=export_cuda_ipc_handle(buffer),
        allocation_bytes=int(buffer.nbytes),
        device_pci_bus_id=cuda_array_pci_bus_id(buffer),
        device_ptr=ptr,
        allocation_base_ptr=base,
        allocation_offset=offset,
        producer_pid=os.getpid(),
    )


async def _timed(clients: list, calls: list, ring: int) -> tuple[float, float]:
    """Return (wall ms, summed server transfer ms) for one pass over calls."""
    server_ms = 0.0

    async def lane(lane_index: int) -> None:
        nonlocal server_ms
        client = clients[lane_index]
        for call in calls[lane_index::ring]:
            response = await call(client, lane_index)
            server_ms += float(response.get("transfer_ms", 0.0))

    cupy.cuda.Device().synchronize()
    start = time.perf_counter()
    await asyncio.gather(*(lane(i) for i in range(ring)))
    return (time.perf_counter() - start) * 1e3, server_ms


async def _run(args: argparse.Namespace, socket_path: str) -> None:
    """Register staging and time every arm."""
    from daser.connector.ipc_client import IPCClientAsync

    record = args.record_bytes
    plane = (record - _PAGE) // (2 * args.layers) // 16 * 16
    group_bytes = 2 * plane * args.group_layers
    groups = args.layers // args.group_layers
    slot = _PAGE + group_bytes + _PAGE
    if args.records * slot > args.staging_bytes:
        raise SystemExit("layer-major group does not fit the staging buffer")
    clients = [IPCClientAsync(socket_path) for _ in range(2)]
    staging = [cupy.empty(args.staging_bytes, dtype=cupy.uint8) for _ in range(2)]
    pid = os.getpid()
    try:
        for index, buffer in enumerate(staging):
            await _register(clients[index], index, buffer)
        per_call = args.staging_bytes // record

        def batch(first: int) -> Any:
            spans = [
                {
                    "target_offset": (i - first) * record,
                    "file_offset": i * record,
                    "nbytes": record,
                }
                for i in range(first, min(first + per_call, args.records))
            ]
            return lambda client, lane: client.transfer_load_registered_cuda(
                lane, pid, len(spans) * record, spans
            )

        index = np.arange(args.records, dtype=np.int64)
        targets = index * slot + _PAGE
        sizes = np.full(args.records, group_bytes, dtype=np.int64)

        def group(g: int) -> Any:
            files = index * record + _PAGE + g * group_bytes
            return lambda client, lane: client.transfer_load_runs(
                lane, pid, files, targets, sizes, _LEASE
            )

        record_major = [batch(first) for first in range(0, args.records, per_call)]
        layer_major = [group(g) for g in range(groups)]
        moved = {
            "record": args.records * record,
            "layer": args.records * group_bytes * groups,
        }
        for name, calls, kind in (
            ("record-major", record_major, "record"),
            ("layer-runs", layer_major, "layer"),
        ):
            for ring in (1, 2):
                await _timed(clients, calls, ring)
                wall, server_ms = min(
                    [await _timed(clients, calls, ring) for _ in range(3)]
                )
                print(
                    f"{name:12s} ring={ring} calls={len(calls):3d} "
                    f"wall={wall:7.1f} ms per_call={wall / len(calls):6.2f} ms "
                    f"server_copy={server_ms / len(calls):6.2f} ms "
                    f"{moved[kind] / wall / 1e6:5.1f} GB/s",
                    flush=True,
                )
    finally:
        for client in clients:
            await client.close()


def main() -> None:
    """Spawn the server, run the worker arms, and stop the server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--records", type=int, default=1024)
    parser.add_argument("--record-bytes", type=int, default=13910016)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--group-layers", type=int, default=1)
    parser.add_argument("--staging-bytes", type=int, default=1610612736)
    parser.add_argument("--gather-min-runs", type=int, default=0)
    args = parser.parse_args()
    context = mp.get_context("spawn")
    ready, stop = context.Queue(), context.Event()
    server = context.Process(target=_server_main, args=(args, ready, stop))
    server.start()
    try:
        tier = ready.get(timeout=600)
        if tier != "l1":
            raise SystemExit(f"bench lease is not all-L1: {tier}")
        asyncio.run(_run(args, os.path.join(args.scratch, "ipc_runs.sock")))
    finally:
        stop.set()
        server.join(timeout=120)
        if server.is_alive():
            server.terminate()


if __name__ == "__main__":
    main()
