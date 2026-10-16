# SPDX-License-Identifier: Apache-2.0
"""ServerCore builders and store helpers shared by server tests."""

# Standard
from pathlib import Path
from typing import Any

# First Party
from daser.metrics import MetricsRegistry
from daser.position.chunk_position import ChunkPositionEncoder
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.chunk_reuse import ChunkReuseIndex
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.core import Allocation, ServerCore, StoreRequest
from daser.server.doc_registry import DocRegistry
from daser.server.ipc import IPCServer
from daser.server.metadata_store import MetadataStore
from daser.server.store_file import ensure_store_file
from daser.server.tiers import L1, L2, StorePlacement, TierLayout

#: Tier grain used by the tests; one raw test record fills one grain.
GRAIN = 4096
RECORD = 4096
BLOCK_TOKENS = 4


def make_layout(
    l2_records: int = 64,
    l1_records: int = 0,
    tp: int = 1,
    l1_record_bytes: int = 0,
) -> TierLayout:
    """Return a layout whose lanes hold the given number of raw records."""
    return TierLayout(
        grain=GRAIN,
        l1_lane_bytes=l1_records * RECORD,
        l2_lane_bytes=l2_records * RECORD,
        tensor_parallel_size=tp,
        l1_record_bytes=l1_record_bytes,
    )


def make_core(
    l2_records: int = 64,
    l1_records: int = 0,
    *,
    tp: int = 1,
    chunk_mode: bool = False,
    registry: MetricsRegistry | None = None,
    layout: TierLayout | None = None,
) -> ServerCore:
    """Create a ServerCore over an empty chunk manager."""
    if chunk_mode:
        retrieval = ChunkReuseIndex(block_tokens=BLOCK_TOKENS)
        position = ChunkPositionEncoder(initial_offset=0)
    else:
        retrieval = PrefixHashIndex(block_tokens=BLOCK_TOKENS)
        position = FixedOffsetEncoder(fixed_offset=0)
    return ServerCore(
        chunk_manager=ChunkManager(MetadataStore(), DocRegistry()),
        retrieval_index=retrieval,
        position_encoder=position,
        tier_layout=layout or make_layout(l2_records, l1_records, tp),
        block_tokens=BLOCK_TOKENS,
        metrics_registry=registry or MetricsRegistry(),
    )


def request(alloc: Allocation, sizes: tuple[int, ...] | None = None) -> StoreRequest:
    """Return a store request for an allocation (raw records by default)."""
    return StoreRequest(
        chunk_key=alloc.chunk_key,
        start_slot=alloc.start_slot,
        num_slots=alloc.num_slots,
        record_sizes=sizes or (RECORD,) * alloc.num_slots,
    )


async def store_rank(
    core: ServerCore,
    alloc: Allocation,
    tp_rank: int = 0,
    sizes: tuple[int, ...] | None = None,
) -> StorePlacement | None:
    """Run one rank's complete store of an allocation, as the IPC path does."""
    (placement,) = await core.place_store([request(alloc, sizes)])
    if placement is None:
        return None
    tp_size = core.tier_layout.tensor_parallel_size
    if placement.l1 is not None:
        await core.store_staged(placement, tp_rank, tp_size)
    if placement.l2 is not None:
        await core.store_persisted(placement, tp_rank, tp_size)
    return placement


async def store_all_ranks(
    core: ServerCore, alloc: Allocation, sizes: tuple[int, ...] | None = None
) -> None:
    """Store an allocation on every TP rank (publishes the chunk)."""
    for rank in range(core.tier_layout.tensor_parallel_size):
        await store_rank(core, alloc, rank, sizes)


async def alloc_and_store(
    core: ServerCore,
    key: str,
    token_count: int = BLOCK_TOKENS,
    parent_key: str = "",
    model_id: str = "m",
) -> Allocation:
    """Allocate a chunk and store it on every rank."""
    alloc = await core.alloc_chunk(key, token_count, model_id, parent_key=parent_key)
    await store_all_ranks(core, alloc)
    return alloc


def ipc_runtime_config(
    tmp_path: Path,
    layout: TierLayout,
    *,
    transfer_mode: str = "iouring",
    staging_records: int = 4,
    **overrides: Any,
) -> dict[str, Any]:
    """Return an IPC runtime config matching ``layout`` (creates the store).

    Args:
        tmp_path: Scratch directory for the socket and the store file.
        layout: Core tier layout the transfer layer is sized from.
        transfer_mode: ``iouring`` or ``gds``.
        staging_records: Staging pool size in records when both tiers exist.
        overrides: Extra runtime-config entries.
    """
    tp = layout.tensor_parallel_size
    both = layout.enabled(L1) and layout.enabled(L2)
    store_path = ""
    if layout.enabled(L2):
        store_path = str(tmp_path / "daser.store")
        ensure_store_file(store_path, layout.l2_lane_bytes * tp)
    config: dict[str, Any] = {
        "socket_path": str(tmp_path / "ipc.sock"),
        "store_path": store_path,
        "slot_size": RECORD * tp,
        "local_slot_size": RECORD,
        "tensor_parallel_size": tp,
        "block_tokens": BLOCK_TOKENS,
        "model_id": "m",
        "transfer_mode": transfer_mode,
        "l1_enabled": layout.enabled(L1),
        "l2_enabled": layout.enabled(L2),
        "tier_grain": GRAIN,
        "l1_lane_bytes": layout.l1_lane_bytes,
        "l2_lane_bytes": layout.l2_lane_bytes,
        "staging_bytes": staging_records * RECORD if both else 0,
        "skip_l2": not layout.enabled(L2),
    }
    config.update(overrides)
    return config


def make_ipc_server(
    tmp_path: Path,
    l2_records: int = 64,
    l1_records: int = 16,
    *,
    core: ServerCore | None = None,
    registry: MetricsRegistry | None = None,
    **config: Any,
) -> IPCServer:
    """Return an IPC server over a fresh core (socket ``tmp_path/ipc.sock``).

    Args:
        tmp_path: Scratch directory for the socket and the store file.
        l2_records: L2 lane capacity in raw records (0 disables L2).
        l1_records: L1 lane capacity in raw records (0 disables L1).
        core: Core to serve instead of a fresh one.
        registry: Metrics registry shared by the core and the server.
        config: ``ipc_runtime_config`` keyword arguments.
    """
    registry = registry or MetricsRegistry()
    core = core or make_core(l2_records, l1_records, registry=registry)
    runtime_config = ipc_runtime_config(tmp_path, core.tier_layout, **config)
    return IPCServer(
        socket_path=runtime_config["socket_path"],
        core=core,
        runtime_config=runtime_config,
        metrics_registry=registry,
    )


def raw_span(
    chunk_key: str,
    start_slot: int,
    num_slots: int = 1,
    *,
    slot_index: int = 0,
    count: int | None = None,
    source_offset: int = 0,
) -> dict[str, Any]:
    """Return a raw store span of ``count`` whole records of one chunk."""
    count = num_slots - slot_index if count is None else count
    return {
        "source_offset": source_offset,
        "nbytes": count * RECORD,
        "chunk_key": chunk_key,
        "start_slot": start_slot,
        "num_slots": num_slots,
        "slot_index": slot_index,
        "packed": False,
    }


def load_span(
    start_slot: int, num_slots: int = 1, target_offset: int = 0
) -> dict[str, int]:
    """Return a raw load span of ``num_slots`` records."""
    return {
        "target_offset": target_offset,
        "start_slot": start_slot,
        "num_slots": num_slots,
        "nbytes": num_slots * RECORD,
    }
