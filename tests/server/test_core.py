# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import random

# Third Party
import pytest

from daser.connector.helpers import (
    ROLLING_PREFIX_SEED,
    hash_tokens,
    rolling_prefix_key,
    rolling_prefix_keys,
)

# First Party
from daser.metrics import MetricsRegistry
from daser.position.chunk_position import ChunkPositionEncoder
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.chunk_reuse import ChunkReuseIndex
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager, SlotsExhausted
from daser.server.core import Allocation, ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.metadata_store import MetadataStore

SLOT_SIZE = 1024
# Packed records are 4 KiB aligned, so their lane uses 4 KiB raw slots.
PACKED_LANE_SLOT = 4096
BLOCK_TOKENS = 4


def first_rolling_key(tokens: list[int]) -> str:
    """Return the first rolling-prefix key for one test block."""
    return rolling_prefix_key(ROLLING_PREFIX_SEED, tokens[:BLOCK_TOKENS])


def place_packed(core: ServerCore, spans: list[dict]) -> list[dict]:
    """Place packed spans in rank 0's lane, as the IPC store path does first."""
    return core.packed_layout.compact(
        spans, core.chunk_manager, local_slot_size=PACKED_LANE_SLOT
    )


def make_core(total_slots: int = 64) -> ServerCore:
    """Create a ServerCore for tests."""
    store = MetadataStore()
    doc_registry = DocRegistry()
    cm = ChunkManager(
        total_slots=total_slots,
        metadata_store=store,
        doc_registry=doc_registry,
    )
    return ServerCore(
        chunk_manager=cm,
        retrieval_index=PrefixHashIndex(block_tokens=BLOCK_TOKENS),
        position_encoder=FixedOffsetEncoder(fixed_offset=0),
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
    )


def make_instrumented_core(
    registry: MetricsRegistry, total_slots: int = 64
) -> ServerCore:
    """Create a ServerCore with a per-test metrics registry."""
    store = MetadataStore()
    doc_registry = DocRegistry()
    cm = ChunkManager(
        total_slots=total_slots,
        metadata_store=store,
        doc_registry=doc_registry,
    )
    return ServerCore(
        chunk_manager=cm,
        retrieval_index=PrefixHashIndex(block_tokens=BLOCK_TOKENS),
        position_encoder=FixedOffsetEncoder(fixed_offset=0),
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
        metrics_registry=registry,
    )


def make_chunk_core(total_slots: int = 64) -> ServerCore:
    """Create a chunk-reuse ServerCore for tests."""
    store = MetadataStore()
    doc_registry = DocRegistry()
    cm = ChunkManager(
        total_slots=total_slots,
        metadata_store=store,
        doc_registry=doc_registry,
    )
    return ServerCore(
        chunk_manager=cm,
        retrieval_index=ChunkReuseIndex(block_tokens=BLOCK_TOKENS),
        position_encoder=ChunkPositionEncoder(initial_offset=0),
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
    )


@pytest.mark.asyncio
async def test_alloc_commit_lookup() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)

    alloc = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    assert alloc.file_offset == alloc.start_slot * SLOT_SIZE

    assert await core.lookup(tokens, "m") == []
    await core.commit_chunk(key)
    chunks = await core.lookup(tokens, "m")

    assert len(chunks) == 1
    assert chunks[0].chunk_key == key
    assert core.is_chunk_committed(key) is True


@pytest.mark.asyncio
async def test_online_lookup_retries_pending_commit() -> None:
    """Online lookup should bridge a store commit arriving mid-lookup."""
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")

    lookup_task = asyncio.create_task(core.lookup(tokens, "m", wait_for_pending=True))
    await asyncio.sleep(0.006)
    await core.commit_chunk(key)

    chunks = await lookup_task
    assert len(chunks) == 1
    assert chunks[0].chunk_key == key


@pytest.mark.asyncio
async def test_lookup_unless_pending_defers_without_waiting() -> None:
    """A pending continuation writer defers the lookup instead of sleeping."""
    registry = MetricsRegistry()
    core = make_instrumented_core(registry)
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")

    started = asyncio.get_running_loop().time()
    assert await core.lookup_unless_pending(tokens, "m") is None
    assert asyncio.get_running_loop().time() - started < 0.002
    assert "daser_cache_lookup_total" not in registry.render_prometheus()

    await core.commit_chunk(key)
    chunks = await core.lookup_unless_pending(tokens, "m")
    assert chunks is not None
    assert [chunk.chunk_key for chunk in chunks] == [key]


@pytest.mark.asyncio
async def test_lookup_unless_pending_ignores_writer_within_gpu_prefix() -> None:
    """A writer whose blocks the caller already computed does not defer."""
    tokens = list(range(1, 13))
    keys = rolling_prefix_keys(tokens, BLOCK_TOKENS)
    core = make_core()
    await core.alloc_chunk(keys[0], token_count=BLOCK_TOKENS, model_id="m")
    await core.commit_chunk(keys[0])
    await core.alloc_chunk(keys[1], token_count=BLOCK_TOKENS, model_id="m")

    # The pending block 1 lies inside the 8 GPU-computed tokens: no defer.
    chunks = await core.lookup_unless_pending(tokens, "m", num_computed_tokens=8)
    assert chunks is not None
    assert [chunk.chunk_key for chunk in chunks] == [keys[0]]
    # With only block 0 computed, the writer can still extend the hit.
    assert await core.lookup_unless_pending(tokens, "m", num_computed_tokens=4) is None


@pytest.mark.asyncio
async def test_lookup_unless_pending_resolves_unrelated_writer() -> None:
    """Another prompt's pending writer does not defer the lookup."""
    core = make_core()
    await core.alloc_chunk(first_rolling_key([5, 6, 7, 8]), token_count=4, model_id="m")

    assert await core.lookup_unless_pending([1, 2, 3, 4], "m") == []


@pytest.mark.asyncio
async def test_online_lookup_ignores_unrelated_pending_commit() -> None:
    """Online lookup should not wait for another prompt's pending writer."""
    core = make_core()
    target_tokens = [1, 2, 3, 4]
    unrelated_tokens = [5, 6, 7, 8]
    unrelated_key = first_rolling_key(unrelated_tokens)
    await core.alloc_chunk(unrelated_key, token_count=4, model_id="m")

    started = asyncio.get_running_loop().time()
    assert await core.lookup(target_tokens, "m", wait_for_pending=True) == []
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.02


@pytest.mark.asyncio
async def test_commit_records_publish_latency() -> None:
    """Commit should observe seconds from writer claim to publication."""
    registry = MetricsRegistry()
    core = make_instrumented_core(registry)
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)

    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await asyncio.sleep(0.01)
    await core.commit_chunk(key)

    rendered = registry.render_prometheus()
    assert "daser_cache_publish_seconds_count 1" in rendered
    assert 'daser_cache_publish_seconds_bucket{le="0.005"} 0' in rendered
    assert "daser_cache_pending_overlap_seconds" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("probe", [False, True])
async def test_publish_probe_records_early_lookup(
    probe: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the probe on, a lookup that beat its writer is timed to commit."""
    if probe:
        monkeypatch.setenv("DASER_PUBLISH_PROBE", "1")
    else:
        monkeypatch.delenv("DASER_PUBLISH_PROBE", raising=False)
    registry = MetricsRegistry()
    core = make_instrumented_core(registry)
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)

    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    assert await core.lookup(tokens, "m") == []
    assert await core.lookup(tokens, "m") == []
    await core.commit_chunk(key)

    rendered = registry.render_prometheus()
    assert ("daser_cache_pending_overlap_seconds_count 1" in rendered) is probe


@pytest.mark.asyncio
async def test_evicted_writer_drops_publish_timing() -> None:
    """An evicted pending writer must not leave publication timing behind."""
    registry = MetricsRegistry()
    core = make_instrumented_core(registry)
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)

    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.evict_chunk(key)
    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.commit_chunk(key)

    rendered = registry.render_prometheus()
    assert "daser_cache_publish_seconds_count 1" in rendered


@pytest.mark.asyncio
async def test_alloc_chunks_returns_contiguous_allocations() -> None:
    """ServerCore allocates multiple chunks in one batch."""
    core = make_core()
    tokens_a = [1, 2, 3, 4]
    tokens_b = [5, 6, 7, 8]
    key_a = first_rolling_key(tokens_a)
    key_b = first_rolling_key(tokens_b)

    allocs = await core.alloc_chunks(
        [
            {"chunk_key": key_a, "token_count": len(tokens_a)},
            {"chunk_key": key_b, "token_count": len(tokens_b)},
        ],
        model_id="m",
    )

    assert [alloc.chunk_key for alloc in allocs] == [key_a, key_b]
    assert [alloc.start_slot for alloc in allocs] == [0, 1]
    assert [alloc.file_offset for alloc in allocs] == [0, SLOT_SIZE]

    await core.commit_chunk(key_a)
    await core.commit_chunk(key_b)
    assert len(await core.lookup(tokens_a, "m")) == 1
    assert len(await core.lookup(tokens_b, "m")) == 1


@pytest.mark.asyncio
async def test_core_records_cache_lookup_and_commit_metrics() -> None:
    """ServerCore should publish cache lookup and commit counters."""
    registry = MetricsRegistry()
    core = make_instrumented_core(registry)
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)

    await core.lookup(tokens, "m")
    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.commit_chunk(key)
    await core.lookup(tokens, "m")

    rendered = registry.render_prometheus()

    assert 'daser_cache_lookup_total{result="miss"} 1.0' in rendered
    assert 'daser_cache_lookup_total{result="hit"} 1.0' in rendered
    assert "daser_cache_requested_tokens_total 8.0" in rendered
    assert "daser_cache_matched_tokens_total 4.0" in rendered
    assert "daser_cache_committed_chunks_total 1.0" in rendered
    assert f"daser_store_l2_bytes_capacity {float(64 * SLOT_SIZE)}" in rendered
    assert "daser_cache_prefix_reuse_tokens_count 1" in rendered


@pytest.mark.asyncio
async def test_core_records_external_prefix_cache_metrics() -> None:
    """ServerCore should publish vLLM-equivalent external prefix cache counters."""
    registry = MetricsRegistry()
    core = make_instrumented_core(registry)

    await core.record_external_prefix_cache(queries=12, hits=8)

    rendered = registry.render_prometheus()
    assert "daser_external_prefix_cache_queries_total 12.0" in rendered
    assert "daser_external_prefix_cache_hits_total 8.0" in rendered


@pytest.mark.asyncio
async def test_wait_for_committed_chunks_completes_after_commit() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)

    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    waiter = asyncio.create_task(core.wait_for_committed_chunks([key], timeout_s=1.0))
    await asyncio.sleep(0)

    assert waiter.done() is False
    await core.commit_chunk(key)
    await waiter


@pytest.mark.asyncio
async def test_wait_for_committed_chunks_times_out() -> None:
    core = make_core()

    with pytest.raises(TimeoutError):
        await core.wait_for_committed_chunks(["missing"], timeout_s=0.001)


@pytest.mark.asyncio
async def test_wait_for_pending_chunks_waits_for_tp_commit_quorum() -> None:
    core = make_core()
    await core.wait_for_pending_chunks(timeout_s=0.001)

    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.commit_chunk(key, tp_rank=0, tp_size=2)

    waiter = asyncio.create_task(core.wait_for_pending_chunks(timeout_s=1.0))
    await asyncio.sleep(0)
    assert waiter.done() is False

    await core.commit_chunk(key, tp_rank=1, tp_size=2)
    await waiter


@pytest.mark.asyncio
async def test_tensor_parallel_commit_waits_for_all_distinct_ranks() -> None:
    """A chunk becomes visible only after every declared TP rank commits."""
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")

    await core.commit_chunk(key, tp_rank=0, tp_size=2)
    await core.commit_chunk(key, tp_rank=0, tp_size=2)
    assert await core.lookup(tokens, "m") == []

    with pytest.raises(ValueError, match="inconsistent TP size"):
        await core.commit_chunk(key, tp_rank=1, tp_size=3)

    await core.commit_chunk(key, tp_rank=1, tp_size=2)
    assert len(await core.lookup(tokens, "m")) == 1


@pytest.mark.asyncio
async def test_store_range_commit_waits_for_complete_l1_coverage() -> None:
    """Server publication waits until split transfer ranges cover a chunk."""
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    alloc = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")

    assert (
        await core.record_store_ranges(
            [
                {
                    "chunk_key": key,
                    "file_offset": alloc.file_offset,
                    "nbytes": SLOT_SIZE // 2,
                    "start_slot": alloc.start_slot,
                    "num_slots": alloc.num_slots,
                }
            ],
            tp_rank=0,
            tp_size=1,
            local_slot_size=SLOT_SIZE,
            rank_stride_bytes=0,
        )
        == []
    )
    assert await core.lookup(tokens, "m") == []

    assert await core.record_store_ranges(
        [
            {
                "chunk_key": key,
                "file_offset": alloc.file_offset + SLOT_SIZE // 2,
                "nbytes": SLOT_SIZE // 2,
                "start_slot": alloc.start_slot,
                "num_slots": alloc.num_slots,
            }
        ],
        tp_rank=0,
        tp_size=1,
        local_slot_size=SLOT_SIZE,
        rank_stride_bytes=0,
    ) == [key]
    assert len(await core.lookup(tokens, "m")) == 1


@pytest.mark.asyncio
async def test_store_range_commit_preserves_tp_quorum() -> None:
    """Complete L1 coverage on one TP rank is not public until quorum."""
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    alloc = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    local_slot_size = SLOT_SIZE // 2
    rank_stride = 64 * local_slot_size

    rank_span = {
        "chunk_key": key,
        "start_slot": alloc.start_slot,
        "num_slots": alloc.num_slots,
    }
    assert await core.record_store_ranges(
        [
            {
                **rank_span,
                "file_offset": alloc.start_slot * local_slot_size,
                "nbytes": local_slot_size,
            }
        ],
        tp_rank=0,
        tp_size=2,
        local_slot_size=local_slot_size,
        rank_stride_bytes=rank_stride,
    ) == [key]
    assert await core.lookup(tokens, "m") == []

    assert await core.record_store_ranges(
        [
            {
                **rank_span,
                "file_offset": rank_stride + alloc.start_slot * local_slot_size,
                "nbytes": local_slot_size,
            }
        ],
        tp_rank=1,
        tp_size=2,
        local_slot_size=local_slot_size,
        rank_stride_bytes=rank_stride,
    ) == [key]
    assert len(await core.lookup(tokens, "m")) == 1


@pytest.mark.asyncio
async def test_online_packed_prefix_publication_uses_physical_slot() -> None:
    """Publish a prefix record when prompt and ring slots differ."""
    core = make_core()
    # Consume the first ring slot so the physical allocation is intentionally
    # different from the prompt-relative synthetic ``:store:7`` slot.
    await core.alloc_chunk(first_rolling_key([9, 10, 11, 12]), 4, "m")
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    alloc = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    assert alloc.num_slots == 1
    assert alloc.start_slot != 7

    ready = await core.record_store_ranges(
        place_packed(
            core,
            [
                {
                    "chunk_key": key,
                    "file_offset": 0,
                    "nbytes": 4096,
                    "start_slot": alloc.start_slot,
                    "num_slots": alloc.num_slots,
                    "logical_slot_start": 7,
                    "logical_slot_count": 1,
                    "packed": True,
                    "mode": "compressed",
                }
            ],
        ),
        tp_rank=0,
        tp_size=1,
        local_slot_size=SLOT_SIZE,
        rank_stride_bytes=0,
    )

    assert ready == [key]
    assert core.packed_slot_refs(7, 1) == []
    assert core.packed_slot_refs(alloc.start_slot, 1) == [
        {
            "slot_id": alloc.start_slot,
            "mode": "compressed",
            "file_offset": 0,
            "stored_length": 4096,
        }
    ]
    assert len(await core.lookup(tokens, "m")) == 1


@pytest.mark.asyncio
async def test_online_packed_publication_is_hidden_until_commit() -> None:
    """Incomplete packed writes cannot become reloadable publications."""
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    alloc = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")

    await core.record_store_ranges(
        place_packed(
            core,
            [
                {
                    "chunk_key": key,
                    "file_offset": 0,
                    "nbytes": 4096,
                    "start_slot": alloc.start_slot,
                    "num_slots": alloc.num_slots,
                    "logical_slot_start": alloc.start_slot,
                    "logical_slot_count": 1,
                    "packed": True,
                    "mode": "compressed",
                }
            ],
        ),
        tp_rank=0,
        tp_size=2,
        local_slot_size=SLOT_SIZE,
        rank_stride_bytes=SLOT_SIZE,
    )

    assert core.packed_slot_refs(alloc.start_slot, 1) == []


@pytest.mark.asyncio
async def test_online_packed_publication_clears_after_capacity_eviction() -> None:
    """Capacity eviction removes stale packed refs before slot reuse."""
    core = make_core(total_slots=1)
    first_tokens = [1, 2, 3, 4]
    first_key = first_rolling_key(first_tokens)
    first_alloc = await core.alloc_chunk(
        first_key, token_count=len(first_tokens), model_id="m"
    )
    await core.record_store_ranges(
        place_packed(
            core,
            [
                {
                    "chunk_key": first_key,
                    "file_offset": 0,
                    "nbytes": 4096,
                    "start_slot": first_alloc.start_slot,
                    "num_slots": first_alloc.num_slots,
                    "logical_slot_start": first_alloc.start_slot,
                    "logical_slot_count": 1,
                    "packed": True,
                    "mode": "compressed",
                }
            ],
        ),
        tp_rank=0,
        tp_size=1,
        local_slot_size=SLOT_SIZE,
        rank_stride_bytes=0,
    )
    assert core.packed_slot_refs(first_alloc.start_slot, 1)

    await core.alloc_chunk(first_rolling_key([5, 6, 7, 8]), token_count=4, model_id="m")

    assert core.packed_slot_refs(first_alloc.start_slot, 1) == []


@pytest.mark.asyncio
async def test_online_packed_publication_clears_after_explicit_eviction() -> None:
    """Explicit chunk eviction removes its packed reload references."""
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    alloc = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.record_store_ranges(
        place_packed(
            core,
            [
                {
                    "chunk_key": key,
                    "file_offset": 0,
                    "nbytes": 4096,
                    "start_slot": alloc.start_slot,
                    "num_slots": alloc.num_slots,
                    "logical_slot_start": alloc.start_slot,
                    "logical_slot_count": 1,
                    "packed": True,
                    "mode": "compressed",
                }
            ],
        ),
        tp_rank=0,
        tp_size=1,
        local_slot_size=SLOT_SIZE,
        rank_stride_bytes=0,
    )
    assert core.packed_slot_refs(alloc.start_slot, 1)

    await core.evict_chunk(key)

    assert core.packed_slot_refs(alloc.start_slot, 1) == []


@pytest.mark.asyncio
async def test_removal_listener_sees_capacity_and_explicit_evictions() -> None:
    """Listeners get every removed key; other chunks keep their packed refs."""
    core = make_core(total_slots=2)
    removed: list[str] = []
    core.add_chunk_removal_listener(removed.append)
    keys = [first_rolling_key([i, i + 1, i + 2, i + 3]) for i in (1, 5, 9)]
    allocs = []
    for key in keys[:2]:
        alloc = await core.alloc_chunk(key, token_count=4, model_id="m")
        allocs.append(alloc)
        await core.record_store_ranges(
            place_packed(
                core,
                [
                    {
                        "chunk_key": key,
                        "file_offset": alloc.start_slot * 4096,
                        "nbytes": 4096,
                        "start_slot": alloc.start_slot,
                        "num_slots": 1,
                        "logical_slot_start": alloc.start_slot,
                        "logical_slot_count": 1,
                        "packed": True,
                        "mode": "compressed",
                    }
                ],
            ),
            tp_rank=0,
            tp_size=1,
            local_slot_size=SLOT_SIZE,
            rank_stride_bytes=0,
        )

    await core.alloc_chunk(keys[2], token_count=4, model_id="m")  # evicts keys[0]
    assert removed == [keys[0]]
    assert core.packed_slot_refs(allocs[1].start_slot, 1)

    await core.evict_chunk(keys[1])
    assert removed == [keys[0], keys[1]]
    assert core.packed_slot_refs(allocs[1].start_slot, 1) == []


@pytest.mark.asyncio
async def test_restored_orphan_committed_chunk_can_be_reused(tmp_path) -> None:
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    original = make_core()
    await original.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await original.commit_chunk(key)
    original.chunk_manager.save(str(tmp_path / "daser.index"))

    store = MetadataStore()
    doc_registry = DocRegistry()
    cm = ChunkManager(
        total_slots=64,
        metadata_store=store,
        doc_registry=doc_registry,
    )
    cm.load(str(tmp_path / "daser.index"))
    restored = ServerCore(
        chunk_manager=cm,
        retrieval_index=PrefixHashIndex(block_tokens=BLOCK_TOKENS),
        position_encoder=FixedOffsetEncoder(fixed_offset=0),
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
    )

    await restored.rebuild_retrieval_index()

    chunks = await restored.lookup(tokens, "m")
    assert len(chunks) == 1
    assert chunks[0].chunk_key == key
    assert await restored.list_documents() == []


@pytest.mark.asyncio
async def test_match_and_alloc_is_idempotent_before_commit() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4, 5, 6]
    key = first_rolling_key(tokens)

    first = await core.match_and_alloc(tokens, key, "m")
    second = await core.match_and_alloc(tokens, key, "m")

    assert first.chunks == []
    assert first.alloc is not None
    assert second.alloc is not None
    assert first.alloc.chunk_key == key
    assert second.alloc.chunk_key == key
    assert second.alloc.start_slot == first.alloc.start_slot
    assert first.alloc.skipped is False
    assert second.alloc.skipped is True


@pytest.mark.asyncio
async def test_alloc_chunk_marks_committed_identical_chunk_as_skipped() -> None:
    """Allocating an already committed identical chunk should not schedule a store."""
    core = make_chunk_core()
    tokens = [1, 2, 3, 4]
    key = hash_tokens(tokens)

    first = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.commit_chunk(key)
    second = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")

    assert second.start_slot == first.start_slot
    assert second.file_offset == first.file_offset
    assert second.skipped is True


@pytest.mark.asyncio
async def test_alloc_chunk_marks_uncommitted_identical_chunk_as_skipped() -> None:
    """Allocating an in-flight identical chunk should not schedule another store."""
    core = make_chunk_core()
    tokens = [1, 2, 3, 4]
    key = hash_tokens(tokens)

    first = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    second = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")

    assert second.start_slot == first.start_slot
    assert second.file_offset == first.file_offset
    assert first.skipped is False
    assert second.skipped is True


@pytest.mark.asyncio
async def test_release_uncommitted_chunk_writer_allows_reallocation() -> None:
    """Canceled in-flight stores should release their duplicate-write claim."""
    core = make_chunk_core()
    tokens = [1, 2, 3, 4]
    key = hash_tokens(tokens)

    first = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    released = await core.release_chunk_writer(
        chunk_key=key,
        start_slot=first.start_slot,
        num_slots=first.num_slots,
    )
    second = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")

    assert released is True
    assert second.skipped is False
    assert second.start_slot == first.start_slot
    assert core.is_current_allocation(key, second.start_slot, second.num_slots)


@pytest.mark.asyncio
async def test_chunk_mode_lookup_returns_multiple_targeted_chunks() -> None:
    core = make_chunk_core()
    doc_a = [1, 2, 3, 4]
    sep = [90, 91, 92, 93]
    doc_b = [5, 6, 7, 8]
    task = [100, 101, 102, 103]
    key_a = hash_tokens(doc_a)
    key_b = hash_tokens(doc_b)

    for tokens in (doc_a, doc_b):
        key = hash_tokens(tokens)
        await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
        await core.commit_chunk(key)

    chunks = await core.lookup(doc_a + sep + doc_b + task, "m")

    assert [chunk.chunk_key for chunk in chunks] == [key_a, key_b]
    assert [chunk.target_token_start for chunk in chunks] == [0, 8]
    assert [chunk.pos_offset for chunk in chunks] == [0, 8]


@pytest.mark.asyncio
async def test_register_list_get_delete_document() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = hash_tokens(tokens)
    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.commit_chunk(key)

    registration = await core.register_document(
        doc_id="doc-1",
        title="first",
        chunk_keys=[key],
        token_count=len(tokens),
        tokens=tokens,
    )
    assert registration.chunk_count_cached == 1

    docs = await core.list_documents()
    assert len(docs) == 1
    assert docs[0].doc_id == "doc-1"
    assert docs[0].chunk_count_cached == 1

    doc = await core.get_document("doc-1")
    assert doc is not None
    assert doc.chunk_keys == [key]
    assert doc.tokens == tokens

    result = await core.delete_document("doc-1")
    assert result.chunks_evicted == 1
    assert await core.get_document("doc-1") is None


@pytest.mark.asyncio
async def test_evict_chunk_flips_document_cached_mask() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = hash_tokens(tokens)
    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.commit_chunk(key)
    await core.register_document(
        doc_id="doc-1",
        title="first",
        chunk_keys=[key],
        token_count=len(tokens),
    )

    await core.evict_chunk(key)

    doc = await core.get_document("doc-1")
    assert doc is not None
    assert doc.cached_mask == [False]
    assert doc.status == "evicted"


@pytest.mark.asyncio
async def test_auto_eviction_removes_lookup_and_updates_doc() -> None:
    core = make_core(total_slots=2)
    first = [1, 2, 3, 4]
    second = [5, 6, 7, 8]
    third = [9, 10, 11, 12]
    first_key = hash_tokens(first)

    for tokens in (first, second):
        key = hash_tokens(tokens)
        await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
        await core.commit_chunk(key)
    await core.register_document(
        doc_id="doc-1",
        title="first",
        chunk_keys=[first_key],
        token_count=len(first),
    )

    third_key = hash_tokens(third)
    await core.alloc_chunk(third_key, token_count=len(third), model_id="m")
    await core.commit_chunk(third_key)

    assert await core.lookup(first, "m") == []
    assert core.is_chunk_committed(first_key) is False
    doc = await core.get_document("doc-1")
    assert doc is not None
    assert doc.cached_mask == [False]


@pytest.mark.asyncio
async def test_capacity_eviction_skips_pending_writer() -> None:
    """A full store evicts committed chunks, never an uncommitted writer."""
    core = make_core(total_slots=2)
    first, second, third = [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]
    first_key = first_rolling_key(first)

    await core.alloc_chunk(first_key, token_count=len(first), model_id="m")
    for tokens in (second, third):
        key = first_rolling_key(tokens)
        await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
        await core.commit_chunk(key)
    await core.commit_chunk(first_key)

    assert len(await core.lookup(first, "m")) == 1
    assert await core.lookup(second, "m") == []
    assert len(await core.lookup(third, "m")) == 1


@pytest.mark.asyncio
async def test_alloc_fails_when_only_pending_writers_remain() -> None:
    """Allocation raises instead of evicting a chunk that is still being written."""
    core = make_core(total_slots=2)
    for tokens in ([1, 2, 3, 4], [5, 6, 7, 8]):
        await core.alloc_chunk(hash_tokens(tokens), token_count=4, model_id="m")
    with pytest.raises(SlotsExhausted):
        await core.alloc_chunk(hash_tokens([9, 10, 11, 12]), 4, "m")


@pytest.mark.asyncio
async def test_late_commit_after_explicit_eviction_is_ignored() -> None:
    core = make_core(total_slots=2)
    first = [1, 2, 3, 4]
    first_key = hash_tokens(first)

    await core.alloc_chunk(first_key, token_count=len(first), model_id="m")
    await core.evict_chunk(first_key)
    await core.commit_chunk(first_key)

    assert await core.lookup(first, "m") == []
    stats = await core.commit_stats()
    assert stats["commit_requests"] == 1
    assert stats["late_evicted_commits"] == 1


@pytest.mark.asyncio
async def test_is_current_allocation_rejects_evicted_or_reused_slot() -> None:
    """Delayed transfer writes are valid only for the current live chunk."""
    core = make_core(total_slots=1)
    first_key = hash_tokens([1, 2, 3, 4])
    first_alloc = await core.alloc_chunk(first_key, token_count=4, model_id="m")
    assert core.is_current_allocation(
        first_key, first_alloc.start_slot, first_alloc.num_slots
    )

    await core.evict_chunk(first_key)
    assert not core.is_current_allocation(
        first_key, first_alloc.start_slot, first_alloc.num_slots
    )
    second_key = hash_tokens([5, 6, 7, 8])
    second = await core.alloc_chunk(second_key, token_count=4, model_id="m")
    assert second.start_slot == first_alloc.start_slot
    assert not core.is_current_allocation(
        first_key, first_alloc.start_slot, first_alloc.num_slots
    )


@pytest.mark.asyncio
async def test_online_lookup_skips_wait_when_gap_has_no_writer() -> None:
    """A pending later block cannot fill an evicted earlier one."""
    core = make_core()
    tokens = list(range(3 * BLOCK_TOKENS))
    keys = rolling_prefix_keys(tokens, BLOCK_TOKENS)
    await core.alloc_chunk(keys[0], token_count=BLOCK_TOKENS, model_id="m")
    await core.commit_chunk(keys[0])
    # keys[1] was evicted and has no writer; keys[2] is still pending.
    await core.alloc_chunk(keys[2], token_count=BLOCK_TOKENS, model_id="m")

    started = asyncio.get_running_loop().time()
    chunks = await core.lookup(tokens, "m", wait_for_pending=True)
    elapsed = asyncio.get_running_loop().time() - started

    assert [chunk.chunk_key for chunk in chunks] == [keys[0]]
    assert elapsed < 0.02


@pytest.mark.asyncio
async def test_online_lookup_waits_for_pending_gap_block() -> None:
    """The writer of the first uncovered block still extends the match."""
    core = make_core()
    tokens = list(range(2 * BLOCK_TOKENS))
    keys = rolling_prefix_keys(tokens, BLOCK_TOKENS)
    await core.alloc_chunk(keys[0], token_count=BLOCK_TOKENS, model_id="m")
    await core.commit_chunk(keys[0])
    await core.alloc_chunk(keys[1], token_count=BLOCK_TOKENS, model_id="m")

    lookup_task = asyncio.create_task(core.lookup(tokens, "m", wait_for_pending=True))
    await asyncio.sleep(0.006)
    await core.commit_chunk(keys[1])

    chunks = await lookup_task
    assert sum(chunk.token_count for chunk in chunks) == 2 * BLOCK_TOKENS


def _place_record(core: ServerCore, key: str) -> dict:
    """Place one 4 KiB packed record for a committed single-slot chunk."""
    meta = core.chunk_manager.store.get(key)
    assert meta is not None
    (span,) = place_packed(
        core,
        [
            {
                "chunk_key": key,
                "file_offset": meta.start_slot * PACKED_LANE_SLOT,
                "nbytes": 4096,
                "start_slot": meta.start_slot,
                "num_slots": 1,
                "logical_slot_start": meta.start_slot,
                "logical_slot_count": 1,
                "packed": True,
                "mode": "compressed",
            }
        ],
    )
    return span


async def _committed_blocks_from(
    core: ServerCore, blocks: list[list[int]]
) -> list[list[int]]:
    """Commit single-block root chunks with one packed record each."""
    for tokens in blocks:
        key = first_rolling_key(tokens)
        await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
        await core.commit_chunk(key)
        _place_record(core, key)
    return blocks


async def _committed_blocks(core: ServerCore, count: int) -> list[list[int]]:
    """Commit ``count`` unrelated single-block chunks."""
    return await _committed_blocks_from(
        core, [[10 * i + j for j in range(BLOCK_TOKENS)] for i in range(count)]
    )


async def _committed_chain(
    core: ServerCore, blocks: int, start: int = 1000
) -> tuple[list[int], list[str]]:
    """Commit one prompt's ``blocks`` chained blocks with parent links."""
    tokens = list(range(start, start + blocks * BLOCK_TOKENS))
    keys = rolling_prefix_keys(tokens, BLOCK_TOKENS)
    parent = ""
    for key in keys:
        await core.alloc_chunk(
            key, token_count=BLOCK_TOKENS, model_id="m", parent_key=parent
        )
        await core.commit_chunk(key)
        _place_record(core, key)
        parent = key
    return tokens, keys


def _resident(core: ServerCore, keys: list[str]) -> list[bool]:
    return [core.chunk_manager.store.get(key) is not None for key in keys]


async def _hit_tokens(core: ServerCore, tokens: list[int]) -> int:
    return sum(chunk.token_count for chunk in await core.lookup(tokens, "m"))


@pytest.mark.asyncio
async def test_reclaim_packed_takes_the_least_recently_used_leaf() -> None:
    """Without an L1 tier, reclaim removes the coldest chunk; a hit protects."""
    core = make_core()
    blocks = await _committed_blocks(core, 3)
    assert len(await core.lookup(blocks[0], "m")) == 1

    assert await core.reclaim_packed()

    assert len(await core.lookup(blocks[0], "m")) == 1
    assert await core.lookup(blocks[1], "m") == []
    assert core.packed_layout.charged_bytes() == 2 * 4096


@pytest.mark.asyncio
async def test_reclaim_packed_keeps_l1_resident_victim_indexed() -> None:
    """A budget victim held by L1 stays indexed and leaves only the budget."""
    core = make_core()
    resident: set[int] = set()
    discarded: list[tuple[int, int]] = []
    core.attach_l1_tier(
        lambda offset, nbytes: offset in resident,
        lambda offset, nbytes: discarded.append((offset, nbytes)),
        lambda offset, nbytes: None,
    )
    blocks = await _committed_blocks(core, 3)
    first_key = first_rolling_key(blocks[0])
    (first,) = core.packed_layout.charged_records(first_key)
    resident.add(first.file_offset)

    assert await core.reclaim_packed()
    assert len(await core.lookup(blocks[0], "m")) == 1
    assert core.packed_layout.held_owner(first.file_offset, 4096) == first_key
    assert core.packed_layout.charged_bytes() == 2 * 4096
    assert core.packed_layout.held_bytes == 4096

    # The held chunk is skipped; the next victim is not resident and leaves.
    assert await core.reclaim_packed()
    assert await core.lookup(blocks[1], "m") == []
    assert discarded == []

    # Removing a held chunk drops its L1 copy before the bytes are reused.
    await core.evict_chunk(first_key)
    assert discarded == [(first.file_offset, 4096)]
    assert core.packed_layout.held_bytes == 0


@pytest.mark.asyncio
async def test_l1_release_of_a_held_record_removes_its_chunk() -> None:
    """When L1 evicts a held record its chunk leaves the index."""
    core = make_core()
    core.attach_l1_tier(lambda o, n: True, lambda o, n: None, lambda o, n: None)
    blocks = await _committed_blocks(core, 2)
    first_key = first_rolling_key(blocks[0])
    (first,) = core.packed_layout.charged_records(first_key)
    assert await core.reclaim_packed()

    core.release_held(first.file_offset, 8192)  # not the held range: ignored
    core.release_held(first.file_offset, 4096)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert await core.lookup(blocks[0], "m") == []
    assert len(await core.lookup(blocks[1], "m")) == 1
    assert core.packed_layout.held_bytes == 0


def _attach_l1(core: ServerCore, resident=None) -> dict[str, list]:
    """Attach a fake L1 tier; ``resident`` limits which offsets it holds."""
    calls: dict[str, list] = {"discarded": [], "released": []}
    core.attach_l1_tier(
        lambda offset, nbytes: resident is None or offset in resident,
        lambda offset, nbytes: calls["discarded"].append((offset, nbytes)),
        lambda offset, nbytes: calls["released"].append((offset, nbytes)),
    )
    return calls


def _offset(core: ServerCore, key: str) -> int:
    meta = core.chunk_manager.store.get(key)
    assert meta is not None
    return meta.start_slot * PACKED_LANE_SLOT


async def _settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_budget_reclaim_holds_a_chain_past_its_held_tail() -> None:
    """A held tail does not block its chain: the frontier moves up."""
    core = make_core()
    _attach_l1(core)
    tokens, keys = await _committed_chain(core, 3)
    for _ in keys:
        assert await core.reclaim_packed()
    assert not await core.reclaim_packed()
    assert core.packed_layout.charged_bytes() == 0
    assert core.packed_layout.held_bytes == 3 * 4096
    assert await _hit_tokens(core, tokens) == 3 * BLOCK_TOKENS


@pytest.mark.asyncio
async def test_extending_a_held_chain_charges_its_held_ancestors() -> None:
    """A child stored under held chunks charges them again, without IO."""
    core = make_core()
    calls = _attach_l1(core)
    _tokens, keys = await _committed_chain(core, 3)
    for _ in keys:
        assert await core.reclaim_packed()

    longer = list(range(1000, 1000 + 4 * BLOCK_TOKENS))
    child = rolling_prefix_keys(longer, BLOCK_TOKENS)[3]
    await core.alloc_chunk(
        child, token_count=BLOCK_TOKENS, model_id="m", parent_key=keys[2]
    )
    await core.commit_chunk(child)
    _place_record(core, child)

    assert sorted(calls["released"]) == sorted((_offset(core, k), 4096) for k in keys)
    assert core.packed_layout.held_bytes == 0
    assert core.packed_layout.charged_bytes() == 4 * 4096
    # The new tail is the frontier again; the chain shrinks from it.
    assert await core.reclaim_packed()
    assert core.packed_layout.held_owner(_offset(core, child), 4096) == child


@pytest.mark.asyncio
async def test_l1_release_inside_a_chain_recharges_instead_of_cutting_it() -> None:
    """L1 evicting a held chunk with children keeps the chain whole."""
    core = make_core()
    _attach_l1(core)
    tokens, keys = await _committed_chain(core, 3)
    for _ in keys:
        assert await core.reclaim_packed()

    core.release_held(_offset(core, keys[1]), 4096)
    await _settle()
    assert _resident(core, keys) == [True, True, True]
    assert await _hit_tokens(core, tokens) == 3 * BLOCK_TOKENS
    # keys[0] was recharged too, so the closure holds.
    assert core.packed_layout.charged_bytes() == 2 * 4096

    core.release_held(_offset(core, keys[2]), 4096)  # a leaf leaves
    await _settle()
    assert _resident(core, keys) == [True, True, False]


@pytest.mark.asyncio
async def test_failed_hold_removes_the_held_subtree_below_the_victim() -> None:
    """A frontier victim L1 cannot hold leaves with its unreachable subtree."""
    core = make_core()
    resident: set[int] = set()
    calls = _attach_l1(core, resident)
    _tokens, keys = await _committed_chain(core, 3)
    resident.add(_offset(core, keys[2]))
    tail_offset = _offset(core, keys[2])
    assert await core.reclaim_packed()  # holds the tail

    assert await core.reclaim_packed()  # middle is not resident
    assert _resident(core, keys) == [True, False, False]
    assert calls["discarded"] == [(tail_offset, 4096)]
    assert core.packed_layout.held_bytes == 0
    assert core.packed_layout.charged_bytes() == 4096


@pytest.mark.asyncio
async def test_budget_reclaim_skips_held_chunks_arena_reclaim_does_not() -> None:
    """The L2 budget only reclaims charged chunks; the arena takes any leaf."""
    core = make_core()
    core.attach_l1_tier(lambda o, n: True, lambda o, n: None, lambda o, n: None)
    blocks = await _committed_blocks(core, 2)
    assert await core.reclaim_packed()  # holds blocks[0]
    assert await core.reclaim_packed()  # holds blocks[1]
    assert not await core.reclaim_packed()
    assert core.packed_layout.charged_bytes() == 0

    pinned = [first_rolling_key(blocks[0])]
    core.pin(pinned)
    assert await core.reclaim_packed(free_address=True)
    assert await core.lookup(blocks[1], "m") == []
    assert not await core.reclaim_packed(free_address=True)
    core.unpin(pinned)
    assert await core.reclaim_packed(free_address=True)
    assert await core.lookup(blocks[0], "m") == []


@pytest.mark.asyncio
async def test_chain_evicts_from_its_cold_tail_and_keeps_its_root() -> None:
    """A prefix chain loses its last block first; lookup keeps the prefix."""
    core = make_core()
    tokens, keys = await _committed_chain(core, 3)
    other = await _committed_blocks(core, 1)
    # Touch the chain root only, as a lookup of the short prefix would.
    assert len(await core.lookup(tokens[:BLOCK_TOKENS], "m")) == 1
    assert len(await core.lookup(other[0], "m")) == 1

    assert await core.reclaim_packed()
    assert _resident(core, keys) == [True, True, False]
    assert await _hit_tokens(core, tokens) == 2 * BLOCK_TOKENS

    # That lookup refreshed the whole remaining chain, so other goes next,
    # then the chain's new tail.
    assert await core.reclaim_packed()
    assert await core.lookup(other[0], "m") == []
    assert await core.reclaim_packed()
    assert _resident(core, keys) == [True, False, False]


@pytest.mark.asyncio
async def test_slot_exhaustion_evicts_cold_leaves_not_chain_heads() -> None:
    """A full store frees a chain's tail before any of its ancestors."""
    core = make_core(total_slots=4)
    tokens, keys = await _committed_chain(core, 3)
    other = await _committed_blocks(core, 1)
    await core.lookup(tokens, "m")

    await _committed_blocks_from(core, [[7000 + j for j in range(BLOCK_TOKENS)]])

    assert _resident(core, keys) == [True, True, True]
    assert await core.lookup(other[0], "m") == []
    fresh = [[8000 + j for j in range(BLOCK_TOKENS)]]
    await _committed_blocks_from(core, fresh)
    assert _resident(core, keys) == [True, True, False]


@pytest.mark.asyncio
async def test_pinned_and_pending_chunks_are_never_reclaimed() -> None:
    """Pins and uncommitted writers are skipped, and so are their ancestors."""
    core = make_core()
    tokens = list(range(3 * BLOCK_TOKENS))
    keys = rolling_prefix_keys(tokens, BLOCK_TOKENS)
    parent = ""
    for key in keys:
        await core.alloc_chunk(key, BLOCK_TOKENS, "m", parent_key=parent)
        parent = key
    for key in keys[:2]:
        await core.commit_chunk(key)
        _place_record(core, key)
    loose = await _committed_blocks(core, 1)
    core.pin([first_rolling_key(loose[0])])

    # keys[2] is pending, so its ancestors are not leaves; loose is pinned.
    assert not await core.reclaim_packed()
    assert _resident(core, keys) == [True, True, True]

    await core.commit_chunk(keys[2])
    _place_record(core, keys[2])
    assert await core.reclaim_packed()
    assert _resident(core, keys) == [True, True, False]


@pytest.mark.asyncio
async def test_orphaned_blocks_are_reclaimed_before_reachable_ones() -> None:
    """Blocks below an evicted head are unreachable and leave first."""
    core = make_core()
    _, keys = await _committed_chain(core, 3)
    other = await _committed_blocks(core, 1)
    await core.lookup(other[0], "m")
    _tokens, newer = await _committed_chain(core, 1, start=5000)

    await core.evict_chunk(keys[0])
    assert await core.reclaim_packed()
    assert await core.reclaim_packed()

    assert _resident(core, keys) == [False, False, False]
    assert len(await core.lookup(other[0], "m")) == 1
    assert _resident(core, newer) == [True]


@pytest.mark.asyncio
async def test_restored_index_rebuilds_eviction_order(tmp_path) -> None:
    """A reloaded index evicts chain tails first, as before the restart."""
    original = make_core(total_slots=3)
    tokens, keys = await _committed_chain(original, 3)
    original.chunk_manager.save(str(tmp_path / "daser.index"))

    cm = ChunkManager(total_slots=3, metadata_store=MetadataStore())
    cm.load(str(tmp_path / "daser.index"))
    restored = ServerCore(
        chunk_manager=cm,
        retrieval_index=PrefixHashIndex(block_tokens=BLOCK_TOKENS),
        position_encoder=FixedOffsetEncoder(fixed_offset=0),
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
    )
    await restored.rebuild_retrieval_index()
    assert await _hit_tokens(restored, tokens) == 3 * BLOCK_TOKENS

    await restored.alloc_chunk(hash_tokens([7, 7, 7, 7]), 4, "m")
    assert _resident(restored, keys) == [True, True, False]
    assert await _hit_tokens(restored, tokens) == 2 * BLOCK_TOKENS


@pytest.mark.asyncio
async def test_store_range_ignores_write_to_a_reallocated_key() -> None:
    """A write that outlived its allocation is dropped, not fatal.

    A chunk can be removed while its store is in flight; a later request may
    then allocate the same key at another slot before the late write reports.
    That write must not publish the new allocation or fail.
    """
    core = make_core(total_slots=4)
    key = "reused"
    stale = await core.alloc_chunk(key, token_count=BLOCK_TOKENS, model_id="m")
    await core.evict_chunk(key)
    await core.alloc_chunk("filler", token_count=BLOCK_TOKENS, model_id="m")
    assert not core.is_current_allocation(key, stale.start_slot, stale.num_slots)
    fresh = await core.alloc_chunk(key, token_count=BLOCK_TOKENS, model_id="m")
    assert fresh.start_slot != stale.start_slot

    def span(alloc: Allocation) -> dict[str, object]:
        return {
            "chunk_key": key,
            "file_offset": alloc.file_offset,
            "nbytes": SLOT_SIZE,
            "start_slot": alloc.start_slot,
            "num_slots": alloc.num_slots,
        }

    geometry = {
        "tp_rank": 0,
        "tp_size": 1,
        "local_slot_size": SLOT_SIZE,
        "rank_stride_bytes": 0,
    }
    assert await core.record_store_ranges([span(stale)], **geometry) == []
    assert core.is_current_allocation(key, fresh.start_slot, fresh.num_slots)
    assert await core.record_store_ranges([span(fresh)], **geometry) == [key]


def _assert_held_closure(core: ServerCore) -> None:
    """Every chunk whose parent is held is itself held; records agree."""
    layout = core.packed_layout
    metas = {meta.chunk_key: meta for meta in core.chunk_manager.store.iter_chunks()}

    def held(key: str) -> bool:
        return bool(layout.held_records(key)) and not layout.charged_records(key)

    for key, meta in metas.items():
        assert not (layout.held_records(key) and layout.charged_records(key))
        if meta.parent_key in metas and held(meta.parent_key):
            assert held(key), (key, meta.parent_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(6))
async def test_held_closure_and_budget_hold_under_random_sessions(seed: int) -> None:
    """Random chain growth, hits, budget reclaim and L1 releases keep invariants.

    After every store the budget loop (as the IPC store path runs it) brings
    charged bytes back under the budget, and the held closure always holds.
    """
    rng = random.Random(seed)
    budget = 12 * 4096
    core = make_core(total_slots=512)
    resident: set[int] = set()
    _attach_l1(core, resident)
    sessions = [
        list(range(10_000 * s, 10_000 * s + 12 * BLOCK_TOKENS)) for s in range(6)
    ]
    lengths = [0] * len(sessions)
    for _step in range(150):
        s = rng.randrange(len(sessions))
        lengths[s] = min(lengths[s] + rng.randint(1, 3), 12)
        tokens = sessions[s][: lengths[s] * BLOCK_TOKENS]
        await core.lookup(tokens, "m")
        parent = ""
        for key in rolling_prefix_keys(tokens, BLOCK_TOKENS):
            if core.chunk_manager.store.get(key) is None:
                await core.alloc_chunk(
                    key, token_count=BLOCK_TOKENS, model_id="m", parent_key=parent
                )
                await core.commit_chunk(key)
                _place_record(core, key)
                if rng.random() < 0.7:
                    resident.add(_offset(core, key))
            parent = key
        while core.packed_layout.charged_bytes() > budget:
            assert await core.reclaim_packed()
        _assert_held_closure(core)
        # L1 evicts a random held record now and then.
        held = [
            record
            for meta in core.chunk_manager.store.iter_chunks()
            for record in core.packed_layout.held_records(meta.chunk_key)
        ]
        if held and rng.random() < 0.3:
            record = rng.choice(held)
            core.release_held(record.file_offset, record.stored_length)
            await _settle()
            _assert_held_closure(core)
