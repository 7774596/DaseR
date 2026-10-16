# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from pathlib import Path

# Third Party
import pytest

# First Party
from daser.connector.helpers import (
    ROLLING_PREFIX_SEED,
    hash_tokens,
    rolling_prefix_key,
    rolling_prefix_keys,
)
from daser.metrics import MetricsRegistry
from daser.server.core import ChunkRecords, ServerCore, StoreRequest
from daser.server.tiers import L1, L2, ChunkUnavailable, TierFull
from tests.server.core_helpers import (
    BLOCK_TOKENS,
    RECORD,
    alloc_and_store,
    make_core,
    make_layout,
    request,
    store_all_ranks,
    store_rank,
)


def first_rolling_key(tokens: list[int]) -> str:
    """Return the first rolling-prefix key for one test block."""
    return rolling_prefix_key(ROLLING_PREFIX_SEED, tokens[:BLOCK_TOKENS])


async def committed_chain(
    core: ServerCore, blocks: int, start: int = 0
) -> tuple[list[int], list[str]]:
    """Store a prefix chain of ``blocks`` single-block chunks."""
    tokens = list(range(start, start + blocks * BLOCK_TOKENS))
    keys = rolling_prefix_keys(tokens, BLOCK_TOKENS)
    parent = ""
    for key in keys:
        await alloc_and_store(core, key, parent_key=parent)
        parent = key
    return tokens, keys


async def fresh_root(core: ServerCore, seed: int) -> list[int]:
    """Store one unrelated single-block root chunk; return its tokens."""
    tokens = [seed * 100 + j for j in range(BLOCK_TOKENS)]
    await alloc_and_store(core, first_rolling_key(tokens))
    return tokens


async def hit_tokens(core: ServerCore, tokens: list[int]) -> int:
    """Return how many prompt tokens a lookup reuses."""
    return sum(chunk.token_count for chunk in await core.lookup(tokens, "m"))


def resident(core: ServerCore, keys: list[str]) -> list[bool]:
    """Return which chunks are still committed."""
    return [core.is_chunk_committed(key) for key in keys]


# ----------------------------------------------------------------------
# Lookup and publication
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alloc_store_lookup() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)

    alloc = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    assert "file_offset" not in alloc.to_dict()
    assert await core.lookup(tokens, "m") == []
    await store_all_ranks(core, alloc)

    chunks = await core.lookup(tokens, "m")
    assert [chunk.chunk_key for chunk in chunks] == [key]
    assert core.is_chunk_committed(key) is True


@pytest.mark.asyncio
async def test_online_lookup_retries_pending_commit() -> None:
    """Online lookup should bridge a store commit arriving mid-lookup."""
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    alloc = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")

    lookup_task = asyncio.create_task(core.lookup(tokens, "m", wait_for_pending=True))
    await asyncio.sleep(0.006)
    await store_all_ranks(core, alloc)

    chunks = await lookup_task
    assert [chunk.chunk_key for chunk in chunks] == [key]


@pytest.mark.asyncio
async def test_lookup_unless_pending_defers_without_waiting() -> None:
    """A pending continuation writer defers the lookup instead of sleeping."""
    registry = MetricsRegistry()
    core = make_core(registry=registry)
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    alloc = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")

    started = asyncio.get_running_loop().time()
    assert await core.lookup_unless_pending(tokens, "m") is None
    assert asyncio.get_running_loop().time() - started < 0.002
    assert "daser_cache_lookup_total" not in registry.render_prometheus()

    await store_all_ranks(core, alloc)
    chunks = await core.lookup_unless_pending(tokens, "m")
    assert chunks is not None
    assert [chunk.chunk_key for chunk in chunks] == [key]


@pytest.mark.asyncio
async def test_lookup_unless_pending_ignores_writer_within_gpu_prefix() -> None:
    """A writer whose blocks the caller already computed does not defer."""
    tokens = list(range(1, 13))
    keys = rolling_prefix_keys(tokens, BLOCK_TOKENS)
    core = make_core()
    await alloc_and_store(core, keys[0])
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
    await core.alloc_chunk(first_rolling_key([5, 6, 7, 8]), token_count=4, model_id="m")

    started = asyncio.get_running_loop().time()
    assert await core.lookup([1, 2, 3, 4], "m", wait_for_pending=True) == []
    assert asyncio.get_running_loop().time() - started < 0.02


@pytest.mark.asyncio
async def test_commit_records_publish_latency() -> None:
    """Commit should observe seconds from writer claim to publication."""
    registry = MetricsRegistry()
    core = make_core(registry=registry)
    tokens = [1, 2, 3, 4]
    alloc = await core.alloc_chunk(first_rolling_key(tokens), 4, "m")
    await asyncio.sleep(0.01)
    await store_all_ranks(core, alloc)

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
    core = make_core(registry=registry)
    tokens = [1, 2, 3, 4]
    alloc = await core.alloc_chunk(first_rolling_key(tokens), 4, "m")
    assert await core.lookup(tokens, "m") == []
    assert await core.lookup(tokens, "m") == []
    await store_all_ranks(core, alloc)

    rendered = registry.render_prometheus()
    assert ("daser_cache_pending_overlap_seconds_count 1" in rendered) is probe


@pytest.mark.asyncio
async def test_evicted_writer_drops_publish_timing() -> None:
    """An evicted pending writer must not leave publication timing behind."""
    registry = MetricsRegistry()
    core = make_core(registry=registry)
    key = first_rolling_key([1, 2, 3, 4])
    await core.alloc_chunk(key, 4, "m")
    await core.evict_chunk(key)
    await alloc_and_store(core, key)

    assert "daser_cache_publish_seconds_count 1" in registry.render_prometheus()


@pytest.mark.asyncio
async def test_alloc_chunks_assigns_fresh_slot_ids() -> None:
    """A batch gets consecutive ids; ids of removed chunks are never reused."""
    core = make_core()
    key_a = first_rolling_key([1, 2, 3, 4])
    key_b = first_rolling_key([5, 6, 7, 8])
    allocs = await core.alloc_chunks(
        [
            {"chunk_key": key_a, "token_count": 4},
            {"chunk_key": key_b, "token_count": 8},
        ],
        model_id="m",
    )
    assert [(a.start_slot, a.num_slots) for a in allocs] == [(0, 1), (1, 2)]
    for alloc in allocs:
        await store_all_ranks(core, alloc)
    assert len(await core.lookup([1, 2, 3, 4], "m")) == 1

    await core.evict_chunk(key_a)
    again = await core.alloc_chunk(key_a, 4, "m")
    assert again.start_slot == 3
    assert not core.is_current_allocation(key_a, 0, 1)


@pytest.mark.asyncio
async def test_core_records_cache_and_tier_metrics() -> None:
    """ServerCore publishes lookup, commit and tier gauges."""
    registry = MetricsRegistry()
    core = make_core(l2_records=8, registry=registry)
    tokens = [1, 2, 3, 4]
    await core.lookup(tokens, "m")
    await alloc_and_store(core, first_rolling_key(tokens))
    await core.lookup(tokens, "m")

    rendered = registry.render_prometheus()
    assert 'daser_cache_lookup_total{result="miss"} 1.0' in rendered
    assert 'daser_cache_lookup_total{result="hit"} 1.0' in rendered
    assert "daser_cache_requested_tokens_total 8.0" in rendered
    assert "daser_cache_matched_tokens_total 4.0" in rendered
    assert "daser_cache_committed_chunks_total 1.0" in rendered
    assert "daser_cache_prefix_reuse_tokens_count 1" in rendered
    assert f'daser_tier_bytes_capacity{{tier="l2"}} {float(8 * RECORD)}' in rendered
    assert f'daser_tier_bytes_used{{tier="l2"}} {float(RECORD)}' in rendered


@pytest.mark.asyncio
async def test_core_records_external_prefix_cache_metrics() -> None:
    registry = MetricsRegistry()
    core = make_core(registry=registry)
    await core.record_external_prefix_cache(queries=12, hits=8)

    rendered = registry.render_prometheus()
    assert "daser_external_prefix_cache_queries_total 12.0" in rendered
    assert "daser_external_prefix_cache_hits_total 8.0" in rendered


@pytest.mark.asyncio
async def test_wait_for_committed_chunks_completes_after_store() -> None:
    core = make_core()
    key = first_rolling_key([1, 2, 3, 4])
    alloc = await core.alloc_chunk(key, 4, "m")
    waiter = asyncio.create_task(core.wait_for_committed_chunks([key], timeout_s=1.0))
    await asyncio.sleep(0)
    assert waiter.done() is False
    await store_all_ranks(core, alloc)
    await waiter


@pytest.mark.asyncio
async def test_wait_for_committed_chunks_times_out() -> None:
    core = make_core()
    with pytest.raises(TimeoutError):
        await core.wait_for_committed_chunks(["missing"], timeout_s=0.001)


@pytest.mark.asyncio
async def test_tensor_parallel_store_waits_for_every_rank() -> None:
    """A chunk publishes only after every TP rank stored it."""
    core = make_core(tp=2)
    await core.wait_for_pending_chunks(timeout_s=0.001)
    tokens = [1, 2, 3, 4]
    alloc = await core.alloc_chunk(first_rolling_key(tokens), 4, "m")
    await store_rank(core, alloc, tp_rank=0)
    await store_rank(core, alloc, tp_rank=0)
    assert await core.lookup(tokens, "m") == []

    waiter = asyncio.create_task(core.wait_for_pending_chunks(timeout_s=1.0))
    await asyncio.sleep(0)
    assert waiter.done() is False
    with pytest.raises(ValueError, match="invalid TP rank"):
        await store_rank(core, alloc, tp_rank=2)
    await store_rank(core, alloc, tp_rank=1)
    await waiter
    assert len(await core.lookup(tokens, "m")) == 1
    # Every rank wrote its own lane of one shared extent.
    stats = core.tier_stats()[L2]
    assert stats.used == RECORD and stats.copies == 1


@pytest.mark.asyncio
async def test_match_and_alloc_is_idempotent_before_commit() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4, 5, 6]
    key = first_rolling_key(tokens)
    first = await core.match_and_alloc(tokens, key, "m")
    second = await core.match_and_alloc(tokens, key, "m")

    assert first.chunks == []
    assert first.alloc is not None and second.alloc is not None
    assert second.alloc.start_slot == first.alloc.start_slot
    assert first.alloc.skipped is False
    assert second.alloc.skipped is True


@pytest.mark.asyncio
async def test_alloc_of_an_owned_chunk_is_skipped() -> None:
    """Allocating an in-flight or committed chunk schedules no second store."""
    core = make_core(chunk_mode=True)
    key = hash_tokens([1, 2, 3, 4])
    first = await core.alloc_chunk(key, 4, "m")
    pending = await core.alloc_chunk(key, 4, "m")
    assert pending.skipped is True and pending.start_slot == first.start_slot
    await store_all_ranks(core, first)
    committed = await core.alloc_chunk(key, 4, "m")
    assert committed.skipped is True and committed.start_slot == first.start_slot


@pytest.mark.asyncio
async def test_release_of_an_unstored_writer_removes_the_chunk() -> None:
    """A canceled store frees the key: the next writer gets new slot ids."""
    core = make_core(chunk_mode=True)
    key = hash_tokens([1, 2, 3, 4])
    first = await core.alloc_chunk(key, 4, "m")
    assert await core.release_chunk_writer(key, first.start_slot, first.num_slots)
    assert not core.is_current_allocation(key, first.start_slot, first.num_slots)

    second = await core.alloc_chunk(key, 4, "m")
    assert second.skipped is False
    assert second.start_slot != first.start_slot
    assert not await core.release_chunk_writer(key, first.start_slot, 1)
    await store_all_ranks(core, second)
    assert not await core.release_chunk_writer(key, second.start_slot, 1)
    assert core.is_chunk_committed(key)


@pytest.mark.asyncio
async def test_chunk_mode_lookup_returns_multiple_targeted_chunks() -> None:
    core = make_core(chunk_mode=True)
    doc_a, sep, doc_b, task = [1, 2, 3, 4], [90, 91, 92, 93], [5, 6, 7, 8], [100] * 4
    for tokens in (doc_a, doc_b):
        await alloc_and_store(core, hash_tokens(tokens))

    chunks = await core.lookup(doc_a + sep + doc_b + task, "m")
    assert [c.chunk_key for c in chunks] == [hash_tokens(doc_a), hash_tokens(doc_b)]
    assert [c.target_token_start for c in chunks] == [0, 8]
    assert [c.pos_offset for c in chunks] == [0, 8]


# ----------------------------------------------------------------------
# Documents
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_list_get_delete_document() -> None:
    registry = MetricsRegistry()
    core = make_core(registry=registry)
    tokens = [1, 2, 3, 4]
    key = hash_tokens(tokens)
    await alloc_and_store(core, key)

    registration = await core.register_document(
        doc_id="doc-1",
        title="first",
        chunk_keys=[key],
        token_count=len(tokens),
        tokens=tokens,
    )
    assert registration.chunk_count_cached == 1
    docs = await core.list_documents()
    assert [doc.doc_id for doc in docs] == ["doc-1"]
    assert docs[0].chunk_count_cached == 1
    doc = await core.get_document("doc-1")
    assert doc is not None and doc.chunk_keys == [key] and doc.tokens == tokens

    result = await core.delete_document("doc-1")
    assert result.chunks_evicted == 1
    assert await core.get_document("doc-1") is None
    assert core.tier_stats()[L2].used == 0
    assert 'daser_tier_bytes_used{tier="l2"} 0' in registry.render_prometheus()


@pytest.mark.asyncio
async def test_evict_chunk_flips_document_cached_mask() -> None:
    core = make_core()
    key = hash_tokens([1, 2, 3, 4])
    await alloc_and_store(core, key)
    await core.register_document(
        doc_id="doc-1", title="first", chunk_keys=[key], token_count=4
    )
    await core.evict_chunk(key)

    doc = await core.get_document("doc-1")
    assert doc is not None
    assert doc.cached_mask == [False]
    assert doc.status == "evicted"


# ----------------------------------------------------------------------
# Capacity eviction
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capacity_eviction_removes_lookup_and_updates_doc() -> None:
    registry = MetricsRegistry()
    core = make_core(l2_records=2, chunk_mode=True, registry=registry)
    first, second, third = [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]
    for tokens in (first, second):
        await alloc_and_store(core, hash_tokens(tokens))
    await core.register_document(
        doc_id="doc-1", title="first", chunk_keys=[hash_tokens(first)], token_count=4
    )
    await alloc_and_store(core, hash_tokens(third))

    assert await core.lookup(first, "m") == []
    assert core.is_chunk_committed(hash_tokens(first)) is False
    assert core.chunk_manager.store.get(hash_tokens(first)) is None
    doc = await core.get_document("doc-1")
    assert doc is not None and doc.cached_mask == [False]
    rendered = registry.render_prometheus()
    assert 'daser_cache_evicted_chunks_total{reason="capacity"} 1.0' in rendered
    assert 'daser_tier_evicted_copies_total{tier="l2"} 1.0' in rendered


@pytest.mark.asyncio
async def test_unstored_writer_takes_no_space_and_is_never_evicted() -> None:
    core = make_core(l2_records=2)
    first, second, third = [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]
    pending = await core.alloc_chunk(first_rolling_key(first), 4, "m")
    for tokens in (second, third):
        await alloc_and_store(core, first_rolling_key(tokens))
    await store_all_ranks(core, pending)

    assert len(await core.lookup(first, "m")) == 1
    assert await core.lookup(second, "m") == []
    assert len(await core.lookup(third, "m")) == 1


@pytest.mark.asyncio
async def test_store_fails_when_only_partial_copies_remain() -> None:
    """Copies not valid on every rank are never evicted; the store raises."""
    core = make_core(l2_records=2, tp=2)
    for tokens in ([1, 2, 3, 4], [5, 6, 7, 8]):
        alloc = await core.alloc_chunk(hash_tokens(tokens), 4, "m")
        await store_rank(core, alloc, tp_rank=0)
    third = await core.alloc_chunk(hash_tokens([9, 10, 11, 12]), 4, "m")
    with pytest.raises(TierFull):
        await core.place_store([request(third)])


@pytest.mark.asyncio
async def test_late_store_after_explicit_eviction_is_ignored() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    alloc = await core.alloc_chunk(hash_tokens(tokens), 4, "m")
    (placement,) = await core.place_store([request(alloc)])
    assert placement is not None
    await core.evict_chunk(hash_tokens(tokens))
    # The in-flight write keeps its extent until it reports.
    assert core.tier_stats()[L2].used == RECORD
    await core.store_persisted(placement, 0, 1)
    assert core.tier_stats()[L2].used == 0

    assert await core.lookup(tokens, "m") == []
    stats = await core.commit_stats()
    assert stats["commit_requests"] == 1
    assert stats["late_evicted_commits"] == 1
    # A placement request for the removed allocation is dropped.
    assert await core.place_store([request(alloc)]) == [None]


@pytest.mark.asyncio
async def test_store_of_a_reallocated_key_ignores_the_stale_allocation() -> None:
    """A write that outlived its allocation is dropped, not fatal."""
    core = make_core()
    key = "reused"
    stale = await core.alloc_chunk(key, BLOCK_TOKENS, "m")
    await core.evict_chunk(key)
    fresh = await core.alloc_chunk(key, BLOCK_TOKENS, "m")
    assert fresh.start_slot != stale.start_slot

    assert await store_rank(core, stale) is None
    assert not core.is_chunk_committed(key)
    assert await store_rank(core, fresh) is not None
    assert core.is_chunk_committed(key)


@pytest.mark.asyncio
async def test_online_lookup_skips_wait_when_gap_has_no_writer() -> None:
    """A pending later block cannot fill an evicted earlier one."""
    core = make_core()
    tokens = list(range(3 * BLOCK_TOKENS))
    keys = rolling_prefix_keys(tokens, BLOCK_TOKENS)
    await alloc_and_store(core, keys[0])
    # keys[1] was evicted and has no writer; keys[2] is still pending.
    await core.alloc_chunk(keys[2], token_count=BLOCK_TOKENS, model_id="m")

    started = asyncio.get_running_loop().time()
    chunks = await core.lookup(tokens, "m", wait_for_pending=True)
    assert [chunk.chunk_key for chunk in chunks] == [keys[0]]
    assert asyncio.get_running_loop().time() - started < 0.02


@pytest.mark.asyncio
async def test_online_lookup_waits_for_pending_gap_block() -> None:
    """The writer of the first uncovered block still extends the match."""
    core = make_core()
    tokens = list(range(2 * BLOCK_TOKENS))
    keys = rolling_prefix_keys(tokens, BLOCK_TOKENS)
    await alloc_and_store(core, keys[0])
    alloc = await core.alloc_chunk(keys[1], token_count=BLOCK_TOKENS, model_id="m")

    lookup_task = asyncio.create_task(core.lookup(tokens, "m", wait_for_pending=True))
    await asyncio.sleep(0.006)
    await store_all_ranks(core, alloc)
    assert sum(chunk.token_count for chunk in await lookup_task) == 2 * BLOCK_TOKENS


@pytest.mark.asyncio
async def test_chain_evicts_from_its_cold_tail_and_keeps_its_root() -> None:
    """A prefix chain loses its last block first; lookup keeps the prefix."""
    core = make_core(l2_records=4)
    tokens, keys = await committed_chain(core, 3)
    other = await fresh_root(core, 50)
    # Touch the chain root only, as a lookup of the short prefix would.
    assert len(await core.lookup(tokens[:BLOCK_TOKENS], "m")) == 1
    assert len(await core.lookup(other, "m")) == 1

    await fresh_root(core, 60)
    assert resident(core, keys) == [True, True, False]
    assert await hit_tokens(core, tokens) == 2 * BLOCK_TOKENS
    # That lookup refreshed the remaining chain: other and the older root 60
    # go next, then the chain's new tail.
    await fresh_root(core, 61)
    assert await core.lookup(other, "m") == []
    await fresh_root(core, 62)
    assert resident(core, keys) == [True, True, False]
    await fresh_root(core, 63)
    assert resident(core, keys) == [True, False, False]


@pytest.mark.asyncio
async def test_pending_child_protects_its_ancestors() -> None:
    """A chain with an unstored tail is not evictable; io pins wait."""
    core = make_core(l2_records=3)
    tokens = list(range(3 * BLOCK_TOKENS))
    keys = rolling_prefix_keys(tokens, BLOCK_TOKENS)
    parent = ""
    allocs = []
    for key in keys:
        allocs.append(await core.alloc_chunk(key, BLOCK_TOKENS, "m", parent_key=parent))
        parent = key
    for alloc in allocs[:2]:
        await store_all_ranks(core, alloc)
    loose = await fresh_root(core, 70)
    loose_meta = core.chunk_manager.store.get(first_rolling_key(loose))
    assert loose_meta is not None
    segments = await core.plan_load(loose_meta.start_slot, 1)

    # keys[2] is unstored, so its ancestors are not leaves; loose is io
    # pinned by the load. The store waits for the load instead of failing.
    store = asyncio.create_task(store_all_ranks(core, allocs[2]))
    await asyncio.sleep(0.01)
    assert not store.done()
    core.release_load(segments)
    await asyncio.wait_for(store, 1)
    assert resident(core, keys) == [True, True, True]
    assert await core.lookup(loose, "m") == []


@pytest.mark.asyncio
async def test_orphaned_blocks_are_evicted_before_reachable_ones() -> None:
    """Blocks below an evicted head are unreachable and leave first."""
    core = make_core(l2_records=5)
    _tokens, keys = await committed_chain(core, 3)
    other = await fresh_root(core, 80)
    await core.lookup(other, "m")
    newer = await fresh_root(core, 81)

    await core.evict_chunk(keys[0])
    for seed in (90, 91, 92):
        await fresh_root(core, seed)
    assert resident(core, keys) == [False, False, False]
    assert len(await core.lookup(other, "m")) == 1
    assert len(await core.lookup(newer, "m")) == 1


# ----------------------------------------------------------------------
# Two tiers: staging, loads, promotion
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_staged_l1_copy_publishes_before_the_l2_write() -> None:
    core = make_core(l2_records=8, l1_records=4)
    tokens = [1, 2, 3, 4]
    alloc = await core.alloc_chunk(first_rolling_key(tokens), 4, "m")
    (placement,) = await core.place_store([request(alloc)])
    assert placement is not None and placement.l1 is not None
    await core.store_staged(placement, 0, 1)
    assert core.is_chunk_committed(alloc.chunk_key)
    assert len(await core.lookup(tokens, "m")) == 1

    # The L1 source stays io pinned until the L2 write ends.
    assert placement.l1.io_pins == 1
    await core.store_persisted(placement, 0, 1)
    assert placement.l1.io_pins == 0 and placement.l2 is not None
    assert placement.l2.valid_ranks == 1
    assert (await core.commit_stats())["commit_requests"] == 1


@pytest.mark.asyncio
async def test_plan_load_spans_adjacent_chunks_from_mid_chunk() -> None:
    core = make_core(l2_records=16)
    first = await core.alloc_chunk("a", 3 * BLOCK_TOKENS, "m")
    second = await core.alloc_chunk("b", 2 * BLOCK_TOKENS, "m")
    await store_all_ranks(core, first, (RECORD, 2 * RECORD, RECORD))
    await store_all_ranks(core, second)

    segments = await core.plan_load(first.start_slot + 1, 3)
    assert [(s.chunk_key, s.slot_index, s.tier) for s in segments] == [
        ("a", 1, L2),
        ("b", 0, L2),
    ]
    a_copy, b_copy = segments[0].copy, segments[1].copy
    assert segments[0].record_sizes == (2 * RECORD, RECORD)
    assert segments[0].lane_offset == a_copy.offset + RECORD
    assert segments[0].nbytes == 3 * RECORD
    assert segments[1].record_sizes == (RECORD,)
    assert segments[1].lane_offset == b_copy.offset
    assert a_copy.io_pins == 1 and b_copy.io_pins == 1
    core.release_load(segments)
    assert a_copy.io_pins == 0 and b_copy.io_pins == 0

    with pytest.raises(ValueError):
        await core.plan_load(first.start_slot, 0)


@pytest.mark.asyncio
async def test_plan_load_rejects_uncommitted_or_unknown_slots() -> None:
    core = make_core()
    committed = await alloc_and_store(core, "a")
    pending = await core.alloc_chunk("b", BLOCK_TOKENS, "m")
    with pytest.raises(ChunkUnavailable):
        await core.plan_load(pending.start_slot, 1)
    with pytest.raises(ChunkUnavailable):
        await core.plan_load(committed.start_slot, 2)
    # Nothing stays pinned after a failed plan.
    segments = await core.plan_load(committed.start_slot, 1)
    assert segments[0].copy.io_pins == 1
    core.release_load(segments)
    with pytest.raises(ChunkUnavailable):
        await core.plan_load(1000, 1)


@pytest.mark.asyncio
async def test_l1_eviction_keeps_the_chunk_readable_from_l2() -> None:
    core = make_core(l2_records=8, l1_records=1)
    await alloc_and_store(core, "a")
    await alloc_and_store(core, "b")
    stats = core.tier_stats()
    assert stats[L1].evicted_copies == 1 and stats[L2].evicted_copies == 0
    meta = core.chunk_manager.store.get("a")
    assert meta is not None
    segments = await core.plan_load(meta.start_slot, 1)
    assert segments[0].tier == L2
    core.release_load(segments)


@pytest.mark.asyncio
async def test_promotion_serves_later_loads_from_l1() -> None:
    core = make_core(l2_records=8, l1_records=1)
    await alloc_and_store(core, "a")
    await alloc_and_store(core, "b")
    pending = await core.alloc_chunk("c", BLOCK_TOKENS, "m")
    uncommitted, promotion = await core.begin_promotions(["c", "a"])
    assert uncommitted is None
    assert promotion is not None
    meta = core.chunk_manager.store.get("a")
    assert meta is not None
    load = asyncio.create_task(core.plan_load(meta.start_slot, 1))
    await asyncio.sleep(0)
    assert not load.done()
    core.finish_promotion(promotion, ok=True)
    segments = await asyncio.wait_for(load, 1)
    assert segments[0].tier == L1
    core.release_load(segments)
    await store_all_ranks(core, pending)


@pytest.mark.asyncio
async def test_lease_keeps_l1_copy_through_l1_pressure() -> None:
    core = make_core(l2_records=8, l1_records=1)
    await alloc_and_store(core, "a")
    lease = core.pin_lease("a")
    assert lease is not None
    await alloc_and_store(core, "b")
    assert core.tier_stats()[L1].admission_skips == 1
    meta = core.chunk_manager.store.get("a")
    assert meta is not None
    segments = await core.plan_load(meta.start_slot, 1)
    assert segments[0].tier == L1
    core.release_load(segments)
    core.unpin_lease(lease)
    await alloc_and_store(core, "c")
    assert core.tier_stats()[L1].evicted_copies == 1
    assert core.pin_lease("missing") is None


@pytest.mark.asyncio
async def test_skip_l2_store_ends_at_the_staged_report() -> None:
    core = make_core(layout=make_layout(l2_records=0, l1_records=2))
    tokens = [1, 2, 3, 4]
    alloc = await core.alloc_chunk(first_rolling_key(tokens), 4, "m")
    (placement,) = await core.place_store([request(alloc)])
    assert placement is not None and placement.l2 is None
    await core.store_staged(placement, 0, 1)
    assert placement.l1 is not None and placement.l1.io_pins == 0
    assert len(await core.lookup(tokens, "m")) == 1
    with pytest.raises(ValueError):
        core.save_index("unused")


@pytest.mark.asyncio
async def test_failed_store_removes_the_uncommitted_chunk() -> None:
    registry = MetricsRegistry()
    core = make_core(registry=registry)
    alloc = await core.alloc_chunk("a", BLOCK_TOKENS, "m")
    (placement,) = await core.place_store([request(alloc)])
    assert placement is not None
    await core.store_failed(placement, 0, 1)
    assert core.chunk_manager.store.get("a") is None
    assert core.tier_stats()[L2].used == 0
    rendered = registry.render_prometheus()
    assert 'daser_cache_evicted_chunks_total{reason="store_failed"} 1.0' in rendered
    retry = await core.alloc_chunk("a", BLOCK_TOKENS, "m")
    assert retry.skipped is False
    await store_all_ranks(core, retry)
    assert core.is_chunk_committed("a")


@pytest.mark.asyncio
async def test_place_store_checks_record_counts() -> None:
    core = make_core()
    alloc = await core.alloc_chunk("a", 2 * BLOCK_TOKENS, "m")
    with pytest.raises(ValueError):
        await core.place_store([request(alloc, (RECORD,))])


@pytest.mark.asyncio
async def test_packed_record_modes_are_checked_and_kept() -> None:
    """Packed modes match the slots, stay fixed per chunk, and are queryable."""
    core = make_core()
    alloc = await core.alloc_chunk("a", 2 * BLOCK_TOKENS, "m")

    def packed(modes: tuple[str, ...]) -> StoreRequest:
        return StoreRequest(
            chunk_key="a",
            start_slot=alloc.start_slot,
            num_slots=2,
            record_sizes=(2 * RECORD, RECORD),
            record_modes=modes,
        )

    for bad in (("raw",), ("raw", "zstd")):
        with pytest.raises(ValueError):
            await core.place_store([packed(bad)])
    assert core.chunk_records("a") is None
    (placement,) = await core.place_store([packed(("raw", "compressed"))])
    assert placement is not None
    with pytest.raises(ValueError, match="changed"):
        await core.place_store([packed(("compressed", "compressed"))])
    await core.store_persisted(placement, 0, 1)
    assert core.chunk_records("a") == ChunkRecords(
        record_sizes=(2 * RECORD, RECORD), record_modes=("raw", "compressed")
    )
    raw = await alloc_and_store(core, "b")
    records = core.chunk_records(raw.chunk_key)
    assert records is not None and records.record_modes == ()


@pytest.mark.asyncio
async def test_full_tiers_track_copies_valid_on_every_rank() -> None:
    core = make_core(l2_records=8, l1_records=1, tp=2)
    alloc = await core.alloc_chunk("a", BLOCK_TOKENS, "m")
    assert core.full_tiers("a") == ()
    await store_rank(core, alloc, 0)
    assert core.full_tiers("a") == ()
    await store_rank(core, alloc, 1)
    assert core.full_tiers("a") == (L1, L2)
    await alloc_and_store(core, "b")  # evicts a's L1 copy
    assert core.full_tiers("a") == (L2,)
    assert core.full_tiers("missing") == ()


@pytest.mark.asyncio
async def test_release_placement_frees_a_duplicate_placement() -> None:
    """A chunk split over requests releases the extra placements' pins."""
    core = make_core(l2_records=8, l1_records=1)
    alloc = await core.alloc_chunk("a", BLOCK_TOKENS, "m")
    (first,) = await core.place_store([request(alloc)])
    (second,) = await core.place_store([request(alloc)])
    assert first is not None and second is not None
    assert (second.l1, second.l2) == (first.l1, first.l2)
    core.release_placement(second)
    await core.store_staged(first, 0, 1)
    await core.store_persisted(first, 0, 1)
    assert core.is_chunk_committed("a")
    # No io pin is left: L1 pressure can evict the copy.
    await alloc_and_store(core, "b")
    assert core.full_tiers("a") == (L2,)


@pytest.mark.asyncio
async def test_begin_promotions_admits_committed_l2_only_chunks() -> None:
    core = make_core(l2_records=8, l1_records=2)
    for key in ("a", "b", "c", "d"):
        await alloc_and_store(core, key)
    pending = await core.alloc_chunk("e", BLOCK_TOKENS, "m")
    assert core.full_tiers("a") == core.full_tiers("b") == (L2,)

    promotions = await core.begin_promotions(["a", "e", "b"])
    assert promotions[1] is None  # uncommitted
    for promotion in (promotions[0], promotions[2]):
        assert promotion is not None
        core.finish_promotion(promotion, ok=True)
    assert L1 in core.full_tiers("a") and L1 in core.full_tiers("b")
    await store_all_ranks(core, pending)


@pytest.mark.asyncio
async def test_resolve_slots_splits_a_lookup_run_into_chunks() -> None:
    """A merged prefix run resolves back to its chunks without pinning."""
    core = make_core()
    tokens, keys = await committed_chain(core, 2)
    wide = await alloc_and_store(core, "wide", 3 * BLOCK_TOKENS)
    (run,) = await core.lookup(tokens, "m")
    assert run.num_slots == 2
    assert core.resolve_slots(run.start_slot, run.num_slots) == [
        (keys[0], 0, 1),
        (keys[1], 0, 1),
    ]
    assert core.resolve_slots(run.start_slot + 1, 3) == [
        (keys[1], 0, 1),
        ("wide", 0, 2),
    ]
    assert core.resolve_slots(wide.start_slot + 1, 1) == [("wide", 1, 1)]
    pending = await core.alloc_chunk("p", BLOCK_TOKENS, "m")
    assert core.resolve_slots(pending.start_slot, 1) is None
    assert core.resolve_slots(10_000, 1) is None
    await store_all_ranks(core, pending)


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restored_index_keeps_lookups_and_eviction_order(tmp_path: Path) -> None:
    """A reloaded index evicts chain tails first, as before the restart."""
    path = str(tmp_path / "daser.index")
    original = make_core(l2_records=3)
    tokens, keys = await committed_chain(original, 3)
    original.save_index(path, cache_reuse_mode="prefix")

    restored = make_core(l2_records=3)
    await restored.load_index(path, expected_cache_reuse_mode="prefix")
    assert await hit_tokens(restored, tokens) == 3 * BLOCK_TOKENS
    assert restored.tier_stats()[L2].used == 3 * RECORD
    # New ids never alias restored ones.
    fresh = await restored.alloc_chunk(hash_tokens([7, 7, 7, 7]), 4, "m")
    assert fresh.start_slot == 3
    await store_all_ranks(restored, fresh)
    assert resident(restored, keys) == [True, True, False]
    assert await hit_tokens(restored, tokens) == 2 * BLOCK_TOKENS


@pytest.mark.asyncio
async def test_restored_document_sees_unsaved_chunks_as_evicted(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "daser.index")
    original = make_core(l2_records=4, l1_records=4)
    await alloc_and_store(original, "saved")
    await original.alloc_chunk("pending", BLOCK_TOKENS, "m")
    await original.register_document(
        doc_id="doc", title="t", chunk_keys=["saved", "pending"], token_count=8
    )
    original.save_index(path)

    restored = make_core(l2_records=4, l1_records=4)
    await restored.load_index(path)
    assert restored.is_chunk_committed("saved")
    assert restored.chunk_manager.store.get("pending") is None
    assert restored.tier_stats()[L1].used == 0
    doc = await restored.get_document("doc")
    assert doc is not None and doc.cached_mask == [True, False]


@pytest.mark.asyncio
async def test_l1_only_chunks_are_not_persisted(tmp_path: Path) -> None:
    path = str(tmp_path / "daser.index")
    core = make_core(l2_records=1, l1_records=4)
    await alloc_and_store(core, "a")
    lease = core.pin_lease("a")
    assert lease is not None
    # b's L2 store evicts a's L2 copy; a stays readable from L1.
    await alloc_and_store(core, "b")
    assert core.is_chunk_committed("a")
    core.save_index(path)
    core.unpin_lease(lease)

    restored = make_core(l2_records=1, l1_records=4)
    await restored.load_index(path)
    assert not restored.is_chunk_committed("a")
    assert restored.is_chunk_committed("b")


@pytest.mark.asyncio
async def test_load_index_rejects_another_layout_or_mode(tmp_path: Path) -> None:
    path = str(tmp_path / "daser.index")
    original = make_core(l2_records=4)
    await alloc_and_store(original, "a")
    original.save_index(path, cache_reuse_mode="prefix")

    for core, mode in (
        (make_core(l2_records=8), "prefix"),
        (make_core(layout=make_layout(l2_records=4, tp=2)), "prefix"),
        (make_core(l2_records=4), "chunk"),
    ):
        with pytest.raises(ValueError):
            await core.load_index(path, expected_cache_reuse_mode=mode)
        # Nothing changed: the core still serves as a cold start.
        assert core.chunk_manager.next_slot == 0
        await alloc_and_store(core, "fresh")

    corrupt = tmp_path / "corrupt.index"
    corrupt.write_bytes(b"\x93not-an-index")
    with pytest.raises(ValueError):
        await make_core(l2_records=4).load_index(str(corrupt))
