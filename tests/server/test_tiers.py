# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import random

# Third Party
import pytest

# First Party
from daser.metrics import MetricsRegistry
from daser.server.tiers import (
    L1,
    L2,
    TIERS,
    ChunkUnavailable,
    L2Placement,
    StorePlacement,
    TierFull,
    TierIndex,
    TierLayout,
    record_offsets,
)

G = 4096


class Harness:
    """A TierIndex with recorded loss callbacks and a blocked-key set."""

    def __init__(
        self,
        l1: int = 0,
        l2: int = 0,
        tp: int = 1,
        l1_record_bytes: int = 0,
    ) -> None:
        self.layout = TierLayout(
            grain=G,
            l1_lane_bytes=l1 * G,
            l2_lane_bytes=l2 * G,
            tensor_parallel_size=tp,
            l1_record_bytes=l1_record_bytes,
        )
        self.blocked: set[str] = set()
        self.lost: list[str] = []
        self.parents: dict[str, str] = {}
        self.tiers = TierIndex(
            self.layout,
            blocked=lambda key: key in self.blocked,
            on_lost=self.lost.append,
            metrics_registry=MetricsRegistry(),
        )

    def add(self, key: str, parent: str = "") -> None:
        self.tiers.add_chunk(key, parent)
        self.parents[key] = parent

    def place(
        self, key: str, records: int = 1, parent: str | None = None
    ) -> StorePlacement:
        if parent is not None:
            self.add(key, parent)
        (placement,) = self.tiers.place_store([(key, (G,) * records)])
        return placement

    def finish(self, placement: StorePlacement, ranks: range | None = None) -> None:
        """Mark every copy valid on ``ranks`` and drop the placement pins."""
        for rank in (
            ranks if ranks is not None else range(self.layout.tensor_parallel_size)
        ):
            for copy in (placement.l1, placement.l2):
                if copy is not None:
                    self.tiers.mark_valid(copy, rank)
        for copy in (placement.l1, placement.l2):
            if copy is not None:
                self.tiers.unpin_io(copy)

    def store(self, key: str, records: int = 1, parent: str = "") -> StorePlacement:
        placement = self.place(key, records, parent)
        self.finish(placement)
        return placement


def assert_invariants(h: Harness, quiescent: bool = True) -> None:
    """Check the tier accounting and residency contract of ``h``.

    Args:
        h: Harness to check.
        quiescent: No store, load or promotion is in flight.
    """
    tiers = h.tiers
    extents: dict[str, list[tuple[int, int]]] = {tier: [] for tier in TIERS}
    copies = {tier: 0 for tier in TIERS}
    for key in h.parents:
        chunk = tiers.chunk(key)
        if chunk is None:
            continue
        for tier in TIERS:
            copy = chunk.copy(tier)
            if copy is None:
                continue
            assert tiers.enabled(tier)
            assert copy.attached and copy.chunk_key == key and copy.tier == tier
            assert copy.offset % G == 0 and copy.length % G == 0
            assert copy.length == h.layout.extent_length(tier, chunk.record_sizes)
            assert copy.offset + copy.length <= h.layout.lane_bytes(tier)
            extents[tier].append((copy.offset, copy.length))
            copies[tier] += 1
            if quiescent:
                assert copy.io_pins == 0
        if quiescent:
            # No partial survivor: every chunk with tier state is present.
            assert tiers.is_present(key), key
            assert any(tiers.is_full(chunk.copy(tier)) for tier in TIERS)
            # Capacity eviction removes leaves only, so ancestors stay.
            parent = h.parents[key]
            if parent:
                assert tiers.chunk(parent) is not None, (key, parent)
    for key in h.lost:
        assert tiers.chunk(key) is None
    for tier in TIERS:
        if not tiers.enabled(tier):
            continue
        stats = tiers.stats(tier)
        assert stats.used <= stats.capacity
        assert stats.used + stats.free == stats.capacity
        assert stats.copies == copies[tier]
        spans = sorted(extents[tier])
        for (start, length), (nxt, _length) in zip(spans, spans[1:], strict=False):
            assert start + length <= nxt
        if quiescent:
            assert stats.used == sum(length for _start, length in spans)


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def test_layout_validates_grain_lanes_and_tiers() -> None:
    with pytest.raises(ValueError):
        TierLayout(grain=3 * G, l1_lane_bytes=0, l2_lane_bytes=3 * G)
    with pytest.raises(ValueError):
        TierLayout(grain=2048, l1_lane_bytes=0, l2_lane_bytes=G)
    with pytest.raises(ValueError):
        TierLayout(grain=2 * G, l1_lane_bytes=0, l2_lane_bytes=3 * G)
    with pytest.raises(ValueError):
        TierLayout(grain=G, l1_lane_bytes=0, l2_lane_bytes=0)
    with pytest.raises(ValueError):
        TierLayout(grain=G, l1_lane_bytes=G, l2_lane_bytes=0, l1_record_bytes=100)


def test_layout_extents_and_lane_offsets() -> None:
    layout = TierLayout(
        grain=4 * G,
        l1_lane_bytes=8 * G,
        l2_lane_bytes=16 * G,
        tensor_parallel_size=2,
        l1_record_bytes=2 * G,
    )
    # Records pack at 4 KiB and the extent rounds up to the grain.
    assert layout.extent_length(L2, (G, G, G)) == 4 * G
    assert layout.extent_length(L2, (5 * G,)) == 8 * G
    # Raw accounting charges L1 at least one raw record per slot.
    assert layout.extent_length(L1, (G, G, G)) == 8 * G
    assert layout.absolute_offset(L2, 1, 4 * G) == 20 * G
    assert layout.absolute_offset(L1, 1, 0) == 8 * G
    with pytest.raises(ValueError):
        layout.absolute_offset(L2, 2, 0)
    assert record_offsets((G, 3 * G, G)) == [0, G, 4 * G]


# ----------------------------------------------------------------------
# Store placement
# ----------------------------------------------------------------------


def test_batch_of_new_chunks_gets_one_contiguous_l2_extent() -> None:
    h = Harness(l2=16)
    h.store("pad", 1)
    for key in ("a", "b", "c"):
        h.add(key)
    placements = h.tiers.place_store([("a", (G,)), ("b", (G, G)), ("c", (G,))])
    offsets = [p.l2.offset for p in placements if p.l2 is not None]
    assert offsets == [G, 2 * G, 4 * G]
    for placement in placements:
        h.finish(placement)
    assert_invariants(h)


def test_batch_falls_back_to_per_chunk_eviction_when_fragmented() -> None:
    h = Harness(l2=4)
    for index in range(4):
        h.store(f"k{index}")
    # Free two non-adjacent grains: no contiguous 2-grain extent exists.
    h.tiers.remove("k0")
    h.tiers.remove("k2")
    h.add("a")
    h.add("b")
    placements = h.tiers.place_store([("a", (G,)), ("b", (G,))])
    assert {p.l2.offset for p in placements if p.l2 is not None} == {0, 2 * G}
    for placement in placements:
        h.finish(placement)
    # A third chunk now needs an eviction, taking the oldest leaf.
    h.store("c", parent="")
    assert h.lost == ["k1"]
    assert_invariants(h)


def test_every_rank_gets_the_same_placement() -> None:
    h = Harness(l1=4, l2=4, tp=2)
    h.add("a")
    first = h.tiers.place_store([("a", (G,))])[0]
    second = h.tiers.place_store([("a", (G,))])[0]
    assert first.l1 is second.l1 and first.l2 is second.l2
    assert first.l2 is not None and first.l2.io_pins == 2
    h.finish(first, range(0, 1))
    assert not h.tiers.is_present("a")
    assert h.tiers.is_rank_valid("a", 0) and not h.tiers.is_rank_valid("a", 1)
    h.finish(second, range(1, 2))
    assert h.tiers.is_present("a")
    assert_invariants(h)


def test_place_store_rejects_bad_requests() -> None:
    h = Harness(l2=4)
    with pytest.raises(ValueError):
        h.tiers.place_store([("unknown", (G,))])
    h.add("a")
    with pytest.raises(ValueError):
        h.tiers.place_store([("a", (G,)), ("a", (G,))])
    with pytest.raises(ValueError):
        h.tiers.place_store([("a", (100,))])
    with pytest.raises(ValueError):
        h.tiers.place_store([("a", ())])
    h.store("b")
    with pytest.raises(ValueError):
        h.tiers.place_store([("b", (G, G))])
    assert_invariants(h)


def test_mark_valid_ignores_detached_copies_and_checks_rank() -> None:
    h = Harness(l2=4)
    placement = h.place("a", parent="")
    assert placement.l2 is not None
    with pytest.raises(ValueError):
        h.tiers.mark_valid(placement.l2, 1)
    h.tiers.remove("a")
    assert h.tiers.mark_valid(placement.l2, 0) is False
    # The pinned extent is freed by the last unpin, not by remove.
    assert h.tiers.stats(L2).used == G
    h.tiers.unpin_io(placement.l2)
    assert h.tiers.stats(L2).used == 0
    with pytest.raises(ValueError):
        h.tiers.unpin_io(placement.l2)


def test_l1_full_of_io_pins_skips_admission_but_l2_takes_the_store() -> None:
    h = Harness(l1=1, l2=4)
    held = h.place("a", parent="")
    h.finish(held)
    h.tiers.pin_io(held.l1)  # an in-flight L1 read
    placement = h.place("b", parent="")
    assert placement.l1 is None and placement.l2 is not None
    assert h.tiers.stats(L1).admission_skips == 1
    h.finish(placement)
    h.tiers.unpin_io(held.l1)
    assert h.tiers.is_present("b")
    assert_invariants(h)


def test_skip_l2_raises_tier_full_when_l1_cannot_admit() -> None:
    h = Harness(l1=2)
    h.store("a")
    h.store("b")
    leases = [h.tiers.pin_lease(key) for key in ("a", "b")]
    h.add("c")
    with pytest.raises(TierFull) as excinfo:
        h.tiers.place_store([("c", (G,))])
    assert excinfo.value.tier == L1 and excinfo.value.io_blocked is False
    # An io pin instead of a lease reports that finishing IO may help.
    for lease in leases:
        assert lease is not None
        h.tiers.unpin_lease(lease)
    a = h.tiers.chunk("a")
    b = h.tiers.chunk("b")
    assert a is not None and b is not None and a.l1 is not None and b.l1 is not None
    h.tiers.pin_io(a.l1)
    h.tiers.pin_io(b.l1)
    with pytest.raises(TierFull) as excinfo:
        h.tiers.place_store([("c", (G,))])
    assert excinfo.value.io_blocked is True
    h.tiers.unpin_io(a.l1)
    h.tiers.unpin_io(b.l1)
    placement = h.place("c")
    h.finish(placement)
    assert h.lost == ["a"]
    assert_invariants(h)


def test_failed_batch_leaves_nothing_placed_or_pinned() -> None:
    h = Harness(l1=2)
    h.store("a")
    lease = h.tiers.pin_lease("a")
    h.add("b")
    h.add("c")
    with pytest.raises(TierFull):
        h.tiers.place_store([("b", (G,)), ("c", (G,))])
    assert h.tiers.chunk("b") is None and h.tiers.chunk("c") is None
    assert h.tiers.stats(L1).used == G
    assert lease is not None
    h.tiers.unpin_lease(lease)
    assert_invariants(h)


def test_discard_incomplete_keeps_full_copies_only() -> None:
    h = Harness(l1=4, l2=4, tp=2)
    placement = h.place("a", parent="")
    assert placement.l1 is not None and placement.l2 is not None
    for rank in range(2):
        h.tiers.mark_valid(placement.l1, rank)
    h.tiers.mark_valid(placement.l2, 0)
    assert h.tiers.discard_incomplete(placement) is True
    h.tiers.unpin_io(placement.l1)
    h.tiers.unpin_io(placement.l2)
    chunk = h.tiers.chunk("a")
    assert chunk is not None and chunk.l2 is None and chunk.l1 is placement.l1
    assert_invariants(h)

    lone = h.place("b", parent="")
    assert h.tiers.discard_incomplete(lone) is False
    h.tiers.unpin_io(lone.l1)
    h.tiers.unpin_io(lone.l2)
    assert h.tiers.chunk("b") is None
    assert h.tiers.stats(L2).used == 0


# ----------------------------------------------------------------------
# Eviction
# ----------------------------------------------------------------------


def test_eviction_takes_leaf_lru_and_keeps_chain_heads() -> None:
    h = Harness(l2=3)
    h.store("a0")
    h.store("a1", parent="a0")
    h.store("b0")
    h.tiers.touch(["a0", "a1"])
    h.store("c0")
    assert h.lost == ["b0"]
    h.store("d0")
    # a0 is not a leaf while a1 exists, so the chain loses its tail.
    assert h.lost == ["b0", "a1"]
    assert_invariants(h)


def test_queue_hint_keeps_queued_chunks_and_counts_hinted_evictions() -> None:
    registry = MetricsRegistry()
    h = Harness(l2=2)
    h.tiers = TierIndex(
        h.layout,
        blocked=lambda key: key in h.blocked,
        on_lost=h.lost.append,
        metrics_registry=registry,
    )
    h.store("a0")
    h.store("b0")
    h.tiers.want("r1", ["a0"])
    h.store("c0")
    # LRU would take a0; the queued request keeps it.
    assert h.lost == ["b0"]
    h.tiers.want("r2", ["c0"])
    h.store("d0")
    # Both remaining copies are hinted: the later hint (c0) goes first.
    assert h.lost == ["b0", "c0"]
    text = registry.render_prometheus()
    assert 'daser_queue_hint_evicted_copies_total{tier="l2"} 1' in text
    assert "daser_queue_hint_requests 2" in text
    assert "daser_queue_hints_total 2" in text
    h.tiers.unwant("r1")
    h.tiers.unwant("r2")
    assert "daser_queue_hint_requests 0" in registry.render_prometheus()
    assert_invariants(h)


def test_copy_with_a_full_pair_is_evicted_even_inside_a_chain() -> None:
    h = Harness(l1=1, l2=4)
    h.store("a0")
    h.store("a1", parent="a0")
    # a1's L1 copy pushed out a0's: a0 has a full L2 copy, so its L1 copy
    # was a candidate although a0 is not a leaf.
    a0 = h.tiers.chunk("a0")
    assert a0 is not None and a0.l1 is None and h.tiers.is_full(a0.l2)
    assert h.lost == []
    assert h.tiers.stats(L1).evicted_copies == 1
    assert_invariants(h)


def test_blocked_and_pinned_chunks_are_not_evicted() -> None:
    h = Harness(l2=2)
    h.store("pending")
    h.store("loaded")
    h.blocked.add("pending")
    h.tiers.pin_io(h.tiers.chunk("loaded").l2)
    h.add("c")
    with pytest.raises(TierFull) as excinfo:
        h.tiers.place_store([("c", (G,))])
    assert excinfo.value.io_blocked is True
    h.blocked.clear()
    h.tiers.refresh("pending")
    h.finish(h.place("c"))
    assert h.lost == ["pending"]
    h.tiers.unpin_io(h.tiers.chunk("loaded").l2)
    assert_invariants(h)


def test_lease_blocks_l1_eviction_but_not_l2() -> None:
    h = Harness(l1=1, l2=1)
    h.store("a")
    lease = h.tiers.pin_lease("a")
    assert lease is not None
    placement = h.place("b", parent="")
    # L2 evicted a's copy (its L1 copy is full); L1 kept the leased copy.
    assert placement.l1 is None and placement.l2 is not None
    h.finish(placement)
    a = h.tiers.chunk("a")
    assert a is not None and a.l2 is None and h.tiers.is_full(a.l1)
    h.tiers.unpin_lease(lease)
    with pytest.raises(ValueError):
        h.tiers.unpin_lease(lease)
    assert_invariants(h)


def test_evicting_a_full_copy_drops_a_partial_survivor() -> None:
    h = Harness(l1=1, l2=4, tp=2)
    placement = h.place("a", parent="")
    assert placement.l1 is not None and placement.l2 is not None
    for rank in range(2):
        h.tiers.mark_valid(placement.l1, rank)
    h.tiers.mark_valid(placement.l2, 0)
    h.tiers.unpin_io(placement.l1)
    h.tiers.unpin_io(placement.l2)
    h.store("b")
    # a's L1 copy was its only full copy; its partial L2 copy cannot serve.
    assert h.lost == ["a"]
    assert h.tiers.stats(L2).used == G
    assert_invariants(h)


def test_remove_releases_unpinned_copies_at_once() -> None:
    h = Harness(l1=2, l2=2)
    h.store("a")
    h.tiers.remove("a")
    assert h.tiers.chunk("a") is None
    assert h.tiers.stats(L1).used == 0 and h.tiers.stats(L2).used == 0
    assert h.lost == []


@pytest.mark.asyncio
async def test_io_released_wakes_on_last_unpin() -> None:
    h = Harness(l2=2)
    h.store("a")
    copy = h.tiers.chunk("a").l2
    await h.tiers.io_released(L2)  # nothing pinned: returns at once
    h.tiers.pin_io(copy)
    h.tiers.pin_io(copy)
    waiter = asyncio.create_task(h.tiers.io_released(L2))
    await asyncio.sleep(0)
    h.tiers.unpin_io(copy)
    await asyncio.sleep(0)
    assert not waiter.done()
    h.tiers.unpin_io(copy)
    await asyncio.wait_for(waiter, 1)


# ----------------------------------------------------------------------
# Loads and promotion
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_plan_prefers_l1_and_pins() -> None:
    h = Harness(l1=1, l2=4)
    h.store("a")
    copy = await h.tiers.read_plan("a")
    assert copy.tier == L1 and copy.io_pins == 1
    h.tiers.unpin_io(copy)
    h.store("b")
    copy = await h.tiers.read_plan("a")
    assert copy.tier == L2
    h.tiers.unpin_io(copy)
    with pytest.raises(ChunkUnavailable):
        await h.tiers.read_plan("missing")
    assert_invariants(h)


@pytest.mark.asyncio
async def test_promotion_pins_block_eviction_and_loads_wait_for_it() -> None:
    h = Harness(l1=1, l2=4)
    h.store("a")
    h.store("b")  # a is now L2-only
    promotion = h.tiers.begin_promotion("a")
    assert promotion is not None and promotion.target.tier == L1
    b = h.tiers.chunk("b")
    assert b is not None and b.l1 is None  # evicted for the promotion
    # The target is pinned and not valid yet: L1 cannot admit b again.
    assert h.tiers.begin_promotion("b") is None
    assert h.tiers.stats(L1).admission_skips == 1

    load = asyncio.create_task(h.tiers.read_plan("a"))
    await asyncio.sleep(0)
    assert not load.done()
    h.tiers.finish_promotion(promotion, ok=True)
    copy = await asyncio.wait_for(load, 1)
    assert copy.tier == L1
    h.tiers.unpin_io(copy)
    assert_invariants(h)


@pytest.mark.asyncio
async def test_failed_promotion_falls_back_to_l2() -> None:
    h = Harness(l1=1, l2=4)
    h.store("a")
    h.store("b")
    promotion = h.tiers.begin_promotion("a")
    assert promotion is not None
    load = asyncio.create_task(h.tiers.read_plan("a"))
    await asyncio.sleep(0)
    h.tiers.finish_promotion(promotion, ok=False)
    copy = await asyncio.wait_for(load, 1)
    assert copy.tier == L2
    h.tiers.unpin_io(copy)
    assert h.tiers.chunk("a").l1 is None
    assert_invariants(h)


@pytest.mark.asyncio
async def test_removing_a_chunk_resolves_its_promotion() -> None:
    h = Harness(l1=1, l2=4)
    h.store("a")
    h.store("b")
    promotion = h.tiers.begin_promotion("a")
    assert promotion is not None
    load = asyncio.create_task(h.tiers.read_plan("a"))
    await asyncio.sleep(0)
    h.tiers.remove("a")
    with pytest.raises(ChunkUnavailable):
        await asyncio.wait_for(load, 1)
    h.tiers.finish_promotion(promotion, ok=True)
    assert h.tiers.stats(L1).used == 0
    assert h.tiers.stats(L2).used == G


def test_begin_promotion_needs_an_l2_only_full_chunk() -> None:
    h = Harness(l1=2, l2=4)
    h.store("a")
    assert h.tiers.begin_promotion("a") is None  # already in L1
    assert h.tiers.begin_promotion("missing") is None
    l2_only = Harness(l2=4)
    l2_only.store("a")
    assert l2_only.tiers.begin_promotion("a") is None  # L1 disabled


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def test_l2_snapshot_round_trip() -> None:
    h = Harness(l1=2, l2=8)
    h.store("a", 2)
    h.store("b", 1, parent="a")
    placements = h.tiers.snapshot_l2(["a", "b", "missing"])
    assert [p.chunk_key for p in placements] == ["a", "b"]

    restored = Harness(l1=2, l2=8)
    restored.add("a")
    restored.add("b", "a")
    restored.tiers.restore_l2(placements)
    assert restored.tiers.stats(L2).used == h.tiers.stats(L2).used
    assert restored.tiers.stats(L1).used == 0
    assert restored.tiers.is_present("a") and restored.tiers.is_present("b")
    assert_invariants(restored)
    # The restored chain keeps leaf-first order.
    for index in range(6):
        restored.store(f"x{index}")
    assert restored.lost == ["b"]


def test_l2_restore_rejects_invalid_placements() -> None:
    h = Harness(l2=4)
    for key in ("a", "b"):
        h.add(key)
    with pytest.raises(ValueError, match="overlap"):
        h.tiers.validate_l2_placements(
            [L2Placement("a", 0, (G, G)), L2Placement("b", G, (G,))]
        )
    with pytest.raises(ValueError):
        h.tiers.validate_l2_placements([L2Placement("a", 3 * G, (G, G))])
    with pytest.raises(ValueError):
        h.tiers.validate_l2_placements([L2Placement("a", 100, (G,))])
    with pytest.raises(ValueError):
        h.tiers.restore_l2([L2Placement("unknown", 0, (G,))])
    h.store("c", parent="")
    with pytest.raises(ValueError):
        h.tiers.validate_l2_placements([L2Placement("a", 2 * G, (G,))])
    no_l2 = Harness(l1=4)
    with pytest.raises(ValueError):
        no_l2.tiers.validate_l2_placements([])


# ----------------------------------------------------------------------
# Randomized invariants
# ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(8))
async def test_random_operations_keep_tier_invariants(seed: int) -> None:
    """Random chain stores, failures, loads, promotions and leases.

    After every step: every chunk with tier state is present (a copy valid
    on every rank), capacity eviction never strands a child without its
    parent, extents are aligned, in the lane and disjoint, and the allocator
    accounts every byte.
    """
    rng = random.Random(seed)
    tp = rng.choice((1, 2))
    l2 = 0 if seed % 4 == 3 else rng.randint(6, 14)
    h = Harness(l1=rng.randint(3, 8), l2=l2, tp=tp)
    chains: list[str] = []
    leases = []
    serial = 0
    for _step in range(300):
        op = rng.random()
        if op < 0.45:
            serial += 1
            key = f"k{serial}"
            tails = [tail for tail in chains if h.tiers.chunk(tail) is not None]
            parent = rng.choice(tails) if tails and rng.random() < 0.7 else ""
            h.add(key, parent)
            sizes = tuple(G * rng.randint(1, 2) for _ in range(rng.randint(1, 2)))
            try:
                placements = h.tiers.place_store([(key, sizes)])
            except TierFull:
                h.tiers.remove(key)
                del h.parents[key]
                continue
            (placement,) = placements
            fail = rng.random() < 0.15
            ranks = range(tp - 1) if fail else range(tp)
            for rank in ranks:
                for copy in (placement.l1, placement.l2):
                    if copy is not None:
                        h.tiers.mark_valid(copy, rank)
            if fail and not h.tiers.discard_incomplete(placement):
                for copy in (placement.l1, placement.l2):
                    if copy is not None:
                        h.tiers.unpin_io(copy)
                h.tiers.remove(key)
                del h.parents[key]
                continue
            for copy in (placement.l1, placement.l2):
                if copy is not None:
                    h.tiers.unpin_io(copy)
            if parent in chains:
                chains.remove(parent)
            chains.append(key)
        elif op < 0.65:
            live = [key for key in h.parents if h.tiers.chunk(key) is not None]
            if live:
                key = rng.choice(live)
                copy = await h.tiers.read_plan(key)
                assert h.tiers.is_full(copy)
                h.tiers.touch_path(key)
                h.tiers.unpin_io(copy)
        elif op < 0.8:
            live = [key for key in h.parents if h.tiers.chunk(key) is not None]
            if live:
                promotion = h.tiers.begin_promotion(rng.choice(live))
                if promotion is not None:
                    assert_invariants(h, quiescent=False)
                    h.tiers.finish_promotion(promotion, ok=rng.random() < 0.8)
        elif op < 0.9:
            live = [key for key in h.parents if h.tiers.chunk(key) is not None]
            if live and len(leases) < 2:
                lease = h.tiers.pin_lease(rng.choice(live))
                if lease is not None:
                    leases.append(lease)
        elif leases:
            h.tiers.unpin_lease(leases.pop(rng.randrange(len(leases))))
        assert_invariants(h)
    for lease in leases:
        h.tiers.unpin_lease(lease)
    assert_invariants(h)
    # The tiers are small enough that capacity eviction ran.
    assert h.lost
