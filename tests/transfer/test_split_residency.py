# SPDX-License-Identifier: Apache-2.0

"""Split-residency replacement: stable per-prefix L1 shares under cyclic reuse."""

import pytest

from daser.replacement import (
    LRUReplacementPolicy,
    ReplacementPolicy,
    SplitResidencyPolicy,
)

PAGE = 4096


def test_protected_segment_never_exceeds_budget() -> None:
    """Large eligible inserts are rejected to probation instead of overflowing."""
    policy = SplitResidencyPolicy[int](protect_fraction=0.5, capacity_bytes=100 * PAGE)
    for key in range(400):
        policy.insert_range(key, key * PAGE, PAGE)
        assert policy.protected_bytes <= 50 * PAGE
    # A single allocation larger than the budget cannot become protected.
    policy.insert_range(10_000, 0, 80 * PAGE)
    assert policy.protected_bytes <= 50 * PAGE


def test_probation_evicted_before_protected() -> None:
    """Every probation key leaves before the first protected key."""
    policy = SplitResidencyPolicy[int](protect_fraction=0.9, capacity_bytes=1 << 40)
    for key in range(200):
        policy.insert_range(key, key * PAGE, PAGE)
    protected = policy.protected_bytes // PAGE
    assert 0 < protected < 200
    order = [policy.evict() for _ in range(200)]
    assert policy.evict() is None
    assert set(order) == set(range(200))
    survivors = SplitResidencyPolicy[int](protect_fraction=0.9, capacity_bytes=1 << 40)
    for key in range(200):
        survivors.insert_range(key, key * PAGE, PAGE)
    for _ in range(200 - protected):
        survivors.evict()
    assert survivors.protected_bytes == protected * PAGE


def test_live_protected_keys_survive_and_stale_ones_are_replaced() -> None:
    """A full protected segment rejects newcomers until its oldest key is stale."""
    capacity = 100 * PAGE
    policy = SplitResidencyPolicy[int](protect_fraction=0.5, capacity_bytes=capacity)
    for key in range(200):
        policy.insert_range(key, key * PAGE, PAGE)
    before = policy.eligibility
    full = policy.protected_bytes
    # Fresh eligible inserts while old keys are still live: budget holds.
    for key in range(200, 260):
        policy.insert_range(key, key * PAGE, PAGE)
    assert policy.protected_bytes <= 50 * PAGE
    assert policy.eligibility <= before
    # After one capacity of inserted bytes nothing old was touched, so old
    # protected keys are stale and newcomers can replace them.
    for key in range(260, 600):
        policy.insert_range(key, key * PAGE, PAGE)
    remaining_old = [k for k in range(200) if k in _protected_keys(policy, 600)]
    assert len(remaining_old) < full // PAGE


def _protected_keys(policy: SplitResidencyPolicy[int], keys: int) -> set[int]:
    """Drain ``policy`` and return the keys evicted after the last probation key."""
    marker = -1
    policy.insert(marker)
    order = [policy.evict() for _ in range(keys + 1)]
    return {key for key in order[order.index(marker) + 1 :] if key is not None}


def test_admits_matches_protected_insert() -> None:
    """``admits`` predicts whether a fresh insert enters the protected segment."""
    policy = SplitResidencyPolicy[int](protect_fraction=0.5, capacity_bytes=1 << 40)
    for key in range(64):
        admitted = policy.admits(key * PAGE)
        before = policy.protected_bytes
        policy.insert_range(key, key * PAGE, PAGE)
        assert (policy.protected_bytes > before) == admitted
    assert 0 < policy.protected_bytes < 64 * PAGE


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
def test_rejects_out_of_range_fraction(fraction: float) -> None:
    """The protected budget must be a proper share of capacity."""
    with pytest.raises(ValueError):
        SplitResidencyPolicy[int](protect_fraction=fraction, capacity_bytes=PAGE)


def _cyclic_session_hits(
    policy: ReplacementPolicy[int], promote_pages: int = 1
) -> list[int]:
    """Replay 4 cyclic sessions of 8 pages over a 24-page L1; count hits/session.

    Misses are promoted in groups of ``promote_pages`` adjacent pages sharing
    one allocation, like a coalesced L2 miss read.
    """
    pytest.importorskip("cupy")
    from daser.transfer.iouring.l1_cache import L1Cache

    sessions, pages, capacity = 4, 8, 24
    cache = L1Cache(capacity * PAGE, PAGE, lambda _key, _data: False, policy)
    hits = [0] * sessions
    try:
        for round_index in range(8):
            for session in range(sessions):
                page = 0
                while page < pages:
                    offset = (session * pages + page) * PAGE
                    found, _missing = cache.resolve_subranges(0, offset, PAGE)
                    if found:
                        cache.record_hits(found)
                        if round_index >= 3:
                            hits[session] += 1
                        page += 1
                        continue
                    group = 1
                    while group < promote_pages and page + group < pages:
                        nxt = offset + group * PAGE
                        if cache.resolve_subranges(0, nxt, PAGE)[0]:
                            break
                        group += 1
                    parent = cache.reserve_untracked(group * PAGE)
                    assert parent is not None
                    children = [
                        ((offset + i * PAGE, PAGE), parent.subslice(i * PAGE, PAGE))
                        for i in range(group)
                    ]
                    parent.close()
                    cache.put_reserved_group(children)
                    page += group
    finally:
        cache.close()
    return hits


@pytest.mark.parametrize("promote_pages", [1, 4])
def test_split_residency_keeps_every_session_partially_resident(
    promote_pages: int,
) -> None:
    """Cyclic reuse beyond capacity: LRU never hits; split keeps every session warm."""
    capacity = 24 * PAGE
    lru = _cyclic_session_hits(LRUReplacementPolicy[int](), promote_pages)
    assert lru == [0, 0, 0, 0]
    split = _cyclic_session_hits(
        SplitResidencyPolicy[int](protect_fraction=0.85, capacity_bytes=capacity),
        promote_pages,
    )
    assert min(split) > 0
