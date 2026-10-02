# SPDX-License-Identifier: Apache-2.0

"""Predicted-next-use replacement under cyclic multi-session reuse."""

import pytest

from daser.replacement.next_use import NextUseReplacementPolicy

_REGION = 100


class _Clock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _policy(clock: _Clock) -> NextUseReplacementPolicy[str]:
    return NextUseReplacementPolicy[str](region_bytes=_REGION, clock=clock)


def test_empty_policy_has_no_victim() -> None:
    """Evicting from an empty policy returns None."""
    assert _policy(_Clock()).evict() is None


def test_rejects_invalid_parameters() -> None:
    """Region size must be positive and the overdue factor above one."""
    with pytest.raises(ValueError):
        NextUseReplacementPolicy[str](region_bytes=0)
    with pytest.raises(ValueError):
        NextUseReplacementPolicy[str](overdue_factor=1.0)


def test_cyclic_sessions_keep_the_next_session() -> None:
    """Round-robin sessions evict the one used most recently, not the LRU.

    Sessions a, b, c are each reused every 3 s. At t=5.5 the next uses are
    a at 6, b at 7 and c at 8, so c is the victim while LRU would pick a.
    """
    clock = _Clock()
    policy = _policy(clock)
    for t, name in enumerate("abc"):
        clock.now = float(t)
        policy.insert_range(name, t * _REGION)
    for t, name in enumerate("abc", start=3):
        clock.now = float(t)
        policy.access(name)
    clock.now = 5.5
    assert policy.evict() == "c"
    assert policy.evict() == "b"


def test_dead_allocation_is_evicted_first() -> None:
    """An allocation idle for over twice its interval is evicted first."""
    clock = _Clock()
    policy = _policy(clock)
    policy.insert_range("dead", 0)
    policy.insert_range("live", _REGION)
    clock.now = 1.0
    policy.access("dead")
    policy.access("live")
    for t in (2.0, 3.0, 4.0):
        clock.now = t
        policy.access("live")
    clock.now = 4.5
    assert policy.evict() == "dead"


def test_region_history_survives_eviction() -> None:
    """Data promoted again into a region recovers that region's interval."""
    clock = _Clock()
    policy = _policy(clock)
    policy.insert_range("old", 0)
    policy.insert_range("other", _REGION)
    clock.now = 1.0
    policy.access("other")
    policy.remove("old")
    clock.now = 10.0
    # Region 0 was last used at 0 s, so the new key predicts reuse at 20 s;
    # "other" (interval 1 s, last 1 s) is overdue and goes first.
    policy.insert_range("again", 50)
    clock.now = 10.5
    assert policy.evict() == "other"
    assert policy.evict() == "again"
    assert policy.evict() is None


def test_ties_evict_the_later_use() -> None:
    """Equal predictions evict the later insertion, the tail of a prefix."""
    clock = _Clock()
    policy = _policy(clock)
    for name in ("head", "mid", "tail"):
        policy.insert(name)
    assert [policy.evict() for _ in range(3)] == ["tail", "mid", "head"]


def test_many_refreshes_keep_victims_consistent() -> None:
    """Repeated accesses never return a removed or stale key."""
    clock = _Clock()
    policy = _policy(clock)
    keys = [f"k{i}" for i in range(8)]
    for i, key in enumerate(keys):
        policy.insert_range(key, i * _REGION)
    for step in range(400):
        clock.now = 1.0 + step * 0.01
        policy.access(keys[step % 8])
    policy.remove("k3")
    victims = [policy.evict() for _ in range(7)]
    assert sorted(victims) == sorted(k for k in keys if k != "k3")
    assert policy.evict() is None
