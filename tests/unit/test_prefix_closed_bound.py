# SPDX-License-Identifier: Apache-2.0
"""Tests for the offline prefix-closed eviction bound."""

import pytest

from benchmarks.research.prefix_closed_bound import simulate


def test_cascade_frees_orphans_for_a_returning_session() -> None:
    """FIFO keeps an unreachable tail; cascade reuses its space for others."""
    # A's second turn appends block 2 after B, so FIFO evicting A's head
    # leaves 2 resident ahead of nothing useful while B's head goes next.
    requests = [
        ([1], 2),
        ([10, 11], 4),
        ([1, 2], 4),
        ([20], 2),
        ([30], 2),
        ([10, 11], 4),
    ]
    fifo_hit, fifo_orphan = simulate(
        requests, 4, "fifo", block_tokens=2, sample_every=1
    )
    cascade_hit, cascade_orphan = simulate(
        requests, 4, "cascade", block_tokens=2, sample_every=1
    )
    # Cascade drops A's tail with its head, so B survives to its return.
    assert cascade_hit > fifo_hit
    assert cascade_orphan == 0.0
    assert fifo_orphan > 0.0


def test_lru_refresh_keeps_reused_prefix() -> None:
    """A re-touched prefix survives under LRU but not FIFO."""
    requests = [([1, 2], 4), ([3], 2), ([1, 2], 4), ([4], 2), ([1, 2], 4)]
    lru_hit, _ = simulate(requests, 3, "lru", block_tokens=2)
    fifo_hit, _ = simulate(requests, 3, "fifo", block_tokens=2)
    assert lru_hit > fifo_hit


def test_unknown_mode_raises() -> None:
    """Only the documented modes are accepted."""
    with pytest.raises(ValueError):
        simulate([], 1, "belady", block_tokens=1)
