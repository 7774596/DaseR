# SPDX-License-Identifier: Apache-2.0
"""Tests for the offline prefix-hit bounds."""

from benchmarks.mooncake.benchmark import TraceRequest
from benchmarks.research.offline_hit_bound import (
    belady_prefix_hit_rate,
    dead_hint_prefix_hit_rate,
    fifo_prefix_hit_rate,
    prefix_hit_rate,
    restarted_hit_rate,
)


def _loop(sessions: int, rounds: int, blocks: int) -> list[TraceRequest]:
    """Round-robin sessions that each resend the same ``blocks``-block prompt."""
    rows = []
    for _ in range(rounds):
        for sid in range(sessions):
            ids = tuple(sid * blocks + b for b in range(blocks))
            rows.append(
                TraceRequest(
                    index=len(rows),
                    timestamp_ms=len(rows),
                    input_length=blocks * 512,
                    output_length=1,
                    hash_ids=ids,
                    session_id=sid,
                )
            )
    return rows


def test_belady_beats_lru_on_a_loop_larger_than_the_cache() -> None:
    """A cyclic scan over capacity thrashes LRU but not the clairvoyant bound."""
    requests = _loop(sessions=3, rounds=4, blocks=4)

    lru, _, _ = prefix_hit_rate(requests, capacity_blocks=8)
    belady, _, _ = belady_prefix_hit_rate(requests, capacity_blocks=8)

    assert lru == 0.0
    assert belady > 0.4


def test_belady_matches_lru_when_everything_fits() -> None:
    """With infinite capacity both policies hit every repeated block."""
    requests = _loop(sessions=2, rounds=3, blocks=2)

    assert belady_prefix_hit_rate(requests, 0) == prefix_hit_rate(requests, 0)


def test_dead_hint_evicts_blocks_a_session_diverged_from() -> None:
    """Blocks dropped by a session's new prefix go before live blocks."""
    # Session 0 touches (1, 2) after session 1, then diverges from it.
    rows = [(1, (7, 8)), (0, (1, 2)), (0, (3, 4)), (1, (7, 8))]
    requests = [
        TraceRequest(
            index=i,
            timestamp_ms=i,
            input_length=len(ids) * 512,
            output_length=1,
            hash_ids=ids,
            session_id=sid,
        )
        for i, (sid, ids) in enumerate(rows)
    ]

    lru, _, _ = prefix_hit_rate(requests, capacity_blocks=4)
    dead, _, _ = dead_hint_prefix_hit_rate(requests, capacity_blocks=4)

    assert lru == 0.0
    assert dead == 0.25


def test_restart_empties_the_cache_before_the_given_request() -> None:
    """A restart turns the next resend of every prompt into a cold miss."""
    requests = _loop(sessions=2, rounds=3, blocks=2)

    warm = prefix_hit_rate(requests, 0)
    cold = restarted_hit_rate(prefix_hit_rate, requests, 0, [4])
    unrestarted = restarted_hit_rate(prefix_hit_rate, requests, 0, [])

    assert unrestarted == warm
    # Rounds 1 and 3 miss (2 of 6 requests hit instead of 4 of 6).
    assert warm[0] == 4 / 6
    assert cold[0] == 2 / 6


def test_fifo_evicts_a_reused_head_that_lru_keeps() -> None:
    """Hits refresh LRU but not FIFO, so FIFO drops the oldest reused prefix."""
    rows = [(0, (1, 2)), (1, (9,)), (0, (1, 2)), (1, (8,)), (0, (1, 2))]
    requests = [
        TraceRequest(
            index=i,
            timestamp_ms=i,
            input_length=len(ids) * 512,
            output_length=1,
            hash_ids=ids,
            session_id=sid,
        )
        for i, (sid, ids) in enumerate(rows)
    ]

    lru, _, _ = prefix_hit_rate(requests, capacity_blocks=3)
    fifo, _, _ = fifo_prefix_hit_rate(requests, capacity_blocks=3)

    assert lru == 0.5
    assert fifo == 0.25


def test_fifo_rewrites_resident_blocks_its_own_insertions_evict() -> None:
    """Each insertion evicts at once, so a resident later block is rewritten."""
    rows = [(1, 2, 3), (4,), (1, 2, 3), (1, 2, 3)]
    requests = [
        TraceRequest(
            index=i,
            timestamp_ms=i,
            input_length=len(ids) * 512,
            output_length=1,
            hash_ids=ids,
        )
        for i, ids in enumerate(rows)
    ]

    # Replaying (1, 2, 3) over [2, 3, 4] evicts 2 and 3 before reaching them,
    # so the store ends as [1, 2, 3] and the last request hits all 3 blocks.
    fifo, hits, _ = fifo_prefix_hit_rate(requests, capacity_blocks=3)

    assert hits == 3 * 512
    assert fifo == 0.3
