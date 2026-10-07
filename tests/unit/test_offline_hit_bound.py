# SPDX-License-Identifier: Apache-2.0
"""Tests for the offline prefix-hit bounds."""

from benchmarks.mooncake.benchmark import TraceRequest
from benchmarks.research.offline_hit_bound import (
    belady_prefix_hit_rate,
    dead_hint_prefix_hit_rate,
    prefix_hit_rate,
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
