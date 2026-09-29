# SPDX-License-Identifier: Apache-2.0

from benchmarks.mooncake.benchmark import MOONCAKE_BLOCK_TOKENS, TraceRequest
from benchmarks.research.admission_sim import (
    block_features,
    learn_reuse_table,
    reused_blocks,
    simulate,
)


def _req(i: int, blocks: tuple[int, ...], output: int = 10) -> TraceRequest:
    return TraceRequest(
        index=i,
        timestamp_ms=float(i),
        input_length=len(blocks) * MOONCAKE_BLOCK_TOKENS,
        output_length=output,
        hash_ids=blocks,
    )


# System prefix 0; session A turns 1-2; session B reuses turn 1 of A.
TRACE = [
    _req(0, (0, 1)),
    _req(1, (0, 1, 2)),
    _req(2, (0, 1, 2, 3)),
    _req(3, (0, 5)),
]


def test_generation_follows_session_depth() -> None:
    features = block_features(TRACE, use_output=False)
    assert features[(0, 0)] == (0, 0)
    assert features[(0, 1)] == (0, 0)
    assert features[(1, 2)] == (1, 0)
    assert features[(2, 3)] == (2, 0)
    assert features[(3, 1)] == (1, 0)
    assert (1, 0) not in features


def test_output_bucket_only_when_requested() -> None:
    trace = [_req(0, (0,), output=600)]
    assert block_features(trace, use_output=True)[(0, 0)] == (0, 2)
    assert block_features(trace, use_output=False)[(0, 0)] == (0, 0)


def test_reuse_table_counts_later_occurrences() -> None:
    reused = reused_blocks(TRACE)
    assert (0, 0) in reused and (1, 2) in reused
    assert (2, 3) not in reused and (3, 1) not in reused
    table = learn_reuse_table(block_features(TRACE, use_output=False), reused, 4)
    assert table[(0, 0)] == 1.0
    assert table[(1, 0)] == 0.5
    assert table[(2, 0)] == 0.0


def test_unbounded_write_through_hits_every_repeat() -> None:
    result = simulate(TRACE, test_start=0, l1_blocks=100, l2_blocks=100, admit=None)
    assert result.hit_rate == 6 / 11
    assert result.l2_written_blocks == 5


def test_oracle_admission_keeps_hits_and_skips_dead_writes() -> None:
    reused = reused_blocks(TRACE)
    result = simulate(
        TRACE,
        test_start=0,
        l1_blocks=0,
        l2_blocks=100,
        admit=lambda i, j: (i, j) in reused,
    )
    assert result.hit_rate == 6 / 11
    assert result.l2_written_blocks == 3


def test_l1_hit_promotes_denied_block_on_eviction() -> None:
    trace = [_req(0, (7,)), _req(1, (7,)), _req(2, (8,)), _req(3, (7,))]
    result = simulate(
        trace, test_start=0, l1_blocks=1, l2_blocks=10, admit=lambda i, j: False
    )
    # Block 7 hits in L1, is promoted to L2 when 8 evicts it, then hits again.
    assert result.hit_rate == 2 / 4
    assert result.l2_written_blocks == 1


def test_warmup_requests_are_not_counted() -> None:
    result = simulate(TRACE, test_start=3, l1_blocks=100, l2_blocks=100, admit=None)
    assert result.hit_rate == 1 / 2
    assert result.l2_written_blocks == 1
