# SPDX-License-Identifier: Apache-2.0

from benchmarks.mooncake.benchmark import MOONCAKE_BLOCK_TOKENS, TraceRequest
from benchmarks.research.admission_sim import (
    TierPolicy,
    auc,
    block_features,
    gpu_reuse_labels,
    learn_reuse_table,
    reused_blocks,
    simulate,
    simulate_tiers,
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


def test_auc_ranks_and_ties() -> None:
    assert auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert auc([1.0, 1.0, 1.0, 1.0], [False, True, False, True]) == 0.5
    assert auc([0.3, 0.4], [True, True]) == 0.5


def test_gpu_reuse_labels_split_by_gpu_residency() -> None:
    local, external, _, _ = gpu_reuse_labels(TRACE, gpu_blocks=100)
    assert external == set()
    assert (0, 0) in local and (1, 2) in local
    local, external, _, _ = gpu_reuse_labels(TRACE, gpu_blocks=0)
    assert local == set()
    assert {(0, 0), (0, 1), (1, 2)} <= external


def test_tiers_serve_gpu_misses_from_external_tier() -> None:
    result = simulate_tiers(
        TRACE, gpu_blocks=0, l1_blocks=100, l2_blocks=100, policy=TierPolicy()
    )
    prompt = sum(r.input_length for r in TRACE)
    assert result.local_hit == 0.0
    assert result.external_hit == 6 * MOONCAKE_BLOCK_TOKENS / prompt
    assert result.l2_written_blocks == 5


def test_zero_denied_budget_loses_reuse_and_counts_ghosts() -> None:
    policy = TierPolicy(min_output=1000, denied_budget_blocks=0)
    result = simulate_tiers(
        TRACE, gpu_blocks=0, l1_blocks=100, l2_blocks=100, policy=policy
    )
    assert result.external_hit == 0.0
    assert result.l2_written_blocks == 0
    # Every re-store of an evicted block is a ghost and is denied again.
    assert (result.denied, result.ghosts) == (11, 6)


def test_denied_hit_is_rescued_to_l2() -> None:
    policy = TierPolicy(min_output=1000, denied_budget_blocks=100)
    result = simulate_tiers(
        TRACE, gpu_blocks=0, l1_blocks=100, l2_blocks=100, policy=policy
    )
    assert result.rescued == 3
    assert result.l2_written_blocks == 3


def test_full_fifo_arena_removes_oldest_block_from_every_tier() -> None:
    trace = [_req(0, (7,)), _req(1, (8,)), _req(2, (7,))]
    lru = simulate_tiers(
        trace, gpu_blocks=0, l1_blocks=100, l2_blocks=1, policy=TierPolicy()
    )
    fifo = simulate_tiers(
        trace,
        gpu_blocks=0,
        l1_blocks=100,
        l2_blocks=1,
        policy=TierPolicy(fifo_arena=True),
    )
    assert lru.external_hit > 0.0
    assert fifo.external_hit == 0.0


def test_exclusive_tiers_hold_l1_blocks_out_of_l2() -> None:
    trace = [_req(0, (7,)), _req(1, (8,)), _req(2, (7,))]
    kwargs = {"gpu_blocks": 0, "l1_blocks": 1, "l2_blocks": 1}
    inclusive = simulate_tiers(trace, policy=TierPolicy(), **kwargs)
    exclusive = simulate_tiers(trace, policy=TierPolicy(exclusive=True), **kwargs)
    # Inclusive L2 already holds 8 when 7 returns; exclusive L2 holds only the
    # block L1 evicted, so 7 hits there and 8 is not written yet.
    assert inclusive.external_hit == 0
    assert exclusive.external_hit == 1 / 3
    assert exclusive.l2_written_blocks == 1
