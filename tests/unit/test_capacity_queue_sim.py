# SPDX-License-Identifier: Apache-2.0
"""Tests for the closed-population KV capacity queue simulator."""

from benchmarks.mooncake.benchmark import TraceRequest
from benchmarks.research.capacity_queue_sim import build_sessions, simulate, summarise


def _row(index: int, session: int, length: int, out: int = 10) -> TraceRequest:
    return TraceRequest(
        index=index,
        timestamp_ms=0.0,
        input_length=length,
        output_length=out,
        hash_ids=(index,),
        session_id=session,
        gap_ms=0.0,
    )


def test_ample_capacity_has_no_queue() -> None:
    """With room for every session, admission never waits."""
    rows = [_row(i, i % 3, 100 * (i + 1)) for i in range(9)]
    jobs = simulate(build_sessions(rows, None, 1000.0, 0.01), 3, 10**9, "fcfs")
    assert len(jobs) == 9
    assert summarise(jobs)["sum_queue_s"] == 0.0


def test_tight_capacity_queues_and_reuse_shortens_prefill() -> None:
    """Capacity for one request queues the rest; reuse only cuts prefill."""
    rows = [_row(0, 0, 1000), _row(1, 1, 1000), _row(2, 0, 1500)]
    tight = summarise(
        simulate(build_sessions(rows, None, 1000.0, 0.01), 2, 1100, "fcfs")
    )
    assert tight["sum_queue_s"] > 0.0
    cold = build_sessions(rows, 0.0, 1000.0, 0.01)
    warm = build_sessions(rows, None, 1000.0, 0.01)
    assert warm[0][1].prefill_s < cold[0][1].prefill_s
    assert warm[0][1].kv_tokens == cold[0][1].kv_tokens


def test_srpt_admits_the_shorter_waiting_job_first() -> None:
    """Under SRPT a small request that fits passes a blocked head of line."""
    rows = [_row(0, 0, 900), _row(1, 1, 900, out=1000), _row(2, 2, 89, out=1)]

    def admit(policy: str) -> dict[int, float]:
        jobs = simulate(build_sessions(rows, 0.0, 1000.0, 0.01), 3, 1000, policy)
        return {j.index: j.admit_s for j in jobs}

    srpt, fcfs = admit("srpt"), admit("fcfs")
    assert srpt[2] == 0.0 and srpt[1] > 0.0
    # FCFS: request 1 (ready first, does not fit) blocks request 2.
    assert fcfs[2] >= fcfs[1] > 0.0
