# SPDX-License-Identifier: Apache-2.0
from benchmarks.research.clamp_idle import clamp_rows, idle_cuts


def test_idle_cuts_only_remove_time_with_nothing_in_flight() -> None:
    # Requests: [0, 10], [5, 20] overlap; idle 20 -> 200; [200, 210]; idle 210 -> 240.
    intervals = [(0.0, 10.0), (5.0, 20.0), (200.0, 210.0), (240.0, 250.0)]
    assert idle_cuts(intervals, 60.0) == [(20.0, 120.0)]
    assert idle_cuts(intervals, 10.0) == [(20.0, 170.0), (210.0, 20.0)]


def test_idle_cuts_union_of_runs_keeps_slower_run_busy() -> None:
    # A slower reference still runs until 150, so only 150 -> 200 is idle.
    fast = [(0.0, 20.0), (200.0, 210.0)]
    slow = [(0.0, 150.0), (200.0, 260.0)]
    assert idle_cuts(fast + slow, 30.0) == [(150.0, 20.0)]


def test_clamp_rows_shifts_later_timestamps_and_keeps_order_and_gaps() -> None:
    rows = [
        {"timestamp": 1000, "gap_ms": 0},
        {"timestamp": 15000, "gap_ms": 4000},
        {"timestamp": 201000, "gap_ms": 0},
        {"timestamp": 205000, "gap_ms": 3000},
    ]
    # Row 1 is gap-paced: recorded at 15 s but sent at 12 s, before the cut.
    out = clamp_rows(rows, [(20.0, 120.0)], send_offsets=[0.0, 12.0, 200.0, 204.0])
    assert [r["timestamp"] for r in out] == [1000, 15000, 81000, 85000]
    assert [r["gap_ms"] for r in out] == [0, 4000, 0, 3000]
    assert rows[2]["timestamp"] == 201000  # input rows untouched


def test_clamp_rows_uses_send_time_not_recorded_timestamp() -> None:
    # A gap-paced row recorded after the cut start but sent before it stays put.
    rows = [{"timestamp": 0}, {"timestamp": 30000}, {"timestamp": 300000}]
    out = clamp_rows(rows, [(25.0, 200.0)], send_offsets=[0.0, 20.0, 290.0])
    assert [r["timestamp"] for r in out] == [0, 30000, 100000]
