# SPDX-License-Identifier: Apache-2.0
"""Tests for the research summary's send-order check and overlap bound."""

import json
from pathlib import Path

from benchmarks.research.summarize import (
    live_clock_rows,
    order_rows,
    overlap_bound_rows,
    request_hit_rows,
)


def _arm(root: Path, name: str, send_offsets: list[float]) -> Path:
    """Write a requests.jsonl whose trace index i was sent at send_offsets[i]."""
    arm_dir = root / name
    arm_dir.mkdir()
    with (arm_dir / "requests.jsonl").open("w") as f:
        for index, offset in enumerate(send_offsets):
            f.write(json.dumps({"trace_index": index, "send_offset_s": offset}) + "\n")
    return arm_dir


def test_order_check_flags_only_reordered_replays(tmp_path: Path) -> None:
    """In-order replays pass; a swapped pair is counted and flagged."""
    in_order = _arm(tmp_path, "in", [0.0, 1.0, 2.0, 3.0])
    swapped = _arm(tmp_path, "swapped", [0.0, 2.0, 1.0, 3.0])

    good, bad = order_rows([in_order, swapped])

    assert "out_of_order=0/4" in good and "ANOMALY" not in good
    assert "out_of_order=2/4" in bad and "ANOMALY" in bad


def _ttft_arm(root: Path, name: str, ttft_ms: list[float]) -> Path:
    """Write a completed requests.jsonl whose trace index i had ttft_ms[i]."""
    arm_dir = root / name / "art" / "arm"
    arm_dir.mkdir(parents=True)
    with (arm_dir / "requests.jsonl").open("w") as f:
        for index, ttft in enumerate(ttft_ms):
            record = {"trace_index": index, "status": "completed", "ttft_ms": ttft}
            f.write(json.dumps(record) + "\n")
    return arm_dir


def test_overlap_bound_hides_at_most_the_oracle_work(tmp_path: Path) -> None:
    """Each request keeps max(oracle, extra wait), never more than its TTFT."""
    oracle = _ttft_arm(tmp_path, "oracle", [1000.0, 1000.0, 1000.0])
    # Extra waits 500 (fully hidden), 3000 (only 1000 hidden), and a request
    # faster than the oracle (noise, kept as measured).
    arm = _ttft_arm(tmp_path, "arm", [1500.0, 4000.0, 900.0])

    header, row = overlap_bound_rows([arm], oracle, first=3)

    assert "on 3 requests" in header and "oracle=3" in header
    assert "ttft=6 " in row
    assert "extra=3.5 " in row
    assert "bound=5 " in row  # 1000 + 3000 + 900
    assert "hidden<=1.5" in row


def test_live_clock_check_flags_poll_counting(tmp_path: Path) -> None:
    """One tick per request passes; a clock counting polls is flagged."""
    arms = []
    for name, ticks in (("ok", 21), ("polls", 80), ("off", None)):
        arm_dir = _arm(tmp_path, name, [float(i) for i in range(20)])
        if ticks is not None:
            (arm_dir / "daser_metrics_before.prom").write_text(
                "daser_arena_live_tick 1.0\n"
            )
            (arm_dir / "daser_metrics_after.prom").write_text(
                f"daser_arena_live_tick {ticks}.0\n"
            )
        arms.append(arm_dir)

    good, bad = live_clock_rows(arms)

    assert "live_ticks=20 requests=20" in good and "ANOMALY" not in good
    assert "live_ticks=79" in bad and "ANOMALY" in bad


def test_request_hit_check_reconciles_cached_tokens(tmp_path: Path) -> None:
    """Per-request cached tokens matching the counters pass; a gap is flagged."""
    arms = []
    for name, cached in (("ok", [0, 80, 90]), ("gap", [0, 0, 0]), ("none", None)):
        arm_dir = tmp_path / name
        arm_dir.mkdir()
        with (arm_dir / "requests.jsonl").open("w") as f:
            for i in range(3):
                record = {"trace_index": i, "prompt_tokens": 100}
                if cached is not None:
                    record["cached_tokens"] = cached[i]
                f.write(json.dumps(record) + "\n")
        arms.append(arm_dir)

    good, bad = request_hit_rows(arms, ["0.567", "0.567", "0.567"])

    assert "n=3 request_hit=0.567" in good and "ANOMALY" not in good
    assert "request_hit=0.000" in bad and "ANOMALY" in bad
