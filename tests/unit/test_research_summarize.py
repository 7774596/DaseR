# SPDX-License-Identifier: Apache-2.0
"""Tests for the research summary's send-order check and overlap bound."""

import json
from pathlib import Path

from benchmarks.research.summarize import order_rows, overlap_bound_rows


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
