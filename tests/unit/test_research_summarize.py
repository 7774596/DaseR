# SPDX-License-Identifier: Apache-2.0
"""Tests for the research summary's send-order check."""

import json
from pathlib import Path

from benchmarks.research.summarize import order_rows


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
