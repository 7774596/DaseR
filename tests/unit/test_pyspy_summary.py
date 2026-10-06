# SPDX-License-Identifier: Apache-2.0
"""Tests for the py-spy raw profile summary."""

from benchmarks.research.pyspy_summary import frame_key, parse, summarize


def test_frame_key_drops_line_number() -> None:
    """Frames from the same function on different lines share one key."""
    assert frame_key("run (transfer/iouring/l2_engine.py:88)") == (
        "run transfer/iouring/l2_engine.py"
    )
    assert frame_key("process 1") == "process 1"


def test_parse_skips_malformed_lines() -> None:
    """Only ``stack count`` lines are kept."""
    assert parse(["a;b 3\n", "\n", "junk\n"]) == [(["a", "b"], 3)]


def test_inclusive_counts_each_frame_once_per_stack() -> None:
    """Recursive DaseR frames count once; non-matching frames are excluded."""
    stacks = parse(
        [
            "proc;f (server/a.py:1);f (server/a.py:2);g (x/y.py:3) 3",
            "proc;h (lib/z.py:4) 1",
        ]
    )
    lines = summarize(stacks)
    assert "incl 0.750 f server/a.py" in lines
    assert not any(line.startswith("incl") and "x/y.py" in line for line in lines)
    assert "self 0.750 g x/y.py" in lines
