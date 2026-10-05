# SPDX-License-Identifier: Apache-2.0
"""Command-line contract of the research agent-trace runner."""

from pathlib import Path

from benchmarks.research.agent_trace_arms import parse_args


def _args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--trace",
        str(tmp_path / "trace.jsonl"),
        "--artifact",
        str(tmp_path / "art"),
        "--scratch",
        str(tmp_path / "scr"),
        "--model",
        "/models/qwen",
        *extra,
    ]


def test_async_scheduling_defaults_off(tmp_path: Path) -> None:
    """Arms keep vLLM async scheduling off unless asked."""
    assert parse_args(_args(tmp_path)).async_scheduling is False


def test_async_scheduling_flag_enables_it(tmp_path: Path) -> None:
    """``--async-scheduling`` turns it on for every arm."""
    assert parse_args(_args(tmp_path, "--async-scheduling")).async_scheduling is True
