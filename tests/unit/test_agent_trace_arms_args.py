# SPDX-License-Identifier: Apache-2.0
"""Command-line contract of the research agent-trace runner."""

import json
from pathlib import Path

import pytest

from benchmarks.research.agent_trace_arms import _manager, parse_args


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


def test_speculative_config_reaches_every_arm_as_one_argument(tmp_path: Path) -> None:
    """The JSON goes to vllm serve as one argv item, for native and DaseR arms."""
    spec = '{"method": "ngram_gpu", "num_speculative_tokens": 5}'
    args = parse_args(_args(tmp_path, "--speculative-config", spec))

    for arm in ("native-apc", "daser-compressed"):
        extra = list(_manager(arm, args).extra_vllm_args)
        flag = extra.index("--speculative-config")
        assert json.loads(extra[flag + 1]) == json.loads(spec)


def test_speculative_config_defaults_off(tmp_path: Path) -> None:
    """Without the flag no arm enables speculation."""
    args = parse_args(_args(tmp_path))

    assert "--speculative-config" not in _manager("native-apc", args).extra_vllm_args


def test_speculative_config_rejects_non_objects(tmp_path: Path) -> None:
    """A malformed config fails at parse time, before any server starts."""
    with pytest.raises(SystemExit):
        parse_args(_args(tmp_path, "--speculative-config", "[5]"))
