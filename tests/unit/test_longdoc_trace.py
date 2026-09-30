# SPDX-License-Identifier: Apache-2.0
"""Tests for the real-text long-document QA trace builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from benchmarks.mooncake.benchmark import load_trace
from benchmarks.research.longdoc_trace import build_trace, main, parse_stage


def test_trace_shares_document_blocks_and_never_question_blocks() -> None:
    """Each request is one document prefix plus a question nobody else asks."""
    rows, stages = build_trace(
        documents=3,
        document_tokens=1024,
        question_tokens=100,
        output_tokens=8,
        warmup_gap_s=2.0,
        stages=[(4.0, 5.0), (8.0, 5.0)],
        seed=1,
    )

    assert [s["name"] for s in stages] == ["warmup", "qps4", "qps8"]
    assert stages[0] == {"name": "warmup", "start": 0, "end": 3, "qps": None}
    assert stages[1]["end"] == stages[2]["start"] and stages[2]["end"] == len(rows)
    timestamps = [row["timestamp"] for row in rows]
    assert timestamps == sorted(timestamps)
    documents = {tuple(row["hash_ids"][:2]) for row in rows}
    assert len(documents) == 3
    questions = [row["hash_ids"][2] for row in rows]
    assert len(set(questions)) == len(rows)
    assert all(row["input_length"] == 1124 for row in rows)
    # Rate roughly matches: stage 2 sends about twice as many as stage 1.
    assert stages[2]["end"] - stages[2]["start"] > stages[1]["end"] - stages[1]["start"]


def test_trace_rejects_documents_that_split_a_block() -> None:
    """A partial last document block would be shared with the question."""
    with pytest.raises(ValueError, match="multiple of 512"):
        build_trace(
            documents=1,
            document_tokens=700,
            question_tokens=1,
            output_tokens=1,
            warmup_gap_s=1.0,
            stages=[(1.0, 1.0)],
            seed=0,
        )


def test_parse_stage_validates_values() -> None:
    """Stages are positive ``qps:seconds`` pairs."""
    assert parse_stage("0.5:300") == (0.5, 300.0)
    for bad in ("1", "0:10", "a:b"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_stage(bad)


def test_cli_trace_is_loadable_by_the_mooncake_replay(tmp_path: Path) -> None:
    """The emitted trace passes the replay's own validation."""
    main(
        [
            "trace",
            "--output-dir",
            str(tmp_path),
            "--documents",
            "2",
            "--document-tokens",
            "512",
            "--stage",
            "2:3",
        ]
    )

    requests = load_trace(tmp_path / "trace.jsonl")
    stages = json.loads((tmp_path / "stages.json").read_text())
    assert len(requests) == stages[-1]["end"]
