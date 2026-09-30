# SPDX-License-Identifier: Apache-2.0
"""Tests for the Mooncake production trace benchmark."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.mooncake import benchmark
from benchmarks.mooncake.benchmark import (
    MOONCAKE_BLOCK_TOKENS,
    CorpusPromptBuilder,
    PromptBuilder,
    TraceRequest,
    load_trace,
    prefix_predecessors,
    replay,
    scheduled_offset_seconds,
    select_requests,
    summarise,
)
from benchmarks.utils.loadgen import RequestResult


def _write_trace(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_trace_validates_order_and_block_count(tmp_path: Path) -> None:
    """Rows must be source ordered and contain one hash per 512-token block."""
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            {
                "timestamp": 0,
                "input_length": 513,
                "output_length": 4,
                "hash_ids": [10, 11],
            },
            {
                "timestamp": 3,
                "input_length": 512,
                "output_length": 2,
                "hash_ids": [10],
            },
        ],
    )

    requests = load_trace(trace)

    assert [request.index for request in requests] == [0, 1]
    assert requests[0].hash_ids == (10, 11)

    _write_trace(
        trace,
        [
            {
                "timestamp": 3,
                "input_length": 512,
                "output_length": 1,
                "hash_ids": [1],
            },
            {
                "timestamp": 2,
                "input_length": 512,
                "output_length": 1,
                "hash_ids": [2],
            },
        ],
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        load_trace(trace)


def test_load_trace_selects_contiguous_rebased_window(tmp_path: Path) -> None:
    """A source window preserves row IDs and relative arrival gaps."""
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            {
                "timestamp": index * 10,
                "input_length": 1,
                "output_length": 1,
                "hash_ids": [index],
            }
            for index in range(4)
        ],
    )

    requests = load_trace(trace, max_requests=2, start_request=1)

    assert [request.index for request in requests] == [1, 2]
    assert [request.timestamp_ms for request in requests] == [0, 10]


def test_load_trace_filters_context_stratum_before_limit(tmp_path: Path) -> None:
    """Context filtering keeps source order and limits retained rows."""
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            {
                "timestamp": index * 10,
                "input_length": length,
                "output_length": 1,
                "hash_ids": [index] * ((length + 511) // 512),
            }
            for index, length in enumerate((10, 600, 20, 700))
        ],
    )

    requests = load_trace(trace, max_requests=2, min_context_tokens=500)

    assert [request.index for request in requests] == [1, 3]
    assert [request.timestamp_ms for request in requests] == [0, 20]


def test_prompt_builder_preserves_full_and_partial_shared_blocks() -> None:
    """Equal hash prefixes yield equal tokens, including a partial last block."""
    builder = PromptBuilder(vocab_size=1024, special_token_ids={0, 1, 2})
    first = TraceRequest(0, 0, 700, 1, (10, 11))
    second = TraceRequest(1, 0, 900, 1, (10, 11))
    different = TraceRequest(2, 0, 700, 1, (10, 12))

    first_prompt = builder.build(first)
    second_prompt = builder.build(second)
    different_prompt = builder.build(different)

    assert len(first_prompt) == 700
    assert first_prompt == second_prompt[:700]
    assert (
        first_prompt[:MOONCAKE_BLOCK_TOKENS] == different_prompt[:MOONCAKE_BLOCK_TOKENS]
    )
    assert (
        first_prompt[MOONCAKE_BLOCK_TOKENS:] != different_prompt[MOONCAKE_BLOCK_TOKENS:]
    )
    assert not ({0, 1, 2} & set(first_prompt))


def test_arrival_scaling_preserves_same_timestamp_bursts() -> None:
    """Scaling changes time units without separating simultaneous arrivals."""
    assert scheduled_offset_seconds(3000, 1.0) == 3.0
    assert scheduled_offset_seconds(3000, 2.0) == 1.5
    assert scheduled_offset_seconds(3000, 4.0) == scheduled_offset_seconds(3000, 4.0)
    with pytest.raises(ValueError, match="positive"):
        scheduled_offset_seconds(0, 0)


def test_context_selection_and_summary_report_skipped_token_mass() -> None:
    """Overflow rows remain visible and successful latency excludes failures."""
    requests = [
        TraceRequest(0, 0, 1000, 10, (1, 2)),
        TraceRequest(1, 1, 2000, 100, (3, 4, 5, 6)),
    ]
    selection = select_requests(requests, max_model_len=1500)

    summary = summarise(
        selection,
        [
            {
                "error": None,
                "ttft_ms": 10,
                "latency_ms": 20,
                "client_queue_ms": 1,
                "arrival_to_first_token_ms": 12,
                "arrival_to_completion_ms": 22,
            },
            {
                "error": "failed",
                "ttft_ms": 0,
                "latency_ms": 0,
                "client_queue_ms": 0,
                "arrival_to_first_token_ms": 0,
                "arrival_to_completion_ms": 0,
            },
        ],
        wall_seconds=2,
        time_scale=1,
        prefix_cache_metrics={
            "vllm:prefix_cache_queries_total": 100,
            "vllm:prefix_cache_hits_total": 60,
        },
    )

    assert [request.index for request in selection.eligible] == [0]
    assert [request.index for request in selection.skipped] == [1]
    assert summary["skipped_context_requests"] == 1
    assert summary["skipped_input_tokens"] == 2000
    assert summary["completed_requests"] == 1
    assert summary["failed_requests"] == 1
    assert summary["ttft_ms"]["p99"] == 10
    assert summary["arrival_to_first_token_ms"]["p99"] == 12
    assert summary["prefix_cache"]["local_hit_rate"] == 0.6


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {"timestamp": 0, "input_length": 0, "output_length": 1, "hash_ids": []},
            "positive",
        ),
        (
            {"timestamp": 0, "input_length": 513, "output_length": 1, "hash_ids": [1]},
            "expected 2 hash IDs",
        ),
        (
            {"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": [-1]},
            "non-negative",
        ),
    ],
)
def test_load_trace_rejects_invalid_rows(
    tmp_path: Path, row: dict[str, object], message: str
) -> None:
    """Invalid lengths and hash metadata fail before replay begins."""
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, [row])

    with pytest.raises(ValueError, match=message):
        load_trace(trace)


def test_replay_aborts_after_failure_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay cancels outstanding requests once the failure budget is spent."""

    async def failing_stream(*args: Any, **kwargs: Any) -> RequestResult:
        return RequestResult(0, "mooncake-toolagent", "", 0.0, 0.0, 0, 0, "boom")

    monkeypatch.setattr(benchmark, "vllm_completion_stream", failing_stream)
    requests = [TraceRequest(i, i * 20, 10, 1, (i,)) for i in range(50)]
    selection = select_requests(requests, max_model_len=100)

    results, _, aborted = asyncio.run(
        replay(
            selection=selection,
            prompt_builder=PromptBuilder(vocab_size=1000, special_token_ids=[]),
            server_url="http://unused",
            served_model_name="m",
            output_path=tmp_path / "requests.jsonl",
            max_inflight=1,
            timeout_seconds=1.0,
            time_scale=1.0,
            seed=0,
            abort_after_failures=3,
        )
    )

    assert aborted
    # Requests finishing in the same wakeup may overshoot the budget slightly.
    assert 3 <= len(results) < len(requests)
    assert summarise(selection, results, 1.0, 1.0, aborted=aborted)["aborted"]


def test_prefix_predecessor_is_the_producer_of_the_deepest_shared_block() -> None:
    """Turns chain to the previous turn; a shared system block does not serialize."""
    requests = [
        TraceRequest(0, 0, 10, 1, (1, 2)),  # session A turn 1
        TraceRequest(1, 0, 10, 1, (1, 3)),  # session B turn 1, shares block 1
        TraceRequest(2, 0, 10, 1, (1, 2, 4)),  # session A turn 2
        TraceRequest(3, 0, 10, 1, (1, 5)),  # session C, shares block 1 only
        TraceRequest(4, 0, 10, 1, (1, 2, 4, 6)),  # session A turn 3
        TraceRequest(5, 0, 10, 1, (9,)),  # unrelated
    ]

    assert prefix_predecessors(requests) == [None, 0, 0, 0, 2, None]


def test_replay_sends_request_after_its_prefix_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With prefix dependencies a later turn is sent only after the earlier one ends."""
    events: list[tuple[str, int]] = []

    async def stream(*args: Any, **kwargs: Any) -> RequestResult:
        sample = args[2]
        kwargs["on_send"]()
        events.append(("send", sample.sample_id))
        await asyncio.sleep(0.05 if sample.sample_id == 0 else 0.0)
        events.append(("done", sample.sample_id))
        # The first turn fails: its successor must still be released.
        error = "boom" if sample.sample_id == 0 else None
        return RequestResult(sample.sample_id, "d", "", 1.0, 1.0, 1, 1, error)

    monkeypatch.setattr(benchmark, "vllm_completion_stream", stream)
    requests = [
        TraceRequest(0, 0, 10, 1, (1,)),
        TraceRequest(1, 0, 10, 1, (1, 2)),
        TraceRequest(2, 0, 10, 1, (7,)),
    ]
    results, _, _ = asyncio.run(
        replay(
            selection=select_requests(requests, max_model_len=100),
            prompt_builder=PromptBuilder(vocab_size=1000, special_token_ids=[]),
            server_url="http://unused",
            served_model_name="m",
            output_path=tmp_path / "requests.jsonl",
            max_inflight=8,
            timeout_seconds=1.0,
            time_scale=1e6,
            seed=0,
            prefix_dependencies=True,
        )
    )

    assert events.index(("done", 0)) < events.index(("send", 1))
    # The independent request does not wait for the slow one.
    assert events.index(("send", 2)) < events.index(("done", 0))
    by_index = {record["trace_index"]: record for record in results}
    assert by_index[1]["dependency_wait_ms"] > 0
    assert by_index[2]["dependency_wait_ms"] == pytest.approx(0, abs=5)


def test_corpus_prompt_builder_uses_real_text_and_keeps_hash_identity() -> None:
    """Blocks carry corpus text; equal hashes match and distinct hashes differ."""
    corpus = list(range(5000, 5000 + 3 * MOONCAKE_BLOCK_TOKENS))
    builder = CorpusPromptBuilder(
        vocab_size=1024, special_token_ids={0, 1, 2}, corpus=corpus
    )
    first = builder.build(TraceRequest(0, 0, 1024, 1, (0, 1)))
    again = builder.build(TraceRequest(1, 0, 700, 1, (0, 1)))
    wrapped = builder.build(TraceRequest(2, 0, 512, 1, (3,)))  # same window as 0

    assert first[:700] == again
    # After the hash marker the block is the corpus window verbatim.
    assert first[3:MOONCAKE_BLOCK_TOKENS] == corpus[3:MOONCAKE_BLOCK_TOKENS]
    assert (
        first[MOONCAKE_BLOCK_TOKENS + 3 :] == corpus[MOONCAKE_BLOCK_TOKENS + 3 : 1024]
    )
    assert wrapped[3:] == first[3:MOONCAKE_BLOCK_TOKENS]
    assert wrapped[:3] != first[:3]
