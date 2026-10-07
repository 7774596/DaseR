# SPDX-License-Identifier: Apache-2.0
"""Tests for the Mooncake production trace benchmark."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from typing import Any

import pytest

from benchmarks.mooncake import benchmark
from benchmarks.mooncake.benchmark import (
    MOONCAKE_BLOCK_TOKENS,
    CorpusPromptBuilder,
    InstanceRouter,
    PromptBuilder,
    SessionTextPromptBuilder,
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


def test_strict_order_replays_in_trace_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict order sends requests one after another in trace order.

    Prefix dependencies alone release every session's first turn at once,
    so the trace interleaving is lost; strict order keeps it.
    """
    sent: list[int] = []

    async def stream(*args: Any, **kwargs: Any) -> RequestResult:
        sample = args[2]
        kwargs["on_send"]()
        sent.append(sample.sample_id)
        await asyncio.sleep(0.01)
        return RequestResult(sample.sample_id, "d", "", 1.0, 1.0, 1, 1, None)

    monkeypatch.setattr(benchmark, "vllm_completion_stream", stream)
    # Two sessions; the trace finishes session A before session B starts.
    requests = [
        TraceRequest(0, 0, 10, 1, (1,)),
        TraceRequest(1, 1, 10, 1, (1, 2)),
        TraceRequest(2, 2, 10, 1, (5,)),
        TraceRequest(3, 3, 10, 1, (5, 6)),
    ]
    asyncio.run(
        replay(
            selection=select_requests(requests, max_model_len=100),
            prompt_builder=PromptBuilder(vocab_size=1000, special_token_ids=[]),
            server_url="http://unused",
            served_model_name="m",
            output_path=tmp_path / "requests.jsonl",
            max_inflight=1,
            timeout_seconds=1.0,
            time_scale=1e6,
            seed=0,
            prefix_dependencies=True,
            strict_order=True,
        )
    )

    # Prefix dependencies alone would send session B's first turn before A's
    # second one.
    assert sent == [0, 1, 2, 3]


def test_ordered_admission_overlaps_requests_in_trace_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordered admission sends in trace order and still overlaps sessions."""
    sent: list[int] = []
    inflight = 0
    peak = 0

    async def stream(*args: Any, **kwargs: Any) -> RequestResult:
        nonlocal inflight, peak
        sample = args[2]
        kwargs["on_send"]()
        sent.append(sample.sample_id)
        inflight += 1
        peak = max(peak, inflight)
        # Later requests finish first, so completion order is not send order.
        await asyncio.sleep(0.05 - 0.01 * sample.sample_id)
        inflight -= 1
        return RequestResult(sample.sample_id, "d", "", 1.0, 1.0, 1, 1, None)

    monkeypatch.setattr(benchmark, "vllm_completion_stream", stream)
    # Two interleaved sessions: A, B, A's second turn, B's second turn.
    requests = [
        TraceRequest(0, 0, 10, 1, (1,)),
        TraceRequest(1, 1, 10, 1, (5,)),
        TraceRequest(2, 2, 10, 1, (1, 2)),
        TraceRequest(3, 3, 10, 1, (5, 6)),
    ]
    asyncio.run(
        replay(
            selection=select_requests(requests, max_model_len=100),
            prompt_builder=PromptBuilder(vocab_size=1000, special_token_ids=[]),
            server_url="http://unused",
            served_model_name="m",
            output_path=tmp_path / "requests.jsonl",
            max_inflight=2,
            timeout_seconds=1.0,
            time_scale=1e6,
            seed=0,
            prefix_dependencies=True,
            ordered_admission=True,
        )
    )

    # B's first turn finishes before A's, but B's second turn still waits for
    # A's second turn to be admitted first.
    assert sent == [0, 1, 2, 3]
    assert peak == 2


def test_strict_order_excludes_ordered_admission(tmp_path: Path) -> None:
    """The two ordering modes cannot be combined."""
    with pytest.raises(ValueError, match="exclusive"):
        asyncio.run(
            replay(
                selection=select_requests([], max_model_len=100),
                prompt_builder=PromptBuilder(vocab_size=1000, special_token_ids=[]),
                server_url="http://unused",
                served_model_name="m",
                output_path=tmp_path / "requests.jsonl",
                max_inflight=2,
                timeout_seconds=1.0,
                time_scale=1e6,
                seed=0,
                strict_order=True,
                ordered_admission=True,
            )
        )


def test_dependency_gap_idles_before_the_dependent_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gap counts as dependency wait, not as client queue time."""

    async def stream(*args: Any, **kwargs: Any) -> RequestResult:
        sample = args[2]
        kwargs["on_send"]()
        return RequestResult(sample.sample_id, "d", "", 1.0, 1.0, 1, 1, None)

    monkeypatch.setattr(benchmark, "vllm_completion_stream", stream)
    requests = [TraceRequest(0, 0, 10, 1, (1,)), TraceRequest(1, 1, 10, 1, (1, 2))]
    results, _, _ = asyncio.run(
        replay(
            selection=select_requests(requests, max_model_len=100),
            prompt_builder=PromptBuilder(vocab_size=1000, special_token_ids=[]),
            server_url="http://unused",
            served_model_name="m",
            output_path=tmp_path / "requests.jsonl",
            max_inflight=1,
            timeout_seconds=1.0,
            time_scale=1e6,
            seed=0,
            strict_order=True,
            dependency_gap_s=0.1,
        )
    )

    by_index = {record["trace_index"]: record for record in results}
    assert by_index[0]["dependency_wait_ms"] == pytest.approx(0, abs=5)
    assert by_index[1]["dependency_wait_ms"] >= 100
    assert by_index[1]["client_queue_ms"] < 50


def test_recorded_gap_paces_dependents_instead_of_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dependent row with gap_ms starts that gap after its producer ends.

    Its own timestamp is ignored, so slow service does not swallow the gap;
    a row without a producer still waits for its timestamp.
    """
    sent_at: dict[int, float] = {}
    done_at: dict[int, float] = {}

    async def stream(*args: Any, **kwargs: Any) -> RequestResult:
        sample = args[2]
        kwargs["on_send"]()
        sent_at[sample.sample_id] = time.perf_counter()
        await asyncio.sleep(0.15 if sample.sample_id == 0 else 0.0)
        done_at[sample.sample_id] = time.perf_counter()
        return RequestResult(sample.sample_id, "d", "", 1.0, 1.0, 1, 1, None)

    monkeypatch.setattr(benchmark, "vllm_completion_stream", stream)
    requests = [
        TraceRequest(0, 0, 10, 1, (1,)),
        # Timestamp 10 ms (before the producer ends), gap 50 ms.
        TraceRequest(1, 10, 10, 1, (1, 2), gap_ms=50),
        # No producer: paced by its 100 ms timestamp alone.
        TraceRequest(2, 100, 10, 1, (7,), gap_ms=0),
    ]
    started = time.perf_counter()
    asyncio.run(
        replay(
            selection=select_requests(requests, max_model_len=100),
            prompt_builder=PromptBuilder(vocab_size=1000, special_token_ids=[]),
            server_url="http://unused",
            served_model_name="m",
            output_path=tmp_path / "requests.jsonl",
            max_inflight=8,
            timeout_seconds=1.0,
            time_scale=1.0,
            seed=0,
            prefix_dependencies=True,
        )
    )

    assert sent_at[1] - done_at[0] >= 0.05
    assert sent_at[1] - done_at[0] < 0.1
    assert sent_at[2] - started >= 0.1
    assert sent_at[2] < done_at[0]


def test_load_trace_reads_optional_gap(tmp_path: Path) -> None:
    """gap_ms is optional and must be non-negative."""
    path = tmp_path / "trace.jsonl"
    row = {"timestamp": 0, "input_length": 10, "output_length": 1, "hash_ids": [1]}
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "gap_ms": 7}) + "\n")
    first, second = load_trace(path)
    assert first.gap_ms is None
    assert second.gap_ms == 7

    path.write_text(json.dumps({**row, "gap_ms": -1}) + "\n")
    with pytest.raises(ValueError, match="gap_ms"):
        load_trace(path)


def test_load_trace_bounds_prewarm_tokens(tmp_path: Path) -> None:
    """prewarm_tokens is optional and must lie within the prompt."""
    path = tmp_path / "trace.jsonl"
    row = {"timestamp": 0, "input_length": 10, "output_length": 1, "hash_ids": [1]}
    path.write_text(json.dumps({**row, "prewarm_tokens": 8}) + "\n")
    assert load_trace(path)[0].prewarm_tokens == 8

    path.write_text(json.dumps({**row, "prewarm_tokens": 11}) + "\n")
    with pytest.raises(ValueError, match="prewarm_tokens"):
        load_trace(path)


def _prewarm_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prewarm: bool,
    warm_s: float,
    chunk_tokens: int = 0,
    max_waiting: int | None = None,
) -> tuple[list[tuple[int, int, str | None, float]], dict[int, dict]]:
    """Replay a producer and a gap-paced user turn with a known prefix."""
    sends: list[tuple[int, int, str | None, float]] = []

    async def stream(*args: Any, **kwargs: Any) -> RequestResult:
        index, prompt, params = args[2].sample_id, args[3], args[4]
        sends.append(
            (index, len(prompt), params.get("request_id"), time.perf_counter())
        )
        if kwargs.get("on_send"):
            kwargs["on_send"]()
        await asyncio.sleep(warm_s if params["max_tokens"] == 1 else 0.0)
        return RequestResult(index, "d", "", 1.0, 2.0, len(prompt), 1)

    monkeypatch.setattr(benchmark, "vllm_completion_stream", stream)
    requests = [
        TraceRequest(0, 0, 1024, 8, (1, 2), session_id=0),
        TraceRequest(
            1, 0, 2048, 8, (1, 3, 4, 5), session_id=0, gap_ms=50, prewarm_tokens=1100
        ),
    ]
    results, _, _ = asyncio.run(
        replay(
            selection=select_requests(requests, max_model_len=100_000),
            prompt_builder=PromptBuilder(vocab_size=1000, special_token_ids=[]),
            server_url="http://a",
            served_model_name="m",
            output_path=tmp_path / "requests.jsonl",
            max_inflight=1,
            timeout_seconds=1.0,
            time_scale=1.0,
            seed=0,
            prefix_dependencies=True,
            prewarm=prewarm,
            prewarm_chunk_tokens=chunk_tokens,
            prewarm_max_waiting=max_waiting,
        )
    )
    return sends, {record["trace_index"]: record for record in results}


def test_prewarm_sends_the_known_prefix_during_the_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prefix goes out when the producer ends; the turn follows its gap."""
    sends, by_index = _prewarm_replay(tmp_path, monkeypatch, True, warm_s=0.0)

    assert [(i, n, rid) for i, n, rid, _ in sends] == [
        (0, 1024, None),
        (1, 1100, "trace-1-warm"),
        (1, 2048, None),
    ]
    assert sends[2][3] - sends[1][3] >= 0.05
    assert by_index[1]["prewarm_error"] is None
    assert by_index[1]["ttft_ms"] < 5.0
    assert "prewarm_ms" not in by_index[0]


def test_prewarm_outlasting_the_gap_counts_toward_ttft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn waits for its prewarm, and that extra wait is part of TTFT."""
    _, by_index = _prewarm_replay(tmp_path, monkeypatch, True, warm_s=0.2)
    assert by_index[1]["prewarm_ms"] >= 200.0
    assert 140.0 <= by_index[1]["ttft_ms"] < 200.0

    sends, by_index = _prewarm_replay(tmp_path, monkeypatch, False, warm_s=0.2)
    assert [i for i, *_ in sends] == [0, 1]
    assert "prewarm_ms" not in by_index[1]


def test_chunked_prewarm_sends_growing_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chunked prewarm covers the known prefix in growing pieces."""
    sends, by_index = _prewarm_replay(
        tmp_path, monkeypatch, True, warm_s=0.0, chunk_tokens=400
    )
    assert [(i, n) for i, n, *_ in sends] == [
        (0, 1024),
        (1, 400),
        (1, 800),
        (1, 1100),
        (1, 2048),
    ]
    assert by_index[1]["prewarm_error"] is None


def test_chunked_prewarm_stops_at_the_end_of_the_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No new piece starts after the gap; the turn prefills the rest."""
    sends, by_index = _prewarm_replay(
        tmp_path, monkeypatch, True, warm_s=0.035, chunk_tokens=400
    )
    assert [(i, n) for i, n, *_ in sends] == [(0, 1024), (1, 400), (1, 800), (1, 2048)]
    assert by_index[1]["ttft_ms"] < 40.0


def test_prewarm_yields_once_the_server_queue_exceeds_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A piece starts only while vLLM's waiting queue is within the bound."""
    queue = iter([0.0, 3.0])

    async def waiting(*args: Any) -> float:
        return next(queue)

    monkeypatch.setattr(benchmark, "_waiting_requests", waiting)
    sends, by_index = _prewarm_replay(
        tmp_path, monkeypatch, True, warm_s=0.0, chunk_tokens=400, max_waiting=2
    )

    assert [(i, n) for i, n, *_ in sends] == [(0, 1024), (1, 400), (1, 2048)]
    assert by_index[1]["prewarm_pieces"] == 1
    assert by_index[1]["prewarm_yielded"] is True

    sends, by_index = _prewarm_replay(
        tmp_path, monkeypatch, True, warm_s=0.0, chunk_tokens=400
    )
    assert len(sends) == 5
    assert by_index[1]["prewarm_pieces"] == 3
    assert by_index[1]["prewarm_yielded"] is False


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


def test_load_trace_reads_optional_session_id(tmp_path: Path) -> None:
    """Rows may carry a session_id; rows without one keep None."""
    trace = tmp_path / "trace.jsonl"
    base = {"input_length": 10, "output_length": 1, "hash_ids": [1]}
    _write_trace(
        trace,
        [{"timestamp": 0, "session_id": 7, **base}, {"timestamp": 1, **base}],
    )

    assert [row.session_id for row in load_trace(trace)] == [7, None]


def _session_request(index: int, session: int, blocks: int) -> TraceRequest:
    hash_ids = tuple(session * 1000 + block for block in range(blocks))
    return TraceRequest(
        index, 0, blocks * MOONCAKE_BLOCK_TOKENS, 8, hash_ids, session_id=session
    )


def test_sticky_route_keeps_a_session_on_its_first_instance() -> None:
    """Later turns stay put even when their instance carries more work."""
    router = InstanceRouter(["http://a", "http://b"], route="sticky")
    first, _ = router.acquire(_session_request(0, session=1, blocks=64))
    other, _ = router.acquire(_session_request(1, session=2, blocks=1))
    later, _ = router.acquire(_session_request(2, session=1, blocks=65))

    assert (first, other, later) == (0, 1, 0)


def test_least_work_route_follows_outstanding_work_across_sessions() -> None:
    """A heavy in-flight request pushes the next turn of its session away."""
    router = InstanceRouter(["http://a", "http://b"], route="least-work")
    first_request = _session_request(0, session=1, blocks=64)
    first, first_work = router.acquire(first_request)
    later, _ = router.acquire(_session_request(1, session=1, blocks=65))
    router.release(first, first_work, first_request)
    after, _ = router.acquire(_session_request(2, session=3, blocks=1))

    assert (first, later, after) == (0, 1, 0)


def test_router_charges_less_work_for_a_seen_prefix() -> None:
    """A repeated prefix is predicted cached, so its prefill is not charged."""
    router = InstanceRouter(["http://a"], route="least-work")
    request = _session_request(0, session=1, blocks=32)
    _, cold = router.acquire(request)
    _, warm = router.acquire(_session_request(1, session=1, blocks=32))

    assert warm < cold / 10


def _route_after_busy_holder(route: str) -> int:
    """Route a prefix extension whose holder is busy with a smaller job."""
    router = InstanceRouter(["http://a", "http://b"], route=route)
    first_request = _session_request(0, session=1, blocks=64)
    first, first_work = router.acquire(first_request)
    router.release(first, first_work, first_request)
    router.acquire(_session_request(1, session=2, blocks=8))
    later, _ = router.acquire(_session_request(2, session=1, blocks=65))
    return later


def test_gpu_affinity_follows_the_instance_holding_the_prefix() -> None:
    """Recompute elsewhere costs more than queueing behind the smaller job."""
    assert _route_after_busy_holder("gpu-affinity") == 0


def test_tier_affinity_balances_when_the_prefix_is_reusable_elsewhere() -> None:
    """A shared-tier hit costs only a load, so the idle instance wins."""
    assert _route_after_busy_holder("tier-affinity") == 1


def test_tier_affinity_breaks_load_ties_by_locality() -> None:
    """With equal outstanding work, the instance holding the prefix is chosen."""
    router = InstanceRouter(["http://a", "http://b"], route="tier-affinity")
    first_request = _session_request(0, session=1, blocks=8)
    other_request = _session_request(1, session=2, blocks=8)
    first, first_work = router.acquire(first_request)
    other, other_work = router.acquire(other_request)
    router.release(first, first_work, first_request)
    router.release(other, other_work, other_request)
    later, _ = router.acquire(_session_request(2, session=2, blocks=9))

    assert (first, other, later) == (0, 1, 1)


def test_affinity_shadow_cache_forgets_evicted_prefixes() -> None:
    """Blocks pushed out of the shadow capacity no longer attract the request."""

    def route_return(local_blocks: int) -> int:
        router = InstanceRouter(
            ["http://a", "http://b"], route="gpu-affinity", local_blocks=local_blocks
        )
        for index, session, blocks in ((0, 1, 64), (1, 3, 64)):
            request = _session_request(index, session, blocks)
            instance, work = router.acquire(request)
            router.release(instance, work, request)
        router.acquire(_session_request(2, session=4, blocks=1))
        later, _ = router.acquire(_session_request(3, session=1, blocks=65))
        return later

    assert route_return(local_blocks=1024) == 0
    assert route_return(local_blocks=64) == 1


def test_router_rejects_bad_affinity_settings() -> None:
    """Non-positive shadow capacity or load rate is rejected."""
    with pytest.raises(ValueError):
        InstanceRouter(["http://a"], route="tier-affinity", local_blocks=0)
    with pytest.raises(ValueError):
        InstanceRouter(["http://a"], route="tier-affinity", load_tokens_per_s=0)


def _subagent_request(index: int, session: int, base: int, blocks: int) -> TraceRequest:
    """Build a request of ``session`` whose blocks start at ``base``."""
    hash_ids = tuple(base * 1000 + block for block in range(blocks))
    return TraceRequest(
        index, 0, blocks * MOONCAKE_BLOCK_TOKENS, 8, hash_ids, session_id=session
    )


def test_hbm_fit_counts_a_shared_in_flight_prefix_once() -> None:
    """A next turn sharing an in-flight prefix fits beside it and stays home."""
    router = InstanceRouter(["http://a", "http://b"], route="hbm-fit", local_blocks=70)
    first, _ = router.acquire(_session_request(0, session=1, blocks=64))
    router.acquire(_session_request(1, session=2, blocks=1))
    later, _ = router.acquire(_session_request(2, session=1, blocks=65))

    assert (first, later) == (0, 0)
    assert router.resident_blocks(0) == 65 + 2


def test_hbm_fit_moves_a_request_that_does_not_fit_beside_in_flight_ones() -> None:
    """Disjoint blocks over capacity go to the instance with room and re-pin."""
    router = InstanceRouter(["http://a", "http://b"], route="hbm-fit", local_blocks=100)
    first, _ = router.acquire(_session_request(0, session=1, blocks=64))
    router.acquire(_session_request(1, session=2, blocks=1))
    moved, _ = router.acquire(_subagent_request(2, session=1, base=5, blocks=50))
    after, _ = router.acquire(_subagent_request(3, session=1, base=6, blocks=1))

    assert (first, moved, after) == (0, 1, 1)


def test_hbm_fit_stays_home_when_the_request_fits_nowhere() -> None:
    """Without room anywhere the session queues on its own instance."""
    router = InstanceRouter(["http://a", "http://b"], route="hbm-fit", local_blocks=80)
    first, _ = router.acquire(_session_request(0, session=1, blocks=64))
    router.acquire(_session_request(1, session=2, blocks=60))
    later, _ = router.acquire(_subagent_request(2, session=1, base=5, blocks=50))

    assert (first, later) == (0, 0)


def test_router_release_frees_resident_blocks() -> None:
    """Finished requests drop their blocks; shared ones stay while referenced."""
    router = InstanceRouter(["http://a"], route="hbm-fit", local_blocks=200)
    a = _session_request(0, session=1, blocks=64)
    b = _session_request(1, session=1, blocks=65)
    _, a_work = router.acquire(a)
    _, b_work = router.acquire(b)
    router.release(0, a_work, a)

    assert router.resident_blocks(0) == 65 + 1
    router.release(0, b_work, b)
    assert router.resident_blocks(0) == 0


def test_kv_cost_prefers_the_cached_prefix_and_spreads_new_sessions() -> None:
    """Dynamo-style cost: missing prefill blocks plus potential active blocks."""
    router = InstanceRouter(["http://a", "http://b"], route="kv-cost", local_blocks=200)
    first, _ = router.acquire(_session_request(0, session=1, blocks=64))
    later, _ = router.acquire(_session_request(1, session=1, blocks=65))
    other, _ = router.acquire(_session_request(2, session=2, blocks=10))

    assert (first, later, other) == (0, 0, 1)


def test_sticky_saturated_drops_affinity_once_home_is_saturated() -> None:
    """Sessions stay home below the work threshold and move above it."""
    for saturation_s, expected in ((1e9, 0), (1e-9, 1)):
        router = InstanceRouter(
            ["http://a", "http://b"],
            route="sticky-saturated",
            saturation_s=saturation_s,
        )
        router.acquire(_session_request(0, session=1, blocks=64))
        later, _ = router.acquire(_session_request(1, session=1, blocks=65))

        assert later == expected
    with pytest.raises(ValueError):
        InstanceRouter(["http://a"], route="sticky-saturated", saturation_s=0)


def _overcommitted_router(**kwargs: Any) -> InstanceRouter:
    """Instance 0 holds two disjoint session-1 requests past its capacity."""
    router = InstanceRouter(
        ["http://a", "http://b"], route="sticky", local_blocks=100, **kwargs
    )
    router.acquire(_session_request(0, session=1, blocks=64))
    router.acquire(_session_request(1, session=2, blocks=1))
    return router


def test_rebind_target_needs_an_overcommitted_home_and_room_elsewhere() -> None:
    """Only a request waiting for GPU KV moves, and only where it fits."""
    router = _overcommitted_router()
    waiting = _subagent_request(2, session=1, base=5, blocks=50)
    assert router.rebind_target(0, waiting) is None
    router.acquire(waiting)

    assert router.rebind_target(0, waiting) == 1
    router.acquire(_subagent_request(3, session=2, base=7, blocks=60))
    assert router.rebind_target(0, waiting) is None


def test_router_move_transfers_the_charge_and_session() -> None:
    """A moved request's blocks leave its old instance and pin the session."""
    router = _overcommitted_router()
    waiting = _subagent_request(2, session=1, base=5, blocks=50)
    _, work = router.acquire(waiting)
    router.move(0, 1, work, waiting)

    assert router.resident_blocks(0) == 64 + 1
    assert router.resident_blocks(1) == 1 + 1 + 50 + 1
    later, _ = router.acquire(_subagent_request(3, session=1, base=6, blocks=1))
    assert later == 1


def test_rebind_wait_counts_only_unproduced_prefill() -> None:
    """The wait budget is the slack plus prefill of tokens never produced."""
    router = _overcommitted_router(rebind_after_s=1.0, prefill_tokens_per_s=512.0)
    assert (
        InstanceRouter(["http://a"]).rebind_wait_s(
            _session_request(0, session=1, blocks=2)
        )
        is None
    )
    # 65 blocks of which 64 were already produced: one block left to prefill.
    assert router.rebind_wait_s(
        _session_request(2, session=1, blocks=65)
    ) == pytest.approx(1.0 + 1.0)
    with pytest.raises(ValueError):
        InstanceRouter(["http://a"], rebind_after_s=-1.0)


def test_replay_rebinds_a_request_waiting_on_an_overcommitted_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The waiting stream is cancelled and resent; TTFT keeps the lost time."""
    sends: list[tuple[int, str, str | None]] = []
    resent = asyncio.Event()

    async def stream(*args: Any, **kwargs: Any) -> RequestResult:
        index = args[2].sample_id
        sends.append((index, args[1], args[4].get("request_id")))
        kwargs["on_send"]()
        if index == 2 and args[1] == "http://a":
            await asyncio.sleep(60)
        if index == 2:
            resent.set()
        kwargs["on_first_token"]()
        if index == 0:
            # Hold instance 0's blocks until the waiting request moves.
            await asyncio.wait_for(resent.wait(), timeout=10)
        await asyncio.sleep(0.01)
        return RequestResult(index, "d", "", 1.0, 5.0, 1, 3)

    monkeypatch.setattr(benchmark, "vllm_completion_stream", stream)
    requests = [
        _session_request(0, session=1, blocks=64),
        _session_request(1, session=2, blocks=1),
        _subagent_request(2, session=1, base=5, blocks=50),
    ]
    selection = select_requests(requests, max_model_len=100_000)
    results, _, _ = asyncio.run(
        replay(
            selection=selection,
            prompt_builder=PromptBuilder(vocab_size=1000, special_token_ids=[]),
            server_url=["http://a", "http://b"],
            served_model_name="m",
            output_path=tmp_path / "requests.jsonl",
            max_inflight=4,
            timeout_seconds=1.0,
            time_scale=1e6,
            seed=0,
            route="sticky",
            route_local_blocks=100,
            rebind_after_s=0.0,
        )
    )

    by_index = {record["trace_index"]: record for record in results}
    assert (by_index[2]["instance"], by_index[2]["rebinds"]) == (1, 1)
    assert (by_index[0]["rebinds"], by_index[1]["rebinds"]) == (0, 0)
    assert [send for send in sends if send[0] == 2] == [
        (2, "http://a", None),
        (2, "http://b", "trace-2-r1"),
    ]
    assert by_index[2]["ttft_ms"] > 200.0


def test_replay_routes_between_instances_and_records_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each request goes to its routed URL; records and summary name the instance."""
    urls: list[str] = []

    async def stream(*args: Any, **kwargs: Any) -> RequestResult:
        urls.append(args[1])
        kwargs["on_send"]()
        await asyncio.sleep(0.01)
        return RequestResult(args[2].sample_id, "d", "", 1.0, 5.0, 1, 3)

    monkeypatch.setattr(benchmark, "vllm_completion_stream", stream)
    requests = [_session_request(i, session=i % 2, blocks=1) for i in range(4)]
    selection = select_requests(requests, max_model_len=10_000)
    results, _, _ = asyncio.run(
        replay(
            selection=selection,
            prompt_builder=PromptBuilder(vocab_size=1000, special_token_ids=[]),
            server_url=["http://a", "http://b"],
            served_model_name="m",
            output_path=tmp_path / "requests.jsonl",
            max_inflight=4,
            timeout_seconds=1.0,
            time_scale=1e6,
            seed=0,
            route="sticky",
        )
    )

    by_index = {record["trace_index"]: record for record in results}
    assert {by_index[i]["instance"] for i in (0, 2)} != {
        by_index[i]["instance"] for i in (1, 3)
    }
    assert sorted(urls) == ["http://a", "http://a", "http://b", "http://b"]
    assert by_index[0]["tpot_ms"] == pytest.approx(2.0)
    summary = summarise(selection, results, 1.0, 1.0)
    assert summary["requests_per_instance"] == {"0": 2, "1": 2}


def test_session_text_prompts_are_stream_prefixes(tmp_path: Path) -> None:
    """Rows read input_length real tokens from their session's offset."""
    path = tmp_path / "trace.jsonl"
    rows = [
        {
            "timestamp": 0,
            "input_length": 3,
            "output_length": 1,
            "hash_ids": [0],
            "token_offset": 4,
        },
        {
            "timestamp": 1,
            "input_length": 5,
            "output_length": 1,
            "hash_ids": [0],
            "token_offset": 4,
        },
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    requests = load_trace(path)
    builder = SessionTextPromptBuilder(list(range(100, 110)))
    assert builder.build(requests[0]) == [104, 105, 106]
    assert builder.build(requests[1]) == [104, 105, 106, 107, 108]

    with pytest.raises(ValueError, match="past the token stream"):
        builder.build(TraceRequest(2, 0, 7, 1, (0,), token_offset=4))
    with pytest.raises(ValueError, match="token_offset"):
        builder.build(TraceRequest(3, 0, 2, 1, (0,)))
    path.write_text(json.dumps({**rows[0], "token_offset": -1}) + "\n")
    with pytest.raises(ValueError, match="token_offset"):
        load_trace(path)
