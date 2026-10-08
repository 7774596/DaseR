# SPDX-License-Identifier: Apache-2.0
"""Tests for the cache-state probe's pass sequence and summary."""

import asyncio

from benchmarks.mooncake.benchmark import TraceRequest
from benchmarks.research.cache_state_probe import (
    probe_request,
    select_probe_requests,
    summarise_probe,
    text_agreement,
)
from benchmarks.utils.loadgen import RequestResult


def _request(index: int, session: int, length: int) -> TraceRequest:
    return TraceRequest(
        index=index,
        timestamp_ms=0.0,
        input_length=length,
        output_length=8,
        hash_ids=tuple(range((length + 511) // 512)),
        session_id=session,
        token_offset=0,
    )


def test_select_takes_each_sessions_longest_fitting_prompt() -> None:
    """One row per session: the longest whose prompt fits the limit."""
    rows = [
        _request(0, 1, 1000),
        _request(1, 1, 3000),
        _request(2, 2, 900),
        _request(3, 1, 9000),
        _request(4, 3, 9000),
    ]
    picked = select_probe_requests(rows, count=5, max_prompt_tokens=4000)
    assert [r.index for r in picked] == [1, 2]
    assert len(select_probe_requests(rows, count=1, max_prompt_tokens=4000)) == 1


def test_text_agreement_is_shared_prefix_over_longer_text() -> None:
    """Equal texts agree fully; divergence counts the common prefix."""
    assert text_agreement("abcd", "abcd") == 1.0
    assert text_agreement("", "") == 1.0
    assert text_agreement("abcd", "abxy") == 0.5
    assert text_agreement("ab", "abcd") == 0.5


def test_probe_resets_only_before_the_external_pass() -> None:
    """Passes run cold, gpu, external; the reset precedes only external."""
    events: list[str] = []
    counters = {"vllm:prefix_cache_hits_total": 0.0}
    # Per-pass counter bumps and decode shapes the fake server reports.
    gpu_hits = {"cold": 0.0, "gpu": 512.0, "external": 0.0}
    steps = {"cold": 4, "gpu": 4, "external": 8}

    async def send(request: TraceRequest, name: str) -> RequestResult:
        events.append(name)
        counters["vllm:prefix_cache_hits_total"] += gpu_hits[name]
        return RequestResult(
            sample_id=request.index,
            dataset="d",
            generated_text="same text",
            ttft_ms=100.0,
            latency_ms=100.0 + 15 * 10.0,
            prompt_tokens=request.input_length,
            completion_tokens=16,
            token_events=steps[name],
        )

    async def snapshot() -> dict[str, float]:
        return dict(counters)

    async def reset() -> None:
        events.append("reset")

    rows = asyncio.run(probe_request(_request(0, 1, 1024), send, snapshot, reset, 0.0))
    assert events == ["cold", "gpu", "reset", "external"]
    assert [r["pass"] for r in rows] == ["cold", "gpu", "external"]
    assert [r["gpu_hit_tokens"] for r in rows] == [0.0, 512.0, 0.0]
    assert [r["acceptance_length"] for r in rows] == [4.0, 4.0, 2.0]
    assert rows[0]["tpot_ms"] == 10.0
    assert all(r["text_agreement"] == 1.0 for r in rows)
    assert [r["text_agreement_gpu"] for r in rows] == [None, 1.0, 1.0]

    summary = summarise_probe(rows)
    assert summary["external"]["acceptance_pooled"] == 2.0
    assert summary["gpu"]["gpu_hit_tokens"] == 512.0
    assert all(summary[p]["off_state_rows"] == 0 for p in ("cold", "gpu", "external"))


def test_summary_flags_external_pass_served_by_gpu_cache() -> None:
    """GPU hits in the external pass mean the reset did not take effect."""
    base = {
        "error": None,
        "input_length": 1024,
        "text_agreement_gpu": None,
        "ttft_ms": 1.0,
        "tpot_ms": 1.0,
        "acceptance_length": 1.0,
        "completion_tokens": 1,
        "token_events": 1,
        "spec_accepted_tokens": 0.0,
        "spec_draft_tokens": 0.0,
        "external_hit_tokens": 0.0,
        "text_agreement": 1.0,
    }
    rows = [
        {**base, "pass": "cold", "gpu_hit_tokens": 256.0},
        {**base, "pass": "gpu", "gpu_hit_tokens": 896.0, "external_hit_tokens": 128.0},
        {**base, "pass": "external", "gpu_hit_tokens": 768.0},
    ]
    summary = summarise_probe(rows)
    assert summary["external"]["off_state_rows"] == 1
    # The connector serving the GPU pass's tail block does not flip it.
    assert summary["gpu"]["off_state_rows"] == 0
    # A shared system-prompt hit (a quarter of the prompt) is still cold.
    assert summary["cold"]["off_state_rows"] == 0
