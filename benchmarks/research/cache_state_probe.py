# SPDX-License-Identifier: Apache-2.0
"""Probe decode speed and speculative acceptance under each prefix-cache state.

Sends the same prompt three times in a row: ``cold`` (nothing cached),
``gpu`` (immediate resend, the vLLM GPU prefix cache serves it) and
``external`` (the GPU prefix cache is reset first, so only a KV connector can
serve it; without a connector this pass is a second cold control). Requests
are greedy with ``ignore_eos``, so all three passes should decode the same
text. Comparing acceptance length (completion tokens per engine step) and
TPOT across passes shows whether restoring KV from an external tier changes
speculative decoding, for example because a drafter's KV is not restored.

Each pass records the request's own timings plus the server's counter deltas
(GPU and external prefix-cache hit tokens, speculative drafts and accepted
tokens), so a pass whose cache state is not the intended one is visible:
``external`` with nonzero GPU hits means the reset did not take effect.

Requires the vLLM server to run with ``VLLM_SERVER_DEV_MODE=1`` (it registers
``/reset_prefix_cache``) and ``--stream-interval 1`` (one stream chunk per
engine step).
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Sequence
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

import httpx
import numpy as np

from benchmarks.mooncake.benchmark import (
    SessionTextPromptBuilder,
    TraceRequest,
    load_trace,
)
from benchmarks.utils.datasets import BenchmarkSample
from benchmarks.utils.loadgen import RequestResult, vllm_completion_stream
from benchmarks.utils.metrics import compute_metric_delta, extract_prometheus_counters

PASSES = ("cold", "gpu", "external")

#: Counters whose per-pass deltas are recorded.
COUNTERS = {
    "gpu_hit_tokens": "vllm:prefix_cache_hits_total",
    "external_hit_tokens": "vllm:external_prefix_cache_hits_total",
    "spec_drafts": "vllm:spec_decode_num_drafts_total",
    "spec_draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "spec_accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}

Send = Callable[[TraceRequest, str], Awaitable[RequestResult]]
Snapshot = Callable[[], Awaitable[dict[str, float]]]
Reset = Callable[[], Awaitable[None]]


def _median(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _ratio(
    rows: Sequence[dict[str, Any]], numerator: str, denominator: str
) -> float | None:
    total = sum(r[denominator] for r in rows)
    return sum(r[numerator] for r in rows) / total if total else None


def select_probe_requests(
    requests: Sequence[TraceRequest],
    count: int,
    max_prompt_tokens: int,
) -> list[TraceRequest]:
    """Pick each session's longest prompt that fits, one per session.

    Args:
        requests: Trace rows in source order.
        count: Maximum rows to return (sessions in first-appearance order).
        max_prompt_tokens: Largest admissible ``input_length``.

    Returns:
        At most ``count`` rows; sessions without a fitting row are skipped.

    Thread-safety:
        Pure function.
    """
    best: dict[int | None, TraceRequest] = {}
    for request in requests:
        if request.input_length > max_prompt_tokens:
            continue
        key = request.session_id if request.session_id is not None else request.index
        current = best.get(key)
        if current is None or request.input_length > current.input_length:
            best[key] = request
    return list(best.values())[:count]


def text_agreement(reference: str, other: str) -> float:
    """Return the shared leading fraction of two generated texts.

    Args:
        reference: Text of the reference pass.
        other: Text of the compared pass.

    Returns:
        Length of the common prefix over the longer text (1.0 when equal,
        including both empty).

    Thread-safety:
        Pure function.
    """
    longest = max(len(reference), len(other))
    if longest == 0:
        return 1.0
    shared = 0
    for left, right in zip(reference, other, strict=False):
        if left != right:
            break
        shared += 1
    return shared / longest


async def probe_request(
    request: TraceRequest,
    send: Send,
    snapshot: Snapshot,
    reset: Reset,
    settle_s: float,
) -> list[dict[str, Any]]:
    """Run the cold, GPU-hit and external-hit passes of one request.

    Args:
        request: Trace row to send three times.
        send: Coroutine sending the row once; the second argument is the
            pass name (used in the request ID).
        snapshot: Coroutine returning the server's Prometheus counters.
        reset: Coroutine resetting the server's GPU prefix cache.
        settle_s: Wait after a pass so asynchronous KV stores finish and
            free their blocks before the next pass (and before a reset).

    Returns:
        One record per pass in :data:`PASSES` order.

    Thread-safety:
        Must run alone against the server: concurrent traffic would both
        pollute the counter deltas and keep blocks that block the reset.
    """
    records: list[dict[str, Any]] = []
    texts: dict[str, str] = {}
    for name in PASSES:
        if name == "external":
            await reset()
        before = await snapshot()
        result = await send(request, name)
        await asyncio.sleep(settle_s)
        delta = compute_metric_delta(before, await snapshot())
        texts[name] = result.generated_text
        decode_tokens = max(result.completion_tokens - 1, 0)
        records.append(
            {
                "index": request.index,
                "session_id": request.session_id,
                "input_length": request.input_length,
                "pass": name,
                "error": result.error,
                "ttft_ms": result.ttft_ms,
                "latency_ms": result.latency_ms,
                "tpot_ms": (
                    (result.latency_ms - result.ttft_ms) / decode_tokens
                    if decode_tokens
                    else None
                ),
                "completion_tokens": result.completion_tokens,
                "token_events": result.token_events,
                "acceptance_length": (
                    result.completion_tokens / result.token_events
                    if result.token_events
                    else None
                ),
                "cached_tokens": result.cached_tokens,
                "text_sha1": hashlib.sha1(result.generated_text.encode()).hexdigest(),
                "text_agreement": text_agreement(texts["cold"], result.generated_text),
                # GPU and external hits both reuse the KV the cold pass wrote,
                # so they can agree with each other where they differ from
                # cold (whose own prefill ran in a different batch shape).
                "text_agreement_gpu": (
                    text_agreement(texts["gpu"], result.generated_text)
                    if "gpu" in texts
                    else None
                ),
                **{key: delta.get(metric, 0.0) for key, metric in COUNTERS.items()},
            }
        )
    return records


def summarise_probe(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-pass medians and validity flags.

    Args:
        records: Records from :func:`probe_request`.

    Returns:
        Mapping from pass name to its count, failures, median TTFT/TPOT,
        median and pooled acceptance length, mean text agreement with the
        cold and GPU passes, and the number of rows whose cache state was not
        the intended one (over half the prompt served by a cache the pass is
        not meant to use: any for cold, external for gpu, GPU for external).

    Thread-safety:
        Pure function.
    """
    summary: dict[str, Any] = {}
    for name in PASSES:
        rows = [r for r in records if r["pass"] == name]
        ok = [r for r in rows if r["error"] is None]
        tokens = sum(r["completion_tokens"] for r in ok)
        events = sum(r["token_events"] for r in ok)
        # Hits from the source a pass is not meant to use: a shared system
        # prompt or the connector's tail block may still hit, so a row is
        # off-state only when that source serves over half the prompt.
        stray = {
            "cold": ("gpu_hit_tokens", "external_hit_tokens"),
            "gpu": ("external_hit_tokens",),
            "external": ("gpu_hit_tokens",),
        }[name]
        off_state = [
            r for r in ok if 2 * sum(r[key] for key in stray) > r["input_length"]
        ]
        summary[name] = {
            "requests": len(rows),
            "failed": len(rows) - len(ok),
            "ttft_ms_p50": _median([r["ttft_ms"] for r in ok]),
            "tpot_ms_p50": _median([r["tpot_ms"] for r in ok]),
            "acceptance_p50": _median([r["acceptance_length"] for r in ok]),
            "acceptance_pooled": tokens / events if events else None,
            "spec_accept_rate": _ratio(ok, "spec_accepted_tokens", "spec_draft_tokens"),
            "gpu_hit_tokens": sum(r["gpu_hit_tokens"] for r in ok),
            "external_hit_tokens": sum(r["external_hit_tokens"] for r in ok),
            "text_agreement_mean": _mean([r["text_agreement"] for r in ok]),
            "text_agreement_gpu_mean": _mean(
                [
                    r["text_agreement_gpu"]
                    for r in ok
                    if r["text_agreement_gpu"] is not None
                ]
            ),
            "off_state_rows": len(off_state),
        }
    return summary


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Probe every selected request against one server and write artifacts.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The summary also written to ``<output-dir>/summary.json``.

    Thread-safety:
        Sends one request at a time; see :func:`probe_request`.
    """
    requests = select_probe_requests(
        load_trace(args.trace),
        args.requests,
        args.max_model_len - args.max_tokens,
    )
    builder = SessionTextPromptBuilder(np.load(args.session_tokens, mmap_mode="r"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:

        async def send(request: TraceRequest, name: str) -> RequestResult:
            sample = BenchmarkSample(
                sample_id=request.index,
                dataset="cache-state-probe",
                context="",
                question="",
                answers=[],
            )
            params = {
                "model": args.served_model_name,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
                "request_id": f"probe-{request.index}-{name}",
            }
            return await vllm_completion_stream(
                client,
                args.server_url,
                sample,
                builder.build(request),
                params,
                asyncio.Semaphore(1),
                args.timeout,
            )

        async def snapshot() -> dict[str, float]:
            response = await client.get(f"{args.server_url}/metrics")
            response.raise_for_status()
            return extract_prometheus_counters(response.text)

        async def reset() -> None:
            # The endpoint always answers 200; a reset blocked by unfreed
            # blocks shows up as GPU hits in the external pass instead.
            response = await client.post(f"{args.server_url}/reset_prefix_cache")
            response.raise_for_status()

        with (args.output_dir / "probe.jsonl").open("w") as out:
            for request in requests:
                rows = await probe_request(
                    request, send, snapshot, reset, args.settle_s
                )
                for row in rows:
                    out.write(json.dumps(row) + "\n")
                out.flush()
                records.extend(rows)
    summary = {"selected": len(requests), "passes": summarise_probe(records)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments without the program name; ``None`` reads ``sys.argv``.

    Returns:
        Parsed namespace.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--session-tokens", type=Path, required=True)
    p.add_argument("--server-url", required=True)
    p.add_argument("--served-model-name", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--requests", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--settle-s", type=float, default=3.0)
    p.add_argument("--timeout", type=float, default=600.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the probe and print its summary as JSON.

    Args:
        argv: Arguments without the program name; ``None`` reads ``sys.argv``.
    """
    summary = asyncio.run(run_probe(parse_args(argv)))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
