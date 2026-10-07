# SPDX-License-Identifier: Apache-2.0
"""Replay Mooncake FAST'25 traces against an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import queue
import statistics
import threading
import time
from typing import Any, Iterable, Sequence

import httpx

from benchmarks.utils.datasets import BenchmarkSample
from benchmarks.utils.loadgen import RequestResult, vllm_completion_stream
from benchmarks.utils.metrics import compute_metric_delta, extract_prometheus_counters

MOONCAKE_BLOCK_TOKENS = 512
_BLOCK_ID_DIGITS = 8
# Marker tokens per corpus block; base >= 1e5 ordinary tokens covers 1e15 IDs.
_CORPUS_ID_DIGITS = 3
_WRITER_STOP = object()
ROUTES = (
    "sticky",
    "least-work",
    "gpu-affinity",
    "tier-affinity",
    "hbm-fit",
    "kv-cost",
    "sticky-saturated",
)


def _prefix_blocks(request: TraceRequest, present: Any) -> int:
    """Count the leading ``request.hash_ids`` contained in ``present``."""
    count = 0
    for hash_id in request.hash_ids:
        if hash_id not in present:
            break
        count += 1
    return count


def _decode_blocks(request: TraceRequest) -> int:
    """Return the 512-token blocks a request's decode adds to GPU KV."""
    return math.ceil(request.output_length / MOONCAKE_BLOCK_TOKENS)


@dataclass(frozen=True)
class TraceRequest:
    """One validated Mooncake request.

    Args:
        index: Zero-based source row index.
        timestamp_ms: Arrival offset in milliseconds.
        input_length: Prompt length in tokens.
        output_length: Requested decode length in tokens.
        hash_ids: Remapped 512-token prefix block hashes.
        session_id: Optional conversation/agent session the request belongs
            to; used only for session-affine routing across instances.
        gap_ms: Optional recorded idle time (tool execution / think time)
            between the prefix producer's completion and this request. With
            prefix dependencies it replaces the arrival timestamp for
            dependent requests, so replay keeps real gaps however long the
            server takes.
        prewarm_tokens: Optional leading prompt tokens known as soon as the
            prefix producer completes (a user turn whose previous query
            chain a thinking-model template re-renders without reasoning).

    Thread-safety:
        Immutable and safe to share across replay tasks.
    """

    index: int
    timestamp_ms: float
    input_length: int
    output_length: int
    hash_ids: tuple[int, ...]
    session_id: int | None = None
    gap_ms: float | None = None
    prewarm_tokens: int | None = None


@dataclass(frozen=True)
class WorkModel:
    """Per-request GPU-seconds estimate used to balance instances.

    Prefill of ``n`` new tokens after ``c`` cached ones costs
    ``n * (linear_flop + attention_flop * (c + n / 2)) / effective_flops``;
    decoding ``o`` tokens at context ``L`` costs
    ``o * (L * kv_bytes_per_token / decode_kv_bps + decode_step_s)``.
    Defaults are calibrated for GLM-4-9B on one H800; only the ratio between
    instances matters for routing.

    Thread-safety:
        Immutable.
    """

    linear_flop: float = 18e9
    attention_flop: float = 655e3
    effective_flops: float = 400e12
    kv_bytes_per_token: float = 80 * 1024
    decode_kv_bps: float = 1.43e12
    decode_step_s: float = 0.003

    def seconds(self, request: TraceRequest, cached_tokens: int) -> float:
        """Return the estimated prefill plus decode GPU seconds of a request.

        Args:
            request: Trace request.
            cached_tokens: Leading prompt tokens expected to be reused.

        Returns:
            Estimated GPU seconds.
        """
        new = request.input_length - cached_tokens
        prefill = (
            new
            * (self.linear_flop + self.attention_flop * (cached_tokens + new / 2))
            / self.effective_flops
        )
        decode = request.output_length * (
            request.input_length * self.kv_bytes_per_token / self.decode_kv_bps
            + self.decode_step_s
        )
        return prefill + decode


class InstanceRouter:
    """Choose a serving instance per request.

    ``sticky`` pins a session to the instance with the least outstanding work
    when its first request arrives, so later turns hit that instance's GPU
    prefix cache (session affinity). ``least-work`` sends every request to the
    instance with the least outstanding work, relying on a shared external
    KV tier for reuse wherever the request lands. Outstanding work is the sum
    of :class:`WorkModel` estimates of admitted, unfinished requests; cached
    prefixes are predicted from ``hash_ids`` already sent to any instance.

    The affinity routes also keep a per-instance shadow LRU of the blocks sent
    there, bounded by the GPU prefix-cache capacity, and pick the instance
    minimizing outstanding work plus this request's cost on it.
    ``gpu-affinity`` (Preble-style) prices a block missing from the instance's
    GPU cache as recompute. ``tier-affinity`` prices it as a load from the
    shared external tier when any instance already produced it, so locality
    only breaks near-ties in load and blocks never seen are recomputed
    anywhere.

    ``hbm-fit`` routes on GPU KV co-residency rather than work: an instance's
    resident demand is the union of in-flight prompt blocks (shared prefixes
    count once, as in vLLM's prefix cache) plus their decode blocks. A session
    stays on its instance while the request fits there beside the in-flight
    ones; otherwise it moves to the least-loaded instance it fits on, whose
    missing prefix the shared external tier serves as a load. When it fits
    nowhere it stays and queues behind the session's own prefix.

    ``kv-cost`` (NVIDIA Dynamo KV router) picks the instance minimizing
    prefill blocks missing from its shadow GPU cache plus its potential
    active blocks (in-flight resident blocks plus this request's new ones).
    ``sticky-saturated`` (llm-d sticky-until-saturated) keeps a session on its
    instance until that instance's outstanding work reaches
    ``saturation_s``, then moves it to the least-loaded instance.

    With ``rebind_after_s`` set, a request still waiting for its first token
    after :meth:`rebind_wait_s` may be re-bound once (:meth:`rebind_target`,
    :meth:`move`): only when its instance's resident demand exceeds
    ``local_blocks`` (it waits for GPU KV, not compute) and another instance
    can hold it. The caller cancels the waiting stream and resends there; the
    shared external tier serves its prefix.

    Args:
        server_urls: Instance base URLs.
        route: One of :data:`ROUTES`.
        work_model: Cost model for outstanding work.
        local_blocks: Per-instance GPU KV capacity in 512-token blocks: the
            shadow prefix-cache bound of the affinity routes and the
            co-residency bound of ``hbm-fit``.
        load_tokens_per_s: External-tier load rate used by
            ``tier-affinity`` to price a non-local reusable block.
        saturation_s: Outstanding-work seconds at which
            ``sticky-saturated`` stops honoring session affinity.
        rebind_after_s: Slack beyond the expected prefill time before a
            waiting request may be re-bound; ``None`` disables re-binding.
        prefill_tokens_per_s: Prefill rate used for the expected wait.

    Thread-safety:
        Single asyncio event loop only; no internal locking.
    """

    def __init__(
        self,
        server_urls: Sequence[str],
        route: str = "sticky",
        work_model: WorkModel | None = None,
        local_blocks: int = 2048,
        load_tokens_per_s: float = 360e3,
        saturation_s: float = 10.0,
        rebind_after_s: float | None = None,
        prefill_tokens_per_s: float = 10e3,
    ) -> None:
        if not server_urls:
            raise ValueError("at least one server URL is required")
        if route not in ROUTES:
            raise ValueError(f"unknown route: {route}")
        if local_blocks <= 0 or load_tokens_per_s <= 0:
            raise ValueError("local_blocks and load_tokens_per_s must be positive")
        if saturation_s <= 0:
            raise ValueError("saturation_s must be positive")
        if rebind_after_s is not None and rebind_after_s < 0:
            raise ValueError("rebind_after_s must be non-negative")
        if prefill_tokens_per_s <= 0:
            raise ValueError("prefill_tokens_per_s must be positive")
        self.server_urls = tuple(url.rstrip("/") for url in server_urls)
        self.route = route
        self.work_model = work_model or WorkModel()
        self.local_blocks = local_blocks
        self.load_tokens_per_s = load_tokens_per_s
        self.saturation_s = saturation_s
        self.rebind_after_s = rebind_after_s
        self.prefill_tokens_per_s = prefill_tokens_per_s
        self._outstanding = [0.0] * len(self.server_urls)
        self._sessions: dict[int, int] = {}
        self._seen: set[int] = set()
        self._local: list[OrderedDict[int, None]] = [
            OrderedDict() for _ in self.server_urls
        ]
        # In-flight prompt block refcounts and decode blocks per instance.
        self._resident: list[dict[int, int]] = [{} for _ in self.server_urls]
        self._decode_blocks = [0] * len(self.server_urls)

    def resident_blocks(self, instance: int) -> int:
        """Return the GPU KV blocks in-flight requests hold on ``instance``.

        Args:
            instance: Instance index.

        Returns:
            Distinct in-flight prompt blocks plus their decode blocks.
        """
        return len(self._resident[instance]) + self._decode_blocks[instance]

    def acquire(self, request: TraceRequest) -> tuple[int, float]:
        """Pick an instance for ``request`` and charge its estimated work.

        Args:
            request: Admitted request.

        Returns:
            Instance index and the charged work, to pass to :meth:`release`.
        """
        cached = self._cached_tokens(request, _prefix_blocks(request, self._seen))
        if self.route in ("gpu-affinity", "tier-affinity"):
            costs = [
                self._affinity_cost(request, i, cached)
                for i in range(len(self.server_urls))
            ]
            instance = min(
                range(len(costs)), key=lambda i: self._outstanding[i] + costs[i]
            )
            work = costs[instance]
            self._touch_local(instance, request.hash_ids)
        elif self.route == "kv-cost":
            work = self.work_model.seconds(request, cached)
            instance = min(
                range(len(self.server_urls)),
                key=lambda i: self._kv_cost(request, i),
            )
            self._touch_local(instance, request.hash_ids)
        else:
            work = self.work_model.seconds(request, cached)
            instance = (
                self._pick_fit(request)
                if self.route == "hbm-fit"
                else self._pick(request)
            )
        self._seen.update(request.hash_ids)
        self._charge(instance, work, request)
        return instance, work

    def release(self, instance: int, work: float, request: TraceRequest) -> None:
        """Remove a finished request's work and resident blocks from its instance.

        Args:
            instance: Index returned by :meth:`acquire`.
            work: Work returned by :meth:`acquire`.
            request: The request passed to :meth:`acquire`.
        """
        self._outstanding[instance] = max(0.0, self._outstanding[instance] - work)
        resident = self._resident[instance]
        for hash_id in request.hash_ids:
            count = resident[hash_id] - 1
            if count:
                resident[hash_id] = count
            else:
                del resident[hash_id]
        self._decode_blocks[instance] -= _decode_blocks(request)

    def rebind_wait_s(self, request: TraceRequest) -> float | None:
        """Return how long ``request`` may wait for its first token.

        Call before :meth:`acquire`, which marks its blocks as produced.

        Args:
            request: Request about to be acquired.

        Returns:
            Expected prefill seconds of its unproduced tokens plus
            ``rebind_after_s``, or ``None`` when re-binding is disabled.
        """
        if self.rebind_after_s is None:
            return None
        cached = self._cached_tokens(request, _prefix_blocks(request, self._seen))
        new = request.input_length - cached
        return self.rebind_after_s + new / self.prefill_tokens_per_s

    def rebind_target(self, instance: int, request: TraceRequest) -> int | None:
        """Return an instance to re-bind a waiting request to, if any.

        Args:
            instance: Instance currently holding ``request``.
            request: Acquired request still waiting for its first token.

        Returns:
            The least-loaded other instance that can hold ``request`` beside
            its in-flight requests when ``instance`` is over its GPU KV
            capacity; otherwise ``None``.
        """
        if self.resident_blocks(instance) <= self.local_blocks:
            return None
        fits = [
            i
            for i in range(len(self.server_urls))
            if i != instance
            and self.resident_blocks(i) + self._new_blocks(request, i)
            <= self.local_blocks
        ]
        if not fits:
            return None
        return min(fits, key=self._outstanding.__getitem__)

    def move(
        self, instance: int, target: int, work: float, request: TraceRequest
    ) -> None:
        """Move an acquired request's charge from ``instance`` to ``target``.

        Args:
            instance: Instance the request was acquired on.
            target: New instance, e.g. from :meth:`rebind_target`.
            work: Work returned by :meth:`acquire`.
            request: The acquired request.
        """
        self.release(instance, work, request)
        self._charge(target, work, request)
        self._touch_local(target, request.hash_ids)
        if request.session_id is not None:
            self._sessions[request.session_id] = target

    def _charge(self, instance: int, work: float, request: TraceRequest) -> None:
        self._outstanding[instance] += work
        resident = self._resident[instance]
        for hash_id in request.hash_ids:
            resident[hash_id] = resident.get(hash_id, 0) + 1
        self._decode_blocks[instance] += _decode_blocks(request)

    def _pick_fit(self, request: TraceRequest) -> int:
        home = self._sessions.get(request.session_id)
        fits = [
            i
            for i in range(len(self.server_urls))
            if self.resident_blocks(i) + self._new_blocks(request, i)
            <= self.local_blocks
        ]
        if home is not None and (home in fits or not fits):
            return home
        candidates = fits or range(len(self.server_urls))
        instance = min(candidates, key=self._outstanding.__getitem__)
        if request.session_id is not None:
            self._sessions[request.session_id] = instance
        return instance

    def _new_blocks(self, request: TraceRequest, instance: int) -> int:
        resident = self._resident[instance]
        missing = sum(1 for hash_id in request.hash_ids if hash_id not in resident)
        return missing + _decode_blocks(request)

    def _kv_cost(self, request: TraceRequest, instance: int) -> int:
        blocks = math.ceil(request.input_length / MOONCAKE_BLOCK_TOKENS)
        overlap = _prefix_blocks(request, self._local[instance])
        prefill = max(0, blocks - overlap)
        return (
            prefill
            + self.resident_blocks(instance)
            + self._new_blocks(request, instance)
        )

    def _pick(self, request: TraceRequest) -> int:
        if self.route == "least-work" or request.session_id is None:
            return self._least_loaded()
        instance = self._sessions.get(request.session_id)
        if instance is None or (
            self.route == "sticky-saturated"
            and self._outstanding[instance] >= self.saturation_s
        ):
            instance = self._least_loaded()
            self._sessions[request.session_id] = instance
        return instance

    def _least_loaded(self) -> int:
        return min(range(len(self._outstanding)), key=self._outstanding.__getitem__)

    def _cached_tokens(self, request: TraceRequest, blocks: int) -> int:
        return max(0, min(blocks * MOONCAKE_BLOCK_TOKENS, request.input_length - 1))

    def _affinity_cost(
        self, request: TraceRequest, instance: int, cached: int
    ) -> float:
        local = self._cached_tokens(
            request, _prefix_blocks(request, self._local[instance])
        )
        if self.route == "gpu-affinity":
            return self.work_model.seconds(request, local)
        return (
            self.work_model.seconds(request, cached)
            + max(0, cached - local) / self.load_tokens_per_s
        )

    def _touch_local(self, instance: int, hash_ids: Sequence[int]) -> None:
        # vLLM frees a request's blocks tail first, so the prefix head is the
        # most recently used and is evicted last.
        lru = self._local[instance]
        for hash_id in reversed(hash_ids):
            lru[hash_id] = None
            lru.move_to_end(hash_id)
        while len(lru) > self.local_blocks:
            lru.popitem(last=False)


@dataclass(frozen=True)
class TraceSelection:
    """Context-limit partition of selected trace requests.

    Args:
        selected: Contiguous source prefix selected for the run.
        eligible: Requests that fit the configured model context.
        skipped: Requests rejected by the context limit.

    Thread-safety:
        Immutable containers of immutable requests.
    """

    selected: tuple[TraceRequest, ...]
    eligible: tuple[TraceRequest, ...]
    skipped: tuple[TraceRequest, ...]


class PromptBuilder:
    """Build deterministic token-ID prompts from Mooncake block hashes.

    Args:
        vocab_size: Model vocabulary size.
        special_token_ids: Token IDs that must not appear in synthetic prompts.

    Thread-safety:
        Safe in the asyncio replay, where calls occur on one event-loop thread.
        The internal LRU cache is not intended for concurrent native threads.
    """

    def __init__(self, vocab_size: int, special_token_ids: Iterable[int]) -> None:
        excluded = set(special_token_ids)
        self._tokens = tuple(
            token_id for token_id in range(vocab_size) if token_id not in excluded
        )
        if len(self._tokens) < 2:
            raise ValueError("tokenizer must expose at least two ordinary token IDs")
        self._max_hash_id = len(self._tokens) ** _BLOCK_ID_DIGITS
        self._blocks: OrderedDict[int, tuple[int, ...]] = OrderedDict()

    def build(self, request: TraceRequest) -> list[int]:
        """Construct the exact-length prompt represented by a trace row.

        Args:
            request: Validated Mooncake request.

        Returns:
            Integer token IDs whose 512-token blocks follow ``hash_ids``.

        Thread-safety:
            See the class-level note. Returned lists are request-owned.
        """

        prompt: list[int] = []
        for hash_id in request.hash_ids:
            prompt.extend(self._block(hash_id))
        del prompt[request.input_length :]
        return prompt

    def _block(self, hash_id: int) -> tuple[int, ...]:
        cached = self._blocks.get(hash_id)
        if cached is not None:
            self._blocks.move_to_end(hash_id)
            return cached
        if hash_id >= self._max_hash_id:
            raise ValueError(
                f"hash ID {hash_id} exceeds deterministic encoding capacity"
            )
        result = self._make_block(hash_id)
        self._blocks[hash_id] = result
        if len(self._blocks) > 8192:
            self._blocks.popitem(last=False)
        return result

    def _id_tokens(self, hash_id: int, digits: int) -> list[int]:
        base = len(self._tokens)
        value = hash_id
        tokens = []
        for _ in range(digits):
            tokens.append(self._tokens[value % base])
            value //= base
        return tokens

    def _make_block(self, hash_id: int) -> tuple[int, ...]:
        block = self._id_tokens(hash_id, _BLOCK_ID_DIGITS)
        base = len(self._tokens)
        state = (hash_id + 1) & ((1 << 64) - 1)
        for position in range(_BLOCK_ID_DIGITS, MOONCAKE_BLOCK_TOKENS):
            state = (state * 6364136223846793005 + 1442695040888963407 + position) & (
                (1 << 64) - 1
            )
            block.append(self._tokens[state % base])
        return tuple(block)


class CorpusPromptBuilder(PromptBuilder):
    """Build prompts whose blocks carry real text from a tokenized corpus.

    Block ``h`` holds the corpus window ``h mod W`` (W = corpus length / 512),
    so consecutive hash IDs read as consecutive text and equal hash prefixes
    still give equal token prefixes. The first ``_CORPUS_ID_DIGITS`` tokens of
    each block encode ``h`` so distinct hash IDs never collide, even when the
    corpus wraps.

    Args:
        vocab_size: Model vocabulary size.
        special_token_ids: Token IDs excluded from the hash-ID marker.
        corpus: 1-D sequence of real token IDs (for example a memory-mapped
            ``numpy`` array) at least one block long.

    Thread-safety:
        Same as :class:`PromptBuilder`.
    """

    def __init__(
        self,
        vocab_size: int,
        special_token_ids: Iterable[int],
        corpus: Sequence[int],
    ) -> None:
        super().__init__(vocab_size, special_token_ids)
        self._windows = len(corpus) // MOONCAKE_BLOCK_TOKENS
        if self._windows < 1:
            raise ValueError("corpus must hold at least one 512-token block")
        self._corpus = corpus

    def _make_block(self, hash_id: int) -> tuple[int, ...]:
        start = (hash_id % self._windows) * MOONCAKE_BLOCK_TOKENS
        text = self._corpus[start + _CORPUS_ID_DIGITS : start + MOONCAKE_BLOCK_TOKENS]
        return tuple(self._id_tokens(hash_id, _CORPUS_ID_DIGITS)) + tuple(
            int(token) for token in text
        )


def load_trace(
    path: Path,
    max_requests: int = 0,
    start_request: int = 0,
    min_context_tokens: int = 0,
) -> list[TraceRequest]:
    """Load and validate a contiguous window of a Mooncake JSONL trace.

    Args:
        path: Local Mooncake JSONL path.
        max_requests: Maximum rows to load; zero loads the complete trace.
        start_request: Zero-based source row where the window begins.
        min_context_tokens: Minimum input-plus-output tokens to retain.

    Returns:
        Validated requests in source order.

    Raises:
        ValueError: If the limit or any selected row violates the trace schema.

    Thread-safety:
        Performs synchronous planning-time file IO before replay starts.
    """

    if max_requests < 0 or start_request < 0 or min_context_tokens < 0:
        raise ValueError("request limits must be non-negative")
    requests: list[TraceRequest] = []
    previous_timestamp = -1.0
    first_timestamp = 0.0
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index < start_request:
                continue
            if max_requests and len(requests) >= max_requests:
                break
            if not line.strip():
                raise ValueError(f"row {index}: blank lines are not valid requests")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"row {index}: invalid JSON: {exc.msg}") from exc
            request = _parse_request(index, payload)
            if request.input_length + request.output_length < min_context_tokens:
                continue
            if request.timestamp_ms < previous_timestamp:
                raise ValueError(f"row {index}: timestamps must be non-decreasing")
            previous_timestamp = request.timestamp_ms
            if not requests:
                first_timestamp = request.timestamp_ms
            requests.append(
                replace(request, timestamp_ms=request.timestamp_ms - first_timestamp)
            )
    if not requests:
        raise ValueError("trace selection is empty")
    return requests


def select_requests(
    requests: Sequence[TraceRequest], max_model_len: int
) -> TraceSelection:
    """Partition requests by the model context limit without changing lengths.

    Args:
        requests: Validated source-order requests.
        max_model_len: Maximum prompt plus output token count.

    Returns:
        Selected, eligible, and skipped immutable request sequences.

    Thread-safety:
        Pure function.
    """

    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")
    selected = tuple(requests)
    eligible: list[TraceRequest] = []
    skipped: list[TraceRequest] = []
    for request in selected:
        destination = (
            eligible
            if request.input_length + request.output_length <= max_model_len
            else skipped
        )
        destination.append(request)
    return TraceSelection(
        selected=selected,
        eligible=tuple(eligible),
        skipped=tuple(skipped),
    )


def scheduled_offset_seconds(timestamp_ms: float, time_scale: float) -> float:
    """Convert a source timestamp to a scaled replay offset.

    Args:
        timestamp_ms: Original arrival offset in milliseconds.
        time_scale: Replay acceleration factor; one preserves source timing.

    Returns:
        Replay offset in seconds.

    Thread-safety:
        Pure function.
    """

    if timestamp_ms < 0:
        raise ValueError("timestamp_ms must be non-negative")
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("time_scale must be finite and positive")
    return timestamp_ms / 1000.0 / time_scale


def prefix_predecessors(requests: Sequence[TraceRequest]) -> list[int | None]:
    """Find the earlier request that first stored each request's reusable prefix.

    For every request, walk its ``hash_ids`` through a trie of earlier
    requests. The deepest shared block was first produced by the request
    that created that trie node; once it completes, the whole shared prefix
    exists. Waiting on that creator keeps cross-request reuse intact when
    arrival timestamps are ignored, without serializing requests that only
    share an early block such as a common system prompt.

    Args:
        requests: Requests in replay order.

    Returns:
        Position (into ``requests``) of each request's predecessor, or None
        when it shares no block with any earlier request.

    Thread-safety:
        Pure function.
    """

    root: dict[int, tuple[int, dict]] = {}
    predecessors: list[int | None] = []
    for position, request in enumerate(requests):
        children = root
        predecessor: int | None = None
        for hash_id in request.hash_ids:
            node = children.get(hash_id)
            if node is None:
                node = (position, {})
                children[hash_id] = node
            else:
                predecessor = node[0]
            children = node[1]
        predecessors.append(predecessor)
    return predecessors


def summarise(
    selection: TraceSelection,
    results: Sequence[dict[str, Any]],
    wall_seconds: float,
    time_scale: float,
    prefix_cache_metrics: dict[str, float] | None = None,
    aborted: bool = False,
) -> dict[str, Any]:
    """Summarize trace coverage and successful request latency.

    Args:
        selection: Context-limit partition used for the replay.
        results: Completed eligible request result dictionaries.
        wall_seconds: Measured replay wall duration.
        time_scale: Configured arrival acceleration factor.
        prefix_cache_metrics: Optional vLLM local/external cache counter deltas.
        aborted: Whether the replay stopped early after too many failures.

    Returns:
        JSON-serializable aggregate metrics.

    Thread-safety:
        Pure function over immutable snapshots.
    """

    successful = [result for result in results if result.get("error") is None]
    ttfts = [float(result["ttft_ms"]) for result in successful]
    latencies = [float(result["latency_ms"]) for result in successful]
    client_queues = [float(result["client_queue_ms"]) for result in successful]
    arrival_ttfts = [
        float(result["arrival_to_first_token_ms"]) for result in successful
    ]
    arrival_latencies = [
        float(result["arrival_to_completion_ms"]) for result in successful
    ]
    tpots = [
        float(result["tpot_ms"])
        for result in successful
        if result.get("tpot_ms") is not None
    ]
    instances: dict[str, int] = {}
    for result in successful:
        key = str(result.get("instance", 0))
        instances[key] = instances.get(key, 0) + 1
    return {
        "selected_requests": len(selection.selected),
        "eligible_requests": len(selection.eligible),
        "skipped_context_requests": len(selection.skipped),
        "skipped_input_tokens": sum(row.input_length for row in selection.skipped),
        "skipped_output_tokens": sum(row.output_length for row in selection.skipped),
        "completed_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "aborted": aborted,
        "wall_seconds": wall_seconds,
        "time_scale": time_scale,
        "achieved_completion_rps": (
            len(successful) / wall_seconds if wall_seconds > 0 else 0.0
        ),
        "ttft_ms": _latency_summary(ttfts),
        "latency_ms": _latency_summary(latencies),
        "client_queue_ms": _latency_summary(client_queues),
        "arrival_to_first_token_ms": _latency_summary(arrival_ttfts),
        "arrival_to_completion_ms": _latency_summary(arrival_latencies),
        "tpot_ms": _latency_summary(tpots),
        "requests_per_instance": instances,
        "prefix_cache": _prefix_cache_summary(prefix_cache_metrics or {}),
    }


async def replay(
    *,
    selection: TraceSelection,
    prompt_builder: PromptBuilder,
    server_url: str | Sequence[str],
    served_model_name: str,
    output_path: Path,
    max_inflight: int,
    timeout_seconds: float,
    time_scale: float,
    seed: int,
    abort_after_failures: int = 0,
    prefix_dependencies: bool = False,
    route: str = "sticky",
    route_local_blocks: int = 2048,
    route_load_tokens_per_s: float = 360e3,
    route_saturation_s: float = 10.0,
    rebind_after_s: float | None = None,
    strict_order: bool = False,
    dependency_gap_s: float = 0.0,
    ordered_admission: bool = False,
    prewarm: bool = False,
    prewarm_chunk_tokens: int = 0,
) -> tuple[list[dict[str, Any]], float, bool]:
    """Replay eligible requests and incrementally write completion records.

    Args:
        selection: Validated context-limit partition.
        prompt_builder: Deterministic token prompt builder.
        server_url: OpenAI-compatible server base URL, or several instance
            URLs to route between.
        served_model_name: Model name sent in completion bodies.
        output_path: JSONL request-result destination.
        max_inflight: Maximum concurrent HTTP requests.
        timeout_seconds: Per-request HTTP timeout.
        time_scale: Arrival acceleration factor.
        seed: Generation seed.
        abort_after_failures: Cancel all outstanding requests once this many
            requests have failed; zero disables early abort.
        prefix_dependencies: Hold each request until its
            :func:`prefix_predecessors` request completes, so closed-loop
            replay (large ``time_scale``) keeps cross-request reuse. The wait
            happens before the concurrency semaphore and is excluded from
            client queue time and TTFT. Dependent requests carrying
            ``gap_ms`` start that (time-scaled) gap after the dependency
            completes instead of at their timestamp.
        route: :class:`InstanceRouter` policy when several URLs are given.
        route_local_blocks: Shadow GPU cache blocks per instance for the
            affinity routes.
        route_load_tokens_per_s: External-tier load rate for
            ``tier-affinity``.
        route_saturation_s: Outstanding-work threshold of
            ``sticky-saturated``.
        rebind_after_s: Re-bind a request waiting for GPU KV to another
            instance this many seconds past its expected prefill time (see
            :class:`InstanceRouter`); ``None`` disables it. Records carry
            ``rebinds`` and the final ``instance``.
        strict_order: Hold each request until the previous one completes, so
            the send order is exactly the trace order (implies prefix
            dependencies). Without it, closed-loop replay sends ready
            requests in semaphore wake-up order, not trace order.
        dependency_gap_s: Idle seconds between a dependency's completion
            and the dependent request (tool execution / think time). The
            gap is excluded from client queue time and TTFT.
        ordered_admission: Admit requests to the concurrency limit in trace
            order: each one waits until the previous trace request was
            admitted (and for its own prefix dependency), so the send order is
            the trace order while up to ``max_inflight`` requests overlap.
        prewarm: When a dependent request carries ``prewarm_tokens``, send
            that prefix (one output token, outside the concurrency limit) as
            soon as its producer completes, so its prefill runs during the
            idle gap. The request itself is sent after the prewarm finishes;
            any wait past the gap counts toward its TTFT. Records carry
            ``prewarm_ms`` and ``prewarm_error``.
        prewarm_chunk_tokens: When positive, prewarm in growing prefixes of
            this many more tokens each (prefix caching reuses the earlier
            ones), so no single prefill holds the GPU for the whole prefix
            and foreground requests queue between pieces. Chunking stops at
            the end of the idle gap; the request prefills the rest itself.
            Zero sends the whole prefix at once.

    Returns:
        Completion-order result dictionaries of finished requests, measured
        wall duration, and whether the replay was aborted early.

    Async/thread-safety:
        Uses one asyncio event loop for scheduling and a dedicated writer
        thread for result IO. Prompt construction is serialized per task after
        admission and does not run in the writer thread.
    """

    if max_inflight <= 0:
        raise ValueError("max_inflight must be positive")
    if abort_after_failures < 0:
        raise ValueError("abort_after_failures must be non-negative")
    if prewarm_chunk_tokens < 0:
        raise ValueError("prewarm_chunk_tokens must be non-negative")
    if strict_order and ordered_admission:
        raise ValueError("strict_order and ordered_admission are exclusive")
    scheduled_offset_seconds(0, time_scale)
    router = InstanceRouter(
        [server_url] if isinstance(server_url, str) else server_url,
        route,
        local_blocks=route_local_blocks,
        load_tokens_per_s=route_load_tokens_per_s,
        saturation_s=route_saturation_s,
        rebind_after_s=rebind_after_s,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _ResultWriter(output_path)
    for request in selection.skipped:
        writer.write(_skipped_record(request))

    outer_sem = asyncio.Semaphore(max_inflight)
    done_events = [asyncio.Event() for _ in selection.eligible]
    admitted_events = [asyncio.Event() for _ in selection.eligible]
    predecessors: list[int | None] = (
        prefix_predecessors(selection.eligible)
        if prefix_dependencies
        else [None] * len(selection.eligible)
    )
    if strict_order:
        predecessors = [
            position - 1 if position else None
            for position in range(len(selection.eligible))
        ]
    started_at = time.perf_counter()
    timeout = httpx.Timeout(timeout_seconds)
    limits = httpx.Limits(
        max_connections=max_inflight,
        max_keepalive_connections=max_inflight,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            tasks = [
                asyncio.create_task(
                    _run_request(
                        request=request,
                        prompt_builder=prompt_builder,
                        client=client,
                        router=router,
                        served_model_name=served_model_name,
                        outer_sem=outer_sem,
                        started_at=started_at,
                        timeout_seconds=timeout_seconds,
                        time_scale=time_scale,
                        seed=seed,
                        writer=writer,
                        wait_for=(
                            None
                            if predecessors[position] is None
                            else done_events[predecessors[position]]
                        ),
                        done=done_events[position],
                        dependency_gap_s=dependency_gap_s,
                        admit_after=(
                            admitted_events[position - 1]
                            if ordered_admission and position
                            else None
                        ),
                        admitted=admitted_events[position],
                        prewarm=prewarm,
                        prewarm_chunk_tokens=prewarm_chunk_tokens,
                    )
                )
                for position, request in enumerate(selection.eligible)
            ]
            results: list[dict[str, Any]] = []
            failures = 0
            aborted = False
            pending: set[asyncio.Task[dict[str, Any]]] = set(tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    record = task.result()
                    results.append(record)
                    failures += record.get("error") is not None
                if abort_after_failures and failures >= abort_after_failures:
                    aborted = bool(pending)
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    break
    finally:
        writer.close()
    return results, time.perf_counter() - started_at, aborted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Mooncake benchmark CLI arguments.

    Args:
        argv: Optional arguments without the executable name.

    Returns:
        Parsed argparse namespace.

    Thread-safety:
        Pure except for argparse's process-level error handling.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--server-url",
        nargs="+",
        default=["http://127.0.0.1:8001"],
        help="One or more instance base URLs; several URLs enable --route.",
    )
    parser.add_argument(
        "--route",
        choices=ROUTES,
        default="sticky",
        help="Instance choice per request when several --server-url are given.",
    )
    parser.add_argument(
        "--route-local-blocks",
        type=int,
        default=2048,
        help="Shadow GPU prefix-cache capacity per instance in 512-token blocks "
        "(affinity routes).",
    )
    parser.add_argument(
        "--route-load-tokens-per-s",
        type=float,
        default=360e3,
        help="External-tier load rate pricing non-local reuse (tier-affinity).",
    )
    parser.add_argument(
        "--rebind-after-s",
        type=float,
        default=None,
        help="Re-bind a request waiting for GPU KV to another instance this "
        "many seconds past its expected prefill time (off by default).",
    )
    parser.add_argument(
        "--prewarm-stripped",
        action="store_true",
        help="Prefill each row's prewarm_tokens prefix as soon as its "
        "prefix producer completes (off by default).",
    )
    parser.add_argument(
        "--prewarm-chunk-tokens",
        type=int,
        default=0,
        help="Prewarm in growing prefixes of this many more tokens each, "
        "stopping at the end of the idle gap (0 sends the whole prefix).",
    )
    parser.add_argument(
        "--route-saturation-s",
        type=float,
        default=10.0,
        help="Outstanding work (s) at which sticky-saturated drops affinity.",
    )
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--start-request", type=int, default=0)
    parser.add_argument("--min-context-tokens", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--abort-after-failures",
        type=int,
        default=0,
        help="Cancel the replay once this many requests fail (0 disables).",
    )
    parser.add_argument("--overflow", choices=("skip", "error"), default="skip")
    parser.add_argument(
        "--corpus-tokens",
        type=Path,
        default=None,
        help=(
            "Optional .npy array of real token IDs (same tokenizer); blocks "
            "then carry real text instead of synthetic tokens."
        ),
    )
    parser.add_argument(
        "--prefix-dependencies",
        action="store_true",
        help=(
            "Send each request only after the earlier request that first "
            "stored its shared hash_ids prefix completes (closed-loop replay)."
        ),
    )
    parser.add_argument(
        "--strict-order",
        action="store_true",
        help=(
            "Send each request only after the previous trace request "
            "completes, so the replay follows the trace order exactly."
        ),
    )
    parser.add_argument(
        "--ordered-admission",
        action="store_true",
        help=(
            "Admit requests to --max-inflight in trace order, so the send "
            "order is the trace order while requests still overlap."
        ),
    )
    parser.add_argument(
        "--dependency-gap-s",
        type=float,
        default=0.0,
        help=(
            "Idle seconds between a dependency's completion and the next "
            "request (models tool execution / think time; not in TTFT)."
        ),
    )
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    """Run the benchmark from parsed CLI arguments.

    Args:
        args: Namespace returned by :func:`parse_args`.

    Returns:
        Zero on complete success, one when any eligible request fails.

    Async/thread-safety:
        Owns one replay event loop and one result-writer thread.
    """

    _reject_tmp_output(args.output_dir)
    requests = load_trace(
        args.trace,
        args.max_requests,
        args.start_request,
        args.min_context_tokens,
    )
    selection = select_requests(requests, args.max_model_len)
    if args.overflow == "error" and selection.skipped:
        raise ValueError(
            f"{len(selection.skipped)} selected requests exceed max model length"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if args.corpus_tokens is None:
        prompt_builder = PromptBuilder(
            vocab_size=len(tokenizer),
            special_token_ids=tokenizer.all_special_ids,
        )
    else:
        import numpy as np

        prompt_builder = CorpusPromptBuilder(
            vocab_size=len(tokenizer),
            special_token_ids=tokenizer.all_special_ids,
            corpus=np.load(args.corpus_tokens, mmap_mode="r"),
        )
    results_path = args.output_dir / "requests.jsonl"
    summary_path = args.output_dir / "summary.json"
    server_urls = [url.rstrip("/") for url in args.server_url]
    before_metrics = await _metrics_snapshot(server_urls)
    results, wall_seconds, aborted = await replay(
        selection=selection,
        prompt_builder=prompt_builder,
        server_url=server_urls,
        served_model_name=args.served_model_name,
        output_path=results_path,
        max_inflight=args.max_inflight,
        timeout_seconds=args.timeout,
        time_scale=args.time_scale,
        seed=args.seed,
        abort_after_failures=args.abort_after_failures,
        prefix_dependencies=args.prefix_dependencies,
        route=args.route,
        route_local_blocks=args.route_local_blocks,
        route_load_tokens_per_s=args.route_load_tokens_per_s,
        route_saturation_s=args.route_saturation_s,
        rebind_after_s=args.rebind_after_s,
        strict_order=args.strict_order,
        dependency_gap_s=args.dependency_gap_s,
        ordered_admission=args.ordered_admission,
        prewarm=args.prewarm_stripped,
        prewarm_chunk_tokens=args.prewarm_chunk_tokens,
    )
    after_metrics = await _metrics_snapshot(server_urls)
    summary = summarise(
        selection,
        results,
        wall_seconds,
        args.time_scale,
        compute_metric_delta(before_metrics, after_metrics),
        aborted=aborted,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["failed_requests"] or aborted else 0


def main(argv: list[str] | None = None) -> None:
    """Run the Mooncake benchmark CLI.

    Args:
        argv: Optional arguments without the executable name.

    Async/thread-safety:
        Creates and owns the process event loop.
    """

    raise SystemExit(asyncio.run(main_async(parse_args(argv))))


def _parse_request(index: int, payload: Any) -> TraceRequest:
    if not isinstance(payload, dict):
        raise ValueError(f"row {index}: request must be a JSON object")
    timestamp = _number(payload.get("timestamp"), index, "timestamp")
    input_length = _integer(payload.get("input_length"), index, "input_length")
    output_length = _integer(payload.get("output_length"), index, "output_length")
    raw_hash_ids = payload.get("hash_ids")
    if not isinstance(raw_hash_ids, list):
        raise ValueError(f"row {index}: hash_ids must be a list")
    hash_ids = tuple(
        _integer(hash_id, index, f"hash_ids[{position}]")
        for position, hash_id in enumerate(raw_hash_ids)
    )
    if timestamp < 0:
        raise ValueError(f"row {index}: timestamp must be non-negative")
    if input_length <= 0 or output_length <= 0:
        raise ValueError(f"row {index}: input and output lengths must be positive")
    expected_blocks = math.ceil(input_length / MOONCAKE_BLOCK_TOKENS)
    if len(hash_ids) != expected_blocks:
        raise ValueError(
            f"row {index}: expected {expected_blocks} hash IDs, got {len(hash_ids)}"
        )
    if any(hash_id < 0 for hash_id in hash_ids):
        raise ValueError(f"row {index}: hash IDs must be non-negative")
    raw_session = payload.get("session_id")
    session_id = (
        None if raw_session is None else _integer(raw_session, index, "session_id")
    )
    raw_gap = payload.get("gap_ms")
    gap_ms = None if raw_gap is None else _number(raw_gap, index, "gap_ms")
    if gap_ms is not None and gap_ms < 0:
        raise ValueError(f"row {index}: gap_ms must be non-negative")
    raw_prewarm = payload.get("prewarm_tokens")
    prewarm_tokens = (
        None if raw_prewarm is None else _integer(raw_prewarm, index, "prewarm_tokens")
    )
    if prewarm_tokens is not None and not 0 < prewarm_tokens <= input_length:
        raise ValueError(f"row {index}: prewarm_tokens must be in [1, input_length]")
    return TraceRequest(
        index=index,
        timestamp_ms=timestamp,
        input_length=input_length,
        output_length=output_length,
        hash_ids=hash_ids,
        session_id=session_id,
        gap_ms=gap_ms,
        prewarm_tokens=prewarm_tokens,
    )


def _integer(value: Any, index: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"row {index}: {field} must be an integer")
    return value


def _number(value: Any, index: int, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"row {index}: {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"row {index}: {field} must be finite")
    return result


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": statistics.mean(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _prefix_cache_summary(metrics: dict[str, float]) -> dict[str, float]:
    local_queries = metrics.get("vllm:prefix_cache_queries_total", 0.0)
    local_hits = metrics.get("vllm:prefix_cache_hits_total", 0.0)
    external_queries = metrics.get("vllm:external_prefix_cache_queries_total", 0.0)
    external_hits = metrics.get("vllm:external_prefix_cache_hits_total", 0.0)
    return {
        "local_query_tokens": local_queries,
        "local_hit_tokens": local_hits,
        "local_hit_rate": local_hits / local_queries if local_queries else 0.0,
        "external_query_tokens": external_queries,
        "external_hit_tokens": external_hits,
        "external_hit_rate": (
            external_hits / external_queries if external_queries else 0.0
        ),
    }


async def _metrics_snapshot(server_urls: Sequence[str]) -> dict[str, float]:
    """Sum Prometheus counters over every instance (missing ones count as 0)."""
    totals: dict[str, float] = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        for server_url in server_urls:
            try:
                response = await client.get(f"{server_url}/metrics")
                response.raise_for_status()
            except httpx.HTTPError:
                continue
            for name, value in extract_prometheus_counters(response.text).items():
                totals[name] = totals.get(name, 0.0) + value
    return totals


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    fraction = rank - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


async def _rebind_when_waiting(
    router: InstanceRouter,
    instance: int,
    request: TraceRequest,
    stream: asyncio.Future[RequestResult],
    first_token: asyncio.Event,
    wait_s: float,
    poll_s: float = 0.25,
) -> int | None:
    """Return a re-bind target once ``request`` has waited ``wait_s``.

    Returns ``None`` as soon as the stream produces a token or finishes.
    """
    deadline = time.perf_counter() + wait_s
    while not stream.done() and not first_token.is_set():
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            target = router.rebind_target(instance, request)
            if target is not None:
                return target
            remaining = poll_s
        await asyncio.wait({stream}, timeout=remaining)
    return None


async def _prewarm_prefix(
    *,
    request: TraceRequest,
    prompt_builder: PromptBuilder,
    client: httpx.AsyncClient,
    router: InstanceRouter,
    served_model_name: str,
    timeout_seconds: float,
    seed: int,
    chunk_tokens: int = 0,
    stop: asyncio.Event | None = None,
) -> tuple[float, str | None]:
    """Prefill ``request``'s known prefix with one-token completions.

    The prefix requests are routed like the request itself (same session), so
    their KV lands where the request will look for it. With ``chunk_tokens``
    the prefix is sent as growing prefixes ``chunk, 2*chunk, ...`` one after
    another, each reusing the previous one's cached KV; no new piece starts
    once ``stop`` is set (the in-flight piece still finishes).

    Args:
        chunk_tokens: Tokens added per piece; zero sends the whole prefix.
        stop: Set when the idle gap ends.

    Returns:
        ``(elapsed milliseconds, first error or None)``.
    """
    tokens = request.prewarm_tokens or 0
    step = chunk_tokens if chunk_tokens > 0 else tokens
    begin = time.perf_counter()
    for end in range(step, tokens + step, step):
        if end > step and stop is not None and stop.is_set():
            break
        error = await _prefill_prefix(
            request=request,
            tokens=min(end, tokens),
            prompt_builder=prompt_builder,
            client=client,
            router=router,
            served_model_name=served_model_name,
            timeout_seconds=timeout_seconds,
            seed=seed,
        )
        if error is not None:
            return (time.perf_counter() - begin) * 1000.0, error
    return (time.perf_counter() - begin) * 1000.0, None


async def _prefill_prefix(
    *,
    request: TraceRequest,
    tokens: int,
    prompt_builder: PromptBuilder,
    client: httpx.AsyncClient,
    router: InstanceRouter,
    served_model_name: str,
    timeout_seconds: float,
    seed: int,
) -> str | None:
    """Send ``request``'s first ``tokens`` tokens as a one-token completion.

    Returns:
        The request error, or ``None`` on success.
    """
    prefix = replace(
        request,
        input_length=tokens,
        output_length=1,
        hash_ids=request.hash_ids[: math.ceil(tokens / MOONCAKE_BLOCK_TOKENS)],
    )
    instance, work = router.acquire(prefix)
    try:
        result = await vllm_completion_stream(
            client,
            router.server_urls[instance],
            BenchmarkSample(
                sample_id=request.index,
                dataset="mooncake-toolagent",
                context="",
                question="",
                answers=[],
            ),
            prompt_builder.build(prefix),
            {
                "model": served_model_name,
                "max_tokens": 1,
                "temperature": 0.0,
                "ignore_eos": True,
                "seed": seed,
                "request_id": f"trace-{request.index}-warm",
            },
            asyncio.Semaphore(1),
            timeout_seconds,
        )
    finally:
        router.release(instance, work, prefix)
    return result.error


async def _run_request(
    *,
    request: TraceRequest,
    prompt_builder: PromptBuilder,
    client: httpx.AsyncClient,
    router: InstanceRouter,
    served_model_name: str,
    outer_sem: asyncio.Semaphore,
    started_at: float,
    timeout_seconds: float,
    time_scale: float,
    seed: int,
    writer: _ResultWriter,
    wait_for: asyncio.Event | None,
    done: asyncio.Event,
    dependency_gap_s: float = 0.0,
    admit_after: asyncio.Event | None = None,
    admitted: asyncio.Event | None = None,
    prewarm: bool = False,
    prewarm_chunk_tokens: int = 0,
) -> dict[str, Any]:
    try:
        return await _run_request_after_dependency(
            prewarm=prewarm,
            prewarm_chunk_tokens=prewarm_chunk_tokens,
            dependency_gap_s=dependency_gap_s,
            admit_after=admit_after,
            admitted=admitted,
            request=request,
            prompt_builder=prompt_builder,
            client=client,
            router=router,
            served_model_name=served_model_name,
            outer_sem=outer_sem,
            started_at=started_at,
            timeout_seconds=timeout_seconds,
            time_scale=time_scale,
            seed=seed,
            writer=writer,
            wait_for=wait_for,
        )
    finally:
        # Failed or cancelled predecessors must not strand their successors.
        done.set()
        if admitted is not None:
            admitted.set()


async def _run_request_after_dependency(
    *,
    request: TraceRequest,
    prompt_builder: PromptBuilder,
    client: httpx.AsyncClient,
    router: InstanceRouter,
    served_model_name: str,
    outer_sem: asyncio.Semaphore,
    started_at: float,
    timeout_seconds: float,
    time_scale: float,
    seed: int,
    writer: _ResultWriter,
    wait_for: asyncio.Event | None,
    dependency_gap_s: float = 0.0,
    admit_after: asyncio.Event | None = None,
    admitted: asyncio.Event | None = None,
    prewarm: bool = False,
    prewarm_chunk_tokens: int = 0,
) -> dict[str, Any]:
    # A dependent request with a recorded gap is paced by its producer's
    # completion, not by its arrival timestamp.
    gap_paced = wait_for is not None and request.gap_ms is not None
    if not gap_paced:
        scheduled_offset = scheduled_offset_seconds(request.timestamp_ms, time_scale)
        await asyncio.sleep(
            max(0.0, started_at + scheduled_offset - time.perf_counter())
        )
    dependency_at = time.perf_counter()
    warm: asyncio.Task[tuple[float, str | None]] | None = None
    gap_over = asyncio.Event()
    if wait_for is not None:
        await wait_for.wait()
        if prewarm and request.prewarm_tokens:
            warm = asyncio.create_task(
                _prewarm_prefix(
                    request=request,
                    prompt_builder=prompt_builder,
                    client=client,
                    router=router,
                    served_model_name=served_model_name,
                    timeout_seconds=timeout_seconds,
                    seed=seed,
                    chunk_tokens=prewarm_chunk_tokens,
                    stop=gap_over,
                )
            )
        # Idle time after the dependency (tool execution / think time).
        gap_s = dependency_gap_s
        if gap_paced:
            gap_s += scheduled_offset_seconds(request.gap_ms, time_scale)
        await asyncio.sleep(gap_s)
    gap_over.set()
    prewarm_ms: float | None = None
    prewarm_error: str | None = None
    warm_wait_s = 0.0
    if warm is not None:
        # The gateway sends the turn once its prefix is resident; a prewarm
        # outlasting the idle gap delays the request's first token.
        warm_from = time.perf_counter()
        prewarm_ms, prewarm_error = await warm
        warm_wait_s = time.perf_counter() - warm_from
    queued_at = time.perf_counter()
    if gap_paced:
        # The request arrives when its gap ends.
        scheduled_offset = queued_at - started_at
    if admit_after is not None:
        await admit_after.wait()
    async with outer_sem:
        admitted_at = time.perf_counter()
        if admitted is not None:
            admitted.set()
        prompt = prompt_builder.build(request)
        sent_at = admitted_at
        wait_s = router.rebind_wait_s(request)
        instance, work = router.acquire(request)
        rebinds = 0
        resent_at: float | None = None
        stream: asyncio.Future[RequestResult] | None = None

        def record_send() -> None:
            nonlocal sent_at, resent_at
            if rebinds:
                resent_at = time.perf_counter()
            else:
                sent_at = time.perf_counter()

        try:
            while True:
                params: dict[str, Any] = {
                    "model": served_model_name,
                    "max_tokens": request.output_length,
                    "temperature": 0.0,
                    "ignore_eos": True,
                    "seed": seed,
                }
                if rebinds:
                    # A re-sent request must not share its aborted
                    # predecessor's ID in the shared DaseR server.
                    params["request_id"] = f"trace-{request.index}-r{rebinds}"
                first_token = asyncio.Event()
                stream = asyncio.ensure_future(
                    vllm_completion_stream(
                        client,
                        router.server_urls[instance],
                        BenchmarkSample(
                            sample_id=request.index,
                            dataset="mooncake-toolagent",
                            context="",
                            question="",
                            answers=[],
                        ),
                        prompt,
                        params,
                        asyncio.Semaphore(1),
                        timeout_seconds,
                        on_send=record_send,
                        on_first_token=first_token.set,
                    )
                )
                target = None
                if wait_s is not None and not rebinds:
                    target = await _rebind_when_waiting(
                        router, instance, request, stream, first_token, wait_s
                    )
                if target is None:
                    result = await stream
                    break
                stream.cancel()
                await asyncio.gather(stream, return_exceptions=True)
                router.move(instance, target, work, request)
                instance = target
                rebinds += 1
        finally:
            if stream is not None and not stream.done():
                stream.cancel()
            router.release(instance, work, request)
        shift_ms = warm_wait_s * 1000.0
        if resent_at is not None:
            # Time spent on the abandoned instance counts toward TTFT.
            shift_ms += (resent_at - sent_at) * 1000.0
        if shift_ms:
            result = replace(
                result,
                ttft_ms=result.ttft_ms + shift_ms,
                latency_ms=result.latency_ms + shift_ms,
            )
    record = _result_record(
        request=request,
        result=result,
        scheduled_offset=scheduled_offset,
        dependency_at=dependency_at,
        queued_at=queued_at,
        admitted_at=admitted_at,
        sent_at=sent_at,
        completed_at=time.perf_counter(),
        started_at=started_at,
    )
    record["instance"] = instance
    record["session_id"] = request.session_id
    if warm is not None:
        record["prewarm_ms"] = prewarm_ms
        record["prewarm_error"] = prewarm_error
    record["rebinds"] = rebinds
    writer.write(record)
    return record


def _result_record(
    *,
    request: TraceRequest,
    result: RequestResult,
    scheduled_offset: float,
    dependency_at: float,
    queued_at: float,
    admitted_at: float,
    sent_at: float,
    completed_at: float,
    started_at: float,
) -> dict[str, Any]:
    payload = asdict(result)
    payload.pop("generated_text", None)
    payload.pop("queue_ms", None)
    decode_tokens = result.completion_tokens - 1
    tpot_ms = (
        (result.latency_ms - result.ttft_ms) / decode_tokens
        if result.error is None and decode_tokens > 0
        else None
    )
    target_at = started_at + scheduled_offset
    arrival_to_send_ms = max(0.0, sent_at - target_at) * 1000.0
    return {
        "status": "completed" if result.error is None else "failed",
        "trace_index": request.index,
        "source_timestamp_ms": request.timestamp_ms,
        "input_length": request.input_length,
        "output_length": request.output_length,
        "scheduled_offset_s": scheduled_offset,
        "queue_enter_offset_s": queued_at - started_at,
        "admitted_offset_s": admitted_at - started_at,
        "send_offset_s": sent_at - started_at,
        "completion_offset_s": completed_at - started_at,
        "dependency_wait_ms": (queued_at - dependency_at) * 1000.0,
        "client_queue_ms": (admitted_at - queued_at) * 1000.0,
        "arrival_to_send_ms": arrival_to_send_ms,
        "arrival_to_first_token_ms": arrival_to_send_ms + result.ttft_ms,
        "arrival_to_completion_ms": max(0.0, completed_at - target_at) * 1000.0,
        "tpot_ms": tpot_ms,
        **payload,
    }


def _skipped_record(request: TraceRequest) -> dict[str, Any]:
    return {
        "status": "skipped_context_limit",
        "trace_index": request.index,
        "source_timestamp_ms": request.timestamp_ms,
        "input_length": request.input_length,
        "output_length": request.output_length,
    }


def _reject_tmp_output(path: Path) -> None:
    resolved = path.resolve()
    if resolved == Path("/tmp") or Path("/tmp") in resolved.parents:
        raise ValueError("benchmark output must use approved scratch, not /tmp")


class _ResultWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, record: dict[str, Any]) -> None:
        if self._error is not None:
            raise RuntimeError("result writer failed") from self._error
        self._queue.put(record)

    def close(self) -> None:
        self._queue.put(_WRITER_STOP)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("result writer failed") from self._error

    def _run(self) -> None:
        try:
            with self._path.open("w", encoding="utf-8") as output:
                while True:
                    record = self._queue.get()
                    if record is _WRITER_STOP:
                        break
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
        except BaseException as exc:
            self._error = exc


if __name__ == "__main__":
    main()
