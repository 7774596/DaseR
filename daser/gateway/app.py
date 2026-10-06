# SPDX-License-Identifier: Apache-2.0
"""Gateway that plans DaseR-aware prefill in front of vLLM.

For a token-ID completion whose prefix DaseR holds, the gateway asks the
server for a split point ``x`` (``split_prefill``): the server starts reading
the stored KV of ``[x, hit)`` and the gateway prefills ``[0, x)`` on vLLM with
DaseR loads disabled. The original request then finds ``[0, x)`` in vLLM's
prefix cache and loads only the already-read back part, so GPU compute and
storage reads overlap instead of the GPU idling during the read.

The gateway can also set each completion's vLLM ``priority`` (lower runs
first; vLLM needs ``--scheduling-policy priority``). ``length`` orders by
prompt tokens. ``residual`` orders by the tokens left to compute after DaseR's
reusable prefix, counting chunks still being written, so a long prompt whose
prefix DaseR holds is not queued as a large job.

Admission control gates large jobs instead of reordering vLLM's queue: a
completion whose work (prompt tokens under ``length``, residual tokens under
``residual``) reaches ``admit_tokens`` waits for one of ``admit_slots`` slots
and holds it until its first streamed chunk, so at most ``admit_slots`` large
prefills share vLLM's token budget while small jobs pass straight through.

Dispatch holds completions in the gateway and forwards them to vLLM only while
fewer than ``dispatch_slots`` are waiting for their first streamed chunk, so
vLLM's own queue stays short and the gateway picks who runs next. ``residual``
releases the held request with the fewest tokens left after DaseR's reusable
prefix first (shortest remaining prefill), ``length`` the shortest prompt, and
``fifo`` keeps arrival order (the hold-back ablation). ``residual-fresh``
re-estimates the residual of the ``dispatch_refresh`` best-ranked held requests
whenever a slot frees, so shared prefixes written while they waited count.
``cost-fresh`` refreshes the same way but ranks by estimated prefill cost: each
of the ``m`` missing tokens pays a linear part plus attention over the ``h``
reusable tokens and the earlier misses, ``m * (1 + (h + m / 2) / T)``, where
``T`` (``dispatch_attn_tokens``) is the context length at which a token's
attention FLOPs equal its linear-layer FLOPs.

Hole filling recovers stored KV that a prefix lookup cannot reach. A rolling
key names its whole prefix, so a block DaseR still holds after an evicted one
is exact KV of this prompt, but vLLM and DaseR both stop matching at the first
missing block. ``fill_holes`` asks the server for such gaps and, for each one
followed by at least ``fill_min_tail_tokens`` held tokens, prefills the prompt
up to the gap's end with DaseR loads and saves on: the shadow loads the held
prefix, computes only the gap and stores it, so the original request then
reaches the held blocks behind the gap instead of recomputing them.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
import heapq
import itertools
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import httpx
from starlette.background import BackgroundTask

from daser.connector.ipc_client import IPCClientAsync
from daser.logging import init_logger

logger = init_logger(__name__)

PRIORITY_MODES = ("none", "length", "residual")
ADMIT_MODES = ("none", "length", "residual")
DISPATCH_MODES = (
    "none",
    "fifo",
    "length",
    "residual",
    "residual-fresh",
    "cost-fresh",
)
_KEEPALIVE_EXPIRY_S = 2.0


def prefill_cost(prompt_tokens: int, residual: int, attn_tokens: float) -> float:
    """Estimate prefill work for ``residual`` missing tokens of a prompt.

    Args:
        prompt_tokens: Prompt length.
        residual: Tokens left to compute after the reusable prefix.
        attn_tokens: Context length at which one token's attention FLOPs equal
            its linear-layer FLOPs.

    Returns:
        Cost in linear-token units: each missing token costs 1 plus its
        attention over the reusable prefix and the misses before it.
    """
    reusable = max(prompt_tokens - residual, 0)
    return residual * (1.0 + (reusable + residual / 2.0) / attn_tokens)


@dataclass(frozen=True)
class GatewayConfig:
    """Gateway runtime configuration.

    Attributes:
        vllm_url: Base URL of the vLLM OpenAI server.
        socket_path: DaseR server IPC socket.
        split: Whether to plan bidirectional prefill; off forwards unchanged.
        compute_tokens_per_s: GPU prefill rate used to size the front part.
        load_bytes_per_s: Stored-byte read rate used to size the back part.
        min_compute_tokens: Smallest front part worth a separate prefill.
        timeout_s: Upstream request timeout.
        priority: vLLM priority policy: ``none`` leaves requests unchanged,
            ``length`` uses prompt tokens and ``residual`` uses prompt tokens
            minus DaseR's reusable prefix.
        admit: Large-job admission policy: ``none`` admits everything,
            ``length`` gates by prompt tokens and ``residual`` by prompt
            tokens minus DaseR's reusable prefix.
        admit_tokens: Work at or above which a request is a large job.
        admit_slots: Large jobs allowed in prefill at once.
        dispatch: Gateway-held release order: ``none`` forwards at once,
            ``fifo`` by arrival, ``length`` by prompt tokens and ``residual``
            by prompt tokens minus DaseR's reusable prefix; ``residual-fresh``
            also re-estimates the best-ranked held requests at each release.
        dispatch_slots: Forwarded completions allowed to await their first
            streamed chunk at once under ``dispatch``.
        dispatch_refresh: Best-ranked held requests re-estimated per release
            under ``residual-fresh`` and ``cost-fresh``.
        dispatch_attn_tokens: Context length at which one token's attention
            FLOPs equal its linear FLOPs (``cost-fresh`` only).
        fill_holes: Recompute gaps that hide stored blocks before forwarding.
        fill_min_tail_tokens: Smallest held run after a gap worth a shadow
            prefill.
        fill_max_holes: Most gaps filled per request (earliest first).
    """

    vllm_url: str
    socket_path: str
    split: bool = False
    compute_tokens_per_s: float = 30000.0
    load_bytes_per_s: float = 15e9
    min_compute_tokens: int = 4096
    timeout_s: float = 3600.0
    priority: str = "none"
    admit: str = "none"
    admit_tokens: int = 8192
    admit_slots: int = 1
    dispatch: str = "none"
    dispatch_slots: int = 4
    dispatch_refresh: int = 8
    dispatch_attn_tokens: float = 13500.0
    fill_holes: bool = False
    fill_min_tail_tokens: int = 4096
    fill_max_holes: int = 2

    def __post_init__(self) -> None:
        """Validate the priority and admission policies.

        Raises:
            ValueError: if a policy is unknown or a slot count is < 1.
        """
        if self.priority not in PRIORITY_MODES:
            raise ValueError(f"unknown gateway priority {self.priority!r}")
        if self.admit not in ADMIT_MODES:
            raise ValueError(f"unknown gateway admit {self.admit!r}")
        if self.admit_slots < 1:
            raise ValueError("gateway admit_slots must be at least 1")
        if self.dispatch not in DISPATCH_MODES:
            raise ValueError(f"unknown gateway dispatch {self.dispatch!r}")
        if self.dispatch_slots < 1:
            raise ValueError("gateway dispatch_slots must be at least 1")
        if self.dispatch_refresh < 1:
            raise ValueError("gateway dispatch_refresh must be at least 1")
        if self.dispatch_attn_tokens <= 0:
            raise ValueError("gateway dispatch_attn_tokens must be positive")
        if self.fill_max_holes < 1:
            raise ValueError("gateway fill_max_holes must be at least 1")
        if self.fill_holes and self.split:
            raise ValueError("gateway fill_holes and split are exclusive")


class SlotDispatcher:
    """Hand out a fixed number of slots, smallest key first.

    Waiters queue by ``(key, arrival)``; a released slot goes to the waiter
    with the smallest key, so the gateway forwards shortest-work-first while
    capping how many forwarded requests are in flight. With ``rekey`` set, a
    release first re-keys the ``refresh`` best-ranked waiters from their
    payloads, so keys that went stale while waiting are corrected before the
    slot is granted.

    Async/thread-safety:
        Use from one event loop; not thread-safe.
    """

    def __init__(
        self,
        slots: int,
        rekey: Callable[[Any], Awaitable[float]] | None = None,
        refresh: int = 8,
    ) -> None:
        """Create a dispatcher.

        Args:
            slots: Slots available at once; must be at least 1.
            rekey: Optional coroutine mapping a waiter's payload to a fresh
                key; failures keep the old key.
            refresh: Best-ranked waiters re-keyed per release under ``rekey``.
        """
        self._free = slots
        self._waiters: list[tuple[float, int, asyncio.Future[None], Any]] = []
        self._order = itertools.count()
        self._rekey = rekey
        self._refresh = refresh
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def waiting(self) -> int:
        """Number of callers still queued for a slot (including cancelled)."""
        return len(self._waiters)

    async def acquire(self, key: float, payload: Any = None) -> Callable[[], None]:
        """Wait for a slot and return an idempotent release callback.

        Args:
            key: Release order; smaller keys get slots first, ties by arrival.
            payload: Passed to ``rekey`` when the dispatcher refreshes keys.

        Returns:
            Callback that returns the slot; safe to call more than once.

        Raises:
            asyncio.CancelledError: if cancelled while waiting; a slot already
                handed to the caller is passed on, not leaked.
        """
        if self._free > 0 and not self._waiters:
            self._free -= 1
        else:
            fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            heapq.heappush(self._waiters, (key, next(self._order), fut, payload))
            try:
                await fut
            except asyncio.CancelledError:
                if fut.done() and not fut.cancelled():
                    self._release()
                raise
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                self._release()

        return release

    def _release(self) -> None:
        """Grant a freed slot, re-keying the best waiters first if enabled."""
        if self._rekey is None or len(self._waiters) < 2:
            self._grant()
            return
        # The slot stays reserved (not free) until the refreshed grant runs, so
        # a newcomer cannot take it while keys are being recomputed.
        task = asyncio.get_running_loop().create_task(self._grant_refreshed())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _grant_refreshed(self) -> None:
        """Re-key the best-ranked live waiters, then grant the reserved slot."""
        assert self._rekey is not None
        async with self._lock:
            heads: list[tuple[float, int, asyncio.Future[None], Any]] = []
            while self._waiters and len(heads) < self._refresh:
                item = heapq.heappop(self._waiters)
                if not item[2].done():
                    heads.append(item)
            keys = await asyncio.gather(
                *(self._rekey(item[3]) for item in heads), return_exceptions=True
            )
            for (old, order, fut, payload), new in zip(heads, keys, strict=True):
                key = old if isinstance(new, BaseException) else float(new)
                heapq.heappush(self._waiters, (key, order, fut, payload))
            self._grant()

    def _grant(self) -> None:
        """Give the slot to the smallest live waiter or return it to the pool."""
        while self._waiters:
            fut = heapq.heappop(self._waiters)[2]
            if not fut.done():
                fut.set_result(None)
                return
        self._free += 1


def build_gateway_app(
    cfg: GatewayConfig,
    transport: httpx.AsyncBaseTransport | None = None,
    ipc: Any | None = None,
) -> FastAPI:
    """Build the gateway FastAPI app.

    Args:
        cfg: Gateway configuration.
        transport: Optional HTTP transport to vLLM (tests inject a mock).
        ipc: Optional async IPC client; defaults to one on ``socket_path``.

    Returns:
        App serving ``POST /v1/completions`` with optional bidirectional
        prefill and forwarding every other path to vLLM unchanged.

    Async/thread-safety:
        Handlers run on the uvicorn event loop; the IPC client serializes its
        calls on one connection.
    """
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["http"] = httpx.AsyncClient(
            base_url=cfg.vllm_url,
            timeout=httpx.Timeout(cfg.timeout_s),
            # Drop idle connections before vLLM's uvicorn (5 s keep-alive)
            # closes them, so a held request is never sent on a socket the
            # server is closing (httpx.ReadError).
            limits=httpx.Limits(keepalive_expiry=_KEEPALIVE_EXPIRY_S),
            transport=transport,
        )
        state["ipc"] = ipc if ipc is not None else IPCClientAsync(cfg.socket_path)
        resp = await state["ipc"].call({"op": "get_runtime_config"})
        config = resp.get("runtime_config", {})
        state["model_id"] = str(config.get("model_id", "default"))
        state["block_tokens"] = int(config.get("block_tokens", 0))
        state["admit"] = asyncio.Semaphore(cfg.admit_slots)
        fresh = cfg.dispatch.endswith("-fresh")

        async def rekey(prompt: list[int]) -> float:
            return _dispatch_key(cfg, prompt, await _residual(state, prompt))

        state["dispatch"] = SlotDispatcher(
            cfg.dispatch_slots,
            rekey=rekey if fresh else None,
            refresh=cfg.dispatch_refresh,
        )
        logger.info(
            "[GATEWAY] ready vllm=%s split=%s priority=%s admit=%s/%d/%d "
            "dispatch=%s/%d compute_tps=%.0f load_bps=%.3g",
            cfg.vllm_url,
            cfg.split,
            cfg.priority,
            cfg.admit,
            cfg.admit_tokens,
            cfg.admit_slots,
            cfg.dispatch,
            cfg.dispatch_slots,
            cfg.compute_tokens_per_s,
            cfg.load_bytes_per_s,
        )
        try:
            yield
        finally:
            await state["http"].aclose()
            await state["ipc"].close()

    app = FastAPI(lifespan=lifespan)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        body = await request.json()
        prompt = body.get("prompt")
        token_prompt = (
            isinstance(prompt, list)
            and bool(prompt)
            and all(isinstance(token, int) for token in prompt)
        )
        residual: int | None = None
        if token_prompt and (
            "residual" in (cfg.priority, cfg.admit)
            or cfg.dispatch.endswith(("residual", "-fresh"))
        ):
            residual = await _residual(state, prompt)
        if token_prompt and cfg.priority != "none":
            body["priority"] = len(prompt) if cfg.priority == "length" else residual
        releases: list[Callable[[], None]] = []
        if token_prompt and cfg.dispatch != "none":
            key = _dispatch_key(cfg, prompt, residual)
            releases.append(await state["dispatch"].acquire(key, prompt))
        if token_prompt and cfg.admit != "none":
            work = len(prompt) if cfg.admit == "length" else residual
            if work is not None and work >= cfg.admit_tokens:
                try:
                    releases.append(await _acquire_large_slot(state["admit"]))
                except BaseException:
                    for fn in releases:
                        fn()
                    raise

        def release_all() -> None:
            for fn in releases:
                fn()

        release = release_all if releases else None
        try:
            if token_prompt and cfg.split:
                await _prefill_front(state, cfg, body, prompt)
            if token_prompt and cfg.fill_holes:
                await _fill_holes(state, cfg, body, prompt)
            return await _forward(state["http"], "/v1/completions", body, release)
        except BaseException:
            if release is not None:
                release()
            raise

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def passthrough(path: str, request: Request) -> Response:
        upstream = await state["http"].request(
            request.method,
            f"/{path}",
            content=await request.body(),
            headers={"content-type": request.headers.get("content-type", "")},
        )
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )

    return app


def _dispatch_key(cfg: GatewayConfig, prompt: list[int], residual: int | None) -> float:
    """Release-order key for a held completion (smaller is released first)."""
    if cfg.dispatch == "fifo":
        return time.monotonic()
    if cfg.dispatch == "length":
        return float(len(prompt))
    if cfg.dispatch == "cost-fresh":
        return prefill_cost(len(prompt), residual or 0, cfg.dispatch_attn_tokens)
    return float(residual or 0)


async def _acquire_large_slot(slots: asyncio.Semaphore) -> Callable[[], None]:
    """Wait for a large-job slot and return an idempotent release callback."""
    await slots.acquire()
    released = False

    def release() -> None:
        nonlocal released
        if not released:
            released = True
            slots.release()

    return release


async def _residual(state: dict[str, Any], prompt: list[int]) -> int:
    """Return prompt tokens left to compute after DaseR's reusable prefix."""
    try:
        reusable = await state["ipc"].estimate_reuse(prompt, state["model_id"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("[GATEWAY] reuse estimate failed: %s", exc)
        reusable = 0
    # vLLM computes at least the last prompt token.
    residual = max(1, len(prompt) - min(reusable, len(prompt)))
    logger.info(
        "[GATEWAY] residual prompt=%d reusable=%d residual=%d",
        len(prompt),
        reusable,
        residual,
    )
    return residual


async def _prefill_front(
    state: dict[str, Any],
    cfg: GatewayConfig,
    body: dict[str, Any],
    prompt: list[int],
) -> None:
    """Prefill the planned front part on vLLM while DaseR reads the back."""
    block_tokens = state["block_tokens"]
    aligned = (len(prompt) // block_tokens) * block_tokens if block_tokens else 0
    if aligned == 0:
        return
    started_at = time.perf_counter()
    try:
        plan = await state["ipc"].split_prefill(
            prompt[:aligned],
            state["model_id"],
            len(prompt),
            cfg.compute_tokens_per_s,
            cfg.load_bytes_per_s,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[GATEWAY] split plan failed: %s", exc)
        return
    front = int(plan.get("compute_tokens", 0))
    if front < cfg.min_compute_tokens:
        return
    planned_at = time.perf_counter()
    shadow = {
        "model": body.get("model", ""),
        "prompt": prompt[:front],
        "max_tokens": 1,
        "temperature": 0.0,
        "kv_transfer_params": {"daser_skip_load": True, "daser_skip_save": True},
    }
    try:
        resp = await state["http"].post("/v1/completions", json=shadow)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[GATEWAY] front prefill failed: %s", exc)
        return
    logger.info(
        "[GATEWAY] split hit=%d front=%d readahead=%s plan_ms=%.1f front_ms=%.1f",
        int(plan.get("hit_tokens", 0)),
        front,
        plan.get("started"),
        (planned_at - started_at) * 1000,
        (time.perf_counter() - planned_at) * 1000,
    )


async def _fill_holes(
    state: dict[str, Any],
    cfg: GatewayConfig,
    body: dict[str, Any],
    prompt: list[int],
) -> None:
    """Recompute gaps that hide stored blocks, so the request reaches them.

    Each shadow prefill ends at a gap's end; it loads the held prefix, which
    an earlier shadow made contiguous, computes the gap and saves it.
    """
    started_at = time.perf_counter()
    try:
        holes = await state["ipc"].prefill_holes(
            prompt, state["model_id"], cfg.fill_min_tail_tokens
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[GATEWAY] hole probe failed: %s", exc)
        return
    holes = holes[: cfg.fill_max_holes]
    if not holes:
        return
    planned_at = time.perf_counter()
    filled = 0
    for start, end in holes:
        # vLLM computes at least one prompt token, so end below the prompt.
        if end >= len(prompt):
            break
        shadow = {
            "model": body.get("model", ""),
            "prompt": prompt[:end],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        try:
            resp = await state["http"].post("/v1/completions", json=shadow)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[GATEWAY] hole fill [%d, %d) failed: %s", start, end, exc)
            break
        filled += end - start
    logger.info(
        "[GATEWAY] fill holes=%d hole_tokens=%d prompt=%d plan_ms=%.1f fill_ms=%.1f",
        len(holes),
        filled,
        len(prompt),
        (planned_at - started_at) * 1000,
        (time.perf_counter() - planned_at) * 1000,
    )


async def _forward(
    http: httpx.AsyncClient,
    path: str,
    body: dict[str, Any],
    on_first_chunk: Callable[[], None] | None = None,
) -> Response:
    """Forward a completion to vLLM, streaming the reply when requested.

    ``on_first_chunk`` runs once the first streamed body chunk (or the whole
    non-streamed reply) arrives, and at the latest when the stream closes.
    """
    if not body.get("stream"):
        try:
            upstream = await http.post(path, json=body)
        finally:
            if on_first_chunk is not None:
                on_first_chunk()
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )
    upstream = await http.send(http.build_request("POST", path, json=body), stream=True)

    async def chunks() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                if on_first_chunk is not None:
                    on_first_chunk()
                yield chunk
        finally:
            if on_first_chunk is not None:
                on_first_chunk()
            await upstream.aclose()

    return StreamingResponse(
        chunks(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        # Runs even when the client disconnects before the body starts.
        background=None if on_first_chunk is None else BackgroundTask(on_first_chunk),
    )
