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
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        """Validate the priority and admission policies.

        Raises:
            ValueError: if a policy is unknown or ``admit_slots`` < 1.
        """
        if self.priority not in PRIORITY_MODES:
            raise ValueError(f"unknown gateway priority {self.priority!r}")
        if self.admit not in ADMIT_MODES:
            raise ValueError(f"unknown gateway admit {self.admit!r}")
        if self.admit_slots < 1:
            raise ValueError("gateway admit_slots must be at least 1")


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
            transport=transport,
        )
        state["ipc"] = ipc if ipc is not None else IPCClientAsync(cfg.socket_path)
        resp = await state["ipc"].call({"op": "get_runtime_config"})
        config = resp.get("runtime_config", {})
        state["model_id"] = str(config.get("model_id", "default"))
        state["block_tokens"] = int(config.get("block_tokens", 0))
        state["admit"] = asyncio.Semaphore(cfg.admit_slots)
        logger.info(
            "[GATEWAY] ready vllm=%s split=%s priority=%s admit=%s/%d/%d "
            "compute_tps=%.0f load_bps=%.3g",
            cfg.vllm_url,
            cfg.split,
            cfg.priority,
            cfg.admit,
            cfg.admit_tokens,
            cfg.admit_slots,
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
        if token_prompt and "residual" in (cfg.priority, cfg.admit):
            residual = await _residual(state, prompt)
        if token_prompt and cfg.priority != "none":
            body["priority"] = len(prompt) if cfg.priority == "length" else residual
        release: Callable[[], None] | None = None
        if token_prompt and cfg.admit != "none":
            work = len(prompt) if cfg.admit == "length" else residual
            if work is not None and work >= cfg.admit_tokens:
                release = await _acquire_large_slot(state["admit"])
        try:
            if token_prompt and cfg.split:
                await _prefill_front(state, cfg, body, prompt)
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
