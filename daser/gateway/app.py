# SPDX-License-Identifier: Apache-2.0
"""Gateway that splits a cached prefix into a computed front and a read back.

For a token-ID completion whose prefix DaseR holds, the gateway asks the
server for a split point ``x`` (``split_prefill``): the server starts reading
the stored KV of ``[x, hit)`` and the gateway prefills ``[0, x)`` on vLLM with
DaseR loads disabled. The original request then finds ``[0, x)`` in vLLM's
prefix cache and loads only the already-read back part, so GPU compute and
storage reads overlap instead of the GPU idling during the read.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import httpx

from daser.connector.ipc_client import IPCClientAsync
from daser.logging import init_logger

logger = init_logger(__name__)


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
    """

    vllm_url: str
    socket_path: str
    split: bool = False
    compute_tokens_per_s: float = 30000.0
    load_bytes_per_s: float = 15e9
    min_compute_tokens: int = 4096
    timeout_s: float = 3600.0


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
        logger.info(
            "[GATEWAY] ready vllm=%s split=%s compute_tps=%.0f load_bps=%.3g",
            cfg.vllm_url,
            cfg.split,
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
        if cfg.split and isinstance(prompt, list) and prompt:
            if all(isinstance(token, int) for token in prompt):
                await _prefill_front(state, cfg, body, prompt)
        return await _forward(state["http"], "/v1/completions", body)

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
    http: httpx.AsyncClient, path: str, body: dict[str, Any]
) -> Response:
    """Forward a completion to vLLM, streaming the reply when requested."""
    if not body.get("stream"):
        upstream = await http.post(path, json=body)
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )
    upstream = await http.send(http.build_request("POST", path, json=body), stream=True)

    async def chunks() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        chunks(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )
