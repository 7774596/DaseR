# SPDX-License-Identifier: Apache-2.0
"""Prefill gateway: computes the planned front, then forwards the request."""

import json
from typing import Any

from fastapi.testclient import TestClient
import httpx
import pytest

from daser.gateway import GatewayConfig, build_gateway_app

BLOCK_TOKENS = 4


class _PlanIPC:
    """Async IPC stub returning a fixed split plan."""

    def __init__(self, compute_tokens: int) -> None:
        self.compute_tokens = compute_tokens
        self.plans: list[tuple[list[int], int]] = []

    async def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Answer the runtime-config probe."""
        assert payload == {"op": "get_runtime_config"}
        return {"runtime_config": {"model_id": "m", "block_tokens": BLOCK_TOKENS}}

    async def split_prefill(
        self,
        tokens: list[int],
        model_id: str,
        external_prefix_queries: int,
        compute_tokens_per_s: float,
        load_bytes_per_s: float,
    ) -> dict[str, Any]:
        """Record the plan request and return the configured split."""
        assert model_id == "m"
        self.plans.append((list(tokens), external_prefix_queries))
        return {"compute_tokens": self.compute_tokens, "hit_tokens": len(tokens)}

    async def close(self) -> None:
        """Nothing to close."""


def _vllm_mock(bodies: list[dict[str, Any]]) -> httpx.MockTransport:
    """vLLM stand-in recording completion bodies and answering metrics."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text="vllm_up 1\n")
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"text": "x"}]})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    ("split", "compute_tokens", "expect_front"),
    [
        (True, 2 * BLOCK_TOKENS, True),
        (True, 0, False),
        (False, 2 * BLOCK_TOKENS, False),
    ],
)
def test_gateway_prefills_front_without_daser_loads(
    split: bool, compute_tokens: int, expect_front: bool
) -> None:
    """With a split plan the front prefill precedes the unchanged request.

    The front request carries only the planned tokens and disables DaseR
    loads and saves; with no plan or the switch off the request is forwarded
    alone and unchanged.
    """
    bodies: list[dict[str, Any]] = []
    ipc = _PlanIPC(compute_tokens)
    cfg = GatewayConfig(
        vllm_url="http://vllm", socket_path="unused", split=split, min_compute_tokens=1
    )
    prompt = list(range(3 * BLOCK_TOKENS + 1))
    request = {"model": "m", "prompt": prompt, "max_tokens": 4}
    with TestClient(
        build_gateway_app(cfg, transport=_vllm_mock(bodies), ipc=ipc)
    ) as client:
        resp = client.post("/v1/completions", json=request)
        metrics = client.get("/metrics")

    assert resp.status_code == 200
    assert metrics.text == "vllm_up 1\n"
    assert bodies[-1] == request
    if expect_front:
        assert ipc.plans == [(prompt[: 3 * BLOCK_TOKENS], len(prompt))]
        front = bodies[0]
        assert front["prompt"] == prompt[:compute_tokens]
        assert front["max_tokens"] == 1
        assert front["kv_transfer_params"] == {
            "daser_skip_load": True,
            "daser_skip_save": True,
        }
        assert len(bodies) == 2
    else:
        assert len(bodies) == 1
