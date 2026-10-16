# SPDX-License-Identifier: Apache-2.0
"""Scheduler protects a queued request's cached prefix until it is scheduled."""

from types import SimpleNamespace
from typing import Any

import pytest

from daser.config import CACHE_REUSE_CHUNK
from daser.connector.scheduler.lifecycle import RequestLifecycle


class _ProtectIPC:
    """Synchronous IPC stub recording protection calls."""

    def __init__(self, budget: int, *, fail: bool = False, keys: int = 2) -> None:
        self.budget = budget
        self.fail = fail
        self.keys = keys
        self.calls: list[tuple[str, str]] = []

    def get_runtime_config(self) -> dict[str, Any]:
        return {
            "slot_size": 32,
            "block_tokens": 4,
            "protect_waiting_bytes": self.budget,
        }

    def protect_prefix(self, lease_id: str, tokens: list[int]) -> int:
        if self.fail:
            raise ConnectionError("server gone")
        self.calls.append(("protect", lease_id))
        return self.keys

    def release_protection(self, lease_id: str) -> None:
        self.calls.append(("release", lease_id))


def _lifecycle(ipc: _ProtectIPC) -> RequestLifecycle:
    lifecycle = RequestLifecycle(
        ipc_client=ipc,
        block_tokens=4,
        slot_size=32,
        model_id="m",
        cache_reuse_mode=CACHE_REUSE_CHUNK,
        runtime_config_ready=True,
    )
    lifecycle.refresh_runtime_config()
    return lifecycle


def _request() -> SimpleNamespace:
    return SimpleNamespace(request_id="req-1", prompt_token_ids=list(range(9)))


@pytest.mark.parametrize("finish", [True, False])
def test_queued_request_is_protected_until_alloc_or_finish(finish: bool) -> None:
    """Arrival protects; allocation or finishing releases exactly once."""
    ipc = _ProtectIPC(budget=1 << 20)
    lifecycle = _lifecycle(ipc)
    request = _request()
    lifecycle.on_new_request(request)
    if finish:
        lifecycle.request_finished(request, [])
    else:
        blocks = SimpleNamespace(blocks=[[]])
        lifecycle.update_state_after_alloc(request, blocks, 0)
    lifecycle.request_finished(request, [])
    assert ipc.calls == [("protect", "req-1"), ("release", "req-1")]


@pytest.mark.parametrize("ipc", [_ProtectIPC(0), _ProtectIPC(1 << 20, keys=0)])
def test_no_protection_sends_no_release(ipc: _ProtectIPC) -> None:
    """With protection off, or nothing protected, no release is sent."""
    lifecycle = _lifecycle(ipc)
    request = _request()
    lifecycle.on_new_request(request)
    lifecycle.request_finished(request, [])
    assert ("release", "req-1") not in ipc.calls
    if ipc.budget == 0:
        assert ipc.calls == []


def test_protect_failure_is_ignored() -> None:
    """An IPC error leaves the request schedulable and unprotected."""
    ipc = _ProtectIPC(1 << 20, fail=True)
    lifecycle = _lifecycle(ipc)
    request = _request()
    lifecycle.on_new_request(request)
    lifecycle.request_finished(request, [])
    assert ipc.calls == []
