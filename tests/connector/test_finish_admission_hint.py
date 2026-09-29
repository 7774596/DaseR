# SPDX-License-Identifier: Apache-2.0
"""Scheduler reports written chunks when finish admission is enabled."""

from types import SimpleNamespace
from typing import Any

import pytest

from daser.config import CACHE_REUSE_CHUNK
from daser.connector.scheduler.lifecycle import RequestLifecycle

BLOCK_TOKENS = 4


class _FakeIPC:
    """Synchronous IPC stub recording allocation and finish calls."""

    def __init__(self, finish_admission: bool) -> None:
        self.finish_admission = finish_admission
        self.finish_calls: list[tuple[list[str], int]] = []

    def get_runtime_config(self) -> dict[str, Any]:
        """Return a minimal server runtime config."""
        return {
            "slot_size": 32,
            "block_tokens": BLOCK_TOKENS,
            "model_id": "m",
            "cache_reuse_mode": CACHE_REUSE_CHUNK,
            "finish_admission": self.finish_admission,
        }

    def alloc_chunk(self, chunk_key: str, token_count: int, model_id: str) -> dict:
        """Return server-style allocation metadata."""
        del token_count, model_id
        return {"chunk_key": chunk_key, "start_slot": 0, "file_offset": 0}

    def finish_request(self, chunk_keys: list[str], output_tokens: int) -> None:
        """Record a finish-admission hint."""
        self.finish_calls.append((list(chunk_keys), output_tokens))


def _request(req_id: str, num_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=req_id,
        prompt_token_ids=list(range(num_tokens)),
        kv_transfer_params={"daser_skip_load": True},
        num_output_tokens=600,
    )


def _run_store_request(ipc: _FakeIPC) -> None:
    lifecycle = RequestLifecycle(
        ipc_client=ipc,
        block_tokens=BLOCK_TOKENS,
        slot_size=32,
        model_id="m",
        cache_reuse_mode=CACHE_REUSE_CHUNK,
        runtime_config_ready=False,
    )
    lifecycle.refresh_runtime_config()
    request = _request("req", 3 * BLOCK_TOKENS)
    assert lifecycle.get_num_new_matched_tokens(request, 0) == (0, False)
    blocks = SimpleNamespace(
        blocks=[[SimpleNamespace(block_id=idx) for idx in range(3)]]
    )
    lifecycle.update_state_after_alloc(request, blocks, 0)
    lifecycle.request_finished(request, [0, 1, 2])
    # A second finish for the same request must not resend the chunks.
    lifecycle.request_finished(request, [0, 1, 2])


@pytest.mark.parametrize("enabled", [True, False])
def test_request_finished_reports_written_chunks(enabled: bool) -> None:
    ipc = _FakeIPC(finish_admission=enabled)
    _run_store_request(ipc)
    if not enabled:
        assert ipc.finish_calls == []
        return
    assert len(ipc.finish_calls) == 1
    chunk_keys, output_tokens = ipc.finish_calls[0]
    assert len(chunk_keys) == 1
    assert output_tokens == 600
