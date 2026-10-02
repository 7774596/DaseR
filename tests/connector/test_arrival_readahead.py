# SPDX-License-Identifier: Apache-2.0
"""Scheduler sends a readahead hint when a request enters the waiting queue."""

from types import SimpleNamespace

import pytest

from daser.config import CACHE_REUSE_CHUNK
from daser.connector.scheduler.lifecycle import RequestLifecycle

BLOCK_TOKENS = 4


class _HintIPC:
    """Synchronous IPC stub recording readahead hints."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.hints: list[tuple[list[int], str, int]] = []

    def readahead_hint(
        self, tokens: list[int], model_id: str, external_prefix_queries: int
    ) -> bool:
        """Record one hint, or fail like a dropped server connection."""
        if self.fail:
            raise ConnectionError("server gone")
        self.hints.append((list(tokens), model_id, external_prefix_queries))
        return True


def _lifecycle(ipc: _HintIPC, *, arrival_readahead: bool) -> RequestLifecycle:
    return RequestLifecycle(
        ipc_client=ipc,
        block_tokens=BLOCK_TOKENS,
        slot_size=32,
        model_id="m",
        cache_reuse_mode=CACHE_REUSE_CHUNK,
        runtime_config_ready=True,
        arrival_readahead=arrival_readahead,
    )


def _request(num_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(request_id="req-1", prompt_token_ids=list(range(num_tokens)))


@pytest.mark.parametrize("arrival_readahead", [True, False])
def test_new_request_hints_block_aligned_prompt(arrival_readahead: bool) -> None:
    """The hint carries the block-aligned prompt; the switch defaults off."""
    ipc = _HintIPC()
    _lifecycle(ipc, arrival_readahead=arrival_readahead).on_new_request(
        _request(2 * BLOCK_TOKENS + 1)
    )
    expected = [(list(range(2 * BLOCK_TOKENS)), "m", 2 * BLOCK_TOKENS + 1)]
    assert ipc.hints == (expected if arrival_readahead else [])


def test_new_request_hint_skips_short_prompts_and_tolerates_errors() -> None:
    """A prompt shorter than one block sends nothing; IPC errors are ignored."""
    ipc = _HintIPC()
    _lifecycle(ipc, arrival_readahead=True).on_new_request(_request(BLOCK_TOKENS - 1))
    assert ipc.hints == []
    _lifecycle(_HintIPC(fail=True), arrival_readahead=True).on_new_request(
        _request(BLOCK_TOKENS)
    )
