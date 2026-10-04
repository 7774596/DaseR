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
        self.hints: list[tuple[list[int], str, int, int]] = []

    def readahead_hint(
        self,
        tokens: list[int],
        model_id: str,
        external_prefix_queries: int,
        num_computed_tokens: int = 0,
    ) -> bool:
        """Record one hint, or fail like a dropped server connection."""
        if self.fail:
            raise ConnectionError("server gone")
        self.hints.append(
            (list(tokens), model_id, external_prefix_queries, num_computed_tokens)
        )
        return True


class _CachedPool:
    """GPU block pool stub holding a set of block hashes."""

    hash_block_size = 2

    def __init__(self, cached: set[int]) -> None:
        self.cached = cached

    def get_cached_block(self, block_hash: int, group_ids: list[int]) -> object:
        """Return a block marker when the hash is cached, else None."""
        assert group_ids == [0]
        return object() if block_hash in self.cached else None


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
    hashes = list(range(num_tokens // _CachedPool.hash_block_size))
    return SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(num_tokens)),
        block_hashes=hashes,
    )


@pytest.mark.parametrize("arrival_readahead", [True, False])
def test_new_request_hints_block_aligned_prompt(arrival_readahead: bool) -> None:
    """The hint carries the block-aligned prompt; the switch defaults off."""
    ipc = _HintIPC()
    _lifecycle(ipc, arrival_readahead=arrival_readahead).on_new_request(
        _request(2 * BLOCK_TOKENS + 1)
    )
    expected = [(list(range(2 * BLOCK_TOKENS)), "m", 2 * BLOCK_TOKENS + 1, 0)]
    assert ipc.hints == (expected if arrival_readahead else [])


def test_new_request_hint_skips_short_prompts_and_tolerates_errors() -> None:
    """A prompt shorter than one block sends nothing; IPC errors are ignored."""
    ipc = _HintIPC()
    _lifecycle(ipc, arrival_readahead=True).on_new_request(_request(BLOCK_TOKENS - 1))
    assert ipc.hints == []
    _lifecycle(_HintIPC(fail=True), arrival_readahead=True).on_new_request(
        _request(BLOCK_TOKENS)
    )


def test_new_request_hint_starts_after_gpu_prefix_hit() -> None:
    """GPU-cached leading blocks are skipped, rounded down to DaseR blocks."""
    ipc = _HintIPC()
    lifecycle = _lifecycle(ipc, arrival_readahead=True)
    # Hashes 0-2 cached (6 tokens), hash 4 cached past a gap: only the
    # contiguous run counts, rounded down to one 4-token DaseR block.
    lifecycle.bind_gpu_block_pool(_CachedPool({0, 1, 2, 4}))
    lifecycle.on_new_request(_request(3 * BLOCK_TOKENS))
    assert ipc.hints == [(list(range(3 * BLOCK_TOKENS)), "m", 3 * BLOCK_TOKENS, 4)]


def test_new_request_fully_gpu_cached_sends_no_hint() -> None:
    """A prompt whose aligned prefix is all GPU-resident needs no readahead."""
    ipc = _HintIPC()
    lifecycle = _lifecycle(ipc, arrival_readahead=True)
    lifecycle.bind_gpu_block_pool(_CachedPool(set(range(8))))
    lifecycle.on_new_request(_request(2 * BLOCK_TOKENS + 1))
    assert ipc.hints == []
