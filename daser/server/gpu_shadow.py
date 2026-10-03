# SPDX-License-Identifier: Apache-2.0
"""Shadow model of which stored chunks the GPU prefix cache still holds.

vLLM reports no prefix-cache evictions to the connector. A worker that runs
one request at a time keeps each served prompt in GPU memory and evicts the
least recently served blocks first, the tail of a prompt before its head.
The server can replay that order from the lookup stream: the shadow keeps
an LRU of chunks weighted by token count and capped at the GPU's KV
capacity. A chunk the shadow holds has a faster copy than L1, so the memory
tier can give up its bytes first.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Iterable
from typing import Any


class GpuResidencyShadow:
    """Token-weighted LRU of the chunks the GPU prefix cache holds.

    Args:
        capacity_tokens: GPU KV capacity in tokens; must be positive.

    Async/thread-safety:
        Not synchronized; the IPC server calls it from its event loop.
    """

    def __init__(self, capacity_tokens: int) -> None:
        if capacity_tokens <= 0:
            raise ValueError("capacity_tokens must be positive")
        self._capacity = capacity_tokens
        self._tokens = 0
        self._chunks: OrderedDict[Hashable, tuple[int, Any]] = OrderedDict()

    @property
    def resident_tokens(self) -> int:
        """Return the tokens the shadow holds."""
        return self._tokens

    def __contains__(self, key: Hashable) -> bool:
        """Return whether the shadow holds ``key``."""
        return key in self._chunks

    def serve(
        self, chunks: Iterable[tuple[Hashable, int, Any]]
    ) -> tuple[list[Any], list[Any]]:
        """Record that the GPU computed or reused a prompt's chunks.

        Args:
            chunks: ``(key, tokens, payload)`` in prompt order. The head
                becomes the most recent, so the tail is evicted first.

        Returns:
            ``(resident, evicted)`` payloads: chunks that entered the GPU or
            were refreshed, head last, and chunks pushed out of capacity.
        """
        resident: list[Any] = []
        for key, tokens, payload in reversed(list(chunks)):
            previous = self._chunks.pop(key, None)
            if previous is not None:
                self._tokens -= previous[0]
            self._chunks[key] = (int(tokens), payload)
            self._tokens += int(tokens)
            resident.append(payload)
        evicted: list[Any] = []
        while self._tokens > self._capacity and self._chunks:
            _key, (tokens, payload) = self._chunks.popitem(last=False)
            self._tokens -= tokens
            evicted.append(payload)
        return resident, evicted

    def drop(self, keys: Iterable[Hashable]) -> list[Any]:
        """Forget chunks the GPU turned out not to hold.

        Args:
            keys: Chunk keys a lookup showed were not computed locally.

        Returns:
            Payloads of the keys the shadow held.
        """
        dropped: list[Any] = []
        for key in keys:
            entry = self._chunks.pop(key, None)
            if entry is not None:
                self._tokens -= entry[0]
                dropped.append(entry[1])
        return dropped
