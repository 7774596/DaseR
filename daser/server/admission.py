# SPDX-License-Identifier: Apache-2.0

"""Finish-time L2 admission for compressed-online stores (research switch).

Most agent KV blocks are written once and never reused, and whether a block
will be reused is not visible when it is stored but is visible when its
request finishes. With this policy a newly stored chunk is published to the
memory tier only ("held"). When its writing request finishes the connector
reports the output length: long outputs, or chunks that were looked up while
held, are written to L2; the rest are evicted, returning their packed arena
bytes. A held chunk that leaves L1 before the decision is written back by the
transfer layer, so no hit is lost to a late decision.

The class is pure bookkeeping. The IPC server applies the returned actions to
the transfer layer and the core.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

# Research switch: minimum output tokens for a finished request's chunks to be
# written to L2. Unset keeps write-through.
FINISH_ADMIT_MIN_OUTPUT_ENV = "DASER_FINISH_ADMIT_MIN_OUTPUT"

# Decisions that arrive before their chunk's store are kept for this many keys.
_MAX_EARLY_DECISIONS = 65536

Range = tuple[int, int]


@dataclass
class AdmissionActions:
    """Transfer and metadata actions produced by one admission event.

    Attributes:
        persist: held ranges to write to L2.
        evict: chunk keys to evict; the removal listener discards their ranges.
        admitted: chunks written to L2 because of a long output.
        rescued: chunks written to L2 only because they were hit while held.
    """

    persist: list[Range] = field(default_factory=list)
    evict: list[str] = field(default_factory=list)
    admitted: int = 0
    rescued: int = 0


class FinishAdmission:
    """Track held chunks until their writing request finishes.

    Args:
        min_output_tokens: output length at or above which a finished
            request's chunks are admitted to L2.

    Async/thread-safety:
        Not synchronized; call only from the server event loop.
    """

    def __init__(self, min_output_tokens: int) -> None:
        if min_output_tokens < 0:
            raise ValueError("min_output_tokens must be nonnegative")
        self.min_output_tokens = min_output_tokens
        self._held: dict[str, list[Range]] = {}
        self._hit: set[str] = set()
        self._early: OrderedDict[str, bool] = OrderedDict()

    @property
    def held_chunks(self) -> int:
        """Return the number of chunks waiting for a decision."""
        return len(self._held)

    def hold(self, chunk_key: str, ranges: list[Range]) -> AdmissionActions:
        """Record a chunk whose store was published to L1 only.

        Args:
            chunk_key: stored chunk.
            ranges: held ``(file_offset, nbytes)`` transfer keys.

        Returns:
            Actions to apply now when the decision already arrived.
        """
        admit = self._early.pop(chunk_key, None)
        self._held[chunk_key] = list(ranges)
        if admit is None:
            return AdmissionActions()
        return self._resolve(chunk_key, admit)

    def mark_hit(self, chunk_keys: list[str]) -> None:
        """Record that held chunks were matched by a lookup.

        Args:
            chunk_keys: chunk keys returned by a lookup.
        """
        self._hit.update(key for key in chunk_keys if key in self._held)

    def finish(self, chunk_keys: list[str], output_tokens: int) -> AdmissionActions:
        """Decide L2 admission for the chunks one finished request wrote.

        Args:
            chunk_keys: chunks allocated for the request's stores.
            output_tokens: generated tokens of the finished request.

        Returns:
            Actions for chunks whose store already completed. Decisions for
            chunks not stored yet are applied by the later :meth:`hold`.
        """
        admit = output_tokens >= self.min_output_tokens
        actions = AdmissionActions()
        for key in chunk_keys:
            if key in self._held:
                part = self._resolve(key, admit)
                actions.persist.extend(part.persist)
                actions.evict.extend(part.evict)
                actions.admitted += part.admitted
                actions.rescued += part.rescued
                continue
            self._early[key] = admit
            self._early.move_to_end(key)
            while len(self._early) > _MAX_EARLY_DECISIONS:
                self._early.popitem(last=False)
        return actions

    def forget(self, chunk_key: str) -> list[Range]:
        """Drop all state for a chunk that left the metadata store.

        Args:
            chunk_key: removed chunk.

        Returns:
            Held ranges the caller must discard from the transfer layer.
        """
        self._hit.discard(chunk_key)
        self._early.pop(chunk_key, None)
        return self._held.pop(chunk_key, [])

    def _resolve(self, chunk_key: str, admit: bool) -> AdmissionActions:
        """Turn one decision for a held chunk into actions."""
        hit = chunk_key in self._hit
        if admit or hit:
            self._hit.discard(chunk_key)
            return AdmissionActions(
                persist=self._held.pop(chunk_key),
                admitted=int(admit),
                rescued=int(not admit),
            )
        # Keep the ranges until the eviction's removal listener discards them.
        return AdmissionActions(evict=[chunk_key])
