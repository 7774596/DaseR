# SPDX-License-Identifier: Apache-2.0

"""Finish-time L2 admission for compressed-online stores (research switch).

Most agent KV blocks are written once and never reused, and whether a block
will be reused is not visible when it is stored but is visible when its
request finishes. With this policy a newly stored chunk is published to the
memory tier only ("held"). When its writing request finishes the connector
reports the output length: long outputs are written to L2. Short-output
chunks are denied L2 but stay resident in memory so a later turn can still
hit them; a hit on a denied chunk rescues it to L2. Denied chunks share a byte
budget and the oldest are evicted when it is exceeded, returning their packed
arena bytes without an SSD write. A held chunk that leaves L1 before either
event is written back by the transfer layer, so no hit is lost to a late
decision.

The class is pure bookkeeping. The IPC server applies the returned actions to
the transfer layer and the core.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

# Research switch: minimum output tokens for a finished request's chunks to be
# written to L2. Unset keeps write-through.
FINISH_ADMIT_MIN_OUTPUT_ENV = "DASER_FINISH_ADMIT_MIN_OUTPUT"

# Research switch: bytes of denied chunks kept resident in memory, as a
# fraction of L1. Zero evicts denied chunks at the decision.
FINISH_ADMIT_DENIED_L1_FRACTION_ENV = "DASER_FINISH_ADMIT_DENIED_L1_FRACTION"
DEFAULT_DENIED_L1_FRACTION = 0.5

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
        rescued: chunks written to L2 because they were hit while held or
            denied.
        denied: chunks refused L2 by a short output.
    """

    persist: list[Range] = field(default_factory=list)
    evict: list[str] = field(default_factory=list)
    admitted: int = 0
    rescued: int = 0
    denied: int = 0

    def extend(self, other: AdmissionActions) -> None:
        """Append another event's actions to this one.

        Args:
            other: actions to merge.
        """
        self.persist.extend(other.persist)
        self.evict.extend(other.evict)
        self.admitted += other.admitted
        self.rescued += other.rescued
        self.denied += other.denied


class FinishAdmission:
    """Track held chunks until their writing request finishes.

    Args:
        min_output_tokens: output length at or above which a finished
            request's chunks are admitted to L2.
        denied_budget_bytes: bytes of denied chunks kept resident in memory.

    Async/thread-safety:
        Not synchronized; call only from the server event loop.
    """

    def __init__(self, min_output_tokens: int, denied_budget_bytes: int = 0) -> None:
        if min_output_tokens < 0:
            raise ValueError("min_output_tokens must be nonnegative")
        if denied_budget_bytes < 0:
            raise ValueError("denied_budget_bytes must be nonnegative")
        self.min_output_tokens = min_output_tokens
        self.denied_budget_bytes = denied_budget_bytes
        self._held: dict[str, list[Range]] = {}
        self._hit: set[str] = set()
        self._denied: OrderedDict[str, int] = OrderedDict()
        self._denied_bytes = 0
        self._early: OrderedDict[str, bool] = OrderedDict()

    @property
    def held_chunks(self) -> int:
        """Return the number of chunks not yet written to L2."""
        return len(self._held)

    @property
    def denied_bytes(self) -> int:
        """Return the resident bytes of denied chunks."""
        return self._denied_bytes

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

    def mark_hit(self, chunk_keys: list[str]) -> AdmissionActions:
        """Record lookup hits; a hit on a denied chunk rescues it to L2.

        Args:
            chunk_keys: chunk keys returned by a lookup.

        Returns:
            Persist actions for rescued denied chunks.
        """
        actions = AdmissionActions()
        for key in chunk_keys:
            if key in self._denied:
                self._drop_denied(key)
                actions.persist.extend(self._held.pop(key))
                actions.rescued += 1
            elif key in self._held:
                self._hit.add(key)
        return actions

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
                if key not in self._denied:
                    actions.extend(self._resolve(key, admit))
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
        self._drop_denied(chunk_key)
        return self._held.pop(chunk_key, [])

    def _resolve(self, chunk_key: str, admit: bool) -> AdmissionActions:
        """Turn one decision for a held chunk into actions."""
        hit = chunk_key in self._hit
        self._hit.discard(chunk_key)
        if admit or hit:
            return AdmissionActions(
                persist=self._held.pop(chunk_key),
                admitted=int(admit),
                rescued=int(not admit),
            )
        nbytes = sum(size for _, size in self._held[chunk_key])
        self._denied[chunk_key] = nbytes
        self._denied_bytes += nbytes
        actions = AdmissionActions(denied=1)
        # Keep the ranges until the eviction's removal listener discards them.
        while self._denied_bytes > self.denied_budget_bytes and self._denied:
            oldest = next(iter(self._denied))
            self._drop_denied(oldest)
            actions.evict.append(oldest)
        return actions

    def _drop_denied(self, chunk_key: str) -> None:
        """Remove a chunk from the denied set and its byte budget."""
        nbytes = self._denied.pop(chunk_key, None)
        if nbytes is not None:
            self._denied_bytes -= nbytes
