# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import math
import os
from typing import Any, Optional

# First Party
from daser.config import EXCLUSIVE_CLEAN_ENV, EXCLUSIVE_TIERS_ENV
from daser.connector.helpers import TokenSequence
from daser.connector.helpers import token_count as count_tokens
from daser.logging import init_logger
from daser.metrics import REGISTRY, MetricsRegistry
from daser.position.base import PositionEncoder
from daser.retrieval.base import RetrievalIndex, RetrievalMatch
from daser.server.chunk_lifecycle import ChunkLifecycle
from daser.server.chunk_manager import ChunkManager, IndexSnapshot
from daser.server.doc_registry import DocEntry, DocRegistry
from daser.server.metadata_store import ChunkMeta
from daser.server.tiers import (
    L2,
    TIERS,
    ChunkUnavailable,
    Demotion,
    Lease,
    Promotion,
    StorePlacement,
    TierCopy,
    TierFull,
    TierIndex,
    TierLayout,
    TierStats,
    record_offsets,
)

logger = init_logger(__name__)

# Research switch: when set to "1", lookups that could have used a still-pending
# writer are recorded so commit can report how long that lookup was early.
# Off by default because it hashes every prompt's candidate keys.
PUBLISH_PROBE_ENV = "DASER_PUBLISH_PROBE"
# Writer claims happen at scheduling time, so publication latency spans the
# request's queueing and prefill (seconds) as well as the save window (ms).
_PUBLISH_BUCKETS = (
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    60.0,
)


@dataclass(frozen=True)
class ChunkInfo:
    """Public chunk metadata returned by cache lookup operations.

    Attributes:
        chunk_key: cache key for the token sequence.
        start_slot: first slot occupied in the store.
        num_slots: number of contiguous slots occupied.
        token_count: number of tokens covered by the chunk.
        pos_offset: position offset assigned by the position encoder.
        target_token_start: token offset in the current prompt where this
            chunk should be loaded.
        model_id: model identifier used for reuse isolation.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    token_count: int
    pos_offset: int
    model_id: str
    target_token_start: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a msgpack/JSON-safe representation.

        Returns:
            Plain dict containing all chunk fields.
        """
        return {
            "chunk_key": self.chunk_key,
            "start_slot": self.start_slot,
            "num_slots": self.num_slots,
            "token_count": self.token_count,
            "pos_offset": self.pos_offset,
            "model_id": self.model_id,
            "target_token_start": self.target_token_start,
        }


@dataclass(frozen=True)
class Allocation:
    """Slot allocation returned to the connector.

    Attributes:
        chunk_key: cache key for the allocated chunk.
        start_slot: first allocated slot.
        num_slots: number of allocated slots.
        pos_offset: position offset assigned by the position encoder.
        token_count: number of tokens covered by this allocation.
        skipped: True when the connector should not write KV: the chunk is
            already committed or owned by another writer.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    pos_offset: int
    token_count: int
    skipped: bool = False

    def to_dict(self, include_chunk_key: bool = True) -> dict[str, Any]:
        """Return a msgpack/JSON-safe representation.

        Args:
            include_chunk_key: include chunk_key in the output when True.

        Returns:
            Plain dict containing allocation fields.
        """
        payload: dict[str, Any] = {
            "start_slot": self.start_slot,
            "num_slots": self.num_slots,
            "pos_offset": self.pos_offset,
            "skipped": self.skipped,
        }
        if include_chunk_key:
            payload["chunk_key"] = self.chunk_key
            payload["token_count"] = self.token_count
        return payload


@dataclass(frozen=True)
class MatchAndAllocResult:
    """Result for combined lookup and conditional allocation.

    Attributes:
        chunks: matching chunks when lookup hits.
        alloc: allocation info when lookup misses and allocation occurs.
    """

    chunks: list[ChunkInfo]
    alloc: Optional[Allocation]

    def to_dict(self) -> dict[str, Any]:
        """Return a msgpack-safe response payload.

        Returns:
            Dict with chunks and optional alloc.
        """
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "alloc": self.alloc.to_dict() if self.alloc is not None else None,
        }


@dataclass(frozen=True)
class DocumentRegistration:
    """Result for registering a document.

    Attributes:
        chunk_count_cached: number of document chunks currently cached.
    """

    chunk_count_cached: int


@dataclass(frozen=True)
class DocumentSummary:
    """Summary returned by document listing.

    Attributes:
        doc_id: unique document identifier.
        title: user-facing title.
        token_count: number of original document tokens.
        chunk_count_total: number of chunk keys referenced by the document.
        chunk_count_cached: number of referenced chunks still cached.
        status: document cache status.
        created_at: unix timestamp when the document was registered.
    """

    doc_id: str
    title: str
    token_count: int
    chunk_count_total: int
    chunk_count_cached: int
    status: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation.

        Returns:
            Plain dict containing summary fields.
        """
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "token_count": self.token_count,
            "chunk_count_total": self.chunk_count_total,
            "chunk_count_cached": self.chunk_count_cached,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class DeleteDocumentResult:
    """Result for deleting a document.

    Attributes:
        chunks_evicted: number of chunks removed because no other doc used them.
    """

    chunks_evicted: int


@dataclass(frozen=True)
class StoreRequest:
    """One chunk a rank is about to store.

    Attributes:
        chunk_key: Allocated chunk.
        start_slot: First slot id of the allocation the writer was given.
        num_slots: Slot count of that allocation.
        record_sizes: Per-slot record bytes on this rank, in slot order
            (``local_slot_size`` for raw slots, the codec's stored length for
            compressed ones); positive 4 KiB multiples, one per slot.
        record_modes: Per-slot codec mode (``raw`` or ``compressed``) of a
            packed chunk, one per slot; empty for the raw storage format.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    record_sizes: tuple[int, ...]
    record_modes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChunkRecords:
    """Record layout of a stored chunk, identical on every rank.

    Attributes:
        record_sizes: Per-slot record bytes.
        record_modes: Per-slot codec mode of a packed chunk; empty for raw.
    """

    record_sizes: tuple[int, ...]
    record_modes: tuple[str, ...]


@dataclass(frozen=True)
class LoadSegment:
    """A run of one chunk's slots read from one tier copy.

    Attributes:
        chunk_key: Chunk owning the slots.
        slot_index: First slot of the run, relative to the chunk start.
        record_sizes: Record bytes of each slot in the run.
        lane_offset: Lane-relative byte offset of the run's first record.
        copy: Copy the run is read from; it carries one io pin until
            ``ServerCore.release_load``.
    """

    chunk_key: str
    slot_index: int
    record_sizes: tuple[int, ...]
    lane_offset: int
    copy: TierCopy

    @property
    def tier(self) -> str:
        """Return the tier the run is read from."""
        return self.copy.tier

    @property
    def nbytes(self) -> int:
        """Return the bytes of the run on one rank."""
        return sum(self.record_sizes)


class ServerCore:
    """Shared control-plane core for DaseR server APIs.

    The core owns all mutable server-side cache and document state: chunk
    ids and metadata (``ChunkManager``), commit/writer state
    (``ChunkLifecycle``), and tier residency (``TierIndex``). The HTTP
    server calls document-level methods directly, while the IPC server calls
    connector cache, store and load methods.

    A chunk is visible to lookup once every TP rank stored it into some tier
    copy and one tier copy is valid on every rank. Tier eviction that drops a
    chunk's last copy removes the chunk (``daser_cache_evicted_chunks_total``
    with ``reason="capacity"``).

    Args:
        chunk_manager: chunk id allocator and persistence coordinator; must
            not index any chunk yet (use ``load_index`` to restore).
        retrieval_index: cache lookup index.
        position_encoder: position-offset policy.
        tier_layout: L1/L2 tier geometry.
        block_tokens: tokens per vLLM block.
        metrics_registry: registry for cache and tier metrics.

    Raises:
        ValueError: if ``chunk_manager`` already indexes chunks.

    Async/thread-safety:
        Intended to run on the DaseR server asyncio event loop. Methods do
        not perform blocking I/O (except ``save_index``). Callers should
        serialize access through the single event loop.
    """

    def __init__(
        self,
        chunk_manager: ChunkManager,
        retrieval_index: RetrievalIndex,
        position_encoder: PositionEncoder,
        tier_layout: TierLayout,
        block_tokens: int = 16,
        metrics_registry: MetricsRegistry | None = None,
    ) -> None:
        if len(chunk_manager.store) or chunk_manager.next_slot:
            raise ValueError("ServerCore needs an empty chunk manager; use load_index")
        self._cm = chunk_manager
        self._ri = retrieval_index
        self._pe = position_encoder
        self._block_tokens = block_tokens
        self._metrics = metrics_registry or REGISTRY
        self._lifecycle = ChunkLifecycle()
        self._commit_requests = 0
        self._late_evicted_commits = 0
        self._lookup_requests = 0
        self._lookup_hits = 0
        self._publish_probe = os.environ.get(PUBLISH_PROBE_ENV) == "1"
        # Keys whose last tier copy was evicted; their retrieval entries are
        # dropped by ``_flush_lost`` right after the evicting operation.
        self._lost: list[str] = []
        # Codec modes of packed chunks with tier state (raw chunks have none).
        self._record_modes: dict[str, tuple[str, ...]] = {}
        clean = os.environ.get(EXCLUSIVE_CLEAN_ENV)
        self._tiers = TierIndex(
            tier_layout,
            blocked=self._has_pending_writer,
            on_lost=self._on_lost,
            metrics_registry=self._metrics,
            exclusive=os.environ.get(EXCLUSIVE_TIERS_ENV) == "1",
            clean_bytes=int(clean) if clean else None,
        )
        # Called after a tier change leaves a demotion deficit (exclusive).
        self._demotion_listener: Callable[[], None] | None = None
        self._tiers.publish_metrics()

    @property
    def chunk_manager(self) -> ChunkManager:
        """Return the owned ChunkManager."""
        return self._cm

    @property
    def tier_layout(self) -> TierLayout:
        """Return the L1/L2 tier geometry."""
        return self._tiers.layout

    @property
    def index_epoch(self) -> int:
        """Return a counter that advances whenever a chunk leaves the index.

        Async/thread-safety:
            Read on the owning server event loop.
        """
        return self._cm.store.removals

    async def lookup(
        self,
        tokens: TokenSequence,
        model_id: str,
        *,
        wait_for_pending: bool = False,
    ) -> list[ChunkInfo]:
        """Look up cached chunks for token IDs.

        Args:
            tokens: prompt token IDs.
            model_id: model identifier.
            wait_for_pending: Retry a miss or incomplete prefix while an
                allocated store writer is still publishing. This is intended
                for online-compressed serving, where the next multi-turn
                lookup can arrive before the asynchronous store commit.

        Returns:
            List of matching chunks, possibly empty.

        Async/thread-safety:
            Performs no blocking I/O and should run on the server event loop.
        """
        matches = await self._ri.lookup(tokens, model_id)
        candidate_keys: set[str] | None = None
        pending_keys = self._lifecycle.pending_write_keys
        if pending_keys and (wait_for_pending or self._publish_probe):
            # Candidate-key generation hashes the complete prompt. Keep its
            # result across retries because the token sequence is immutable;
            # only index visibility changes between attempts.
            candidate_keys = self._ri.candidate_keys(tokens, model_id)
        if (
            self._publish_probe
            and candidate_keys is not None
            and self._lookup_needs_pending_retry(tokens, matches, candidate_keys)
        ):
            self._lifecycle.note_pending_overlap(candidate_keys & pending_keys)
        if wait_for_pending and self._lookup_needs_pending_retry(
            tokens, matches, candidate_keys
        ):
            # Store transfer and retrieval-index publication run on this same
            # asyncio loop. Short cooperative sleeps let those tasks advance
            # without putting a synchronous wait on the request path. The
            # bounded backoff covers normal transfer jitter while preserving a
            # finite miss latency when a writer fails or is evicted.
            for delay_s in (0.002, 0.004, 0.008, 0.016, 0.032, 0.064):
                if not self._lifecycle.pending_write_keys:
                    break
                await asyncio.sleep(delay_s)
                matches = await self._ri.lookup(tokens, model_id)
                if not self._lookup_needs_pending_retry(
                    tokens, matches, candidate_keys
                ):
                    break
        return self._record_lookup(tokens, matches)

    async def lookup_unless_pending(
        self,
        tokens: TokenSequence,
        model_id: str,
        num_computed_tokens: int = 0,
    ) -> list[ChunkInfo] | None:
        """Look up cached chunks, deferring when a pending writer may extend them.

        Unlike ``lookup(wait_for_pending=True)`` this never sleeps: a caller
        on a synchronous path (the vLLM scheduler) gets an immediate answer
        and can retry on its own schedule instead of blocking for the
        writer's transfer and commit.

        Args:
            tokens: prompt token IDs.
            model_id: model identifier.
            num_computed_tokens: tokens the caller already holds locally (the
                vLLM GPU prefix cache). A writer whose blocks end within them
                cannot add external hits, so it does not defer the lookup.

        Returns:
            None when the contiguous matches stop before the block-aligned
            prompt length and an allocated writer holds the key of the first
            block past both the matches and ``num_computed_tokens``;
            the deferred attempt is not counted in lookup metrics. Otherwise
            the matching chunks, exactly as ``lookup`` returns them.

        Async/thread-safety:
            Performs no blocking I/O and should run on the server event loop.
        """
        matches = await self._ri.lookup(tokens, model_id)
        if self._lifecycle.pending_write_keys and self._lookup_needs_pending_retry(
            tokens,
            matches,
            self._ri.candidate_keys(tokens, model_id),
            num_computed_tokens=num_computed_tokens,
        ):
            return None
        return self._record_lookup(tokens, matches)

    async def record_external_prefix_cache(self, queries: int, hits: int) -> None:
        """Record vLLM-equivalent external prefix cache token counters.

        Args:
            queries: Number of prompt tokens vLLM queried through the KV
                connector external prefix cache path.
            hits: Number of queried tokens vLLM accepted as external prefix
                cache hits.

        Returns:
            None.

        Async/thread-safety:
            Performs no blocking I/O and should run on the server event loop.
        """
        queries = max(0, int(queries))
        hits = max(0, min(int(hits), queries))
        self._metrics.counter(
            "daser_external_prefix_cache_queries_total",
            "External prefix cache queries from DaseR KV connector, "
            "in terms of queried tokens.",
        ).inc(queries)
        self._metrics.counter(
            "daser_external_prefix_cache_hits_total",
            "External prefix cache hits from DaseR KV connector, "
            "in terms of cached tokens accepted by vLLM.",
        ).inc(hits)

    def queue_hint(self, request_id: str, tokens: TokenSequence) -> int:
        """Keep a gateway-queued request's prefix resident until it finishes.

        Hints are ordered by arrival (FIFO admission); see "排队提示" in
        ``docs/design/tier-residency.md``. Re-hinting an id replaces it.

        Args:
            request_id: Gateway request id, later passed to ``queue_release``.
            tokens: Prompt token IDs the request will send.

        Returns:
            Number of prefix-chain keys hinted (0 when the retrieval index has
            no fixed match order).

        Async/thread-safety:
            Pure CPU work on the server event loop: hashes the prompt once.
        """
        keys = self._ri.chain_keys(tokens)
        self._tiers.want(request_id, keys)
        return len(keys)

    def queue_release(self, request_id: str) -> None:
        """Drop a queue hint once its request finishes (or is abandoned).

        Args:
            request_id: Id given to ``queue_hint``; unknown ids are ignored.

        Async/thread-safety:
            Runs on the server event loop.
        """
        self._tiers.unwant(request_id)

    async def alloc_chunk(
        self, chunk_key: str, token_count: int, model_id: str, parent_key: str = ""
    ) -> Allocation:
        """Allocate slots for a new chunk or reuse a compatible allocation.

        Args:
            chunk_key: cache key for the token sequence.
            token_count: number of tokens in this chunk.
            model_id: model identifier.
            parent_key: previous chunk of the prefix chain, or empty.

        Returns:
            Allocation metadata for connector writes.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        allocations = await self.alloc_chunks(
            [
                {
                    "chunk_key": chunk_key,
                    "token_count": token_count,
                    "parent_key": parent_key,
                }
            ],
            model_id,
        )
        return allocations[0]

    async def alloc_chunks(
        self,
        chunks: list[dict[str, Any]],
        model_id: str,
    ) -> list[Allocation]:
        """Allocate slots for multiple chunks in one server event-loop turn.

        Args:
            chunks: chunk descriptors with ``chunk_key`` and ``token_count``,
                and optionally ``parent_key`` (previous chunk of the prefix
                chain; empty or missing for a root).
            model_id: model identifier.

        Returns:
            Allocation metadata in the same order as ``chunks``.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        allocations: list[Allocation] = []
        for chunk in chunks:
            chunk_key = str(chunk["chunk_key"])
            token_count = int(chunk["token_count"])
            num_slots = self._slots_for(token_count)
            should_write = not self._has_store_owner(chunk_key, token_count, model_id)
            meta = self._alloc_or_get_chunk(
                chunk_key=chunk_key,
                token_count=token_count,
                num_slots=num_slots,
                model_id=model_id,
                parent_key=str(chunk.get("parent_key", "")),
            )
            if should_write:
                self._lifecycle.mark_write_owner(chunk_key)
            allocations.append(
                self._allocation(
                    meta,
                    token_count=token_count,
                    num_slots=num_slots,
                    skipped=not should_write,
                )
            )
        return allocations

    async def match_and_alloc(
        self, tokens: list[int], chunk_key: str, model_id: str
    ) -> MatchAndAllocResult:
        """Run lookup and allocate a future store target on miss.

        Args:
            tokens: full prompt token IDs.
            chunk_key: key for the aligned prefix; empty disables allocation.
            model_id: model identifier.

        Returns:
            Lookup chunks on hit, or allocation on miss.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        chunks = await self.lookup(tokens, model_id)
        if chunks:
            return MatchAndAllocResult(chunks=chunks, alloc=None)
        if not chunk_key:
            return MatchAndAllocResult(chunks=[], alloc=None)
        aligned = (len(tokens) // self._block_tokens) * self._block_tokens
        if aligned == 0:
            return MatchAndAllocResult(chunks=[], alloc=None)
        num_slots = self._slots_for(aligned)
        should_write = not self._has_store_owner(chunk_key, aligned, model_id)
        meta = self._alloc_or_get_chunk(
            chunk_key=chunk_key,
            token_count=aligned,
            num_slots=num_slots,
            model_id=model_id,
        )
        if should_write:
            self._lifecycle.mark_write_owner(chunk_key)
        return MatchAndAllocResult(
            chunks=[],
            alloc=self._allocation(
                meta,
                token_count=aligned,
                num_slots=num_slots,
                skipped=not should_write,
            ),
        )

    async def place_store(
        self, requests: list[StoreRequest]
    ) -> list[StorePlacement | None]:
        """Place one rank's chunk stores in the tiers and pin the copies.

        Placement is per chunk and idempotent: every TP rank and every
        writer retry of one allocation gets the same copies (see
        ``TierIndex.place_store``). Evicts tier candidates as needed; when
        only in-flight IO blocks the space, waits for it and retries.

        Args:
            requests: Chunks to store, each from a writer allocation.

        Returns:
            One entry per request: the placement (every copy carries one io
            pin, released by ``store_staged`` / ``store_persisted`` /
            ``store_failed``), or None for an allocation that no longer
            exists (the write must be dropped).

        Raises:
            ValueError: if record sizes do not match the allocation or a
                chunk appears twice.
            TierFull: if a tier cannot fit a chunk once IO is idle.

        Async/thread-safety:
            Runs on the server event loop; may await ``io_released``.
        """
        while True:
            live = []
            for index, request in enumerate(requests):
                if not self.is_current_allocation(
                    request.chunk_key, request.start_slot, request.num_slots
                ):
                    continue
                if len(request.record_sizes) != request.num_slots:
                    raise ValueError(
                        f"chunk {request.chunk_key} has {request.num_slots} slots "
                        f"but {len(request.record_sizes)} record sizes"
                    )
                self._check_record_modes(request)
                live.append((index, request))
            try:
                placements = self._tiers.place_store(
                    [(r.chunk_key, tuple(r.record_sizes)) for _i, r in live]
                )
            except TierFull as exc:
                await self._after_tier_change()
                if not exc.io_blocked:
                    raise
                await self._tiers.io_released(exc.tier)
                continue
            for _index, request in live:
                if request.record_modes:
                    self._record_modes[request.chunk_key] = tuple(request.record_modes)
            await self._after_tier_change()
            result: list[StorePlacement | None] = [None] * len(requests)
            for (index, _request), placement in zip(live, placements, strict=True):
                result[index] = placement
            return result

    def release_placement(self, placement: StorePlacement) -> None:
        """Release a placement's io pins without reporting a store outcome.

        For a duplicate placement of the same chunk and rank (a chunk whose
        records arrived in several store requests): the request that reports
        the outcome uses ``store_staged`` / ``store_persisted`` /
        ``store_failed``; every other request releases its pins here once its
        IO finished.

        Args:
            placement: Placement from ``place_store``.

        Async/thread-safety:
            Runs on the server event loop.
        """
        for copy in (placement.l1, placement.l2):
            if copy is not None:
                self._tiers.unpin_io(copy)
        self._notify_demotions()

    async def store_staged(
        self, placement: StorePlacement, tp_rank: int, tp_size: int
    ) -> None:
        """Record that ``tp_rank``'s bytes reached the L1 copy.

        Without an L2 copy this ends the rank's store: the L1 pin is released
        and the rank counts as one commit request. With L2 the chunk may
        already publish (it is present once L1 is valid on every rank).

        Args:
            placement: Placement from ``place_store``.
            tp_rank: Rank whose device-to-host copy finished.
            tp_size: Server TP size.

        Raises:
            ValueError: if the rank is invalid or the placement has no L1
                copy.

        Async/thread-safety:
            Runs on the server event loop; may await the retrieval insert.
        """
        self._check_rank(tp_rank, tp_size)
        if placement.l1 is None:
            raise ValueError(f"store of {placement.chunk_key} has no L1 copy")
        valid = self._tiers.mark_valid(placement.l1, tp_rank)
        if placement.l2 is None:
            self._tiers.unpin_io(placement.l1)
            await self._finish_rank(placement.chunk_key, valid, tp_rank, tp_size)
            self._notify_demotions()
        elif valid:
            await self._maybe_publish(placement.chunk_key, tp_rank, tp_size)

    async def store_persisted(
        self, placement: StorePlacement, tp_rank: int, tp_size: int
    ) -> None:
        """Record that ``tp_rank``'s bytes reached the L2 copy; end its store.

        Releases the rank's pins of both copies and counts one commit
        request; the chunk publishes once every rank reported and a copy is
        valid on every rank.

        Args:
            placement: Placement from ``place_store`` with an L2 copy.
            tp_rank: Rank whose L2 write finished.
            tp_size: Server TP size.

        Raises:
            ValueError: if the rank is invalid or the placement has no L2
                copy.

        Async/thread-safety:
            Runs on the server event loop; may await the retrieval insert.
        """
        self._check_rank(tp_rank, tp_size)
        if placement.l2 is None:
            raise ValueError(f"store of {placement.chunk_key} has no L2 copy")
        valid = self._tiers.mark_valid(placement.l2, tp_rank)
        l1 = placement.l1
        valid = valid or (l1 is not None and l1.attached and l1.is_valid(tp_rank))
        self._tiers.unpin_io(placement.l2)
        if l1 is not None:
            self._tiers.unpin_io(l1)
        await self._finish_rank(placement.chunk_key, valid, tp_rank, tp_size)
        self._notify_demotions()

    async def store_failed(
        self, placement: StorePlacement, tp_rank: int, tp_size: int
    ) -> None:
        """Release a rank's store that failed before it ended.

        Copies not valid on every rank are dropped; an uncommitted chunk left
        without copies is removed (``reason="store_failed"``), so a retry
        allocates it again. Call instead of ``store_persisted`` (or, without
        L2, instead of ``store_staged``).

        Args:
            placement: Placement from ``place_store``.
            tp_rank: Rank whose store failed.
            tp_size: Server TP size.

        Raises:
            ValueError: if the rank is invalid.

        Async/thread-safety:
            Runs on the server event loop.
        """
        self._check_rank(tp_rank, tp_size)
        key = placement.chunk_key
        copies = [c for c in (placement.l1, placement.l2) if c is not None]
        live = any(copy.attached for copy in copies)
        remains = self._tiers.discard_incomplete(placement)
        for copy in copies:
            self._tiers.unpin_io(copy)
        if self._tiers.chunk(key) is None:
            self._record_modes.pop(key, None)
        if live and not remains and not self._lifecycle.is_committed(key):
            await self._remove_chunk(key, reason="store_failed")
        await self._after_tier_change()

    async def plan_load(self, start_slot: int, num_slots: int) -> list[LoadSegment]:
        """Resolve a committed slot range to tier reads and pin them.

        The range may start inside a chunk and span adjacent chunks. Each
        chunk is read from its L1 copy when valid, after an in-flight
        promotion, else from L2.

        Args:
            start_slot: First slot id.
            num_slots: Number of slots.

        Returns:
            Segments in slot order; release with ``release_load``.

        Raises:
            ValueError: if ``num_slots < 1``.
            ChunkUnavailable: if a slot belongs to no committed chunk or the
                chunk left the index meanwhile (no pins are kept).

        Async/thread-safety:
            Runs on the server event loop; may await promotions.
        """
        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        segments: list[LoadSegment] = []
        slot = start_slot
        end = start_slot + num_slots
        try:
            while slot < end:
                owner = self._cm.resolve_slot(slot)
                if owner is None or not self._lifecycle.is_committed(
                    owner[0].chunk_key
                ):
                    raise ChunkUnavailable(f"slot {slot} has no committed chunk")
                meta, index = owner
                copy = await self._tiers.read_plan(meta.chunk_key)
                chunk = self._tiers.chunk(meta.chunk_key)
                if self._cm.store.get(meta.chunk_key) is not meta or chunk is None:
                    self._tiers.unpin_io(copy)
                    raise ChunkUnavailable(f"chunk {meta.chunk_key} left the index")
                count = min(meta.num_slots - index, end - slot)
                sizes = chunk.record_sizes
                segments.append(
                    LoadSegment(
                        chunk_key=meta.chunk_key,
                        slot_index=index,
                        record_sizes=sizes[index : index + count],
                        lane_offset=copy.offset + record_offsets(sizes)[index],
                        copy=copy,
                    )
                )
                slot += count
        except BaseException:
            self.release_load(segments)
            raise
        return segments

    def resolve_slots(
        self, start_slot: int, num_slots: int
    ) -> list[tuple[str, int, int]] | None:
        """Return the committed chunks covering a slot range, without pins.

        A lookup result may merge adjacent chunks into one run; this splits
        such a range back into its chunks.

        Args:
            start_slot: First slot id.
            num_slots: Number of slots (positive).

        Returns:
            ``(chunk_key, first slot index in the chunk, slot count)`` per
            chunk in slot order, or None when a slot has no committed chunk.

        Async/thread-safety:
            Reads in-memory state on the server event loop; O(chunks).
        """
        parts: list[tuple[str, int, int]] = []
        slot = start_slot
        end = start_slot + num_slots
        while slot < end:
            owner = self._cm.resolve_slot(slot)
            if owner is None or not self._lifecycle.is_committed(owner[0].chunk_key):
                return None
            meta, index = owner
            count = min(meta.num_slots - index, end - slot)
            parts.append((meta.chunk_key, index, count))
            slot += count
        return parts

    def release_load(self, segments: list[LoadSegment]) -> None:
        """Release the io pins of segments from ``plan_load``.

        Args:
            segments: Segments whose reads finished or were abandoned.
        """
        for segment in segments:
            self._tiers.unpin_io(segment.copy)
        self._notify_demotions()

    def finish_promotion(self, promotion: Promotion, ok: bool) -> None:
        """Complete a promotion started by ``begin_promotions``.

        Args:
            promotion: The promotion.
            ok: Whether the L2 -> L1 read succeeded on every rank.
        """
        self._tiers.finish_promotion(promotion, ok)
        self._tiers.publish_metrics()
        self._notify_demotions()

    @property
    def exclusive_tiers(self) -> bool:
        """Return whether L1 and L2 hold disjoint chunks."""
        return self._tiers.exclusive

    def set_demotion_listener(self, listener: Callable[[], None] | None) -> None:
        """Register the callback run when a tier change leaves a deficit.

        With exclusive tiers the IPC server (which owns the transfer layer)
        starts demotions from it; see ``demotion_deficit``.

        Args:
            listener: Synchronous callback, or None to clear it.
        """
        self._demotion_listener = listener

    def demotion_deficit(self) -> int:
        """Return the L1 lane bytes to demote (see ``TierIndex``)."""
        return self._tiers.demotion_deficit()

    def l1_room(self) -> int:
        """Return the L1 lane bytes a promotion can take without a demotion.

        See ``TierIndex.l1_room``; an upper bound with exclusive tiers.
        """
        return self._tiers.l1_room()

    def add_l1_demand(self, lane_bytes: int) -> None:
        """Raise (negative: lower) the L1 room demotions must keep.

        Raising it starts demotions through the demotion listener.

        Args:
            lane_bytes: L1 lane bytes to add (see ``TierIndex.add_l1_demand``).
        """
        self._tiers.add_l1_demand(lane_bytes)
        self._notify_demotions()

    async def begin_demotions(self, max_bytes: int) -> list[Demotion]:
        """Admit the coldest dirty L1 chunks into L2 for L1 -> L2 writes.

        Args:
            max_bytes: L1 lane bytes to admit at most (one more chunk may
                cross it).

        Returns:
            Demotions; both copies are io pinned until ``finish_demotion``.

        Async/thread-safety:
            Runs on the server event loop; flushes chunks lost to evictions.
        """
        demotions = self._tiers.begin_demotions(max_bytes)
        await self._flush_lost()
        self._tiers.publish_metrics()
        return demotions

    def finish_demotion(self, demotion: Demotion, ok: bool) -> None:
        """Complete a demotion started by ``begin_demotions``.

        Args:
            demotion: The demotion.
            ok: Whether the L1 -> L2 write succeeded on every rank.
        """
        self._tiers.finish_demotion(demotion, ok)
        self._tiers.publish_metrics()

    def pin_lease(self, chunk_key: str) -> Lease | None:
        """Keep a chunk's L1 copy from eviction until ``unpin_lease``.

        Args:
            chunk_key: Chunk key.

        Returns:
            The lease, or None when the chunk has no tier copy.
        """
        return self._tiers.pin_lease(chunk_key)

    def unpin_lease(self, lease: Lease) -> None:
        """Release a lease from ``pin_lease``.

        Args:
            lease: The lease; releasing one of a removed chunk is a no-op.
        """
        self._tiers.unpin_lease(lease)
        self._notify_demotions()

    async def begin_promotions(self, chunk_keys: list[str]) -> list[Promotion | None]:
        """Admit committed L2-only chunks into L1 for L2 -> L1 reads.

        One metrics and lost-chunk flush covers the batch.

        Args:
            chunk_keys: Chunk keys, in admission order.

        Returns:
            One entry per key: the promotion (both copies io pinned until
            ``finish_promotion``), or None when the chunk is not committed,
            already has an L1 copy, or L1 cannot admit it.

        Async/thread-safety:
            Runs on the server event loop; loads of a promoted chunk wait
            for its promotion.
        """
        promotions = [
            self._tiers.begin_promotion(key)
            if self._lifecycle.is_committed(key)
            else None
            for key in chunk_keys
        ]
        await self._after_tier_change()
        return promotions

    def chunk_records(self, chunk_key: str) -> ChunkRecords | None:
        """Return the record layout of a chunk with tier state.

        Args:
            chunk_key: Chunk key.

        Returns:
            Record sizes and (packed) modes, or None when the chunk has no
            tier copy.

        Async/thread-safety:
            Reads in-memory state on the server event loop.
        """
        chunk = self._tiers.chunk(chunk_key)
        if chunk is None:
            return None
        return ChunkRecords(
            record_sizes=chunk.record_sizes,
            record_modes=self._record_modes.get(chunk_key, ()),
        )

    def promotion_pending(self, chunk_key: str) -> bool:
        """Return whether a load of the chunk would wait for a promotion.

        Args:
            chunk_key: Chunk key.

        Returns:
            True while an L2 -> L1 read of the chunk is in flight.

        Async/thread-safety:
            Reads in-memory state on the server event loop; takes no pin, so
            the answer is a hint that may change at the next await.
        """
        chunk = self._tiers.chunk(chunk_key)
        return (
            chunk is not None
            and chunk.l1 is not None
            and (chunk.l1.promotion is not None)
        )

    def full_tiers(self, chunk_key: str) -> tuple[str, ...]:
        """Return the tiers holding a copy of the chunk valid on every rank.

        Args:
            chunk_key: Chunk key.

        Returns:
            Subset of ``(L1, L2)`` in tier order; empty when none.

        Async/thread-safety:
            Reads in-memory state on the server event loop.
        """
        chunk = self._tiers.chunk(chunk_key)
        if chunk is None:
            return ()
        return tuple(tier for tier in TIERS if self._tiers.is_full(chunk.copy(tier)))

    def tier_stats(self) -> dict[str, TierStats]:
        """Return the state of every enabled tier, keyed by tier name."""
        return {
            tier: self._tiers.stats(tier) for tier in TIERS if self._tiers.enabled(tier)
        }

    def is_chunk_committed(self, chunk_key: str) -> bool:
        """Return whether a chunk key has been committed.

        Args:
            chunk_key: cache key to check.

        Returns:
            True after the last rank's store published the chunk and before it is
            evicted or removed.

        Async/thread-safety:
            Reads in-memory state on the server event loop. It performs no
            blocking I/O.
        """
        return self._lifecycle.is_committed(chunk_key)

    def is_chunk_reusable(
        self,
        chunk_key: str,
        token_count: int,
        model_id: str,
    ) -> bool:
        """Return whether an identical committed chunk can be reused.

        Args:
            chunk_key: cache key to check.
            token_count: number of tokens expected for this chunk.
            model_id: model identifier expected for reuse isolation.

        Returns:
            True when a committed metadata entry exists for the same key,
            token count, slot count, and model. False for misses, in-flight
            uncommitted allocations, and incompatible metadata.

        Async/thread-safety:
            Reads in-memory state on the server event loop. It performs no
            blocking I/O.
        """
        return self._meta_matches(
            chunk_key, token_count, model_id, self._lifecycle.committed
        )

    async def wait_for_committed_chunks(
        self,
        chunk_keys: list[str],
        timeout_s: float,
    ) -> None:
        """Wait until all chunk keys have been committed.

        Args:
            chunk_keys: cache keys to wait for.
            timeout_s: maximum wait time in seconds.

        Raises:
            TimeoutError: if any chunk key is still uncommitted at timeout.

        Async/thread-safety:
            Must run on the server event loop. Waiters are completed by
            store reports on the same event loop.
        """
        await self._lifecycle.wait_for_committed(chunk_keys, timeout_s)

    async def wait_for_pending_chunks(self, timeout_s: float) -> None:
        """Wait for currently allocated store writers to commit.

        Args:
            timeout_s: maximum seconds to wait.

        Raises:
            TimeoutError: if any pending chunk remains uncommitted at timeout.

        Async/thread-safety:
            Snapshots lifecycle state and waits without blocking the server
            event loop. Writers registered after the snapshot are not included.
        """
        pending = list(self._lifecycle.write_owners - self._lifecycle.committed)
        await self._lifecycle.wait_for_committed(pending, timeout_s)

    async def commit_stats(self) -> dict[str, int]:
        """Return connector commit counters for benchmark synchronization.

        Returns:
            Dict containing total processed commit requests and the subset that
            arrived after the chunk had already been evicted.

        Async/thread-safety:
            Reads in-memory counters on the server event loop.
        """
        return {
            "commit_requests": self._commit_requests,
            "late_evicted_commits": self._late_evicted_commits,
            "lookup_requests": self._lookup_requests,
            "lookup_hits": self._lookup_hits,
        }

    async def live_allocations(self, allocations: list[dict[str, Any]]) -> list[str]:
        """Return chunk keys that still own their allocated slot ranges.

        Args:
            allocations: Dicts with ``chunk_key``, ``start_slot``, and
                ``num_slots`` fields.

        Returns:
            Chunk keys whose current metadata still matches the supplied slot
            allocation.

        Async/thread-safety:
            Reads in-memory metadata on the server event loop.
        """
        live: list[str] = []
        for alloc in allocations:
            chunk_key = str(alloc.get("chunk_key", ""))
            if not chunk_key:
                continue
            if self.is_current_allocation(
                chunk_key=chunk_key,
                start_slot=int(alloc.get("start_slot", -1)),
                num_slots=int(alloc.get("num_slots", 0)),
            ):
                live.append(chunk_key)
        return live

    def is_current_allocation(
        self,
        chunk_key: str,
        start_slot: int,
        num_slots: int,
    ) -> bool:
        """Return whether a delayed write still targets the live chunk.

        Args:
            chunk_key: chunk key associated with the write span.
            start_slot: first slot the connector was told to write.
            num_slots: number of slots allocated for the chunk.

        Returns:
            True when the chunk still exists and still owns the same slot range.

        Async/thread-safety:
            Reads in-memory metadata on the server event loop.
        """
        meta = self._cm.store.get(chunk_key)
        return (
            meta is not None
            and meta.start_slot == start_slot
            and meta.num_slots == num_slots
        )

    async def release_chunk_writer(
        self,
        chunk_key: str,
        start_slot: int,
        num_slots: int,
    ) -> bool:
        """Release an uncommitted writer claim for a canceled store.

        A chunk without a copy valid on every rank is removed (nothing could
        ever publish it); otherwise only the claim is dropped.

        Args:
            chunk_key: chunk key associated with the canceled store.
            start_slot: first slot the connector was told to write.
            num_slots: number of slots allocated for the chunk.

        Returns:
            True when an uncommitted writer claim was released, False when the
            chunk was already committed or no longer matches the allocation.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        if self._lifecycle.is_committed(chunk_key):
            return False
        if not self.is_current_allocation(chunk_key, start_slot, num_slots):
            return False
        if self._tiers.is_present(chunk_key):
            self._lifecycle.discard_owner(chunk_key)
            self._tiers.refresh(chunk_key)
        else:
            await self._remove_chunk(chunk_key, reason="writer_released")
        return True

    async def evict_chunk(self, chunk_key: str) -> None:
        """Evict a chunk from retrieval, metadata and tier state.

        Args:
            chunk_key: cache key for the chunk.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop. In-flight
            IO keeps the chunk's extents until it releases them.
        """
        await self._remove_chunk(chunk_key, reason="explicit")

    def save_index(self, path: str, cache_reuse_mode: str | None = None) -> None:
        """Persist the committed chunks that have an L2 copy.

        L1-only and uncommitted chunks are not saved; documents keep their
        references and see those chunks as evicted after a restore.

        Args:
            path: absolute file path to write.
            cache_reuse_mode: cache reuse mode that produced the chunks.

        Raises:
            ValueError: if L2 is disabled.

        Async/thread-safety:
            Blocking file IO with fsync; call at shutdown after transfers
            drained.
        """
        if not self._tiers.enabled(L2):
            raise ValueError("cannot persist an index without an L2 tier")
        committed = [
            meta.chunk_key
            for meta in self._cm.store.iter_chunks()
            if self._lifecycle.is_committed(meta.chunk_key)
        ]
        placements = self._tiers.snapshot_l2(committed)
        metas = []
        for placement in placements:
            meta = self._cm.store.get(placement.chunk_key)
            assert meta is not None
            metas.append(meta)
        registry = self._cm.doc_registry
        layout = self._tiers.layout
        self._cm.save(
            path,
            IndexSnapshot(
                cache_reuse_mode=cache_reuse_mode,
                next_slot=self._cm.next_slot,
                tier_grain=layout.grain,
                l2_lane_bytes=layout.l2_lane_bytes,
                tensor_parallel_size=layout.tensor_parallel_size,
                metas=metas,
                l2=placements,
                docs=registry.to_dict() if registry is not None else {},
            ),
        )

    async def load_index(
        self, path: str, expected_cache_reuse_mode: str | None = None
    ) -> None:
        """Restore chunks saved by ``save_index`` as committed L2 copies.

        Args:
            path: absolute file path to read.
            expected_cache_reuse_mode: When set, an index from another mode is
                rejected.

        Raises:
            ValueError: if the index format, reuse mode, tier grain, L2 lane
                size or TP size differ, the core already holds chunks, or the
                saved chunks are inconsistent. Nothing changes on error.
            OSError: if the file cannot be read.

        Async/thread-safety:
            Blocking file read; call at startup before serving requests.
        """
        snapshot = ChunkManager.read_snapshot(path, expected_cache_reuse_mode)
        layout = self._tiers.layout
        saved = (
            snapshot.tier_grain,
            snapshot.l2_lane_bytes,
            snapshot.tensor_parallel_size,
        )
        current = (layout.grain, layout.l2_lane_bytes, layout.tensor_parallel_size)
        if saved != current:
            raise ValueError(
                f"tier layout mismatch: index has (grain, l2_lane, tp)={saved}, "
                f"server has {current}"
            )
        sizes = {p.chunk_key: p.record_sizes for p in snapshot.l2}
        if len(sizes) != len(snapshot.l2) or set(sizes) != {
            meta.chunk_key for meta in snapshot.metas
        }:
            raise ValueError("index chunks and L2 placements differ")
        for meta in snapshot.metas:
            if len(sizes[meta.chunk_key]) != meta.num_slots:
                raise ValueError(f"record count mismatch for chunk: {meta.chunk_key}")
            if meta.parent_key == meta.chunk_key:
                raise ValueError(f"chunk is its own parent: {meta.chunk_key}")
        self._tiers.validate_l2_placements(snapshot.l2)
        self._cm.restore(snapshot)
        for meta in snapshot.metas:
            self._tiers.add_chunk(meta.chunk_key, meta.parent_key)
        self._tiers.restore_l2(snapshot.l2)
        for meta in snapshot.metas:
            await self._ri.insert(meta)
            self._lifecycle.mark_committed(meta.chunk_key)
        self._tiers.publish_metrics()
        logger.info("[CORE] restored %d chunks from %s", len(snapshot.metas), path)

    async def register_document(
        self,
        doc_id: str,
        title: str,
        chunk_keys: list[str],
        token_count: int,
        tokens: Optional[list[int]] = None,
        text: Optional[str] = None,
    ) -> DocumentRegistration:
        """Register a document and attach it to cached chunks.

        Args:
            doc_id: unique document identifier.
            title: user-facing title.
            chunk_keys: chunk keys in document order.
            token_count: number of original document tokens.
            tokens: optional full token sequence for prompt reconstruction.
            text: optional original document text for UI inspection.

        Returns:
            Document registration result.

        Raises:
            RuntimeError: if no DocRegistry is attached.
            ValueError: if doc_id already exists.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        registry = self._require_doc_registry()
        if registry.get(doc_id) is not None:
            raise ValueError(f"doc_id already exists: {doc_id}")

        cached_mask: list[bool] = []
        for key in chunk_keys:
            meta = self._cm.store.get(key)
            if meta is None:
                cached_mask.append(False)
                continue
            if doc_id not in meta.doc_ids:
                meta.doc_ids.append(doc_id)
            cached_mask.append(True)

        entry = DocEntry(
            doc_id=doc_id,
            title=title,
            token_count=token_count,
            chunk_keys=list(chunk_keys),
            cached_mask=cached_mask,
            status="ready" if any(cached_mask) else "evicted",
            tokens=list(tokens) if tokens is not None else None,
            text=text,
        )
        registry.insert(entry)
        cached = sum(cached_mask)
        logger.info(
            "[CORE] register_document doc_id=%s chunks=%d cached=%d",
            doc_id,
            len(chunk_keys),
            cached,
        )
        return DocumentRegistration(chunk_count_cached=cached)

    async def list_documents(self) -> list[DocumentSummary]:
        """Return summaries for all registered documents.

        Returns:
            Document summaries in registry order.

        Async/thread-safety:
            Reads in-memory state on the server event loop.
        """
        registry = self._cm.doc_registry
        if registry is None:
            return []
        summaries: list[DocumentSummary] = []
        for entry in registry.all_entries():
            summaries.append(
                DocumentSummary(
                    doc_id=entry.doc_id,
                    title=entry.title,
                    token_count=entry.token_count,
                    chunk_count_total=len(entry.chunk_keys),
                    chunk_count_cached=sum(entry.cached_mask),
                    status=entry.status,
                    created_at=entry.created_at,
                )
            )
        return summaries

    async def get_document(self, doc_id: str) -> Optional[DocEntry]:
        """Return a registered document.

        Args:
            doc_id: document identifier.

        Returns:
            DocEntry when found, otherwise None.

        Async/thread-safety:
            Reads in-memory state on the server event loop.
        """
        registry = self._cm.doc_registry
        if registry is None:
            return None
        return registry.get(doc_id)

    async def delete_document(self, doc_id: str) -> DeleteDocumentResult:
        """Delete a document and evict chunks it solely references.

        Args:
            doc_id: document identifier.

        Returns:
            Deletion result.

        Raises:
            RuntimeError: if no DocRegistry is attached.
            ValueError: if doc_id is unknown.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        registry = self._require_doc_registry()
        entry = registry.remove(doc_id)
        if entry is None:
            raise ValueError(f"doc_id not found: {doc_id}")

        chunks_evicted = 0
        for key in entry.chunk_keys:
            if self._detach_doc_from_chunk(doc_id, key):
                await self._ri.remove(key)
                chunks_evicted += 1
        if chunks_evicted:
            self._tiers.publish_metrics()
        logger.info(
            "[CORE] delete_document doc_id=%s chunks_evicted=%d",
            doc_id,
            chunks_evicted,
        )
        return DeleteDocumentResult(chunks_evicted=chunks_evicted)

    def _require_doc_registry(self) -> DocRegistry:
        """Return the attached DocRegistry or raise a public operation error."""
        registry = self._cm.doc_registry
        if registry is None:
            raise RuntimeError("doc registry not enabled")
        return registry

    def _mark_chunk_evicted_in_docs(self, meta: ChunkMeta) -> None:
        """Mark a chunk as evicted in every referencing document.

        Args:
            meta: chunk metadata being evicted.
        """
        registry = self._cm.doc_registry
        if registry is None:
            return
        for doc_id in list(meta.doc_ids):
            registry.mark_chunk_evicted(doc_id, meta.chunk_key)

    def _detach_doc_from_chunk(self, doc_id: str, chunk_key: str) -> bool:
        """Detach a document reference and remove unreferenced chunks.

        Args:
            doc_id: document being deleted.
            chunk_key: referenced chunk key.

        Returns:
            True when the chunk became unreferenced and was removed.
        """
        meta = self._cm.store.get(chunk_key)
        if meta is None:
            self._lifecycle.discard(chunk_key)
            return False
        if doc_id in meta.doc_ids:
            meta.doc_ids.remove(doc_id)
        if meta.doc_ids:
            return False
        self._cm.remove(chunk_key)
        self._tiers.remove(chunk_key)
        self._record_modes.pop(chunk_key, None)
        self._lifecycle.discard(chunk_key)
        return True

    async def _remove_chunk(self, chunk_key: str, *, reason: str) -> None:
        """Remove one chunk from retrieval, metadata, tier and lifecycle state.

        Args:
            chunk_key: cache key for the chunk.
            reason: ``daser_cache_evicted_chunks_total`` label value.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        await self._ri.remove(chunk_key)
        meta = self._cm.store.get(chunk_key)
        if meta is not None:
            self._mark_chunk_evicted_in_docs(meta)
            self._cm.remove(chunk_key)
            self._tiers.remove(chunk_key)
            self._record_modes.pop(chunk_key, None)
        self._lifecycle.discard(chunk_key)
        self._count_eviction(reason)
        self._tiers.publish_metrics()
        logger.debug("[CORE] remove chunk key=%s reason=%s", chunk_key[:8], reason)

    def _on_lost(self, chunk_key: str) -> None:
        """Drop a chunk whose last tier copy was evicted (``TierIndex`` hook).

        Runs synchronously inside the evicting tier call; the retrieval entry
        is removed by ``_flush_lost`` before the core yields.
        """
        meta = self._cm.store.get(chunk_key)
        if meta is not None:
            self._mark_chunk_evicted_in_docs(meta)
            self._cm.remove(chunk_key)
        self._record_modes.pop(chunk_key, None)
        self._lifecycle.discard(chunk_key)
        self._count_eviction("capacity")
        self._lost.append(chunk_key)

    async def _flush_lost(self) -> None:
        """Remove the retrieval entries of chunks lost to tier eviction."""
        while self._lost:
            chunk_key = self._lost.pop()
            # A chunk allocated again under the key owns the entry now.
            if self._cm.store.get(chunk_key) is None:
                await self._ri.remove(chunk_key)

    async def _after_tier_change(self) -> None:
        """Finish an operation that may have evicted tier copies."""
        await self._flush_lost()
        self._tiers.publish_metrics()
        self._notify_demotions()

    def _notify_demotions(self) -> None:
        """Run the demotion listener while L1 lacks free or clean bytes.

        Called after tier changes and whenever io pins or leases drop, since
        a dirty L1 copy becomes demotable only once unpinned.
        """
        if self._demotion_listener is not None and self._tiers.demotion_deficit():
            self._demotion_listener()

    def _count_eviction(self, reason: str) -> None:
        """Count one chunk that left the index."""
        self._metrics.counter(
            "daser_cache_evicted_chunks_total",
            "Chunks evicted from cache metadata.",
        ).inc(labels={"reason": reason})

    async def _finish_rank(
        self, chunk_key: str, valid: bool, tp_rank: int, tp_size: int
    ) -> None:
        """Count a rank's ended store and publish the chunk if it can.

        Args:
            chunk_key: Stored chunk.
            valid: Whether the rank's bytes landed in a copy still attached
                to the chunk (False for a write to a removed allocation).
            tp_rank: Rank whose store ended.
            tp_size: Server TP size.
        """
        self._commit_requests += 1
        if not valid:
            self._late_evicted_commits += 1
            self._metrics.counter(
                "daser_cache_late_evicted_commits_total",
                "Commit requests ignored because the allocation was evicted.",
            ).inc()
            logger.debug("[CORE] ignore late store of evicted key=%s", chunk_key[:8])
            return
        await self._maybe_publish(chunk_key, tp_rank, tp_size)

    async def _maybe_publish(self, chunk_key: str, tp_rank: int, tp_size: int) -> None:
        """Record a rank's valid store and publish once the chunk is present.

        The TP quorum (``ChunkLifecycle.record_commit_shard``) counts ranks
        whose bytes reached a tier copy; the chunk publishes when every rank
        arrived and some copy is valid on every rank. A quorum reached before
        that is re-checked on the next rank event.
        """
        if not self._lifecycle.record_commit_shard(chunk_key, tp_rank, tp_size):
            return
        meta = self._cm.store.get(chunk_key)
        if meta is None or not self._tiers.is_present(chunk_key):
            self._lifecycle.abort_publish(chunk_key)
            return
        try:
            await self._ri.insert(meta)
        except BaseException:
            self._lifecycle.abort_publish(chunk_key)
            raise
        publish_s, overlap_wait_s = self._lifecycle.mark_committed(chunk_key)
        self._tiers.touch([chunk_key])
        self._metrics.counter(
            "daser_cache_committed_chunks_total",
            "Chunks committed and published to the retrieval index.",
        ).inc()
        if publish_s is not None:
            self._metrics.histogram(
                "daser_cache_publish_seconds",
                "Seconds from store-writer claim to retrieval publication.",
                buckets=_PUBLISH_BUCKETS,
            ).observe(publish_s)
        if overlap_wait_s is not None:
            self._metrics.histogram(
                "daser_cache_pending_overlap_seconds",
                "Seconds from the first lookup that needed a pending chunk to "
                "its publication.",
                buckets=_PUBLISH_BUCKETS,
            ).observe(overlap_wait_s)
        logger.debug(
            "[CORE] commit key=%s tp_rank=%d tp_size=%d",
            chunk_key[:8],
            tp_rank,
            tp_size,
        )

    def _has_pending_writer(self, chunk_key: str) -> bool:
        """Return whether a chunk has an uncommitted store writer."""
        return chunk_key in self._lifecycle.write_owners and not (
            self._lifecycle.is_committed(chunk_key)
        )

    def _check_record_modes(self, request: StoreRequest) -> None:
        """Reject packed record modes that are malformed or changed.

        Raises:
            ValueError: if the modes do not match the slot count, name an
                unknown mode, or differ from the chunk's stored modes.
        """
        modes = tuple(request.record_modes)
        if modes and (
            len(modes) != request.num_slots
            or any(mode not in ("raw", "compressed") for mode in modes)
        ):
            raise ValueError(f"invalid record modes for chunk {request.chunk_key}")
        if self._tiers.chunk(request.chunk_key) is not None and modes != (
            self._record_modes.get(request.chunk_key, ())
        ):
            raise ValueError(f"record modes changed for chunk: {request.chunk_key}")

    def _check_rank(self, tp_rank: int, tp_size: int) -> None:
        """Reject a rank outside the server's TP geometry."""
        expected = self._tiers.layout.tensor_parallel_size
        if tp_size != expected or not 0 <= tp_rank < tp_size:
            raise ValueError(
                f"invalid TP rank {tp_rank} for size {tp_size} (server tp={expected})"
            )

    def _record_lookup(
        self, tokens: TokenSequence, matches: list[Any]
    ) -> list[ChunkInfo]:
        """Count a resolved lookup and convert its matches to chunk info.

        Args:
            tokens: Prompt token IDs used for the lookup.
            matches: Retrieval matches the lookup resolved to.

        Returns:
            Chunk info for ``matches``.
        """
        self._lookup_requests += 1
        if matches:
            self._lookup_hits += 1
        chunks = [self._chunk_info(match) for match in matches]
        # A merged prefix hit is keyed by its last block; touch the chain.
        for chunk in chunks:
            self._tiers.touch_path(chunk.chunk_key)
        self._metrics.counter(
            "daser_cache_lookup_total",
            "Cache lookup requests by result.",
        ).inc(labels={"result": "hit" if chunks else "miss"})
        self._metrics.counter(
            "daser_cache_requested_tokens_total",
            "Prompt tokens checked for cache reuse.",
        ).inc(count_tokens(tokens))
        self._metrics.counter(
            "daser_cache_matched_tokens_total",
            "Prompt tokens matched by cache lookup.",
        ).inc(sum(chunk.token_count for chunk in chunks))
        if chunks:
            self._metrics.histogram(
                "daser_cache_prefix_reuse_tokens",
                "Tokens reused per cache hit.",
                buckets=(16, 64, 128, 256, 512, 1024, 2048, 4096),
            ).observe(sum(chunk.token_count for chunk in chunks))
        return chunks

    def _lookup_needs_pending_retry(
        self,
        tokens: TokenSequence,
        matches: list[Any],
        candidate_keys: set[str] | None = None,
        *,
        num_computed_tokens: int = 0,
    ) -> bool:
        """Return whether an incomplete lookup may be completed by a writer.

        Args:
            tokens: Prompt token IDs used for the lookup.
            matches: Retrieval matches returned for the current lookup.
            candidate_keys: Precomputed candidates for ``tokens``. If omitted,
                candidates are generated for this check.
            num_computed_tokens: Tokens the caller already holds locally. A
                writer of a block inside them cannot add external hits.

        Returns:
            True when the contiguous matches stop before the block-aligned
            prompt length and a pending writer holds a key that could extend
            them past ``num_computed_tokens``.
        """
        pending_keys = self._lifecycle.pending_write_keys
        if not pending_keys:
            return False
        if candidate_keys is None:
            candidate_keys = self._ri.candidate_keys(tokens, "")
        if not candidate_keys.intersection(pending_keys):
            return False
        aligned = (count_tokens(tokens) // self._block_tokens) * self._block_tokens
        if aligned <= 0:
            return False
        covered = 0
        for match in matches:
            target_start = int(getattr(match, "target_token_start", 0))
            token_count = int(getattr(match.meta, "token_count", 0))
            if target_start > covered:
                break
            covered = max(covered, target_start + token_count)
            if covered >= aligned:
                return False
        # A writer of a later block cannot fill a gap left by an evicted one,
        # and one inside the caller's local prefix adds no external hit.
        local = (max(0, int(num_computed_tokens)) // self._block_tokens) * (
            self._block_tokens
        )
        blockers = self._ri.continuation_keys(tokens, max(covered, local), "")
        if blockers is None:
            blockers = candidate_keys
        return bool(blockers.intersection(pending_keys))

    def _has_store_owner(
        self,
        chunk_key: str,
        token_count: int,
        model_id: str,
    ) -> bool:
        """Return whether a compatible writer already owns a chunk key.

        Args:
            chunk_key: cache key to inspect.
            token_count: number of tokens expected for this chunk.
            model_id: model identifier expected for reuse isolation.

        Returns:
            True for committed chunks and for uncommitted allocations whose
            first writer has already claimed the store target.
        """
        return self._meta_matches(
            chunk_key, token_count, model_id, self._lifecycle.write_owners
        )

    def _slots_for(self, token_count: int) -> int:
        """Return the number of KV slots needed for ``token_count`` tokens."""
        return math.ceil(token_count / self._block_tokens)

    def _meta_matches(
        self,
        chunk_key: str,
        token_count: int,
        model_id: str,
        membership: set[str],
    ) -> bool:
        """Return whether stored metadata matches and is in ``membership``.

        Args:
            chunk_key: cache key to inspect.
            token_count: expected token count.
            model_id: expected model identifier.
            membership: set the key must belong to (committed or write-owner).

        Returns:
            True when a metadata entry exists for the key, is in ``membership``,
            and matches the token count, slot count, and model.
        """
        meta = self._cm.store.get(chunk_key)
        if meta is None or chunk_key not in membership:
            return False
        return (
            meta.token_count == token_count
            and meta.num_slots == self._slots_for(token_count)
            and meta.model_id == model_id
        )

    def _alloc_or_get_chunk(
        self,
        chunk_key: str,
        token_count: int,
        num_slots: int,
        model_id: str,
        parent_key: str = "",
    ) -> ChunkMeta:
        """Allocate a chunk or reuse a compatible in-flight allocation.

        Allocation takes fresh slot ids only; tier space is reserved when the
        chunk is stored (``place_store``).

        Args:
            chunk_key: cache key for the token sequence.
            token_count: number of tokens covered by the chunk.
            num_slots: number of slots required.
            model_id: model identifier.
            parent_key: previous chunk of the prefix chain, or empty.

        Returns:
            Existing or newly allocated chunk metadata.

        Raises:
            ValueError: if an existing chunk has incompatible metadata or the
                chunk names itself as parent.
        """
        existing = self._cm.store.get(chunk_key)
        if existing is not None:
            if (
                existing.token_count != token_count
                or existing.num_slots != num_slots
                or existing.model_id != model_id
            ):
                raise ValueError(
                    f"chunk_key already exists with incompatible metadata: {chunk_key}"
                )
            self._tiers.touch([chunk_key])
            return existing
        if parent_key == chunk_key:
            raise ValueError("a chunk cannot be its own parent")
        pos_offset = self._pe.assign_offset(chunk_key, token_count)
        meta = self._cm.alloc(
            chunk_key=chunk_key,
            num_slots=num_slots,
            token_count=token_count,
            model_id=model_id,
            pos_offset=pos_offset,
            parent_key=parent_key,
        )
        self._tiers.add_chunk(chunk_key, parent_key)
        return meta

    def _chunk_info(self, match: RetrievalMatch) -> ChunkInfo:
        """Convert a RetrievalMatch to ChunkInfo."""
        meta = match.meta
        pos_offset = self._pe.get_offset(meta, match.target_token_start)
        return ChunkInfo(
            chunk_key=meta.chunk_key,
            start_slot=meta.start_slot,
            num_slots=meta.num_slots,
            token_count=meta.token_count,
            pos_offset=pos_offset,
            model_id=meta.model_id,
            target_token_start=match.target_token_start,
        )

    def _allocation(
        self,
        meta: ChunkMeta,
        token_count: int,
        num_slots: int,
        skipped: bool = False,
    ) -> Allocation:
        """Convert ChunkMeta to Allocation."""
        return Allocation(
            chunk_key=meta.chunk_key,
            start_slot=meta.start_slot,
            num_slots=num_slots,
            pos_offset=meta.pos_offset,
            token_count=token_count,
            skipped=skipped,
        )
