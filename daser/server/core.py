# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import math
import os
from typing import Any, Callable, Optional

# First Party
from daser.config import PREFIX_CLOSED_EVICTION_ENV
from daser.connector.helpers import TokenSequence
from daser.connector.helpers import token_count as count_tokens
from daser.logging import init_logger
from daser.metrics import REGISTRY, MetricsRegistry
from daser.position.base import PositionEncoder
from daser.retrieval.base import RetrievalIndex, RetrievalMatch
from daser.server.chunk_lifecycle import ChunkLifecycle
from daser.server.chunk_manager import ChunkManager
from daser.server.doc_registry import DocEntry, DocRegistry
from daser.server.ghost_admission import GhostAdmission
from daser.server.metadata_store import ChunkMeta

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

# Distinct recent prompts remembered so adaptive ghost admission feeds each
# prompt to its shadow caches once despite repeated scheduler polls.
_GHOST_PROMPT_MEMORY = 4096


@dataclass(frozen=True)
class ChunkInfo:
    """Public chunk metadata returned by cache lookup operations.

    Attributes:
        chunk_key: cache key for the token sequence.
        start_slot: first slot occupied in the ring buffer.
        num_slots: number of contiguous slots occupied.
        token_count: number of tokens covered by the chunk.
        pos_offset: position offset assigned by the position encoder.
        target_token_start: token offset in the current prompt where this
            chunk should be loaded.
        model_id: model identifier used for reuse isolation.
        file_offset: byte offset in the KV store file.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    token_count: int
    pos_offset: int
    model_id: str
    file_offset: int
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
            "file_offset": self.file_offset,
            "target_token_start": self.target_token_start,
        }


@dataclass(frozen=True)
class Allocation:
    """Slot allocation returned to the connector.

    Attributes:
        chunk_key: cache key for the allocated chunk.
        start_slot: first allocated slot.
        num_slots: number of allocated slots.
        file_offset: byte offset in the KV store file.
        pos_offset: position offset assigned by the position encoder.
        token_count: number of tokens covered by this allocation.
        skipped: True when the connector should not write KV: the chunk is
            already committed or owned by another writer, or ghost admission
            declined its first store (then no slots are allocated and
            ``start_slot`` and ``file_offset`` are -1).
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    file_offset: int
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
            "file_offset": self.file_offset,
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


class ServerCore:
    """Shared control-plane core for DaseR server APIs.

    The core owns all mutable server-side cache and document state. The
    HTTP server calls document-level methods directly, while the IPC server
    calls connector cache methods.

    Args:
        chunk_manager: ring-buffer allocator and persistence coordinator.
        retrieval_index: cache lookup index.
        position_encoder: position-offset policy.
        slot_size: bytes per KV slot.
        block_tokens: tokens per vLLM block.

    Async/thread-safety:
        Intended to run on the DaseR server asyncio event loop. Methods do
        not perform blocking I/O. Callers should serialize access through
        the single event loop.
    """

    def __init__(
        self,
        chunk_manager: ChunkManager,
        retrieval_index: RetrievalIndex,
        position_encoder: PositionEncoder,
        slot_size: int,
        block_tokens: int = 16,
        metrics_registry: MetricsRegistry | None = None,
    ) -> None:
        self._cm = chunk_manager
        self._ri = retrieval_index
        self._pe = position_encoder
        self._slot_size = slot_size
        self._block_tokens = block_tokens
        self._metrics = metrics_registry or REGISTRY
        self._lifecycle = ChunkLifecycle()
        self._commit_requests = 0
        self._late_evicted_commits = 0
        self._lookup_requests = 0
        self._lookup_hits = 0
        self._packed_slots: dict[int, dict[str, Any]] = {}
        self._packed_slot_owners: dict[int, str] = {}
        self._packed_slots_by_owner: dict[str, set[int]] = {}
        self._removal_listeners: list[Callable[[str], None]] = []
        self._publish_probe = os.environ.get(PUBLISH_PROBE_ENV) == "1"
        # Prefix-closed eviction: key -> keys a lookup reaches only through
        # it, learned from looked-up prompts (research switch only). Edges to
        # children that are neither stored, pending nor learned since the last
        # prune are dropped periodically, so it stays proportional to the
        # cache instead of to every prompt ever seen.
        self._prefix_closed = os.environ.get(PREFIX_CLOSED_EVICTION_ENV) == "1"
        self._chain_children: dict[str, set[str]] = {}
        self._chain_prompt: int | None = None
        self._chain_edges = 0
        self._chain_fresh: set[str] = set()
        self._chain_limit = 2 * chunk_manager.total_slots
        self._ghost_admission = GhostAdmission.from_env(chunk_manager.total_slots)
        # Schedulers poll waiting prompts every step, interleaved, so the
        # shadow caches see each distinct prompt once.
        self._ghost_prompts: OrderedDict[int, None] = OrderedDict()
        self._record_capacity_metrics()

    @property
    def chunk_manager(self) -> ChunkManager:
        """Return the owned ChunkManager."""
        return self._cm

    @property
    def index_epoch(self) -> int:
        """Return a counter that advances whenever a chunk leaves the index.

        Async/thread-safety:
            Read on the owning server event loop.
        """
        return self._cm.store.removals

    def add_chunk_removal_listener(self, listener: Callable[[str], None]) -> None:
        """Register a callback for chunks that leave the metadata store.

        Args:
            listener: Called with the chunk key after the chunk is removed by
                ring eviction, explicit eviction or document deletion.

        Async/thread-safety:
            Listeners run synchronously on the server event loop and must not
            block or re-enter the core.
        """
        self._removal_listeners.append(listener)

    async def rebuild_retrieval_index(self) -> None:
        """Reinsert restored chunks into the retrieval index.

        Async/thread-safety:
            Run during startup before serving requests.
        """
        # Packed refs are an in-memory publication artifact, not part of the
        # persisted chunk index. A restart must begin without reloadable
        # compressed records until the worker republishes them.
        self._packed_slots.clear()
        self._packed_slot_owners.clear()
        self._packed_slots_by_owner.clear()
        for meta in list(self._cm.store.iter_chunks()):
            await self._ri.insert(meta)
            self._lifecycle.mark_committed(meta.chunk_key)

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

    async def estimate_reusable_tokens(
        self, tokens: TokenSequence, model_id: str
    ) -> int:
        """Estimate how many leading prompt tokens a lookup will reuse.

        Counts committed chunks and chunks a store writer is still publishing,
        so a follow-up request arriving before its producer's commit is not
        mistaken for a miss.  Unlike :meth:`lookup` this records no lookup
        metrics, touches no replacement state and starts no reads.

        Args:
            tokens: prompt token IDs.
            model_id: model identifier.

        Returns:
            Contiguous reusable token count from the prompt start.

        Async/thread-safety:
            Performs no blocking I/O and should run on the server event loop.
        """
        return await self._ri.reusable_tokens(
            tokens, model_id, self._lifecycle.pending_write_keys
        )

    def prefill_holes(
        self, tokens: TokenSequence, model_id: str, min_tail_tokens: int
    ) -> list[tuple[int, int]]:
        """Find missing block runs that hide stored blocks behind them.

        A prefix lookup stops at the first missing block, yet a rolling key
        names its whole prefix, so a stored block after a gap still holds
        this prompt's exact KV. Recomputing only the gap makes it reachable.
        Like :meth:`estimate_reusable_tokens` this counts pending writers as
        held and records no metrics.

        Args:
            tokens: prompt token IDs.
            model_id: model identifier.
            min_tail_tokens: Smallest held run after a gap worth recovering.

        Returns:
            ``(start, end)`` token spans of the gaps to recompute, in prompt
            order; each is followed by at least ``min_tail_tokens`` held
            tokens.

        Async/thread-safety:
            Performs no blocking I/O and should run on the server event loop.
        """
        held = self._ri.resident_blocks(
            tokens, model_id, self._lifecycle.pending_write_keys
        )
        block = self._block_tokens
        holes: list[tuple[int, int]] = []
        i = 0
        while i < len(held):
            if held[i]:
                i += 1
                continue
            start = i
            while i < len(held) and not held[i]:
                i += 1
            tail = i
            while tail < len(held) and held[tail]:
                tail += 1
            if i < len(held) and (tail - i) * block >= min_tail_tokens:
                holes.append((start * block, i * block))
        return holes

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
        if self._prefix_closed:
            self._learn_chain(tokens)
        if self._ghost_admission is not None and self._ghost_admission.adaptive:
            prompt = hash(tokens if isinstance(tokens, bytes) else tuple(tokens))
            if prompt not in self._ghost_prompts:
                self._ghost_prompts[prompt] = None
                if len(self._ghost_prompts) > _GHOST_PROMPT_MEMORY:
                    self._ghost_prompts.popitem(last=False)
                self._ghost_admission.observe(self._ri.chain_keys(tokens))
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
        self._record_capacity_metrics()
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

    async def alloc_chunk(
        self, chunk_key: str, token_count: int, model_id: str
    ) -> Allocation:
        """Allocate slots for a new chunk or reuse a compatible allocation.

        Args:
            chunk_key: cache key for the token sequence.
            token_count: number of tokens in this chunk.
            model_id: model identifier.

        Returns:
            Allocation metadata for connector writes.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        allocations = await self.alloc_chunks(
            [{"chunk_key": chunk_key, "token_count": token_count}],
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
            chunks: chunk descriptors with ``chunk_key`` and ``token_count``.
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
            if should_write and not self._admit_store(chunk_key):
                allocations.append(
                    self._declined_allocation(chunk_key, token_count, num_slots)
                )
                continue
            meta = await self._alloc_or_get_chunk(
                chunk_key=chunk_key,
                token_count=token_count,
                num_slots=num_slots,
                model_id=model_id,
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
        if should_write and not self._admit_store(chunk_key):
            return MatchAndAllocResult(
                chunks=[],
                alloc=self._declined_allocation(chunk_key, aligned, num_slots),
            )
        meta = await self._alloc_or_get_chunk(
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

    async def commit_chunk(
        self, chunk_key: str, tp_rank: int = 0, tp_size: int = 1
    ) -> None:
        """Mark a chunk as committed and visible to lookup.

        Args:
            chunk_key: cache key for the chunk.
            tp_rank: tensor-parallel rank whose shard finished storing.
            tp_size: total tensor-parallel ranks required before publication.

        Raises:
            ValueError: if the chunk was not allocated.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        meta = self._cm.store.get(chunk_key)
        if meta is None:
            if self._lifecycle.is_evicted(chunk_key):
                self._commit_requests += 1
                self._late_evicted_commits += 1
                self._metrics.counter(
                    "daser_cache_late_evicted_commits_total",
                    "Commit requests ignored because the allocation was evicted.",
                ).inc()
                logger.debug(
                    "[CORE] ignore late commit for evicted key=%s",
                    chunk_key[:8],
                )
                return
            raise ValueError(f"chunk_key not found: {chunk_key}")
        self._commit_requests += 1
        if not self._lifecycle.record_commit_shard(chunk_key, tp_rank, tp_size):
            return
        try:
            await self._ri.insert(meta)
        except BaseException:
            self._lifecycle.abort_publish(chunk_key)
            raise
        publish_s, overlap_wait_s = self._lifecycle.mark_committed(chunk_key)
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
        self._record_capacity_metrics()
        logger.debug(
            "[CORE] commit_chunk key=%s tp_rank=%d tp_size=%d",
            chunk_key[:8],
            tp_rank,
            tp_size,
        )

    async def record_store_ranges(
        self,
        spans: list[dict[str, Any]],
        tp_rank: int,
        tp_size: int,
        local_slot_size: int,
        rank_stride_bytes: int,
    ) -> list[str]:
        """Record completed transfer ranges and publish fully written chunks.

        Args:
            spans: Accepted store spans with chunk allocation metadata.
            tp_rank: Tensor-parallel rank that completed the transfer.
            tp_size: Total tensor-parallel ranks required for publication.
            local_slot_size: Bytes occupied by one rank-local KV slot.
            rank_stride_bytes: File-byte distance between TP rank lanes.

        Returns:
            Chunk keys whose rank shard became completely written and whose
            existing commit path was invoked.

        Raises:
            ValueError: if transfer geometry or packed span metadata is invalid.

        Async/thread-safety:
            Runs on the server event loop. It only updates control-plane state
            and awaits the existing retrieval-index commit operation. Spans
            whose chunk was evicted or re-allocated since the write started
            are skipped.
        """
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid TP rank {tp_rank} for size {tp_size}")
        if local_slot_size <= 0:
            raise ValueError("local_slot_size must be positive")
        if tp_size > 1 and rank_stride_bytes <= 0:
            raise ValueError("rank_stride_bytes must be positive for TP stores")

        ready: list[str] = []
        ready_set: set[str] = set()
        for span in spans:
            chunk_key = str(span.get("chunk_key", ""))
            if not chunk_key:
                continue
            start_slot = int(span.get("start_slot", -1))
            num_slots = int(span.get("num_slots", 0))
            # Eviction can remove a chunk while its store is in flight, and a
            # later request may allocate the same key elsewhere before this
            # write reports. Such a write belongs to no live allocation.
            if not self.is_current_allocation(chunk_key, start_slot, num_slots):
                logger.debug(
                    "[CORE] drop store range of stale allocation key=%s slot=%d",
                    chunk_key[:8],
                    start_slot,
                )
                continue
            expected_start = tp_rank * rank_stride_bytes + start_slot * local_slot_size
            expected_end = expected_start + num_slots * local_slot_size
            range_start = int(span["file_offset"])
            range_end = range_start + int(span["nbytes"])
            if bool(span.get("packed", False)):
                logical_start = int(span.get("logical_slot_start", -1))
                logical_count = int(span.get("logical_slot_count", 0))
                if logical_count != 1:
                    raise ValueError("packed store spans must describe one slot")
                slot_index = 0 if num_slots == 1 else logical_start
                complete = self._lifecycle.record_written_slot(
                    chunk_key,
                    tp_rank,
                    slot_index,
                    num_slots,
                )
                physical_slot = start_slot + slot_index
                self._clear_packed_slot(physical_slot)
                self._packed_slots[physical_slot] = {
                    "slot_id": physical_slot,
                    "mode": str(span.get("mode", "compressed")),
                    "file_offset": range_start,
                    "stored_length": int(span["nbytes"]),
                }
                self._packed_slot_owners[physical_slot] = chunk_key
                self._packed_slots_by_owner.setdefault(chunk_key, set()).add(
                    physical_slot
                )
            else:
                complete = self._lifecycle.record_written_range(
                    chunk_key,
                    tp_rank,
                    expected_start,
                    expected_end,
                    range_start,
                    range_end,
                )
            if complete and chunk_key not in ready_set:
                ready.append(chunk_key)
                ready_set.add(chunk_key)

        for chunk_key in ready:
            await self.commit_chunk(chunk_key, tp_rank=tp_rank, tp_size=tp_size)
        return ready

    def packed_slot_refs(
        self, start_slot: int, num_slots: int
    ) -> list[dict[str, int | str]]:
        """Return online packed records for a logical slot interval.

        Args:
            start_slot: First logical store slot.
            num_slots: Number of slots requested.

        Returns:
            Ordered record metadata, or an empty list until every record has
            been published by the online store path.
        """
        if start_slot < 0 or num_slots <= 0:
            return []
        refs = [
            self._packed_slots.get(start_slot + index) for index in range(num_slots)
        ]
        if any(ref is None for ref in refs):
            return []
        owners = {
            self._packed_slot_owners.get(start_slot + index)
            for index in range(num_slots)
        }
        if None in owners:
            return []
        for owner in owners:
            if (
                owner is None
                or self._cm.store.get(owner) is None
                or not self._lifecycle.is_committed(owner)
            ):
                self._clear_packed_refs(owner)
                return []
        return [dict(ref) for ref in refs if ref is not None]

    def packed_slot_owners(self, start_slot: int, num_slots: int) -> list[str]:
        """Return the chunks owning online packed records in a slot interval.

        A coalesced prefix match reports only its last key; this recovers
        every stored chunk it spans.

        Args:
            start_slot: First logical store slot.
            num_slots: Number of slots in the interval.

        Returns:
            Distinct owner chunk keys in slot order; slots without a packed
            record are skipped.

        Async/thread-safety:
            Reads in-memory state on the server event loop. It performs no
            blocking I/O.
        """
        owners: dict[str, None] = {}
        for slot in range(start_slot, start_slot + max(0, num_slots)):
            owner = self._packed_slot_owners.get(slot)
            if owner is not None:
                owners[owner] = None
        return list(owners)

    def is_chunk_committed(self, chunk_key: str) -> bool:
        """Return whether a chunk key has been committed.

        Args:
            chunk_key: cache key to check.

        Returns:
            True after ``commit_chunk`` has published the chunk and before it is
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
            ``commit_chunk`` on the same event loop.
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
        self._clear_packed_refs(chunk_key)
        self._lifecycle.discard_owner(chunk_key)
        return True

    async def evict_chunk(self, chunk_key: str) -> None:
        """Evict a chunk from retrieval and metadata state.

        Args:
            chunk_key: cache key for the chunk.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        await self._remove_chunk(chunk_key, reason="explicit")

    async def evict_oldest_chunk(self, protected_keys: set[str]) -> ChunkMeta | None:
        """Evict a chunk to relieve physical byte pressure.

        Compressed-online stores over-provision logical slots, so the packed
        arena can fill before the ring runs out of slots. By default this
        advances the ring tail past skip blocks and evicts the first chunk
        found (FIFO), using the same cleanup as automatic ring eviction.

        Args:
            protected_keys: Chunk keys that must stay resident, e.g. the
                chunks being placed by the current store call.

        Returns:
            Metadata of the evicted chunk, or None when the ring is empty or
            only protected chunks remain in reach.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        store = self._cm.store
        while len(store):
            entry = store.get_slot_entry(self._cm.tail_slot)
            meta = store.get(entry.chunk_key) if entry.kind == "chunk" else None
            if meta is not None and meta.chunk_key in protected_keys:
                return None
            self._cm.evict_oldest()
            if meta is not None:
                await self._drain_ring_evictions(frozenset(protected_keys))
                return meta
        return None

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
        logger.info(
            "[CORE] delete_document doc_id=%s chunks_evicted=%d",
            doc_id,
            chunks_evicted,
        )
        return DeleteDocumentResult(chunks_evicted=chunks_evicted)

    async def _drain_ring_evictions(
        self, protected: frozenset[str] = frozenset()
    ) -> None:
        """Remove ring-evicted chunks from retrieval index.

        Args:
            protected: Chunk keys prefix-closed eviction must keep, e.g. the
                chunk whose allocation evicted the ring tail.

        Async/thread-safety:
            Runs on the server event loop after allocation.
        """
        evicted = self._cm.drain_evicted_chunk_keys()
        for chunk_key in evicted:
            self._clear_packed_refs(chunk_key)
            await self._ri.remove(chunk_key)
            self._lifecycle.mark_evicted(chunk_key)
            self._notify_chunk_removed(chunk_key)
            self._metrics.counter(
                "daser_cache_evicted_chunks_total",
                "Chunks evicted from cache metadata.",
            ).inc(labels={"reason": "capacity"})
            logger.debug("[CORE] removed auto-evicted chunk key=%s", chunk_key[:8])
        if self._prefix_closed and evicted:
            await self._remove_orphans(evicted, protected)
        self._record_capacity_metrics()

    def _record_capacity_metrics(self) -> None:
        """Publish current byte capacity gauges."""
        total_slots = self._cm.physical_slots
        used_slots = min(total_slots, self._cm.total_slots - self._cm.free_slots)
        self._metrics.gauge(
            "daser_store_l2_bytes_capacity",
            "Total L2 store capacity in bytes.",
        ).set(total_slots * self._slot_size)
        self._metrics.gauge(
            "daser_store_l2_bytes_used",
            "Currently used L2 store bytes.",
        ).set(used_slots * self._slot_size)

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
            self._clear_packed_refs(chunk_key)
            self._lifecycle.discard(chunk_key)
            return False
        if doc_id in meta.doc_ids:
            meta.doc_ids.remove(doc_id)
        if meta.doc_ids:
            return False
        self._clear_packed_refs(chunk_key)
        self._cm.store.remove(chunk_key)
        self._lifecycle.discard(chunk_key)
        self._notify_chunk_removed(chunk_key)
        return True

    def _clear_packed_slot(self, slot_id: int) -> None:
        """Remove one packed publication before a physical slot is reused."""
        self._packed_slots.pop(slot_id, None)
        owner = self._packed_slot_owners.pop(slot_id, None)
        if owner is not None:
            slots = self._packed_slots_by_owner.get(owner)
            if slots is not None:
                slots.discard(slot_id)
                if not slots:
                    del self._packed_slots_by_owner[owner]

    def _clear_packed_refs(self, chunk_key: str | None) -> None:
        """Remove every packed publication owned by one chunk."""
        if not chunk_key:
            return
        for slot_id in list(self._packed_slots_by_owner.get(chunk_key, ())):
            self._clear_packed_slot(slot_id)

    def _notify_chunk_removed(self, chunk_key: str) -> None:
        """Tell registered listeners that a chunk left the metadata store."""
        for listener in self._removal_listeners:
            listener(chunk_key)

    async def _remove_chunk(self, chunk_key: str, *, reason: str) -> None:
        """Remove one chunk from retrieval, metadata and lifecycle state.

        Args:
            chunk_key: cache key for the chunk.
            reason: ``daser_cache_evicted_chunks_total`` label value.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        await self._ri.remove(chunk_key)
        self._clear_packed_refs(chunk_key)
        meta = self._cm.store.get(chunk_key)
        if meta is not None:
            self._mark_chunk_evicted_in_docs(meta)
            self._cm.store.remove(chunk_key)
            self._notify_chunk_removed(chunk_key)
        self._lifecycle.mark_evicted(chunk_key)
        self._metrics.counter(
            "daser_cache_evicted_chunks_total",
            "Chunks evicted from cache metadata.",
        ).inc(labels={"reason": reason})
        self._record_capacity_metrics()
        logger.debug("[CORE] remove chunk key=%s reason=%s", chunk_key[:8], reason)

    def _learn_chain(self, tokens: TokenSequence) -> None:
        """Record which keys a lookup of ``tokens`` reaches through which.

        Schedulers poll one prompt many times, so a prompt identical to the
        last linked one is skipped before hashing its chain.
        """
        prompt = hash(tokens if isinstance(tokens, bytes) else tuple(tokens))
        if prompt == self._chain_prompt:
            return
        self._chain_prompt = prompt
        chain = self._ri.chain_keys(tokens)
        for parent, child in zip(chain, chain[1:], strict=False):
            children = self._chain_children.setdefault(parent, set())
            if child not in children:
                children.add(child)
                self._chain_edges += 1
            self._chain_fresh.add(child)
        if self._chain_edges > self._chain_limit:
            self._prune_chain()

    def _prune_chain(self) -> None:
        """Drop reachability edges that can no longer guide a cascade.

        An edge matters only while its child is stored, still being written,
        or about to be (learned since the last prune); any other edge is
        relearned by the next lookup that reaches it. The next prune waits
        for as many new edges as are cached (at least two per slot), so the
        map stays within three times that and pruning is amortised O(1).
        """
        pending = self._lifecycle.pending_write_keys
        store = self._cm.store
        edges = cached = 0
        for parent in list(self._chain_children):
            children = set()
            for child in self._chain_children[parent]:
                if child in pending or store.get(child) is not None:
                    cached += 1
                elif child not in self._chain_fresh:
                    continue
                children.add(child)
            if children:
                self._chain_children[parent] = children
                edges += len(children)
            else:
                del self._chain_children[parent]
        self._chain_edges = edges
        self._chain_fresh = set()
        # Size the next generation by the cached edges, not the kept fresh
        # ones, so fresh-but-never-stored edges cannot ratchet the limit up.
        self._chain_limit = edges + max(cached, 2 * self._cm.total_slots)
        self._metrics.gauge(
            "daser_prefix_chain_edges",
            "Reachability edges kept for prefix-closed eviction.",
        ).set(edges)

    async def _remove_orphans(
        self, evicted: list[str], protected: frozenset[str]
    ) -> None:
        """Remove resident chunks that only evicted chunks led to.

        A lookup stops at the first missing key, so every key reached through
        an evicted one is dead weight until that key is stored again. Keys
        still being written (or ``protected``) are kept, and so are their
        descendants.

        Args:
            evicted: Keys just removed by capacity eviction.
            protected: Keys that must stay resident.
        """
        keep = protected | self._lifecycle.pending_write_keys
        stack = list(evicted)
        seen = set(stack)
        while stack:
            for child in tuple(self._chain_children.get(stack.pop(), ())):
                if child in seen or child in keep:
                    continue
                seen.add(child)
                stack.append(child)
                if self._cm.store.get(child) is not None:
                    await self._remove_chunk(child, reason="orphan")

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

    async def _alloc_or_get_chunk(
        self,
        chunk_key: str,
        token_count: int,
        num_slots: int,
        model_id: str,
    ) -> ChunkMeta:
        """Allocate a chunk or reuse a compatible in-flight allocation.

        Args:
            chunk_key: cache key for the token sequence.
            token_count: number of tokens covered by the chunk.
            num_slots: number of slots required.
            model_id: model identifier.

        Returns:
            Existing or newly allocated chunk metadata.

        Raises:
            ValueError: if an existing chunk has incompatible metadata.
            RuntimeError: if allocation does not populate metadata.
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
            return existing

        pos_offset = self._pe.assign_offset(chunk_key, token_count)
        start_slot = self._cm.alloc(
            chunk_key=chunk_key,
            num_slots=num_slots,
            token_count=token_count,
            model_id=model_id,
            pos_offset=pos_offset,
        )
        await self._drain_ring_evictions(frozenset((chunk_key,)))
        meta = self._cm.store.get(chunk_key)
        if meta is None:
            raise RuntimeError(
                f"allocation succeeded at slot {start_slot} but metadata is missing"
            )
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
            file_offset=meta.start_slot * self._slot_size,
            target_token_start=match.target_token_start,
        )

    def _admit_store(self, chunk_key: str) -> bool:
        """Return whether ghost admission lets the first writer store a key.

        Args:
            chunk_key: key with no store owner yet.

        Returns:
            True when ghost admission is disabled, the key is already
            allocated, or the filter admits it.
        """
        if self._ghost_admission is None or self._cm.store.get(chunk_key) is not None:
            return True
        admitted = self._ghost_admission.admit(chunk_key)
        self._metrics.counter(
            "daser_ghost_admission_total",
            "First stores of unallocated keys by ghost admission result.",
        ).inc(labels={"result": "admitted" if admitted else "declined"})
        return admitted

    def _declined_allocation(
        self, chunk_key: str, token_count: int, num_slots: int
    ) -> Allocation:
        """Return the skipped allocation of a store ghost admission declined."""
        return Allocation(
            chunk_key=chunk_key,
            start_slot=-1,
            num_slots=num_slots,
            file_offset=-1,
            pos_offset=0,
            token_count=token_count,
            skipped=True,
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
            file_offset=meta.start_slot * self._slot_size,
            pos_offset=meta.pos_offset,
            token_count=token_count,
            skipped=skipped,
        )
