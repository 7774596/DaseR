# SPDX-License-Identifier: Apache-2.0
"""Chunk residency in the L1 (pinned host) and L2 (store file) tiers.

A chunk has at most one copy per tier. A copy is one grain-aligned extent
holding the chunk's records back to back at 4 KiB offsets, at the same
lane-relative offset on every TP rank lane. Each tier has its own TLSF
allocator and its own leaf-first LRU heap over the shared chunk tree. See
``docs/design/tier-residency.md``.
"""

# Standard
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

# First Party
from daser.logging import init_logger
from daser.metrics import REGISTRY, MetricsRegistry
from daser.server.allocator import TlsfAllocator
from daser.server.eviction import EvictionTree

logger = init_logger(__name__)

L1 = "l1"
L2 = "l2"
TIERS = (L1, L2)
#: Eviction-tree heap of L1-only chunks waiting to be written to L2
#: (exclusive tiers).
_DEMOTE = "demote"
#: Alignment of every record inside a copy (O_DIRECT / GDS).
RECORD_ALIGNMENT = 4096


def _round_up(value: int, multiple: int) -> int:
    """Return ``value`` rounded up to a multiple of ``multiple``."""
    return -(-value // multiple) * multiple


def _other(tier: str) -> str:
    """Return the tier that is not ``tier``."""
    return L2 if tier == L1 else L1


def record_offsets(record_sizes: tuple[int, ...]) -> list[int]:
    """Return each record's byte offset inside its chunk copy.

    Args:
        record_sizes: Per-slot record sizes in bytes, in slot order.

    Returns:
        ``offsets[i] = sum(record_sizes[:i])``.

    Async/thread-safety:
        Pure helper safe to call from any thread.
    """
    offsets = []
    total = 0
    for size in record_sizes:
        offsets.append(total)
        total += size
    return offsets


class TierFull(MemoryError):
    """No tier space can be freed for a store placement.

    Attributes:
        tier: Tier that ran out of space.
        nbytes: Extent bytes that did not fit.
        io_blocked: True when in-flight IO pins bytes of the tier, so space
            may appear once it finishes (see ``TierIndex.io_released``).
    """

    def __init__(self, tier: str, nbytes: int, io_blocked: bool) -> None:
        super().__init__(
            f"{tier} cannot place {nbytes} bytes (io_blocked={io_blocked})"
        )
        self.tier = tier
        self.nbytes = nbytes
        self.io_blocked = io_blocked


class ChunkUnavailable(LookupError):
    """A chunk has no readable copy (removed, or never fully stored)."""


@dataclass(frozen=True)
class TierLayout:
    """Static geometry of the tiers, fixed at server startup.

    Attributes:
        grain: Allocation unit of both tiers in bytes; a power of two and a
            multiple of 4096.
        l1_lane_bytes: Bytes of each TP rank's L1 lane; 0 disables L1.
        l2_lane_bytes: Bytes of each TP rank's L2 lane; 0 disables L2.
        tensor_parallel_size: Number of rank lanes per tier.
        l1_record_bytes: When positive, every L1 record occupies at least
            this many bytes (raw-equivalent L1 accounting); 0 uses the
            stored size.

    Raises:
        ValueError: if a field is invalid or both tiers are disabled.
    """

    grain: int
    l1_lane_bytes: int
    l2_lane_bytes: int
    tensor_parallel_size: int = 1
    l1_record_bytes: int = 0

    def __post_init__(self) -> None:
        grain = self.grain
        if grain <= 0 or grain & (grain - 1) or grain % RECORD_ALIGNMENT:
            raise ValueError(
                f"tier grain must be a power of two multiple of "
                f"{RECORD_ALIGNMENT}, got {grain}"
            )
        for name in ("l1_lane_bytes", "l2_lane_bytes"):
            value = getattr(self, name)
            if value < 0 or value % grain:
                raise ValueError(
                    f"{name} must be a non-negative multiple of {grain}, got {value}"
                )
        if not self.l1_lane_bytes and not self.l2_lane_bytes:
            raise ValueError("at least one tier must be enabled")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if self.l1_record_bytes < 0 or self.l1_record_bytes % RECORD_ALIGNMENT:
            raise ValueError("l1_record_bytes must be a non-negative 4 KiB multiple")

    def lane_bytes(self, tier: str) -> int:
        """Return the lane size of ``tier`` in bytes (0 when disabled).

        Args:
            tier: ``L1`` or ``L2``.
        """
        return self.l1_lane_bytes if tier == L1 else self.l2_lane_bytes

    def enabled(self, tier: str) -> bool:
        """Return whether ``tier`` has capacity.

        Args:
            tier: ``L1`` or ``L2``.
        """
        return self.lane_bytes(tier) > 0

    def absolute_offset(self, tier: str, rank: int, lane_offset: int) -> int:
        """Return the byte offset of ``lane_offset`` on ``rank``'s lane.

        Args:
            tier: ``L1`` (pinned arena offset) or ``L2`` (file offset).
            rank: TP rank in ``[0, tensor_parallel_size)``.
            lane_offset: Lane-relative byte offset.

        Returns:
            ``rank * lane_bytes(tier) + lane_offset``.

        Raises:
            ValueError: if ``rank`` is out of range.
        """
        if not 0 <= rank < self.tensor_parallel_size:
            raise ValueError(f"invalid TP rank {rank}")
        return rank * self.lane_bytes(tier) + lane_offset

    def extent_length(self, tier: str, record_sizes: tuple[int, ...]) -> int:
        """Return the extent bytes a chunk copy occupies in ``tier``.

        Args:
            tier: ``L1`` or ``L2``.
            record_sizes: Per-slot record sizes in bytes.

        Returns:
            The packed size rounded up to the grain; in L1 at least
            ``len(record_sizes) * l1_record_bytes``.
        """
        length = sum(record_sizes)
        if tier == L1:
            length = max(length, len(record_sizes) * self.l1_record_bytes)
        return _round_up(length, self.grain)


@dataclass(eq=False)
class TierCopy:
    """One chunk's extent in one tier, shared by every rank lane.

    Attributes:
        chunk_key: Owning chunk.
        tier: ``L1`` or ``L2``.
        offset: Lane-relative extent start; multiple of the grain.
        length: Extent bytes; multiple of the grain.
        valid_ranks: Bit ``r`` is set once rank ``r``'s bytes are in place.
        io_pins: In-flight reads or writes using the extent; it is never
            released or evicted while positive.
        attached: False once the copy left its chunk; its extent is released
            when the last io pin drops.
        promotion: L1 only: resolves (True on success) when the L2 -> L1
            read filling this copy finishes.
    """

    chunk_key: str
    tier: str
    offset: int
    length: int
    valid_ranks: int = 0
    io_pins: int = 0
    attached: bool = True
    promotion: asyncio.Future[bool] | None = None

    def is_valid(self, rank: int) -> bool:
        """Return whether ``rank``'s bytes are in place.

        Args:
            rank: TP rank.
        """
        return bool(self.valid_ranks >> rank & 1)


@dataclass(eq=False)
class ChunkTiers:
    """Tier copies of one chunk.

    Attributes:
        record_sizes: Per-slot record sizes in bytes (same on every rank).
        l1: L1 copy, if any.
        l2: L2 copy, if any.
        lease_pins: Request leases keeping the L1 copy resident.
    """

    record_sizes: tuple[int, ...]
    l1: TierCopy | None = None
    l2: TierCopy | None = None
    lease_pins: int = 0

    def copy(self, tier: str) -> TierCopy | None:
        """Return the copy in ``tier``.

        Args:
            tier: ``L1`` or ``L2``.
        """
        return self.l1 if tier == L1 else self.l2


@dataclass(frozen=True)
class StorePlacement:
    """Where one rank writes a chunk; returned by ``TierIndex.place_store``.

    Every copy in the placement carries one io pin taken for the caller,
    released by ``TierIndex.unpin_io``.

    Attributes:
        chunk_key: Stored chunk.
        record_sizes: Per-slot record sizes in bytes.
        l1: L1 destination (the D2H snapshot is kept there), or None when L1
            did not admit the chunk (the store goes through staging).
        l2: L2 destination, or None when L2 is disabled.
    """

    chunk_key: str
    record_sizes: tuple[int, ...]
    l1: TierCopy | None
    l2: TierCopy | None


@dataclass(frozen=True)
class Promotion:
    """An admitted L2 -> L1 copy; returned by ``TierIndex.begin_promotion``.

    Both copies carry one io pin until ``TierIndex.finish_promotion``.

    Attributes:
        chunk_key: Promoted chunk.
        record_sizes: Per-slot record sizes in bytes.
        source: Valid L2 copy to read.
        target: New L1 copy to fill.
    """

    chunk_key: str
    record_sizes: tuple[int, ...]
    source: TierCopy
    target: TierCopy


@dataclass(frozen=True)
class Demotion:
    """An admitted L1 -> L2 write; returned by ``TierIndex.begin_demotions``.

    Exclusive tiers only. Both copies carry one io pin until
    ``TierIndex.finish_demotion``.

    Attributes:
        chunk_key: Demoted chunk.
        record_sizes: Per-slot record sizes in bytes.
        source: Valid L1 copy to write from.
        target: New L2 copy to fill.
    """

    chunk_key: str
    record_sizes: tuple[int, ...]
    source: TierCopy
    target: TierCopy


@dataclass(frozen=True)
class Lease:
    """A request lease keeping one chunk's L1 copy from eviction.

    Attributes:
        chunk_key: Leased chunk.
        chunk: The chunk instance the lease pins; a later chunk under the
            same key is unaffected by releasing this lease.
    """

    chunk_key: str
    chunk: ChunkTiers


@dataclass(frozen=True)
class L2Placement:
    """Persisted L2 location of one chunk.

    Attributes:
        chunk_key: Chunk key.
        offset: Lane-relative extent start.
        record_sizes: Per-slot record sizes in bytes.
    """

    chunk_key: str
    offset: int
    record_sizes: tuple[int, ...]


@dataclass(frozen=True)
class TierStats:
    """Point-in-time state of one tier; byte fields are per rank lane.

    Attributes:
        capacity: Lane bytes.
        used: Bytes of copies (including retired copies still io pinned).
        free: Allocator free bytes; ``used + free == capacity``.
        holes: Maximal free extents.
        largest_free: Longest free extent.
        copies: Copies attached to chunks.
        evicted_copies: Copies evicted since startup.
        evicted_bytes: Extent bytes evicted since startup.
        extra_evicted_bytes: Evicted bytes beyond the deficits that triggered
            them (fragmentation cost).
        admission_skips: Chunks this tier did not admit (L1 only).
    """

    capacity: int
    used: int
    free: int
    holes: int
    largest_free: int
    copies: int
    evicted_copies: int
    evicted_bytes: int
    extra_evicted_bytes: int
    admission_skips: int


@dataclass
class _TierState:
    """Mutable per-tier bookkeeping."""

    allocator: TlsfAllocator
    used: int = 0
    pinned: int = 0
    copies: int = 0
    evicted_copies: int = 0
    evicted_bytes: int = 0
    extra_evicted_bytes: int = 0
    admission_skips: int = 0
    io_waiters: list[asyncio.Future[None]] = field(default_factory=list)


class TierIndex:
    """Per-chunk tier copies, tier allocators and eviction.

    Chunk nodes are added at allocation (``add_chunk``) and get tier state at
    their first store placement. Eviction takes leaf-first LRU candidates
    per tier: a chunk's copy in tier T is a candidate when it is valid on
    every rank, not io pinned, not lease pinned (L1), not ``blocked``, and
    either the other tier holds a valid copy or the chunk is a leaf. Evicting
    a chunk's last copy removes it from the index and calls ``on_lost``.

    With ``exclusive`` the tiers hold different chunks (see "独占分层" in
    the design doc): a store goes to L1 only when L1 admits it, else to L2
    only; a promotion drops its L2 source; an L1-only (dirty) copy is a
    candidate only when the chunk is an orphan, and leaves L1 through a
    demotion (``begin_demotions`` / ``finish_demotion``) that writes it to
    L2 first.

    Args:
        layout: Tier geometry.
        blocked: Returns True for a chunk that must keep its copies (pending
            writer). Call ``refresh`` when it turns False.
        on_lost: Called synchronously with the key of a chunk whose last
            copy was evicted, after its tier state and tree node are gone.
        metrics_registry: Registry for tier counters and gauges.
        exclusive: Keep L1 and L2 disjoint; needs both tiers.
        clean_bytes: Exclusive only: L1 lane bytes to keep free or clean
            (``demotion_deficit``); None means an eighth of the L1 lane.

    Raises:
        ValueError: if ``exclusive`` is set without both tiers.

    Async/thread-safety:
        Owned by the server event loop. Every method is synchronous except
        ``io_released`` and ``read_plan``; no method blocks.
    """

    def __init__(
        self,
        layout: TierLayout,
        blocked: Callable[[str], bool],
        on_lost: Callable[[str], None],
        metrics_registry: MetricsRegistry | None = None,
        exclusive: bool = False,
        clean_bytes: int | None = None,
    ) -> None:
        if exclusive and not (layout.enabled(L1) and layout.enabled(L2)):
            raise ValueError("exclusive tiers need both L1 and L2")
        self._layout = layout
        self._blocked = blocked
        self._on_lost = on_lost
        self._metrics = metrics_registry or REGISTRY
        self._tp_mask = (1 << layout.tensor_parallel_size) - 1
        self._tiers = {
            tier: _TierState(TlsfAllocator(layout.lane_bytes(tier), layout.grain))
            for tier in TIERS
            if layout.enabled(tier)
        }
        self._chunks: dict[str, ChunkTiers] = {}
        self._exclusive = exclusive
        self._clean_target = (
            layout.l1_lane_bytes // 8 if clean_bytes is None else clean_bytes
        )
        # Exclusive: L1 lane bytes of chunks full in both tiers, and of
        # chunks being demoted.
        self._clean_bytes = 0
        self._demoting_bytes = 0
        heaps = tuple(self._tiers) + ((_DEMOTE,) if exclusive else ())
        self._tree = EvictionTree(heaps, self._is_candidate)

    @property
    def layout(self) -> TierLayout:
        """Return the tier geometry."""
        return self._layout

    def enabled(self, tier: str) -> bool:
        """Return whether ``tier`` has capacity.

        Args:
            tier: ``L1`` or ``L2``.
        """
        return tier in self._tiers

    # ------------------------------------------------------------------
    # Chunk tree
    # ------------------------------------------------------------------

    def add_chunk(self, key: str, parent_key: str = "") -> None:
        """Add an allocated chunk as the most recently used tree node.

        Args:
            key: Chunk key; must not be a node.
            parent_key: Previous chunk of its prefix chain, or empty.

        Raises:
            ValueError: if ``key`` is already a node or its own parent.
        """
        self._tree.add(key, parent_key)

    def touch(self, keys: list[str]) -> None:
        """Mark chunks as just used (later keys more recent).

        Args:
            keys: Chunk keys; unknown keys are ignored.
        """
        self._tree.touch(keys)

    def touch_path(self, key: str) -> None:
        """Mark a chunk and its ancestors as just used.

        Args:
            key: Last chunk of a prefix hit; unknown keys are ignored.
        """
        self._tree.touch_path(key)

    def refresh(self, key: str) -> None:
        """Re-queue a chunk after its ``blocked`` state cleared.

        Args:
            key: Chunk key; unknown keys are ignored.
        """
        self._tree.refresh(key)

    def is_leaf(self, key: str) -> bool:
        """Return whether no child node of ``key`` exists.

        Args:
            key: Chunk key.
        """
        return self._tree.is_leaf(key)

    def want(self, hint_id: str, keys: list[str]) -> None:
        """Keep a queued request's prefix chunks resident until it finishes.

        See ``EvictionTree.want``; hinted chunks are evicted after every
        unhinted one, latest hint first.

        Args:
            hint_id: Request id.
            keys: Prefix-chain keys the request will look up.
        """
        self._tree.want(hint_id, keys)
        self._metrics.counter("daser_queue_hints_total", "Queue hints received.").inc()
        self._metrics.counter(
            "daser_queue_hint_keys_total", "Chunk keys named by queue hints."
        ).inc(len(keys))
        self._publish_hint_metrics()

    def unwant(self, hint_id: str) -> None:
        """Drop a queue hint; its chunks become most recently used.

        Args:
            hint_id: Id given to ``want``; unknown ids are ignored.
        """
        self._tree.unwant(hint_id)
        self._publish_hint_metrics()

    def remove(self, key: str) -> None:
        """Drop a chunk's copies and tree node (explicit eviction, deletion).

        An in-flight promotion of the chunk resolves False; extents still io
        pinned are released when their last pin drops. Outstanding leases
        become no-ops. ``on_lost`` is not called.

        Args:
            key: Chunk key; unknown keys are ignored.
        """
        chunk = self._chunks.pop(key, None)
        if chunk is not None:
            for tier in TIERS:
                copy = chunk.copy(tier)
                if copy is not None:
                    self._detach(chunk, copy)
        self._tree.remove(key)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def chunk(self, key: str) -> ChunkTiers | None:
        """Return the tier state of a chunk, if it has any copy.

        Args:
            key: Chunk key.

        Returns:
            The live state; callers must treat it as read-only.
        """
        return self._chunks.get(key)

    def is_full(self, copy: TierCopy | None) -> bool:
        """Return whether ``copy`` is attached and valid on every rank.

        Args:
            copy: Copy or None.
        """
        return copy is not None and copy.attached and copy.valid_ranks == self._tp_mask

    def is_present(self, key: str) -> bool:
        """Return whether some tier copy of ``key`` is valid on every rank.

        Args:
            key: Chunk key.
        """
        chunk = self._chunks.get(key)
        return chunk is not None and (self.is_full(chunk.l1) or self.is_full(chunk.l2))

    def is_rank_valid(self, key: str, rank: int) -> bool:
        """Return whether some tier copy of ``key`` holds ``rank``'s bytes.

        Args:
            key: Chunk key.
            rank: TP rank.
        """
        chunk = self._chunks.get(key)
        if chunk is None:
            return False
        return any(
            copy is not None and copy.attached and copy.is_valid(rank)
            for copy in (chunk.l1, chunk.l2)
        )

    def stats(self, tier: str) -> TierStats:
        """Return the current state of an enabled tier.

        Args:
            tier: ``L1`` or ``L2``.

        Raises:
            KeyError: if ``tier`` is disabled.
        """
        state = self._tiers[tier]
        allocator = state.allocator
        return TierStats(
            capacity=allocator.size,
            used=state.used,
            free=allocator.free_bytes,
            holes=allocator.holes,
            largest_free=allocator.largest_free,
            copies=state.copies,
            evicted_copies=state.evicted_copies,
            evicted_bytes=state.evicted_bytes,
            extra_evicted_bytes=state.extra_evicted_bytes,
            admission_skips=state.admission_skips,
        )

    def publish_metrics(self) -> None:
        """Set the per-tier gauges (bytes summed over all rank lanes).

        Async/thread-safety:
            O(1) per tier; runs after every tier change on the event loop.
            ``largest_free`` scans a size class, so it is only reported by
            ``stats``.
        """
        tp = self._layout.tensor_parallel_size
        for tier, state in self._tiers.items():
            labels = {"tier": tier}
            self._metrics.gauge(
                "daser_tier_bytes_capacity", "Tier capacity in bytes."
            ).set(state.allocator.size * tp, labels=labels)
            self._metrics.gauge(
                "daser_tier_bytes_used", "Tier bytes used by chunk copies."
            ).set(state.used * tp, labels=labels)
            self._metrics.gauge(
                "daser_tier_free_holes", "Disjoint free extents per tier lane."
            ).set(state.allocator.holes, labels=labels)
        if self._exclusive:
            self._metrics.gauge(
                "daser_tier_l1_clean_bytes",
                "Exclusive tiers: L1 bytes of chunks also full in L2.",
            ).set(self._clean_bytes * tp)
            self._metrics.gauge(
                "daser_tier_demoting_bytes",
                "Exclusive tiers: L1 bytes being written to L2.",
            ).set(self._demoting_bytes * tp)

    # ------------------------------------------------------------------
    # Stores
    # ------------------------------------------------------------------

    def place_store(
        self, chunks: list[tuple[str, tuple[int, ...]]]
    ) -> list[StorePlacement]:
        """Place a batch of chunk stores for one rank and pin the copies.

        A chunk with tier state reuses it (other TP ranks and writer
        retries of the same allocation see the same offsets, including an L1
        skip). A new chunk gets an L2 extent (one contiguous reservation for
        all new chunks of the batch if possible, else per chunk, evicting L2
        candidates) and an L1 extent when L1 can admit it after evicting L1
        candidates; otherwise its store skips L1. Without L2 an L1 shortage
        fails the batch. With exclusive tiers a new chunk gets the L1 extent
        alone when L1 admits it, else the L2 extent alone; when L2 cannot
        place it either, the coldest dirty L1 leaf without a lease is dropped
        (its only copy) and L1 is tried again.

        Args:
            chunks: ``(chunk_key, record_sizes)`` per chunk; every key must be
                a tree node. Record sizes are positive 4 KiB multiples.

        Returns:
            One placement per chunk, in order. Every copy carries one io pin
            for the caller.

        Raises:
            ValueError: if a key is not a node, appears twice, or its sizes
                are invalid or differ from an existing placement.
            TierFull: if a new chunk cannot be placed. Nothing of this call
                stays pinned or placed; chunks evicted for it stay evicted.
        """
        seen: set[str] = set()
        for key, sizes in chunks:
            if key not in self._tree:
                raise ValueError(f"store of unallocated chunk: {key}")
            if key in seen:
                raise ValueError(f"chunk stored twice in one batch: {key}")
            seen.add(key)
            self._validate_sizes(sizes)
            existing = self._chunks.get(key)
            if existing is not None and existing.record_sizes != tuple(sizes):
                raise ValueError(f"record sizes changed for chunk: {key}")

        fresh = [
            (key, tuple(sizes)) for key, sizes in chunks if key not in self._chunks
        ]
        pinned: list[TierCopy] = []
        created: dict[str, ChunkTiers] = {}
        try:
            for key, _sizes in chunks:
                existing = self._chunks.get(key)
                if existing is None:
                    continue
                for copy in (existing.l1, existing.l2):
                    if copy is not None:
                        self._pin(copy)
                        pinned.append(copy)
            for key, sizes in fresh:
                created[key] = ChunkTiers(record_sizes=sizes)
            if self._exclusive:
                # L1 first; only the chunks L1 does not admit go to L2. When
                # L2 has no room either, dirty L1 leaves are dropped to make
                # L1 room before the store fails.
                while True:
                    self._place_l1(fresh, created, pinned)
                    fresh = [(k, s) for k, s in fresh if created[k].l1 is None]
                    try:
                        l2_offsets = self._reserve_l2(fresh)
                        break
                    except TierFull:
                        if not self._drop_dirty_leaf():
                            raise
            else:
                l2_offsets = self._reserve_l2(fresh) if L2 in self._tiers else {}
            for key, sizes in fresh:
                chunk = created[key]
                if key in l2_offsets:
                    chunk.l2 = self._new_copy(key, L2, l2_offsets[key], sizes)
                    self._pin(chunk.l2)
                    pinned.append(chunk.l2)
            if not self._exclusive:
                self._place_l1(fresh, created, pinned)
        except TierFull as exc:
            # Created copies are detached, so the unpin releases their extents.
            for copy in reversed(pinned):
                if copy.chunk_key in created:
                    copy.attached = False
                self._unpin(copy)
            raise TierFull(
                exc.tier, exc.nbytes, self._io_blocked(exc.tier, exc.nbytes)
            ) from None
        for key, chunk in created.items():
            self._chunks[key] = chunk
            self._tiers_count(chunk, +1)
        placements = []
        for key, _sizes in chunks:
            chunk = self._chunks[key]
            placements.append(
                StorePlacement(
                    chunk_key=key,
                    record_sizes=chunk.record_sizes,
                    l1=chunk.l1,
                    l2=chunk.l2,
                )
            )
        return placements

    def mark_valid(self, copy: TierCopy, rank: int) -> bool:
        """Record that ``rank``'s bytes of ``copy`` are in place.

        Args:
            copy: Copy from a placement or promotion.
            rank: TP rank.

        Returns:
            False if the copy left its chunk meanwhile (nothing changes).

        Raises:
            ValueError: if ``rank`` is out of range.
        """
        if not 0 <= rank < self._layout.tensor_parallel_size:
            raise ValueError(f"invalid TP rank {rank}")
        if not copy.attached:
            return False
        copy.valid_ranks |= 1 << rank
        if copy.valid_ranks == self._tp_mask:
            # Both this copy and the other tier's copy may now be candidates.
            self._tree.refresh(copy.chunk_key)
        return True

    def discard_incomplete(self, placement: StorePlacement) -> bool:
        """Drop the copies of a failed store that are not valid on every rank.

        Their extents are released once the last io pin drops. A chunk left
        without copies loses its tier state (its tree node stays).

        Args:
            placement: Placement of the failed store.

        Returns:
            Whether the chunk still has a copy.
        """
        chunk = self._chunks.get(placement.chunk_key)
        if chunk is None:
            return False
        for copy in (placement.l1, placement.l2):
            if copy is not None and copy.attached and not self.is_full(copy):
                self._detach(chunk, copy)
        if chunk.l1 is None and chunk.l2 is None:
            del self._chunks[placement.chunk_key]
            return False
        return True

    # ------------------------------------------------------------------
    # IO pins
    # ------------------------------------------------------------------

    def pin_io(self, copy: TierCopy) -> None:
        """Keep ``copy``'s extent from eviction and release during an IO.

        Args:
            copy: Attached copy.

        Raises:
            ValueError: if the copy left its chunk.
        """
        if not copy.attached:
            raise ValueError("cannot pin a detached copy")
        self._pin(copy)

    def unpin_io(self, copy: TierCopy) -> None:
        """Release one io pin of ``copy``.

        A detached copy's extent is freed at its last pin; an attached one
        may become an eviction candidate. Wakes ``io_released`` waiters.

        Args:
            copy: Copy pinned by ``pin_io``, ``place_store``,
                ``begin_promotion`` or ``read_plan``.

        Raises:
            ValueError: if the copy has no io pin.
        """
        if copy.io_pins <= 0:
            raise ValueError("unpin of a copy without io pins")
        self._unpin(copy)

    async def io_released(self, tier: str) -> None:
        """Wait until some io pin of ``tier`` drops to zero.

        Args:
            tier: Enabled tier.

        Async/thread-safety:
            Resolved by ``unpin_io`` on the same event loop. Returns at once
            when the tier has no pinned bytes.
        """
        state = self._tiers[tier]
        if not state.pinned:
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        state.io_waiters.append(future)
        try:
            await future
        finally:
            if future in state.io_waiters:
                state.io_waiters.remove(future)

    # ------------------------------------------------------------------
    # Loads and promotion
    # ------------------------------------------------------------------

    async def read_plan(self, key: str) -> TierCopy:
        """Return the copy a load of ``key`` reads, with one io pin taken.

        L1 when valid on every rank; else waits for an in-flight promotion
        and re-checks; else L2.

        Args:
            key: Chunk key.

        Returns:
            A copy valid on every rank; release with ``unpin_io``.

        Raises:
            ChunkUnavailable: if no copy is valid on every rank.

        Async/thread-safety:
            Awaits only promotion futures; the pin is taken in the same event
            loop step as the final check. Cancellation does not cancel the
            promotion.
        """
        while True:
            chunk = self._chunks.get(key)
            if chunk is None:
                raise ChunkUnavailable(key)
            if self.is_full(chunk.l1):
                assert chunk.l1 is not None
                self._pin(chunk.l1)
                return chunk.l1
            if chunk.l1 is not None and chunk.l1.promotion is not None:
                await asyncio.shield(chunk.l1.promotion)
                continue
            if self.is_full(chunk.l2):
                assert chunk.l2 is not None
                self._pin(chunk.l2)
                return chunk.l2
            raise ChunkUnavailable(key)

    def begin_promotion(self, key: str) -> Promotion | None:
        """Admit an L2-only chunk into L1 for an L2 -> L1 read.

        Evicts L1 candidates as needed. Loads arriving meanwhile wait for the
        promotion (``read_plan``).

        Args:
            key: Chunk key.

        Returns:
            The promotion, or None when L1 is disabled, the chunk has an L1
            copy, no full L2 copy, or L1 cannot admit it (counted as an
            admission skip).

        Async/thread-safety:
            Must run on the event loop (creates a future).
        """
        chunk = self._chunks.get(key)
        if (
            L1 not in self._tiers
            or chunk is None
            or chunk.l1 is not None
            or not self.is_full(chunk.l2)
        ):
            return None
        source = chunk.l2
        assert source is not None
        # L1 eviction only detaches the victims' own copies, so neither this
        # chunk nor its L2 source can disappear while L1 admits it.
        length = self._layout.extent_length(L1, chunk.record_sizes)
        offset = self._reserve(L1, length)
        if offset is None:
            self._count_admission_skip()
            return None
        self._pin(source)
        target = self._new_copy(key, L1, offset, chunk.record_sizes)
        target.promotion = asyncio.get_running_loop().create_future()
        chunk.l1 = target
        self._tiers[L1].copies += 1
        self._pin(target)
        return Promotion(
            chunk_key=key,
            record_sizes=chunk.record_sizes,
            source=source,
            target=target,
        )

    def finish_promotion(self, promotion: Promotion, ok: bool) -> None:
        """Complete a promotion: validate or drop the L1 copy, unpin both.

        Args:
            promotion: Value from ``begin_promotion``.
            ok: Whether the L2 -> L1 read succeeded on every rank.
        """
        target = promotion.target
        future = target.promotion
        target.promotion = None
        chunk = self._chunks.get(promotion.chunk_key)
        if target.attached and chunk is not None:
            if ok:
                if self._exclusive and chunk.l2 is promotion.source:
                    # Exclusive: the chunk moves to L1; in-flight reads keep
                    # the source extent until their pins drop.
                    self._detach(chunk, promotion.source)
                target.valid_ranks = self._tp_mask
            else:
                self._detach(chunk, target)
        self._unpin(target)
        self._unpin(promotion.source)
        if future is not None and not future.done():
            future.set_result(ok and target.attached)
        self._tree.refresh(promotion.chunk_key)

    # ------------------------------------------------------------------
    # Demotion (exclusive tiers)
    # ------------------------------------------------------------------

    @property
    def exclusive(self) -> bool:
        """Return whether L1 and L2 hold disjoint chunks."""
        return self._exclusive

    def demotion_deficit(self) -> int:
        """Return the L1 lane bytes to demote to restore the clean target.

        Returns:
            ``max(0, target - (free + clean + demoting))`` in lane bytes; 0
            without exclusive tiers.
        """
        if not self._exclusive:
            return 0
        free = self._tiers[L1].allocator.free_bytes
        covered = free + self._clean_bytes + self._demoting_bytes
        return max(0, self._clean_target - covered)

    def begin_demotions(self, max_bytes: int) -> list[Demotion]:
        """Admit the coldest dirty L1 chunks into L2 for L1 -> L2 writes.

        Chunks leave the demotion heap in eviction order, tail first: a
        chunk is a candidate only once no child chunk is dirty. Each gets an L2
        extent, evicting only L2 candidates of strictly lower priority, so a
        cold chunk never pushes out a hotter or hinted one. A leaf that does
        not fit is the coldest chunk of both tiers and is dropped from L1
        instead (it may be the chunk's last copy); a non-leaf that does not
        fit ends the batch.

        Args:
            max_bytes: Stop once the admitted L1 lane bytes reach this.

        Returns:
            Admitted demotions; both copies of each carry one io pin until
            ``finish_demotion``.

        Raises:
            RuntimeError: without exclusive tiers.
        """
        if not self._exclusive:
            raise RuntimeError("demotion needs exclusive tiers")
        demotions: list[Demotion] = []
        admitted = 0
        while admitted < max_bytes:
            key = self._tree.pop_victim(_DEMOTE)
            if key is None:
                break
            chunk = self._chunks[key]
            source = chunk.l1
            assert source is not None
            length = self._layout.extent_length(L2, chunk.record_sizes)
            offset = self._reserve(L2, length, below=key)
            if offset is None:
                offset = self._reserve_over_subtree(key, length)
            if offset is None:
                if self._tree.is_leaf(key) and not chunk.lease_pins:
                    self._evict(L1, key)
                    self._count_demotion("dropped")
                    continue
                self._count_demotion("stalled")
                self._tree.refresh(key)
                break
            target = self._new_copy(key, L2, offset, chunk.record_sizes)
            chunk.l2 = target
            self._tiers[L2].copies += 1
            self._pin(source)
            self._pin(target)
            # No longer dirty: its parent may now demote (tail first).
            self._refresh_parent(key)
            self._demoting_bytes += source.length
            admitted += source.length
            demotions.append(
                Demotion(
                    chunk_key=key,
                    record_sizes=chunk.record_sizes,
                    source=source,
                    target=target,
                )
            )
        return demotions

    def finish_demotion(self, demotion: Demotion, ok: bool) -> None:
        """Complete a demotion: validate or drop the L2 copy, unpin both.

        On success the L1 copy is clean and may be evicted at once.

        Args:
            demotion: Value from ``begin_demotions``.
            ok: Whether the L1 -> L2 write succeeded on every rank.
        """
        source, target = demotion.source, demotion.target
        self._demoting_bytes -= source.length
        chunk = self._chunks.get(demotion.chunk_key)
        if target.attached and chunk is not None:
            if ok:
                target.valid_ranks = self._tp_mask
                if self._is_clean(chunk):
                    self._clean_bytes += source.length
                nbytes = target.length * self._layout.tensor_parallel_size
                self._metrics.counter(
                    "daser_tier_demoted_bytes_total",
                    "L1 bytes written to L2 by demotions (all rank lanes).",
                ).inc(nbytes)
            else:
                self._detach(chunk, target)
        self._count_demotion("ok" if ok else "failed")
        self._unpin(target)
        self._unpin(source)
        self._tree.refresh(demotion.chunk_key)

    # ------------------------------------------------------------------
    # Leases
    # ------------------------------------------------------------------

    def pin_lease(self, key: str) -> Lease | None:
        """Keep a chunk's L1 copy from eviction until ``unpin_lease``.

        Args:
            key: Chunk key.

        Returns:
            The lease, or None when the chunk has no tier state.
        """
        chunk = self._chunks.get(key)
        if chunk is None:
            return None
        chunk.lease_pins += 1
        return Lease(chunk_key=key, chunk=chunk)

    def unpin_lease(self, lease: Lease) -> None:
        """Release a lease taken by ``pin_lease``.

        Args:
            lease: Lease to release; a lease of a removed chunk is a no-op.

        Raises:
            ValueError: if the chunk has no lease pin left.
        """
        chunk = lease.chunk
        if chunk.lease_pins <= 0:
            raise ValueError("unpin of a chunk without lease pins")
        chunk.lease_pins -= 1
        if chunk.lease_pins == 0 and self._chunks.get(lease.chunk_key) is chunk:
            self._tree.refresh(lease.chunk_key)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def snapshot_l2(self, keys: list[str]) -> list[L2Placement]:
        """Return the L2 placements of the given chunks that are full in L2.

        Args:
            keys: Chunk keys to consider (typically the committed ones).

        Returns:
            Placements of chunks whose L2 copy is valid on every rank.
        """
        placements = []
        for key in keys:
            chunk = self._chunks.get(key)
            if chunk is not None and self.is_full(chunk.l2):
                assert chunk.l2 is not None
                placements.append(
                    L2Placement(
                        chunk_key=key,
                        offset=chunk.l2.offset,
                        record_sizes=chunk.record_sizes,
                    )
                )
        return placements

    def validate_l2_placements(self, placements: list[L2Placement]) -> None:
        """Check that persisted placements can be restored into this index.

        Args:
            placements: Saved placements.

        Raises:
            ValueError: if L2 is disabled, a key is repeated or already has
                tier state, sizes are invalid, or extents are misaligned, out
                of the lane, overlap each other or overlap used space.
        """
        if L2 not in self._tiers:
            raise ValueError("cannot restore L2 placements without an L2 tier")
        lane = self._layout.l2_lane_bytes
        extents = []
        keys = set()
        for placement in placements:
            key = placement.chunk_key
            if key in self._chunks or key in keys:
                raise ValueError(f"invalid restored chunk: {key}")
            keys.add(key)
            self._validate_sizes(placement.record_sizes)
            length = self._layout.extent_length(L2, placement.record_sizes)
            offset = placement.offset
            if offset < 0 or offset % self._layout.grain or offset + length > lane:
                raise ValueError(f"invalid L2 extent for chunk: {key}")
            extents.append((offset, length))
        extents.sort()
        for (start, length), (next_start, _length) in zip(
            extents, extents[1:], strict=False
        ):
            if start + length > next_start:
                raise ValueError("restored L2 extents overlap")
        if placements and self._tiers[L2].used:
            raise ValueError("cannot restore L2 placements into a used L2 tier")

    def restore_l2(self, placements: list[L2Placement]) -> None:
        """Rebuild L2 copies (valid on every rank) from persisted placements.

        Validates every placement (``validate_l2_placements``) before
        changing anything.

        Args:
            placements: Saved placements; every key must be a tree node.

        Raises:
            ValueError: if a key is not a tree node or validation fails.
        """
        for placement in placements:
            if placement.chunk_key not in self._tree:
                raise ValueError(
                    f"restored chunk is not allocated: {placement.chunk_key}"
                )
        self.validate_l2_placements(placements)
        allocator = self._tiers[L2].allocator
        for placement in placements:
            sizes = tuple(placement.record_sizes)
            length = self._layout.extent_length(L2, sizes)
            allocator.take(placement.offset, length)
            chunk = ChunkTiers(record_sizes=sizes)
            chunk.l2 = self._new_copy(placement.chunk_key, L2, placement.offset, sizes)
            chunk.l2.valid_ranks = self._tp_mask
            self._chunks[placement.chunk_key] = chunk
            self._tiers[L2].copies += 1
            self._tree.refresh(placement.chunk_key)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_candidate(self, tier: str, key: str) -> bool:
        """Return whether ``key``'s copy in ``tier`` may be evicted now.

        For the ``_DEMOTE`` heap: whether its L1 copy may be demoted now.
        """
        chunk = self._chunks.get(key)
        if chunk is None:
            return False
        if tier == _DEMOTE:
            source = chunk.l1
            return (
                chunk.l2 is None
                and source is not None
                and not source.io_pins
                and self.is_full(source)
                and not self._tree.is_orphan(key)
                and not self._blocked(key)
                and not self._has_dirty_child(key)
            )
        copy = chunk.copy(tier)
        if copy is None or copy.io_pins or not self.is_full(copy):
            return False
        if tier == L1 and chunk.lease_pins:
            return False
        if self._blocked(key):
            return False
        if self.is_full(chunk.copy(_other(tier))):
            return True
        if self._exclusive and tier == L1:
            # A dirty copy leaves L1 through a demotion; orphans are dropped.
            return self._tree.is_orphan(key)
        return self._tree.is_leaf(key)

    def _has_dirty_child(self, key: str) -> bool:
        """Return whether a child chunk of ``key`` is in L1 with no L2 copy.

        Demotion runs tail first: a parent demoted before its dirty children
        would sit in L2 as a non-leaf, which is no L2 candidate until those
        children leave, so L2 could fill with chunks it cannot evict.
        """
        chunks = self._chunks
        for child in self._tree.children(key):
            state = chunks.get(child)
            if state is not None and state.l1 is not None and state.l2 is None:
                return True
        return False

    def _reserve_over_subtree(self, key: str, length: int) -> int | None:
        """Reserve an L2 extent for demoting ``key`` by evicting its subtree.

        Called when no L2 candidate of lower priority than ``key`` is left. A
        queue hint names a whole prefix with one seq, so ``key``'s hinted
        descendants tie with it and the ``below`` floor keeps them; dropping
        ``key`` instead would orphan them, and leaving it dirty stalls
        demotion, so L1 stops admitting and promoting. Evicting the deepest
        L2 candidates below ``key`` first keeps the longest prefix.

        Returns:
            Lane offset, or None when no L2 candidate is left below ``key``.
        """
        allocator = self._tiers[L2].allocator
        if length > allocator.size:
            return None
        while (offset := allocator.reserve(length)) is None:
            victim = self._deepest_l2_candidate(key)
            if victim is None:
                return None
            self._evict(L2, victim)
            self._metrics.counter(
                "daser_tier_demotion_subtree_evictions_total",
                "L2 copies below a demoting chunk evicted to make room for it.",
            ).inc()
        return offset

    def _deepest_l2_candidate(self, key: str) -> str | None:
        """Return an L2 candidate below ``key`` with no candidate below it."""
        found: str | None = None
        stack = [(child, 1) for child in self._tree.children(key)]
        depth = 0
        seen = {key}
        while stack:
            node, level = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if level > depth and self._is_candidate(L2, node):
                found, depth = node, level
            stack.extend((child, level + 1) for child in self._tree.children(node))
        return found

    def _refresh_parent(self, key: str) -> None:
        """Re-queue ``key``'s parent: it may have lost its last dirty child."""
        if self._exclusive:
            parent = self._tree.parent(key)
            if parent:
                self._tree.refresh(parent)

    def _drop_dirty_leaf(self) -> bool:
        """Drop the coldest dirty L1 leaf without a lease (exclusive tiers).

        Last resort of a store L2 cannot place: the dropped chunk loses its
        only copy, which beats failing the request.

        Returns:
            Whether a chunk was dropped.
        """
        skipped: list[str] = []
        dropped = False
        while (key := self._tree.pop_victim(_DEMOTE)) is not None:
            if self._tree.is_leaf(key) and not self._chunks[key].lease_pins:
                self._evict(L1, key)
                self._count_demotion("dropped")
                dropped = True
                break
            skipped.append(key)
        for key in skipped:
            self._tree.refresh(key)
        return dropped

    def _validate_sizes(self, sizes: tuple[int, ...]) -> None:
        """Reject an empty or misaligned record size list."""
        if not sizes or any(s <= 0 or s % RECORD_ALIGNMENT for s in sizes):
            raise ValueError(
                f"record sizes must be positive {RECORD_ALIGNMENT}-byte multiples"
            )

    def _new_copy(
        self, key: str, tier: str, offset: int, sizes: tuple[int, ...]
    ) -> TierCopy:
        """Account a reserved extent as a new copy of ``key``."""
        length = self._layout.extent_length(tier, sizes)
        self._tiers[tier].used += length
        return TierCopy(chunk_key=key, tier=tier, offset=offset, length=length)

    def _tiers_count(self, chunk: ChunkTiers, delta: int) -> None:
        """Adjust per-tier attached-copy counts for a chunk's copies."""
        for tier in TIERS:
            if chunk.copy(tier) is not None:
                self._tiers[tier].copies += delta

    def _place_l1(
        self,
        fresh: list[tuple[str, tuple[int, ...]]],
        created: dict[str, ChunkTiers],
        pinned: list[TierCopy],
    ) -> None:
        """Give each new chunk an L1 copy when L1 admits it.

        Raises:
            TierFull: if L1 cannot admit a chunk and L2 is disabled.
        """
        for key, sizes in fresh:
            length = self._layout.extent_length(L1, sizes)
            offset = self._reserve(L1, length) if L1 in self._tiers else None
            if offset is None:
                if L2 not in self._tiers:
                    raise TierFull(L1, length, self._io_blocked(L1, length))
                if L1 in self._tiers:
                    self._count_admission_skip()
                continue
            chunk = created[key]
            chunk.l1 = self._new_copy(key, L1, offset, sizes)
            self._pin(chunk.l1)
            pinned.append(chunk.l1)

    def _reserve_l2(self, fresh: list[tuple[str, tuple[int, ...]]]) -> dict[str, int]:
        """Reserve L2 extents for new chunks, contiguously when possible.

        Returns:
            Lane offset per key.

        Raises:
            TierFull: if some chunk does not fit after eviction; extents
                reserved by this call are released first.
        """
        if not fresh:
            return {}
        allocator = self._tiers[L2].allocator
        lengths = [self._layout.extent_length(L2, sizes) for _key, sizes in fresh]
        offsets: dict[str, int] = {}
        start = allocator.reserve(sum(lengths)) if len(fresh) > 1 else None
        if start is not None:
            for (key, _sizes), length in zip(fresh, lengths, strict=True):
                offsets[key] = start
                start += length
            return offsets
        try:
            for (key, _sizes), length in zip(fresh, lengths, strict=True):
                offset = self._reserve(L2, length)
                if offset is None:
                    raise TierFull(L2, length, io_blocked=False)
                offsets[key] = offset
        except TierFull:
            for (key, _sizes), length in zip(fresh, lengths, strict=True):
                if key in offsets:
                    allocator.release(offsets[key], length)
            raise
        return offsets

    def _reserve(self, tier: str, length: int, below: str | None = None) -> int | None:
        """Reserve ``length`` bytes in ``tier``, evicting candidates.

        Args:
            tier: Tier to reserve in.
            length: Extent bytes.
            below: Evict only candidates of lower priority than this node.

        Returns:
            Lane offset, or None when no candidate is left.
        """
        state = self._tiers[tier]
        if length > state.allocator.size:
            return None
        evicted = 0
        while True:
            offset = state.allocator.reserve(length)
            if offset is not None:
                if evicted:
                    state.extra_evicted_bytes += max(0, evicted - length)
                    self._metrics.counter(
                        "daser_tier_extra_evicted_bytes_total",
                        "Evicted bytes beyond the deficit that triggered them.",
                    ).inc(max(0, evicted - length), labels={"tier": tier})
                return offset
            victim = self._tree.pop_victim(tier, below=below)
            if victim is None:
                return None
            evicted += self._evict(tier, victim)

    def _evict(self, tier: str, key: str) -> int:
        """Evict ``key``'s copy in ``tier``; drop the chunk if it was the last.

        Returns:
            Extent bytes freed.
        """
        chunk = self._chunks[key]
        copy = chunk.copy(tier)
        assert copy is not None
        length = copy.length
        self._detach(chunk, copy)
        state = self._tiers[tier]
        state.evicted_copies += 1
        state.evicted_bytes += length
        labels = {"tier": tier}
        self._metrics.counter(
            "daser_tier_evicted_copies_total", "Chunk copies evicted per tier."
        ).inc(labels=labels)
        self._metrics.counter(
            "daser_tier_evicted_bytes_total", "Extent bytes evicted per tier."
        ).inc(length, labels=labels)
        if self._tree.is_wanted(key):
            self._metrics.counter(
                "daser_queue_hint_evicted_copies_total",
                "Evicted copies a queued request still needed.",
            ).inc(labels=labels)
        survivor = chunk.copy(_other(tier))
        if survivor is not None and not self.is_full(survivor):
            # A partial copy cannot serve reads; the chunk is gone either way.
            self._detach(chunk, survivor)
        if chunk.l1 is None and chunk.l2 is None:
            del self._chunks[key]
            self._tree.remove(key)
            self._on_lost(key)
        else:
            # The surviving copy may no longer be a candidate (its pair is
            # gone); its stale heap entry is validated at pop time.
            self._tree.refresh(key)
        return length

    def _detach(self, chunk: ChunkTiers, copy: TierCopy) -> None:
        """Take ``copy`` off its chunk; free the extent unless io pinned."""
        if self._exclusive and self._is_clean(chunk):
            assert chunk.l1 is not None
            self._clean_bytes -= chunk.l1.length
        if copy.tier == L1:
            chunk.l1 = None
            self._refresh_parent(copy.chunk_key)
        else:
            chunk.l2 = None
        if copy.attached:
            self._tiers[copy.tier].copies -= 1
        copy.attached = False
        if copy.promotion is not None:
            future = copy.promotion
            copy.promotion = None
            if not future.done():
                future.set_result(False)
        if copy.io_pins == 0:
            self._release(copy)

    def _is_clean(self, chunk: ChunkTiers) -> bool:
        """Return whether both of a chunk's copies are full."""
        return self.is_full(chunk.l1) and self.is_full(chunk.l2)

    def _release(self, copy: TierCopy) -> None:
        """Return a detached, unpinned copy's extent to its allocator."""
        state = self._tiers[copy.tier]
        state.allocator.release(copy.offset, copy.length)
        state.used -= copy.length

    def _pin(self, copy: TierCopy) -> None:
        """Take one io pin."""
        if copy.io_pins == 0:
            self._tiers[copy.tier].pinned += copy.length
        copy.io_pins += 1

    def _unpin(self, copy: TierCopy) -> None:
        """Drop one io pin; release or re-queue the copy at zero."""
        copy.io_pins -= 1
        if copy.io_pins:
            return
        state = self._tiers[copy.tier]
        state.pinned -= copy.length
        if not copy.attached:
            self._release(copy)
        else:
            self._tree.refresh(copy.chunk_key)
        waiters, state.io_waiters = state.io_waiters, []
        for future in waiters:
            if not future.done():
                future.set_result(None)

    def _io_blocked(self, tier: str, length: int) -> bool:
        """Return whether finishing IO might make ``length`` bytes fit."""
        state = self._tiers[tier]
        return state.pinned > 0 and length <= state.allocator.size

    def _count_admission_skip(self) -> None:
        """Count one chunk L1 did not admit."""
        self._tiers[L1].admission_skips += 1
        self._metrics.counter(
            "daser_tier_l1_admission_skips_total",
            "Chunks L1 could not admit (stored or read through staging).",
        ).inc()

    def _count_demotion(self, outcome: str) -> None:
        """Count one demotion by outcome (ok, failed, dropped, stalled)."""
        self._metrics.counter(
            "daser_tier_demotions_total",
            "Exclusive-tier L1 -> L2 demotions by outcome.",
        ).inc(labels={"outcome": outcome})

    def _publish_hint_metrics(self) -> None:
        """Set the queue-hint gauges."""
        self._metrics.gauge(
            "daser_queue_hint_requests", "Queued requests with a live hint."
        ).set(self._tree.hinted_requests)
        self._metrics.gauge(
            "daser_queue_hint_keys", "Chunk keys named by a live queue hint."
        ).set(self._tree.hinted_keys)
