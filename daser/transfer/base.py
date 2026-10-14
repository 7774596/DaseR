# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

TransferTier = Literal["l1", "mixed", "l2"]


@dataclass
class TransferStats:
    """Common transfer counters.

    Attributes:
        l1_hits: number of reads served from the memory tier.
        l1_misses: number of reads not found in the memory tier.
        l2_reads: number of reads issued to the SSD tier.
        l2_writes: number of writes issued to the SSD tier.
        l2_write_bytes: bytes written to the SSD tier.
        held_evict_persists: dirty held ranges written to L2 because they
            left the memory tier (exclusive tiers).
        prefetch_requests: number of host-tier prefetch operations.
        prefetch_l1_bytes: requested bytes already resident in L1.
        prefetch_l2_bytes: requested bytes read from L2.
        readahead_read_bytes: bytes read from L2 by lookup readahead.
        readahead_served_bytes: load bytes served from readahead leases.
        readahead_dropped_bytes: readahead bytes released before any load
            used them (overlapping store or budget reclaim).
        readahead_contended_bytes: of the read bytes, those read while
            another readahead also had a read in flight.
        l2_write_contended_bytes: L2 write bytes submitted while a
            readahead was still reading.
        promotion_wait_l2_write_us: time L2 promotions waited for L1 space
            on a pending L2 write that holds pool memory.
        promotion_wait_promotion_us: same, on another in-flight promotion.
        promotion_wait_pool_us: same, on a pool release.
        load_readahead_wait_us: grouped-load time spent waiting for in-flight
            readaheads that cover the requested spans.
        load_resolve_us: grouped-load time spent resolving spans against
            readahead leases and L1 under the metadata lock.
        load_pending_write_us: grouped-load time spent waiting for L2 writes
            of the requested ranges to finish.
        load_l2_us: grouped-load time spent reading L1 misses from L2.
        load_copy_drain_us: grouped-load time spent draining the L1 hit copy
            after the L2 phase finished.
        load_ra_window_us: readahead group time spent waiting for a slot of
            the per-readahead read window.
        load_l2_reserve_us: L2 miss-batch time (readahead and load) spent
            reserving L1 promotions and waiting for overlapping writes before
            the last read is submitted.
        load_l2_io_us: L2 miss-batch time from the last submission until
            every read completed and was published.
        dst_copies: pinned-to-CUDA destination copy submissions.
        dst_copy_runs: contiguous runs those submissions issued.
        dst_copy_bytes: bytes those submissions copied.
        l1_evicted_bytes: accounted bytes L1 replacement evicted.
        l1_evicted_pinned_bytes: of those, still owned by an L2 writer or a
            lease, so residency was dropped without freeing pool memory.
        l1_evicted_unread_bytes: of those, never hit since insertion.
        store_lease_wait_us: time stores waited for a request lease that
            did not yet retain the stored range.
        store_lease_wait_max_us: longest such wait of a single store.
    """

    l1_hits: int = 0
    l1_misses: int = 0
    l2_reads: int = 0
    l2_writes: int = 0
    l2_write_bytes: int = 0
    held_evict_persists: int = 0
    prefetch_requests: int = 0
    prefetch_l1_bytes: int = 0
    prefetch_l2_bytes: int = 0
    readahead_read_bytes: int = 0
    readahead_served_bytes: int = 0
    readahead_dropped_bytes: int = 0
    readahead_contended_bytes: int = 0
    l2_write_contended_bytes: int = 0
    promotion_wait_l2_write_us: int = 0
    promotion_wait_promotion_us: int = 0
    promotion_wait_pool_us: int = 0
    load_readahead_wait_us: int = 0
    load_resolve_us: int = 0
    load_pending_write_us: int = 0
    load_l2_us: int = 0
    load_copy_drain_us: int = 0
    load_ra_window_us: int = 0
    load_l2_reserve_us: int = 0
    load_l2_io_us: int = 0
    dst_copies: int = 0
    dst_copy_runs: int = 0
    dst_copy_bytes: int = 0
    l1_evicted_bytes: int = 0
    l1_evicted_pinned_bytes: int = 0
    l1_evicted_unread_bytes: int = 0
    store_lease_wait_us: int = 0
    store_lease_wait_max_us: int = 0


class LeaseIncompleteError(RuntimeError):
    """A leased prefetch could not retain every requested byte.

    The requested ranges stopped holding the looked-up bytes (for example
    their chunk was evicted and its slots re-allocated after the lookup), so
    the lease was released. Callers recover by looking the request up again.
    """


@dataclass(frozen=True)
class PrefetchResult:
    """Byte attribution for one host-tier prefetch operation."""

    requested_bytes: int
    l1_bytes: int
    l2_bytes: int


class TransferLayer(ABC):
    """Abstract server-owned KV transfer layer.

    Concrete implementations hide whether storage is GDS direct-to-GPU or
    io_uring through pinned host memory.

    Capability surface:
        Backends advertise optional behavior through attributes and overridable
        methods rather than ad-hoc duck typing. ``coalesce_store_spans`` lets a
        backend opt into adjacent store-span coalescing; ``stats`` and
        ``l1_bytes_used`` expose tiering counters; ``drain`` waits for
        background work; ``store_bytes_grouped``/``load_bytes_grouped`` execute
        multi-span batches and default to looping over the single-span methods.

    Async/thread-safety:
        Methods are coroutine-compatible and should be called from the server
        asyncio event loop. Implementations may offload blocking syscalls to an
        executor.
    """

    #: When True the server coalesces adjacent store spans before dispatch.
    coalesce_store_spans: bool = False
    #: When True the backend has a memory tier that can hold records: packed
    #: store spans carrying ``hold=True`` are published to it only (their SSD
    #: write waits for eviction) and ``hold_resident`` can keep L2 records.
    supports_held_stores: bool = False

    @property
    def stats(self) -> TransferStats:
        """Return tiering counters for this backend.

        Returns:
            The backend's ``_stats`` snapshot when it maintains one, otherwise
            zeroed counters for backends without a memory tier.
        """
        stats = getattr(self, "_stats", None)
        return stats if stats is not None else TransferStats()

    @property
    def l1_bytes_used(self) -> int:
        """Return L1 memory-tier bytes currently resident.

        Returns:
            The backend's ``_l1_used`` counter, or 0 for backends without a
            memory tier.
        """
        return int(getattr(self, "_l1_used", 0))

    def debug_state(self) -> dict[str, Any]:
        """Return a small snapshot of in-flight work for stall diagnostics.

        Returns:
            Counts of in-flight loads, leases and writes; empty for backends
            that keep no such state.

        Async/thread-safety:
            May be called from a diagnostic thread without the backend lock;
            values are best-effort and must not be used for control flow.
        """
        return {}

    @abstractmethod
    async def load_bytes(self, dst: Any, file_offset: int, nbytes: int) -> int:
        """Load bytes into ``dst``.

        Args:
            dst: writable buffer or GPU array.
            file_offset: L2 byte offset.
            nbytes: number of bytes to load.

        Returns:
            Number of bytes loaded.
        """

    @abstractmethod
    async def store_bytes(
        self,
        src: Any,
        file_offset: int,
        nbytes: int,
        *,
        accounted_nbytes: int | None = None,
    ) -> int:
        """Store bytes from ``src``.

        Args:
            src: readable buffer or GPU array.
            file_offset: L2 byte offset.
            nbytes: number of bytes to store.
            accounted_nbytes: Optional host-tier capacity charge. Backends
                without a host tier may ignore it.

        Returns:
            Number of bytes stored.
        """

    async def store_bytes_grouped(self, src: Any, spans: list[dict[str, int]]) -> int:
        """Store multiple spans from ``src``.

        Args:
            src: readable buffer or GPU array spanning all source offsets.
            spans: span dicts with ``source_offset``, ``nbytes``, ``file_offset``.

        Returns:
            Total number of bytes stored.

        Async/thread-safety:
            Default implementation loops over ``store_bytes``. Backends may
            override to batch the spans into a single backend call.
        """
        total = 0
        for span in spans:
            source_offset = int(span.get("source_offset", 0))
            nbytes = int(span["nbytes"])
            file_offset = int(span["file_offset"])
            total += await self.store_bytes(
                src[source_offset : source_offset + nbytes],
                file_offset,
                nbytes,
                accounted_nbytes=int(span.get("accounted_nbytes", nbytes)),
            )
        return total

    async def load_bytes_grouped(self, dst: Any, spans: list[dict[str, int]]) -> int:
        """Load multiple spans into ``dst``.

        Args:
            dst: writable buffer or GPU array spanning all target offsets.
            spans: span dicts with ``target_offset``, ``nbytes``, ``file_offset``.

        Returns:
            Total number of bytes loaded.

        Async/thread-safety:
            Default implementation loops over ``load_bytes``. Backends may
            override to batch the spans into a single backend call.
        """
        total = 0
        view = memoryview(dst) if isinstance(dst, (bytearray, bytes)) else dst
        for span in spans:
            target_offset = int(span.get("target_offset", 0))
            nbytes = int(span["nbytes"])
            file_offset = int(span["file_offset"])
            total += await self.load_bytes(
                view[target_offset : target_offset + nbytes], file_offset, nbytes
            )
        return total

    async def prefetch_bytes_grouped(
        self,
        spans: list[dict[str, int]],
        lease_id: str | None = None,
    ) -> PrefetchResult:
        """Promote storage spans into an optional host-memory tier.

        Args:
            spans: Storage spans containing ``file_offset`` and ``nbytes``.
            lease_id: Optional request identifier whose promoted bytes must
                remain stable until a later load consumes them.

        Returns:
            Byte attribution for the request.

        Raises:
            NotImplementedError: If this backend has no host-memory tier.

        Async/thread-safety:
            Implementations run on the server event loop and must await all
            blocking I/O through their normal async backend path.
        """
        raise NotImplementedError("transfer backend does not support host prefetch")

    async def start_readahead(self, spans: list[dict[str, int]]) -> bool:
        """Begin reading a request's external spans ahead of its load.

        Args:
            spans: Storage spans containing ``file_offset``, ``nbytes`` and
                optional ``accounted_nbytes``, in load order.

        Returns:
            True when background readahead was started; the default backend
            has none and returns False.

        Async/thread-safety:
            Must return without waiting for the reads; implementations run on
            the server event loop.
        """
        return False

    async def wait_readahead_idle(self) -> None:
        """Wait until no readahead is reading.

        Async/thread-safety:
            Returns at once for backends without readahead; implementations
            run on the server event loop.
        """
        return None

    async def resident_prefix(self, groups: list[list[dict[str, int]]]) -> int:
        """Count leading span groups whose bytes are all host-resident.

        A memory-only store cannot recover evicted bytes, so the control
        plane uses this to stop a lookup at the first chunk that left memory.

        Args:
            groups: Ordered span lists (``file_offset``/``nbytes``), one per
                lookup chunk.

        Returns:
            Number of leading groups that are fully resident. Their recency
            is refreshed so they are the last candidates for replacement
            while the matching load is in flight.

        Raises:
            NotImplementedError: If this backend has no host-memory tier.

        Async/thread-safety:
            Classification and recency refresh must be atomic with respect
            to cache eviction and overlapping stores.
        """
        raise NotImplementedError("transfer backend does not support host residency")

    async def classify_and_acquire_lease(
        self,
        lease_id: str,
        spans: list[dict[str, int]],
    ) -> TransferTier:
        """Classify host-tier residency and atomically lease an all-L1 window.

        Args:
            lease_id: Request identifier used by later load and release calls.
            spans: Storage spans containing ``file_offset`` and ``nbytes``.

        Returns:
            ``l1`` when every byte is resident, ``mixed`` for partial L1
            residency, or ``l2`` when no requested byte is resident.

        Raises:
            NotImplementedError: If this backend has no host-memory tier.

        Async/thread-safety:
            Classification and all-L1 lease acquisition must be atomic with
            respect to cache eviction and overlapping stores.
        """
        raise NotImplementedError("transfer backend does not support host leases")

    async def load_leased_bytes_grouped(
        self,
        dst: Any,
        spans: list[dict[str, int]],
        lease_id: str,
    ) -> int:
        """Load spans from bytes retained by a request lease.

        Args:
            dst: Writable destination buffer.
            spans: Span dicts with target offset, file offset, and byte count.
            lease_id: Request identifier returned through lookup admission.

        Returns:
            Total number of bytes loaded.

        Async/thread-safety:
            Backends with host leases must keep the lease alive after this call;
            the IPC boundary releases ranges only after destination sync.
        """
        return await self.load_bytes_grouped(dst, spans)

    async def load_leased_runs(
        self,
        dst: Any,
        file_offsets: Any,
        target_offsets: Any,
        nbytes: Any,
        lease_id: str,
    ) -> int:
        """Copy many small runs retained by a request lease in one call.

        Args:
            dst: Flat CUDA byte destination.
            file_offsets: ``int64`` array of physical run starts.
            target_offsets: ``int64`` array of destination offsets.
            nbytes: ``int64`` array of run lengths.
            lease_id: Request lease retaining every run.

        Returns:
            Bytes copied, after the copies completed on the device.

        Raises:
            NotImplementedError: If this backend has no host-memory tier.

        Async/thread-safety:
            Runs on the owning event loop; the lease ranges are not consumed.
        """
        raise NotImplementedError("transfer backend does not support leased runs")

    async def release_lease_ranges(
        self,
        lease_id: str,
        spans: list[dict[str, int]],
    ) -> None:
        """Release successfully staged physical ranges from a request lease.

        Args:
            lease_id: Request identifier owning the retained bytes.
            spans: Physical storage ranges safe to release after staging sync.

        Returns:
            None.

        Async/thread-safety:
            Implementations must make repeated and overlapping releases
            idempotent.
        """
        return None

    async def release_lease(self, lease_id: str) -> None:
        """Release every remaining range owned by a request lease.

        Args:
            lease_id: Request identifier to clean up.

        Returns:
            None.

        Async/thread-safety:
            Implementations must make cleanup idempotent so cancellation can
            race with load completion safely.
        """
        return None

    def hold_resident(self, file_offset: int, nbytes: int) -> bool:
        """Keep a record's memory-tier copy as the only copy the server needs.

        The range must already be written (or being written) to L2. On
        success it stays in the memory tier until replacement evicts it, then
        the held-release listener is called.

        Args:
            file_offset: Start of one packed record.
            nbytes: Record length.

        Returns:
            Whether the whole record was resident and is now held; always
            False for backends without a memory tier.

        Async/thread-safety:
            Call on the server event loop. Does not await.
        """
        del file_offset, nbytes
        return False

    def set_held_release_listener(  # noqa: B027
        self, listener: Callable[[int, int], None]
    ) -> None:
        """Register the callback for held ranges leaving the memory tier.

        Args:
            listener: Called with ``(file_offset, nbytes)`` after a held range
                was evicted; a dirty range already has its L2 write ordered.

        Async/thread-safety:
            The listener runs synchronously on the server event loop inside
            transfer bookkeeping and must not await or call the transfer.
        """

    def discard_held(self, file_offset: int, nbytes: int) -> None:  # noqa: B027
        """Drop a held range without writing it to the SSD tier.

        Args:
            file_offset: Start of the held range.
            nbytes: Range length. A range no longer held is ignored.

        Async/thread-safety:
            Must be called on the server event loop before the file range is
            reused by another store. Does not await.
        """

    async def drain(self) -> None:  # noqa: B027
        """Wait for any background transfer work to complete.

        Async/thread-safety:
            Default implementation is an intentional no-op for backends with no
            background work. Backends with async write-back override this.
        """

    @abstractmethod
    def close(self) -> None:
        """Release file handles and backend resources."""
