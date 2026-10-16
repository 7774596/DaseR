# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import bisect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import time
from typing import Any

# First Party
from daser.logging import init_logger
from daser.replacement import ReplacementPolicy
from daser.replacement.bip import BIPReplacementPolicy
from daser.transfer.base import (
    LeaseIncompleteError,
    PrefetchResult,
    TransferLayer,
    TransferStats,
    TransferTier,
)
from daser.transfer.iouring import copy_ops
from daser.transfer.iouring.l1_cache import L1Cache, L1RangeHit
from daser.transfer.iouring.l2_engine import L2IoEngine
from daser.transfer.iouring.native import NativeIOUring
from daser.transfer.iouring.pinned_pool import PinnedMemorySlice

# Keep source snapshots bounded so a packed store does not monopolize the
# pinned pool while preserving the existing request-local staging budget.
# Keep one packed source allocation large enough to cover a multi-record run.
# This reduces L2 write submissions and lets capacity eviction reclaim the
# physical group as one unit; the staging pool still bounds total in-flight
# bytes and callers split oversized requests before this limit.
_PACKED_SOURCE_COPY_BYTES = 128 * 1024 * 1024


_PACKED_READ_BATCH_BYTES = 128 * 1024 * 1024
# Small sparse reads share a submission, but retain independent completion and
# promotion. Large coalesced extents continue through the splitting single-IO
# API; neither path reads the holes between immutable packed records.
_PACKED_READ_BATCH_COUNT = 64
# A readahead not consumed this long after it started lost its load (aborted
# or trimmed request) and is reclaimed; it outlives a queued request's wait.
_READAHEAD_LIFETIME_S = 120.0
#: A promotion reservation still waiting after this long logs the tier state.
_STUCK_WAIT_LOG_S = 30.0
#: Re-check interval for reservations waiting on an exhausted pool: readahead
#: pieces landing make readaheads reclaimable without notifying pool waiters.
_POOL_RETRY_S = 0.05
# Packed L1 inserts most promoted ranges at LRU and one in this many at MRU,
# so a scan larger than L1 cannot flush the reused working set.
_PACKED_L1_BIP_INTERVAL = 32

logger = init_logger(__name__)

_DIRECT_IO_ALIGNMENT = 4096


def _l1_replacement_policy(bip_enabled: bool) -> ReplacementPolicy[int] | None:
    """Select the L1 allocation replacement policy.

    Args:
        bip_enabled: Use BIP instead of the L1 cache's default LRU.

    Returns:
        The policy instance, or None for the L1 cache's default LRU.
    """
    if bip_enabled:
        return BIPReplacementPolicy[int](mru_interval=_PACKED_L1_BIP_INTERVAL)
    return None


async def _drain_destination_copies(copies: list[asyncio.Task[None]]) -> None:
    """Defer cancellation until DMA tasks have stopped using source pages."""
    # A failed copy must not make gather return while another DMA still owns
    # its source. Collect every outcome before propagating the first error.
    pending = asyncio.gather(*copies, return_exceptions=True)
    cancellation: asyncio.CancelledError | None = None
    while not pending.done():
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError as exc:
            cancellation = exc
    for result in pending.result():
        if isinstance(result, BaseException):
            raise result
    if cancellation is not None:
        raise cancellation


@dataclass(frozen=True)
class _LeasedSliceRange:
    """Map one physical byte range to bytes retained in a pinned slice."""

    file_offset: int
    nbytes: int
    data: PinnedMemorySlice
    source_offset: int


@dataclass
class _RequestLease:
    """Retain the unconsumed physical ranges for one inference request."""

    remaining: list[tuple[int, int]]
    hits: list[_LeasedSliceRange]
    slice_ids: set[int]
    released: asyncio.Future[None]
    active_loads: int = 0
    release_all: bool = False
    budget_bytes: int = 0
    # (starts, ends, host bases) of ``hits`` for vectorized run lookup;
    # rebuilt lazily and cleared whenever ``hits`` changes.
    run_index: tuple[Any, Any, Any] | None = None


@dataclass
class _Readahead:
    """Track one lookup-triggered L2 readahead retained by a request lease.

    ``pieces`` pairs the physical ranges of each bounded read group with the
    future completed once that group is attached to the lease (or failed).
    """

    ranges: list[tuple[int, int]]
    pieces: list[tuple[list[tuple[int, int]], asyncio.Future[None]]]
    started: float


@dataclass
class _Promotion:
    """One in-flight L2-to-L1 promotion holding a reserved pinned slice."""

    # L2 byte range read, so an overlapping prefetch waits for these bytes
    # instead of reading them a second time.
    key: tuple[int, int]
    # Cache mutation epoch at reservation; a newer overlapping store makes
    # the promoted bytes stale.
    epoch: int
    # Completed when the promotion ends, waking stores waiting for the pool.
    done: asyncio.Future[None]


class _ReadaheadStarved(Exception):
    """A readahead promotion found the L1 pool exhausted and gave up.

    Only readahead leases and resident pins hold the pool then, so waiting
    could deadlock against the load the readahead serves; that load reads the
    range through the normal path instead.
    """


@dataclass
class _LoadPlan:
    """Accumulate one grouped load's span resolution across its batches.

    ``copies`` holds the destination-copy tasks submitted so far; the load
    drains them before returning, also on error or cancellation.
    """

    total: int = 0
    misses: list[dict[str, int]] = field(default_factory=list)
    pending: list[asyncio.Task[None]] = field(default_factory=list)
    copies: list[asyncio.Task[None]] = field(default_factory=list)


def _stripe_extents(nbytes: int, stripe_bytes: int) -> list[tuple[int, int]]:
    """Split one aligned read into submission-sized ``(offset, size)`` parts.

    Args:
        nbytes: Length of the read in bytes.
        stripe_bytes: Target part length; parts grow beyond it when the read
            would otherwise need more than one ring submission batch.

    Returns:
        Contiguous parts covering ``[0, nbytes)``, each a multiple of the
        direct-IO alignment except possibly the last.
    """
    per_part = max(stripe_bytes, -(-nbytes // _PACKED_READ_BATCH_COUNT))
    per_part = -(-per_part // _DIRECT_IO_ALIGNMENT) * _DIRECT_IO_ALIGNMENT
    return [
        (offset, min(per_part, nbytes - offset))
        for offset in range(0, nbytes, per_part)
    ]


def _group_by_bytes(
    misses: list[dict[str, Any]], max_bytes: int
) -> list[list[dict[str, Any]]]:
    """Split ordered misses into consecutive groups of about ``max_bytes``.

    A miss longer than ``max_bytes`` is first cut into aligned fragments,
    with its capacity charge distributed over them, so one long contiguous
    extent still completes group by group.
    """
    piece = max(
        _DIRECT_IO_ALIGNMENT, max_bytes // _DIRECT_IO_ALIGNMENT * (_DIRECT_IO_ALIGNMENT)
    )
    fragments: list[dict[str, Any]] = []
    for miss in misses:
        nbytes = int(miss["nbytes"])
        if nbytes <= piece:
            fragments.append(miss)
            continue
        start = int(miss["file_offset"])
        cut = [
            {
                "target_offset": 0,
                "file_offset": start + offset,
                "nbytes": min(piece, nbytes - offset),
            }
            for offset in range(0, nbytes, piece)
        ]
        _assign_accounting_charges(cut, int(miss["accounted_nbytes"]), nbytes)
        fragments.extend(cut)
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for miss in fragments:
        nbytes = int(miss["nbytes"])
        if current and size + nbytes > max_bytes:
            groups.append(current)
            current, size = [], 0
        current.append(miss)
        size += nbytes
    if current:
        groups.append(current)
    return groups


def _ranges_overlap(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> bool:
    """Return whether any range in ``left`` intersects any range in ``right``."""
    return any(
        start < other + other_size and other < start + size
        for start, size in left
        for other, other_size in right
    )


def _resolve_from_hits(
    hits: tuple[list[int], list[_LeasedSliceRange]],
    target_offset: int,
    file_offset: int,
    nbytes: int,
) -> tuple[list[tuple[int, PinnedMemorySlice, int, int]], list[tuple[int, int, int]]]:
    """Split one load span between retained slice ranges and uncovered bytes.

    Args:
        hits: Start offsets and the matching retained ranges, both sorted by
            ``file_offset``.
        target_offset: Destination offset of the span.
        file_offset: Physical start of the span.
        nbytes: Span length in bytes.

    Returns:
        ``(chunks, uncovered)``: grouped-copy chunks served by retained ranges
        and the ``(target_offset, file_offset, nbytes)`` gaps they leave.

    Async/thread-safety:
        Pure CPU planning; call while the transfer lock is held.
    """
    starts, ranges = hits
    if not ranges:
        return [], [(target_offset, file_offset, nbytes)]
    chunks: list[tuple[int, PinnedMemorySlice, int, int]] = []
    uncovered: list[tuple[int, int, int]] = []
    end = file_offset + nbytes
    cursor = file_offset
    index = max(0, bisect.bisect_right(starts, cursor) - 1)
    while cursor < end and index < len(ranges):
        hit = ranges[index]
        index += 1
        hit_end = hit.file_offset + hit.nbytes
        if hit_end <= cursor:
            continue
        if hit.file_offset >= end:
            break
        if hit.file_offset > cursor:
            uncovered.append(
                (target_offset + cursor - file_offset, cursor, hit.file_offset - cursor)
            )
            cursor = hit.file_offset
        covered = min(end, hit_end) - cursor
        chunks.append(
            (
                target_offset + cursor - file_offset,
                hit.data,
                hit.source_offset + cursor - hit.file_offset,
                covered,
            )
        )
        cursor += covered
    if cursor < end:
        uncovered.append((target_offset + cursor - file_offset, cursor, end - cursor))
    return chunks, uncovered


def _split_runs_at_slices(
    files: Any, targets: Any, sizes: Any, starts: Any, ends: Any
) -> tuple[Any, Any, Any, Any]:
    """Split runs that cross contiguous slice edges into per-slice runs.

    Readahead pieces become L1 slices at their own boundaries, so a record
    run can span two adjacent slices of one lease.

    Args:
        files: ``int64`` physical run starts.
        targets: ``int64`` destination offsets.
        sizes: ``int64`` run lengths.
        starts: Sorted ``int64`` slice starts.
        ends: ``int64`` slice ends matching ``starts``.

    Returns:
        ``(files, targets, sizes, index)`` with every run inside slice
        ``index``.

    Raises:
        KeyError: If a run starts outside every slice or crosses a gap.
    """
    import numpy

    index = numpy.searchsorted(starts, files, side="right") - 1
    if not len(starts) or (index < 0).any() or (files >= ends[index]).any():
        raise KeyError("request lease does not cover every load run")
    if not (files + sizes > ends[index]).any():
        return files, targets, sizes, index
    out: list[tuple[int, int, int, int]] = []
    for file_offset, target, size, slot in zip(
        files.tolist(), targets.tolist(), sizes.tolist(), index.tolist(), strict=True
    ):
        while True:
            piece = min(size, int(ends[slot]) - file_offset)
            out.append((file_offset, target, piece, slot))
            size -= piece
            if size == 0:
                break
            file_offset, target, slot = file_offset + piece, target + piece, slot + 1
            if slot >= len(starts) or int(starts[slot]) != file_offset:
                raise KeyError("request lease does not cover every load run")
    split = numpy.array(out, dtype=numpy.int64).T
    return split[0], split[1], split[2], split[3]


def _normalize_ranges(spans: list[dict[str, int]]) -> list[tuple[int, int]]:
    """Return sorted, merged positive physical ranges from transfer spans."""
    ranges = sorted(
        (int(span["file_offset"]), int(span["nbytes"]))
        for span in spans
        if int(span["nbytes"]) > 0
    )
    merged: list[tuple[int, int]] = []
    for start, size in ranges:
        end = start + size
        if merged and start <= merged[-1][0] + merged[-1][1]:
            previous_start, previous_size = merged[-1]
            merged[-1] = (
                previous_start,
                max(previous_start + previous_size, end) - previous_start,
            )
        else:
            merged.append((start, size))
    return merged


def _normalize_accounted_ranges(
    spans: list[dict[str, int]],
    max_touching_charge: int,
) -> list[dict[str, int]]:
    """Return merged ranges while summing per-record capacity charges.

    Args:
        spans: Positive transfer spans with validated accounting charges.
        max_touching_charge: Largest charge a merge of merely adjacent spans
            may reach. Overlapping spans always merge; adjacent spans stay
            separate beyond this bound so raw accounting never produces a
            single range charged more than L1 holds.

    Returns:
        Sorted, merged spans whose charge covers every source record in the
        merged physical range.

    Async/thread-safety:
        Pure CPU bookkeeping; safe on the owning asyncio event loop.
    """
    normalized = [
        {
            "file_offset": int(span["file_offset"]),
            "nbytes": int(span["nbytes"]),
            "accounted_nbytes": int(span["accounted_nbytes"]),
        }
        for span in spans
        if int(span["nbytes"]) > 0
    ]
    normalized.sort(key=lambda span: span["file_offset"])
    merged: list[dict[str, int]] = []
    for span in normalized:
        if not merged:
            merged.append(span)
            continue
        previous = merged[-1]
        previous_end = previous["file_offset"] + previous["nbytes"]
        current_end = span["file_offset"] + span["nbytes"]
        charge = previous["accounted_nbytes"] + span["accounted_nbytes"]
        if span["file_offset"] < previous_end or (
            span["file_offset"] == previous_end and charge <= max_touching_charge
        ):
            previous["nbytes"] = (
                max(previous_end, current_end) - previous["file_offset"]
            )
            previous["accounted_nbytes"] = charge
            continue
        merged.append(span)
    return merged


def _assign_accounting_charges(
    misses: list[dict[str, Any]],
    accounted_nbytes: int,
    physical_nbytes: int,
) -> None:
    """Distribute one span's capacity charge across uncached fragments.

    Args:
        misses: Physical miss fragments modified in place.
        accounted_nbytes: Capacity charge for the complete requested span.
        physical_nbytes: Stored bytes in the complete requested span.

    Raises:
        ValueError: If the charge is smaller than physical bytes or the span
            geometry is invalid.

    Async/thread-safety:
        Pure CPU bookkeeping; safe on the server event loop.
    """
    if accounted_nbytes < physical_nbytes or physical_nbytes <= 0:
        raise ValueError("invalid per-entry L1 accounting geometry")
    remaining_charge = accounted_nbytes
    remaining_bytes = physical_nbytes
    for index, miss in enumerate(misses):
        miss_bytes = int(miss["nbytes"])
        if miss_bytes <= 0 or miss_bytes > remaining_bytes:
            raise ValueError("invalid L1 miss fragment")
        if index == len(misses) - 1:
            charge = remaining_charge
        else:
            charge = max(
                miss_bytes,
                (accounted_nbytes * miss_bytes + physical_nbytes - 1)
                // physical_nbytes,
            )
            charge = min(charge, remaining_charge - (remaining_bytes - miss_bytes))
        miss["accounted_nbytes"] = charge
        remaining_charge -= charge
        remaining_bytes -= miss_bytes


def _coalesce_load_misses(
    misses: list[dict[str, Any]],
    max_bytes: int,
    *,
    preserve_targets: bool,
) -> list[dict[str, Any]]:
    """Merge adjacent L2 misses into bounded physical reads.

    Packed records are variable-length but are usually close in the append
    arena. Treating every record as an independent io_uring request adds a
    submission and pinned-slice allocation per record. A merged span may carry
    a ``parts`` list when its destination offsets are disjoint; the caller
    reads the physical extent once and publishes each part independently.
    This preserves the decoder's staging layout while allowing requests from
    adjacent logical chunks to share one disk operation.

    Args:
        misses: Positive L2 miss spans in destination order.
        max_bytes: Maximum size of one promoted L1 range.
        preserve_targets: Preserve destination subranges for CUDA copies.

    Returns:
        Equivalent miss spans with contiguous ranges merged.

    Async/thread-safety:
        Pure CPU planning; safe to call while the transfer lock is held or
        from the owning asyncio event loop.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive for L2 miss coalescing")
    normalized = [
        {
            "target_offset": int(miss.get("target_offset", 0)),
            "file_offset": int(miss["file_offset"]),
            "nbytes": int(miss["nbytes"]),
            "accounted_nbytes": int(miss.get("accounted_nbytes", miss["nbytes"])),
        }
        for miss in misses
    ]
    result: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(normalized):
        run = [normalized[cursor]]
        cursor += 1
        while cursor < len(normalized):
            previous = run[-1]
            current = normalized[cursor]
            if current["file_offset"] != previous["file_offset"] + previous["nbytes"]:
                break
            run.append(current)
            cursor += 1

        # Merge even a two-record run when both physical and destination
        # ranges are adjacent. Fragmented packed loads often leave short
        # runs; keeping those separate adds allocation and IO overhead.
        run_cursor = 0
        while run_cursor < len(run):
            chunk: list[dict[str, int]] = []
            chunk_bytes = 0
            physical_bytes = 0
            while run_cursor < len(run):
                item = run[run_cursor]
                item_end = item["file_offset"] + item["nbytes"]
                chunk_start = chunk[0]["file_offset"] if chunk else item["file_offset"]
                physical_bytes = item_end - chunk_start
                if chunk and physical_bytes > max_bytes:
                    break
                chunk.append(item)
                chunk_bytes += item["nbytes"]
                run_cursor += 1
            if len(chunk) < 2:
                result.extend(chunk)
                continue
            start_file = chunk[0]["file_offset"]
            end_file = chunk[-1]["file_offset"] + chunk[-1]["nbytes"]
            if preserve_targets and any(
                item["target_offset"]
                != chunk[0]["target_offset"] + (item["file_offset"] - start_file)
                for item in chunk[1:]
            ):
                result.append(
                    {
                        "target_offset": chunk[0]["target_offset"],
                        "file_offset": start_file,
                        "nbytes": end_file - start_file,
                        "parts": [
                            {
                                "target_offset": item["target_offset"],
                                "file_offset": item["file_offset"],
                                "nbytes": item["nbytes"],
                                "accounted_nbytes": item["accounted_nbytes"],
                            }
                            for item in chunk
                        ],
                    }
                )
            else:
                result.append(
                    {
                        "target_offset": chunk[0]["target_offset"],
                        "file_offset": start_file,
                        "nbytes": end_file - start_file,
                        "accounted_nbytes": sum(
                            int(item["accounted_nbytes"]) for item in chunk
                        ),
                    }
                )
    return result


def _coalesce_load_spans(
    spans: list[dict[str, int]],
    max_bytes: int,
) -> list[dict[str, int]]:
    """Merge destination-aware adjacent spans for packed load planning.

    Args:
        spans: Load spans containing file and destination offsets.
        max_bytes: Maximum merged span size; keeps each L2 miss promotable
            into L1.

    Returns:
        Input-order spans merged only when both byte ranges are contiguous
        and the merged span stays within ``max_bytes``.

    Async/thread-safety:
        Pure CPU planning; safe while the transfer metadata lock is held.
    """
    merged: list[dict[str, int]] = []
    for span in spans:
        item = {
            "target_offset": int(span.get("target_offset", 0)),
            "file_offset": int(span["file_offset"]),
            "nbytes": int(span["nbytes"]),
            "accounted_nbytes": int(span.get("accounted_nbytes", span["nbytes"])),
        }
        if item["nbytes"] <= 0:
            continue
        if merged:
            previous = merged[-1]
            if (
                item["file_offset"] == previous["file_offset"] + previous["nbytes"]
                and item["target_offset"]
                == previous["target_offset"] + previous["nbytes"]
                and previous["nbytes"] + item["nbytes"] <= max_bytes
            ):
                previous["nbytes"] += item["nbytes"]
                previous["accounted_nbytes"] += item["accounted_nbytes"]
                continue
        merged.append(item)
    return merged


def _subtract_ranges(
    ranges: list[tuple[int, int]],
    removals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Subtract physical ranges, preserving any non-overlapping fragments."""
    result = list(ranges)
    for remove_start, remove_size in removals:
        remove_end = remove_start + remove_size
        next_result: list[tuple[int, int]] = []
        for start, size in result:
            end = start + size
            if end <= remove_start or remove_end <= start:
                next_result.append((start, size))
                continue
            if start < remove_start:
                next_result.append((start, remove_start - start))
            if remove_end < end:
                next_result.append((remove_end, end - remove_end))
        result = next_result
    return result


def _lease_retains_range(lease: _RequestLease, file_offset: int, nbytes: int) -> bool:
    """Return whether a lease's pinned slices cover its part of a byte range.

    Args:
        lease: Request lease to inspect.
        file_offset: Start of the range.
        nbytes: Length of the range.

    Returns:
        True when every unconsumed lease byte inside the range is backed by
        one of the lease's retained slices.
    """
    end = file_offset + nbytes
    uncovered = [
        (max(start, file_offset), min(start + size, end) - max(start, file_offset))
        for start, size in lease.remaining
        if start < end and file_offset < start + size
    ]
    for hit in lease.hits:
        if not uncovered:
            break
        uncovered = _subtract_ranges(uncovered, [(hit.file_offset, hit.nbytes)])
    return not uncovered


def _subtract_leased_hits(
    hits: list[_LeasedSliceRange],
    removals: list[tuple[int, int]],
) -> list[_LeasedSliceRange]:
    """Subtract consumed ranges while preserving pinned-slice source offsets."""
    result = list(hits)
    for remove_start, remove_size in removals:
        remove_end = remove_start + remove_size
        next_result: list[_LeasedSliceRange] = []
        for hit in result:
            start = hit.file_offset
            end = start + hit.nbytes
            if end <= remove_start or remove_end <= start:
                next_result.append(hit)
                continue
            if start < remove_start:
                next_result.append(
                    _LeasedSliceRange(
                        file_offset=start,
                        nbytes=remove_start - start,
                        data=hit.data,
                        source_offset=hit.source_offset,
                    )
                )
            if remove_end < end:
                next_result.append(
                    _LeasedSliceRange(
                        file_offset=remove_end,
                        nbytes=end - remove_end,
                        data=hit.data,
                        source_offset=hit.source_offset + remove_end - start,
                    )
                )
        result = next_result
    return result


class TieredIOUringTransferLayer(TransferLayer):
    """Async L1 pinned-memory + L2 SSD transfer layer.

    The implementation uses Linux io_uring for L2 positioned file I/O. GPU
    CUDA-IPC copies are added at the server IPC boundary; this layer owns the
    tiering and L2 persistence policy.

    Args:
        path: existing L2 store file of at least ``l2_bytes`` (created by the
            server); unused when ``skip_l2``.
        l1_bytes: maximum memory-tier bytes.
        l2_bytes: SSD-tier capacity.
        io_workers: number of native io_uring rings and executor threads used
            for L2 operations.
        bip_enabled: Enable BIP replacement for L1 allocations.
        l1_accounting: Capacity unit for L1 residency, either stored bytes or
            raw-equivalent bytes carried by each exact record.
        prefetch_lease_budget_bytes: Maximum accounted bytes retained by
            in-flight prefetch leases (0 disables the budget). A prefetch that
            would exceed it waits until earlier leases are released, so staged
            requests never outgrow L1 and lose their retained slices.
        l2_read_stripe_bytes: Split an L2 read larger than the packed batch
            limit into parts of about this many bytes submitted together on
            one ring, so a single request keeps many IOs in flight (0 keeps
            one serial read per extent).
        readahead_piece_bytes: Research switch. When positive, a lookup that
            hits L2 starts reading the request's external spans into L1 in
            groups of about this many bytes, ``readahead_window`` groups in
            flight, retained by an internal lease until the load consumes
            them. The SSD queue depth then no longer depends on how many
            worker staging buffers are in flight. 0 disables readahead.
        readahead_window: Read groups in flight per readahead.
        l2_write_gate: Research switch. L2 writes run on their own executor
            and pause between 16 MiB pieces while any L2 read is in native IO
            (bounded per piece), covering writes already issued when a read
            starts; see ``L2IoEngine``.
        readahead_budget_fraction: Share of L1 that readahead leases may
            retain at once (default half).

    Async/thread-safety:
        Public async methods serialize tier metadata with an asyncio lock.
        io_uring completion waits are offloaded with ``run_in_executor``.
    """

    coalesce_store_spans = True

    def __init__(
        self,
        path: str,
        l1_bytes: int,
        l2_bytes: int,
        io_workers: int = 8,
        skip_l2: bool = False,
        bip_enabled: bool = False,
        l1_accounting: str = "stored",
        prefetch_lease_budget_bytes: int = 0,
        l2_read_stripe_bytes: int = 0,
        readahead_piece_bytes: int = 0,
        readahead_window: int = 4,
        l2_write_gate: bool = False,
        readahead_budget_fraction: float = 0.5,
    ) -> None:
        if not 0.0 < readahead_budget_fraction < 1.0:
            raise ValueError("readahead_budget_fraction must be in (0, 1)")
        if readahead_piece_bytes < 0 or readahead_window <= 0:
            raise ValueError(
                "readahead_piece_bytes must be >= 0 and readahead_window > 0"
            )
        if l1_bytes <= 0:
            raise ValueError("l1_bytes must be positive")
        if not skip_l2 and l2_bytes <= 0:
            raise ValueError("l2_bytes must be positive")
        if not skip_l2 and l1_bytes > l2_bytes:
            raise ValueError("l1_bytes must not exceed l2_bytes")
        if io_workers <= 0:
            raise ValueError("io_workers must be positive")
        if l1_accounting not in ("stored", "raw"):
            raise ValueError("l1_accounting must be 'stored' or 'raw'")
        if not 0 <= prefetch_lease_budget_bytes <= l1_bytes:
            raise ValueError("prefetch_lease_budget_bytes must be in [0, l1_bytes]")
        if l2_read_stripe_bytes < 0 or l2_read_stripe_bytes % _DIRECT_IO_ALIGNMENT:
            raise ValueError(
                "l2_read_stripe_bytes must be a non-negative multiple of "
                f"{_DIRECT_IO_ALIGNMENT}"
            )
        self._l2: L2IoEngine | None = None
        if not skip_l2:
            self._l2 = L2IoEngine(path, l2_bytes, io_workers, write_gate=l2_write_gate)
        self.bip_enabled = bool(bip_enabled)
        self.l1_accounting = l1_accounting
        self._l1_bytes = l1_bytes
        self._l2_bytes = l2_bytes
        self._prefetch_lease_budget = prefetch_lease_budget_bytes
        self._l2_read_stripe = l2_read_stripe_bytes
        self._readahead_piece = 0 if skip_l2 else readahead_piece_bytes
        self._readahead_window = readahead_window
        self._readahead_budget = int(l1_bytes * readahead_budget_fraction)
        # Lease ID -> readahead; insertion order is age for budget reclaim.
        self._readaheads: dict[str, _Readahead] = {}
        self._readahead_tasks: set[asyncio.Task[None]] = set()
        self._readahead_seq = 0
        # Lease ID -> readahead reads in flight, for the contention count.
        self._readahead_reading: dict[str, int] = {}
        # Running readaheads, and an event set while there are none.
        self._readahead_running = 0
        self._readahead_idle = asyncio.Event()
        self._readahead_idle.set()
        # Latest L2 write per range, which a later write of an overlapping
        # range waits for, and the source buffer of every write still in
        # flight (superseded ones included) keyed by ``id()``.
        self._pending_l2: dict[tuple[int, int], asyncio.Task[None]] = {}
        self._l2_write_buffers: dict[int, PinnedMemorySlice] = {}
        self._held_release_listener: Callable[[int, int], None] | None = None
        # In-flight promotions by reservation identifier (``id()`` of the slice).
        self._promotions: dict[int, _Promotion] = {}
        self._cache_epoch = 0
        self._cache_mutations: list[tuple[int, int, int]] = []
        self._request_leases: dict[str, _RequestLease] = {}
        self._leased_slices: dict[int, tuple[PinnedMemorySlice, int]] = {}
        self._pending_h2d: set[asyncio.Task[None]] = set()
        # Grouped loads in flight, reported by ``debug_state``.
        self._loads_active = 0
        self._l1 = L1Cache(
            l1_bytes,
            alignment=_DIRECT_IO_ALIGNMENT,
            pinned_predicate=self._is_slice_pinned,
            replacement_policy=_l1_replacement_policy(self.bip_enabled),
            on_held_release=self._release_held,
        )
        self._l2_errors: list[BaseException] = []
        self._lock = asyncio.Lock()
        # Signalled under ``_lock`` whenever a lease is dropped.
        self._lease_released = asyncio.Condition(self._lock)
        self._stats = TransferStats()
        logger.info(
            "[TRANSFER:iouring] path=%s l1=%d l2=%d direct_io=%s "
            "io_workers=%d skip_l2=%s bip_enabled=%s l1_accounting=%s "
            "prefetch_lease_budget=%d l2_read_stripe=%d readahead_piece=%d "
            "readahead_window=%d l2_write_gate=%s",
            path,
            l1_bytes,
            l2_bytes,
            not skip_l2,
            io_workers,
            skip_l2,
            self.bip_enabled,
            self.l1_accounting,
            prefetch_lease_budget_bytes,
            l2_read_stripe_bytes,
            self._readahead_piece,
            self._readahead_window,
            l2_write_gate,
        )

    async def load_bytes(self, dst: Any, file_offset: int, nbytes: int) -> int:
        """Load bytes into a writable destination buffer.

        Args:
            dst: writable byte buffer.
            file_offset: byte offset in L2.
            nbytes: number of bytes to load.

        Returns:
            Number of bytes loaded.
        """
        self._check_range(file_offset, nbytes)
        pending: list[asyncio.Task[None]] = []
        l1_hits: list[L1RangeHit] = []
        misses: list[dict[str, int]] = []
        copy_future: asyncio.Task[None] | None = None
        async with self._lock:
            self._raise_l2_error_locked()
            l1_hits, misses = self._l1.resolve_subranges(
                target_offset=0,
                file_offset=file_offset,
                nbytes=nbytes,
            )
            if self._l2 is None and misses:
                self._stats.l1_misses += len(misses)
                raise KeyError(
                    "skip_l2 cache miss for range "
                    f"[{file_offset}, {file_offset + nbytes})"
                )
            if l1_hits:
                self._l1.record_hits(l1_hits)
                self._stats.l1_hits += 1 if self._l2 is None else len(l1_hits)
                copy_future = self._copy_grouped_to_dst(
                    dst,
                    [
                        (
                            hit.target_offset,
                            hit.data,
                            hit.source_offset,
                            hit.nbytes,
                        )
                        for hit in l1_hits
                    ],
                )
            for miss in misses:
                self._stats.l1_misses += 1
                pending.extend(
                    self._find_pending_l2_locked(
                        int(miss["file_offset"]),
                        int(miss["nbytes"]),
                    )
                )

        try:
            if pending:
                await asyncio.gather(*pending)
            if misses:
                await self._load_l2_misses_grouped(dst, misses)
        finally:
            if copy_future is not None:
                await _drain_destination_copies([copy_future])
        return nbytes

    async def load_bytes_grouped(
        self,
        dst: Any,
        spans: list[dict[str, int]],
    ) -> int:
        """Load multiple spans, coalescing L1 hits into one destination copy.

        Args:
            dst: writable byte buffer.
            spans: span dicts with target_offset, file_offset, and nbytes.

        Returns:
            Number of bytes loaded.

        Async/thread-safety:
            Uses the same metadata lock as ``load_bytes``. L2 misses still use
            native io_uring through the executor.
        """
        # Coalesce the corresponding request spans so one adjacent
        # allocation is resolved in a single metadata walk.
        spans = _coalesce_load_spans(spans, self._l1_bytes)
        plan = _LoadPlan()
        mark = time.perf_counter()
        self._loads_active += 1
        try:
            if self._readaheads:
                await self._wait_for_readaheads(spans)
            mark = self._add_phase_us("load_readahead_wait_us", mark)
            await self._resolve_load_batch(dst, spans, plan)
            mark = self._add_phase_us("load_resolve_us", mark)
            if plan.pending:
                await asyncio.gather(*plan.pending)
            mark = self._add_phase_us("load_pending_write_us", mark)
            if plan.misses:
                await self._load_l2_misses_grouped(dst, plan.misses)
            mark = self._add_phase_us("load_l2_us", mark)
        finally:
            try:
                if plan.copies:
                    await _drain_destination_copies(plan.copies)
            finally:
                self._loads_active -= 1
        self._add_phase_us("load_copy_drain_us", mark)
        return plan.total

    async def store_bytes(
        self,
        src: Any,
        file_offset: int,
        nbytes: int,
        *,
        accounted_nbytes: int | None = None,
    ) -> int:
        """Store bytes into L1 immediately and schedule L2 persistence.

        Args:
            src: readable byte buffer.
            file_offset: byte offset in L2.
            nbytes: number of bytes to store.
            accounted_nbytes: Optional raw-equivalent L1 capacity charge.

        Returns:
            Number of bytes stored.
        """
        self._check_range(file_offset, nbytes)
        accounting = (
            nbytes
            if accounted_nbytes is None
            else self._accounted_nbytes({"accounted_nbytes": accounted_nbytes}, nbytes)
        )
        key = (file_offset, nbytes)
        if self._l2 is None:
            while True:
                async with self._lock:
                    waiters = self._overlapping_lease_waiters_locked(
                        file_offset, nbytes
                    )
                    if not waiters:
                        hit = self._l1.find(file_offset)
                        if hit is not None:
                            hit_key, cached, target_offset = hit
                            if target_offset + nbytes <= len(cached):
                                self._copy_src_to_pinned_at(
                                    src, cached, target_offset, nbytes
                                )
                                self._l1.touch(hit_key)
                                self._record_cache_mutation_locked(file_offset, nbytes)
                                return nbytes
                        data = self._l1.reserve_or_raise(
                            key,
                            nbytes,
                            preserve_overlaps=True,
                            accounted_nbytes=accounting,
                        )
                        try:
                            self._copy_src_to_pinned_at(src, data, 0, nbytes)
                        except BaseException:
                            data.close()
                            raise
                        self._record_cache_mutation_locked(file_offset, nbytes)
                        self._l1.put(key, data, accounted_nbytes=accounting)
                        return nbytes
                await asyncio.gather(*waiters)

        lease_wait_since = await self._wait_for_overlapping_leases(file_offset, nbytes)
        data = await self._reserve_l1_buffer(key, nbytes, accounting)
        try:
            # The source snapshot is immutable while this copy runs. Offload
            # the potentially millisecond-scale CUDA-to-host transfer so the
            # server event loop can continue servicing lookup/load IPC.
            await asyncio.to_thread(self._copy_src_to_pinned, src, data, nbytes)
        except BaseException:
            data.close()
            raise
        try:
            while True:
                async with self._lock:
                    self._raise_l2_error_locked()
                    waiters = self._overlapping_lease_waiters_locked(
                        file_offset, nbytes, retained_ok=True
                    )
                    if not waiters:
                        self._record_cache_mutation_locked(file_offset, nbytes)
                        # Pin the buffer as a pending L2 write before publishing:
                        # an over-budget raw charge or a BIP insertion at the
                        # LRU end can evict (and close) it inside put().
                        self._persist_locked(key, data)
                        self._l1.put(key, data, accounted_nbytes=accounting)
                        # Let the newly created task submit its executor work
                        # before a synchronous shutdown can close the
                        # executor.  This yields only once; it does not wait
                        # for the SSD write and keeps the store API async.
                        break
                lease_wait_since = await self._wait_lease_waiters(
                    waiters, lease_wait_since
                )
            await asyncio.sleep(0)
            return nbytes
        except BaseException:
            async with self._lock:
                if not self._l1.contains_slice(data):
                    data.close()
                    self._l1.notify_pool_waiters()
            raise

    async def store_bytes_grouped(
        self,
        src: Any,
        spans: list[dict[str, Any]],
    ) -> int:
        """Store multiple spans through the L1/L2 tiering path.

        Args:
            src: readable byte buffer or CuPy ndarray.
            spans: span dicts with source_offset, file_offset, and nbytes.

        Returns:
            Total number of bytes stored.

        Async/thread-safety:
            In normal tiered mode this uses ``store_bytes`` for each span so L1
            visibility and asynchronous L2 scheduling keep the same ordering
            guarantees as single-span stores. In ``skip_l2`` mode it serializes
            L1 metadata once for the whole group because no pending L2 writer
            can retain evicted pool slices.
        """
        if self._l2 is None:
            return await self._store_bytes_grouped_l1_only(src, spans)

        # Online packed records are compact in the worker staging buffer even
        # when their file extents are separated by allocation envelopes.  A
        # single GPU-to-host copy for each contiguous source run removes the
        # per-record CUDA memcpy and L1-admission overhead.  Raw spans keep the
        # master one-span-at-a-time path for identical eviction semantics.
        if spans and all(bool(span.get("packed", False)) for span in spans):
            return await self._store_packed_bytes_grouped(src, spans)

        total = 0
        for span in spans:
            source_offset = int(span.get("source_offset", 0))
            nbytes = int(span["nbytes"])
            file_offset = int(span["file_offset"])
            source = self._slice_src(src, source_offset, nbytes)
            total += await self.store_bytes(
                source,
                file_offset,
                nbytes,
                accounted_nbytes=self._accounted_nbytes(span, nbytes),
            )
        return total

    async def _store_packed_bytes_grouped(
        self,
        src: Any,
        spans: list[dict[str, Any]],
    ) -> int:
        """Store packed spans with one host snapshot per contiguous source run.

        Packed records use a compact source layout but retain independent L1
        keys because their destination file ranges may cross allocation
        envelopes.  Each source-contiguous run therefore receives one
        reference-counted pinned allocation, whose child slices are published
        as independent cache entries and L2 writes.

        Args:
            src: Readable worker-owned CUDA staging buffer.
            spans: Packed spans with source and file offsets.

        Returns:
            Total packed bytes admitted to L1 and scheduled for L2.

        Async/thread-safety:
            Pool metadata and publication are serialized by the transfer lock;
            CUDA memcpy runs outside that lock on an executor thread. Caller
            cancellation waits for the actual copy before freeing its pages.
        """
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_end = -1
        current_bytes = 0
        copy_limit = min(_PACKED_SOURCE_COPY_BYTES, self._l1_bytes)
        mapped_nbytes = getattr(src, "nbytes", None)
        if mapped_nbytes is not None:
            mapped_nbytes = int(mapped_nbytes)
        for span in spans:
            source_offset = int(span.get("source_offset", 0))
            nbytes = int(span["nbytes"])
            if nbytes <= 0:
                continue
            if current and (
                source_offset != current_end
                or current_bytes + nbytes > copy_limit
                or (
                    mapped_nbytes is not None and source_offset + nbytes > mapped_nbytes
                )
            ):
                groups.append(current)
                current = []
                current_bytes = 0
            current.append(span)
            current_end = source_offset + nbytes
            current_bytes += nbytes
        if current:
            groups.append(current)

        total = 0
        for group in groups:
            total += await self._store_packed_source_group(src, group)
        return total

    async def _store_packed_source_group(
        self,
        src: Any,
        spans: list[dict[str, Any]],
        *,
        lease_wait_since: float | None = None,
    ) -> int:
        """Admit one source-contiguous packed run and schedule its L2 writes.

        ``lease_wait_since`` carries the start of a lease wait across the
        publication retry so the wait is accounted as one episode.
        """
        if not spans:
            return 0
        total = sum(int(span["nbytes"]) for span in spans)
        copy_limit = min(_PACKED_SOURCE_COPY_BYTES, self._l1_bytes)
        if total > copy_limit and len(spans) > 1:
            # Retain a defensive bound for callers that provide a grouped
            # list directly, while never splitting a logical record.
            groups: list[list[dict[str, Any]]] = []
            current: list[dict[str, Any]] = []
            current_size = 0
            for span in spans:
                nbytes = int(span["nbytes"])
                if current and current_size + nbytes > copy_limit:
                    groups.append(current)
                    current = []
                    current_size = 0
                current.append(span)
                current_size += nbytes
            if current:
                groups.append(current)
            return sum(
                [await self._store_packed_source_group(src, group) for group in groups]
            )
        source_start = int(spans[0].get("source_offset", 0))
        source_end = source_start + total
        if source_start < 0 or total <= 0:
            raise ValueError("packed source spans must be positive")
        if any(
            int(span.get("source_offset", 0)) < 0
            or int(span.get("source_offset", 0)) + int(span["nbytes"]) > source_end
            for span in spans
        ):
            raise ValueError("packed source spans are not contiguous")
        keys = [(int(span["file_offset"]), int(span["nbytes"])) for span in spans]
        accounting = [
            self._accounted_nbytes(span, int(span["nbytes"])) for span in spans
        ]

        while True:
            async with self._lock:
                self._raise_l2_error_locked()
                for file_offset, nbytes in keys:
                    self._check_range(file_offset, nbytes)
                lease_waiters = {
                    waiter
                    for file_offset, nbytes in keys
                    for waiter in self._overlapping_lease_waiters_locked(
                        file_offset, nbytes, retained_ok=True
                    )
                }
                waiters = set()
                if lease_waiters:
                    data = None
                else:
                    for file_offset, nbytes in keys:
                        self._l1.drop_overlapping(
                            file_offset,
                            nbytes,
                            preserve_remainder=True,
                        )
                    data = self._l1.reserve_untracked(total)
                    if data is None:
                        waiters = {
                            task
                            for task in self._pending_l2.values()
                            if not task.done()
                        }
                        waiters.update(
                            promotion.done
                            for promotion in self._promotions.values()
                            if not promotion.done.done()
                        )
                        if not waiters:
                            waiter = asyncio.get_event_loop().create_future()
                            self._l1.register_pool_waiter(waiter)
                            waiters = {waiter}
            if lease_waiters:
                lease_wait_since = await self._wait_lease_waiters(
                    lease_waiters, lease_wait_since
                )
                continue
            if waiters:
                await asyncio.gather(*waiters)
                continue
            if data is None:
                continue
            break

        try:
            source = self._slice_src(src, source_start, total)
            source_nbytes = int(getattr(source, "nbytes", len(source)))
            if source_nbytes < total:
                raise ValueError(
                    "packed source slice is shorter than requested copy: "
                    f"source_nbytes={source_nbytes} total={total} "
                    f"source_start={source_start}"
                )
            # Keep blocking CUDA-to-host staging out of the asyncio event loop;
            # L1 metadata publication remains serialized below this await.
            snapshot = asyncio.create_task(
                asyncio.to_thread(self._copy_src_to_pinned, source, data, total)
            )
            # Cancelling an executor await does not stop its CUDA memcpy.
            # Keep both the exported source and pinned destination alive until
            # the real copy ends, including repeated cancellation requests.
            await _drain_destination_copies([snapshot])
        except BaseException:
            # Release the reservation before logging so a diagnostic failure
            # cannot strand stores waiting for these pinned pages.
            async with self._lock:
                data.close()
                self._l1.notify_pool_waiters()
            logger.warning(
                "[TRANSFER] packed CUDA-to-pinned copy failed: "
                "source_start=%d total=%d spans=%d",
                source_start,
                total,
                len(spans),
                exc_info=True,
            )
            raise

        children: list[tuple[tuple[int, int], PinnedMemorySlice]] = []
        cursor = 0
        try:
            async with self._lock:
                self._raise_l2_error_locked()
                waiters = {
                    waiter
                    for file_offset, nbytes in keys
                    for waiter in self._overlapping_lease_waiters_locked(
                        file_offset, nbytes, retained_ok=True
                    )
                }
                if waiters:
                    # A lease may have arrived while the snapshot memcpy was
                    # in flight. Retry publication after releasing the group.
                    pass
                else:
                    for file_offset, nbytes in keys:
                        self._l1.drop_overlapping(
                            file_offset,
                            nbytes,
                            preserve_remainder=True,
                        )
                    for key in keys:
                        child = data.subslice(cursor, key[1])
                        children.append((key, child))
                        cursor += key[1]
                    # Pin every child as pending-write before publishing:
                    # when accounted charges exceed physical bytes,
                    # publication can evict this group's own allocation,
                    # and an unpinned child would be closed before its
                    # bytes reach L2.
                    for key, child in children:
                        self._persist_locked(key, child)
                    self._l1.put_reserved_group(children, accounted_nbytes=accounting)
                    for key, _child in children:
                        self._record_cache_mutation_locked(*key)
                    data.close()
                    # Ensure every scheduled write has entered the executor
                    # before callers can close the layer synchronously.
                    await asyncio.sleep(0)
                    return total
            data.close()
            lease_wait_since = await self._wait_lease_waiters(waiters, lease_wait_since)
        except BaseException:
            data.close()
            for _key, child in children:
                child.close()
            raise
        return await self._store_packed_source_group(
            src, spans, lease_wait_since=lease_wait_since
        )

    async def prefetch_bytes_grouped(
        self,
        spans: list[dict[str, int]],
        lease_id: str | None = None,
    ) -> PrefetchResult:
        """Promote L2-missing portions of spans into the pinned L1 tier.

        Args:
            spans: Aligned storage spans containing ``file_offset`` and
                ``nbytes``.
            lease_id: Optional request identifier that blocks overlapping
                stores and retains promoted slices for the later GPU load.

        Returns:
            Requested bytes split between existing L1 data and L2 reads.

        Raises:
            NotImplementedError: If the L2 tier is disabled.

        Async/thread-safety:
            Metadata is protected by the transfer lock; io_uring reads are
            awaited through the existing executor-backed read path.
        """
        if self._l2 is None:
            raise NotImplementedError("prefetch requires the io_uring L2 tier")

        accounted_spans = [
            {
                "file_offset": int(span["file_offset"]),
                "nbytes": int(span["nbytes"]),
                "accounted_nbytes": self._accounted_nbytes(span, int(span["nbytes"])),
            }
            for span in spans
            if int(span["nbytes"]) > 0
        ]
        normalized_spans = _normalize_accounted_ranges(accounted_spans, self._l1_bytes)
        requested_ranges = _normalize_ranges(accounted_spans)
        requested_bytes = sum(size for _start, size in requested_ranges)
        if lease_id is not None and requested_bytes > self._l1_bytes:
            raise MemoryError(
                f"request lease needs {requested_bytes} bytes but L1 capacity is "
                f"{self._l1_bytes}"
            )
        budget = self._prefetch_lease_budget if lease_id is not None else 0
        charge = sum(span["accounted_nbytes"] for span in normalized_spans)
        if budget and charge > budget:
            raise MemoryError(
                f"prefetch lease needs {charge} accounted bytes but the lease "
                f"budget is {budget}"
            )
        l1_bytes = 0
        misses: list[dict[str, int]] = []
        pending: list[asyncio.Task[None]] = []
        try:
            async with self._lock:
                if lease_id is not None:
                    # Promotions reserve L1 one read batch at a time, so two
                    # leases whose ranges exceed L1 together could each hold
                    # part of the pool and wait for the other forever. Admit a
                    # lease only once its physical bytes fit beside the others.
                    await self._lease_released.wait_for(
                        lambda: (
                            self._leased_physical_bytes_locked(lease_id)
                            + requested_bytes
                            <= self._l1_bytes
                            and (
                                not budget
                                or self._leased_budget_bytes_locked(lease_id) + charge
                                <= budget
                            )
                        )
                    )
                self._raise_l2_error_locked()
                if lease_id is not None:
                    self._replace_request_lease_locked(lease_id, requested_ranges, [])
                    self._request_leases[lease_id].budget_bytes = (
                        charge if budget else 0
                    )
                for span in normalized_spans:
                    self._check_range(span["file_offset"], span["nbytes"])
            # Resolve against L1; bytes another request is already promoting
            # are awaited and resolved again, so concurrent prefetches of one
            # document read it from L2 once and share the promoted slice.
            unresolved: list[dict[str, int]] = normalized_spans
            while unresolved:
                promotions: list[asyncio.Future[None]] = []
                deferred: list[dict[str, int]] = []
                async with self._lock:
                    self._raise_l2_error_locked()
                    for span in unresolved:
                        file_offset = span["file_offset"]
                        nbytes = span["nbytes"]
                        l1_hits, span_misses = self._l1.resolve_subranges(
                            target_offset=0,
                            file_offset=file_offset,
                            nbytes=nbytes,
                        )
                        if l1_hits:
                            self._l1.record_hits(l1_hits)
                            l1_bytes += sum(hit.nbytes for hit in l1_hits)
                            if lease_id is not None:
                                self._attach_l1_hits_to_lease_locked(lease_id, l1_hits)
                        if span_misses:
                            _assign_accounting_charges(
                                span_misses,
                                span["accounted_nbytes"],
                                nbytes,
                            )
                        for miss in span_misses:
                            entry = {
                                "file_offset": int(miss["file_offset"]),
                                "nbytes": int(miss["nbytes"]),
                                "accounted_nbytes": int(miss["accounted_nbytes"]),
                            }
                            in_flight = self._find_pending_promotions_locked(
                                entry["file_offset"], entry["nbytes"]
                            )
                            if in_flight:
                                promotions.extend(in_flight)
                                deferred.append(entry)
                                continue
                            pending.extend(
                                self._find_pending_l2_locked(
                                    entry["file_offset"], entry["nbytes"]
                                )
                            )
                            misses.append({"target_offset": 0, **entry})
                if promotions:
                    await asyncio.gather(*set(promotions))
                unresolved = deferred

            if pending:
                await asyncio.gather(*set(pending))
            l2_bytes = sum(int(miss["nbytes"]) for miss in misses)
            if misses:
                await self._load_l2_misses_grouped(None, misses, lease_id=lease_id)
            async with self._lock:
                if lease_id is not None:
                    self._require_complete_lease_locked(lease_id)
                self._stats.prefetch_requests += 1
                self._stats.prefetch_l1_bytes += l1_bytes
                self._stats.prefetch_l2_bytes += l2_bytes
            return PrefetchResult(requested_bytes, l1_bytes, l2_bytes)
        except BaseException:
            if lease_id is not None:
                await self.release_lease(lease_id)
            raise

    async def wait_readahead_idle(self) -> None:
        """Wait until every started readahead has finished reading.

        Async/thread-safety:
            Runs on the server event loop; a readahead started while waiting
            extends the wait.
        """
        await self._readahead_idle.wait()

    async def start_readahead(self, spans: list[dict[str, int]]) -> bool:
        """Start reading a looked-up request's L2 spans into a retained lease.

        Resident bytes are retained at once; L2 misses are grouped into reads
        of about ``readahead_piece_bytes`` with ``readahead_window`` groups in
        flight. A later ``load_bytes_grouped`` that overlaps a group waits for
        it and copies the retained bytes, falling back to the normal L1/L2
        path for anything the readahead did not cover.

        Args:
            spans: Storage spans with ``file_offset``, ``nbytes`` and optional
                ``accounted_nbytes``.

        Returns:
            True when a new readahead was registered. False when readahead is
            disabled, every span is already being read ahead, or not even the
            first new span fits the readahead budget (``readahead_budget_
            fraction`` of L1, less the bytes earlier readaheads still retain).
            A request larger than the
            remaining budget reads ahead its leading spans. Readaheads are
            reclaimed only once older than their lifetime, so a queued
            request's readahead survives later lookups.

        Async/thread-safety:
            Registers under the transfer lock and returns without waiting for
            reads. Readahead leases never block stores: a store overlapping
            one drops it (see ``_overlapping_lease_waiters_locked``).
        """
        if not self._readahead_piece or self._l2 is None:
            return False
        accounted = [
            {
                "file_offset": int(span["file_offset"]),
                "nbytes": int(span["nbytes"]),
                "accounted_nbytes": self._accounted_nbytes(span, int(span["nbytes"])),
            }
            for span in spans
            if int(span["nbytes"]) > 0
        ]
        if not accounted:
            return False
        async with self._lock:
            self._raise_l2_error_locked()
            self._prune_readaheads_locked()
            # Readaheads past their lifetime lost their load (an aborted or
            # trimmed request); live ones may belong to a request that is
            # loading or still queued, so they keep their budget.
            now = time.perf_counter()
            for lease_id in [
                lease_id
                for lease_id, item in self._readaheads.items()
                if now - item.started > _READAHEAD_LIFETIME_S
            ]:
                self._drop_readahead_locked(lease_id, "expired")
            covered = [rng for item in self._readaheads.values() for rng in item.ranges]
            budget = self._readahead_budget - self._readahead_bytes_locked()
            # Spans arrive in load order: read ahead the leading spans that
            # are not read ahead already and fit the remaining budget; the
            # tail goes to the normal load path.
            kept: list[dict[str, int]] = []
            for span in accounted:
                if not _subtract_ranges(
                    [(span["file_offset"], span["nbytes"])], covered
                ):
                    continue
                if span["nbytes"] > budget:
                    break
                kept.append(span)
                budget -= span["nbytes"]
            accounted = kept
            ranges = _normalize_ranges(accounted)
            if not ranges:
                return False
            self._readahead_seq += 1
            lease_id = f"readahead-{self._readahead_seq}"
            self._replace_request_lease_locked(lease_id, ranges, [])
            misses: list[dict[str, Any]] = []
            for span in _normalize_accounted_ranges(accounted, self._l1_bytes):
                self._check_range(span["file_offset"], span["nbytes"])
                hits, span_misses = self._l1.resolve_subranges(
                    target_offset=0,
                    file_offset=span["file_offset"],
                    nbytes=span["nbytes"],
                )
                if hits:
                    self._l1.record_hits(hits)
                    self._attach_l1_hits_to_lease_locked(lease_id, hits)
                if span_misses:
                    _assign_accounting_charges(
                        span_misses, span["accounted_nbytes"], span["nbytes"]
                    )
                for miss in span_misses:
                    offset, nbytes = int(miss["file_offset"]), int(miss["nbytes"])
                    # Bytes another reader or writer is producing stay with
                    # the normal load path instead of being read twice.
                    if self._find_pending_promotions_locked(
                        offset, nbytes
                    ) or self._find_pending_l2_locked(offset, nbytes):
                        continue
                    misses.append(
                        {
                            "target_offset": 0,
                            "file_offset": offset,
                            "nbytes": nbytes,
                            "accounted_nbytes": int(miss["accounted_nbytes"]),
                        }
                    )
            groups = _group_by_bytes(
                _coalesce_load_misses(
                    misses, self._readahead_piece, preserve_targets=False
                ),
                self._readahead_piece,
            )
            loop = asyncio.get_running_loop()
            readahead = _Readahead(
                ranges=ranges,
                pieces=[
                    (_normalize_ranges(group), loop.create_future()) for group in groups
                ],
                started=time.perf_counter(),
            )
            self._readaheads[lease_id] = readahead
        if groups:
            # Count the readahead as reading from registration, so an idle
            # wait right after start_readahead covers it.
            self._readahead_running += 1
            self._readahead_idle.clear()
            task = asyncio.create_task(self._run_readahead(lease_id, readahead, groups))
            self._readahead_tasks.add(task)
            task.add_done_callback(self._readahead_task_done)
        return True

    async def resident_prefix(self, groups: list[list[dict[str, int]]]) -> int:
        """Count leading span groups fully resident in L1 and refresh them.

        Args:
            groups: Ordered span lists, one per lookup chunk.

        Returns:
            Number of leading groups whose every byte is an L1 hit.

        Async/thread-safety:
            Runs under the transfer metadata lock, so a group counted here
            is resident at return time; later stores may still evict it.
        """
        async with self._lock:
            resident = 0
            for group in groups:
                hits: list[L1RangeHit] = []
                for file_offset, nbytes in _normalize_ranges(group):
                    self._check_range(file_offset, nbytes)
                    span_hits, misses = self._l1.resolve_subranges(
                        target_offset=0,
                        file_offset=file_offset,
                        nbytes=nbytes,
                    )
                    if misses:
                        return resident
                    hits.extend(span_hits)
                self._l1.record_hits(hits)
                resident += 1
            return resident

    async def classify_and_acquire_lease(
        self,
        lease_id: str,
        spans: list[dict[str, int]],
    ) -> TransferTier:
        """Classify exact spans and atomically retain an all-L1 request window."""
        requested_ranges = _normalize_ranges(spans)
        if not requested_ranges:
            return "l2"
        async with self._lock:
            self._raise_l2_error_locked()
            existing = self._request_leases.get(lease_id)
            if existing is not None:
                if existing.remaining == requested_ranges:
                    return "l1"
                self._release_request_lease_locked(lease_id)

            hits: list[L1RangeHit] = []
            l1_bytes = 0
            requested_bytes = 0
            has_miss = False
            for file_offset, nbytes in requested_ranges:
                self._check_range(file_offset, nbytes)
                requested_bytes += nbytes
                span_hits, misses = self._l1.resolve_subranges(
                    target_offset=0,
                    file_offset=file_offset,
                    nbytes=nbytes,
                )
                hits.extend(span_hits)
                l1_bytes += sum(hit.nbytes for hit in span_hits)
                has_miss = has_miss or bool(misses)
            if not has_miss:
                self._l1.record_hits(hits)
                self._replace_request_lease_locked(lease_id, requested_ranges, hits)
                return "l1"
            return "mixed" if l1_bytes else "l2"

    async def load_leased_bytes_grouped(
        self,
        dst: Any,
        spans: list[dict[str, int]],
        lease_id: str,
    ) -> int:
        """Copy request-leased L1 bytes without re-resolving cache metadata."""
        total = 0
        chunks: list[tuple[int, PinnedMemorySlice, int, int]] = []
        copy_future: asyncio.Task[None] | None = None
        async with self._lock:
            self._raise_l2_error_locked()
            lease = self._request_leases.get(lease_id)
            if lease is None:
                raise KeyError(f"unknown transfer lease: {lease_id}")
            lease.active_loads += 1
            try:
                for span in spans:
                    target_offset = int(span.get("target_offset", 0))
                    file_offset = int(span["file_offset"])
                    nbytes = int(span["nbytes"])
                    self._check_range(file_offset, nbytes)
                    total += nbytes
                    chunks.extend(
                        self._resolve_leased_range_locked(
                            lease,
                            target_offset,
                            file_offset,
                            nbytes,
                        )
                    )
                if chunks:
                    self._stats.l1_hits += len(chunks)
                    copy_future = self._copy_grouped_to_dst(dst, chunks)
            except BaseException:
                lease.active_loads -= 1
                if lease.release_all and lease.active_loads == 0:
                    self._release_request_lease_locked(lease_id, force=True)
                raise
        if copy_future is not None:
            await _drain_destination_copies([copy_future])
        return total

    async def load_leased_runs(
        self,
        dst: Any,
        file_offsets: Any,
        target_offsets: Any,
        nbytes: Any,
        lease_id: str,
    ) -> int:
        """Copy many leased L1 runs with one vectorized resolution.

        Layer-wise delivery copies one plane group of every record of a
        request per call, i.e. thousands of small runs. Each run must lie in
        L1 slices retained by the lease; runs are mapped to their slices with
        one sorted search instead of per-span metadata walks, and a run that
        crosses into the adjacent retained slice is copied in two pieces.

        Args:
            dst: Flat CUDA byte destination (optionally with
                ``copy_stream_ptr``).
            file_offsets: ``int64`` array of physical run starts.
            target_offsets: ``int64`` array of destination offsets.
            nbytes: ``int64`` array of run lengths.
            lease_id: Request lease retaining every run. The copies run on
                the copy engines, so they do not compete with a concurrent
                forward pass for SMs.

        Returns:
            Bytes copied, after the copies completed on the device.

        Raises:
            KeyError: If the lease is unknown or a run is not inside
                contiguous retained slices.
            ValueError: If the arrays are empty, unequal or out of bounds.

        Async/thread-safety:
            Runs on the owning event loop. The lease is pinned for the whole
            call, so its slices cannot be released while copies are in flight;
            the lease ranges are not consumed.
        """
        import numpy

        files = numpy.asarray(file_offsets, dtype=numpy.int64)
        sizes = numpy.broadcast_to(
            numpy.asarray(nbytes, dtype=numpy.int64), files.shape
        )
        targets = numpy.asarray(target_offsets, dtype=numpy.int64)
        async with self._lock:
            self._raise_l2_error_locked()
            lease = self._request_leases.get(lease_id)
            if lease is None:
                raise KeyError(f"unknown transfer lease: {lease_id}")
            if lease.run_index is None:
                hits = sorted(lease.hits, key=lambda hit: hit.file_offset)
                starts = numpy.array(
                    [hit.file_offset for hit in hits], dtype=numpy.int64
                )
                lease.run_index = (
                    starts,
                    starts
                    + numpy.array([hit.nbytes for hit in hits], dtype=numpy.int64),
                    numpy.array(
                        [hit.data.ptr_at(hit.source_offset) for hit in hits],
                        dtype=numpy.uint64,
                    ),
                )
            starts, ends, bases = lease.run_index
            files, targets, sizes, index = _split_runs_at_slices(
                files, targets, sizes, starts, ends
            )
            sources = bases[index] + (files - starts[index]).astype(numpy.uint64)
            # Pin the lease before enqueueing: a partial submission still
            # reads its slices, and the finally below releases the pin only
            # after the device event completes.
            lease.active_loads += 1
            try:
                event = copy_ops.copy_pointer_runs_to_cuda_dst(
                    dst,
                    targets,
                    sources,
                    sizes,
                )
            except BaseException:
                lease.active_loads -= 1
                raise
            self._stats.l1_hits += len(files)
        try:
            while not event.done:
                await asyncio.sleep(0.0005)
        finally:
            async with self._lock:
                lease.active_loads -= 1
                if lease.release_all and lease.active_loads == 0:
                    self._release_request_lease_locked(lease_id, force=True)
        return int(sizes.sum())

    async def release_lease_ranges(
        self,
        lease_id: str,
        spans: list[dict[str, int]],
    ) -> None:
        """Release staged physical ranges after the IPC destination is synced."""
        removals = _normalize_ranges(spans)
        async with self._lock:
            lease = self._request_leases.get(lease_id)
            if lease is None:
                return
            if lease.active_loads > 0:
                lease.active_loads -= 1
            if lease.release_all:
                if lease.active_loads == 0:
                    self._release_request_lease_locked(lease_id, force=True)
                return
            self._consume_lease_ranges_locked(lease_id, removals)

    async def release_lease(self, lease_id: str) -> None:
        """Idempotently release every remaining range for one request."""
        async with self._lock:
            self._release_request_lease_locked(lease_id)

    def hold_resident(self, file_offset: int, nbytes: int) -> bool:
        """Mark a record's resident L1 range held; see ``TransferLayer``.

        Async/thread-safety:
            Must run on the owning event loop. It does not await, so it is
            atomic with respect to locked sections; a split keeps every byte
            resident and its owners unchanged.
        """
        return self._l1.hold_clean((file_offset, nbytes))

    def set_held_release_listener(self, listener: Callable[[int, int], None]) -> None:
        """Register the held-release callback; see ``TransferLayer``."""
        self._held_release_listener = listener

    def release_hold(self, file_offset: int, nbytes: int) -> None:
        """Keep a held L1 range as an ordinary entry; see ``TransferLayer``.

        Async/thread-safety:
            Must run on the owning event loop. It does not await, so it is
            atomic with respect to locked sections.
        """
        self._l1.release_hold((file_offset, nbytes))

    def discard_held(self, file_offset: int, nbytes: int) -> None:
        """Drop a held L1 range.

        Async/thread-safety:
            Must run on the owning event loop before the file range is reused.
            It does not await, so it is atomic with respect to locked sections.
        """
        self._l1.discard_held((file_offset, nbytes))

    async def drain(self) -> None:
        """Wait until all pending L2 writes and destination copies complete.

        Async/thread-safety:
            Must be called from the owning asyncio event loop before shutdown
            when durable L2 contents are required.
        """
        while True:
            async with self._lock:
                pending = (
                    list(self._pending_l2.values())
                    + list(self._pending_h2d)
                    + list(self._readahead_tasks)
                )
                self._raise_l2_error_locked()
            if not pending:
                return
            await _drain_destination_copies(pending)

    @property
    def stats(self) -> TransferStats:
        """Return tiering counters, refreshing the L1 eviction totals.

        Returns:
            The live counters object; eviction fields mirror ``L1Cache``.
        """
        evicted = self._l1.eviction_bytes
        self._stats.l1_evicted_bytes = evicted["bytes"]
        self._stats.l1_evicted_pinned_bytes = evicted["pinned_bytes"]
        self._stats.l1_evicted_unread_bytes = evicted["unread_bytes"]
        return self._stats

    def debug_state(self) -> dict[str, Any]:
        """Return in-flight loads, leases, readaheads and writes.

        Returns:
            Counts plus per-lease ``(remaining bytes, active loads)``.

        Async/thread-safety:
            Read from a diagnostic thread without the transfer lock; dicts are
            copied first, so values are best-effort.
        """
        leases = dict(self._request_leases)
        return {
            "loads_active": self._loads_active,
            "leases": {
                lease_id: (
                    sum(size for _start, size in lease.remaining),
                    lease.active_loads,
                )
                for lease_id, lease in leases.items()
            },
            "readaheads": len(dict(self._readaheads)),
            "readahead_running": self._readahead_running,
            "pending_l2_writes": sum(
                not task.done() for task in list(self._pending_l2.values())
            ),
            "pending_l1_promotions": len(dict(self._promotions)),
            "l1_bytes_used": self._l1.bytes_used,
        }

    def close(self) -> None:
        """Close the L2 file handle after pending writes are drained/cancelled."""
        if self._pending_h2d:
            raise RuntimeError("drain destination copies before closing the transfer")
        for lease in self._request_leases.values():
            if not lease.released.done():
                lease.released.set_result(None)
        self._request_leases.clear()
        self._readaheads.clear()
        self._leased_slices.clear()
        # Cancellation only stops the asyncio bookkeeping task; the blocking
        # executor operation may still be reading its pinned source buffer.
        # Keep every pending buffer alive until ``L2IoEngine.close`` joins the
        # executor, otherwise an asynchronous store can persist bytes from a
        # recycled/cleared slice.  This path is shutdown-only and may block.
        pending_buffers = list(self._l2_write_buffers.values())
        for task in self._pending_l2.values():
            task.cancel()
        self._pending_l2.clear()
        self._l2_write_buffers.clear()
        if self._l2 is not None:
            self._l2.close()
        for pending_buffer in pending_buffers:
            if not self._l1.contains_slice(pending_buffer):
                pending_buffer.close()
        self._l1.close()

    def _is_slice_pinned(self, key: tuple[int, int], data: PinnedMemorySlice) -> bool:
        """Return whether an L2 writer or request lease owns ``data``."""
        del key
        return id(data) in self._l2_write_buffers or id(data) in self._leased_slices

    def _release_held(self, key: tuple[int, int]) -> None:
        """Hand a held range leaving L1 back to the server."""
        if self._held_release_listener is not None:
            self._held_release_listener(*key)

    def _persist_locked(
        self, key: tuple[int, int], data: PinnedMemorySlice
    ) -> asyncio.Task[None]:
        """Schedule one L2 write for an L1 range and pin its buffer.

        The write waits for earlier pending writes of overlapping ranges and
        owns ``data`` until it ends, also after a newer write of the same
        range replaces it in ``_pending_l2``.

        Returns:
            The write task.
        """
        previous = self._find_pending_l2_locked(*key)
        task = self._schedule_l2_write_locked(key, key[0], data, previous)
        self._pending_l2[key] = task
        self._l2_write_buffers[id(data)] = data
        return task

    @property
    def l2_timing_us(self) -> dict[str, int]:
        """Return L2 executor queue and native IO microseconds by kind.

        Returns:
            ``read_queue``/``read_io``/``write_queue``/``write_io`` totals,
            empty when L2 is disabled.
        """
        return {} if self._l2 is None else self._l2.timing_us

    @property
    def l2_read_contention(self) -> dict[tuple[int, bool], tuple[int, int, int]]:
        """Return L2 read jobs grouped by SSD concurrency at job start.

        Returns:
            ``{(read concurrency bucket, write in flight): (jobs, io_us,
            bytes)}`` totals, empty when L2 is disabled.
        """
        return {} if self._l2 is None else self._l2.read_contention

    @property
    def l1_bytes_used(self) -> int:
        """Return resident L1 memory-tier bytes."""
        return self._l1.bytes_used

    def _accounted_nbytes(self, span: dict[str, Any], nbytes: int) -> int:
        """Return the configured L1 capacity charge for one physical span.

        Args:
            span: Transfer span with optional raw-equivalent accounting bytes.
            nbytes: Physical stored bytes represented by the span.

        Returns:
            Stored bytes in normal mode, or the exact raw-equivalent charge in
            ``raw`` mode.

        Raises:
            ValueError: If the span reports an invalid accounting charge.
        """
        if nbytes <= 0:
            raise ValueError("transfer spans must have positive byte lengths")
        charge = (
            nbytes
            if self.l1_accounting == "stored"
            else int(span.get("accounted_nbytes", nbytes))
        )
        if charge < nbytes:
            raise ValueError(
                f"L1 accounting charge {charge} is smaller than stored bytes {nbytes}"
            )
        return charge

    def _check_range(self, file_offset: int, nbytes: int) -> None:
        """Validate an L2 byte range.

        Args:
            file_offset: byte offset in L2.
            nbytes: byte count.

        Raises:
            ValueError: when the range is invalid or exceeds L2.
        """
        if file_offset < 0 or nbytes < 0:
            raise ValueError("file_offset and nbytes must be non-negative")
        if self._l2 is None:
            if nbytes > self._l1_bytes:
                raise ValueError(
                    f"range {nbytes} bytes exceeds L1 capacity {self._l1_bytes}"
                )
            return
        if file_offset + nbytes > self._l2_bytes:
            raise ValueError(
                f"range [{file_offset}, {file_offset + nbytes}) exceeds "
                f"L2 capacity {self._l2_bytes}"
            )
        if (
            file_offset % _DIRECT_IO_ALIGNMENT != 0
            or nbytes % _DIRECT_IO_ALIGNMENT != 0
        ):
            raise ValueError(
                "O_DIRECT io_uring ranges must be aligned to "
                f"{_DIRECT_IO_ALIGNMENT} bytes: offset={file_offset} nbytes={nbytes}"
            )

    def _find_pending_l2_locked(
        self,
        file_offset: int,
        nbytes: int,
    ) -> list[asyncio.Task[None]]:
        """Return pending L2 writes overlapping the requested byte span."""
        end = file_offset + nbytes
        return [
            task
            for key, task in self._pending_l2.items()
            if key[0] < end and file_offset < key[0] + key[1]
        ]

    def _find_pending_promotions_locked(
        self,
        file_offset: int,
        nbytes: int,
    ) -> list[asyncio.Future[None]]:
        """Return waiters of in-flight L2 promotions overlapping a span."""
        end = file_offset + nbytes
        return [
            promotion.done
            for promotion in self._promotions.values()
            if promotion.key[0] < end
            and file_offset < promotion.key[0] + promotion.key[1]
            and not promotion.done.done()
        ]

    def _read_l2_into(
        self,
        file_offset: int,
        dst: PinnedMemorySlice,
        uring: NativeIOUring,
    ) -> int:
        """Blocking io_uring L2 read into pinned memory."""
        if self._l2 is None:
            raise RuntimeError("L2 reads are disabled when skip_l2 is true")
        return self._l2.read_into(file_offset, dst.view(), uring)

    def _write_l2(
        self,
        file_offset: int,
        data: PinnedMemorySlice,
        uring: NativeIOUring,
    ) -> None:
        """Blocking io_uring L2 write."""
        if self._l2 is None:
            raise RuntimeError("L2 writes are disabled when skip_l2 is true")
        written = self._l2.write(file_offset, data.view(), uring)
        if written != len(data):
            raise IOError(f"short io_uring write: {written} != {len(data)}")

    def _schedule_l2_write_locked(
        self,
        key: tuple[int, int],
        file_offset: int,
        data: PinnedMemorySlice,
        previous: list[asyncio.Task[None]],
    ) -> asyncio.Task[None]:
        """Schedule one L2 write and start independent IO immediately."""
        if self._l2 is None:
            raise RuntimeError("L2 writes are disabled when skip_l2 is true")
        if previous:
            return asyncio.create_task(
                self._write_l2_async(key, file_offset, data, previous)
            )

        loop = asyncio.get_event_loop()
        future = asyncio.create_task(self._run_l2_write(loop, file_offset, data))
        return asyncio.create_task(self._track_l2_write(key, data, future))

    async def _track_l2_write(
        self,
        key: tuple[int, int],
        data: PinnedMemorySlice,
        future: asyncio.Future[None],
    ) -> None:
        """Publish completion for an already-submitted L2 write."""
        current = asyncio.current_task()
        try:
            await future
            async with self._lock:
                self._stats.l2_writes += 1
                self._stats.l2_write_bytes += key[1]
        except BaseException as exc:
            async with self._lock:
                self._l2_errors.append(exc)
            raise
        finally:
            async with self._lock:
                self._finish_l2_write_locked(key, data, current)

    async def _write_l2_async(
        self,
        key: tuple[int, int],
        file_offset: int,
        data: PinnedMemorySlice,
        previous: list[asyncio.Task[None]],
    ) -> None:
        """Persist one L2 write and publish completion to waiters."""
        current = asyncio.current_task()
        try:
            if previous:
                await asyncio.gather(*previous)
            loop = asyncio.get_event_loop()
            if self._l2 is None:
                raise RuntimeError("L2 writes are disabled when skip_l2 is true")
            await self._run_l2_write(loop, file_offset, data)
            async with self._lock:
                self._stats.l2_writes += 1
                self._stats.l2_write_bytes += key[1]
        except BaseException as exc:
            async with self._lock:
                self._l2_errors.append(exc)
            raise
        finally:
            async with self._lock:
                self._finish_l2_write_locked(key, data, current)

    def _finish_l2_write_locked(
        self,
        key: tuple[int, int],
        data: PinnedMemorySlice,
        task: asyncio.Task[None] | None,
    ) -> None:
        """Forget one ended L2 write and free its buffer if nothing owns it."""
        if self._pending_l2.get(key) is task:
            del self._pending_l2[key]
        self._l2_write_buffers.pop(id(data), None)
        self._close_unowned_slice_locked(data)

    async def _run_l2_write(
        self,
        loop: asyncio.AbstractEventLoop,
        file_offset: int,
        data: PinnedMemorySlice,
    ) -> None:
        """Execute one blocking SSD write on the L2 executor.

        Args:
            loop: Owning asyncio loop used to submit executor work.
            file_offset: Absolute byte offset in the L2 store.
            data: Pinned host slice containing the immutable payload.

        Async/thread-safety:
            Metadata locks are never held across this await.
        """
        if self._l2 is None:
            raise RuntimeError("L2 writes are disabled when skip_l2 is true")
        if not self._readahead_idle.is_set():
            self._stats.l2_write_contended_bytes += len(data)
        await loop.run_in_executor(
            self._l2.write_executor,
            self._l2.timed("write", self._write_l2),
            file_offset,
            data,
            self._next_uring(),
        )

    def _raise_l2_error_locked(self) -> None:
        """Raise and clear the first asynchronous L2 write failure."""
        if not self._l2_errors:
            return
        error = self._l2_errors.pop(0)
        raise RuntimeError("asynchronous io_uring L2 write failed") from error

    async def _load_l2_misses_grouped(
        self,
        dst: Any | None,
        misses: list[dict[str, Any]],
        lease_id: str | None = None,
    ) -> None:
        """Read grouped L2 misses concurrently, then promote in request order."""
        # Online packed records are variable-length, but adjacent records are
        # laid out as one aligned physical range.  Merge those misses before
        # admission so a long prefix uses one io_uring read and one pinned L1
        # slice instead of one operation per record.  The destination-aware
        # check keeps decoder staging offsets correct; prefetch has no target
        # buffer and can merge on physical adjacency alone.
        misses = _coalesce_load_misses(
            misses,
            self._l1_bytes,
            preserve_targets=dst is not None,
        )
        start = 0
        while start < len(misses):
            batch = self._next_l2_miss_batch(misses, start)
            await self._load_l2_miss_batch(dst, batch, lease_id=lease_id)
            start += len(batch)

    async def _load_l2_miss_batch(
        self,
        dst: Any | None,
        misses: list[dict[str, int]],
        lease_id: str | None = None,
    ) -> None:
        """Read one bounded L2 miss batch and promote it to L1."""
        loop = asyncio.get_event_loop()
        if self._l2 is None:
            raise RuntimeError("L2 reads are disabled when skip_l2 is true")
        reads: list[tuple[dict[str, int], PinnedMemorySlice, int, int]] = []
        future_to_read: dict[
            asyncio.Future[int],
            tuple[dict[str, int], PinnedMemorySlice, int, int],
        ] = {}
        batch_jobs: list[asyncio.Future[int]] = []
        copy_futures: list[asyncio.Task[None]] = []
        queued: list[tuple[dict[str, int], PinnedMemorySlice, int, int]] = []

        def submit_queued() -> None:
            if not queued:
                return
            assert self._l2 is not None
            if len(queued) == 1:
                span, pinned, promotion_id, epoch = queued.pop()
                future = loop.run_in_executor(
                    self._l2.executor,
                    self._l2.timed("read", self._read_l2_into),
                    int(span["file_offset"]),
                    pinned,
                    self._next_uring(),
                )
                future_to_read[future] = (span, pinned, promotion_id, epoch)
                return
            job, completions = self._l2.submit_read_batch(
                [
                    (int(span["file_offset"]), pinned.view())
                    for span, pinned, _promotion, _epoch in queued
                ]
            )
            batch_jobs.append(job)
            future_to_read.update(zip(completions, queued, strict=True))
            queued.clear()

        def parts_for(span: dict[str, Any]) -> list[tuple[int, int, int, int, int]]:
            """Expand one physical read into destination/cache subranges."""
            parts = span.get("parts")
            if not parts:
                return [
                    (
                        int(span.get("target_offset", 0)),
                        int(span["file_offset"]),
                        int(span["nbytes"]),
                        0,
                        self._accounted_nbytes(span, int(span["nbytes"])),
                    )
                ]
            start = int(span["file_offset"])
            return [
                (
                    int(part["target_offset"]),
                    int(part["file_offset"]),
                    int(part["nbytes"]),
                    int(part["file_offset"]) - start,
                    self._accounted_nbytes(part, int(part["nbytes"])),
                )
                for part in parts
            ]

        # Futures whose completion has been processed; the batch runs in
        # reserve/drain rounds, and a round drains only what is still pending.
        processed: set[asyncio.Future[int]] = set()

        def outstanding() -> set[asyncio.Future[int]]:
            return set(future_to_read) - processed

        mark = time.perf_counter()
        try:
            next_miss = 0
            while next_miss < len(misses):
                while next_miss < len(misses):
                    span = misses[next_miss]
                    nbytes = int(span["nbytes"])
                    key = (int(span["file_offset"]), nbytes)
                    accounted_nbytes = self._accounted_nbytes(span, nbytes)
                    reservation = await self._reserve_l1_promotion(
                        key,
                        nbytes,
                        accounted_nbytes,
                        own_promotions={read[2] for read in reads},
                        lease_id=lease_id,
                        # Holding unfinished promotions while waiting for
                        # another batch's would deadlock two batches that
                        # wait on each other: finish this round first.
                        defer=bool(queued) or bool(outstanding()),
                    )
                    if reservation is None:
                        break
                    next_miss += 1
                    pinned, promotion_id, epoch, pending_writes = reservation
                    reads.append((span, pinned, promotion_id, epoch))
                    if pending_writes:
                        submit_queued()
                        await asyncio.gather(*set(pending_writes))
                    if nbytes <= _PACKED_READ_BATCH_BYTES:
                        queued.append((span, pinned, promotion_id, epoch))
                        if len(queued) >= _PACKED_READ_BATCH_COUNT:
                            submit_queued()
                        continue
                    submit_queued()
                    if self._l2_read_stripe:
                        # The batch job completes only after every part's CQE is
                        # drained, and raises the first part error after that.
                        view = pinned.view()
                        file_offset = int(span["file_offset"])
                        extents = _stripe_extents(nbytes, self._l2_read_stripe)
                        future, parts = self._l2.submit_read_batch(
                            [
                                (file_offset + offset, view[offset : offset + size])
                                for offset, size in extents
                            ]
                        )
                        batch_jobs.extend(parts)
                    else:
                        future = loop.run_in_executor(
                            self._l2.executor,
                            self._l2.timed("read", self._read_l2_into),
                            int(span["file_offset"]),
                            pinned,
                            self._next_uring(),
                        )
                    future_to_read[future] = (span, pinned, promotion_id, epoch)

                submit_queued()
                mark = self._add_phase_us("load_l2_reserve_us", mark)
                pending = outstanding()
                while pending:
                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    processed.update(done)
                    for future in done:
                        future.result()
                        span, pinned, promotion_id, epoch = future_to_read[future]
                        parts = parts_for(span)
                        if dst is not None:
                            # One physical read can represent several disjoint
                            # logical records. Submit those records as one grouped
                            # destination copy so the CUDA IPC copy path records
                            # one completion event and retains the pinned source
                            # once. This preserves each target/source offset while
                            # removing one host launch and event per part.
                            chunks = [
                                (target_offset, pinned, source_offset, nbytes)
                                for (
                                    target_offset,
                                    _file_offset,
                                    nbytes,
                                    source_offset,
                                    _accounted_nbytes,
                                ) in parts
                            ]
                            copy_future = self._copy_grouped_to_dst(dst, chunks)
                            if copy_future is not None:
                                copy_futures.append(copy_future)
                        async with self._lock:
                            self._raise_l2_error_locked()
                            self._stats.l2_reads += 1
                            stale_parts = [
                                (file_offset, nbytes)
                                for (
                                    _target_offset,
                                    file_offset,
                                    nbytes,
                                    _source_offset,
                                    _accounted_nbytes,
                                ) in parts
                                if self._promotion_is_stale_locked(
                                    file_offset, nbytes, epoch
                                )
                            ]
                            # A contiguous coalesced extent is one immutable L1
                            # object. Keeping that parent instead of one child
                            # slice per packed record lets later grouped H2D loads
                            # merge adjacent records into a single memcpy while
                            # retaining source offsets for each logical part.
                            contiguous = (
                                not stale_parts
                                and parts[0][1] == int(span["file_offset"])
                                and parts[-1][1] + parts[-1][2]
                                == int(span["file_offset"]) + int(span["nbytes"])
                                and all(
                                    current[1] == previous[1] + previous[2]
                                    for previous, current in zip(
                                        parts, parts[1:], strict=False
                                    )
                                )
                            )
                            if contiguous:
                                parent_key = (
                                    int(span["file_offset"]),
                                    int(span["nbytes"]),
                                )
                                parent_accounted_nbytes = sum(part[4] for part in parts)
                                # Retain lease references before publishing: an
                                # over-budget raw charge or a BIP insertion at the
                                # LRU end can evict (and close) the promoted slice
                                # inside put() before the lease could pin it.
                                if lease_id is not None:
                                    self._attach_l1_hits_to_lease_locked(
                                        lease_id,
                                        [
                                            L1RangeHit(
                                                target_offset=target_offset,
                                                key=parent_key,
                                                data=pinned,
                                                source_offset=source_offset,
                                                nbytes=nbytes,
                                            )
                                            for (
                                                target_offset,
                                                _file_offset,
                                                nbytes,
                                                source_offset,
                                                _accounted_nbytes,
                                            ) in parts
                                        ],
                                    )
                                self._l1.put(
                                    parent_key,
                                    pinned,
                                    accounted_nbytes=parent_accounted_nbytes,
                                )
                            else:
                                for (
                                    target_offset,
                                    file_offset,
                                    nbytes,
                                    source_offset,
                                    accounted_nbytes,
                                ) in parts:
                                    if (file_offset, nbytes) in stale_parts:
                                        # Another promotion or store already
                                        # published these bytes; retain that
                                        # copy so the lease stays complete.
                                        if lease_id is not None:
                                            resident, _missing = (
                                                self._l1.resolve_subranges(
                                                    target_offset=target_offset,
                                                    file_offset=file_offset,
                                                    nbytes=nbytes,
                                                )
                                            )
                                            self._attach_l1_hits_to_lease_locked(
                                                lease_id, resident
                                            )
                                        continue
                                    key = (file_offset, nbytes)
                                    child = pinned.subslice(source_offset, nbytes)
                                    if lease_id is not None:
                                        self._attach_l1_hits_to_lease_locked(
                                            lease_id,
                                            [
                                                L1RangeHit(
                                                    target_offset=target_offset,
                                                    key=key,
                                                    data=child,
                                                    source_offset=0,
                                                    nbytes=nbytes,
                                                )
                                            ],
                                        )
                                    self._l1.put(
                                        key,
                                        child,
                                        accounted_nbytes=accounted_nbytes,
                                    )
                            if stale_parts:
                                self._close_unowned_slice_locked(pinned)
                                self._l1.notify_pool_waiters()
                            self._finish_l1_promotion_locked(promotion_id)
                mark = self._add_phase_us("load_l2_io_us", mark)
        finally:
            # CQ callbacks can report success before the submitting thread has
            # drained its other reads. Wait for actual executor completion on
            # every exit path before returning any pinned storage to the pool.
            if batch_jobs:
                await asyncio.gather(*batch_jobs, return_exceptions=True)
            if future_to_read:
                await asyncio.gather(*future_to_read, return_exceptions=True)
            try:
                if copy_futures:
                    await _drain_destination_copies(copy_futures)
            finally:
                async with self._lock:
                    live_buffers = self._l1.resident_slice_ids() | set(
                        self._leased_slices
                    )
                    for _span, pinned, promotion_id, _epoch in reads:
                        self._finish_l1_promotion_locked(promotion_id)
                        if id(pinned) not in live_buffers:
                            pinned.close()
                            self._l1.notify_pool_waiters()

    def _next_l2_miss_batch(
        self,
        misses: list[dict[str, int]],
        start: int,
    ) -> list[dict[str, int]]:
        """Return a miss batch bounded by L1 capacity and io_uring count."""
        ring_count = self._l2.ring_count if self._l2 is not None else 1
        # A large read occupies a ring on its own (striped or single), while
        # small reads share one submission of up to _PACKED_READ_BATCH_COUNT
        # entries; counting each small read as a full ring would cap scattered
        # small misses at ``ring_count`` reads in flight.
        capacity = ring_count * _PACKED_READ_BATCH_COUNT
        batch: list[dict[str, int]] = []
        batch_bytes = 0
        batch_cost = 0
        for span in misses[start:]:
            nbytes = int(span["nbytes"])
            cost = 1 if nbytes <= _PACKED_READ_BATCH_BYTES else _PACKED_READ_BATCH_COUNT
            if batch and (
                batch_bytes + nbytes > self._l1_bytes or batch_cost + cost > capacity
            ):
                break
            batch.append(span)
            batch_bytes += nbytes
            batch_cost += cost
        return batch

    def _next_uring(self) -> NativeIOUring:
        """Return the next native io_uring instance for one L2 operation."""
        if self._l2 is None:
            raise RuntimeError("io_uring rings are disabled when skip_l2 is true")
        return self._l2.next_uring()

    def _copy_grouped_to_dst(
        self,
        dst: Any,
        chunks: list[tuple[int, PinnedMemorySlice, int, int]],
    ) -> asyncio.Task[None] | None:
        """Submit copies while borrowing CUDA sources until their event completes."""
        event = copy_ops.destination_copy_event(dst, chunks)
        if event is None:
            copy_ops.copy_grouped_to_dst(dst, chunks)
            return None
        # Borrow the original slices, not just Python objects: L1 eviction can
        # explicitly close and recycle a still-referenced slice. The existing
        # lease registry also makes in-place overwrites preserve these bytes.
        sources = {id(data): data for _target, data, _offset, _size in chunks}
        for data_id, data in sources.items():
            existing = self._leased_slices.get(data_id)
            self._leased_slices[data_id] = (
                data,
                1 if existing is None else existing[1] + 1,
            )
        error: BaseException | None = None
        try:
            runs = copy_ops.copy_grouped_to_dst(dst, chunks)
            self._stats.dst_copies += 1
            self._stats.dst_copy_runs += runs
            self._stats.dst_copy_bytes += sum(chunk[3] for chunk in chunks)
        except BaseException as exc:
            error = exc
        finally:
            # A partial submission can still own source pages. Record and
            # drain its event even when a later chunk raises during enqueue.
            copy_ops.record_destination_copy_event(dst, event)
            completion = asyncio.create_task(
                self._finish_destination_copy(event, tuple(sources), error)
            )
            self._pending_h2d.add(completion)
            completion.add_done_callback(self._pending_h2d.discard)
        return completion

    async def _finish_destination_copy(
        self,
        event: Any,
        source_ids: tuple[int, ...],
        error: BaseException | None,
    ) -> None:
        """Release borrowed DMA sources on the owning loop after device completion."""
        try:
            while not event.done:
                await asyncio.sleep(0.0005)
        finally:
            async with self._lock:
                for data_id in source_ids:
                    self._release_slice_reference_locked(data_id)
                self._l1.notify_pool_waiters()
        if error is not None:
            raise error

    def _slice_src(self, src: Any, offset: int, nbytes: int) -> Any:
        """Return a readable source slice."""
        return copy_ops.slice_src(src, offset, nbytes)

    def _copy_src_to_pinned(
        self,
        src: Any,
        pinned: PinnedMemorySlice,
        nbytes: int,
    ) -> None:
        """Copy bytes from a CPU or CuPy source into pinned host memory."""
        copy_ops.copy_src_to_pinned(src, pinned, 0, nbytes)

    def _copy_src_to_pinned_at(
        self,
        src: Any,
        pinned: PinnedMemorySlice,
        target_offset: int,
        nbytes: int,
    ) -> None:
        """Copy bytes from a CPU or CUDA source into pinned host memory."""
        copy_ops.copy_src_to_pinned(src, pinned, target_offset, nbytes)

    async def _store_bytes_grouped_l1_only(
        self,
        src: Any,
        spans: list[dict[str, Any]],
    ) -> int:
        """Store grouped spans in L1 without scheduling L2 persistence."""
        while True:
            async with self._lock:
                self._raise_l2_error_locked()
                waiters = {
                    waiter
                    for span in spans
                    for waiter in self._overlapping_lease_waiters_locked(
                        int(span["file_offset"]),
                        int(span["nbytes"]),
                    )
                }
                if not waiters:
                    total = 0
                    for span in spans:
                        source_offset = int(span.get("source_offset", 0))
                        nbytes = int(span["nbytes"])
                        file_offset = int(span["file_offset"])
                        accounting = self._accounted_nbytes(span, nbytes)
                        self._check_range(file_offset, nbytes)
                        key = (file_offset, nbytes)
                        hit = self._l1.find(file_offset)
                        source = self._slice_src(src, source_offset, nbytes)
                        if hit is not None:
                            hit_key, cached, target_offset = hit
                            if target_offset + nbytes <= len(cached):
                                self._copy_src_to_pinned_at(
                                    source, cached, target_offset, nbytes
                                )
                                self._l1.touch(hit_key)
                                self._record_cache_mutation_locked(file_offset, nbytes)
                                total += nbytes
                                continue
                        data = self._l1.reserve_or_raise(
                            key,
                            nbytes,
                            preserve_overlaps=True,
                            accounted_nbytes=accounting,
                        )
                        try:
                            self._copy_src_to_pinned_at(source, data, 0, nbytes)
                        except BaseException:
                            data.close()
                            raise
                        self._record_cache_mutation_locked(file_offset, nbytes)
                        self._l1.put(key, data, accounted_nbytes=accounting)
                        total += nbytes
                    return total
            await asyncio.gather(*waiters)

    async def _reserve_l1_buffer(
        self,
        key: tuple[int, int],
        nbytes: int,
        accounted_nbytes: int,
    ) -> PinnedMemorySlice:
        """Reserve pinned L1 space, waiting for pending L2 victims if needed.

        Args:
            key: L1 byte-range key being inserted.
            nbytes: Number of logical bytes needed.
            accounted_nbytes: L1 capacity charge for the range.

        Returns:
            A pinned slice leased from the preallocated pool.

        Async/thread-safety:
            Mutates L1 metadata under the transfer lock. If evicted victims are
            still owned by pending L2 writes, releases the lock and waits for
            those writes to free their pool slices.
        """
        while True:
            async with self._lock:
                self._raise_l2_error_locked()
                data = self._l1.reserve(
                    key,
                    nbytes,
                    accounted_nbytes=accounted_nbytes,
                )
                if data is not None:
                    return data
                # Pool exhausted with no evictable victim: an in-flight L2
                # write still pins a slice. Wait for one to finish, then retry.
                wait_for: asyncio.Future[None] | None = next(
                    (task for task in self._pending_l2.values() if not task.done()),
                    None,
                )
                if wait_for is None:
                    wait_for = next(
                        (
                            promotion.done
                            for promotion in self._promotions.values()
                            if not promotion.done.done()
                        ),
                        None,
                    )
                if wait_for is None:
                    if self._reclaim_idle_readahead_locked():
                        continue
                    wait_for = asyncio.get_event_loop().create_future()
                    self._l1.register_pool_waiter(wait_for)
                    timeout: float | None = _POOL_RETRY_S
                else:
                    timeout = None
            await asyncio.wait({wait_for}, timeout=timeout)

    async def _reserve_l1_promotion(
        self,
        key: tuple[int, int],
        nbytes: int,
        accounted_nbytes: int,
        own_promotions: set[int],
        lease_id: str | None = None,
        defer: bool = False,
    ) -> (
        tuple[
            PinnedMemorySlice,
            int,
            int,
            list[asyncio.Task[None]],
        ]
        | None
    ):
        """Reserve and register a pinned slice for one L2 promotion.

        Args:
            key: L2 range being promoted.
            nbytes: Number of bytes to reserve.
            accounted_nbytes: L1 capacity charge for the range.
            own_promotions: Reservation identifiers the caller already holds
                in the same read batch. Those finish only after this call
                returns, so waiting on them would never wake.
            lease_id: Request lease the promotion reads into, if any.
            defer: Return None instead of waiting when the pool is full. A
                caller holding unfinished promotions sets it: waiting would
                hold them while another batch may wait on them in turn.

        Returns:
            The pinned slice, reservation identifier, cache mutation epoch, and
            L2 writes that must finish before the read starts; None when
            ``defer`` is set and no slice is free.

        Raises:
            _ReadaheadStarved: ``lease_id`` is a readahead and the pool is
                exhausted with nothing in flight to free it.

        Async/thread-safety:
            The reservation and its waiter are registered while the transfer
            lock is held. Other stores therefore cannot mistake an in-flight
            promotion for free pool capacity.
        """
        since = time.perf_counter()
        logged = False
        while True:
            async with self._lock:
                self._raise_l2_error_locked()
                data = self._l1.reserve(
                    key,
                    nbytes,
                    accounted_nbytes=accounted_nbytes,
                )
                if data is not None:
                    promotion_id = id(data)
                    self._promotions[promotion_id] = _Promotion(
                        key=key,
                        epoch=self._cache_epoch,
                        done=asyncio.get_event_loop().create_future(),
                    )
                    pending_writes = self._find_pending_l2_locked(key[0], key[1])
                    return (
                        data,
                        promotion_id,
                        self._cache_epoch,
                        pending_writes,
                    )
                if defer:
                    return None
                cause = "l2_write"
                wait_for: asyncio.Future[None] | None = next(
                    (task for task in self._pending_l2.values() if not task.done()),
                    None,
                )
                if wait_for is None:
                    cause = "promotion"
                    wait_for = next(
                        (
                            promotion.done
                            for promotion_id, promotion in self._promotions.items()
                            if promotion_id not in own_promotions
                            and not promotion.done.done()
                        ),
                        None,
                    )
                if wait_for is None:
                    cause = "pool"
                    if lease_id is not None and lease_id in self._readaheads:
                        # Readaheads are best effort; the load they serve may be
                        # what holds the pool, so do not wait on it.
                        raise _ReadaheadStarved(f"{lease_id} needs {nbytes} bytes")
                    if self._reclaim_idle_readahead_locked(exclude=lease_id):
                        continue
                    wait_for = asyncio.get_event_loop().create_future()
                    self._l1.register_pool_waiter(wait_for)
            started = time.perf_counter()
            try:
                if cause == "pool":
                    # Poll: readahead pieces landing do not wake pool waiters.
                    await asyncio.wait({wait_for}, timeout=_POOL_RETRY_S)
                    waited_s = started - since
                    if not logged and waited_s >= _STUCK_WAIT_LOG_S:
                        logged = True
                        # Diagnostic for promotion stalls that never wake.
                        logger.warning(
                            "[TRANSFER:iouring] promotion of %d bytes waiting "
                            "%.0f s on pool: l1_used=%d/%d "
                            "readahead_retained=%d readaheads=%d",
                            nbytes,
                            waited_s,
                            self._l1.bytes_used,
                            self._l1_bytes,
                            self._readahead_bytes_locked(),
                            len(self._readaheads),
                        )
                else:
                    await wait_for
            finally:
                field_name = f"promotion_wait_{cause}_us"
                setattr(
                    self._stats,
                    field_name,
                    getattr(self._stats, field_name)
                    + int((time.perf_counter() - started) * 1e6),
                )

    def _record_cache_mutation_locked(self, file_offset: int, nbytes: int) -> None:
        """Record a newly published store for concurrent promotion checks."""
        self._cache_epoch += 1
        self._cache_mutations.append(
            (self._cache_epoch, file_offset, file_offset + nbytes)
        )

    def _promotion_is_stale_locked(
        self,
        file_offset: int,
        nbytes: int,
        epoch: int,
    ) -> bool:
        """Return whether a newer overlapping store invalidated a promotion."""
        end = file_offset + nbytes
        if self._l1.has_overlap(file_offset, nbytes):
            return True
        return any(
            mutation_epoch > epoch
            and mutation_start < end
            and file_offset < mutation_end
            for mutation_epoch, mutation_start, mutation_end in self._cache_mutations
        )

    def _finish_l1_promotion_locked(self, promotion_id: int) -> None:
        """Release one promotion reservation and wake blocked pool users."""
        promotion = self._promotions.pop(promotion_id, None)
        if promotion is not None and not promotion.done.done():
            promotion.done.set_result(None)
        if self._promotions:
            oldest_epoch = min(p.epoch for p in self._promotions.values())
            self._cache_mutations = [
                mutation
                for mutation in self._cache_mutations
                if mutation[0] > oldest_epoch
            ]
        else:
            self._cache_mutations.clear()

    def _replace_request_lease_locked(
        self,
        lease_id: str,
        ranges: list[tuple[int, int]],
        hits: list[L1RangeHit],
    ) -> None:
        """Replace one request lease while the transfer metadata lock is held."""
        if not lease_id:
            raise ValueError("lease_id must not be empty")
        existing = self._request_leases.get(lease_id)
        if existing is not None and existing.active_loads:
            raise RuntimeError(f"cannot replace active transfer lease: {lease_id}")
        self._release_request_lease_locked(lease_id)
        self._request_leases[lease_id] = _RequestLease(
            remaining=list(ranges),
            hits=[],
            slice_ids=set(),
            released=asyncio.get_running_loop().create_future(),
        )
        self._attach_l1_hits_to_lease_locked(lease_id, hits)

    def _attach_l1_hits_to_lease_locked(
        self,
        lease_id: str,
        hits: list[L1RangeHit],
    ) -> None:
        """Retain resident slice references for one admitted request."""
        lease = self._request_leases.get(lease_id)
        if lease is None:
            raise RuntimeError(
                f"transfer lease was released during prefetch: {lease_id}"
            )
        new_ids: set[int] = set()
        for hit in hits:
            lease.hits.append(
                _LeasedSliceRange(
                    file_offset=hit.key[0] + hit.source_offset,
                    nbytes=hit.nbytes,
                    data=hit.data,
                    source_offset=hit.source_offset,
                )
            )
            data_id = id(hit.data)
            if data_id not in lease.slice_ids:
                new_ids.add(data_id)
                lease.slice_ids.add(data_id)
        lease.hits.sort(key=lambda item: item.file_offset)
        lease.run_index = None
        for data_id in new_ids:
            data = next(hit.data for hit in hits if id(hit.data) == data_id)
            existing = self._leased_slices.get(data_id)
            if existing is None:
                self._leased_slices[data_id] = (data, 1)
            else:
                self._leased_slices[data_id] = (existing[0], existing[1] + 1)

    def _require_complete_lease_locked(self, lease_id: str) -> None:
        """Raise unless retained slice ranges cover the whole request lease."""
        lease = self._request_leases.get(lease_id)
        if lease is None:
            raise RuntimeError(
                f"transfer lease was released during prefetch: {lease_id}"
            )
        uncovered = list(lease.remaining)
        for hit in lease.hits:
            uncovered = _subtract_ranges(
                uncovered,
                [(hit.file_offset, hit.nbytes)],
            )
        if uncovered:
            raise LeaseIncompleteError(
                f"prefetch did not retain complete lease: {lease_id}"
            )

    def _resolve_leased_range_locked(
        self,
        lease: _RequestLease,
        target_offset: int,
        file_offset: int,
        nbytes: int,
    ) -> list[tuple[int, PinnedMemorySlice, int, int]]:
        """Resolve one load span exclusively through retained lease references."""
        chunks: list[tuple[int, PinnedMemorySlice, int, int]] = []
        cursor = file_offset
        end = file_offset + nbytes
        while cursor < end:
            covering = next(
                (
                    hit
                    for hit in lease.hits
                    if hit.file_offset <= cursor < hit.file_offset + hit.nbytes
                ),
                None,
            )
            if covering is None:
                raise KeyError(
                    f"request lease does not cover load range [{file_offset}, {end})"
                )
            covered = min(end, covering.file_offset + covering.nbytes) - cursor
            chunks.append(
                (
                    target_offset + cursor - file_offset,
                    covering.data,
                    covering.source_offset + cursor - covering.file_offset,
                    covered,
                )
            )
            cursor += covered
        return chunks

    def _consume_lease_ranges_locked(
        self,
        lease_id: str,
        removals: list[tuple[int, int]],
    ) -> None:
        """Remove staged ranges and release slice references no longer needed."""
        lease = self._request_leases.get(lease_id)
        if lease is None:
            return
        old_slice_ids = set(lease.slice_ids)
        lease.remaining = _subtract_ranges(lease.remaining, removals)
        lease.hits = _subtract_leased_hits(lease.hits, removals)
        lease.run_index = None
        lease.slice_ids = {id(hit.data) for hit in lease.hits}
        for data_id in old_slice_ids - lease.slice_ids:
            self._release_slice_reference_locked(data_id)
        if lease.remaining:
            # Consumed ranges no longer count against leased L1 capacity.
            self._lease_released.notify_all()
            return
        self._request_leases.pop(lease_id, None)
        if not lease.released.done():
            lease.released.set_result(None)
        self._l1.notify_pool_waiters()
        self._lease_released.notify_all()

    def _release_request_lease_locked(
        self,
        lease_id: str,
        *,
        force: bool = False,
    ) -> None:
        """Drop one complete lease and wake overlapping store waiters."""
        lease = self._request_leases.get(lease_id)
        if lease is None:
            return
        if lease.active_loads and not force:
            lease.release_all = True
            return
        self._request_leases.pop(lease_id, None)
        for data_id in lease.slice_ids:
            self._release_slice_reference_locked(data_id)
        if not lease.released.done():
            lease.released.set_result(None)
        self._l1.notify_pool_waiters()
        self._lease_released.notify_all()

    def _leased_physical_bytes_locked(self, exclude: str | None) -> int:
        """Return physical L1 bytes still retained by live request leases.

        Args:
            exclude: Lease ID left out of the sum, because a prefetch for the
                same request replaces that lease.

        Returns:
            Sum of the unconsumed range sizes over the remaining leases.
        """
        return sum(
            size
            for lease_id, lease in self._request_leases.items()
            if lease_id != exclude
            for _start, size in lease.remaining
        )

    def _leased_budget_bytes_locked(self, exclude: str | None) -> int:
        """Return accounted bytes charged by live prefetch leases.

        Args:
            exclude: Lease ID left out of the sum, because a prefetch for the
                same request replaces that lease's charge.

        Returns:
            Sum of ``budget_bytes`` over the remaining request leases.
        """
        return sum(
            lease.budget_bytes
            for lease_id, lease in self._request_leases.items()
            if lease_id != exclude
        )

    def _release_slice_reference_locked(self, data_id: int) -> None:
        """Release one request's reference to a retained pinned slice."""
        retained = self._leased_slices.get(data_id)
        if retained is None:
            return
        data, count = retained
        if count > 1:
            self._leased_slices[data_id] = (data, count - 1)
            return
        self._leased_slices.pop(data_id, None)
        self._close_unowned_slice_locked(data)

    def _close_unowned_slice_locked(self, data: PinnedMemorySlice) -> None:
        """Close a detached slice after its final writer/lease reference leaves."""
        if self._l1.contains_slice(data):
            return
        if id(data) in self._l2_write_buffers:
            return
        if id(data) in self._leased_slices:
            return
        data.close()
        self._l1.notify_pool_waiters()

    def _overlapping_lease_waiters_locked(
        self,
        file_offset: int,
        nbytes: int,
        *,
        retained_ok: bool = False,
    ) -> list[asyncio.Future[None]]:
        """Return active request-lease waiters overlapping one store range.

        Readahead leases are speculative: an overlapping store drops them
        instead of waiting, so an unconsumed readahead never blocks a writer.

        Args:
            file_offset: Start of the store range.
            nbytes: Length of the store range.
            retained_ok: The store writes a fresh slice instead of mutating
                resident bytes in place, so a lease whose own pinned slices
                already cover the overlap (its loads read those, never L1 or
                L2) does not block it. Such a lease can be held for as long as
                its request waits for GPU blocks, which may in turn wait for
                this store's request to free its blocks.

        Returns:
            ``released`` futures of the leases the store must wait for. With
            ``retained_ok`` a lease may stop blocking without being released,
            so callers wait with :meth:`_wait_lease_waiters`.
        """
        end = file_offset + nbytes
        for lease_id in [
            lease_id
            for lease_id in self._readaheads
            if _ranges_overlap(
                self._request_leases[lease_id].remaining
                if lease_id in self._request_leases
                else [],
                [(file_offset, nbytes)],
            )
        ]:
            self._drop_readahead_locked(lease_id, "store")
        return [
            lease.released
            for lease in self._request_leases.values()
            if any(
                start < end and file_offset < start + size
                for start, size in lease.remaining
            )
            and not (retained_ok and _lease_retains_range(lease, file_offset, nbytes))
        ]

    async def _wait_for_overlapping_leases(
        self,
        file_offset: int,
        nbytes: int,
    ) -> float | None:
        """Wait until no request lease blocks a fresh-slice store range.

        Returns:
            The ``time.perf_counter()`` start of the lease wait, or None when
            the store did not wait.
        """
        since: float | None = None
        while True:
            async with self._lock:
                waiters = self._overlapping_lease_waiters_locked(
                    file_offset, nbytes, retained_ok=True
                )
            if not waiters:
                return since
            since = await self._wait_lease_waiters(waiters, since)

    async def _wait_lease_waiters(
        self, waiters: Iterable[asyncio.Future[None]], since: float | None
    ) -> float:
        """Wait for a blocking lease to be released or to retain its range.

        A prefetch attaching its promoted slices makes a lease stop blocking
        ``retained_ok`` stores without resolving ``released``, so the caller
        rechecks at the pool retry interval.

        Args:
            waiters: Release futures of the blocking leases.
            since: Start of this store's lease wait, or None on its first wait.

        Returns:
            The wait start to pass to the next call of the same store. The
            time waited is added to ``store_lease_wait_us`` and the store's
            wait so far updates ``store_lease_wait_max_us``.
        """
        mark = time.perf_counter()
        since = mark if since is None else since
        await asyncio.wait(set(waiters), timeout=_POOL_RETRY_S)
        now = time.perf_counter()
        self._stats.store_lease_wait_us += int((now - mark) * 1_000_000)
        self._stats.store_lease_wait_max_us = max(
            self._stats.store_lease_wait_max_us, int((now - since) * 1_000_000)
        )
        return since

    async def _run_readahead(
        self,
        lease_id: str,
        readahead: _Readahead,
        groups: list[list[dict[str, Any]]],
    ) -> None:
        """Read readahead groups in order with a bounded number in flight."""
        await self._read_readahead_groups(lease_id, readahead, groups)

    def _readahead_task_done(self, task: asyncio.Task[None]) -> None:
        """Forget a finished readahead task and mark idleness when last."""
        self._readahead_tasks.discard(task)
        self._readahead_running -= 1
        if not self._readahead_running:
            self._readahead_idle.set()

    async def _read_readahead_groups(
        self,
        lease_id: str,
        readahead: _Readahead,
        groups: list[list[dict[str, Any]]],
    ) -> None:
        """Read the groups of one readahead and log its throughput."""
        window = asyncio.Semaphore(self._readahead_window)
        read_bytes = 0
        failed = 0
        # Physical L2 reads (one per coalesced miss) across all groups.
        reads = sum(len(group) for group in groups)

        async def read(
            group: list[dict[str, Any]], future: asyncio.Future[None]
        ) -> None:
            nonlocal read_bytes, failed
            mark = time.perf_counter()
            try:
                async with window:
                    self._add_phase_us("load_ra_window_us", mark)
                    read_bytes += await self._read_readahead_group(lease_id, group)
            except Exception as exc:  # noqa: BLE001
                # A dropped lease or a stale range only costs the readahead:
                # the waiting load re-resolves through the normal path.
                failed += 1
                logger.debug("[TRANSFER:iouring] readahead group failed: %s", exc)
            finally:
                if not future.done():
                    future.set_result(None)

        await asyncio.gather(
            *(
                read(group, future)
                for group, (_ranges, future) in zip(
                    groups, readahead.pieces, strict=True
                )
            )
        )
        elapsed = time.perf_counter() - readahead.started
        logger.info(
            "[TRANSFER:iouring] readahead %s groups=%d reads=%d failed=%d read=%d "
            "elapsed_ms=%.1f gbps=%.2f",
            lease_id,
            len(groups),
            reads,
            failed,
            read_bytes,
            elapsed * 1000,
            read_bytes / elapsed / 1e9 if elapsed > 0 else 0.0,
        )

    async def _read_readahead_group(
        self, lease_id: str, group: list[dict[str, Any]]
    ) -> int:
        """Read one readahead group into its lease and count contention.

        Returns the bytes read, 0 when the lease is already gone. Bytes count
        as contended when another readahead has a read in flight at the start
        or the end of this one.
        """
        if lease_id not in self._request_leases:
            return 0
        reading = self._readahead_reading
        reading[lease_id] = reading.get(lease_id, 0) + 1
        contended = len(reading) > 1
        try:
            await self._load_l2_miss_batch(None, group, lease_id=lease_id)
            contended = contended or len(reading) > 1
        finally:
            reading[lease_id] -= 1
            if not reading[lease_id]:
                del reading[lease_id]
        group_bytes = sum(int(miss["nbytes"]) for miss in group)
        self._stats.readahead_read_bytes += group_bytes
        if contended:
            self._stats.readahead_contended_bytes += group_bytes
        return group_bytes

    async def _resolve_load_batch(
        self, dst: Any, spans: list[dict[str, int]], plan: _LoadPlan
    ) -> None:
        """Resolve load spans against readahead leases and L1, under the lock.

        Retained and L1-resident bytes are submitted as one destination copy
        appended to ``plan.copies``; L1 misses and the pending L2 writes
        they must wait for are appended to ``plan``.
        """
        merged_l1: list[tuple[int, PinnedMemorySlice, int, int]] = []
        async with self._lock:
            self._raise_l2_error_locked()
            readahead_hits = self._readahead_hits_locked()
            for span in spans:
                target_offset = int(span.get("target_offset", 0))
                file_offset = int(span["file_offset"])
                nbytes = int(span["nbytes"])
                accounted_nbytes = self._accounted_nbytes(span, nbytes)
                self._check_range(file_offset, nbytes)
                plan.total += nbytes
                leased, uncovered = _resolve_from_hits(
                    readahead_hits, target_offset, file_offset, nbytes
                )
                if leased:
                    self._l1.mark_read(file_offset, nbytes)
                    self._stats.l1_hits += len(leased)
                    self._stats.readahead_served_bytes += sum(
                        chunk[3] for chunk in leased
                    )
                    merged_l1.extend(leased)
                l1_hits = []
                span_misses = []
                for sub_target, sub_offset, sub_nbytes in uncovered:
                    sub_hits, sub_misses = self._l1.resolve_subranges(
                        target_offset=sub_target,
                        file_offset=sub_offset,
                        nbytes=sub_nbytes,
                    )
                    l1_hits.extend(sub_hits)
                    span_misses.extend(sub_misses)
                if self._l2 is None and span_misses:
                    self._stats.l1_misses += 1
                    raise KeyError(
                        "skip_l2 cache miss for range "
                        f"[{file_offset}, {file_offset + nbytes})"
                    )
                if span_misses:
                    _assign_accounting_charges(
                        span_misses,
                        accounted_nbytes,
                        nbytes,
                    )
                if l1_hits:
                    self._l1.record_hits(l1_hits)
                    self._stats.l1_hits += 1 if self._l2 is None else len(l1_hits)
                    merged_l1.extend(
                        (
                            hit.target_offset,
                            hit.data,
                            hit.source_offset,
                            hit.nbytes,
                        )
                        for hit in l1_hits
                    )
                for miss in span_misses:
                    self._stats.l1_misses += 1
                    plan.pending.extend(
                        self._find_pending_l2_locked(
                            int(miss["file_offset"]),
                            int(miss["nbytes"]),
                        )
                    )
                    plan.misses.append(
                        {
                            "target_offset": int(miss["target_offset"]),
                            "file_offset": int(miss["file_offset"]),
                            "nbytes": int(miss["nbytes"]),
                            "accounted_nbytes": int(miss["accounted_nbytes"]),
                        }
                    )
            if merged_l1:
                copy_future = self._copy_grouped_to_dst(dst, merged_l1)
                if copy_future is not None:
                    plan.copies.append(copy_future)
            if self._readaheads:
                # The copy above borrows every leased slice it reads, so the
                # served ranges can leave their readahead lease right away.
                self._consume_readaheads_locked(_normalize_ranges(spans))

    def _pending_readahead_pieces(
        self, spans: list[dict[str, Any]]
    ) -> list[tuple[list[tuple[int, int]], asyncio.Future[None]]]:
        """Return in-flight readahead pieces overlapping the load spans."""
        ranges = _normalize_ranges(spans)
        return [
            (piece_ranges, future)
            for readahead in list(self._readaheads.values())
            for piece_ranges, future in readahead.pieces
            if not future.done() and _ranges_overlap(piece_ranges, ranges)
        ]

    async def _wait_for_readaheads(self, spans: list[dict[str, Any]]) -> None:
        """Wait for in-flight readahead groups overlapping the load spans."""
        waiters = [future for _ranges, future in self._pending_readahead_pieces(spans)]
        if waiters:
            await asyncio.gather(*waiters)

    def _add_phase_us(self, field: str, mark: float) -> float:
        """Charge the time since ``mark`` to a load-phase counter.

        Args:
            field: ``TransferStats`` microsecond counter to increment.
            mark: ``time.perf_counter()`` value at the start of the phase.

        Returns:
            The current ``time.perf_counter()`` value, the next phase's mark.
        """
        now = time.perf_counter()
        setattr(
            self._stats,
            field,
            getattr(self._stats, field) + int((now - mark) * 1_000_000),
        )
        return now

    def _readahead_hits_locked(self) -> tuple[list[int], list[_LeasedSliceRange]]:
        """Return the retained readahead ranges sorted by physical offset."""
        self._prune_readaheads_locked()
        ranges = sorted(
            (
                hit
                for lease_id in self._readaheads
                for hit in self._request_leases[lease_id].hits
            ),
            key=lambda hit: hit.file_offset,
        )
        return [hit.file_offset for hit in ranges], ranges

    def _consume_readaheads_locked(self, removals: list[tuple[int, int]]) -> None:
        """Release loaded ranges from every readahead lease covering them."""
        for lease_id in list(self._readaheads):
            lease = self._request_leases.get(lease_id)
            if lease is not None and _ranges_overlap(lease.remaining, removals):
                self._consume_lease_ranges_locked(lease_id, removals)
        self._prune_readaheads_locked()

    def _reclaim_idle_readahead_locked(self, exclude: str | None = None) -> bool:
        """Drop the oldest readahead with no read or load in flight.

        A reservation calls this when the pool is exhausted and no pending L2
        write or promotion will free space. Otherwise an unconsumed readahead
        (its request aborted, trimmed or GPU-hit) keeps its slices until it
        expires, and only a later readahead checks the expiry. Dropping it
        costs at most a re-read from L2.

        Args:
            exclude: Lease identifier never dropped (the caller's own).

        Returns:
            True when a readahead was dropped.
        """
        idle = []
        for lease_id, item in self._readaheads.items():
            lease = self._request_leases.get(lease_id)
            if (
                lease_id != exclude
                and (lease is None or not lease.active_loads)
                and all(future.done() for _ranges, future in item.pieces)
            ):
                idle.append((item.started, lease_id))
        if not idle:
            return False
        self._drop_readahead_locked(min(idle)[1], "pool")
        return True

    def _readahead_bytes_locked(self) -> int:
        """Return physical bytes still retained by readahead leases."""
        return sum(
            size
            for lease_id in self._readaheads
            if lease_id in self._request_leases
            for _start, size in self._request_leases[lease_id].remaining
        )

    def _drop_readahead_locked(self, lease_id: str, reason: str) -> None:
        """Forget one readahead and release its retained slices.

        In-flight groups of a dropped readahead fail when they try to attach
        to the released lease; their reads are closed by the normal cleanup.
        ``reason`` labels the drop in the log.
        """
        self._readaheads.pop(lease_id, None)
        lease = self._request_leases.get(lease_id)
        if lease is not None:
            dropped = sum(hit.nbytes for hit in lease.hits)
            self._stats.readahead_dropped_bytes += dropped
            logger.info(
                "[TRANSFER:iouring] readahead %s dropped reason=%s bytes=%d",
                lease_id,
                reason,
                dropped,
            )
        self._release_request_lease_locked(lease_id, force=True)

    def _prune_readaheads_locked(self) -> None:
        """Forget readaheads whose lease was fully consumed or released."""
        for lease_id in [
            lease_id
            for lease_id in self._readaheads
            if lease_id not in self._request_leases
        ]:
            del self._readaheads[lease_id]
