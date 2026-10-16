# SPDX-License-Identifier: Apache-2.0

"""Transfer-layer contract: move bytes between server-chosen locations.

The server decides every location. A transfer backend copies bytes between a
worker staging buffer (CUDA IPC mapping or CPU buffer), the pinned L1 arena
and the L2 store file, and keeps only transient state: in-flight IO, staging
slices and copy events. It never caches bytes by address, never decides
residency and never evicts.
"""

# Standard
from abc import ABC, abstractmethod
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

#: Tier a load reads from.
LoadSource = Literal["l1", "l2"]


@dataclass(frozen=True)
class Move:
    """One contiguous byte range to move between server-chosen locations.

    Attributes:
        buffer_offset: Byte offset in the worker staging buffer; ignored by
            ``promote``, which has no buffer.
        nbytes: Byte count, positive.
        l1_offset: Absolute byte offset in the pinned L1 arena, or None.
        l2_offset: Absolute byte offset in the L2 store file (rank lane
            already applied by the server), or None.
    """

    buffer_offset: int
    nbytes: int
    l1_offset: int | None = None
    l2_offset: int | None = None


@dataclass(frozen=True)
class StoreHandle:
    """Completion of one ``TransferLayer.store`` call.

    Attributes:
        staged: Resolves once every source byte has been consumed: each L1
            copy is valid and the worker staging buffer may be reused.
        persisted: Resolves once every L2 write finished; carries the error
            of any failed snapshot or write. Without L2 work it resolves
            together with ``staged``.

    Errors:
        A failed source snapshot fails ``staged`` (no L1 copy of the batch is
        valid) and ``persisted``; a failed L2 write fails ``persisted`` only.
        Errors are marked retrieved, so awaiting neither future does not
        warn; ``drain`` re-raises them.

    Async/thread-safety:
        Both futures belong to the server event loop. Awaiting a future from
        a task that may be cancelled cancels the future itself, so such
        callers wrap it in ``asyncio.shield``; the IO continues either way.
    """

    staged: asyncio.Future[None]
    persisted: asyncio.Future[None]


@dataclass
class TransferStats:
    """Cumulative data-movement counters of one transfer backend.

    Attributes:
        l2_reads: physical L2 read operations completed.
        l2_read_bytes: bytes those reads returned.
        l2_writes: physical L2 write operations completed.
        l2_write_bytes: bytes those writes persisted.
        d2h_copies: staging-buffer-to-host snapshot calls (store side).
        d2h_bytes: bytes those snapshots copied.
        dst_copies: host-to-destination copy submissions (load side).
        dst_copy_runs: contiguous CUDA runs those submissions issued.
        dst_copy_bytes: bytes those submissions copied.
        staging_waits: staging-pool reservations that had to wait.
        staging_wait_us: total time those reservations waited.
    """

    l2_reads: int = 0
    l2_read_bytes: int = 0
    l2_writes: int = 0
    l2_write_bytes: int = 0
    d2h_copies: int = 0
    d2h_bytes: int = 0
    dst_copies: int = 0
    dst_copy_runs: int = 0
    dst_copy_bytes: int = 0
    staging_waits: int = 0
    staging_wait_us: int = 0


class TransferLayer(ABC):
    """Abstract server-owned KV data mover.

    Capability surface:
        ``store``, ``load``, ``drain`` and ``close`` are required. Backends
        without a pinned L1 arena keep the default ``promote`` and
        ``load_runs``, which raise ``NotImplementedError``; the server checks
        the backend's capabilities at startup instead of probing types.

    Async/thread-safety:
        Call every method from the server asyncio event loop. Blocking
        syscalls, CUDA copies and completion waits are offloaded or polled
        without blocking the loop. Callers must keep the buffer and every
        referenced extent alive and unchanged until the operation (for a
        store: ``StoreHandle.persisted``) completes.
    """

    @property
    @abstractmethod
    def stats(self) -> TransferStats:
        """Return this backend's cumulative counters.

        Returns:
            The live counters object; read it on the server event loop.
        """

    @property
    def l2_timing_us(self) -> dict[str, int]:
        """Return cumulative L2 executor queue and native IO microseconds.

        Returns:
            Totals keyed by phase (for io_uring ``read_queue``, ``read_io``,
            ``write_queue``, ``write_io``, ``write_gate``); empty for backends
            that do not measure them.
        """
        return {}

    @property
    def l2_read_contention(self) -> dict[tuple[int, bool], tuple[int, int, int]]:
        """Return L2 read jobs grouped by the SSD concurrency they met.

        Returns:
            ``{(read concurrency bucket, write in flight): (jobs, io_us,
            bytes)}``; empty for backends that do not measure it.
        """
        return {}

    def debug_state(self) -> dict[str, Any]:
        """Return a small snapshot of in-flight work for stall diagnostics.

        Returns:
            Backend-specific counts; empty for backends that keep none.

        Async/thread-safety:
            May be read from a diagnostic thread; values are best-effort and
            must not drive control flow.
        """
        return {}

    @abstractmethod
    async def store(self, buffer: Any, moves: Sequence[Move]) -> StoreHandle:
        """Start copying staging-buffer ranges into L1 and/or L2.

        A move with ``l1_offset`` is snapshotted into that arena extent, which
        becomes the source of its L2 write when ``l2_offset`` is set. A move
        without ``l1_offset`` goes to L2 through a transient staging slice.

        Args:
            buffer: Readable staging buffer (CUDA array wrapper or bytes-like).
            moves: Ranges with ``buffer_offset`` and at least one tier offset.

        Returns:
            Handle whose ``staged`` and ``persisted`` futures report progress.

        Raises:
            ValueError: If a move is misaligned, out of bounds or names a tier
                the backend does not have. Nothing is submitted then.
            RuntimeError: If the layer is closing.

        Async/thread-safety:
            Returns once the work is scheduled; it does not wait for IO.
        """

    @abstractmethod
    async def load(
        self, buffer: Any, moves: Sequence[Move], source: LoadSource
    ) -> None:
        """Copy tier ranges into the staging buffer.

        With ``source="l1"`` every move reads its ``l1_offset``. With
        ``source="l2"`` every move reads its ``l2_offset``; a move that also
        has ``l1_offset`` fills that arena extent with the same read before
        the copy to ``buffer`` (promotion on read).

        Args:
            buffer: Writable destination buffer.
            moves: Ranges to read.
            source: Tier to read from.

        Raises:
            ValueError: If a move is invalid for ``source``.
            IOError: On short or failed reads.

        Async/thread-safety:
            Returns after every byte reached ``buffer`` (CUDA copies have
            completed on the device). Cancellation is deferred until the
            submitted IO and copies stopped using the extents.
        """

    async def promote(self, moves: Sequence[Move]) -> None:
        """Read L2 ranges into L1 arena extents without touching a GPU.

        Args:
            moves: Ranges with both ``l2_offset`` and ``l1_offset``.

        Raises:
            NotImplementedError: If the backend has no L1 arena.

        Async/thread-safety:
            Cancellation is deferred until the submitted reads completed.
        """
        raise NotImplementedError("transfer backend has no L1 arena to promote into")

    async def load_runs(
        self,
        buffer: Any,
        l1_offsets: Any,
        targets: Any,
        lengths: Any,
    ) -> None:
        """Copy many small L1 arena runs into a flat CUDA destination.

        Args:
            buffer: Flat CUDA byte destination, optionally exposing
                ``copy_stream_ptr``.
            l1_offsets: ``int64`` array of absolute arena offsets.
            targets: ``int64`` array of destination byte offsets.
            lengths: ``int64`` array of run lengths.

        Raises:
            NotImplementedError: If the backend has no L1 arena.

        Async/thread-safety:
            Returns after the copies completed on the device.
        """
        raise NotImplementedError("transfer backend has no L1 arena for layer runs")

    @abstractmethod
    async def drain(self) -> None:
        """Wait for every in-flight store, load, promotion and copy.

        Raises:
            RuntimeError: For the first background store failure not yet
                raised by an earlier ``drain``.

        Async/thread-safety:
            Cancelling ``drain`` does not cancel the work it waits for.
        """

    @abstractmethod
    async def close(self) -> None:
        """Drain outstanding work, then release files and pinned memory.

        Async/thread-safety:
            Idempotent. Operations started after ``close`` began raise
            ``RuntimeError``.
        """
