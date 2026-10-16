# SPDX-License-Identifier: Apache-2.0

"""kvikio/cuFile transfer backend: direct L2 IO between GPU buffers and SSD.

GDS has no pinned L1 arena, so it implements the L2 subset of
``TransferLayer``: ``store`` (``staged`` and ``persisted`` are the same
completion), ``load(source="l2")``, ``drain`` and ``close``.
"""

# Standard
import asyncio
from collections.abc import Coroutine, Sequence
import enum
import os
from typing import Any

# Third Party
import kvikio
import kvikio.cufile
import kvikio.defaults

# First Party
from daser.logging import init_logger
from daser.transfer.base import (
    LoadSource,
    Move,
    StoreHandle,
    TransferLayer,
    TransferStats,
)
from daser.transfer.moves import (
    MoveKind,
    buffer_nbytes,
    coalesce_moves,
    validate_moves,
)
from daser.transfer.tasks import consume_exception, resolve, settle

logger = init_logger(__name__)

_COMPAT_READ_TASK_SIZE = 4 << 20
_COMPAT_WRITE_TASK_SIZE = 64 << 20


class TransferBackend(enum.Enum):
    """Active IO backend for GDSTransferLayer."""

    GDS = "gds"
    COMPAT = "compat"


class GDSTransferLayer(TransferLayer):
    """Async NVMe<->GPU IO using kvikio.

    Args:
        path: Existing store file.
        l2_bytes: File bytes addressable by ``l2_offset``; defaults to the
            file size.
        nthreads: kvikio thread-pool size used in compat mode.

    Async/thread-safety:
        Call from one event loop. kvikio futures are awaited on the default
        executor. The backend is selected once during construction.
    """

    def __init__(
        self, path: str, *, l2_bytes: int | None = None, nthreads: int = 64
    ) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Store file not found: {path}")
        size = os.path.getsize(path)
        self._l2_bytes = size if l2_bytes is None else l2_bytes
        if not 0 < self._l2_bytes <= size:
            raise ValueError(f"l2_bytes must be in (0, {size}], got {l2_bytes}")

        kvikio.defaults.set("compat_mode", kvikio.CompatMode.OFF)
        try:
            self._file = kvikio.cufile.CuFile(path, "r+")
            self._backend = TransferBackend.GDS
        except RuntimeError as exc:
            logger.warning(
                "[TRANSFER:gds] direct cuFile open failed, falling back to compat: %s",
                exc,
            )
            kvikio.defaults.set("compat_mode", kvikio.CompatMode.ON)
            self._backend = TransferBackend.COMPAT
            kvikio.defaults.set("num_threads", nthreads)
            self._file = kvikio.cufile.CuFile(path, "r+")
        self._stats = TransferStats()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._errors: list[BaseException] = []
        self._close_task: asyncio.Task[None] | None = None

        logger.info(
            "[TRANSFER:gds] backend=%s nthreads=%d task_size=%d path=%s",
            self._backend.name,
            nthreads,
            kvikio.defaults.get("task_size"),
            path,
        )

    @property
    def backend(self) -> TransferBackend:
        """The active IO backend."""
        return self._backend

    @property
    def stats(self) -> TransferStats:
        """Return the live cumulative counters."""
        return self._stats

    def debug_state(self) -> dict[str, Any]:
        """Return the number of in-flight operations."""
        return {"in_flight": len(self._tasks), "closing": self._close_task is not None}

    async def store(self, buffer: Any, moves: Sequence[Move]) -> StoreHandle:
        """Write staging-buffer ranges to L2.

        Args:
            buffer: CuPy source buffer.
            moves: Ranges with ``l2_offset``; ``l1_offset`` is rejected.

        Returns:
            Handle whose ``staged`` and ``persisted`` are one future that
            completes when every write ended.

        Raises:
            ValueError: For an invalid move; nothing is submitted.
            RuntimeError: If the layer is closing.
        """
        self._check_open()
        self._validate(moves, "store", buffer)
        done = asyncio.get_running_loop().create_future()
        done.add_done_callback(consume_exception)
        self._spawn(self._run_store(buffer, list(moves), done))
        return StoreHandle(staged=done, persisted=done)

    async def load(
        self, buffer: Any, moves: Sequence[Move], source: LoadSource
    ) -> None:
        """Read L2 ranges into ``buffer``.

        Args:
            buffer: CuPy destination buffer.
            moves: Ranges with ``l2_offset``.
            source: Must be ``"l2"``.

        Raises:
            ValueError: For ``source="l1"`` or an invalid move.
            RuntimeError: If the layer is closing.
        """
        if source != "l2":
            raise ValueError("GDS transfer has no L1 arena; load source must be 'l2'")
        self._check_open()
        self._validate(moves, "l2", buffer)
        runs = coalesce_moves(moves, ("l2", "buffer"))
        await settle([self._spawn(self._io(buffer, runs, write=False))])

    async def drain(self) -> None:
        """Wait for in-flight IO, then raise the first store failure.

        Raises:
            RuntimeError: For the first store failure since the previous drain.
        """
        while self._tasks:
            await asyncio.wait(set(self._tasks))
        if self._errors:
            first = self._errors[0]
            self._errors.clear()
            raise RuntimeError("asynchronous GDS store failed") from first

    async def close(self) -> None:
        """Drain all IO, then close the kvikio file handle; idempotent."""
        if self._close_task is None:
            self._close_task = asyncio.get_running_loop().create_task(self._run_close())
        await asyncio.shield(self._close_task)

    def _check_open(self) -> None:
        """Raise if close began."""
        if self._close_task is not None:
            raise RuntimeError("GDS transfer layer is closing")

    def _validate(self, moves: Sequence[Move], kind: MoveKind, buffer: Any) -> None:
        """Validate a move batch against the L2 file (no L1 arena)."""
        validate_moves(
            moves,
            kind,
            buffer_bytes=buffer_nbytes(buffer),
            l1_bytes=0,
            l2_bytes=self._l2_bytes,
        )

    def _spawn(self, body: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """Start a tracked task that ``drain`` and ``close`` wait for."""
        task = asyncio.get_running_loop().create_task(body)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run_store(
        self, buffer: Any, moves: list[Move], done: asyncio.Future[None]
    ) -> None:
        """Write every move and resolve the handle future."""
        try:
            await self._io(buffer, coalesce_moves(moves, ("buffer", "l2")), write=True)
        except asyncio.CancelledError as exc:
            resolve(done, exc)
            raise
        except Exception as exc:
            self._errors.append(exc)
            logger.warning("[TRANSFER:gds] store failed: %r", exc)
            resolve(done, exc)
        else:
            resolve(done, None)

    async def _io(self, buffer: Any, runs: list[Move], *, write: bool) -> None:
        """Submit every run to kvikio, then wait for all completions."""
        if not runs:
            return
        loop = asyncio.get_running_loop()
        if self._backend == TransferBackend.COMPAT:
            kvikio.defaults.set(
                "task_size",
                _COMPAT_WRITE_TASK_SIZE if write else _COMPAT_READ_TASK_SIZE,
            )
        submit = self._file.pwrite if write else self._file.pread
        futures = []
        submit_error: Exception | None = None
        try:
            for run in runs:
                futures.append(
                    submit(
                        buffer[run.buffer_offset : run.buffer_offset + run.nbytes],
                        run.nbytes,
                        run.l2_offset,
                    )
                )
        except Exception as exc:
            submit_error = exc
        # Already submitted IO keeps using the buffer: wait for it even when a
        # later submission failed.
        waits = [loop.run_in_executor(None, future.get) for future in futures]
        await settle(waits)
        if submit_error is not None:
            raise submit_error
        for run, wait in zip(runs, waits, strict=True):
            result = wait.result()
            if int(result) != run.nbytes:
                raise IOError(f"short GDS IO: {result} != {run.nbytes}")
        if write:
            self._stats.l2_writes += len(runs)
            self._stats.l2_write_bytes += sum(run.nbytes for run in runs)
        else:
            self._stats.l2_reads += len(runs)
            self._stats.l2_read_bytes += sum(run.nbytes for run in runs)

    async def _run_close(self) -> None:
        """Shutdown body shared by every ``close`` caller."""
        while self._tasks:
            await asyncio.wait(set(self._tasks))
        for error in self._errors:
            logger.warning("[TRANSFER:gds] unreported store failure: %r", error)
        self._errors.clear()
        self._file.close()
        logger.debug("[TRANSFER:gds] file closed")
