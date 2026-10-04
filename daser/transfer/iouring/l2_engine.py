# SPDX-License-Identifier: Apache-2.0

"""io_uring positioned I/O engine for the L2 SSD tier.

This engine owns the O_DIRECT file descriptor, the pool of native io_uring
rings, and the executor used to offload blocking completion waits. It holds no
tiering or L1 state: callers pass absolute byte offsets and memoryviews. Write
scheduling and the pending-write bookkeeping that ties a write to an L1 pool
slice stay in the transfer-layer orchestrator.
"""

# Standard
import asyncio
import concurrent.futures
import os
import threading
import time
from typing import Any

# First Party
from daser.transfer.iouring.native import NativeIOUring


class L2IoEngine:
    """Round-robin io_uring positioned reads/writes against the L2 store file.

    Args:
        path: pre-allocated or creatable L2 store file.
        l2_bytes: SSD-tier capacity; the file is truncated to this size.
        io_workers: number of native io_uring rings and executor threads.

    Async/thread-safety:
        Ring selection is serialized by an internal lock. Blocking syscalls are
        meant to be offloaded onto ``executor`` by the caller, which awaits the
        completion on its event loop.
    """

    def __init__(self, path: str, l2_bytes: int, io_workers: int) -> None:
        if l2_bytes <= 0:
            raise ValueError("l2_bytes must be positive")
        if io_workers <= 0:
            raise ValueError("io_workers must be positive")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a+b") as f:
            f.truncate(l2_bytes)
        self._fd = os.open(path, os.O_RDWR | os.O_DIRECT)
        self._urings = [NativeIOUring(entries=64) for _ in range(io_workers)]
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=io_workers,
            thread_name_prefix="daser-iouring",
        )
        self._uring_lock = threading.Lock()
        self._next_uring_index = 0
        # Microseconds read/write jobs waited for an executor thread and spent
        # in native IO, keyed "read_queue", "read_io", "write_queue",
        # "write_io"; guarded by ``_timing_lock`` (executor threads add).
        self._timing_lock = threading.Lock()
        self._timing_us = dict.fromkeys(
            ("read_queue", "read_io", "write_queue", "write_io"), 0
        )

    @property
    def ring_count(self) -> int:
        """Return the number of io_uring rings (used to bound read batches)."""
        return len(self._urings)

    @property
    def executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return the executor used to offload blocking io_uring waits."""
        return self._executor

    @property
    def timing_us(self) -> dict[str, int]:
        """Return cumulative executor queue and native IO microseconds.

        Returns:
            A copy keyed ``read_queue``, ``read_io``, ``write_queue`` and
            ``write_io``. Queue time runs from submission to an executor
            thread starting the job; IO time is the native call itself.

        Async/thread-safety:
            Safe from any thread; executor threads update under a lock.
        """
        with self._timing_lock:
            return dict(self._timing_us)

    def timed(self, kind: str, fn: Any) -> Any:
        """Wrap a blocking job so its queue and IO time are counted.

        Args:
            kind: ``"read"`` or ``"write"``.
            fn: blocking callable to run on the executor.

        Returns:
            A callable with the same arguments and result as ``fn``.
        """
        submitted = time.perf_counter()

        def run(*args: Any) -> Any:
            start = time.perf_counter()
            try:
                return fn(*args)
            finally:
                end = time.perf_counter()
                with self._timing_lock:
                    self._timing_us[f"{kind}_queue"] += int(
                        (start - submitted) * 1_000_000
                    )
                    self._timing_us[f"{kind}_io"] += int((end - start) * 1_000_000)

        return run

    def next_uring(self) -> NativeIOUring:
        """Return the next native io_uring instance for one L2 operation."""
        with self._uring_lock:
            uring = self._urings[self._next_uring_index]
            self._next_uring_index = (self._next_uring_index + 1) % len(self._urings)
            return uring

    def read_into(self, file_offset: int, dst: memoryview, uring: NativeIOUring) -> int:
        """Blocking io_uring L2 read into ``dst``.

        Args:
            file_offset: byte offset in the L2 store file.
            dst: writable memoryview to fill.
            uring: ring to submit on (from ``next_uring``).

        Returns:
            Number of bytes read.
        """
        return uring.read_into(self._fd, file_offset, dst)

    def write(self, file_offset: int, data: memoryview, uring: NativeIOUring) -> int:
        """Blocking io_uring L2 write of ``data``.

        Args:
            file_offset: byte offset in the L2 store file.
            data: readable memoryview to persist.
            uring: ring to submit on (from ``next_uring``).

        Returns:
            Number of bytes written.
        """
        return uring.write(self._fd, file_offset, data)

    def submit_read_batch(
        self, reads: list[tuple[int, memoryview]]
    ) -> tuple[asyncio.Future[int], list[asyncio.Future[int]]]:
        """Offload sparse reads and expose individual completion notifications.

        Args:
            reads: Nonempty aligned (file offset, writable buffer) pairs,
                bounded by the native ring capacity and per-read size limit.

        Returns:
            Executor completion and input-ordered per-read futures. Individual
            futures become ready as CQEs arrive, permitting immediate copies.

        Async/thread-safety:
            Call on the owning event loop. The caller must retain all buffers
            and await the executor completion before releasing them, even if
            a read fails or the enclosing request is cancelled. Do not cancel
            the returned futures. All future mutations occur on this loop.
        """
        loop = asyncio.get_running_loop()
        completions: list[asyncio.Future[int]] = [loop.create_future() for _ in reads]

        def publish(index: int, nbytes: int) -> None:
            if not completions[index].done():
                completions[index].set_result(nbytes)

        def on_complete(index: int, nbytes: int) -> None:
            loop.call_soon_threadsafe(publish, index, nbytes)

        submitted = loop.run_in_executor(
            self._executor,
            self.timed("read", self.next_uring().read_batch_into),
            self._fd,
            reads,
            on_complete,
        )

        def finish(future: asyncio.Future[int]) -> None:
            try:
                future.result()
            except BaseException as error:
                for completion in completions:
                    if not completion.done():
                        completion.set_exception(error)

        submitted.add_done_callback(finish)
        return submitted, completions

    def close(self) -> None:
        """Shut down the executor, close rings, and close the file descriptor."""
        self._executor.shutdown(wait=True)
        for uring in self._urings:
            uring.close()
        os.close(self._fd)
