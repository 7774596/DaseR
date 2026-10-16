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
from daser.transfer.tasks import settle

#: Write-gate granularity: a gated write is issued in pieces of this size and
#: re-checks for in-flight reads between pieces.
_GATE_CHUNK_BYTES = 16 << 20
#: Longest a gated write piece waits for reads to drain before it proceeds,
#: so a continuous read stream cannot starve writes that hold L1 memory.
_GATE_MAX_WAIT_S = 0.05
#: Largest read-concurrency bucket; deeper concurrency is counted here.
_CONTENTION_MAX_BUCKET = 32


def _contention_bucket(level: int) -> int:
    """Return the power-of-two bucket (1, 2, 4, ... 32) holding ``level``."""
    bucket = 1
    while bucket < level and bucket < _CONTENTION_MAX_BUCKET:
        bucket <<= 1
    return bucket


def _gate_pieces(
    writes: list[tuple[int, memoryview]], capacity: int
) -> list[list[tuple[int, memoryview]]]:
    """Pack extents into gated write pieces.

    Args:
        writes: (file offset, source) pairs.
        capacity: most IOs one native batch takes.

    Returns:
        Native batches of at most ``capacity`` IOs and ``_GATE_CHUNK_BYTES``
        in total, in input order; longer extents are split at that size.
    """
    pieces: list[list[tuple[int, memoryview]]] = []
    batch: list[tuple[int, memoryview]] = []
    nbytes = 0
    for file_offset, data in writes:
        view = memoryview(data).cast("B")
        for start in range(0, len(view), _GATE_CHUNK_BYTES):
            part = view[start : start + _GATE_CHUNK_BYTES]
            if batch and (
                len(batch) == capacity or nbytes + len(part) > _GATE_CHUNK_BYTES
            ):
                pieces.append(batch)
                batch, nbytes = [], 0
            batch.append((file_offset + start, part))
            nbytes += len(part)
    if batch:
        pieces.append(batch)
    return pieces


class L2IoEngine:
    """Round-robin io_uring positioned reads/writes against the L2 store file.

    Args:
        path: existing L2 store file, created by the server
            (``daser.server.store_file``); it is opened, never created or
            resized.
        l2_bytes: SSD-tier capacity; the file must be at least this large.
        io_workers: number of native io_uring rings and executor threads.
        write_gate: Research switch. Writes run on their own executor; each
            ``write_batch`` call is one job issuing native batches of at most
            ``_GATE_CHUNK_BYTES`` in turn, and before each batch it waits, at
            most ``_GATE_MAX_WAIT_S``, while any read job is in native IO,
            because concurrent O_DIRECT writes cut the read bandwidth of the
            shared devices by half or more.

    Async/thread-safety:
        Ring selection is serialized by an internal lock. Blocking syscalls are
        meant to be offloaded onto ``executor`` (``write_executor`` for writes)
        by the caller, which awaits the completion on its event loop.
    """

    def __init__(
        self, path: str, l2_bytes: int, io_workers: int, write_gate: bool = False
    ) -> None:
        if l2_bytes <= 0:
            raise ValueError("l2_bytes must be positive")
        if io_workers <= 0:
            raise ValueError("io_workers must be positive")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"L2 store file {path} does not exist; the server creates it "
                "before starting the transfer layer"
            )
        existing = os.path.getsize(path)
        if existing < l2_bytes:
            raise ValueError(
                f"L2 store file {path} has size {existing}, need {l2_bytes}"
            )
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
            ("read_queue", "read_io", "write_queue", "write_io", "write_gate"), 0
        )
        self._write_gate = bool(write_gate)
        self._write_executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=io_workers, thread_name_prefix="daser-iouring-w"
            )
            if self._write_gate
            else self._executor
        )
        # Read jobs currently in native IO; gated writes wait on the condition
        # until it reaches zero.
        self._reads_cond = threading.Condition()
        self._reads_active = 0
        # Read jobs by (concurrency bucket, write in flight) at job start:
        # [jobs, native IO microseconds, bytes]; guarded by ``_timing_lock``,
        # as is the count of write jobs in native IO.
        self._read_contention: dict[tuple[int, bool], list[int]] = {}
        self._writes_active = 0

    @property
    def ring_count(self) -> int:
        """Return the number of io_uring rings (used to bound read batches)."""
        return len(self._urings)

    @property
    def executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return the executor used to offload blocking io_uring reads."""
        return self._executor

    @property
    def write_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return the executor for blocking writes (``executor`` ungated)."""
        return self._write_executor

    @property
    def timing_us(self) -> dict[str, int]:
        """Return cumulative executor queue and native IO microseconds.

        Returns:
            A copy keyed ``read_queue``, ``read_io``, ``write_queue``,
            ``write_io`` and ``write_gate``. Queue time runs from submission
            to an executor thread starting the job; IO time is the native call
            itself, including ``write_gate`` time writes waited for reads.

        Async/thread-safety:
            Safe from any thread; executor threads update under a lock.
        """
        with self._timing_lock:
            return dict(self._timing_us)

    @property
    def read_contention(self) -> dict[tuple[int, bool], tuple[int, int, int]]:
        """Return cumulative read jobs grouped by the contention they met.

        A read job is classified when it enters native IO by how many read
        jobs were then in native IO, itself included, rounded up to a power
        of two and capped at 32, and by whether a write job was in native IO.
        The per-group IO time over bytes gives read latency and bandwidth as
        a function of SSD concurrency.

        Returns:
            ``{(bucket, write_in_flight): (jobs, io_us, bytes)}``.

        Async/thread-safety:
            Safe from any thread; executor threads update under a lock.
        """
        with self._timing_lock:
            return {key: tuple(value) for key, value in self._read_contention.items()}

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
            reading = kind == "read"
            key = None
            if reading:
                with self._reads_cond:
                    self._reads_active += 1
                    level = self._reads_active
                with self._timing_lock:
                    key = (_contention_bucket(level), self._writes_active > 0)
            else:
                with self._timing_lock:
                    self._writes_active += 1
            nbytes = 0
            try:
                result = fn(*args)
                if isinstance(result, int):
                    nbytes = result
                return result
            finally:
                if reading:
                    with self._reads_cond:
                        self._reads_active -= 1
                        if not self._reads_active:
                            self._reads_cond.notify_all()
                end = time.perf_counter()
                io_us = int((end - start) * 1_000_000)
                with self._timing_lock:
                    self._timing_us[f"{kind}_queue"] += int(
                        (start - submitted) * 1_000_000
                    )
                    self._timing_us[f"{kind}_io"] += io_us
                    if key is None:
                        self._writes_active -= 1
                    else:
                        group = self._read_contention.setdefault(key, [0, 0, 0])
                        group[0] += 1
                        group[1] += io_us
                        group[2] += nbytes

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

    async def write_batch(self, writes: list[tuple[int, memoryview]]) -> int:
        """Persist many extents with as few executor jobs as possible.

        Ungated, the writes are submitted in native batches of up to one
        ring's capacity, one executor job per batch, so they do not queue
        ahead of reads one job per extent. With ``write_gate`` the call is a
        single job that issues gate-sized native batches one after another,
        waiting for in-flight reads before each.

        Args:
            writes: Aligned (file offset, source) pairs; sources follow the
                ``NativeIOUring.write_batch_from`` buffer contract.

        Returns:
            Bytes written.

        Raises:
            IOError: On a short write (raised after every job ended).
            OSError: On a failed write.

        Async/thread-safety:
            Call on the owning event loop. The caller must keep every source
            unchanged until this returns, also on errors and cancellation.
        """
        if not writes:
            return 0
        loop = asyncio.get_running_loop()
        if self._write_gate:
            jobs = [
                loop.run_in_executor(
                    self._write_executor,
                    self.timed("write", self._write_gated),
                    _gate_pieces(writes, self._urings[0].batch_capacity),
                    self.next_uring(),
                )
            ]
        else:
            jobs = []
            for start in range(0, len(writes), self._urings[0].batch_capacity):
                uring = self.next_uring()
                batch = writes[start : start + uring.batch_capacity]
                jobs.append(
                    loop.run_in_executor(
                        self._write_executor,
                        self.timed("write", uring.write_batch_from),
                        self._fd,
                        batch,
                    )
                )
        await settle(jobs)
        return sum(job.result() for job in jobs)

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
        """Shut down the executors, close rings, and close the file descriptor."""
        self._executor.shutdown(wait=True)
        if self._write_executor is not self._executor:
            self._write_executor.shutdown(wait=True)
        for uring in self._urings:
            uring.close()
        os.close(self._fd)

    def _write_gated(
        self, pieces: list[list[tuple[int, memoryview]]], uring: NativeIOUring
    ) -> int:
        """Blocking: issue ``pieces`` in turn, each after the read gate.

        Returns:
            Bytes written.
        """
        written = 0
        for piece in pieces:
            waited = self._wait_reads_drained()
            if waited:
                with self._timing_lock:
                    self._timing_us["write_gate"] += waited
            written += uring.write_batch_from(self._fd, piece)
        return written

    def _wait_reads_drained(self) -> int:
        """Block until no read job is in native IO, bounded by the gate cap.

        Returns:
            Microseconds waited.
        """
        with self._reads_cond:
            if not self._reads_active:
                return 0
            start = time.perf_counter()
            self._reads_cond.wait_for(
                lambda: not self._reads_active, timeout=_GATE_MAX_WAIT_S
            )
            return int((time.perf_counter() - start) * 1_000_000)
