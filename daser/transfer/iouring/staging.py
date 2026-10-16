# SPDX-License-Identifier: Apache-2.0

"""Async reservation front end of the pinned staging pool.

Staging carries L2 bytes that have no arena extent: stores the server did not
admit to L1 and L2 loads it did not promote. Every slice is released by the
IO or copy that owns it, never by another request's progress, so a waiter
only depends on in-flight IO and cannot deadlock.
"""

# Standard
import asyncio

# First Party
from daser.transfer.iouring import pinned_pool


class StagingPool:
    """Fixed pinned staging pool with waiting reservations.

    Args:
        capacity: Positive pool size in bytes (multiple of 4096).

    Async/thread-safety:
        Event-loop only: reserve, release and close from the transfer loop.
        A caller may wait in ``reserve`` only when every slice it already
        holds belongs to submitted IO; otherwise it must ``try_reserve`` and
        submit what it holds first.
    """

    def __init__(self, capacity: int) -> None:
        self._pool = pinned_pool.PinnedMemoryPool(capacity)
        self._waiters: list[asyncio.Future[None]] = []

    @property
    def capacity(self) -> int:
        """Return the pool size in bytes."""
        return self._pool.capacity

    @property
    def used_bytes(self) -> int:
        """Return the bytes held by live slices."""
        return self._pool.used_bytes

    @property
    def waiters(self) -> int:
        """Return the number of reservations currently waiting."""
        return len(self._waiters)

    def try_reserve(self, nbytes: int) -> pinned_pool.PinnedMemorySlice | None:
        """Reserve a slice without waiting.

        Args:
            nbytes: Positive byte count, at most ``capacity``.

        Returns:
            The slice, or None while the pool cannot hold it.
        """
        return self._pool.allocate(nbytes)

    async def reserve(self, nbytes: int) -> pinned_pool.PinnedMemorySlice:
        """Reserve a slice, waiting for in-flight IO to release space.

        Args:
            nbytes: Positive byte count, at most ``capacity``.

        Returns:
            The slice; close it through ``release``.

        Raises:
            ValueError: If ``nbytes`` exceeds the pool capacity.
        """
        if nbytes > self._pool.capacity:
            raise ValueError(
                f"staging request {nbytes} exceeds capacity {self._pool.capacity}"
            )
        while True:
            data = self._pool.allocate(nbytes)
            if data is not None:
                return data
            waiter = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)
            try:
                await waiter
            finally:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)

    def release(self, data: pinned_pool.PinnedMemorySlice) -> None:
        """Return a slice and wake every waiting reservation to retry.

        Args:
            data: Slice from this pool whose IO and copies have completed.
        """
        data.close()
        waiters, self._waiters = self._waiters, []
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)

    def close(self) -> None:
        """Free the pinned backing; idempotent. Call after every slice closed."""
        self._pool.close()
