# SPDX-License-Identifier: Apache-2.0
"""Replacement by predicted next use from per-region reuse intervals."""

from collections.abc import Callable
import heapq
import time
from typing import Generic, TypeVar

from daser.replacement.base import ReplacementPolicy

K = TypeVar("K")

# Weight of a new observation in the default-interval moving average.
_EWMA_WEIGHT = 0.25


class NextUseReplacementPolicy(ReplacementPolicy[K], Generic[K]):
    """Evict the allocation whose predicted next use is furthest away.

    Each allocation is tied to the L2 region of its first inserted range.
    The region remembers when it was last used, so data promoted again after
    an eviction recovers its reuse interval. The prediction is the last use
    plus that interval; allocations without history use a moving average of
    observed intervals. Allocations idle for more than ``overdue_factor``
    intervals are presumed dead and evicted first, then the allocation with
    the latest predicted use. Under cyclic multi-session reuse this
    approximates Belady, where LRU evicts the session needed next.

    Args:
        region_bytes: L2 bytes per history region.
        overdue_factor: Idle intervals after which an allocation is dead.
        clock: Monotonic time source. A logical clock (for example one tick
            per request lookup) keeps long cold prefills from stretching
            intervals the way wall time does.

    Async/thread-safety:
        The cache owner must serialize calls. The policy owns only keys; it
        never releases buffers or decides whether a transfer may proceed.
    """

    def __init__(
        self,
        region_bytes: int = 256 * 2**20,
        overdue_factor: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if region_bytes <= 0 or overdue_factor <= 1.0:
            raise ValueError("region_bytes must be > 0 and overdue_factor > 1")
        self._region_bytes = region_bytes
        self._overdue = overdue_factor
        self._clock = clock
        self._default_interval = 0.0
        self._region_last: dict[int, float] = {}
        # key -> (region or None, last use, interval, version)
        self._keys: dict[K, tuple[int | None, float, float, int]] = {}
        self._version = 0
        # Lazy heaps of (-predicted use, -version, key) and (deadline,
        # version, key). Ties evict the later use, the tail of a prefix.
        self._furthest: list[tuple[float, int, K]] = []
        self._deadline: list[tuple[float, int, K]] = []

    def insert(self, key: K) -> None:
        """Track a key with no L2 placement.

        Args:
            key: Physical allocation identity.
        """
        if key not in self._keys:
            self._track(key, None, self._clock())

    def insert_range(self, key: K, file_offset: int, nbytes: int = 0) -> None:
        """Track a key under the L2 region of its first inserted range.

        Args:
            key: Physical allocation identity; later children are ignored.
            file_offset: L2 byte offset of the inserted range.
            nbytes: Unused; residency size does not change the prediction.
        """
        del nbytes
        if key not in self._keys:
            self._track(key, file_offset // self._region_bytes, self._clock())

    def access(self, key: K) -> None:
        """Record a real read and refresh the key's reuse interval.

        Args:
            key: Resident physical allocation identity; unknown keys are
                ignored.
        """
        entry = self._keys.get(key)
        if entry is not None:
            self._track(key, entry[0], self._clock())

    def remove(self, key: K) -> None:
        """Forget a key; its region keeps the last-use history.

        Args:
            key: Physical allocation identity to forget.
        """
        self._keys.pop(key, None)

    def evict(self) -> K | None:
        """Remove a dead allocation, else the one needed furthest ahead.

        Returns:
            Victim identity, or None when empty. The owner remains responsible
            for writer, lease, and DMA barriers before releasing its storage.
        """
        if not self._keys:
            return None
        if len(self._furthest) > 4 * len(self._keys) + 64:
            self._rebuild_heaps()
        now = self._clock()
        while self._deadline and not self._live(self._deadline[0]):
            heapq.heappop(self._deadline)
        if self._deadline and self._deadline[0][0] < now:
            key = heapq.heappop(self._deadline)[2]
        else:
            while not self._live(self._furthest[0]):
                heapq.heappop(self._furthest)
            key = heapq.heappop(self._furthest)[2]
        del self._keys[key]
        return key

    def _track(self, key: K, region: int | None, now: float) -> None:
        """Record a use of ``key`` at ``now`` and push its predictions."""
        entry = self._keys.get(key)
        last = entry[1] if entry is not None else self._region_last.get(region)
        if last is not None and now > last:
            interval = now - last
            self._default_interval = (
                interval
                if self._default_interval <= 0.0
                else self._default_interval
                + _EWMA_WEIGHT * (interval - self._default_interval)
            )
        elif entry is not None:
            interval = entry[2]
        else:
            interval = self._default_interval
        if region is not None:
            self._region_last[region] = now
        self._version += 1
        self._keys[key] = (region, now, interval, self._version)
        self._push(key, now, interval, self._version)

    def _push(self, key: K, last: float, interval: float, version: int) -> None:
        """Push one key's predicted use and dead-line onto the heaps."""
        heapq.heappush(self._furthest, (-(last + interval), -version, key))
        if interval > 0.0:
            deadline = last + self._overdue * interval
            heapq.heappush(self._deadline, (deadline, version, key))

    def _live(self, item: tuple[float, int, K]) -> bool:
        """Return whether a heap item matches the key's current version."""
        entry = self._keys.get(item[2])
        return entry is not None and entry[3] == abs(item[1])

    def _rebuild_heaps(self) -> None:
        """Drop stale heap items once they outnumber live keys."""
        self._furthest = []
        self._deadline = []
        for key, (_region, last, interval, version) in self._keys.items():
            self._push(key, last, interval, version)
