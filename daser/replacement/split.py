# SPDX-License-Identifier: Apache-2.0

"""Bandwidth-split residency: keep a stable share of every prefix in L1.

Recency policies keep or drop whole sessions: a cyclic set of agent sessions
larger than L1 leaves some hits fully memory-resident and the rest fully on
SSD. Spreading the same resident bytes evenly over every active prefix keeps
each load's memory share stable, so no session falls back to a pure SSD load.

Allocations whose stable storage-offset hash falls below an eligibility
threshold enter a byte-budgeted protected segment; all others enter an LRU
probation segment that is always evicted first. The protected segment never
self-thrashes under cyclic reuse: when it is full, the oldest protected
allocation is demoted only if it is stale (not touched during two capacities'
worth of inserted bytes); otherwise the allocation with the highest hash value
yields its place. The protected set therefore converges to the lowest-hash
slice of every active prefix regardless of arrival order. Contention lowers
the eligibility threshold and spare budget raises it, so admission tracks
``budget / active working set`` without knowing either.
"""

# Standard
from collections import OrderedDict
from typing import Generic, TypeVar

# First Party
from daser.replacement.base import ReplacementPolicy

K = TypeVar("K")

_HASH_MULTIPLIER = 0x9E3779B97F4A7C15
_HASH_MASK = (1 << 64) - 1
_ELIGIBILITY_STEP = 0.98
_ELIGIBILITY_MAX = 0.95
_ELIGIBILITY_MIN = 0.01
# A protected key is stale after this many capacities of inserted bytes with
# no access. Cyclic reuse distance is bounded by the unprotected part of the
# active working set, which stays under two capacities for a set that is at
# most twice L1; finished prefixes still leave within that bounded age.
_STALE_CAPACITIES = 2


def _offset_class(file_offset: int, granularity: int) -> float:
    """Map a storage offset to a stable pseudo-uniform value in ``[0, 1)``.

    Args:
        file_offset: L2 byte offset that names the cached range.
        granularity: Bytes per hashed unit; offsets inside one unit share a class.

    Returns:
        Fibonacci-hash value of ``file_offset // granularity`` scaled to [0, 1).
    """
    unit = (file_offset // granularity) & _HASH_MASK
    return ((unit * _HASH_MULTIPLIER) & _HASH_MASK) / float(1 << 64)


class SplitResidencyPolicy(ReplacementPolicy[K], Generic[K]):
    """Byte-budgeted protected segment chosen by offset hash, LRU probation.

    Args:
        protect_fraction: Protected byte budget as a share of
            ``capacity_bytes``, in ``(0, 1)``.
        capacity_bytes: Cache capacity; sets the budget and the staleness age.
        granularity: Bytes per hashed offset unit.

    Async/thread-safety:
        Not thread-safe by itself. Owners call it from one event loop or guard
        it with their own lock, like the other replacement policies.
    """

    def __init__(
        self,
        protect_fraction: float,
        capacity_bytes: int,
        granularity: int = 4096,
    ) -> None:
        if not 0.0 < protect_fraction < 1.0:
            raise ValueError("protect_fraction must be in (0, 1)")
        if capacity_bytes <= 0 or granularity <= 0:
            raise ValueError("capacity_bytes and granularity must be positive")
        self._budget = int(protect_fraction * capacity_bytes)
        self._stale_age = _STALE_CAPACITIES * capacity_bytes
        self._granularity = granularity
        self._eligibility = min(protect_fraction, _ELIGIBILITY_MAX)
        self._protected: OrderedDict[K, None] = OrderedDict()
        self._probation: OrderedDict[K, None] = OrderedDict()
        self._sizes: dict[K, int] = {}
        self._last_access: dict[K, int] = {}
        self._classes: dict[K, float] = {}
        self._protected_bytes = 0
        self._clock = 0

    @property
    def eligibility(self) -> float:
        """Return the current offset-hash threshold for protected admission."""
        return self._eligibility

    @property
    def protected_bytes(self) -> int:
        """Return the bytes currently tracked in the protected segment."""
        return self._protected_bytes

    def insert(self, key: K) -> None:
        """Insert or refresh a key without placement hints (probation segment).

        Args:
            key: Cache key to track.
        """
        if self._refresh(key):
            return
        self._track(key, 0, protected=False)

    def insert_range(self, key: K, file_offset: int, nbytes: int = 0) -> None:
        """Insert a key or grow a tracked one, classifying new keys by offset.

        Args:
            key: Cache key to track.
            file_offset: Stable L2 byte offset of the key's first cached range.
            nbytes: Bytes added to the key's resident size by this insert.
        """
        self._clock += nbytes
        if key in self._sizes:
            self._sizes[key] += nbytes
            if key in self._protected:
                self._protected_bytes += nbytes
            self._refresh(key)
            if key in self._protected:
                self._enforce_budget(key)
            return
        value = _offset_class(file_offset, self._granularity)
        self._classes[key] = value
        eligible = value < self._eligibility
        self._track(key, nbytes, protected=eligible)
        if eligible:
            self._enforce_budget(key)

    def access(self, key: K) -> None:
        """Mark an existing key as recently used within its segment.

        Args:
            key: Cache key that was read or written.
        """
        self._refresh(key)

    def remove(self, key: K) -> None:
        """Remove a key from replacement tracking.

        Args:
            key: Cache key to forget.
        """
        nbytes = self._sizes.pop(key, None)
        if nbytes is None:
            return
        del self._last_access[key]
        self._classes.pop(key, None)
        if key in self._protected:
            del self._protected[key]
            self._protected_bytes -= nbytes
        else:
            self._probation.pop(key, None)

    def evict(self) -> K | None:
        """Return the oldest probation key, else the oldest protected key.

        Returns:
            Victim key, or None when both segments are empty.
        """
        for segment in (self._probation, self._protected):
            if segment:
                key = next(iter(segment))
                self.remove(key)
                return key
        return None

    def _track(self, key: K, nbytes: int, *, protected: bool) -> None:
        """Start tracking a new key at the MRU end of one segment."""
        self._sizes[key] = nbytes
        self._last_access[key] = self._clock
        if protected:
            self._protected[key] = None
            self._protected_bytes += nbytes
        else:
            self._probation[key] = None

    def _refresh(self, key: K) -> bool:
        """Move a tracked key to its segment's MRU end and stamp its access.

        Args:
            key: Cache key to refresh.

        Returns:
            True when the key was already tracked.
        """
        for segment in (self._protected, self._probation):
            if key in segment:
                segment.move_to_end(key)
                self._last_access[key] = self._clock
                return True
        return False

    def _enforce_budget(self, key: K) -> None:
        """Restore the protected byte budget after ``key`` entered or grew.

        Stale protected keys are demoted to the probation LRU end. Among live
        keys, the one with the highest hash value (possibly ``key`` itself)
        moves to the probation MRU end, and the eligibility threshold drops;
        spare budget raises it.
        """
        while self._protected_bytes > self._budget:
            oldest = next(iter(self._protected))
            if oldest != key and (
                self._clock - self._last_access[oldest] >= self._stale_age
            ):
                self._demote(oldest, to_lru_end=True)
                continue
            self._eligibility = max(
                _ELIGIBILITY_MIN, self._eligibility * _ELIGIBILITY_STEP
            )
            victim = max(self._protected, key=self._classes.__getitem__)
            self._demote(victim, to_lru_end=False)
            if victim == key:
                return
        if self._protected_bytes < 0.9 * self._budget:
            self._eligibility = min(
                _ELIGIBILITY_MAX, self._eligibility / _ELIGIBILITY_STEP
            )

    def _demote(self, key: K, *, to_lru_end: bool) -> None:
        """Move a protected key into probation."""
        del self._protected[key]
        self._protected_bytes -= self._sizes[key]
        self._probation[key] = None
        if to_lru_end:
            self._probation.move_to_end(key, last=False)
