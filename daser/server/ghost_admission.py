# SPDX-License-Identifier: Apache-2.0

"""Ghost admission to the external tier (research switch).

When tenants share the external tier, a tenant whose prompts are mostly
one-time content (a scan) pushes out a tenant with a small, heavily reused
working set. A ghost filter in front of the store keeps scans out: the first
store of an unseen key only records the key as a ghost and writes nothing.
The key is admitted when it is stored again while still a ghost. The packed
arena and its FIFO replacement are unchanged.

A fixed filter costs every first-reuse miss, which hurts a single tenant whose
whole working set fits. The adaptive mode decides with a miniature simulation
(Waldspurger et al., ATC'17): two shadow FIFO caches of ``capacity / sample``
keys see a ``1 / sample`` key sample of every looked-up prompt. One always
admits and the other admits on ghost. The real filter is on while the ghost
shadow has fewer decayed misses. The shadows replace in insertion order like
the packed arena; LRU shadows overrate always-admit and keep the filter off
when the arena is losing a reused working set to scans.
"""

from __future__ import annotations

from collections import OrderedDict
import os
import zlib

# Research switch: "fixed" always filters first stores through the ghost list;
# "adaptive" filters only while the miniature simulation favours it. Unset
# admits every store.
GHOST_ADMIT_ENV = "DASER_GHOST_ADMIT"
GHOST_ADMIT_MODES = ("fixed", "adaptive")

# Shadow caches see one key in this many.
DEFAULT_SAMPLE = 16
# Per-prompt decay of shadow miss counts, so the choice follows phase changes.
_MISS_DECAY = 0.999


def _insert(cache: OrderedDict[str, None], key: str, capacity: int) -> None:
    """Insert ``key`` as the newest entry, dropping the oldest past capacity."""
    cache[key] = None
    cache.move_to_end(key)
    if len(cache) > capacity:
        cache.popitem(last=False)


class GhostAdmission:
    """Admit a chunk key to the external tier on its second store.

    Args:
        capacity: chunks the external tier holds; the ghost list keeps as
            many keys.
        adaptive: filter only while the miniature simulation favours it.
        sample: one key in ``sample`` feeds the shadow caches.

    Async/thread-safety:
        Not synchronized; call only from the server event loop.
    """

    def __init__(
        self, capacity: int, *, adaptive: bool = False, sample: int = DEFAULT_SAMPLE
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if sample <= 0:
            raise ValueError("sample must be positive")
        self.capacity = capacity
        self.adaptive = adaptive
        self.sample = sample
        self.admitted = 0
        self.denied = 0
        self._ghosts: OrderedDict[str, None] = OrderedDict()
        self._shadow_capacity = max(1, capacity // sample)
        # Shadow tiers: [always admit, admit on ghost], and the latter's ghosts.
        self._shadows: tuple[OrderedDict[str, None], OrderedDict[str, None]] = (
            OrderedDict(),
            OrderedDict(),
        )
        self._shadow_ghosts: OrderedDict[str, None] = OrderedDict()
        self._shadow_misses = [0.0, 0.0]

    @classmethod
    def from_env(cls, capacity: int) -> GhostAdmission | None:
        """Build the filter selected by ``DASER_GHOST_ADMIT``.

        Args:
            capacity: chunks the external tier holds.

        Returns:
            The configured filter, or None when the switch is unset.

        Raises:
            ValueError: for an unknown mode.
        """
        mode = os.environ.get(GHOST_ADMIT_ENV, "")
        if not mode:
            return None
        if mode not in GHOST_ADMIT_MODES:
            raise ValueError(f"{GHOST_ADMIT_ENV} must be one of {GHOST_ADMIT_MODES}")
        return cls(capacity, adaptive=mode == "adaptive")

    @property
    def filtering(self) -> bool:
        """Return whether first stores are currently held back as ghosts."""
        if not self.adaptive:
            return True
        return self._shadow_misses[1] < self._shadow_misses[0]

    def observe(self, chunk_keys: list[str]) -> None:
        """Feed one looked-up prompt's keys to the shadow caches.

        Call once per distinct prompt; a no-op for the fixed filter.

        Args:
            chunk_keys: keys of the prompt's blocks in match order.
        """
        if not self.adaptive:
            return
        admit_all, filtered = self._shadows
        for key in chunk_keys:
            if zlib.crc32(key.encode()) % self.sample:
                continue
            if key not in admit_all:
                self._shadow_misses[0] += 1
                _insert(admit_all, key, self._shadow_capacity)
            if key in filtered:
                continue
            self._shadow_misses[1] += 1
            if self._shadow_ghosts.pop(key, False) is None:
                _insert(filtered, key, self._shadow_capacity)
            else:
                _insert(self._shadow_ghosts, key, self._shadow_capacity)
        self._shadow_misses[0] *= _MISS_DECAY
        self._shadow_misses[1] *= _MISS_DECAY

    def admit(self, chunk_key: str) -> bool:
        """Decide whether the first writer of an unstored key may write it.

        Args:
            chunk_key: key about to be allocated.

        Returns:
            True to store the chunk. False records the key as a ghost; the
            caller writes nothing and a later store of the key is admitted
            while the ghost is remembered.
        """
        if self._ghosts.pop(chunk_key, False) is None or not self.filtering:
            self.admitted += 1
            return True
        _insert(self._ghosts, chunk_key, self.capacity)
        self.denied += 1
        return False
