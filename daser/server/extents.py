# SPDX-License-Identifier: Apache-2.0
"""Address-ordered free ranges shared by the slot allocator and packed arena."""

from bisect import bisect_left


class FreeExtents:
    """Address-ordered free ranges of ``[start, end)``.

    Adjacent ranges are always coalesced, so the list length equals the number
    of holes between live allocations. Units are the caller's (slots or bytes).

    Args:
        start: First address of the managed space.
        end: Exclusive end address of the managed space.

    Async/thread-safety:
        Not thread-safe; owned by one server event loop.
    """

    def __init__(self, start: int, end: int) -> None:
        if end <= start:
            raise ValueError("free extent space must be non-empty")
        self.start = start
        self.end = end
        self.free = end - start
        self._starts: list[int] = [start]
        self._ends: list[int] = [end]

    @property
    def size(self) -> int:
        """Return the size of the managed space."""
        return self.end - self.start

    @property
    def holes(self) -> int:
        """Return the number of disjoint free ranges."""
        return len(self._starts)

    def reserve(self, length: int) -> int | None:
        """Carve ``length`` units from the first range that fits.

        Args:
            length: Positive number of units.

        Returns:
            Start address of the reserved range, or None when no free range
            is long enough.
        """
        for index, (start, end) in enumerate(
            zip(self._starts, self._ends, strict=True)
        ):
            if end - start < length:
                continue
            if end - start == length:
                del self._starts[index]
                del self._ends[index]
            else:
                self._starts[index] = start + length
            self.free -= length
            return start
        return None

    def release(self, offset: int, length: int) -> None:
        """Return ``[offset, offset + length)`` and merge it with neighbours.

        Args:
            offset: Start of a previously reserved range.
            length: Its length.
        """
        end = offset + length
        index = bisect_left(self._starts, offset)
        merge_prev = index > 0 and self._ends[index - 1] == offset
        merge_next = index < len(self._starts) and self._starts[index] == end
        if merge_prev and merge_next:
            self._ends[index - 1] = self._ends[index]
            del self._starts[index]
            del self._ends[index]
        elif merge_prev:
            self._ends[index - 1] = end
        elif merge_next:
            self._starts[index] = offset
        else:
            self._starts.insert(index, offset)
            self._ends.insert(index, end)
        self.free += length

    def take(self, offset: int, length: int) -> None:
        """Reserve the exact range ``[offset, offset + length)``.

        Used to rebuild allocator state from persisted allocations.

        Args:
            offset: Start of the range.
            length: Its length.

        Raises:
            ValueError: If the range is not entirely free.
        """
        end = offset + length
        index = bisect_left(self._starts, offset + 1) - 1
        if index < 0 or self._ends[index] < end or self._starts[index] > offset:
            raise ValueError(f"range [{offset}, {end}) is not free")
        start, stop = self._starts[index], self._ends[index]
        del self._starts[index]
        del self._ends[index]
        if end < stop:
            self._starts.insert(index, end)
            self._ends.insert(index, stop)
        if start < offset:
            self._starts.insert(index, start)
            self._ends.insert(index, offset)
        self.free -= length
