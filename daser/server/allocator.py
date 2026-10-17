# SPDX-License-Identifier: Apache-2.0

"""Two-level segregated-fit (TLSF) byte allocator for one storage tier.

Every offset, length, free block and split remainder is a multiple of the
allocator grain, so any range the allocator hands out or keeps free stays
usable for O_DIRECT and GDS transfers. Internally sizes are counted in grain
units.

Free blocks live in size classes: the first level is ``floor(log2(units))``
and the second level splits each power-of-two range into 8 equal
subdivisions (sizes below 8 units get one exact class each). Two integer
bitmaps record which classes are non-empty. ``start -> length`` and
``end -> start`` maps find neighbours for O(1) coalescing.

A one-byte-per-grain free map (1 MiB per 64 GiB of capacity at the default
64 KiB grain) lets ``release`` reject double frees and lets ``take`` find
the free block covering a range. Its checks are C-level byte scans bounded
by the extent (``release``) or the covering free block (``take``), so no
operation depends on the number of free blocks.
"""

#: Second-level subdivisions per power-of-two range (log2 = 3).
_SL_LOG2 = 3
_SL_COUNT = 1 << _SL_LOG2
#: O_DIRECT / GDS alignment every grain must be a multiple of.
_IO_ALIGNMENT = 4096


def _size_class(units: int) -> tuple[int, int]:
    """Return the (first, second) level class holding a block of ``units``.

    Args:
        units: positive block size in grain units.

    Returns:
        ``(fl, sl)``. ``fl == 0`` holds the exact sizes 1..7; for
        ``fl >= 1`` the class covers
        ``[(8 + sl) << (fl - 1), (9 + sl) << (fl - 1))``.
    """
    if units < _SL_COUNT:
        return 0, units
    top = units.bit_length() - 1
    shift = top - _SL_LOG2
    return top - _SL_LOG2 + 1, (units >> shift) - _SL_COUNT


def _good_fit_class(units: int) -> tuple[int, int]:
    """Return the smallest class whose every block holds ``units``.

    Args:
        units: positive request size in grain units.

    Returns:
        ``(fl, sl)`` of the first class whose lower bound is ``>= units``.
    """
    if units < _SL_COUNT:
        return 0, units
    step = 1 << (units.bit_length() - 1 - _SL_LOG2)
    return _size_class(units + step - 1)


class TlsfAllocator:
    """Grain-aligned extent allocator over ``[0, size)`` bytes.

    Args:
        size: managed capacity in bytes; positive multiple of ``grain``.
        grain: allocation unit in bytes; a power of two and a multiple of
            4096.

    Raises:
        ValueError: if ``grain`` or ``size`` is invalid.

    Async/thread-safety:
        Not synchronized. Callers use one allocator from a single thread
        (the server event loop).
    """

    def __init__(self, size: int, grain: int) -> None:
        if grain <= 0 or grain & (grain - 1) or grain % _IO_ALIGNMENT:
            raise ValueError(
                f"grain must be a power of two multiple of {_IO_ALIGNMENT}, got {grain}"
            )
        if size <= 0 or size % grain:
            raise ValueError(
                f"size must be a positive multiple of grain {grain}, got {size}"
            )
        self._grain = grain
        self._units = size // grain
        max_fl = _size_class(self._units)[0]
        self._fl_bitmap = 0
        self._sl_bitmaps = [0] * (max_fl + 1)
        # (fl, sl) -> insertion-ordered set of block starts in that class.
        self._bins: dict[tuple[int, int], dict[int, None]] = {}
        self._length_of: dict[int, int] = {}
        self._start_of_end: dict[int, int] = {}
        # One byte per grain unit: 1 = free, 0 = reserved.
        self._free_map = bytearray(b"\x01") * self._units
        self._free_units = 0
        self._insert_free(0, self._units)

    @property
    def size(self) -> int:
        """Return the managed capacity in bytes."""
        return self._units * self._grain

    @property
    def grain(self) -> int:
        """Return the allocation unit in bytes."""
        return self._grain

    @property
    def free_bytes(self) -> int:
        """Return the total free bytes."""
        return self._free_units * self._grain

    @property
    def holes(self) -> int:
        """Return the number of maximal free extents."""
        return len(self._length_of)

    @property
    def largest_free(self) -> int:
        """Return the length in bytes of the largest free extent (0 if none).

        Async/thread-safety:
            Scans only the highest non-empty size class.
        """
        if not self._fl_bitmap:
            return 0
        fl = self._fl_bitmap.bit_length() - 1
        sl = self._sl_bitmaps[fl].bit_length() - 1
        bucket = self._bins[(fl, sl)]
        return max(self._length_of[start] for start in bucket) * self._grain

    def reserve(self, length: int) -> int | None:
        """Allocate ``length`` contiguous bytes.

        Uses a good fit (the first non-empty class whose blocks all fit); if
        none exists, scans the request's own class for a block that fits, so
        a request fails only when no free extent is long enough.

        Args:
            length: positive multiple of ``grain``.

        Returns:
            Byte offset of the reserved extent, or ``None`` if no free extent
            is long enough.

        Raises:
            ValueError: if ``length`` is not a positive multiple of ``grain``.
        """
        units = self._to_units(length, "length")
        found = self._find_class(*_good_fit_class(units))
        if found is not None:
            start = next(iter(self._bins[found]))
        else:
            start = self._exact_fit(units)
            if start is None:
                return None
        block_units = self._remove_free(start)
        if block_units > units:
            self._insert_free(start + units, block_units - units)
        self._free_map[start : start + units] = bytes(units)
        return start * self._grain

    def release(self, offset: int, length: int) -> None:
        """Return ``[offset, offset + length)`` to the free space.

        The range may be any reserved sub-range (an extent reserved once can
        be released in independent pieces). Adjacent free blocks coalesce.

        Args:
            offset: byte offset; multiple of ``grain``.
            length: positive multiple of ``grain``.

        Raises:
            ValueError: if the range is misaligned, out of bounds, or overlaps
                free space (double free).
        """
        start, units = self._check_range(offset, length)
        end = start + units
        if self._free_map.find(1, start, end) != -1:
            raise ValueError(f"release of [{offset}, +{length}) overlaps free")
        self._free_map[start:end] = b"\x01" * units
        left = self._start_of_end.get(start)
        if left is not None:
            self._remove_free(left)
            start = left
        if end in self._length_of:
            end += self._remove_free(end)
        self._insert_free(start, end - start)

    def take(self, offset: int, length: int) -> None:
        """Reserve the exact range ``[offset, offset + length)``.

        Used to rebuild allocator state from persisted placements at startup.

        Args:
            offset: byte offset; multiple of ``grain``.
            length: positive multiple of ``grain``.

        Raises:
            ValueError: if the range is misaligned, out of bounds, or not
                entirely free.
        """
        start, units = self._check_range(offset, length)
        if self._free_map.find(0, start, start + units) != -1:
            raise ValueError(f"take of [{offset}, +{length}) is not free")
        # The covering free block starts right after the last reserved unit.
        block = self._free_map.rfind(0, 0, start) + 1
        block_end = block + self._length_of[block]
        self._remove_free(block)
        self._free_map[start : start + units] = bytes(units)
        if block < start:
            self._insert_free(block, start - block)
        if start + units < block_end:
            self._insert_free(start + units, block_end - start - units)

    def _to_units(self, nbytes: int, name: str) -> int:
        """Convert a positive grain multiple to grain units."""
        if nbytes <= 0 or nbytes % self._grain:
            raise ValueError(
                f"{name} must be a positive multiple of grain {self._grain}, "
                f"got {nbytes}"
            )
        return nbytes // self._grain

    def _check_range(self, offset: int, length: int) -> tuple[int, int]:
        """Validate a byte range and return it as ``(start, units)``."""
        units = self._to_units(length, "length")
        if offset < 0 or offset % self._grain:
            raise ValueError(
                f"offset must be a non-negative multiple of grain {self._grain}, "
                f"got {offset}"
            )
        start = offset // self._grain
        if start + units > self._units:
            raise ValueError(
                f"range [{offset}, +{length}) exceeds allocator size {self.size}"
            )
        return start, units

    def _find_class(self, fl: int, sl: int) -> tuple[int, int] | None:
        """Return the first non-empty class at or above ``(fl, sl)``."""
        if fl >= len(self._sl_bitmaps):
            return None
        sl_map = self._sl_bitmaps[fl] & (-1 << sl)
        if not sl_map:
            fl_map = self._fl_bitmap & (-1 << (fl + 1))
            if not fl_map:
                return None
            fl = (fl_map & -fl_map).bit_length() - 1
            sl_map = self._sl_bitmaps[fl]
        return fl, (sl_map & -sl_map).bit_length() - 1

    def _exact_fit(self, units: int) -> int | None:
        """Return a block start in ``units``' own class holding ``units``."""
        bucket = self._bins.get(_size_class(units))
        if bucket is None:
            return None
        for start in bucket:
            if self._length_of[start] >= units:
                return start
        return None

    def _insert_free(self, start: int, units: int) -> None:
        """Record ``[start, start + units)`` as one free block."""
        fl, sl = _size_class(units)
        self._bins.setdefault((fl, sl), {})[start] = None
        self._sl_bitmaps[fl] |= 1 << sl
        self._fl_bitmap |= 1 << fl
        self._length_of[start] = units
        self._start_of_end[start + units] = start
        self._free_units += units

    def _remove_free(self, start: int) -> int:
        """Drop the free block starting at ``start`` and return its units."""
        units = self._length_of.pop(start)
        del self._start_of_end[start + units]
        fl, sl = _size_class(units)
        bucket = self._bins[(fl, sl)]
        del bucket[start]
        if not bucket:
            del self._bins[(fl, sl)]
            self._sl_bitmaps[fl] &= ~(1 << sl)
            if not self._sl_bitmaps[fl]:
                self._fl_bitmap &= ~(1 << fl)
        self._free_units -= units
        return units
