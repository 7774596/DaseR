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

``ZonedAllocator`` keeps one TLSF allocator per zone (a set of ranges with
one physical placement), so its extents never cross a zone.
"""

# Standard
import bisect
from collections.abc import Collection

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
        free_ranges: Initially free ``(offset, length)`` byte ranges, sorted,
            disjoint, grain-aligned and inside ``[0, size)``; the rest starts
            reserved and is never handed out unless released. None frees the
            whole range.

    Raises:
        ValueError: if ``grain``, ``size`` or ``free_ranges`` is invalid.

    Async/thread-safety:
        Not synchronized. Callers use one allocator from a single thread
        (the server event loop).
    """

    def __init__(
        self,
        size: int,
        grain: int,
        free_ranges: list[tuple[int, int]] | None = None,
    ) -> None:
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
        self._free_units = 0
        if free_ranges is None:
            # One byte per grain unit: 1 = free, 0 = reserved.
            self._free_map = bytearray(b"\x01") * self._units
            self._insert_free(0, self._units)
            return
        self._free_map = bytearray(self._units)
        previous_end = 0
        for offset, length in free_ranges:
            start, units = self._check_range(offset, length)
            if start < previous_end:
                raise ValueError("free_ranges must be sorted and disjoint")
            previous_end = start + units
            self._free_map[start:previous_end] = b"\x01" * units
            self._insert_free(start, units)

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


class ZonedAllocator:
    """Grain-aligned extent allocator whose extents never cross a zone.

    A zone is a set of byte ranges with one physical placement (for the L2
    store file: one device). Each zone has its own ``TlsfAllocator`` over the
    whole range with only the zone's ranges initially free, so a reservation
    always lies inside one zone and free blocks of different zones never
    coalesce. See ``docs/design/l2-device-spread.md``.

    Args:
        size: managed capacity in bytes; positive multiple of ``grain``.
        grain: allocation unit in bytes (see ``TlsfAllocator``).
        zones: ``(start, end, zone)`` runs, sorted and covering
            ``[0, size)`` without gaps; bounds are grain multiples, zone ids
            are ``0..n-1``. Empty means a single zone.

    Raises:
        ValueError: if ``zones`` does not tile ``[0, size)`` or its ids are
            not dense.

    Async/thread-safety:
        Not synchronized; owned by the server event loop.
    """

    def __init__(
        self, size: int, grain: int, zones: tuple[tuple[int, int, int], ...] = ()
    ) -> None:
        runs = list(zones) or [(0, size, 0)]
        ids = {zone for _start, _end, zone in runs}
        if ids != set(range(len(ids))):
            raise ValueError(f"zone ids must be 0..n-1, got {sorted(ids)}")
        position = 0
        for start, end, _zone in runs:
            if start != position or end <= start:
                raise ValueError(f"zones must tile [0, {size}) in order")
            position = end
        if position != size:
            raise ValueError(f"zones must tile [0, {size}) in order")
        self._starts = [start for start, _end, _zone in runs]
        self._runs = runs
        self._zones = [
            TlsfAllocator(
                size,
                grain,
                [(start, end - start) for start, end, z in runs if z == zone],
            )
            for zone in range(len(ids))
        ]

    @property
    def size(self) -> int:
        """Return the managed capacity in bytes."""
        return self._zones[0].size

    @property
    def grain(self) -> int:
        """Return the allocation unit in bytes."""
        return self._zones[0].grain

    @property
    def zone_count(self) -> int:
        """Return the number of zones."""
        return len(self._zones)

    @property
    def free_bytes(self) -> int:
        """Return the total free bytes over all zones."""
        return sum(zone.free_bytes for zone in self._zones)

    @property
    def holes(self) -> int:
        """Return the number of maximal free extents over all zones."""
        return sum(zone.holes for zone in self._zones)

    @property
    def largest_free(self) -> int:
        """Return the longest free extent in bytes over all zones."""
        return max(zone.largest_free for zone in self._zones)

    def zone_free_bytes(self, zone: int) -> int:
        """Return the free bytes of one zone.

        Args:
            zone: zone id.
        """
        return self._zones[zone].free_bytes

    def zone_of(self, offset: int) -> int:
        """Return the zone holding byte ``offset``.

        Args:
            offset: byte offset in ``[0, size)``.

        Raises:
            ValueError: if ``offset`` is out of range.
        """
        if not 0 <= offset < self.size:
            raise ValueError(f"offset {offset} outside [0, {self.size})")
        return self._runs[bisect.bisect_right(self._starts, offset) - 1][2]

    def reserve(self, length: int, avoid: Collection[int] = ()) -> int | None:
        """Allocate ``length`` contiguous bytes inside one zone.

        Zones not in ``avoid`` are tried from most to least free bytes.

        Args:
            length: positive multiple of ``grain``.
            avoid: zone ids not to use.

        Returns:
            Byte offset of the extent, or None if no allowed zone has a free
            extent that long.

        Raises:
            ValueError: if ``length`` is not a positive multiple of ``grain``.
        """
        order = sorted(
            (zone for zone in range(len(self._zones)) if zone not in avoid),
            key=lambda zone: -self._zones[zone].free_bytes,
        )
        if length <= 0 or length % self.grain:
            raise ValueError(
                f"length must be a positive multiple of grain {self.grain}, "
                f"got {length}"
            )
        for zone in order:
            offset = self._zones[zone].reserve(length)
            if offset is not None:
                return offset
        return None

    def release(self, offset: int, length: int) -> None:
        """Free ``[offset, offset + length)``, split at zone bounds.

        Args:
            offset: byte offset; multiple of ``grain``.
            length: positive multiple of ``grain``.

        Raises:
            ValueError: if the range is misaligned, out of bounds, or overlaps
                free space.
        """
        for zone, start, part in self._split(offset, length):
            self._zones[zone].release(start, part)

    def take(self, offset: int, length: int) -> None:
        """Reserve the exact range ``[offset, offset + length)``.

        A range crossing zone bounds is taken piecewise (persisted placements
        may predate the current zone map).

        Args:
            offset: byte offset; multiple of ``grain``.
            length: positive multiple of ``grain``.

        Raises:
            ValueError: if the range is misaligned, out of bounds, or not
                entirely free.
        """
        for zone, start, part in self._split(offset, length):
            self._zones[zone].take(start, part)

    def _split(self, offset: int, length: int) -> list[tuple[int, int, int]]:
        """Return ``(zone, offset, length)`` pieces of a range, by zone run."""
        if length <= 0 or offset < 0 or offset + length > self.size:
            raise ValueError(
                f"range [{offset}, +{length}) outside allocator size {self.size}"
            )
        pieces = []
        index = bisect.bisect_right(self._starts, offset) - 1
        end = offset + length
        while offset < end:
            _start, run_end, zone = self._runs[index]
            part_end = min(end, run_end)
            pieces.append((zone, offset, part_end - offset))
            offset = part_end
            index += 1
        return pieces
