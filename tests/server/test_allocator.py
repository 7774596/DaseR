# SPDX-License-Identifier: Apache-2.0

# Standard
import random

# Third Party
import pytest

# First Party
from daser.server.allocator import TlsfAllocator, ZonedAllocator

GRAIN = 4096


class _Reference:
    """Brute-force model: the set of free grain units."""

    def __init__(self, units: int) -> None:
        self.units = units
        self.free = set(range(units))

    def runs(self) -> list[tuple[int, int]]:
        """Return maximal free runs as ``(start, length)`` in units."""
        runs: list[tuple[int, int]] = []
        for unit in sorted(self.free):
            if runs and runs[-1][0] + runs[-1][1] == unit:
                runs[-1] = (runs[-1][0], runs[-1][1] + 1)
            else:
                runs.append((unit, 1))
        return runs

    def is_free(self, start: int, length: int) -> bool:
        return all(unit in self.free for unit in range(start, start + length))

    def is_used(self, start: int, length: int) -> bool:
        return not any(unit in self.free for unit in range(start, start + length))

    def mark(self, start: int, length: int, free: bool) -> None:
        span = set(range(start, start + length))
        if free:
            self.free |= span
        else:
            self.free -= span


def _assert_matches(alloc: TlsfAllocator, ref: _Reference) -> None:
    runs = ref.runs()
    assert alloc.free_bytes == len(ref.free) * GRAIN
    assert alloc.holes == len(runs)
    assert alloc.largest_free == max((n for _, n in runs), default=0) * GRAIN
    assert alloc.free_bytes % GRAIN == 0
    assert alloc.largest_free % GRAIN == 0


def _drain(alloc: TlsfAllocator, ref: _Reference) -> None:
    """Reserve every free byte, largest extent first, checking each result.

    Matching the reference hole by hole shows that every free remainder the
    allocator kept is a grain-aligned extent the reference also has free.
    """
    while alloc.largest_free:
        length = alloc.largest_free
        offset = alloc.reserve(length)
        assert offset is not None
        assert offset % GRAIN == 0
        assert ref.is_free(offset // GRAIN, length // GRAIN)
        ref.mark(offset // GRAIN, length // GRAIN, free=False)
        _assert_matches(alloc, ref)
    assert not ref.free
    assert alloc.reserve(GRAIN) is None


def _random_length(rng: random.Random, units: int) -> int:
    if rng.random() < 0.7:
        return rng.randint(1, 12)
    return rng.randint(1, units)


@pytest.mark.parametrize(("seed", "units"), [(0, 512), (1, 1000), (2, 77), (3, 4096)])
def test_random_sequences_match_reference_model(seed: int, units: int) -> None:
    """reserve/release/take agree with a brute-force free-unit model."""
    rng = random.Random(seed)
    alloc = TlsfAllocator(units * GRAIN, GRAIN)
    ref = _Reference(units)
    reserved: list[tuple[int, int]] = []
    for _ in range(4000):
        op = rng.random()
        if op < 0.45:
            length = _random_length(rng, units)
            offset = alloc.reserve(length * GRAIN)
            fits = any(n >= length for _, n in ref.runs())
            assert (offset is not None) == fits
            if offset is not None:
                assert offset % GRAIN == 0
                start = offset // GRAIN
                assert ref.is_free(start, length)
                ref.mark(start, length, free=False)
                reserved.append((start, length))
        elif op < 0.85 and reserved:
            # Release a random piece of a reserved extent; keep the rest.
            start, length = reserved.pop(rng.randrange(len(reserved)))
            lo = rng.randrange(length)
            hi = rng.randint(lo + 1, length)
            alloc.release((start + lo) * GRAIN, (hi - lo) * GRAIN)
            ref.mark(start + lo, hi - lo, free=True)
            if lo:
                reserved.append((start, lo))
            if hi < length:
                reserved.append((start + hi, length - hi))
        else:
            length = rng.randint(1, min(units, 16))
            start = rng.randrange(units - length + 1)
            if ref.is_free(start, length):
                alloc.take(start * GRAIN, length * GRAIN)
                ref.mark(start, length, free=False)
                reserved.append((start, length))
            else:
                with pytest.raises(ValueError):
                    alloc.take(start * GRAIN, length * GRAIN)
        _assert_matches(alloc, ref)
    _drain(alloc, ref)


def test_reserve_finds_a_fitting_block_below_its_good_fit_class() -> None:
    """A 17-grain hole serves a 17-grain request though good fit rounds up."""
    alloc = TlsfAllocator(17 * GRAIN, GRAIN)
    assert alloc.reserve(17 * GRAIN) == 0
    assert alloc.free_bytes == 0

    alloc = TlsfAllocator(16 * GRAIN, GRAIN)
    assert alloc.reserve(17 * GRAIN) is None
    assert alloc.free_bytes == 16 * GRAIN


def test_release_coalesces_neighbours() -> None:
    alloc = TlsfAllocator(8 * GRAIN, GRAIN)
    offsets = [alloc.reserve(2 * GRAIN) for _ in range(4)]
    assert offsets == [0, 2 * GRAIN, 4 * GRAIN, 6 * GRAIN]
    alloc.release(0, 2 * GRAIN)
    alloc.release(4 * GRAIN, 2 * GRAIN)
    assert (alloc.holes, alloc.largest_free) == (2, 2 * GRAIN)
    alloc.release(2 * GRAIN, 2 * GRAIN)
    assert (alloc.holes, alloc.largest_free) == (1, 6 * GRAIN)
    alloc.release(6 * GRAIN, 2 * GRAIN)
    assert (alloc.holes, alloc.free_bytes) == (1, alloc.size)


def test_take_splits_the_covering_free_block() -> None:
    alloc = TlsfAllocator(10 * GRAIN, GRAIN)
    alloc.take(3 * GRAIN, 4 * GRAIN)
    assert (alloc.holes, alloc.free_bytes, alloc.largest_free) == (
        2,
        6 * GRAIN,
        3 * GRAIN,
    )
    with pytest.raises(ValueError, match="not free"):
        alloc.take(2 * GRAIN, 2 * GRAIN)
    with pytest.raises(ValueError, match="not free"):
        alloc.take(6 * GRAIN, GRAIN)
    assert alloc.free_bytes == 6 * GRAIN


@pytest.mark.parametrize(
    ("offset", "length"),
    [
        (2, 2),  # exactly a free block
        (1, 2),  # tail overlaps the free block's head
        (3, 2),  # head overlaps the free block's tail
        (1, 4),  # strictly contains the free block
        (2, 1),  # inside the free block
    ],
)
def test_double_free_is_rejected_without_mutation(offset: int, length: int) -> None:
    alloc = TlsfAllocator(8 * GRAIN, GRAIN)
    assert alloc.reserve(8 * GRAIN) == 0
    alloc.release(2 * GRAIN, 2 * GRAIN)
    with pytest.raises(ValueError, match="overlaps free"):
        alloc.release(offset * GRAIN, length * GRAIN)
    assert (alloc.free_bytes, alloc.holes) == (2 * GRAIN, 1)


@pytest.mark.parametrize(
    ("offset", "length"),
    [(GRAIN // 2, GRAIN), (0, GRAIN + 1), (0, 0), (-GRAIN, GRAIN), (0, 9 * GRAIN)],
)
def test_misaligned_or_out_of_bounds_ranges_are_rejected(
    offset: int, length: int
) -> None:
    alloc = TlsfAllocator(8 * GRAIN, GRAIN)
    assert alloc.reserve(8 * GRAIN) == 0
    with pytest.raises(ValueError):
        alloc.release(offset, length)
    with pytest.raises(ValueError):
        alloc.take(offset, length)
    assert alloc.free_bytes == 0


@pytest.mark.parametrize("length", [0, -GRAIN, GRAIN - 1, 3 * GRAIN])
def test_reserve_rejects_non_grain_lengths(length: int) -> None:
    alloc = TlsfAllocator(8 * 2 * GRAIN, 2 * GRAIN)
    with pytest.raises(ValueError):
        alloc.reserve(length)
    assert alloc.free_bytes == alloc.size


def test_reserve_larger_than_capacity_returns_none() -> None:
    alloc = TlsfAllocator(4 * GRAIN, GRAIN)
    assert alloc.reserve(5 * GRAIN) is None
    assert alloc.free_bytes == 4 * GRAIN


@pytest.mark.parametrize("grain", [0, 1024, 4096 * 3, 6144, -4096])
def test_invalid_grain_is_rejected(grain: int) -> None:
    with pytest.raises(ValueError, match="grain"):
        TlsfAllocator(4096 * 12, grain)


@pytest.mark.parametrize("size", [0, GRAIN * 64 + GRAIN, -GRAIN * 64])
def test_size_must_be_a_positive_grain_multiple(size: int) -> None:
    with pytest.raises(ValueError, match="size"):
        TlsfAllocator(size, GRAIN * 64)


def test_default_grain_offsets_stay_aligned() -> None:
    grain = 64 << 10
    alloc = TlsfAllocator(1000 * grain, grain)
    assert (alloc.size, alloc.grain) == (1000 * grain, grain)
    offsets = [alloc.reserve(3 * grain) for _ in range(5)]
    assert all(offset is not None and offset % grain == 0 for offset in offsets)
    assert alloc.free_bytes == 985 * grain


# ----------------------------------------------------------------------
# Initial free ranges and zones
# ----------------------------------------------------------------------


def test_free_ranges_limit_what_is_handed_out() -> None:
    alloc = TlsfAllocator(
        16 * GRAIN, GRAIN, [(2 * GRAIN, 3 * GRAIN), (8 * GRAIN, GRAIN)]
    )
    assert alloc.free_bytes == 4 * GRAIN
    assert alloc.holes == 2
    assert alloc.reserve(4 * GRAIN) is None
    assert alloc.reserve(3 * GRAIN) == 2 * GRAIN
    assert alloc.reserve(GRAIN) == 8 * GRAIN
    assert alloc.reserve(GRAIN) is None
    # Space outside the initial ranges counts as reserved: it may be released.
    alloc.release(0, 2 * GRAIN)
    assert alloc.reserve(2 * GRAIN) == 0


@pytest.mark.parametrize(
    "ranges",
    [[(GRAIN, GRAIN), (0, GRAIN)], [(0, 2 * GRAIN), (GRAIN, GRAIN)], [(0, 100)]],
)
def test_free_ranges_must_be_sorted_disjoint_and_aligned(ranges) -> None:
    with pytest.raises(ValueError):
        TlsfAllocator(4 * GRAIN, GRAIN, ranges)


ZONES = ((0, 4 * GRAIN, 0), (4 * GRAIN, 8 * GRAIN, 1), (8 * GRAIN, 12 * GRAIN, 0))


def _zone_of_ref(offset: int) -> int:
    for start, end, zone in ZONES:
        if start <= offset < end:
            return zone
    raise AssertionError(offset)


def test_zoned_reservations_stay_inside_one_zone() -> None:
    alloc = ZonedAllocator(12 * GRAIN, GRAIN, ZONES)
    assert alloc.zone_count == 2
    assert alloc.free_bytes == 12 * GRAIN
    # Zone 0 is split in two runs of 4, so 5 units fit nowhere.
    assert alloc.reserve(5 * GRAIN) is None
    offsets = [alloc.reserve(2 * GRAIN) for _ in range(6)]
    assert None not in offsets
    for offset in offsets:
        assert _zone_of_ref(offset) == _zone_of_ref(offset + 2 * GRAIN - 1)
        assert alloc.zone_of(offset) == _zone_of_ref(offset)
    assert alloc.free_bytes == 0
    assert alloc.reserve(GRAIN) is None


def test_zoned_reserve_avoids_zones_and_prefers_the_freest() -> None:
    alloc = ZonedAllocator(12 * GRAIN, GRAIN, ZONES)
    # Zone 0 has 8 free units, zone 1 has 4.
    assert alloc.zone_of(alloc.reserve(GRAIN)) == 0
    assert alloc.zone_of(alloc.reserve(GRAIN, avoid={0})) == 1
    assert alloc.zone_free_bytes(1) == 3 * GRAIN
    assert alloc.reserve(4 * GRAIN, avoid={0}) is None
    assert alloc.reserve(GRAIN, avoid={0, 1}) is None
    with pytest.raises(ValueError):
        alloc.reserve(100, avoid={0, 1})


def test_zoned_release_and_take_split_at_zone_bounds() -> None:
    alloc = ZonedAllocator(12 * GRAIN, GRAIN, ZONES)
    # A restored extent may cross zones; it is taken and freed piecewise.
    alloc.take(3 * GRAIN, 6 * GRAIN)
    assert alloc.free_bytes == 6 * GRAIN
    assert alloc.zone_free_bytes(1) == 0
    with pytest.raises(ValueError):
        alloc.take(2 * GRAIN, 2 * GRAIN)
    alloc.release(3 * GRAIN, 6 * GRAIN)
    assert alloc.free_bytes == 12 * GRAIN
    with pytest.raises(ValueError):
        alloc.release(3 * GRAIN, GRAIN)
    with pytest.raises(ValueError):
        alloc.release(11 * GRAIN, 2 * GRAIN)
    # Free blocks of different zones never coalesce.
    assert alloc.holes == 3
    assert alloc.largest_free == 4 * GRAIN


@pytest.mark.parametrize(
    "zones",
    [
        ((0, 4 * GRAIN, 0),),
        ((0, 4 * GRAIN, 0), (5 * GRAIN, 8 * GRAIN, 1)),
        ((0, 4 * GRAIN, 0), (4 * GRAIN, 8 * GRAIN, 2)),
        ((4 * GRAIN, 8 * GRAIN, 0), (0, 4 * GRAIN, 1)),
    ],
)
def test_zones_must_tile_the_range_with_dense_ids(zones) -> None:
    with pytest.raises(ValueError):
        ZonedAllocator(8 * GRAIN, GRAIN, zones)


@pytest.mark.parametrize("seed", range(4))
def test_one_zone_matches_tlsf(seed: int) -> None:
    rng = random.Random(seed)
    plain = TlsfAllocator(64 * GRAIN, GRAIN)
    zoned = ZonedAllocator(64 * GRAIN, GRAIN)
    live: list[tuple[int, int]] = []
    for _ in range(400):
        if live and rng.random() < 0.45:
            offset, length = live.pop(rng.randrange(len(live)))
            plain.release(offset, length)
            zoned.release(offset, length)
        else:
            length = rng.randint(1, 6) * GRAIN
            offset = plain.reserve(length)
            assert zoned.reserve(length) == offset
            if offset is not None:
                live.append((offset, length))
        assert (zoned.free_bytes, zoned.holes, zoned.largest_free) == (
            plain.free_bytes,
            plain.holes,
            plain.largest_free,
        )
