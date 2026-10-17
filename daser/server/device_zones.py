# SPDX-License-Identifier: Apache-2.0

"""Unprivileged discovery of which block device holds each L2 file range.

On a filesystem spanning several devices without striping (btrfs ``single``
data, LVM linear volumes) one file extent lies on one device. The probe maps
each FIEMAP extent of the store file to a device by issuing one small
O_DIRECT IO inside it and watching which device's ``/sys/block/<dev>/stat``
sector counter moved. Unwritten (preallocated) extents get a zero write,
which leaves their contents unchanged; written extents are only read. The
result is a zone map for ``daser.server.allocator.ZonedAllocator``. See
``docs/design/l2-device-spread.md``.
"""

# Standard
import array
import fcntl
import mmap
import os
import struct
import time

# First Party
from daser.logging import init_logger

logger = init_logger(__name__)

#: ``_IOWR('f', 11, struct fiemap)``.
FS_IOC_FIEMAP = 0xC020660B
FIEMAP_FLAG_SYNC = 0x1
FIEMAP_EXTENT_LAST = 0x1
FIEMAP_EXTENT_UNWRITTEN = 0x800
_FIEMAP_HEADER = struct.Struct("=QQIIII")
_FIEMAP_EXTENT = struct.Struct("=QQQQQIIII")
_FIEMAP_BATCH = 512
#: Probe IO size; attribution needs one device to move by at least this.
PROBE_BYTES = 64 << 10
#: Probe attempts per extent before it is left unattributed.
PROBE_ATTEMPTS = 4
#: Device runs shorter than this are merged into a neighbour.
MIN_ZONE_RUN_BYTES = 16 << 20
_VIRTUAL_PREFIXES = ("loop", "ram", "zram", "sr", "fd", "nbd")
_SECTOR = 512


def leaf_block_devices(sys_block: str = "/sys/block") -> list[str]:
    """Return the physical block devices listed under ``sys_block``.

    A device counts when it is not a loop/ram/optical-style virtual device,
    has no ``slaves`` (device-mapper and md devices do) and nonzero size.

    Args:
        sys_block: sysfs block directory.

    Returns:
        Sorted device names; empty if ``sys_block`` is unreadable.
    """
    try:
        names = os.listdir(sys_block)
    except OSError:
        return []
    devices = []
    for name in sorted(names):
        if name.startswith(_VIRTUAL_PREFIXES):
            continue
        base = os.path.join(sys_block, name)
        try:
            if os.listdir(os.path.join(base, "slaves")):
                continue
            with open(os.path.join(base, "size")) as handle:
                if int(handle.read().strip() or 0) == 0:
                    continue
        except (OSError, ValueError):
            continue
        devices.append(name)
    return devices


def build_zones(
    extents: list[tuple[int, int, str | None]],
    length: int,
    grain: int,
    min_run: int = MIN_ZONE_RUN_BYTES,
) -> tuple[tuple[int, int, int], ...]:
    """Turn attributed file extents into a zone map over ``[0, length)``.

    Unattributed extents and gaps inherit the previous known device (the
    next one at the start). Bounds are rounded down to ``grain``; adjacent
    runs of one device merge and runs shorter than ``min_run`` join their
    predecessor. Zone ids follow sorted device names.

    Args:
        extents: ``(offset, length, device or None)`` sorted by offset.
        length: covered byte length; a multiple of ``grain``.
        grain: alignment of every bound.
        min_run: shortest run kept on its own.

    Returns:
        ``(start, end, zone)`` runs tiling ``[0, length)``, or ``()`` when
        fewer than two devices remain.
    """
    runs: list[list] = []
    position = 0
    for offset, size, device in extents:
        start, end = max(offset, position), min(offset + size, length)
        if end <= start:
            continue
        if start > position:
            runs.append([position, start, None])
        runs.append([start, end, device])
        position = end
    if position < length:
        runs.append([position, length, None])
    known = [run[2] for run in runs if run[2] is not None]
    if not known:
        return ()
    previous = known[0]
    for run in runs:
        if run[2] is None:
            run[2] = previous
        previous = run[2]
    # Merge equal neighbours, then absorb short runs into their predecessor.
    same: list[list] = []
    for run in runs:
        if same and same[-1][2] == run[2]:
            same[-1][1] = run[1]
        else:
            same.append(run)
    kept: list[list] = []
    for run in same:
        if kept and (kept[-1][2] == run[2] or run[1] - run[0] < min_run):
            kept[-1][1] = run[1]
        else:
            kept.append(run)
    if len(kept) > 1 and kept[0][1] - kept[0][0] < min_run:
        kept[1][0] = 0
        kept.pop(0)
    collapsed: list[list] = []
    for run in kept:
        if collapsed and collapsed[-1][2] == run[2]:
            collapsed[-1][1] = run[1]
        else:
            collapsed.append(run)
    # Runs are now at least ``min_run`` long, so rounding cannot empty one.
    for left, right in zip(collapsed, collapsed[1:], strict=False):
        bound = right[0] - right[0] % grain
        left[1] = right[0] = bound
    devices = sorted({run[2] for run in collapsed})
    if len(devices) < 2:
        return ()
    ids = {device: index for index, device in enumerate(devices)}
    return tuple((start, end, ids[device]) for start, end, device in collapsed)


def probe_device_zones(
    path: str, length: int, grain: int, sys_block: str = "/sys/block"
) -> tuple[tuple[int, int, int], ...]:
    """Map ``[0, length)`` of the store file at ``path`` to device zones.

    Args:
        path: existing store file; no other process may write it meanwhile.
        length: bytes to map from offset 0 (the rank 0 L2 lane); a multiple
            of ``grain``.
        grain: allocation unit every zone bound is aligned to.
        sys_block: sysfs block directory (tests point it elsewhere).

    Returns:
        ``(start, end, zone)`` runs for ``ZonedAllocator``, or ``()`` (one
        zone) when the file lies on one device or cannot be attributed.

    Async/thread-safety:
        Blocking; call at startup before the transfer layer opens the file.
        Only unwritten extents are written, with zeros.
    """
    started = time.perf_counter()
    devices = leaf_block_devices(sys_block)
    if len(devices) < 2:
        return ()
    try:
        fd = os.open(path, os.O_RDWR | os.O_DIRECT)
    except OSError as exc:
        logger.warning("[SERVER] L2 device probe cannot open %s: %s", path, exc)
        return ()
    buffer = mmap.mmap(-1, PROBE_BYTES)
    attributed: list[tuple[int, int, str | None]] = []
    try:
        extents = _fiemap(fd, length)
        for offset, size, unwritten in extents:
            nbytes = min(PROBE_BYTES, size - size % 4096)
            if nbytes <= 0:
                attributed.append((offset, size, None))
                continue
            device = _probe_extent(
                fd, offset, memoryview(buffer)[:nbytes], unwritten, devices, sys_block
            )
            attributed.append((offset, size, device))
    except OSError as exc:
        logger.warning("[SERVER] L2 device probe failed on %s: %s", path, exc)
        return ()
    finally:
        buffer.close()
        os.close(fd)
    zones = build_zones(attributed, length, grain)
    unknown = sum(device is None for _offset, _size, device in attributed)
    sizes: dict[int, int] = {}
    for start, end, zone in zones:
        sizes[zone] = sizes.get(zone, 0) + end - start
    logger.info(
        "[SERVER] L2 device zones: %d zones %s over %d extents (%d unattributed) "
        "in %.2fs",
        max(1, len(sizes)),
        {
            zone: f"{nbytes / (1 << 30):.1f}GiB"
            for zone, nbytes in sorted(sizes.items())
        },
        len(attributed),
        unknown,
        time.perf_counter() - started,
    )
    return zones


def _read_sectors(devices: list[str], sys_block: str, field: int) -> dict[str, int]:
    """Return one ``stat`` sector field per device (2 = read, 6 = written)."""
    counts = {}
    for device in devices:
        with open(os.path.join(sys_block, device, "stat")) as handle:
            counts[device] = int(handle.read().split()[field])
    return counts


def _probe_extent(
    fd: int,
    offset: int,
    view: memoryview,
    unwritten: bool,
    devices: list[str],
    sys_block: str,
) -> str | None:
    """Return the one device a probe IO at ``offset`` lands on, or None.

    The first attempt on an unwritten extent writes zeros (reads of
    unwritten space do no device IO); every later attempt reads what that
    write placed. Devices moving by at least the probe size are intersected
    across attempts until one is left.
    """
    sectors = len(view) // _SECTOR
    candidates: set[str] | None = None
    write = unwritten
    for _attempt in range(PROBE_ATTEMPTS):
        field = 6 if write else 2
        before = _read_sectors(devices, sys_block, field)
        if write:
            os.pwritev(fd, [view], offset)
        else:
            os.preadv(fd, [view], offset)
        after = _read_sectors(devices, sys_block, field)
        write = False
        moved = {d for d in devices if after[d] - before[d] >= sectors}
        candidates = moved if candidates is None else candidates & moved
        if len(candidates) == 1:
            return next(iter(candidates))
        if not candidates:
            return None
    return None


def _fiemap(fd: int, length: int) -> list[tuple[int, int, bool]]:
    """Return ``(offset, length, unwritten)`` file extents within ``length``."""
    extents: list[tuple[int, int, bool]] = []
    start = 0
    while start < length:
        request = bytearray(_FIEMAP_HEADER.size + _FIEMAP_BATCH * _FIEMAP_EXTENT.size)
        _FIEMAP_HEADER.pack_into(
            request, 0, start, length - start, FIEMAP_FLAG_SYNC, 0, _FIEMAP_BATCH, 0
        )
        buffer = array.array("B", request)
        fcntl.ioctl(fd, FS_IOC_FIEMAP, buffer, True)
        mapped = _FIEMAP_HEADER.unpack_from(buffer, 0)[3]
        if not mapped:
            break
        last = False
        for index in range(mapped):
            fields = _FIEMAP_EXTENT.unpack_from(
                buffer, _FIEMAP_HEADER.size + index * _FIEMAP_EXTENT.size
            )
            logical, _physical, size, flags = fields[0], fields[1], fields[2], fields[5]
            extents.append((logical, size, bool(flags & FIEMAP_EXTENT_UNWRITTEN)))
            start = logical + size
            last = bool(flags & FIEMAP_EXTENT_LAST)
        if last:
            break
    return extents
