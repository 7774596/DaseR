# SPDX-License-Identifier: Apache-2.0
"""GPU-resident copy of stored bytes a predicted load will read.

The stash holds stored (compressed when the store is compressed) bytes in one
device allocation on the load destination's GPU. A predicted session's bytes
are copied host-to-device while the current request still computes, so its
load copies them device-to-device instead of over PCIe on the critical path.
"""

import bisect
from dataclasses import dataclass
from typing import Any


@dataclass
class StashRange:
    """Map one physical byte range to bytes held in the device stash."""

    file_offset: int
    nbytes: int
    data: "DeviceStashSlice"
    source_offset: int


class DeviceStashSlice:
    """Byte view of the stash allocation, shaped like a pinned slice.

    Grouped copies read sources through ``ptr_at`` only, so stash bytes can
    join the same copy as pinned L1 bytes. ``close`` is a no-op: the stash
    owns its allocation and reuses it on the next fill.

    Args:
        base_ptr: Device pointer of the stash allocation.
        size: Allocation size in bytes.
    """

    on_device = True

    def __init__(self, base_ptr: int, size: int) -> None:
        self._base_ptr = base_ptr
        self._size = size

    def ptr_at(self, offset: int = 0) -> int:
        """Return the device pointer of a byte offset into the stash.

        Args:
            offset: Byte offset inside the allocation.

        Returns:
            Integer device pointer.
        """
        if not 0 <= offset <= self._size:
            raise ValueError("offset is outside the device stash")
        return self._base_ptr + offset

    def close(self) -> None:
        """Keep the allocation; the stash owns and reuses it."""
        return None


class _StashTarget:
    """Copy destination over the stash allocation for grouped H2D copies."""

    def __init__(self, array: Any, stream: Any) -> None:
        self._array = array
        self._stream = stream

    def __getitem__(self, item: Any) -> Any:
        return self._array[item]

    @property
    def copy_stream_ptr(self) -> int:
        return int(self._stream.ptr)


class DeviceStash:
    """Fixed-capacity device buffer of physical byte ranges.

    Args:
        capacity_bytes: Size of the device allocation, made lazily on the
            first fill.

    Async/thread-safety:
        Not synchronized. The transfer layer mutates entries under its
        metadata lock and only refills after every copy reading the stash
        has drained.
    """

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.capacity_bytes = int(capacity_bytes)
        self._array: Any | None = None
        self._slice: DeviceStashSlice | None = None
        self._stream: Any | None = None
        self._entries: list[StashRange] = []
        self._starts: list[int] = []
        # Ranges stored while a fill is copying; None when no fill runs.
        self._dirty: list[tuple[int, int]] | None = None

    @property
    def resident_bytes(self) -> int:
        """Return the bytes currently published in the stash."""
        return sum(entry.nbytes for entry in self._entries)

    def hits(self) -> tuple[list[int], list[StashRange]]:
        """Return published ranges and their starts, sorted by file offset."""
        return self._starts, self._entries

    def clear(self) -> None:
        """Unpublish every range so no new load reads the stash."""
        self._entries = []
        self._starts = []

    def target(self, device_id: int) -> tuple[_StashTarget, DeviceStashSlice]:
        """Return the copy destination and source view, allocating on demand.

        Args:
            device_id: CUDA device of the load destinations.

        Returns:
            ``(target, view)``: a grouped-copy destination over the whole
            allocation and the slice later loads copy from.
        """
        if self._array is None:
            import cupy

            with cupy.cuda.Device(device_id):
                memory = cupy.cuda.Memory(self.capacity_bytes)
                pointer = cupy.cuda.MemoryPointer(memory, 0)
                self._array = cupy.ndarray(
                    (self.capacity_bytes,), dtype=cupy.uint8, memptr=pointer
                )
                self._stream = cupy.cuda.Stream(non_blocking=True)
            self._slice = DeviceStashSlice(
                int(self._array.data.ptr), self.capacity_bytes
            )
        assert self._slice is not None
        return _StashTarget(self._array, self._stream), self._slice

    def begin_fill(self) -> None:
        """Start recording stores that land while a fill copies."""
        self._dirty = []

    def publish(self, ranges: list[StashRange]) -> int:
        """Publish filled ranges, except those a store overwrote meanwhile.

        Args:
            ranges: Filled ranges, each backed by bytes already on the device.

        Returns:
            Number of bytes published.
        """
        dirty = self._dirty or []
        self._dirty = None
        kept = [
            entry
            for entry in ranges
            if not any(
                start < entry.file_offset + entry.nbytes
                and entry.file_offset < start + size
                for start, size in dirty
            )
        ]
        kept.sort(key=lambda entry: entry.file_offset)
        self._entries = kept
        self._starts = [entry.file_offset for entry in kept]
        return sum(entry.nbytes for entry in kept)

    def invalidate(self, file_offset: int, nbytes: int) -> None:
        """Forget stashed bytes a store is about to overwrite.

        Args:
            file_offset: Physical start of the stored range.
            nbytes: Length of the stored range.
        """
        if self._dirty is not None:
            self._dirty.append((file_offset, nbytes))
        if not self._entries:
            return
        end = file_offset + nbytes
        index = max(0, bisect.bisect_right(self._starts, file_offset) - 1)
        if not any(
            entry.file_offset < end and file_offset < entry.file_offset + entry.nbytes
            for entry in self._entries[index:]
            if entry.file_offset < end
        ):
            return
        self._entries = [
            entry
            for entry in self._entries
            if not (
                entry.file_offset < end
                and file_offset < entry.file_offset + entry.nbytes
            )
        ]
        self._starts = [entry.file_offset for entry in self._entries]

    def close(self) -> None:
        """Release the device allocation."""
        self.clear()
        self._array = None
        self._slice = None
        self._stream = None
