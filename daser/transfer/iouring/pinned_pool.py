# SPDX-License-Identifier: Apache-2.0

"""Exact-size CUDA pinned host buffers and a first-fit staging sub-allocator."""

# Standard
from collections.abc import Callable


def _align_up(value: int, alignment: int) -> int:
    """Round a byte count up to an alignment boundary."""
    return -(-value // alignment) * alignment


class PinnedMemoryBuffer:
    """One dedicated CUDA page-locked host allocation of exactly ``size`` bytes.

    The bytes come from ``cupy.cuda.PinnedMemory`` (``cudaHostAlloc``), not
    CuPy's caching pinned pool, which rounds requests up to powers of two and
    keeps freed blocks; the physical backing therefore equals the configured
    capacity. The pages are portable across CUDA contexts.

    Args:
        size: Positive number of bytes to allocate and lock.

    Async/thread-safety:
        Not internally synchronized; callers serialize mutation of the bytes.
        ``close`` drops this owner's reference, which frees the pages
        synchronously once no memoryview from ``view`` remains alive.
    """

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("pinned buffer size must be positive")
        import cupy

        self._size = size
        memory = cupy.cuda.PinnedMemory(size, cupy.cuda.runtime.hostAllocPortable)
        self._pointer: cupy.cuda.PinnedMemoryPointer | None = (
            cupy.cuda.PinnedMemoryPointer(memory, 0)
        )

    def view(self) -> memoryview:
        """Return a writable byte memoryview over the whole allocation.

        Returns:
            Memoryview of ``len(self)`` bytes; it keeps the pages alive until
            released.

        Raises:
            RuntimeError: If the buffer is closed.
        """
        if self._pointer is None:
            raise RuntimeError("pinned buffer is closed")
        return memoryview(self._pointer).cast("B")[: self._size]

    def ptr_at(self, offset: int = 0) -> int:
        """Return the host pointer of a byte offset inside the allocation.

        Args:
            offset: Byte offset in ``[0, len(self)]``.

        Returns:
            Integer host pointer usable by CUDA runtime copy APIs.

        Raises:
            ValueError: If ``offset`` is outside the buffer.
            RuntimeError: If the buffer is closed.
        """
        if self._pointer is None:
            raise RuntimeError("pinned buffer is closed")
        if not 0 <= offset <= self._size:
            raise ValueError("offset is outside pinned buffer")
        return int(self._pointer.ptr) + offset

    def close(self) -> None:
        """Drop the allocation; idempotent.

        Thread-safety:
            Call after every IO and DMA using the pages has completed.
        """
        self._pointer = None

    def __len__(self) -> int:
        """Return the allocation size in bytes."""
        return self._size


class PinnedMemorySlice:
    """Byte range of a ``PinnedMemoryPool`` backing, returned on ``close``.

    Args:
        owner: Backing pinned buffer.
        offset: Byte offset inside ``owner``.
        size: Slice size in bytes.
        release: Called once with ``(offset, size)`` when the slice closes.

    Async/thread-safety:
        Not internally synchronized; the owning pool's caller serializes
        allocation and release.
    """

    def __init__(
        self,
        owner: PinnedMemoryBuffer,
        offset: int,
        size: int,
        release: Callable[[int, int], None],
    ) -> None:
        if offset < 0 or size <= 0 or offset + size > len(owner):
            raise ValueError("slice is outside pinned buffer")
        self._owner = owner
        self._offset = offset
        self._size = size
        self._release: Callable[[int, int], None] | None = release

    def view(self) -> memoryview:
        """Return a writable byte memoryview over the slice.

        Raises:
            RuntimeError: If the slice is closed.
        """
        if self._release is None:
            raise RuntimeError("pinned slice is closed")
        return self._owner.view()[self._offset : self._offset + self._size]

    def ptr_at(self, offset: int = 0) -> int:
        """Return the host pointer of a byte offset inside the slice.

        Args:
            offset: Byte offset in ``[0, len(self)]``.

        Returns:
            Integer host pointer usable by CUDA runtime copy APIs.

        Raises:
            ValueError: If ``offset`` is outside the slice.
            RuntimeError: If the slice is closed.
        """
        if self._release is None:
            raise RuntimeError("pinned slice is closed")
        if not 0 <= offset <= self._size:
            raise ValueError("offset is outside pinned slice")
        return self._owner.ptr_at(self._offset + offset)

    def close(self) -> None:
        """Return the slice to its pool; idempotent.

        Thread-safety:
            Call only after no thread or CUDA operation uses the slice.
        """
        release, self._release = self._release, None
        if release is not None:
            release(self._offset, self._size)

    def __len__(self) -> int:
        """Return the slice size in bytes."""
        return self._size


class PinnedMemoryPool:
    """Fixed-capacity first-fit sub-allocator over one pinned buffer.

    Args:
        capacity: Bytes to allocate, rounded up to ``alignment``.
        alignment: Alignment of every slice offset and reserved length.

    Async/thread-safety:
        Synchronous and unlocked: call ``allocate`` and close slices from one
        thread (the transfer event loop).
    """

    def __init__(self, capacity: int, alignment: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if alignment <= 0:
            raise ValueError("alignment must be positive")
        self._alignment = alignment
        self._capacity = _align_up(capacity, alignment)
        self._memory = PinnedMemoryBuffer(self._capacity)
        self._free: list[tuple[int, int]] = [(0, self._capacity)]
        self._used = 0
        self._closed = False

    @property
    def capacity(self) -> int:
        """Return the pool size in bytes."""
        return self._capacity

    @property
    def used_bytes(self) -> int:
        """Return the bytes held by open slices, alignment included."""
        return self._used

    def allocate(self, size: int) -> PinnedMemorySlice | None:
        """Reserve a slice of ``size`` bytes.

        Args:
            size: Positive byte count.

        Returns:
            The slice, or None when no free range holds the aligned request.

        Raises:
            ValueError: If ``size`` is not positive.
            RuntimeError: If the pool is closed.
        """
        if self._closed:
            raise RuntimeError("pinned pool is closed")
        if size <= 0:
            raise ValueError("size must be positive")
        aligned = _align_up(size, self._alignment)
        for index, (offset, free_size) in enumerate(self._free):
            if free_size < aligned:
                continue
            if free_size > aligned:
                self._free[index] = (offset + aligned, free_size - aligned)
            else:
                self._free.pop(index)
            self._used += aligned
            return PinnedMemorySlice(self._memory, offset, size, self._release)
        return None

    def close(self) -> None:
        """Release the backing allocation; idempotent.

        Thread-safety:
            Call after every slice is closed and no IO uses the pool.
        """
        if self._closed:
            return
        self._closed = True
        self._free = []
        self._memory.close()

    def _release(self, offset: int, size: int) -> None:
        """Return an aligned range to the free list, merging neighbours."""
        aligned = _align_up(size, self._alignment)
        self._used -= aligned
        if self._closed:
            return
        merged: list[tuple[int, int]] = []
        for start, length in sorted([*self._free, (offset, aligned)]):
            if merged and start == merged[-1][0] + merged[-1][1]:
                merged[-1] = (merged[-1][0], merged[-1][1] + length)
            else:
                merged.append((start, length))
        self._free = merged
