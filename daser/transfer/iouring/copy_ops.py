# SPDX-License-Identifier: Apache-2.0

"""Stateless copy/marshalling helpers for the io_uring transfer layer.

These functions move bytes between CPU buffers, CuPy device arrays, and the
pinned host slices used by the L1 tier. They hold no tiering state and take no
locks, so they are safe to call from either the event loop or executor threads
as long as the caller owns the buffers passed in.
"""

# Standard
import ctypes
from typing import Any

# First Party
from daser.transfer.iouring.pinned_pool import PinnedMemorySlice

#: One grouped copy chunk: (target_offset, source_slice, source_offset, nbytes).
CopyChunk = tuple[int, PinnedMemorySlice, int, int]


def _merge_cuda_runs(dst: Any, chunks: list[CopyChunk]) -> list[tuple[int, int, int]]:
    """Return ``(dst_ptr, src_ptr, nbytes)`` runs with adjacent chunks merged."""
    merged: list[tuple[int, int, int]] = []
    for target_offset, data, source_offset, nbytes in sorted(
        chunks, key=lambda item: item[0]
    ):
        source_ptr = data.ptr_at(source_offset)
        if merged:
            prev_target, prev_source, prev_nbytes = merged[-1]
            if (
                target_offset == prev_target + prev_nbytes
                and source_ptr == prev_source + prev_nbytes
            ):
                merged[-1] = (prev_target, prev_source, prev_nbytes + nbytes)
                continue
        merged.append((target_offset, source_ptr, nbytes))
    end = max(target + nbytes for target, _source, nbytes in merged)
    whole = slice_dst(dst, 0, end)
    base_ptr = cuda_array_ptr(whole)
    if (
        base_ptr is not None
        and getattr(whole, "ndim", 0) == 1
        and getattr(whole, "itemsize", 0) == 1
        and getattr(getattr(whole, "flags", None), "c_contiguous", False)
        and int(whole.size) >= end
    ):
        # A flat byte array: offsets are pointer arithmetic, so a copy with
        # thousands of runs does not build one CuPy view per run.
        return [(base_ptr + target, source, n) for target, source, n in merged]
    runs = []
    for target_offset, source_ptr, nbytes in merged:
        dst_ptr = cuda_array_ptr(slice_dst(dst, target_offset, nbytes))
        if dst_ptr is None:
            raise TypeError("grouped CUDA copy target lost CUDA array interface")
        runs.append((dst_ptr, source_ptr, nbytes))
    return runs


#: ``cudaMemcpySrcAccessOrderStream``: sources are read in stream order.
_SRC_ACCESS_ORDER_STREAM = 1


class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _MemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("src_access_order", ctypes.c_int),
        ("src_loc_hint", _MemLocation),
        ("dst_loc_hint", _MemLocation),
        ("flags", ctypes.c_uint),
    ]


_memcpy_batch: Any = None


def _batch_memcpy_fn() -> Any:
    """Return ``cudaMemcpyBatchAsync`` from the process CUDA runtime.

    CuPy and torch already load ``libcudart.so.12``, so this binds the same
    runtime instead of a second copy. Requires CUDA 12.8+.
    """
    global _memcpy_batch
    if _memcpy_batch is None:
        fn = ctypes.CDLL("libcudart.so.12").cudaMemcpyBatchAsync
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p] * 3 + [
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        _memcpy_batch = fn
    return _memcpy_batch


def _launch_memcpy_batch(
    dst_ptrs: Any, src_ptrs: Any, sizes: Any, stream_ptr: int
) -> None:
    """Queue every run as one ``cudaMemcpyBatchAsync`` on the copy engines.

    One call replaces a Python loop of ``cudaMemcpyAsync`` launches, so the
    submitting thread spends O(1) instead of O(runs) per group. The stream
    must not be the legacy default stream.
    """
    import numpy

    dsts = numpy.ascontiguousarray(dst_ptrs, dtype=numpy.uint64)
    srcs = numpy.ascontiguousarray(src_ptrs, dtype=numpy.uint64)
    lengths = numpy.ascontiguousarray(sizes, dtype=numpy.uint64)
    attrs = _MemcpyAttributes(src_access_order=_SRC_ACCESS_ORDER_STREAM)
    attrs_index = ctypes.c_size_t(0)
    fail_index = ctypes.c_size_t(0)
    err = _batch_memcpy_fn()(
        dsts.ctypes.data,
        srcs.ctypes.data,
        lengths.ctypes.data,
        len(lengths),
        ctypes.addressof(attrs),
        ctypes.addressof(attrs_index),
        1,
        ctypes.byref(fail_index),
        stream_ptr,
    )
    if err:
        raise RuntimeError(
            f"cudaMemcpyBatchAsync failed with error {err} at run {fail_index.value}"
        )


def copy_pointer_runs_to_cuda_dst(
    dst: Any,
    target_offsets: Any,
    source_ptrs: Any,
    nbytes: Any,
) -> Any:
    """Queue host-pointer runs into a flat CUDA byte destination.

    Every run goes to the copy engines in one ``cudaMemcpyBatchAsync``, or
    one ``cudaMemcpyAsync`` per run on the legacy default stream, which batch
    copies do not accept.

    Args:
        dst: Flat CuPy-like byte destination, optionally exposing
            ``copy_stream_ptr``.
        target_offsets: ``int64`` array of destination byte offsets.
        source_ptrs: ``uint64`` array of pinned, device-mapped host pointers.
        nbytes: ``int64`` array of run lengths.

    Raises:
        ValueError: If the runs are malformed.

    Returns:
        A CUDA event recorded on the destination stream after the copies.

    Async/thread-safety:
        Enqueues work only. The caller keeps every source alive until the
        returned event completes. Offsets are pointer arithmetic, so no
        per-run destination view is built.
    """
    import cupy
    from cupy.cuda import runtime
    import numpy

    targets = numpy.asarray(target_offsets, dtype=numpy.int64)
    sources = numpy.asarray(source_ptrs, dtype=numpy.uint64)
    sizes = numpy.asarray(nbytes, dtype=numpy.int64)
    if not (len(targets) == len(sources) == len(sizes)) or not len(targets):
        raise ValueError("pointer runs must be nonempty and of equal length")
    end = int((targets + sizes).max())
    whole = slice_dst(dst, 0, end)
    base_ptr = cuda_array_ptr(whole)
    if base_ptr is None or int(whole.size) < end or int(targets.min()) < 0:
        raise ValueError("pointer runs exceed the CUDA destination")
    dst_ptrs = targets.astype(numpy.uint64) + numpy.uint64(base_ptr)
    stream_ptr = int(getattr(dst, "copy_stream_ptr", 0))
    with cupy.cuda.Device(int(whole.device.id)):
        if stream_ptr:
            _launch_memcpy_batch(dst_ptrs, sources, sizes, stream_ptr)
        else:
            for d, s, n in zip(
                dst_ptrs.tolist(), sources.tolist(), sizes.tolist(), strict=True
            ):
                runtime.memcpyAsync(d, s, n, runtime.memcpyHostToDevice, stream_ptr)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(cupy.cuda.ExternalStream(stream_ptr))
    return event


def destination_copy_event(dst: Any, chunks: list[CopyChunk]) -> Any | None:
    """Create an unrecorded event for a CUDA destination copy, or return None.

    Args:
        dst: Byte-addressable destination with an optional private copy stream.
        chunks: Nonempty source/destination ranges about to be copied.

    Returns:
        CUDA event on the destination device, or None for synchronous CPU copies.

    Async/thread-safety:
        Call on the transfer event loop before submission. The caller must retain
        sources, record this event after submission and poll it before release.
    """
    if not chunks:
        return None
    target = slice_dst(dst, chunks[0][0], chunks[0][3])
    if cuda_array_ptr(target) is None:
        return None
    import cupy

    with cupy.cuda.Device(int(target.device.id)):
        return cupy.cuda.Event(disable_timing=True)


def record_destination_copy_event(dst: Any, event: Any) -> None:
    """Record a completion event after copies on the destination's stream.

    Args:
        dst: Destination that supplied the private copy stream, or stream zero.
        event: Event created by destination_copy_event before submission.

    Returns:
        None.

    Async/thread-safety:
        Call on the same transfer loop immediately after the grouped submission.
        Recording is nonblocking and must also run after a partial copy failure.
    """
    import cupy

    target = slice_dst(dst, 0, 0)
    with cupy.cuda.Device(int(target.device.id)):
        event.record(cupy.cuda.ExternalStream(int(getattr(dst, "copy_stream_ptr", 0))))


def cuda_array_ptr(dst: Any) -> int | None:
    """Return a CUDA device pointer for a CuPy-like array destination.

    Args:
        dst: candidate destination object.

    Returns:
        Integer device pointer when ``dst`` exposes the CuPy ``data.ptr``
        interface, otherwise None.
    """
    data = getattr(dst, "data", None)
    ptr = getattr(data, "ptr", None)
    if ptr is None:
        return None
    return int(ptr)


def slice_dst(dst: Any, offset: int, nbytes: int) -> Any:
    """Return a writable destination slice for ``[offset, offset + nbytes)``.

    Args:
        dst: writable buffer or CuPy ndarray.
        offset: byte offset into ``dst``.
        nbytes: number of bytes the slice must cover.

    Returns:
        A sliced view appropriate for the destination type.
    """
    if hasattr(dst, "set"):
        try:
            return dst[offset : offset + nbytes]
        except (TypeError, KeyError, IndexError):
            if offset == 0:
                return dst
            raise
    if isinstance(dst, bytearray | memoryview):
        return memoryview(dst).cast("B")[offset : offset + nbytes]
    try:
        return dst[offset : offset + nbytes]
    except (TypeError, KeyError, IndexError):
        pass
    return memoryview(dst).cast("B")[offset : offset + nbytes]


def slice_src(src: Any, offset: int, nbytes: int) -> Any:
    """Return a readable source slice for ``[offset, offset + nbytes)``.

    Args:
        src: readable buffer or CuPy ndarray.
        offset: byte offset into ``src``.
        nbytes: number of bytes the slice must cover.

    Returns:
        A sliced view appropriate for the source type.
    """
    if hasattr(src, "get"):
        return src[offset : offset + nbytes]
    try:
        return src[offset : offset + nbytes]
    except (TypeError, KeyError, IndexError):
        pass
    return memoryview(src).cast("B")[offset : offset + nbytes]


def copy_src_to_pinned(
    src: Any,
    pinned: PinnedMemorySlice,
    target_offset: int,
    nbytes: int,
) -> None:
    """Copy bytes from a CPU or CuPy source into pinned host memory.

    Args:
        src: readable byte buffer or CuPy ndarray.
        pinned: destination slice leased from the L1 pool.
        target_offset: byte offset into ``pinned`` to write at.
        nbytes: number of bytes to copy.
    """
    if hasattr(src, "data") and getattr(src.data, "ptr", None) is not None:
        import cupy
        from cupy.cuda import runtime

        with cupy.cuda.Device(int(src.device.id)):
            runtime.memcpy(
                pinned.ptr_at(target_offset),
                int(src.data.ptr),
                nbytes,
                runtime.memcpyDeviceToHost,
            )
        return
    pinned.view()[target_offset : target_offset + nbytes] = memoryview(src).cast("B")[
        :nbytes
    ]


def coalesce_copy_chunks(chunks: list[CopyChunk]) -> list[CopyChunk]:
    """Merge adjacent copies sharing one source slice and contiguous offsets.

    Args:
        chunks: grouped copy chunks to merge.

    Returns:
        Chunks sorted by target offset with adjacent runs merged.
    """
    ordered = sorted(chunks, key=lambda item: item[0])
    merged: list[CopyChunk] = []
    for target_offset, data, source_offset, nbytes in ordered:
        if not merged:
            merged.append((target_offset, data, source_offset, nbytes))
            continue
        prev_target, prev_data, prev_source, prev_nbytes = merged[-1]
        if (
            prev_data is data
            and target_offset == prev_target + prev_nbytes
            and source_offset == prev_source + prev_nbytes
        ):
            merged[-1] = (prev_target, prev_data, prev_source, prev_nbytes + nbytes)
            continue
        merged.append((target_offset, data, source_offset, nbytes))
    return merged


def copy_grouped_to_cuda_dst(dst: Any, chunks: list[CopyChunk]) -> int:
    """Copy grouped pinned ranges into a CUDA destination.

    Args:
        dst: CuPy ndarray destination.
        chunks: grouped copy chunks from the L1 tier.

    Returns:
        Number of merged runs copied.

    Async/thread-safety:
        Work is queued on the destination's copy stream.
    """
    import cupy
    from cupy.cuda import runtime

    runs = _merge_cuda_runs(dst, chunks)
    stream_ptr = int(getattr(dst, "copy_stream_ptr", 0))
    device_id = int(slice_dst(dst, 0, 0).device.id)
    # Device-resident sources (the device stash) copy device-to-device.
    on_device = any(getattr(chunk[1], "on_device", False) for chunk in chunks)
    with cupy.cuda.Device(device_id):
        if on_device:
            for dst_ptr, source_ptr, nbytes in runs:
                runtime.memcpyAsync(
                    dst_ptr, source_ptr, nbytes, runtime.memcpyDefault, stream_ptr
                )
            return len(runs)
        for dst_ptr, source_ptr, nbytes in runs:
            runtime.memcpyAsync(
                dst_ptr, source_ptr, nbytes, runtime.memcpyHostToDevice, stream_ptr
            )
    return len(runs)


def copy_grouped_to_dst(dst: Any, chunks: list[CopyChunk]) -> int:
    """Copy source chunks into the destination without staging repacks.

    Args:
        dst: writable buffer or CuPy ndarray destination.
        chunks: grouped copy chunks from the L1 tier.

    Returns:
        Number of merged runs copied into a CUDA destination, else 0.
    """
    if not chunks:
        return 0
    first_target = slice_dst(dst, chunks[0][0], chunks[0][3])
    if cuda_array_ptr(first_target) is not None:
        return copy_grouped_to_cuda_dst(dst, chunks)

    for target_offset, data, source_offset, nbytes in coalesce_copy_chunks(chunks):
        target = slice_dst(dst, target_offset, nbytes)
        if hasattr(target, "set"):
            import numpy

            host = numpy.frombuffer(
                data.view()[source_offset : source_offset + nbytes],
                dtype=numpy.uint8,
                count=nbytes,
            )
            target.set(host)
            continue
        dst_view = memoryview(dst).cast("B")
        dst_view[target_offset : target_offset + nbytes] = data.view()[
            source_offset : source_offset + nbytes
        ]
    return 0
