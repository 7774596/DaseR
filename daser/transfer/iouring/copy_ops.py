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

#: Zero-copy gather: each thread block copies whole pieces with 16-byte loads
#: straight from pinned host memory, so small scattered pieces reach PCIe
#: bandwidth where one cudaMemcpyAsync per piece is launch-bound.
_GATHER_SOURCE = r"""
extern "C" __global__ void daser_gather16(
    const unsigned long long* src, const unsigned long long* dst,
    const unsigned long long* nbytes, int count) {
  for (int piece = blockIdx.x; piece < count; piece += gridDim.x) {
    const uint4* s = (const uint4*)src[piece];
    uint4* d = (uint4*)dst[piece];
    unsigned long long n = nbytes[piece] / 16;
    for (unsigned long long i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
  }
}
"""
#: Layer-group gather: the same copy, but every record contributes the range
#: at one shared offset, so the per-request record table is uploaded once and
#: each layer group launches with scalar offsets only.
_LAYER_GATHER_SOURCE = r"""
extern "C" __global__ void daser_layer_gather16(
    const unsigned long long* src, const unsigned long long* dst,
    unsigned long long src_offset, unsigned long long dst_offset,
    unsigned long long nbytes, int count) {
  unsigned long long n = nbytes / 16;
  for (int rec = blockIdx.x; rec < count; rec += gridDim.x) {
    const uint4* s = (const uint4*)(src[rec] + src_offset);
    uint4* d = (uint4*)(dst[rec] + dst_offset);
    for (unsigned long long i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
  }
}
"""
_GATHER_BLOCKS = 1056
_GATHER_THREADS = 512
_gather_kernel: Any = None
_layer_gather_kernel: Any = None


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


#: Per-run copy modes for ``copy_pointer_runs_to_cuda_dst``.
COPY_MODE_GATHER = "gather"
COPY_MODE_MEMCPY = "memcpy"
COPY_MODE_BATCH = "batch"
COPY_MODES = (COPY_MODE_GATHER, COPY_MODE_MEMCPY, COPY_MODE_BATCH)
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


def _launch_gather(meta: Any, stream_ptr: int) -> None:
    """Queue one zero-copy gather kernel for 16-byte-aligned runs.

    ``meta`` is a ``(3, runs)`` uint64 array of destination pointers, source
    pointers and lengths. Must run under the destination's
    ``cupy.cuda.Device`` context.
    """
    global _gather_kernel
    import cupy
    import numpy

    if _gather_kernel is None:
        _gather_kernel = cupy.RawKernel(_GATHER_SOURCE, "daser_gather16")
    count = int(meta.shape[1])
    stream = cupy.cuda.ExternalStream(stream_ptr)
    with stream:
        device_meta = cupy.asarray(numpy.ascontiguousarray(meta, dtype=numpy.uint64))
        _gather_kernel(
            (min(_GATHER_BLOCKS, count),),
            (_GATHER_THREADS,),
            (
                device_meta[1],
                device_meta[0],
                device_meta[2],
                numpy.int32(count),
            ),
        )


class LayerGatherPlan:
    """Per-request record table for layer-group copies into a CUDA buffer.

    Record ``i`` starts at host address ``src_ptrs[i]`` (pinned, mapped) and
    its packed destination block starts at ``dst_ptrs[i]``. The table is
    uploaded once; each ``copy`` then moves ``[src_offset, src_offset +
    nbytes)`` of every record to ``dst_offset`` of its destination block with
    one kernel launch and no per-record host work.

    Args:
        src_ptrs: Host pointers of the record starts.
        dst_ptrs: Device pointers of the destination blocks.
        stream_ptr: CUDA stream the copies are queued on.
        device_id: CUDA device owning the destination.

    Async/thread-safety:
        Not thread-safe. The caller keeps the source slices and destination
        alive until the stream passes the last ``copy``; the table itself is
        a stream-ordered allocation held by this object.
    """

    def __init__(
        self,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        stream_ptr: int,
        device_id: int,
    ) -> None:
        if len(src_ptrs) != len(dst_ptrs) or not src_ptrs:
            raise ValueError("src_ptrs and dst_ptrs must be equal and nonempty")
        if any((s | d) % 16 for s, d in zip(src_ptrs, dst_ptrs, strict=True)):
            raise ValueError("record pointers must be 16-byte aligned")
        global _layer_gather_kernel
        import cupy
        import numpy

        self._count = len(src_ptrs)
        self._stream_ptr = stream_ptr
        self._device_id = device_id
        with cupy.cuda.Device(device_id):
            if _layer_gather_kernel is None:
                _layer_gather_kernel = cupy.RawKernel(
                    _LAYER_GATHER_SOURCE, "daser_layer_gather16"
                )
            with cupy.cuda.ExternalStream(stream_ptr):
                self._src = cupy.asarray(numpy.array(src_ptrs, dtype=numpy.uint64))
                self._dst = cupy.asarray(numpy.array(dst_ptrs, dtype=numpy.uint64))

    @property
    def count(self) -> int:
        """Return the number of records in the plan."""
        return self._count

    def copy(self, src_offset: int, dst_offset: int, nbytes: int) -> None:
        """Queue one layer-group copy from every record.

        Args:
            src_offset: Byte offset of the range inside each record.
            dst_offset: Byte offset inside each destination block.
            nbytes: Range length; offsets and length must be 16-byte multiples.

        Returns:
            None. The copy is queued on the plan's stream.
        """
        if (src_offset | dst_offset | nbytes) % 16:
            raise ValueError("layer gather offsets and length must be 16-aligned")
        import cupy
        import numpy

        with cupy.cuda.Device(self._device_id):
            with cupy.cuda.ExternalStream(self._stream_ptr):
                _layer_gather_kernel(
                    (min(_GATHER_BLOCKS, self._count),),
                    (_GATHER_THREADS,),
                    (
                        self._src,
                        self._dst,
                        numpy.uint64(src_offset),
                        numpy.uint64(dst_offset),
                        numpy.uint64(nbytes),
                        numpy.int32(self._count),
                    ),
                )


def copy_pointer_runs_to_cuda_dst(
    dst: Any,
    target_offsets: Any,
    source_ptrs: Any,
    nbytes: Any,
    gather_min_runs: int = 0,
    copy_mode: str = COPY_MODE_GATHER,
) -> Any:
    """Queue host-pointer runs into a flat CUDA byte destination.

    Args:
        dst: Flat CuPy-like byte destination, optionally exposing
            ``copy_stream_ptr``.
        target_offsets: ``int64`` array of destination byte offsets.
        source_ptrs: ``uint64`` array of pinned, device-mapped host pointers.
        nbytes: ``int64`` array of run lengths.
        gather_min_runs: Use one zero-copy gather launch when at least this
            many runs are given and all are 16-byte aligned (0 disables).
        copy_mode: ``gather`` applies ``gather_min_runs``; ``memcpy`` issues
            one copy-engine ``cudaMemcpyAsync`` per run; ``batch`` submits
            every run in one copy-engine ``cudaMemcpyBatchAsync`` (falls back
            to ``memcpy`` on the legacy default stream).

    Raises:
        ValueError: If the runs are malformed or the copy mode is unknown.

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
    if copy_mode not in COPY_MODES:
        raise ValueError(f"unknown copy mode: {copy_mode}")
    end = int((targets + sizes).max())
    whole = slice_dst(dst, 0, end)
    base_ptr = cuda_array_ptr(whole)
    if base_ptr is None or int(whole.size) < end or int(targets.min()) < 0:
        raise ValueError("pointer runs exceed the CUDA destination")
    dst_ptrs = targets.astype(numpy.uint64) + numpy.uint64(base_ptr)
    stream_ptr = int(getattr(dst, "copy_stream_ptr", 0))
    with cupy.cuda.Device(int(whole.device.id)):
        aligned = not (
            numpy.bitwise_or.reduce(dst_ptrs | sources | sizes.astype(numpy.uint64))
            % 16
        )
        if (
            copy_mode == COPY_MODE_GATHER
            and gather_min_runs
            and len(sizes) >= gather_min_runs
            and aligned
        ):
            _launch_gather(
                numpy.stack([dst_ptrs, sources, sizes.astype(numpy.uint64)]),
                stream_ptr,
            )
        elif copy_mode == COPY_MODE_BATCH and stream_ptr:
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


def copy_grouped_to_cuda_dst(
    dst: Any, chunks: list[CopyChunk], gather_min_runs: int = 0
) -> int:
    """Copy grouped pinned ranges into a CUDA destination.

    Args:
        dst: CuPy ndarray destination.
        chunks: grouped copy chunks from the L1 tier.
        gather_min_runs: Use one zero-copy gather kernel instead of one
            ``cudaMemcpyAsync`` per run when at least this many merged runs
            remain and every run is 16-byte aligned (0 disables).

    Returns:
        Number of merged runs copied.

    Async/thread-safety:
        Work is queued on the destination's copy stream. Gather metadata is a
        stream-ordered CuPy allocation, so it stays valid until the kernel runs.
    """
    import cupy
    from cupy.cuda import runtime

    runs = _merge_cuda_runs(dst, chunks)
    stream_ptr = int(getattr(dst, "copy_stream_ptr", 0))
    device_id = int(slice_dst(dst, 0, 0).device.id)
    with cupy.cuda.Device(device_id):
        if (
            gather_min_runs
            and len(runs) >= gather_min_runs
            and all(not (d | s | n) % 16 for d, s, n in runs)
        ):
            import numpy

            _launch_gather(numpy.array(runs, dtype=numpy.uint64).T, stream_ptr)
            return len(runs)
        for dst_ptr, source_ptr, nbytes in runs:
            runtime.memcpyAsync(
                dst_ptr, source_ptr, nbytes, runtime.memcpyHostToDevice, stream_ptr
            )
    return len(runs)


def copy_grouped_to_dst(
    dst: Any, chunks: list[CopyChunk], gather_min_runs: int = 0
) -> None:
    """Copy source chunks into the destination without staging repacks.

    Args:
        dst: writable buffer or CuPy ndarray destination.
        chunks: grouped copy chunks from the L1 tier.
        gather_min_runs: CUDA destinations only; see
            ``copy_grouped_to_cuda_dst``.
    """
    if not chunks:
        return
    first_target = slice_dst(dst, chunks[0][0], chunks[0][3])
    if cuda_array_ptr(first_target) is not None:
        copy_grouped_to_cuda_dst(dst, chunks, gather_min_runs)
        return

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
