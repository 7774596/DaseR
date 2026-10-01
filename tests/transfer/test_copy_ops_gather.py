# SPDX-License-Identifier: Apache-2.0

"""Grouped pinned-to-CUDA copies with and without the zero-copy gather."""

import pytest

from daser.transfer.iouring import copy_ops
from daser.transfer.iouring.pinned_pool import PinnedMemoryPool

cupy = pytest.importorskip("cupy")


def _scattered_copy(piece: int, gather_min_runs: int) -> tuple[bytes, bytes, int]:
    """Copy eight scattered pinned pieces to permuted GPU offsets.

    Returns the expected bytes, the GPU result and the reported run count.
    """
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")
    count = 8
    stride = 3 * piece + 512
    pool = PinnedMemoryPool(stride * count + piece)
    try:
        source = pool.allocate(stride * count + piece)
        assert source is not None
        view = source.view()
        for i in range(len(view)):
            view[i] = (i * 7 + 3) % 251
        order = [5, 2, 7, 0, 3, 6, 1, 4]
        chunks: list[copy_ops.CopyChunk] = [
            (slot * piece, source, i * stride, piece) for i, slot in enumerate(order)
        ]
        expected = bytearray(count * piece)
        for target, _data, offset, nbytes in chunks:
            expected[target : target + nbytes] = view[offset : offset + nbytes]
        dst = cupy.zeros(count * piece, dtype=cupy.uint8)
        runs = copy_ops.copy_grouped_to_cuda_dst(dst, chunks, gather_min_runs)
        cupy.cuda.Device().synchronize()
        result = cupy.asnumpy(dst).tobytes()
        source.close()
        return bytes(expected), result, runs
    finally:
        pool.close()


@pytest.mark.parametrize("gather_min_runs", [0, 2, 9])
def test_aligned_scattered_copy_matches_source(gather_min_runs: int) -> None:
    """Gathered, memcpy and below-threshold copies produce identical bytes."""
    expected, result, runs = _scattered_copy(4096, gather_min_runs)
    assert runs == 8
    assert result == expected


def test_unaligned_pieces_fall_back_to_memcpy() -> None:
    """Pieces that are not 16-byte multiples still copy correctly."""
    expected, result, runs = _scattered_copy(4100, 2)
    assert runs == 8
    assert result == expected
