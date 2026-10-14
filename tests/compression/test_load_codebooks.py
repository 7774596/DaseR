# SPDX-License-Identifier: Apache-2.0
"""Codebook layout consumed by the compressed-KV load kernels."""

import pytest

from daser.compression import CompressedStoreGeometry, default_online_codebooks
from daser.compression.format import CODEBOOK_ENTRIES

tilelang_kv = pytest.importorskip("daser.ops.tilelang_compressed_kv")

_PACKED_ROW_BYTES = 16


def _codebooks(tile_scalars: int) -> tuple[bytes, int]:
    geometry = CompressedStoreGeometry(
        num_slots=1,
        slot_size=2 * 2 * 16 * 2 * 128 * 2,
        block_tokens=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=128,
        tile_scalars=tile_scalars,
    )
    return default_online_codebooks(geometry), geometry.num_layers * 2


def test_word_parallel_rows_place_each_level_in_one_byte_permute_pair() -> None:
    """Primary entries fill bytes 0-6 and secondary entries bytes 8-14."""
    codebooks, planes = _codebooks(256)
    packed = tilelang_kv.pack_load_codebooks(codebooks, planes, 256)
    assert len(packed) == planes * _PACKED_ROW_BYTES
    for plane in range(planes):
        row = codebooks[plane * CODEBOOK_ENTRIES : (plane + 1) * CODEBOOK_ENTRIES]
        out = packed[plane * _PACKED_ROW_BYTES : (plane + 1) * _PACKED_ROW_BYTES]
        assert out[:7] == row[:7]
        assert out[8:15] == row[7:14]
        assert out[7] == 0 and out[15] == 0


def test_generic_tiles_keep_the_codebooks_unchanged() -> None:
    """Tiles other than 256 scalars decode with the original rows."""
    codebooks, planes = _codebooks(1024)
    assert tilelang_kv.pack_load_codebooks(codebooks, planes, 1024) == codebooks


def test_packing_rejects_mismatched_codebook_length() -> None:
    """A codebook blob for another plane count is refused."""
    with pytest.raises(ValueError, match="codebook bytes"):
        tilelang_kv.pack_load_codebooks(bytes(CODEBOOK_ENTRIES * 3), 4, 256)
