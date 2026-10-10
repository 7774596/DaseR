# SPDX-License-Identifier: Apache-2.0
"""Padded codebook layout of the branchless compressed-KV decoder."""

import pytest

from daser.compression import CompressedStoreGeometry, default_online_codebooks
from daser.compression.format import CODEBOOK_ENTRIES

tilelang_kv = pytest.importorskip("daser.ops.tilelang_compressed_kv")


def test_packed_rows_place_each_level_in_one_byte_permute_pair() -> None:
    """Primary entries fill bytes 0-6 and secondary entries bytes 8-14."""
    geometry = CompressedStoreGeometry(
        num_slots=1,
        slot_size=2 * 2 * 16 * 2 * 128 * 2,
        block_tokens=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=128,
        tile_scalars=256,
    )
    codebooks = default_online_codebooks(geometry)
    planes = geometry.num_layers * 2
    packed = tilelang_kv.pack_branchless_codebooks(codebooks, planes)
    width = tilelang_kv.BRANCHLESS_CODEBOOK_BYTES
    assert len(packed) == planes * width
    for plane in range(planes):
        row = codebooks[plane * CODEBOOK_ENTRIES : (plane + 1) * CODEBOOK_ENTRIES]
        out = packed[plane * width : (plane + 1) * width]
        assert out[:7] == row[:7]
        assert out[8:15] == row[7:14]
        assert out[7] == 0 and out[15] == 0


def test_packing_rejects_mismatched_codebook_length() -> None:
    """A codebook blob for another plane count is refused."""
    with pytest.raises(ValueError, match="codebook bytes"):
        tilelang_kv.pack_branchless_codebooks(bytes(CODEBOOK_ENTRIES * 3), 4)
