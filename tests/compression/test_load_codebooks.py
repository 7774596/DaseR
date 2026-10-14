# SPDX-License-Identifier: Apache-2.0
"""Codebook layout consumed by the compressed-KV load kernels."""

import pytest

from daser.compression import CompressedStoreGeometry, default_online_codebooks
from daser.compression.format import CODEBOOK_ENTRIES

tilelang_kv = pytest.importorskip("daser.ops.tilelang_compressed_kv")

_PACKED_ROW_BYTES = 16


def _codebooks() -> tuple[bytes, int]:
    geometry = CompressedStoreGeometry(
        num_slots=1,
        slot_size=2 * 2 * 16 * 2 * 128 * 2,
        block_tokens=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=128,
    )
    return default_online_codebooks(geometry), geometry.num_layers * 2


def test_rows_place_each_level_in_one_byte_permute_pair() -> None:
    """Primary entries fill bytes 0-6 and secondary entries bytes 8-14."""
    codebooks, planes = _codebooks()
    packed = tilelang_kv.pack_load_codebooks(codebooks, planes)
    assert len(packed) == planes * _PACKED_ROW_BYTES
    for plane in range(planes):
        row = codebooks[plane * CODEBOOK_ENTRIES : (plane + 1) * CODEBOOK_ENTRIES]
        out = packed[plane * _PACKED_ROW_BYTES : (plane + 1) * _PACKED_ROW_BYTES]
        assert out[:7] == row[:7]
        assert out[8:15] == row[7:14]
        assert out[7] == 0 and out[15] == 0


def test_packing_rejects_mismatched_codebook_length() -> None:
    """A codebook blob for another plane count is refused."""
    with pytest.raises(ValueError, match="codebook bytes"):
        tilelang_kv.pack_load_codebooks(bytes(CODEBOOK_ENTRIES * 3), 4)


@pytest.mark.parametrize("compile_kernel", ["load", "store"])
def test_kernels_reject_tiles_other_than_the_online_size(compile_kernel: str) -> None:
    """Only the online 256-scalar tile has a TileLang codec kernel."""
    geometry = {"num_blocks": 2, "num_planes": 4, "plane_scalars": 2048}
    with pytest.raises(ValueError, match="tile_scalars"):
        if compile_kernel == "load":
            tilelang_kv.compile_load_kernel(
                **geometry, head_dim=128, tile_scalars=1024, staging_bytes=4096
            )
        else:
            tilelang_kv.compile_store_kernels(
                **geometry, head_dim=128, tile_scalars=1024
            )
