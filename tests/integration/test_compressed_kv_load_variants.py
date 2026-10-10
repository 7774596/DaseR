# SPDX-License-Identifier: Apache-2.0
"""Research load variants of the 256-scalar fused compressed-KV decoder."""

import numpy as np
import pytest
import torch

from daser.compression import (
    CompressedStoreGeometry,
    SlotMode,
    default_online_codebooks,
    encode_slot,
)
from daser.config import (
    LOAD_REGISTER_CODEBOOK_ENV,
    LOAD_TILE_GROUPS_ENV,
    LOAD_WORD_READS_ENV,
)
from daser.ops.compressed_kv import FusedCompressedKVDecoder
from daser.ops.kv_layout import allocate_cross_layer_kv_cache

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _geometry() -> CompressedStoreGeometry:
    """Odd head_dim keeps every lane and record field byte-misaligned."""
    return CompressedStoreGeometry(
        num_slots=2,
        slot_size=2 * 2 * 128 * 4 * 127 * 2,
        block_tokens=128,
        num_layers=2,
        num_kv_heads=4,
        head_dim=127,
        tile_scalars=256,
    )


def _slot(geometry: CompressedStoreGeometry, seed: int) -> bytes:
    """Return a slot whose high bytes hit primary, secondary and raw escapes."""
    rng = np.random.default_rng(seed)
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = rng.integers(0, 256, geometry.slot_size // 2, dtype=np.uint8)
    common = np.array([0x3E, 0x3F, 0x40, 0xBF, 0xBE, 0x3D, 0xC0], dtype=np.uint8)
    rare = np.array([0x3C, 0xBD, 0x41, 0xC1], dtype=np.uint8)
    high = common[rng.integers(0, len(common), geometry.slot_size // 2)]
    pick = rng.random(geometry.slot_size // 2)
    high[pick < 0.15] = rare[rng.integers(0, len(rare), int((pick < 0.15).sum()))]
    high[pick < 0.01] = rng.integers(0, 256, int((pick < 0.01).sum()), dtype=np.uint8)
    raw[1::2] = high
    return raw.tobytes()


def _cache(num_blocks: int, geometry: CompressedStoreGeometry) -> torch.Tensor:
    cache = allocate_cross_layer_kv_cache(
        num_blocks,
        geometry.num_layers,
        geometry.block_tokens,
        geometry.num_kv_heads,
        geometry.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    cache.zero_()
    return cache


def _restore(
    staging: torch.Tensor,
    offsets: list[int],
    modes: list[int],
    geometry: CompressedStoreGeometry,
    codebooks: bytes,
) -> torch.Tensor:
    cache = _cache(len(offsets), geometry)
    decoder = FusedCompressedKVDecoder(
        kv_cache=cache,
        codebooks=codebooks,
        tile_scalars=geometry.tile_scalars,
        ring_depth=1,
        max_slots_per_buffer=len(offsets),
    )
    stream = torch.cuda.Stream()
    restored = decoder.decode(
        staging=staging,
        staging_offsets=offsets,
        block_ids=list(range(len(offsets))),
        modes=modes,
        buffer_index=0,
        stream=stream,
    )
    stream.synchronize()
    assert restored == len(offsets)
    return cache


@pytest.mark.parametrize("lead", [0, 1, 2, 3])
def test_word_reads_restore_misaligned_records_byte_exact(
    monkeypatch: pytest.MonkeyPatch, lead: int
) -> None:
    """Records at every byte phase, the last ending at the extent, are exact."""
    geometry = _geometry()
    codebooks = default_online_codebooks(geometry)
    raws = [_slot(geometry, seed) for seed in (11, 12)]
    encoded = [
        encode_slot(raw, slot_id=i, geometry=geometry, codebooks=codebooks)
        for i, raw in enumerate(raws)
    ]
    assert all(e.mode is SlotMode.COMPRESSED for e in encoded)
    first = b"\xa5" * lead + encoded[0].payload
    # Pad so the second record ends exactly at a 4-byte aligned extent.
    pad = (-(len(first) + len(encoded[1].payload))) % 4
    blob = first + b"\x5a" * pad + encoded[1].payload
    offsets = [lead, len(first) + pad]
    staging = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy()).cuda()
    monkeypatch.setenv(LOAD_WORD_READS_ENV, "1")
    cache = _restore(staging, offsets, [1, 1], geometry, codebooks)
    for block, raw in enumerate(raws):
        assert cache[block].view(torch.uint8).cpu().numpy().tobytes() == raw


def test_word_reads_match_byte_path_with_raw_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed compressed/raw restores decode identically with the switch on/off."""
    geometry = _geometry()
    codebooks = default_online_codebooks(geometry)
    compressed = encode_slot(
        _slot(geometry, 21), slot_id=0, geometry=geometry, codebooks=codebooks
    )
    fallback = _slot(geometry, 22)
    blob = b"\x00" + compressed.payload + fallback
    blob += b"\x00" * ((-len(blob)) % 4)
    offsets = [1, 1 + len(compressed.payload)]
    staging = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy()).cuda()
    results = []
    for switch in ("0", "1"):
        monkeypatch.setenv(LOAD_WORD_READS_ENV, switch)
        results.append(_restore(staging, offsets, [1, 0], geometry, codebooks))
    assert torch.equal(results[0].view(torch.uint16), results[1].view(torch.uint16))
    assert results[1][1].view(torch.uint8).cpu().numpy().tobytes() == fallback


def test_word_reads_reject_misaligned_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    """A staging view that is not 4-byte aligned is refused before launch."""
    geometry = _geometry()
    codebooks = default_online_codebooks(geometry)
    encoded = encode_slot(
        _slot(geometry, 31), slot_id=0, geometry=geometry, codebooks=codebooks
    )
    backing = torch.zeros(len(encoded.payload) + 8, dtype=torch.uint8, device="cuda")
    staging = backing[1 : 1 + len(encoded.payload)]
    monkeypatch.setenv(LOAD_WORD_READS_ENV, "1")
    with pytest.raises(ValueError, match="4-byte aligned"):
        _restore(staging, [0], [1], geometry, codebooks)


@pytest.mark.parametrize("word_reads", ["0", "1"])
@pytest.mark.parametrize("groups", [2, 3, 5])
def test_tile_groups_restore_byte_exact(
    monkeypatch: pytest.MonkeyPatch, groups: int, word_reads: str
) -> None:
    """Coarsened CTAs, including groups past the last tile, restore exactly."""
    geometry = _geometry()
    codebooks = default_online_codebooks(geometry)
    raws = [_slot(geometry, seed) for seed in (41, 42, 43)]
    encoded = encode_slot(raws[0], slot_id=0, geometry=geometry, codebooks=codebooks)
    second = encode_slot(raws[2], slot_id=1, geometry=geometry, codebooks=codebooks)
    blob = b"\x01\x02\x03" + encoded.payload + raws[1] + second.payload
    blob += b"\x00" * ((-len(blob)) % 4)
    offsets = [3, 3 + len(encoded.payload), 3 + len(encoded.payload) + len(raws[1])]
    staging = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy()).cuda()
    monkeypatch.setenv(LOAD_WORD_READS_ENV, word_reads)
    monkeypatch.setenv(LOAD_TILE_GROUPS_ENV, str(groups))
    cache = _restore(staging, offsets, [1, 0, 1], geometry, codebooks)
    for block, raw in enumerate(raws):
        assert cache[block].view(torch.uint8).cpu().numpy().tobytes() == raw


@pytest.mark.parametrize(("word_reads", "groups"), [("0", "1"), ("1", "3")])
def test_register_codebook_restore_byte_exact(
    monkeypatch: pytest.MonkeyPatch, word_reads: str, groups: str
) -> None:
    """Register codebooks decode primary, secondary and raw escapes exactly."""
    geometry = _geometry()
    codebooks = default_online_codebooks(geometry)
    raws = [_slot(geometry, seed) for seed in (51, 52)]
    encoded = [
        encode_slot(raw, slot_id=i, geometry=geometry, codebooks=codebooks)
        for i, raw in enumerate(raws)
    ]
    blob = b"\x07\x07" + encoded[0].payload + encoded[1].payload
    blob += b"\x00" * ((-len(blob)) % 4)
    offsets = [2, 2 + len(encoded[0].payload)]
    staging = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy()).cuda()
    monkeypatch.setenv(LOAD_WORD_READS_ENV, word_reads)
    monkeypatch.setenv(LOAD_TILE_GROUPS_ENV, groups)
    monkeypatch.setenv(LOAD_REGISTER_CODEBOOK_ENV, "1")
    cache = _restore(staging, offsets, [1, 1], geometry, codebooks)
    for block, raw in enumerate(raws):
        assert cache[block].view(torch.uint8).cpu().numpy().tobytes() == raw


def test_tile_groups_reject_non_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero coarsening factor is refused when the decoder compiles."""
    geometry = _geometry()
    monkeypatch.setenv(LOAD_TILE_GROUPS_ENV, "0")
    with pytest.raises(ValueError, match="positive integer"):
        FusedCompressedKVDecoder(
            kv_cache=_cache(1, geometry),
            codebooks=default_online_codebooks(geometry),
            tile_scalars=geometry.tile_scalars,
            ring_depth=1,
            max_slots_per_buffer=1,
        )
