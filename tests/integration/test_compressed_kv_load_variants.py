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
from daser.compression.format import PREFIX_GROUP_TILES, SlotHeader
from daser.config import (
    LOAD_BRANCHLESS_DECODE_ENV,
    LOAD_PIPELINED_GROUPS_ENV,
    LOAD_REGISTER_CODEBOOK_ENV,
    LOAD_TILE_GROUPS_ENV,
    LOAD_UNCHECKED_BOUNDS_ENV,
    LOAD_VECTOR_STORES_ENV,
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


@pytest.mark.parametrize("unchecked", ["0", "1"])
@pytest.mark.parametrize("lead", [0, 1, 2, 3])
def test_word_reads_restore_misaligned_records_byte_exact(
    monkeypatch: pytest.MonkeyPatch, lead: int, unchecked: str
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
    monkeypatch.setenv(LOAD_UNCHECKED_BOUNDS_ENV, unchecked)
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


@pytest.mark.parametrize(
    ("word_reads", "pipelined"), [("0", "0"), ("1", "0"), ("1", "1")]
)
@pytest.mark.parametrize("groups", [1, 2, 3, 5])
def test_tile_groups_restore_byte_exact(
    monkeypatch: pytest.MonkeyPatch, groups: int, word_reads: str, pipelined: str
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
    monkeypatch.setenv(LOAD_PIPELINED_GROUPS_ENV, pipelined)
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


@pytest.mark.parametrize(
    ("word_reads", "groups", "register", "pipelined"),
    [
        ("0", "1", "0", "0"),
        ("1", "1", "0", "0"),
        ("1", "3", "1", "0"),
        ("0", "2", "1", "0"),
        ("1", "3", "1", "1"),
    ],
)
def test_grouped_prefix_records_restore_beside_version_one(
    monkeypatch: pytest.MonkeyPatch,
    word_reads: str,
    groups: str,
    register: str,
    pipelined: str,
) -> None:
    """Version-2 group prefixes decode exactly next to v1 and raw records."""
    geometry = _geometry()
    codebooks = default_online_codebooks(geometry)
    raws = [_slot(geometry, seed) for seed in (61, 62, 63)]
    grouped = encode_slot(
        raws[0],
        slot_id=0,
        geometry=geometry,
        codebooks=codebooks,
        prefix_group_tiles=PREFIX_GROUP_TILES,
    )
    per_tile = encode_slot(raws[2], slot_id=1, geometry=geometry, codebooks=codebooks)
    assert len(grouped.payload) <= len(per_tile.payload)
    blob = b"\x05" + grouped.payload + raws[1] + per_tile.payload
    blob += b"\x00" * ((-len(blob)) % 4)
    offsets = [1, 1 + len(grouped.payload), 1 + len(grouped.payload) + len(raws[1])]
    staging = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy()).cuda()
    monkeypatch.setenv(LOAD_WORD_READS_ENV, word_reads)
    monkeypatch.setenv(LOAD_TILE_GROUPS_ENV, groups)
    monkeypatch.setenv(LOAD_REGISTER_CODEBOOK_ENV, register)
    monkeypatch.setenv(LOAD_PIPELINED_GROUPS_ENV, pipelined)
    cache = _restore(staging, offsets, [1, 0, 1], geometry, codebooks)
    for block, raw in enumerate(raws):
        assert cache[block].view(torch.uint8).cpu().numpy().tobytes() == raw


@pytest.mark.parametrize(
    ("word_reads", "groups", "register", "unchecked", "branchless", "pipelined"),
    [
        ("0", "1", "0", "0", "0", "0"),
        ("1", "1", "1", "0", "0", "0"),
        ("1", "2", "1", "0", "0", "0"),
        ("1", "1", "1", "1", "0", "0"),
        ("1", "2", "0", "1", "0", "0"),
        ("1", "1", "0", "0", "1", "0"),
        ("1", "2", "0", "1", "1", "0"),
        ("1", "2", "1", "1", "0", "1"),
        ("1", "3", "0", "1", "1", "1"),
    ],
)
def test_vector_stores_restore_byte_exact_with_fanout(
    monkeypatch: pytest.MonkeyPatch,
    word_reads: str,
    groups: str,
    register: str,
    unchecked: str,
    branchless: str,
    pipelined: str,
) -> None:
    """128-bit stores restore v2, raw and v1 records, a partial tile and fanout."""
    # 3 tokens x 5 heads x 128 = 1920 scalars: the eighth tile is half
    # empty. Eight layers keep the slot 4 KiB aligned and large enough that
    # both records stay compressed instead of falling back to raw.
    geometry = CompressedStoreGeometry(
        num_slots=3,
        slot_size=8 * 2 * 3 * 5 * 128 * 2,
        block_tokens=3,
        num_layers=8,
        num_kv_heads=5,
        head_dim=128,
        tile_scalars=256,
    )
    codebooks = default_online_codebooks(geometry)
    raws = [_slot(geometry, seed) for seed in (71, 72, 73)]
    grouped = encode_slot(
        raws[0],
        slot_id=0,
        geometry=geometry,
        codebooks=codebooks,
        prefix_group_tiles=PREFIX_GROUP_TILES,
    )
    per_tile = encode_slot(raws[2], slot_id=2, geometry=geometry, codebooks=codebooks)
    assert grouped.mode is SlotMode.COMPRESSED
    assert per_tile.mode is SlotMode.COMPRESSED
    blob = b"\x03" + grouped.payload + raws[1] + per_tile.payload
    blob += b"\x00" * ((-len(blob)) % 4)
    offsets = [1, 1 + len(grouped.payload), 1 + len(grouped.payload) + len(raws[1])]
    staging = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy()).cuda()
    monkeypatch.setenv(LOAD_WORD_READS_ENV, word_reads)
    monkeypatch.setenv(LOAD_TILE_GROUPS_ENV, groups)
    monkeypatch.setenv(LOAD_REGISTER_CODEBOOK_ENV, register)
    monkeypatch.setenv(LOAD_VECTOR_STORES_ENV, "1")
    monkeypatch.setenv(LOAD_UNCHECKED_BOUNDS_ENV, unchecked)
    monkeypatch.setenv(LOAD_BRANCHLESS_DECODE_ENV, branchless)
    monkeypatch.setenv(LOAD_PIPELINED_GROUPS_ENV, pipelined)
    cache = _cache(5, geometry)
    decoder = FusedCompressedKVDecoder(
        kv_cache=cache,
        codebooks=codebooks,
        tile_scalars=geometry.tile_scalars,
        ring_depth=1,
        max_slots_per_buffer=5,
    )
    stream = torch.cuda.Stream()
    # Blocks 3 and 4 repeat sources 0 and 2, which selects the fanout kernel.
    decoder.decode(
        staging=staging,
        staging_offsets=offsets + [offsets[0], offsets[2]],
        block_ids=[0, 1, 2, 3, 4],
        modes=[1, 0, 1, 1, 1],
        buffer_index=0,
        stream=stream,
    )
    stream.synchronize()
    for block, raw in enumerate(raws + [raws[0], raws[2]]):
        assert cache[block].view(torch.uint8).cpu().numpy().tobytes() == raw


def test_vector_stores_reject_head_dims_not_divisible_by_eight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 128-bit store layout needs whole eight-scalar words per head."""
    geometry = _geometry()
    monkeypatch.setenv(LOAD_VECTOR_STORES_ENV, "1")
    with pytest.raises(ValueError, match="divisible by 8"):
        FusedCompressedKVDecoder(
            kv_cache=_cache(1, geometry),
            codebooks=default_online_codebooks(geometry),
            tile_scalars=geometry.tile_scalars,
            ring_depth=1,
            max_slots_per_buffer=1,
        )


@pytest.mark.parametrize(
    ("unchecked", "branchless", "pipelined"),
    [("0", "0", "0"), ("1", "0", "0"), ("1", "1", "0"), ("1", "0", "1")],
)
@pytest.mark.parametrize("region", ["header", "prefix", "payload"])
def test_corrupt_records_only_change_their_own_destination(
    monkeypatch: pytest.MonkeyPatch,
    unchecked: str,
    branchless: str,
    pipelined: str,
    region: str,
) -> None:
    """Bit flips in a record never fault or write outside its destination."""
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
    record = encode_slot(
        _slot(geometry, 81),
        slot_id=0,
        geometry=geometry,
        codebooks=codebooks,
        prefix_group_tiles=PREFIX_GROUP_TILES,
    ).payload
    first = SlotHeader.parse(record).descriptors[0]
    prefix = first.prefix_offset  # descriptor offsets are slot-relative
    regions = {
        # Plane descriptors drive every per-plane record offset.
        "header": range(120, 120 + 48 * 2 * geometry.num_layers),
        "prefix": range(prefix, prefix + 16),
        "payload": range(geometry.header_bytes, len(record)),
    }
    monkeypatch.setenv(LOAD_WORD_READS_ENV, "1")
    monkeypatch.setenv(LOAD_REGISTER_CODEBOOK_ENV, "1")
    monkeypatch.setenv(LOAD_VECTOR_STORES_ENV, "1")
    monkeypatch.setenv(LOAD_UNCHECKED_BOUNDS_ENV, unchecked)
    monkeypatch.setenv(LOAD_BRANCHLESS_DECODE_ENV, branchless)
    monkeypatch.setenv(LOAD_PIPELINED_GROUPS_ENV, pipelined)
    cache = _cache(3, geometry)
    decoder = FusedCompressedKVDecoder(
        kv_cache=cache,
        codebooks=codebooks,
        tile_scalars=geometry.tile_scalars,
        ring_depth=1,
        max_slots_per_buffer=1,
    )
    stream = torch.cuda.Stream()
    rng = np.random.default_rng(91)
    sentinel = torch.full_like(cache[0], 0x5A5A, dtype=torch.int16).view(cache.dtype)
    for _ in range(16):
        corrupt = bytearray(record)
        for offset in rng.choice(regions[region], size=8, replace=False):
            corrupt[offset] ^= 1 << int(rng.integers(0, 8))
        blob = bytes(corrupt) + b"\x00" * ((-len(corrupt)) % 4)
        staging = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy()).cuda()
        cache[0].copy_(sentinel)
        cache[2].copy_(sentinel)
        decoder.decode(
            staging=staging,
            staging_offsets=[0],
            block_ids=[1],
            modes=[1],
            buffer_index=0,
            stream=stream,
        )
        # An illegal address would surface here as a CUDA error.
        stream.synchronize()
        assert torch.equal(cache[0], sentinel)
        assert torch.equal(cache[2], sentinel)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_branchless_decode_restores_lanes_full_of_raw_escapes(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    """Lanes whose eight scalars are all raw escapes restore byte-exact."""
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
    rng = np.random.default_rng(97)
    scalars = geometry.slot_size // 2
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = rng.integers(0, 256, scalars, dtype=np.uint8)
    high = np.full(scalars, 0x3F, dtype=np.uint8)
    # Every eighth lane gets eight bytes outside both codebook levels; its
    # neighbours keep the slot compressible.
    lane = np.arange(scalars) // 8
    noisy = lane % 8 == 1
    high[noisy] = rng.integers(0xD0, 0xF0, int(noisy.sum()), dtype=np.uint8)
    raw[1::2] = high
    encoded = encode_slot(
        raw.tobytes(),
        slot_id=0,
        geometry=geometry,
        codebooks=codebooks,
        prefix_group_tiles=PREFIX_GROUP_TILES if version == "v2" else 1,
    )
    assert encoded.mode == SlotMode.COMPRESSED
    blob = b"\x01\x02" + encoded.payload
    blob += b"\x00" * ((-len(blob)) % 4)
    staging = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy()).cuda()
    monkeypatch.setenv(LOAD_WORD_READS_ENV, "1")
    monkeypatch.setenv(LOAD_VECTOR_STORES_ENV, "1")
    monkeypatch.setenv(LOAD_UNCHECKED_BOUNDS_ENV, "1")
    monkeypatch.setenv(LOAD_BRANCHLESS_DECODE_ENV, "1")
    cache = _restore(staging, [2], [1], geometry, codebooks)
    assert cache[0].view(torch.uint8).cpu().numpy().tobytes() == raw.tobytes()


def test_vector_stores_build_and_restore_past_the_int32_bit_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache above 2**31 bits builds and restores its last block exactly."""
    geometry = CompressedStoreGeometry(
        num_slots=1,
        slot_size=2 * 2 * 16 * 8 * 128 * 2,
        block_tokens=16,
        num_layers=2,
        num_kv_heads=8,
        head_dim=128,
        tile_scalars=256,
    )
    # 2100 blocks x 128 KiB is about 275 MB, past 2**31 bits (256 MiB).
    num_blocks = 2100
    assert num_blocks * geometry.slot_size * 8 > 2**31
    codebooks = default_online_codebooks(geometry)
    raw = _slot(geometry, 81)
    encoded = encode_slot(raw, slot_id=0, geometry=geometry, codebooks=codebooks)
    assert encoded.mode is SlotMode.COMPRESSED
    staging = torch.from_numpy(
        np.frombuffer(encoded.payload, dtype=np.uint8).copy()
    ).cuda()
    monkeypatch.setenv(LOAD_WORD_READS_ENV, "1")
    monkeypatch.setenv(LOAD_VECTOR_STORES_ENV, "1")
    cache = _cache(num_blocks, geometry)
    decoder = FusedCompressedKVDecoder(
        kv_cache=cache,
        codebooks=codebooks,
        tile_scalars=geometry.tile_scalars,
        ring_depth=1,
        max_slots_per_buffer=1,
    )
    stream = torch.cuda.Stream()
    last = num_blocks - 1
    decoder.decode(
        staging=staging,
        staging_offsets=[0],
        block_ids=[last],
        modes=[1],
        buffer_index=0,
        stream=stream,
    )
    stream.synchronize()
    assert cache[last].view(torch.uint8).cpu().numpy().tobytes() == raw
