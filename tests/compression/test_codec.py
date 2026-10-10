# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

from daser.compression import (
    CompressedStoreGeometry,
    SlotMode,
    decode_slot,
    default_online_codebooks,
    encode_slot,
)
from daser.compression.format import (
    CODEBOOK_ENTRIES,
    FORMAT_VERSION,
    GROUP_PREFIX_FORMAT_VERSION,
    IO_ALIGNMENT,
    ONLINE_TILE_SCALARS,
    PREFIX_GROUP_TILES,
    SlotHeader,
    digest_bytes,
    max_slot_layers,
    prefix_table_bytes,
)


def _geometry(num_slots: int = 2) -> CompressedStoreGeometry:
    return CompressedStoreGeometry(
        num_slots=num_slots,
        slot_size=2 * 2 * 128 * 4 * 127 * 2,
        block_tokens=128,
        num_layers=2,
        num_kv_heads=4,
        head_dim=127,
        tile_scalars=1024,
    )


def _slot(
    geometry: CompressedStoreGeometry,
    *,
    seed: int,
    escaped_high_byte: int | None = None,
) -> bytes:
    rng = np.random.default_rng(seed)
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = rng.integers(0, 256, geometry.slot_size // 2, dtype=np.uint8)
    common = np.array([0x3E, 0x3F, 0x40, 0xBF], dtype=np.uint8)
    raw[1::2] = common[
        rng.integers(0, len(common), geometry.slot_size // 2, dtype=np.uint8)
    ]
    if escaped_high_byte is not None:
        raw[1::2][::997] = escaped_high_byte
    return raw.tobytes()


def test_reference_codec_is_byte_exact_with_escapes_and_tail_tile() -> None:
    geometry = _geometry()
    evaluation = _slot(geometry, seed=2, escaped_high_byte=0x7E)
    codebooks = default_online_codebooks(geometry)

    encoded = encode_slot(
        evaluation,
        slot_id=1,
        geometry=geometry,
        codebooks=codebooks,
    )

    assert encoded.mode is SlotMode.COMPRESSED
    assert len(encoded.payload) < geometry.slot_size
    assert len(encoded.payload) % IO_ALIGNMENT == 0
    header = SlotHeader.parse(
        encoded.payload,
        expected_geometry=geometry,
        expected_slot_id=1,
        expected_codebook_hash=digest_bytes(codebooks),
    )
    assert header.descriptors[0].scalar_count % geometry.tile_scalars == 512
    # Only the complete slot is an O_DIRECT transfer. Plane records are
    # intentionally packed back-to-back so their internal padding is not
    # counted as stored KV bytes.
    assert any(
        descriptor.record_offset % IO_ALIGNMENT for descriptor in header.descriptors[1:]
    )
    assert any(
        descriptor.record_length % IO_ALIGNMENT for descriptor in header.descriptors
    )
    assert sum(item.escape_count for item in header.descriptors) > 0
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=1,
            geometry=geometry,
            codebooks=codebooks,
        )
        == evaluation
    )


def test_three_bit_codec_treats_unrepresentable_symbols_as_escapes() -> None:
    """Three-bit packing must preserve codebook entries 7 through 14."""
    geometry = _geometry(num_slots=1)
    codebooks = default_online_codebooks(geometry)
    table = np.frombuffer(codebooks, dtype=np.uint8)[:CODEBOOK_ENTRIES]
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = 17
    high = np.full(geometry.slot_size // 2, table[0], dtype=np.uint8)
    high[::997] = table[10]
    raw[1::2] = high

    encoded = encode_slot(
        raw.tobytes(), slot_id=0, geometry=geometry, codebooks=codebooks
    )

    assert encoded.mode is SlotMode.COMPRESSED
    header = SlotHeader.parse(
        encoded.payload,
        expected_geometry=geometry,
        expected_slot_id=0,
        expected_codebook_hash=digest_bytes(codebooks),
    )
    assert sum(item.escape_count for item in header.descriptors) > 0
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )
        == raw.tobytes()
    )


def test_three_bit_escape_stream_round_trips_the_eighth_secondary_entry() -> None:
    """The narrow escape stream promotes codebook entry fourteen to raw bytes."""
    geometry = _geometry(num_slots=1)
    codebooks = default_online_codebooks(geometry)
    table = np.frombuffer(codebooks, dtype=np.uint8)[:CODEBOOK_ENTRIES]
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = 23
    high = np.full(geometry.slot_size // 2, table[0], dtype=np.uint8)
    high[::997] = table[14]
    raw[1::2] = high

    encoded = encode_slot(
        raw.tobytes(), slot_id=0, geometry=geometry, codebooks=codebooks
    )

    header = SlotHeader.parse(
        encoded.payload,
        expected_geometry=geometry,
        expected_slot_id=0,
        expected_codebook_hash=digest_bytes(codebooks),
    )
    assert sum(item.escape_count for item in header.descriptors) > 0
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )
        == raw.tobytes()
    )


def test_online_codebook_is_generic_for_all_geometries() -> None:
    """The startup default is model-independent and plane-major."""
    geometry = CompressedStoreGeometry(
        num_slots=1,
        slot_size=36 * 2 * 128 * 8 * 128 * 2,
        block_tokens=128,
        num_layers=36,
        num_kv_heads=8,
        head_dim=128,
    )

    tables = np.frombuffer(default_online_codebooks(geometry), dtype=np.uint8).reshape(
        geometry.plane_count, CODEBOOK_ENTRIES
    )

    assert tables.shape == (72, CODEBOOK_ENTRIES)
    assert all(len(np.unique(row)) == CODEBOOK_ENTRIES for row in tables)
    assert np.all(tables == tables[0])
    assert tables[0].tolist() == [
        63,
        191,
        62,
        190,
        64,
        192,
        61,
        189,
        60,
        188,
        59,
        187,
        58,
        186,
        57,
    ]


def test_incompressible_slot_uses_explicit_raw_mode() -> None:
    geometry = _geometry(num_slots=1)
    rng = np.random.default_rng(4)
    raw = rng.integers(0, 256, geometry.slot_size, dtype=np.uint8).tobytes()
    codebooks = default_online_codebooks(geometry)

    encoded = encode_slot(raw, slot_id=0, geometry=geometry, codebooks=codebooks)

    assert encoded.mode is SlotMode.RAW
    assert encoded.payload == raw
    assert (
        decode_slot(
            encoded.payload,
            mode=SlotMode.RAW,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )
        == raw
    )


def test_slot_header_rejects_corrupt_descriptor() -> None:
    geometry = _geometry(num_slots=1)
    slot = _slot(geometry, seed=30)
    codebooks = default_online_codebooks(geometry)
    encoded = encode_slot(slot, slot_id=0, geometry=geometry, codebooks=codebooks)
    assert encoded.mode is SlotMode.COMPRESSED
    payload = bytearray(encoded.payload)
    payload[128:136] = b"\xff" * 8

    with pytest.raises(ValueError):
        decode_slot(
            payload,
            mode=SlotMode.COMPRESSED,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )


def test_grouped_prefix_version_round_trips_and_shrinks_records() -> None:
    """Version 2 stores one prefix per eight tiles and stays byte exact."""
    geometry = replace(_geometry(), tile_scalars=ONLINE_TILE_SCALARS)
    codebooks = default_online_codebooks(geometry)
    raw = _slot(geometry, seed=41, escaped_high_byte=0x7E)
    tiles = (geometry.plane_scalars + ONLINE_TILE_SCALARS - 1) // ONLINE_TILE_SCALARS
    assert tiles % PREFIX_GROUP_TILES

    v1 = encode_slot(raw, slot_id=0, geometry=geometry, codebooks=codebooks)
    v2 = encode_slot(
        raw,
        slot_id=0,
        geometry=geometry,
        codebooks=codebooks,
        prefix_group_tiles=PREFIX_GROUP_TILES,
    )

    assert v2.mode is SlotMode.COMPRESSED
    first = SlotHeader.parse(v1.payload, expected_geometry=geometry)
    second = SlotHeader.parse(v2.payload, expected_geometry=geometry)
    assert first.format_version == FORMAT_VERSION
    assert second.format_version == GROUP_PREFIX_FORMAT_VERSION
    saved = prefix_table_bytes(tiles, 1) - prefix_table_bytes(tiles, PREFIX_GROUP_TILES)
    assert saved > 0
    assert all(
        a.record_length - b.record_length == saved
        for a, b in zip(first.descriptors, second.descriptors, strict=True)
    )
    for encoded in (v1, v2):
        assert (
            decode_slot(
                encoded.payload,
                mode=encoded.mode,
                slot_id=0,
                geometry=geometry,
                codebooks=codebooks,
            )
            == raw
        )


def test_grouped_prefix_tables_hold_group_boundary_ranks() -> None:
    """Each version-2 entry equals the version-1 prefix at its group start."""
    geometry = replace(_geometry(num_slots=1), tile_scalars=ONLINE_TILE_SCALARS)
    codebooks = default_online_codebooks(geometry)
    raw = _slot(geometry, seed=43, escaped_high_byte=0x7E)
    tiles = (geometry.plane_scalars + ONLINE_TILE_SCALARS - 1) // ONLINE_TILE_SCALARS
    groups = (tiles + PREFIX_GROUP_TILES - 1) // PREFIX_GROUP_TILES
    v1 = encode_slot(raw, slot_id=0, geometry=geometry, codebooks=codebooks)
    v2 = encode_slot(
        raw,
        slot_id=0,
        geometry=geometry,
        codebooks=codebooks,
        prefix_group_tiles=PREFIX_GROUP_TILES,
    )
    for a, b in zip(
        SlotHeader.parse(v1.payload).descriptors,
        SlotHeader.parse(v2.payload).descriptors,
        strict=True,
    ):
        per_tile = np.frombuffer(
            v1.payload, dtype="<u4", count=2 * (tiles + 1), offset=a.prefix_offset
        ).reshape(2, tiles + 1)
        grouped = np.frombuffer(
            v2.payload, dtype="<u4", count=2 * (groups + 1), offset=b.prefix_offset
        ).reshape(2, groups + 1)
        boundaries = np.minimum(np.arange(groups + 1) * PREFIX_GROUP_TILES, tiles)
        assert (grouped == per_tile[:, boundaries]).all()


def test_grouped_prefix_version_requires_online_tiles() -> None:
    geometry = _geometry(num_slots=1)
    codebooks = default_online_codebooks(geometry)
    raw = _slot(geometry, seed=45)

    with pytest.raises(ValueError, match="256-scalar"):
        encode_slot(
            raw,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
            prefix_group_tiles=PREFIX_GROUP_TILES,
        )
    with pytest.raises(ValueError, match="granularity"):
        encode_slot(
            raw, slot_id=0, geometry=geometry, codebooks=codebooks, prefix_group_tiles=4
        )
    payload = bytearray(
        encode_slot(raw, slot_id=0, geometry=geometry, codebooks=codebooks).payload
    )
    payload[8] = GROUP_PREFIX_FORMAT_VERSION
    with pytest.raises(ValueError, match="256-scalar"):
        SlotHeader.parse(payload, expected_geometry=geometry)
    payload[8] = 3
    with pytest.raises(ValueError, match="version"):
        SlotHeader.parse(payload, expected_geometry=geometry)


def _layer_geometry(num_layers: int) -> CompressedStoreGeometry:
    return CompressedStoreGeometry(
        num_slots=1,
        slot_size=num_layers * 2 * 16 * 8 * 128 * 2,
        block_tokens=16,
        num_layers=num_layers,
        num_kv_heads=8,
        head_dim=128,
        tile_scalars=ONLINE_TILE_SCALARS,
    )


def test_geometry_rejects_layers_beyond_the_header_page() -> None:
    # A 48-layer MoE model needs 96 plane descriptors; one page holds 82.
    assert max_slot_layers() == 41
    geometry = _layer_geometry(max_slot_layers())
    slot = _slot(geometry, seed=7)
    encoded = encode_slot(
        slot,
        slot_id=0,
        geometry=geometry,
        codebooks=default_online_codebooks(geometry),
    )
    assert encoded.mode is SlotMode.COMPRESSED
    with pytest.raises(ValueError, match="at most 41 layers"):
        _layer_geometry(max_slot_layers() + 1)
