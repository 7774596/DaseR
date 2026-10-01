# SPDX-License-Identifier: Apache-2.0

"""Layer-major staging plans for packed compressed records."""

import numpy as np
import pytest

from daser.compression import (
    CompressedStoreGeometry,
    SlotMode,
    default_online_codebooks,
    encode_slot,
)
from daser.compression.format import IO_ALIGNMENT
from daser.compression.layer_major import (
    plan_layer_group,
    plan_layer_groups,
    rebased_headers,
    stage_layer_group,
    staged_descriptor_patch,
)


def _geometry() -> CompressedStoreGeometry:
    return CompressedStoreGeometry(
        num_slots=2,
        slot_size=3 * 2 * 128 * 2 * 64 * 2,
        block_tokens=128,
        num_layers=3,
        num_kv_heads=2,
        head_dim=64,
        tile_scalars=1024,
    )


def _records(geometry: CompressedStoreGeometry) -> list[bytes]:
    codebooks = default_online_codebooks(geometry)
    records = []
    for seed in range(2):
        rng = np.random.default_rng(seed)
        raw = np.empty(geometry.slot_size, dtype=np.uint8)
        raw[0::2] = rng.integers(0, 256, geometry.slot_size // 2, dtype=np.uint8)
        raw[1::2] = np.array([0x3E, 0x3F, 0x40, 0xBF], dtype=np.uint8)[
            rng.integers(0, 4, geometry.slot_size // 2, dtype=np.uint8)
        ]
        encoded = encode_slot(
            raw.tobytes(), slot_id=seed, geometry=geometry, codebooks=codebooks
        )
        assert encoded.mode is SlotMode.COMPRESSED
        records.append(encoded.payload)
    return records


def _headers(records: list[bytes]) -> np.ndarray:
    return np.stack(
        [np.frombuffer(r, dtype=np.uint8, count=IO_ALIGNMENT) for r in records]
    )


def test_full_range_covers_every_plane_byte() -> None:
    """The full plane range spans each record from its first plane on."""
    records = _records(_geometry())
    plan = plan_layer_group(_headers(records), (0, 6))
    assert (plan.source_begin == IO_ALIGNMENT).all()
    assert (plan.source_end <= [len(r) for r in records]).all()
    assert (plan.staging_offsets % IO_ALIGNMENT == 0).all()
    assert (plan.data_offsets % 16 == plan.source_begin % 16).all()


def test_groups_partition_records_and_stay_aligned() -> None:
    """Adjacent groups tile each record's plane bytes without gaps."""
    records = _records(_geometry())
    headers = _headers(records)
    groups = [plan_layer_group(headers, (first, 2)) for first in (0, 2, 4)]
    for left, right in zip(groups, groups[1:], strict=False):
        assert (left.source_end == right.source_begin).all()
    for plan in groups:
        assert (plan.data_offsets % 16 == plan.source_begin % 16).all()
        assert plan.staging_bytes % IO_ALIGNMENT == 0
        assert plan.staging_bytes < sum(len(r) for r in records)


def test_staged_group_reads_the_same_plane_bytes() -> None:
    """Rebased descriptors address the staged copy of the same plane bytes."""
    records = _records(_geometry())
    headers = _headers(records)
    plan = plan_layer_group(headers, (2, 2))
    image = stage_layer_group(records, plan)
    shifted = rebased_headers(headers, plan)
    assert np.array_equal(headers[:, :120], shifted[:, :120])
    original = headers[:, 120 : 120 + 6 * 48].view(np.uint32).reshape(2, -1, 12)
    moved = shifted[:, 120 : 120 + 6 * 48].view(np.uint32).reshape(2, -1, 12)
    assert np.array_equal(original[:, :2], moved[:, :2])
    assert np.array_equal(original[:, 4:], moved[:, 4:])
    for index, record in enumerate(records):
        page = int(plan.staging_offsets[index])
        assert bytes(image[page : page + IO_ALIGNMENT]) == shifted[index].tobytes()
        for plane in (2, 3):
            start, length = (int(w) for w in original[index, plane, 3:5])
            staged = int(moved[index, plane, 3])
            assert (
                bytes(image[page + staged : page + staged + length])
                == record[start : start + length]
            )


def test_shared_layout_keeps_header_pages_fixed() -> None:
    """Groups planned together share slot offsets sized for the largest group."""
    records = _records(_geometry())
    headers = _headers(records)
    ranges = [(0, 2), (2, 2), (4, 2)]
    shared = plan_layer_groups(headers, ranges)
    alone = [plan_layer_group(headers, group) for group in ranges]
    for plan, single in zip(shared, alone, strict=True):
        assert np.array_equal(plan.staging_offsets, shared[0].staging_offsets)
        assert plan.staging_bytes == shared[0].staging_bytes
        assert plan.staging_bytes >= single.staging_bytes
        assert np.array_equal(plan.source_begin, single.source_begin)
        assert (plan.data_offsets % 16 == plan.source_begin % 16).all()
    with pytest.raises(ValueError):
        plan_layer_groups(headers, [])


def test_descriptor_patch_turns_staged_headers_into_rebased_ones() -> None:
    """Patching staged original pages yields the staged group image's pages."""
    records = _records(_geometry())
    headers = _headers(records)
    for plan in plan_layer_groups(headers, [(0, 2), (2, 2), (4, 2)]):
        staging = np.zeros(plan.staging_bytes, dtype=np.uint8)
        for index, page in enumerate(plan.staging_offsets):
            staging[page : page + IO_ALIGNMENT] = headers[index]
        indices, values = staged_descriptor_patch(headers, plan)
        assert values.dtype == np.int32
        staging.view(np.int32)[indices] = values
        image = np.frombuffer(stage_layer_group(records, plan), dtype=np.uint8)
        for page in plan.staging_offsets:
            assert np.array_equal(
                staging[page : page + IO_ALIGNMENT],
                image[page : page + IO_ALIGNMENT],
            )


def test_rejects_raw_records_and_bad_ranges() -> None:
    """Non-packed pages and out-of-table ranges are refused."""
    records = _records(_geometry())
    headers = _headers(records)
    with pytest.raises(ValueError):
        plan_layer_group(np.zeros_like(headers), (0, 2))
    with pytest.raises(ValueError):
        plan_layer_group(headers, (-1, 2))
    with pytest.raises(ValueError):
        plan_layer_group(headers, (0, 0))
    with pytest.raises(ValueError):
        plan_layer_group(headers, (0, 100))
    with pytest.raises(ValueError):
        plan_layer_group(headers[:, :100], (0, 2))
