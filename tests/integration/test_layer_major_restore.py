# SPDX-License-Identifier: Apache-2.0

"""Layer-major staged plane groups restore byte-exact through the GPU decoder."""

import numpy as np
import pytest
import torch

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
    stage_layer_group,
)
from daser.ops.compressed_kv import FusedCompressedKVDecoder
from daser.ops.kv_layout import allocate_cross_layer_kv_cache

_LAYERS = 4


def _geometry(tile_scalars: int) -> CompressedStoreGeometry:
    return CompressedStoreGeometry(
        num_slots=3,
        slot_size=_LAYERS * 2 * 128 * 4 * 128 * 2,
        block_tokens=128,
        num_layers=_LAYERS,
        num_kv_heads=4,
        head_dim=128,
        tile_scalars=tile_scalars,
    )


def _raw_slot(geometry: CompressedStoreGeometry, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = rng.integers(0, 256, geometry.slot_size // 2, dtype=np.uint8)
    common = np.array([0x3E, 0x3F, 0x40, 0xBF], dtype=np.uint8)
    raw[1::2] = common[
        rng.integers(0, len(common), geometry.slot_size // 2, dtype=np.uint8)
    ]
    raw[1::2][::997] = 0x7E
    return raw.tobytes()


def _kv_cache(num_blocks: int, geometry: CompressedStoreGeometry) -> torch.Tensor:
    kv_cache = allocate_cross_layer_kv_cache(
        num_blocks,
        geometry.num_layers,
        geometry.block_tokens,
        geometry.num_kv_heads,
        geometry.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    kv_cache.zero_()
    return kv_cache


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tile_scalars", [256, 1024])
@pytest.mark.parametrize("group_layers", [1, 2])
def test_layer_groups_restore_like_whole_records(
    tile_scalars: int, group_layers: int
) -> None:
    """Restoring every layer group from layer-major staging equals one full
    record-major restore, and a group touches only its own layers."""
    geometry = _geometry(tile_scalars)
    codebooks = default_online_codebooks(geometry)
    encoded = [
        encode_slot(
            _raw_slot(geometry, seed),
            slot_id=seed,
            geometry=geometry,
            codebooks=codebooks,
        )
        for seed in range(3)
    ]
    assert all(item.mode is SlotMode.COMPRESSED for item in encoded)
    payloads = [item.payload for item in encoded]
    blocks = [2, 0, 1]
    stream = torch.cuda.Stream()
    with torch.inference_mode():
        expected = _kv_cache(3, geometry)
        actual = _kv_cache(3, geometry)
        reference = FusedCompressedKVDecoder(
            kv_cache=expected,
            codebooks=codebooks,
            tile_scalars=tile_scalars,
            ring_depth=1,
            max_slots_per_buffer=3,
        )
        decoder = FusedCompressedKVDecoder(
            kv_cache=actual,
            codebooks=codebooks,
            tile_scalars=tile_scalars,
            ring_depth=1,
            max_slots_per_buffer=3,
        )
    whole = torch.from_numpy(np.frombuffer(b"".join(payloads), dtype=np.uint8).copy())
    offsets = np.cumsum([0] + [len(p) for p in payloads[:-1]]).tolist()
    plan = reference.prepare(
        staging=whole.cuda(),
        staging_offsets=offsets,
        block_ids=blocks,
        modes=[1, 1, 1],
        buffer_index=0,
        stream=stream,
    )
    assert plan is not None
    plan.submit()
    stream.synchronize()

    headers = np.stack(
        [np.frombuffer(p, dtype=np.uint8, count=IO_ALIGNMENT) for p in payloads]
    )
    for first_layer in range(0, _LAYERS, group_layers):
        group = plan_layer_group(headers, (2 * first_layer, 2 * group_layers))
        assert group.staging_bytes < sum(len(p) for p in payloads)
        image = stage_layer_group(payloads, group)
        staging = torch.from_numpy(np.frombuffer(image, dtype=np.uint8).copy()).cuda()
        restore = decoder.prepare(
            staging=staging,
            staging_offsets=group.staging_offsets.tolist(),
            block_ids=blocks,
            modes=[1, 1, 1],
            buffer_index=0,
            stream=stream,
            plane_range=group.plane_range,
        )
        assert restore is not None
        restore.submit()
        stream.synchronize()
        done = first_layer + group_layers
        assert torch.equal(actual[:, :done], expected[:, :done])
        assert not actual[:, done:].any()


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tile_scalars", [256, 1024])
def test_one_prepare_restores_groups_from_separate_staging(tile_scalars: int) -> None:
    """One prepared plan decodes each group from its own staging buffer."""
    geometry = _geometry(tile_scalars)
    codebooks = default_online_codebooks(geometry)
    payloads = [
        encode_slot(
            _raw_slot(geometry, seed),
            slot_id=seed,
            geometry=geometry,
            codebooks=codebooks,
        ).payload
        for seed in range(3)
    ]
    blocks = [1, 2, 0]
    stream = torch.cuda.Stream()
    with torch.inference_mode():
        expected = _kv_cache(3, geometry)
        actual = _kv_cache(3, geometry)
        decoders = [
            FusedCompressedKVDecoder(
                kv_cache=cache,
                codebooks=codebooks,
                tile_scalars=tile_scalars,
                ring_depth=1,
                max_slots_per_buffer=3,
            )
            for cache in (expected, actual)
        ]
    whole = torch.from_numpy(
        np.frombuffer(b"".join(payloads), dtype=np.uint8).copy()
    ).cuda()
    offsets = np.cumsum([0] + [len(p) for p in payloads[:-1]]).tolist()
    reference = decoders[0].prepare(
        staging=whole,
        staging_offsets=offsets,
        block_ids=blocks,
        modes=[1, 1, 1],
        buffer_index=0,
        stream=stream,
    )
    assert reference is not None
    reference.submit()

    headers = np.stack(
        [np.frombuffer(p, dtype=np.uint8, count=IO_ALIGNMENT) for p in payloads]
    )
    groups = plan_layer_groups(headers, [(0, 2), (2, 4), (6, 2)])
    stagings = [
        torch.from_numpy(
            np.frombuffer(stage_layer_group(payloads, group), dtype=np.uint8).copy()
        ).cuda()
        for group in groups
    ]
    plan = decoders[1].prepare(
        staging=stagings[0],
        staging_offsets=groups[0].staging_offsets.tolist(),
        block_ids=blocks,
        modes=[1, 1, 1],
        buffer_index=0,
        stream=stream,
    )
    assert plan is not None
    with pytest.raises(ValueError):
        plan.submit_planes(2, 4, stagings[1])
    for group, staging in zip(groups, stagings, strict=True):
        assert plan.submit_planes(*group.plane_range, staging) == 3
    assert plan.submit() == 0
    with pytest.raises(ValueError):
        plan.submit_planes(0, 2, stagings[0])
    stream.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_partial_plane_range_rejects_raw_records() -> None:
    """Raw fallback records cannot be restored one plane group at a time."""
    geometry = _geometry(1024)
    codebooks = default_online_codebooks(geometry)
    with torch.inference_mode():
        decoder = FusedCompressedKVDecoder(
            kv_cache=_kv_cache(1, geometry),
            codebooks=codebooks,
            tile_scalars=1024,
            ring_depth=1,
            max_slots_per_buffer=1,
        )
    staging = torch.zeros(geometry.slot_size, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError):
        decoder.prepare(
            staging=staging,
            staging_offsets=[0],
            block_ids=[0],
            modes=[0],
            buffer_index=0,
            stream=torch.cuda.Stream(),
            plane_range=(0, 2),
        )
