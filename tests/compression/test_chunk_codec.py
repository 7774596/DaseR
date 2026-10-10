# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from daser.compression import (
    CompressedStoreGeometry,
    SlotMode,
    decode_slot,
    default_online_codebooks,
    encode_slot,
)
from daser.compression.format import IO_ALIGNMENT, ONLINE_TILE_SCALARS
from daser.ops.chunk_codec import (
    KVMajorChunkCodec,
    chunk_capacity_bytes,
    kv_major_to_physical,
    physical_to_kv_major,
)
from daser.ops.kv_layout import allocate_cross_layer_kv_cache, physical_kv_view

_LAYERS, _TOKENS, _HEADS, _HEAD_DIM = 2, 16, 4, 64


def _chunk(seed: int) -> torch.Tensor:
    """Return a compressible kv-major BF16 chunk ``[2, L, T, H, D]``."""
    rng = np.random.default_rng(seed)
    scalars = 2 * _LAYERS * _TOKENS * _HEADS * _HEAD_DIM
    raw = np.empty(scalars * 2, dtype=np.uint8)
    raw[0::2] = rng.integers(0, 256, scalars, dtype=np.uint8)
    common = np.array([0x3E, 0x3F, 0x40, 0xBF], dtype=np.uint8)
    raw[1::2] = common[rng.integers(0, len(common), scalars, dtype=np.uint8)]
    return (
        torch.from_numpy(raw)
        .view(torch.bfloat16)
        .view(2, _LAYERS, _TOKENS, _HEADS, _HEAD_DIM)
    )


def test_capacity_class_is_aligned_and_bounded_by_raw() -> None:
    raw = 18 * 1024 * 1024
    capacity = chunk_capacity_bytes(raw, 0.75)
    assert capacity % IO_ALIGNMENT == 0
    assert raw * 0.75 <= capacity < raw * 0.75 + IO_ALIGNMENT
    assert chunk_capacity_bytes(IO_ALIGNMENT, 0.01) == IO_ALIGNMENT
    assert chunk_capacity_bytes(raw, 1.0) == raw


@pytest.mark.parametrize(
    ("raw", "ratio"), [(0, 0.75), (IO_ALIGNMENT + 2, 0.75), (IO_ALIGNMENT, 0.0)]
)
def test_capacity_class_rejects_invalid_inputs(raw: int, ratio: float) -> None:
    with pytest.raises(ValueError):
        chunk_capacity_bytes(raw, ratio)


def test_layout_mapping_round_trips_through_physical_blnhc() -> None:
    chunk = _chunk(1)
    cache = allocate_cross_layer_kv_cache(
        2, _LAYERS, _TOKENS, _HEADS, _HEAD_DIM, dtype=torch.bfloat16, device="cpu"
    )
    physical = physical_kv_view(cache)
    physical[1].copy_(kv_major_to_physical(chunk))
    # Each DaseR codec plane (layer, kv) must hold the engine's chunk[kv, layer].
    for layer in range(_LAYERS):
        for kv in range(2):
            assert torch.equal(cache[1, layer, kv], chunk[kv, layer])
    assert torch.equal(physical_to_kv_major(physical[1]), chunk)


def test_layout_mapping_preserves_bytes_through_reference_codec() -> None:
    geometry = CompressedStoreGeometry(
        num_slots=1,
        slot_size=_LAYERS * 2 * _TOKENS * _HEADS * _HEAD_DIM * 2,
        block_tokens=_TOKENS,
        num_layers=_LAYERS,
        num_kv_heads=_HEADS,
        head_dim=_HEAD_DIM,
        tile_scalars=ONLINE_TILE_SCALARS,
    )
    codebooks = default_online_codebooks(geometry)
    chunk = _chunk(2)
    cache = allocate_cross_layer_kv_cache(
        1, _LAYERS, _TOKENS, _HEADS, _HEAD_DIM, dtype=torch.bfloat16, device="cpu"
    )
    physical_kv_view(cache)[0].copy_(kv_major_to_physical(chunk))
    slot_bytes = cache[0].contiguous().view(torch.uint8).numpy().tobytes()
    encoded = encode_slot(slot_bytes, slot_id=0, geometry=geometry, codebooks=codebooks)
    assert encoded.mode is SlotMode.COMPRESSED
    decoded = decode_slot(
        encoded.payload,
        mode=encoded.mode,
        slot_id=0,
        geometry=geometry,
        codebooks=codebooks,
    )
    restored = allocate_cross_layer_kv_cache(
        1, _LAYERS, _TOKENS, _HEADS, _HEAD_DIM, dtype=torch.bfloat16, device="cpu"
    )
    restored[0].copy_(
        torch.frombuffer(bytearray(decoded), dtype=torch.bfloat16).view(
            restored[0].shape
        )
    )
    assert torch.equal(physical_to_kv_major(physical_kv_view(restored)[0]), chunk)


def test_layout_helpers_reject_wrong_kv_axis() -> None:
    with pytest.raises(ValueError):
        kv_major_to_physical(torch.empty(3, 1, 1, 1, 1))
    with pytest.raises(ValueError):
        physical_to_kv_major(torch.empty(1, 1, 1, 3, 1))


def test_codec_requires_cuda_device() -> None:
    with pytest.raises(ValueError):
        KVMajorChunkCodec(
            num_layers=_LAYERS,
            chunk_tokens=_TOKENS,
            num_kv_heads=_HEADS,
            head_dim=_HEAD_DIM,
            device=torch.device("cpu"),
            capacity_ratio=0.75,
            max_chunks=1,
        )
