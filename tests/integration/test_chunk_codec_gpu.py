# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from daser.compression.format import IO_ALIGNMENT
from daser.ops.chunk_codec import ChunkCapacityError, KVMajorChunkCodec

_LAYERS, _TOKENS, _HEADS, _HEAD_DIM = 4, 64, 4, 128


def _codec(max_chunks: int, capacity_ratio: float = 0.75) -> KVMajorChunkCodec:
    return KVMajorChunkCodec(
        num_layers=_LAYERS,
        chunk_tokens=_TOKENS,
        num_kv_heads=_HEADS,
        head_dim=_HEAD_DIM,
        device=torch.device("cuda"),
        capacity_ratio=capacity_ratio,
        max_chunks=max_chunks,
    )


def _chunk(codec: KVMajorChunkCodec, seed: int, *, compressible: bool) -> torch.Tensor:
    """Return one contiguous CUDA uint8 kv-major chunk buffer."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, codec.raw_bytes, dtype=np.uint8)
    if compressible:
        common = np.array([0x3E, 0x3F, 0x40, 0xBF], dtype=np.uint8)
        high = common[rng.integers(0, len(common), codec.raw_bytes // 2)]
        high[::997] = 0x7E
        raw[1::2] = high
    return torch.from_numpy(raw).cuda()


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch", [1, 3])
def test_chunk_codec_round_trips_through_capacity_records(batch: int) -> None:
    codec = _codec(max_chunks=3)
    stream = torch.cuda.Stream()
    originals = [_chunk(codec, seed, compressible=True) for seed in range(batch)]
    chunks = [chunk.clone() for chunk in originals]
    lengths = codec.encode(chunks, stream)
    assert all(0 < length <= codec.capacity_bytes for length in lengths)
    # Engines move only the capacity prefix; the rest of the buffer is reused.
    records = [chunk[: codec.capacity_bytes].clone() for chunk in chunks]
    restored = [torch.full_like(chunk, 0xA5) for chunk in originals]
    for target, record in zip(restored, records, strict=True):
        target[: codec.capacity_bytes].copy_(record)
    codec.decode(restored, stream)
    stream.synchronize()
    for target, original in zip(restored, originals, strict=True):
        assert torch.equal(target, original)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_codec_decode_reuses_rings_across_batches() -> None:
    codec = _codec(max_chunks=2)
    stream = torch.cuda.Stream()
    originals = [_chunk(codec, seed, compressible=True) for seed in range(6)]
    encoded = [chunk.clone() for chunk in originals]
    for start in range(0, 6, 2):
        codec.encode(encoded[start : start + 2], stream)
    for start in range(0, 6, 2):
        codec.decode(encoded[start : start + 2], stream)
    stream.synchronize()
    for target, original in zip(encoded, originals, strict=True):
        assert torch.equal(target, original)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_codec_fails_closed_when_record_exceeds_capacity() -> None:
    codec = _codec(max_chunks=2)
    stream = torch.cuda.Stream()
    chunks = [
        _chunk(codec, 1, compressible=True),
        _chunk(codec, 2, compressible=False),
    ]
    with pytest.raises(ChunkCapacityError):
        codec.encode(chunks, stream)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_codec_rejects_mismatched_buffers() -> None:
    codec = _codec(max_chunks=1)
    stream = torch.cuda.Stream()
    good = _chunk(codec, 1, compressible=True)
    with pytest.raises(ValueError):
        codec.encode([good, good.clone()], stream)
    with pytest.raises(ValueError):
        codec.encode([good[:-IO_ALIGNMENT]], stream)
