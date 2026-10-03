# SPDX-License-Identifier: Apache-2.0

"""In-forward layer-wise restore matches a record-major restore byte for byte."""

from typing import Any

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
from daser.config import LAYERWISE_SLOTS_ENV
from daser.connector.metadata import CompressedLoadSlot, ReqLoadSpec
from daser.connector.worker import load as load_module
from daser.connector.worker.load import LoadPipeline
from daser.connector.worker.memory import FixedCudaStagingPool
from daser.connector.worker.staging import CROSS_LAYER_KV_CACHE_KEY
from daser.ops.compressed_kv import FusedCompressedKVDecoder
from daser.ops.kv_layout import allocate_cross_layer_kv_cache

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

# More layers than the deepest tested slot ring, so slots get refilled.
_LAYERS = 8
_TILE_SCALARS = 1024


def _geometry() -> CompressedStoreGeometry:
    return CompressedStoreGeometry(
        num_slots=3,
        slot_size=_LAYERS * 2 * 128 * 4 * 128 * 2,
        block_tokens=128,
        num_layers=_LAYERS,
        num_kv_heads=4,
        head_dim=128,
        tile_scalars=_TILE_SCALARS,
    )


def _payloads(geometry: CompressedStoreGeometry, codebooks: bytes) -> list[bytes]:
    payloads = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        raw = np.empty(geometry.slot_size, dtype=np.uint8)
        raw[0::2] = rng.integers(0, 256, geometry.slot_size // 2, dtype=np.uint8)
        common = np.array([0x3E, 0x3F, 0x40, 0xBF], dtype=np.uint8)
        raw[1::2] = common[rng.integers(0, 4, geometry.slot_size // 2, dtype=np.uint8)]
        raw[1::2][::997] = 0x7E
        encoded = encode_slot(
            raw.tobytes(), slot_id=seed, geometry=geometry, codebooks=codebooks
        )
        assert encoded.mode is SlotMode.COMPRESSED
        payloads.append(encoded.payload)
    return payloads


def _kv_cache(geometry: CompressedStoreGeometry) -> torch.Tensor:
    return allocate_cross_layer_kv_cache(
        3,
        geometry.num_layers,
        geometry.block_tokens,
        geometry.num_kv_heads,
        geometry.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).zero_()


class _FakeLoadClient:
    """Async IPC stub that serves leased runs from a device-resident store."""

    store: torch.Tensor
    pool: FixedCudaStagingPool
    fail_on_call: int = -1
    calls: list[int] = []
    released: list[str] = []

    def __init__(self, socket_path: str) -> None:
        del socket_path

    async def init_transfer(self) -> dict[str, Any]:
        """Accept transfer initialization."""
        return {}

    async def register_staging_cuda(self, **kwargs: Any) -> dict[str, Any]:
        """Accept staging registration."""
        del kwargs
        return {}

    async def transfer_load_runs(
        self,
        buffer_index: int,
        producer_pid: int,
        file_offsets: Any,
        target_offsets: Any,
        nbytes: Any,
        lease_id: str,
        *,
        copy_mode: str = "gather",
    ) -> dict[str, Any]:
        """Copy runs into the staging buffer like the server's L1 path."""
        del producer_pid, copy_mode
        assert lease_id == "req"
        calls = type(self).calls
        calls.append(buffer_index)
        if len(calls) == type(self).fail_on_call:
            raise RuntimeError("injected transfer failure")
        buffer = type(self).pool.buffer(buffer_index)
        store = type(self).store
        total = 0
        for src, dst, size in zip(file_offsets, target_offsets, nbytes, strict=True):
            src, dst, size = int(src), int(dst), int(size)
            assert 0 <= src and src + size <= store.numel()
            assert 0 <= dst and dst + size <= buffer.numel()
            buffer[dst : dst + size].copy_(store[src : src + size])
            total += size
        torch.cuda.synchronize()
        return {"bytes": total}

    async def release_transfer_lease(self, lease_id: str) -> dict[str, Any]:
        """Record the worker-side lease release."""
        type(self).released.append(lease_id)
        return {}

    async def close(self) -> None:
        """Close nothing."""


def _restore(
    monkeypatch: pytest.MonkeyPatch, fail_on_call: int = -1
) -> tuple[torch.Tensor, torch.Tensor, LoadPipeline, list[int]]:
    geometry = _geometry()
    codebooks = default_online_codebooks(geometry)
    payloads = _payloads(geometry, codebooks)
    padded = [
        p + bytes(-len(p) % IO_ALIGNMENT) for p in payloads
    ]  # records start 4 KiB aligned in the store
    offsets = np.cumsum([IO_ALIGNMENT] + [len(p) for p in padded[:-1]]).tolist()
    store = torch.zeros(offsets[-1] + len(padded[-1]), dtype=torch.uint8)
    for offset, payload in zip(offsets, padded, strict=True):
        store[offset : offset + len(payload)] = torch.frombuffer(
            bytearray(payload), dtype=torch.uint8
        )
    blocks = [2, 0, 1]

    expected = _kv_cache(geometry)
    reference = FusedCompressedKVDecoder(
        kv_cache=expected,
        codebooks=codebooks,
        tile_scalars=_TILE_SCALARS,
        ring_depth=1,
        max_slots_per_buffer=3,
    )
    stream = torch.cuda.Stream()
    whole = torch.from_numpy(np.frombuffer(b"".join(padded), dtype=np.uint8).copy())
    plan = reference.prepare(
        staging=whole.cuda(),
        staging_offsets=[offset - IO_ALIGNMENT for offset in offsets],
        block_ids=blocks,
        modes=[1, 1, 1],
        buffer_index=0,
        stream=stream,
    )
    assert plan is not None
    plan.submit()
    stream.synchronize()

    pool = FixedCudaStagingPool(torch.device("cuda"), 4 << 20, 2)
    _FakeLoadClient.store = store.cuda()
    _FakeLoadClient.pool = pool
    _FakeLoadClient.fail_on_call = fail_on_call
    _FakeLoadClient.calls = []
    _FakeLoadClient.released = []
    monkeypatch.setattr(load_module, "IPCClientAsync", _FakeLoadClient)

    actual = _kv_cache(geometry)
    pipeline = LoadPipeline("/unused/daser.sock", client_count=2)
    pipeline.configure(
        kv_caches={"layers": actual},
        layer_names=[f"layer.{index}" for index in range(_LAYERS)],
        local_slot_size=geometry.slot_size,
        rank_stride_bytes=geometry.slot_size,
        tp_rank=0,
        staging_pool=pool,
        load_key_scale=1.0,
        load_value_scale=1.0,
        rope_delta_scale=1.0,
        rope_base=10000.0,
        rope_rotary_dim=0,
        rope_is_neox_style=True,
    )
    pipeline.configure_compression(
        storage_format="compressed-online",
        codebooks=codebooks,
        tile_scalars=_TILE_SCALARS,
    )
    pipeline.initialize_transfer()
    spec = ReqLoadSpec(
        chunk_key="chunk",
        start_slot=0,
        num_slots=3,
        block_ids=blocks,
        file_offset=offsets[0],
        token_count=3 * geometry.block_tokens,
        lease_id="req",
        compressed_slots=[
            CompressedLoadSlot(index, "compressed", offset, len(payload))
            for index, (offset, payload) in enumerate(zip(offsets, padded, strict=True))
        ],
    )
    pipeline.start_layerwise({"req": spec})
    for layer in range(_LAYERS):
        pipeline.wait_for_layer(layer)
        # Attention of this layer runs on the current stream after the wait.
        torch.cuda.current_stream().synchronize()
        if fail_on_call < 0:
            assert torch.equal(actual[:, : layer + 1], expected[:, : layer + 1])
    return actual, expected, pipeline, blocks


@pytest.mark.parametrize("slots", ["1", "2", "64"])
def test_layerwise_restore_matches_record_major(
    monkeypatch: pytest.MonkeyPatch, slots: str
) -> None:
    """Every layer is restored exactly before its wait, at any slot depth."""
    monkeypatch.setenv(LAYERWISE_SLOTS_ENV, slots)
    actual, expected, pipeline, _ = _restore(monkeypatch)
    pipeline.start_layerwise({})
    try:
        assert torch.equal(actual, expected)
        assert pipeline.take_invalid_block_ids() == set()
        # One header read, then one transfer per layer on alternating lanes.
        assert _FakeLoadClient.calls[0] == 1
        lanes = [layer % 2 for layer in range(_LAYERS)]
        assert sorted(_FakeLoadClient.calls[1:]) == sorted(lanes)
        if slots == "1":
            assert _FakeLoadClient.calls == [1, *lanes]
        assert _FakeLoadClient.released == ["req"]
    finally:
        pipeline.shutdown()


def test_layerwise_failure_invalidates_blocks_and_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-forward transfer failure reports every destination block."""
    _, _, pipeline, blocks = _restore(monkeypatch, fail_on_call=3)
    try:
        assert pipeline.take_invalid_block_ids() == set(blocks)
        assert pipeline.take_invalid_block_ids() == set()
        pipeline.start_layerwise({})
        assert _FakeLoadClient.released == ["req"]
    finally:
        pipeline.shutdown()


@pytest.mark.parametrize("slots", ["1", "2"])
def test_raw_layerwise_restore_matches_slot_copy(
    monkeypatch: pytest.MonkeyPatch, slots: str
) -> None:
    """Raw slot-major records restore each layer exactly before its wait."""
    monkeypatch.setenv(LAYERWISE_SLOTS_ENV, slots)
    geometry = _geometry()
    generator = torch.Generator().manual_seed(0)
    raw = torch.randint(
        0, 256, (3 * geometry.slot_size,), dtype=torch.uint8, generator=generator
    )
    # Two raw chunks; the second starts at its own file offset.
    first_offset = IO_ALIGNMENT
    second_offset = first_offset + 2 * geometry.slot_size + IO_ALIGNMENT
    store = torch.zeros(second_offset + geometry.slot_size, dtype=torch.uint8)
    store[first_offset : first_offset + 2 * geometry.slot_size] = raw[
        : 2 * geometry.slot_size
    ]
    store[second_offset:] = raw[2 * geometry.slot_size :]
    blocks = [2, 0, 1]

    expected = _kv_cache(geometry)
    slots = raw.cuda().view(expected.dtype).view(3, *expected.shape[1:])
    expected.index_copy_(0, torch.tensor(blocks, device="cuda"), slots)

    pool = FixedCudaStagingPool(torch.device("cuda"), 4 << 20, 2)
    _FakeLoadClient.store = store.cuda()
    _FakeLoadClient.pool = pool
    _FakeLoadClient.fail_on_call = -1
    _FakeLoadClient.calls = []
    _FakeLoadClient.released = []
    monkeypatch.setattr(load_module, "IPCClientAsync", _FakeLoadClient)

    actual = _kv_cache(geometry)
    pipeline = LoadPipeline("/unused/daser.sock", client_count=2)
    pipeline.configure(
        kv_caches={CROSS_LAYER_KV_CACHE_KEY: actual},
        layer_names=[f"layer.{index}" for index in range(_LAYERS)],
        local_slot_size=geometry.slot_size,
        rank_stride_bytes=geometry.slot_size,
        tp_rank=0,
        staging_pool=pool,
        load_key_scale=1.0,
        load_value_scale=1.0,
        rope_delta_scale=1.0,
        rope_base=10000.0,
        rope_rotary_dim=0,
        rope_is_neox_style=True,
    )
    pipeline.initialize_transfer()
    specs = {
        "req:load:0": ReqLoadSpec(
            chunk_key="a",
            start_slot=0,
            num_slots=2,
            block_ids=blocks[:2],
            file_offset=first_offset,
            token_count=2 * geometry.block_tokens,
            lease_id="req",
        ),
        "req:load:1": ReqLoadSpec(
            chunk_key="b",
            start_slot=2,
            num_slots=1,
            block_ids=blocks[2:],
            file_offset=second_offset,
            token_count=geometry.block_tokens,
            lease_id="req",
        ),
    }
    bits = actual.view(torch.int16)
    want = expected.view(torch.int16)
    try:
        pipeline.start_layerwise(specs)
        for layer in range(_LAYERS):
            pipeline.wait_for_layer(layer)
            torch.cuda.current_stream().synchronize()
            # Random bytes include NaN patterns: compare bits, not values.
            assert torch.equal(bits[:, : layer + 1], want[:, : layer + 1])
        pipeline.start_layerwise({})
        assert torch.equal(bits, want)
        assert pipeline.take_invalid_block_ids() == set()
        # No header read: one transfer per layer on alternating lanes.
        lanes = [layer % 2 for layer in range(_LAYERS)]
        assert sorted(_FakeLoadClient.calls) == sorted(lanes)
        if slots == "1":
            assert _FakeLoadClient.calls == lanes
        assert _FakeLoadClient.released == ["req"]
    finally:
        pipeline.shutdown()
