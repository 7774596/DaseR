# SPDX-License-Identifier: Apache-2.0

"""Layer-group restore microbenchmark for layer-major staged packed records.

Encodes packed records at a real model geometry, stages each layer group
layer-major (header page plus the group's planes) and restores it with the
fused decoder's ``plane_range``. Prints, per group size, the staging bytes,
host prepare+submit cost and GPU decode time per group against one
record-major restore of every plane.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from daser.compression import (
    CompressedStoreGeometry,
    SlotMode,
    default_online_codebooks,
    encode_slot,
)
from daser.compression.layer_major import (
    plan_layer_group,
    plan_layer_groups,
    record_header_bytes,
    stage_layer_group,
)
from daser.ops.compressed_kv import FusedCompressedKVDecoder
from daser.ops.kv_layout import allocate_cross_layer_kv_cache


def _record(geometry: CompressedStoreGeometry, seed: int, codebooks: bytes) -> bytes:
    """Encode one BF16-like record whose high bytes follow the codebook."""
    rng = np.random.default_rng(seed)
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = rng.integers(0, 256, geometry.slot_size // 2, dtype=np.uint8)
    raw[1::2] = np.array([0x3C, 0x3D, 0x3E, 0xBC, 0xBD, 0xBE], dtype=np.uint8)[
        rng.integers(0, 6, geometry.slot_size // 2, dtype=np.uint8)
    ]
    encoded = encode_slot(
        raw.tobytes(), slot_id=seed, geometry=geometry, codebooks=codebooks
    )
    if encoded.mode is not SlotMode.COMPRESSED:
        raise RuntimeError("synthetic record did not compress")
    return encoded.payload


def _time_restores(
    decoder: FusedCompressedKVDecoder,
    stagings: list[tuple[torch.Tensor, list[int], tuple[int, int] | None]],
    blocks: list[int],
    stream: torch.cuda.Stream,
    repeats: int,
) -> tuple[float, float]:
    """Return best (host ms per restore, GPU ms per restore) over ``repeats``."""
    best_host, best_gpu = float("inf"), float("inf")
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    modes = [1] * len(blocks)
    for _ in range(repeats):
        stream.synchronize()
        host = 0.0
        start.record(stream)
        for staging, offsets, plane_range in stagings:
            began = time.perf_counter()
            plan = decoder.prepare(
                staging=staging,
                staging_offsets=offsets,
                block_ids=blocks,
                modes=modes,
                buffer_index=0,
                stream=stream,
                plane_range=plane_range,
            )
            assert plan is not None
            plan.submit()
            host += time.perf_counter() - began
        end.record(stream)
        stream.synchronize()
        best_host = min(best_host, host * 1e3 / len(stagings))
        best_gpu = min(best_gpu, start.elapsed_time(end) / len(stagings))
    return best_host, best_gpu


def _time_grouped(
    decoder: FusedCompressedKVDecoder,
    stagings: list[tuple[torch.Tensor, list[int], tuple[int, int]]],
    blocks: list[int],
    stream: torch.cuda.Stream,
    repeats: int,
) -> tuple[float, float]:
    """Time one ``prepare`` plus one ``submit_planes`` per group.

    Returns best (host ms per group, GPU ms per group) over ``repeats``.
    """
    best_host, best_gpu = float("inf"), float("inf")
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for _ in range(repeats):
        stream.synchronize()
        start.record(stream)
        began = time.perf_counter()
        plan = decoder.prepare(
            staging=stagings[0][0],
            staging_offsets=stagings[0][1],
            block_ids=blocks,
            modes=[1] * len(blocks),
            buffer_index=0,
            stream=stream,
        )
        assert plan is not None
        for staging, _offsets, plane_range in stagings:
            plan.submit_planes(*plane_range, staging)
        host = time.perf_counter() - began
        end.record(stream)
        stream.synchronize()
        best_host = min(best_host, host * 1e3 / len(stagings))
        best_gpu = min(best_gpu, start.elapsed_time(end) / len(stagings))
    return best_host, best_gpu


def main() -> None:
    """Run the layer-group restore sweep and print one row per group size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=64)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-tokens", type=int, default=128)
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4, 12, 36])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    slot_size = args.layers * 2 * args.block_tokens * args.heads * args.head_dim * 2
    geometry = CompressedStoreGeometry(
        num_slots=args.records,
        slot_size=slot_size,
        block_tokens=args.block_tokens,
        num_layers=args.layers,
        num_kv_heads=args.heads,
        head_dim=args.head_dim,
    )
    codebooks = default_online_codebooks(geometry)
    unique = [_record(geometry, seed, codebooks) for seed in range(4)]
    payloads = [unique[i % len(unique)] for i in range(args.records)]
    record_bytes = sum(len(p) for p in payloads)
    headers = np.stack(
        [
            np.frombuffer(p, dtype=np.uint8, count=record_header_bytes(p))
            for p in payloads
        ]
    )
    blocks = list(np.random.default_rng(7).permutation(args.records).tolist())
    stream = torch.cuda.Stream()
    with torch.inference_mode():
        kv_cache = allocate_cross_layer_kv_cache(
            args.records,
            args.layers,
            args.block_tokens,
            args.heads,
            args.head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        decoder = FusedCompressedKVDecoder(
            kv_cache=kv_cache,
            codebooks=codebooks,
            tile_scalars=geometry.tile_scalars,
            ring_depth=1,
            max_slots_per_buffer=args.records,
        )
    whole = torch.from_numpy(
        np.frombuffer(b"".join(payloads), dtype=np.uint8).copy()
    ).cuda()
    whole_offsets = np.concatenate(
        ([0], np.cumsum([len(p) for p in payloads])[:-1])
    ).tolist()
    _time_restores(decoder, [(whole, whole_offsets, None)], blocks, stream, 1)
    host, gpu = _time_restores(
        decoder, [(whole, whole_offsets, None)], blocks, stream, args.repeats
    )
    print(
        f"records={args.records} record_mb={record_bytes / args.records / 1e6:.2f} "
        f"ratio={record_bytes / (args.records * slot_size):.3f} "
        f"tile={geometry.tile_scalars}"
    )
    print(
        f"record-major all planes: host {host:.2f} ms gpu {gpu:.2f} ms "
        f"({record_bytes / gpu / 1e6:.0f} GB/s packed)"
    )
    print(
        "group_layers groups staged/record host_ms/group gpu_ms/group "
        "gpu_total_ms | shared: staged/record host_ms/group gpu_ms/group"
    )
    for group_layers in args.groups:
        stagings = []
        staged = 0
        for first in range(0, args.layers, group_layers):
            count = min(group_layers, args.layers - first)
            plan = plan_layer_group(headers, (2 * first, 2 * count))
            image = stage_layer_group(payloads, plan)
            staged += plan.staging_bytes
            tensor = torch.from_numpy(np.frombuffer(image, dtype=np.uint8).copy())
            stagings.append(
                (tensor.cuda(), plan.staging_offsets.tolist(), plan.plane_range)
            )
        host, gpu = _time_restores(decoder, stagings, blocks, stream, args.repeats)
        del stagings
        ranges = [
            (2 * first, 2 * min(group_layers, args.layers - first))
            for first in range(0, args.layers, group_layers)
        ]
        shared = plan_layer_groups(headers, ranges)
        # Each group's descriptors only cover its own planes, so every group
        # needs its own staged image; production refills a two-buffer ring.
        grouped = [
            (
                torch.from_numpy(
                    np.frombuffer(
                        stage_layer_group(payloads, plan), dtype=np.uint8
                    ).copy()
                ).cuda(),
                plan.staging_offsets.tolist(),
                plan.plane_range,
            )
            for plan in shared
        ]
        _time_grouped(decoder, grouped, blocks, stream, 1)
        shared_host, shared_gpu = _time_grouped(
            decoder, grouped, blocks, stream, args.repeats
        )
        print(
            f"{group_layers:12d} {len(ranges):6d} {staged / record_bytes:13.3f} "
            f"{host:12.3f} {gpu:11.3f} {gpu * len(ranges):12.2f} | "
            f"{shared[0].staging_bytes * len(ranges) / record_bytes:13.3f} "
            f"{shared_host:12.3f} {shared_gpu:11.3f}"
        )
        del grouped


if __name__ == "__main__":
    main()
