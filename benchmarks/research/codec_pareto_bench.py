# SPDX-License-Identifier: Apache-2.0

"""Same-card codec Pareto microbenchmark: stored ratio against GPU throughput.

Loads captured BF16 KV slots (``[slots, layers, 2, tokens, heads, head_dim]``
uint16 ``.npy``), calibrates per-plane codebooks on the first slots and
measures, on the remaining slots of the same card:

- the raw device-to-device copy bound;
- DaseR online encode (``FusedOnlineKVPacker.pack_into``) and decode
  (``FusedCompressedKVDecoder``) for each record format;
- optionally nvCOMP algorithms with and without a BF16 byte shuffle, the
  shuffle and unshuffle kernels included in the timed region;
- optionally the SplitZip GPU codec, calibrated on the same slots as DaseR;
- optionally single-core CPU zstd over byte-shuffled slots.

Every decode row is checked byte-exact. One JSON object per arm is printed
and the Pareto front (lowest ratio for a given decode throughput) is marked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

from daser.compression.calibration import calibrate_codebooks
from daser.compression.format import (
    GROUP_PREFIX_ENV,
    ONLINE_TILE_SCALARS,
    CompressedStoreGeometry,
)
from daser.ops.compressed_kv import FusedCompressedKVDecoder, FusedOnlineKVPacker
from daser.ops.kv_layout import allocate_cross_layer_kv_cache


def byte_shuffle(data: torch.Tensor) -> torch.Tensor:
    """Split 16-bit scalars into a low-byte stream followed by a high-byte stream.

    Args:
        data (torch.Tensor): Contiguous byte tensor of even length.

    Returns:
        torch.Tensor: New byte tensor, all low bytes then all high bytes.
    """
    return data.view(-1, 2).t().contiguous().view(-1)


def byte_unshuffle(data: torch.Tensor) -> torch.Tensor:
    """Invert ``byte_shuffle``.

    Args:
        data (torch.Tensor): Shuffled byte tensor of even length.

    Returns:
        torch.Tensor: New byte tensor with interleaved little-endian scalars.
    """
    return data.view(2, -1).t().contiguous().view(-1)


def pareto_front(rows: list[dict[str, Any]]) -> list[str]:
    """Return arms not dominated in (lower ratio, higher decode throughput).

    Args:
        rows (list[dict[str, Any]]): Rows with ``arm``, ``ratio`` and
            ``dec_gbs``; rows lacking either metric are ignored.

    Returns:
        list[str]: Non-dominated arm names, fastest first.
    """
    points = [
        r for r in rows if r.get("ratio") is not None and r.get("dec_gbs") is not None
    ]
    points.sort(key=lambda r: (-r["dec_gbs"], r["ratio"]))
    front, best_ratio = [], float("inf")
    for row in points:
        if row["ratio"] < best_ratio:
            front.append(row["arm"])
            best_ratio = row["ratio"]
    return front


def _best_ms(fn: Callable[[], Any], stream: torch.cuda.Stream, repeats: int) -> float:
    """Return the best event-timed milliseconds of ``fn`` on ``stream``."""
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record(stream)
        fn()
        end.record(stream)
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def _emit(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    """Record and print one result row."""
    rows.append(row)
    print(json.dumps(row), flush=True)


def _memcpy_row(raw: np.ndarray, repeats: int) -> dict[str, Any]:
    """Time one device-to-device copy of the evaluation slots (the bound)."""
    source = torch.from_numpy(raw.reshape(-1).view(np.uint8)).cuda()
    target = torch.empty_like(source)
    ms = _best_ms(lambda: target.copy_(source), torch.cuda.current_stream(), repeats)
    return {"arm": "memcpy", "ratio": 1.0, "dec_gbs": raw.nbytes / ms / 1e6}


def _daser_rows(
    raw: np.ndarray, books: bytes, formats: list[str], repeats: int
) -> list[dict[str, Any]]:
    """Encode and decode the evaluation slots with every record format."""
    rows: list[dict[str, Any]] = []
    n, layers, _, tokens, heads, dim = raw.shape
    slot_size = raw[0].nbytes
    total = raw.nbytes
    stream = torch.cuda.Stream()
    source = torch.from_numpy(raw.view(np.int16)).cuda()
    kv_cache = allocate_cross_layer_kv_cache(
        n, layers, tokens, heads, dim, dtype=torch.bfloat16, device="cuda"
    )
    staging = torch.empty(n * slot_size, dtype=torch.uint8, device="cuda")
    blocks = list(range(n))
    for fmt in formats:
        os.environ[GROUP_PREFIX_ENV] = "1" if fmt == "v2" else "0"
        kv_cache.view(torch.int16).copy_(source)
        packer = FusedOnlineKVPacker(
            kv_cache=kv_cache,
            codebooks=books,
            tile_scalars=ONLINE_TILE_SCALARS,
            max_slots_per_buffer=n,
        )

        def pack(packer: FusedOnlineKVPacker = packer) -> list[Any]:
            return packer.pack_into(
                staging=staging,
                block_ids=blocks,
                logical_slots=blocks,
                slot_stride=slot_size,
                stream=stream,
            )

        slots = pack()
        torch.cuda.synchronize()
        stored = sum(s.stored_length for s in slots)
        modes = [int(s.mode) for s in slots]
        # Records are packed back to back; their staging offsets are cumulative.
        offsets = [s.source_offset for s in slots]
        best = float("inf")
        for _ in range(repeats):  # pack_into synchronizes to read lengths
            torch.cuda.synchronize()
            began = time.perf_counter()
            pack()
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - began)
        _emit(
            rows,
            {
                "arm": f"daser_{fmt}_encode",
                "ratio": stored / total,
                "enc_gbs": total / best / 1e9,
            },
        )
        decoder = FusedCompressedKVDecoder(
            kv_cache=kv_cache,
            codebooks=books,
            tile_scalars=ONLINE_TILE_SCALARS,
            ring_depth=1,
            max_slots_per_buffer=n,
        )

        def decode(
            decoder: FusedCompressedKVDecoder = decoder,
            offsets: list[int] = offsets,
            modes: list[int] = modes,
        ) -> int:
            plan = decoder.prepare(
                staging=staging,
                staging_offsets=offsets,
                block_ids=blocks,
                modes=modes,
                buffer_index=0,
                stream=stream,
            )
            return plan.submit() if plan is not None else 0

        kv_cache.zero_()
        decode()
        torch.cuda.synchronize()
        exact = torch.equal(kv_cache.view(torch.int16), source)
        ms = _best_ms(decode, stream, repeats)
        _emit(
            rows,
            {
                "arm": f"daser_{fmt}_decode",
                "ratio": stored / total,
                "dec_gbs": total / ms / 1e6,
                "exact": exact,
            },
        )
    os.environ.pop(GROUP_PREFIX_ENV, None)
    return rows


def _nvcomp_rows(
    raw: np.ndarray, nvcomp_path: str, algorithms: list[str], repeats: int
) -> list[dict[str, Any]]:
    """Time nvCOMP batched slot compression with and without byte shuffle."""
    import nvidia

    nvidia.__path__.append(nvcomp_path)
    from nvidia import nvcomp

    rows: list[dict[str, Any]] = []
    stream = torch.cuda.current_stream()
    slots = [torch.from_numpy(s.reshape(-1).view(np.uint8)).cuda() for s in raw]
    total = raw.nbytes
    # BF16-aware modes; ANS with "<f2" is nvCOMP's best ratio on KV (R194).
    options = {
        "Cascaded": {"data_type": "<u2"},
        "Bitcomp": {"data_type": "<u2"},
        "ANS": {"data_type": "<f2"},
    }
    for algorithm in algorithms:
        for shuffled in (False, True):
            arm = f"nvcomp_{'shuf_' if shuffled else ''}{algorithm}"
            try:
                codec = nvcomp.Codec(
                    algorithm=algorithm,
                    cuda_stream=stream.cuda_stream,
                    **({} if shuffled else options.get(algorithm, {})),
                )
                prep = byte_shuffle if shuffled else (lambda t: t)
                post = byte_unshuffle if shuffled else (lambda t: t)

                def encode(codec: Any = codec, prep: Callable = prep) -> Any:
                    return codec.encode([nvcomp.as_array(prep(s)) for s in slots])

                encoded = encode()
                torch.cuda.synchronize()

                def decode(
                    codec: Any = codec, encoded: Any = encoded, post: Callable = post
                ) -> tuple[Any, list[torch.Tensor]]:
                    # DLPack views do not own nvCOMP buffers: return the
                    # arrays too so the views stay valid while in use.
                    out = codec.decode(encoded)
                    return out, [
                        post(torch.from_dlpack(o).view(torch.uint8).reshape(-1))
                        for o in out
                    ]

                _, decoded = decode()
                exact = all(
                    torch.equal(d, s) for d, s in zip(decoded, slots, strict=True)
                )
                _emit(
                    rows,
                    {
                        "arm": arm,
                        "ratio": sum(e.buffer_size for e in encoded) / total,
                        "enc_gbs": total / _best_ms(encode, stream, repeats) / 1e6,
                        "dec_gbs": total / _best_ms(decode, stream, repeats) / 1e6,
                        "exact": exact,
                    },
                )
            except Exception as exc:  # report unsupported options and continue
                _emit(rows, {"arm": arm, "error": repr(exc)[:200]})
    return rows


def _splitzip_rows(
    raw: np.ndarray, calibration: np.ndarray, splitzip_path: str, repeats: int
) -> list[dict[str, Any]]:
    """Time the SplitZip GPU codec, calibrated on the DaseR calibration slots."""
    sys.path.insert(0, splitzip_path)
    from codec_gpu import ChunkLocalSplitZipGPU

    codec = ChunkLocalSplitZipGPU(device="cuda", chunk_size=1024)
    codec.calibrate(torch.from_numpy(calibration.view(np.int16)).cuda().view(-1))
    data = torch.from_numpy(raw.view(np.int16)).cuda().view(-1)
    encoded = codec.encode(data)
    exact = torch.equal(codec.decode(encoded).view(torch.int16), data)
    stream = torch.cuda.current_stream()
    kv = torch.empty_like(data)

    def decode_into_kv() -> None:
        # SplitZip decodes to a fresh contiguous tensor; restoring it needs at
        # least one more pass into the KV buffer, which DaseR's restore fuses.
        kv.copy_(codec.decode(encoded).view(torch.int16))

    return [
        {
            "arm": "splitzip_then_copy",
            "ratio": encoded.compressed_bytes / raw.nbytes,
            "dec_gbs": raw.nbytes / _best_ms(decode_into_kv, stream, repeats) / 1e6,
            "exact": exact,
        },
        {
            "arm": "splitzip",
            "ratio": encoded.compressed_bytes / raw.nbytes,
            "enc_gbs": raw.nbytes
            / _best_ms(lambda: codec.encode(data), stream, repeats)
            / 1e6,
            "dec_gbs": raw.nbytes
            / _best_ms(lambda: codec.decode(encoded), stream, repeats)
            / 1e6,
            "exact": exact,
        },
    ]


def _cpu_zstd_rows(raw: np.ndarray, level: int) -> list[dict[str, Any]]:
    """Time single-core zstd over byte-shuffled slots (per-core throughput)."""
    import zstandard

    compressor = zstandard.ZstdCompressor(level=level)
    decompressor = zstandard.ZstdDecompressor()
    slots = [
        byte_shuffle(torch.from_numpy(s.reshape(-1).view(np.uint8))).numpy().tobytes()
        for s in raw
    ]
    began = time.perf_counter()
    encoded = [compressor.compress(s) for s in slots]
    middle = time.perf_counter()
    for chunk in encoded:
        decompressor.decompress(chunk)
    ended = time.perf_counter()
    return [
        {
            "arm": f"cpu_shuf_zstd{level}_per_core",
            "ratio": sum(map(len, encoded)) / raw.nbytes,
            "enc_gbs": raw.nbytes / (middle - began) / 1e9,
            "dec_gbs": raw.nbytes / (ended - middle) / 1e9,
        }
    ]


def main() -> None:
    """Run the Pareto sweep and print one JSON row per arm plus the front."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", required=True, help="captured KV .npy")
    parser.add_argument("--model-id", default="captured")
    parser.add_argument("--calibration-slots", type=int, default=8)
    parser.add_argument("--formats", nargs="+", default=["v1", "v2"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--nvcomp-path",
        default=None,
        help="nvidia namespace directory that holds the nvcomp package",
    )
    parser.add_argument(
        "--nvcomp-algorithms", nargs="+", default=["LZ4", "Bitcomp", "ANS", "Zstd"]
    )
    parser.add_argument(
        "--cpu-zstd-level", type=int, default=None, help="add a CPU zstd row"
    )
    parser.add_argument(
        "--splitzip-path",
        default=None,
        help="directory holding the SplitZip codec_gpu module",
    )
    args = parser.parse_args()

    slots = np.load(args.slots)
    count, layers, _, tokens, heads, dim = slots.shape
    geometry = CompressedStoreGeometry(
        num_slots=count,
        slot_size=slots[0].nbytes,
        block_tokens=tokens,
        num_layers=layers,
        num_kv_heads=heads,
        head_dim=dim,
        tile_scalars=ONLINE_TILE_SCALARS,
    )
    ncal = args.calibration_slots
    books = calibrate_codebooks(
        (slots[i].tobytes() for i in range(ncal)),
        model_id=args.model_id,
        geometry=geometry,
    ).codebooks
    evaluate = np.ascontiguousarray(slots[ncal:])
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        _emit(rows, _memcpy_row(evaluate, args.repeats))
        rows += _daser_rows(evaluate, books, args.formats, args.repeats)
        if args.nvcomp_path:
            rows += _nvcomp_rows(
                evaluate, args.nvcomp_path, args.nvcomp_algorithms, args.repeats
            )
        if args.splitzip_path:
            for row in _splitzip_rows(
                evaluate, slots[:ncal], args.splitzip_path, args.repeats
            ):
                _emit(rows, row)
    if args.cpu_zstd_level is not None:
        for row in _cpu_zstd_rows(evaluate, args.cpu_zstd_level):
            _emit(rows, row)
    print(json.dumps({"pareto_front": pareto_front(rows)}), file=sys.stdout)


if __name__ == "__main__":
    main()
