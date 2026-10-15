# SPDX-License-Identifier: Apache-2.0

"""Same-card codec Pareto microbenchmark: stored ratio against restore throughput.

Loads captured BF16 KV slots (``[slots, layers, 2, tokens, heads, head_dim]``
uint16 ``.npy``), calibrates per-plane codebooks on the first slots and places
the remaining slots (repeated cyclically up to ``--eval-blocks``) at seeded
random blocks of a larger paged KV pool, as a paged allocator would. Every arm
restores into those scattered paged blocks, ``--blocks-per-call`` blocks per
codec call, on the same card:

- the contiguous device-to-device copy bound and the raw paged copy;
- DaseR online encode (``FusedOnlineKVPacker.pack_into``) and decode
  (``FusedCompressedKVDecoder``), which read and write the paged blocks;
- optionally nvCOMP algorithms with and without a BF16 byte shuffle, and the
  SplitZip GPU codec calibrated on the same slots as DaseR. These codecs work
  on contiguous buffers, so restore adds a copy into the paged blocks
  (``dec_gbs``); ``dec_contig_gbs`` stops at the codec's own output;
- optionally single-core CPU zstd over byte-shuffled slots.

Every restore is checked byte-exact. One JSON object per arm and call size is
printed, then the Pareto front (lowest ratio for a given ``dec_gbs``) per call
size.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

from daser.compression.calibration import calibrate_codebooks
from daser.compression.format import ONLINE_TILE_SCALARS, CompressedStoreGeometry
from daser.ops.compressed_kv import FusedCompressedKVDecoder, FusedOnlineKVPacker
from daser.ops.kv_layout import allocate_cross_layer_kv_cache, physical_kv_view


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


def scattered_block_ids(count: int, pool_blocks: int, seed: int) -> list[int]:
    """Return distinct pool blocks in seeded random order, like a paged allocator.

    Args:
        count (int): Number of destination blocks.
        pool_blocks (int): Size of the paged KV pool to draw from.
        seed (int): Permutation seed.

    Returns:
        list[int]: ``count`` distinct block IDs in ``[0, pool_blocks)``.

    Raises:
        ValueError: If the pool is smaller than ``count``.
    """
    if not 0 < count <= pool_blocks:
        raise ValueError("block count must be positive and fit in the pool")
    return np.random.default_rng(seed).permutation(pool_blocks)[:count].tolist()


def call_groups(count: int, blocks_per_call: int) -> list[slice]:
    """Split ``count`` blocks into consecutive restore calls.

    Args:
        count (int): Total blocks restored per timed pass.
        blocks_per_call (int): Blocks handed to one codec call.

    Returns:
        list[slice]: Consecutive slices covering ``range(count)``; the last
        one may be short.
    """
    if count <= 0 or blocks_per_call <= 0:
        raise ValueError("block counts must be positive")
    return [
        slice(first, min(first + blocks_per_call, count))
        for first in range(0, count, blocks_per_call)
    ]


def _best_ms(
    run: Callable[[Any], Any],
    stream: torch.cuda.Stream,
    repeats: int,
    prepare: Callable[[], Any] = lambda: None,
) -> float:
    """Return the best event-timed milliseconds of ``run(prepare())``.

    ``prepare`` runs before the start event of every repeat, so host-side
    setup that the serving path overlaps with payload transfer stays untimed.
    """
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    run(prepare())
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        state = prepare()
        torch.cuda.synchronize()
        start.record(stream)
        run(state)
        end.record(stream)
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def _emit(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    """Record and print one result row."""
    rows.append(row)
    print(json.dumps(row), flush=True)


class _PagedKV:
    """Scattered destination blocks of one paged cross-layer KV pool.

    ``rows`` is the contiguous physical ``[pool_blocks, slot_scalars]`` int16
    view of ``kv_cache``; ``reference`` holds the expected physical bytes of
    ``block_ids`` in order. External codecs compress and restore those
    physical rows, which is what a codec outside the KV layout would see.
    """

    def __init__(
        self, slots: torch.Tensor, pool_blocks: int, block_ids: list[int]
    ) -> None:
        """Allocate the pool and write logical ``slots`` at ``block_ids``."""
        _, layers, _, tokens, heads, dim = slots.shape
        self.kv_cache = allocate_cross_layer_kv_cache(
            pool_blocks, layers, tokens, heads, dim, dtype=torch.bfloat16, device="cuda"
        )
        self.rows = (
            physical_kv_view(self.kv_cache).view(torch.int16).view(pool_blocks, -1)
        )
        self.block_ids = block_ids
        self.ids = torch.tensor(block_ids, dtype=torch.long, device="cuda")
        self.kv_cache.zero_()
        self.kv_cache.view(torch.int16)[self.ids] = slots
        self.reference = self.rows[self.ids].clone()
        self.slot_bytes = int(self.reference[0].nbytes)
        self.total = int(self.reference.nbytes)

    def clear(self) -> None:
        """Zero the destination blocks before a correctness pass."""
        self.rows.index_fill_(0, self.ids, 0)

    def exact(self) -> bool:
        """Return whether every destination block matches its source bytes."""
        torch.cuda.synchronize()
        return torch.equal(self.rows[self.ids], self.reference)


def _memcpy_row(kv: _PagedKV, repeats: int) -> dict[str, Any]:
    """Time one contiguous device-to-device copy of every slot (the bound)."""
    target = torch.empty_like(kv.reference)
    ms = _best_ms(
        lambda _: target.copy_(kv.reference), torch.cuda.current_stream(), repeats
    )
    return {"arm": "memcpy", "ratio": 1.0, "dec_gbs": kv.total / ms / 1e6}


def _raw_copy_rows(kv: _PagedKV, ks: list[int], repeats: int) -> list[dict[str, Any]]:
    """Time the uncompressed restore: contiguous staging into paged blocks."""
    rows = []
    stream = torch.cuda.current_stream()
    for k in ks:
        groups = call_groups(len(kv.block_ids), k)

        def run(_: Any, groups: list[slice] = groups) -> None:
            for g in groups:
                kv.rows.index_copy_(0, kv.ids[g], kv.reference[g])

        kv.clear()
        run(None)
        rows.append(
            {
                "arm": "raw_copy",
                "blocks_per_call": k,
                "ratio": 1.0,
                "dec_gbs": kv.total / _best_ms(run, stream, repeats) / 1e6,
                "exact": kv.exact(),
            }
        )
    return rows


def _daser_rows(
    kv: _PagedKV, books: bytes, ks: list[int], repeats: int
) -> list[dict[str, Any]]:
    """Encode from and decode into the scattered paged blocks with DaseR."""
    rows: list[dict[str, Any]] = []
    count = len(kv.block_ids)
    stream = torch.cuda.Stream()
    staging = torch.empty(count * kv.slot_bytes, dtype=torch.uint8, device="cuda")
    scratch = torch.empty(max(ks) * kv.slot_bytes, dtype=torch.uint8, device="cuda")
    packer = FusedOnlineKVPacker(
        kv_cache=kv.kv_cache,
        codebooks=books,
        tile_scalars=ONLINE_TILE_SCALARS,
        max_slots_per_buffer=count,
    )
    slots = packer.pack_into(
        staging=staging,
        block_ids=kv.block_ids,
        logical_slots=list(range(count)),
        slot_stride=kv.slot_bytes,
        stream=stream,
    )
    torch.cuda.synchronize()
    ratio = sum(s.stored_length for s in slots) / kv.total
    modes = [int(s.mode) for s in slots]
    # Records are packed back to back; their staging offsets are cumulative.
    offsets = [s.source_offset for s in slots]
    # One metadata ring per call so a pass prepares every call up front.
    decoder = FusedCompressedKVDecoder(
        kv_cache=kv.kv_cache,
        codebooks=books,
        tile_scalars=ONLINE_TILE_SCALARS,
        ring_depth=count,
        max_slots_per_buffer=count,
    )
    for k in ks:
        groups = call_groups(count, k)
        best = float("inf")
        for _ in range(repeats + 1):  # pack_into synchronizes to read lengths
            torch.cuda.synchronize()
            began = time.perf_counter()
            for g in groups:
                ids = kv.block_ids[g]
                packer.pack_into(
                    staging=scratch,
                    block_ids=ids,
                    logical_slots=list(range(len(ids))),
                    slot_stride=kv.slot_bytes,
                    stream=stream,
                )
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - began)

        def prepare(groups: list[slice] = groups) -> list[Any]:
            return [
                decoder.prepare(
                    staging=staging,
                    staging_offsets=offsets[g],
                    block_ids=kv.block_ids[g],
                    modes=modes[g],
                    buffer_index=index,
                    stream=stream,
                )
                for index, g in enumerate(groups)
            ]

        def run(plans: list[Any]) -> None:
            for plan in plans:
                plan.submit()

        kv.clear()
        # The decoder runs on its own stream: order it after the clear, or
        # the first launches race the zero fill of their destination blocks.
        stream.wait_stream(torch.cuda.current_stream())
        run(prepare())
        exact = kv.exact()
        _emit(
            rows,
            {
                "arm": "daser",
                "blocks_per_call": k,
                "ratio": ratio,
                "enc_gbs": kv.total / best / 1e9,
                "dec_gbs": kv.total / _best_ms(run, stream, repeats, prepare) / 1e6,
                "exact": exact,
            },
        )
    return rows


def _nvcomp_rows(
    kv: _PagedKV, nvcomp_path: str, algorithms: list[str], ks: list[int], repeats: int
) -> list[dict[str, Any]]:
    """Time nvCOMP per-block compression; restore copies into the paged blocks.

    ``dec_contig_gbs`` stops at nvCOMP's own output buffers (plus unshuffle);
    ``dec_gbs`` adds the copy into the scattered paged blocks, with the byte
    unshuffle fused into that copy.
    """
    import nvidia

    nvidia.__path__.append(nvcomp_path)
    from nvidia import nvcomp

    rows: list[dict[str, Any]] = []
    stream = torch.cuda.current_stream()
    blocks = [kv.rows[b].view(torch.uint8) for b in kv.block_ids]
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
            except Exception as exc:  # report unsupported options and continue
                _emit(rows, {"arm": arm, "error": repr(exc)[:200]})
                continue
            prep = byte_shuffle if shuffled else (lambda t: t)
            for k in ks:
                groups = call_groups(len(blocks), k)

                def encode(
                    _: Any, codec: Any = codec, prep: Callable = prep, groups=groups
                ) -> list[Any]:
                    return [
                        codec.encode([nvcomp.as_array(prep(b)) for b in blocks[g]])
                        for g in groups
                    ]

                try:
                    encoded = encode(None)
                    torch.cuda.synchronize()
                    stored = sum(e.buffer_size for batch in encoded for e in batch)

                    def contig(
                        _: Any, codec: Any = codec, encoded=encoded, shuffled=shuffled
                    ) -> list[Any]:
                        # DLPack views do not own nvCOMP buffers: return the
                        # arrays so the views stay valid while in use.
                        kept = []
                        for batch in encoded:
                            out = codec.decode(batch)
                            kept.append(out)
                            if shuffled:
                                kept += [
                                    byte_unshuffle(
                                        torch.from_dlpack(o).view(torch.uint8)
                                    )
                                    for o in out
                                ]
                        return kept

                    def paged(
                        _: Any,
                        codec: Any = codec,
                        encoded=encoded,
                        shuffled=shuffled,
                        groups=groups,
                    ) -> list[Any]:
                        kept = []
                        for batch, g in zip(encoded, groups, strict=True):
                            out = codec.decode(batch)
                            kept.append(out)
                            for o, dst in zip(out, blocks[g], strict=True):
                                src = torch.from_dlpack(o).view(torch.uint8).reshape(-1)
                                if shuffled:
                                    dst.view(-1, 2).copy_(src.view(2, -1).t())
                                else:
                                    dst.copy_(src)
                        return kept

                    kv.clear()
                    paged(None)
                    exact = kv.exact()
                    _emit(
                        rows,
                        {
                            "arm": arm,
                            "blocks_per_call": k,
                            "ratio": stored / kv.total,
                            "enc_gbs": kv.total
                            / _best_ms(encode, stream, repeats)
                            / 1e6,
                            "dec_gbs": kv.total
                            / _best_ms(paged, stream, repeats)
                            / 1e6,
                            "dec_contig_gbs": kv.total
                            / _best_ms(contig, stream, repeats)
                            / 1e6,
                            "exact": exact,
                        },
                    )
                    del encoded
                except Exception as exc:  # report failures and continue
                    _emit(
                        rows,
                        {"arm": arm, "blocks_per_call": k, "error": repr(exc)[:200]},
                    )
    return rows


def _splitzip_rows(
    kv: _PagedKV,
    calibration: torch.Tensor,
    splitzip_path: str,
    ks: list[int],
    repeats: int,
) -> list[dict[str, Any]]:
    """Time the SplitZip GPU codec, calibrated on the DaseR calibration slots.

    SplitZip takes and returns contiguous tensors, so encode gathers the paged
    blocks first and restore copies the decoded tensor into the paged blocks
    (``dec_gbs``); ``dec_contig_gbs`` stops at SplitZip's output buffer.
    """
    sys.path.insert(0, splitzip_path)
    from codec_gpu import ChunkLocalSplitZipGPU

    codec = ChunkLocalSplitZipGPU(device="cuda", chunk_size=1024)
    codec.calibrate(calibration.view(-1))
    stream = torch.cuda.current_stream()
    rows = []
    for k in ks:
        groups = call_groups(len(kv.block_ids), k)

        def encode(_: Any, groups: list[slice] = groups) -> list[Any]:
            return [codec.encode(kv.rows.index_select(0, kv.ids[g])) for g in groups]

        encoded = encode(None)
        stored = sum(e.compressed_bytes for e in encoded)

        def contig(_: Any, encoded: list[Any] = encoded) -> None:
            for e in encoded:
                codec.decode(e)

        def paged(
            _: Any, encoded: list[Any] = encoded, groups: list[slice] = groups
        ) -> None:
            for e, g in zip(encoded, groups, strict=True):
                out = codec.decode(e).view(torch.int16).view(g.stop - g.start, -1)
                kv.rows.index_copy_(0, kv.ids[g], out)

        kv.clear()
        paged(None)
        rows.append(
            {
                "arm": "splitzip",
                "blocks_per_call": k,
                "ratio": stored / kv.total,
                "enc_gbs": kv.total / _best_ms(encode, stream, repeats) / 1e6,
                "dec_gbs": kv.total / _best_ms(paged, stream, repeats) / 1e6,
                "dec_contig_gbs": kv.total / _best_ms(contig, stream, repeats) / 1e6,
                "exact": kv.exact(),
            }
        )
        del encoded
    return rows


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
    """Run the Pareto sweep and print one JSON row per arm plus the fronts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", required=True, help="captured KV .npy")
    parser.add_argument("--model-id", default="captured")
    parser.add_argument("--calibration-slots", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--eval-blocks",
        type=int,
        default=256,
        help="destination blocks per timed pass (evaluation slots repeat cyclically)",
    )
    parser.add_argument("--blocks-per-call", type=int, nargs="+", default=[1, 16, 256])
    parser.add_argument(
        "--pool-factor",
        type=int,
        default=2,
        help="paged pool size as a multiple of --eval-blocks",
    )
    parser.add_argument("--seed", type=int, default=0)
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
    ks = sorted(set(args.blocks_per_call))
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        source = torch.from_numpy(evaluate.view(np.int16)).cuda()
        cyclic = torch.arange(args.eval_blocks, device="cuda") % source.shape[0]
        block_ids = scattered_block_ids(
            args.eval_blocks, args.pool_factor * args.eval_blocks, args.seed
        )
        kv = _PagedKV(source[cyclic], args.pool_factor * args.eval_blocks, block_ids)
        del source, cyclic
        _emit(rows, _memcpy_row(kv, args.repeats))
        for row in _raw_copy_rows(kv, ks, args.repeats):
            _emit(rows, row)
        rows += _daser_rows(kv, books, ks, args.repeats)
        if args.nvcomp_path:
            rows += _nvcomp_rows(
                kv, args.nvcomp_path, args.nvcomp_algorithms, ks, args.repeats
            )
        if args.splitzip_path:
            calibration = torch.from_numpy(slots[:ncal].view(np.int16)).cuda()
            for row in _splitzip_rows(
                kv, calibration, args.splitzip_path, ks, args.repeats
            ):
                _emit(rows, row)
    if args.cpu_zstd_level is not None:
        for row in _cpu_zstd_rows(evaluate, args.cpu_zstd_level):
            _emit(rows, row)
    for k in ks:
        front = pareto_front([r for r in rows if r.get("blocks_per_call") == k])
        print(json.dumps({"blocks_per_call": k, "pareto_front": front}))


if __name__ == "__main__":
    main()
