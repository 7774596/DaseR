# SPDX-License-Identifier: Apache-2.0
"""Per-layer step cost of FragPack attention vs FA3 at a serving geometry.

Fills one single-layer :class:`HbmKvStore` with captured KV (one TP rank's
heads) through the public step API and keeps the same pages raw for FA3,
then times, interleaved per round, the per-layer attention work of one engine
step:

- decode: FA3 paged (vLLM's FULL-graph setting: AOT scheduler metadata, at
  most ``--fa3-max-splits`` splits) vs the fused kernel, on batches whose
  context lengths are drawn from ``--lengths`` (e.g. the replay's prompt
  lengths) or given explicitly;
- prefill: one chunk over a cached context, FA3 over the raw paged cache vs
  ``gather`` + FA3 over the BF16 workspace;
- the per-layer ``write`` + ``encode`` of a decode-only step.

``--raw-records`` calibrates the codebooks on random bytes, so the store
keeps every record raw and the fused kernel reads the format without the
byte savings.

Usage:
    python -m benchmarks.research.fragpack_step_bench --dump KV.npy \
        --heads 0:2 --out OUT.json [--lengths lengths.npy]
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import math
from pathlib import Path
import statistics
from typing import Callable

import numpy as np
import torch

from daser.compression import attn_page as ap
from daser.compression.attn_tables import calibrate, device_tables
from daser.logging import init_logger
from daser.ops.fused_attn_decode import (
    GQA,
    RecordTables,
    decode_attention,
    decode_units,
    workspace_elems,
)
from daser.ops.hbm_kv import HbmKvStore, page_bytes

logger = init_logger(__name__)

T = ap.PAGE_TOKENS
D = ap.HEAD_DIM
FILL_BLOCKS = 64


def _time_rounds(fns: list[Callable[[], None]], rounds: int, inner: int) -> list[float]:
    """Return the median microseconds per call of each function.

    Args:
        fns: Zero-argument callables enqueueing GPU work.
        rounds: Rounds; each round times every function once (interleaved).
        inner: Calls per timed sample.

    Returns:
        One median per function.
    """
    for fn in fns:
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    samples: list[list[float]] = [[] for _ in fns]
    for _ in range(rounds):
        for i, fn in enumerate(fns):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(inner):
                fn()
            end.record()
            end.synchronize()
            samples[i].append(start.elapsed_time(end) * 1e3 / inner)
    return [statistics.median(s) for s in samples]


def _fill(store: HbmKvStore, kv: torch.Tensor) -> None:
    """Write and encode every block of ``kv`` (``[NB, 2, T, H, D]``)."""
    no_live = np.zeros(0, np.int32)
    for b0 in range(0, kv.shape[0], FILL_BLOCKS):
        n = min(FILL_BLOCKS, kv.shape[0] - b0)
        blocks = torch.arange(b0, b0 + n, device="cuda")
        slots = (blocks[:, None] * T + torch.arange(T, device="cuda")).reshape(-1)
        store.prepare(slots, slots.numel(), no_live)
        part = kv[b0 : b0 + n]
        store.write(
            0,
            part[:, 0].reshape(-1, *part.shape[3:]),
            part[:, 1].reshape(-1, *part.shape[3:]),
            slots,
        )
        store.encode(0, n)
        store.commit()
    torch.cuda.synchronize()


def _batches(args: argparse.Namespace, max_ctx: int) -> list[list[int]]:
    """Return decode batches (context lengths) to time."""
    out = [[int(x) for x in b.split("/")] for b in args.decode.split(",") if b]
    if args.lengths:
        pool = np.load(args.lengths).astype(np.int64)
        pool = pool[(pool > 0) & (pool <= max_ctx)]
        rng = np.random.default_rng(0)
        for _ in range(args.samples):
            out.append(sorted(int(x) for x in rng.choice(pool, args.batch)))
    return out


def _decode_rows(
    store: HbmKvStore,
    kc: torch.Tensor,
    vc: torch.Tensor,
    heads: int,
    batches: list[list[int]],
    args: argparse.Namespace,
) -> list[dict]:
    """Time FA3 vs fused decode attention for each batch."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from vllm.vllm_flash_attn.flash_attn_interface import get_scheduler_metadata

    nb = kc.shape[0]
    scale = D**-0.5
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    tables = RecordTables(*store.layer_tables(0))
    rows = []
    for lens in batches:
        b = len(lens)
        pages = [math.ceil(x / T) for x in lens]
        width = max(pages)
        perm = torch.randperm(nb, device="cuda", dtype=torch.int32)
        bt = torch.zeros(b, width, dtype=torch.int32, device="cuda")
        used = 0
        for i, p in enumerate(pages):
            bt[i, :p] = perm[used : used + p]
            used += p
        q = torch.randn(b, heads * GQA, D, dtype=torch.bfloat16, device="cuda")
        seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
        cu = torch.arange(b + 1, dtype=torch.int32, device="cuda")
        out_fa, out_fp = torch.empty_like(q), torch.empty_like(q)
        auto = decode_units(b, heads, sms)
        sweep = [auto] + [int(u) for u in args.units.split(",") if u]
        opart, ml, plan = workspace_elems(heads, max(sweep))
        f32 = dict(dtype=torch.float32, device="cuda")
        part = (torch.empty(opart, **f32), torch.empty(ml, **f32))
        # One plan per unit count, written once: a step plans at its first
        # layer and every layer reuses the plan, timed separately below.
        ws = [
            (*part, torch.empty(plan, dtype=torch.int32, device="cuda")) for _ in sweep
        ]
        for u, w in zip(sweep, ws, strict=True):
            decode_attention(q, tables, bt, seq, scale, u, out_fp, w, True)
        meta = get_scheduler_metadata(
            batch_size=b,
            max_seqlen_q=1,
            max_seqlen_k=max(lens),
            num_heads_q=heads * GQA,
            num_heads_kv=heads,
            headdim=D,
            cache_seqlens=seq,
            qkv_dtype=torch.bfloat16,
            cu_seqlens_q=cu,
            page_size=T,
            causal=True,
            num_splits=args.fa3_max_splits,
        )

        fa3 = partial(
            flash_attn_varlen_func,
            q=q,
            k=kc,
            v=vc,
            out=out_fa,
            cu_seqlens_q=cu,
            max_seqlen_q=1,
            seqused_k=seq,
            max_seqlen_k=max(lens),
            softmax_scale=scale,
            causal=True,
            block_table=bt,
            scheduler_metadata=meta,
            num_splits=args.fa3_max_splits,
            fa_version=3,
        )
        fused = [
            partial(decode_attention, q, tables, bt, seq, scale, u, out_fp, w, False)
            for u, w in zip(sweep, ws, strict=True)
        ]
        replan = partial(
            decode_attention, q, tables, bt, seq, scale, auto, out_fp, ws[0], True
        )
        t_fa, t_plan, *t_us = _time_rounds([fa3, replan, *fused], args.rounds, 10)
        t_fp = t_us[0]
        raw_bytes = 2 * sum(pages) * T * heads * D * 2
        rows.append(
            {
                "kind": "decode",
                "lens": lens,
                "units": auto,
                "sweep_us": dict(
                    zip(sweep[1:], (round(t, 1) for t in t_us[1:]), strict=True)
                ),
                "us_fa3": round(t_fa, 1),
                "us_fused": round(t_fp, 1),
                "us_plan": round(t_plan - t_fp, 1),
                "fused_over_fa3": round(t_fp / t_fa, 3),
                "raw_gbs_fa3": round(raw_bytes / t_fa / 1e3),
                "raw_gbs_fused": round(raw_bytes / t_fp / 1e3),
                "maxdiff": float((out_fa.float() - out_fp.float()).abs().max()),
            }
        )
        logger.info("%s", rows[-1])
    return rows


def _prefill_rows(
    store: HbmKvStore,
    kc: torch.Tensor,
    vc: torch.Tensor,
    heads: int,
    configs: list[tuple[int, int]],
    rounds: int,
) -> list[dict]:
    """Time one prefill chunk: FA3 over raw pages vs gather + FA3 workspace."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    nb = kc.shape[0]
    scale = D**-0.5
    rows = []
    for ctx, chunk in configs:
        pages = math.ceil(ctx / T)
        src = torch.randperm(nb, device="cuda", dtype=torch.int32)[:pages]
        wk = torch.empty(pages, T, heads, D, dtype=torch.bfloat16, device="cuda")
        wv = torch.empty_like(wk)
        q = torch.randn(chunk, heads * GQA, D, dtype=torch.bfloat16, device="cuda")
        out = torch.empty_like(q)
        cu_q = torch.tensor([0, chunk], dtype=torch.int32, device="cuda")
        seq = torch.tensor([ctx], dtype=torch.int32, device="cuda")
        bt_raw = src.view(1, -1)
        bt_ws = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, -1)

        fa = partial(
            flash_attn_varlen_func,
            q=q,
            out=out,
            cu_seqlens_q=cu_q,
            max_seqlen_q=chunk,
            seqused_k=seq,
            max_seqlen_k=ctx,
            softmax_scale=scale,
            causal=True,
            fa_version=3,
        )
        t_raw, t_g, t_ws = _time_rounds(
            [
                partial(fa, k=kc, v=vc, block_table=bt_raw),
                partial(store.gather, 0, src, wk, wv),
                partial(fa, k=wk, v=wv, block_table=bt_ws),
            ],
            rounds,
            3,
        )
        rows.append(
            {
                "kind": "prefill",
                "ctx": ctx,
                "chunk": chunk,
                "us_fa3_raw": round(t_raw, 1),
                "us_gather": round(t_g, 1),
                "us_fa3_ws": round(t_ws, 1),
                "extra_us": round(t_g + t_ws - t_raw, 1),
            }
        )
        logger.info("%s", rows[-1])
    return rows


def _write_encode_row(store: HbmKvStore, heads: int, batch: int, rounds: int) -> dict:
    """Time one layer's ``write`` + ``encode`` of a decode-only step."""
    nb = store.num_blocks
    # Decode tokens land mid-block (no close) except the bound passed to encode.
    slots = (torch.arange(batch, device="cuda") * 7 + 3) * T + 5
    key = torch.randn(batch, heads, D, dtype=torch.bfloat16, device="cuda")
    store.prepare(slots, batch, np.zeros(0, np.int32))
    assert int(slots.max()) < nb * T

    t_w, t_e = _time_rounds(
        [partial(store.write, 0, key, key, slots), partial(store.encode, 0, batch)],
        rounds,
        20,
    )
    store.commit()
    return {
        "kind": "write_encode",
        "B": batch,
        "us_write": round(t_w, 1),
        "us_encode": round(t_e, 1),
    }


def main() -> None:
    """Run the benchmark and write JSON results."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dump", required=True, help="uint16 [S, L, 2, 128, H, 128] KV")
    p.add_argument("--out", required=True)
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--heads", default="0:2", help="KV head slice of one TP rank")
    p.add_argument("--calib-pages", type=int, default=8)
    p.add_argument("--blocks", type=int, default=16384)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--fa3-max-splits", type=int, default=32)
    p.add_argument("--units", default="", help="extra unit counts to time")
    p.add_argument("--decode", default="", help="batches like 8000/64000,...")
    p.add_argument("--lengths", help="int .npy of context lengths to sample")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--raw-records", action="store_true")
    p.add_argument("--prefill", default="16000x2048,100000x2048,100000x8192")
    args = p.parse_args()
    torch.manual_seed(0)
    h0, h1 = (int(x) for x in args.heads.split(":"))
    dump = np.load(args.dump, mmap_mode="r")
    layer = np.asarray(dump[:, args.layer : args.layer + 1, :, :, h0:h1])
    heads = h1 - h0
    calib = layer[: args.calib_pages]
    if args.raw_records:
        calib = np.random.default_rng(0).integers(0, 1 << 16, calib.shape, np.uint16)
    table = calibrate(calib)
    content = layer[args.calib_pages :, 0]
    idx = np.arange(args.blocks) % content.shape[0]
    kv = torch.from_numpy(content[idx].view(np.int16)).cuda().view(torch.bfloat16)
    nb = args.blocks
    cache = torch.zeros(nb, page_bytes(heads), dtype=torch.uint8, device="cuda")
    store = HbmKvStore(
        [cache],
        heads,
        device_tables(table, torch.device("cuda")),
        FILL_BLOCKS * T,
        8,
        # Raw records all live in overflow.
        2 * heads * nb + 4096 if args.raw_records else 4096,
    )
    _fill(store, kv)
    st = store.stats()
    kc, vc = kv[:, 0].contiguous(), kv[:, 1].contiguous()
    del kv
    head = {
        "gpu": torch.cuda.get_device_name(0),
        "heads": heads,
        "blocks": nb,
        "fallbacks": st.fallbacks,
        "spills": st.spills,
        "raw_records": args.raw_records,
    }
    logger.info("%s", head)
    rows = _decode_rows(
        store, kc, vc, heads, _batches(args, nb * T // args.batch), args
    )
    pre = [tuple(map(int, c.split("x"))) for c in args.prefill.split(",") if c]
    rows += _prefill_rows(store, kc, vc, heads, pre, args.rounds)
    rows.append(_write_encode_row(store, heads, args.batch, args.rounds))
    logger.info("%s", rows[-1])
    Path(args.out).write_text(json.dumps({"head": head, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
