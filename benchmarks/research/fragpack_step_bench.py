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
- the per-layer ``write`` + ``encode`` of a decode-only step, both while
  blocks stay open and when their final token closes them.

CUDA-event samples include host launch gaps in the default eager mode.
``--cuda-graph`` replays captured GPU work instead. Decode planning is paid
once per engine step, not once per layer; its reported difference estimate
is not an isolated kernel measurement. Setup, validation, prepare and commit
are outside the timed write/encode scope.
``--warmup-ms`` warms each decode/prefill arm for at least that CUDA-event
duration before sampling; stateful write/encode retains three rebuilt steps.

``--raw-records`` calibrates the codebooks on random bytes, so the store
keeps every record raw and the fused kernel reads the format without the
byte savings.

Usage:
    python -m benchmarks.research.fragpack_step_bench --dump KV.npy \
        --heads 0:2 --out OUT.json [--lengths lengths.npy]
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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


def _summary(samples: list[float]) -> dict:
    """Retain samples and their distribution in microseconds per call."""
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "p10_us": float(np.percentile(samples, 10)),
        "p90_us": float(np.percentile(samples, 90)),
        "stdev_us": statistics.pstdev(samples),
    }


def _time_rounds(
    fns: list[Callable[[], None]],
    rounds: int,
    inner: int,
    cuda_graph: bool,
    invalidate: Callable[[int], None],
    validate: Callable[[int], None],
    warmup_ms: float,
) -> list[list[float]]:
    """Return interleaved CUDA-event samples for stateless/repeatable calls.

    Args:
        fns: Zero-argument callables enqueueing GPU work.
        rounds: Rounds; each round times every function once (interleaved).
        inner: Calls per timed sample.
        cuda_graph: Capture ``inner`` calls and replay once per sample.
        invalidate: Poison arm outputs before its untimed validation run.
        validate: Check one arm's outputs after its untimed run/replay.
        warmup_ms: Minimum CUDA-event warmup milliseconds per repeatable arm,
            after capture and validation; excluded from returned samples.

    Returns:
        Per-call microseconds, grouped by function. Caller must not pass a
        stateful encode that would consume the same close list repeatedly.
    """
    for fn in fns:
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graphs = []
    if cuda_graph:
        for fn in fns:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(inner):
                    fn()
            graphs.append(graph)
    # Validate the actual timing mode once per arm. Poisoning first prevents
    # a missing graph write from passing on an earlier arm's shared output.
    # This also warms the graph replay path before collecting timed samples.
    for i, fn in enumerate(fns):
        invalidate(i)
        if cuda_graph:
            graphs[i].replay()
        else:
            fn()
        validate(i)
    torch.cuda.synchronize()
    for i, fn in enumerate(fns):
        elapsed_ms = 0.0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        while elapsed_ms < warmup_ms:
            start.record()
            # Batch ten timing-sized runs to reduce host synchronization gaps.
            # The threshold is a minimum: the final batch may exceed it.
            for _ in range(10):
                if cuda_graph:
                    graphs[i].replay()
                else:
                    for _ in range(inner):
                        fn()
            end.record()
            end.synchronize()
            elapsed_ms += start.elapsed_time(end)
    samples: list[list[float]] = [[] for _ in fns]
    for _ in range(rounds):
        for i, fn in enumerate(fns):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if cuda_graph:
                graphs[i].replay()
            else:
                for _ in range(inner):
                    fn()
            end.record()
            end.synchronize()
            samples[i].append(start.elapsed_time(end) * 1e3 / inner)
    return samples


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
        if not lens or min(lens) < 0 or max(lens) == 0 or sum(pages) > nb:
            raise ValueError(f"decode batch needs 1..{nb} distinct cache pages: {lens}")
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
        sweep = list(
            dict.fromkeys([auto] + [int(u) for u in args.units.split(",") if u])
        )
        if min(sweep) < b:
            raise ValueError(f"every --units value must cover the batch size {b}")
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
        fa3()
        reference = out_fa.clone()
        maxdiff_by_units = {}

        def invalidate(
            arm: int, outputs: tuple[torch.Tensor, ...] = (out_fa, out_fp)
        ) -> None:
            """Poison the output written by this FA3 or fused arm."""
            outputs[0 if arm == 0 else 1].fill_(float("nan"))

        def validate(
            arm: int,
            outputs: tuple[torch.Tensor, ...] = (out_fa, out_fp),
            expected: torch.Tensor = reference,
            units: list[int] = sweep,
            errors: dict[int, float] = maxdiff_by_units,
        ) -> None:
            """Check every unit count and replan in the selected timing mode."""
            actual = outputs[0 if arm == 0 else 1]
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
            if arm >= 2:
                errors[units[arm - 2]] = float(
                    (expected.float() - actual.float()).abs().max()
                )

        samples = _time_rounds(
            [fa3, replan, *fused],
            args.rounds,
            10,
            args.cuda_graph,
            invalidate,
            validate,
            args.warmup_ms,
        )
        t_fa, t_plan, *t_us = [statistics.median(s) for s in samples]
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
                "us_fused_with_plan": round(t_plan, 1),
                "us_plan_delta_estimate": round(t_plan - t_fp, 1),
                "plan_timing_scope": (
                    "difference of medians, not an isolated kernel; one plan per "
                    "engine step, reused by all layers; noise may make delta negative"
                ),
                "fused_over_fa3": round(t_fp / t_fa, 3),
                "raw_gbs_fa3": round(raw_bytes / t_fa / 1e3),
                "raw_gbs_fused": round(raw_bytes / t_fp / 1e3),
                "maxdiff": maxdiff_by_units[auto],
                "maxdiff_by_units": maxdiff_by_units,
                "validation_atol": 2e-2,
                "validation_rtol": 2e-2,
                "validation_mode": "cuda_graph" if args.cuda_graph else "eager",
                "timing": dict(
                    zip(
                        [
                            "fa3",
                            "fused_with_plan",
                            *[f"fused_units_{u}" for u in sweep],
                        ],
                        [_summary(s) for s in samples],
                        strict=True,
                    )
                ),
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
    cuda_graph: bool,
    warmup_ms: float,
) -> list[dict]:
    """Time one prefill chunk: FA3 over raw pages vs gather + FA3 workspace."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    nb = kc.shape[0]
    scale = D**-0.5
    rows = []
    for ctx, chunk in configs:
        pages = math.ceil(ctx / T)
        if not 0 < chunk <= ctx or pages > nb:
            raise ValueError(f"prefill needs 0 < chunk <= ctx <= {nb * T}")
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
        fa(k=kc, v=vc, block_table=bt_raw)
        reference = out.clone()

        def invalidate(
            arm: int, outputs: tuple[torch.Tensor, ...] = (out, wk, wv)
        ) -> None:
            """Poison each gather or attention output before validation."""
            if arm == 1:
                outputs[1].fill_(float("nan"))
                outputs[2].fill_(float("nan"))
            else:
                outputs[0].fill_(float("nan"))

        def validate(
            arm: int,
            outputs: tuple[torch.Tensor, ...] = (out, wk, wv),
            expected: torch.Tensor = reference,
            blocks: torch.Tensor = src,
        ) -> None:
            """Require bit-identical gathered KV and matching prefill output."""
            if arm == 1:
                for actual, raw in ((outputs[1], kc), (outputs[2], vc)):
                    if not torch.equal(
                        actual.view(torch.int16), raw[blocks].view(torch.int16)
                    ):
                        raise AssertionError("prefill gather differs from raw KV bits")
            else:
                torch.testing.assert_close(outputs[0], expected, atol=2e-2, rtol=2e-2)

        samples = _time_rounds(
            [
                partial(fa, k=kc, v=vc, block_table=bt_raw),
                partial(store.gather, 0, src, wk, wv),
                partial(fa, k=wk, v=wv, block_table=bt_ws),
            ],
            rounds,
            3,
            cuda_graph,
            invalidate,
            validate,
            warmup_ms,
        )
        t_raw, t_g, t_ws = [statistics.median(s) for s in samples]
        rows.append(
            {
                "kind": "prefill",
                "ctx": ctx,
                "chunk": chunk,
                "us_fa3_raw": round(t_raw, 1),
                "us_gather": round(t_g, 1),
                "us_fa3_ws": round(t_ws, 1),
                "extra_us": round(t_g + t_ws - t_raw, 1),
                "timing_scope": "separate component medians; extra_us is an estimate",
                "validation_mode": "cuda_graph" if cuda_graph else "eager",
                "gather_bitexact": True,
                "timing": dict(
                    zip(
                        ["fa3_raw", "gather", "fa3_workspace"],
                        [_summary(s) for s in samples],
                        strict=True,
                    )
                ),
            }
        )
        logger.info("%s", rows[-1])
    return rows


def _write_encode_row(
    source: HbmKvStore,
    kc: torch.Tensor,
    vc: torch.Tensor,
    batch: int,
    rounds: int,
    cuda_graph: bool,
    offset: int,
) -> dict:
    """Time valid open/closing steps, rebuilding staging before every sample.

    The dedicated store uses captured KV and the same codebooks as attention.
    Each sample claims fresh staging through prepare(offset 0), writes a
    prefix, then prepares the single-token step with those blocks live.
    Reclaiming the blocks resets arena fill and frees old overflow records;
    replaying encode repeatedly against a single close list would instead
    append duplicate records until the arena/overflow fills.
    """
    if not 0 < batch <= kc.shape[0]:
        raise ValueError("write/encode batch must fit in the captured cache")
    heads = source.kv_heads
    cache = torch.empty(batch, page_bytes(heads), dtype=torch.uint8, device="cuda")
    store = HbmKvStore(
        [cache], heads, source.tables, batch * T, batch, 2 * heads * batch
    )
    blocks = torch.arange(batch, dtype=torch.int32, device="cuda")
    bases = blocks.to(torch.int64) * T
    live = np.arange(batch, dtype=np.int32)
    no_live = np.zeros(0, dtype=np.int32)
    wk = torch.empty(batch, T, heads, D, dtype=torch.bfloat16, device="cuda")
    wv = torch.empty_like(wk)
    case = "close_block" if offset == T - 1 else "mid_block"
    seed_slots = (bases[:, None] + torch.arange(offset, device="cuda")).flatten()
    slots = bases + offset
    seed_k = kc[:batch, :offset].reshape(-1, heads, D).contiguous()
    seed_v = vc[:batch, :offset].reshape(-1, heads, D).contiguous()
    key = torch.empty(batch, heads, D, dtype=torch.bfloat16, device="cuda")
    value = torch.empty_like(key)
    expected_k = kc[:batch, : offset + 1].clone()
    expected_v = vc[:batch, : offset + 1].clone()
    expected_encoded = 2 * heads * batch if offset == T - 1 else 0

    def setup(sample_index: int) -> None:
        """Reclaim and seed blocks outside timing on the owning stream."""
        store.prepare(seed_slots, seed_slots.numel(), no_live)
        store.write(0, seed_k, seed_v, seed_slots)
        store.commit()
        store.prepare(slots, batch, live)
        # Alternate the appended token so a missing write cannot pass by
        # reading the token left in recycled staging by an earlier sample.
        key.copy_(kc[:batch, offset])
        value.copy_(vc[:batch, offset])
        if sample_index % 2:
            key.neg_()
            value.neg_()
        expected_k[:, offset].copy_(key)
        expected_v[:, offset].copy_(value)

    def validate(before: dict[str, int]) -> dict[str, int]:
        """Require real encode work and bit-identical written KV."""
        store.commit()
        after = asdict(store.stats())
        delta = {name: after[name] - old for name, old in before.items()}
        if delta["encoded"] != expected_encoded:
            raise AssertionError(f"{case}: invalid encode counter delta {delta}")
        store.gather(0, blocks, wk, wv)
        for actual, expected in ((wk, expected_k), (wv, expected_v)):
            if not torch.equal(
                actual[:, : offset + 1].view(torch.int16), expected.view(torch.int16)
            ):
                raise AssertionError(f"{case}: gathered KV differs from written bits")
        return delta

    for warmup in range(3):
        setup(warmup)
        before = asdict(store.stats())
        store.write(0, key, value, slots)
        store.encode(0, batch)
        validate(before)

    # External event nodes preserve timestamps when captured. The graph
    # contains one write and one encode only; public prepare always runs
    # outside capture/replay and no state is patched through private fields.
    start, middle, end = [
        torch.cuda.Event(enable_timing=True, external=cuda_graph) for _ in range(3)
    ]

    def measured_step() -> None:
        """Enqueue the two per-layer operations with CUDA-event boundaries."""
        start.record()
        store.write(0, key, value, slots)
        middle.record()
        store.encode(0, batch)
        end.record()

    graph = None
    if cuda_graph:
        setup(2)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            measured_step()
    write_samples, encode_samples, deltas = [], [], []
    for iteration in range(rounds):
        setup(iteration + 3)
        before = asdict(store.stats())
        if graph is None:
            measured_step()
        else:
            graph.replay()
        end.synchronize()
        write_samples.append(start.elapsed_time(middle) * 1e3)
        encode_samples.append(middle.elapsed_time(end) * 1e3)
        deltas.append(validate(before))
    row = {
        "kind": "write_encode",
        "case": case,
        "B": batch,
        "warmup_steps": 3,
        "us_write": round(statistics.median(write_samples), 1),
        "us_encode": round(statistics.median(encode_samples), 1),
        "timing_scope": (
            "one layer; prepare, prefix seed, commit and validation excluded"
        ),
        "timing": {
            "write": _summary(write_samples),
            "encode": _summary(encode_samples),
        },
        "encoded_records_per_sample": expected_encoded,
        "counter_deltas": deltas,
        "gather_bitexact": True,
        "verified_tokens_per_block": offset + 1,
    }
    logger.info("%s", row)
    # Release the final open staging through the same API used between steps.
    store.prepare(torch.empty(0, dtype=torch.int64, device="cuda"), 0, no_live)
    store.commit()
    torch.cuda.synchronize()
    return row


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
    p.add_argument(
        "--warmup-ms",
        type=float,
        default=100.0,
        help="minimum CUDA-event warmup ms per decode/prefill arm (default: 100)",
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="time captured GPU work; default: eager",
    )
    p.add_argument("--fa3-max-splits", type=int, default=32)
    p.add_argument("--units", default="", help="extra unit counts to time")
    p.add_argument("--decode", default="", help="batches like 8000/64000,...")
    p.add_argument("--lengths", help="int .npy of context lengths to sample")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--raw-records", action="store_true")
    p.add_argument("--prefill", default="16000x2048,100000x2048,100000x8192")
    args = p.parse_args()
    if args.rounds < 1 or args.batch < 1 or args.blocks < 1:
        p.error("--rounds, --batch and --blocks must be positive")
    if not math.isfinite(args.warmup_ms) or args.warmup_ms < 0:
        p.error("--warmup-ms must be finite and non-negative")
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
        "timing_mode": "cuda_graph" if args.cuda_graph else "eager",
        "timing_scope": (
            "CUDA-event GPU time; eager samples include host launch gaps; "
            "decode and prefill are single-layer microbenchmarks"
        ),
        "rounds": args.rounds,
        "warmup_ms": args.warmup_ms,
        "warmup_scope": (
            "minimum CUDA-event elapsed time per decode/prefill arm after capture "
            "and validation, checked in batches of ten timing-sized runs; "
            "excluded from samples; write_encode uses three rebuilt-state steps"
        ),
    }
    logger.info("%s", head)
    rows = _decode_rows(
        store, kc, vc, heads, _batches(args, nb * T // args.batch), args
    )
    pre = [tuple(map(int, c.split("x"))) for c in args.prefill.split(",") if c]
    rows += _prefill_rows(
        store, kc, vc, heads, pre, args.rounds, args.cuda_graph, args.warmup_ms
    )
    rows += [
        _write_encode_row(
            store, kc, vc, args.batch, args.rounds, args.cuda_graph, offset
        )
        for offset in (5, T - 1)
    ]
    Path(args.out).write_text(json.dumps({"head": head, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
