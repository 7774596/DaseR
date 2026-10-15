# SPDX-License-Identifier: Apache-2.0
"""Time a single long-prefix restore per (length, prefix fraction, tier) cell.

For every cell the probe sends, one request at a time against an otherwise
idle server:

1. ``write``: the prefix ``P = fraction * total`` tokens (one output token),
   so the KV connector stores it;
2. for the ``ssd`` tier only, unrelated ``flush`` requests totalling
   ``--flush-tokens`` so the prefix leaves the host L1 tier and must be read
   back from SSD (L2 must be large enough to keep it);
3. a GPU prefix-cache reset, then ``external``: the full ``total``-token
   prompt, whose first ``P`` tokens only the connector can serve;
4. ``gpu``: the same prefix with a different suffix and no reset, so the GPU
   prefix cache serves the prefix and only the suffix is computed. It is the
   compute-only control ``C`` of the cell; ``external - gpu`` is the exposed
   restore time. For a full-prompt prefix both passes send the same prompt,
   so their greedy outputs must match: a mismatch means the restored KV was
   wrong when the forward read it.

Without a connector ``external`` is a full recompute. Every prompt starts
with a random header unique to its cell and repeat, so cells never share
cached blocks. Each pass records its TTFT, decode inter-token gaps and the
deltas of the vLLM (and optionally DaseR) counters, so a pass that hit the
wrong cache is visible.

Requires ``VLLM_SERVER_DEV_MODE=1`` on the server (``/reset_prefix_cache``).
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from typing import Any

import httpx
import numpy as np

from benchmarks.utils.datasets import BenchmarkSample
from benchmarks.utils.loadgen import RequestResult, vllm_completion_stream
from benchmarks.utils.metrics import compute_metric_delta, extract_prometheus_counters

TIERS = ("l1", "ssd")
HEADER_TOKENS = 16

#: vLLM counters recorded per pass.
VLLM_COUNTERS = {
    "gpu_hit_tokens": "vllm:prefix_cache_hits_total",
    "external_hit_tokens": "vllm:external_prefix_cache_hits_total",
}

#: DaseR counter-name prefixes recorded per pass (labels kept).
DASER_PREFIXES = (
    "daser_l1_hits_total",
    "daser_l1_misses_total",
    "daser_l2_read_bytes_total",
    "daser_l2_reads_total",
    "daser_load_phase_seconds_total",
    "daser_readahead_bytes_total",
)


@dataclass(frozen=True)
class Cell:
    """One measured configuration.

    Attributes:
        total_tokens: Prompt length of the ``external`` pass.
        prefix_tokens: Tokens stored by ``write`` and restored by ``external``.
        tier: ``l1`` or ``ssd``.
        repeat: Repeat index within the cell.
    """

    total_tokens: int
    prefix_tokens: int
    tier: str
    repeat: int


def plan_cells(
    totals: Sequence[int],
    fractions: Sequence[float],
    tiers: Sequence[str],
    repeats: int,
    block_tokens: int,
) -> list[Cell]:
    """Expand the sweep into cells with block-aligned prefixes.

    Args:
        totals: Prompt lengths in tokens.
        fractions: Cached prefix fractions in ``(0, 1]``.
        tiers: Host tiers to measure, each in :data:`TIERS`.
        repeats: Repeats per configuration (the summary reports medians).
        block_tokens: KV block size; prefixes are rounded down to it, and a
            full-prompt prefix stays the whole prompt.

    Returns:
        Cells in sweep order, repeats innermost.

    Raises:
        ValueError: On an unknown tier, a fraction outside ``(0, 1]`` or a
            prefix shorter than one block.

    Thread-safety:
        Pure function.
    """
    cells = []
    for total in totals:
        for fraction in fractions:
            if not 0 < fraction <= 1:
                raise ValueError(f"prefix fraction {fraction} not in (0, 1]")
            prefix = total if fraction == 1 else int(total * fraction)
            prefix -= prefix % block_tokens if prefix < total else 0
            if prefix < block_tokens:
                raise ValueError(f"prefix of {total}x{fraction} below one block")
            for tier in tiers:
                if tier not in TIERS:
                    raise ValueError(f"unknown tier {tier}")
                cells.extend(Cell(total, prefix, tier, r) for r in range(repeats))
    return cells


class PromptSource:
    """Build cell prompts from a real token stream plus unique headers.

    Args:
        tokens: 1-D real token IDs; prompt bodies are slices of it.
        seed: Seed of the header generator.
        vocab: Exclusive upper bound of header token IDs.

    Thread-safety:
        Not thread-safe (owns a random generator).
    """

    def __init__(self, tokens: np.ndarray, seed: int, vocab: int = 100_000) -> None:
        self._tokens = tokens
        self._rng = np.random.default_rng(seed)
        self._vocab = vocab
        self._cursor = 0

    def header(self) -> list[int]:
        """Return a fresh random header that no other prompt shares."""
        return self._rng.integers(1000, self._vocab, HEADER_TOKENS).tolist()

    def body(self, length: int) -> list[int]:
        """Return the next ``length`` stream tokens, wrapping at the end.

        Args:
            length: Tokens to return; must not exceed the stream length.

        Returns:
            Real token IDs.
        """
        if length > len(self._tokens):
            raise ValueError("token stream is shorter than one prompt body")
        if self._cursor + length > len(self._tokens):
            self._cursor = 0
        start, self._cursor = self._cursor, self._cursor + length
        return self._tokens[start : self._cursor].tolist()


def cell_prompts(cell: Cell, source: PromptSource) -> dict[str, list[int]]:
    """Return the write, external and GPU-control prompts of one cell.

    All three share the header and the prefix; ``external`` continues it with
    the stream, ``gpu`` with a different suffix of the same length (for a
    full-prompt prefix both equal ``write``).

    Args:
        cell: Cell to build.
        source: Prompt token source.

    Returns:
        Mapping from pass name to prompt token IDs, each ``total_tokens`` long
        except ``write`` (``prefix_tokens``).
    """
    head = source.header()
    body = source.body(cell.total_tokens - HEADER_TOKENS)
    full = head + body
    prefix = full[: cell.prefix_tokens]
    other = source.body(cell.total_tokens - cell.prefix_tokens)
    return {
        "write": prefix,
        "external": full,
        "gpu": prefix + other if cell.prefix_tokens < cell.total_tokens else full,
    }


def flush_lengths(flush_tokens: int, max_prompt: int) -> list[int]:
    """Split the flush volume into prompts no longer than ``max_prompt``.

    Args:
        flush_tokens: Total flush tokens.
        max_prompt: Longest admissible prompt.

    Returns:
        Prompt lengths summing to ``flush_tokens`` (empty for zero).
    """
    if flush_tokens < 0 or max_prompt <= HEADER_TOKENS:
        raise ValueError("flush volume and prompt limit must be positive")
    lengths = [max_prompt] * (flush_tokens // max_prompt)
    if flush_tokens % max_prompt:
        lengths.append(max(flush_tokens % max_prompt, HEADER_TOKENS + 1))
    return lengths


def summarise(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return per-configuration medians of the external and GPU passes.

    Args:
        records: Measured pass records (``external`` and ``gpu``).

    Returns:
        One row per (total, prefix, tier) with median TTFTs, the exposed
        restore time ``external - gpu`` and the failure / off-state counts.
        A pass is off-state when the cache it must not use served over half
        of its prefix (GPU hits for ``external``, external hits for ``gpu``).
        Full-prompt prefixes also count the repeats whose two greedy outputs
        differ (``output_mismatch``).

    Thread-safety:
        Pure function.
    """
    keys = sorted({(r["total_tokens"], r["prefix_tokens"], r["tier"]) for r in records})
    rows = []
    for total, prefix, tier in keys:
        cell = [
            r
            for r in records
            if (r["total_tokens"], r["prefix_tokens"], r["tier"])
            == (total, prefix, tier)
        ]
        out: dict[str, Any] = {
            "total_tokens": total,
            "prefix_tokens": prefix,
            "tier": tier,
        }
        for name, stray in (
            ("external", "gpu_hit_tokens"),
            ("gpu", "external_hit_tokens"),
        ):
            ok = [r for r in cell if r["pass"] == name and r["error"] is None]
            ttfts = [r["ttft_ms"] for r in ok]
            out[f"{name}_ttft_ms_p50"] = statistics.median(ttfts) if ttfts else None
            out[f"{name}_failed"] = sum(
                1 for r in cell if r["pass"] == name and r["error"] is not None
            )
            out[f"{name}_off_state"] = sum(1 for r in ok if 2 * r[stray] > prefix)
            gaps = sorted(g for r in ok for g in r["itl_ms"])
            out[f"{name}_itl_ms_p99"] = (
                gaps[min(len(gaps) - 1, int(0.99 * len(gaps)))] if gaps else None
            )
        if (
            out["external_ttft_ms_p50"] is not None
            and out["gpu_ttft_ms_p50"] is not None
        ):
            out["restore_ms_p50"] = out["external_ttft_ms_p50"] - out["gpu_ttft_ms_p50"]
        if prefix == total:
            texts: dict[int, dict[str, str]] = {}
            for r in cell:
                if r["error"] is None:
                    texts.setdefault(r["repeat"], {})[r["pass"]] = r["text"]
            out["output_mismatch"] = sum(
                1 for t in texts.values() if len(t) == 2 and len(set(t.values())) > 1
            )
        rows.append(out)
    return rows


def compare_arms(
    summaries: dict[str, Sequence[dict[str, Any]]],
    raw: str,
    cmp: str,
    raw_lw: str,
    cmp_lw: str,
) -> list[dict[str, Any]]:
    """Join four arms' summaries into the layer-wise pipelining gates.

    Per cell, ``C`` is the compute-only control (median ``gpu`` TTFT of the
    non-layer-wise arms) and ``IO_x = external_x - C``. A perfect layer
    pipeline hides ``min(IO, C)``, so ``(IO + C) / max(IO, C)`` bounds the
    layer-wise speedup of each arm.

    Args:
        summaries: Arm name to :func:`summarise` rows.
        raw: Raw-record arm without layer-wise loads.
        cmp: Compressed arm without layer-wise loads.
        raw_lw: Raw-record arm with layer-wise loads.
        cmp_lw: Compressed arm with layer-wise loads.

    Returns:
        One row per cell present in all four arms with ``io_*_ms``, ``c_ms``,
        the bounds ``bound_raw`` / ``bound_cmp``, measured speedups over
        ``raw`` (``s_cmp``, ``s_raw_lw``, ``s_cmp_lw``), ``lw_gain_cmp``
        (``cmp / cmp_lw``) and ``super_additive`` (``s_cmp_lw`` above the
        product of the two single-factor speedups).

    Thread-safety:
        Pure function.
    """

    def by_cell(arm: str) -> dict[tuple[int, int, str], dict[str, Any]]:
        return {
            (r["total_tokens"], r["prefix_tokens"], r["tier"]): r
            for r in summaries[arm]
        }

    arms = {name: by_cell(name) for name in (raw, cmp, raw_lw, cmp_lw)}
    rows = []
    for key in sorted(set.intersection(*(set(a) for a in arms.values()))):
        ext = {name: arms[name][key]["external_ttft_ms_p50"] for name in arms}
        gpus = [arms[name][key]["gpu_ttft_ms_p50"] for name in (raw, cmp)]
        if None in ext.values() or None in gpus:
            continue
        c = statistics.median(gpus)
        io_raw, io_cmp = max(ext[raw] - c, 0.0), max(ext[cmp] - c, 0.0)
        s_cmp, s_raw_lw = ext[raw] / ext[cmp], ext[raw] / ext[raw_lw]
        s_cmp_lw = ext[raw] / ext[cmp_lw]
        rows.append(
            {
                "total_tokens": key[0],
                "prefix_tokens": key[1],
                "tier": key[2],
                "c_ms": c,
                "io_raw_ms": io_raw,
                "io_cmp_ms": io_cmp,
                "bound_raw": (io_raw + c) / max(io_raw, c),
                "bound_cmp": (io_cmp + c) / max(io_cmp, c),
                "s_cmp": s_cmp,
                "s_raw_lw": s_raw_lw,
                "s_cmp_lw": s_cmp_lw,
                "lw_gain_cmp": ext[cmp] / ext[cmp_lw],
                "super_additive": s_cmp_lw > s_cmp * s_raw_lw,
            }
        )
    return rows


async def run_probe(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Measure every cell against one server and write the artifacts.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The summary rows also written to ``<output-dir>/summary.json``.

    Thread-safety:
        Sends one request at a time; concurrent traffic would pollute the
        counter deltas and keep blocks that defeat the prefix-cache reset.
    """
    cells = plan_cells(
        args.total_tokens,
        args.prefix_fractions,
        args.tiers,
        args.repeats,
        args.block_tokens,
    )
    longest = max(c.total_tokens for c in cells)
    if longest + args.max_tokens > args.max_model_len:
        raise ValueError("longest probe prompt exceeds --max-model-len")
    source = PromptSource(np.load(args.session_tokens, mmap_mode="r"), args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:

        async def send(
            prompt: list[int], request_id: str, max_tokens: int
        ) -> RequestResult:
            sample = BenchmarkSample(
                sample_id=0,
                dataset="prefix-restore-probe",
                context="",
                question="",
                answers=[],
            )
            params = {
                "model": args.served_model_name,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
                "request_id": request_id,
            }
            return await vllm_completion_stream(
                client,
                args.server_url,
                sample,
                prompt,
                params,
                asyncio.Semaphore(1),
                args.timeout,
            )

        async def snapshot() -> dict[str, float]:
            counters = {}
            for url in filter(None, (args.server_url, args.daser_url)):
                response = await client.get(f"{url}/metrics")
                response.raise_for_status()
                counters.update(extract_prometheus_counters(response.text))
            return counters

        async def reset() -> None:
            response = await client.post(f"{args.server_url}/reset_prefix_cache")
            response.raise_for_status()

        with (args.output_dir / "probe.jsonl").open("w") as out:
            for index, cell in enumerate(cells):
                prompts = cell_prompts(cell, source)
                tag = f"pr{index}-{cell.total_tokens}-{cell.prefix_tokens}-{cell.tier}"
                await send(prompts["write"], f"{tag}-write", 1)
                await asyncio.sleep(args.settle_s)
                if cell.tier == "ssd":
                    for n, length in enumerate(
                        flush_lengths(
                            args.flush_tokens,
                            min(args.flush_prompt_tokens, args.max_model_len - 1),
                        )
                    ):
                        await send(
                            source.header() + source.body(length - HEADER_TOKENS),
                            f"{tag}-flush{n}",
                            1,
                        )
                    await asyncio.sleep(args.settle_s)
                await reset()
                for name in ("external", "gpu"):
                    before = await snapshot()
                    result = await send(prompts[name], f"{tag}-{name}", args.max_tokens)
                    await asyncio.sleep(args.settle_s)
                    delta = compute_metric_delta(before, await snapshot())
                    row = {
                        "total_tokens": cell.total_tokens,
                        "prefix_tokens": cell.prefix_tokens,
                        "tier": cell.tier,
                        "repeat": cell.repeat,
                        "pass": name,
                        "error": result.error,
                        "ttft_ms": result.ttft_ms,
                        "latency_ms": result.latency_ms,
                        "completion_tokens": result.completion_tokens,
                        "text": result.generated_text,
                        "cached_tokens": result.cached_tokens,
                        "itl_ms": list(result.token_gaps_ms),
                        **{k: delta.get(m, 0.0) for k, m in VLLM_COUNTERS.items()},
                        "daser": {
                            k: v
                            for k, v in delta.items()
                            if k.startswith(DASER_PREFIXES) and v
                        },
                    }
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    records.append(row)
    summary = summarise(records)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments without the program name; ``None`` reads ``sys.argv``.

    Returns:
        Parsed namespace.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--session-tokens", type=Path, required=True)
    p.add_argument("--server-url", required=True)
    p.add_argument("--daser-url", default=None)
    p.add_argument("--served-model-name", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--total-tokens", type=int, nargs="+", default=[32768])
    p.add_argument("--prefix-fractions", type=float, nargs="+", default=[0.5, 0.9, 1.0])
    p.add_argument("--tiers", nargs="+", choices=TIERS, default=list(TIERS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--block-tokens", type=int, default=128)
    p.add_argument(
        "--flush-tokens",
        type=int,
        default=0,
        help="tokens of unrelated prompts sent before an ssd-tier read",
    )
    p.add_argument(
        "--flush-prompt-tokens",
        type=int,
        default=32768,
        help="longest flush prompt (shorter prompts prefill faster)",
    )
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--settle-s", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=1800.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the probe and print its summary as JSON.

    Args:
        argv: Arguments without the program name; ``None`` reads ``sys.argv``.
    """
    print(json.dumps(asyncio.run(run_probe(parse_args(argv))), indent=2))


if __name__ == "__main__":
    main()
