# SPDX-License-Identifier: Apache-2.0

"""Upper bound of non-prefix KV reuse on real-token agent traces.

Prefix caching only reuses a request's longest stored block-aligned prefix.
This tool measures how much of the remaining prefill (tokens a prefix cache
must compute) reappears verbatim, as a span of at least ``W`` tokens, in a
request that finished earlier. Such spans are candidates for approximate
non-prefix reuse (position re-encoding plus selective recompute); the number
is an unbounded-store upper bound, not a hit rate.

Input: gzip JSON lines, one per session, in replay order::

    {"sid": "<dataset>__<task>__<model>", "iters": [[n_msgs, gap, out], ...],
     "msgs": [{"role": ..., "content": ..., "tool_calls": ...}, ...]}

where iteration ``i`` sends the first ``n_msgs`` messages (agent traces with
cumulative message lists collapse to their longest list). Messages are
rendered as ``<|role|>\\n`` + content (+ tool-call JSON) and tokenized once.

Accounting per request ``r`` with token range ``[0, end_r)``:
    * ``prefix``: longest 128-token-aligned prefix stored by an earlier
      request of any session (vLLM/DaseR chained block hashing);
    * ``fresh``: prefill tokens after the prefix that are not the previous
      turn's assistant output (decode already produced their KV);
    * ``covered``: fresh tokens inside a window of ``W`` tokens whose exact
      content ended in an earlier request of the same scope: ``session``
      (same session), ``task`` (sessions of one task across models) or
      ``all``.
Compute-weighted shares weight a token at position ``p`` by
``1 + p / attn_break_even`` (linear layers plus attention over the context).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import json

import numpy as np

_BLOCK = 128
_MIX = np.uint64(0x9E3779B97F4A7C15)
_BASE = np.uint64(0x100000001B3)
_SCOPES = ("session", "task", "all")


@dataclass
class Trace:
    """Token stream of all sessions concatenated in replay order.

    Attributes:
        tokens: ``int64`` token ids.
        fresh: ``bool`` per token, False for assistant-output tokens.
        session: ``int64`` session index per token.
        task: ``int64`` task index per token.
        session_start: ``int64`` global start offset per session.
        req_end: ``int64`` global end offset per request (non-decreasing).
        req_session: ``int64`` session index per request.
    """

    tokens: np.ndarray
    fresh: np.ndarray
    session: np.ndarray
    task: np.ndarray
    session_start: np.ndarray
    req_end: np.ndarray
    req_session: np.ndarray


def _render(message: dict) -> str:
    """Render one chat message as the text a serving engine would see."""
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    text = f"<|{message.get('role')}|>\n{content or ''}"
    if message.get("tool_calls"):
        text += json.dumps(message["tool_calls"], sort_keys=True)
    return text


def _window_hashes(tokens: np.ndarray, window: int) -> np.ndarray:
    """Return a 64-bit content hash of every ``window``-token span.

    Args:
        tokens: ``int64`` token ids.
        window: Span length, a power of two.

    Returns:
        ``uint64`` array; entry ``i`` hashes ``tokens[i : i + window]``.
    """
    if window & (window - 1):
        raise ValueError("window must be a power of two")
    h = (tokens.astype(np.uint64) + np.uint64(1)) * _MIX
    power, width = _BASE, 1
    with np.errstate(over="ignore"):
        while width < window:
            h = h[:-width] * power + h[width:]
            power = power * power
            width *= 2
    return h


def build_trace(sessions: list[dict], encode: object) -> Trace:
    """Tokenize sessions into a :class:`Trace`.

    Args:
        sessions: Session records (see module docstring), in replay order.
        encode: Callable mapping text to a list of token ids.

    Returns:
        The concatenated trace.
    """
    cache: dict[str, list[int]] = {}
    tasks: dict[str, int] = {}
    tokens, fresh, session, task, starts = [], [], [], [], []
    req_end, req_session = [], []
    offset = 0
    for s_idx, rec in enumerate(sessions):
        t_idx = tasks.setdefault(rec["sid"].rsplit("__", 1)[0], len(tasks))
        cum, ids, roles = [0], [], []
        for message in rec["msgs"]:
            text = _render(message)
            piece = cache.get(text)
            if piece is None:
                piece = cache[text] = list(encode(text))
            ids.extend(piece)
            roles.extend([message.get("role") != "assistant"] * len(piece))
            cum.append(len(ids))
        # Assistant tokens of the first request's history were never decoded
        # here; only later turns' assistant outputs come from decode.
        n_msgs = sorted({int(it[0]) for it in rec["iters"]})
        first_end = cum[n_msgs[0]]
        is_fresh = np.asarray(roles, dtype=bool)
        is_fresh[:first_end] = True
        starts.append(offset)
        tokens.append(np.asarray(ids, dtype=np.int64))
        fresh.append(is_fresh)
        session.append(np.full(len(ids), s_idx, dtype=np.int64))
        task.append(np.full(len(ids), t_idx, dtype=np.int64))
        req_end.extend(offset + cum[n] for n in n_msgs)
        req_session.extend([s_idx] * len(n_msgs))
        offset += len(ids)
    return Trace(
        tokens=np.concatenate(tokens),
        fresh=np.concatenate(fresh),
        session=np.concatenate(session),
        task=np.concatenate(task),
        session_start=np.asarray(starts, dtype=np.int64),
        req_end=np.asarray(req_end, dtype=np.int64),
        req_session=np.asarray(req_session, dtype=np.int64),
    )


def prefix_hits(trace: Trace) -> np.ndarray:
    """Return each request's global prefix-cache hit end offset.

    Args:
        trace: Concatenated trace.

    Returns:
        ``int64`` per request: global offset where its prefill starts.
    """
    hit = trace.session_start[trace.req_session].copy()
    block_h = _window_hashes(trace.tokens, _BLOCK)
    bounds = [*trace.session_start[1:], len(trace.tokens)]
    chains, ends = [], []
    with np.errstate(over="ignore"):
        for start, stop in zip(trace.session_start, bounds, strict=True):
            acc = np.uint64(0)
            for i, h in enumerate(block_h[start : stop - _BLOCK + 1 : _BLOCK]):
                acc = (acc ^ h) * _MIX + np.uint64(i + 1)
                chains.append(acc)
                ends.append(start + _BLOCK * (i + 1))
    chain_h = np.asarray(chains, dtype=np.uint64)
    chain_end = np.asarray(ends, dtype=np.int64)
    chain_time = np.searchsorted(trace.req_end, chain_end, side="left")
    first = _first_time(chain_h, np.zeros_like(chain_h), chain_time)
    for r in range(len(trace.req_end)):
        base = trace.session_start[trace.req_session[r]]
        if r and trace.req_session[r - 1] == trace.req_session[r]:
            hit[r] = base + (trace.req_end[r - 1] - base) // _BLOCK * _BLOCK
        lo = np.searchsorted(chain_end, hit[r] + 1, side="left")
        hi = np.searchsorted(chain_end, trace.req_end[r], side="right")
        stored = np.nonzero(first[lo:hi] < r)[0]
        if len(stored):
            hit[r] = max(hit[r], chain_end[lo + stored[-1]])
    return hit


def span_coverage(trace: Trace, hit: np.ndarray, window: int, scope: str) -> np.ndarray:
    """Return a per-token mask of tokens a stored earlier span covers.

    Args:
        trace: Concatenated trace.
        hit: Per-request prefill start from :func:`prefix_hits`.
        window: Minimum reused span length in tokens (power of two).
        scope: ``session``, ``task`` or ``all``.

    Returns:
        ``bool`` per token; True when some ``window``-token span containing
        it, inside its request's prefill, ended in an earlier request of the
        same scope.
    """
    n = len(trace.tokens)
    if n < window:
        return np.zeros(n, dtype=bool)
    h = _window_hashes(trace.tokens, window)
    pos = np.arange(len(h), dtype=np.int64)
    valid = trace.session[pos] == trace.session[pos + window - 1]
    group = {
        "session": trace.session,
        "task": trace.task,
        "all": np.zeros(n, dtype=np.int64),
    }[scope][pos]
    time = np.searchsorted(trace.req_end, pos + window, side="left")
    h, group, time, pos = h[valid], group[valid], time[valid], pos[valid]
    first = _first_time(h, group.astype(np.uint64), time)
    # Token positions of the request that owns each window's last token.
    lo = np.maximum(pos, hit[time])
    stored = (first < time) & (lo < pos + window)
    diff = np.zeros(n + 1, dtype=np.int64)
    np.add.at(diff, lo[stored], 1)
    np.add.at(diff, pos[stored] + window, -1)
    return np.cumsum(diff[:-1]) > 0


def summarize(
    trace: Trace,
    hit: np.ndarray,
    covered: dict[tuple[str, int], np.ndarray],
    attn_break_even: float,
) -> dict[str, float]:
    """Aggregate token and compute-weighted shares over all requests.

    Args:
        trace: Concatenated trace.
        hit: Per-request prefill start.
        covered: ``(scope, window)`` -> per-token coverage mask.
        attn_break_even: Context length where per-token attention FLOPs equal
            linear-layer FLOPs.

    Returns:
        Totals (``ctx``, ``prefill``, ``fresh`` tokens) and ``<scope>@<W>``
        token shares plus ``<scope>@<W>w`` weighted shares of fresh tokens.
    """
    session_pos = np.arange(len(trace.tokens)) - trace.session_start[trace.session]
    weight = 1.0 + session_pos / attn_break_even
    in_prefill = np.zeros(len(trace.tokens) + 1, dtype=np.int64)
    np.add.at(in_prefill, hit, 1)
    np.add.at(in_prefill, trace.req_end, -1)
    count = np.cumsum(in_prefill[:-1])  # requests prefilling each token
    fresh_n = count * trace.fresh
    ctx = float((trace.req_end - trace.session_start[trace.req_session]).sum())
    out = {
        "ctx": ctx,
        "prefill": float(count.sum()),
        "fresh": float(fresh_n.sum()),
    }
    fresh_w = float((fresh_n * weight).sum())
    for (scope, window), mask in covered.items():
        out[f"{scope}@{window}"] = float((fresh_n * mask).sum()) / max(out["fresh"], 1)
        out[f"{scope}@{window}w"] = float((fresh_n * mask * weight).sum()) / max(
            fresh_w, 1
        )
    return out


def _first_time(h: np.ndarray, group: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Return, per element, the earliest ``time`` of its ``(h, group)`` key."""
    order = np.lexsort((time, group, h))
    hs, gs = h[order], group[order]
    head = np.ones(len(order), dtype=bool)
    head[1:] = (hs[1:] != hs[:-1]) | (gs[1:] != gs[:-1])
    run = np.cumsum(head) - 1
    first_sorted = time[order][head][run]
    first = np.empty_like(time)
    first[order] = first_sorted
    return first


def main() -> None:
    """Tokenize the trace and print span-reuse shares per dataset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", required=True, help="collapsed sessions .jsonl.gz")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer directory")
    parser.add_argument("--windows", default="128,512,2048")
    parser.add_argument("--attn-break-even", type=float, default=28672.0)
    parser.add_argument(
        "--split",
        choices=["dataset", "model"],
        default="dataset",
        help="replay each dataset, or each dataset:model, as its own deployment",
    )
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    with gzip.open(args.trace, "rt") as f:
        sessions = [json.loads(line) for line in f]
    windows = [int(w) for w in args.windows.split(",")]
    cache: dict[str, list[int]] = {}

    def encode(text: str) -> list[int]:
        if text not in cache:
            cache[text] = tok.encode(text, add_special_tokens=False)
        return cache[text]

    def part(sid: str) -> str:
        name = sid.split("__", 1)[0]
        return f"{name}:{sid.rsplit('__', 1)[1]}" if args.split == "model" else name

    datasets = sorted({part(s["sid"]) for s in sessions})
    header = ["dataset", "req", "ctxM", "prefillM", "freshM"] + [
        f"{s[0]}{w}" for s in _SCOPES for w in windows
    ]
    print(" ".join(f"{h:>9}" for h in header))
    for name in [*datasets, "all"]:
        subset = [s for s in sessions if name == "all" or part(s["sid"]) == name]
        trace = build_trace(subset, encode)
        hit = prefix_hits(trace)
        covered = {
            (scope, w): span_coverage(trace, hit, w, scope)
            for scope in _SCOPES
            for w in windows
        }
        res = summarize(trace, hit, covered, args.attn_break_even)
        cells = [name, str(len(trace.req_end))] + [
            f"{res[k] / 1e6:.1f}" for k in ("ctx", "prefill", "fresh")
        ]
        for suffix in ("", "w"):
            row = cells + [
                f"{res[f'{s}@{w}{suffix}']:.3f}" for s in _SCOPES for w in windows
            ]
            row[0] = name + ("(w)" if suffix else "")
            print(" ".join(f"{c:>9}" for c in row))


if __name__ == "__main__":
    main()
