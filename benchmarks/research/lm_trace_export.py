# SPDX-License-Identifier: Apache-2.0
"""Export real-text agent sessions as a gap-paced replay trace.

Input is the collapsed lmcache-agentic file (one JSON line per OpenHands
session: ``sid``, cumulative ``msgs`` and per-iteration ``iters`` rows
``[n_messages, pre_gap_s, output_length]``). Each session is rendered message
by message with the target tokenizer into one token stream, so iteration i's
prompt, the first ``n_messages`` messages plus the assistant header, is a
prefix of that stream. All sampled streams are written back to back to
``--tokens`` (``.npy``); each row carries ``token_offset``, its session's
start, for :class:`benchmarks.mooncake.benchmark.SessionTextPromptBuilder`.

Rows keep Mooncake fields: ``hash_ids`` name 512-token blocks by
``(session, block index, block length)`` (prompts of a session are prefixes
of one stream, so equal blocks get equal IDs), ``gap_ms`` is the recorded
idle time before the iteration (capped at ``--gap-cap-s``) and ``timestamp``
merges sessions started at Poisson offsets with about ``--concurrency``
overlapping. Recorded gaps are short (the agent waits on its own decode, not
on tools), so a session's span is estimated as its gaps plus its output at
``--decode-step-s`` seconds per token; without that, every session would
start at once. Unlike hash-only traces the prompts are real code and tool
output, so decode and prompt-lookup speculation see real repetition.
"""

import argparse
import gzip
import json
import math
import random
from typing import Any

import numpy as np

BLOCK_TOKENS = 512


def _render(message: dict[str, Any]) -> str:
    """Render one chat message as plain role-tagged text.

    Args:
        message: OpenAI-style message (``role``, ``content``, optional
            ``tool_calls``).

    Returns:
        ``<|role|>\\n`` followed by the content and any tool calls as JSON.
    """
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    text = f"<|{message.get('role')}|>\n{content or ''}"
    if message.get("tool_calls"):
        text += json.dumps(message["tool_calls"], sort_keys=True)
    return text


def session_rows(
    session: dict[str, Any],
    encode: Any,
    header: list[int],
    max_ctx: int,
    max_output: int,
    gap_cap_s: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Tokenize one session and build its per-iteration rows.

    Args:
        session: collapsed session record (``msgs``, ``iters``).
        encode: ``str -> list[int]`` tokenizer call without special tokens.
        header: token IDs of the assistant header ``<|assistant|>\\n``.
        max_ctx: drop iterations whose prompt plus output exceeds this.
        max_output: cap on ``output_length`` (0 keeps the recorded value).
        gap_cap_s: upper bound on a recorded idle gap in seconds.

    Returns:
        ``(stream, rows)``: the session token stream, and rows with
        ``input_length`` (the prompt's end in the stream), ``output_length``
        and ``gap_ms`` in iteration order.
    """
    stream: list[int] = []
    offsets = []
    for message in session["msgs"]:
        offsets.append(len(stream))
        stream.extend(encode(_render(message)))
    offsets.append(len(stream))
    rows = []
    for n_messages, pre_gap, output_length in session["iters"]:
        end = offsets[min(n_messages, len(session["msgs"]))]
        if stream[end : end + len(header)] == header:
            end += len(header)
        out = max(1, min(int(output_length), max_output or int(output_length)))
        if end == 0 or end + out > max_ctx:
            continue
        rows.append(
            {
                "input_length": end,
                "output_length": out,
                "gap_ms": round(min(max(float(pre_gap), 0.0), gap_cap_s) * 1000),
            }
        )
    return stream, rows


def session_span_s(rows: list[dict[str, Any]], decode_step_s: float) -> float:
    """Estimate how long one session runs when served alone.

    Args:
        rows: the session's rows (``gap_ms``, ``output_length``).
        decode_step_s: estimated seconds per generated token.

    Returns:
        Idle gaps plus decode time in seconds.
    """
    return sum(r["gap_ms"] for r in rows) / 1000 + decode_step_s * sum(
        r["output_length"] for r in rows
    )


def export(
    path: str,
    tokenizer: str,
    sessions: int,
    concurrency: float,
    max_ctx: int,
    max_output: int,
    gap_cap_s: float,
    seed: int,
    min_requests: int = 2,
    decode_step_s: float = 0.03,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build the token stream and merged replay rows.

    Args:
        path: collapsed lmcache-agentic ``.jsonl.gz``.
        tokenizer: tokenizer path of the served model.
        sessions: number of sessions to sample.
        concurrency: mean number of overlapping sessions.
        max_ctx: drop iterations whose prompt plus output exceeds this.
        max_output: cap on ``output_length`` (0 keeps the recorded value).
        gap_cap_s: upper bound on a recorded idle gap in seconds.
        seed: RNG seed for sampling and start offsets.
        min_requests: only keep sessions with at least this many rows.
        decode_step_s: estimated seconds per generated token, used only to
            space session starts (see :func:`session_span_s`).

    Returns:
        ``(tokens, rows)``: all session streams back to back (int32) and
        Mooncake rows with ``session_id``, ``gap_ms`` and ``token_offset`` in
        arrival order.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    header = encode("<|assistant|>\n")
    with gzip.open(path, "rt") as f:
        records = [json.loads(line) for line in f]
    rng = random.Random(seed)
    rng.shuffle(records)
    chains = []
    for record in records:
        if len(chains) == sessions:
            break
        stream, rows = session_rows(
            record, encode, header, max_ctx, max_output, gap_cap_s
        )
        if len(rows) >= min_requests:
            chains.append((stream, rows))
    spans = [session_span_s(rows, decode_step_s) for _, rows in chains]
    rate = concurrency / max(sum(spans) / max(len(spans), 1), 1e-9)
    tokens: list[int] = []
    merged = []
    t0 = 0.0
    for sid, (stream, rows) in enumerate(chains):
        t0 += rng.expovariate(rate)
        offset = len(tokens)
        tokens.extend(stream)
        t = t0
        for i, row in enumerate(rows):
            t += row["gap_ms"] / 1000 if i else 0.0
            n = row["input_length"]
            hash_ids = [
                (sid, b, min(BLOCK_TOKENS, n - b * BLOCK_TOKENS))
                for b in range(math.ceil(n / BLOCK_TOKENS))
            ]
            merged.append((t, sid, i, {**row, "hash_ids": hash_ids}, offset))
    merged.sort(key=lambda item: (item[0], item[1], item[2]))
    table: dict = {}
    out = []
    for t, sid, i, row, offset in merged:
        out.append(
            {
                "timestamp": round(t * 1000),
                "input_length": row["input_length"],
                "output_length": row["output_length"],
                "hash_ids": [
                    table.setdefault(key, len(table)) for key in row["hash_ids"]
                ],
                "session_id": sid,
                "gap_ms": row["gap_ms"] if i else 0,
                "token_offset": offset,
            }
        )
    return np.asarray(tokens, dtype=np.int32), out


def main() -> None:
    """CLI entry point: write the trace and token stream, print a summary."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions-file", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tokens", required=True, help="Output .npy token stream.")
    p.add_argument("--sessions", type=int, default=64)
    p.add_argument("--concurrency", type=float, default=8.0)
    p.add_argument("--max-ctx", type=int, default=131072)
    p.add_argument("--max-output", type=int, default=0)
    p.add_argument("--gap-cap-s", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--min-requests", type=int, default=2)
    p.add_argument("--decode-step-s", type=float, default=0.03)
    a = p.parse_args()
    tokens, rows = export(
        a.sessions_file,
        a.tokenizer,
        a.sessions,
        a.concurrency,
        a.max_ctx,
        a.max_output,
        a.gap_cap_s,
        a.seed,
        a.min_requests,
        a.decode_step_s,
    )
    np.save(a.tokens, tokens)
    with open(a.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    ctx = sorted(r["input_length"] for r in rows)
    print(
        f"requests={len(rows)} sessions={len({r['session_id'] for r in rows})} "
        f"tokens={len(tokens) / 1e6:.2f}M ctx_p50={ctx[len(ctx) // 2]} "
        f"ctx_p90={ctx[len(ctx) * 9 // 10]} "
        f"out_sum={sum(r['output_length'] for r in rows)}"
    )


if __name__ == "__main__":
    main()
