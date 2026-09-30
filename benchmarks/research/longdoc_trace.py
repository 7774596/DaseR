# SPDX-License-Identifier: Apache-2.0

"""Build real-text long-document QA workloads in Mooncake trace format.

Two subcommands:

``corpus``
    Tokenize plain-text files or ShareGPT JSON into a flat ``.npy`` array of
    token IDs for ``benchmarks.mooncake.benchmark --corpus-tokens``.

``trace``
    Emit a Mooncake-format ``trace.jsonl`` for repeated questions over N long
    documents, plus ``stages.json`` naming the row range of every stage. A
    warmup stage prefills each document once at a fixed gap; each following
    stage sends Poisson arrivals at a fixed rate, every request asking a fresh
    question about a uniformly chosen document. Document ``d`` owns the
    consecutive hash IDs ``d * stride + k``, so with ``--corpus-tokens`` its
    blocks read as consecutive corpus text; question blocks use their own
    hash range and are never shared.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any

from benchmarks.mooncake.benchmark import MOONCAKE_BLOCK_TOKENS

# Hash-ID spacing between documents; larger than any document's block count.
_DOC_STRIDE = 1 << 20


def parse_stage(text: str) -> tuple[float, float]:
    """Parse ``<qps>:<seconds>`` into a positive rate and duration.

    Args:
        text: Stage specification, for example ``"0.5:300"``.

    Returns:
        ``(qps, seconds)``.

    Raises:
        argparse.ArgumentTypeError: On malformed or non-positive values.
    """
    try:
        qps, seconds = (float(part) for part in text.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"stage must be <qps>:<seconds>: {text}"
        ) from exc
    if qps <= 0 or seconds <= 0:
        raise argparse.ArgumentTypeError(f"stage values must be positive: {text}")
    return qps, seconds


def build_trace(
    *,
    documents: int,
    document_tokens: int,
    question_tokens: int,
    output_tokens: int,
    warmup_gap_s: float,
    stages: list[tuple[float, float]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return trace rows and stage ranges for a long-document QA workload.

    Args:
        documents: Number of distinct documents.
        document_tokens: Tokens per document; must be a multiple of 512 so
            the question never shares a block with the document.
        question_tokens: Tokens per question suffix.
        output_tokens: Decode tokens per request.
        warmup_gap_s: Gap between warmup requests (one per document).
        stages: ``(qps, seconds)`` for each measured stage, in order.
        seed: Seed for arrivals and document choice.

    Returns:
        ``(rows, stage_ranges)``; rows are Mooncake trace objects in
        non-decreasing timestamp order, and each stage range has ``name``,
        ``start`` and ``end`` row indices (end exclusive) plus ``qps``.
    """
    if document_tokens <= 0 or document_tokens % MOONCAKE_BLOCK_TOKENS:
        raise ValueError("document_tokens must be a positive multiple of 512")
    if documents <= 0 or question_tokens <= 0 or output_tokens <= 0:
        raise ValueError("documents, question and output tokens must be positive")
    rng = random.Random(seed)
    doc_blocks = document_tokens // MOONCAKE_BLOCK_TOKENS
    question_blocks = math.ceil(question_tokens / MOONCAKE_BLOCK_TOKENS)
    question_base = documents * _DOC_STRIDE
    rows: list[dict[str, Any]] = []

    def add(timestamp_s: float, doc: int) -> None:
        first_question = question_base + len(rows) * question_blocks
        rows.append(
            {
                "timestamp": round(timestamp_s * 1000.0, 3),
                "input_length": document_tokens + question_tokens,
                "output_length": output_tokens,
                "hash_ids": [doc * _DOC_STRIDE + k for k in range(doc_blocks)]
                + list(range(first_question, first_question + question_blocks)),
            }
        )

    ranges: list[dict[str, Any]] = []
    for doc in range(documents):
        add(doc * warmup_gap_s, doc)
    ranges.append({"name": "warmup", "start": 0, "end": len(rows), "qps": None})
    clock = documents * warmup_gap_s
    for qps, seconds in stages:
        start, stage_end = len(rows), clock + seconds
        clock += rng.expovariate(qps)
        while clock < stage_end:
            add(clock, rng.randrange(documents))
            clock += rng.expovariate(qps)
        clock = stage_end
        ranges.append(
            {"name": f"qps{qps:g}", "start": start, "end": len(rows), "qps": qps}
        )
    return rows, ranges


def build_corpus(paths: list[Path], tokenizer_path: str, max_tokens: int) -> list[int]:
    """Tokenize text or ShareGPT JSON files into one flat token list.

    Args:
        paths: ``.txt`` files (read as-is) or ShareGPT ``.json`` files (every
            human/gpt turn, rendered with the tokenizer's chat template).
        tokenizer_path: Tokenizer name or path, loaded with remote code.
        max_tokens: Stop after this many tokens (0 = no limit).

    Returns:
        Token IDs without special tokens between documents.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokens: list[int] = []
    for path in paths:
        if path.suffix == ".json":
            for row in json.loads(path.read_text()):
                messages = [
                    {
                        "role": "user" if turn["from"] == "human" else "assistant",
                        "content": turn["value"],
                    }
                    for turn in row.get("conversations", [])
                    if turn.get("from") in ("human", "gpt")
                ]
                if messages:
                    rendered = tokenizer.apply_chat_template(messages, tokenize=True)
                    if not isinstance(rendered, list):
                        rendered = rendered["input_ids"]
                    tokens.extend(rendered)
                if max_tokens and len(tokens) >= max_tokens:
                    break
        else:
            tokens.extend(
                tokenizer(path.read_text(), add_special_tokens=False)["input_ids"]
            )
        if max_tokens and len(tokens) >= max_tokens:
            return tokens[:max_tokens]
    return tokens


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    corpus = sub.add_parser("corpus", help="tokenize text into a .npy corpus")
    corpus.add_argument("--tokenizer", required=True)
    corpus.add_argument("--input", type=Path, nargs="+", required=True)
    corpus.add_argument("--output", type=Path, required=True)
    corpus.add_argument("--max-tokens", type=int, default=0)
    trace = sub.add_parser("trace", help="emit trace.jsonl and stages.json")
    trace.add_argument("--output-dir", type=Path, required=True)
    trace.add_argument("--documents", type=int, required=True)
    trace.add_argument("--document-tokens", type=int, required=True)
    trace.add_argument("--question-tokens", type=int, default=256)
    trace.add_argument("--output-tokens", type=int, default=32)
    trace.add_argument("--warmup-gap", type=float, default=10.0)
    trace.add_argument(
        "--stage",
        type=parse_stage,
        action="append",
        required=True,
        help="<qps>:<seconds>; repeat for successive stages",
    )
    trace.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "corpus":
        import numpy as np

        tokens = build_corpus(args.input, args.tokenizer, args.max_tokens)
        np.save(args.output, np.asarray(tokens, dtype=np.int32))
        print(f"wrote {len(tokens)} tokens to {args.output}")
        return
    rows, ranges = build_trace(
        documents=args.documents,
        document_tokens=args.document_tokens,
        question_tokens=args.question_tokens,
        output_tokens=args.output_tokens,
        warmup_gap_s=args.warmup_gap,
        stages=args.stage,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (args.output_dir / "stages.json").write_text(json.dumps(ranges, indent=2) + "\n")
    for stage in ranges:
        print(f"{stage['name']}: rows {stage['start']}..{stage['end']}")


if __name__ == "__main__":
    main()
