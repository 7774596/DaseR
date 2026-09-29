# SPDX-License-Identifier: Apache-2.0

"""Offline two-tier admission simulator for external KV tiers.

Replays a Mooncake trace over 512-token hash blocks through an L1 (DRAM) LRU
and an L2 (SSD) LRU. Every new block enters L1. Write-through copies every new
block to L2 at store time. Any other policy writes a block to L2 only when the
admission predicate accepts it, or when it is evicted from L1 after a hit
there. The report is prefix hit rate and bytes written to L2 over a test
window. The admission table is learned on the preceding warmup window of the
same length, so the test window never informs its own policy.

A compression ratio scales the bytes per block: both tiers hold
``1 / ratio`` times more blocks, and each L2 write costs ``ratio`` times the
raw bytes.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmarks.mooncake.benchmark import (
    MOONCAKE_BLOCK_TOKENS,
    TraceRequest,
    load_trace,
)
from benchmarks.utils.sizing import parse_size_bytes

# (request index, block index) of a block first written by that request.
BlockRef = tuple[int, int]
# (session generation, output-length bucket).
Feature = tuple[int, int]
AdmitFn = Callable[[int, int], bool]

MAX_GENERATION = 2
OUTPUT_BUCKET_TOKENS = 250
MAX_OUTPUT_BUCKET = 3


def block_features(
    requests: Sequence[TraceRequest], *, use_output: bool
) -> dict[BlockRef, Feature]:
    """Return the admission feature of every first-seen block.

    Generation is 0 for a request that matches no seen block, and otherwise
    the generation of the last matched block plus one, clipped at 2. A shared
    system prefix is therefore generation 0, a first turn's new blocks are 1,
    and later turns of the same session are 2.

    Args:
        requests: trace rows in arrival order.
        use_output: include the output-length bucket, which is only known when
            the request finishes. Store-time policies must pass False.

    Returns:
        Feature per (request index, block index) for blocks the request is the
        first to write.
    """
    seen: dict[int, int] = {}
    features: dict[BlockRef, Feature] = {}
    for i, request in enumerate(requests):
        blocks = request.hash_ids
        matched = 0
        while matched < len(blocks) and blocks[matched] in seen:
            matched += 1
        generation = (
            min(seen[blocks[matched - 1]] + 1, MAX_GENERATION) if matched else 0
        )
        bucket = (
            min(request.output_length // OUTPUT_BUCKET_TOKENS, MAX_OUTPUT_BUCKET)
            if use_output
            else 0
        )
        for j in range(len(blocks)):
            if blocks[j] not in seen:
                features[(i, j)] = (generation, bucket)
                seen[blocks[j]] = generation
    return features


def reused_blocks(requests: Sequence[TraceRequest]) -> set[BlockRef]:
    """Return block references whose hash appears again in a later request.

    Args:
        requests: trace rows in arrival order.

    Returns:
        Set of (request index, block index) with a later occurrence.
    """
    later: set[int] = set()
    reused: set[BlockRef] = set()
    for i in range(len(requests) - 1, -1, -1):
        blocks = requests[i].hash_ids
        for j, block in enumerate(blocks):
            if block in later:
                reused.add((i, j))
        later.update(blocks)
    return reused


def learn_reuse_table(
    features: dict[BlockRef, Feature],
    reused: set[BlockRef],
    end_request: int,
) -> dict[Feature, float]:
    """Estimate reuse probability per feature from requests before a cutoff.

    Args:
        features: output of :func:`block_features`.
        reused: output of :func:`reused_blocks` over the same requests.
        end_request: only blocks written by requests ``< end_request`` count.

    Returns:
        Fraction of first-seen blocks that are reused, per feature.
    """
    counts: dict[Feature, list[int]] = defaultdict(lambda: [0, 0])
    for ref, feature in features.items():
        if ref[0] < end_request:
            counts[feature][0] += 1
            counts[feature][1] += ref in reused
    return {feature: hit / total for feature, (total, hit) in counts.items()}


@dataclass(frozen=True)
class SimResult:
    """Outcome of one simulated test window.

    Attributes:
        hit_rate: prefix-hit tokens over prompt tokens in the test window.
        l2_written_blocks: blocks written to L2 during the test window.
    """

    hit_rate: float
    l2_written_blocks: int


def simulate(
    requests: Sequence[TraceRequest],
    *,
    test_start: int,
    l1_blocks: int,
    l2_blocks: int,
    admit: AdmitFn | None,
) -> SimResult:
    """Replay requests through L1 and L2 LRUs with one L2 admission policy.

    Args:
        requests: warmup rows followed by test rows, in arrival order.
        test_start: index of the first test request; earlier rows only warm
            the caches.
        l1_blocks: L1 capacity in hash blocks.
        l2_blocks: L2 capacity in hash blocks.
        admit: ``admit(i, j)`` for block ``j`` first written by request ``i``;
            None selects write-through. Blocks re-written after eviction are
            always admitted, since their reuse has already been observed.

    Returns:
        Hit rate and L2 writes over requests ``>= test_start``.
    """
    l1: OrderedDict[int, bool] = OrderedDict()
    l2: OrderedDict[int, None] = OrderedDict()
    first_seen: set[int] = set()
    hit_tokens = prompt_tokens = written = 0
    for i, request in enumerate(requests):
        counted = i >= test_start
        matched = 0
        for block in request.hash_ids:
            if block in l1:
                l1[block] = True
                l1.move_to_end(block)
            elif block in l2:
                l2.move_to_end(block)
            else:
                break
            matched += 1
        if counted:
            prompt_tokens += request.input_length
            hit_tokens += min(matched * MOONCAKE_BLOCK_TOKENS, request.input_length)
        for j, block in enumerate(request.hash_ids):
            if block in l1 or block in l2:
                continue
            new = block not in first_seen
            first_seen.add(block)
            l1[block] = False
            if admit is None or not new or admit(i, j):
                l2[block] = None
                written += counted
        while len(l1) > l1_blocks:
            block, hit_in_l1 = l1.popitem(last=False)
            if hit_in_l1 and block not in l2:
                l2[block] = None
                written += counted
        while len(l2) > l2_blocks:
            l2.popitem(last=False)
    return SimResult(hit_tokens / max(prompt_tokens, 1), written)


def main() -> None:
    """CLI entry point: one line per (L2 capacity, storage format)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument(
        "--window", type=int, required=True, help="Warmup and test length."
    )
    parser.add_argument("--kv-bytes-per-token", type=int, default=80 * 1024)
    parser.add_argument("--l1", default="64gib")
    parser.add_argument("--l2", nargs="+", default=["128gib", "256gib"])
    parser.add_argument("--compress-ratio", type=float, default=0.704)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.2, 0.3])
    args = parser.parse_args()
    requests = list(load_trace(args.trace, 2 * args.window, 0, 0))
    reused = reused_blocks(requests)
    tables = {
        use_output: learn_reuse_table(
            block_features(requests, use_output=use_output), reused, args.window
        )
        for use_output in (False, True)
    }
    features = {
        use_output: block_features(requests, use_output=use_output)
        for use_output in (False, True)
    }

    def table_policy(use_output: bool, threshold: float) -> AdmitFn:
        table, feats = tables[use_output], features[use_output]
        return lambda i, j: table.get(feats[(i, j)], 1.0) >= threshold

    policies: list[tuple[str, AdmitFn | None]] = [("wt", None)]
    for threshold in args.thresholds:
        policies.append((f"store{threshold}", table_policy(False, threshold)))
        policies.append((f"finish{threshold}", table_policy(True, threshold)))
    policies.append(("oracle", lambda i, j: (i, j) in reused))

    raw_block_bytes = MOONCAKE_BLOCK_TOKENS * args.kv_bytes_per_token
    for l2_text in args.l2:
        for name, ratio in (("raw", 1.0), ("cmp", args.compress_ratio)):
            block_bytes = raw_block_bytes * ratio
            cells = []
            for label, admit in policies:
                result = simulate(
                    requests,
                    test_start=args.window,
                    l1_blocks=int(parse_size_bytes(args.l1) / block_bytes),
                    l2_blocks=int(parse_size_bytes(l2_text) / block_bytes),
                    admit=admit,
                )
                gib = result.l2_written_blocks * block_bytes / 2**30
                cells.append(f"{label} {result.hit_rate:.3f}/{gib:.0f}")
            print(f"l1={args.l1} l2={l2_text} {name}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
