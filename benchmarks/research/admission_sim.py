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

With ``--gpu-kv`` the first window is replayed cold through a GPU prefix cache
in front of L1 and L2, with the server's finish-time admission semantics
(denied-block FIFO budget, held write-back, ghost re-stores, optional packed
arena FIFO). It also reports how well per-block signals separate reused from
dead blocks (ROC AUC), for any reuse and for reuse after GPU eviction.
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


def gpu_reuse_labels(
    requests: Sequence[TraceRequest], gpu_blocks: int
) -> tuple[set[BlockRef], set[BlockRef], list[float], list[float]]:
    """Split block reuse into GPU-local and external reuse.

    The GPU prefix cache is an LRU of ``gpu_blocks`` hash blocks. A later
    prefix match on a block counts as local reuse when the block is still on
    the GPU and as external reuse when it was evicted, i.e. the reuse only an
    external tier with unbounded capacity can serve.

    Args:
        requests: trace rows in arrival order.
        gpu_blocks: GPU prefix-cache capacity in hash blocks.

    Returns:
        ``(local, external, local_gaps_s, external_gaps_s)``: first-writer
        references with at least one local or external reuse, and the time
        since the block's previous access for every local or external reuse.
    """
    writer: dict[int, BlockRef] = {}
    last_access: dict[int, float] = {}
    gpu: OrderedDict[int, None] = OrderedDict()
    local: set[BlockRef] = set()
    external: set[BlockRef] = set()
    local_gaps: list[float] = []
    external_gaps: list[float] = []
    for i, request in enumerate(requests):
        now = request.timestamp_ms / 1000
        on_gpu = True
        for block in request.hash_ids:
            if block not in writer:
                break
            on_gpu = on_gpu and block in gpu
            gap = now - last_access[block]
            if on_gpu:
                local.add(writer[block])
                local_gaps.append(gap)
            else:
                external.add(writer[block])
                external_gaps.append(gap)
        for j, block in enumerate(request.hash_ids):
            writer.setdefault(block, (i, j))
            last_access[block] = now
            gpu[block] = None
            gpu.move_to_end(block)
        while len(gpu) > gpu_blocks:
            gpu.popitem(last=False)
    return local, external, local_gaps, external_gaps


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Return the ROC AUC of scores for binary labels, averaging ties.

    Args:
        scores: one score per sample; higher should mean positive.
        labels: one label per sample.

    Returns:
        Probability that a random positive outranks a random negative; 0.5
        when either class is empty.
    """
    order = sorted(range(len(scores)), key=lambda k: scores[k])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        for k in order[start : end + 1]:
            ranks[k] = (start + end) / 2 + 1
        start = end + 1
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return 0.5
    rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


@dataclass(frozen=True)
class TierPolicy:
    """L2 admission policy of the three-tier simulator.

    Attributes:
        min_output: output length at or above which a request's new blocks are
            written to L2 at once; 0 is write-through.
        denied_budget_blocks: denied blocks kept in L1 under a FIFO budget;
            None bounds them by L1 capacity only (plain LRU).
        admit_ghosts: write a re-stored block to L2 when it was denied and
            evicted before.
        persist_evicted_denied: write a denied block back to L2 when L1 LRU
            evicts it before the budget does (the server's held write-back).
        fifo_arena: model the server's packed arena. Every stored block takes
            arena space in store order whether or not it is written; a full
            arena removes its oldest block from the index, L1 included, and
            hits do not refresh arena order.
        exclusive: keep admitted blocks in L1 only and write them to L2 when
            L1 evicts them (the server's exclusive tiers), instead of writing
            them at store time.
    """

    min_output: int = 0
    denied_budget_blocks: int | None = None
    admit_ghosts: bool = False
    persist_evicted_denied: bool = True
    fifo_arena: bool = False
    exclusive: bool = False


@dataclass(frozen=True)
class TierResult:
    """Outcome of one three-tier replay.

    Attributes:
        local_hit: GPU prefix-hit tokens over prompt tokens.
        external_hit: L1/L2 prefix-hit tokens over prompt tokens.
        l2_hit: part of ``external_hit`` from blocks found in L2 only.
        l2_written_blocks: blocks written to L2.
        ghosts: re-stores of denied blocks evicted by the budget.
        denied: blocks refused L2 at store time.
        rescued: denied blocks written to L2 because of an external hit.
    """

    local_hit: float
    external_hit: float
    l2_hit: float
    l2_written_blocks: int
    ghosts: int
    denied: int
    rescued: int


def simulate_tiers(
    requests: Sequence[TraceRequest],
    *,
    gpu_blocks: int,
    l1_blocks: int,
    l2_blocks: int,
    policy: TierPolicy,
) -> TierResult:
    """Replay requests through a GPU prefix cache, L1 and L2.

    Every request's blocks enter the GPU LRU. The external tiers serve the
    prefix continuation after the GPU match. A new block enters L1 and, when
    admitted, L2. A denied block stays in L1 only; any external prefix match
    on it rescues it to L2. Re-stored blocks are judged again by their writer's
    output length, as in the server.

    Args:
        requests: trace rows in arrival order; all rows are counted.
        gpu_blocks: GPU prefix-cache capacity in hash blocks.
        l1_blocks: L1 capacity in hash blocks.
        l2_blocks: L2 capacity in hash blocks.
        policy: L2 admission policy.

    Returns:
        Local and external hit rates, L2 writes and ghost count.
    """
    gpu: OrderedDict[int, None] = OrderedDict()
    l1: OrderedDict[int, None] = OrderedDict()
    l2: OrderedDict[int, None] = OrderedDict()
    denied: OrderedDict[int, None] = OrderedDict()
    # Admitted blocks held in L1 until it evicts them (exclusive only).
    pending: set[int] = set()
    ghosts: set[int] = set()
    # Store order of arena-resident blocks (fifo_arena only).
    arena: OrderedDict[int, None] = OrderedDict()

    def remove(block: int) -> None:
        arena.pop(block, None)
        l1.pop(block, None)
        l2.pop(block, None)
        denied.pop(block, None)
        pending.discard(block)

    local_tokens = external_tokens = prompt_tokens = written = ghost_count = 0
    denied_count = rescued = l2_only_blocks = 0

    def write_l2(block: int) -> None:
        nonlocal written
        l2[block] = None
        l2.move_to_end(block)
        written += 1

    for request in requests:
        blocks = request.hash_ids
        prompt_tokens += request.input_length
        n_local = 0
        while n_local < len(blocks) and blocks[n_local] in gpu:
            n_local += 1
        n_ext = 0
        while n_ext < len(blocks) and (blocks[n_ext] in l1 or blocks[n_ext] in l2):
            block = blocks[n_ext]
            if block in l1:
                l1.move_to_end(block)
            else:
                if not policy.fifo_arena:
                    l2.move_to_end(block)
                l2_only_blocks += n_ext >= n_local
            if block in denied:
                del denied[block]
                if policy.exclusive:
                    pending.add(block)
                else:
                    write_l2(block)
                rescued += 1
            n_ext += 1
        local = min(n_local * MOONCAKE_BLOCK_TOKENS, request.input_length)
        matched = min(max(n_local, n_ext) * MOONCAKE_BLOCK_TOKENS, request.input_length)
        local_tokens += local
        external_tokens += matched - local
        admit = request.output_length >= policy.min_output
        for block in blocks:
            gpu[block] = None
            gpu.move_to_end(block)
            if block in l1 or block in l2:
                continue
            l1[block] = None
            if policy.fifo_arena:
                arena[block] = None
            ghost = block in ghosts
            ghosts.discard(block)
            ghost_count += ghost
            if admit or (ghost and policy.admit_ghosts):
                if policy.exclusive:
                    pending.add(block)
                else:
                    write_l2(block)
                continue
            denied[block] = None
            denied_count += 1
            budget = policy.denied_budget_blocks
            while budget is not None and len(denied) > budget:
                oldest = next(iter(denied))
                remove(oldest)
                ghosts.add(oldest)
        while len(gpu) > gpu_blocks:
            gpu.popitem(last=False)
        while len(l1) > l1_blocks:
            block, _ = l1.popitem(last=False)
            if block in pending:
                pending.discard(block)
                write_l2(block)
            elif block in denied:
                del denied[block]
                if policy.persist_evicted_denied:
                    write_l2(block)
                else:
                    arena.pop(block, None)
                    ghosts.add(block)
        if policy.fifo_arena:
            while len(arena) > l2_blocks:
                remove(next(iter(arena)))
        while len(l2) > l2_blocks:
            l2.popitem(last=False)
    return TierResult(
        local_tokens / max(prompt_tokens, 1),
        external_tokens / max(prompt_tokens, 1),
        l2_only_blocks * MOONCAKE_BLOCK_TOKENS / max(prompt_tokens, 1),
        written,
        ghost_count,
        denied_count,
        rescued,
    )


def tier_report(
    requests: Sequence[TraceRequest],
    *,
    gpu_blocks_list: Sequence[int],
    l1_blocks: int,
    l2_blocks: int,
    block_gib: float,
) -> None:
    """Log reuse-label AUCs and three-tier policy results per GPU capacity.

    Args:
        requests: trace rows in arrival order.
        gpu_blocks_list: GPU prefix-cache capacities to report.
        l1_blocks: L1 capacity in stored blocks.
        l2_blocks: L2 capacity in stored blocks.
        block_gib: GiB written per stored block.
    """
    reused = reused_blocks(requests)
    features = block_features(requests, use_output=False)
    refs = sorted(features)
    signals: dict[str, Callable[[BlockRef], float]] = {
        "output": lambda ref: requests[ref[0]].output_length,
        "input": lambda ref: requests[ref[0]].input_length,
        "gen": lambda ref: features[ref][0],
        "pos": lambda ref: ref[1] / len(requests[ref[0]].hash_ids),
    }
    half = l1_blocks // 2
    policies = {
        "wt": TierPolicy(),
        "a1b": TierPolicy(min_output=500, denied_budget_blocks=half),
        "a1b+g": TierPolicy(
            min_output=500, denied_budget_blocks=half, admit_ghosts=True
        ),
        "hitonly": TierPolicy(min_output=1 << 30, persist_evicted_denied=False),
        "ex": TierPolicy(exclusive=True),
        "ex+a1b+g": TierPolicy(
            min_output=500,
            denied_budget_blocks=half,
            admit_ghosts=True,
            exclusive=True,
        ),
    }
    policies.update(
        {
            f"{name}/fifo": TierPolicy(
                min_output=policy.min_output,
                denied_budget_blocks=policy.denied_budget_blocks,
                admit_ghosts=policy.admit_ghosts,
                persist_evicted_denied=policy.persist_evicted_denied,
                fifo_arena=True,
                exclusive=policy.exclusive,
            )
            for name, policy in list(policies.items())
        }
    )
    for gpu_blocks in gpu_blocks_list:
        local, external, local_gaps, ext_gaps = gpu_reuse_labels(requests, gpu_blocks)
        cells = []
        for label_name, positives in (("any", reused), ("ext", external)):
            labels = [ref in positives for ref in refs]
            aucs = " ".join(
                f"{name}={auc([fn(ref) for ref in refs], labels):.2f}"
                for name, fn in signals.items()
            )
            cells.append(f"{label_name}[{sum(labels)}] {aucs}")
        local_labels = [float(ref in local) for ref in refs]
        cells.append(
            f"ext|local {auc(local_labels, [ref in external for ref in refs]):.2f}"
        )
        for name, gaps in (("gap_local", local_gaps), ("gap_ext", ext_gaps)):
            ordered = sorted(gaps) or [0.0]
            p50 = ordered[len(ordered) // 2]
            p90 = ordered[int(len(ordered) * 0.9)]
            cells.append(f"{name} p50={p50:.0f}s p90={p90:.0f}s")
        print(f"gpu={gpu_blocks} " + "  ".join(cells))
        results = []
        for name, policy in policies.items():
            result = simulate_tiers(
                requests,
                gpu_blocks=gpu_blocks,
                l1_blocks=l1_blocks,
                l2_blocks=l2_blocks,
                policy=policy,
            )
            results.append(
                f"{name} loc={result.local_hit:.3f} ext={result.external_hit:.3f}"
                f"(l2 {result.l2_hit:.3f})"
                f" w={result.l2_written_blocks * block_gib:.0f}"
                f" g={result.ghosts} d={result.denied} r={result.rescued}"
            )
        print(f"gpu={gpu_blocks}   " + "  ".join(results))


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
    parser.add_argument(
        "--gpu-kv",
        nargs="+",
        default=[],
        help="GPU prefix-cache sizes (raw KV). When set, replay the first "
        "--window requests cold through GPU, L1 and L2 and report reuse-label "
        "AUCs and the compressed three-tier policies instead.",
    )
    args = parser.parse_args()
    raw_block_bytes = MOONCAKE_BLOCK_TOKENS * args.kv_bytes_per_token
    if args.gpu_kv:
        block_bytes = raw_block_bytes * args.compress_ratio
        tier_report(
            load_trace(args.trace, args.window, 0, 0),
            gpu_blocks_list=[
                int(parse_size_bytes(size) / raw_block_bytes) for size in args.gpu_kv
            ],
            l1_blocks=int(parse_size_bytes(args.l1) / block_bytes),
            l2_blocks=int(parse_size_bytes(args.l2[0]) / block_bytes),
            block_gib=block_bytes / 2**30,
        )
        return
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
