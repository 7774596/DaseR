# SPDX-License-Identifier: Apache-2.0
"""Offline prefix-hit bound of a block cache under several replacement policies.

Usage::

    python -m benchmarks.research.prefix_closed_bound <trace.jsonl> \
        --first 300 --kib-per-token 144 --gib 48 64 96

The trace is ``hash_ids`` JSONL (one chained block id per ``block_tokens``
tokens). A request hits its leading resident blocks only, like a rolling-prefix
index. ``fifo`` evicts in insertion order; ``cascade`` also drops the resident
descendants of each victim (they are unreachable until it returns); ``lru``
refreshes every block a request touches; ``belady`` evicts the furthest next
use (clairvoyant, with admission bypass) as the replacement ceiling. The
report gives the token hit ratio and the share of resident blocks that are
unreachable ("orphan"), sampled when the cache is at least 95% full. It
models one tier, without the GPU cache or host L1, so it only ranks policies.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import heapq
import json
from pathlib import Path
from typing import Sequence

MODES = ("fifo", "cascade", "lru", "belady")


def simulate(
    requests: Sequence[tuple[Sequence[int], int]],
    capacity_blocks: int,
    mode: str,
    block_tokens: int,
    sample_every: int = 10,
) -> tuple[float, float]:
    """Replay requests through one block cache.

    Args:
        requests: ``(hash_ids, input_length)`` per request, in arrival order.
        capacity_blocks: Resident block budget.
        mode: One of :data:`MODES`.
        block_tokens: Tokens per block.
        sample_every: Sample the orphan share every this many requests.

    Returns:
        ``(token hit ratio, mean orphan share of a full cache)``.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "belady":
        return _simulate_belady(requests, capacity_blocks, block_tokens), 0.0
    cache: OrderedDict[int, None] = OrderedDict()
    parent: dict[int, int | None] = {}
    children: dict[int | None, set[int]] = {}
    hit = total = 0
    orphan_samples: list[float] = []

    def drop(block: int) -> None:
        stack = [block]
        seen = {block}
        while stack:
            victim = stack.pop()
            cache.pop(victim, None)
            if mode == "cascade":
                # A re-stored parent can sit between a missing grandparent
                # and resident grandchildren, so walk every descendant.
                stack.extend(k for k in children.get(victim, ()) if k not in seen)
                seen.update(children.get(victim, ()))

    for index, (ids, length) in enumerate(requests, start=1):
        total += length
        lead = 0
        for block in ids:
            if block not in cache:
                break
            lead += 1
        hit += min(lead * block_tokens, length)
        prev: int | None = None
        for block in ids:
            if block not in parent:
                parent[block] = prev
                children.setdefault(prev, set()).add(block)
            if block in cache:
                if mode == "lru":
                    cache.move_to_end(block)
            else:
                cache[block] = None
                while len(cache) > capacity_blocks:
                    drop(next(iter(cache)))
            prev = block
        if index % sample_every == 0 and len(cache) >= 0.95 * capacity_blocks:
            orphan_samples.append(_orphan_share(cache, parent))
    share = sum(orphan_samples) / len(orphan_samples) if orphan_samples else 0.0
    return hit / max(total, 1), share


def _simulate_belady(
    requests: Sequence[tuple[Sequence[int], int]],
    capacity_blocks: int,
    block_tokens: int,
) -> float:
    """Return the token hit ratio of clairvoyant replacement with bypass.

    Each request inserts its missing blocks, then the cache evicts the
    blocks whose next use lies furthest ahead (never-reused blocks first,
    including ones just inserted, which acts as admission bypass). A chained
    child never recurs before its parent, so this also stays prefix-closed.
    """
    uses: dict[int, list[int]] = {}
    for index, (ids, _) in enumerate(requests):
        for block in ids:
            seen = uses.setdefault(block, [])
            if not seen or seen[-1] != index:
                seen.append(index)
    cursor = dict.fromkeys(uses, 0)
    never = len(requests)
    cache: dict[int, int] = {}
    heap: list[tuple[int, int]] = []
    hit = total = 0
    for index, (ids, length) in enumerate(requests):
        total += length
        lead = 0
        for block in ids:
            if block not in cache:
                break
            lead += 1
        hit += min(lead * block_tokens, length)
        for block in ids:
            positions = uses[block]
            while cursor[block] < len(positions) and positions[cursor[block]] <= index:
                cursor[block] += 1
            nxt = positions[cursor[block]] if cursor[block] < len(positions) else never
            if cache.get(block) != nxt:
                cache[block] = nxt
                heapq.heappush(heap, (-nxt, block))
        while len(cache) > capacity_blocks:
            neg, block = heapq.heappop(heap)
            if cache.get(block) == -neg:
                del cache[block]
    return hit / max(total, 1)


def _orphan_share(
    cache: OrderedDict[int, None], parent: dict[int, int | None]
) -> float:
    """Return the share of resident blocks with a non-resident ancestor."""
    reachable: dict[int, bool] = {}

    def ok(block: int) -> bool:
        path = []
        node: int | None = block
        while node is not None and node not in reachable:
            if node not in cache:
                break
            path.append(node)
            node = parent[node]
        result = node is None or reachable.get(node, False)
        for item in path:
            reachable[item] = result
        return result

    return sum(not ok(b) for b in cache) / max(len(cache), 1)


def main(argv: list[str] | None = None) -> None:
    """Print one line per capacity with the hit and orphan share of each mode.

    Args:
        argv: Optional argument list; defaults to ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--first", type=int, default=300)
    parser.add_argument("--block-tokens", type=int, default=512)
    parser.add_argument("--kib-per-token", type=float, default=144.0)
    parser.add_argument("--gib", type=float, nargs="+", required=True)
    args = parser.parse_args(argv)
    with args.trace.open() as handle:
        rows = [
            json.loads(line) for _, line in zip(range(args.first), handle, strict=False)
        ]
    requests = [(row["hash_ids"], int(row["input_length"])) for row in rows]
    block_gib = args.block_tokens * args.kib_per_token / 2**20
    for gib in args.gib:
        capacity = int(gib / block_gib)
        cells = []
        for mode in MODES:
            hit, orphan = simulate(requests, capacity, mode, args.block_tokens)
            cells.append(f"{mode} hit {hit:.3f} orphan {orphan:.3f}")
        print(f"{gib:g} GiB ({capacity} blocks): " + " | ".join(cells))


if __name__ == "__main__":
    main()
