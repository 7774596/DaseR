# SPDX-License-Identifier: Apache-2.0

"""Offline LRU prefix-hit upper bound for a Mooncake trace window.

Simulates one inclusive LRU cache over Mooncake 512-token hash blocks with a
byte capacity, and reports the fraction of prompt tokens served as a reusable
prefix. Because LRU is a stack algorithm, the hit rate at capacity ``C`` is an
upper bound for any GPU + L1 + L2 hierarchy whose total capacity is ``C``.
``--policy belady`` instead evicts the block whose next use is farthest
(deepest prefix position first among ties): a clairvoyant bound for what any
replacement policy could reach at the same capacity. ``--policy lru-dead``
ranks a realizable hint: blocks a session's prefix diverged from are evicted
first. ``--restart-at`` empties the cache before the given requests to price
a cold cache-service restart.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Callable
import heapq
from pathlib import Path

from benchmarks.mooncake.benchmark import MOONCAKE_BLOCK_TOKENS, load_trace
from benchmarks.utils.sizing import parse_size_bytes


def prefix_hit_rate(requests: list, capacity_blocks: int) -> tuple[float, int, int]:
    """Replay requests through an LRU of ``capacity_blocks`` hash blocks.

    Args:
        requests: ``TraceRequest`` rows in arrival order.
        capacity_blocks: Cache capacity in 512-token blocks (``<=0`` = infinite).

    Returns:
        ``(hit_rate, hit_tokens, prompt_tokens)`` where hits count only the
        leading resident blocks of each prompt.
    """
    cache: OrderedDict[int, None] = OrderedDict()
    hit_tokens = 0
    prompt_tokens = 0
    for request in requests:
        prompt_tokens += request.input_length
        leading = 0
        for block in request.hash_ids:
            if block not in cache:
                break
            leading += 1
        # The final block may be partial; never count beyond the prompt.
        hit_tokens += min(leading * MOONCAKE_BLOCK_TOKENS, request.input_length)
        for block in request.hash_ids:
            cache[block] = None
            cache.move_to_end(block)
        if capacity_blocks > 0:
            while len(cache) > capacity_blocks:
                cache.popitem(last=False)
    return hit_tokens / max(prompt_tokens, 1), hit_tokens, prompt_tokens


def dead_hint_prefix_hit_rate(
    requests: list, capacity_blocks: int
) -> tuple[float, int, int]:
    """Replay requests through an LRU that evicts session-dead blocks first.

    When a session's request no longer contains blocks of that session's
    previous request (the prefix diverged, e.g. a template stripped earlier
    reasoning), those blocks are moved to the eviction end. This needs no
    future knowledge: a serving gateway sees the divergence on arrival.

    Args:
        requests: ``TraceRequest`` rows in arrival order.
        capacity_blocks: Cache capacity in 512-token blocks (``<=0`` = infinite).

    Returns:
        ``(hit_rate, hit_tokens, prompt_tokens)`` with the same leading-prefix
        hit rule as :func:`prefix_hit_rate`.
    """
    cache: OrderedDict[int, None] = OrderedDict()
    previous: dict[object, list[int]] = {}
    hit_tokens = 0
    prompt_tokens = 0
    for request in requests:
        prompt_tokens += request.input_length
        leading = 0
        for block in request.hash_ids:
            if block not in cache:
                break
            leading += 1
        hit_tokens += min(leading * MOONCAKE_BLOCK_TOKENS, request.input_length)
        current = set(request.hash_ids)
        for block in previous.get(request.session_id, []):
            if block not in current and block in cache:
                cache.move_to_end(block, last=False)
        for block in request.hash_ids:
            cache[block] = None
            cache.move_to_end(block)
        if capacity_blocks > 0:
            while len(cache) > capacity_blocks:
                cache.popitem(last=False)
        previous[request.session_id] = request.hash_ids
    return hit_tokens / max(prompt_tokens, 1), hit_tokens, prompt_tokens


def belady_prefix_hit_rate(
    requests: list, capacity_blocks: int
) -> tuple[float, int, int]:
    """Replay requests through a clairvoyant farthest-next-use cache.

    Args:
        requests: ``TraceRequest`` rows in arrival order.
        capacity_blocks: Cache capacity in 512-token blocks (``<=0`` = infinite).

    Returns:
        ``(hit_rate, hit_tokens, prompt_tokens)`` with the same leading-prefix
        hit rule as :func:`prefix_hit_rate`. Ties in next use evict the block
        deepest in its prompt, so shared prefixes outlive their tails.
    """
    never = len(requests)
    uses: dict[int, list[int]] = {}
    for index, request in enumerate(requests):
        for block in request.hash_ids:
            uses.setdefault(block, []).append(index)
    cursor = {block: 0 for block in uses}
    resident: dict[int, int] = {}
    heap: list[tuple[int, int, int]] = []
    hit_tokens = 0
    prompt_tokens = 0
    for index, request in enumerate(requests):
        prompt_tokens += request.input_length
        leading = 0
        for block in request.hash_ids:
            if block not in resident:
                break
            leading += 1
        hit_tokens += min(leading * MOONCAKE_BLOCK_TOKENS, request.input_length)
        for position, block in enumerate(request.hash_ids):
            times = uses[block]
            while cursor[block] < len(times) and times[cursor[block]] <= index:
                cursor[block] += 1
            nxt = times[cursor[block]] if cursor[block] < len(times) else never
            resident[block] = nxt
            heapq.heappush(heap, (-nxt, -position, block))
        if capacity_blocks > 0:
            while len(resident) > capacity_blocks:
                neg_next, _, block = heapq.heappop(heap)
                if resident.get(block) == -neg_next:
                    del resident[block]
    return hit_tokens / max(prompt_tokens, 1), hit_tokens, prompt_tokens


def restarted_hit_rate(
    simulate: Callable[[list, int], tuple[float, int, int]],
    requests: list,
    capacity_blocks: int,
    restart_at: list[int],
) -> tuple[float, int, int]:
    """Replay ``requests`` with the cache emptied before each restart index.

    Models a cache service that loses its contents (cold restart) before
    request ``restart_at[i]``: each segment is replayed on an empty cache.

    Args:
        simulate: One of the policy functions in this module.
        requests: ``TraceRequest`` rows in arrival order.
        capacity_blocks: Cache capacity in 512-token blocks (``<=0`` = infinite).
        restart_at: Request indices before which the cache is emptied.

    Returns:
        ``(hit_rate, hit_tokens, prompt_tokens)`` summed over the segments.
    """
    bounds = [0, *sorted(i for i in restart_at if 0 < i < len(requests))]
    bounds.append(len(requests))
    hit_tokens = 0
    prompt_tokens = 0
    for start, end in zip(bounds, bounds[1:], strict=False):
        _, hits, tokens = simulate(requests[start:end], capacity_blocks)
        hit_tokens += hits
        prompt_tokens += tokens
    return hit_tokens / max(prompt_tokens, 1), hit_tokens, prompt_tokens


_POLICIES = {
    "lru": prefix_hit_rate,
    "lru-dead": dead_hint_prefix_hit_rate,
    "belady": belady_prefix_hit_rate,
}
_POLICY_HELP = (
    "lru: plain LRU; lru-dead: LRU that evicts blocks a session's prefix "
    "diverged from first; belady: clairvoyant farthest next use."
)


def main() -> None:
    """CLI entry point: print one line per capacity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--start-request", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--kv-bytes-per-token", type=int, default=80 * 1024)
    parser.add_argument(
        "--capacities",
        nargs="+",
        default=["24gib", "64gib", "128gib", "256gib", "512gib", "0"],
        help="Byte capacities; 0 means infinite.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[1.0],
        help="Stored/raw byte ratios (one column each, e.g. 1.0 0.737).",
    )
    parser.add_argument(
        "--policy", choices=tuple(_POLICIES), default="lru", help=_POLICY_HELP
    )
    parser.add_argument(
        "--restart-at",
        nargs="*",
        type=int,
        default=[],
        help="Request indices (within the window) before which the cache is "
        "emptied, modelling a cold cache-service restart.",
    )
    args = parser.parse_args()
    policy = _POLICIES[args.policy]

    def simulate(rows: list, capacity_blocks: int) -> tuple[float, int, int]:
        return restarted_hit_rate(policy, rows, capacity_blocks, args.restart_at)

    requests = list(load_trace(args.trace, args.max_requests, args.start_request, 0))
    unique = len({block for request in requests for block in request.hash_ids})
    total_tokens = sum(request.input_length for request in requests)
    block_bytes = MOONCAKE_BLOCK_TOKENS * args.kv_bytes_per_token
    print(
        f"requests={len(requests)} span_s={requests[-1].timestamp_ms / 1000:.0f} "
        f"unique_kv_gib={unique * block_bytes / 2**30:.0f}"
    )
    for text in args.capacities:
        capacity = parse_size_bytes(text)
        cells = []
        for ratio in args.ratios:
            stored = max(1, int(block_bytes * ratio))
            rate, hits, _ = simulate(requests, capacity // stored)
            cells.append(f"r{ratio:g}: hit={rate:.3f} miss_tok={total_tokens - hits}")
        print(f"cap={text:>8} " + "  ".join(cells))


if __name__ == "__main__":
    main()
