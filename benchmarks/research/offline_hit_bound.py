# SPDX-License-Identifier: Apache-2.0

"""Offline LRU prefix-hit upper bound for a Mooncake trace window.

Simulates one inclusive LRU cache over Mooncake 512-token hash blocks with a
byte capacity, and reports the fraction of prompt tokens served as a reusable
prefix. Because LRU is a stack algorithm, the hit rate at capacity ``C`` is an
upper bound for any GPU + L1 + L2 hierarchy whose total capacity is ``C``.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
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
    args = parser.parse_args()
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
            rate, hits, _ = prefix_hit_rate(requests, capacity // stored)
            cells.append(f"r{ratio:g}: hit={rate:.3f} miss_tok={total_tokens - hits}")
        print(f"cap={text:>8} " + "  ".join(cells))


if __name__ == "__main__":
    main()
