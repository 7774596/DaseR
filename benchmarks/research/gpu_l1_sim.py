# SPDX-License-Identifier: Apache-2.0
"""Offline GPU / L1 / L2 sim: does L1 waste capacity on GPU-resident blocks?

Replays a trace in file order (strict closed loop, one request at a time).
The GPU prefix cache is a block LRU in raw bytes; after each request its
blocks are touched tail first, so vLLM-style eviction drops a request's tail
before its head. L2 holds everything. L1 is a byte-capacity LRU of stored
blocks (compressed or raw) under one of three policies:

- ``inclusive``: every stored block enters L1 (current DaseR behaviour).
- ``gpu-aware``: L1 evicts blocks that are currently GPU-resident first.
- ``bip-*``: the same with BIP insertion (one in 32 new blocks at the MRU
  end), DaseR's default L1 order.
- ``shadow``: what the server can implement: a block moves to the L1 LRU
  end when it becomes GPU-resident and to the MRU end when the GPU evicts it.
- ``demote``: ``gpu-aware`` plus, when the GPU evicts a block that L1 no
  longer holds, a background L2 read brings it back into L1.

Reports, per request window, the GPU-local share, the external share served
from L1 and from L2 (critical-path SSD bytes), and background L2 bytes.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

from benchmarks.mooncake.benchmark import MOONCAKE_BLOCK_TOKENS, load_trace
from benchmarks.utils.sizing import parse_size_bytes


def simulate(
    requests: list,
    gpu_blocks: int,
    l1_blocks: int,
    policy: str,
    warmup: int,
    promote: bool = False,
) -> dict[str, float]:
    """Replay ``requests`` and return token shares and background block reads.

    Args:
        requests: ``TraceRequest`` rows in replay order.
        gpu_blocks: GPU prefix-cache capacity in blocks.
        l1_blocks: L1 capacity in blocks (already scaled by the codec ratio).
        policy: ``inclusive``, ``gpu-aware`` or ``demote`` with LRU
            insertion, or the same with a ``bip-`` prefix for BIP insertion.
        warmup: Requests excluded from the shares.
        promote: Blocks a load reads from L2 also enter L1 (DaseR loads and
            readaheads fill L1), with the same insertion policy as stores.

    Returns:
        ``loc``, ``l1`` and ``l2`` shares of prompt blocks after warmup, and
        ``bg_l2`` background blocks per measured request.
    """
    gpu: OrderedDict[int, None] = OrderedDict()
    l1: OrderedDict[int, None] = OrderedDict()
    l2: set[int] = set()
    bip = policy.startswith("bip")
    bip_interval = 32
    gpu_aware = "gpu-aware" in policy or "demote" in policy
    demote = "demote" in policy
    shadow = "shadow" in policy
    totals = {"loc": 0, "l1": 0, "l2": 0, "all": 0, "bg": 0}

    inserts = 0

    def l1_insert(block: int) -> None:
        nonlocal inserts
        l1[block] = None
        inserts = (inserts + 1) % bip_interval
        # BIP: most new blocks enter at the LRU end (DaseR's default L1).
        l1.move_to_end(block, last=not bip or inserts == 0)
        while len(l1) > l1_blocks:
            victim = None
            if gpu_aware:
                victim = next((b for b in l1 if b in gpu and b != block), None)
            if victim is None:
                victim = next(iter(l1))
            del l1[victim]

    for index, request in enumerate(requests):
        blocks = list(request.hash_ids)
        measured = index >= warmup
        local = 0
        while local < len(blocks) and blocks[local] in gpu:
            local += 1
        external = local
        served_l1 = served_l2 = 0
        from_l2: list[int] = []
        while external < len(blocks) and blocks[external] in l2:
            if blocks[external] in l1:
                served_l1 += 1
                l1.move_to_end(blocks[external])
            else:
                served_l2 += 1
                from_l2.append(blocks[external])
            external += 1
        if promote:
            for block in from_l2:
                l1_insert(block)
        if measured:
            totals["loc"] += local
            totals["l1"] += served_l1
            totals["l2"] += served_l2
            totals["all"] += len(blocks)
        # Store: new blocks are written through to L1 and L2.
        for block in blocks:
            if block not in l2:
                l2.add(block)
                l1_insert(block)
        # GPU: the request's blocks become most recent, head last.
        for block in reversed(blocks):
            gpu[block] = None
            gpu.move_to_end(block)
            if shadow and block in l1:
                # GPU-resident: next L1 victim.
                l1.move_to_end(block, last=False)
        while len(gpu) > gpu_blocks:
            victim, _ = gpu.popitem(last=False)
            if shadow and victim in l1:
                # Left the GPU: L1 now holds its only fast copy.
                l1.move_to_end(victim)
            if demote and victim not in l1:
                l1_insert(victim)
                if measured:
                    totals["bg"] += 1
    n = max(totals["all"], 1)
    return {
        "loc": totals["loc"] / n,
        "l1": totals["l1"] / n,
        "l2": totals["l2"] / n,
        "bg_l2": totals["bg"] / max(len(requests) - warmup, 1),
    }


def main() -> None:
    """Parse arguments and print one row per (L1 format, policy)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--kv-bytes-per-token", type=int, default=144 * 1024)
    parser.add_argument("--gpu", default="48gib")
    parser.add_argument("--l1", nargs="+", default=["64gib"])
    parser.add_argument("--ratio", type=float, default=0.737)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument(
        "--promote", action="store_true", help="L2 load reads also fill L1."
    )
    args = parser.parse_args()
    requests = load_trace(args.trace, 0, 0, 0)
    block_bytes = MOONCAKE_BLOCK_TOKENS * args.kv_bytes_per_token
    gpu_blocks = parse_size_bytes(args.gpu) // block_bytes
    for l1_size in args.l1:
        for fmt, ratio in (("raw", 1.0), ("cmp", args.ratio)):
            l1_blocks = int(parse_size_bytes(l1_size) / (block_bytes * ratio))
            for policy in (
                "inclusive",
                "gpu-aware",
                "bip-inclusive",
                "bip-gpu-aware",
                "shadow",
                "bip-shadow",
            ):
                r = simulate(
                    requests, gpu_blocks, l1_blocks, policy, args.warmup, args.promote
                )
                print(
                    f"l1={l1_size} {fmt} {policy:13s} loc={r['loc']:.3f} "
                    f"l1={r['l1']:.3f} l2={r['l2']:.3f} "
                    f"bg_l2_blocks/req={r['bg_l2']:.1f}"
                )


if __name__ == "__main__":
    main()
