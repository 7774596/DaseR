# SPDX-License-Identifier: Apache-2.0
"""Reorder a session trace to follow cc-traces' real session-switch order.

The input trace (TraceLab-style rows with ``session_id``) keeps each session's
requests in order; only the interleaving across sessions changes. cc-traces
sessions start at Poisson offsets (about ``--concurrency`` overlapping) with
their real request offsets, and the merged stream's session sequence, with
consecutive repeats collapsed, drives the order (turns left once too few
sessions remain to alternate are appended at the end): each new cc session claims
a free trace session (one with requests left whose previous cc session has
ended), and later requests of an unmapped cc session are skipped. Only the
switching pattern is borrowed; trace sessions keep their own prompts.
Timestamps are rewritten to ``index * --step-ms`` for closed-loop replay.
"""

import argparse
from collections import Counter
import json
import random
import struct

from daser.server.session_predictor import SessionPredictor


def _real_switch_sequence(
    path: str, sessions: int, concurrency: float, min_ctx: int, seed: int
) -> list[int]:
    """Return the merged cc-traces session sequence, repeats collapsed."""
    chains = []
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            reqs = [r for r in s["requests"] if "in" in r and r["in"] >= min_ctx]
            if len(reqs) >= 2:
                chains.append(reqs)
    rng = random.Random(seed)
    rng.shuffle(chains)
    chains = chains[:sessions]
    mean_span = sum(c[-1]["t"] - c[0]["t"] for c in chains) / len(chains)
    rate = concurrency / mean_span
    t0, stream = 0.0, []
    for sid, reqs in enumerate(chains):
        t0 += rng.expovariate(rate)
        stream += [(t0 + r["t"] - reqs[0]["t"], sid) for r in reqs]
    stream.sort()
    seq: list[int] = []
    for _t, sid in stream:
        if not seq or seq[-1] != sid:
            seq.append(sid)
    return seq


def _replay_accuracy(order: list[int], depth: int) -> float:
    """Share of session switches whose session ``SessionPredictor`` predicted."""
    pred = SessionPredictor()
    turns: Counter[int] = Counter()
    prompts: dict[bytes, int] = {}
    predicted: set[int] = set()
    hits = switches = 0
    for index, sid in enumerate(order):
        if index and turns[sid]:
            switches += 1
            hits += sid in predicted
        tokens = [sid * 1_000_000 + i for i in range(8 * (turns[sid] + 1))]
        packed = struct.pack(f"{len(tokens)}i", *tokens)
        turns[sid] += 1
        prompts[packed] = sid
        pred.observe(packed, "m")
        predicted = {prompts[p] for p, _m in pred.predict(depth)}
    return hits / max(switches, 1)


def main() -> None:
    """Parse arguments, write the reordered trace and print its accuracy."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="Input session trace (jsonl).")
    ap.add_argument("--cc-traces", required=True, help="cc-traces traces.jsonl.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--sessions", type=int, default=64)
    ap.add_argument("--concurrency", type=float, default=4.0)
    ap.add_argument("--min-ctx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step-ms", type=int, default=1000)
    args = ap.parse_args()
    with open(args.trace) as f:
        rows = [json.loads(line) for line in f]
    by_session: dict[int, list[dict]] = {}
    for row in rows:
        by_session.setdefault(int(row["session_id"]), []).append(row)
    trace_sessions = sorted(by_session)
    seq = _real_switch_sequence(
        args.cc_traces, args.sessions, args.concurrency, args.min_ctx, args.seed
    )
    last_seen = {cc: index for index, cc in enumerate(seq)}
    mapping: dict[int, int] = {}
    free = list(trace_sessions)
    used: Counter[int] = Counter()
    order: list[int] = []
    for index, cc in enumerate(seq):
        if cc not in mapping:
            if not free:
                continue
            mapping[cc] = free.pop(0)
        sid = mapping[cc]
        if sid < 0:
            continue
        if not order or order[-1] != sid:
            order.append(sid)
            used[sid] += 1
        if used[sid] == len(by_session[sid]):
            mapping[cc] = -1
        elif index == last_seen[cc]:
            # The cc session ended: its trace session continues under the
            # next new cc session.
            free.append(sid)
        if len(order) == len(rows):
            break
    # The last live sessions run out of partners to alternate with; their
    # remaining turns follow back to back (the GPU cache serves those).
    tail = len(rows) - len(order)
    for sid in trace_sessions:
        order += [sid] * (len(by_session[sid]) - used[sid])
    emitted: Counter[int] = Counter()
    with open(args.output, "w") as f:
        for index, sid in enumerate(order):
            row = dict(by_session[sid][emitted[sid]])
            emitted[sid] += 1
            row["timestamp"] = index * args.step_ms
            f.write(json.dumps(row) + "\n")
    print("order", " ".join(map(str, order)), f"tail={tail}")
    for depth in (1, 2, 3):
        print(f"depth={depth} switch_accuracy={_replay_accuracy(order, depth):.3f}")


if __name__ == "__main__":
    main()
