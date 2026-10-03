# SPDX-License-Identifier: Apache-2.0
"""Export cc-traces sessions as a Mooncake-format closed-loop replay trace.

cc-traces rows carry per-session (``hash_id_scope: local``) chained 64-token
hash IDs. Each output 512-token block takes a global ID keyed by
``(session, model, last chained hash, hashes in block)``: chained hashes make
the last hash identify the whole prefix, so blocks shared within a session
keep one ID while different sessions and models never alias. Requests longer
than ``--max-ctx`` are dropped (the model cannot serve them), and so are
requests shorter than ``--min-ctx`` (to replay only long-context tails).
Sessions start
at Poisson offsets with about ``--concurrency`` overlapping and keep their real
request offsets; the merged stream is written in time order with
``timestamp = index * --step-ms`` and ``session_id`` for strict-order replay.
``--max-output`` caps decode length so replays measure TTFT, not decode.
``--round-robin K`` instead models K closed-loop agents sharing one server
(tool waits longer than the other agents' service): K sessions are live and
take turns one request each; a finished session's slot goes to the next one.
"""

import argparse
import json
import math
import random

CC_BLOCK_TOKENS = 64
OUT_BLOCK_TOKENS = 512
GROUP = OUT_BLOCK_TOKENS // CC_BLOCK_TOKENS


def _block_ids(
    hash_ids: list[int], tokens: int, scope: tuple[int, str], table: dict
) -> list[int]:
    """Map one request's 64-token chained hashes to global 512-token IDs.

    Args:
        hash_ids: cc-traces chained hash IDs of the request (local scope).
        tokens: request input length in tokens.
        scope: ``(session index, model)`` namespace for the local IDs.
        table: shared key -> global ID map, extended in place.

    Returns:
        One global ID per 512-token block of the request.
    """
    ids = []
    for start in range(0, math.ceil(tokens / OUT_BLOCK_TOKENS) * GROUP, GROUP):
        group = hash_ids[start : start + GROUP]
        key = (*scope, group[-1], len(group))
        ids.append(table.setdefault(key, len(table)))
    return ids


def _poisson_order(
    chains: list[list[dict]], concurrency: float, rng: random.Random
) -> list[tuple[int, int]]:
    """Merge sessions started at Poisson offsets on their real timelines.

    Args:
        chains: per-session request lists.
        concurrency: mean number of overlapping sessions.
        rng: seeded RNG for the start offsets.

    Returns:
        ``(session, request index)`` pairs in merged time order.
    """
    mean_span = sum(c[-1]["t"] - c[0]["t"] for c in chains) / len(chains)
    rate = concurrency / max(mean_span, 1e-9)
    t0, stream = 0.0, []
    for sid, reqs in enumerate(chains):
        t0 += rng.expovariate(rate)
        stream += [(t0 + r["t"] - reqs[0]["t"], sid, i) for i, r in enumerate(reqs)]
    stream.sort()
    return [(sid, i) for _, sid, i in stream]


def _round_robin_order(chains: list[list[dict]], live: int) -> list[tuple[int, int]]:
    """Interleave ``live`` sessions one request per turn, refilling freed slots.

    Args:
        chains: per-session request lists.
        live: number of concurrently live sessions.

    Returns:
        ``(session, request index)`` pairs in replay order.
    """
    pending = list(range(len(chains)))
    slots = [[pending.pop(0), 0] for _ in range(min(live, len(pending)))]
    order = []
    while slots:
        for slot in list(slots):
            sid, i = slot
            order.append((sid, i))
            slot[1] += 1
            if slot[1] == len(chains[sid]):
                if pending:
                    slot[:] = [pending.pop(0), 0]
                else:
                    slots.remove(slot)
    return order


def export(
    path: str,
    sessions: int,
    concurrency: float,
    max_ctx: int,
    min_peak_ctx: int,
    seed: int,
    step_ms: int,
    max_output: int = 0,
    min_ctx: int = 0,
    min_requests: int = 1,
    round_robin: int = 0,
) -> list[dict]:
    """Build the merged replay rows.

    Args:
        path: cc-traces ``traces.jsonl``.
        sessions: number of sessions to sample.
        concurrency: mean number of overlapping sessions.
        max_ctx: drop requests whose input plus output exceeds this.
        min_peak_ctx: only sample sessions reaching this input length.
        seed: RNG seed for sampling and start offsets.
        step_ms: timestamp spacing of consecutive output rows.
        max_output: cap on ``output_length`` (0 keeps the recorded value).
        min_ctx: drop requests whose input is shorter than this.
        min_requests: only sample sessions keeping at least this many requests.
        round_robin: when positive, interleave this many live sessions in
            turn instead of merging Poisson-offset real timelines.

    Returns:
        Mooncake rows (``timestamp``, ``input_length``, ``output_length``,
        ``hash_ids``, ``session_id``) in replay order.
    """
    chains = []
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            reqs = [
                r
                for r in s["requests"]
                if "in" in r
                and r.get("hash_ids")
                and min_ctx <= r["in"]
                and r["in"] + r["out"] <= max_ctx
            ]
            if len(reqs) >= min_requests and max(r["in"] for r in reqs) >= min_peak_ctx:
                chains.append(reqs)
    rng = random.Random(seed)
    rng.shuffle(chains)
    chains = chains[:sessions]
    stream = (
        _round_robin_order(chains, round_robin)
        if round_robin > 0
        else _poisson_order(chains, concurrency, rng)
    )
    table: dict = {}
    rows = []
    for index, (sid, i) in enumerate(stream):
        r = chains[sid][i]
        rows.append(
            {
                "timestamp": index * step_ms,
                "input_length": r["in"],
                "output_length": max(1, min(r["out"], max_output or r["out"])),
                "hash_ids": _block_ids(
                    r["hash_ids"], r["in"], (sid, r["model"]), table
                ),
                "session_id": sid,
            }
        )
    return rows


def main() -> None:
    """CLI entry point: write the exported trace and print a one-line summary."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc-traces", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sessions", type=int, default=8)
    p.add_argument("--concurrency", type=float, default=4.0)
    p.add_argument("--max-ctx", type=int, default=262144)
    p.add_argument("--min-peak-ctx", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--step-ms", type=int, default=1000)
    p.add_argument("--max-output", type=int, default=0)
    p.add_argument("--min-ctx", type=int, default=0)
    p.add_argument("--min-requests", type=int, default=1)
    p.add_argument("--round-robin", type=int, default=0)
    a = p.parse_args()
    rows = export(
        a.cc_traces,
        a.sessions,
        a.concurrency,
        a.max_ctx,
        a.min_peak_ctx,
        a.seed,
        a.step_ms,
        a.max_output,
        a.min_ctx,
        a.min_requests,
        a.round_robin,
    )
    with open(a.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    blocks = len({h for row in rows for h in row["hash_ids"]})
    tokens = sum(row["input_length"] for row in rows)
    print(
        f"requests={len(rows)} sessions={len({r['session_id'] for r in rows})} "
        f"unique_blocks={blocks} ({blocks * OUT_BLOCK_TOKENS / 1e6:.2f}M tok) "
        f"input_tok={tokens / 1e6:.2f}M"
    )


if __name__ == "__main__":
    main()
