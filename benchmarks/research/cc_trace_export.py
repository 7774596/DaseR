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
``--real-time-gap-cap-s C`` instead keeps real timing: idle gaps are capped
at C seconds, timestamps are merged arrivals in milliseconds, and each row's
``gap_ms`` (idle time after the previous request of the same session and
model) lets prefix-dependent replay start it that long after its producer
completes.
``--reasoning-strip-think-s S`` models a thinking-model template that drops
past reasoning at user turns (idle longer than S seconds): the previous query
chain is re-rendered, so its blocks get new IDs, and the user-turn row carries
``prewarm_tokens``, the prefix known as soon as that chain ends.
``--max-output`` caps decode length so replays measure TTFT, not decode.
``--round-robin K`` instead models K closed-loop agents sharing one server
(tool waits longer than the other agents' service): K sessions are live and
take turns one request each; a finished session's slot goes to the next one.
"""

import argparse
import bisect
import json
import math
import random

CC_BLOCK_TOKENS = 64
OUT_BLOCK_TOKENS = 512
GROUP = OUT_BLOCK_TOKENS // CC_BLOCK_TOKENS


def _block_ids(
    hash_ids: list[int],
    tokens: int,
    scope: tuple[int, str],
    table: dict,
    epochs: list[tuple[int, int]] | None = None,
) -> list[int]:
    """Map one request's 64-token chained hashes to global 512-token IDs.

    Args:
        hash_ids: cc-traces chained hash IDs of the request (local scope).
        tokens: request input length in tokens.
        scope: ``(session index, model)`` namespace for the local IDs.
        table: shared key -> global ID map, extended in place.
        epochs: optional reasoning-strip rewrite points ``(token offset,
            epoch)`` sorted by offset, first offset 0. A block ending past an
            offset takes that epoch, so blocks rewritten at a user turn get
            new IDs while older and later requests of the same epoch agree.

    Returns:
        One global ID per 512-token block of the request.
    """
    ids = []
    offsets = [o for o, _ in epochs] if epochs else []
    for start in range(0, math.ceil(tokens / OUT_BLOCK_TOKENS) * GROUP, GROUP):
        group = hash_ids[start : start + GROUP]
        key = (*scope, group[-1], len(group))
        if epochs:
            end = start * CC_BLOCK_TOKENS + len(group) * CC_BLOCK_TOKENS
            key += (epochs[bisect.bisect_left(offsets, end) - 1][1],)
        ids.append(table.setdefault(key, len(table)))
    return ids


def _strip_reasoning(
    reqs: list[dict], think_s: float
) -> tuple[list[list[tuple[int, int]]], list[int]]:
    """Model a thinking-model chat template that strips past reasoning.

    Templates such as Qwen3's drop ``reasoning_content`` of every assistant
    message before the last real user query. When a new user turn arrives
    (``think_time`` above ``think_s``), the previous query chain is re-rendered
    without reasoning, so the prompt diverges where that chain started and its
    whole tail is recomputed, while the old chain KV is never read again.

    Args:
        reqs: one session's requests in time order.
        think_s: idle seconds above which a request starts a new user turn.

    Returns:
        ``(epochs, prewarm)`` per request: the rewrite points to pass to
        :func:`_block_ids`, and the leading tokens of the request that are
        known when the previous chain ends (0 when not a user turn). The
        known prefix is capped at the previous request's input plus output,
        an upper bound on the stripped chain because reasoning is part of
        the recorded output.
    """
    state: dict[str, dict] = {}
    epochs_out, prewarm_out = [], []
    for r in reqs:
        s = state.get(r["model"])
        prewarm = 0
        if s is None:
            s = state[r["model"]] = {
                "epochs": [(0, 0)],
                "next": 1,
                "start": r["in"],
                "prev": r,
            }
        elif r.get("think_time", 0.0) > think_s:
            # Re-render diverges at the previous chain's first assistant turn.
            start = s["start"]
            kept = [e for e in s["epochs"] if e[0] < start]
            s["epochs"] = [*kept, (start, s["next"])]
            s["next"] += 1
            prev = s["prev"]
            prewarm = min(r["in"], prev["in"] + prev["out"])
            s["start"] = r["in"]
        s["prev"] = r
        epochs_out.append(list(s["epochs"]))
        prewarm_out.append(prewarm)
    return epochs_out, prewarm_out


def _poisson_order(
    chains: list[list[dict]],
    concurrency: float,
    rng: random.Random,
) -> list[tuple[float, int, int]]:
    """Merge sessions started at Poisson offsets on their real timelines.

    Args:
        chains: per-session request lists.
        concurrency: mean number of overlapping sessions.
        rng: seeded RNG for the start offsets.

    Returns:
        ``(arrival seconds, session, request index)`` in merged time order.
    """
    mean_span = sum(c[-1]["t"] - c[0]["t"] for c in chains) / len(chains)
    rate = concurrency / max(mean_span, 1e-9)
    t0, stream = 0.0, []
    for sid, reqs in enumerate(chains):
        t0 += rng.expovariate(rate)
        stream += [(t0 + r["t"] - reqs[0]["t"], sid, i) for i, r in enumerate(reqs)]
    stream.sort()
    return stream


def _idle_gap(prev: dict, cur: dict, gap_cap_s: float) -> float:
    """Return the capped idle seconds between ``prev`` finishing and ``cur``.

    Args:
        prev: earlier request (``t`` start seconds, ``api_time`` service).
        cur: later request.
        gap_cap_s: upper bound on a gap in seconds (user think time can run
            for hours; tool gaps are seconds).

    Returns:
        Idle seconds, clamped to ``[0, gap_cap_s]``.
    """
    idle = cur["t"] - prev["t"] - prev.get("api_time", 0.0)
    return min(max(idle, 0.0), gap_cap_s)


def _chain_gaps(reqs: list[dict], gap_cap_s: float) -> list[float]:
    """Return each request's capped gap after the previous one of its model.

    A session's main agent and its subagents use different models and hash
    namespaces, so a request's prefix producer is the previous request of the
    same model, not the previous request of the session.

    Args:
        reqs: one session's requests in time order.
        gap_cap_s: upper bound on a gap in seconds.

    Returns:
        Gap seconds per request; the first request of each model gets 0.
    """
    last: dict[str, dict] = {}
    gaps = []
    for r in reqs:
        prev = last.get(r["model"])
        gaps.append(0.0 if prev is None else _idle_gap(prev, r, gap_cap_s))
        last[r["model"]] = r
    return gaps


def _compress_timeline(reqs: list[dict], gap_cap_s: float) -> list[dict]:
    """Rewrite request offsets ``t`` with idle gaps capped at ``gap_cap_s``.

    Args:
        reqs: one session's requests in time order.
        gap_cap_s: upper bound on a gap in seconds.

    Returns:
        Copies of the requests with compressed ``t`` (service time kept).
    """
    out, t = [], 0.0
    for prev, r in zip([None, *reqs], reqs, strict=False):
        if prev is not None:
            t += prev.get("api_time", 0.0) + _idle_gap(prev, r, gap_cap_s)
        out.append({**r, "t": t})
    return out


def _round_robin_order(
    chains: list[list[dict]], live: int
) -> list[tuple[float, int, int]]:
    """Interleave ``live`` sessions one request per turn, refilling freed slots.

    Args:
        chains: per-session request lists.
        live: number of concurrently live sessions.

    Returns:
        ``(0.0, session, request index)`` in replay order (no arrival times).
    """
    pending = list(range(len(chains)))
    slots = [[pending.pop(0), 0] for _ in range(min(live, len(pending)))]
    order = []
    while slots:
        for slot in list(slots):
            sid, i = slot
            order.append((0.0, sid, i))
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
    real_time_gap_cap_s: float = 0.0,
    reasoning_strip_think_s: float = 0.0,
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
        real_time_gap_cap_s: when positive, emit real-time rows: idle gaps
            are capped at this many seconds, ``timestamp`` is the merged
            arrival in milliseconds (not ``index * step_ms``) and each row
            carries ``gap_ms``, its capped idle time after the session's
            previous request, for gap-paced closed-loop replay.
        reasoning_strip_think_s: when positive, model a template that strips
            past reasoning at user turns (``think_time`` above this many
            seconds): rewritten blocks get new IDs, and each user-turn row
            carries ``prewarm_tokens``, its prefix known when the previous
            chain ends.

    Returns:
        Mooncake rows (``timestamp``, ``input_length``, ``output_length``,
        ``hash_ids``, ``session_id``, ``gap_ms`` in real-time mode and
        ``prewarm_tokens`` on stripped user turns) in replay order.
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
    real_time = real_time_gap_cap_s > 0
    if real_time:
        if round_robin > 0:
            raise ValueError("real-time export merges timelines; drop round_robin")
        chains = [_compress_timeline(c, real_time_gap_cap_s) for c in chains]
    stream = (
        _round_robin_order(chains, round_robin)
        if round_robin > 0
        else _poisson_order(chains, concurrency, rng)
    )
    gaps = [_chain_gaps(c, real_time_gap_cap_s) for c in chains] if real_time else []
    strip = (
        [_strip_reasoning(c, reasoning_strip_think_s) for c in chains]
        if reasoning_strip_think_s > 0
        else []
    )
    table: dict = {}
    rows = []
    for index, (arrival, sid, i) in enumerate(stream):
        r = chains[sid][i]
        row = {
            "timestamp": round(arrival * 1000) if real_time else index * step_ms,
            "input_length": r["in"],
            "output_length": max(1, min(r["out"], max_output or r["out"])),
            "hash_ids": _block_ids(
                r["hash_ids"],
                r["in"],
                (sid, r["model"]),
                table,
                strip[sid][0][i] if strip else None,
            ),
            "session_id": sid,
        }
        if real_time:
            row["gap_ms"] = round(gaps[sid][i] * 1000)
        if strip and strip[sid][1][i]:
            row["prewarm_tokens"] = strip[sid][1][i]
        rows.append(row)
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
    p.add_argument(
        "--real-time-gap-cap-s",
        type=float,
        default=0.0,
        help="Emit real-time rows (ms timestamps + gap_ms) with gaps capped here.",
    )
    p.add_argument(
        "--reasoning-strip-think-s",
        type=float,
        default=0.0,
        help="Strip past reasoning at user turns idle longer than this (seconds).",
    )
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
        a.real_time_gap_cap_s,
        a.reasoning_strip_think_s,
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
