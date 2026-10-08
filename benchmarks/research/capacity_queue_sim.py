# SPDX-License-Identifier: Apache-2.0
"""Closed-population GPU KV capacity queue simulator (ranking only).

Replays a session trace as a closed population of ``K`` live sessions against
one engine whose GPU KV cache holds ``capacity`` tokens. An admitted request
holds ``input + output`` tokens until it completes; its prefill (only the
uncached tokens) runs on one serial prefill server at ``prefill_tok_s``, then
it decodes ``output`` tokens at ``tpot_s`` each. A session's next request
becomes ready ``gap_ms`` after the previous one completes; when a session's
last request completes, the next trace session starts.

Waiting requests are admitted head-of-line in policy order while they fit:
``fcfs`` (ready time, vLLM's default), ``srpt`` (shortest prefill + decode
first) or ``small`` (fewest KV tokens first). Comparing policies bounds how
much of the measured capacity queue admission reordering could remove; the
``--hit`` sweep shows how much the prefix hit rate of an external KV tier
shortens prefill under the same capacity. Results only rank candidates: the
model ignores chunked-prefill interleaving and decode slowdown with batch.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import heapq
import json
from pathlib import Path

from benchmarks.mooncake.benchmark import TraceRequest, load_trace

POLICIES = ("fcfs", "srpt", "small")


@dataclass
class SimJob:
    """One request in the simulation.

    Args:
        index: Trace row index.
        session: Position of the owning session in start order.
        kv_tokens: GPU KV tokens held while admitted.
        prefill_s: Serial prefill time of the uncached tokens.
        decode_s: Decode time of the output tokens.
    """

    index: int
    session: int
    kv_tokens: int
    prefill_s: float
    decode_s: float
    gap_s: float
    ready_s: float = 0.0
    admit_s: float = 0.0
    first_token_s: float = 0.0
    done_s: float = 0.0


def build_sessions(
    requests: list[TraceRequest],
    hit: float | None,
    prefill_tok_s: float,
    tpot_s: float,
) -> list[list[SimJob]]:
    """Group trace rows into per-session job lists.

    Args:
        requests: Trace rows in source order.
        hit: Target prefix hit rate; ``None`` assumes every request reuses its
            session's previous context exactly. Otherwise the reused part of
            each prompt is scaled uniformly so the overall hit rate is ``hit``.
        prefill_tok_s: Serial prefill throughput in uncached tokens/s.
        tpot_s: Per-output-token decode time.

    Returns:
        Sessions in first-appearance order, each a list of jobs in trace order.
    """
    rows: dict[int, list[TraceRequest]] = defaultdict(list)
    for request in requests:
        key = request.session_id if request.session_id is not None else -request.index
        rows[key].append(request)
    reuse: dict[int, int] = {}
    for group in rows.values():
        context = 0
        for request in group:
            reuse[request.index] = min(context, request.input_length)
            context = request.input_length + request.output_length
    total = sum(r.input_length for r in requests)
    reused = sum(reuse.values())
    scale = 1.0 if hit is None or reused == 0 else min(hit * total / reused, 1.0)
    sessions = []
    for position, group in enumerate(rows.values()):
        jobs = []
        for request in group:
            uncached = request.input_length - scale * reuse[request.index]
            jobs.append(
                SimJob(
                    index=request.index,
                    session=position,
                    kv_tokens=request.input_length + request.output_length,
                    prefill_s=uncached / prefill_tok_s,
                    decode_s=request.output_length * tpot_s,
                    gap_s=(request.gap_ms or 0.0) / 1000.0,
                )
            )
        sessions.append(jobs)
    return sessions


def simulate(
    sessions: list[list[SimJob]],
    population: int,
    capacity: int,
    policy: str,
    ramp_s: float = 0.0,
    max_requests: int = 0,
) -> list[SimJob]:
    """Run the closed population and return completed jobs.

    Args:
        sessions: Output of :func:`build_sessions` (mutated in place).
        population: Live sessions ``K``.
        capacity: GPU KV capacity in tokens; a job larger than it is admitted
            alone.
        policy: One of :data:`POLICIES`.
        ramp_s: The first ``population`` sessions start evenly over this span.
        max_requests: Stop after this many completions (0 = all).

    Returns:
        Completed jobs in completion order.

    Thread-safety:
        Pure computation over the given objects.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    events: list[tuple[float, int, str, int]] = []  # (time, seq, kind, session)
    seq = 0
    cursor = [0] * len(sessions)
    next_session = min(population, len(sessions))
    for position in range(next_session):
        start = ramp_s * position / max(population, 1)
        heapq.heappush(events, (start, seq, "ready", position))
        seq += 1
    waiting: list[SimJob] = []
    used = 0
    prefill_free = 0.0
    done: list[SimJob] = []

    def key(job: SimJob) -> tuple[float, int]:
        if policy == "srpt":
            return (job.prefill_s + job.decode_s, job.index)
        if policy == "small":
            return (float(job.kv_tokens), job.index)
        return (job.ready_s, job.index)

    while events:
        now, _, kind, position = heapq.heappop(events)
        if kind == "ready":
            job = sessions[position][cursor[position]]
            job.ready_s = now
            waiting.append(job)
        else:
            job = sessions[position][cursor[position]]
            used -= job.kv_tokens
            done.append(job)
            if max_requests and len(done) >= max_requests:
                break
            cursor[position] += 1
            if cursor[position] < len(sessions[position]):
                gap = sessions[position][cursor[position]].gap_s
                heapq.heappush(events, (now + gap, seq, "ready", position))
                seq += 1
            elif next_session < len(sessions):
                heapq.heappush(events, (now, seq, "ready", next_session))
                seq += 1
                next_session += 1
        waiting.sort(key=key)
        while waiting and (used + waiting[0].kv_tokens <= capacity or used == 0):
            job = waiting.pop(0)
            used += job.kv_tokens
            job.admit_s = now
            prefill_free = max(prefill_free, now) + job.prefill_s
            job.first_token_s = prefill_free
            job.done_s = job.first_token_s + job.decode_s
            heapq.heappush(events, (job.done_s, seq, "done", job.session))
            seq += 1
    return done


def summarise(jobs: list[SimJob], trim_frac: float = 0.0) -> dict[str, float]:
    """Return queue and TTFT sums over the jobs after a warmup trim.

    Args:
        jobs: Completed jobs in completion order.
        trim_frac: Leading fraction of completions dropped as warmup.

    Returns:
        Request count, summed TTFT and admission queue seconds, their ratio,
        and the TTFT p90.
    """
    kept = jobs[int(len(jobs) * trim_frac) :]
    ttft = sorted(j.first_token_s - j.ready_s for j in kept)
    queue = sum(j.admit_s - j.ready_s for j in kept)
    total = sum(ttft)
    return {
        "requests": len(kept),
        "sum_ttft_s": total,
        "sum_queue_s": queue,
        "q_over_ttft": queue / total if total else 0.0,
        "ttft_p90_s": ttft[int(0.9 * len(ttft))] if ttft else 0.0,
    }


def main(argv: list[str] | None = None) -> None:
    """Print one JSON line per (hit rate, policy) cell.

    Args:
        argv: Arguments without the program name; ``None`` reads ``sys.argv``.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--population", type=int, required=True)
    p.add_argument("--capacity-tokens", type=int, required=True)
    p.add_argument("--prefill-tok-s", type=float, required=True)
    p.add_argument("--tpot-ms", type=float, required=True)
    p.add_argument("--hit", type=float, nargs="+", default=[0.815, 0.95])
    p.add_argument("--policies", nargs="+", default=list(POLICIES))
    p.add_argument("--ramp-s", type=float, default=120.0)
    p.add_argument("--max-requests", type=int, default=3000)
    p.add_argument("--trim-frac", type=float, default=0.2)
    args = p.parse_args(argv)
    requests = load_trace(args.trace)
    for hit in args.hit:
        for policy in args.policies:
            sessions = build_sessions(
                requests, hit, args.prefill_tok_s, args.tpot_ms / 1000.0
            )
            jobs = simulate(
                sessions,
                args.population,
                args.capacity_tokens,
                policy,
                args.ramp_s,
                args.max_requests,
            )
            row = {"hit": hit, "policy": policy, **summarise(jobs, args.trim_frac)}
            print(
                json.dumps(
                    {
                        k: round(v, 3) if isinstance(v, float) else v
                        for k, v in row.items()
                    }
                )
            )


if __name__ == "__main__":
    main()
