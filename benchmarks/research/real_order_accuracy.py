# SPDX-License-Identifier: Apache-2.0
"""Offline next-session prediction accuracy on cc-traces real arrival order.

Sessions start at Poisson offsets chosen so that about ``--concurrency``
sessions overlap; within a session the real request offsets ``t`` are kept.
The merged stream is fed to ``SessionPredictor`` (hash ids packed as tokens),
and we report how often the next request's session was among the predicted
``depth`` sessions, counting only requests that continue a known session.
"""

import argparse
import json
import random
import struct

from daser.server.session_predictor import SessionPredictor


class TimePredictor:
    """Rank other live sessions by last arrival + own gap EWMA (past info only)."""

    def __init__(self, stale: float = 3.0) -> None:
        self.last: dict[int, float] = {}
        self.gap: dict[int, float] = {}
        self.stale = stale

    def observe(self, t: float, sid: int) -> None:
        """Record an arrival of ``sid`` at time ``t``."""
        if sid in self.last:
            g = t - self.last[sid]
            self.gap[sid] = (
                g if sid not in self.gap else self.gap[sid] + 0.25 * (g - self.gap[sid])
            )
        self.last[sid] = t

    def predict(self, t: float, current: int, depth: int) -> set[int]:
        """Return the ``depth`` sessions expected back soonest after ``t``."""
        known = [g for g in self.gap.values()]
        default = sorted(known)[len(known) // 2] if known else 0.0
        cands = []
        for sid, last in self.last.items():
            if sid == current:
                continue
            g = self.gap.get(sid, default)
            if t - last > self.stale * max(g, 1e-3) + 60.0:
                continue
            cands.append((max(last + g, t), sid))
        cands.sort()
        return {sid for _eta, sid in cands[:depth]}


def main() -> None:
    """Parse arguments, build the merged stream and print accuracy rows."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--sessions", type=int, default=64)
    ap.add_argument("--concurrency", type=float, nargs="+", default=[4, 8, 16])
    ap.add_argument("--min-ctx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sessions = []
    with open(args.path) as f:
        for line in f:
            s = json.loads(line)
            # Subagents run inside the parent session on the same clock; each
            # is its own prompt chain, so it becomes its own pseudo-session.
            chains = [[r for r in s["requests"] if "in" in r]]
            chains += [r["requests"] for r in s["requests"] if "requests" in r]
            group = []
            for chain in chains:
                reqs = [r for r in chain if "in" in r and r["in"] >= args.min_ctx]
                if len(reqs) >= 2:
                    group.append(reqs)
            if group:
                sessions.append(group)
    rng = random.Random(args.seed)
    rng.shuffle(sessions)
    sessions = sessions[: args.sessions]
    mean_span = sum(g[0][-1]["t"] - g[0][0]["t"] for g in sessions) / len(sessions)
    for conc in args.concurrency:
        # Poisson session starts: rate so that `conc` sessions overlap on average.
        rate = conc / mean_span
        t0, stream = 0.0, []
        sid = 0
        for group in sessions:
            t0 += rng.expovariate(rate)
            base = group[0][0]["t"]
            for reqs in group:
                for r in reqs:
                    stream.append((t0 + r["t"] - base, sid, r["hash_ids"]))
                sid += 1
        stream.sort(key=lambda x: x[0])
        for depth in (1, 2, 3, 4):
            pred = SessionPredictor()
            seen: dict[int, bool] = {}
            last_pred: set[int] = set()
            ids: dict[int, int] = {}
            hits = cont = switch = switch_hits = 0
            prev_sid = None
            for _t, sid, hashes in stream:
                # hash_id_scope is local: make ids unique per session.
                tokens = struct.pack(
                    f"{len(hashes)}q", *((sid << 40) | h for h in hashes)
                )
                if sid in seen:
                    cont += 1
                    hit = sid in last_pred
                    hits += hit
                    if sid != prev_sid:
                        switch += 1
                        switch_hits += hit
                seen[sid] = True
                if not pred.observe(tokens, "m"):
                    continue
                predicted = pred.predict(depth)
                # Map predicted prompts back to sessions by exact prompt match.
                ids[hash(tokens)] = sid
                last_pred = {ids.get(hash(p), -1) for p, _m in predicted}
                prev_sid = sid
            tp = TimePredictor()
            t_last: set[int] = set()
            t_hits = t_switch_hits = 0
            seen_t: set[int] = set()
            prev = None
            for t, sid, _hashes in stream:
                if sid in seen_t:
                    hit = sid in t_last
                    t_hits += hit
                    if sid != prev:
                        t_switch_hits += hit
                seen_t.add(sid)
                tp.observe(t, sid)
                t_last = tp.predict(t, sid, depth)
                prev = sid
            print(
                f"conc={conc:g} depth={depth} "
                f"time_acc_switch={t_switch_hits / max(switch, 1):.3f}"
            )
            print(
                f"conc={conc:g} depth={depth} requests={len(stream)} "
                f"continuing={cont} acc={hits / max(cont, 1):.3f} "
                f"switches={switch} acc_switch={switch_hits / max(switch, 1):.3f}"
            )


if __name__ == "__main__":
    main()
