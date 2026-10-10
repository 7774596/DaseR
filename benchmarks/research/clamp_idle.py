# SPDX-License-Identifier: Apache-2.0
"""Clamp whole-system idle stretches of a real-time trace.

In real-time replay a session's first request is paced by its ``timestamp``
and every later request by its ``gap_ms`` after its dependency completes, so a
stretch with nothing in flight is set by the next session start's timestamp.
Given reference runs' ``requests.jsonl`` (send/completion offsets; pass the
slowest arms so a stretch is idle in all of them), every idle stretch longer
than ``--max-idle-s`` is shortened to that length by
shifting all later timestamps back by the excess. Rows paced by ``gap_ms``
keep their gaps; arrival order and every in-flight overlap are unchanged
because only time with zero requests in flight is removed.
"""

import argparse
import json


def idle_cuts(
    intervals: list[tuple[float, float]], max_idle_s: float
) -> list[tuple[float, float]]:
    """Return ``(start, excess)`` for each idle stretch longer than the cap.

    Args:
        intervals: ``(send, completion)`` offsets in seconds of every request.
        max_idle_s: Longest idle stretch to keep.

    Returns:
        One ``(start_s, excess_s)`` per stretch with zero requests in flight
        lasting more than ``max_idle_s``, in time order; ``excess_s`` is the
        part to remove.
    """
    cuts = []
    ordered = sorted(intervals)
    busy_until = ordered[0][1]
    for send, done in ordered[1:]:
        if send - busy_until > max_idle_s:
            cuts.append((busy_until, send - busy_until - max_idle_s))
        busy_until = max(busy_until, done)
    return cuts


def clamp_rows(
    rows: list[dict], cuts: list[tuple[float, float]], send_offsets: list[float]
) -> list[dict]:
    """Shift trace timestamps back by the idle excess that precedes each send.

    Gap-paced rows are sent before their recorded timestamp, so the reference
    send offset, not the timestamp, decides which side of a cut a row is on.

    Args:
        rows: Trace rows with ms ``timestamp`` in replay order.
        cuts: ``(start_s, excess_s)`` from :func:`idle_cuts`.
        send_offsets: Reference send offset in seconds of each row.

    Returns:
        New rows; a row sent after a cut's start moves back by that cut's
        excess (cumulatively over all earlier cuts).
    """
    out = []
    for row, sent in zip(rows, send_offsets, strict=True):
        shift_s = sum(excess for start, excess in cuts if sent > start)
        out.append(dict(row, timestamp=round(row["timestamp"] - shift_s * 1000.0)))
    return out


def main() -> None:
    """Write the clamped trace and report the idle time removed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument(
        "--reference", required=True, nargs="+", help="requests.jsonl of runs"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-idle-s", type=float, default=60.0)
    args = parser.parse_args()
    with open(args.trace) as f:
        rows = [json.loads(line) for line in f]
    intervals = []
    sends: dict[int, float] = {}
    for path in args.reference:
        with open(path) as f:
            ref = [json.loads(line) for line in f]
        intervals += [(r["send_offset_s"], r["completion_offset_s"]) for r in ref]
        for r in ref:
            i = r["trace_index"]
            sends[i] = min(sends.get(i, r["send_offset_s"]), r["send_offset_s"])
    cuts = idle_cuts(intervals, args.max_idle_s)
    count = len(sends)
    if sorted(sends) != list(range(count)):
        raise ValueError("references must cover a leading range of trace rows")
    clamped = clamp_rows(rows[:count], cuts, [sends[i] for i in range(count)])
    with open(args.output, "w") as f:
        for row in clamped:
            f.write(json.dumps(row) + "\n")
    removed = sum(excess for _, excess in cuts)
    print(
        f"rows {len(clamped)} cuts {len(cuts)} idle removed {removed:.0f} s "
        f"(cap {args.max_idle_s:g} s)"
    )


if __name__ == "__main__":
    main()
