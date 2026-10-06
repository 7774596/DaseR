# SPDX-License-Identifier: Apache-2.0
"""Summarize GPU busy time, idle gaps, kernel mix and copy overlap from nsys.

Usage::

    python -m benchmarks.research.nsys_gpu_busy <report.sqlite> [...]

Each argument is an ``nsys export -t sqlite`` file. One block per report:
window length, GPU busy share (union of kernel intervals), idle gaps between
kernels (count and total above 1 ms / 10 ms), kernel-time shares by category,
the top kernels, and H2D/D2H bytes with the share of copy time that overlaps
a running kernel.

With ``--requests <requests.jsonl>`` the idle gaps above 1 ms are also split
by request phase: waiting for the first token (external load + prefill),
decoding, or no request in flight. ``--server <report.sqlite>`` adds how much
of each class the server process's H2D copies cover; it is aligned through
both sessions' ``TARGET_INFO_SESSION_START_TIME``. Request times are
``time.monotonic()`` seconds, converted with ``--mono-to-epoch-s`` (default:
this host's current ``time.time() - time.monotonic()``, valid on the host
that recorded the run, before a reboot).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Iterable, Sequence

# Ordered: the first matching pattern names the category.
CATEGORIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("attention", re.compile(r"flash|fmha|attn|attention|paged", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|cublas|xmma|nvjet|matmul|wgmma", re.I)),
    ("codec", re.compile(r"daser|codec|bitplane|unpack|decompress|plane", re.I)),
    ("copy", re.compile(r"copy|gather|scatter|reshape_and_cache|memcpy", re.I)),
    ("norm/act", re.compile(r"norm|silu|gelu|act_and_mul|rotary|rope", re.I)),
)
_COPY_KINDS = {1: "H2D", 2: "D2H"}
_PHASES = ("first_token", "decode", "no_request")


def merge(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping ``(start, end)`` intervals.

    Args:
        intervals: Intervals in any order.

    Returns:
        Disjoint intervals sorted by start.
    """
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap_ns(
    spans: Sequence[tuple[int, int]], busy: Sequence[tuple[int, int]]
) -> int:
    """Return the total time of ``spans`` covered by disjoint sorted ``busy``.

    Args:
        spans: Disjoint sorted intervals to measure.
        busy: Disjoint sorted covering intervals.

    Returns:
        Covered nanoseconds.
    """
    total, j = 0, 0
    for start, end in spans:
        while j < len(busy) and busy[j][1] <= start:
            j += 1
        k = j
        while k < len(busy) and busy[k][0] < end:
            total += min(end, busy[k][1]) - max(start, busy[k][0])
            k += 1
    return total


def category(name: str) -> str:
    """Return the kernel category of a demangled kernel name."""
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


def covered_ns(start: int, end: int, intervals: Sequence[tuple[int, int]]) -> int:
    """Return how much of ``[start, end)`` the disjoint ``intervals`` cover."""
    return sum(max(0, min(e, end) - max(s, start)) for s, e in intervals)


def phase_gaps(
    busy: Sequence[tuple[int, int]],
    first_token: Sequence[tuple[int, int]],
    decode: Sequence[tuple[int, int]],
    copies: Sequence[tuple[int, int]] = (),
    min_gap_ns: int = 1_000_000,
) -> dict[str, tuple[int, int, int]]:
    """Classify idle gaps between kernel intervals by request phase.

    Args:
        busy: Disjoint sorted kernel intervals.
        first_token: Disjoint sorted intervals with a request waiting for its
            first token (send to first token).
        decode: Disjoint sorted intervals with a request decoding.
        copies: Disjoint sorted copy intervals to measure inside each class.
        min_gap_ns: Only gaps longer than this are classified.

    Returns:
        ``{phase: (gap count, idle ns, copy ns inside)}`` for ``first_token``,
        ``decode`` and ``no_request``; a gap goes to the phase covering most
        of it.
    """
    out = {phase: [0, 0, 0] for phase in _PHASES}
    for prev, nxt in zip(busy, busy[1:], strict=False):
        start, end = prev[1], nxt[0]
        if end - start <= min_gap_ns:
            continue
        waiting = covered_ns(start, end, first_token)
        decoding = covered_ns(start, end, decode)
        idle = end - start - waiting - decoding
        if waiting >= max(decoding, idle):
            phase = "first_token"
        elif decoding >= idle:
            phase = "decode"
        else:
            phase = "no_request"
        out[phase][0] += 1
        out[phase][1] += end - start
        out[phase][2] += covered_ns(start, end, copies)
    return {phase: (n, ns, cp) for phase, (n, ns, cp) in out.items()}


def _copy_lines(
    copies: Sequence[tuple[int, int, int, int]], busy: Sequence[tuple[int, int]]
) -> list[str]:
    """Return one line per copy direction with bytes, time and kernel overlap.

    Args:
        copies: ``(start, end, bytes, copyKind)`` memcpy rows.
        busy: Disjoint sorted kernel intervals (may be empty).

    Returns:
        Report lines for the H2D and D2H copies present.
    """
    lines: list[str] = []
    for kind, label in _COPY_KINDS.items():
        rows = [(s, e, b) for s, e, b, k in copies if k == kind]
        if not rows:
            continue
        spans = merge((s, e) for s, e, _ in rows)
        copy_ns = sum(e - s for s, e in spans)
        lines.append(
            f"  {label}: n={len(rows)} GB={sum(b for _, _, b in rows) / 1e9:.1f} "
            f"time={copy_ns / 1e9:.2f}s overlap_with_kernels="
            f"{overlap_ns(spans, busy) / max(copy_ns, 1):.3f}"
        )
    return lines


def summarize(path: Path, top: int = 6) -> list[str]:
    """Return report lines for one nsys sqlite export.

    Args:
        path: ``nsys export -t sqlite`` output.
        top: Number of individual kernels to list.

    Returns:
        Human-readable lines.
    """
    con = sqlite3.connect(path)
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master")}
    # A process that only copies (e.g. the DaseR server) has no kernel table.
    kernels = (
        con.execute(
            "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName)"
        ).fetchall()
        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables
        else []
    )
    copies = (
        con.execute(
            "SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ).fetchall()
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables
        else []
    )
    con.close()
    busy = merge((start, end) for start, end, _ in kernels)
    if not busy:
        return [f"{path.name}: no kernels"] + _copy_lines(copies, busy)
    window = busy[-1][1] - busy[0][0]
    busy_ns = sum(end - start for start, end in busy)
    gaps = [b[0] - a[1] for a, b in zip(busy, busy[1:], strict=False)]
    big = [g for g in gaps if g > 1_000_000]
    huge = [g for g in gaps if g > 10_000_000]
    lines = [
        f"{path.name}: window={window / 1e9:.1f}s busy={busy_ns / window:.3f} "
        f"idle={1 - busy_ns / window:.3f} gaps>1ms n={len(big)} "
        f"sum={sum(big) / 1e9:.2f}s gaps>10ms n={len(huge)} "
        f"sum={sum(huge) / 1e9:.2f}s"
    ]
    by_cat: dict[str, int] = {}
    by_name: dict[str, int] = {}
    kernel_ns = 0
    for start, end, name in kernels:
        span = end - start
        kernel_ns += span
        by_cat[category(name)] = by_cat.get(category(name), 0) + span
        by_name[name] = by_name.get(name, 0) + span
    lines.append(
        "  kernel share: "
        + " ".join(
            f"{label}={ns / kernel_ns:.3f}"
            for label, ns in sorted(by_cat.items(), key=lambda kv: -kv[1])
        )
    )
    for name, ns in sorted(by_name.items(), key=lambda kv: -kv[1])[:top]:
        lines.append(f"    {ns / kernel_ns:.3f} {category(name):9s} {name[:110]}")
    return lines + _copy_lines(copies, busy)


def _session_start_ns(path: Path) -> int:
    """Return a report's session start as epoch nanoseconds."""
    con = sqlite3.connect(path)
    start = con.execute(
        "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
    ).fetchone()[0]
    con.close()
    return int(start)


def _intervals(path: Path, query: str) -> list[tuple[int, int]]:
    """Return merged ``(start, end)`` rows of ``query`` (empty if no table)."""
    con = sqlite3.connect(path)
    try:
        rows = con.execute(query).fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    return merge((int(s), int(e)) for s, e in rows)


def request_phase_lines(
    report: Path,
    requests: Path,
    server: Path | None,
    mono_to_epoch_s: float,
) -> list[str]:
    """Return idle-gap lines split by request phase for one vLLM report.

    Args:
        report: vLLM process ``nsys export -t sqlite`` output.
        requests: Replay ``requests.jsonl`` with ``sent_at_s``,
            ``first_token_at_s`` (monotonic seconds) and ``latency_ms``.
        server: Optional server-process report whose H2D copies are measured
            inside each gap class.
        mono_to_epoch_s: ``time.time() - time.monotonic()`` of the recording
            host.

    Returns:
        One line per phase plus the window length.
    """
    origin = _session_start_ns(report)
    busy = _intervals(report, "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL")
    if not busy:
        return [f"{report.name}: no kernels"]

    def to_ns(mono_s: float) -> int:
        return int((mono_s + mono_to_epoch_s) * 1e9) - origin

    first_token: list[tuple[int, int]] = []
    decode: list[tuple[int, int]] = []
    with requests.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("first_token_at_s") is None:
                continue
            sent, first = record["sent_at_s"], record["first_token_at_s"]
            first_token.append((to_ns(sent), to_ns(first)))
            decode.append((to_ns(first), to_ns(sent + record["latency_ms"] / 1e3)))
    copies: list[tuple[int, int]] = []
    if server is not None:
        shift = _session_start_ns(server) - origin
        copies = [
            (s + shift, e + shift)
            for s, e in _intervals(
                server,
                "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = 1",
            )
        ]
    window = busy[-1][1] - busy[0][0]
    lines = [f"  request phases (gaps >1 ms, window {window / 1e9:.1f}s):"]
    for phase, (count, idle, copy) in phase_gaps(
        busy, merge(first_token), merge(decode), copies
    ).items():
        lines.append(
            f"    {phase:11s} n={count} idle={idle / 1e9:.2f}s "
            f"share={idle / window:.3f} server_h2d_inside={copy / 1e9:.2f}s"
        )
    return lines


def main(argv: list[str] | None = None) -> None:
    """Print one summary block per sqlite report.

    Args:
        argv: Optional argument list; defaults to ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--requests", type=Path, default=None)
    parser.add_argument("--server", type=Path, default=None)
    parser.add_argument(
        "--mono-to-epoch-s", type=float, default=time.time() - time.monotonic()
    )
    args = parser.parse_args(argv)
    for report in args.reports:
        print("\n".join(summarize(report, args.top)))
        if args.requests is not None and report != args.server:
            print(
                "\n".join(
                    request_phase_lines(
                        report, args.requests, args.server, args.mono_to_epoch_s
                    )
                )
            )


if __name__ == "__main__":
    main()
