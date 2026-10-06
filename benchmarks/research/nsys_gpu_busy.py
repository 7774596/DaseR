# SPDX-License-Identifier: Apache-2.0
"""Summarize GPU busy time, idle gaps, kernel mix and copy overlap from nsys.

Usage::

    python -m benchmarks.research.nsys_gpu_busy <report.sqlite> [...]

Each argument is an ``nsys export -t sqlite`` file. One block per report:
window length, GPU busy share (union of kernel intervals), idle gaps between
kernels (count and total above 1 ms / 10 ms), kernel-time shares by category,
the top kernels, and H2D/D2H bytes with the share of copy time that overlaps
a running kernel.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sqlite3
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


def main(argv: list[str] | None = None) -> None:
    """Print one summary block per sqlite report.

    Args:
        argv: Optional argument list; defaults to ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args(argv)
    for report in args.reports:
        print("\n".join(summarize(report, args.top)))


if __name__ == "__main__":
    main()
