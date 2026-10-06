# SPDX-License-Identifier: Apache-2.0
"""Summarize a ``py-spy record -f raw`` profile by thread, self and inclusive time.

Usage::

    python -m benchmarks.research.pyspy_summary <profile.txt> [--match REGEX]

Each line of a raw (collapsed) profile is ``frame;frame;...;frame <count>``,
root first. The report lists sample shares per root (process/thread), the
top leaf (self) frames, and the top inclusive frames whose source path
matches ``--match`` (DaseR packages by default). Inclusive counts a frame at
most once per stack, so recursion does not inflate it.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
from typing import Iterable

DEFAULT_MATCH = r"\((?:server|ipc|transfer|cache|codec|connector)/"
_FRAME = re.compile(r"^(?P<name>.*?) \((?P<path>[^():]+):\d+\)$")


def frame_key(frame: str) -> str:
    """Return ``"<function> <path>"`` for a py-spy frame, dropping the line.

    Args:
        frame: One frame such as ``"run (transfer/iouring/l2_engine.py:88)"``.

    Returns:
        The frame without its line number, or the frame unchanged if it has
        no ``(path:line)`` suffix.
    """
    match = _FRAME.match(frame)
    return f"{match['name']} {match['path']}" if match else frame


def parse(lines: Iterable[str]) -> list[tuple[list[str], int]]:
    """Parse raw collapsed-stack lines into ``(frames, count)`` pairs.

    Args:
        lines: Lines of a ``py-spy record -f raw`` output.

    Returns:
        Root-first frame lists with their sample counts; blank or malformed
        lines are skipped.
    """
    stacks: list[tuple[list[str], int]] = []
    for line in lines:
        stack, _, count = line.rstrip("\n").rpartition(" ")
        if stack and count.isdigit():
            stacks.append((stack.split(";"), int(count)))
    return stacks


def within(
    stacks: list[tuple[list[str], int]], pattern: str
) -> list[tuple[list[str], int]]:
    """Keep the stacks that contain a frame matching ``pattern``.

    Args:
        stacks: Output of :func:`parse`.
        pattern: Regex searched in each frame.

    Returns:
        The matching stacks, unchanged.
    """
    regex = re.compile(pattern)
    return [(f, c) for f, c in stacks if any(regex.search(x) for x in f)]


def summarize(
    stacks: list[tuple[list[str], int]], match: str = DEFAULT_MATCH, top: int = 12
) -> list[str]:
    """Return report lines for parsed stacks.

    Args:
        stacks: Output of :func:`parse`.
        match: Regex a frame must match to appear in the inclusive list.
        top: Rows per section.

    Returns:
        Human-readable lines with sample shares.
    """
    total = sum(count for _, count in stacks)
    if not total:
        return ["no samples"]
    pattern = re.compile(match)
    roots: Counter[str] = Counter()
    leaves: Counter[str] = Counter()
    inclusive: Counter[str] = Counter()
    for frames, count in stacks:
        roots[frames[0][:80]] += count
        leaves[frame_key(frames[-1])] += count
        for key in {frame_key(f) for f in frames if pattern.search(f)}:
            inclusive[key] += count
    lines = [f"samples={total}"]
    for title, counter in (("root", roots), ("self", leaves), ("incl", inclusive)):
        for key, count in counter.most_common(top):
            lines.append(f"{title} {count / total:.3f} {key[:140]}")
    return lines


def main(argv: list[str] | None = None) -> None:
    """Print the summary of one raw py-spy profile.

    Args:
        argv: Optional argument list; defaults to ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--match", default=DEFAULT_MATCH)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--within",
        default=None,
        help="keep only stacks with a frame matching this regex (e.g. one "
        "thread's loop); shares are then relative to those stacks",
    )
    args = parser.parse_args(argv)
    # Native frames can carry non-UTF-8 symbol bytes.
    with args.profile.open(errors="replace") as handle:
        stacks = parse(handle)
    if args.within is not None:
        stacks = within(stacks, args.within)
    print("\n".join(summarize(stacks, args.match, args.top)))


if __name__ == "__main__":
    main()
