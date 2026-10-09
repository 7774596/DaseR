# SPDX-License-Identifier: Apache-2.0

"""Sample DaseR tier counters during a replay and attribute them per request.

``CounterTrace`` polls a DaseR ``/metrics`` endpoint on a background thread and
appends ``{"t": perf_counter, "c": {name: value}}`` lines for the tier
counters only. With serial replay the counter deltas between the sample
before a request is sent and the first sample after its first token are that
request's load traffic (lookup readahead included: it reads the request's own
L2 ranges), which ``classify`` turns into the tier that served the request's
external hit: ``none``, ``l1``, ``l2`` or ``mixed``. Windows that overlap
another request's load phase, or in which readahead bytes were dropped
unused, are flagged so callers can exclude them. Timestamps use
``time.perf_counter`` like the replay's ``sent_at_s``; both are the
system-wide monotonic clock on Linux.

Usage::

    python -m benchmarks.research.counter_trace requests.jsonl counters.jsonl
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
import threading
import time
from typing import Any
import urllib.request

#: Counter families the trace keeps (series of other families are dropped).
TIER_COUNTERS = (
    "daser_l1_hits_total",
    "daser_l1_misses_total",
    "daser_l2_read_bytes_total",
    "daser_readahead_bytes_total",
    "daser_transfer_bytes_total",
)
_L1_HITS = "daser_l1_hits_total"
_RA_DROPPED = 'daser_readahead_bytes_total{outcome="dropped"}'


def _parse(text: str) -> dict[str, float]:
    """Return the ``TIER_COUNTERS`` series of a Prometheus text exposition."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith(TIER_COUNTERS):
            continue
        key, _, value = line.rpartition(" ")
        try:
            values[key] = float(value)
        except ValueError:
            continue
    return values


def _ssd_read_bytes(counters: dict[str, float]) -> float:
    """Sum L2 read bytes over the SSD contention buckets of one sample."""
    return sum(
        value
        for key, value in counters.items()
        if key.startswith("daser_l2_read_bytes_total")
    )


class CounterTrace:
    """Background poller of DaseR tier counters.

    Args:
        url: DaseR base URL (``/metrics`` is appended).
        path: JSON-lines output file, truncated on start.
        interval_s: seconds between polls.

    Thread-safety:
        ``start`` and ``stop`` are called from one thread; the poller thread
        owns the output file until ``stop`` returns.
    """

    def __init__(self, url: str, path: Path, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._url = f"{url}/metrics"
        self._path = path
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Start polling on the background thread."""
        self._thread.start()

    def stop(self) -> None:
        """Take a final sample and stop polling."""
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        """Poll until stopped; failed polls are skipped."""
        with self._path.open("w") as out:
            while True:
                stopping = self._stop.is_set()
                try:
                    with urllib.request.urlopen(self._url, timeout=10) as response:
                        text = response.read().decode()
                except OSError:
                    text = ""
                if text:
                    sample = {"t": time.perf_counter(), "c": _parse(text)}
                    out.write(json.dumps(sample) + "\n")
                    out.flush()
                if stopping:
                    return
                self._stop.wait(self._interval_s)


def classify(
    requests: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    l2_bytes_per_token: float | None = None,
    l2_share: float = 0.8,
) -> list[dict[str, Any]]:
    """Attribute counter deltas to each request and name its serving tier.

    A request's window runs from the last sample at or before ``sent_at_s``
    to the first sample at or after ``first_token_at_s`` (the load finishes
    before the first token; decode moves no tier counters).

    Args:
        requests: replay records with ``sample_id``, ``sent_at_s`` and
            ``first_token_at_s``.
        samples: ``CounterTrace`` lines sorted by ``t``.

    Returns:
        One dict per request with a sample bracketing it: ``sample_id``,
        ``l1_hits``, ``ssd_read_bytes``, ``tier`` (``none`` when neither
        moved, ``l1``/``l2`` when only one did, ``mixed`` when both did) and
        ``clean``: false when the window overlaps another request's
        ``[sent_at_s, first_token_at_s]`` or readahead bytes were dropped.

    With ``l2_bytes_per_token`` (stored L2 bytes per token, compression
    included) the tier follows the byte share instead: ``ssd_share`` is SSD
    bytes over ``prompt_tokens * l2_bytes_per_token``; ``l2`` when it is at
    least ``l2_share``, ``l1`` when L1 hit and it is at most
    ``1 - l2_share``, else ``mixed``. L1 hit counts are chunks, not bytes,
    and a chunk fetched from SSD may also count as an L1 hit once promoted,
    so "both moved" alone does not mean a mixed load.
    """
    times = [sample["t"] for sample in samples]
    phases = sorted(
        (float(r["sent_at_s"]), float(r["first_token_at_s"]), r["sample_id"])
        for r in requests
    )
    rows = []
    for record in requests:
        sent = float(record["sent_at_s"])
        first = float(record["first_token_at_s"])
        lo = bisect.bisect_right(times, sent) - 1
        hi = bisect.bisect_left(times, first)
        if lo < 0 or hi >= len(samples):
            continue
        start, end = times[lo], times[hi]
        overlap = any(
            other != record["sample_id"] and o_sent < end and o_first > start
            for o_sent, o_first, other in phases
        )
        before, after = samples[lo]["c"], samples[hi]["c"]
        dropped = after.get(_RA_DROPPED, 0.0) - before.get(_RA_DROPPED, 0.0)
        hits = after.get(_L1_HITS, 0.0) - before.get(_L1_HITS, 0.0)
        ssd = _ssd_read_bytes(after) - _ssd_read_bytes(before)
        tier = {
            (False, False): "none",
            (True, False): "l1",
            (False, True): "l2",
            (True, True): "mixed",
        }[(hits > 0, ssd > 0)]
        row = {
            "sample_id": record["sample_id"],
            "l1_hits": hits,
            "ssd_read_bytes": ssd,
            "tier": tier,
            "clean": not overlap and dropped <= 0,
        }
        if l2_bytes_per_token is not None and tier != "none":
            share = ssd / (float(record["prompt_tokens"]) * l2_bytes_per_token)
            row["ssd_share"] = share
            if share >= l2_share:
                row["tier"] = "l2"
            elif hits > 0 and share <= 1.0 - l2_share:
                row["tier"] = "l1"
            else:
                row["tier"] = "mixed"
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    """Print per-request tiers as JSON lines for a replay and its trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("counters", type=Path)
    parser.add_argument(
        "--l2-bytes-per-token",
        type=float,
        help="Stored L2 bytes per token; tiers then follow the SSD byte share.",
    )
    parser.add_argument("--l2-share", type=float, default=0.8)
    args = parser.parse_args(argv)
    requests = [
        json.loads(line) for line in args.requests.read_text().splitlines() if line
    ]
    samples = [
        json.loads(line) for line in args.counters.read_text().splitlines() if line
    ]
    ordered = sorted(samples, key=lambda s: s["t"])
    for row in classify(requests, ordered, args.l2_bytes_per_token, args.l2_share):
        print(json.dumps(row))


if __name__ == "__main__":
    main()
