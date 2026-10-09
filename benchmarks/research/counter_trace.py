# SPDX-License-Identifier: Apache-2.0

"""Sample DaseR tier counters during a replay and attribute them per request.

``CounterTrace`` polls a DaseR ``/metrics`` endpoint on a background thread and
appends ``{"t": perf_counter, "c": {name: value}}`` lines for the tier
counters only. With serial replay (one request in flight) the counter deltas
between the sample before a request is sent and the first sample after it
completes are that request's tier traffic, which ``classify`` turns into the
tier that served the request's external hit: ``none``, ``l1``, ``l2`` or
``mixed``. Timestamps use ``time.perf_counter`` like the replay's
``sent_at_s``; both are the system-wide monotonic clock on Linux.

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
    requests: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attribute counter deltas to each request and name its serving tier.

    A request's window runs from the last sample at or before ``sent_at_s``
    to the first sample at or after its completion. The window is only
    exclusive to the request when requests do not overlap, i.e. serial replay.

    Args:
        requests: replay records with ``sample_id``, ``sent_at_s`` and
            ``latency_ms``.
        samples: ``CounterTrace`` lines sorted by ``t``.

    Returns:
        One dict per request with a sample bracketing it: ``sample_id``,
        ``l1_hits``, ``ssd_read_bytes`` and ``tier`` (``none`` when neither
        moved, ``l1``/``l2`` when only one did, ``mixed`` when both did).
    """
    times = [sample["t"] for sample in samples]
    rows = []
    for record in requests:
        sent = float(record["sent_at_s"])
        done = sent + float(record["latency_ms"]) / 1000.0
        lo = bisect.bisect_right(times, sent) - 1
        hi = bisect.bisect_left(times, done)
        if lo < 0 or hi >= len(samples):
            continue
        before, after = samples[lo]["c"], samples[hi]["c"]
        hits = after.get(_L1_HITS, 0.0) - before.get(_L1_HITS, 0.0)
        ssd = _ssd_read_bytes(after) - _ssd_read_bytes(before)
        tier = {
            (False, False): "none",
            (True, False): "l1",
            (False, True): "l2",
            (True, True): "mixed",
        }[(hits > 0, ssd > 0)]
        rows.append(
            {
                "sample_id": record["sample_id"],
                "l1_hits": hits,
                "ssd_read_bytes": ssd,
                "tier": tier,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    """Print per-request tiers as JSON lines for a replay and its trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("counters", type=Path)
    args = parser.parse_args(argv)
    requests = [
        json.loads(line) for line in args.requests.read_text().splitlines() if line
    ]
    samples = [
        json.loads(line) for line in args.counters.read_text().splitlines() if line
    ]
    for row in classify(requests, sorted(samples, key=lambda s: s["t"])):
        print(json.dumps(row))


if __name__ == "__main__":
    main()
