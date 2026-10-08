# SPDX-License-Identifier: Apache-2.0

"""Opt-in worker-side timeline of async external KV loads.

Enabled by the scheduler probe's ``DASER_SCHED_PROBE_INTERVAL_S``; the probe
then logs cumulative totals at that interval. For each async load it splits
the worker's share of the scheduler-side ``load_wait`` into:

- io: from ``start_load_kv`` handing the load to the load thread until its
  future completes (dispatcher queueing, IPC, server transfer, restore).
- report: from that completion until the next ``get_finished`` poll returns
  the request as ``finished_recving`` (step cadence on the worker).

It also reports the mean interval between ``get_finished`` polls (one per
model step), so the remainder of the scheduler-side wait (allocation to
``start_load_kv`` and ``finished_recving`` to the scheduler seeing it) can be
read in steps.
"""

from __future__ import annotations

from collections.abc import Iterable
import os
import threading
import time

from daser.logging import init_logger

logger = init_logger(__name__)

PROBE_INTERVAL_ENV = "DASER_SCHED_PROBE_INTERVAL_S"


class LoadTimeline:
    """Accumulate per-load io / report times and log them periodically.

    Async/thread-safety:
        :meth:`on_start` and :meth:`on_collect` run on the vLLM worker thread;
        :meth:`on_done` runs on the load thread (future callback). A lock
        guards the shared state.
    """

    def __init__(self, interval_s: float) -> None:
        """Create a timeline that logs every ``interval_s`` seconds.

        Args:
            interval_s: Logging interval in seconds (must be positive).
        """
        self._interval_s = interval_s
        self._next_log = time.monotonic() + interval_s
        self._lock = threading.Lock()
        self._started: dict[str, float] = {}
        self._done: dict[str, float] = {}
        self._loads = 0
        self._io_s = 0.0
        self._io_max = 0.0
        self._report_s = 0.0
        self._report_max = 0.0
        self._polls = 0
        self._poll_span_s = 0.0
        self._last_poll: float | None = None

    @classmethod
    def from_env(cls) -> LoadTimeline | None:
        """Build a timeline from ``DASER_SCHED_PROBE_INTERVAL_S``.

        Returns:
            A timeline when the variable is a positive float, else None.
        """
        try:
            interval_s = float(os.environ.get(PROBE_INTERVAL_ENV, "0"))
        except ValueError:
            return None
        return cls(interval_s) if interval_s > 0 else None

    def on_start(self, req_ids: Iterable[str], now: float | None = None) -> None:
        """Record that loads were handed to the load thread.

        Args:
            req_ids: Base request IDs of the started loads.
            now: Monotonic timestamp; defaults to the current time.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            for req_id in req_ids:
                self._started.setdefault(req_id, now)

    def on_done(self, req_ids: Iterable[str], now: float | None = None) -> None:
        """Record that the loads' futures completed.

        Args:
            req_ids: Base request IDs sharing the completed future.
            now: Monotonic timestamp; defaults to the current time.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            for req_id in req_ids:
                if req_id in self._started:
                    self._done.setdefault(req_id, now)

    def on_collect(self, finished: Iterable[str], now: float | None = None) -> None:
        """Record one ``get_finished`` poll and the loads it reported.

        Args:
            finished: Base request IDs reported as ``finished_recving``.
            now: Monotonic timestamp; defaults to the current time.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._last_poll is not None:
                self._polls += 1
                self._poll_span_s += now - self._last_poll
            self._last_poll = now
            for req_id in finished:
                started = self._started.pop(req_id, None)
                done = self._done.pop(req_id, None)
                if started is None:
                    continue
                done = now if done is None else done
                io, report = done - started, now - done
                self._loads += 1
                self._io_s += io
                self._io_max = max(self._io_max, io)
                self._report_s += report
                self._report_max = max(self._report_max, report)
            line = self._summary() if now >= self._next_log else None
            if line is not None:
                self._next_log = now + self._interval_s
        if line is not None:
            logger.info("[LOADPROBE] %s", line)

    def summary(self) -> str:
        """Return the cumulative totals as a compact ``key=value`` string."""
        with self._lock:
            return self._summary()

    def _summary(self) -> str:
        """Format totals; the caller holds the lock."""
        step_ms = self._poll_span_s / self._polls * 1000 if self._polls else 0.0
        return (
            f"loads={self._loads} io_s={self._io_s:.1f} io_max_s={self._io_max:.2f} "
            f"report_s={self._report_s:.1f} report_max_s={self._report_max:.2f} "
            f"open={len(self._started)} polls={self._polls} step_ms={step_ms:.1f}"
        )
