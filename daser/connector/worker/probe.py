# SPDX-License-Identifier: Apache-2.0

"""Opt-in worker-side breakdown of how long finished saves hold vLLM blocks.

Enabled by setting ``DASER_STORE_PROBE_INTERVAL_S`` to a positive number of
seconds. Each finished save is timed through these points (monotonic clock):

- finished: ``collect_finished`` first sees the request as finished.
- submitted: the save is handed to the store event loop (end of that poll).
- admitted: the save acquired the store-capacity semaphore.
- ready: vLLM blocks became releasable (packed: snapshot in staging; raw:
  every batch staged, written, and committed).
- emitted: ``collect_finished`` returned the request ID to vLLM.

Segments: ``dispatch`` = submitted - finished, ``fifo`` = admitted -
submitted, ``service`` = ready - admitted, ``lag`` = emitted - ready, and
``hold`` = emitted - finished. Batch-level lease wait, staging, and IPC write
times are accumulated separately because packed groups share batches.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time

from daser.logging import init_logger

logger = init_logger(__name__)

STORE_PROBE_INTERVAL_ENV = "DASER_STORE_PROBE_INTERVAL_S"
_SEGMENTS = ("dispatch", "fifo", "service", "lag", "hold")
_BATCH_PHASES = ("lease_wait", "stage", "write")


@dataclass
class SaveTimeline:
    """Monotonic timestamps of one finished save (0.0 means not reached)."""

    finished: float = 0.0
    submitted: float = 0.0
    admitted: float = 0.0
    ready: float = 0.0


@dataclass
class _Totals:
    """Count, sum, and max of one duration series."""

    count: int = 0
    seconds: float = 0.0
    max_seconds: float = 0.0

    def add(self, seconds: float) -> None:
        """Accumulate one non-negative duration."""
        seconds = max(0.0, seconds)
        self.count += 1
        self.seconds += seconds
        self.max_seconds = max(self.max_seconds, seconds)

    def summary(self) -> str:
        """Return ``n/sum/mean/max`` in seconds."""
        mean = self.seconds / self.count if self.count else 0.0
        return (
            f"n={self.count} s={self.seconds:.1f} mean={mean:.3f} "
            f"max={self.max_seconds:.2f}"
        )


class StoreProbe:
    """Accumulate save-hold segments and log cumulative totals periodically.

    Async/thread-safety:
        ``on_emitted`` runs on the worker thread; ``on_batch`` runs on the
        store event loop. A lock guards the shared totals.
    """

    def __init__(self, interval_s: float) -> None:
        """Create a probe that logs at most every ``interval_s`` seconds.

        Args:
            interval_s: Logging interval in seconds (must be positive).
        """
        self._interval_s = interval_s
        self._next_log = time.monotonic() + interval_s
        self._lock = threading.Lock()
        self._segments = {name: _Totals() for name in _SEGMENTS}
        self._batches = {name: _Totals() for name in _BATCH_PHASES}
        self._batch_bytes = 0
        self._incomplete = 0

    @classmethod
    def from_env(cls) -> StoreProbe | None:
        """Build a probe from ``DASER_STORE_PROBE_INTERVAL_S``.

        Returns:
            A probe when the variable is a positive float, else None.
        """
        try:
            interval_s = float(os.environ.get(STORE_PROBE_INTERVAL_ENV, "0"))
        except ValueError:
            return None
        return cls(interval_s) if interval_s > 0 else None

    def on_batch(
        self, lease_wait_s: float, stage_s: float, write_s: float, nbytes: int
    ) -> None:
        """Record one staged batch.

        Args:
            lease_wait_s: Time waiting for a staging lease permit.
            stage_s: Copy or pack plus stream synchronization time.
            write_s: IPC transfer, server write, and commit time.
            nbytes: Staged bytes.
        """
        with self._lock:
            self._batches["lease_wait"].add(lease_wait_s)
            self._batches["stage"].add(stage_s)
            self._batches["write"].add(write_s)
            self._batch_bytes += nbytes

    def on_emitted(self, timeline: SaveTimeline, emitted: float) -> None:
        """Record the segments of a save whose blocks were just released.

        Args:
            timeline: Timestamps collected for the save.
            emitted: Time ``collect_finished`` returned the request.
        """
        points = (timeline.finished, timeline.submitted, timeline.admitted)
        with self._lock:
            if not all(points) or not timeline.ready:
                self._incomplete += 1
            else:
                self._segments["dispatch"].add(timeline.submitted - timeline.finished)
                self._segments["fifo"].add(timeline.admitted - timeline.submitted)
                self._segments["service"].add(timeline.ready - timeline.admitted)
                self._segments["lag"].add(emitted - timeline.ready)
                self._segments["hold"].add(emitted - timeline.finished)
        self._maybe_log()

    def _maybe_log(self) -> None:
        """Log cumulative totals when the interval elapsed."""
        now = time.monotonic()
        with self._lock:
            if now < self._next_log:
                return
            self._next_log = now + self._interval_s
            segments = " | ".join(
                f"{name} {self._segments[name].summary()}" for name in _SEGMENTS
            )
            batches = " | ".join(
                f"{name} {self._batches[name].summary()}" for name in _BATCH_PHASES
            )
            gib = self._batch_bytes / 2**30
            incomplete = self._incomplete
        logger.info(
            "[STORE-PROBE] %s || batches %s | gib=%.1f | incomplete=%d",
            segments,
            batches,
            gib,
            incomplete,
        )
