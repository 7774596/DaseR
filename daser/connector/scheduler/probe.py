# SPDX-License-Identifier: Apache-2.0

"""Opt-in scheduler-side accounting of GPU block holds caused by the connector.

Enabled by setting ``DASER_SCHED_PROBE_INTERVAL_S`` to a positive number of
seconds; the probe then logs cumulative totals at that interval. It measures:

- lookup: wall time spent inside ``get_num_new_matched_tokens`` (synchronous
  IPC on the scheduler thread).
- load wait: time from an async external hit until ``finished_recving``,
  weighted by the request's allocated blocks (block-seconds held while the
  request sits in ``WAITING_FOR_REMOTE_KVS``).
- save hold: time from a delayed ``request_finished`` until
  ``finished_sending``, weighted by the request's blocks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import os
import time

from daser.logging import init_logger

logger = init_logger(__name__)

PROBE_INTERVAL_ENV = "DASER_SCHED_PROBE_INTERVAL_S"


@dataclass
class _HoldTotals:
    """Cumulative interval statistics for one hold kind."""

    count: int = 0
    seconds: float = 0.0
    block_seconds: float = 0.0
    max_seconds: float = 0.0
    open: dict[str, tuple[float, int]] = field(default_factory=dict)

    def start(self, req_id: str, now: float, num_blocks: int) -> None:
        """Open an interval for ``req_id`` unless one is already open."""
        self.open.setdefault(req_id, (now, num_blocks))

    def set_blocks(self, req_id: str, num_blocks: int) -> None:
        """Update the block weight of an open interval."""
        if req_id in self.open:
            self.open[req_id] = (self.open[req_id][0], num_blocks)

    def close(self, req_id: str, now: float) -> None:
        """Close the interval for ``req_id`` if it is open."""
        entry = self.open.pop(req_id, None)
        if entry is None:
            return
        elapsed = now - entry[0]
        self.count += 1
        self.seconds += elapsed
        self.block_seconds += elapsed * entry[1]
        self.max_seconds = max(self.max_seconds, elapsed)

    def summary(self) -> str:
        """Return a compact ``key=value`` summary."""
        return (
            f"n={self.count} s={self.seconds:.1f} blk_s={self.block_seconds:.0f} "
            f"max_s={self.max_seconds:.2f} open={len(self.open)}"
        )


class SchedulerProbe:
    """Accumulate connector-induced scheduler costs and log them periodically.

    Async/thread-safety:
        All methods are called on vLLM's scheduler thread; no locking.
    """

    def __init__(self, interval_s: float) -> None:
        """Create a probe that logs every ``interval_s`` seconds.

        Args:
            interval_s: Logging interval in seconds (must be positive).
        """
        self._interval_s = interval_s
        self._next_log = time.monotonic() + interval_s
        self._lookup_calls = 0
        self._lookup_seconds = 0.0
        self._lookup_max = 0.0
        self._pending_retries = 0
        self._load_wait = _HoldTotals()
        self._save_hold = _HoldTotals()

    @classmethod
    def from_env(cls) -> SchedulerProbe | None:
        """Build a probe from ``DASER_SCHED_PROBE_INTERVAL_S``.

        Returns:
            A probe when the variable is a positive float, else None.
        """
        try:
            interval_s = float(os.environ.get(PROBE_INTERVAL_ENV, "0"))
        except ValueError:
            return None
        return cls(interval_s) if interval_s > 0 else None

    def on_lookup(
        self, req_id: str, elapsed_s: float, result: tuple[int | None, bool]
    ) -> None:
        """Record one ``get_num_new_matched_tokens`` call.

        Args:
            req_id: vLLM request ID.
            elapsed_s: Wall time spent in the call.
            result: The ``(num_external_tokens, is_async)`` return value.
        """
        self._lookup_calls += 1
        self._lookup_seconds += elapsed_s
        self._lookup_max = max(self._lookup_max, elapsed_s)
        num_tokens, is_async = result
        if num_tokens is None:
            self._pending_retries += 1
        elif is_async and num_tokens > 0:
            self._load_wait.start(req_id, time.monotonic(), 0)
        self._maybe_log()

    def on_alloc(self, req_id: str, num_blocks: int) -> None:
        """Attach the allocated block count to an open load wait.

        Args:
            req_id: vLLM request ID.
            num_blocks: Blocks allocated to the request.
        """
        self._load_wait.set_blocks(req_id, num_blocks)

    def on_request_finished(self, req_id: str, num_blocks: int, delayed: bool) -> None:
        """Open a save-hold interval when block freeing is delayed.

        Args:
            req_id: vLLM request ID.
            num_blocks: Blocks the scheduler would otherwise free now.
            delayed: Whether the connector asked vLLM to delay freeing.
        """
        if delayed:
            self._save_hold.start(req_id, time.monotonic(), num_blocks)

    def on_connector_output(
        self, finished_sending: Iterable[str], finished_recving: Iterable[str]
    ) -> None:
        """Close intervals for requests whose transfers completed.

        Args:
            finished_sending: Request IDs whose saves completed.
            finished_recving: Request IDs whose loads completed.
        """
        now = time.monotonic()
        for req_id in finished_sending:
            self._save_hold.close(req_id, now)
        for req_id in finished_recving:
            self._load_wait.close(req_id, now)
        self._maybe_log()

    def _maybe_log(self) -> None:
        """Log cumulative totals when the interval elapsed."""
        now = time.monotonic()
        if now < self._next_log:
            return
        self._next_log = now + self._interval_s
        logger.info(
            "[PROBE] lookup n=%d s=%.2f max_ms=%.1f pending_retries=%d | "
            "load_wait %s | save_hold %s",
            self._lookup_calls,
            self._lookup_seconds,
            self._lookup_max * 1000,
            self._pending_retries,
            self._load_wait.summary(),
            self._save_hold.summary(),
        )
