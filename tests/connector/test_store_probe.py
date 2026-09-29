# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
import logging

import pytest

from daser.connector.worker.probe import (
    STORE_PROBE_INTERVAL_ENV,
    SaveTimeline,
    StoreProbe,
)


class _Capture(logging.Handler):
    """Collect formatted probe log lines."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.fixture
def probe_log() -> Iterator[_Capture]:
    """Attach a capture handler to the store probe logger."""
    handler = _Capture()
    probe_logger = logging.getLogger("daser.connector.worker.probe")
    probe_logger.addHandler(handler)
    previous = probe_logger.level
    probe_logger.setLevel(logging.INFO)
    yield handler
    probe_logger.removeHandler(handler)
    probe_logger.setLevel(previous)


def test_from_env_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe is off unless the interval variable is a positive number."""
    monkeypatch.delenv(STORE_PROBE_INTERVAL_ENV, raising=False)
    assert StoreProbe.from_env() is None
    monkeypatch.setenv(STORE_PROBE_INTERVAL_ENV, "-1")
    assert StoreProbe.from_env() is None
    monkeypatch.setenv(STORE_PROBE_INTERVAL_ENV, "60")
    assert StoreProbe.from_env() is not None


def test_segments_partition_the_hold(probe_log: _Capture) -> None:
    """dispatch + fifo + service + lag equals the total hold."""
    probe = StoreProbe(interval_s=1e-9)
    probe.on_batch(lease_wait_s=0.5, stage_s=0.25, write_s=1.0, nbytes=2**30)
    timeline = SaveTimeline(finished=10.0, submitted=10.5, admitted=12.0, ready=13.0)
    probe.on_emitted(timeline, emitted=14.0)

    line = probe_log.lines[-1]
    assert "dispatch n=1 s=0.5" in line
    assert "fifo n=1 s=1.5" in line
    assert "service n=1 s=1.0" in line
    assert "lag n=1 s=1.0" in line
    assert "hold n=1 s=4.0" in line
    assert "lease_wait n=1 s=0.5" in line
    assert "write n=1 s=1.0" in line
    assert "gib=1.0" in line
    assert "incomplete=0" in line


def test_incomplete_timeline_is_counted_not_timed(probe_log: _Capture) -> None:
    """Saves missing a timestamp are reported as incomplete."""
    probe = StoreProbe(interval_s=1e-9)
    probe.on_emitted(SaveTimeline(finished=1.0, submitted=2.0), emitted=3.0)

    line = probe_log.lines[-1]
    assert "hold n=0" in line
    assert "incomplete=1" in line
