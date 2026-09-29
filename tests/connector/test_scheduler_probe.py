# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
import logging

import pytest

from daser.connector.scheduler.probe import PROBE_INTERVAL_ENV, SchedulerProbe


class _Capture(logging.Handler):
    """Collect formatted probe log lines."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.fixture
def probe_log() -> Iterator[_Capture]:
    """Attach a capture handler to the probe logger."""
    handler = _Capture()
    probe_logger = logging.getLogger("daser.connector.scheduler.probe")
    probe_logger.addHandler(handler)
    previous = probe_logger.level
    probe_logger.setLevel(logging.INFO)
    yield handler
    probe_logger.removeHandler(handler)
    probe_logger.setLevel(previous)


def test_from_env_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe is off unless the interval variable is a positive number."""
    monkeypatch.delenv(PROBE_INTERVAL_ENV, raising=False)
    assert SchedulerProbe.from_env() is None
    monkeypatch.setenv(PROBE_INTERVAL_ENV, "0")
    assert SchedulerProbe.from_env() is None
    monkeypatch.setenv(PROBE_INTERVAL_ENV, "bogus")
    assert SchedulerProbe.from_env() is None
    monkeypatch.setenv(PROBE_INTERVAL_ENV, "30")
    assert SchedulerProbe.from_env() is not None


def test_probe_accounts_load_wait_and_save_hold(probe_log: _Capture) -> None:
    """Async hits and delayed frees are closed by transfer completions."""
    probe = SchedulerProbe(interval_s=1e-9)
    probe.on_lookup("load", 0.002, (256, True))
    probe.on_alloc("load", 4)
    probe.on_lookup("miss", 0.001, (0, False))
    probe.on_lookup("retry", 0.001, (None, True))
    probe.on_request_finished("save", 3, delayed=True)
    probe.on_request_finished("free", 3, delayed=False)
    probe.on_connector_output(finished_sending=["save"], finished_recving=["load"])

    line = probe_log.lines[-1]
    assert "lookup n=3" in line
    assert "pending_retries=1" in line
    assert "load_wait n=1" in line
    assert "save_hold n=1" in line
    assert line.count("open=0") == 2


def test_probe_keeps_unfinished_intervals_open(probe_log: _Capture) -> None:
    """Intervals without a completion stay open and are not counted."""
    probe = SchedulerProbe(interval_s=1e-9)
    probe.on_lookup("load", 0.001, (128, True))
    probe.on_request_finished("save", 2, delayed=True)
    probe.on_connector_output(finished_sending=[], finished_recving=[])

    line = probe_log.lines[-1]
    assert "load_wait n=0" in line and "save_hold n=0" in line
    assert line.count("open=1") == 2
