# SPDX-License-Identifier: Apache-2.0
"""Tests for the worker-side async load timeline."""

import pytest

from daser.connector.worker.load_probe import LoadTimeline


def test_timeline_splits_io_from_report_lag() -> None:
    """io runs from start to future completion, report from there to the poll."""
    timeline = LoadTimeline(interval_s=1e9)
    timeline.on_collect([], now=0.0)
    timeline.on_start(["a", "b"], now=1.0)
    timeline.on_done(["a"], now=1.2)
    timeline.on_collect([], now=1.5)
    timeline.on_done(["b"], now=1.6)
    timeline.on_collect(["a", "b"], now=2.0)
    summary = timeline.summary()
    assert "loads=2" in summary
    assert "io_s=0.8" in summary  # 0.2 + 0.6
    assert "report_s=1.2" in summary  # 0.8 + 0.4
    assert "open=0" in summary
    assert "polls=2 step_ms=1000.0" in summary


def test_timeline_ignores_unstarted_and_counts_open_loads() -> None:
    """Completions without a start are dropped; unfinished loads stay open."""
    timeline = LoadTimeline(interval_s=1e9)
    timeline.on_done(["x"], now=0.5)
    timeline.on_start(["y"], now=1.0)
    timeline.on_collect(["x"], now=2.0)
    summary = timeline.summary()
    assert "loads=0" in summary and "open=1" in summary


def test_timeline_disabled_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timeline is off unless the probe interval is positive."""
    monkeypatch.delenv("DASER_SCHED_PROBE_INTERVAL_S", raising=False)
    assert LoadTimeline.from_env() is None
    monkeypatch.setenv("DASER_SCHED_PROBE_INTERVAL_S", "30")
    assert LoadTimeline.from_env() is not None
