# SPDX-License-Identifier: Apache-2.0
"""Admission timestamp tracing of the scheduler-side request lifecycle."""

import logging
from types import SimpleNamespace

import pytest

from daser.connector.scheduler.lifecycle import RequestLifecycle


class _MissIPC:
    """IPC stub that never finds a cached chunk."""

    def lookup(self, *args, **kwargs):
        return []

    def lookup_versioned(self, *args, **kwargs):
        return [], 0

    def index_epoch(self) -> int:
        return 0


def _lifecycle(admit_trace: bool) -> RequestLifecycle:
    return RequestLifecycle(
        ipc_client=_MissIPC(),
        block_tokens=4,
        slot_size=32,
        model_id="model",
        cache_reuse_mode="chunk",
        runtime_config_ready=True,
        admit_trace=admit_trace,
    )


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="a",
        prompt_token_ids=list(range(12)),
        kv_transfer_params={"daser_skip_save": True},
    )


def _admit_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "[ADMIT]" in r.getMessage()]


def _drive(lifecycle: RequestLifecycle) -> None:
    request = _request()
    lifecycle.get_num_new_matched_tokens(request, 0)
    lifecycle.get_num_new_matched_tokens(request, 0)  # allocation retry
    blocks = SimpleNamespace(blocks=[[SimpleNamespace(block_id=i) for i in range(3)]])
    lifecycle.update_state_after_alloc(request, blocks, 0)
    lifecycle.update_connector_output(
        SimpleNamespace(finished_sending=(), finished_recving=("a",))
    )


def test_trace_logs_first_lookup_alloc_and_load(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each event is logged once with a timestamp; retries add no lookup line."""
    caplog.set_level(logging.INFO)
    _drive(_lifecycle(admit_trace=True))
    lines = _admit_lines(caplog)
    assert [line.split()[1:3] for line in lines] == [
        ["lookup", "a"],
        ["alloc", "a"],
        ["loaded", "a"],
    ]
    times = [float(line.rsplit("t=", 1)[1]) for line in lines]
    assert times == sorted(times)


def test_trace_off_by_default_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """Without the switch no admission line is written."""
    caplog.set_level(logging.INFO)
    lifecycle = _lifecycle(admit_trace=False)
    _drive(lifecycle)
    lifecycle.trace_match("a", 0, 8, 12)
    assert _admit_lines(caplog) == []


def test_trace_match_logs_only_changed_splits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Retries with the same split are silent; a resume lookup is logged."""
    caplog.set_level(logging.INFO)
    lifecycle = _lifecycle(admit_trace=True)
    lifecycle.trace_match("a", 0, None, 12)
    lifecycle.trace_match("a", 0, 8, 12)
    lifecycle.trace_match("a", 0, 8, 12)  # allocation retry
    lifecycle.trace_match("a", 4, 4, 12)  # resumed after preemption
    fields = [line.split()[2:6] for line in _admit_lines(caplog)]
    assert fields == [
        ["a", "local=0", "ext=None", "total=12"],
        ["a", "local=0", "ext=8", "total=12"],
        ["a", "local=4", "ext=4", "total=12"],
    ]
