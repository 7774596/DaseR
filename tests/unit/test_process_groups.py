# SPDX-License-Identifier: Apache-2.0
"""Tests for recording and stopping service process groups."""

import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from benchmarks.research.reap import reap
from benchmarks.utils.servers import (
    descendant_process_groups,
    pgid_file_for,
    read_process_groups,
    record_process_groups,
    stop_from_pid_file,
)


def _spawn_service_with_setsid_child() -> tuple[subprocess.Popen[bytes], int]:
    """Start a service that runs a child in its own session, like py-spy."""
    service = subprocess.Popen(
        ["sh", "-c", "setsid sleep 60 & echo $!; sleep 60"],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    assert service.stdout is not None
    child = int(service.stdout.readline())
    deadline = time.monotonic() + 5
    while os.getpgid(child) != child and time.monotonic() < deadline:
        time.sleep(0.05)  # setsid execs after the fork
    return service, child


def _gone(pid: int) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            return True
        if stat.rsplit(")", 1)[1].split()[0] == "Z":
            return True
        time.sleep(0.05)
    return False


def test_descendant_groups_include_setsid_children_not_the_caller() -> None:
    service, child = _spawn_service_with_setsid_child()
    try:
        groups = descendant_process_groups([service.pid])
        assert service.pid in groups
        assert child in groups
        assert os.getpgrp() not in groups
    finally:
        os.killpg(service.pid, 9)
        os.killpg(child, 9)
        service.wait()


def test_recorded_group_survives_its_leader_exit(tmp_path: Path) -> None:
    """A group recorded once is kept after its root no longer leads to it."""
    service, child = _spawn_service_with_setsid_child()
    pgids = tmp_path / "pgids"
    try:
        record_process_groups(pgids, [service.pid])
        os.killpg(service.pid, 9)
        service.wait()
        assert child in record_process_groups(pgids, [service.pid])
    finally:
        os.killpg(child, 9)


def test_stop_from_pid_file_stops_setsid_children(tmp_path: Path) -> None:
    service, child = _spawn_service_with_setsid_child()
    pid_file = tmp_path / "pids.json"
    pid_file.write_text(json.dumps([{"pid": service.pid, "cmd": ["sh"]}]))

    stop_from_pid_file(pid_file)
    service.wait(timeout=5)

    assert _gone(child)
    assert child in read_process_groups(pgid_file_for(pid_file))


def test_reap_stops_groups_left_by_a_killed_runner(tmp_path: Path) -> None:
    """Groups recorded under a run dir are stopped even with no runner alive."""
    service, child = _spawn_service_with_setsid_child()
    arm = tmp_path / "scr" / "arm"
    arm.mkdir(parents=True)
    record_process_groups(arm / "pgids", [service.pid])

    assert reap([tmp_path], gpu=None, settle_s=0) == []
    service.wait(timeout=20)

    assert _gone(child)


def test_stop_skips_the_callers_own_group(tmp_path: Path) -> None:
    pgids = tmp_path / "pgids"
    pgids.write_text(f"{os.getpgrp()}\n")
    from benchmarks.utils.servers import stop_process_groups

    assert stop_process_groups(pgids, grace_s=0) == []


@pytest.mark.parametrize("text", ["", "\n", "x\n"])
def test_read_ignores_empty_and_malformed_lines(tmp_path: Path, text: str) -> None:
    path = tmp_path / "pgids"
    path.write_text(text)
    assert read_process_groups(path) == []


def test_watchdog_anomaly_stops_groups_its_runner_left(tmp_path: Path) -> None:
    """After an anomaly, groups still alive past the grace period are killed."""
    from benchmarks.research import watchdog

    service, child = _spawn_service_with_setsid_child()
    out = tmp_path / "art"
    scr = tmp_path / "scr"
    out.mkdir()
    (scr / "logs").mkdir(parents=True)
    (scr / "pids.json").write_text(json.dumps([{"pid": service.pid}]))
    record_process_groups(pgid_file_for(scr / "pids.json"), [service.pid])
    args = ["--output-dir", str(out), "--service-dir", str(scr)]
    args += ["--startup-timeout", "0", "--reap-grace-s", "0.5"]

    # No replay client ever appears: an anomaly with no runner to tear down.
    with pytest.raises(SystemExit) as exit_info:
        watchdog.main(args)
    service.wait(timeout=5)

    assert exit_info.value.code == 1
    assert _gone(child)
