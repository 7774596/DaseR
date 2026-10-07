# SPDX-License-Identifier: Apache-2.0
"""Tests for the agent-trace arm watchdog."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from benchmarks.research.watchdog import parse_args, watch


def test_wrapper_exit_is_not_a_service_exit_while_its_group_runs(
    tmp_path: Path,
) -> None:
    """A service whose wrapper exited but whose server still runs is healthy."""
    out = tmp_path / "art"
    service = tmp_path / "scr"
    out.mkdir()
    (service / "logs").mkdir(parents=True)
    # Stand-in replay client: the watchdog finds it by module name and dir.
    replay = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(2)",
            "benchmarks.mooncake.benchmark",
            str(out.resolve()),
        ]
    )
    # The wrapper (like ``nsys profile --duration``) leaves its server running.
    wrapper = subprocess.Popen(
        ["sh", "-c", "sleep 60 & echo $!"],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    assert wrapper.stdout is not None
    wrapper.stdout.readline()  # the server has started
    wrapper.wait(timeout=5)
    (service / "pids.json").write_text(json.dumps([{"pid": wrapper.pid}]))
    try:
        ok, message = watch(
            parse_args(
                [
                    "--output-dir",
                    str(out),
                    "--service-dir",
                    str(service),
                    "--vllm-url",
                    "http://127.0.0.1:9",
                    "--interval",
                    "0.4",
                    "--data-path",
                    str(tmp_path),
                    "--min-free-data-gib",
                    "0",
                    "--min-free-root-gib",
                    "0",
                    "--no-stop",
                ]
            )
        )
    finally:
        os.killpg(wrapper.pid, signal.SIGKILL)
        replay.wait(timeout=5)
    assert ok, message
    assert message.startswith("replay exited")
