# SPDX-License-Identifier: Apache-2.0
"""Tests for on-demand stall dumps."""

# Standard
from pathlib import Path
import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    import asyncio, os, sys, threading, time
    from daser import stalldump

    trigger = sys.argv[1]
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    async def stuck_waiter():
        await asyncio.Event().wait()

    asyncio.run_coroutine_threadsafe(stuck_waiter(), loop)
    stalldump.register_loop("demo", loop)
    stalldump.register_state("demo", lambda: {"pending": ["req-1"]})
    time.sleep(1.5)
    with open(trigger, "w") as f:
        f.write("go")
    os.utime(trigger, (time.time() + 5, time.time() + 5))
    time.sleep(2.5)
    """
)


def _run(tmp_path: Path, enabled: bool) -> str:
    trigger = tmp_path / "dump"
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).parents[1])}
    if enabled:
        env["DASER_STALL_DUMP_FILE"] = str(trigger)
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(trigger)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return result.stdout + result.stderr


def test_touching_trigger_dumps_tasks_and_state(tmp_path: Path) -> None:
    output = _run(tmp_path, enabled=True)
    assert "[STALLDUMP]" in output
    assert "stuck_waiter" in output
    assert "state demo: {'pending': ['req-1']}" in output


def test_disabled_by_default(tmp_path: Path) -> None:
    assert "[STALLDUMP]" not in _run(tmp_path, enabled=False)
