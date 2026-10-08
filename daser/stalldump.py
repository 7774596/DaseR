# SPDX-License-Identifier: Apache-2.0
"""On-demand stall dumps of thread stacks, asyncio tasks and component state.

Set ``DASER_STALL_DUMP_FILE`` to a path; every process that imported a
component calling :func:`register_loop` or :func:`register_state` then
watches that file from a daemon thread. Touching the file (changing its
mtime) logs, at WARNING, every thread's stack, every pending task of the
registered event loops with its await stack, and each registered state
snapshot. Unset (the default), nothing is started.

Diagnostic only: task stacks are read from another thread without the
loop's cooperation, so a snapshot can be slightly inconsistent.
"""

# Standard
import asyncio
from collections.abc import Callable
import io
import os
import sys
import threading
import time
import traceback
from typing import Any

# First Party
from daser.logging import init_logger

logger = init_logger(__name__)

_POLL_S = 1.0
_lock = threading.Lock()
_loops: dict[str, asyncio.AbstractEventLoop] = {}
_states: dict[str, Callable[[], dict[str, Any]]] = {}
_watcher: threading.Thread | None = None


def _dump_file() -> str:
    """Return the trigger file path, empty when dumps are disabled."""
    return os.environ.get("DASER_STALL_DUMP_FILE", "")


def _format_dump() -> str:
    """Render thread stacks, registered loop tasks and state snapshots."""
    out = io.StringIO()
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    # The documented CPython hook for other threads' stacks.
    for ident, frame in sys._current_frames().items():  # noqa: SLF001
        out.write(f"--- thread {names.get(ident, '?')} ({ident})\n")
        out.write("".join(traceback.format_stack(frame)))
    with _lock:
        loops = dict(_loops)
        states = dict(_states)
    for name, loop in loops.items():
        try:
            tasks = list(asyncio.all_tasks(loop))
        except RuntimeError as exc:  # set changed while iterating
            out.write(f"--- loop {name}: tasks unavailable ({exc})\n")
            continue
        out.write(f"--- loop {name}: {len(tasks)} pending tasks\n")
        for task in tasks:
            out.write(f"task {task.get_name()} {task.get_coro()!r}\n")
            buffer = io.StringIO()
            task.print_stack(limit=12, file=buffer)
            out.write(buffer.getvalue())
    for name, state in states.items():
        try:
            out.write(f"--- state {name}: {state()!r}\n")
        except Exception as exc:  # noqa: BLE001 - a diagnostic must not raise
            out.write(f"--- state {name}: failed ({exc!r})\n")
    return out.getvalue()


def _watch(path: str) -> None:
    """Log one dump whenever the trigger file's mtime changes."""
    seen = os.path.getmtime(path) if os.path.exists(path) else None
    while True:
        time.sleep(_POLL_S)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime == seen:
            continue
        seen = mtime
        logger.warning("[STALLDUMP] pid=%d\n%s", os.getpid(), _format_dump())


def _ensure_watcher() -> None:
    """Start the trigger-file watcher once per process when enabled."""
    global _watcher
    path = _dump_file()
    if not path or _watcher is not None:
        return
    _watcher = threading.Thread(
        target=_watch, args=(path,), daemon=True, name="daser-stalldump"
    )
    _watcher.start()


def register_loop(name: str, loop: asyncio.AbstractEventLoop) -> None:
    """Include an event loop's pending tasks in stall dumps.

    Args:
        name: Label printed with the loop's tasks.
        loop: Event loop whose tasks are listed.

    Async/thread-safety:
        Thread-safe; a no-op unless ``DASER_STALL_DUMP_FILE`` is set.
    """
    if not _dump_file():
        return
    with _lock:
        _loops[name] = loop
    _ensure_watcher()


def register_state(name: str, state: Callable[[], dict[str, Any]]) -> None:
    """Include a component state snapshot in stall dumps.

    Args:
        name: Label printed with the snapshot.
        state: Callable returning a small dict of counts and identifiers; it
            runs on the watcher thread and must not block.

    Async/thread-safety:
        Thread-safe; a no-op unless ``DASER_STALL_DUMP_FILE`` is set.
    """
    if not _dump_file():
        return
    with _lock:
        _states[name] = state
    _ensure_watcher()
