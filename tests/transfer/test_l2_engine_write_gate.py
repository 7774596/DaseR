# SPDX-License-Identifier: Apache-2.0

# Standard
import mmap
from pathlib import Path
import threading
import time

# Third Party
import pytest

# First Party
from daser.transfer.iouring.l2_engine import L2IoEngine
from tests.transfer.store_files import create_store_file

PAGE = 4096


def _page(fill: int) -> memoryview:
    buf = mmap.mmap(-1, PAGE)
    buf.write(bytes([fill]) * PAGE)
    return memoryview(buf)


def _hold_read(engine: L2IoEngine, release: threading.Event) -> threading.Thread:
    """Run a read job that stays in native IO until ``release`` is set."""
    started = threading.Event()

    def blocking_read() -> None:
        started.set()
        release.wait()

    thread = threading.Thread(target=engine.timed("read", blocking_read))
    thread.start()
    assert started.wait(5)
    return thread


def test_gated_write_waits_bounded_for_in_flight_read(tmp_path: Path) -> None:
    engine = L2IoEngine(
        create_store_file(str(tmp_path / "l2"), 4 * PAGE), 4 * PAGE, 2, write_gate=True
    )
    release = threading.Event()
    reader = _hold_read(engine, release)
    start = time.perf_counter()
    assert engine.write(0, _page(7), engine.next_uring()) == PAGE
    waited = time.perf_counter() - start
    release.set()
    reader.join()
    # The read never drains, so the write proceeds after the bounded wait.
    assert waited >= 0.04
    assert engine.timing_us["write_gate"] >= 40_000
    assert engine.write_executor is not engine.executor
    dst = memoryview(mmap.mmap(-1, PAGE))
    assert engine.read_into(0, dst, engine.next_uring()) == PAGE
    assert bytes(dst) == bytes([7]) * PAGE
    engine.close()


def test_gated_write_resumes_when_reads_drain(tmp_path: Path) -> None:
    engine = L2IoEngine(
        create_store_file(str(tmp_path / "l2"), 4 * PAGE), 4 * PAGE, 2, write_gate=True
    )
    release = threading.Event()
    reader = _hold_read(engine, release)
    threading.Timer(0.01, release.set).start()
    assert engine.write(0, _page(3), engine.next_uring()) == PAGE
    reader.join()
    assert 0 < engine.timing_us["write_gate"] < 40_000
    engine.close()


def test_ungated_write_ignores_reads_and_shares_executor(tmp_path: Path) -> None:
    engine = L2IoEngine(create_store_file(str(tmp_path / "l2"), 4 * PAGE), 4 * PAGE, 2)
    release = threading.Event()
    reader = _hold_read(engine, release)
    assert engine.write(0, _page(1), engine.next_uring()) == PAGE
    release.set()
    reader.join()
    assert engine.timing_us["write_gate"] == 0
    assert engine.write_executor is engine.executor
    engine.close()


def test_engine_opens_existing_file_without_creating_or_resizing(
    tmp_path: Path,
) -> None:
    """The server owns the store file; the engine must only open it."""
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        L2IoEngine(str(missing), 4 * PAGE, 1)
    assert not missing.exists()

    short = create_store_file(str(tmp_path / "short"), 2 * PAGE)
    with pytest.raises(ValueError, match="has size"):
        L2IoEngine(short, 4 * PAGE, 1)
    assert Path(short).stat().st_size == 2 * PAGE

    larger = create_store_file(str(tmp_path / "larger"), 8 * PAGE)
    engine = L2IoEngine(larger, 4 * PAGE, 1)
    engine.close()
    assert Path(larger).stat().st_size == 8 * PAGE
