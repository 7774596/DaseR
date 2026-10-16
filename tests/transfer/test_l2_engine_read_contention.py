# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import mmap
from pathlib import Path
import threading

# First Party
from daser.transfer.iouring.l2_engine import L2IoEngine
from tests.transfer.store_files import create_store_file

PAGE = 4096


def _hold(engine: L2IoEngine, kind: str, release: threading.Event) -> threading.Thread:
    """Run a ``kind`` job that stays in native IO until ``release`` is set."""
    started = threading.Event()

    def blocking_job() -> int:
        started.set()
        release.wait()
        return PAGE

    thread = threading.Thread(target=engine.timed(kind, blocking_job))
    thread.start()
    assert started.wait(5)
    return thread


def test_lone_read_counts_in_bucket_one_with_bytes(tmp_path: Path) -> None:
    engine = L2IoEngine(create_store_file(str(tmp_path / "l2"), 4 * PAGE), 4 * PAGE, 2)
    dst = memoryview(mmap.mmap(-1, PAGE))
    assert engine.timed("read", engine.read_into)(0, dst, engine.next_uring()) == PAGE
    contention = engine.read_contention
    assert set(contention) == {(1, False)}
    jobs, io_us, nbytes = contention[(1, False)]
    assert (jobs, nbytes) == (1, PAGE)
    assert io_us >= 0
    engine.close()


def test_reads_bucket_by_concurrency_and_write_activity(tmp_path: Path) -> None:
    engine = L2IoEngine(create_store_file(str(tmp_path / "l2"), 4 * PAGE), 4 * PAGE, 4)
    release = threading.Event()
    held = [_hold(engine, "read", release) for _ in range(2)]
    writer = _hold(engine, "write", release)
    # Third concurrent read, started while a write is in native IO.
    engine.timed("read", lambda: PAGE)()
    release.set()
    for thread in (*held, writer):
        thread.join()
    contention = engine.read_contention
    assert contention[(4, True)][0] == 1
    assert contention[(4, True)][2] == PAGE
    # The two held reads started with no write in flight.
    assert sum(contention[key][0] for key in contention if not key[1]) == 2
    engine.close()


def test_writes_do_not_count_as_reads(tmp_path: Path) -> None:
    engine = L2IoEngine(create_store_file(str(tmp_path / "l2"), 4 * PAGE), 4 * PAGE, 2)
    data = memoryview(mmap.mmap(-1, PAGE))
    assert asyncio.run(engine.write_batch([(0, data)])) == PAGE
    assert engine.read_contention == {}
    engine.close()
