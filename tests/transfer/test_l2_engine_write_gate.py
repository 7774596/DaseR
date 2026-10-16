# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from collections.abc import Sequence
import mmap
from pathlib import Path
import threading
import time

# Third Party
import pytest

# First Party
from daser.transfer.iouring.l2_engine import L2IoEngine
from daser.transfer.iouring.native import NativeIOUring
from tests.transfer.store_files import create_store_file

PAGE = 4096
MIB = 1024 * 1024


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
    assert asyncio.run(engine.write_batch([(0, _page(7))])) == PAGE
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
    assert asyncio.run(engine.write_batch([(0, _page(3))])) == PAGE
    reader.join()
    assert 0 < engine.timing_us["write_gate"] < 40_000
    engine.close()


def test_ungated_write_ignores_reads_and_shares_executor(tmp_path: Path) -> None:
    engine = L2IoEngine(create_store_file(str(tmp_path / "l2"), 4 * PAGE), 4 * PAGE, 2)
    release = threading.Event()
    reader = _hold_read(engine, release)
    assert asyncio.run(engine.write_batch([(0, _page(1))])) == PAGE
    release.set()
    reader.join()
    assert engine.timing_us["write_gate"] == 0
    assert engine.write_executor is engine.executor
    engine.close()


@pytest.mark.parametrize("gated", [False, True])
def test_write_batch_persists_every_extent(tmp_path: Path, gated: bool) -> None:
    """``write_batch`` writes extents beyond one ring's capacity, gated or not."""
    count = 70
    engine = L2IoEngine(
        create_store_file(str(tmp_path / "l2"), 2 * count * PAGE),
        2 * count * PAGE,
        2,
        write_gate=gated,
    )
    writes = [(2 * i * PAGE, _page(i % 251)) for i in range(count)]
    try:
        assert asyncio.run(engine.write_batch(writes)) == count * PAGE
        assert asyncio.run(engine.write_batch([])) == 0
        for i in (0, 63, 64, count - 1):
            dst = memoryview(mmap.mmap(-1, PAGE))
            assert engine.read_into(2 * i * PAGE, dst, engine.next_uring()) == PAGE
            assert bytes(dst) == bytes([i % 251]) * PAGE
        assert engine.timing_us["write_io"] > 0
    finally:
        engine.close()


def test_gated_batch_waits_once_per_piece_not_per_extent(tmp_path: Path) -> None:
    """70 page extents form two gated pieces (64 + 6), so two bounded waits."""
    count = 70
    engine = L2IoEngine(
        create_store_file(str(tmp_path / "l2"), count * PAGE),
        count * PAGE,
        2,
        write_gate=True,
    )
    release = threading.Event()
    reader = _hold_read(engine, release)
    try:
        writes = [(i * PAGE, _page(i)) for i in range(count)]
        assert asyncio.run(engine.write_batch(writes)) == count * PAGE
        # Per-extent gating would wait about 70 x 50 ms.
        assert 80_000 <= engine.timing_us["write_gate"] < 1_000_000
    finally:
        release.set()
        reader.join()
        engine.close()


def test_gated_batch_splits_into_gate_sized_pieces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gated pieces hold at most 16 MiB; a longer extent is split."""
    original = NativeIOUring.write_batch_from
    pieces: list[list[tuple[int, int]]] = []

    def recording(
        self: NativeIOUring, fd: int, writes: Sequence[tuple[int, memoryview]]
    ) -> int:
        pieces.append([(offset, len(data)) for offset, data in writes])
        return original(self, fd, writes)

    monkeypatch.setattr(NativeIOUring, "write_batch_from", recording)
    big = 16 * MIB + PAGE
    size = big + PAGE
    engine = L2IoEngine(
        create_store_file(str(tmp_path / "l2"), size), size, 2, write_gate=True
    )
    source = mmap.mmap(-1, big)
    source.write(bytes(range(256)) * (big // 256))
    try:
        written = asyncio.run(
            engine.write_batch([(0, memoryview(source)), (big, _page(9))])
        )
        assert written == size
        assert pieces == [[(0, 16 * MIB)], [(16 * MIB, PAGE), (big, PAGE)]]
        dst = memoryview(mmap.mmap(-1, 2 * PAGE))
        assert engine.read_into(16 * MIB, dst, engine.next_uring()) == 2 * PAGE
        assert bytes(dst[:PAGE]) == source[16 * MIB : big]
        assert bytes(dst[PAGE:]) == bytes([9]) * PAGE
    finally:
        engine.close()


def test_write_batch_raises_after_all_jobs_end(tmp_path: Path) -> None:
    """An out-of-range extent fails the batch; the engine stays usable."""
    engine = L2IoEngine(create_store_file(str(tmp_path / "l2"), 2 * PAGE), 2 * PAGE, 1)
    try:
        with pytest.raises(ValueError):
            asyncio.run(engine.write_batch([(0, _page(1)), (-PAGE, _page(2))]))
        assert asyncio.run(engine.write_batch([(PAGE, _page(5))])) == PAGE
        dst = memoryview(mmap.mmap(-1, PAGE))
        assert engine.read_into(PAGE, dst, engine.next_uring()) == PAGE
        assert bytes(dst) == bytes([5]) * PAGE
    finally:
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
