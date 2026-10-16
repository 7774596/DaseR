# SPDX-License-Identifier: Apache-2.0

"""Public contract of the io_uring data mover (arena, staging, L2)."""

# Standard
import asyncio
from collections.abc import Callable, Sequence
import errno
import os
from pathlib import Path
import threading
from typing import Any

# Third Party
import numpy
import pytest

# First Party
from daser.transfer import Move
from daser.transfer.iouring import TieredIOUringTransferLayer, copy_ops
from daser.transfer.iouring.native import NativeIOUring
from tests.transfer.cuda import require_cuda
from tests.transfer.store_files import create_store_file

PAGE = 4096
MIB = 1024 * 1024


def _pattern(nbytes: int, seed: int) -> bytes:
    """Return ``nbytes`` of a seed-dependent byte pattern."""
    values = numpy.arange(nbytes, dtype=numpy.uint32) * 7 + seed
    return (values % 251).astype(numpy.uint8).tobytes()


def _layer(
    tmp_path: Path,
    *,
    l1: int = 0,
    staging: int = 4 * PAGE,
    l2: int = 16 * PAGE,
    **kwargs: Any,
) -> TieredIOUringTransferLayer:
    """Build a layer over a fresh store file of ``l2`` bytes."""
    path = create_store_file(str(tmp_path / "l2.store"), l2)
    return TieredIOUringTransferLayer(
        path, l1_bytes=l1, staging_bytes=staging, l2_bytes=l2, **kwargs
    )


def _write_file(path: Path, offset: int, data: bytes) -> None:
    """Write ``data`` into an existing store file at ``offset``."""
    fd = os.open(path, os.O_WRONLY)
    try:
        assert os.pwrite(fd, data, offset) == len(data)
    finally:
        os.close(fd)


def _read_file(path: Path, offset: int, nbytes: int) -> bytes:
    """Read ``nbytes`` of a store file at ``offset``."""
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(nbytes)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"l1_bytes": 100, "staging_bytes": PAGE}, "l1_bytes"),
        ({"l1_bytes": 0, "staging_bytes": 100}, "staging_bytes"),
        ({"l1_bytes": 0, "staging_bytes": 0}, "staging_bytes"),
        (
            {"l1_bytes": 0, "staging_bytes": PAGE, "l2_read_stripe_bytes": 10},
            "l2_read_stripe_bytes",
        ),
        ({"l1_bytes": 0, "staging_bytes": PAGE, "io_workers": 0}, "io_workers"),
        ({"l1_bytes": 0, "staging_bytes": 0, "skip_l2": True}, "l1_bytes"),
        ({"l1_bytes": PAGE, "staging_bytes": PAGE, "skip_l2": True}, "staging"),
    ],
)
def test_constructor_rejects_invalid_sizes(
    tmp_path: Path, kwargs: dict[str, Any], match: str
) -> None:
    path = create_store_file(str(tmp_path / "l2.store"), 4 * PAGE)
    with pytest.raises(ValueError, match=match):
        TieredIOUringTransferLayer(path, l2_bytes=4 * PAGE, **kwargs)


def test_constructor_needs_server_created_store_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.store"
    with pytest.raises(FileNotFoundError):
        TieredIOUringTransferLayer(
            str(missing), l1_bytes=0, staging_bytes=PAGE, l2_bytes=PAGE
        )
    assert not missing.exists()
    short = create_store_file(str(tmp_path / "short.store"), PAGE)
    with pytest.raises(ValueError, match="has size"):
        TieredIOUringTransferLayer(
            short, l1_bytes=0, staging_bytes=PAGE, l2_bytes=2 * PAGE
        )


def test_staging_store_writes_file_bytes_with_o_direct(tmp_path: Path) -> None:
    """A staging store persists the exact bytes at the requested L2 offsets."""

    async def scenario() -> None:
        layer = _layer(tmp_path)
        payload = _pattern(3 * PAGE, 1)
        try:
            handle = await layer.store(
                payload,
                [
                    Move(0, PAGE, l2_offset=8 * PAGE),
                    Move(PAGE, 2 * PAGE, l2_offset=2 * PAGE),
                ],
            )
            await handle.staged
            await handle.persisted
            await layer.drain()
        finally:
            await layer.close()
        store = tmp_path / "l2.store"
        assert _read_file(store, 8 * PAGE, PAGE) == payload[:PAGE]
        assert _read_file(store, 2 * PAGE, 2 * PAGE) == payload[PAGE:]
        assert layer.stats.l2_writes == 2
        assert layer.stats.l2_write_bytes == 3 * PAGE

    asyncio.run(scenario())


def test_staging_roundtrip_larger_than_staging_pool(tmp_path: Path) -> None:
    """Stores and loads larger than the pool stream through waiting slices."""

    async def scenario() -> None:
        layer = _layer(tmp_path, staging=2 * PAGE)
        payload = _pattern(8 * PAGE, 2)
        moves = [Move(0, 8 * PAGE, l2_offset=4 * PAGE)]
        try:
            handle = await layer.store(payload, moves)
            await asyncio.wait_for(handle.persisted, 5)
            assert layer.stats.l2_writes == 4
            waits_after_store = layer.stats.staging_waits
            assert waits_after_store >= 1
            dst = bytearray(8 * PAGE)
            await asyncio.wait_for(layer.load(dst, moves, "l2"), 5)
            assert bytes(dst) == payload
            assert layer.stats.staging_waits > waits_after_store
            assert layer.debug_state()["staging_used_bytes"] == 0
        finally:
            await layer.close()

    asyncio.run(scenario())


def test_contiguous_moves_coalesce_into_one_io(tmp_path: Path) -> None:
    """Moves contiguous in buffer and L2 share one write, read and copy."""

    async def scenario() -> None:
        layer = _layer(tmp_path)
        payload = _pattern(2 * PAGE, 3)
        moves = [Move(PAGE, PAGE, l2_offset=PAGE), Move(0, PAGE, l2_offset=0)]
        try:
            await (await layer.store(payload, moves)).persisted
            assert layer.stats.l2_writes == 1
            assert layer.stats.d2h_copies == 1
            dst = bytearray(2 * PAGE)
            await layer.load(dst, moves, "l2")
            assert bytes(dst) == payload
            assert layer.stats.l2_reads == 1
            assert layer.stats.dst_copies == 1
        finally:
            await layer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("tier", ["staging", "arena"])
def test_staged_resolves_before_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier: str
) -> None:
    """``staged`` resolves after the snapshot, ``persisted`` after the write."""
    original = NativeIOUring.write_batch_from
    started = threading.Event()
    release = threading.Event()

    def held_write(
        self: NativeIOUring, fd: int, writes: Sequence[tuple[int, memoryview]]
    ) -> int:
        started.set()
        assert release.wait(10)
        return original(self, fd, writes)

    monkeypatch.setattr(NativeIOUring, "write_batch_from", held_write)

    async def scenario() -> None:
        layer = _layer(tmp_path, l1=4 * PAGE)
        payload = _pattern(PAGE, 4)
        l1 = 0 if tier == "arena" else None
        move = Move(0, PAGE, l1_offset=l1, l2_offset=PAGE)
        try:
            handle = await layer.store(payload, [move])
            await asyncio.wait_for(handle.staged, 5)
            assert await asyncio.to_thread(started.wait, 5)
            assert not handle.persisted.done()
            if tier == "arena":
                # Arena bytes are valid once staged, before L2 persisted.
                resident = bytearray(PAGE)
                await layer.load(resident, [Move(0, PAGE, l1_offset=0)], "l1")
                assert bytes(resident) == payload
            release.set()
            await asyncio.wait_for(handle.persisted, 5)
            restored = bytearray(PAGE)
            await layer.load(restored, [Move(0, PAGE, l2_offset=PAGE)], "l2")
            assert bytes(restored) == payload
        finally:
            release.set()
            await layer.close()

    asyncio.run(scenario())


def test_large_arena_store_writes_while_later_groups_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arena snapshots run in 128 MiB groups; L2 writes start group by group."""
    original_copy = copy_ops.copy_src_to_pinned
    original_write = NativeIOUring.write_batch_from
    copies: list[int] = []
    first_written = threading.Event()

    def copy(
        src: Any, pinned: copy_ops.HostMemory, target_offset: int, nbytes: int
    ) -> None:
        copies.append(nbytes)
        if len(copies) == 2:
            # The second group's D2H overlaps the first group's L2 write.
            assert first_written.wait(10)
        original_copy(src, pinned, target_offset, nbytes)

    def write(
        self: NativeIOUring, fd: int, writes: Sequence[tuple[int, memoryview]]
    ) -> int:
        written = original_write(self, fd, writes)
        first_written.set()
        return written

    monkeypatch.setattr(copy_ops, "copy_src_to_pinned", copy)
    monkeypatch.setattr(NativeIOUring, "write_batch_from", write)
    nbytes = 128 * MIB + 2 * PAGE

    async def scenario() -> None:
        layer = _layer(tmp_path, l1=nbytes, l2=nbytes)
        payload = _pattern(nbytes, 12)
        try:
            handle = await layer.store(
                payload, [Move(0, nbytes, l1_offset=0, l2_offset=0)]
            )
            await asyncio.wait_for(handle.persisted, 20)
            assert copies == [128 * MIB, 2 * PAGE]
            assert layer.stats.d2h_copies == 2
            assert layer.stats.l2_writes == 2
        finally:
            await layer.close()
        assert _read_file(tmp_path / "l2.store", 0, nbytes) == payload

    asyncio.run(scenario())


def test_persist_error_surfaces_on_handle_and_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed L2 write fails ``persisted`` only and is raised once by drain."""

    def failing_write(
        self: NativeIOUring, fd: int, writes: Sequence[tuple[int, memoryview]]
    ) -> int:
        raise OSError(errno.EIO, "injected write failure")

    monkeypatch.setattr(NativeIOUring, "write_batch_from", failing_write)

    async def scenario() -> None:
        layer = _layer(tmp_path)
        try:
            handle = await layer.store(b"e" * PAGE, [Move(0, PAGE, l2_offset=0)])
            await asyncio.wait_for(handle.staged, 5)
            with pytest.raises(OSError, match="injected write failure"):
                await asyncio.wait_for(handle.persisted, 5)
            with pytest.raises(RuntimeError, match="store failed") as raised:
                await layer.drain()
            assert isinstance(raised.value.__cause__, OSError)
            await layer.drain()
            assert layer.stats.l2_writes == 0
            assert layer.debug_state()["staging_used_bytes"] == 0
        finally:
            await layer.close()

    asyncio.run(scenario())


def test_promote_then_load_from_arena(tmp_path: Path) -> None:
    """``promote`` fills arena extents that ``load("l1")`` then serves."""

    async def scenario() -> None:
        layer = _layer(tmp_path, l1=4 * PAGE)
        payload = _pattern(2 * PAGE, 5)
        _write_file(tmp_path / "l2.store", 6 * PAGE, payload)
        try:
            await layer.promote(
                [
                    Move(0, PAGE, l1_offset=3 * PAGE, l2_offset=6 * PAGE),
                    Move(0, PAGE, l1_offset=PAGE, l2_offset=7 * PAGE),
                ]
            )
            assert layer.stats.l2_reads == 2
            assert layer.stats.dst_copies == 0
            dst = bytearray(2 * PAGE)
            await layer.load(
                dst,
                [
                    Move(0, PAGE, l1_offset=3 * PAGE),
                    Move(PAGE, PAGE, l1_offset=PAGE),
                ],
                "l1",
            )
            assert bytes(dst) == payload
            assert layer.stats.l2_reads == 2
        finally:
            await layer.close()

    asyncio.run(scenario())


def test_l2_load_with_arena_fill(tmp_path: Path) -> None:
    """An L2 load with ``l1_offset`` also fills that arena extent."""

    async def scenario() -> None:
        layer = _layer(tmp_path, l1=4 * PAGE)
        payload = _pattern(2 * PAGE, 6)
        _write_file(tmp_path / "l2.store", 0, payload)
        try:
            dst = bytearray(2 * PAGE)
            await layer.load(
                dst,
                [
                    Move(0, PAGE, l1_offset=2 * PAGE, l2_offset=0),
                    Move(PAGE, PAGE, l2_offset=PAGE),
                ],
                "l2",
            )
            assert bytes(dst) == payload
            reads = layer.stats.l2_reads
            resident = bytearray(PAGE)
            await layer.load(resident, [Move(0, PAGE, l1_offset=2 * PAGE)], "l1")
            assert bytes(resident) == payload[:PAGE]
            assert layer.stats.l2_reads == reads
        finally:
            await layer.close()

    asyncio.run(scenario())


def test_skip_l2_arena_only_roundtrip(tmp_path: Path) -> None:
    """Without L2 the arena is the only tier; L2 moves are rejected."""

    async def scenario() -> None:
        layer = TieredIOUringTransferLayer(
            str(tmp_path / "unused.store"),
            l1_bytes=4 * PAGE,
            staging_bytes=0,
            l2_bytes=0,
            skip_l2=True,
        )
        payload = _pattern(3 * PAGE, 7)
        try:
            handle = await layer.store(
                payload,
                [Move(0, PAGE, l1_offset=3 * PAGE), Move(PAGE, 2 * PAGE, l1_offset=0)],
            )
            await handle.staged
            await handle.persisted
            dst = bytearray(3 * PAGE)
            await layer.load(
                dst,
                [Move(0, PAGE, l1_offset=3 * PAGE), Move(PAGE, 2 * PAGE, l1_offset=0)],
                "l1",
            )
            assert bytes(dst) == payload
            assert layer.stats.l2_writes == 0
            assert layer.l2_timing_us == {}
            with pytest.raises(ValueError, match="no L2 tier"):
                await layer.store(payload, [Move(0, PAGE, l2_offset=0)])
            with pytest.raises(ValueError, match="no L2 tier"):
                await layer.load(dst, [Move(0, PAGE, l2_offset=0)], "l2")
            with pytest.raises(NotImplementedError):
                await layer.promote([Move(0, PAGE, l1_offset=0, l2_offset=0)])
        finally:
            await layer.close()
        assert not (tmp_path / "unused.store").exists()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("op", "move", "match"),
    [
        ("store", Move(0, PAGE, l2_offset=100), "aligned"),
        ("store", Move(0, 100, l2_offset=0), "aligned"),
        ("store", Move(0, PAGE, l1_offset=100, l2_offset=0), "aligned"),
        ("store", Move(0, PAGE, l2_offset=16 * PAGE), "L2 store"),
        ("store", Move(0, PAGE, l1_offset=4 * PAGE), "arena"),
        ("store", Move(PAGE, PAGE, l2_offset=0), "buffer"),
        ("store", Move(0, 0, l2_offset=0), "positive"),
        ("store", Move(0, PAGE), "no tier"),
        ("l1", Move(0, PAGE, l2_offset=0), "l1_offset"),
        ("l2", Move(0, PAGE, l1_offset=0), "l2_offset"),
        ("l2", Move(0, PAGE, l2_offset=-PAGE), "L2 store"),
        ("promote", Move(0, PAGE, l2_offset=0), "l1_offset"),
    ],
)
def test_invalid_moves_rejected_before_any_io(
    tmp_path: Path, op: str, move: Move, match: str
) -> None:
    """A bad move fails the whole batch before anything is submitted."""

    async def scenario() -> None:
        layer = _layer(tmp_path, l1=4 * PAGE)
        good = Move(0, PAGE, l1_offset=0 if op == "l1" else None, l2_offset=0)
        try:
            with pytest.raises(ValueError, match=match):
                if op == "store":
                    await layer.store(b"v" * PAGE, [good, move])
                elif op == "promote":
                    await layer.promote([move])
                else:
                    await layer.load(bytearray(PAGE), [good, move], op)
            assert layer.stats.l2_reads == layer.stats.l2_writes == 0
            assert layer.stats.d2h_copies == layer.stats.dst_copies == 0
        finally:
            await layer.close()

    asyncio.run(scenario())


def test_load_rejects_unknown_source(tmp_path: Path) -> None:
    async def scenario() -> None:
        layer = _layer(tmp_path)
        try:
            with pytest.raises(ValueError, match="source"):
                await layer.load(bytearray(PAGE), [], "l3")  # type: ignore[arg-type]
        finally:
            await layer.close()

    asyncio.run(scenario())


def test_load_runs_rejects_invalid_runs(tmp_path: Path) -> None:
    """``load_runs`` validates the run arrays against the arena."""

    async def scenario() -> None:
        layer = _layer(tmp_path, l1=4 * PAGE)
        empty = numpy.zeros(0, dtype=numpy.int64)
        one = numpy.array([0], dtype=numpy.int64)
        size = numpy.array([PAGE], dtype=numpy.int64)
        try:
            with pytest.raises(ValueError, match="nonempty"):
                await layer.load_runs(None, empty, empty, empty)
            with pytest.raises(ValueError, match="equal length"):
                await layer.load_runs(None, one, numpy.array([0, 1]), size)
            with pytest.raises(ValueError, match="arena"):
                await layer.load_runs(None, numpy.array([4 * PAGE]), one, size)
            with pytest.raises(ValueError, match="arena"):
                await layer.load_runs(None, one, one, numpy.array([0]))
            with pytest.raises(ValueError, match="destination"):
                await layer.load_runs(bytearray(PAGE), one, numpy.array([1]), size)
            with pytest.raises(ValueError, match="destination"):
                await layer.load_runs(None, one, numpy.array([-1]), size)
        finally:
            await layer.close()
        arena_less = _layer(tmp_path)
        try:
            with pytest.raises(NotImplementedError):
                await arena_less.load_runs(None, one, one, size)
        finally:
            await arena_less.close()

    asyncio.run(scenario())


def test_load_runs_copies_arena_runs_to_cuda(tmp_path: Path) -> None:
    cupy = require_cuda()

    async def scenario() -> None:
        layer = _layer(tmp_path, l1=8 * PAGE)
        payload = _pattern(8 * PAGE, 8)
        try:
            await (await layer.store(payload, [Move(0, 8 * PAGE, l1_offset=0)])).staged
            dst = cupy.zeros(3 * PAGE, dtype=cupy.uint8)
            await layer.load_runs(
                dst,
                numpy.array([5 * PAGE, 0, 2 * PAGE], dtype=numpy.int64),
                numpy.array([0, PAGE, 2 * PAGE], dtype=numpy.int64),
                numpy.array([PAGE, PAGE, PAGE], dtype=numpy.int64),
            )
            restored = cupy.asnumpy(dst).tobytes()
            expected = payload[5 * PAGE : 6 * PAGE] + payload[:PAGE]
            assert restored == expected + payload[2 * PAGE : 3 * PAGE]
            assert layer.stats.dst_copy_runs == 3
        finally:
            await layer.close()

    asyncio.run(scenario())


def test_grouped_arena_load_merges_cuda_copy_runs(tmp_path: Path) -> None:
    """Arena chunks contiguous in source and destination share one CUDA run."""
    cupy = require_cuda()

    async def scenario() -> None:
        layer = _layer(tmp_path, l1=8 * PAGE)
        payload = _pattern(8 * PAGE, 9)
        try:
            await (await layer.store(payload, [Move(0, 8 * PAGE, l1_offset=0)])).staged
            dst = cupy.zeros(4 * PAGE, dtype=cupy.uint8)
            await layer.load(
                dst,
                [
                    Move(0, PAGE, l1_offset=0),
                    Move(PAGE, PAGE, l1_offset=PAGE),
                    Move(2 * PAGE, PAGE, l1_offset=5 * PAGE),
                    Move(3 * PAGE, PAGE, l1_offset=6 * PAGE),
                ],
                "l1",
            )
            restored = cupy.asnumpy(dst).tobytes()
            assert restored == payload[: 2 * PAGE] + payload[5 * PAGE : 7 * PAGE]
            assert layer.stats.dst_copies == 1
            assert layer.stats.dst_copy_runs == 2
        finally:
            await layer.close()

    asyncio.run(scenario())


def test_many_small_reads_split_into_ring_sized_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scattered reads are submitted in native batches of at most 64."""
    original = NativeIOUring.read_batch_into
    batches: list[int] = []

    def counting(
        self: NativeIOUring,
        fd: int,
        reads: Sequence[tuple[int, memoryview]],
        on_complete: Callable[[int, int], None] | None = None,
    ) -> int:
        batches.append(len(reads))
        return original(self, fd, reads, on_complete)

    monkeypatch.setattr(NativeIOUring, "read_batch_into", counting)
    count = 100

    async def scenario() -> None:
        layer = _layer(tmp_path, staging=256 * PAGE, l2=2 * count * PAGE, io_workers=2)
        payload = _pattern(count * PAGE, 10)
        moves = [Move(i * PAGE, PAGE, l2_offset=2 * i * PAGE) for i in range(count)]
        try:
            await (await layer.store(payload, moves)).persisted
            dst = bytearray(count * PAGE)
            await layer.load(dst, moves, "l2")
            assert bytes(dst) == payload
            assert layer.stats.l2_reads == count
        finally:
            await layer.close()

    asyncio.run(scenario())
    assert sum(batches) == count
    assert max(batches) <= 64


def test_scattered_arena_store_writes_in_ring_sized_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A group's scattered L2 runs share native write batches of at most 64."""
    original = NativeIOUring.write_batch_from
    batches: list[int] = []

    def counting(
        self: NativeIOUring, fd: int, writes: Sequence[tuple[int, memoryview]]
    ) -> int:
        batches.append(len(writes))
        return original(self, fd, writes)

    monkeypatch.setattr(NativeIOUring, "write_batch_from", counting)
    count = 100

    async def scenario() -> None:
        layer = _layer(tmp_path, l1=count * PAGE, l2=2 * count * PAGE, io_workers=2)
        payload = _pattern(count * PAGE, 13)
        moves = [
            Move(i * PAGE, PAGE, l1_offset=i * PAGE, l2_offset=2 * i * PAGE)
            for i in range(count)
        ]
        try:
            await (await layer.store(payload, moves)).persisted
            assert layer.stats.l2_writes == count
            assert layer.stats.l2_write_bytes == count * PAGE
        finally:
            await layer.close()
        store = tmp_path / "l2.store"
        for i in (0, 63, 64, count - 1):
            expected = payload[i * PAGE : (i + 1) * PAGE]
            assert _read_file(store, 2 * i * PAGE, PAGE) == expected

    asyncio.run(scenario())
    assert sorted(batches) == [36, 64]


@pytest.mark.parametrize("stripe", [0, 16 * MIB])
def test_large_arena_read_striped_or_serial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stripe: int
) -> None:
    """A read above 128 MiB is striped into one submission when enabled."""
    original = NativeIOUring.read_batch_into
    batches: list[int] = []

    def counting(
        self: NativeIOUring,
        fd: int,
        reads: Sequence[tuple[int, memoryview]],
        on_complete: Callable[[int, int], None] | None = None,
    ) -> int:
        batches.append(len(reads))
        return original(self, fd, reads, on_complete)

    monkeypatch.setattr(NativeIOUring, "read_batch_into", counting)
    nbytes = 130 * MIB

    async def scenario() -> None:
        layer = _layer(tmp_path, l1=nbytes, l2=nbytes, l2_read_stripe_bytes=stripe)
        payload = _pattern(nbytes, 11)
        _write_file(tmp_path / "l2.store", 0, payload)
        try:
            dst = bytearray(nbytes)
            await layer.load(dst, [Move(0, nbytes, l1_offset=0, l2_offset=0)], "l2")
            assert dst == payload
            assert layer.stats.l2_reads == 1
        finally:
            await layer.close()

    asyncio.run(scenario())
    assert batches == ([] if stripe == 0 else [9])


@pytest.mark.parametrize("gated", [False, True])
def test_write_gate_waits_for_in_flight_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gated: bool
) -> None:
    """With the gate on, an L2 write waits (bounded) while a read is in IO."""
    original = NativeIOUring.read_batch_into
    started = threading.Event()
    release = threading.Event()

    def held_read(
        self: NativeIOUring,
        fd: int,
        reads: Sequence[tuple[int, memoryview]],
        on_complete: Callable[[int, int], None] | None = None,
    ) -> int:
        started.set()
        assert release.wait(10)
        return original(self, fd, reads, on_complete)

    monkeypatch.setattr(NativeIOUring, "read_batch_into", held_read)

    async def scenario() -> None:
        layer = _layer(tmp_path, l2_write_gate=gated)
        loading = asyncio.create_task(
            layer.load(bytearray(PAGE), [Move(0, PAGE, l2_offset=0)], "l2")
        )
        try:
            assert await asyncio.to_thread(started.wait, 5)
            handle = await layer.store(b"w" * PAGE, [Move(0, PAGE, l2_offset=PAGE)])
            await asyncio.wait_for(handle.persisted, 5)
            assert not loading.done()
            waited = layer.l2_timing_us["write_gate"]
            assert waited >= 40_000 if gated else waited == 0
            release.set()
            await asyncio.wait_for(loading, 5)
        finally:
            release.set()
            await asyncio.gather(loading, return_exceptions=True)
            await layer.close()

    asyncio.run(scenario())


def test_close_drains_then_rejects_new_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``close`` waits for in-flight stores, is idempotent and final."""
    original = NativeIOUring.write_batch_from
    release = threading.Event()

    def held_write(
        self: NativeIOUring, fd: int, writes: Sequence[tuple[int, memoryview]]
    ) -> int:
        assert release.wait(10)
        return original(self, fd, writes)

    monkeypatch.setattr(NativeIOUring, "write_batch_from", held_write)

    async def scenario() -> None:
        layer = _layer(tmp_path)
        handle = await layer.store(b"c" * PAGE, [Move(0, PAGE, l2_offset=0)])
        closing = asyncio.create_task(layer.close())
        await asyncio.sleep(0.02)
        assert not closing.done()
        assert layer.debug_state()["closing"]
        with pytest.raises(RuntimeError, match="closing"):
            await layer.store(b"c" * PAGE, [Move(0, PAGE, l2_offset=0)])
        with pytest.raises(RuntimeError, match="closing"):
            await layer.load(bytearray(PAGE), [Move(0, PAGE, l2_offset=0)], "l2")
        release.set()
        await asyncio.wait_for(closing, 5)
        assert handle.persisted.done() and handle.persisted.exception() is None
        await layer.close()

    try:
        asyncio.run(scenario())
    finally:
        release.set()
    assert _read_file(tmp_path / "l2.store", 0, PAGE) == b"c" * PAGE
