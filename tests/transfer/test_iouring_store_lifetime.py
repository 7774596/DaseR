# SPDX-License-Identifier: Apache-2.0
"""Protect pinned store destinations while executor copies are still running."""

import asyncio
from pathlib import Path
import threading
from typing import Any

import pytest

from daser.transfer.iouring import TieredIOUringTransferLayer, copy_ops
from daser.transfer.iouring.l1_cache import L1Cache
from daser.transfer.iouring.native import NativeIOUring
from daser.transfer.iouring.pinned_pool import PinnedMemorySlice


async def _store(
    layer: TieredIOUringTransferLayer,
    data: bytes,
    file_offset: int,
    mode: str,
    *,
    accounted_nbytes: int | None = None,
) -> int:
    """Exercise one public store entry point with identical physical geometry."""
    if mode == "single":
        return await layer.store_bytes(
            data, file_offset, len(data), accounted_nbytes=accounted_nbytes
        )
    span = {
        "source_offset": 0,
        "file_offset": file_offset,
        "nbytes": len(data),
        "packed": mode == "packed",
    }
    if accounted_nbytes is not None:
        span["accounted_nbytes"] = accounted_nbytes
    return await layer.store_bytes_grouped(data, [span])


@pytest.mark.parametrize("mode", ["single", "grouped", "packed"])
@pytest.mark.parametrize("outcome", ["success", "error", "cancel", "cancel_twice"])
def test_store_drains_snapshot_before_releasing_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, mode: str
) -> None:
    """A pending copy survives cancellation and wakes stores waiting for its pages."""
    started = threading.Event()
    release = threading.Event()
    ended = threading.Event()
    original = copy_ops.copy_src_to_pinned

    def copy(
        src: Any, pinned: PinnedMemorySlice, target_offset: int, nbytes: int
    ) -> None:
        if bytes(src[:1]) == b"a":
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test copy was not released")
            # Access the actual public view only when the simulated DMA ends.
            # Closing/recycling the allocation before this point is invalid.
            assert len(pinned.view()) == nbytes
            original(src, pinned, target_offset, nbytes)
            ended.set()
            if outcome == "error":
                raise RuntimeError("snapshot copy failed after writing")
        else:
            original(src, pinned, target_offset, nbytes)

    monkeypatch.setattr(copy_ops, "copy_src_to_pinned", copy)

    async def scenario() -> None:
        layer = TieredIOUringTransferLayer(
            path=str(tmp_path / "store-lifetime.store"),
            l1_bytes=4096,
            l2_bytes=8192,
        )
        storing = asyncio.create_task(_store(layer, b"a" * 4096, 0, mode))
        replacing: asyncio.Task[int] | None = None
        try:
            assert await asyncio.to_thread(started.wait, 5)
            replacing = asyncio.create_task(_store(layer, b"b" * 4096, 4096, mode))
            await asyncio.sleep(0.01)
            assert not replacing.done()
            if outcome.startswith("cancel"):
                storing.cancel()
                await asyncio.sleep(0.01)
                assert not storing.done()
                if outcome == "cancel_twice":
                    storing.cancel()
                    await asyncio.sleep(0.01)
                    assert not storing.done()
            assert not ended.is_set()
            release.set()
            if outcome.startswith("cancel"):
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(storing, 5)
            elif outcome == "error":
                with pytest.raises(RuntimeError, match="snapshot copy failed"):
                    await asyncio.wait_for(storing, 5)
            else:
                assert await asyncio.wait_for(storing, 5) == 4096
            assert ended.is_set()
            assert await asyncio.wait_for(replacing, 5) == 4096
            restored = bytearray(4096)
            assert await layer.load_bytes(restored, 4096, 4096) == 4096
            assert restored == b"b" * 4096
        finally:
            release.set()
            await asyncio.gather(
                storing, *([replacing] if replacing else []), return_exceptions=True
            )
            await layer.drain()
            layer.close()

    asyncio.run(scenario())


def test_grouped_store_streams_through_smaller_l1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A contiguous source larger than L1 remains storable without a larger pool."""
    copy_sizes: list[int] = []
    original = copy_ops.copy_src_to_pinned

    def copy(
        src: Any, pinned: PinnedMemorySlice, target_offset: int, nbytes: int
    ) -> None:
        copy_sizes.append(nbytes)
        original(src, pinned, target_offset, nbytes)

    monkeypatch.setattr(copy_ops, "copy_src_to_pinned", copy)

    async def scenario() -> None:
        layer = TieredIOUringTransferLayer(
            path=str(tmp_path / "small-pool.store"),
            l1_bytes=8192,
            l2_bytes=24576,
        )
        source = b"a" * 4096 + b"b" * 4096 + b"c" * 4096
        spans = [
            {
                "source_offset": i * 4096,
                "file_offset": i * 8192,
                "nbytes": 4096,
                "packed": True,
            }
            for i in range(3)
        ]
        try:
            assert await asyncio.wait_for(
                layer.store_bytes_grouped(source, spans), 5
            ) == len(source)
            assert copy_sizes == [8192, 4096]
            await layer.drain()
            for i in range(3):
                restored = bytearray(4096)
                assert await layer.load_bytes(restored, i * 8192, 4096) == 4096
                assert restored == source[i * 4096 : (i + 1) * 4096]
        finally:
            await layer.drain()
            layer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["single", "grouped", "packed"])
def test_cancelled_drain_preserves_ordered_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Repeated drain cancellation preserves both writers and their old buffers."""
    started = threading.Event()
    release = threading.Event()
    writes: list[bytes] = []
    original = NativeIOUring.write

    def write(
        ring: NativeIOUring, fd: int, file_offset: int, data: bytes | memoryview
    ) -> int:
        payload = bytes(data)
        if payload == b"a" * 4096:
            started.set()
            if not release.wait(5):
                raise TimeoutError("first write was not released")
        assert bytes(data) == payload
        result = original(ring, fd, file_offset, data)
        writes.append(payload)
        return result

    monkeypatch.setattr(NativeIOUring, "write", write)

    async def scenario() -> None:
        path = str(tmp_path / "overwrite.store")
        options = {"path": path, "l1_bytes": 8192, "l2_bytes": 16384}
        layer = TieredIOUringTransferLayer(**options)
        draining: asyncio.Task[None] | None = None
        try:
            assert await _store(layer, b"a" * 4096, 0, mode) == 4096
            assert await asyncio.to_thread(started.wait, 5)
            # The later store is accepted into a separate snapshot while the
            # old same-key writer still owns its original allocation.
            assert (
                await asyncio.wait_for(_store(layer, b"b" * 4096, 0, mode), 5) == 4096
            )
            assert not writes
            draining = asyncio.create_task(layer.drain())
            for _ in range(2):
                await asyncio.sleep(0.01)
                draining.cancel()
                await asyncio.sleep(0.01)
                assert not draining.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(draining, 5)
            assert writes == [b"a" * 4096, b"b" * 4096]
            assert layer.stats.l2_writes == 2
            # Reusing the remaining pool page also detects a retained old
            # same-key slice that was forgotten by range-based bookkeeping.
            assert (
                await asyncio.wait_for(_store(layer, b"c" * 4096, 4096, mode), 5)
                == 4096
            )
            await layer.drain()
        finally:
            release.set()
            if draining is not None:
                await asyncio.gather(draining, return_exceptions=True)
            try:
                await layer.drain()
            finally:
                layer.close()
        reader = TieredIOUringTransferLayer(**options)
        try:
            data = bytearray(8192)
            assert await reader.load_bytes(data, 0, 8192) == 8192
            assert data == b"b" * 4096 + b"c" * 4096
        finally:
            await reader.drain()
            reader.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["single", "grouped", "packed"])
def test_cancel_after_publication_keeps_evicted_write_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Cancellation after self-eviction cannot close a pending SSD source."""
    started = threading.Event()
    release = threading.Event()
    original_write = NativeIOUring.write
    original_put = L1Cache.put_reserved_group if mode == "packed" else L1Cache.put

    def write(
        ring: NativeIOUring, fd: int, file_offset: int, data: bytes | memoryview
    ) -> int:
        if file_offset == 32768:
            started.set()
            if not release.wait(5):
                raise TimeoutError("published write was not released")
            assert bytes(data) == b"z" * 4096
        return original_write(ring, fd, file_offset, data)

    def publish(cache: L1Cache, *args: Any, **kwargs: Any) -> None:
        original_put(cache, *args, **kwargs)
        # Cancel at the public cache publication boundary, before the store's
        # final scheduling yield. The write remains a transfer-owned operation.
        task = asyncio.current_task()
        assert task is not None
        task.cancel()

    monkeypatch.setattr(NativeIOUring, "write", write)

    async def scenario() -> None:
        options = {
            "path": str(tmp_path / "cancel-published.store"),
            "l1_bytes": 16384,
            "l2_bytes": 65536,
            "l1_accounting": "raw",
            "bip_enabled": True,
        }
        layer = TieredIOUringTransferLayer(**options)
        try:
            await _store(layer, b"a" * 4096, 0, mode, accounted_nbytes=12288)
            await layer.drain()
            with monkeypatch.context() as patch:
                patch.setattr(
                    L1Cache,
                    "put_reserved_group" if mode == "packed" else "put",
                    publish,
                )
                storing = asyncio.create_task(
                    _store(layer, b"z" * 4096, 32768, mode, accounted_nbytes=8192)
                )
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(storing, 5)
            assert await asyncio.to_thread(started.wait, 5)
            # This allocation pressures/reuses freed pages; a wrongly closed
            # published source would be recycled before its native write ends.
            replacing = asyncio.create_task(_store(layer, b"c" * 12288, 4096, mode))
            await asyncio.sleep(0.01)
            release.set()
            assert await asyncio.wait_for(replacing, 5) == 12288
            await layer.drain()
        finally:
            release.set()
            try:
                await layer.drain()
            finally:
                layer.close()
        reader = TieredIOUringTransferLayer(**options)
        try:
            data = bytearray(4096)
            assert await reader.load_bytes(data, 32768, 4096) == 4096
            assert data == b"z" * 4096
        finally:
            await reader.drain()
            reader.close()

    asyncio.run(scenario())


def test_drain_reports_write_error_after_other_native_writes_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An already-recorded failure cannot let drain abandon another SSD owner."""
    failed_started = threading.Event()
    other_started = threading.Event()
    fail = threading.Event()
    release = threading.Event()
    failed = threading.Event()
    other_finished = threading.Event()
    original = NativeIOUring.write

    def write(
        ring: NativeIOUring, fd: int, file_offset: int, data: bytes | memoryview
    ) -> int:
        if file_offset == 0:
            failed_started.set()
            if not fail.wait(5):
                raise TimeoutError("failed write was not released")
            failed.set()
            raise OSError("injected SSD write failure")
        other_started.set()
        if not release.wait(5):
            raise TimeoutError("other write was not released")
        result = original(ring, fd, file_offset, data)
        other_finished.set()
        return result

    monkeypatch.setattr(NativeIOUring, "write", write)

    async def scenario() -> None:
        layer = TieredIOUringTransferLayer(
            path=str(tmp_path / "write-error.store"), l1_bytes=8192, l2_bytes=16384
        )
        draining: asyncio.Task[None] | None = None
        try:
            await layer.store_bytes(b"a" * 4096, 0, 4096)
            await layer.store_bytes(b"b" * 4096, 4096, 4096)
            assert await asyncio.to_thread(failed_started.wait, 5)
            assert await asyncio.to_thread(other_started.wait, 5)
            fail.set()
            assert await asyncio.to_thread(failed.wait, 5)
            # Let the failure complete its event-loop bookkeeping so drain
            # encounters a recorded error alongside an unrelated active write.
            await asyncio.sleep(0.02)
            draining = asyncio.create_task(layer.drain())
            await asyncio.sleep(0.02)
            assert not draining.done()
            assert not other_finished.is_set()
            release.set()
            with pytest.raises(
                RuntimeError, match="asynchronous io_uring L2 write failed"
            ):
                await asyncio.wait_for(draining, 5)
            assert other_finished.is_set()
            assert layer.stats.l2_writes == 1
        finally:
            fail.set()
            release.set()
            if draining is not None:
                await asyncio.gather(draining, return_exceptions=True)
            try:
                await layer.drain()
            except RuntimeError:
                pass
            layer.close()

    asyncio.run(scenario())
