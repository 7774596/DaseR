# SPDX-License-Identifier: Apache-2.0
"""Protect pinned store snapshots while executor copies are still running."""

# Standard
import asyncio
from pathlib import Path
import threading
from typing import Any

# Third Party
import pytest

# First Party
from daser.transfer import Move
from daser.transfer.iouring import TieredIOUringTransferLayer, copy_ops
from tests.transfer.store_files import create_store_file

PAGE = 4096


def _layer(path: Path, staging: int, l2: int) -> TieredIOUringTransferLayer:
    """Build a staging-only layer over a fresh store file."""
    return TieredIOUringTransferLayer(
        create_store_file(str(path), l2),
        l1_bytes=0,
        staging_bytes=staging,
        l2_bytes=l2,
    )


@pytest.mark.parametrize("outcome", ["success", "error"])
def test_pending_snapshot_keeps_staging_slice_until_copy_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """A store waiting for staging proceeds only after the slow snapshot ends."""
    started = threading.Event()
    release = threading.Event()
    ended = threading.Event()
    original = copy_ops.copy_src_to_pinned

    def copy(
        src: Any, pinned: copy_ops.HostMemory, target_offset: int, nbytes: int
    ) -> None:
        if bytes(src[:1]) == b"a":
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test copy was not released")
            # The slice must still be live when the simulated DMA lands.
            original(src, pinned, target_offset, nbytes)
            ended.set()
            if outcome == "error":
                raise RuntimeError("snapshot copy failed after writing")
        else:
            original(src, pinned, target_offset, nbytes)

    monkeypatch.setattr(copy_ops, "copy_src_to_pinned", copy)

    async def scenario() -> None:
        layer = _layer(tmp_path / "store-lifetime.store", PAGE, 2 * PAGE)
        try:
            first = await layer.store(b"a" * PAGE, [Move(0, PAGE, l2_offset=0)])
            assert await asyncio.to_thread(started.wait, 5)
            second = await layer.store(b"b" * PAGE, [Move(0, PAGE, l2_offset=PAGE)])
            await asyncio.sleep(0.01)
            assert not first.staged.done()
            assert not second.staged.done()
            assert layer.debug_state()["staging_waiters"] == 1
            release.set()
            if outcome == "error":
                for future in (first.staged, first.persisted):
                    with pytest.raises(RuntimeError, match="snapshot copy failed"):
                        await asyncio.wait_for(future, 5)
            else:
                await asyncio.wait_for(first.persisted, 5)
            assert ended.is_set()
            await asyncio.wait_for(second.persisted, 5)
            if outcome == "error":
                with pytest.raises(RuntimeError, match="store failed"):
                    await layer.drain()
            restored = bytearray(PAGE)
            await layer.load(restored, [Move(0, PAGE, l2_offset=PAGE)], "l2")
            assert restored == b"b" * PAGE
            assert layer.stats.l2_writes == (1 if outcome == "error" else 2)
        finally:
            release.set()
            await layer.close()

    asyncio.run(scenario())


def test_store_streams_through_smaller_staging_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scattered moves larger than staging are snapshotted in reserved groups."""
    copy_sizes: list[int] = []
    original = copy_ops.copy_src_to_pinned

    def copy(
        src: Any, pinned: copy_ops.HostMemory, target_offset: int, nbytes: int
    ) -> None:
        copy_sizes.append(nbytes)
        original(src, pinned, target_offset, nbytes)

    monkeypatch.setattr(copy_ops, "copy_src_to_pinned", copy)

    async def scenario() -> None:
        layer = _layer(tmp_path / "small-pool.store", 2 * PAGE, 6 * PAGE)
        source = b"a" * PAGE + b"b" * PAGE + b"c" * PAGE
        moves = [Move(i * PAGE, PAGE, l2_offset=2 * i * PAGE) for i in range(3)]
        try:
            handle = await layer.store(source, moves)
            await asyncio.wait_for(handle.persisted, 5)
            # Two slices fill the pool and share one snapshot; the third
            # waits for a write to release its slice.
            assert layer.stats.d2h_copies == 2
            assert copy_sizes == [PAGE] * 3
            assert layer.stats.staging_waits == 1
            for i in range(3):
                restored = bytearray(PAGE)
                await layer.load(
                    restored, [Move(0, PAGE, l2_offset=2 * i * PAGE)], "l2"
                )
                assert restored == source[i * PAGE : (i + 1) * PAGE]
        finally:
            await layer.close()

    asyncio.run(scenario())
