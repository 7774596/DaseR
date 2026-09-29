# SPDX-License-Identifier: Apache-2.0
"""Held packed stores stay in L1 until persisted, discarded or evicted."""

import asyncio
from pathlib import Path

from daser.transfer.iouring import TieredIOUringTransferLayer

PAGE = 4096


def _span(file_offset: int, *, hold: bool) -> dict[str, object]:
    return {
        "source_offset": 0,
        "file_offset": file_offset,
        "nbytes": PAGE,
        "packed": True,
        "hold": hold,
    }


def _layer(tmp_path: Path, l1_pages: int) -> TieredIOUringTransferLayer:
    return TieredIOUringTransferLayer(
        path=str(tmp_path / "held.store"),
        l1_bytes=l1_pages * PAGE,
        l2_bytes=8 * PAGE,
    )


def test_held_store_serves_l1_without_l2_write(tmp_path: Path) -> None:
    async def scenario() -> None:
        layer = _layer(tmp_path, 4)
        await layer.store_bytes_grouped(b"a" * PAGE, [_span(0, hold=True)])
        await layer.drain()
        assert layer.stats.l2_writes == 0
        dst = bytearray(PAGE)
        assert await layer.load_bytes(dst, 0, PAGE) == PAGE
        assert bytes(dst) == b"a" * PAGE
        layer.close()

    asyncio.run(scenario())


def test_persist_held_writes_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        layer = _layer(tmp_path, 4)
        await layer.store_bytes_grouped(b"a" * PAGE, [_span(0, hold=True)])
        layer.persist_held([(0, PAGE)])
        layer.persist_held([(0, PAGE)])
        await layer.drain()
        assert layer.stats.l2_writes == 1
        assert layer.stats.l2_write_bytes == PAGE
        layer.close()

    asyncio.run(scenario())


def test_discard_held_drops_l1_without_l2_write(tmp_path: Path) -> None:
    async def scenario() -> None:
        layer = _layer(tmp_path, 4)
        await layer.store_bytes_grouped(b"a" * PAGE, [_span(0, hold=True)])
        assert layer.held_bytes == PAGE
        layer.discard_held([(0, PAGE)])
        await layer.drain()
        assert layer.stats.l2_writes == 0
        assert layer.l1_bytes_used == 0
        assert layer.held_bytes == 0
        layer.close()

    asyncio.run(scenario())


def test_held_range_evicted_from_l1_is_written_back(tmp_path: Path) -> None:
    async def scenario() -> None:
        layer = _layer(tmp_path, 1)
        await layer.store_bytes_grouped(b"a" * PAGE, [_span(0, hold=True)])
        # A second store needs the only L1 page and evicts the held range.
        await layer.store_bytes_grouped(b"b" * PAGE, [_span(PAGE, hold=False)])
        await layer.drain()
        assert layer.stats.held_evict_persists == 1
        assert layer.stats.l2_writes == 2
        assert layer.held_bytes == 0
        dst = bytearray(PAGE)
        assert await layer.load_bytes(dst, 0, PAGE) == PAGE
        assert bytes(dst) == b"a" * PAGE
        layer.close()

    asyncio.run(scenario())
