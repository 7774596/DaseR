# SPDX-License-Identifier: Apache-2.0
"""Held packed stores stay in L1 until discarded or evicted."""

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


def test_discard_held_drops_l1_without_l2_write(tmp_path: Path) -> None:
    async def scenario() -> None:
        layer = _layer(tmp_path, 4)
        released: list[tuple[int, int]] = []
        layer.set_held_release_listener(lambda *key: released.append(key))
        await layer.store_bytes_grouped(b"a" * PAGE, [_span(0, hold=True)])
        layer.discard_held(0, PAGE)
        await layer.drain()
        assert layer.stats.l2_writes == 0
        assert layer.l1_bytes_used == 0
        assert released == []
        layer.close()

    asyncio.run(scenario())


def test_held_range_evicted_from_l1_is_written_back(tmp_path: Path) -> None:
    async def scenario() -> None:
        layer = _layer(tmp_path, 1)
        released: list[tuple[int, int]] = []
        layer.set_held_release_listener(lambda *key: released.append(key))
        await layer.store_bytes_grouped(b"a" * PAGE, [_span(0, hold=True)])
        # A second store needs the only L1 page and evicts the held range.
        await layer.store_bytes_grouped(b"b" * PAGE, [_span(PAGE, hold=False)])
        await layer.drain()
        assert released == [(0, PAGE)]
        assert layer.stats.held_evict_persists == 1
        assert layer.stats.l2_writes == 2
        dst = bytearray(PAGE)
        assert await layer.load_bytes(dst, 0, PAGE) == PAGE
        assert bytes(dst) == b"a" * PAGE
        layer.close()

    asyncio.run(scenario())


def test_hold_resident_keeps_a_written_record_without_another_write(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        layer = _layer(tmp_path, 1)
        released: list[tuple[int, int]] = []
        layer.set_held_release_listener(lambda *key: released.append(key))
        await layer.store_bytes_grouped(b"a" * PAGE, [_span(0, hold=False)])
        await layer.drain()
        assert layer.hold_resident(0, PAGE)
        assert not layer.hold_resident(PAGE, PAGE)  # not resident
        # Evicting the clean held page reports it without writing it again.
        await layer.store_bytes_grouped(b"b" * PAGE, [_span(PAGE, hold=False)])
        await layer.drain()
        assert released == [(0, PAGE)]
        assert layer.stats.held_evict_persists == 0
        assert layer.stats.l2_writes == 2
        dst = bytearray(PAGE)
        assert await layer.load_bytes(dst, 0, PAGE) == PAGE
        assert bytes(dst) == b"a" * PAGE
        layer.close()

    asyncio.run(scenario())


def test_l2_timing_counts_read_and_write_jobs(tmp_path: Path) -> None:
    async def scenario() -> None:
        layer = _layer(tmp_path, 1)
        await layer.store_bytes_grouped(b"a" * PAGE, [_span(0, hold=False)])
        await layer.store_bytes_grouped(b"b" * PAGE, [_span(PAGE, hold=False)])
        await layer.drain()
        dst = bytearray(PAGE)
        # The first page left the one-page L1, so this load reads L2.
        assert await layer.load_bytes(dst, 0, PAGE) == PAGE
        timing = layer.l2_timing_us
        assert set(timing) == {
            "read_queue",
            "read_io",
            "write_queue",
            "write_io",
            "write_gate",
        }
        assert timing["write_io"] > 0
        assert timing["read_io"] > 0
        layer.close()

    asyncio.run(scenario())
