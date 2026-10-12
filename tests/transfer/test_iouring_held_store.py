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


def _preclean_layer(
    tmp_path: Path, l1_pages: int, fraction: float
) -> TieredIOUringTransferLayer:
    return TieredIOUringTransferLayer(
        path=str(tmp_path / "held.store"),
        l1_bytes=l1_pages * PAGE,
        l2_bytes=8 * PAGE,
        held_preclean_fraction=fraction,
    )


def test_preclean_writes_oldest_held_over_limit_and_keeps_it_resident(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        layer = _preclean_layer(tmp_path, 4, 0.5)
        for page, fill in enumerate((b"a", b"b")):
            await layer.store_bytes_grouped(
                fill * PAGE, [_span(page * PAGE, hold=True)]
            )
        await layer.drain()
        # At the limit (2 of 4 pages held) nothing is written yet.
        assert layer.stats.l2_writes == 0
        await layer.store_bytes_grouped(b"c" * PAGE, [_span(2 * PAGE, hold=True)])
        await layer.drain()
        assert layer.stats.held_preclean_bytes == PAGE
        assert layer.stats.l2_writes == 1
        assert layer.held_bytes == 2 * PAGE
        assert layer.l1_bytes_used == 3 * PAGE
        # The precleaned range is no longer held: discarding it is a no-op.
        layer.discard_held([(0, PAGE)])
        assert layer.l1_bytes_used == 3 * PAGE
        dst = bytearray(PAGE)
        assert await layer.load_bytes(dst, 0, PAGE) == PAGE
        assert bytes(dst) == b"a" * PAGE
        layer.close()

    asyncio.run(scenario())


def test_precleaned_range_leaves_l1_without_another_write(tmp_path: Path) -> None:
    async def scenario() -> None:
        layer = _preclean_layer(tmp_path, 2, 0.5)
        await layer.store_bytes_grouped(b"a" * PAGE, [_span(0, hold=True)])
        await layer.store_bytes_grouped(b"b" * PAGE, [_span(PAGE, hold=True)])
        await layer.drain()
        assert layer.stats.held_preclean_bytes == PAGE
        # A third store evicts the clean oldest page: only its own write.
        await layer.store_bytes_grouped(b"c" * PAGE, [_span(2 * PAGE, hold=False)])
        await layer.drain()
        assert layer.stats.held_evict_persists == 0
        assert layer.stats.l2_writes == 2
        dst = bytearray(PAGE)
        assert await layer.load_bytes(dst, 0, PAGE) == PAGE
        assert bytes(dst) == b"a" * PAGE
        layer.close()

    asyncio.run(scenario())


def test_preclean_fraction_must_be_below_one(tmp_path: Path) -> None:
    try:
        _preclean_layer(tmp_path, 2, 1.0)
    except ValueError:
        return
    raise AssertionError("held_preclean_fraction 1.0 was accepted")


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
