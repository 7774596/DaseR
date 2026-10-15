# SPDX-License-Identifier: Apache-2.0
"""Held records stay in L1 until discarded or evicted."""

import asyncio
from pathlib import Path

from daser.transfer.iouring import TieredIOUringTransferLayer

PAGE = 4096


def _span(file_offset: int) -> dict[str, object]:
    return {
        "source_offset": 0,
        "file_offset": file_offset,
        "nbytes": PAGE,
        "packed": True,
    }


def _layer(tmp_path: Path, l1_pages: int) -> TieredIOUringTransferLayer:
    return TieredIOUringTransferLayer(
        path=str(tmp_path / "held.store"),
        l1_bytes=l1_pages * PAGE,
        l2_bytes=8 * PAGE,
    )


def test_discard_held_drops_l1_without_release(tmp_path: Path) -> None:
    async def scenario() -> None:
        layer = _layer(tmp_path, 4)
        released: list[tuple[int, int]] = []
        layer.set_held_release_listener(lambda *key: released.append(key))
        await layer.store_bytes_grouped(b"a" * PAGE, [_span(0)])
        await layer.drain()
        assert layer.hold_resident(0, PAGE)
        layer.discard_held(0, PAGE)
        assert layer.l1_bytes_used == 0
        assert released == []
        layer.close()

    asyncio.run(scenario())


def test_hold_resident_keeps_a_written_record_without_another_write(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        layer = _layer(tmp_path, 1)
        released: list[tuple[int, int]] = []
        layer.set_held_release_listener(lambda *key: released.append(key))
        await layer.store_bytes_grouped(b"a" * PAGE, [_span(0)])
        await layer.drain()
        assert layer.hold_resident(0, PAGE)
        assert not layer.hold_resident(PAGE, PAGE)  # not resident
        # Evicting the clean held page reports it without writing it again.
        await layer.store_bytes_grouped(b"b" * PAGE, [_span(PAGE)])
        await layer.drain()
        assert released == [(0, PAGE)]
        assert layer.stats.l2_writes == 2
        dst = bytearray(PAGE)
        assert await layer.load_bytes(dst, 0, PAGE) == PAGE
        assert bytes(dst) == b"a" * PAGE
        layer.close()

    asyncio.run(scenario())


def test_l2_timing_counts_read_and_write_jobs(tmp_path: Path) -> None:
    async def scenario() -> None:
        layer = _layer(tmp_path, 1)
        await layer.store_bytes_grouped(b"a" * PAGE, [_span(0)])
        await layer.store_bytes_grouped(b"b" * PAGE, [_span(PAGE)])
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
