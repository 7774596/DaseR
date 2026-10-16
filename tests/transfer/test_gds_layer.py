# SPDX-License-Identifier: Apache-2.0

"""Public contract of the GDS (kvikio) L2-only transfer backend.

kvikio falls back to its POSIX compat mode without a GPU, so the contract is
exercised with host buffers.
"""

# Standard
import asyncio
from pathlib import Path

# Third Party
import numpy
import pytest

# First Party
from daser.transfer import Move
from tests.transfer.store_files import create_store_file

gds = pytest.importorskip("daser.transfer.gds")

PAGE = 4096


def _layer(tmp_path: Path, nbytes: int = 16 * PAGE) -> "gds.GDSTransferLayer":
    """Build a GDS layer over a fresh store file."""
    return gds.GDSTransferLayer(create_store_file(str(tmp_path / "l2.store"), nbytes))


def test_store_and_load_roundtrip(tmp_path: Path) -> None:
    """``staged`` and ``persisted`` complete together; loads read L2 bytes."""

    async def scenario() -> None:
        layer = _layer(tmp_path)
        assert layer.backend in (gds.TransferBackend.GDS, gds.TransferBackend.COMPAT)
        src = (numpy.arange(3 * PAGE) % 251).astype(numpy.uint8)
        moves = [
            Move(0, PAGE, l2_offset=8 * PAGE),
            Move(PAGE, 2 * PAGE, l2_offset=PAGE),
        ]
        try:
            handle = await layer.store(src, moves)
            assert handle.staged is handle.persisted
            await asyncio.wait_for(handle.persisted, 10)
            await layer.drain()
            dst = numpy.zeros(3 * PAGE, dtype=numpy.uint8)
            await layer.load(dst, moves, "l2")
            assert (dst == src).all()
            assert layer.stats.l2_write_bytes == layer.stats.l2_read_bytes == 3 * PAGE
        finally:
            await layer.close()
            await layer.close()

    asyncio.run(scenario())


def test_rejects_l1_and_unsupported_operations(tmp_path: Path) -> None:
    """GDS has no arena: L1 offsets, L1 loads, promote and runs are refused."""

    async def scenario() -> None:
        layer = _layer(tmp_path)
        buffer = numpy.zeros(PAGE, dtype=numpy.uint8)
        try:
            with pytest.raises(ValueError, match="no L1 arena"):
                await layer.store(buffer, [Move(0, PAGE, l1_offset=0, l2_offset=0)])
            with pytest.raises(ValueError, match="'l2'"):
                await layer.load(buffer, [Move(0, PAGE, l1_offset=0)], "l1")
            with pytest.raises(ValueError, match="aligned"):
                await layer.load(buffer, [Move(0, PAGE, l2_offset=100)], "l2")
            with pytest.raises(NotImplementedError):
                await layer.promote([Move(0, PAGE, l1_offset=0, l2_offset=0)])
            with pytest.raises(NotImplementedError):
                await layer.load_runs(buffer, [0], [0], [PAGE])
            assert layer.stats.l2_reads == layer.stats.l2_writes == 0
        finally:
            await layer.close()
        with pytest.raises(RuntimeError, match="closing"):
            await layer.store(buffer, [Move(0, PAGE, l2_offset=0)])

    asyncio.run(scenario())


def test_constructor_checks_file_and_size(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        gds.GDSTransferLayer(str(tmp_path / "missing.store"))
    path = create_store_file(str(tmp_path / "l2.store"), PAGE)
    with pytest.raises(ValueError, match="l2_bytes"):
        gds.GDSTransferLayer(path, l2_bytes=2 * PAGE)
