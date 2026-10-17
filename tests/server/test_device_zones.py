# SPDX-License-Identifier: Apache-2.0

# Standard
import os
from pathlib import Path

# Third Party
import pytest

# First Party
from daser.server.device_zones import (
    MIN_ZONE_RUN_BYTES,
    build_zones,
    leaf_block_devices,
    probe_device_zones,
)
from daser.server.store_file import ensure_store_file

M = MIN_ZONE_RUN_BYTES
G = 64 << 10


def _fake_device(root: Path, name: str, size: int = 100, slaves: tuple = ()) -> None:
    base = root / name
    (base / "slaves").mkdir(parents=True)
    for slave in slaves:
        (base / "slaves" / slave).touch()
    (base / "size").write_text(f"{size}\n")
    (base / "stat").write_text(" ".join(["0"] * 17) + "\n")


def _tiles(zones: tuple, length: int) -> None:
    position = 0
    for start, end, _zone in zones:
        assert start == position and end > start
        assert start % G == 0 and end % G == 0 or end == length
        position = end
    assert position == length


def test_build_zones_merges_fills_and_numbers_devices() -> None:
    extents = [
        (0, 2 * M, "b"),
        (2 * M, M // 4, "b"),  # same device: merged
        (2 * M + M // 4, 3 * M, "a"),
        (6 * M, M, None),  # unknown: inherits "a"
        (8 * M, 2 * M, "b"),  # gap 7M..8M inherits "a"
    ]
    zones = build_zones(extents, 10 * M, G)
    assert zones == (
        (0, 2 * M + M // 4, 1),
        (2 * M + M // 4, 8 * M, 0),
        (8 * M, 10 * M, 1),
    )
    _tiles(zones, 10 * M)


def test_build_zones_absorbs_short_runs_and_aligns_bounds() -> None:
    extents = [
        (0, M // 2, "a"),  # short first run joins the next
        (M // 2, 2 * M + 4096, "b"),
        (2 * M + M // 2 + 4096, M // 4, "c"),  # short: joins "b"
        (2 * M + M // 2 + 4096 + M // 4, 3 * M, "a"),
    ]
    length = 6 * M
    zones = build_zones(extents, length, G)
    _tiles(zones, length)
    assert [zone for _s, _e, zone in zones] == [1, 0]
    assert zones[0][1] % G == 0


def test_build_zones_is_empty_for_one_device_or_none_known() -> None:
    assert build_zones([(0, 4 * M, "a"), (4 * M, 4 * M, "a")], 8 * M, G) == ()
    assert build_zones([(0, 4 * M, None)], 4 * M, G) == ()
    assert build_zones([], 4 * M, G) == ()


def test_leaf_block_devices_skips_virtual_stacked_and_empty(tmp_path: Path) -> None:
    _fake_device(tmp_path, "nvme0n1")
    _fake_device(tmp_path, "sda")
    _fake_device(tmp_path, "loop0")
    _fake_device(tmp_path, "dm-0", slaves=("sda",))
    _fake_device(tmp_path, "nvme9n1", size=0)
    assert leaf_block_devices(str(tmp_path)) == ["nvme0n1", "sda"]
    assert leaf_block_devices(str(tmp_path / "missing")) == []


def test_probe_without_moving_counters_is_one_zone(tmp_path: Path) -> None:
    sys_block = tmp_path / "sys"
    _fake_device(sys_block, "nvme0n1")
    _fake_device(sys_block, "nvme1n1")
    path = str(tmp_path / "store.bin")
    ensure_store_file(path, 8 * M)
    assert probe_device_zones(path, 8 * M, G, str(sys_block)) == ()
    # The probe never changes file contents.
    with open(path, "rb") as handle:
        assert handle.read(1 << 20) == bytes(1 << 20)


def test_probe_on_the_real_store_file_tiles_the_lane(tmp_path: Path) -> None:
    path = str(tmp_path / "store.bin")
    length = 4 << 30
    try:
        ensure_store_file(path, length)
        os.close(os.open(path, os.O_RDONLY | os.O_DIRECT))
    except OSError as exc:
        pytest.skip(f"O_DIRECT store file unsupported here: {exc}")
    first = probe_device_zones(path, length, G)
    if first:
        _tiles(first, length)
        assert probe_device_zones(path, length, G) == first
    with open(path, "rb") as handle:
        assert handle.read(1 << 20) == bytes(1 << 20)
    os.remove(path)
