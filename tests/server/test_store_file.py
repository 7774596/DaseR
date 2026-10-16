# SPDX-License-Identifier: Apache-2.0

# Standard
import array
import errno
import fcntl
import os
from pathlib import Path

# Third Party
import pytest

# First Party
from daser.server import store_file
from daser.server.store_file import ensure_store_file

SIZE = 64 << 10


def _flags(path: Path) -> int:
    flags = array.array("i", [0])
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.ioctl(fd, store_file.FS_IOC_GETFLAGS, flags, True)
    finally:
        os.close(fd)
    return flags[0]


def _filesystem_supports_nocow(directory: Path) -> bool:
    """Probe with ``chattr +C`` semantics on a fresh empty file."""
    probe = directory / "nocow.probe"
    fd = os.open(probe, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        flags = array.array("i", [0])
        fcntl.ioctl(fd, store_file.FS_IOC_GETFLAGS, flags, True)
        flags[0] |= store_file.FS_NOCOW_FL
        fcntl.ioctl(fd, store_file.FS_IOC_SETFLAGS, flags, False)
        fcntl.ioctl(fd, store_file.FS_IOC_GETFLAGS, flags, True)
        return bool(flags[0] & store_file.FS_NOCOW_FL)
    except OSError:
        return False
    finally:
        os.close(fd)
        probe.unlink()


def test_new_file_is_created_at_size_and_preallocated(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "daser.store"
    ensure_store_file(str(path), SIZE)
    stat = path.stat()
    assert stat.st_size == SIZE
    assert stat.st_blocks * 512 >= SIZE


def test_new_file_is_nocow_when_the_filesystem_supports_it(tmp_path: Path) -> None:
    if not _filesystem_supports_nocow(tmp_path):
        pytest.skip("scratch filesystem does not support NOCOW")
    path = tmp_path / "daser.store"
    ensure_store_file(str(path), SIZE)
    assert _flags(path) & store_file.FS_NOCOW_FL


def test_nocow_failure_is_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_ioctl(*_args: object) -> None:
        raise OSError(errno.ENOTTY, "inappropriate ioctl")

    monkeypatch.setattr(store_file.fcntl, "ioctl", failing_ioctl)
    path = tmp_path / "daser.store"
    ensure_store_file(str(path), SIZE)
    assert path.stat().st_size == SIZE


def test_unsupported_fallocate_falls_back_to_a_sparse_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unsupported(*_args: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "not supported")

    monkeypatch.setattr(store_file.os, "posix_fallocate", unsupported)
    path = tmp_path / "daser.store"
    ensure_store_file(str(path), SIZE)
    assert path.stat().st_size == SIZE


def test_failed_preallocation_removes_the_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_space(*_args: object) -> None:
        raise OSError(errno.ENOSPC, "no space left")

    monkeypatch.setattr(store_file.os, "posix_fallocate", no_space)
    path = tmp_path / "daser.store"
    with pytest.raises(OSError, match="no space"):
        ensure_store_file(str(path), SIZE)
    assert not path.exists()


def test_existing_file_of_the_right_size_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "daser.store"
    path.write_bytes(b"k" * SIZE)
    ensure_store_file(str(path), SIZE)
    assert path.read_bytes() == b"k" * SIZE


def test_existing_larger_file_is_truncated(tmp_path: Path) -> None:
    path = tmp_path / "daser.store"
    path.write_bytes(b"k" * (2 * SIZE))
    ensure_store_file(str(path), SIZE)
    assert path.read_bytes() == b"k" * SIZE


def test_existing_smaller_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "daser.store"
    path.write_bytes(b"k" * (SIZE - 1))
    with pytest.raises(ValueError, match="has size"):
        ensure_store_file(str(path), SIZE)
    assert path.stat().st_size == SIZE - 1


@pytest.mark.parametrize("size", [0, -SIZE])
def test_non_positive_size_is_rejected(tmp_path: Path, size: int) -> None:
    path = tmp_path / "daser.store"
    with pytest.raises(ValueError, match="positive"):
        ensure_store_file(str(path), size)
    assert not path.exists()
