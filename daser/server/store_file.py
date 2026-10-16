# SPDX-License-Identifier: Apache-2.0

"""Creation and validation of the L2 KV store file.

A new store file is created empty, marked NOCOW (``chattr +C``) while it is
still empty, and preallocated to its full size. On copy-on-write filesystems
such as btrfs this keeps the file in few extents under overwrite churn
instead of fragmenting it, which roughly doubles sustained O_DIRECT write
bandwidth. NOCOW is a best-effort hint: filesystems without the attribute
only log a warning.
"""

# Standard
import array
import errno
import fcntl
import os

# First Party
from daser.logging import init_logger

logger = init_logger(__name__)

#: ``FS_IOC_GETFLAGS`` / ``FS_IOC_SETFLAGS`` (``_IOR``/``_IOW('f', 1|2, long)``
#: with the generic Linux ioctl encoding used by x86_64 and aarch64).
FS_IOC_GETFLAGS = 0x80086601
FS_IOC_SETFLAGS = 0x40086602
#: Inode flag that disables copy-on-write for file data (``chattr +C``).
FS_NOCOW_FL = 0x00800000


def _set_nocow(fd: int, path: str) -> bool:
    """Try to mark the open, still empty file ``fd`` NOCOW.

    Args:
        fd: open descriptor of an empty regular file.
        path: file path, for log messages.

    Returns:
        True when the flag reads back as set, False otherwise. Failures are
        logged and never raised.
    """
    flags = array.array("i", [0])
    try:
        fcntl.ioctl(fd, FS_IOC_GETFLAGS, flags, True)
        flags[0] |= FS_NOCOW_FL
        fcntl.ioctl(fd, FS_IOC_SETFLAGS, flags, False)
        fcntl.ioctl(fd, FS_IOC_GETFLAGS, flags, True)
    except OSError as exc:
        logger.warning("[SERVER] cannot set NOCOW on store file %s: %s", path, exc)
        return False
    if not flags[0] & FS_NOCOW_FL:
        logger.warning(
            "[SERVER] filesystem ignored NOCOW on store file %s; "
            "overwrites may fragment it",
            path,
        )
        return False
    return True


def _create_store_file(path: str, size: int) -> None:
    """Create ``path`` exclusively, mark it NOCOW and preallocate ``size``.

    Args:
        path: store file path that must not exist.
        size: file size in bytes.

    Raises:
        OSError: if the file cannot be created or preallocated; the partial
            file is removed.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        nocow = _set_nocow(fd, path)
        try:
            os.posix_fallocate(fd, 0, size)
            preallocated = True
        except OSError as exc:
            if exc.errno not in (errno.EOPNOTSUPP, errno.EINVAL):
                raise
            logger.warning(
                "[SERVER] fallocate unsupported for store file %s (%s); "
                "creating a sparse file",
                path,
                exc,
            )
            os.ftruncate(fd, size)
            preallocated = False
    except BaseException:
        os.close(fd)
        os.unlink(path)
        raise
    os.close(fd)
    logger.info(
        "[SERVER] created store file %s: %d bytes, nocow=%s, preallocated=%s",
        path,
        size,
        nocow,
        preallocated,
    )


def ensure_store_file(path: str, size: int) -> None:
    """Make ``path`` an L2 store file of exactly ``size`` bytes.

    A missing file is created NOCOW and preallocated. An existing larger file
    is truncated down to ``size``; an existing smaller file is rejected.
    Existing files keep their attributes and contents.

    Args:
        path: store file path; missing parent directories are created.
        size: required file size in bytes; positive.

    Raises:
        ValueError: if ``size`` is not positive or an existing file is
            smaller than ``size``.
        OSError: if creation or preallocation fails.

    Async/thread-safety:
        Blocking filesystem calls; run at startup, not on an event loop that
        serves requests. Not safe against concurrent callers on one path.
    """
    if size <= 0:
        raise ValueError(f"store file size must be positive, got {size}")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(path):
        _create_store_file(path, size)
        return
    existing = os.path.getsize(path)
    if existing < size:
        raise ValueError(
            f"existing store file {path} has size {existing}, expected {size}"
        )
    if existing > size:
        logger.warning(
            "[SERVER] truncating store file %s from %d to aligned size %d",
            path,
            existing,
            size,
        )
        os.truncate(path, size)
