# SPDX-License-Identifier: Apache-2.0

"""Test helper that provides the L2 store file the server creates at startup.

Transfer code must not import ``daser.server``, so transfer tests create the
file with this helper instead of ``daser.server.store_file``.
"""

# Standard
import os


def create_store_file(path: str, nbytes: int) -> str:
    """Create ``path`` as a sparse file of at least ``nbytes`` bytes.

    Args:
        path: store file path; missing parent directories are created.
        nbytes: minimum file size in bytes.

    Returns:
        ``path``, for inline use in constructor calls.

    Async/thread-safety:
        Blocking filesystem helper for test setup.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "ab") as f:
        if f.tell() < nbytes:
            f.truncate(nbytes)
    return path
