# SPDX-License-Identifier: Apache-2.0

"""Skip helper for transfer tests that need a CUDA device."""

# Standard
from typing import Any

# Third Party
import pytest


def require_cuda() -> Any:
    """Return the CuPy module, or skip the test without a CUDA device.

    Returns:
        The imported ``cupy`` module with at least one visible device.

    Async/thread-safety:
        Synchronous test helper; call at test start.
    """
    cupy = pytest.importorskip("cupy")
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except Exception:  # cupy.cuda.runtime.CUDARuntimeError: no device/driver
        count = 0
    if count == 0:
        pytest.skip("CUDA device required")
    return cupy
