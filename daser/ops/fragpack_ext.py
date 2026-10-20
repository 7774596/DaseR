# SPDX-License-Identifier: Apache-2.0
"""JIT build of the FragPack CUDA extension (decode attention + HBM KV store)."""

from __future__ import annotations

import functools
from pathlib import Path
from types import ModuleType

_DIR = Path(__file__).parent
_SOURCES = ("fragpack_ext.cpp", "fused_attn_decode.cu", "hbm_kv.cu")


@functools.cache
def extension() -> ModuleType:
    """Build (once per process) and return the FragPack extension module.

    Returns:
        The compiled module exposing ``fused_decode`` and the ``hbm_*`` ops.

    Async/thread-safety:
        Cached; the first call compiles (minutes) under torch's file lock.
    """
    from torch.utils.cpp_extension import load

    return load(
        name="daser_fragpack",
        sources=[str(_DIR / s) for s in _SOURCES],
        extra_include_paths=[str(_DIR)],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )


__all__ = ["extension"]
