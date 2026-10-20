# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin: register the FragPack backend as the CUSTOM backend.

Expose it through a ``vllm.general_plugins`` entry point
(``daser_fragpack = daser.attention.plugin:register``).
"""

from __future__ import annotations


def register() -> None:
    """Register ``fragpack_backend.FragPackAttentionBackend`` as CUSTOM.

    Returns:
        None.

    Async/thread-safety:
        Called by vLLM in every process at startup; only records a class path,
        so it imports nothing heavy.
    """
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "daser.attention.fragpack_backend.FragPackAttentionBackend",
    )
