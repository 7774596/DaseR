# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin: registers the AF1 backend as ``--attention-backend CUSTOM``.

Expose it through a ``vllm.general_plugins`` entry point
(``daser_af1 = daser.attention.plugin:register``).
"""

from __future__ import annotations


def register() -> None:
    """Register :class:`~daser.attention.af1_backend.Af1AttentionBackend` as CUSTOM.

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
        AttentionBackendEnum.CUSTOM, "daser.attention.af1_backend.Af1AttentionBackend"
    )
