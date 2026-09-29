# SPDX-License-Identifier: Apache-2.0

"""A vLLM KV connector that never loads or saves anything.

Used as a benchmark control arm: it makes vLLM run its connector code paths
(metadata build, worker hooks, ``get_finished`` polling) without any external
lookup, transfer, or GPU copy, so the per-step cost of merely having a KV
connector can be separated from the cost of DaseR's work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)

if TYPE_CHECKING:
    import torch
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


class NoopConnector(KVConnectorBase_V1):
    """KV connector whose every hook is a no-op and reports zero hits.

    Async/thread-safety:
        Stateless; safe for vLLM's scheduler and worker roles.
    """

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Do nothing: there is never anything to load.

        Args:
            forward_context: vLLM forward context for this step.
            **kwargs: Unused hook arguments.
        """

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Return immediately.

        Args:
            layer_name: Layer reported by vLLM.
        """

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Do nothing: KV is never saved.

        Args:
            layer_name: Layer reported by vLLM.
            kv_layer: Layer KV tensor.
            attn_metadata: Attention metadata.
            **kwargs: Unused hook arguments.
        """

    def wait_for_save(self) -> None:
        """Return immediately."""

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Report no external hit.

        Args:
            request: Request being scheduled.
            num_computed_tokens: Tokens already computed locally.

        Returns:
            ``(0, False)``: no external tokens, no async load.
        """
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        """Do nothing after allocation.

        Args:
            request: Request that received blocks.
            blocks: Allocated blocks.
            num_external_tokens: Always 0 for this connector.
        """

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        """Return empty metadata.

        Args:
            scheduler_output: Scheduler output for this step.

        Returns:
            An empty ``KVConnectorMetadata``.
        """
        return KVConnectorMetadata()
