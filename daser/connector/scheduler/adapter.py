# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

from daser.connector.metadata import DaserConnectorMeta


class SchedulerConnectorMixin:
    """Adapt vLLM scheduler hooks to the request lifecycle interface."""

    def on_new_request(self, request: "Request") -> None:
        """Forward vLLM's waiting-queue arrival to the request lifecycle."""
        self._request_lifecycle.on_new_request(request)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Return DaseR cache credit for one request."""
        probe = getattr(self, "_scheduler_probe", None)
        started = time.perf_counter()
        result = self._request_lifecycle.get_num_new_matched_tokens(
            request,
            num_computed_tokens,
        )
        if probe is not None:
            probe.on_lookup(request.request_id, time.perf_counter() - started, result)
        return result

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Bind allocated vLLM blocks to pending lifecycle work."""
        probe = getattr(self, "_scheduler_probe", None)
        if probe is not None:
            probe.on_alloc(request.request_id, len(blocks.blocks[0]))
        self._request_lifecycle.update_state_after_alloc(
            request,
            blocks,
            num_external_tokens,
        )

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> DaserConnectorMeta:
        """Build worker metadata from request lifecycle state."""
        return self._request_lifecycle.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Return whether worker save completion still holds request blocks."""
        result = self._request_lifecycle.request_finished(request, block_ids)
        probe = getattr(self, "_scheduler_probe", None)
        if probe is not None:
            probe.on_request_finished(request.request_id, len(block_ids), result[0])
        return result

    def update_connector_output(self, connector_output: Any) -> None:
        """Apply worker transfer completions to request lifecycle state."""
        self._request_lifecycle.update_connector_output(connector_output)
        probe = getattr(self, "_scheduler_probe", None)
        if probe is not None:
            probe.on_connector_output(
                getattr(connector_output, "finished_sending", None) or (),
                getattr(connector_output, "finished_recving", None) or (),
            )
