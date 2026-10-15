# SPDX-License-Identifier: Apache-2.0
"""Scheduler hand-off of synchronous layer-wise compressed loads."""

from types import SimpleNamespace
from typing import Any

import pytest

from daser.connector.ipc_client import IPCClientAsync, PrefetchLookupResult
from daser.connector.scheduler.lifecycle import RequestLifecycle

BLOCK_TOKENS = 4
SLOT_SIZE = 4096


class _LookupIPC:
    """Synchronous IPC stub returning one leased L1 hit."""

    def __init__(
        self, modes: list[str], raw: bool = False, pos_offset: int = 0
    ) -> None:
        self.modes = modes
        self.raw = raw
        self.pos_offset = pos_offset
        self.released: list[str] = []

    def lookup_with_prefetch(
        self, lease_id: str, tokens: list[int], model_id: str, **kwargs: Any
    ) -> PrefetchLookupResult:
        """Return a two-slot chunk whose records use ``self.modes``.

        With ``raw`` the chunk is a raw slot-major chunk without packed
        record metadata.
        """
        del lease_id, tokens, model_id, kwargs
        slots = [
            {
                "slot_id": 10 + index,
                "mode": mode,
                "file_offset": (10 + index) * SLOT_SIZE,
                "stored_length": SLOT_SIZE,
            }
            for index, mode in enumerate(self.modes)
        ]
        return PrefetchLookupResult(
            chunks=[
                {
                    "chunk_key": "cached",
                    "start_slot": 10,
                    "num_slots": len(slots),
                    "file_offset": 10 * SLOT_SIZE,
                    "token_count": len(slots) * BLOCK_TOKENS,
                    "target_token_start": 0,
                    "pos_offset": self.pos_offset,
                    "compressed_slots": [] if self.raw else slots,
                }
            ],
            spans=[],
            tier="l1",
        )

    def release_transfer_lease(self, lease_id: str) -> None:
        """Record a scheduler-side lease release."""
        self.released.append(lease_id)


def _lifecycle(ipc: _LookupIPC, layerwise: bool) -> RequestLifecycle:
    return RequestLifecycle(
        ipc_client=ipc,
        socket_path="/unused/daser.sock",
        block_tokens=BLOCK_TOKENS,
        slot_size=SLOT_SIZE,
        model_id="model",
        cache_reuse_mode="chunk",
        runtime_config_ready=True,
        prefetch_max_requests=1,
        layerwise_load=layerwise,
    )


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req",
        prompt_token_ids=list(range(3 * BLOCK_TOKENS)),
        kv_transfer_params={"daser_skip_save": True},
    )


def _schedule(lifecycle: RequestLifecycle) -> tuple[Any, Any]:
    request = _request()
    matched = lifecycle.get_num_new_matched_tokens(request, 0)
    blocks = SimpleNamespace(
        blocks=[[SimpleNamespace(block_id=index) for index in (5, 6, 7)]]
    )
    lifecycle.update_state_after_alloc(request, blocks, 2 * BLOCK_TOKENS)
    meta = lifecycle.build_connector_meta(
        SimpleNamespace(num_scheduled_tokens={"req": BLOCK_TOKENS})
    )
    return matched, meta


def test_leased_packed_hit_loads_synchronously_and_hands_off_lease() -> None:
    """A fully packed L1 hit is a sync load whose lease moves to the worker."""
    ipc = _LookupIPC(["compressed", "compressed"])
    lifecycle = _lifecycle(ipc, layerwise=True)

    matched, meta = _schedule(lifecycle)

    assert matched == (2 * BLOCK_TOKENS, False)
    assert meta.layerwise_req_ids == {"req"}
    spec = meta.reqs_to_load["req"]
    assert spec.block_ids == [5, 6]
    assert spec.lease_id == "req"
    assert [slot.mode for slot in spec.compressed_slots] == ["compressed"] * 2
    # The scheduler no longer owns the lease, so finishing or shutting down
    # must not release it underneath the in-flight forward pass.
    lifecycle.shutdown()
    assert ipc.released == []
    assert not lifecycle.build_connector_meta(
        SimpleNamespace(num_scheduled_tokens={})
    ).layerwise_req_ids


def test_leased_raw_hit_loads_synchronously() -> None:
    """A leased raw slot-major L1 hit also takes the layer-wise path."""
    ipc = _LookupIPC(["raw", "raw"], raw=True)
    lifecycle = _lifecycle(ipc, layerwise=True)

    matched, meta = _schedule(lifecycle)

    assert matched == (2 * BLOCK_TOKENS, False)
    assert meta.layerwise_req_ids == {"req"}
    spec = meta.reqs_to_load["req"]
    assert spec.block_ids == [5, 6]
    assert spec.lease_id == "req"
    assert spec.compressed_slots == []


def test_raw_hit_with_position_shift_keeps_the_async_load_path() -> None:
    """Raw layer-wise copies cannot apply RoPE shifts, so shifted hits wait."""
    ipc = _LookupIPC(["raw", "raw"], raw=True, pos_offset=3)
    lifecycle = _lifecycle(ipc, layerwise=True)

    matched, meta = _schedule(lifecycle)

    assert matched == (2 * BLOCK_TOKENS, True)
    assert not meta.layerwise_req_ids
    lifecycle.shutdown()


@pytest.mark.parametrize(
    ("layerwise", "modes"),
    [(False, ["compressed", "compressed"]), (True, ["compressed", "raw"])],
)
def test_other_hits_keep_the_async_load_path(layerwise: bool, modes: list[str]) -> None:
    """Switch off or any raw record keeps the asynchronous record-major load."""
    ipc = _LookupIPC(modes)
    lifecycle = _lifecycle(ipc, layerwise=layerwise)

    matched, meta = _schedule(lifecycle)

    assert matched == (2 * BLOCK_TOKENS, True)
    assert not meta.layerwise_req_ids
    assert meta.reqs_to_load["req"].lease_id == "req"
    lifecycle.shutdown()
    assert ipc.released == ["req"]


def test_layerwise_load_keeps_vllm_cudagraph_mode() -> None:
    """Layer-wise loads no longer force PIECEWISE graphs on every step."""
    connector = pytest.importorskip("daser.connector.daser_connector")
    cls = connector.DaserConnector
    assert not cls.requires_piecewise_for_cudagraph({"layerwise_load": True})


def _forward_context(mode: Any, num_tokens: int, uniform: bool) -> Any:
    from vllm.forward_context import BatchDescriptor, ForwardContext

    return ForwardContext(
        no_compile_layers={},
        attn_metadata={"layer.0": object()},
        slot_mapping={},
        cudagraph_runtime_mode=mode,
        batch_descriptor=BatchDescriptor(num_tokens, 1, uniform),
    )


@pytest.mark.parametrize("num_speculative_tokens", [0, 2])
def test_full_graph_steps_need_the_whole_restore_up_front(
    num_speculative_tokens: int,
) -> None:
    """A uniform-decode FULL replay skips the layer hooks, also with spec decode.

    vLLM replays a full graph when every request schedules exactly
    ``1 + num_speculative_tokens`` tokens; a layer-wise hit leaving that many
    tokens to compute lands there.
    """
    runtime = pytest.importorskip("daser.connector.worker.runtime")
    from vllm.config import CUDAGraphMode

    decode_len = 1 + num_speculative_tokens
    full = _forward_context(CUDAGraphMode.FULL, decode_len, uniform=True)
    assert runtime.forward_skips_layer_waits(full)


@pytest.mark.parametrize("mode", ["PIECEWISE", "NONE"])
def test_prefill_steps_wait_per_layer(mode: str) -> None:
    """PIECEWISE and eager forwards run ``wait_for_layer_load`` per layer."""
    runtime = pytest.importorskip("daser.connector.worker.runtime")
    from vllm.config import CUDAGraphMode

    context = _forward_context(CUDAGraphMode[mode], 512, uniform=False)
    assert not runtime.forward_skips_layer_waits(context)


def test_steps_without_attention_need_the_whole_restore_up_front() -> None:
    """No-forward steps (and model runner V2 full replays) run no attention."""
    runtime = pytest.importorskip("daser.connector.worker.runtime")
    from vllm.forward_context import ForwardContext

    empty = ForwardContext(no_compile_layers={}, attn_metadata=None, slot_mapping={})
    assert runtime.forward_skips_layer_waits(empty)
    assert runtime.forward_skips_layer_waits(None)


async def test_async_client_releases_transfer_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker-side client sends the release op for its lease."""
    sent: list[dict[str, Any]] = []

    async def fake_call(self: IPCClientAsync, payload: dict[str, Any]) -> dict:
        del self
        sent.append(payload)
        return {"status": "ok"}

    monkeypatch.setattr(IPCClientAsync, "call", fake_call)
    client = IPCClientAsync("/unused/daser.sock")

    assert await client.release_transfer_lease("req") == {"status": "ok"}
    assert sent == [{"op": "release_transfer_lease", "lease_id": "req"}]
