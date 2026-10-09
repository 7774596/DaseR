# SPDX-License-Identifier: Apache-2.0
"""Decode save: finished requests also store the full blocks they generated."""

from types import SimpleNamespace
from typing import Any

from daser.config import CACHE_REUSE_PREFIX
from daser.connector.helpers import rolling_prefix_keys
from daser.connector.scheduler.lifecycle import RequestLifecycle

BLOCK_TOKENS = 4
PROMPT = list(range(100, 110))  # two full blocks and a two-token tail
OUTPUT = list(range(200, 207))  # 17 tokens in all, 16 computed at finish


class _FakeIPC:
    """Synchronous IPC stub recording store allocations."""

    def __init__(self) -> None:
        self.allocs: list[str] = []
        self.live = True

    def get_runtime_config(self) -> dict[str, Any]:
        """Return a minimal prefix-mode server runtime config."""
        return {
            "slot_size": 32,
            "block_tokens": BLOCK_TOKENS,
            "model_id": "m",
            "cache_reuse_mode": CACHE_REUSE_PREFIX,
        }

    def alloc_chunk(self, chunk_key: str, token_count: int, model_id: str) -> dict:
        """Record and return server-style allocation metadata."""
        del token_count, model_id
        self.allocs.append(chunk_key)
        return {"chunk_key": chunk_key, "start_slot": 0, "file_offset": 0}

    def alloc_chunks(self, chunks: list[dict], model_id: str) -> list[dict]:
        """Allocate each chunk through ``alloc_chunk``."""
        return [
            self.alloc_chunk(str(chunk["chunk_key"]), 0, model_id) for chunk in chunks
        ]

    def live_allocations(self, chunks: list[dict[str, Any]]) -> set[str]:
        """Report every allocation live unless ``live`` is cleared."""
        return {str(chunk["chunk_key"]) for chunk in chunks} if self.live else set()


def _output(scheduled: dict[str, int] | None = None) -> SimpleNamespace:
    return SimpleNamespace(num_scheduled_tokens=scheduled or {})


def _admitted(
    decode_save: bool,
) -> tuple[RequestLifecycle, _FakeIPC, SimpleNamespace]:
    ipc = _FakeIPC()
    lifecycle = RequestLifecycle(
        ipc_client=ipc,
        block_tokens=BLOCK_TOKENS,
        slot_size=32,
        model_id="m",
        cache_reuse_mode=CACHE_REUSE_PREFIX,
        runtime_config_ready=False,
        decode_save=decode_save,
    )
    lifecycle.refresh_runtime_config()
    request = SimpleNamespace(
        request_id="req",
        prompt_token_ids=list(PROMPT),
        output_token_ids=list(OUTPUT),
        kv_transfer_params={"daser_skip_load": True},
        num_output_tokens=len(OUTPUT),
        num_computed_tokens=len(PROMPT) + len(OUTPUT) - 1,
    )
    assert lifecycle.get_num_new_matched_tokens(request, 0) == (0, False)
    blocks = SimpleNamespace(
        blocks=[[SimpleNamespace(block_id=idx) for idx in range(3)]]
    )
    lifecycle.update_state_after_alloc(request, blocks, 0)
    return lifecycle, ipc, request


def test_decode_save_off_stores_only_the_prompt() -> None:
    lifecycle, ipc, request = _admitted(decode_save=False)
    assert len(ipc.allocs) == 2
    meta = lifecycle.build_connector_meta(_output({"req": len(PROMPT)}))
    assert sorted(meta.reqs_to_store) == ["req:store:0", "req:store:1"]
    lifecycle.update_connector_output(
        SimpleNamespace(finished_sending={"req"}, finished_recving=None)
    )
    assert lifecycle.request_finished(request, [0, 1, 2, 3]) == (False, None)
    assert len(ipc.allocs) == 2


def test_decode_save_stores_generated_blocks_after_the_prompt_slots() -> None:
    lifecycle, ipc, request = _admitted(decode_save=True)
    lifecycle.build_connector_meta(_output({"req": len(PROMPT)}))
    lifecycle.update_connector_output(
        SimpleNamespace(finished_sending={"req"}, finished_recving=None)
    )

    assert lifecycle.request_finished(request, [0, 1, 2, 3]) == (True, None)
    meta = lifecycle.build_connector_meta(_output())
    assert meta.writeback_req_ids == {"req"}
    assert sorted(meta.reqs_to_store) == ["req:store:2", "req:store:3"]
    assert meta.reqs_to_store["req:store:2"].block_ids == [2]
    assert meta.reqs_to_store["req:store:3"].block_ids == [3]
    # The keys chain over prompt plus output, so a next turn that replays the
    # output after this prompt looks them up as its own prefix slots.
    keys = rolling_prefix_keys(PROMPT + OUTPUT, BLOCK_TOKENS)
    assert meta.reqs_to_store["req:store:2"].chunk_key == keys[2]
    assert meta.reqs_to_store["req:store:3"].chunk_key == keys[3]
    assert ipc.allocs[:2] == keys[:2]

    lifecycle.update_connector_output(
        SimpleNamespace(finished_sending={"req"}, finished_recving=None)
    )
    meta = lifecycle.build_connector_meta(_output())
    assert meta.writeback_req_ids == set()
    assert meta.reqs_to_store == {}


def test_decode_save_skips_when_nothing_past_the_prompt_is_full() -> None:
    lifecycle, ipc, request = _admitted(decode_save=True)
    lifecycle.build_connector_meta(_output({"req": len(PROMPT)}))
    lifecycle.update_connector_output(
        SimpleNamespace(finished_sending={"req"}, finished_recving=None)
    )
    request.output_token_ids = OUTPUT[:1]
    request.num_computed_tokens = len(PROMPT)
    assert lifecycle.request_finished(request, [0, 1, 2]) == (False, None)
    assert len(ipc.allocs) == 2


def test_dropped_decode_stores_leave_a_pending_prompt_save_to_report() -> None:
    lifecycle, ipc, request = _admitted(decode_save=True)
    # The prompt stores are published but the worker has not finished them.
    meta = lifecycle.build_connector_meta(_output({"req": len(PROMPT)}))
    assert sorted(meta.reqs_to_store) == ["req:store:0", "req:store:1"]

    assert lifecycle.request_finished(request, [0, 1, 2, 3]) == (True, None)
    ipc.live = False  # the decode allocations were evicted before publishing
    meta = lifecycle.build_connector_meta(_output())
    assert meta.reqs_to_store == {}
    # Reporting "req" unsaved would free blocks the prompt save still reads.
    assert meta.writeback_req_ids == set()
