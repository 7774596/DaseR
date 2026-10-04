# SPDX-License-Identifier: Apache-2.0
"""Write-back stores: finished requests hold blocks until the free watermark."""

from types import SimpleNamespace
from typing import Any

from daser.config import CACHE_REUSE_CHUNK
from daser.connector.metadata import DaserConnectorMeta, ReqStoreSpec
from daser.connector.scheduler.lifecycle import RequestLifecycle
from daser.connector.worker.runtime import WorkerRuntime

BLOCK_TOKENS = 4


class _FakeIPC:
    """Synchronous IPC stub recording store allocations."""

    def __init__(self) -> None:
        self.allocs: list[str] = []

    def get_runtime_config(self) -> dict[str, Any]:
        """Return a minimal server runtime config."""
        return {
            "slot_size": 32,
            "block_tokens": BLOCK_TOKENS,
            "model_id": "m",
            "cache_reuse_mode": CACHE_REUSE_CHUNK,
        }

    def alloc_chunk(self, chunk_key: str, token_count: int, model_id: str) -> dict:
        """Record and return server-style allocation metadata."""
        del token_count, model_id
        self.allocs.append(chunk_key)
        return {"chunk_key": chunk_key, "start_slot": 0, "file_offset": 0}

    def live_allocations(self, chunks: list[dict[str, Any]]) -> set[str]:
        """Report every allocation as still live."""
        return {str(chunk["chunk_key"]) for chunk in chunks}


class _FakePool:
    """GPU block pool stub exposing free count and per-block ref counts."""

    def __init__(self, num_blocks: int, free: int) -> None:
        self.blocks = [SimpleNamespace(ref_cnt=0) for _ in range(num_blocks)]
        self.free = free

    def get_num_free_blocks(self) -> int:
        """Return the configured free-block count."""
        return self.free


class _Output:
    """Scheduler output with nothing scheduled."""

    num_scheduled_tokens: dict[str, int] = {}


def _finished_request(
    low_blocks: int, pool: _FakePool
) -> tuple[RequestLifecycle, _FakeIPC, SimpleNamespace]:
    ipc = _FakeIPC()
    lifecycle = RequestLifecycle(
        ipc_client=ipc,
        block_tokens=BLOCK_TOKENS,
        slot_size=32,
        model_id="m",
        cache_reuse_mode=CACHE_REUSE_CHUNK,
        runtime_config_ready=False,
        writeback_low_blocks=low_blocks,
    )
    lifecycle.refresh_runtime_config()
    lifecycle.bind_gpu_block_pool(pool)
    request = SimpleNamespace(
        request_id="req",
        prompt_token_ids=list(range(3 * BLOCK_TOKENS)),
        kv_transfer_params={"daser_skip_load": True},
        num_output_tokens=2,
        num_computed_tokens=3 * BLOCK_TOKENS + 2,
    )
    assert lifecycle.get_num_new_matched_tokens(request, 0) == (0, False)
    blocks = SimpleNamespace(
        blocks=[[SimpleNamespace(block_id=idx) for idx in range(4)]]
    )
    lifecycle.update_state_after_alloc(request, blocks, 0)
    for block in pool.blocks[:4]:
        block.ref_cnt = 1
    return lifecycle, ipc, request


def test_writeback_off_allocates_at_prefill() -> None:
    pool = _FakePool(num_blocks=8, free=0)
    lifecycle, ipc, request = _finished_request(0, pool)
    assert len(ipc.allocs) == 1
    meta = lifecycle.build_connector_meta(_Output())
    assert meta.writeback_req_ids == set()
    assert lifecycle.request_finished(request, [0, 1, 2, 3]) == (False, None)


def test_writeback_holds_until_free_blocks_fall_below_watermark() -> None:
    pool = _FakePool(num_blocks=8, free=10)
    lifecycle, ipc, request = _finished_request(4, pool)
    # The prefix is still GPU-resident: nothing is allocated in DaseR yet.
    assert ipc.allocs == []
    assert lifecycle.request_finished(request, [0, 1, 2, 3]) == (True, None)
    meta = lifecycle.build_connector_meta(_Output())
    assert meta.writeback_req_ids == set()
    assert meta.reqs_to_store == {}

    pool.free = 1
    meta = lifecycle.build_connector_meta(_Output())
    assert meta.writeback_req_ids == {"req"}
    assert list(meta.reqs_to_store) == ["req"]
    assert meta.reqs_to_store["req"].block_ids == [0, 1, 2]
    assert len(ipc.allocs) == 1

    lifecycle.update_connector_output(
        SimpleNamespace(finished_sending={"req"}, finished_recving=None)
    )
    meta = lifecycle.build_connector_meta(_Output())
    assert meta.writeback_req_ids == set()
    assert meta.reqs_to_store == {}


def test_writeback_release_shared_with_live_request_writes_nothing() -> None:
    pool = _FakePool(num_blocks=8, free=10)
    lifecycle, ipc, request = _finished_request(4, pool)
    assert lifecycle.request_finished(request, [0, 1, 2, 3]) == (True, None)
    # A live continuation shares the whole prompt; its own store covers it.
    for block in pool.blocks[:3]:
        block.ref_cnt = 2
    pool.free = 0
    meta = lifecycle.build_connector_meta(_Output())
    assert meta.writeback_req_ids == {"req"}
    assert meta.reqs_to_store == {}
    assert ipc.allocs == []


def _spec(block_ids: list[int]) -> ReqStoreSpec:
    return ReqStoreSpec(
        chunk_key="k",
        start_slot=0,
        num_slots=len(block_ids),
        block_ids=block_ids,
        file_offset=0,
        token_count=len(block_ids) * BLOCK_TOKENS,
    )


class _StorePipeline:
    """Store pipeline stub that finishes every queued request immediately."""

    def __init__(self) -> None:
        self.queued: dict[str, ReqStoreSpec] = {}

    def queue_finished(self, reqs_to_store: dict[str, ReqStoreSpec]) -> None:
        """Record queued store specs."""
        self.queued.update(reqs_to_store)

    def collect_finished(self, finished_req_ids: set[str]) -> set[str]:
        """Return queued requests that vLLM reported finished."""
        return {req_id for req_id in self.queued if req_id in finished_req_ids}


class _LoadPipeline:
    """Load pipeline stub with no completions."""

    def collect_finished(self) -> set[str]:
        """Return no finished loads."""
        return set()


def test_worker_reports_writeback_releases_on_no_forward_step() -> None:
    runtime = WorkerRuntime.__new__(WorkerRuntime)
    runtime._store_pipeline = _StorePipeline()  # noqa: SLF001
    runtime._load_pipeline = _LoadPipeline()  # noqa: SLF001
    runtime._ensure_transfer_ready = lambda: True  # noqa: SLF001
    meta = DaserConnectorMeta(
        reqs_to_store={"saved": _spec([0, 1]), "live": _spec([2])},
        writeback_req_ids={"saved", "superseded"},
    )
    runtime.bind_connector_metadata(meta)
    # vLLM skips wait_for_save on no-forward steps; get_finished alone must
    # queue the released store and report both releases.
    sending, recving = runtime.get_finished(set())
    assert sending == {"saved", "superseded"}
    assert recving is None
    assert set(runtime._store_pipeline.queued) == {"saved"}  # noqa: SLF001

    runtime.bind_connector_metadata(meta)
    runtime.wait_for_save()
    assert set(runtime._store_pipeline.queued) == {"saved", "live"}  # noqa: SLF001
