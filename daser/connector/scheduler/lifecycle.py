# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import logging
import math
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

from daser import stalldump
from daser.config import STORAGE_FORMAT_RAW
from daser.connector.helpers import PendingStore
from daser.connector.ipc_client import PrefetchLookupResult
from daser.connector.metadata import DaserConnectorMeta, ReqLoadSpec, ReqStoreSpec
from daser.connector.scheduler.planning import (
    _base_req_id,
    _computed_tokens_after_step,
    _contiguous_prefix_tokens,
    _get_kv_transfer_flag,
    _load_spec_from_chunk,
    _matches_request_or_store_id,
    _merge_adjacent_load_specs,
    _store_slot_index,
    _trim_chunk_to_external_window,
)
from daser.connector.scheduler.reuse import build_cache_reuse_strategy
from daser.logging import init_logger

logger = init_logger(__name__)

# How long a request may keep deferring its lookup while a pending DaseR
# writer could extend the match. It matches the server-side retry budget
# (2+4+8+16+32+64 ms) but is spent across scheduler steps instead of blocking
# one step.
PENDING_LOOKUP_DEFER_S = 0.126

# Upper bound on how long a request waits for an in-flight load of the same
# prefix before loading its own copy (guards against a stalled leader).
COALESCE_MAX_WAIT_S = 10.0

# Scheduler steps an in-flight load stays joinable after its worker reported
# completion: vLLM caches the landed blocks when it promotes the leader in the
# next schedule() call, which may visit a follower first.
COALESCE_LANDING_STEPS = 1


def _prefix_digest(tokens: list[int], end: int) -> int:
    """Return an in-process digest of ``tokens[:end]``.

    Args:
        tokens: request prompt token IDs.
        end: prefix length in tokens.

    Returns:
        Hash of the prefix. A collision only makes a request wait for a load
        it cannot reuse; vLLM's own block hashes decide what is actually hit.
    """
    return hash(tuple(tokens[:end]))


def _prefetch_external_spans(
    socket_path: str,
    lease_id: str,
    spans: list[dict[str, int]],
) -> dict[str, int]:
    """Prefetch storage spans over a dedicated synchronous IPC connection.

    The RPC can legitimately wait for other leases to free L1 space, which
    lasts as long as those requests run. The connection therefore has no
    socket timeout: a client-side timeout would abandon (and, via the retry,
    resend) a prefetch the server is still executing under this lease.
    The scheduler only polls the returned future, so it never blocks here.
    """
    from daser.connector.ipc_client import IPCClientSync

    client = IPCClientSync(socket_path, timeout=None)
    try:
        return client.transfer_prefetch(spans, lease_id=lease_id)
    finally:
        client.close()


class RequestLifecycle:
    """Own scheduler request state and synchronous IPC orchestration.

    Async/thread-safety:
        These methods run on vLLM's scheduler thread and use the synchronous
        IPC client owned by the connector instance.
    """

    def __init__(
        self,
        *,
        ipc_client: Any,
        block_tokens: int,
        slot_size: int,
        model_id: str,
        cache_reuse_mode: str,
        runtime_config_ready: bool,
        socket_path: str = "",
        prefetch_max_requests: int = 0,
        storage_format: str | None = None,
        coalesce_loads: bool = False,
        layerwise_load: bool = False,
        admit_trace: bool = False,
    ) -> None:
        self._ipc_sync = ipc_client
        # Research switch: requests whose first lookup was traced, or None
        # when admission tracing is off (see _trace_admit).
        self._admit_traced: set[str] | None = set() if admit_trace else None
        # Last (local, external) token split traced per request (see trace_match).
        self._admit_matches: dict[str, tuple[int, int | None]] | None = (
            {} if admit_trace else None
        )
        # Why the last get_num_new_matched_tokens call returned None ("" when
        # it returned a count); read by the scheduler probe.
        self.last_defer_reason = ""
        # Last lookup per request kept for vLLM's allocation retries:
        # (num_computed_tokens, prefix length, index epoch, chunks).
        self._retry_lookups: dict[str, tuple[int, int, int, list[dict[str, Any]]]] = {}
        # Server index epoch already read in this scheduler step, if any.
        self._step_epoch: int | None = None
        self._block_tokens = block_tokens
        self._slot_size = slot_size
        self._model_id = model_id
        self._cache_reuse_mode = cache_reuse_mode
        self._runtime_config_ready = runtime_config_ready
        self._declared_storage_format = storage_format
        self._socket_path = socket_path
        self._prefetch_max_requests = prefetch_max_requests
        self._cache_reuse_strategy = build_cache_reuse_strategy(
            cache_reuse_mode,
            block_tokens,
        )
        self._pending_loads: dict[str, dict[str, Any]] = {}
        self._pending_stores: dict[str, dict[str, Any]] = {}
        self._pending_alloc: dict[str, PendingStore] = {}
        self._pending_async_saves: set[str] = set()
        self._lookup_defer_deadlines: dict[str, float] = {}
        self._req_tokens: dict[str, list[int]] = {}
        self._prefetch_futures: dict[str, Future[dict[str, int]]] = {}
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_signatures: dict[str, tuple[tuple[int, int], ...]] = {}
        # Lookup results parked across scheduler steps, keyed by request and
        # paired with the vLLM computed-token count their spans start at.
        self._prefetch_lookup_results: dict[str, tuple[int, PrefetchLookupResult]] = {}
        self._leased_request_ids: set[str] = set()
        # Research switch (server ``protect_waiting_bytes`` > 0): requests
        # whose cached prefix the server protects while they wait in vLLM's
        # queue; released once vLLM allocates their blocks without an
        # external load, the load lands, or the request is preempted or
        # finishes.
        self._protect_waiting = False
        self._protected_request_ids: set[str] = set()
        # Research switch: fully leased packed or raw hits load synchronously
        # and are restored layer by layer inside the forward pass. The worker
        # then owns the request lease, so these IDs never report
        # finished_recving.
        self._layerwise_load = layerwise_load
        self._layerwise_request_ids: set[str] = set()
        # In-flight load coalescing: vLLM only publishes externally loaded
        # blocks to its prefix cache after the load lands, so a concurrent
        # request for the same prefix would allocate and load its own copy.
        # Leaders map to (prefix end, prefix digest, landing steps left; -1
        # while the load is still in flight); followers map to (awaited end,
        # first deferral time).
        self._coalesce_loads = coalesce_loads
        self._inflight_loads: dict[str, tuple[int, int, int]] = {}
        self._coalesce_followers: dict[str, tuple[int, float]] = {}
        self._follower_digests: dict[str, dict[int, int]] = {}
        self._coalesce_stats = {"followers": 0, "joined": 0, "missed": 0}
        stalldump.register_state(
            "scheduler",
            lambda: {
                "pending_async_saves": sorted(self._pending_async_saves),
                "pending_loads": sorted(self._pending_loads),
                "pending_stores": sorted(self._pending_stores),
                "pending_alloc": len(self._pending_alloc),
                "defer_deadlines": dict(self._lookup_defer_deadlines),
                "inflight_loads": dict(self._inflight_loads),
                "now": time.monotonic(),
            },
        )

    def on_new_request(self, request: "Request") -> None:
        """Protect a request's cached prefix while it waits to be scheduled.

        When the server enables queued-request protection, the prompt is
        sent with the request id as lease, so the prefix indexed at arrival
        is not evicted before the scheduling lookup or, for a request that
        loads external KV, before that load lands. Best effort: failures
        are logged and the request is scheduled normally.

        Args:
            request: vLLM Request entering the waiting queue.

        Thread-safety:
            Runs on the scheduler thread through its synchronous IPC client.
        """
        if not getattr(self, "_runtime_config_ready", True):
            self._refresh_runtime_config()
        if not getattr(self, "_protect_waiting", False):
            return
        req_id = request.request_id
        try:
            protected = self._ipc_sync.protect_prefix(
                req_id, list(request.prompt_token_ids)
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[CONNECTOR] protect failed req=%s: %s", req_id[:8], exc)
            return
        if protected:
            self._protected_request_ids.add(req_id)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> "tuple[int | None, bool]":
        """Query DaseR for cached KV matching request tokens.

        Args:
            request: vLLM Request with prompt_token_ids.
            num_computed_tokens: tokens already in vLLM's KV cache.

        Returns:
            (num_external_tokens, is_async) - (0, False) on miss. A
            ``None`` count sets :attr:`last_defer_reason`.
        """
        self.last_defer_reason = ""
        self._trace_admit("lookup", request.request_id)
        tokens = list(request.prompt_token_ids)
        self._req_tokens[request.request_id] = tokens
        if not getattr(self, "_runtime_config_ready", True):
            self._refresh_runtime_config()

        prefetch_limit = int(getattr(self, "_prefetch_max_requests", 0))
        socket_path = str(getattr(self, "_socket_path", ""))
        prefetch_enabled = prefetch_limit > 0 and bool(socket_path)
        prefetch_futures = getattr(self, "_prefetch_futures", {})
        prefetch_future = prefetch_futures.get(request.request_id)
        parked = getattr(self, "_prefetch_lookup_results", {}).get(request.request_id)
        prefetch_lookup_result = None
        if parked is not None:
            parked_start, prefetch_lookup_result = parked
        if parked is not None and parked_start != num_computed_tokens:
            # vLLM's local prefix hit moved since the lookup (for example the
            # GPU blocks were evicted), so the parked spans and any lease built
            # from them no longer match the window the worker will load.
            if prefetch_future is not None and not prefetch_future.done():
                self.last_defer_reason = "prefetch"
                return None, True
            if (
                prefetch_future is not None
                and not prefetch_future.cancelled()
                and prefetch_future.exception() is not None
            ):
                logger.warning(
                    "[CONNECTOR] prefetch failed req=%s parked_tokens=%d: %s",
                    request.request_id[:8],
                    _contiguous_prefix_tokens(
                        prefetch_lookup_result.chunks, parked_start
                    ),
                    prefetch_future.exception(),
                )
            prefetch_futures.pop(request.request_id, None)
            prefetch_future = None
            getattr(self, "_prefetch_lookup_results", {}).pop(request.request_id, None)
            getattr(self, "_prefetch_signatures", {}).pop(request.request_id, None)
            self._release_transfer_lease(request.request_id, force=True)
            prefetch_lookup_result = None
        if prefetch_future is not None:
            if not prefetch_future.done():
                self.last_defer_reason = "prefetch"
                return None, True
            try:
                prefetch_future.result()
                getattr(self, "_leased_request_ids", set()).add(request.request_id)
            except Exception as exc:  # noqa: BLE001
                # parked_tokens bounds what the fresh lookup below can lose.
                logger.warning(
                    "[CONNECTOR] prefetch failed req=%s parked_tokens=%d: %s",
                    request.request_id[:8],
                    0
                    if prefetch_lookup_result is None
                    else _contiguous_prefix_tokens(
                        prefetch_lookup_result.chunks, num_computed_tokens
                    ),
                    exc,
                )
                getattr(self, "_prefetch_lookup_results", {}).pop(
                    request.request_id, None
                )
                self._release_transfer_lease(request.request_id, force=True)
                prefetch_lookup_result = None
            prefetch_futures.pop(request.request_id, None)

        if getattr(self, "_coalesce_loads", False) and prefetch_lookup_result is None:
            # A fresh lookup supersedes any load this request registered
            # before (for example when vLLM could not allocate its blocks).
            self._inflight_loads.pop(request.request_id, None)
            if self._wait_for_inflight_load(
                request.request_id, tokens, num_computed_tokens
            ):
                # vLLM keeps the request waiting; once the in-flight load lands
                # its blocks are a local prefix hit instead of a second load.
                self.last_defer_reason = "coalesce"
                return None, False

        start = num_computed_tokens
        available = len(tokens) - start
        if available < self._block_tokens:
            getattr(self, "_prefetch_lookup_results", {}).pop(request.request_id, None)
            self._release_transfer_lease(request.request_id)
            self._record_external_prefix_cache_miss(available)
            return 0, False

        skip_load = bool(_get_kv_transfer_flag(request, "daser_skip_load"))
        if skip_load:
            getattr(self, "_prefetch_lookup_results", {}).pop(request.request_id, None)
            self._release_transfer_lease(request.request_id)
            logger.debug("[CONNECTOR] skip load req=%s", request.request_id[:8])
            self._record_external_prefix_cache_miss(available)
            full_aligned = (len(tokens) // self._block_tokens) * self._block_tokens
            skip_save = bool(_get_kv_transfer_flag(request, "daser_skip_save"))
            pending_store = (
                None
                if skip_save
                else self._reuse_strategy().prepare_store(tokens, full_aligned)
            )
            if pending_store is not None:
                self._pending_alloc[request.request_id] = pending_store
            return 0, False

        aligned = (available // self._block_tokens) * self._block_tokens
        prefix = tokens[: start + aligned]
        full_aligned = (len(tokens) // self._block_tokens) * self._block_tokens
        skip_save = bool(_get_kv_transfer_flag(request, "daser_skip_save"))

        if prefetch_lookup_result is None:
            try:
                if prefetch_enabled:
                    prefetch_lookup_result = self._lookup_with_prefetch_metrics(
                        request.request_id,
                        prefix,
                        self._model_id,
                        max(0, len(tokens) - num_computed_tokens),
                        num_computed_tokens,
                    )
                    chunks = prefetch_lookup_result.chunks
                    if prefetch_lookup_result.tier == "l1":
                        getattr(self, "_leased_request_ids", set()).add(
                            request.request_id
                        )
                else:
                    deferred_chunks = self._lookup_deferring_pending(
                        request.request_id,
                        prefix,
                        max(0, len(tokens) - num_computed_tokens),
                        num_computed_tokens,
                    )
                    if deferred_chunks is None:
                        # vLLM keeps the request waiting and asks again next
                        # step, so the scheduler never blocks on the writer.
                        self.last_defer_reason = "pending_writer"
                        return None, False
                    chunks = deferred_chunks
            except Exception as exc:
                logger.warning("[CONNECTOR] lookup failed: %s", exc)
                self._release_transfer_lease(request.request_id, force=True)
                self._runtime_config_ready = False
                return 0, False
        else:
            chunks = prefetch_lookup_result.chunks

        if not chunks:
            getattr(self, "_prefetch_lookup_results", {}).pop(request.request_id, None)
            pending_store = (
                None
                if skip_save
                else self._reuse_strategy().prepare_store(tokens, full_aligned)
            )
            if pending_store is not None:
                self._pending_alloc[request.request_id] = pending_store
            logger.debug("[CONNECTOR] cache miss req=%s", request.request_id[:8])
            return 0, False

        pending_store = (
            None
            if skip_save
            else self._reuse_strategy().prepare_store(
                tokens,
                full_aligned,
                chunks,
            )
        )
        if pending_store is not None:
            self._pending_alloc[request.request_id] = pending_store

        extra_tokens = _contiguous_prefix_tokens(chunks, num_computed_tokens)
        if extra_tokens <= 0:
            getattr(self, "_prefetch_lookup_results", {}).pop(request.request_id, None)
            self._release_transfer_lease(request.request_id)
            return 0, False

        available = len(tokens) - num_computed_tokens
        if extra_tokens >= available:
            extra_tokens = available - 1
            if extra_tokens <= 0:
                getattr(self, "_prefetch_lookup_results", {}).pop(
                    request.request_id, None
                )
                self._release_transfer_lease(request.request_id)
                return 0, False

        if getattr(self, "_coalesce_loads", False):
            end = num_computed_tokens + extra_tokens
            current = self._inflight_loads.get(request.request_id)
            if current is None or current[0] != end:
                self._inflight_loads[request.request_id] = (
                    end,
                    _prefix_digest(tokens, end),
                    -1,
                )

        if (
            prefetch_enabled
            and prefetch_lookup_result is not None
            and prefetch_lookup_result.tier in ("mixed", "l2")
        ):
            spans = prefetch_lookup_result.spans
            signature = tuple(
                sorted(
                    (int(span["file_offset"]), int(span["nbytes"])) for span in spans
                )
            )
            signatures = getattr(self, "_prefetch_signatures", {})
            if signature and signatures.get(request.request_id) != signature:
                active = sum(not future.done() for future in prefetch_futures.values())
                if active >= prefetch_limit:
                    getattr(self, "_prefetch_lookup_results", {})[
                        request.request_id
                    ] = (num_computed_tokens, prefetch_lookup_result)
                    self.last_defer_reason = "prefetch_limit"
                    return None, True
                executor = getattr(self, "_prefetch_executor", None)
                if executor is None:
                    executor = ThreadPoolExecutor(
                        max_workers=prefetch_limit,
                        thread_name_prefix="daser-prefetch",
                    )
                    self._prefetch_executor = executor
                signatures[request.request_id] = signature
                self._prefetch_signatures = signatures
                getattr(self, "_prefetch_lookup_results", {})[request.request_id] = (
                    num_computed_tokens,
                    prefetch_lookup_result,
                )
                prefetch_futures[request.request_id] = executor.submit(
                    _prefetch_external_spans,
                    socket_path,
                    request.request_id,
                    spans,
                )
                self._prefetch_futures = prefetch_futures
                self.last_defer_reason = "prefetch"
                return None, True

        if len(chunks) == 1:
            self._pending_loads[request.request_id] = dict(
                chunks[0],
                num_computed_tokens=num_computed_tokens,
                lease_id=(
                    request.request_id
                    if request.request_id in getattr(self, "_leased_request_ids", set())
                    else ""
                ),
            )
        else:
            self._pending_loads[request.request_id] = {
                str(i): dict(
                    chunk,
                    num_computed_tokens=num_computed_tokens,
                    lease_id=(
                        request.request_id
                        if request.request_id
                        in getattr(self, "_leased_request_ids", set())
                        else ""
                    ),
                )
                for i, chunk in enumerate(chunks)
            }

        logger.debug(
            "[CONNECTOR] cache hit req=%s chunks=%d prefix_tokens=%d",
            request.request_id[:8],
            len(chunks),
            extra_tokens,
        )
        getattr(self, "_prefetch_lookup_results", {}).pop(request.request_id, None)
        layerwise_ids = getattr(self, "_layerwise_request_ids", set())
        if self._layerwise_eligible(request.request_id, chunks):
            layerwise_ids.add(request.request_id)
            return extra_tokens, False
        layerwise_ids.discard(request.request_id)
        return extra_tokens, True

    def trace_match(
        self,
        req_id: str,
        num_computed_tokens: int,
        num_external_tokens: int | None,
        num_prompt_tokens: int,
    ) -> None:
        """Log a lookup's local/external/total token split under admit tracing.

        A line is written only when the split differs from the last one traced
        for the request, so allocation retries add nothing while a preempted
        request's resume lookup does.

        Args:
            req_id: vLLM request id.
            num_computed_tokens: tokens vLLM already holds locally (GPU prefix
                cache) at this lookup.
            num_external_tokens: DaseR credit returned by the lookup, ``None``
                when the request was deferred.
            num_prompt_tokens: prompt length of the request.

        Async/thread-safety:
            Called on the scheduler thread only; no-op when tracing is off.
        """
        matches = getattr(self, "_admit_matches", None)
        split = (num_computed_tokens, num_external_tokens)
        if matches is None or matches.get(req_id) == split:
            return
        matches[req_id] = split
        logger.info(
            "[ADMIT] match %s local=%d ext=%s total=%d t=%.3f",
            req_id,
            num_computed_tokens,
            num_external_tokens,
            num_prompt_tokens,
            time.monotonic(),
        )

    def shutdown(self) -> None:
        """Stop scheduler-side prefetch workers and release their resources.

        Async/thread-safety:
            Called by vLLM after scheduler traffic stops. It waits for active
        IPC calls to finish so their sockets are closed cleanly.
        """
        futures = getattr(self, "_prefetch_futures", {})
        cleanup_ids = (
            set(futures)
            | set(getattr(self, "_prefetch_lookup_results", {}))
            | set(getattr(self, "_leased_request_ids", set()))
        )
        for future in futures.values():
            future.cancel()
        executor = getattr(self, "_prefetch_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._prefetch_executor = None
        futures.clear()
        for req_id in cleanup_ids:
            self._release_transfer_lease(req_id, force=True)
        getattr(self, "_prefetch_lookup_results", {}).clear()
        getattr(self, "_prefetch_signatures", {}).clear()
        if getattr(self, "_coalesce_loads", False):
            self._log_coalesce_stats()

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Record block IDs for requests that will load or store KV.

        Args:
            request: vLLM Request.
            blocks: vLLM KV cache block allocation for this request.
            num_external_tokens: tokens from DaseR (0 if miss).
        """
        req_id = request.request_id
        self._trace_admit("alloc", req_id)
        if not num_external_tokens:
            # A request loading external KV keeps its prefix protected until
            # the load lands, so its chunks cannot be evicted and their bytes
            # reused while the read is queued.
            self._release_protection(req_id)
        block_ids: list[int] = [blk.block_id for blk in blocks.blocks[0]]
        getattr(self, "_retry_lookups", {}).pop(req_id, None)

        if req_id in self._pending_loads:
            chunks = self._pending_loads[req_id]
            if "chunk_key" in chunks:
                chunk = chunks
                if not _trim_chunk_to_external_window(
                    chunk=chunk,
                    block_ids=block_ids,
                    external_start=int(chunk.get("num_computed_tokens", 0)),
                    num_external_tokens=num_external_tokens,
                    block_tokens=self._block_tokens,
                    slot_size=self._slot_size,
                ):
                    del self._pending_loads[req_id]
                    self._release_transfer_lease(req_id)
                    self._record_pending_store_blocks(req_id, block_ids)
                    return
                logger.debug(
                    "[CONNECTOR] load blocks req=%s blocks=%s",
                    req_id,
                    chunk["block_ids"],
                )
                self._record_pending_store_blocks(req_id, block_ids)
                return
            for key, chunk in list(chunks.items()):
                if not _trim_chunk_to_external_window(
                    chunk=chunk,
                    block_ids=block_ids,
                    external_start=int(chunk.get("num_computed_tokens", 0)),
                    num_external_tokens=num_external_tokens,
                    block_tokens=self._block_tokens,
                    slot_size=self._slot_size,
                ):
                    logger.debug(
                        "[CONNECTOR] skip load req=%s key=%s target=%d slots=%d",
                        req_id[:8],
                        chunk.get("chunk_key", "")[:8],
                        int(chunk.get("target_token_start", 0)),
                        int(chunk["num_slots"]),
                    )
                    del chunks[key]
                    continue
                logger.debug(
                    "[CONNECTOR] load blocks req=%s key=%s blocks=%s",
                    req_id,
                    chunk.get("chunk_key", "")[:8],
                    chunk["block_ids"],
                )
            if not chunks:
                self._pending_loads.pop(req_id, None)
                self._release_transfer_lease(req_id)
        self._record_pending_store_blocks(req_id, block_ids)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> DaserConnectorMeta:
        """Package pending load/store specs into connector metadata.

        Args:
            scheduler_output: vLLM SchedulerOutput for this step.

        Returns:
            DaserConnectorMeta with reqs_to_load and reqs_to_store.
        """
        meta = DaserConnectorMeta()
        self._age_landed_loads()
        self._step_epoch = None
        # Published stores have left _pending_stores but still reference model
        # blocks on the worker. Capture their cancellation before dropping the
        # scheduler hold; the worker must discard those unsent specifications.
        meta.cancelled_store_req_ids = self._pending_async_save_ids().intersection(
            getattr(scheduler_output, "preempted_req_ids", None) or ()
        )
        self._drop_preempted_pending_state(scheduler_output)
        scheduled_ids: set[str] = set(scheduler_output.num_scheduled_tokens.keys())
        computed_after = _computed_tokens_after_step(scheduler_output)
        self._record_cached_store_blocks(scheduler_output)

        for req_id, chunks in list(self._pending_loads.items()):
            if req_id in getattr(self, "_layerwise_request_ids", set()):
                self._hand_off_layerwise_load(req_id, chunks, meta)
                continue
            if "chunk_key" in chunks:
                chunk = chunks
                if "block_ids" in chunk:
                    meta.reqs_to_load[req_id] = _load_spec_from_chunk(chunk)
                    del self._pending_loads[req_id]
                continue
            ready = True
            load_specs: list[ReqLoadSpec] = []
            for chunk in chunks.values():
                if "block_ids" not in chunk:
                    ready = False
                    continue
                load_specs.append(_load_spec_from_chunk(chunk))
            merged_specs = _merge_adjacent_load_specs(load_specs, self._slot_size)
            for idx, spec in enumerate(merged_specs):
                load_id = req_id if len(merged_specs) == 1 else f"{req_id}:load:{idx}"
                meta.reqs_to_load[load_id] = spec
            if ready:
                del self._pending_loads[req_id]

        for req_id, alloc in list(self._pending_stores.items()):
            scheduled_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if scheduled_tokens <= 0 and ":store:" in req_id:
                scheduled_tokens = scheduler_output.num_scheduled_tokens.get(
                    _base_req_id(req_id),
                    0,
                )
            base_req_id = _base_req_id(req_id)
            computed_tokens = computed_after.get(base_req_id, scheduled_tokens)
            slot_index = _store_slot_index(req_id)
            required_tokens = (
                (slot_index + 1) * self._block_tokens
                if slot_index is not None
                else int(alloc["token_count"])
            )
            should_store = (
                "block_ids" in alloc
                and base_req_id in scheduled_ids
                and scheduled_tokens > 0
                and computed_tokens >= required_tokens
            )
            if should_store:
                logical_slot_start = _store_slot_index(req_id)
                meta.reqs_to_store[req_id] = ReqStoreSpec(
                    chunk_key=alloc["chunk_key"],
                    start_slot=alloc["start_slot"],
                    num_slots=alloc["num_slots"],
                    block_ids=alloc["block_ids"],
                    file_offset=alloc["file_offset"],
                    token_count=alloc["token_count"],
                    logical_slot_start=(
                        logical_slot_start if logical_slot_start is not None else 0
                    ),
                )
                del self._pending_stores[req_id]

        if meta.reqs_to_store:
            meta.reqs_to_store = self._filter_live_store_specs(meta.reqs_to_store)
            pending_async_saves = self._pending_async_save_ids()
            for req_id in meta.reqs_to_store:
                pending_async_saves.add(_base_req_id(req_id))
        if logger.isEnabledFor(logging.DEBUG):
            for req_id, load_spec in meta.reqs_to_load.items():
                logger.debug(
                    "[CONNECTOR] meta LOAD  req=%s start_slot=%d blocks=%d tokens=%d",
                    req_id[:8],
                    load_spec.start_slot,
                    len(load_spec.block_ids),
                    load_spec.token_count,
                )
            for req_id, store_spec in meta.reqs_to_store.items():
                logger.debug(
                    "[CONNECTOR] meta STORE req=%s start_slot=%d blocks=%d tokens=%d",
                    req_id[:8],
                    store_spec.start_slot,
                    len(store_spec.block_ids),
                    store_spec.token_count,
                )
        return meta

    def refresh_runtime_config(self) -> None:
        """Refresh scheduler geometry and reuse policy from the DaseR server.

        Returns:
            None.

        Async/thread-safety:
            Called on the scheduler thread and performs synchronous control-plane
            IPC during connector initialization or recovery, never worker IO.
        """
        self._refresh_runtime_config()

    def _drop_preempted_pending_state(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """Discard pending scheduler state whose KV blocks were preempted.

        Args:
            scheduler_output: vLLM SchedulerOutput for this step.

        Async/thread-safety:
            Runs on the scheduler thread before metadata is handed to workers.
        """
        preempted_req_ids: set[str] = getattr(
            scheduler_output, "preempted_req_ids", set()
        )
        pending_async_saves = self._pending_async_save_ids()
        for req_id in preempted_req_ids:
            base_req_id = str(req_id)
            getattr(self, "_retry_lookups", {}).pop(base_req_id, None)
            pending_async_saves.discard(base_req_id)
            self._release_transfer_lease(base_req_id, force=True)
            self._release_protection(base_req_id)
            getattr(self, "_prefetch_lookup_results", {}).pop(base_req_id, None)
            getattr(self, "_prefetch_signatures", {}).pop(base_req_id, None)
            self._cancel_prefetch_future(base_req_id)
            for pending_req_id in list(self._pending_loads):
                if _matches_request_or_store_id(pending_req_id, base_req_id):
                    self._pending_loads.pop(pending_req_id, None)
            for pending_req_id in list(self._pending_stores):
                if _matches_request_or_store_id(pending_req_id, base_req_id):
                    self._drop_pending_store(pending_req_id)
            for pending_req_id in list(self._pending_alloc):
                if _matches_request_or_store_id(pending_req_id, base_req_id):
                    self._pending_alloc.pop(pending_req_id, None)

    def _filter_live_store_specs(
        self,
        specs: dict[str, ReqStoreSpec],
    ) -> dict[str, ReqStoreSpec]:
        """Drop store specs whose server allocation was already evicted.

        Args:
            specs: Store specs built for the current scheduler step.

        Returns:
            Specs that still own their allocated server slot ranges.
        """
        try:
            live_keys = self._ipc_sync.live_allocations(
                [
                    {
                        "chunk_key": spec.chunk_key,
                        "start_slot": spec.start_slot,
                        "num_slots": spec.num_slots,
                    }
                    for spec in specs.values()
                ]
            )
        except Exception as exc:
            logger.warning("[CONNECTOR] live_allocations failed: %s", exc)
            return specs
        return {
            req_id: spec
            for req_id, spec in specs.items()
            if spec.chunk_key in live_keys
        }

    def _record_cached_store_blocks(self, scheduler_output: "SchedulerOutput") -> None:
        """Append blocks from later chunked-prefill steps to store trackers.

        Args:
            scheduler_output: vLLM SchedulerOutput for this step.
        """
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached_reqs is None:
            return

        req_ids = getattr(cached_reqs, "req_ids", [])
        new_block_ids = getattr(cached_reqs, "new_block_ids", [])
        for req_id, block_groups in zip(req_ids, new_block_ids, strict=False):
            pending_store = self._pending_alloc.get(req_id)
            if pending_store is None or block_groups is None:
                continue
            if not block_groups:
                continue
            block_group = block_groups[0]
            if block_group is None:
                continue
            block_ids = list(block_group)
            if (
                req_id in getattr(cached_reqs, "resumed_req_ids", set())
                and block_ids[: len(pending_store.block_ids)] == pending_store.block_ids
            ):
                pending_store.block_ids = block_ids
            else:
                pending_store.block_ids.extend(block_ids)
            needed_slots = math.ceil(pending_store.token_count / self._block_tokens)
            if len(pending_store.block_ids) > needed_slots:
                pending_store.block_ids = pending_store.block_ids[:needed_slots]
            self._maybe_allocate_pending_store(req_id, pending_store)

    def _lookup_with_external_prefix_metrics(
        self,
        tokens: list[int],
        model_id: str,
        queries: int,
        num_computed_tokens: int,
        defer_pending: bool | None = None,
    ) -> list[dict[str, Any]] | None:
        """Run lookup while passing vLLM external-prefix query count when supported.

        Args:
            tokens: token prefix sent to DaseR lookup.
            model_id: model identifier.
            queries: vLLM external prefix query token count.
            num_computed_tokens: tokens already computed locally by vLLM.
            defer_pending: Pending-writer policy forwarded to
                ``IPCClientSync.lookup``; None keeps the server default.

        Returns:
            Matching chunk dicts returned by the IPC client, or None when the
            server deferred the lookup.

        Thread-safety:
            Runs on the scheduler thread and uses the synchronous IPC client.
        """
        try:
            return self._ipc_sync.lookup(
                tokens,
                model_id,
                external_prefix_queries=queries,
                num_computed_tokens=num_computed_tokens,
                defer_pending=defer_pending,
            )
        except TypeError as exc:
            if "external_prefix_queries" not in str(exc):
                raise
            return self._ipc_sync.lookup(tokens, model_id)

    def _lookup_deferring_pending(
        self,
        request_id: str,
        tokens: list[int],
        queries: int,
        num_computed_tokens: int,
    ) -> list[dict[str, Any]] | None:
        """Look up without blocking the scheduler on a pending DaseR writer.

        The first deferred attempt starts a ``PENDING_LOOKUP_DEFER_S`` budget
        for the request. Within it the server answers pending instead of
        waiting; after it the lookup resolves with whatever is committed.
        While another request's budget is running, a request without one is
        deferred without a lookup, so deferral never reorders vLLM's queue.

        Args:
            request_id: vLLM request ID owning the defer budget.
            tokens: Token prefix sent to the DaseR lookup.
            queries: vLLM external prefix query token count.
            num_computed_tokens: Tokens already computed locally by vLLM.

        Returns:
            Matching chunk dicts, or None when the lookup is deferred.

        Thread-safety:
            Runs on the scheduler thread and uses the synchronous IPC client.
        """
        deadlines = self._defer_deadlines()
        now = time.monotonic()
        if request_id not in deadlines and any(now < d for d in deadlines.values()):
            # An earlier request is deferred on a pending writer. Deferring
            # this one too keeps vLLM's FCFS order: otherwise it could take
            # the blocks the earlier request needs while it waits for its
            # remote KV, and vLLM's waiting loop stops at the earlier
            # request's failed allocation every step (deadlock).
            return None
        reused = self._reuse_retry_lookup(request_id, len(tokens), num_computed_tokens)
        if reused is not None:
            return reused
        deadline = deadlines.setdefault(request_id, now + PENDING_LOOKUP_DEFER_S)
        chunks, epoch = self._lookup_versioned(
            tokens,
            queries,
            num_computed_tokens,
            defer_pending=now < deadline,
        )
        if epoch is not None and self._step_epoch is None:
            self._step_epoch = epoch
        if chunks is not None and epoch is not None:
            self._retry_lookups[request_id] = (
                num_computed_tokens,
                len(tokens),
                epoch,
                [dict(chunk) for chunk in chunks],
            )
        if chunks is not None:
            deadlines.pop(request_id, None)
        return chunks

    def _trace_admit(self, event: str, req_id: str) -> None:
        """Log ``event`` for ``req_id`` with a monotonic timestamp if tracing.

        ``lookup`` is logged once per request (its first call); allocation
        retries do not repeat it. The clock matches the replay client's.
        """
        traced = getattr(self, "_admit_traced", None)
        if traced is None or (event == "lookup" and req_id in traced):
            return
        traced.add(req_id)
        logger.info("[ADMIT] %s %s t=%.3f", event, req_id, time.monotonic())

    def _lookup_versioned(
        self,
        tokens: list[int],
        queries: int,
        num_computed_tokens: int,
        *,
        defer_pending: bool,
    ) -> tuple[list[dict[str, Any]] | None, int | None]:
        """Look up chunks together with the index epoch they belong to.

        Args:
            tokens: Token prefix sent to the DaseR lookup.
            queries: vLLM external prefix query token count.
            num_computed_tokens: Tokens already computed locally by vLLM.
            defer_pending: Whether the server may answer pending.

        Returns:
            ``(chunks, epoch)``; ``chunks`` is None when the lookup was
            deferred, and ``epoch`` is None when the result must not be
            reused (deferred, or a client that reports no epoch).

        Thread-safety:
            Runs on the scheduler thread and uses the synchronous IPC client.
        """
        lookup_versioned = getattr(self._ipc_sync, "lookup_versioned", None)
        if lookup_versioned is None:
            chunks = self._lookup_with_external_prefix_metrics(
                tokens,
                self._model_id,
                queries,
                num_computed_tokens,
                defer_pending=defer_pending,
            )
            return chunks, None
        return lookup_versioned(
            tokens,
            self._model_id,
            external_prefix_queries=queries,
            num_computed_tokens=num_computed_tokens,
            defer_pending=defer_pending,
        )

    def _reuse_retry_lookup(
        self,
        request_id: str,
        prefix_len: int,
        num_computed_tokens: int,
    ) -> list[dict[str, Any]] | None:
        """Return the previous lookup for an allocation retry if still valid.

        Args:
            request_id: vLLM request ID.
            prefix_len: Length of the token prefix this call would look up.
            num_computed_tokens: vLLM's local prefix hit for this call.

        Returns:
            A copy of the kept chunk list when the request asks again for the
            same window and the server index epoch is unchanged, else None
            (a stale entry is dropped and a fresh lookup follows).

        Thread-safety:
            Runs on the scheduler thread. The epoch is read at most once per
            scheduler step, so a removal later in the same step is seen next
            step; that is the same window as between any lookup and its load.
        """
        retry_lookups = getattr(self, "_retry_lookups", {})
        entry = retry_lookups.get(request_id)
        if entry is None:
            return None
        start, length, epoch, chunks = entry
        if start != num_computed_tokens or length != prefix_len:
            retry_lookups.pop(request_id, None)
            return None
        if getattr(self, "_step_epoch", None) is None:
            self._step_epoch = self._ipc_sync.index_epoch()
        if self._step_epoch != epoch:
            retry_lookups.pop(request_id, None)
            return None
        return [dict(chunk) for chunk in chunks]

    def _defer_deadlines(self) -> dict[str, float]:
        """Return per-request pending-lookup deadlines, creating the map lazily."""
        deadlines = getattr(self, "_lookup_defer_deadlines", None)
        if deadlines is None:
            deadlines = self._lookup_defer_deadlines = {}
        return deadlines

    def _lookup_with_prefetch_metrics(
        self,
        lease_id: str,
        tokens: list[int],
        model_id: str,
        queries: int,
        num_computed_tokens: int,
    ) -> PrefetchLookupResult:
        """Run the atomic lookup/tier-classification scheduler RPC.

        Args:
            lease_id: Base vLLM request ID for an all-L1 lease.
            tokens: Token prefix sent to the DaseR lookup.
            model_id: Model identifier.
            queries: vLLM external-prefix query token count.
            num_computed_tokens: Tokens already resident in local vLLM KV.

        Returns:
            Validated lookup chunks, exact physical spans, and tier state.

        Thread-safety:
            Runs on the scheduler thread through its synchronous IPC client.
        """
        return self._ipc_sync.lookup_with_prefetch(
            lease_id,
            tokens,
            model_id,
            external_prefix_queries=queries,
            num_computed_tokens=num_computed_tokens,
        )

    def _layerwise_eligible(self, req_id: str, chunks: list[dict[str, Any]]) -> bool:
        """Return whether a hit can be restored layer by layer in-forward.

        Only requests leased in L1 whose records are either all packed or
        all raw qualify: the worker then reads each layer's bytes straight
        from L1. Raw chunks must not need a position shift.
        """
        if not getattr(self, "_layerwise_load", False):
            return False
        if req_id not in getattr(self, "_leased_request_ids", set()):
            return False
        if not chunks:
            return False
        if all(chunk.get("compressed_slots") for chunk in chunks):
            return all(
                slot.get("mode") == "compressed"
                for chunk in chunks
                for slot in chunk["compressed_slots"]
            )
        return not any(
            chunk.get("compressed_slots") or chunk.get("pos_offset", 0)
            for chunk in chunks
        )

    def _hand_off_layerwise_load(
        self,
        req_id: str,
        chunks: dict[str, Any],
        meta: DaserConnectorMeta,
    ) -> None:
        """Publish a synchronous layer-wise load and its lease to the worker.

        vLLM computes this request in the step it was scheduled, so every
        allocated chunk is emitted now and nothing stays pending; trimmed
        chunks were already dropped by ``update_state_after_alloc``.
        """
        del self._pending_loads[req_id]
        self._layerwise_request_ids.discard(req_id)
        pending = [chunks] if "chunk_key" in chunks else list(chunks.values())
        specs = [
            _load_spec_from_chunk(chunk) for chunk in pending if "block_ids" in chunk
        ]
        merged = _merge_adjacent_load_specs(specs, self._slot_size)
        for idx, spec in enumerate(merged):
            load_id = req_id if len(merged) == 1 else f"{req_id}:load:{idx}"
            meta.reqs_to_load[load_id] = spec
        # The worker releases the lease after its last layer; the scheduler
        # must not release it while the forward pass still reads L1.
        self._leased_request_ids.discard(req_id)
        meta.layerwise_req_ids.add(req_id)

    def _release_transfer_lease(self, req_id: str, *, force: bool = False) -> None:
        """Best-effort idempotent cleanup for scheduler-owned lease state."""
        leased_ids: set[str] = getattr(self, "_leased_request_ids", set())
        if not force and req_id not in leased_ids:
            return
        release = getattr(self._ipc_sync, "release_transfer_lease", None)
        try:
            if release is not None:
                release(req_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[CONNECTOR] release_transfer_lease failed req=%s: %s",
                req_id[:8],
                exc,
            )
        finally:
            leased_ids.discard(req_id)

    def _release_protection(self, req_id: str) -> None:
        """Release a request's queued-prefix protection, if it holds one."""
        protected = getattr(self, "_protected_request_ids", None)
        if not protected or req_id not in protected:
            return
        protected.discard(req_id)
        try:
            self._ipc_sync.release_protection(req_id)
        except Exception as exc:  # noqa: BLE001
            # The server releases leaked leases after a timeout.
            logger.debug(
                "[CONNECTOR] release protection failed req=%s: %s", req_id[:8], exc
            )

    def _cancel_prefetch_future(self, req_id: str) -> bool:
        """Cancel queued prefetch or release its lease after an active RPC exits."""
        future = getattr(self, "_prefetch_futures", {}).pop(req_id, None)
        if future is None:
            return False
        if not future.cancel():
            future.add_done_callback(
                lambda _future: self._release_transfer_lease(req_id, force=True)
            )
        return True

    def _record_external_prefix_cache_miss(self, queries: int) -> None:
        """Record a connector external-prefix miss when lookup is skipped.

        Args:
            queries: vLLM external prefix query token count.

        Returns:
            None.

        Thread-safety:
            Runs on the scheduler thread and uses the synchronous IPC client
            when it supports the diagnostic operation.
        """
        recorder = getattr(self._ipc_sync, "record_external_prefix_cache", None)
        if recorder is None:
            return
        recorder(max(0, int(queries)), 0)

    def _record_pending_store_blocks(self, req_id: str, block_ids: list[int]) -> None:
        """Record request KV blocks for a pending scheduler store.

        Args:
            req_id: vLLM request ID.
            block_ids: KV block IDs allocated for the request.
        """
        pending_store = self._pending_alloc.get(req_id)
        if pending_store is None:
            return
        requested_tokens = pending_store.token_count
        pending_store.block_ids = block_ids[
            : math.ceil(requested_tokens / self._block_tokens)
        ]
        self._maybe_allocate_pending_store(req_id, pending_store)

    def _refresh_runtime_config(self) -> None:
        """Refresh scheduler geometry and reuse policy over owned sync IPC."""
        try:
            config = self._ipc_sync.get_runtime_config()
        except Exception as exc:  # noqa: BLE001
            logger.info("[CONNECTOR] runtime config unavailable: %s", exc)
            return
        self._slot_size = int(config.get("slot_size", self._slot_size))
        self._tensor_parallel_size = int(config.get("tensor_parallel_size", 1))
        self._rank_stride_bytes = int(config.get("rank_stride_bytes", 0))
        server_storage_format = str(config.get("storage_format", STORAGE_FORMAT_RAW))
        if (
            getattr(self, "_declared_storage_format", None) is not None
            and server_storage_format != self._declared_storage_format
        ):
            raise ValueError(
                "connector storage_format does not match DaseR server: "
                f"{self._declared_storage_format} != {server_storage_format}"
            )
        block_tokens = int(config.get("block_tokens", self._block_tokens))
        self._model_id = str(config.get("model_id", self._model_id))
        self._protect_waiting = int(config.get("protect_waiting_bytes", 0)) > 0
        cache_reuse_mode = str(config.get("cache_reuse_mode", self._cache_reuse_mode))
        if (
            cache_reuse_mode != self._cache_reuse_mode
            or block_tokens != self._block_tokens
        ):
            self._cache_reuse_mode = cache_reuse_mode
            self._block_tokens = block_tokens
            self._cache_reuse_strategy = build_cache_reuse_strategy(
                cache_reuse_mode,
                self._block_tokens,
            )
        else:
            self._block_tokens = block_tokens
        self._runtime_config_ready = bool(self._slot_size)

    def _init_reuse_strategy(self) -> None:
        """Initialize the scheduler cache reuse strategy from current config."""
        self._cache_reuse_strategy = build_cache_reuse_strategy(
            "chunk",
            self._block_tokens,
        )

    def _reuse_strategy(self) -> Any:
        """Return the configured cache reuse strategy.

        Returns:
            Cache reuse strategy initialized from connector runtime config.
        """
        strategy = getattr(self, "_cache_reuse_strategy", None)
        if strategy is None:
            self._init_reuse_strategy()
            strategy = self._cache_reuse_strategy
        return strategy

    def allocate_store_chunk(
        self,
        chunk_key: str,
        token_count: int,
        parent_key: str = "",
    ) -> dict[str, Any]:
        """Allocate server metadata for a pending scheduler store.

        Args:
            chunk_key: cache key to allocate.
            token_count: number of tokens covered by the allocation.
            parent_key: previous chunk of the prefix chain, or empty.

        Returns:
            Mutable server allocation metadata.
        """
        return self._ipc_sync.alloc_chunk(
            chunk_key, token_count, self._model_id, parent_key
        )

    def allocate_store_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Allocate server metadata for multiple pending scheduler stores.

        Args:
            chunks: chunk descriptors with chunk_key, token_count and
                optional parent_key.

        Returns:
            Mutable server allocation metadata for each chunk.
        """
        alloc_chunks = getattr(self._ipc_sync, "alloc_chunks", None)
        if alloc_chunks is None:
            return [
                {
                    **self.allocate_store_chunk(
                        str(chunk["chunk_key"]),
                        int(chunk["token_count"]),
                        str(chunk.get("parent_key", "")),
                    ),
                    "chunk_key": str(chunk["chunk_key"]),
                }
                for chunk in chunks
            ]
        return alloc_chunks(chunks, self._model_id)

    def set_pending_store(self, req_id: str, alloc: dict[str, Any]) -> None:
        """Record an allocated store for later connector metadata packaging.

        Args:
            req_id: vLLM request ID or synthetic store work ID.
            alloc: mutable allocation metadata.
        """
        self._pending_stores[req_id] = alloc

    def has_pending_store(self, req_id: str) -> bool:
        """Return whether a pending store entry already exists.

        Args:
            req_id: vLLM request ID or synthetic store work ID.

        Returns:
            True when the store is already pending.
        """
        return req_id in self._pending_stores

    def drop_pending_alloc(self, req_id: str) -> None:
        """Remove pending allocation state for a request.

        Args:
            req_id: vLLM request ID.
        """
        self._pending_alloc.pop(req_id, None)
        getattr(self, "_prefetch_signatures", {}).pop(req_id, None)
        getattr(self, "_prefetch_lookup_results", {}).pop(req_id, None)
        self._release_transfer_lease(req_id)

    def _drop_pending_store(self, req_id: str) -> None:
        """Remove a pending store and release its server writer claim.

        Args:
            req_id: vLLM request ID or synthetic store work ID.
        """
        alloc = self._pending_stores.pop(req_id, None)
        if alloc is None:
            return
        release = getattr(self._ipc_sync, "release_chunk_writer", None)
        if release is None:
            return
        try:
            release(
                str(alloc["chunk_key"]),
                int(alloc["start_slot"]),
                int(alloc["num_slots"]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CONNECTOR] release_chunk_writer failed: %s", exc)

    def _discard_pending_request(self, req_id: str) -> None:
        """Clear scheduler-side pending state for a request.

        Args:
            req_id: vLLM request ID.
        """
        self._pending_loads.pop(req_id, None)
        getattr(self, "_retry_lookups", {}).pop(req_id, None)
        if req_id in self._pending_stores:
            self._drop_pending_store(req_id)
        for pending_req_id in list(self._pending_stores):
            if pending_req_id.startswith(f"{req_id}:store:"):
                self._drop_pending_store(pending_req_id)
        self._pending_alloc.pop(req_id, None)
        getattr(self, "_layerwise_request_ids", set()).discard(req_id)
        getattr(self, "_prefetch_signatures", {}).pop(req_id, None)
        getattr(self, "_prefetch_lookup_results", {}).pop(req_id, None)
        had_prefetch = self._cancel_prefetch_future(req_id)
        self._release_transfer_lease(req_id, force=had_prefetch)

    def _pending_async_save_ids(self) -> set[str]:
        """Return request IDs whose worker-side saves are still pending.

        Returns:
            Mutable set of base vLLM request IDs.

        Thread-safety:
            Runs on the scheduler thread. The lazy initialization supports
            tests and mixin probes that do not call ``DaserConnector.__init__``.
        """
        pending = getattr(self, "_pending_async_saves", None)
        if pending is None:
            pending = set()
            self._pending_async_saves = pending
        return pending

    def _maybe_allocate_pending_store(
        self, req_id: str, pending_store: PendingStore
    ) -> None:
        """Allocate a DaseR chunk once a pending store has full KV coverage.

        Args:
            req_id: vLLM request ID being tracked.
            pending_store: store tracker for the request.
        """
        requested_tokens = pending_store.token_count
        strategy = self._reuse_strategy()
        if not strategy.ready_to_allocate(pending_store):
            return
        tokens = self._req_tokens.get(req_id, [])
        if len(tokens) < requested_tokens:
            return
        plan = strategy.plan_store(
            req_id,
            pending_store,
            tokens,
            set(self._pending_stores),
        )
        if plan.invalid:
            self._pending_alloc.pop(req_id, None)
            return
        if plan.intents:
            try:
                if len(plan.intents) == 1 and plan.intents[0].req_id == req_id:
                    intent = plan.intents[0]
                    allocations = [
                        self.allocate_store_chunk(
                            intent.chunk_key,
                            intent.token_count,
                            intent.parent_key,
                        )
                    ]
                else:
                    allocations = self.allocate_store_chunks(
                        [
                            {
                                "chunk_key": intent.chunk_key,
                                "token_count": intent.token_count,
                                "parent_key": intent.parent_key,
                            }
                            for intent in plan.intents
                        ]
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[CONNECTOR] store allocation failed: %s", exc)
                return
            if len(allocations) != len(plan.intents):
                logger.warning(
                    "[CONNECTOR] allocation returned %d entries for %d intents",
                    len(allocations),
                    len(plan.intents),
                )
                return
            for intent, alloc in zip(plan.intents, allocations, strict=True):
                if bool(alloc.get("skipped", False)):
                    continue
                alloc["chunk_key"] = str(alloc.get("chunk_key", intent.chunk_key))
                alloc["token_count"] = intent.token_count
                alloc["num_slots"] = len(intent.block_ids)
                alloc["block_ids"] = intent.block_ids
                self._pending_stores[intent.req_id] = alloc
        pending_store.rolling_key = plan.next_key
        pending_store.rolling_slot_index = plan.next_slot
        if plan.complete:
            pending_store.chunk_key = plan.next_key
            self._pending_alloc.pop(req_id, None)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> "tuple[bool, dict[str, Any] | None]":
        """Clean up per-request state after inference completes.

        Args:
            request: finished vLLM Request.
            block_ids: block IDs being freed.

        Returns:
            (True, None) when DaseR is still storing this request's KV blocks,
            otherwise (False, None).
        """
        self._release_protection(request.request_id)
        self._defer_deadlines().pop(request.request_id, None)
        getattr(self, "_inflight_loads", {}).pop(request.request_id, None)
        getattr(self, "_coalesce_followers", {}).pop(request.request_id, None)
        getattr(self, "_follower_digests", {}).pop(request.request_id, None)
        if request.request_id in self._pending_async_save_ids():
            return True, None
        self._req_tokens.pop(request.request_id, None)
        self._discard_pending_request(request.request_id)
        return False, None

    def update_connector_output(self, connector_output: Any) -> None:
        """Update scheduler state from worker-side transfer completions.

        Args:
            connector_output: vLLM KVConnectorOutput carrying finished request
                IDs from workers.

        Async/thread-safety:
            Runs on vLLM's scheduler thread after worker connector polling.
        """
        pending_async_saves = self._pending_async_save_ids()
        for req_id in getattr(connector_output, "finished_sending", None) or ():
            pending_async_saves.discard(req_id)
            self._req_tokens.pop(req_id, None)
            self._discard_pending_request(req_id)
        inflight_loads = getattr(self, "_inflight_loads", {})
        for req_id in getattr(connector_output, "finished_recving", None) or ():
            self._trace_admit("loaded", req_id)
            self._release_protection(req_id)
            landed = inflight_loads.get(req_id)
            if landed is not None:
                inflight_loads[req_id] = (
                    landed[0],
                    landed[1],
                    COALESCE_LANDING_STEPS,
                )
            getattr(self, "_prefetch_lookup_results", {}).pop(req_id, None)
            self._release_transfer_lease(req_id)
            if req_id not in self._pending_alloc:
                self._req_tokens.pop(req_id, None)
            for pending_req_id in list(self._pending_loads):
                if _matches_request_or_store_id(pending_req_id, req_id):
                    self._pending_loads.pop(pending_req_id, None)

    def _wait_for_inflight_load(
        self,
        req_id: str,
        tokens: list[int],
        num_computed_tokens: int,
    ) -> bool:
        """Return whether a request should wait for another request's load.

        A request waits while another request is loading the same prompt
        prefix beyond this request's local hit; after that load lands, vLLM's
        prefix cache serves the prefix and this request loads nothing twice.

        Args:
            req_id: vLLM request ID asking for external matches.
            tokens: request prompt token IDs.
            num_computed_tokens: tokens already hit in vLLM's local cache.

        Returns:
            True when the caller should keep the request waiting this step.

        Async/thread-safety:
            Runs on vLLM's scheduler thread only.
        """
        now = time.monotonic()
        followers = self._coalesce_followers
        waiting = followers.get(req_id)
        if waiting is None or now - waiting[1] < COALESCE_MAX_WAIT_S:
            digests = self._follower_digests.setdefault(req_id, {})
            for leader_id, (end, digest, _) in self._inflight_loads.items():
                if leader_id == req_id or not num_computed_tokens < end < len(tokens):
                    continue
                if end not in digests:
                    digests[end] = _prefix_digest(tokens, end)
                if digests[end] != digest:
                    continue
                if waiting is None:
                    followers[req_id] = (num_computed_tokens, now)
                    self._coalesce_stats["followers"] += 1
                    if self._coalesce_stats["followers"] % 16 == 1:
                        self._log_coalesce_stats()
                return True
        if waiting is not None:
            followers.pop(req_id, None)
            outcome = "joined" if num_computed_tokens > waiting[0] else "missed"
            self._coalesce_stats[outcome] += 1
            if outcome == "missed" or self._coalesce_stats["joined"] % 16 == 1:
                self._log_coalesce_stats()
        self._follower_digests.pop(req_id, None)
        return False

    def _age_landed_loads(self) -> None:
        """Forget landed loads once vLLM has had a step to cache their blocks."""
        inflight_loads = getattr(self, "_inflight_loads", {})
        for req_id, (end, digest, steps) in list(inflight_loads.items()):
            if steps < 0:
                continue
            if steps == 0:
                inflight_loads.pop(req_id, None)
            else:
                inflight_loads[req_id] = (end, digest, steps - 1)

    def _log_coalesce_stats(self) -> None:
        """Log cumulative in-flight load coalescing counters."""
        stats = self._coalesce_stats
        logger.info(
            "[CONNECTOR] load coalescing: followers=%d joined=%d missed=%d inflight=%d",
            stats["followers"],
            stats["joined"],
            stats["missed"],
            len(self._inflight_loads),
        )
