# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
import os
import threading
import time
from typing import Any

import cupy
import torch

from daser import stalldump
from daser.config import (
    STORAGE_FORMAT_COMPRESSED_ONLINE,
    STORAGE_FORMAT_FRAGPACK,
    STORAGE_FORMAT_RAW,
)
from daser.connector.helpers import base_req_id
from daser.connector.ipc_client import IPCClientAsync
from daser.connector.metadata import ReqStoreSpec, StoreWriteSpan
from daser.connector.worker.memory import (
    DEFAULT_STORE_STAGING_BYTES,
    CudaStagingLease,
    FixedCudaStagingPool,
)
from daser.connector.worker.probe import SaveTimeline, StoreProbe
from daser.connector.worker.staging import (
    CROSS_LAYER_KV_CACHE_KEY,
    copy_cross_layer_kv_cache_to_staging,
    record_cuda_event,
)
from daser.logging import init_logger
from daser.ops.compressed_kv import (
    FusedOnlineKVPacker,
    OnlinePackedSlot,
)
from daser.ops.fragpack_codec import FragPackBlockExporter
from daser.transfer.cuda_ipc import (
    cuda_allocation_base_and_offset,
    cuda_array_pci_bus_id,
    cuda_array_pointer,
    export_cuda_ipc_handle,
)

logger = init_logger(__name__)

# Bound each online codec launch so a large first-turn prefix yields to vLLM
# between batches while still amortizing fixed metadata and kernel overhead.
# A batch is also capped at one staging lease, so this value changes neither
# the persisted slot geometry nor transfer ownership.
DEFAULT_ONLINE_PACK_BATCH_SLOTS = 85


def _merge_store_requests(
    saves: Sequence[_DeferredFinishedSave],
) -> dict[str, ReqStoreSpec]:
    """Merge independent deferred saves without changing request order."""
    merged: dict[str, ReqStoreSpec] = {}
    for save in saves:
        for req_id, spec in save.reqs_to_store.items():
            if req_id in merged and merged[req_id] != spec:
                raise RuntimeError(f"conflicting deferred store request: {req_id}")
            merged[req_id] = spec
    return merged


class _TicketGate:
    """Admit numbered tickets strictly in increasing order.

    A ticket's holder awaits ``wait(ticket)`` and calls ``done(ticket)`` when
    its turn ends; ``done`` may also retire a ticket that never waited (its
    work was skipped) and is idempotent.

    Async/thread-safety:
        Used only on the store event loop.
    """

    def __init__(self) -> None:
        self._next = 0
        self._retired: set[int] = set()
        self._waiters: dict[int, asyncio.Event] = {}

    async def wait(self, ticket: int) -> None:
        """Return once every lower ticket is done."""
        if ticket <= self._next:
            return
        event = self._waiters.setdefault(ticket, asyncio.Event())
        await event.wait()

    def done(self, ticket: int) -> None:
        """Retire ``ticket`` and wake the next ticket once it is admissible."""
        if ticket < self._next:
            return
        self._retired.add(ticket)
        while self._next in self._retired:
            self._retired.remove(self._next)
            self._next += 1
        event = self._waiters.pop(self._next, None)
        if event is not None:
            event.set()


@dataclass
class _DeferredFinishedSave:
    """Hold request store work until vLLM reports it finished.

    ``first_ticket`` numbers the first of ``planned_batches``, the staging
    batches fixed at submission, when stores are ordered (packed records with
    TP > 1); otherwise both are ``None``.
    """

    reqs_to_store: dict[str, ReqStoreSpec]
    finished: bool = False
    future: Any | None = None
    producer_event: torch.cuda.Event | None = None
    producer_event_order: int = 0
    snapshot_done: bool = False
    completion_emitted: bool = False
    timeline: SaveTimeline = field(default_factory=SaveTimeline)
    first_ticket: int | None = None
    planned_batches: list[tuple[list[int], list[StoreWriteSpan]]] | None = None


class StorePipeline:
    """Own the complete worker store state machine.

    Args:
        socket_path: DaseR server Unix socket path.
        online_pack_batch_slots: Maximum raw slots encoded by one
            compressed-online codec launch. Ignored by raw storage.

    Raises:
        ValueError: If ``online_pack_batch_slots`` is not positive.

    Async/thread-safety:
        Public methods are called on the vLLM worker thread. Snapshot, IPC, and
        commit execute on the private store thread and its fixed CUDA stream.
        With packed records and TP > 1 the server places a chunk only after
        every rank declared it, so each rank must transfer chunks in the same
        order: batches are numbered at submission (in the scheduler's order,
        identical on every rank) and both staging leases and transfers are
        granted in that order.
    """

    def __init__(
        self,
        socket_path: str,
        online_pack_batch_slots: int = DEFAULT_ONLINE_PACK_BATCH_SLOTS,
    ) -> None:
        if online_pack_batch_slots <= 0:
            raise ValueError("online_pack_batch_slots must be positive")
        self._online_pack_batch_slots = int(online_pack_batch_slots)
        self._client = IPCClientAsync(socket_path)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="daser-store-io",
        )
        self._staging_bytes = 0
        self._staging_pool: FixedCudaStagingPool | None = None
        self._staging_buffer_indices: dict[int, int] = {}
        self._pending_finished_saves: dict[str, _DeferredFinishedSave] = {}
        self._pending_writer_releases: list[Future[None]] = []
        self._store_capacity = 1
        self._store_semaphore: asyncio.Semaphore | None = None
        self._staging_lease_semaphore: asyncio.Semaphore | None = None
        self._next_ticket = 0
        self._lease_gate: _TicketGate | None = None
        self._transfer_gate: _TicketGate | None = None
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._local_slot_size = 0
        self._tp_rank = 0
        self._tp_size = 1
        self._cuda_stream: torch.cuda.Stream | None = None
        # ``FusedOnlineKVPacker`` owns reusable host/device metadata buffers.
        # Keep staging launches serialized while allowing the asyncio store
        # loop to progress transfers for an already-completed batch.
        self._stage_lock = threading.Lock()
        self._storage_format = STORAGE_FORMAT_RAW
        # Packed formats: the online codec, or the FragPack block exporter (same
        # ``pack_into`` contract). None stores raw slots.
        self._online_packer: FusedOnlineKVPacker | FragPackBlockExporter | None = None
        self._producer_event_order = 0
        self._probe = StoreProbe.from_env()
        self._thread.start()
        stalldump.register_loop("worker-store", self._loop)
        stalldump.register_state(
            "worker-store",
            lambda: {
                req_id: (
                    save.finished,
                    save.future is not None and save.future.done(),
                    save.snapshot_done,
                    save.completion_emitted,
                )
                for req_id, save in dict(self._pending_finished_saves).items()
            },
        )

    def configure(
        self,
        *,
        kv_caches: dict[str, torch.Tensor],
        layer_names: list[str],
        local_slot_size: int,
        tp_rank: int,
        tp_size: int,
        staging_bytes: int,
        staging_pool: FixedCudaStagingPool,
    ) -> None:
        """Configure immutable KV layout and staging state.

        Args:
            kv_caches: Registered vLLM KV tensors.
            layer_names: Stable storage layer order.
            local_slot_size: Bytes stored per slot by this TP rank.
            tp_rank: Current tensor-parallel rank.
            tp_size: Tensor-parallel world size.
            staging_bytes: Maximum bytes per store batch.
            staging_pool: Fixed store staging buffers.

        Async/thread-safety:
            Called once on the worker thread before request traffic.
        """
        self._kv_caches = kv_caches
        self._layer_names = list(layer_names)
        self._local_slot_size = local_slot_size
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        self._staging_bytes = staging_bytes
        self._staging_pool = staging_pool
        self._staging_buffer_indices.clear()
        self._store_capacity = staging_pool.depth
        self._store_semaphore = None
        self._staging_lease_semaphore = None
        self._next_ticket = 0
        self._lease_gate = None
        self._transfer_gate = None

    @property
    def max_online_pack_slots(self) -> int:
        """Return the largest raw slot count encoded by one codec launch.

        Returns:
            ``min(online_pack_batch_slots, slots per store buffer)``, or ``0``
            before staging is configured.
        """
        if self._staging_pool is None or self._local_slot_size <= 0:
            return 0
        buffer_slots = max(1, self._staging_pool.buffer_bytes // self._local_slot_size)
        return min(buffer_slots, self._online_pack_batch_slots)

    def configure_compression(
        self,
        *,
        storage_format: str,
        codebooks: bytes,
        tile_scalars: int,
    ) -> None:
        """Bind the store-side packer of the server's storage format.

        Args:
            storage_format: ``raw``, ``compressed-online`` (the startup-warmed
                online packer bound to ``codebooks``) or ``fragpack`` (block blob
                exporter; codebooks unused).
            codebooks: Server-owned plane-major codebook bytes.
            tile_scalars: Codec tile quantum.

        Async/thread-safety:
            Called once on the worker thread before store submissions. Kernel
            construction only reuses the registration-time warm cache.
        """
        self._storage_format = storage_format
        self._online_packer = None
        if storage_format == STORAGE_FORMAT_FRAGPACK:
            if self._staging_pool is None:
                raise ValueError("fragpack storage requires configured store staging")
            self._online_packer = FragPackBlockExporter(self._local_slot_size)
            logger.info(
                "[CONNECTOR] fragpack export configured batch_slots=%d slot_size=%d",
                self.max_online_pack_slots,
                self._local_slot_size,
            )
            return
        if storage_format != STORAGE_FORMAT_COMPRESSED_ONLINE:
            if storage_format != STORAGE_FORMAT_RAW:
                raise ValueError(f"unknown storage format: {storage_format}")
            return
        if self._staging_pool is None or len(self._kv_caches) != 1:
            raise ValueError("compressed-online requires cross-layer KV staging")
        kv_cache = next(iter(self._kv_caches.values()))
        self._online_packer = FusedOnlineKVPacker(
            kv_cache=kv_cache,
            codebooks=codebooks,
            tile_scalars=tile_scalars,
            max_slots_per_buffer=self.max_online_pack_slots,
        )
        logger.info(
            "[CONNECTOR] online pack configured batch_slots=%d "
            "slot_size=%d tile_scalars=%d",
            self.max_online_pack_slots,
            self._local_slot_size,
            tile_scalars,
        )

    def initialize_transfer(self) -> None:
        """Initialize the store IPC client and register fixed CUDA buffers.

        Async/thread-safety:
            Called on the worker thread during startup and waits only for the
            store loop's initialization future.
        """
        self._submit(self._client.init_transfer()).result(timeout=120.0)
        self._register_staging_buffers()

    def configure_rank_geometry(self, tp_rank: int, tp_size: int) -> None:
        """Apply the server-finalized tensor-parallel rank.

        Args:
            tp_rank: Current tensor-parallel rank.
            tp_size: Tensor-parallel world size used for commit coordination.

        Async/thread-safety:
            Called on the worker thread after runtime-config refresh and before
            any store is submitted.
        """
        self._tp_rank = tp_rank
        self._tp_size = tp_size

    def queue_finished(
        self,
        reqs_to_store: dict[str, ReqStoreSpec],
    ) -> None:
        """Queue stores until vLLM reports their requests finished.

        Args:
            reqs_to_store: Store metadata for the current worker step.
        Async/thread-safety:
            Called on the worker thread to accumulate immutable store intent.
            In packed mode, the event is recorded at this point so deferred
            packing waits only for the current forward pass. Raw mode keeps
            the legacy late event capture in ``_submit_save``.
        """
        producer_event: torch.cuda.Event | None = None
        if getattr(self, "_online_packer", None) is not None:
            sample = next(iter(getattr(self, "_kv_caches", {}).values()), None)
            if sample is not None:
                producer_event = record_cuda_event(sample)
        producer_event_order = 0
        if producer_event is not None:
            producer_event_order = getattr(self, "_producer_event_order", 0) + 1
            self._producer_event_order = producer_event_order
        for req_id, spec in reqs_to_store.items():
            base_id = base_req_id(req_id)
            save = self._pending_finished_saves.get(base_id)
            if save is None:
                save = _DeferredFinishedSave(
                    {},
                    producer_event=producer_event,
                    producer_event_order=producer_event_order,
                )
                self._pending_finished_saves[base_id] = save
            elif producer_event is not None:
                # A request may publish multiple chunks over several worker
                # steps. The newest event orders every earlier publication on
                # the same producer stream while covering the latest chunk.
                save.producer_event = producer_event
                save.producer_event_order = producer_event_order
            save.reqs_to_store[req_id] = spec

    def cancel_pending(self, req_ids: set[str]) -> None:
        """Discard preempted, unsent stores and asynchronously release writers.

        Args:
            req_ids: Base request IDs canceled by scheduler metadata.
        Returns:
            None. Missing or already submitted saves are left alone.
        Async/thread-safety:
            Called on the worker thread before new step stores are queued.
            Unsubmitted saves cannot be reading KV on the private IO loop.
            Release RPC futures have no model-block completion semantics and
            are reaped separately by collect_finished or shutdown.
        """
        for req_id in req_ids:
            save = self._pending_finished_saves.get(req_id)
            if save is None or save.future is not None:
                continue
            del self._pending_finished_saves[req_id]
            self._pending_writer_releases.append(
                self._submit(
                    self._release_writer_claims(tuple(save.reqs_to_store.values()))
                )
            )

    def collect_finished(self, finished_req_ids: set[str]) -> set[str]:
        """Submit newly finished stores and collect releasable requests.

        Args:
            finished_req_ids: Requests vLLM finished in this step.

        Returns:
            Request IDs whose vLLM KV snapshot is complete (packed mode) or
            whose full store and commit lifecycle has completed (raw mode).

        Async/thread-safety:
            Called on the worker thread. Store work runs on the private loop.
        """
        self._reap_writer_releases()
        finished: set[str] = set()
        packed_mode = getattr(self, "_online_packer", None) is not None
        for req_id in finished_req_ids:
            save = self._pending_finished_saves.get(req_id)
            if save is not None:
                save.finished = True
                self._stamp((save,), "finished", keep_first=True)

        for req_id, save in list(self._pending_finished_saves.items()):
            # Packed mode has a deliberate two-phase completion contract:
            # snapshot_done means every source block is safely copied/encoded
            # into a leased staging buffer, while ``future`` remains pending
            # until server transfer and commit finish.  This lets vLLM release
            # the original KV blocks without exposing mutable source memory to
            # the asynchronous IPC transfer.
            if packed_mode and save.snapshot_done and not save.completion_emitted:
                save.completion_emitted = True
                finished.add(req_id)
                self._record_emitted(save)

            if save.future is not None and save.future.done():
                try:
                    save.future.result(timeout=120.0)
                finally:
                    del self._pending_finished_saves[req_id]
                # A raw save is only releasable after its full future.  Packed
                # saves may already have emitted the snapshot completion on a
                # previous poll, so no duplicate request ID is returned here.
                if not packed_mode:
                    finished.add(req_id)
                    self._record_emitted(save)

        # Submit the entire finished FIFO after observing the current state.
        # This preserves the raw path's one-poll dispatch boundary while the
        # private event loop applies the staging-depth bound for queued saves.
        ready_saves = [
            save
            for save in self._pending_finished_saves.values()
            if save.finished and save.future is None
        ]
        self._stamp(ready_saves, "submitted")
        if packed_mode:
            for group in self._group_packed_saves(ready_saves):
                self._submit_packed_save_group(group)
        else:
            for save in ready_saves:
                self._submit_save(save)
        return finished

    def shutdown(self) -> None:
        """Finish queued stores, close IPC, and stop the store loop.

        Async/thread-safety:
            Called once on the worker thread after request traffic stops.
        """
        first_error: BaseException | None = None
        try:
            for future in self._pending_writer_releases:
                try:
                    future.result(timeout=120.0)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
            self._pending_writer_releases.clear()
            submitted_ids = [
                req_id
                for req_id, save in self._pending_finished_saves.items()
                if save.future is not None
            ]
            for req_id in submitted_ids:
                save = self._pending_finished_saves[req_id]
                try:
                    if save.future is not None:
                        save.future.result(timeout=120.0)
                except BaseException as exc:  # preserve cleanup during shutdown
                    if first_error is None:
                        first_error = exc
                finally:
                    del self._pending_finished_saves[req_id]
            for req_id in list(self._pending_finished_saves):
                save = self._pending_finished_saves[req_id]
                try:
                    self._submit_save(save)
                    assert save.future is not None
                    save.future.result(timeout=120.0)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                finally:
                    del self._pending_finished_saves[req_id]
            try:
                self._submit(self._client.close()).result(timeout=5.0)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)
        self._cuda_stream = None
        if first_error is not None:
            raise first_error

    def _submit(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _release_writer_claims(self, specs: tuple[ReqStoreSpec, ...]) -> None:
        """Attempt every canceled allocation release and propagate failures."""
        results = await asyncio.gather(
            *(
                self._client.release_chunk_writer(
                    spec.chunk_key, spec.start_slot, spec.num_slots
                )
                for spec in specs
            ),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    def _reap_writer_releases(self) -> None:
        """Observe completed releases without returning request completions."""
        completed: list[Future[None]] = []
        pending: list[Future[None]] = []
        for future in self._pending_writer_releases:
            (completed if future.done() else pending).append(future)
        self._pending_writer_releases = pending
        first_error: BaseException | None = None
        for future in completed:
            try:
                future.result()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _stamp(
        self,
        saves: Sequence[_DeferredFinishedSave],
        point: str,
        *,
        keep_first: bool = False,
    ) -> None:
        """Set timeline ``point`` on ``saves`` when the store probe is on."""
        if getattr(self, "_probe", None) is None:
            return
        now = time.monotonic()
        for save in saves:
            if not (keep_first and getattr(save.timeline, point)):
                setattr(save.timeline, point, now)

    def _record_emitted(self, save: _DeferredFinishedSave) -> None:
        """Report a save whose blocks are being released to the probe."""
        probe = getattr(self, "_probe", None)
        if probe is not None:
            probe.on_emitted(save.timeline, time.monotonic())

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit_save(self, save: _DeferredFinishedSave) -> None:
        """Capture producer ordering and enqueue one save in FIFO order."""
        producer_event = save.producer_event
        if producer_event is None:
            # Preserve raw behavior and lightweight test probes that
            # do not expose a packed producer event.
            sample = next(iter(self._kv_caches.values()), None)
            producer_event = record_cuda_event(sample) if sample is not None else None
        self._reserve_tickets(save)
        save.future = self._submit(self._run_bounded_save(save, producer_event))

    def _group_packed_saves(
        self,
        saves: list[_DeferredFinishedSave],
    ) -> list[tuple[_DeferredFinishedSave, ...]]:
        """Greedily combine packed saves that fit in one staging batch."""
        groups: list[tuple[_DeferredFinishedSave, ...]] = []
        current: list[_DeferredFinishedSave] = []
        for save in saves:
            if not current:
                current.append(save)
                continue
            candidate = [*current, save]
            candidate_save = _DeferredFinishedSave(
                reqs_to_store=_merge_store_requests(candidate)
            )
            if len(self._plan_finished_save(candidate_save)) <= 1:
                current.append(save)
                continue
            groups.append(tuple(current))
            current = [save]
        if current:
            groups.append(tuple(current))
        return groups

    def _submit_packed_save_group(
        self,
        saves: tuple[_DeferredFinishedSave, ...],
    ) -> None:
        """Submit one packed staging operation and fan completion to members."""
        if not saves:
            return
        if len(saves) == 1:
            self._submit_save(saves[0])
            return

        merged = _DeferredFinishedSave(
            reqs_to_store=_merge_store_requests(saves),
            finished=True,
        )
        producer = max(saves, key=lambda save: save.producer_event_order)
        producer_event = producer.producer_event
        member_futures: tuple[Future[None], ...] = tuple(Future() for _save in saves)
        for save, member_future in zip(saves, member_futures, strict=True):
            save.future = member_future
        self._reserve_tickets(merged)

        group_future = self._submit(
            self._run_bounded_save_group(merged, producer_event, saves)
        )

        def complete_members(completed: Any) -> None:
            try:
                completed.result()
            except BaseException as exc:
                for member_future in member_futures:
                    member_future.set_exception(exc)
            else:
                for member_future in member_futures:
                    member_future.set_result(None)

        group_future.add_done_callback(complete_members)

    async def _run_bounded_save(
        self,
        save: _DeferredFinishedSave,
        producer_event: torch.cuda.Event | None,
    ) -> None:
        """Run one save while preserving FIFO staging-buffer admission.

        Args:
            save: Finished request save to transfer.
            producer_event: CUDA event ordering the KV snapshot.

        Async/thread-safety:
            Runs on the private store event loop. The semaphore prevents more
            saves from entering synchronous staging acquisition than there are
            preallocated buffers; semaphore waiters are admitted in FIFO order.
        """
        await self._run_admitted(save, producer_event, (save,))

    async def _run_bounded_save_group(
        self,
        merged: _DeferredFinishedSave,
        producer_event: torch.cuda.Event | None,
        members: tuple[_DeferredFinishedSave, ...],
    ) -> None:
        """Run one cross-request packed save under normal store admission."""
        await self._run_admitted(merged, producer_event, members)

    async def _run_admitted(
        self,
        save: _DeferredFinishedSave,
        producer_event: torch.cuda.Event | None,
        members: tuple[_DeferredFinishedSave, ...],
    ) -> None:
        """Admit and run one save; a failed store leaves its chunks uncached.

        Args:
            save: Save (or merged group save) to transfer.
            producer_event: CUDA event ordering the KV snapshot.
            members: Request saves whose completion this save reports.

        Async/thread-safety:
            Runs on the private store event loop. Unordered saves pass a FIFO
            semaphore of the staging depth; ordered saves are bounded by the
            ticket-ordered lease grant instead, since a semaphore would admit
            them out of order. A store failure is logged, the writer claims
            are released and the source blocks are reported free, so one
            failed chunk never stops the engine.
        """
        first_ticket = getattr(save, "first_ticket", None)

        async def run() -> None:
            self._stamp(members, "admitted")
            if members == (save,):
                await self._store_finished_save(save, producer_event)
            else:
                await self._store_finished_save(
                    save, producer_event, snapshot_targets=members
                )

        try:
            if first_ticket is None:
                semaphore = self._store_semaphore
                if semaphore is None:
                    semaphore = asyncio.Semaphore(max(1, self._store_capacity))
                    self._store_semaphore = semaphore
                async with semaphore:
                    await run()
            else:
                await run()
        except Exception as exc:  # noqa: BLE001
            await self._store_failed(save, members, exc)
        finally:
            if first_ticket is not None:
                # Retire the tickets of batches that never ran (failure).
                lease_gate, transfer_gate = self._gates()
                for index in range(len(save.planned_batches or ())):
                    lease_gate.done(first_ticket + index)
                    transfer_gate.done(first_ticket + index)

    def _plan_finished_save(
        self,
        save: _DeferredFinishedSave,
    ) -> list[tuple[list[int], list[StoreWriteSpan]]]:
        if not self._kv_caches or self._staging_pool is None:
            return []
        max_batch_bytes = self._staging_bytes
        if getattr(self, "_online_packer", None) is not None:
            # A large first-turn prefix can contain 16+ blocks. Compressing
            # the whole prefix in one CUDA launch monopolizes memory
            # bandwidth while other conversations are prefilling. Small
            # online batches let vLLM work between codec launches; each batch
            # still follows the same server-owned span and publication path.
            max_batch_bytes = min(
                max_batch_bytes,
                self._online_pack_batch_slots * self._local_slot_size,
            )
        return build_staging_store_batches(
            save.reqs_to_store,
            self._local_slot_size,
            max_batch_bytes=max_batch_bytes,
        )

    async def _store_finished_save(
        self,
        save: _DeferredFinishedSave,
        producer_event: torch.cuda.Event | None,
        *,
        snapshot_targets: tuple[_DeferredFinishedSave, ...] | None = None,
    ) -> None:
        batches = getattr(save, "planned_batches", None)
        if batches is None:
            batches = self._plan_finished_save(save)
        snapshot_saves = snapshot_targets or (save,)
        if not batches:
            self._stamp(snapshot_saves, "ready")
            if getattr(self, "_online_packer", None) is not None:
                for snapshot_save in snapshot_saves:
                    snapshot_save.snapshot_done = True
            return

        packed_mode = getattr(self, "_online_packer", None) is not None

        # Raw mode intentionally keeps the master completion barrier: each
        # staging lease is returned immediately after its transfer call.  The
        # packed path below is the only mode that decouples source snapshot
        # completion from server transfer completion.
        if not packed_mode:
            for block_ids, spans in batches:
                staged = await self._stage_batch_acquired(
                    block_ids, spans, producer_event
                )
                try:
                    await self._write_staged(staged)
                finally:
                    self._release_staged(staged)
            self._stamp((save,), "ready")
            return

        overlap_batches = (
            len(batches) > 1 and int(getattr(self, "_store_capacity", 1)) > 1
        )
        # The staging pool has two independent leases in the production
        # configuration. Start the next (single producer-locked) pack as soon
        # as the current staging snapshot is ready so its CUDA work overlaps
        # the server's asynchronous L1/L2 admission. The current lease remains
        # live until transfer completion, and the next stage task is drained on
        # errors before its lease can be reclaimed.
        transfer_tasks: list[asyncio.Task[list[str]]] = []
        pending_stage: asyncio.Task[StagedStoreBatch] | None = None
        first_ticket = getattr(save, "first_ticket", None)

        def stage(index: int) -> Any:
            block_ids, spans = batches[index]
            if first_ticket is None:
                return self._stage_batch_acquired(block_ids, spans, producer_event)
            return self._stage_batch_acquired(
                block_ids, spans, producer_event, ticket=first_ticket + index
            )

        def transfer(index: int, staged: StagedStoreBatch) -> Any:
            if first_ticket is None:
                return self._transfer_staged(staged)
            return self._transfer_staged(staged, ticket=first_ticket + index)

        try:
            if overlap_batches:
                pending_stage = asyncio.create_task(stage(0))
            for index in range(len(batches)):
                if pending_stage is None:
                    staged = await stage(index)
                else:
                    staged = await pending_stage
                    pending_stage = None
                if overlap_batches and index + 1 < len(batches):
                    pending_stage = asyncio.create_task(stage(index + 1))
                # The lease is intentionally captured by the transfer task.
                # Its release occurs only after the server has consumed the
                # CUDA IPC mapping and completed L1/L2 write plus commit.
                transfer_tasks.append(asyncio.create_task(transfer(index, staged)))

            # All source blocks are now immutable snapshots in worker-owned
            # staging leases.  ``collect_finished`` can release vLLM blocks
            # while transfer tasks continue in the background.
            self._stamp(snapshot_saves, "ready")
            for snapshot_save in snapshot_saves:
                snapshot_save.snapshot_done = True

            results = await asyncio.gather(*transfer_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        except BaseException:
            if pending_stage is not None:
                await self._drain_staged_task(pending_stage)
            if transfer_tasks:
                await asyncio.gather(*transfer_tasks, return_exceptions=True)
            raise

    async def _transfer_staged(
        self, staged: "StagedStoreBatch", *, ticket: int | None = None
    ) -> list[str]:
        """Transfer one packed snapshot and release its staging lease.

        Args:
            staged: Snapshot whose lease must remain valid for the full IPC
                transfer and server-side commit.
            ticket: Batch number of an ordered store: the transfer starts
                after every lower-numbered transfer finished.

        Returns:
            Chunk keys accepted by the server transfer layer.

        Async/thread-safety:
            Runs on the private store event loop.  The lease is released in a
            ``finally`` block so transfer failures cannot exhaust the fixed
            staging pool or leave pinned memory retained indefinitely.
        """
        try:
            if ticket is None:
                return await self._write_staged(staged)
            gate = self._gates()[1]
            await gate.wait(ticket)
            try:
                return await self._write_staged(staged)
            finally:
                gate.done(ticket)
        finally:
            self._release_staged(staged)

    async def _stage_batch_acquired(
        self,
        block_ids: list[int],
        spans: list[StoreWriteSpan],
        producer_event: torch.cuda.Event | None,
        *,
        ticket: int | None = None,
    ) -> "StagedStoreBatch":
        """Acquire a bounded staging lease and snapshot one store batch.

        Args:
            block_ids: Physical vLLM blocks in logical order.
            spans: Server-owned write spans for this batch.
            producer_event: CUDA event ordering the live KV snapshot.
            ticket: Batch number of an ordered store: the lease is granted
                after every lower-numbered batch got its lease.

        Returns:
            A staged batch that owns one lease-semaphore permit until transfer
            completion.

        Async/thread-safety:
            Runs on the private store event loop.  CUDA staging itself is
            delegated to an executor thread and remains serialized by the
            packer lock; the permit bounds all in-flight leases globally.
        """
        lease_semaphore = getattr(self, "_staging_lease_semaphore", None)
        if lease_semaphore is None:
            pool = self._staging_pool
            depth = pool.depth if pool is not None else self._store_capacity
            lease_semaphore = asyncio.Semaphore(max(1, depth))
            self._staging_lease_semaphore = lease_semaphore
        wait_started = time.monotonic()
        if ticket is None:
            await lease_semaphore.acquire()
        else:
            gate = self._gates()[0]
            await gate.wait(ticket)
            try:
                await lease_semaphore.acquire()
            finally:
                gate.done(ticket)
        stage_started = time.monotonic()
        try:
            staged = await asyncio.to_thread(
                self._stage_batch_serialized,
                block_ids,
                spans,
                producer_event,
            )
        except BaseException:
            lease_semaphore.release()
            raise
        return StagedStoreBatch(
            buffer=staged.buffer,
            spans=staged.spans,
            lease=staged.lease,
            lease_permit=lease_semaphore,
            lease_wait_s=stage_started - wait_started,
            stage_s=time.monotonic() - stage_started,
        )

    async def _drain_staged_task(
        self,
        task: asyncio.Task["StagedStoreBatch"],
    ) -> None:
        """Finish and release a prefetched staging task during error cleanup."""
        try:
            staged = await asyncio.shield(task)
        except BaseException:
            return
        self._release_staged(staged)

    def _release_staged(self, staged: "StagedStoreBatch") -> None:
        """Release a staged buffer and its global lease permit exactly once."""
        stage_lock = getattr(self, "_stage_lock", None)
        if stage_lock is None:
            staged.lease.release()
        else:
            with stage_lock:
                staged.lease.release()
        if staged.lease_permit is not None:
            staged.lease_permit.release()

    def _stage_batch_serialized(
        self,
        block_ids: list[int],
        spans: list[StoreWriteSpan],
        producer_event: torch.cuda.Event | None,
    ) -> "StagedStoreBatch":
        """Stage one batch off the store event-loop thread.

        Args:
            block_ids: Physical vLLM blocks in logical order.
            spans: Server-owned write spans for this batch.
            producer_event: CUDA event ordering the live KV snapshot.

        Returns:
            A leased staging batch whose bytes are ready for IPC transfer.

        Async/thread-safety:
            Runs in an executor thread.  The lock protects reusable packer
            metadata buffers; CUDA stream ordering still protects the staging
            bytes before ``_write_cuda_buffer`` is called.
        """
        stage_lock = getattr(self, "_stage_lock", None)
        if stage_lock is None:
            # Lightweight tests may construct the pipeline with ``__new__``
            # and replace staging entirely; preserve that public helper
            # contract without requiring the full runtime initializer.
            return self._stage_batch(block_ids, spans, producer_event)
        with stage_lock:
            return self._stage_batch(block_ids, spans, producer_event)

    def _stage_batch(
        self,
        block_ids: list[int],
        spans: list[StoreWriteSpan],
        producer_event: torch.cuda.Event | None,
    ) -> "StagedStoreBatch":
        if self._staging_pool is None:
            raise RuntimeError("store staging pool is not configured")
        sample = next(iter(self._kv_caches.values()))
        nbytes = len(block_ids) * self._local_slot_size
        lease = self._staging_pool.acquire(nbytes)
        staging = lease.view
        stream = self._cuda_stream
        packed_mode = self._online_packer is not None
        if sample.device.type == "cuda" and stream is None:
            torch.cuda.set_device(sample.device)
            stream = torch.cuda.Stream(device=sample.device)
            self._cuda_stream = stream
        stage_started = time.perf_counter()
        pack_ms = 0.0
        if stream is not None:
            if producer_event is not None:
                stream.wait_event(producer_event)
            with torch.cuda.stream(stream):
                if self._online_packer is not None:
                    pack_started = time.perf_counter()
                    records = self._online_packer.pack_into(
                        staging=staging,
                        block_ids=block_ids,
                        logical_slots=_logical_slots_for_batch(block_ids, spans),
                        slot_stride=self._local_slot_size,
                        stream=stream,
                    )
                    if sum(record.stored_length for record in records) > nbytes:
                        raise RuntimeError(
                            "online packed staging layout exceeds capacity"
                        )
                    pack_ms = (time.perf_counter() - pack_started) * 1000
                    spans = _packed_store_spans(
                        spans,
                        records,
                        self._local_slot_size,
                    )
                else:
                    self._copy_blocks(staging, block_ids, sample)
            sync_started = time.perf_counter()
            stream.synchronize()
            sync_ms = (time.perf_counter() - sync_started) * 1000
        else:
            self._copy_blocks(staging, block_ids, sample)
            sync_ms = 0.0
        logger.debug(
            "[CONNECTOR] store stage: slots=%d bytes=%d packed=%s total_ms=%.3f "
            "pack_ms=%.3f sync_ms=%.3f",
            len(block_ids),
            nbytes,
            packed_mode,
            (time.perf_counter() - stage_started) * 1000,
            pack_ms,
            sync_ms,
        )
        return StagedStoreBatch(staging, spans, lease)

    def _copy_blocks(
        self,
        staging: torch.Tensor,
        block_ids: list[int],
        sample: torch.Tensor,
    ) -> None:
        block_index = torch.tensor(block_ids, dtype=torch.long, device=sample.device)
        copy_cross_layer_kv_cache_to_staging(
            staging,
            self._kv_caches[CROSS_LAYER_KV_CACHE_KEY],
            block_ids,
            len(self._layer_names),
            self._local_slot_size,
            block_index,
        )

    async def _write_staged(self, staged: "StagedStoreBatch") -> list[str]:
        """Write one staged batch and report its phases to the store probe."""
        started = time.monotonic()
        chunk_keys = await self._write_cuda_buffer(staged)
        probe = getattr(self, "_probe", None)
        if probe is not None:
            probe.on_batch(
                staged.lease_wait_s,
                staged.stage_s,
                time.monotonic() - started,
                int(staged.buffer.numel() * staged.buffer.element_size()),
            )
        return chunk_keys

    async def _write_cuda_buffer(self, staged: "StagedStoreBatch") -> list[str]:
        transfer_started = time.perf_counter()
        slot_size = getattr(self, "_local_slot_size", 0)
        raw_equivalent_bytes = (
            sum(span.num_slots * slot_size for span in staged.spans)
            if slot_size > 0
            else staged.buffer.nbytes
        )
        packed_bytes = _store_payload_nbytes(staged.buffer.nbytes, staged.spans)
        if staged.buffer.device.type == "cuda":
            torch.cuda.set_device(staged.buffer.device)
        buffer_index = self._staging_buffer_indices.get(staged.lease.tensor.data_ptr())
        spans = [
            {
                "source_offset": span.source_offset,
                "nbytes": span.nbytes,
                "chunk_key": span.chunk_key,
                "start_slot": span.start_slot,
                "num_slots": span.num_slots,
                "slot_index": span.slot_index,
                "packed": span.packed,
                "mode": span.packed_mode or "raw",
            }
            for span in staged.spans
        ]
        if buffer_index is not None:
            chunk_keys = await self._client.transfer_store_registered_cuda(
                buffer_index=buffer_index,
                producer_pid=os.getpid(),
                spans=spans,
                tp_rank=self._tp_rank,
                tp_size=self._tp_size,
            )
        else:
            cp_buffer = cupy.asarray(staged.buffer)
            device_ptr = cuda_array_pointer(cp_buffer)
            allocation_base, allocation_offset = cuda_allocation_base_and_offset(
                device_ptr
            )
            chunk_keys = await self._client.transfer_store_cuda(
                cuda_ipc_handle=export_cuda_ipc_handle(cp_buffer),
                nbytes=packed_bytes,
                device_pci_bus_id=cuda_array_pci_bus_id(cp_buffer),
                device_ptr=device_ptr,
                allocation_base_ptr=allocation_base,
                allocation_offset=allocation_offset,
                producer_pid=os.getpid(),
                tp_rank=self._tp_rank,
                tp_size=self._tp_size,
                spans=spans,
            )
        logger.debug(
            "[CONNECTOR] store transfer complete: spans=%d chunks=%d "
            "raw_bytes=%d packed_bytes=%d elapsed_ms=%.3f",
            len(staged.spans),
            len(chunk_keys),
            raw_equivalent_bytes,
            packed_bytes,
            (time.perf_counter() - transfer_started) * 1000,
        )
        return chunk_keys

    def _reserve_tickets(self, save: _DeferredFinishedSave) -> None:
        """Number ``save``'s staging batches when stores must be ordered.

        Called on the worker thread in submission order, which is the
        scheduler's order and therefore the same on every TP rank.
        """
        if getattr(self, "_online_packer", None) is None:
            return
        if getattr(self, "_tp_size", 1) <= 1:
            return
        save.planned_batches = self._plan_finished_save(save)
        save.first_ticket = getattr(self, "_next_ticket", 0)
        self._next_ticket = save.first_ticket + len(save.planned_batches)

    def _gates(self) -> tuple[_TicketGate, _TicketGate]:
        """Return the lease and transfer gates of ordered stores."""
        if (
            getattr(self, "_lease_gate", None) is None
            or getattr(self, "_transfer_gate", None) is None
        ):
            self._lease_gate = _TicketGate()
            self._transfer_gate = _TicketGate()
        assert self._lease_gate is not None and self._transfer_gate is not None
        return self._lease_gate, self._transfer_gate

    async def _store_failed(
        self,
        save: _DeferredFinishedSave,
        members: tuple[_DeferredFinishedSave, ...],
        exc: Exception,
    ) -> None:
        """Leave a failed save's chunks uncached and free its source blocks."""
        logger.warning(
            "[CONNECTOR] store of %d chunk(s) failed, left uncached: %s",
            len(save.reqs_to_store),
            exc,
        )
        try:
            await self._release_writer_claims(tuple(save.reqs_to_store.values()))
        except Exception as release_exc:  # noqa: BLE001
            logger.warning(
                "[CONNECTOR] writer release after a failed store failed: %s",
                release_exc,
            )
        if getattr(self, "_online_packer", None) is not None:
            stream = getattr(self, "_cuda_stream", None)
            if stream is not None:
                # A staging kernel may still read the source blocks.
                await asyncio.to_thread(stream.synchronize)
            for member in members:
                member.snapshot_done = True

    def _register_staging_buffers(self) -> None:
        """Map fixed store buffers before request traffic; fail startup on error."""
        pool = self._staging_pool
        if pool is None or self._staging_buffer_indices:
            return
        for index in range(pool.depth):
            tensor = pool.buffer(index)
            if not tensor.is_cuda:
                continue
            cp_tensor = cupy.asarray(tensor)
            device_ptr = cuda_array_pointer(cp_tensor)
            allocation_base, allocation_offset = cuda_allocation_base_and_offset(
                device_ptr
            )
            self._submit(
                self._client.register_staging_cuda(
                    direction="store",
                    buffer_index=index,
                    cuda_ipc_handle=export_cuda_ipc_handle(cp_tensor),
                    allocation_bytes=tensor.nbytes,
                    device_pci_bus_id=cuda_array_pci_bus_id(cp_tensor),
                    device_ptr=device_ptr,
                    allocation_base_ptr=allocation_base,
                    allocation_offset=allocation_offset,
                    producer_pid=os.getpid(),
                )
            ).result(timeout=120.0)
            self._staging_buffer_indices[tensor.data_ptr()] = index
        logger.info(
            "[CONNECTOR] registered %d store staging buffers",
            len(self._staging_buffer_indices),
        )


@dataclass(frozen=True)
class StagedStoreBatch:
    """Hold a worker CUDA snapshot until its async store completes."""

    buffer: torch.Tensor
    spans: list[StoreWriteSpan]
    lease: CudaStagingLease
    lease_permit: asyncio.Semaphore | None = None
    lease_wait_s: float = 0.0
    stage_s: float = 0.0


def _store_payload_nbytes(
    buffer_nbytes: int,
    spans: list[StoreWriteSpan],
) -> int:
    """Return the CUDA IPC mapping length for one store batch.

    Packed records are compacted inside a raw-sized staging lease. Mapping only
    through the final packed source byte avoids exposing and registering the
    unused tail while preserving the raw path's existing allocation size.

    Args:
        buffer_nbytes: Logical byte capacity of the leased staging view.
        spans: Store spans describing source-buffer ranges.

    Returns:
        The logical mapping length sent to the server-owned transfer layer.

    Raises:
        ValueError: If a span falls outside the leased staging view.

    Async/thread-safety:
        Pure CPU validation; safe to call from the store event-loop thread.
    """
    if buffer_nbytes < 0:
        raise ValueError("store staging buffer size must be non-negative")
    max_end = 0
    has_packed_span = False
    for span in spans:
        source_offset = int(span.source_offset)
        nbytes = int(span.nbytes)
        if source_offset < 0 or nbytes < 0:
            raise ValueError("store span source range must be non-negative")
        end = source_offset + nbytes
        if end > buffer_nbytes:
            raise ValueError("store span exceeds staging buffer")
        max_end = max(max_end, end)
        has_packed_span = has_packed_span or bool(span.packed)
    if not has_packed_span:
        return buffer_nbytes
    return max_end


def build_staging_store_batches(
    reqs_to_store: dict[str, ReqStoreSpec],
    slot_size: int,
    max_batch_bytes: int = DEFAULT_STORE_STAGING_BYTES,
) -> list[tuple[list[int], list[StoreWriteSpan]]]:
    """Split store requests into bounded slot-major staging batches.

    Args:
        reqs_to_store: Request IDs mapped to store specifications.
        slot_size: Bytes stored for one rank-local KV slot.
        max_batch_bytes: Maximum GPU staging bytes in one batch.

    Returns:
        Ordered block ID and server write-span batches. A chunk larger than
        one batch is split across batches; its spans name the first slot
        they carry with ``slot_index``.

    Async/thread-safety:
        Pure CPU planning; safe to call from worker or store-loop threads.
    """
    if slot_size <= 0:
        raise ValueError("slot_size must be positive")
    max_slots = max(1, max_batch_bytes // slot_size)
    batches: list[tuple[list[int], list[StoreWriteSpan]]] = []
    batch_blocks: list[int] = []
    batch_spans: list[StoreWriteSpan] = []
    written_specs: set[tuple[str, int, int, int]] = set()

    def flush_batch() -> None:
        nonlocal batch_blocks, batch_spans
        if batch_blocks:
            batches.append((batch_blocks, batch_spans))
        batch_blocks = []
        batch_spans = []

    for spec in reqs_to_store.values():
        source_key = (
            spec.chunk_key,
            spec.start_slot,
            spec.num_slots,
            len(spec.block_ids),
        )
        if source_key in written_specs:
            continue
        written_specs.add(source_key)
        cursor = 0
        while cursor < len(spec.block_ids):
            if len(batch_blocks) >= max_slots:
                flush_batch()
            available = max_slots - len(batch_blocks)
            take = min(available, len(spec.block_ids) - cursor)
            if take <= 0:
                flush_batch()
                continue
            source_slot = len(batch_blocks)
            part = spec.block_ids[cursor : cursor + take]
            batch_blocks.extend(part)
            batch_spans.append(
                StoreWriteSpan(
                    source_offset=source_slot * slot_size,
                    nbytes=take * slot_size,
                    slot_index=cursor,
                    chunk_key=spec.chunk_key,
                    start_slot=spec.start_slot,
                    num_slots=spec.num_slots,
                    logical_slot_start=spec.logical_slot_start + cursor,
                    logical_slot_count=take,
                )
            )
            cursor += take
    flush_batch()
    return batches


def _logical_slots_for_batch(
    block_ids: list[int], spans: list[StoreWriteSpan]
) -> list[int]:
    """Expand bounded write spans into one logical slot ID per source block."""
    slots: list[int] = []
    for span in spans:
        count = span.logical_slot_count
        start = span.logical_slot_start
        if start < 0 or count <= 0:
            raise ValueError("store span is missing logical slot metadata")
        slots.extend(range(start, start + count))
    if len(slots) != len(block_ids):
        raise ValueError("store span slot count does not match source blocks")
    return slots


def _packed_store_spans(
    spans: list[StoreWriteSpan],
    packed: list[OnlinePackedSlot],
    slot_size: int,
) -> list[StoreWriteSpan]:
    """Build one transfer span per online packed record.

    The server places every record of a chunk back to back inside the
    chunk's tier extents; each span names its record by ``slot_index``.

    Args:
        spans: Original raw-stride spans grouped by allocation and batch.
        packed: One packed record description per source slot, in span order.
        slot_size: Raw bytes reserved for one rank-local slot.

    Returns:
        One packed transfer span per logical slot.

    Raises:
        ValueError: If span metadata, slot lengths, or record ordering is
            inconsistent.

    Async/thread-safety:
        Pure CPU planning; safe to call from the store worker thread.
    """
    if slot_size <= 0:
        raise ValueError("slot_size must be positive")

    result: list[StoreWriteSpan] = []
    packed_cursor = 0
    for source in spans:
        count = source.logical_slot_count
        if count <= 0 or source.logical_slot_start < 0:
            raise ValueError("store span is missing logical slot metadata")
        source_records = packed[packed_cursor : packed_cursor + count]
        if len(source_records) != count:
            raise ValueError("packed slot count does not match store spans")
        for relative, record in enumerate(source_records):
            if record.logical_slot != source.logical_slot_start + relative:
                raise ValueError("packed slots are not ordered within store span")
            if record.stored_length <= 0:
                raise ValueError("packed record must not be empty")
            result.append(
                StoreWriteSpan(
                    source_offset=record.source_offset,
                    nbytes=record.stored_length,
                    slot_index=source.slot_index + relative,
                    chunk_key=source.chunk_key,
                    start_slot=source.start_slot,
                    num_slots=source.num_slots,
                    logical_slot_start=record.logical_slot,
                    logical_slot_count=1,
                    packed=True,
                    packed_mode=record.mode.name.lower(),
                )
            )
        packed_cursor += count

    if packed_cursor != len(packed):
        raise ValueError("packed slot count does not match store spans")
    return result
