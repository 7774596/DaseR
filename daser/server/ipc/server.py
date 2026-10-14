# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
import contextlib
from dataclasses import asdict
import os
import threading
import time
from typing import Any

from daser import stalldump
from daser.config import (
    DEFAULT_LOAD_READAHEAD_BYTES,
    EXCLUSIVE_TIERS_ENV,
    L2_WRITE_GATE_ENV,
    LOAD_READAHEAD_ENV,
    LOAD_READAHEAD_WINDOW_ENV,
    READAHEAD_BUDGET_FRACTION_ENV,
    STORAGE_FORMAT_COMPRESSED_ONLINE,
)
from daser.connector.helpers import TokenSequence
from daser.ipc_protocol import read_frame, write_frame

# First Party
from daser.logging import init_logger
from daser.metrics import REGISTRY, MetricsRegistry
from daser.server.core import ChunkInfo, ServerCore
from daser.server.packed_layout import PackedArenaFull
from daser.transfer import LeaseIncompleteError, TransferLayer
from daser.transfer.cuda_ipc import open_cuda_ipc_buffer
from daser.transfer.iouring import TieredIOUringTransferLayer

logger = init_logger(__name__)

# Bound adjacent packed writes independently of logical commit granularity.
# The configured L1 capacity below further limits each merge: the transfer
# layer cannot split an individual span to fit its pinned pool. Raw spans
# keep the existing unbounded adjacency coalescing contract.
_PACKED_STORE_COALESCE_BYTES = 128 * 1024 * 1024


_CUDA_IPC_CACHE_LIMIT = 16
# Grouped-load phases exported as daser_load_phase_seconds_total{phase}; each
# maps to a TransferStats load_<phase>_us counter.
_LOAD_PHASES = (
    "readahead_wait",
    "resolve",
    "pending_write",
    "l2",
    "copy_drain",
    "ra_window",
    "l2_reserve",
    "l2_io",
)


def _external_prefix_hits(
    chunks: list[ChunkInfo], num_computed_tokens: int, queries: int
) -> int:
    """Return tokens vLLM will count as external prefix cache hits.

    Args:
        chunks: DaseR lookup chunks.
        num_computed_tokens: tokens vLLM already computed locally.
        queries: vLLM external prefix query token count.

    Returns:
        Contiguous external-prefix hit tokens using vLLM connector semantics.
    """
    covered_until = num_computed_tokens
    for chunk in sorted(chunks, key=lambda item: int(item.target_token_start)):
        target_start = int(chunk.target_token_start)
        target_end = target_start + int(chunk.token_count)
        if target_end <= covered_until:
            continue
        if target_start > covered_until:
            break
        covered_until = target_end
    hits = covered_until - num_computed_tokens
    if hits >= queries:
        hits = queries - 1
    return max(0, min(hits, queries))


def _external_load_window(
    target_start: int,
    target_end: int,
    external_start: int,
    external_end: int,
    block_tokens: int,
) -> tuple[int, int]:
    """Return the block-aligned token window the worker loads from one chunk.

    Must match the connector's load planning: the start rounds up, and the end
    rounds up to cover a partially admitted final block (vLLM admits at most
    ``prompt_len - 1`` external tokens) but never past the chunk.

    Args:
        target_start: First prompt token the chunk covers.
        target_end: Token after the last prompt token the chunk covers.
        external_start: Token offset where external loading begins.
        external_end: Token after the last externally admitted token.
        block_tokens: Tokens per KV block.

    Returns:
        ``(load_start, load_end)``; empty when ``load_end <= load_start``.
    """
    load_start = max(target_start, external_start)
    load_end = min(target_end, external_end)
    load_start = ((load_start + block_tokens - 1) // block_tokens) * block_tokens
    load_end = ((load_end + block_tokens - 1) // block_tokens) * block_tokens
    return load_start, min(load_end, target_end)


def _prefetch_spans_from_chunks(
    chunks: list[ChunkInfo],
    *,
    external_start: int,
    external_tokens: int,
    block_tokens: int,
    slot_size: int,
    tensor_parallel_size: int,
    rank_stride_bytes: int,
) -> list[dict[str, int]]:
    """Translate an admitted external KV window into physical TP-lane spans.

    Args:
        chunks: Server lookup chunks covering the prompt prefix.
        external_start: Token offset where external cache loading begins.
        external_tokens: Number of externally admitted tokens.
        block_tokens: Tokens stored in one logical cache slot.
        slot_size: Aggregate bytes per logical slot across TP ranks.
        tensor_parallel_size: Number of physical rank lanes.
        rank_stride_bytes: Byte distance between adjacent rank lanes.

    Returns:
        Sorted block-aligned physical storage ranges for host-tier admission.
    """
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive for prefetch lookup")
    if external_tokens <= 0 or external_start % block_tokens != 0:
        return []
    if slot_size <= 0 or tensor_parallel_size <= 0:
        raise ValueError("invalid transfer geometry for prefetch lookup")
    if slot_size % tensor_parallel_size:
        raise ValueError("slot_size must divide evenly across tensor-parallel ranks")
    local_slot_size = slot_size // tensor_parallel_size
    external_end = external_start + external_tokens
    spans: list[dict[str, int]] = []
    for chunk in sorted(chunks, key=lambda item: int(item.target_token_start)):
        target_start = int(chunk.target_token_start)
        target_end = target_start + int(chunk.token_count)
        load_start, load_end = _external_load_window(
            target_start, target_end, external_start, external_end, block_tokens
        )
        if load_end <= load_start:
            continue
        start_slot = int(chunk.start_slot) + (
            (load_start - target_start) // block_tokens
        )
        nbytes = ((load_end - load_start) // block_tokens) * local_slot_size
        for rank in range(tensor_parallel_size):
            spans.append(
                {
                    "file_offset": rank * rank_stride_bytes
                    + start_slot * local_slot_size,
                    "nbytes": nbytes,
                }
            )
    return spans


def _coalesce_transfer_spans(
    spans: list[dict[str, Any]],
    *,
    max_packed_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Merge adjacent transfer spans without changing byte contents.

    Held spans are never merged: each held L1 range is one packed record.

    Args:
        spans: transfer spans with source_offset, file_offset, and nbytes.
        max_packed_bytes: Optional upper bound for one coalesced packed span.
            Raw spans and individual packed spans larger than the bound are
            unchanged.

    Returns:
        Coalesced spans sorted by source and file offset.
    """
    if max_packed_bytes is not None and max_packed_bytes <= 0:
        raise ValueError("max_packed_bytes must be positive when provided")
    normalized = [
        {
            "source_offset": int(span.get("source_offset", 0)),
            "file_offset": int(span["file_offset"]),
            "nbytes": int(span["nbytes"]),
            "packed": bool(span.get("packed", False)),
            "accounted_nbytes": int(span.get("accounted_nbytes", span["nbytes"])),
            "hold": bool(span.get("hold", False)),
        }
        for span in spans
        if int(span["nbytes"]) > 0
    ]
    normalized.sort(key=lambda span: (span["source_offset"], span["file_offset"]))
    if not normalized:
        return []

    merged = [normalized[0]]
    for span in normalized[1:]:
        prev = merged[-1]
        prev_source_end = prev["source_offset"] + prev["nbytes"]
        prev_file_end = prev["file_offset"] + prev["nbytes"]
        if (
            span["source_offset"] == prev_source_end
            and span["file_offset"] == prev_file_end
            and span["packed"] == prev["packed"]
            and not span["hold"]
            and not prev["hold"]
            and (
                not span["packed"]
                or max_packed_bytes is None
                or prev["nbytes"] + span["nbytes"] <= max_packed_bytes
            )
        ):
            prev["nbytes"] += span["nbytes"]
            prev["accounted_nbytes"] += span["accounted_nbytes"]
        else:
            merged.append(span)
    return merged


class IPCServer:
    """IPC server over Unix socket + msgpack.

    This server is the internal IPC interface for vLLM DaserConnector. It only
    exposes connector cache operations and delegates all business logic to
    ServerCore.

    Args:
        socket_path: Unix socket path.
        core: shared DaseR server core.
        runtime_config: connector runtime values returned by
            ``get_runtime_config``.

    Async/thread-safety:
        Must be started and stopped from the server asyncio event loop.
    """

    def __init__(
        self,
        socket_path: str,
        core: ServerCore,
        runtime_config: dict[str, Any] | None = None,
        metrics_registry: MetricsRegistry | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._core = core
        self._runtime_config = runtime_config or {}
        self._metrics = metrics_registry or REGISTRY
        self._server: asyncio.AbstractServer | None = None
        self._transfer: TransferLayer | None = None
        self._transfer_lock = threading.Lock()
        # Online packed stores publish asynchronously after the worker has
        # released its source snapshot, so a successor lookup can observe the
        # writer key during the short transfer/commit interval. Retry for this
        # format; raw keeps immediate-miss semantics.
        self._lookup_wait_pending = (
            self._runtime_config.get("storage_format")
            == STORAGE_FORMAT_COMPRESSED_ONLINE
        )
        logger.info("lookup wait_for_pending=%s", self._lookup_wait_pending)
        # L2 readahead group size started by plain lookups (0 disables).
        self._readahead_piece = int(
            os.environ.get(LOAD_READAHEAD_ENV) or DEFAULT_LOAD_READAHEAD_BYTES
        )
        self._prev_readahead_bytes: dict[str, int] = {}
        self._prev_promotion_wait_us: dict[str, int] = {}
        self._prev_store_lease_wait_us = 0
        self._prev_load_phase_us: dict[str, int] = {}
        self._prev_dst_copy: dict[str, int] = {}
        self._prev_l1_evicted: dict[str, int] = {}
        # Packed records are tiered across L1 and an L2 budget when both
        # tiers exist; see docs/design/tier-residency.md. Exclusive tiers
        # store new records in L1 only. Tiers attach with the transfer layer.
        self._packed_tiers = self._runtime_config.get(
            "storage_format"
        ) == STORAGE_FORMAT_COMPRESSED_ONLINE and not bool(
            self._runtime_config.get("skip_l2", False)
        )
        self._exclusive_tiers = (
            self._packed_tiers and os.environ.get(EXCLUSIVE_TIERS_ENV) != "0"
        )
        if self._exclusive_tiers:
            self._runtime_config["exclusive_tiers"] = True
        self._cuda_ipc_cache: OrderedDict[
            tuple[int, int, int, int | None], "_CachedCudaArray"
        ] = OrderedDict()
        self._load_staging_buffers: dict[tuple[int, int], _CachedCudaArray] = {}
        self._store_staging_buffers: dict[tuple[int, int], _CachedCudaArray] = {}
        self._op_handlers: dict[
            str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
        ] = {
            "lookup": self._op_lookup,
            "index_epoch": self._op_index_epoch,
            "lookup_prefetch": self._op_lookup_prefetch,
            "record_external_prefix_cache": self._op_record_external_prefix_cache,
            "get_runtime_config": self._op_get_runtime_config,
            "alloc_chunk": self._op_alloc_chunk,
            "alloc_chunks": self._op_alloc_chunks,
            "match_and_alloc": self._op_match_and_alloc,
            "commit_chunk": self._op_commit_chunk,
            "commit_chunks": self._op_commit_chunks,
            "commit_stats": self._op_commit_stats,
            "live_allocations": self._op_live_allocations,
            "transfer_drain": self._op_transfer_drain,
            "transfer_prefetch": self._transfer_prefetch,
            "init_transfer": self._op_init_transfer,
            "transfer_store": self._transfer_store,
            "transfer_load": self._transfer_load,
            "transfer_load_runs": self._transfer_load_runs,
            "register_load_staging": self._register_load_staging,
            "register_store_staging": self._register_store_staging,
            "evict_chunk": self._op_evict_chunk,
            "release_chunk_writer": self._op_release_chunk_writer,
            "release_transfer_lease": self._op_release_transfer_lease,
        }

    async def start(self) -> None:
        """Start listening on the Unix socket.

        Async/thread-safety:
            Removes a stale socket path and starts an asyncio Unix server.
        """
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        stalldump.register_loop("server", asyncio.get_running_loop())
        stalldump.register_state(
            "server-transfer",
            lambda: {} if self._transfer is None else self._transfer.debug_state(),
        )
        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=self._socket_path
        )
        logger.info("[IPC] listening on %s", self._socket_path)

    async def initialize_transfer(self) -> None:
        """Eagerly create the transfer layer off the event loop.

        Offloads the blocking ``_ensure_transfer`` call (which may allocate
        pinned memory pools, open io_uring rings, etc.) to a thread so the
        server is fully provisioned before the first inference request.

        Async/thread-safety:
            Must be called from the server asyncio event loop during startup.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_transfer)

    async def drain_transfer(self) -> None:
        """Wait for server-owned transfer-layer background work.

        Async/thread-safety:
            Runs on the server asyncio event loop. The transfer layer is
            initialized on demand, then its async ``drain`` method is awaited
            when present.
        """
        transfer = self._ensure_transfer()
        await transfer.drain()

    async def stop(self) -> None:
        """Stop the server and remove the socket file.

        Async/thread-safety:
            Closes the asyncio server on the current event loop.
        """
        await self.stop_accepting()
        await self.close()
        logger.info("[IPC] server stopped")

    async def stop_accepting(self) -> None:
        """Stop accepting new IPC connections and remove the socket path.

        Async/thread-safety:
            Closes the asyncio listener on the current event loop. Existing
            transfer resources remain open until ``close`` is called.
        """
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        logger.info("[IPC] server stopped accepting")

    async def close(self) -> None:
        """Drain and close transfer resources owned by the IPC server.

        Async/thread-safety:
            Runs on the server asyncio event loop after new IPC work has been
            rejected.
        """
        if self._transfer is not None:
            await self._transfer.drain()
            self._transfer.close()
            self._transfer = None
        self._close_cuda_ipc_cache()
        logger.info("[IPC] resources closed")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle one connector connection with multiple frames.

        Args:
            reader: request stream reader.
            writer: response stream writer.

        Async/thread-safety:
            Runs one coroutine per client connection.
        """
        try:
            while True:
                try:
                    msg = await read_frame(reader)
                except asyncio.IncompleteReadError:
                    return
                response = await self._dispatch(msg)
                await write_frame(writer, response)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[IPC] error handling request: %s", exc)
            try:
                await write_frame(writer, {"error": str(exc)})
            except Exception:
                pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _dispatch(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one decoded connector request.

        Args:
            msg: request dict containing an ``op`` key.

        Returns:
            Response dict suitable for msgpack encoding.

        Async/thread-safety:
            Calls ServerCore on the same asyncio event loop.
        """
        op = str(msg.get("op", "unknown"))
        ipc_labels = {"op": op}
        started = time.perf_counter()
        status = "error"
        # Requests still running: a stalled op stays visible on /metrics.
        inflight = self._metrics.gauge(
            "daser_ipc_requests_inflight",
            "IPC requests currently being handled, by operation.",
        )
        inflight.inc(labels=ipc_labels)
        try:
            handler = self._op_handlers.get(op)
            if handler is None:
                return {"error": f"unknown op: {op}"}
            response = await handler(msg)
            ok = response.get("ok", True) is not False and "error" not in response
            status = "ok" if ok else "error"
            return response
        except Exception as exc:  # noqa: BLE001
            logger.exception("[IPC] request failed: %s", exc)
            return {"error": str(exc)}
        finally:
            inflight.inc(-1.0, labels=ipc_labels)
            elapsed = time.perf_counter() - started
            self._metrics.counter(
                "daser_ipc_requests_total",
                "IPC requests by operation and status.",
            ).inc(labels={**ipc_labels, "status": status})
            self._metrics.histogram(
                "daser_ipc_request_duration_seconds",
                "IPC request latency by operation.",
                buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
            ).observe(elapsed, labels=ipc_labels)

    async def _op_lookup(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``lookup`` request, recording external prefix counters.

        ``defer_pending`` selects how a lookup that a pending writer may
        extend is answered when pending waits are enabled: absent keeps the
        server-side wait, True replies ``{"pending": True}`` at once so a
        synchronous caller can retry later, and False resolves immediately.
        """
        tokens = self._lookup_tokens(msg)
        # Read before the lookup awaits: a removal during it advances the
        # epoch past the reply's, so a caller never reuses a stale result.
        epoch = self._core.index_epoch
        defer_pending = msg.get("defer_pending")
        if self._lookup_wait_pending and defer_pending is not None:
            chunks = await self._lookup_now(
                tokens,
                msg["model_id"],
                defer=bool(defer_pending),
                num_computed_tokens=int(msg.get("num_computed_tokens", 0)),
            )
            if chunks is None:
                return {"chunks": [], "pending": True}
        else:
            chunks = await self._lookup_core(tokens, msg["model_id"])
        if "external_prefix_queries" in msg:
            queries = int(msg.get("external_prefix_queries", 0))
            num_computed_tokens = int(msg.get("num_computed_tokens", 0))
            hits = _external_prefix_hits(
                chunks,
                num_computed_tokens=num_computed_tokens,
                queries=queries,
            )
            await self._core.record_external_prefix_cache(queries=queries, hits=hits)
            if self._readahead_piece and chunks and hits > 0:
                spans = self._external_spans(chunks, num_computed_tokens, hits)
                if spans:
                    await self._ensure_transfer().start_readahead(spans)
        return {"chunks": self._chunk_payloads(chunks), "index_epoch": epoch}

    async def _op_index_epoch(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle an ``index_epoch`` request.

        Returns:
            ``{"index_epoch": int}``; a lookup reply carrying the same value
            still describes chunks that are all present in the index.
        """
        return {"index_epoch": self._core.index_epoch}

    async def _op_lookup_prefetch(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Lookup, classify exact external spans, and lease an all-L1 result."""
        lease_id = str(msg.get("lease_id", ""))
        if not lease_id:
            raise ValueError("lookup_prefetch requires lease_id")
        tokens = self._lookup_tokens(msg)
        chunks = await self._lookup_core(tokens, msg["model_id"])
        queries = int(msg.get("external_prefix_queries", 0))
        num_computed_tokens = int(msg.get("num_computed_tokens", 0))
        hits = _external_prefix_hits(chunks, num_computed_tokens, queries)
        await self._core.record_external_prefix_cache(queries=queries, hits=hits)
        if not chunks or hits <= 0:
            return {"chunks": self._chunk_payloads(chunks), "spans": []}

        spans = self._external_spans(chunks, num_computed_tokens, hits)
        if not spans:
            return {"chunks": self._chunk_payloads(chunks), "spans": []}
        transfer = self._ensure_transfer()
        tier = await transfer.classify_and_acquire_lease(lease_id, spans)
        return {
            "chunks": self._chunk_payloads(chunks),
            "spans": spans,
            "tier": tier,
        }

    @staticmethod
    def _lookup_tokens(msg: dict[str, Any]) -> TokenSequence:
        """Decode the compact token representation used by scheduler lookup.

        Args:
            msg: IPC request containing ``token_bytes`` or legacy ``tokens``.

        Returns:
            Token IDs as packed native-int bytes or a legacy list.

        Raises:
            ValueError: if the packed representation is malformed.
        """
        packed = msg.get("token_bytes")
        if packed is not None:
            if not isinstance(packed, bytes) or len(packed) % 4:
                raise ValueError("lookup token_bytes must be int32 aligned bytes")
            return packed
        return msg["tokens"]

    async def _lookup_core(
        self, tokens: TokenSequence, model_id: str
    ) -> list[ChunkInfo]:
        """Run an immediate lookup against the committed retrieval index.

        Args:
            tokens: Prompt token IDs.
            model_id: Model identifier used for cache isolation.

        Returns:
            Retrieval chunks returned by the server core.

        Async/thread-safety:
            Runs on the IPC server event loop. Online stores are published to
            the retrieval index only after their transfer commits; an
            uncommitted record is therefore a safe miss and must not make the
            request wait for a background writer. The compatibility fallback
            is limited to test doubles or older public core implementations
            that do not expose the keyword.
        """
        try:
            chunks = await self._core.lookup(
                tokens,
                model_id,
                wait_for_pending=self._lookup_wait_pending,
            )
        except TypeError as exc:
            if "wait_for_pending" not in str(exc):
                raise
            chunks = await self._core.lookup(tokens, model_id)
        chunks = await self._memory_resident_chunks(chunks)
        return chunks

    async def _lookup_now(
        self,
        tokens: TokenSequence,
        model_id: str,
        *,
        defer: bool,
        num_computed_tokens: int = 0,
    ) -> list[ChunkInfo] | None:
        """Run a lookup that never waits for pending writers.

        Args:
            tokens: Prompt token IDs.
            model_id: Model identifier used for cache isolation.
            defer: Return None instead of chunks when a pending writer may
                extend the match.
            num_computed_tokens: Tokens the caller's GPU prefix cache already
                holds; writers ending within them do not defer.

        Returns:
            Retrieval chunks, or None for a deferred lookup.

        Async/thread-safety:
            Runs on the IPC server event loop without sleeping, so a
            synchronous scheduler RPC is never held for a writer commit.
        """
        if defer:
            chunks = await self._core.lookup_unless_pending(
                tokens, model_id, num_computed_tokens
            )
            if chunks is None:
                return None
        else:
            chunks = await self._core.lookup(tokens, model_id)
        chunks = await self._memory_resident_chunks(chunks)
        return chunks

    async def _memory_resident_chunks(self, chunks: list[ChunkInfo]) -> list[ChunkInfo]:
        """Trim a memory-only lookup to the chunks whose bytes are still in L1.

        With ``skip_l2`` the index and L1 replacement are independent: pinned
        pool pressure can evict a committed range that the index still lists,
        and nothing can refill it. Advertising such a chunk makes the worker
        load fail, so the match stops at the first chunk that left memory.

        Args:
            chunks: Lookup chunks covering the prompt prefix.

        Returns:
            The leading chunks that are fully L1-resident; all chunks when L2
            is enabled.

        Async/thread-safety:
            Runs on the IPC server event loop. Residency is checked and
            refreshed atomically in the transfer layer; a store racing the
            later load can still evict a counted chunk.
        """
        if not chunks or not self._runtime_config.get("skip_l2", False):
            return chunks
        ordered = sorted(chunks, key=lambda item: int(item.target_token_start))
        groups: list[list[dict[str, int]]] = []
        for chunk in ordered:
            spans = self._chunk_spans(chunk)
            if not spans:
                break
            groups.append(spans)
        resident = await self._ensure_transfer().resident_prefix(groups)
        return ordered[:resident]

    async def _op_record_external_prefix_cache(
        self, msg: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle a standalone ``record_external_prefix_cache`` request."""
        await self._core.record_external_prefix_cache(
            queries=int(msg.get("queries", 0)),
            hits=int(msg.get("hits", 0)),
        )
        return {"ok": True}

    async def _op_get_runtime_config(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``get_runtime_config`` request."""
        return {"runtime_config": dict(self._runtime_config)}

    async def _op_alloc_chunk(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a single ``alloc_chunk`` request."""
        alloc = await self._core.alloc_chunk(
            msg["chunk_key"], int(msg["token_count"]), msg["model_id"]
        )
        return alloc.to_dict(include_chunk_key=False)

    async def _op_alloc_chunks(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a batched ``alloc_chunks`` request."""
        allocs = await self._core.alloc_chunks(
            list(msg.get("chunks", [])), msg["model_id"]
        )
        return {
            "allocations": [alloc.to_dict(include_chunk_key=True) for alloc in allocs]
        }

    async def _op_match_and_alloc(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``match_and_alloc`` request."""
        result = await self._core.match_and_alloc(
            msg["tokens"], msg.get("chunk_key", ""), msg["model_id"]
        )
        return result.to_dict()

    async def _op_commit_chunk(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a single ``commit_chunk`` request."""
        await self._core.commit_chunk(
            msg["chunk_key"],
            tp_rank=int(msg.get("tp_rank", 0)),
            tp_size=int(msg.get("tp_size", 1)),
        )
        return {"ok": True}

    async def _op_commit_chunks(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a batched ``commit_chunks`` request."""
        for chunk_key in msg.get("chunk_keys", []):
            await self._core.commit_chunk(
                chunk_key,
                tp_rank=int(msg.get("tp_rank", 0)),
                tp_size=int(msg.get("tp_size", 1)),
            )
        return {"ok": True}

    async def _op_commit_stats(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``commit_stats`` request."""
        return {"commit_stats": await self._core.commit_stats()}

    async def _op_live_allocations(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``live_allocations`` request."""
        live = await self._core.live_allocations(list(msg.get("allocations", [])))
        return {"chunk_keys": live}

    async def _op_transfer_drain(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``transfer_drain`` request."""
        transfer = self._transfer
        if transfer is not None:
            await transfer.drain()
        return {"ok": True}

    async def _transfer_prefetch(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Promote storage spans into the server-owned host-memory tier.

        Args:
            msg: IPC request containing ``spans`` with file offsets and sizes.

        Returns:
            Requested, L1-resident, and L2-read byte counts.

        Async/thread-safety:
            Runs on the IPC event loop and delegates to the transfer layer's
            asynchronous prefetch capability.
        """
        transfer = self._ensure_transfer()
        lease_id = str(msg.get("lease_id", "")) or None
        spans = list(msg.get("spans", []))
        operations = self._metrics.counter(
            "daser_prefetch_operations_total",
            "Host-tier prefetch operations by result.",
        )
        try:
            result = await transfer.prefetch_bytes_grouped(spans, lease_id=lease_id)
        except LeaseIncompleteError as exc:
            # Expected when the looked-up chunk lost its slots before the
            # prefetch ran; the connector releases the lease and looks again.
            operations.inc(labels={"status": "incomplete"})
            logger.warning("[IPC] transfer_prefetch: %s", exc)
            return {"error": str(exc)}
        operations.inc(labels={"status": "ok"})
        bytes_counter = self._metrics.counter(
            "daser_prefetch_bytes_total",
            "Host-tier prefetch bytes by tier.",
        )
        bytes_counter.inc(result.requested_bytes, labels={"tier": "requested"})
        bytes_counter.inc(result.l1_bytes, labels={"tier": "l1"})
        bytes_counter.inc(result.l2_bytes, labels={"tier": "l2"})
        return {
            "ok": True,
            "requested_bytes": result.requested_bytes,
            "l1_bytes": result.l1_bytes,
            "l2_bytes": result.l2_bytes,
        }

    async def _op_init_transfer(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle an ``init_transfer`` request.

        Waits off the event loop: during startup the eager initializer may
        hold the transfer lock while pinning L1 for minutes, and blocking the
        loop on it would stall every other client's request past its timeout.
        """
        await self.initialize_transfer()
        return {"ok": True}

    async def _op_evict_chunk(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle an ``evict_chunk`` request."""
        await self._core.evict_chunk(msg["chunk_key"])
        return {"ok": True}

    async def _op_release_chunk_writer(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``release_chunk_writer`` request."""
        released = await self._core.release_chunk_writer(
            chunk_key=str(msg["chunk_key"]),
            start_slot=int(msg["start_slot"]),
            num_slots=int(msg["num_slots"]),
        )
        return {"released": released}

    async def _op_release_transfer_lease(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Idempotently release remaining host-tier bytes for one request."""
        lease_id = str(msg.get("lease_id", ""))
        if not lease_id:
            raise ValueError("release_transfer_lease requires lease_id")
        transfer = self._transfer
        if transfer is not None:
            await transfer.release_lease(lease_id)
        return {"ok": True}

    async def _transfer_store(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Store one or more spans through the server-owned transfer layer.

        Args:
            msg: IPC request with ``payload`` and ``spans``.

        Returns:
            Response dict with total bytes stored.

        Async/thread-safety:
            Runs on the IPC event loop and awaits transfer-layer operations.
        """
        payload = msg.get("payload", {})
        spans = list(msg.get("spans", []))
        started = time.perf_counter()
        backend = str(self._runtime_config.get("transfer_mode", "gds"))
        transfer = self._ensure_transfer()
        total = 0
        layout = self._core.packed_layout
        hold = self._exclusive_tiers and transfer.supports_held_stores
        held_ranges: list[tuple[int, int]] = []
        stored_chunk_keys: list[str] = []
        accepted_spans: list[dict[str, Any]] = []
        place_ms = 0.0
        reclaims = 0
        tp_rank = int(msg.get("tp_rank", 0))
        tp_size = int(msg.get("tp_size", 1))
        # The worker may export a compact view whose reported length ends at
        # its last record.  Preserve the IPC view's byte-addressable contract
        # when source offsets are sparse or reordered by sizing the mapping
        # through the furthest submitted source byte.  This does not transfer
        # or retain extra data; it only prevents a valid span from being
        # sliced beyond the mapped CUDA view.
        if "cuda_ipc_handle" in payload and spans:
            payload = dict(payload)
            required_nbytes = max(
                int(span.get("source_offset", 0)) + int(span["nbytes"])
                for span in spans
                if int(span["nbytes"]) > 0
            )
            payload["nbytes"] = max(int(payload.get("nbytes", 0)), required_nbytes)
        if "store_staging_buffer_index" in payload:
            buffer_key = (
                int(payload["producer_pid"]),
                int(payload["store_staging_buffer_index"]),
            )
            try:
                buffer = self._store_staging_buffers[buffer_key]
            except KeyError as exc:
                raise ValueError(
                    f"store staging buffer is not registered: {buffer_key}"
                ) from exc
            for span in spans:
                offset, nbytes = int(span.get("source_offset", 0)), int(span["nbytes"])
                if offset < 0 or nbytes < 0 or offset + nbytes > buffer.nbytes:
                    raise ValueError(
                        "store source span exceeds registered staging buffer"
                    )
        elif "cuda_ipc_handle" in payload:
            # Store mappings are one-shot views whose source extent can vary
            # for every compacted batch.  Do not reuse a shorter cached view
            # from an earlier batch of the same staging allocation.
            buffer = self._open_cuda_ipc_payload(
                payload=payload,
                nbytes_key="nbytes",
                cache_mapping=False,
            )
        else:
            buffer = self._payload_buffer(payload)
        write_hold: int | None = None
        try:
            live_spans: list[dict[str, Any]] = []
            for span in spans:
                nbytes = int(span["nbytes"])
                file_offset = int(span["file_offset"])
                chunk_key = str(span.get("chunk_key", ""))
                if chunk_key:
                    live = self._core.is_current_allocation(
                        chunk_key=chunk_key,
                        start_slot=int(span.get("start_slot", -1)),
                        num_slots=int(span.get("num_slots", 0)),
                    )
                    if not live:
                        logger.debug(
                            "[IPC] skip stale transfer_store key=%s offset=%d bytes=%d",
                            chunk_key[:8],
                            file_offset,
                            nbytes,
                        )
                        continue
                    stored_chunk_keys.append(chunk_key)
                live_spans.append(span)

            if self._runtime_config.get(
                "storage_format"
            ) == STORAGE_FORMAT_COMPRESSED_ONLINE and any(
                bool(span.get("packed", False)) for span in live_spans
            ):
                capacity = int(
                    self._runtime_config.get(
                        "l2_size_bytes",
                        self._runtime_config.get("total_store_bytes", 0),
                    )
                )
                if capacity <= 0:
                    capacity = int(self._runtime_config.get("slot_size", 0)) * int(
                        self._runtime_config.get("total_slots", 0)
                    )
                local_slot_size = int(
                    self._runtime_config.get(
                        "local_slot_size",
                        int(self._runtime_config.get("slot_size", 0))
                        // max(1, tp_size),
                    )
                )
                rank_stride_bytes = int(
                    self._runtime_config.get("rank_stride_bytes", 0)
                )
                # Logical slots are over-provisioned for compressed records,
                # so the physical arena can fill first. Remove the oldest
                # records (never the ones being stored) until the reported
                # deficit is freed, then retry placement.
                rank_base = tp_rank * rank_stride_bytes
                protected_keys = {str(span.get("chunk_key", "")) for span in live_spans}
                place_started = time.perf_counter()
                while True:
                    try:
                        live_spans = layout.compact(
                            live_spans,
                            self._core.chunk_manager,
                            local_slot_size=local_slot_size,
                            rank_base=rank_base,
                            arena_size=rank_stride_bytes or capacity // max(1, tp_size),
                            hold=hold,
                        )
                        break
                    except PackedArenaFull as exc:
                        # Chunk removal releases placements inside the core,
                        # so free space can be read directly.
                        target = (
                            layout.free_bytes(rank_base=rank_base) + exc.deficit_bytes
                        )
                        while layout.free_bytes(rank_base=rank_base) < target:
                            if not await self._core.reclaim_packed(
                                protected_keys, rank_base=rank_base, free_address=True
                            ):
                                raise
                            reclaims += 1
                # Keep L2 within its budget; L1-resident victims stay indexed.
                l2_budget = int(self._runtime_config.get("l2_budget_bytes", 0))
                while (
                    l2_budget
                    and layout.charged_bytes(rank_base=rank_base) > l2_budget
                    and await self._core.reclaim_packed(
                        protected_keys, rank_base=rank_base
                    )
                ):
                    reclaims += 1
                place_ms = (time.perf_counter() - place_started) * 1000
                self._record_packed_ratios(live_spans, local_slot_size)
                # Another store may evict these chunks while this one awaits
                # its IO; keep their bytes from being placed again until the
                # transfer layer has this write ordered.
                write_hold = layout.begin_writes(live_spans, rank_base=rank_base)

            for span in live_spans:
                chunk_key = str(span.get("chunk_key", ""))
                if not chunk_key:
                    continue
                accepted_spans.append(
                    {
                        "chunk_key": chunk_key,
                        "file_offset": int(span["file_offset"]),
                        "nbytes": int(span["nbytes"]),
                        "start_slot": int(span.get("start_slot", -1)),
                        "num_slots": int(span.get("num_slots", 0)),
                        "logical_slot_start": int(span.get("logical_slot_start", -1)),
                        "logical_slot_count": int(span.get("logical_slot_count", 0)),
                        "packed": bool(span.get("packed", False)),
                        "mode": str(span.get("mode", "compressed")),
                    }
                )

            if hold:
                # Only records placed held go to L1 alone; a retried record
                # that L1 already released stays charged.
                live_spans = [
                    {**span, "hold": True}
                    if span.get("packed", False)
                    and layout.is_held(int(span["file_offset"]), int(span["nbytes"]))
                    else span
                    for span in live_spans
                ]
                held_ranges = [
                    (int(span["file_offset"]), int(span["nbytes"]))
                    for span in live_spans
                    if span.get("hold")
                ]
            if transfer.coalesce_store_spans:
                store_spans = _coalesce_transfer_spans(
                    live_spans,
                    max_packed_bytes=min(
                        _PACKED_STORE_COALESCE_BYTES,
                        int(
                            self._runtime_config.get(
                                "l1_size_bytes", _PACKED_STORE_COALESCE_BYTES
                            )
                        ),
                    ),
                )
            else:
                store_spans = live_spans
            io_started = time.perf_counter()
            total = await transfer.store_bytes_grouped(buffer, store_spans)
            logger.info(
                "[IPC] transfer_store spans=%d packed=%d dispatched=%d bytes=%d "
                "place_ms=%.1f reclaims=%d io_ms=%.1f total_ms=%.1f",
                len(live_spans),
                sum(bool(span.get("packed", False)) for span in live_spans),
                len(store_spans),
                sum(int(span["nbytes"]) for span in store_spans),
                place_ms,
                reclaims,
                (time.perf_counter() - io_started) * 1000,
                (time.perf_counter() - started) * 1000,
            )
        finally:
            # A record removed while its held store was in flight was dropped
            # before its L1 entry existed. Discard that entry while the write
            # hold still keeps its offset from being placed again.
            for file_offset, nbytes in held_ranges:
                if not layout.is_held(file_offset, nbytes):
                    transfer.discard_held(file_offset, nbytes)
            if write_hold is not None:
                self._end_packed_writes(write_hold)
            if (
                isinstance(buffer, _UncachedCudaArray)
                and "store_staging_buffer_index" not in payload
            ):
                buffer.close()
        if accepted_spans:
            configured_tp_size = int(
                self._runtime_config.get("tensor_parallel_size", tp_size)
            )
            if configured_tp_size != tp_size:
                raise ValueError(
                    f"transfer TP size {tp_size} != configured {configured_tp_size}"
                )
            slot_size = int(self._runtime_config.get("slot_size", 0))
            local_slot_size = int(
                self._runtime_config.get(
                    "local_slot_size", slot_size // max(1, tp_size)
                )
            )
            rank_stride_bytes = int(self._runtime_config.get("rank_stride_bytes", 0))
            await self._core.record_store_ranges(
                accepted_spans,
                tp_rank=tp_rank,
                tp_size=tp_size,
                local_slot_size=local_slot_size,
                rank_stride_bytes=rank_stride_bytes,
            )
        self._record_transfer_metrics(
            op="store",
            backend=backend,
            status="ok",
            nbytes=total,
            elapsed_s=time.perf_counter() - started,
        )
        return {"ok": True, "bytes": total, "chunk_keys": stored_chunk_keys}

    async def _transfer_load(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Load one or more spans through the server-owned transfer layer.

        Args:
            msg: IPC request with ``payload`` and ``spans``.

        Returns:
            Response dict with total bytes loaded and optional bytes data.

        Async/thread-safety:
            Runs on the IPC event loop and awaits transfer-layer operations.
        """
        payload = msg.get("payload", {})
        spans = list(msg.get("spans", []))
        lease_id = str(msg.get("lease_id", "")) or None
        started_total = time.perf_counter()
        backend = str(self._runtime_config.get("transfer_mode", "gds"))
        transfer = self._ensure_transfer()
        total_size = self._payload_size(payload, spans)
        open_ms = 0.0
        if payload.get("return_data"):
            buffer: Any = bytearray(total_size)
        else:
            open_start = time.perf_counter()
            buffer = self._payload_buffer(payload)
            open_ms = (time.perf_counter() - open_start) * 1000

        total = 0
        load_ms = 0.0
        sync_ms = 0.0
        close_ms = 0.0
        close_one_shot_buffer = isinstance(buffer, _UncachedCudaArray) and (
            "load_staging_buffer_index" not in payload
        )
        leased_load_started = False
        try:
            before = asdict(transfer.stats)
            started = time.perf_counter()
            load_start = time.perf_counter()
            if lease_id is None:
                total = await transfer.load_bytes_grouped(buffer, spans)
            else:
                total = await transfer.load_leased_bytes_grouped(
                    buffer,
                    spans,
                    lease_id,
                )
                leased_load_started = True
            load_ms = (time.perf_counter() - load_start) * 1000
            synchronize = getattr(buffer, "synchronize", None)
            if synchronize is not None:
                sync_start = time.perf_counter()
                # CUDA stream synchronization is a blocking runtime call.  It
                # must finish before releasing a leased L1 range, but waiting
                # on the IPC event loop would serialize unrelated load/store
                # requests behind the slowest destination copy.
                await asyncio.to_thread(synchronize)
                sync_ms = (time.perf_counter() - sync_start) * 1000
                self._metrics.counter(
                    "daser_load_phase_seconds_total",
                    "Grouped-load wall time by phase.",
                ).inc(sync_ms / 1000, labels={"phase": "sync"})
            if lease_id is not None:
                await transfer.release_lease_ranges(lease_id, spans)
            elapsed_ms = (time.perf_counter() - started) * 1000
            after = asdict(transfer.stats)
            stats_delta = {
                key: int(after.get(key, 0)) - int(before.get(key, 0))
                for key in set(before) | set(after)
            }
            logger.debug(
                "[IPC] transfer_load timing: spans=%d bytes=%d total_size=%d "
                "open_ms=%.3f load_ms=%.3f sync_ms=%.3f elapsed_ms=%.3f "
                "stats_delta=%s",
                len(spans),
                total,
                total_size,
                open_ms,
                load_ms,
                sync_ms,
                elapsed_ms,
                stats_delta,
            )
            response: dict[str, Any] = {"ok": True, "bytes": total}
            if payload.get("return_data"):
                response["data"] = bytes(buffer)
            else:
                response["transfer_ms"] = elapsed_ms
                response["transfer_open_ms"] = open_ms
                response["transfer_load_ms"] = load_ms
                response["transfer_sync_ms"] = sync_ms
                response["transfer_stats_delta"] = stats_delta
            self._record_transfer_metrics(
                op="load",
                backend=backend,
                status="ok",
                nbytes=total,
                elapsed_s=time.perf_counter() - started_total,
            )
            return response
        except BaseException:
            if lease_id is not None:
                if leased_load_started:
                    await transfer.release_lease_ranges(lease_id, [])
                await transfer.release_lease(lease_id)
            raise
        finally:
            if close_one_shot_buffer:
                close = getattr(buffer, "close", None)
                close_start = time.perf_counter()
                if close is not None:
                    close()
                close_ms = (time.perf_counter() - close_start) * 1000
                logger.info(
                    "[IPC] transfer_load close timing: bytes=%d close_ms=%.3f",
                    total,
                    close_ms,
                )

    async def _transfer_load_runs(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Copy a leased run table into a registered load staging buffer.

        Layer-wise delivery copies one plane group of every record of a
        request per call. ``runs`` packs the group as one ``(3, n)`` int64
        array (file offsets, target offsets, lengths) so the request costs
        O(1) Python regardless of the record count.

        Args:
            msg: IPC request with ``payload`` (registered staging buffer),
                ``lease_id`` and ``runs`` bytes.

        Returns:
            ``{"ok": True, "bytes": n, "transfer_ms": ms}`` once the copies
            completed on the device.

        Raises:
            ValueError: If the run table is malformed or the buffer unknown.

        Async/thread-safety:
            Runs on the IPC event loop. The lease is retained, not consumed;
            the worker releases it with ``release_transfer_lease`` after the
            last group, so concurrent groups on different staging buffers
            overlap.
        """
        import numpy

        started = time.perf_counter()
        lease_id = str(msg.get("lease_id", ""))
        packed = msg.get("runs")
        payload = msg.get("payload", {})
        if not lease_id:
            raise ValueError("transfer_load_runs requires lease_id")
        if "load_staging_buffer_index" not in payload:
            raise ValueError("transfer_load_runs requires a registered buffer")
        if not isinstance(packed, bytes) or not packed or len(packed) % 24:
            raise ValueError("transfer_load_runs runs must be (3, n) int64 bytes")
        runs = numpy.frombuffer(packed, dtype=numpy.int64).reshape(3, -1)
        total = await self._ensure_transfer().load_leased_runs(
            self._payload_buffer(payload),
            runs[0],
            runs[1],
            runs[2],
            lease_id,
        )
        elapsed_s = time.perf_counter() - started
        self._record_transfer_metrics(
            op="load",
            backend=str(self._runtime_config.get("transfer_mode", "gds")),
            status="ok",
            nbytes=total,
            elapsed_s=elapsed_s,
        )
        return {"ok": True, "bytes": total, "transfer_ms": elapsed_s * 1000}

    async def _register_load_staging(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Register one fixed CUDA load staging buffer for indexed reuse.

        Args:
            msg: IPC request whose payload describes a worker-owned fixed CUDA
                staging allocation.

        Returns:
            ``{"ok": True}`` after the CUDA IPC mapping is open and cached.

        Async/thread-safety:
            Runs on the IPC event loop during worker initialization. Replacing
            an existing index closes the old mapping only after the new mapping
            is available.
        """
        return self._register_staging(msg, self._load_staging_buffers)

    async def _register_store_staging(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Map a fixed worker store buffer during startup, outside timed traffic."""
        return self._register_staging(msg, self._store_staging_buffers)

    def _register_staging(
        self,
        msg: dict[str, Any],
        buffers: dict[tuple[int, int], "_CachedCudaArray"],
    ) -> dict[str, Any]:
        """Open a worker pool mapping; its owner retains it until server shutdown."""
        payload = msg.get("payload", {})
        buffer_index = int(payload["buffer_index"])
        producer_pid = int(payload["producer_pid"])
        if buffer_index < 0 or int(payload["allocation_bytes"]) <= 0:
            raise ValueError("staging index must be non-negative and capacity positive")
        buffer_key = (producer_pid, buffer_index)
        opened = self._open_cuda_ipc_payload(
            payload=payload,
            nbytes_key="allocation_bytes",
            cache_mapping=False,
        )
        previous = buffers.get(buffer_key)
        buffers[buffer_key] = opened
        if previous is not None:
            previous.close()
        return {"ok": True}

    def _record_transfer_metrics(
        self,
        op: str,
        backend: str,
        status: str,
        nbytes: int,
        elapsed_s: float,
    ) -> None:
        """Record transfer operation metrics and GB/s log output.

        Args:
            op: Transfer operation, ``load`` or ``store``.
            backend: Configured transfer backend (used for logging only).
            status: Operation status.
            nbytes: Bytes transferred.
            elapsed_s: Operation latency in seconds.
        """
        labels = {"op": op}
        self._metrics.counter(
            "daser_transfer_operations_total",
            "Transfer operations by operation and status.",
        ).inc(labels={**labels, "status": status})
        self._metrics.counter(
            "daser_transfer_bytes_total",
            "Transfer bytes by operation.",
        ).inc(nbytes, labels=labels)
        self._metrics.histogram(
            "daser_transfer_duration_seconds",
            "Transfer operation latency by operation.",
            buckets=(
                0.0005,
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
            ),
        ).observe(elapsed_s, labels=labels)
        self._metrics.histogram(
            "daser_transfer_chunk_size_bytes",
            "Transfer size per operation in bytes.",
            buckets=(65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456),
        ).observe(nbytes, labels=labels)
        self._record_tier_metrics()
        throughput_gbps = (nbytes / elapsed_s / 1_000_000_000) if elapsed_s > 0 else 0.0
        logger.debug(
            "[IPC] transfer_%s summary backend=%s status=%s bytes=%d "
            "elapsed_ms=%.3f throughput_gbps=%.3f",
            op,
            backend,
            status,
            nbytes,
            elapsed_s * 1000,
            throughput_gbps,
        )

    def _record_packed_ratios(
        self, spans: list[dict[str, Any]], local_slot_size: int
    ) -> None:
        """Observe each placed packed record's size relative to a raw slot.

        Args:
            spans: Spans returned by packed placement for one store.
            local_slot_size: Bytes of one raw rank-local slot.

        Async/thread-safety:
            Call on the server event loop. A retried store observes its
            records again.
        """
        histogram = self._metrics.histogram(
            "daser_packed_record_ratio",
            "Stored bytes of each placed packed record over one raw slot, by mode.",
            buckets=tuple(i / 100 for i in range(50, 101, 2)),
        )
        for span in spans:
            if span.get("packed", False):
                histogram.observe(
                    int(span["nbytes"]) / local_slot_size,
                    labels={"mode": str(span.get("mode", "compressed"))},
                )

    def _end_packed_writes(self, token: int) -> None:
        """End one packed write hold and count bytes freed late by it."""
        freed = self._core.packed_layout.end_writes(token)
        if freed:
            self._metrics.counter(
                "daser_packed_deferred_free_bytes_total",
                "Packed bytes freed when a store finished after its chunk left.",
            ).inc(freed)

    def _record_tier_metrics(self) -> None:
        """Publish L1 cache metrics and the cumulative L2 read counter."""
        transfer = self._transfer
        if transfer is None:
            return
        stats = transfer.stats
        current_hits = stats.l1_hits
        current_misses = stats.l1_misses
        current_l2_reads = stats.l2_reads
        l2_write_bytes = stats.l2_write_bytes - getattr(self, "_prev_l2_write_bytes", 0)
        if l2_write_bytes > 0:
            self._metrics.counter(
                "daser_l2_write_bytes_total", "Bytes written to the L2 storage tier."
            ).inc(l2_write_bytes)
        self._prev_l2_write_bytes = stats.l2_write_bytes
        held_evicts = stats.held_evict_persists - getattr(
            self, "_prev_held_evict_persists", 0
        )
        if held_evicts > 0:
            self._metrics.counter(
                "daser_held_writebacks_total",
                "Exclusive-tier records written to L2 when L1 evicted them.",
            ).inc(held_evicts)
        self._prev_held_evict_persists = stats.held_evict_persists
        timing = getattr(transfer, "l2_timing_us", None) or {}
        prev_timing = getattr(self, "_prev_l2_timing_us", {})
        for key, current in timing.items():
            delta = current - prev_timing.get(key, 0)
            if delta > 0:
                kind, phase = key.split("_")
                self._metrics.counter(
                    "daser_l2_executor_seconds_total",
                    "L2 executor queue wait and native IO time.",
                ).inc(delta / 1_000_000, labels={"kind": kind, "phase": phase})
        self._prev_l2_timing_us = timing
        contention = getattr(transfer, "l2_read_contention", None) or {}
        prev_contention = getattr(self, "_prev_l2_read_contention", {})
        for (bucket, writing), current in contention.items():
            labels = {"inflight": str(bucket), "write": str(writing).lower()}
            prev = prev_contention.get((bucket, writing), (0, 0, 0))
            for name, description, delta in (
                (
                    "daser_l2_read_jobs_total",
                    "L2 read jobs by read concurrency and write activity at start.",
                    current[0] - prev[0],
                ),
                (
                    "daser_l2_read_io_seconds_total",
                    "L2 read native IO time by read concurrency and write activity.",
                    (current[1] - prev[1]) / 1_000_000,
                ),
                (
                    "daser_l2_read_bytes_total",
                    "L2 bytes read by read concurrency and write activity.",
                    current[2] - prev[2],
                ),
            ):
                if delta > 0:
                    self._metrics.counter(name, description).inc(delta, labels=labels)
        self._prev_l2_read_contention = contention
        for cause, current in (
            ("l2_write", stats.promotion_wait_l2_write_us),
            ("promotion", stats.promotion_wait_promotion_us),
            ("pool", stats.promotion_wait_pool_us),
        ):
            delta = current - self._prev_promotion_wait_us.get(cause, 0)
            if delta > 0:
                self._metrics.counter(
                    "daser_promotion_wait_seconds_total",
                    "Time L2 promotions waited for L1 space, by what they waited on.",
                ).inc(delta / 1e6, labels={"cause": cause})
            self._prev_promotion_wait_us[cause] = current
        for outcome, current in (
            ("read", stats.readahead_read_bytes),
            ("served", stats.readahead_served_bytes),
            ("dropped", stats.readahead_dropped_bytes),
            ("contended", stats.readahead_contended_bytes),
            ("write_contended", stats.l2_write_contended_bytes),
        ):
            delta = current - self._prev_readahead_bytes.get(outcome, 0)
            if delta > 0:
                self._metrics.counter(
                    "daser_readahead_bytes_total",
                    "Lookup readahead bytes by outcome.",
                ).inc(delta, labels={"outcome": outcome})
            self._prev_readahead_bytes[outcome] = current
        for phase in _LOAD_PHASES:
            current = getattr(stats, f"load_{phase}_us")
            delta = current - self._prev_load_phase_us.get(phase, 0)
            if delta > 0:
                self._metrics.counter(
                    "daser_load_phase_seconds_total",
                    "Grouped-load wall time by phase.",
                ).inc(delta / 1_000_000, labels={"phase": phase})
            self._prev_load_phase_us[phase] = current
        for kind, current in (
            ("all", stats.l1_evicted_bytes),
            ("pinned", stats.l1_evicted_pinned_bytes),
            ("unread", stats.l1_evicted_unread_bytes),
        ):
            delta = current - self._prev_l1_evicted.get(kind, 0)
            if delta > 0:
                self._metrics.counter(
                    "daser_l1_evicted_bytes_total",
                    "Bytes L1 replacement evicted: all, still pinned, never read.",
                ).inc(delta, labels={"kind": kind})
            self._prev_l1_evicted[kind] = current
        for unit, current in (
            ("copies", stats.dst_copies),
            ("runs", stats.dst_copy_runs),
            ("bytes", stats.dst_copy_bytes),
        ):
            delta = current - self._prev_dst_copy.get(unit, 0)
            if delta > 0:
                self._metrics.counter(
                    "daser_dst_copy_total",
                    "L1-to-destination copy submissions, runs and bytes.",
                ).inc(delta, labels={"unit": unit})
            self._prev_dst_copy[unit] = current
        lease_wait_delta = stats.store_lease_wait_us - self._prev_store_lease_wait_us
        if lease_wait_delta > 0:
            self._metrics.counter(
                "daser_store_lease_wait_seconds_total",
                "Time stores waited for a request lease to retain their range.",
            ).inc(lease_wait_delta / 1e6)
        self._prev_store_lease_wait_us = stats.store_lease_wait_us
        self._metrics.gauge(
            "daser_store_lease_wait_max_seconds",
            "Longest lease wait of a single store.",
        ).set(stats.store_lease_wait_max_us / 1e6)
        prev_hits = getattr(self, "_prev_l1_hits", 0)
        prev_misses = getattr(self, "_prev_l1_misses", 0)
        prev_l2_reads = getattr(self, "_prev_l2_reads", 0)
        delta_hits = current_hits - prev_hits
        delta_misses = current_misses - prev_misses
        delta_l2_reads = current_l2_reads - prev_l2_reads
        if delta_hits > 0:
            self._metrics.counter("daser_l1_hits_total", "L1 memory cache hits.").inc(
                delta_hits
            )
        if delta_misses > 0:
            self._metrics.counter(
                "daser_l1_misses_total", "L1 memory cache misses."
            ).inc(delta_misses)
        if delta_l2_reads > 0:
            self._metrics.counter(
                "daser_l2_reads_total", "Reads served from the L2 storage tier."
            ).inc(delta_l2_reads)
        self._prev_l1_hits = current_hits
        self._prev_l1_misses = current_misses
        self._prev_l2_reads = current_l2_reads
        l1_used = transfer.l1_bytes_used
        l1_capacity = int(self._runtime_config.get("l1_size_bytes", 0))
        self._metrics.gauge("daser_l1_bytes_used", "L1 memory cache bytes in use.").set(
            l1_used
        )
        self._metrics.gauge(
            "daser_l1_bytes_capacity", "L1 memory cache total capacity."
        ).set(l1_capacity)
        if self._packed_tiers:
            layout = self._core.packed_layout
            self._metrics.gauge(
                "daser_packed_l2_charged_bytes",
                "Rank-0 packed bytes counted against the L2 budget.",
            ).set(layout.charged_bytes())
            self._metrics.gauge(
                "daser_packed_l1_held_bytes",
                "Packed bytes indexed only through L1 residency.",
            ).set(layout.held_bytes)
            self._metrics.gauge(
                "daser_packed_arena_free_bytes",
                "Unreserved bytes of the rank-0 packed arena lane.",
            ).set(self._core.packed_layout.free_bytes())

    def _ensure_transfer(self) -> TransferLayer:
        """Return the server-owned transfer layer, creating it on first use.

        Protected by a threading lock so that concurrent calls from the
        eager ``initialize_transfer`` thread-pool path and the event-loop
        IPC request path cannot create duplicate transfer layer instances.
        """
        if self._transfer is not None:
            return self._transfer
        with self._transfer_lock:
            if self._transfer is not None:
                return self._transfer
            mode = str(self._runtime_config.get("transfer_mode", "gds"))
            path = str(self._runtime_config.get("store_path", ""))
            skip_l2 = bool(self._runtime_config.get("skip_l2", False))
            if mode == "gds":
                if skip_l2:
                    raise ValueError("skip_l2 is incompatible with gds transfer")
                from daser.transfer.gds import GDSTransferLayer

                self._transfer = GDSTransferLayer(path)
            elif mode == "iouring":
                l2_bytes = int(
                    self._runtime_config.get(
                        "l2_size_bytes",
                        self._runtime_config.get("total_store_bytes", 0),
                    )
                )
                if l2_bytes <= 0:
                    slot_size = int(self._runtime_config.get("slot_size", 0))
                    total_slots = int(self._runtime_config.get("total_slots", 0))
                    l2_bytes = slot_size * total_slots
                self._transfer = TieredIOUringTransferLayer(
                    path=path,
                    l1_bytes=int(self._runtime_config.get("l1_size_bytes", l2_bytes)),
                    l2_bytes=l2_bytes,
                    skip_l2=skip_l2,
                    bip_enabled=bool(self._runtime_config.get("bip_enabled", False)),
                    l1_accounting=str(
                        self._runtime_config.get("l1_accounting", "stored")
                    ),
                    prefetch_lease_budget_bytes=int(
                        self._runtime_config.get("prefetch_lease_budget_bytes", 0)
                    ),
                    l2_read_stripe_bytes=int(
                        self._runtime_config.get("l2_read_stripe_bytes", 0)
                    ),
                    readahead_piece_bytes=self._readahead_piece,
                    readahead_window=int(
                        os.environ.get(LOAD_READAHEAD_WINDOW_ENV, "4") or 4
                    ),
                    l2_write_gate=os.environ.get(L2_WRITE_GATE_ENV) != "0",
                    readahead_budget_fraction=float(
                        os.environ.get(READAHEAD_BUDGET_FRACTION_ENV) or 0.5
                    ),
                )
            else:
                raise ValueError(f"unknown transfer_mode: {mode}")
            if self._packed_tiers and self._transfer.supports_held_stores:
                self._core.attach_l1_tier(
                    self._transfer.hold_resident, self._transfer.discard_held
                )
                self._transfer.set_held_release_listener(self._on_held_release)
        return self._transfer

    def _on_held_release(self, file_offset: int, nbytes: int) -> None:
        """Return a held record that left L1 to the L2 FIFO.

        Exclusive tiers enter it as newest (it was just demoted); inclusive
        tiers as oldest, so the next reclaim drops it as LMCache would.
        """
        self._core.packed_layout.charge_held(
            file_offset, nbytes, newest=self._exclusive_tiers
        )

    def _chunk_payloads(self, chunks: list[ChunkInfo]) -> list[dict[str, Any]]:
        """Serialize lookup chunks and attach published online packed slot refs."""
        payloads: list[dict[str, Any]] = []
        for chunk in chunks:
            payload = chunk.to_dict()
            if (
                self._runtime_config.get("storage_format")
                == STORAGE_FORMAT_COMPRESSED_ONLINE
            ):
                online_refs = self._core.packed_slot_refs(
                    chunk.start_slot,
                    chunk.num_slots,
                )
                if len(online_refs) == chunk.num_slots:
                    payload["compressed_slots"] = online_refs
            payloads.append(payload)
        return payloads

    def _external_spans(
        self,
        chunks: list[ChunkInfo],
        num_computed_tokens: int,
        hits: int,
    ) -> list[dict[str, int]]:
        """Return the physical spans a worker load of ``hits`` tokens reads."""
        block_tokens = int(self._runtime_config.get("block_tokens", 0))
        if self._runtime_config.get("storage_format") == (
            STORAGE_FORMAT_COMPRESSED_ONLINE
        ):
            return self._online_compressed_prefetch_spans(
                chunks,
                external_start=num_computed_tokens,
                external_tokens=hits,
                block_tokens=block_tokens,
            )
        return _prefetch_spans_from_chunks(
            chunks,
            external_start=num_computed_tokens,
            external_tokens=hits,
            block_tokens=block_tokens,
            slot_size=int(self._runtime_config.get("slot_size", 0)),
            tensor_parallel_size=int(
                self._runtime_config.get("tensor_parallel_size", 1)
            ),
            rank_stride_bytes=int(self._runtime_config.get("rank_stride_bytes", 0)),
        )

    def _chunk_spans(self, chunk: ChunkInfo) -> list[dict[str, int]]:
        """Return the physical spans that store one whole chunk."""
        block_tokens = int(self._runtime_config.get("block_tokens", 0))
        window = {
            "external_start": int(chunk.target_token_start),
            "external_tokens": int(chunk.token_count),
            "block_tokens": block_tokens,
        }
        if self._runtime_config.get("storage_format") == (
            STORAGE_FORMAT_COMPRESSED_ONLINE
        ):
            return self._online_compressed_prefetch_spans([chunk], **window)
        return _prefetch_spans_from_chunks(
            [chunk],
            **window,
            slot_size=int(self._runtime_config.get("slot_size", 0)),
            tensor_parallel_size=int(
                self._runtime_config.get("tensor_parallel_size", 1)
            ),
            rank_stride_bytes=int(self._runtime_config.get("rank_stride_bytes", 0)),
        )

    def _online_compressed_prefetch_spans(
        self,
        chunks: list[ChunkInfo],
        *,
        external_start: int,
        external_tokens: int,
        block_tokens: int,
    ) -> list[dict[str, int]]:
        """Translate an online packed window into exact physical read spans.

        Args:
            chunks: Committed lookup chunks covering the prompt prefix.
            external_start: Token offset where external cache loading begins.
            external_tokens: Number of externally admitted tokens.
            block_tokens: Tokens stored in one logical cache slot.

        Returns:
            Packed file offsets and lengths for the aligned external window.
            An empty list is returned when a committed chunk has not published
            every packed slot yet; callers then skip lease-based prefetch until
            the immutable online metadata is complete.

        Async/thread-safety:
            Runs on the server asyncio event loop and only reads control-plane
            metadata. It performs no I/O or blocking operations.
        """
        if block_tokens <= 0:
            raise ValueError("block_tokens must be positive for prefetch lookup")
        local_slot_size = int(
            self._runtime_config.get(
                "local_slot_size",
                int(self._runtime_config.get("slot_size", 0))
                // max(
                    1,
                    int(self._runtime_config.get("tensor_parallel_size", 1)),
                ),
            )
        )
        if local_slot_size <= 0:
            raise ValueError("local_slot_size must be positive for prefetch lookup")
        if external_tokens <= 0 or external_start % block_tokens != 0:
            return []
        external_end = external_start + external_tokens
        spans: list[dict[str, int]] = []
        for chunk in sorted(chunks, key=lambda item: item.target_token_start):
            target_start = int(chunk.target_token_start)
            target_end = target_start + int(chunk.token_count)
            load_start, load_end = _external_load_window(
                target_start, target_end, external_start, external_end, block_tokens
            )
            if load_end <= load_start:
                continue
            start_slot = int(chunk.start_slot) + (
                (load_start - target_start) // block_tokens
            )
            num_slots = (load_end - load_start) // block_tokens
            refs = self._core.packed_slot_refs(start_slot, num_slots)
            if len(refs) != num_slots:
                return []
            spans.extend(
                {
                    "file_offset": int(ref["file_offset"]),
                    "nbytes": int(ref["stored_length"]),
                    "accounted_nbytes": local_slot_size,
                }
                for ref in refs
            )
        return spans

    def _payload_buffer(self, payload: dict[str, Any]) -> Any:
        """Return a byte-addressable buffer for an IPC transfer payload."""
        if "data" in payload:
            return bytearray(payload["data"])
        if "load_staging_buffer_index" in payload:
            buffer_index = int(payload["load_staging_buffer_index"])
            producer_pid = int(payload["producer_pid"])
            buffer_key = (producer_pid, buffer_index)
            try:
                return self._load_staging_buffers[buffer_key]
            except KeyError as exc:
                raise ValueError(
                    "unknown load staging buffer: "
                    f"producer_pid={producer_pid} index={buffer_index}"
                ) from exc
        if "cuda_ipc_handle" in payload:
            return self._open_cuda_ipc_payload(payload=payload, nbytes_key="nbytes")
        raise ValueError("transfer payload requires data or cuda_ipc_handle")

    def _open_cuda_ipc_payload(
        self,
        *,
        payload: dict[str, Any],
        nbytes_key: str,
        cache_mapping: bool = True,
    ) -> "_CachedCudaArray":
        """Open or reuse a CUDA IPC payload mapping.

        Args:
            payload: CUDA IPC payload with handle, device, allocation base, and
                offset fields.
            nbytes_key: Payload field that identifies the mapping size.
            cache_mapping: If True, reuse the general CUDA IPC LRU cache. Fixed
                registered staging buffers pass False because their lifetime is
                owned by ``_load_staging_buffers`` and must not be LRU-evicted.

        Returns:
            Cached or one-shot CUDA array wrapper for the mapped allocation.
        """
        local_ptr = None
        producer_pid = int(payload.get("producer_pid", -1))
        device_ptr = int(payload["device_ptr"])
        nbytes = int(payload[nbytes_key])
        pci_bus_id = (
            str(payload["device_pci_bus_id"])
            if "device_pci_bus_id" in payload
            else None
        )
        allocation_offset = int(payload.get("allocation_offset", 0))
        allocation_base_ptr = int(
            payload.get("allocation_base_ptr", device_ptr - allocation_offset)
        )
        if producer_pid == os.getpid():
            local_ptr = allocation_base_ptr
        if local_ptr is None and cache_mapping:
            key = (
                producer_pid,
                allocation_base_ptr,
                nbytes + allocation_offset,
                pci_bus_id,
            )
            cached = self._cuda_ipc_cache.get(key)
            if cached is None:
                self._evict_cuda_ipc_cache_if_needed()
                opened = open_cuda_ipc_buffer(
                    handle=payload["cuda_ipc_handle"],
                    nbytes=nbytes,
                    pci_bus_id=pci_bus_id,
                    local_ptr=None,
                    allocation_offset=allocation_offset,
                )
                cached = _CachedCudaArray(opened)
                self._cuda_ipc_cache[key] = cached
            else:
                self._cuda_ipc_cache.move_to_end(key)
            return cached
        opened = open_cuda_ipc_buffer(
            handle=payload["cuda_ipc_handle"],
            nbytes=nbytes,
            pci_bus_id=pci_bus_id,
            local_ptr=local_ptr,
            allocation_offset=allocation_offset,
        )
        return _UncachedCudaArray(opened)

    def _evict_cuda_ipc_cache_if_needed(self) -> None:
        """Evict one cached CUDA IPC mapping when the cache is full."""
        if len(self._cuda_ipc_cache) < _CUDA_IPC_CACHE_LIMIT:
            return
        _key, cached = self._cuda_ipc_cache.popitem(last=False)
        cached.close()

    def _close_cuda_ipc_cache(self) -> None:
        """Close all cached CUDA IPC mappings."""
        for cached in self._load_staging_buffers.values():
            cached.close()
        self._load_staging_buffers.clear()
        for cached in self._store_staging_buffers.values():
            cached.close()
        self._store_staging_buffers.clear()
        for cached in self._cuda_ipc_cache.values():
            cached.close()
        self._cuda_ipc_cache.clear()

    def _payload_size(
        self, payload: dict[str, Any], spans: list[dict[str, Any]]
    ) -> int:
        """Return destination payload size for transfer_load."""
        if "nbytes" in payload:
            return int(payload["nbytes"])
        max_end = 0
        for span in spans:
            max_end = max(
                max_end,
                int(span.get("target_offset", 0)) + int(span["nbytes"]),
            )
        return max_end


class _CachedCudaArray:
    """Sliceable wrapper for a cached CUDA IPC buffer."""

    def __init__(self, opened: Any) -> None:
        self._opened = opened
        self._copy_stream: Any | None = None

    def __getitem__(self, item: Any) -> Any:
        """Return a CuPy array slice."""
        return self._opened.array[item]

    def synchronize(self) -> None:
        """Synchronize CUDA writes issued through the opened array."""
        import cupy
        from cupy.cuda import runtime

        with cupy.cuda.Device(int(self._opened.array.device.id)):
            stream_ptr = (
                int(self._copy_stream.ptr) if self._copy_stream is not None else 0
            )
            runtime.streamSynchronize(stream_ptr)

    @property
    def nbytes(self) -> int:
        """Return the byte length of the mapped CUDA view."""
        return int(self._opened.array.nbytes)

    @property
    def copy_stream_ptr(self) -> int:
        """Return the private stream used for asynchronous H2D copies.

        Returns:
            CUDA stream pointer, or zero for non-CUDA test doubles.

        Async/thread-safety:
            Called on the IPC server event loop while a destination mapping is
            being prepared. Stream creation is lazy and each mapping is used
            by one staging lease at a time, so no cross-request locking is
            needed here.
        """
        array = self._opened.array
        if getattr(getattr(array, "data", None), "ptr", None) is None:
            return 0
        if self._copy_stream is None:
            import cupy

            with cupy.cuda.Device(int(array.device.id)):
                self._copy_stream = cupy.cuda.Stream(non_blocking=True)
        return int(self._copy_stream.ptr)

    def close(self) -> None:
        """Close the CUDA IPC handle."""
        self._opened.close()


class _UncachedCudaArray(_CachedCudaArray):
    """Sliceable wrapper that owns a one-shot CUDA IPC buffer."""
