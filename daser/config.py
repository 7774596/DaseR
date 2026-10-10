# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, replace
import json
import os

from daser.compression.format import ONLINE_TILE_SCALARS, CompressedStoreGeometry

BLOCK_TOKENS = 16
DEFAULT_IOURING_L1_BYTES = 1024 * 1024 * 1024

#: Cache reuse mode selecting the retrieval index + position encoder pair.
CACHE_REUSE_PREFIX = "prefix"
CACHE_REUSE_CHUNK = "chunk"
CACHE_REUSE_MODES = (CACHE_REUSE_PREFIX, CACHE_REUSE_CHUNK)
DEFAULT_CACHE_REUSE_MODE = CACHE_REUSE_CHUNK
STORAGE_FORMAT_RAW = "raw"
STORAGE_FORMAT_COMPRESSED_ONLINE = "compressed-online"
STORAGE_FORMATS = (
    STORAGE_FORMAT_RAW,
    STORAGE_FORMAT_COMPRESSED_ONLINE,
)
L1_ACCOUNTING_STORED = "stored"
L1_ACCOUNTING_RAW = "raw"
L1_ACCOUNTING_MODES = (L1_ACCOUNTING_STORED, L1_ACCOUNTING_RAW)
#: Research switch for CLOCK second-chance eviction of the packed arena.
ARENA_SECOND_CHANCE_ENV = "DASER_ARENA_SECOND_CHANCE"
#: Logical ring headroom under second chance. Spared chunks leave skip holes
#: behind the hand, so the ring must outlast the byte arena or its FIFO
#: allocation eviction removes referenced chunks before the hand sees them.
ARENA_SECOND_CHANCE_SLOT_FACTOR = 2
#: Research switch for loop-aware eviction of the packed arena: a positive
#: integer N makes chunks idle for more than N prompts go first (oldest
#: allocation first), otherwise the newest chunk outside the current prompt.
ARENA_LIVE_VICTIM_ENV = "DASER_ARENA_LIVE_VICTIM_IDLE"
#: Logical ring headroom under live-victim eviction. Kept prefixes stay at the
#: ring tail while churn happens near the head, so the ring must span several
#: arenas or FIFO slot reuse sweeps the kept prefixes away.
ARENA_LIVE_VICTIM_SLOT_FACTOR = 16
#: Research switch for prefix-closed eviction: when FIFO capacity eviction
#: removes a chunk, also remove its resident descendants, which no lookup can
#: reach until the chunk is stored again.
PREFIX_CLOSED_EVICTION_ENV = "DASER_PREFIX_CLOSED_EVICTION"
#: Research switch for exclusive L1/L2 tiers under compressed-online storage.
#: New packed records live only in L1 and reach L2 when L1 evicts them, so the
#: packed arena spans L1 + L2 while L2 holds only records that left L1.
EXCLUSIVE_TIERS_ENV = "DASER_EXCLUSIVE_TIERS"
#: Research control for compressed-online storage: ``1`` makes the worker
#: store every packed record through the raw fallback, so the packed layout
#: and exclusive tiers can be measured without compression.
PACKED_FORCE_RAW_ENV = "DASER_PACKED_FORCE_RAW"
#: Research switch: ``1`` compiles the 256-scalar compressed load kernel with
#: ``uint32`` word reads of the packed record fields instead of byte loads.
LOAD_WORD_READS_ENV = "DASER_LOAD_WORD_READS"

#: Research switch: a positive integer ``K`` makes each 256-scalar compressed
#: load CTA decode ``K`` consecutive eight-tile groups of one plane (unset = 1).
LOAD_TILE_GROUPS_ENV = "DASER_LOAD_TILE_GROUPS"

#: Research switch: ``1`` makes the 256-scalar compressed load kernel keep the
#: plane codebook in registers and select escape bytes without memory reads.
LOAD_REGISTER_CODEBOOK_ENV = "DASER_LOAD_REGISTER_CODEBOOK"

#: Research switch: ``1`` makes the 256-scalar compressed load kernel write
#: each lane's eight decoded scalars with one 128-bit store.
LOAD_VECTOR_STORES_ENV = "DASER_LOAD_VECTOR_STORES"
#: Research switch for striped L2 reads: copy each stripe part to the
#: destination as soon as its completion arrives instead of after the extent.
STRIPE_PIPELINE_ENV = "DASER_STRIPE_PIPELINE"
#: Research switch for lookup-triggered L2 readahead: read group size in
#: bytes (0 or unset disables), and the number of groups kept in flight.
LOAD_READAHEAD_ENV = "DASER_LOAD_READAHEAD"
LOAD_READAHEAD_WINDOW_ENV = "DASER_LOAD_READAHEAD_WINDOW"
#: Research switch: ``1`` makes a load copy each readahead piece as soon as it
#: is retained, overlapping host-to-device copies with the remaining reads.
LOAD_READAHEAD_STREAM_ENV = "DASER_LOAD_READAHEAD_STREAM"
#: Research switch: ``1`` makes the scheduler connector send a readahead hint
#: when a request enters vLLM's waiting queue, so its SSD read overlaps the
#: requests ahead of it instead of starting when it is scheduled.
ARRIVAL_READAHEAD_ENV = "DASER_ARRIVAL_READAHEAD"
#: Research switch: positive block count makes the scheduler connector hold a
#: finished request's vLLM blocks and write its prompt KV to DaseR only when
#: free GPU blocks fall below this count (write-back instead of write-through).
WRITEBACK_LOW_BLOCKS_ENV = "DASER_WRITEBACK_LOW_BLOCKS"
#: Research switch: ``1`` makes the scheduler connector also store the full
#: KV blocks a finished request computed past its prompt (the generated
#: tokens), so a next turn whose prompt replays that output can load it.
DECODE_SAVE_ENV = "DASER_DECODE_SAVE"
#: Research switch: ``1`` logs a monotonic timestamp per request at its first
#: lookup, its block allocation and its load completion (scheduler side).
ADMIT_TRACE_ENV = "DASER_ADMIT_TRACE"
#: Research switch: issue L1-to-GPU copies with at least this many
#: discontiguous runs as one zero-copy gather kernel (0 or unset disables).
GATHER_MIN_RUNS_ENV = "DASER_GATHER_MIN_RUNS"
#: Research switch: layer-wise ``transfer_load_runs`` copy mode, ``gather``
#: (SM gather kernel, default), ``memcpy`` or ``batch`` (copy engines). The
#: copies run while the forward pass occupies the SMs.
LAYERWISE_COPY_ENV = "DASER_LAYERWISE_COPY"
#: Research switch: layer-wise restore slots per staging buffer (default 1).
#: More slots let later layers' transfers run while earlier layers restore;
#: a request gets as many as its per-layer bytes fit.
LAYERWISE_SLOTS_ENV = "DASER_LAYERWISE_SLOTS"
#: Research switch: ``1`` times how long the forward stream idles at each
#: layer's restore and logs the sum per layer-wise load as ``gpu_stall_ms``.
LAYERWISE_GPU_STALL_ENV = "DASER_LAYERWISE_GPU_STALL"
#: Research switch: ``1`` submits each request-level L1-to-GPU copy as one
#: ``cudaMemcpyBatchAsync`` over all its runs instead of one copy per run.
BATCH_DST_COPY_ENV = "DASER_BATCH_DST_COPY"
#: Research switch: ``1`` replaces the L1 policy with predicted-next-use
#: eviction from per-L2-region reuse intervals.
L1_NEXT_USE_ENV = "DASER_L1_NEXT_USE"
#: Research switch: ``1`` makes L1 replacement skip allocations an L2 writer
#: or lease still owns instead of dropping their residency.
L1_SKIP_PINNED_ENV = "DASER_L1_SKIP_PINNED"
#: Research switch: ``1`` makes readaheads share one window of reads in
#: flight, granted to the readahead with the fewest bytes left.
READAHEAD_SRPT_ENV = "DASER_READAHEAD_SRPT"
#: Research switch: ``1`` holds L2 writes while a readahead is reading (up to
#: a bounded delay), so writeback does not share the SSD with the next load.
L2_WRITE_YIELD_ENV = "DASER_L2_WRITE_YIELD"
#: Research switch: ``1`` runs L2 writes on their own executor in 16 MiB
#: pieces that pause (bounded) while any L2 read is in native IO.
L2_WRITE_GATE_ENV = "DASER_L2_WRITE_GATE"
#: Research switch: ``1`` reads load spans no readahead covers straight to the
#: destination through transient windows while the load waits for readahead.
READAHEAD_TAIL_BYPASS_ENV = "DASER_READAHEAD_TAIL_BYPASS"
#: Research switch: fraction in (0, 1) of L1 that held (not yet written)
#: ranges may fill before the oldest are written to L2 while no readahead is
#: reading, so later evictions under a readahead free clean memory instead of
#: writing to the SSD the readahead reads from.
HELD_PRECLEAN_FRACTION_ENV = "DASER_HELD_PRECLEAN_FRACTION"
#: Research switch: ``1`` predicts the agent session that returns next from
#: the lookup stream and reads its stored prefix ahead after each load, so the
#: SSD read overlaps the current request's prefill and decode.
SESSION_PREFETCH_ENV = "DASER_SESSION_PREFETCH"
#: Research switch: number of upcoming sessions the next-session prefetch
#: reads ahead (default 1), soonest first, within the readahead budget.
SESSION_PREFETCH_DEPTH_ENV = "DASER_SESSION_PREFETCH_DEPTH"
#: Research switch: chance in [0, 1] that each next-session prediction is
#: kept rather than replaced by a random wrong session (default 1.0).
SESSION_PREDICT_ACCURACY_ENV = "DASER_SESSION_PREDICT_ACCURACY"
#: Research switch: share of L1 that readahead leases may retain at once
#: (default 0.5).
READAHEAD_BUDGET_FRACTION_ENV = "DASER_READAHEAD_BUDGET_FRACTION"
#: Research switch: bytes of a GPU buffer the next-session prefetch fills
#: with the soonest predicted session's stored bytes (default 0 = off).
DEVICE_STASH_BYTES_ENV = "DASER_DEVICE_STASH_BYTES"
#: Research switch: predicted sessions the device stash holds, soonest first
#: (default 1).
DEVICE_STASH_SESSIONS_ENV = "DASER_DEVICE_STASH_SESSIONS"
#: Research switch: readaheads skip spans the device stash holds whole.
READAHEAD_SKIP_STASHED_ENV = "DASER_READAHEAD_SKIP_STASHED"
#: Research switch: byte phases K of a streaming lease prefetch (default 0 =
#: off). The prefetch replies once the lease is admitted and resolved; L2
#: misses fill in the background phase by phase (bytes [k/K, (k+1)/K) of
#: every record), and leased run loads wait until their runs are covered.
STREAMING_ADMISSION_PHASES_ENV = "DASER_STREAMING_ADMISSION_PHASES"
#: Research switch for split residency (``--l1-split-fraction``): ``1`` reads
#: L2 misses outside the protected class through transient pinned windows that
#: never enter L1, so SSD traffic cannot evict the protected segment.
L1_SPLIT_BYPASS_ENV = "DASER_L1_SPLIT_BYPASS"


@dataclass(frozen=True)
class ModelGeometry:
    """KV cache geometry derived from a HuggingFace model config.

    Attributes:
        num_kv_heads: number of KV attention heads.
        head_dim: per-head hidden dimension.
        num_layers: transformer layer count.
        dtype_bytes: bytes per scalar element.
        dtype_name: normalized scalar dtype from the model configuration.
    """

    num_kv_heads: int
    head_dim: int
    num_layers: int
    dtype_bytes: int
    dtype_name: str

    def local_slot_size_for_block_tokens(
        self, block_tokens: int, tensor_parallel_size: int = 1
    ) -> int:
        """Return bytes stored by one TP rank for one vLLM KV block.

        Args:
            block_tokens: Number of tokens in each vLLM KV block.
            tensor_parallel_size: Number of vLLM tensor-parallel ranks.

        Returns:
            Bytes required by one rank for one KV slot.

        Raises:
            ValueError: if ``tensor_parallel_size`` is not positive.
        """
        if tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if self.num_kv_heads >= tensor_parallel_size:
            compatible = self.num_kv_heads % tensor_parallel_size == 0
        else:
            compatible = tensor_parallel_size % self.num_kv_heads == 0
        if not compatible:
            raise ValueError(
                "tensor_parallel_size must divide num_kv_heads or be divisible "
                "by num_kv_heads"
            )
        local_kv_heads = max(1, self.num_kv_heads // tensor_parallel_size)
        return (
            local_kv_heads
            * self.head_dim
            * 2  # K and V
            * self.num_layers
            * block_tokens
            * self.dtype_bytes
        )

    def slot_size_for_block_tokens(
        self, block_tokens: int, tensor_parallel_size: int = 1
    ) -> int:
        """Return bytes required for one vLLM KV block across all layers.

        Args:
            block_tokens: Number of tokens in each vLLM KV block.
            tensor_parallel_size: Number of vLLM tensor-parallel ranks.

        Returns:
            Aggregate bytes required for one KV slot across all TP ranks.
        """
        return (
            self.local_slot_size_for_block_tokens(block_tokens, tensor_parallel_size)
            * tensor_parallel_size
        )


def arena_live_victim_idle() -> int:
    """Return the live-victim idle threshold in prompts, 0 when disabled.

    Reads ``DASER_ARENA_LIVE_VICTIM_IDLE``; unset, empty or non-positive values
    disable loop-aware arena eviction.

    Returns:
        Idle threshold N (> 0) or 0.
    """
    value = os.environ.get(ARENA_LIVE_VICTIM_ENV, "")
    return max(0, int(value)) if value else 0


def _dtype_bytes(dtype: object) -> int:
    """Return storage bytes for a HuggingFace dtype string.

    Args:
        dtype: raw dtype value from model config.

    Returns:
        Number of bytes per element.
    """
    dtype_str = str(dtype or "").lower()
    if "float8" in dtype_str or "fp8" in dtype_str:
        raise ValueError("FP8 KV cache geometry is not supported")
    if "float32" in dtype_str or "fp32" in dtype_str:
        return 4
    return 2


def _dtype_name(dtype: object) -> str:
    """Return a normalized HuggingFace scalar dtype name.

    Args:
        dtype: Raw dtype value from model config.

    Returns:
        Lower-case dtype name without a ``torch.`` prefix.

    Async/thread-safety:
        Pure helper safe to call from any thread.
    """
    name = str(dtype or "").lower().removeprefix("torch.")
    aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    return aliases.get(name, name)


def model_geometry_from_path(model_path: str) -> ModelGeometry:
    """Derive KV cache geometry from ``model_path/config.json``.

    Args:
        model_path: HuggingFace model directory containing ``config.json``.

    Returns:
        ModelGeometry used by the DaseR server runtime config.

    Raises:
        ValueError: if required model geometry fields are missing or invalid.
    """
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, encoding="utf-8") as f:
        payload = json.load(f)

    hidden_size = int(payload.get("hidden_size", 0))
    num_attention_heads = int(payload.get("num_attention_heads", 0))
    num_kv_heads = int(
        payload.get("num_key_value_heads")
        or payload.get("multi_query_group_num")
        or num_attention_heads
    )
    num_layers = int(
        payload.get("num_hidden_layers")
        or payload.get("num_layers")
        or payload.get("n_layer")
        or 0
    )
    head_dim = int(payload.get("head_dim") or payload.get("kv_channels") or 0)
    if head_dim == 0 and hidden_size > 0 and num_attention_heads > 0:
        head_dim = hidden_size // num_attention_heads

    missing = [
        name
        for name, value in [
            ("num_key_value_heads", num_kv_heads),
            ("head_dim", head_dim),
            ("num_hidden_layers", num_layers),
        ]
        if value <= 0
    ]
    if missing:
        raise ValueError(
            f"model config {config_path} is missing valid KV geometry: "
            + ", ".join(missing)
        )

    raw_dtype = payload.get("torch_dtype")
    return ModelGeometry(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        dtype_bytes=_dtype_bytes(raw_dtype),
        dtype_name=_dtype_name(raw_dtype),
    )


@dataclass
class DaserConfig:
    """Top-level configuration for DaseR.

    Paths are derived from ``store_dir`` so the server owns all KV store
    parameters and can expose them to the connector over IPC.

    Attributes:
        model_path: HuggingFace model path used by vLLM and tokenizer loading.
        vllm_model_id: model identifier served by vLLM's OpenAI API.
        store_dir: directory containing DaseR data files.
        total_store_bytes: total KV store capacity.
        total_slots: number of fixed-size slots in the ring buffer.
        ipc_socket_path: Unix socket path for Connector <-> Server IPC.
        log_level: log level string passed to init_logger.
        block_tokens: tokens per vLLM block (default 16).
        model_id: identifier string for the model, used to prevent
                  cross-model cache reuse.
        cache_reuse_mode: retrieval/position strategy selected at startup.
        transfer_mode: server-owned transfer layer selected at startup.
        l1_size_bytes: memory-tier capacity for iouring mode.
        skip_l2: when True, keep all transfer bytes in volatile L1 memory and
            do not allocate or persist SSD-tier store files.
        tensor_parallel_size: vLLM tensor-parallel rank count used by the
            physical KV store layout.
        storage_format: startup-immutable raw or experimental compressed mode.
        bip_enabled: whether the L1 cache uses BIP replacement.
        prefetch_lease_budget_bytes: cap on accounted bytes retained by
            in-flight scheduler prefetch leases in iouring mode (0 = no cap).
        l2_read_stripe_bytes: part size for splitting one large iouring L2
            read into concurrently submitted IOs (0 = one serial read).
        l1_split_fraction: research switch for iouring L1 split residency;
            protected-segment share of L1 (0 = off).
        extra_kv_layers: attention layers the vLLM worker registers beyond
            the model's own, for example an EAGLE drafter's; they share the
            target layers' KV shape and are stored in every slot.
    """

    model_path: str = ""
    vllm_model_id: str = ""
    store_dir: str = "/mnt/xfs/daser"
    total_store_bytes: int = 10 * 1024 * 1024 * 1024
    ipc_socket_path: str = "/tmp/daser.sock"
    log_level: str = "INFO"

    block_tokens: int = BLOCK_TOKENS
    cache_reuse_mode: str = DEFAULT_CACHE_REUSE_MODE
    transfer_mode: str = "iouring"
    l1_size_bytes: int = DEFAULT_IOURING_L1_BYTES
    skip_l2: bool = False
    tensor_parallel_size: int = 1
    storage_format: str = STORAGE_FORMAT_RAW
    bip_enabled: bool = True
    l1_accounting: str = L1_ACCOUNTING_STORED
    prefetch_lease_budget_bytes: int = 0
    l2_read_stripe_bytes: int = 0
    l1_split_fraction: float = 0.0
    extra_kv_layers: int = 0

    def __post_init__(self) -> None:
        """Validate startup-immutable transfer settings."""
        if self.l1_accounting not in L1_ACCOUNTING_MODES:
            raise ValueError(
                f"l1_accounting must be one of {', '.join(L1_ACCOUNTING_MODES)}"
            )
        if self.prefetch_lease_budget_bytes < 0:
            raise ValueError("prefetch_lease_budget_bytes must be non-negative")
        if self.l2_read_stripe_bytes < 0:
            raise ValueError("l2_read_stripe_bytes must be non-negative")
        if not 0.0 <= self.l1_split_fraction < 1.0:
            raise ValueError("l1_split_fraction must be in [0, 1)")
        if self.extra_kv_layers < 0:
            raise ValueError("extra_kv_layers must be non-negative")

    @property
    def store_path(self) -> str:
        """Absolute path to the KV data file."""
        return os.path.join(self.store_dir, "daser.store")

    @property
    def index_path(self) -> str:
        """Absolute path to the serialized metadata index."""
        return os.path.join(self.store_dir, "daser.index")

    @property
    def persists_index(self) -> bool:
        """Whether ``daser.index`` describes the store across restarts.

        Memory-only mode has no durable bytes. Online packed slot records live
        only in server memory and overwrite raw slot envelopes, so a snapshot
        cannot describe a compressed-online store after restart.
        """
        return not self.skip_l2 and self.storage_format == STORAGE_FORMAT_RAW

    @property
    def model_id(self) -> str:
        """Model identifier used for cache isolation."""
        model_id = self.vllm_model_id or self.model_path
        if self.tensor_parallel_size == 1:
            return model_id
        return f"{model_id}::tp{self.tensor_parallel_size}"

    @property
    def total_slots(self) -> int:
        """Return logical ring slots available to the selected storage format."""
        physical_slots = self.physical_total_slots
        if self.storage_format != STORAGE_FORMAT_COMPRESSED_ONLINE:
            return physical_slots
        packed_size = self.resolved_online_packed_slot_size()
        local_bytes = self.resolved_local_slot_size()
        logical_bytes = physical_slots * local_bytes
        slots = max(physical_slots, logical_bytes // packed_size)
        if os.environ.get(ARENA_SECOND_CHANCE_ENV) == "1":
            slots *= ARENA_SECOND_CHANCE_SLOT_FACTOR
        elif arena_live_victim_idle():
            slots *= ARENA_LIVE_VICTIM_SLOT_FACTOR
        return slots

    @property
    def physical_total_slots(self) -> int:
        """Return raw-envelope slots represented by the on-disk L2 file."""
        return self.total_store_bytes // self.resolved_slot_size()

    @property
    def aligned_store_bytes(self) -> int:
        """Return store capacity rounded down to a whole number of slots."""
        return self.physical_total_slots * self.resolved_slot_size()

    def resolved_online_packed_slot_size(self) -> int:
        """Return the minimum aligned size used for logical packed capacity.

        Returns:
            Minimum aligned bytes representable by one compressed-online
            rank-local KV record. Physical extents are allocated from each
            record's actual packed length.

        Raises:
            ValueError: If the model geometry cannot fit the online envelope.
        """
        geometry = self.model_geometry()
        local_slot_size = self.resolved_local_slot_size()
        return CompressedStoreGeometry(
            num_slots=self.physical_total_slots,
            slot_size=local_slot_size,
            block_tokens=self.block_tokens,
            num_layers=geometry.num_layers,
            num_kv_heads=max(1, geometry.num_kv_heads // self.tensor_parallel_size),
            head_dim=geometry.head_dim,
            dtype_bytes=geometry.dtype_bytes,
            tile_scalars=ONLINE_TILE_SCALARS,
        ).online_min_stored_length

    def model_geometry(self) -> ModelGeometry:
        """Return the KV geometry of every layer the worker registers.

        Returns:
            Geometry from ``model_path/config.json`` with ``extra_kv_layers``
            added to its layer count.
        """
        geometry = model_geometry_from_path(self.model_path)
        return replace(geometry, num_layers=geometry.num_layers + self.extra_kv_layers)

    @property
    def l2_size_bytes(self) -> int:
        """Return SSD-tier capacity in bytes."""
        return self.aligned_store_bytes

    def resolved_slot_size(self) -> int:
        """Return bytes per KV slot derived from the model config.

        Returns:
            Slot size in bytes.
        """
        return self.model_geometry().slot_size_for_block_tokens(
            self.block_tokens, self.tensor_parallel_size
        )

    def resolved_local_slot_size(self) -> int:
        """Return bytes stored by one TP rank for one logical KV slot.

        Returns:
            Per-rank KV slot bytes derived from model and TP configuration.
        """
        return self.model_geometry().local_slot_size_for_block_tokens(
            self.block_tokens, self.tensor_parallel_size
        )

    def runtime_config(self) -> dict[str, object]:
        """Return connector runtime config owned by DaseR server.

        Returns:
            Dict returned to DaserConnector over IPC.
        """
        return {
            "socket_path": self.ipc_socket_path,
            "store_path": "" if self.skip_l2 else self.store_path,
            "slot_size": self.resolved_slot_size(),
            "local_slot_size": self.resolved_local_slot_size(),
            "tensor_parallel_size": self.tensor_parallel_size,
            "rank_stride_bytes": self.physical_total_slots
            * self.resolved_local_slot_size(),
            "block_tokens": self.block_tokens,
            "model_id": self.model_id,
            "cache_reuse_mode": self.cache_reuse_mode,
            "transfer_mode": self.transfer_mode,
            "l1_size_bytes": self.l1_size_bytes,
            "l2_size_bytes": self.l2_size_bytes,
            "total_slots": self.total_slots,
            "total_store_bytes": self.l2_size_bytes,
            "skip_l2": self.skip_l2,
            "storage_format": self.storage_format,
            "bip_enabled": self.bip_enabled,
            "l1_accounting": self.l1_accounting,
            "prefetch_lease_budget_bytes": self.prefetch_lease_budget_bytes,
            "l2_read_stripe_bytes": self.l2_read_stripe_bytes,
            "l1_split_fraction": self.l1_split_fraction,
        }
