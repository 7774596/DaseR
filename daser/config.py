# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, replace
import json
import os

BLOCK_TOKENS = 16
DEFAULT_IOURING_L1_BYTES = 1024 * 1024 * 1024
DEFAULT_L2_BYTES = 10 * 1024 * 1024 * 1024
#: Allocation unit of both tiers (``--tier-grain``).
DEFAULT_TIER_GRAIN = 64 * 1024
#: Upper bound of the pinned staging pool carved out of the L1 budget.
MAX_STAGING_BYTES = 256 * 1024 * 1024
_IO_ALIGNMENT = 4096

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
#: Research control for compressed-online storage: ``1`` makes the worker
#: store every packed record through the raw fallback, so the packed layout
#: and L1 holds can be measured without compression.
PACKED_FORCE_RAW_ENV = "DASER_PACKED_FORCE_RAW"
#: Lookup-triggered L2 readahead: read group size in bytes (default 1 GiB,
#: ``0`` disables), and the number of groups kept in flight.
LOAD_READAHEAD_ENV = "DASER_LOAD_READAHEAD"
DEFAULT_LOAD_READAHEAD_BYTES = 1024 * 1024 * 1024
LOAD_READAHEAD_WINDOW_ENV = "DASER_LOAD_READAHEAD_WINDOW"
#: Research switch: ``1`` logs a monotonic timestamp per request at its first
#: lookup, its block allocation and its load completion (scheduler side).
ADMIT_TRACE_ENV = "DASER_ADMIT_TRACE"
#: Research switch: ``1`` times how long the forward stream idles at each
#: layer's restore and logs the sum per layer-wise load as ``gpu_stall_ms``.
LAYERWISE_GPU_STALL_ENV = "DASER_LAYERWISE_GPU_STALL"
#: L2 writes run on their own executor in 16 MiB pieces that pause (bounded)
#: while any L2 read is in native IO (default on; ``0`` disables).
L2_WRITE_GATE_ENV = "DASER_L2_WRITE_GATE"
#: Research switch: ``1`` keeps L1 and L2 disjoint (store to one tier, demote
#: L1 victims to L2); needs both tiers. ``DASER_EXCLUSIVE_CLEAN`` sets the L1
#: lane bytes kept free or clean (default an eighth of the L1 lane).
EXCLUSIVE_TIERS_ENV = "DASER_EXCLUSIVE_TIERS"
EXCLUSIVE_CLEAN_ENV = "DASER_EXCLUSIVE_CLEAN"


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


def _round_down(value: int, alignment: int) -> int:
    """Return ``value`` rounded down to a multiple of ``alignment``."""
    return value // alignment * alignment


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
        l2_size_bytes: requested SSD-tier capacity across all TP ranks; the
            store file holds ``l2_lane_bytes * tensor_parallel_size`` bytes.
            Ignored when ``skip_l2`` is set.
        ipc_socket_path: Unix socket path for Connector <-> Server IPC.
        log_level: log level string passed to init_logger.
        block_tokens: tokens per vLLM block (default 16).
        model_id: identifier string for the model, used to prevent
                  cross-model cache reuse.
        cache_reuse_mode: retrieval/position strategy selected at startup.
        transfer_mode: server-owned transfer layer selected at startup.
        l1_size_bytes: pinned-memory budget for iouring mode (L1 lanes plus
            the staging pool); 0 or gds disables L1.
        skip_l2: when True, keep all transfer bytes in volatile L1 memory and
            do not allocate or persist SSD-tier store files.
        tier_grain: allocation unit of both tiers in bytes; a power of two
            and a multiple of 4096.
        tensor_parallel_size: vLLM tensor-parallel rank count used by the
            physical KV store layout.
        storage_format: startup-immutable raw or experimental compressed mode.
        prefetch_lease_budget_bytes: cap on L1 bytes pinned by
            in-flight scheduler prefetch leases in iouring mode (0 = no cap).
        l2_read_stripe_bytes: part size for splitting one large iouring L2
            read into concurrently submitted IOs (0 = one serial read).
        extra_kv_layers: attention layers the vLLM worker registers beyond
            the model's own, for example an EAGLE drafter's; they share the
            target layers' KV shape and are stored in every slot.
    """

    model_path: str = ""
    vllm_model_id: str = ""
    store_dir: str = "/mnt/xfs/daser"
    l2_size_bytes: int = DEFAULT_L2_BYTES
    ipc_socket_path: str = "/tmp/daser.sock"
    log_level: str = "INFO"

    block_tokens: int = BLOCK_TOKENS
    cache_reuse_mode: str = DEFAULT_CACHE_REUSE_MODE
    transfer_mode: str = "iouring"
    l1_size_bytes: int = DEFAULT_IOURING_L1_BYTES
    skip_l2: bool = False
    tier_grain: int = DEFAULT_TIER_GRAIN
    tensor_parallel_size: int = 1
    storage_format: str = STORAGE_FORMAT_RAW
    l1_accounting: str = L1_ACCOUNTING_STORED
    prefetch_lease_budget_bytes: int = 0
    l2_read_stripe_bytes: int = 0
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
        if self.extra_kv_layers < 0:
            raise ValueError("extra_kv_layers must be non-negative")
        grain = self.tier_grain
        if grain <= 0 or grain & (grain - 1) or grain % _IO_ALIGNMENT:
            raise ValueError(
                f"tier_grain must be a power of two multiple of {_IO_ALIGNMENT}, "
                f"got {grain}"
            )
        if self.l1_size_bytes < 0 or self.l2_size_bytes < 0:
            raise ValueError("tier sizes must be non-negative")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")

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

        Memory-only mode has no durable bytes. Compressed-online records carry
        per-slot codec metadata that lives only in server memory, so only raw
        chunks with an L2 copy are persisted.
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
    def l1_enabled(self) -> bool:
        """Whether the pinned-memory L1 tier exists (iouring with a budget)."""
        return self.transfer_mode == "iouring" and self.l1_size_bytes > 0

    @property
    def l2_enabled(self) -> bool:
        """Whether the SSD L2 tier exists (not ``skip_l2``)."""
        return not self.skip_l2

    @property
    def staging_bytes(self) -> int:
        """Return the pinned staging pool size carved out of the L1 budget.

        Stores not admitted to L1 and loads not promoted to L1 stage through
        this pool, so it exists only when both tiers do.
        """
        if not (self.l1_enabled and self.l2_enabled):
            return 0
        grain = self.tier_grain
        return max(
            grain,
            _round_down(min(MAX_STAGING_BYTES, self.l1_size_bytes // 8), grain),
        )

    @property
    def l1_lane_bytes(self) -> int:
        """Return each TP rank's L1 lane bytes (0 when L1 is disabled)."""
        if not self.l1_enabled:
            return 0
        budget = max(0, self.l1_size_bytes - self.staging_bytes)
        return _round_down(budget // self.tensor_parallel_size, self.tier_grain)

    @property
    def l2_lane_bytes(self) -> int:
        """Return each TP rank's L2 lane bytes (0 when L2 is disabled)."""
        if not self.l2_enabled:
            return 0
        return _round_down(
            self.l2_size_bytes // self.tensor_parallel_size, self.tier_grain
        )

    @property
    def store_file_bytes(self) -> int:
        """Return the store file size: one L2 lane per TP rank."""
        return self.l2_lane_bytes * self.tensor_parallel_size

    @property
    def l1_arena_bytes(self) -> int:
        """Return the pinned L1 arena size: one L1 lane per TP rank."""
        return self.l1_lane_bytes * self.tensor_parallel_size

    @property
    def l1_record_bytes(self) -> int:
        """Return the minimum bytes charged per L1 record.

        Raw L1 accounting of compressed-online records charges every record
        its 4 KiB-aligned raw slot size; otherwise records are charged their
        stored size (0).
        """
        if (
            self.storage_format != STORAGE_FORMAT_COMPRESSED_ONLINE
            or self.l1_accounting != L1_ACCOUNTING_RAW
        ):
            return 0
        local = self.resolved_local_slot_size()
        return -(-local // _IO_ALIGNMENT) * _IO_ALIGNMENT

    def model_geometry(self) -> ModelGeometry:
        """Return the KV geometry of every layer the worker registers.

        Returns:
            Geometry from ``model_path/config.json`` with ``extra_kv_layers``
            added to its layer count.
        """
        geometry = model_geometry_from_path(self.model_path)
        return replace(geometry, num_layers=geometry.num_layers + self.extra_kv_layers)

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
            "block_tokens": self.block_tokens,
            "model_id": self.model_id,
            "cache_reuse_mode": self.cache_reuse_mode,
            "transfer_mode": self.transfer_mode,
            "l1_size_bytes": self.l1_size_bytes,
            "l2_size_bytes": self.store_file_bytes,
            "l1_enabled": self.l1_enabled,
            "l2_enabled": self.l2_enabled,
            "tier_grain": self.tier_grain,
            "l1_lane_bytes": self.l1_lane_bytes,
            "l2_lane_bytes": self.l2_lane_bytes,
            "staging_bytes": self.staging_bytes,
            "skip_l2": self.skip_l2,
            "storage_format": self.storage_format,
            "l1_accounting": self.l1_accounting,
            "prefetch_lease_budget_bytes": self.prefetch_lease_budget_bytes,
            "l2_read_stripe_bytes": self.l2_read_stripe_bytes,
        }
