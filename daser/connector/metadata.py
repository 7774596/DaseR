# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass(frozen=True)
class CompressedLoadSlot:
    """One packed slot record carried to the worker load path.

    The server resolves the record's tier location from ``slot_id``.

    Attributes:
        slot_id: DaseR slot id of the record.
        mode: Explicit ``raw`` or ``compressed`` record mode. When TP ranks
            stored the slot in different modes this is ``raw`` (the
            scheduler must not assume a packed record) and ``rank_modes``
            holds each rank's mode; workers decode only after ``for_rank``.
        stored_length: Aligned record bytes to transfer (the largest record
            of any rank; a rank's own record may end earlier).
        rank_modes: Per-rank modes of a slot the ranks stored differently;
            empty when ``mode`` holds on every rank.
    """

    slot_id: int
    mode: Literal["raw", "compressed"]
    stored_length: int
    rank_modes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.slot_id < 0 or self.mode not in ("raw", "compressed"):
            raise ValueError("invalid compressed load slot identity")
        if self.stored_length <= 0:
            raise ValueError("invalid compressed load slot length")
        if any(mode not in ("raw", "compressed") for mode in self.rank_modes):
            raise ValueError("invalid compressed load slot rank mode")

    def for_rank(self, tp_rank: int) -> "CompressedLoadSlot":
        """Return the slot as one TP rank stored it.

        Args:
            tp_rank: Worker's tensor-parallel rank.

        Returns:
            ``self`` when the ranks agree, else a slot whose ``mode`` is the
            rank's own and that carries no ``rank_modes``.

        Raises:
            ValueError: if ``tp_rank`` has no recorded mode.
        """
        if not self.rank_modes:
            return self
        if not 0 <= tp_rank < len(self.rank_modes):
            raise ValueError(f"compressed load slot has no mode for rank {tp_rank}")
        return replace(
            self,
            mode=self.rank_modes[tp_rank],  # type: ignore[arg-type]
            rank_modes=(),
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CompressedLoadSlot":
        """Validate a server lookup payload for scheduler/worker handoff."""
        return cls(
            slot_id=int(payload["slot_id"]),
            mode=str(payload["mode"]),  # type: ignore[arg-type]
            stored_length=int(payload["stored_length"]),
            rank_modes=tuple(str(mode) for mode in payload.get("rank_modes", ())),
        )


@dataclass
class ReqLoadSpec:
    """Load specification for one request.

    Attributes:
        chunk_key: xxh3_128 of the cached token sequence.
        start_slot: first DaseR slot for this chunk.
        num_slots: number of slots in the chunk.
        block_ids: vLLM block IDs allocated to hold the loaded KV.
        token_count: number of tokens covered.
        target_token_start: token offset where this chunk starts in the
            current prompt.
        pos_offset: target-aware position offset returned by the server.
        lease_id: Base request ID retaining host-tier bytes, or empty when the
            load follows the ordinary non-prefetch path.
        compressed_slots: Ordered packed slot records in compressed-online
            mode; empty for the raw path.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    block_ids: list[int]
    token_count: int
    target_token_start: int = 0
    pos_offset: int = 0
    lease_id: str = ""
    compressed_slots: list[CompressedLoadSlot] = field(default_factory=list)


@dataclass
class ReqStoreSpec:
    """Store specification for one request.

    Attributes:
        chunk_key: xxh3_128 of this request's token sequence.
        start_slot: first DaseR slot allocated for this chunk.
        num_slots: number of slots allocated.
        block_ids: vLLM block IDs whose KV to save.
        token_count: number of tokens to store.
        logical_slot_start: first prompt-relative slot represented by this
            allocation. This is distinct from ``start_slot``, the physical
            slot allocated by DaseR.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    block_ids: list[int]
    token_count: int
    logical_slot_start: int = 0


@dataclass(frozen=True)
class StoreWriteSpan:
    """One contiguous source-buffer slice holding records of one chunk.

    The server places the chunk and resolves where the records go.

    Attributes:
        source_offset: Byte offset in the CUDA IPC source buffer.
        nbytes: Number of bytes to write: whole raw slots, or one packed
            record.
        slot_index: Index of the first record within the chunk allocation.
        chunk_key: Chunk key of the allocation.
        start_slot: First slot id allocated for chunk_key.
        num_slots: Number of slots allocated for chunk_key.
        logical_slot_start: First prompt-relative slot of the span.
        logical_slot_count: Prompt-relative slots the span covers.
        packed: Whether the span is one packed (codec) record.
        packed_mode: ``raw`` or ``compressed`` mode of a packed record.
    """

    source_offset: int
    nbytes: int
    slot_index: int
    chunk_key: str = ""
    start_slot: int = -1
    num_slots: int = 0
    logical_slot_start: int = -1
    logical_slot_count: int = 0
    packed: bool = False
    packed_mode: str = ""


@dataclass
class DaserConnectorMeta(KVConnectorMetadata):
    """Metadata passed from scheduler to worker each scheduling step.

    Attributes:
        reqs_to_load: req_id -> ReqLoadSpec for cache hits.
        reqs_to_store: req_id -> ReqStoreSpec for new chunks to persist.
        cancelled_store_req_ids: Preempted base IDs whose published, unsent
            worker stores must be discarded before this step's new stores.
        layerwise_req_ids: Base IDs whose loads vLLM scheduled synchronously
            in this step; the worker restores them layer by layer inside the
            forward pass and owns their transfer leases.
    """

    reqs_to_load: dict[str, ReqLoadSpec] = field(default_factory=dict)
    reqs_to_store: dict[str, ReqStoreSpec] = field(default_factory=dict)
    cancelled_store_req_ids: set[str] = field(default_factory=set)
    layerwise_req_ids: set[str] = field(default_factory=set)
