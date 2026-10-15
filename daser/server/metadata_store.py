# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import asdict, dataclass, field
import time
from typing import Optional

# Third Party
import msgpack

# First Party
from daser.logging import init_logger

logger = init_logger(__name__)


@dataclass
class ChunkMeta:
    """Metadata for one cached KV chunk.

    Attributes:
        chunk_key: xxh3_128(token_ids) or document ID identifying this chunk.
        start_slot: index of the first slot in daser.store.
        num_slots: number of contiguous slots occupied.
        token_count: number of tokens whose KV is stored.
        pos_offset: position encoding offset applied at load time.
        model_id: model identifier, prevents cross-model reuse.
        created_at: unix timestamp of insertion.
        doc_ids: list of doc_ids that reference this chunk (empty when
            the chunk belongs to no registered document). Serves as the
            back-pointer used by cascading eviction.
        parent_key: key of the chunk a prefix lookup must pass before this
            one (the previous block of the same rolling-prefix chain), or
            empty for a chain root. Drives leaf-first eviction.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    token_count: int
    pos_offset: int
    model_id: str
    created_at: float = 0.0
    doc_ids: list[str] = field(default_factory=list)
    parent_key: str = ""

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()


class MetadataStore:
    """In-memory chunk index: chunk_key -> ChunkMeta.

    Can be serialized to / deserialized from a msgpack file. Slot occupancy
    is owned by ``ChunkManager``, which rebuilds it from this index.
    """

    def __init__(self) -> None:
        self._chunk_index: dict[str, ChunkMeta] = {}
        self._removals = 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def insert(self, meta: ChunkMeta) -> None:
        """Insert a chunk into the index.

        Args:
            meta: ChunkMeta to insert.

        Raises:
            ValueError: if chunk_key already exists.
        """
        if meta.chunk_key in self._chunk_index:
            raise ValueError(f"chunk_key already exists: {meta.chunk_key}")
        self._chunk_index[meta.chunk_key] = meta
        logger.debug(
            "[INDEX] insert chunk_key=%s start_slot=%d num_slots=%d",
            meta.chunk_key,
            meta.start_slot,
            meta.num_slots,
        )

    def remove(self, chunk_key: str) -> ChunkMeta:
        """Remove a chunk from the index.

        Args:
            chunk_key: key of the chunk to remove.

        Returns:
            The removed metadata.

        Raises:
            KeyError: if chunk_key is not found.
        """
        meta = self._chunk_index.pop(chunk_key, None)
        if meta is None:
            raise KeyError(f"chunk_key not found: {chunk_key}")
        self._removals += 1
        logger.debug("[INDEX] remove chunk_key=%s", chunk_key)
        return meta

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def removals(self) -> int:
        """Return how many chunks have left the index since construction.

        A lookup result stays valid while this count is unchanged: chunks are
        only invalidated by removal, and a key cannot be re-inserted without
        being removed first.
        """
        return self._removals

    def get(self, chunk_key: str) -> Optional[ChunkMeta]:
        """Look up a chunk by key.

        Args:
            chunk_key: the key to look up.

        Returns:
            ChunkMeta if found, None otherwise.
        """
        return self._chunk_index.get(chunk_key)

    def __len__(self) -> int:
        """Return number of chunks currently stored."""
        return len(self._chunk_index)

    def iter_chunks(self):
        """Yield every stored ChunkMeta.

        Returns:
            Iterator over ChunkMeta values.
        """
        return iter(self._chunk_index.values())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Serialize the index to msgpack bytes.

        Returns:
            Encoded chunk index.
        """
        payload = {
            "chunk_index": {k: asdict(v) for k, v in self._chunk_index.items()},
        }
        return msgpack.packb(payload, use_bin_type=True)

    def load_bytes(self, data: bytes) -> None:
        """Replace the index with one decoded by :meth:`to_bytes`.

        Args:
            data: Encoded chunk index.
        """
        payload = msgpack.unpackb(data, raw=False)
        # Every previously returned lookup result is invalid after a reload.
        self._removals += 1
        self._chunk_index = {
            k: ChunkMeta(**v) for k, v in payload["chunk_index"].items()
        }
        logger.info("[INDEX] loaded %d chunks", len(self._chunk_index))
