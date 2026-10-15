# SPDX-License-Identifier: Apache-2.0

# Standard
import os
from typing import TYPE_CHECKING

# Third Party
import msgpack

# First Party
from daser.logging import init_logger
from daser.server.extents import FreeExtents
from daser.server.metadata_store import ChunkMeta, MetadataStore

if TYPE_CHECKING:
    from daser.server.doc_registry import DocRegistry

logger = init_logger(__name__)


class SlotsExhausted(MemoryError):
    """Raised when no free slot range is long enough for an allocation."""


class ChunkManager:
    """Slot allocator and chunk index owner for DaseR's KV store.

    Slots are handed out first-fit from address-ordered free ranges. The
    manager never chooses what to evict: ``alloc`` raises ``SlotsExhausted``
    and the caller removes chunks (see ``docs/design/tier-residency.md``).
    A removed chunk's slots are reusable at once, unless a write hold still
    covers it.

    Args:
        total_slots: number of logical slots.
        metadata_store: MetadataStore instance this manager operates on.
        doc_registry: optional DocRegistry persisted with the index.
        physical_slots: raw-envelope slots the on-disk L2 file represents;
            defaults to ``total_slots``.

    Async/thread-safety:
        Not thread-safe; owned by the server event loop.
    """

    def __init__(
        self,
        total_slots: int,
        metadata_store: MetadataStore,
        doc_registry: "DocRegistry | None" = None,
        physical_slots: int | None = None,
    ) -> None:
        self._total_slots = total_slots
        self._physical_slots = physical_slots or total_slots
        if self._physical_slots <= 0 or self._physical_slots > total_slots:
            raise ValueError("physical_slots must be within logical slot capacity")
        self._store = metadata_store
        self._doc_registry = doc_registry
        self._free = FreeExtents(0, total_slots)
        # Write holds: token -> chunk metas, and per-meta hold counts. A
        # removed meta with holds keeps its slots until the count drops to 0.
        self._holds: dict[int, list[ChunkMeta]] = {}
        self._hold_counts: dict[int, int] = {}
        self._retired: dict[int, ChunkMeta] = {}
        self._next_hold = 0
        self._rebuild_free()

    @property
    def total_slots(self) -> int:
        """Return logical slot capacity."""
        return self._total_slots

    @property
    def physical_slots(self) -> int:
        """Return raw-envelope slots represented by the physical L2 lane."""
        return self._physical_slots

    @property
    def doc_registry(self) -> "DocRegistry | None":
        """Return the attached DocRegistry if any."""
        return self._doc_registry

    @property
    def store(self) -> MetadataStore:
        """The underlying MetadataStore."""
        return self._store

    @property
    def free_slots(self) -> int:
        """Return slots not owned by a chunk or a pending write hold."""
        return self._free.free

    @property
    def free_holes(self) -> int:
        """Return the number of disjoint free slot ranges (fragmentation)."""
        return self._free.holes

    def alloc(
        self,
        chunk_key: str,
        num_slots: int,
        token_count: int,
        model_id: str,
        pos_offset: int,
        parent_key: str = "",
    ) -> ChunkMeta:
        """Allocate contiguous slots for a new chunk.

        Args:
            chunk_key: unique identifier for the chunk.
            num_slots: number of contiguous slots required.
            token_count: number of tokens in this chunk.
            model_id: model identifier for cache invalidation.
            pos_offset: position encoding offset for this chunk.
            parent_key: previous chunk of the prefix chain, or empty.

        Returns:
            The inserted chunk metadata.

        Raises:
            ValueError: if num_slots is not in ``[1, total_slots]``.
            SlotsExhausted: if no free range is long enough; nothing changed.
        """
        if not 0 < num_slots <= self._total_slots:
            raise ValueError(
                f"num_slots={num_slots} must be within total_slots={self._total_slots}"
            )
        start_slot = self._free.reserve(num_slots)
        if start_slot is None:
            raise SlotsExhausted(
                f"no {num_slots} contiguous free slots "
                f"(free={self._free.free} holes={self._free.holes})"
            )
        meta = ChunkMeta(
            chunk_key=chunk_key,
            start_slot=start_slot,
            num_slots=num_slots,
            token_count=token_count,
            pos_offset=pos_offset,
            model_id=model_id,
            parent_key=parent_key,
        )
        self._store.insert(meta)
        logger.debug(
            "[CHUNK] alloc chunk_key=%s start=%d num=%d free=%d",
            chunk_key,
            start_slot,
            num_slots,
            self._free.free,
        )
        return meta

    def remove(self, chunk_key: str) -> ChunkMeta:
        """Remove a chunk from the index and free its slots.

        Slots of a chunk under a write hold are freed when the hold ends.

        Args:
            chunk_key: key of a stored chunk.

        Returns:
            The removed metadata.

        Raises:
            KeyError: if the chunk is not stored.
        """
        meta = self._store.remove(chunk_key)
        if self._hold_counts.get(id(meta)):
            self._retired[id(meta)] = meta
        else:
            self._free.release(meta.start_slot, meta.num_slots)
        return meta

    def hold_writes(self, chunk_keys: list[str]) -> int:
        """Keep the slots of stored chunks reserved until a write ends.

        Args:
            chunk_keys: Keys a transfer is about to write; unknown keys are
                ignored.

        Returns:
            Token to pass to ``end_writes`` exactly once.

        Async/thread-safety:
            Call on the server event loop before yielding to the transfer.
        """
        metas = []
        for key in dict.fromkeys(chunk_keys):
            meta = self._store.get(key)
            if meta is not None:
                metas.append(meta)
                self._hold_counts[id(meta)] = self._hold_counts.get(id(meta), 0) + 1
        token = self._next_hold
        self._next_hold += 1
        self._holds[token] = metas
        return token

    def end_writes(self, token: int) -> None:
        """End a write hold and free slots whose chunk left meanwhile.

        Args:
            token: Value returned by ``hold_writes``.

        Raises:
            KeyError: If the token is unknown or already ended.
        """
        for meta in self._holds.pop(token):
            count = self._hold_counts[id(meta)] - 1
            if count:
                self._hold_counts[id(meta)] = count
                continue
            del self._hold_counts[id(meta)]
            retired = self._retired.pop(id(meta), None)
            if retired is not None:
                self._free.release(retired.start_slot, retired.num_slots)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, cache_reuse_mode: str | None = None) -> None:
        """Persist the chunk index and document registry to path.

        Args:
            path: absolute file path to write.
            cache_reuse_mode: optional cache reuse mode that created the
                persisted metadata.
        """
        tmp_path = path + ".tmp"
        payload = {
            "total_slots": self._total_slots,
            "index": self._store.to_bytes(),
            "cache_reuse_mode": cache_reuse_mode,
            "doc_registry": (
                self._doc_registry.to_dict() if self._doc_registry is not None else {}
            ),
        }
        try:
            with open(tmp_path, "wb") as f:
                f.write(msgpack.packb(payload, use_bin_type=True))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        logger.info(
            "[CHUNK] state saved to %s (chunks=%d docs=%d)",
            path,
            len(self._store),
            len(self._doc_registry) if self._doc_registry is not None else 0,
        )

    def load(
        self,
        path: str,
        expected_cache_reuse_mode: str | None = None,
    ) -> None:
        """Restore the chunk index from path and rebuild slot occupancy.

        Args:
            path: absolute file path to read.
            expected_cache_reuse_mode: optional startup cache reuse mode. When
                set, persisted metadata from a different mode is rejected.

        Raises:
            ValueError: on a reuse-mode or slot-capacity mismatch, or when
                persisted chunks overlap.
        """
        with open(path, "rb") as f:
            payload = msgpack.unpackb(f.read(), raw=False)

        saved_mode = payload.get("cache_reuse_mode")
        if expected_cache_reuse_mode is not None and saved_mode is None:
            raise ValueError(
                "cache reuse mode missing from index; cold start is required"
            )
        if (
            expected_cache_reuse_mode is not None
            and saved_mode != expected_cache_reuse_mode
        ):
            raise ValueError(
                "cache reuse mode mismatch: "
                f"index was saved with {saved_mode!r}, "
                f"server started with {expected_cache_reuse_mode!r}"
            )
        if payload.get("total_slots") != self._total_slots:
            raise ValueError(
                f"slot capacity mismatch: index has {payload.get('total_slots')}, "
                f"server has {self._total_slots}"
            )
        self._store.load_bytes(payload["index"])
        self._rebuild_free()
        if self._doc_registry is not None:
            self._doc_registry.load_dict(payload.get("doc_registry", {}) or {})

        logger.info(
            "[CHUNK] state loaded from %s (chunks=%d docs=%d)",
            path,
            len(self._store),
            len(self._doc_registry) if self._doc_registry is not None else 0,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rebuild_free(self) -> None:
        """Recompute free slot ranges from the chunk index."""
        self._free = FreeExtents(0, self._total_slots)
        for meta in sorted(self._store.iter_chunks(), key=lambda m: m.start_slot):
            self._free.take(meta.start_slot, meta.num_slots)
