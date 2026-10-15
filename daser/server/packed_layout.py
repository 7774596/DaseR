# SPDX-License-Identifier: Apache-2.0
"""Placement and L1/L2 residency of online records in a packed byte arena.

See ``docs/design/tier-residency.md``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from daser.server.chunk_manager import ChunkManager
from daser.server.extents import FreeExtents
from daser.server.metadata_store import ChunkMeta


class PackedArenaFull(MemoryError):
    """Raised when a packed batch does not fit the arena's free byte ranges.

    Attributes:
        deficit_bytes: Lower bound on bytes that must be reclaimed before a
            retry of the same batch can succeed.
    """

    def __init__(self, deficit_bytes: int, message: str) -> None:
        """Record the reclaim target.

        Args:
            deficit_bytes: Positive number of bytes to free before retrying.
            message: Human-readable error message.
        """
        super().__init__(message)
        self.deficit_bytes = deficit_bytes


@dataclass(eq=False)
class _Placement:
    """Retain the exact allocation generation and its immutable representation.

    ``writers`` counts open write holds; a placement dropped while a write
    holds it is ``retired`` and keeps its bytes until the last hold ends.
    ``written`` marks a record whose store has reported, so it may be loaded.
    ``held`` marks a record kept resident by L1 alone: it is outside the L2
    budget until its chunk leaves the index.
    """

    owner: ChunkMeta
    file_offset: int
    nbytes: int
    mode: str
    writers: int = 0
    retired: bool = False
    written: bool = False
    held: bool = False


@dataclass(frozen=True)
class PackedRecord:
    """Physical location of one written packed slot.

    Attributes:
        slot_id: Logical slot the record belongs to.
        chunk_key: Chunk that owns the slot.
        mode: Stored representation, e.g. ``"compressed"`` or ``"raw"``.
        file_offset: Byte offset of the record in the rank lane.
        stored_length: Aligned byte length of the record.
    """

    slot_id: int
    chunk_key: str
    mode: str
    file_offset: int
    stored_length: int

    def to_ref(self) -> dict[str, int | str]:
        """Return the slot ref a lookup payload carries to the worker."""
        return {
            "slot_id": self.slot_id,
            "mode": self.mode,
            "file_offset": self.file_offset,
            "stored_length": self.stored_length,
        }


def _span_slot(span: dict[str, Any]) -> int:
    """Return the logical slot a single-slot packed span describes."""
    start = int(span.get("start_slot", -1))
    count = int(span.get("num_slots", 0))
    return start + (0 if count == 1 else int(span.get("logical_slot_start", -1)))


class OnlinePackedLayout:
    """Place packed records in free byte ranges and track their tier state.

    One server event loop owns this bounded metadata map, which is also the
    only chunk -> physical record index: a placement becomes loadable once its
    store reports through ``mark_written``. No method performs IO or suspends;
    transfer and commit visibility remain the caller's responsibility.
    Free space is tracked incrementally: callers report chunks that left the
    metadata store through ``release``, so placement cost does not grow with
    the number of live records. Bytes under an open write hold stay reserved
    after their owner leaves, so a later record cannot be placed over bytes a
    late write may still target.

    Every live placement is either charged (counted by ``charged_bytes``) or
    held (resident in L1 only). The caller picks victims and enforces the L2
    budget with ``charged_records``/``hold``; ``held_owner`` maps an L1
    release back to its chunk.

    Args:
        on_held_drop: Called with ``(file_offset, nbytes)`` when a held
            placement is dropped, before its bytes can be placed again, so
            the caller can discard the L1 copy.
    """

    def __init__(self, on_held_drop: Callable[[int, int], None] | None = None) -> None:
        self._placements: dict[tuple[int, int], _Placement] = {}
        self._keys_by_chunk: dict[str, set[tuple[int, int]]] = {}
        self._by_offset: dict[int, tuple[int, int]] = {}
        self._arenas: dict[int, FreeExtents] = {}
        self._holds: dict[int, list[tuple[int, _Placement]]] = {}
        self._next_hold = 0
        self._charged_bytes: dict[int, int] = {}
        self._held_bytes = 0
        self._on_held_drop = on_held_drop

    def compact(
        self,
        spans: list[dict[str, Any]],
        manager: ChunkManager,
        *,
        local_slot_size: int,
        rank_base: int = 0,
        arena_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Place source-contiguous records into free physical byte ranges.

        Args:
            spans: Packed spans carrying current logical allocation metadata.
            manager: Public owner of current allocation identities.
            local_slot_size: Positive aligned byte size of one raw local slot.
            rank_base: Nonnegative start of this rank's physical lane.
            arena_size: Physical byte capacity of this rank lane. When omitted,
                derive it from the manager's logical slot count for unit tests.

        Returns:
            Copied spans with server-assigned physical file offsets.

        Raises:
            ValueError: On stale ownership, invalid slot geometry, duplicate
                records, a changed representation of an existing generation or
                a lane whose arena size changes.
            PackedArenaFull: When the new records do not fit the free byte
                ranges. No live placement is changed, so the call can be
                retried after reclaiming ``deficit_bytes``.

        Async/thread-safety:
            Call on the server event loop, without yielding between validation
            and transfer reservation. Failed transfer keeps its placement for
            an idempotent retry; this method does not publish cache visibility.
            Costs O(len(spans) x free holes), independent of live records.
        """
        if (
            local_slot_size <= 0
            or local_slot_size % 4096
            or rank_base < 0
            or rank_base % 4096
        ):
            raise ValueError("invalid packed layout geometry")
        if arena_size is None:
            arena_size = manager.total_slots * local_slot_size
        if arena_size <= 0 or arena_size % 4096:
            raise ValueError("invalid packed arena size")
        arena = self._arena(rank_base, arena_size)
        result = [dict(span) for span in spans]
        records: list[tuple[int, int, ChunkMeta, _Placement | None]] = []
        seen: set[int] = set()
        stale: list[tuple[tuple[int, int], _Placement]] = []
        # Validate the whole call before changing any remembered placement.
        for index, span in enumerate(result):
            if not bool(span.get("packed", False)):
                continue
            owner = manager.store.get(str(span.get("chunk_key", "")))
            start = int(span.get("start_slot", -1))
            count = int(span.get("num_slots", 0))
            logical = int(span.get("logical_slot_start", -1))
            slot = _span_slot(span)
            relative = slot - start
            if (
                owner is None
                or owner.start_slot != start
                or owner.num_slots != count
                or logical < 0
                or not 0 <= relative < count
                or int(span.get("logical_slot_count", 0)) != 1
            ):
                raise ValueError("packed layout requires a current slot allocation")
            nbytes = int(span["nbytes"])
            if not 0 < nbytes <= local_slot_size or nbytes % 4096:
                raise ValueError("packed record must fit one aligned raw slot")
            if slot in seen:
                raise ValueError("packed layout repeats a logical slot")
            seen.add(slot)
            prior = self._placements.get((rank_base, slot))
            if prior is not None and prior.owner is not owner:
                # An earlier generation of this slot that was never released.
                stale.append(((rank_base, slot), prior))
                prior = None
            if prior is not None and (
                prior.nbytes != nbytes
                or prior.mode != str(span.get("mode", "compressed"))
            ):
                raise ValueError("packed retry changes an assigned representation")
            records.append((index, slot, owner, prior))
        for key, placement in stale:
            self._drop(key, placement)

        pending: dict[tuple[int, int], _Placement] = {}
        run: list[tuple[int, int, ChunkMeta]] = []
        unplaced_bytes = sum(
            int(result[index]["nbytes"])
            for index, _slot, _owner, prior in records
            if prior is None
        )

        def reserve_run() -> list[int] | None:
            total_bytes = sum(int(result[index]["nbytes"]) for index, _, _ in run)
            cursor = arena.reserve(total_bytes)
            if cursor is not None:
                offsets = []
                for index, _, _ in run:
                    offsets.append(cursor)
                    cursor += int(result[index]["nbytes"])
                return offsets
            # Fragmented arena: place records individually.
            offsets = []
            for index, _, _ in run:
                nbytes = int(result[index]["nbytes"])
                offset = arena.reserve(nbytes)
                if offset is None:
                    # ``offsets`` holds only the records reserved so far.
                    for (prev_index, _, _), prev in zip(run, offsets, strict=False):
                        arena.release(prev, int(result[prev_index]["nbytes"]))
                    return None
                offsets.append(offset)
            return offsets

        def finish_run() -> None:
            nonlocal unplaced_bytes
            if not run:
                return
            offsets = reserve_run()
            if offsets is None:
                # Roll back earlier runs so a failed call reserves nothing.
                for placement in pending.values():
                    arena.release(placement.file_offset, placement.nbytes)
                free_bytes = arena.free
                need = int(result[run[0][0]]["nbytes"])
                raise PackedArenaFull(
                    max(need, unplaced_bytes - free_bytes),
                    f"packed arena exhausted: need={need} "
                    f"unplaced={unplaced_bytes} free={free_bytes}",
                )
            for (index, slot, owner), cursor in zip(run, offsets, strict=True):
                span = result[index]
                nbytes = int(span["nbytes"])
                span["file_offset"] = cursor
                unplaced_bytes -= nbytes
                pending[(rank_base, slot)] = _Placement(
                    owner,
                    cursor,
                    nbytes,
                    str(span.get("mode", "compressed")),
                )
            run.clear()

        for index, slot, owner, prior in records:
            if prior is not None:
                finish_run()
                result[index]["file_offset"] = prior.file_offset
                continue
            if run:
                prev_index, _prev_slot, _prev_owner = run[-1]
                prev_span = result[prev_index]
                if index != prev_index + 1 or int(
                    result[index]["source_offset"]
                ) != int(prev_span["source_offset"]) + int(prev_span["nbytes"]):
                    finish_run()
            run.append((index, slot, owner))
        finish_run()
        for key, placement in pending.items():
            self._register(key, placement)
        return result

    def begin_writes(self, spans: list[dict[str, Any]], *, rank_base: int = 0) -> int:
        """Keep the bytes of placed packed spans reserved until a write ends.

        Args:
            spans: Spans returned by ``compact`` for one transfer. Non-packed
                spans are ignored.
            rank_base: Start of the rank lane the spans were placed in.

        Returns:
            Hold token to pass to ``end_writes`` exactly once.

        Raises:
            ValueError: If a packed span does not match a current placement.

        Async/thread-safety:
            Call on the server event loop right after ``compact``, before
            yielding, so no release can run in between.
        """
        held: list[tuple[int, _Placement]] = []
        for span in spans:
            if not bool(span.get("packed", False)):
                continue
            placement = self._placements.get((rank_base, _span_slot(span)))
            if placement is None or placement.file_offset != int(span["file_offset"]):
                raise ValueError("write hold requires a current packed placement")
            held.append((rank_base, placement))
        for _base, placement in held:
            placement.writers += 1
        token = self._next_hold
        self._next_hold += 1
        self._holds[token] = held
        return token

    def end_writes(self, token: int) -> int:
        """End a write hold and free bytes whose owner left meanwhile.

        Args:
            token: Value returned by ``begin_writes``.

        Returns:
            Bytes freed here because their owner was removed during the hold.

        Raises:
            KeyError: If the token is unknown or already ended.

        Async/thread-safety:
            Call on the server event loop once the transfer layer has the
            write ordered (returned or failed).
        """
        freed = 0
        for rank_base, placement in self._holds.pop(token):
            placement.writers -= 1
            if placement.retired and placement.writers == 0:
                self._arenas[rank_base].release(placement.file_offset, placement.nbytes)
                freed += placement.nbytes
        return freed

    def release(self, chunk_key: str, manager: ChunkManager) -> None:
        """Reclaim the bytes of a chunk that is no longer a current allocation.

        Args:
            chunk_key: Key that left the metadata store.
            manager: Public owner of current allocation identities.

        Async/thread-safety:
            Call on the server event loop after the chunk is removed from
            ``manager.store``. Placements of a newer generation under the same
            key are kept. Costs O(slots of the chunk).
        """
        current = manager.store.get(chunk_key)
        for key in list(self._keys_by_chunk.get(chunk_key, ())):
            placement = self._placements.get(key)
            if placement is not None and placement.owner is not current:
                self._drop(key, placement)

    def mark_written(
        self, chunk_key: str, slot: int, file_offset: int, *, rank_base: int = 0
    ) -> bool:
        """Mark the placement a finished store wrote as loadable.

        Args:
            chunk_key: Chunk the store wrote.
            slot: Logical slot of the record.
            file_offset: Physical offset the store wrote.
            rank_base: Start of the rank lane the record was placed in.

        Returns:
            Whether a matching current placement was marked. False means the
            write targeted bytes that were since released or re-placed.

        Async/thread-safety:
            Call on the server event loop after the transfer completed.
        """
        placement = self._placements.get((rank_base, slot))
        if (
            placement is None
            or placement.owner.chunk_key != chunk_key
            or placement.file_offset != file_offset
        ):
            return False
        placement.written = True
        return True

    def clear_written(self, chunk_key: str) -> None:
        """Hide a chunk's written records until its next store reports.

        Args:
            chunk_key: Chunk whose uncommitted writer was abandoned.

        Async/thread-safety:
            Call on the server event loop. Placements stay reserved, so a
            retried store reuses them.
        """
        for key in self._keys_by_chunk.get(chunk_key, ()):
            self._placements[key].written = False

    def written_records(
        self, start_slot: int, num_slots: int, manager: ChunkManager
    ) -> list[PackedRecord] | None:
        """Return the rank-0 written records of a logical slot interval.

        Args:
            start_slot: First logical slot.
            num_slots: Number of slots requested.
            manager: Public owner of current allocation identities.

        Returns:
            One record per slot in order, or None unless every slot has a
            written placement whose owner is still the current allocation.

        Async/thread-safety:
            Pure lookup on the server event loop.
        """
        if start_slot < 0 or num_slots <= 0:
            return None
        records: list[PackedRecord] = []
        for slot in range(start_slot, start_slot + num_slots):
            placement = self._placements.get((0, slot))
            if placement is None or not placement.written:
                return None
            owner = placement.owner
            if manager.store.get(owner.chunk_key) is not owner:
                return None
            records.append(self._record((0, slot), placement))
        return records

    def charged_records(self, chunk_key: str) -> list[PackedRecord]:
        """Return the charged (L2-budgeted) records of a chunk.

        Args:
            chunk_key: Chunk to inspect.

        Returns:
            Its charged records, in no particular order; empty when every
            record is held or the chunk has no placement.

        Async/thread-safety:
            Pure lookup on the server event loop; O(records of the chunk).
        """
        return [
            self._record(key, placement)
            for key in self._keys_by_chunk.get(chunk_key, ())
            if not (placement := self._placements[key]).held
        ]

    def hold(self, file_offset: int) -> None:
        """Move a charged record out of the L2 budget; L1 now keeps it.

        Args:
            file_offset: Offset of a charged record, e.g. from
                ``charged_records``.

        Raises:
            ValueError: If no charged record starts at ``file_offset``.

        Async/thread-safety:
            Call on the server event loop after L1 marked the range held.
        """
        key = self._by_offset.get(file_offset)
        placement = None if key is None else self._placements[key]
        if placement is None or placement.held:
            raise ValueError("hold requires a charged packed placement")
        lane = key[0]
        self._charged_bytes[lane] -= placement.nbytes
        placement.held = True
        self._held_bytes += placement.nbytes

    def held_owner(self, file_offset: int, nbytes: int) -> str | None:
        """Return the chunk owning the held record at exactly this range.

        Args:
            file_offset: Start of a range L1 released.
            nbytes: Range length.

        Returns:
            The owner chunk key, or None when no live held record matches
            (e.g. it was already dropped).

        Async/thread-safety:
            Pure lookup on the server event loop.
        """
        key = self._by_offset.get(file_offset)
        if key is None:
            return None
        placement = self._placements[key]
        if not placement.held or placement.nbytes != nbytes:
            return None
        return placement.owner.chunk_key

    def charged_bytes(self, *, rank_base: int = 0) -> int:
        """Return bytes of a lane's records counted against the L2 budget.

        Args:
            rank_base: Start of the rank lane.

        Returns:
            Charged bytes of the lane.

        Async/thread-safety:
            Pure lookup on the server event loop.
        """
        return self._charged_bytes.get(rank_base, 0)

    @property
    def held_bytes(self) -> int:
        """Return bytes of held (L1-only) records across all lanes."""
        return self._held_bytes

    def free_bytes(self, *, rank_base: int = 0) -> int:
        """Return unreserved bytes of a rank lane.

        Args:
            rank_base: Start of the rank lane to inspect.

        Returns:
            Free bytes, or 0 before the lane's first placement.

        Async/thread-safety:
            Pure lookup on the server event loop.
        """
        arena = self._arenas.get(rank_base)
        return 0 if arena is None else arena.free

    def _arena(self, rank_base: int, arena_size: int) -> FreeExtents:
        """Return the free-extent list of a lane, creating it on first use."""
        arena = self._arenas.get(rank_base)
        if arena is None:
            arena = FreeExtents(rank_base, rank_base + arena_size)
            self._arenas[rank_base] = arena
        elif arena.end - arena.start != arena_size:
            raise ValueError("packed arena size changed for a rank lane")
        return arena

    def _record(self, key: tuple[int, int], placement: _Placement) -> PackedRecord:
        """Return the public view of one placement."""
        return PackedRecord(
            key[1],
            placement.owner.chunk_key,
            placement.mode,
            placement.file_offset,
            placement.nbytes,
        )

    def _register(self, key: tuple[int, int], placement: _Placement) -> None:
        """Index one new placement; new records are charged."""
        lane = key[0]
        self._placements[key] = placement
        self._keys_by_chunk.setdefault(placement.owner.chunk_key, set()).add(key)
        self._by_offset[placement.file_offset] = key
        self._charged_bytes[lane] = self._charged_bytes.get(lane, 0) + placement.nbytes

    def _drop(self, key: tuple[int, int], placement: _Placement) -> None:
        """Forget one placement and return its bytes unless a write holds them.

        A held placement's L1 copy is discarded first, so no L1 entry can
        alias a record placed later at the same offset.
        """
        del self._placements[key]
        del self._by_offset[placement.file_offset]
        chunk_key = placement.owner.chunk_key
        keys = self._keys_by_chunk.get(chunk_key)
        if keys is not None:
            keys.discard(key)
            if not keys:
                del self._keys_by_chunk[chunk_key]
        lane = key[0]
        if placement.held:
            self._held_bytes -= placement.nbytes
            if self._on_held_drop is not None:
                self._on_held_drop(placement.file_offset, placement.nbytes)
        else:
            self._charged_bytes[lane] -= placement.nbytes
        if placement.writers:
            placement.retired = True
            return
        self._arenas[key[0]].release(placement.file_offset, placement.nbytes)
