# SPDX-License-Identifier: Apache-2.0

"""Chunk commit/ownership state for the DaseR control plane."""

# Standard
import asyncio
import time


class ChunkLifecycle:
    """Track commit, write-ownership, and commit-waiter state.

    ServerCore keeps parallel sets in lockstep: which chunk keys are committed
    and visible to lookup, and which have an active store writer. This class
    owns those sets and the commit-waiter futures so the transitions stay
    consistent in one place.

    Async/thread-safety:
        All methods run on the server asyncio event loop. Commit waiters are
        completed from the same loop by ``mark_committed``.
    """

    def __init__(self) -> None:
        self._committed: set[str] = set()
        self._write_owners: set[str] = set()
        self._commit_waiters: dict[str, set[asyncio.Future[None]]] = {}
        self._commit_shards: dict[str, tuple[int, set[int]]] = {}
        self._publishing: set[str] = set()
        # Publication timing: when a writer claimed each pending key, and when
        # a lookup first found that key still pending (monotonic seconds).
        self._owner_since: dict[str, float] = {}
        self._first_overlap: dict[str, float] = {}

    def is_committed(self, chunk_key: str) -> bool:
        """Return whether ``chunk_key`` is committed and visible to lookup."""
        return chunk_key in self._committed

    @property
    def write_owners(self) -> set[str]:
        """Return the live write-owner key set (used for reuse predicates)."""
        return self._write_owners

    @property
    def committed(self) -> set[str]:
        """Return the committed key set (used for reuse predicates)."""
        return self._committed

    @property
    def pending_write_keys(self) -> set[str]:
        """Return a snapshot of allocated keys not yet published.

        Returns:
            A new set containing keys claimed by a store writer but not yet
            committed to the retrieval index.

        Async/thread-safety:
            Runs on the server event loop. The returned copy can be inspected
            by a caller without mutating lifecycle state.
        """
        return self._write_owners - self._committed

    def mark_write_owner(self, chunk_key: str) -> None:
        """Record that a store writer claimed ``chunk_key``."""
        self._write_owners.add(chunk_key)
        if chunk_key not in self._committed:
            self._owner_since.setdefault(chunk_key, time.monotonic())

    def note_pending_overlap(self, chunk_keys: set[str]) -> None:
        """Record the first time a lookup needed still-pending ``chunk_keys``.

        Args:
            chunk_keys: Pending writer keys that a lookup could have used.

        Async/thread-safety:
            Runs on the server event loop; only the earliest overlap per key is
            kept, so repeated retries do not move the timestamp.
        """
        now = time.monotonic()
        for key in chunk_keys:
            if key in self._owner_since:
                self._first_overlap.setdefault(key, now)

    def mark_committed(self, chunk_key: str) -> tuple[float | None, float | None]:
        """Mark ``chunk_key`` committed, claim ownership, and wake waiters.

        Returns:
            ``(publish_s, overlap_wait_s)``: seconds from write-owner claim to
            publication, and from the first lookup that found the key pending
            to publication. Either is ``None`` when not observed.
        """
        now = time.monotonic()
        since = self._owner_since.pop(chunk_key, None)
        overlap = self._first_overlap.pop(chunk_key, None)
        self._committed.add(chunk_key)
        self._write_owners.add(chunk_key)
        self._commit_shards.pop(chunk_key, None)
        self._publishing.discard(chunk_key)
        self._clear_timing(chunk_key)
        self._notify_commit_waiters(chunk_key)
        return (
            None if since is None else now - since,
            None if overlap is None else now - overlap,
        )

    def record_commit_shard(self, chunk_key: str, tp_rank: int, tp_size: int) -> bool:
        """Record one TP rank and return whether this call should publish.

        Args:
            chunk_key: Cache key for the pending chunk.
            tp_rank: Tensor-parallel rank that completed its store.
            tp_size: Required number of tensor-parallel ranks.

        Returns:
            True exactly once when all distinct ranks have arrived.

        Raises:
            ValueError: if rank metadata is invalid or changes mid-commit.

        Async/thread-safety:
            Runs on the server event loop. ``_publishing`` prevents another
            request from publishing while the retrieval insert awaits.
        """
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid TP rank {tp_rank} for size {tp_size}")
        if chunk_key in self._committed or chunk_key in self._publishing:
            return False
        expected_size, ranks = self._commit_shards.setdefault(
            chunk_key, (tp_size, set())
        )
        if expected_size != tp_size:
            raise ValueError(
                f"inconsistent TP size for chunk: {tp_size} != {expected_size}"
            )
        ranks.add(tp_rank)
        if len(ranks) != tp_size:
            return False
        self._publishing.add(chunk_key)
        return True

    def abort_publish(self, chunk_key: str) -> None:
        """Allow a failed retrieval-index publish to be retried."""
        self._publishing.discard(chunk_key)

    def discard_owner(self, chunk_key: str) -> None:
        """Release a store-writer claim without committing the chunk."""
        self._write_owners.discard(chunk_key)
        self._commit_shards.pop(chunk_key, None)
        self._publishing.discard(chunk_key)
        self._clear_timing(chunk_key)

    def discard(self, chunk_key: str) -> None:
        """Drop committed and write-owner state of a chunk that left the index."""
        self._committed.discard(chunk_key)
        self._write_owners.discard(chunk_key)
        self._commit_shards.pop(chunk_key, None)
        self._publishing.discard(chunk_key)
        self._clear_timing(chunk_key)

    async def wait_for_committed(
        self,
        chunk_keys: list[str],
        timeout_s: float,
    ) -> None:
        """Wait until every key in ``chunk_keys`` is committed.

        Args:
            chunk_keys: keys to await; duplicates and already-committed keys
                are ignored.
            timeout_s: maximum seconds to wait.

        Raises:
            TimeoutError: if any key is still uncommitted at timeout.
        """
        pending = [
            key for key in dict.fromkeys(chunk_keys) if key not in self._committed
        ]
        if not pending:
            return

        loop = asyncio.get_running_loop()
        futures: dict[str, asyncio.Future[None]] = {}
        for key in pending:
            future: asyncio.Future[None] = loop.create_future()
            self._commit_waiters.setdefault(key, set()).add(future)
            futures[key] = future
        try:
            await asyncio.wait_for(
                asyncio.gather(*futures.values()),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError("timed out waiting for committed chunks") from exc
        finally:
            for key, future in futures.items():
                waiters = self._commit_waiters.get(key)
                if waiters is None:
                    continue
                waiters.discard(future)
                if not waiters:
                    self._commit_waiters.pop(key, None)

    def _clear_timing(self, chunk_key: str) -> None:
        """Drop publication-timing state for a chunk."""
        self._owner_since.pop(chunk_key, None)
        self._first_overlap.pop(chunk_key, None)

    def _notify_commit_waiters(self, chunk_key: str) -> None:
        """Complete and clear any commit waiters for ``chunk_key``."""
        waiters = self._commit_waiters.pop(chunk_key, set())
        for future in waiters:
            if not future.done():
                future.set_result(None)
