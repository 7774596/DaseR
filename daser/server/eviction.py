# SPDX-License-Identifier: Apache-2.0
"""Leaf-first eviction order per tier over prefix-chain parent links.

See ``docs/design/tier-residency.md``.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
import heapq

# Priority of a chunk no lookup can reach (an ancestor left the index).
_ORPHAN = -1


@dataclass
class _Node:
    """One chunk: parent link, last access tick, reachability."""

    parent_key: str
    tick: int
    orphan: bool = False


class EvictionTree:
    """Order eviction candidates of each tier by orphan state and recency.

    Every allocated chunk is a node linked to its ``parent_key``. Lookup stops
    at the first missing key, so a node below an absent ancestor is
    unreachable (an orphan) and goes first. Children are indexed by parent key
    whether or not the parent is present, so a parent stored again re-adopts
    them. A node is a leaf when no child node exists at all, including
    uncommitted ones.

    One recency tick is shared by all tiers. Each tier has its own lazy
    min-heap of ``(priority, seq, key)``; the caller-supplied
    ``is_candidate(tier, key)`` predicate decides membership. Entries are
    validated when popped, so updates only push: call ``refresh`` after any
    event that can make a chunk a candidate. Removing a node re-queues its
    parent, which may have become a leaf. Per-pop cost is amortised
    O(log n).

    Args:
        tiers: Names of the tiers that get a candidate heap.
        is_candidate: Returns whether a present node may be evicted from a
            tier now. May call ``is_leaf``; must not mutate the tree.

    Async/thread-safety:
        Not thread-safe; owned by the server event loop.
    """

    def __init__(
        self,
        tiers: tuple[str, ...],
        is_candidate: Callable[[str, str], bool],
    ) -> None:
        self._is_candidate = is_candidate
        self._nodes: dict[str, _Node] = {}
        self._children: dict[str, set[str]] = {}
        self._heaps: dict[str, list[tuple[int, int, str]]] = {t: [] for t in tiers}
        self._seq = 0
        self._tick = 0

    def __len__(self) -> int:
        """Return the number of nodes."""
        return len(self._nodes)

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` is a node."""
        return key in self._nodes

    def add(self, key: str, parent_key: str = "") -> None:
        """Insert a newly allocated chunk as the most recently used node.

        Args:
            key: Chunk key; must not be present.
            parent_key: Previous chunk of its prefix chain, or empty.

        Raises:
            ValueError: If ``key`` is already a node or its own parent.
        """
        if key in self._nodes:
            raise ValueError(f"eviction node already exists: {key}")
        if key == parent_key:
            raise ValueError("a chunk cannot be its own parent")
        parent = self._nodes.get(parent_key) if parent_key else None
        orphan = bool(parent_key) and (parent is None or parent.orphan)
        self._nodes[key] = _Node(parent_key, self._next_tick(), orphan)
        if parent_key:
            self._children.setdefault(parent_key, set()).add(key)
        self._requeue(key)
        if key in self._children:
            # Children stored under this key earlier are re-adopted.
            self._mark_subtree(key, orphan)

    def remove(self, key: str) -> None:
        """Forget a chunk that left the index.

        Its parent is re-queued (it may have become a leaf); nodes below it
        (if any) become orphans and are queued first.

        Args:
            key: Chunk key; unknown keys are ignored.
        """
        node = self._nodes.pop(key, None)
        if node is None:
            return
        parent = node.parent_key
        if parent:
            siblings = self._children.get(parent)
            if siblings is not None:
                siblings.discard(key)
                if not siblings:
                    del self._children[parent]
            if parent in self._nodes:
                self._requeue(parent)
        if key in self._children:
            self._mark_subtree(key, True)

    def touch(self, keys: list[str]) -> None:
        """Mark nodes as just used.

        Args:
            keys: Chunk keys; unknown keys are ignored. Within one call, later
                keys count as more recent.
        """
        for key in keys:
            node = self._nodes.get(key)
            if node is None:
                continue
            node.tick = self._next_tick()
            self._requeue(key)

    def touch_path(self, key: str) -> None:
        """Mark a node and all its present ancestors as just used.

        A prefix hit ending at ``key`` used every block before it; ancestors
        get older ticks than their descendants.

        Args:
            key: Last chunk of a hit; unknown keys are ignored.

        Async/thread-safety:
            O(depth of ``key``), on the server event loop.
        """
        path = []
        seen = set()
        while key and key not in seen and key in self._nodes:
            seen.add(key)
            path.append(key)
            key = self._nodes[key].parent_key
        self.touch(path[::-1])

    def refresh(self, key: str) -> None:
        """Re-queue a node after an event that may make it a candidate.

        Args:
            key: Chunk key; unknown keys are ignored.
        """
        if key in self._nodes:
            self._requeue(key)

    def is_leaf(self, key: str) -> bool:
        """Return whether no child node of ``key`` exists.

        Args:
            key: Chunk key.
        """
        return not self._children.get(key)

    def pop_victim(self, tier: str) -> str | None:
        """Return the lowest-priority candidate of ``tier``.

        Orphans come first, then the least recently used candidate. The entry
        is consumed; the caller evicts the chunk's copy in ``tier`` (and
        ``refresh``-es it if it keeps the node).

        Args:
            tier: Tier name given at construction.

        Returns:
            The victim key, or None when no candidate is left.
        """
        heap = self._heaps[tier]
        while heap:
            priority, _seq, key = heapq.heappop(heap)
            if key not in self._nodes or self._priority(key) != priority:
                # Removed, or a newer entry with the current priority exists.
                continue
            if self._is_candidate(tier, key):
                return key
        return None

    def _priority(self, key: str) -> int:
        """Return the eviction priority of a node (lower goes first)."""
        node = self._nodes[key]
        return _ORPHAN if node.orphan else node.tick

    def _requeue(self, key: str) -> None:
        """Queue a present node in every tier it is a candidate of."""
        if sum(map(len, self._heaps.values())) > 4 * len(self._nodes) + 2048:
            self._rebuild_heaps()
        self._seq += 1
        entry = (self._priority(key), self._seq, key)
        for tier, heap in self._heaps.items():
            if self._is_candidate(tier, key):
                heapq.heappush(heap, entry)

    def _rebuild_heaps(self) -> None:
        """Replace every heap with one entry per current candidate."""
        for tier in self._heaps:
            entries = []
            for key in self._nodes:
                if self._is_candidate(tier, key):
                    self._seq += 1
                    entries.append((self._priority(key), self._seq, key))
            heapq.heapify(entries)
            self._heaps[tier] = entries

    def _mark_subtree(self, key: str, orphan: bool) -> None:
        """Set reachability below ``key`` and re-queue its nodes.

        Runs only when a non-leaf leaves or a parent is stored again, so its
        O(subtree) cost is off the common eviction path.
        """
        stack = list(self._children.get(key, ()))
        seen = {key}
        while stack:
            child = stack.pop()
            node = self._nodes.get(child)
            if node is None or child in seen:
                continue
            seen.add(child)
            node.orphan = orphan
            self._requeue(child)
            stack.extend(self._children.get(child, ()))

    def _next_tick(self) -> int:
        """Return a fresh access tick."""
        self._tick += 1
        return self._tick
