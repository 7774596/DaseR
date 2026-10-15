# SPDX-License-Identifier: Apache-2.0
"""Leaf-first eviction order over prefix-chain parent links.

See ``docs/design/tier-residency.md``.
"""

from collections.abc import Callable
from dataclasses import dataclass
import heapq

# Priority of a chunk no lookup can reach (an ancestor left the index).
_ORPHAN = -1


@dataclass
class _Node:
    """One indexed chunk: parent link, last access tick, reachability."""

    parent_key: str
    tick: int
    orphan: bool = False
    # Every record lives only in L1 (no L2 budget to reclaim).
    held: bool = False


class EvictionTree:
    """Pick the least recently used leaf of the prefix-chain forest.

    Each indexed chunk is a node linked to its ``parent_key``. Only a leaf (no
    child present) may be evicted, so a chain shrinks from its cold tail and
    its root leaves last. Lookup stops at the first missing key, so a node
    below an absent ancestor is unreachable (an orphan) and evicted first.
    Children are indexed by parent key whether or not the parent is present,
    so a parent stored again re-adopts them.

    Candidates live in two lazy min-heaps of ``(priority, seq, key)``: every
    leaf, and leaves that are not held (L2 budget reclaim). Entries are
    validated when popped, so updates only push. Per-pop cost is amortised
    O(log n).

    Async/thread-safety:
        Not thread-safe; owned by the server event loop.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, _Node] = {}
        self._children: dict[str, set[str]] = {}
        self._heap: list[tuple[int, int, str]] = []
        self._charged_heap: list[tuple[int, int, str]] = []
        self._seq = 0
        self._tick = 0

    def __len__(self) -> int:
        """Return the number of nodes."""
        return len(self._nodes)

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` is a node."""
        return key in self._nodes

    def add(self, key: str, parent_key: str = "") -> None:
        """Insert a newly indexed chunk as the most recently used node.

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
        if self._is_leaf(key):
            self._push(key)
        else:
            # Children stored under this key earlier are re-adopted.
            self._mark_subtree(key, orphan)

    def remove(self, key: str) -> None:
        """Forget a chunk that left the index.

        Its parent may become a leaf; nodes below it (if any) become orphans
        and their leaves are queued first.

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
            if parent in self._nodes and self._is_leaf(parent):
                self._push(parent)
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
            if self._is_leaf(key):
                self._push(key)

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
        """Re-queue a node a previous ``pop_victim`` skipped.

        Args:
            key: Chunk key; unknown or non-leaf keys are ignored.
        """
        if key in self._nodes and self._is_leaf(key):
            self._push(key)

    def mark_held(self, key: str) -> None:
        """Exclude a node from ``pop_victim(charged_only=True)``.

        Args:
            key: Chunk key whose records all left the L2 budget; unknown keys
                are ignored. A held node is still a leaf candidate for every
                other trigger and still keeps its parent from being a leaf.
        """
        node = self._nodes.get(key)
        if node is not None:
            node.held = True

    def pop_victim(
        self, skip: Callable[[str], bool], *, charged_only: bool = False
    ) -> str | None:
        """Return the lowest-priority evictable leaf without removing it.

        Args:
            skip: Returns True for a key that may not be evicted now. Its entry
                is dropped from the heap being popped; call ``refresh`` once
                the condition clears.
            charged_only: Ignore held nodes.

        Returns:
            The victim key, or None when no evictable leaf is left. The caller
            removes the chunk (and calls ``remove``) or holds it.
        """
        heap = self._charged_heap if charged_only else self._heap
        while heap:
            priority, _seq, key = heapq.heappop(heap)
            node = self._nodes.get(key)
            if node is None or not self._is_leaf(key):
                continue
            if charged_only and node.held:
                continue
            if self._priority(key) != priority:
                self._push(key)
                continue
            if skip(key):
                continue
            return key
        return None

    def _is_leaf(self, key: str) -> bool:
        """Return whether no child of ``key`` is present."""
        return not self._children.get(key)

    def _priority(self, key: str) -> int:
        """Return the eviction priority of a node (lower goes first)."""
        node = self._nodes[key]
        return _ORPHAN if node.orphan else node.tick

    def _push(self, key: str) -> None:
        """Queue a leaf at its current priority, compacting stale entries."""
        if len(self._heap) + len(self._charged_heap) > 4 * len(self._nodes) + 2048:
            self._rebuild_heaps()
        self._seq += 1
        entry = (self._priority(key), self._seq, key)
        heapq.heappush(self._heap, entry)
        if not self._nodes[key].held:
            heapq.heappush(self._charged_heap, entry)

    def _rebuild_heaps(self) -> None:
        """Replace both heaps with one entry per current leaf."""
        entries = []
        charged = []
        for key, node in self._nodes.items():
            if self._is_leaf(key):
                self._seq += 1
                entry = (self._priority(key), self._seq, key)
                entries.append(entry)
                if not node.held:
                    charged.append(entry)
        heapq.heapify(entries)
        heapq.heapify(charged)
        self._heap = entries
        self._charged_heap = charged

    def _mark_subtree(self, key: str, orphan: bool) -> None:
        """Set reachability of every node below ``key`` and re-queue leaves.

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
            grandchildren = self._children.get(child)
            if grandchildren:
                stack.extend(grandchildren)
            else:
                self._push(child)

    def _next_tick(self) -> int:
        """Return a fresh access tick."""
        self._tick += 1
        return self._tick
