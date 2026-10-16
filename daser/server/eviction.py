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
    # Every record lives only in L1 (outside the L2 budget).
    held: bool = False
    # Present children that are not held.
    charged_kids: int = 0


class EvictionTree:
    """Pick the least recently used leaf of the prefix-chain forest.

    Each indexed chunk is a node linked to its ``parent_key``. Only a leaf (no
    child present) may be evicted, so a chain shrinks from its cold tail and
    its root leaves last. Lookup stops at the first missing key, so a node
    below an absent ancestor is unreachable (an orphan) and evicted first.
    Children are indexed by parent key whether or not the parent is present,
    so a parent stored again re-adopts them.

    Held closure: every descendant of a held node is held. The caller keeps
    it by recharging held ancestors (``held_path``) before adding a child or
    recharging a node. L2 budget reclaim then takes the charged frontier: a
    node that is not held and has no child that is not held (a leaf is the
    common case), so held tails never block their chain's budget.

    Candidates live in two lazy min-heaps of ``(priority, seq, key)``: every
    leaf, and the charged frontier. Entries are validated when popped, so
    updates only push. Per-pop cost is amortised O(log n).

    Async/thread-safety:
        Not thread-safe; owned by the server event loop.
    """

    def __init__(self) -> None:
        """Create an empty tree."""
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
            ValueError: If ``key`` is already a node or its own parent, or the
                parent is held (recharge ``held_path(parent_key)`` first).
        """
        if key in self._nodes:
            raise ValueError(f"eviction node already exists: {key}")
        if key == parent_key:
            raise ValueError("a chunk cannot be its own parent")
        parent = self._nodes.get(parent_key) if parent_key else None
        if parent is not None and parent.held:
            raise ValueError("a held chunk cannot gain a child; recharge it first")
        orphan = bool(parent_key) and (parent is None or parent.orphan)
        node = _Node(parent_key, self._next_tick(), orphan)
        self._nodes[key] = node
        if parent_key:
            self._children.setdefault(parent_key, set()).add(key)
        if parent is not None:
            parent.charged_kids += 1
            self._requeue(parent_key)
        node.charged_kids = sum(
            1
            for child in self._children.get(key, ())
            if child in self._nodes and not self._nodes[child].held
        )
        self._requeue(key)
        if key in self._children:
            # Children stored under this key earlier are re-adopted.
            self._mark_subtree(key, orphan)

    def remove(self, key: str) -> None:
        """Forget a chunk that left the index.

        Its parent may become a candidate; nodes below it (if any) become
        orphans and are queued first.

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
            parent_node = self._nodes.get(parent)
            if parent_node is not None:
                if not node.held:
                    parent_node.charged_kids -= 1
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
        self.touch(self._path(key, held_only=False)[::-1])

    def refresh(self, key: str) -> None:
        """Re-queue a node a previous ``pop_victim`` skipped.

        Args:
            key: Chunk key; unknown keys or non-candidates are ignored.
        """
        if key in self._nodes:
            self._requeue(key)

    def mark_held(self, key: str) -> None:
        """Take a node out of the L2 budget (all its records are held).

        Args:
            key: Charged-frontier chunk key (every child is held); unknown or
                already held keys are ignored. A held node is still a leaf
                candidate for every other trigger and still keeps its parent
                from being a leaf.

        Raises:
            ValueError: If a child of ``key`` is not held.
        """
        node = self._nodes.get(key)
        if node is None or node.held:
            return
        if node.charged_kids:
            raise ValueError("only a node whose children are all held can be held")
        node.held = True
        self._requeue(key)
        parent = self._nodes.get(node.parent_key) if node.parent_key else None
        if parent is not None:
            parent.charged_kids -= 1
            self._requeue(node.parent_key)

    def mark_charged(self, key: str) -> None:
        """Count a held node against the L2 budget again.

        Args:
            key: Held chunk key whose parent is not held; unknown or charged
                keys are ignored.

        Raises:
            ValueError: If the parent is held (recharge ``held_path`` from the
                top down).
        """
        node = self._nodes.get(key)
        if node is None or not node.held:
            return
        parent = self._nodes.get(node.parent_key) if node.parent_key else None
        if parent is not None and parent.held:
            raise ValueError("recharge held ancestors first")
        node.held = False
        self._requeue(key)
        if parent is not None:
            parent.charged_kids += 1
            self._requeue(node.parent_key)

    def held_path(self, key: str) -> list[str]:
        """Return ``key`` and its held ancestors up to the first charged one.

        Args:
            key: Chunk key; an unknown or charged key gives an empty list.

        Returns:
            Held keys from the top of the run down to ``key``, the order in
            which ``mark_charged`` accepts them.
        """
        return self._path(key, held_only=True)[::-1]

    def descendants(self, key: str) -> list[str]:
        """Return the present nodes below ``key``, deepest first.

        Args:
            key: Chunk key (present or not).

        Returns:
            Keys ordered so that each comes before its parent; removing them
            in order never orphans a node that is still present.
        """
        order: list[str] = []
        seen = {key}
        stack = [(key, False)]
        while stack:
            current, expanded = stack.pop()
            if expanded:
                if current != key:
                    order.append(current)
                continue
            stack.append((current, True))
            for child in self._children.get(current, ()):
                if child in self._nodes and child not in seen:
                    seen.add(child)
                    stack.append((child, False))
        return order

    def is_leaf(self, key: str) -> bool:
        """Return whether no child of ``key`` is present.

        Args:
            key: Chunk key.
        """
        return not self._children.get(key)

    def pop_victim(
        self, skip: Callable[[str], bool], *, charged_only: bool = False
    ) -> str | None:
        """Return the lowest-priority evictable candidate without removing it.

        Args:
            skip: Returns True for a key that may not be evicted now. Its entry
                is dropped from the heap being popped; call ``refresh`` once
                the condition clears.
            charged_only: Take the charged frontier (L2 budget reclaim)
                instead of leaves.

        Returns:
            The victim key, or None when no evictable candidate is left. The
            caller removes the chunk (and calls ``remove``) or holds it.
        """
        heap = self._charged_heap if charged_only else self._heap
        while heap:
            priority, _seq, key = heapq.heappop(heap)
            if key not in self._nodes or not self._is_candidate(key, charged_only):
                continue
            if self._priority(key) != priority:
                self._requeue(key)
                continue
            if skip(key):
                continue
            return key
        return None

    def _path(self, key: str, *, held_only: bool) -> list[str]:
        """Return ``key`` and its present (held) ancestors, bottom up."""
        path = []
        seen = set()
        while key and key not in seen and key in self._nodes:
            if held_only and not self._nodes[key].held:
                break
            seen.add(key)
            path.append(key)
            key = self._nodes[key].parent_key
        return path

    def _is_candidate(self, key: str, charged_only: bool) -> bool:
        """Return whether a present node is a leaf (or charged frontier)."""
        if charged_only:
            node = self._nodes[key]
            return not node.held and node.charged_kids == 0
        return self.is_leaf(key)

    def _priority(self, key: str) -> int:
        """Return the eviction priority of a node (lower goes first)."""
        node = self._nodes[key]
        return _ORPHAN if node.orphan else node.tick

    def _requeue(self, key: str) -> None:
        """Queue a present node in the heaps it is a candidate of."""
        if len(self._heap) + len(self._charged_heap) > 4 * len(self._nodes) + 2048:
            self._rebuild_heaps()
        leaf = self._is_candidate(key, False)
        frontier = self._is_candidate(key, True)
        self._seq += 1
        entry = (self._priority(key), self._seq, key)
        if leaf:
            heapq.heappush(self._heap, entry)
        if frontier:
            heapq.heappush(self._charged_heap, entry)

    def _rebuild_heaps(self) -> None:
        """Replace both heaps with one entry per current candidate."""
        entries = []
        charged = []
        for key in self._nodes:
            self._seq += 1
            entry = (self._priority(key), self._seq, key)
            if self._is_candidate(key, False):
                entries.append(entry)
            if self._is_candidate(key, True):
                charged.append(entry)
        heapq.heapify(entries)
        heapq.heapify(charged)
        self._heap = entries
        self._charged_heap = charged

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
