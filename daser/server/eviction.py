# SPDX-License-Identifier: Apache-2.0
"""Leaf-first eviction order per tier over prefix-chain parent links.

See ``docs/design/tier-residency.md``.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
import heapq

# Subtracted from an orphan's tick so every orphan sorts before every
# reachable node while orphans keep their recency order.
_ORPHAN_BASE = 1 << 62
# Added to a queue-hinted node's priority (minus its earliest hint seq) so
# every hinted node sorts after every unhinted one.
_HINT_BASE = 1 << 61
# Heap entries allowed beyond two per node before stale ones are dropped.
_COMPACT_SLACK = 16


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
    min-heap of ``(priority, seq, key)``, priority being ``(orphan first,
    tick)``; the caller-supplied ``is_candidate(tier, key)`` predicate decides
    membership. A tier queues at most one live entry per node (the one whose
    ``seq`` it records); others are superseded and dropped when popped.
    Candidacy is checked when an entry is queued and again when it is popped,
    so call ``refresh`` after any event that can make a chunk a candidate.
    Removing a node re-queues its parent, which may have become a leaf.

    Recency refreshes (``touch``) only raise priorities, so they leave a
    queued entry in place: it surfaces early and ``pop_victim`` re-queues it
    at the node's current priority. The victim is still the candidate with
    the lowest current priority, while a hit on a queued node costs O(1)
    instead of a predicate call and a heap push. Priority decreases (orphan
    marking) re-queue at once. Superseded entries are compacted away, so a
    tier holds at most about two entries per node.

    Queue hints (``want``/``unwant``) name keys a queued or running request
    looks up, from the time it is queued until it finishes. A hinted
    reachable node sorts after every unhinted one; among hinted nodes the one
    whose earliest hint came last goes first, so the tier keeps what the
    queue needs soonest. Hints are kept per key, so
    a chunk added while hinted inherits its hint.

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
        # Per tier: key -> seq of its live heap entry.
        self._queued: dict[str, dict[str, int]] = {t: {} for t in tiers}
        self._seq = 0
        self._tick = 0
        # Queue hints: hint id -> (seq, keys) and key -> seqs hinting it.
        self._hints: dict[str, tuple[int, list[str]]] = {}
        self._wanted: dict[str, set[int]] = {}
        self._hint_seq = 0

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
        for queued in self._queued.values():
            queued.pop(key, None)
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

        A node already queued in a tier keeps its entry there (corrected when
        popped); a node not queued in a tier is queued if it is a candidate,
        so a touch also acts as ``refresh`` (commit relies on this once a
        pending writer is gone).

        Args:
            keys: Chunk keys; unknown keys are ignored. Within one call, later
                keys count as more recent.

        Async/thread-safety:
            O(1) per key when the key is queued in every tier.
        """
        nodes = self._nodes
        self._touch_nodes(
            [(key, node) for key in keys if (node := nodes.get(key)) is not None]
        )

    def touch_path(self, key: str) -> None:
        """Mark a node and all its present ancestors as just used.

        A prefix hit ending at ``key`` used every block before it; ancestors
        get older ticks than their descendants.

        Args:
            key: Last chunk of a hit; unknown keys are ignored.

        Async/thread-safety:
            O(depth of ``key``), on the server event loop.
        """
        nodes = self._nodes
        # Insertion-ordered; also stops at a parent-link cycle.
        path: dict[str, _Node] = {}
        node = nodes.get(key)
        while node is not None and key not in path:
            path[key] = node
            key = node.parent_key
            node = nodes.get(key) if key else None
        self._touch_nodes(list(reversed(path.items())))

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

    def children(self, key: str) -> tuple[str, ...]:
        """Return the child nodes of ``key``.

        Args:
            key: Chunk key.
        """
        return tuple(self._children.get(key, ()))

    def parent(self, key: str) -> str:
        """Return the parent key of node ``key`` (empty for a chain head).

        Args:
            key: Chunk key; unknown keys return an empty string.
        """
        node = self._nodes.get(key)
        return node.parent_key if node is not None else ""

    @property
    def hinted_requests(self) -> int:
        """Return the number of live queue hints."""
        return len(self._hints)

    @property
    def hinted_keys(self) -> int:
        """Return the number of keys named by at least one live hint."""
        return len(self._wanted)

    def is_wanted(self, key: str) -> bool:
        """Return whether a live queue hint names ``key``.

        Args:
            key: Chunk key (need not be a node).
        """
        return key in self._wanted

    def want(self, hint_id: str, keys: list[str]) -> None:
        """Record that a queued request will look up ``keys``.

        Hints are ordered by arrival: a later hint is assumed to be admitted
        later. A hint with an id already present replaces the old one.

        Args:
            hint_id: Caller-chosen request id.
            keys: Prefix-chain keys the request will look up; keys need not
                be nodes yet.

        Async/thread-safety:
            O(len(keys)); only raises priorities, so queued entries are
            corrected lazily by ``pop_victim``.
        """
        if hint_id in self._hints:
            self.unwant(hint_id)
        self._hint_seq += 1
        seq = self._hint_seq
        keys = list(dict.fromkeys(keys))
        self._hints[hint_id] = (seq, keys)
        for key in keys:
            self._wanted.setdefault(key, set()).add(seq)

    def unwant(self, hint_id: str) -> None:
        """Drop a queue hint once its request finishes (or is abandoned).

        Keys no longer hinted by anyone get fresh ticks in prefix order (the
        request just used them) and are re-queued at their lower priority at
        once.

        Args:
            hint_id: Id given to ``want``; unknown ids are ignored.

        Async/thread-safety:
            O(len(keys)) plus one heap push per lowered node and tier.
        """
        hint = self._hints.pop(hint_id, None)
        if hint is None:
            return
        seq, keys = hint
        for key in keys:
            seqs = self._wanted.get(key)
            if seqs is None:
                continue
            earliest = min(seqs)
            seqs.discard(seq)
            if not seqs:
                del self._wanted[key]
                node = self._nodes.get(key)
                if node is not None:
                    node.tick = self._next_tick()
                    self._lower(key)
            elif earliest == seq and key in self._nodes:
                self._lower(key)

    def is_orphan(self, key: str) -> bool:
        """Return whether ``key`` is a node below an absent ancestor.

        Args:
            key: Chunk key; unknown keys are not orphans.
        """
        node = self._nodes.get(key)
        return node is not None and node.orphan

    def pop_victim(self, tier: str, below: str | None = None) -> str | None:
        """Return the lowest-priority candidate of ``tier``.

        Orphans come first, then the least recently used unhinted
        candidate, then the hinted candidate hinted latest. The entry
        is consumed; the caller evicts the chunk's copy in ``tier`` (and
        ``refresh``-es it if it keeps the node). A popped non-candidate is
        consumed too and returns only after ``refresh``.

        Args:
            tier: Tier name given at construction.
            below: When set, a node; only a candidate whose priority is
                strictly lower than this node's is returned, and the lowest
                entry stays queued otherwise.

        Returns:
            The victim key, or None when no candidate is left (below the
            floor).
        """
        heap = self._heaps[tier]
        queued = self._queued[tier]
        floor = self._priority(below) if below is not None else None
        while heap:
            priority, seq, key = heap[0]
            if floor is not None and priority >= floor and queued.get(key) == seq:
                # Priorities only rise while queued, so nothing is below.
                if priority == self._priority(key):
                    return None
            heapq.heappop(heap)
            if queued.get(key) != seq:
                # Removed, consumed, or superseded by a newer entry.
                continue
            current = self._priority(key)
            if priority != current:
                # Touched since queued: move the entry to its true place.
                self._seq += 1
                queued[key] = self._seq
                heapq.heappush(heap, (current, self._seq, key))
                continue
            del queued[key]
            if self._is_candidate(tier, key):
                return key
        return None

    def _touch_nodes(self, path: list[tuple[str, _Node]]) -> None:
        """Give each node a fresh tick in order; queue new candidates only."""
        queues = self._queued.items()
        is_candidate = self._is_candidate
        tick = self._tick
        for key, node in path:
            tick += 1
            node.tick = tick
            for tier, queued in queues:
                if key not in queued and is_candidate(tier, key):
                    self._push(tier, key)
        self._tick = tick

    def _priority(self, key: str) -> int:
        """Return the eviction priority of a node (lower goes first)."""
        node = self._nodes[key]
        if node.orphan:
            return node.tick - _ORPHAN_BASE
        seqs = self._wanted.get(key)
        return _HINT_BASE - min(seqs) if seqs else node.tick

    def _requeue(self, key: str) -> None:
        """Queue a present node in every tier it is a new candidate of.

        A tier that already queues the node keeps its entry: that entry's
        priority is at most the current one and candidacy is rechecked when
        it is popped.
        """
        for tier, queued in self._queued.items():
            if key not in queued and self._is_candidate(tier, key):
                self._push(tier, key)

    def _lower(self, key: str) -> None:
        """Re-queue a node whose priority just decreased, in every tier.

        A queued entry would surface too late, so it is superseded at once.
        """
        for tier, queued in self._queued.items():
            if key in queued or self._is_candidate(tier, key):
                self._push(tier, key)

    def _push(self, tier: str, key: str) -> None:
        """Push ``key``'s current priority as its live entry in ``tier``.

        Superseded entries stay in the heap until popped; once they exceed
        one per node the heap is compacted, so a tier holds at most about
        one live and one stale entry per node.
        """
        heap = self._heaps[tier]
        queued = self._queued[tier]
        if len(heap) > 2 * len(self._nodes) + _COMPACT_SLACK:
            heap[:] = [entry for entry in heap if queued.get(entry[2]) == entry[1]]
            heapq.heapify(heap)
        self._seq += 1
        queued[key] = self._seq
        heapq.heappush(heap, (self._priority(key), self._seq, key))

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
            lowered = orphan and not node.orphan
            node.orphan = orphan
            if lowered:
                self._lower(child)
            else:
                self._requeue(child)
            stack.extend(self._children.get(child, ()))

    def _next_tick(self) -> int:
        """Return a fresh access tick."""
        self._tick += 1
        return self._tick
