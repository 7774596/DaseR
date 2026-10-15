# SPDX-License-Identifier: Apache-2.0
"""Leaf-first eviction order over prefix-chain parent links.

See ``docs/design/tier-residency.md``.
"""

from collections.abc import Callable
from dataclasses import dataclass
import heapq
import time

# Priority of a chunk no lookup can reach (an ancestor left the index).
_ORPHAN = -1
#: Eviction policies: leaf LRU, or return-aware thread selection.
POLICY_LRU = "lru"
POLICY_RETURN = "return"
POLICIES = (POLICY_LRU, POLICY_RETURN)
# Return-aware policy: arrivals closer than this belong to one visit.
_VISIT_GAP_S = 1.0
# A thread silent for this many periods counts as finished.
_OVERDUE_PERIODS = 3.0
# Return period assumed before any thread has two visits.
_DEFAULT_PERIOD_S = 30.0
# EWMA weight of the newest inter-arrival gap.
_PERIOD_ALPHA = 0.5


@dataclass
class _Node:
    """One indexed chunk: parent link, last access tick, reachability."""

    parent_key: str
    tick: int
    thread: str
    orphan: bool = False
    # Every record lives only in L1 (no L2 budget to reclaim).
    held: bool = False


@dataclass
class _Thread:
    """Arrival history of one prefix-chain thread (return-aware policy)."""

    last_seen: float
    period: float | None = None


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

    With ``policy="return"`` the victim is chosen per thread instead (see
    ``docs/design/tier-residency.md``): a node belongs to the thread of its
    parent, or to its parent key (own key for a root) when no parent is
    indexed. Each store or prefix hit of a thread is an arrival; the thread's
    return period is an EWMA of arrival gaps. Orphan leaves still go first; then threads
    silent for more than three periods (longest overdue first), then the
    thread whose predicted return is latest; within it, the oldest leaf.

    Async/thread-safety:
        Not thread-safe; owned by the server event loop.
    """

    def __init__(
        self,
        policy: str = POLICY_LRU,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty tree.

        Args:
            policy: ``"lru"`` (default) or ``"return"``.
            clock: Monotonic seconds, used by the return-aware policy.

        Raises:
            ValueError: For an unknown policy.
        """
        if policy not in POLICIES:
            raise ValueError(f"unknown eviction policy: {policy}")
        self._policy = policy
        self._clock = clock
        self._nodes: dict[str, _Node] = {}
        self._children: dict[str, set[str]] = {}
        self._heap: list[tuple[int, int, str]] = []
        self._charged_heap: list[tuple[int, int, str]] = []
        self._seq = 0
        self._tick = 0
        # Return-aware state: leaves per thread and arrival history.
        self._thread_leaves: dict[str, set[str]] = {}
        self._thread_sizes: dict[str, int] = {}
        self._threads: dict[str, _Thread] = {}

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
        thread = parent.thread if parent is not None else (parent_key or key)
        self._nodes[key] = _Node(parent_key, self._next_tick(), thread, orphan)
        self._thread_sizes[thread] = self._thread_sizes.get(thread, 0) + 1
        if parent is not None:
            self._thread_leaves.get(parent.thread, set()).discard(parent_key)
        if parent_key:
            self._children.setdefault(parent_key, set()).add(key)
        if self._is_leaf(key):
            self._thread_leaves.setdefault(thread, set()).add(key)
            self._push(key)
        else:
            # Children stored under this key earlier are re-adopted.
            self._mark_subtree(key, orphan, thread)
        self._arrive(thread)

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
        self._drop_from_thread(key, node.thread)
        parent = node.parent_key
        if parent:
            siblings = self._children.get(parent)
            if siblings is not None:
                siblings.discard(key)
                if not siblings:
                    del self._children[parent]
            if parent in self._nodes and self._is_leaf(parent):
                parent_node = self._nodes[parent]
                self._thread_leaves.setdefault(parent_node.thread, set()).add(parent)
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
        if key in self._nodes:
            self._arrive(self._nodes[key].thread)
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
        if self._policy == POLICY_RETURN:
            return self._pop_return(skip, charged_only)
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

    def _mark_subtree(self, key: str, orphan: bool, thread: str | None = None) -> None:
        """Set reachability (and optionally thread) below ``key``; re-queue leaves.

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
            if thread is not None and node.thread != thread:
                self._drop_from_thread(child, node.thread)
                node.thread = thread
                self._thread_sizes[thread] = self._thread_sizes.get(thread, 0) + 1
                if self._is_leaf(child):
                    self._thread_leaves.setdefault(thread, set()).add(child)
            grandchildren = self._children.get(child)
            if grandchildren:
                stack.extend(grandchildren)
            else:
                self._push(child)

    def _next_tick(self) -> int:
        """Return a fresh access tick."""
        self._tick += 1
        return self._tick

    def _drop_from_thread(self, key: str, thread: str) -> None:
        """Forget ``key`` in its thread's leaf set and node count."""
        leaves = self._thread_leaves.get(thread)
        if leaves is not None:
            leaves.discard(key)
            if not leaves:
                del self._thread_leaves[thread]
        size = self._thread_sizes.get(thread, 0) - 1
        if size > 0:
            self._thread_sizes[thread] = size
        else:
            # Arrival history outlives the nodes: a session whose chain was
            # evicted keeps its period when it returns.
            self._thread_sizes.pop(thread, None)

    def _arrive(self, thread: str) -> None:
        """Record a store of ``thread``; gaps update its return period."""
        if self._policy != POLICY_RETURN:
            return
        now = self._clock()
        state = self._threads.get(thread)
        if state is None:
            if len(self._threads) > 4 * len(self._thread_sizes) + 1024:
                self._prune_threads()
            self._threads[thread] = _Thread(now)
            return
        gap = now - state.last_seen
        if gap < _VISIT_GAP_S:
            return
        state.period = (
            gap
            if state.period is None
            else _PERIOD_ALPHA * gap + (1 - _PERIOD_ALPHA) * state.period
        )
        state.last_seen = now

    def _prune_threads(self) -> None:
        """Drop the older half of the histories of threads with no nodes."""
        idle = sorted(
            (state.last_seen, thread)
            for thread, state in self._threads.items()
            if thread not in self._thread_sizes
        )
        for _last_seen, thread in idle[: len(idle) // 2 + 1]:
            del self._threads[thread]

    def _pop_return(
        self, skip: Callable[[str], bool], charged_only: bool
    ) -> str | None:
        """Pick a victim leaf with the return-aware thread order."""
        now = self._clock()
        periods = [t.period for t in self._threads.values() if t.period is not None]
        default = sum(periods) / len(periods) if periods else _DEFAULT_PERIOD_S
        best: tuple[tuple[int, float, int], str] | None = None
        for thread, leaves in self._thread_leaves.items():
            state = self._threads.get(thread)
            period = default
            last_seen = now
            if state is not None:
                last_seen = state.last_seen
                if state.period is not None:
                    period = state.period
            silent = now - last_seen
            for key in leaves:
                node = self._nodes[key]
                if charged_only and node.held:
                    continue
                if node.orphan:
                    rank = (2, 0.0, -node.tick)
                elif silent > _OVERDUE_PERIODS * period:
                    rank = (1, silent, -node.tick)
                else:
                    rank = (0, last_seen + period, -node.tick)
                if (best is None or rank > best[0]) and not skip(key):
                    best = (rank, key)
        return None if best is None else best[1]
