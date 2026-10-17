# SPDX-License-Identifier: Apache-2.0

# Standard
import random
import tracemalloc

# Third Party
import pytest

from daser.server.eviction import EvictionTree


class Membership:
    """Candidate predicate driven by a per-tier exclusion set.

    By default every node that is a leaf of the tree is a candidate of every
    tier, matching the symmetric closure of ``TierIndex``.
    """

    def __init__(self) -> None:
        self.blocked: dict[str, set[str]] = {"l1": set(), "l2": set()}
        self.tree: EvictionTree | None = None
        self.leaf_only = True

    def __call__(self, tier: str, key: str) -> bool:
        if key in self.blocked[tier]:
            return False
        assert self.tree is not None
        return not self.leaf_only or self.tree.is_leaf(key)


def make_tree() -> tuple[EvictionTree, Membership]:
    member = Membership()
    tree = EvictionTree(("l1", "l2"), member)
    member.tree = tree
    return tree, member


def drain(tree: EvictionTree, tier: str = "l2") -> list[str]:
    """Evict until no candidate is left, returning the victim order."""
    order = []
    while (victim := tree.pop_victim(tier)) is not None:
        order.append(victim)
        tree.remove(victim)
    return order


def chain(tree: EvictionTree, keys: list[str]) -> None:
    parent = ""
    for key in keys:
        tree.add(key, parent)
        parent = key


def test_chain_evicts_from_tail_and_root_last() -> None:
    tree, _ = make_tree()
    chain(tree, ["a0", "a1", "a2"])
    assert drain(tree) == ["a2", "a1", "a0"]


def test_parent_waits_for_its_last_child() -> None:
    tree, _ = make_tree()
    tree.add("root")
    tree.add("left", "root")
    tree.add("right", "root")
    tree.touch(["root", "left"])
    assert drain(tree) == ["right", "left", "root"]


def test_least_recently_used_leaf_goes_first() -> None:
    tree, _ = make_tree()
    chain(tree, ["a0", "a1"])
    chain(tree, ["b0", "b1"])
    tree.touch(["a0", "a1"])
    assert tree.pop_victim("l2") == "b1"


def test_touch_path_refreshes_ancestors_older_than_the_hit() -> None:
    tree, member = make_tree()
    member.leaf_only = False
    chain(tree, ["a0", "a1", "a2"])
    tree.add("b0")
    tree.touch_path("a2")
    # b0 is now the oldest; the hit chain keeps root-first recency.
    assert drain(tree) == ["b0", "a0", "a1", "a2"]


def test_tiers_have_independent_candidates() -> None:
    tree, member = make_tree()
    tree.add("a")
    tree.add("b")
    member.blocked["l1"].add("a")
    assert tree.pop_victim("l1") == "b"
    assert tree.pop_victim("l2") == "a"
    # Both tiers share one recency order.
    tree.refresh("b")
    assert tree.pop_victim("l2") == "b"


def test_skipped_candidate_is_dropped_until_refresh() -> None:
    tree, member = make_tree()
    tree.add("pinned")
    tree.add("free")
    member.blocked["l2"].add("pinned")
    assert tree.pop_victim("l2") == "free"
    tree.remove("free")
    assert tree.pop_victim("l2") is None
    member.blocked["l2"].clear()
    # The stale entry was consumed; only refresh re-queues it.
    assert tree.pop_victim("l2") is None
    tree.refresh("pinned")
    assert tree.pop_victim("l2") == "pinned"


def test_skipped_child_keeps_parent_resident() -> None:
    tree, member = make_tree()
    chain(tree, ["a0", "a1"])
    member.blocked["l2"].add("a1")
    assert tree.pop_victim("l2") is None
    assert "a0" in tree


def test_removing_a_leaf_requeues_its_parent() -> None:
    tree, _ = make_tree()
    chain(tree, ["a0", "a1"])
    assert tree.pop_victim("l2") == "a1"
    tree.remove("a1")
    assert tree.is_leaf("a0")
    assert tree.pop_victim("l2") == "a0"


def test_orphans_go_first_and_parent_return_readopts() -> None:
    tree, _ = make_tree()
    chain(tree, ["a0", "a1", "a2"])
    tree.add("b0")
    tree.touch(["a1", "a2"])
    tree.remove("a0")
    # a2 is unreachable, so it goes before the older reachable b0.
    assert tree.pop_victim("l2") == "a2"
    tree.refresh("a2")

    tree.add("a0")
    # Re-adopted nodes fall back to plain leaf LRU.
    tree.touch(["a1", "a2"])
    assert drain(tree) == ["b0", "a2", "a1", "a0"]


def test_child_added_under_missing_parent_is_orphan() -> None:
    tree, _ = make_tree()
    tree.add("root")
    tree.add("child", "missing")
    tree.add("grandchild", "child")
    # The older root would go first if the new chain were reachable.
    assert drain(tree) == ["grandchild", "child", "root"]


def test_add_rejects_duplicates_and_self_parent() -> None:
    tree, _ = make_tree()
    tree.add("a")
    with pytest.raises(ValueError):
        tree.add("a")
    with pytest.raises(ValueError):
        tree.add("b", "b")
    tree.remove("unknown")
    tree.touch(["unknown"])
    tree.touch_path("unknown")
    tree.refresh("unknown")
    assert len(tree) == 1
    assert "a" in tree and "unknown" not in tree


def test_repeated_touches_keep_chain_order() -> None:
    tree, _ = make_tree()
    chain(tree, [f"k{i}" for i in range(10)])
    for _ in range(5000):
        tree.touch(["k9"])
    assert drain(tree) == [f"k{i}" for i in reversed(range(10))]


class Reference:
    """Brute-force model: scan every node for the min (orphan-first, tick).

    Mirrors the tree's tick sequence (one per ``add`` and per touched node)
    and derives orphan and leaf state from parent links on every query.
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.tick: dict[str, int] = {}
        self.clock = 0
        self.blocked: dict[str, set[str]] = {"l1": set(), "l2": set()}

    def add(self, key: str, parent: str) -> None:
        self.parent[key] = parent
        self._stamp(key)

    def remove(self, key: str) -> None:
        self.parent.pop(key, None)
        self.tick.pop(key, None)

    def touch(self, keys: list[str]) -> None:
        for key in keys:
            if key in self.parent:
                self._stamp(key)

    def touch_path(self, key: str) -> None:
        path = []
        while key and key in self.parent and key not in path:
            path.append(key)
            key = self.parent[key]
        self.touch(path[::-1])

    def is_leaf(self, key: str) -> bool:
        return all(parent != key for parent in self.parent.values())

    def orphan(self, key: str) -> bool:
        parent = self.parent[key]
        return bool(parent) and (parent not in self.parent or self.orphan(parent))

    def is_candidate(self, tier: str, key: str) -> bool:
        if key in self.blocked[tier]:
            return False
        # L1 is leaf-only; L2 admits any unblocked node, so tiers differ.
        return tier == "l2" or self.is_leaf(key)

    def victim(self, tier: str) -> str | None:
        candidates = [k for k in self.parent if self.is_candidate(tier, k)]
        if not candidates:
            return None
        return min(candidates, key=lambda k: (not self.orphan(k), self.tick[k]))

    def _stamp(self, key: str) -> None:
        self.clock += 1
        self.tick[key] = self.clock


@pytest.mark.parametrize("seed", range(40))
def test_victims_match_brute_force_reference(seed: int) -> None:
    rng = random.Random(seed)
    ref = Reference()
    tree = EvictionTree(("l1", "l2"), ref.is_candidate)
    # The predicate reads the reference, so it is updated before the tree.
    # Parents always have a lower index, so parent links never form a cycle;
    # a parent may be absent, which makes the node an orphan.
    universe = [f"k{i}" for i in range(80)]

    for _step in range(800):
        op = rng.random()
        present = list(ref.parent)
        if op < 0.2:
            index = rng.randrange(len(universe))
            key = universe[index]
            if key not in ref.parent:
                # Mostly chains, so orphaning reaches deep subtrees.
                if index and rng.random() < 0.8:
                    parent = universe[index - 1]
                else:
                    parent = rng.choice([""] + universe[:index])
                ref.add(key, parent)
                tree.add(key, parent)
        elif op < 0.3 and present:
            key = rng.choice(present)
            ref.remove(key)
            tree.remove(key)
        elif op < 0.45:
            keys = rng.sample(universe, rng.randrange(1, 5))
            tree.touch(keys)
            ref.touch(keys)
        elif op < 0.65:
            key = rng.choice(universe)
            tree.touch_path(key)
            ref.touch_path(key)
        elif op < 0.75:
            # Candidacy toggles; becoming a candidate is followed by refresh.
            key, tier = rng.choice(universe), rng.choice(("l1", "l2"))
            if key in ref.blocked[tier]:
                ref.blocked[tier].discard(key)
                tree.refresh(key)
            else:
                ref.blocked[tier].add(key)
        elif op < 0.8:
            key = rng.choice(universe)
            tree.refresh(key)
        elif op < 0.88 and present:
            # A stored-again parent: its subtree is orphaned, then re-adopted.
            key = rng.choice(present)
            parent = ref.parent[key]
            ref.remove(key)
            tree.remove(key)
            ref.add(key, parent)
            tree.add(key, parent)
        else:
            tier = rng.choice(("l1", "l2"))
            expected = ref.victim(tier)
            assert tree.pop_victim(tier) == expected
            if expected is None:
                continue
            if rng.random() < 0.5:
                ref.remove(expected)
                tree.remove(expected)
            else:
                # The caller keeps the node, e.g. it lost only this tier's copy.
                if rng.random() < 0.5:
                    ref.blocked[tier].add(expected)
                tree.refresh(expected)
        for key in universe:
            assert tree.is_leaf(key) == ref.is_leaf(key) or key not in ref.parent

    for tier in ("l1", "l2"):
        while (expected := ref.victim(tier)) is not None:
            assert tree.pop_victim(tier) == expected
            ref.remove(expected)
            tree.remove(expected)
        assert tree.pop_victim(tier) is None


@pytest.mark.parametrize("seed", range(10))
def test_victims_match_reference_while_stale_entries_pile_up(seed: int) -> None:
    rng = random.Random(seed)
    ref = Reference()
    tree = EvictionTree(("l1", "l2"), ref.is_candidate)
    keys = [f"c{i}" for i in range(60)]
    for parent, key in zip([""] + keys[:-1], keys, strict=True):
        ref.add(key, parent)
        tree.add(key, parent)

    for _step in range(400):
        op = rng.random()
        if op < 0.4:
            # Re-storing a chain node orphans and re-adopts its whole subtree.
            key = rng.choice([k for k in keys if k in ref.parent])
            parent = ref.parent[key]
            ref.remove(key)
            tree.remove(key)
            ref.add(key, parent)
            tree.add(key, parent)
        elif op < 0.7:
            key = rng.choice(keys)
            ref.touch_path(key)
            tree.touch_path(key)
        elif op < 0.8:
            key, tier = rng.choice(keys), rng.choice(("l1", "l2"))
            if key in ref.blocked[tier]:
                ref.blocked[tier].discard(key)
                tree.refresh(key)
            else:
                ref.blocked[tier].add(key)
        else:
            tier = rng.choice(("l1", "l2"))
            expected = ref.victim(tier)
            assert tree.pop_victim(tier) == expected
            if expected is not None:
                tree.refresh(expected)

    for tier in ("l1", "l2"):
        while (expected := ref.victim(tier)) is not None:
            assert tree.pop_victim(tier) == expected
            ref.remove(expected)
            tree.remove(expected)


def test_touching_a_queued_path_skips_the_predicate() -> None:
    calls = 0

    def always(tier: str, key: str) -> bool:
        nonlocal calls
        calls += 1
        return True

    tree = EvictionTree(("l1", "l2"), always)
    keys = [f"d{i}" for i in range(100)]
    chain(tree, keys)
    calls = 0
    for _ in range(1000):
        tree.touch_path(keys[-1])
    # Every node is already queued in both tiers, so nothing is re-checked.
    assert calls == 0
    order = []
    while (victim := tree.pop_victim("l1")) is not None:
        order.append(victim)
        tree.remove(victim)
    assert order == keys
    # One check per node when it is finally popped at its true priority.
    assert calls == len(keys)


def test_repeated_touches_and_orphaning_keep_heap_memory_bounded() -> None:
    tree, member = make_tree()
    member.leaf_only = False
    keys = [f"d{i}" for i in range(100)]

    def cycle() -> None:
        tree.touch_path(keys[-1])
        # Orphaning the subtree lowers every priority and re-queues at once.
        tree.remove(keys[0])
        tree.add(keys[0])

    chain(tree, keys)
    tracemalloc.start()
    try:
        for _ in range(100):
            cycle()
        warm, _peak = tracemalloc.get_traced_memory()
        for _ in range(1000):
            cycle()
        after, _peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Unbounded queueing would add ~100 heap entries (>7 KiB) per cycle.
    assert after - warm < 64 * 1024
    assert drain(tree) == keys[1:] + keys[:1]


def test_queue_hint_keeps_hinted_chain_after_unhinted() -> None:
    """Hinted nodes go after every unhinted one, even more recent ones."""
    tree, _ = make_tree()
    chain(tree, ["a0", "a1"])
    chain(tree, ["b0", "b1"])
    tree.want("r1", ["a0", "a1"])
    assert tree.is_wanted("a1") and not tree.is_wanted("b1")
    assert tree.hinted_requests == 1 and tree.hinted_keys == 2
    assert drain(tree) == ["b1", "b0", "a1", "a0"]


def test_queue_hint_evicts_latest_hint_first() -> None:
    """Among hinted chains, the one that will be admitted last goes first."""
    tree, _ = make_tree()
    chain(tree, ["a0", "a1"])
    chain(tree, ["b0", "b1"])
    tree.want("first", ["b0", "b1"])
    tree.want("second", ["a0", "a1"])
    assert drain(tree) == ["a1", "a0", "b1", "b0"]


def test_queue_hint_shared_prefix_follows_earliest_hint() -> None:
    """A key hinted by several requests keeps the earliest one's rank."""
    tree, _ = make_tree()
    chain(tree, ["p", "a"])
    tree.add("b", "p")
    tree.want("first", ["p", "a"])
    tree.want("second", ["p", "b"])
    assert drain(tree) == ["b", "a", "p"]


def test_queue_hint_orphans_still_go_first() -> None:
    """A hint cannot rescue an unreachable node."""
    tree, _ = make_tree()
    tree.add("x", "gone")
    chain(tree, ["a0"])
    tree.want("r", ["x"])
    assert drain(tree) == ["x", "a0"]


def test_queue_release_makes_keys_most_recent_at_once() -> None:
    """Released keys rank as just used, not by their pre-hint tick."""
    tree, _ = make_tree()
    chain(tree, ["a0", "a1"])
    chain(tree, ["b0", "b1"])
    tree.want("r", ["a0", "a1"])
    # Queue the hinted priority, then release: the lower priority must apply
    # immediately even though entries for the hinted rank are queued.
    assert tree.pop_victim("l2") == "b1"
    tree.refresh("b1")
    tree.unwant("r")
    assert not tree.is_wanted("a0") and tree.hinted_keys == 0
    assert drain(tree) == ["b1", "b0", "a1", "a0"]
    tree.unwant("r")  # unknown ids are ignored


def test_queue_release_of_earliest_hint_lowers_shared_key() -> None:
    """Dropping the earliest of two hints re-ranks a shared key by the later."""
    tree, _ = make_tree()
    chain(tree, ["a0"])
    chain(tree, ["b0"])
    tree.want("first", ["a0"])
    tree.want("second", ["b0"])
    tree.want("third", ["a0"])
    # Surface both at their hinted rank so the queued entries are current.
    assert tree.pop_victim("l2") == "b0"
    tree.refresh("b0")
    tree.unwant("first")
    # a0 now ranks by "third" (after "second"), so it goes first.
    assert drain(tree) == ["a0", "b0"]


def test_queue_hint_applies_to_keys_added_later() -> None:
    """A chunk committed while hinted inherits the hint."""
    tree, _ = make_tree()
    tree.want("r", ["a0"])
    chain(tree, ["b0"])
    chain(tree, ["a0"])
    chain(tree, ["c0"])
    assert drain(tree) == ["b0", "c0", "a0"]


def test_queue_hint_replaces_same_id() -> None:
    """Re-hinting an id drops its old keys."""
    tree, _ = make_tree()
    chain(tree, ["a0"])
    chain(tree, ["b0"])
    tree.want("r", ["a0"])
    tree.want("r", ["b0"])
    assert tree.hinted_requests == 1 and not tree.is_wanted("a0")
    assert drain(tree) == ["a0", "b0"]
