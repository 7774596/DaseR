# SPDX-License-Identifier: Apache-2.0

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
