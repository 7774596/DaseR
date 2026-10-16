# SPDX-License-Identifier: Apache-2.0

import pytest

from daser.server.eviction import EvictionTree


def never(key: str) -> bool:
    return False


def drain(tree: EvictionTree, skip=never) -> list[str]:
    """Evict until empty, returning the victim order."""
    order = []
    while (victim := tree.pop_victim(skip)) is not None:
        order.append(victim)
        tree.remove(victim)
    return order


def chain(tree: EvictionTree, keys: list[str]) -> None:
    parent = ""
    for key in keys:
        tree.add(key, parent)
        parent = key


def test_chain_evicts_from_tail_and_root_last():
    tree = EvictionTree()
    chain(tree, ["a0", "a1", "a2"])
    assert drain(tree) == ["a2", "a1", "a0"]


def test_parent_waits_for_its_last_child():
    tree = EvictionTree()
    tree.add("root")
    tree.add("left", "root")
    tree.add("right", "root")
    tree.touch(["root", "left"])
    assert drain(tree) == ["right", "left", "root"]


def test_least_recently_used_leaf_goes_first():
    tree = EvictionTree()
    chain(tree, ["a0", "a1"])
    chain(tree, ["b0", "b1"])
    tree.touch(["a0", "a1"])
    assert tree.pop_victim(never) == "b1"


def test_touch_of_a_hit_chain_protects_its_root():
    tree = EvictionTree()
    chain(tree, ["a0", "a1", "a2"])
    tree.add("b0")
    tree.touch(["a0"])
    # a2 is the coldest leaf even though the chain root was just hit.
    assert tree.pop_victim(never) == "a2"


def test_skipped_leaf_is_dropped_until_refresh():
    tree = EvictionTree()
    tree.add("pinned")
    tree.add("free")
    assert tree.pop_victim(lambda key: key == "pinned") == "free"
    tree.remove("free")
    assert tree.pop_victim(never) is None
    tree.refresh("pinned")
    assert tree.pop_victim(never) == "pinned"


def test_skipped_child_keeps_parent_resident():
    tree = EvictionTree()
    chain(tree, ["a0", "a1"])
    assert tree.pop_victim(lambda key: key == "a1") is None
    assert "a0" in tree


def test_orphans_go_first_and_parent_return_readopts():
    tree = EvictionTree()
    chain(tree, ["a0", "a1", "a2"])
    tree.add("b0")
    tree.touch(["a1", "a2"])
    tree.remove("a0")
    # a2 is unreachable, so it goes before the older reachable b0.
    assert tree.pop_victim(never) == "a2"

    tree.add("a0")
    # Re-adopted nodes fall back to plain leaf LRU.
    tree.touch(["a1", "a2"])
    assert drain(tree) == ["b0", "a2", "a1", "a0"]


def test_child_added_under_missing_parent_is_orphan():
    tree = EvictionTree()
    tree.add("root")
    tree.add("child", "missing")
    tree.add("grandchild", "child")
    # The older root would go first if the new chain were reachable.
    assert drain(tree) == ["grandchild", "child", "root"]


def test_add_rejects_duplicates_and_self_parent():
    tree = EvictionTree()
    tree.add("a")
    with pytest.raises(ValueError):
        tree.add("a")
    with pytest.raises(ValueError):
        tree.add("b", "b")
    tree.remove("unknown")
    tree.touch(["unknown"])
    tree.refresh("unknown")
    assert len(tree) == 1


def test_repeated_touches_keep_chain_order():
    tree = EvictionTree()
    chain(tree, [f"k{i}" for i in range(10)])
    for _ in range(5000):
        tree.touch(["k9"])
    assert drain(tree) == [f"k{i}" for i in reversed(range(10))]


def test_charged_frontier_moves_past_held_children():
    tree = EvictionTree()
    chain(tree, ["a0", "a1", "a2"])
    assert tree.pop_victim(never, charged_only=True) == "a2"
    tree.mark_held("a2")
    assert tree.pop_victim(never, charged_only=True) == "a1"
    tree.mark_held("a1")
    assert tree.pop_victim(never, charged_only=True) == "a0"
    # Leaf eviction still takes the held tail first.
    assert tree.pop_victim(never) == "a2"


def test_held_closure_is_enforced():
    tree = EvictionTree()
    chain(tree, ["a0", "a1"])
    with pytest.raises(ValueError):
        tree.mark_held("a0")  # its child is charged
    tree.mark_held("a1")
    tree.mark_held("a0")
    with pytest.raises(ValueError):
        tree.add("a2", "a1")  # held parent
    with pytest.raises(ValueError):
        tree.mark_charged("a1")  # held parent
    assert tree.held_path("a1") == ["a0", "a1"]
    for key in tree.held_path("a1"):
        tree.mark_charged(key)
    tree.add("a2", "a1")
    assert tree.pop_victim(never, charged_only=True) == "a2"


def test_descendants_are_deepest_first():
    tree = EvictionTree()
    chain(tree, ["r", "x", "y"])
    tree.add("z", "r")
    order = tree.descendants("r")
    assert sorted(order) == ["x", "y", "z"]
    assert order.index("y") < order.index("x")
    assert tree.descendants("y") == []
