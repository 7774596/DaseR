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


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def visit(
    tree: EvictionTree, clock: FakeClock, at: float, key: str, parent: str
) -> None:
    clock.now = at
    tree.add(key, parent)


def return_tree() -> tuple[EvictionTree, FakeClock]:
    clock = FakeClock()
    return EvictionTree(policy="return", clock=clock), clock


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        EvictionTree(policy="mru")


def test_return_policy_evicts_the_thread_that_returns_last():
    # A and B alternate with period 10. LRU would evict A, which returns
    # next; the return-aware order evicts B (predicted at 35, A at 30).
    tree, clock = return_tree()
    visit(tree, clock, 0, "a0", "")
    visit(tree, clock, 5, "b0", "")
    visit(tree, clock, 10, "a1", "a0")
    visit(tree, clock, 15, "b1", "b0")
    visit(tree, clock, 20, "a2", "a1")
    visit(tree, clock, 25, "b2", "b1")
    clock.now = 26
    assert tree.pop_victim(never) == "b2"
    assert EvictionTree().pop_victim(never) is None


def test_return_policy_evicts_overdue_threads_first():
    tree, clock = return_tree()
    visit(tree, clock, 0, "a0", "")
    visit(tree, clock, 10, "a1", "a0")
    for t in range(10):
        visit(tree, clock, 12 + 10 * t, f"b{t}", f"b{t - 1}" if t else "")
    # A (period 10) has been silent for 101 s; B returns every 10 s.
    clock.now = 111
    assert drain(tree)[:2] == ["a1", "a0"]


def test_return_policy_still_evicts_orphans_first():
    tree, clock = return_tree()
    visit(tree, clock, 0, "a0", "")
    visit(tree, clock, 10, "a1", "a0")
    visit(tree, clock, 20, "x1", "missing")
    clock.now = 21
    assert tree.pop_victim(never) == "x1"


def test_return_policy_takes_oldest_leaf_of_the_chosen_thread():
    tree, clock = return_tree()
    visit(tree, clock, 0, "root", "")
    visit(tree, clock, 0.1, "old", "root")
    visit(tree, clock, 0.2, "new", "root")
    clock.now = 1
    assert drain(tree) == ["old", "new", "root"]


def test_return_policy_counts_prefix_hits_as_arrivals():
    tree, clock = return_tree()
    visit(tree, clock, 0, "a0", "")
    visit(tree, clock, 1, "b0", "")
    visit(tree, clock, 10, "a1", "a0")
    visit(tree, clock, 11, "b1", "b0")
    # A hit on A's chain at 20 moves its predicted return past B's (21).
    clock.now = 20
    tree.touch_path("a1")
    clock.now = 20.5
    assert tree.pop_victim(never) == "a1"


def test_return_policy_charged_only_skips_held_leaves():
    tree, clock = return_tree()
    visit(tree, clock, 0, "a0", "")
    visit(tree, clock, 5, "b0", "")
    tree.mark_held("b0")
    clock.now = 6
    assert tree.pop_victim(never, charged_only=True) == "a0"
    assert tree.pop_victim(never) == "b0"


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
