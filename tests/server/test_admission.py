# SPDX-License-Identifier: Apache-2.0
"""Finish-time admission decisions for held chunks."""

from daser.server.admission import FinishAdmission

R1 = [(0, 4096)]
R2 = [(4096, 4096), (8192, 4096)]


def test_long_output_persists_held_ranges() -> None:
    policy = FinishAdmission(500)
    policy.hold("a", R1)
    policy.hold("b", R2)
    actions = policy.finish(["a", "b"], output_tokens=500)
    assert actions.persist == R1 + R2
    assert actions.evict == []
    assert actions.admitted == 2
    assert policy.held_chunks == 0


def test_short_output_evicts_and_forget_returns_ranges() -> None:
    policy = FinishAdmission(500)
    policy.hold("a", R1)
    actions = policy.finish(["a"], output_tokens=10)
    assert actions.persist == []
    assert actions.evict == ["a"]
    # The core's removal listener then asks for the ranges to discard.
    assert policy.forget("a") == R1
    assert policy.held_chunks == 0


def test_hit_while_held_rescues_short_output() -> None:
    policy = FinishAdmission(500)
    policy.hold("a", R1)
    policy.mark_hit(["a", "unknown"])
    actions = policy.finish(["a"], output_tokens=10)
    assert actions.persist == R1
    assert actions.rescued == 1
    assert actions.admitted == 0


def test_decision_before_store_applies_on_hold() -> None:
    policy = FinishAdmission(500)
    assert policy.finish(["a", "b"], output_tokens=10).evict == []
    assert policy.hold("a", R1).evict == ["a"]
    policy.forget("b")
    # A forgotten early decision no longer applies to a later store.
    assert policy.hold("b", R2).evict == []
    assert policy.held_chunks == 2


def test_forget_unknown_chunk_is_empty() -> None:
    assert FinishAdmission(1).forget("missing") == []
