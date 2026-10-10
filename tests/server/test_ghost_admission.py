# SPDX-License-Identifier: Apache-2.0
"""Ghost admission filter in front of the external tier."""

import pytest

from daser.server.ghost_admission import GHOST_ADMIT_ENV, GhostAdmission


def test_fixed_filter_admits_on_second_store() -> None:
    policy = GhostAdmission(capacity=4)
    assert policy.admit("a") is False
    assert policy.admit("a") is True
    # An admitted key leaves the ghost list; a later first store is new.
    assert policy.admit("a") is False
    assert (policy.admitted, policy.denied) == (1, 2)


def test_ghost_list_is_bounded_by_capacity() -> None:
    policy = GhostAdmission(capacity=2)
    for key in ("a", "b", "c"):
        assert policy.admit(key) is False
    assert policy.admit("a") is False  # evicted from the ghost list
    assert policy.admit("c") is True


def test_adaptive_admits_until_scans_dominate() -> None:
    policy = GhostAdmission(capacity=64, adaptive=True, sample=1)
    assert policy.filtering is False
    assert policy.admit("x") is True
    # A hot set that fits: always-admit hits after the first pass, the
    # ghost-filtered shadow pays one extra miss per key.
    hot = [f"h{i}" for i in range(16)]
    for _ in range(4):
        policy.observe(hot)
    assert policy.filtering is False
    # The hot set interleaved with one-time keys several times the capacity:
    # always-admit loses the hot set, the filtered shadow keeps it.
    for i in range(200):
        policy.observe(hot + [f"s{i}-{j}" for j in range(64)])
    assert policy.filtering is True
    assert policy.admit("y") is False


def test_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GHOST_ADMIT_ENV, raising=False)
    assert GhostAdmission.from_env(8) is None
    monkeypatch.setenv(GHOST_ADMIT_ENV, "adaptive")
    policy = GhostAdmission.from_env(8)
    assert policy is not None and policy.adaptive
    monkeypatch.setenv(GHOST_ADMIT_ENV, "bogus")
    with pytest.raises(ValueError):
        GhostAdmission.from_env(8)
