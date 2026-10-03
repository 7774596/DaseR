# SPDX-License-Identifier: Apache-2.0
"""Tests for the shadow model of GPU prefix-cache residency."""

import pytest

from daser.server.gpu_shadow import GpuResidencyShadow


def test_serve_keeps_heads_and_evicts_tails_of_older_prompts() -> None:
    """Capacity drops the oldest prompt's tail before its head."""
    shadow = GpuResidencyShadow(capacity_tokens=30)
    resident, evicted = shadow.serve([("a0", 10, "A0"), ("a1", 10, "A1")])
    assert resident == ["A1", "A0"]
    assert evicted == []
    _resident, evicted = shadow.serve([("b0", 10, "B0"), ("b1", 10, "B1")])
    assert evicted == ["A1"]
    assert "a0" in shadow and "a1" not in shadow
    assert shadow.resident_tokens == 30


def test_serve_refreshes_a_shared_chunk_without_double_counting() -> None:
    """A chunk served again moves to the front and keeps one token charge."""
    shadow = GpuResidencyShadow(capacity_tokens=20)
    shadow.serve([("s", 10, "S"), ("a", 10, "A")])
    _resident, evicted = shadow.serve([("s", 10, "S"), ("b", 10, "B")])
    assert evicted == ["A"]
    assert shadow.resident_tokens == 20


def test_drop_forgets_chunks_the_gpu_did_not_hold() -> None:
    """Only held keys are returned; unknown keys are ignored."""
    shadow = GpuResidencyShadow(capacity_tokens=100)
    shadow.serve([("a", 10, "A"), ("b", 10, "B")])
    assert shadow.drop(["b", "zz"]) == ["B"]
    assert shadow.resident_tokens == 10


def test_capacity_must_be_positive() -> None:
    """A zero capacity is rejected rather than evicting everything."""
    with pytest.raises(ValueError, match="positive"):
        GpuResidencyShadow(0)
