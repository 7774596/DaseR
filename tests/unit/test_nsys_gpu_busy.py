# SPDX-License-Identifier: Apache-2.0
"""Tests for the nsys GPU busy/idle summary helpers."""

from benchmarks.research.nsys_gpu_busy import category, merge, overlap_ns


def test_merge_joins_overlapping_and_touching_intervals() -> None:
    """Overlapping or touching spans collapse; disjoint ones stay apart."""
    assert merge([(5, 7), (0, 2), (1, 3), (3, 4)]) == [(0, 4), (5, 7)]


def test_overlap_counts_only_covered_copy_time() -> None:
    """A copy half inside a kernel interval reports half its time."""
    assert overlap_ns([(0, 10), (20, 30)], [(5, 25)]) == 10


def test_category_names_attention_before_gemm() -> None:
    """FlashAttention kernels built on cutlass count as attention."""
    assert category("cutlass::device_kernel<flash::FlashAttnFwdSm90<>>") == "attention"
    assert category("nvjet_tst_256x8_64x6_4x1_v_bz_TNT") == "gemm"
    assert category("unrelated_kernel") == "other"
