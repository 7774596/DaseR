# SPDX-License-Identifier: Apache-2.0
"""Tests for the nsys GPU busy/idle summary helpers."""

from pathlib import Path
import sqlite3

from benchmarks.research.nsys_gpu_busy import category, merge, overlap_ns, summarize


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


def test_copy_only_report_still_lists_copies(tmp_path: Path) -> None:
    """A process without kernels (DaseR server) still reports its copies."""
    db = tmp_path / "copy_only.sqlite"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY "
        "(start INT, end INT, bytes INT, copyKind INT)"
    )
    con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (0, 10, 2000000000, 2)")
    con.commit()
    con.close()
    lines = summarize(db)
    assert lines[0] == "copy_only.sqlite: no kernels"
    assert lines[1].startswith("  D2H: n=1 GB=2.0 ")
