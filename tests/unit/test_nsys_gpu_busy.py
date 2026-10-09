# SPDX-License-Identifier: Apache-2.0
"""Tests for the nsys GPU busy/idle summary helpers."""

from pathlib import Path
import sqlite3

from benchmarks.research.nsys_gpu_busy import (
    category,
    merge,
    overlap_ns,
    phase_gaps,
    summarize,
)


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


def test_cuda_graph_replays_count_as_busy(tmp_path: Path) -> None:
    """Graph-traced replays fill the gaps between eager kernels."""
    db = tmp_path / "graphs.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE StringIds (id INT, value TEXT)")
    con.execute("INSERT INTO StringIds VALUES (1, 'flash_fwd_kernel')")
    con.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
        "(start INT, end INT, demangledName INT, shortName INT)"
    )
    con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 2000000, 1, 1)")
    con.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (8000000, 10000000, 1, 1)"
    )
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INT, end INT)")
    con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (2000000, 8000000)")
    con.commit()
    con.close()
    lines = summarize(db)
    assert " busy=1.000 " in lines[0]
    assert "gaps>1ms n=0 " in lines[0]
    assert lines[1] == "  kernel share: cuda_graph=0.600 attention=0.400"


def test_phase_gaps_assigns_each_gap_to_its_dominant_phase() -> None:
    """Gaps go to first-token, decode or idle by coverage; copies are measured."""
    busy = [
        (0, 10_000_000),
        (15_000_000, 20_000_000),
        (30_000_000, 31_000_000),
        (40_000_000, 50_000_000),
    ]
    result = phase_gaps(
        busy,
        first_token=[(10_000_000, 15_000_000)],
        decode=[(20_000_000, 29_000_000)],
        copies=[(11_000_000, 13_000_000)],
    )
    assert result["first_token"] == (1, 5_000_000, 2_000_000)
    assert result["decode"] == (1, 10_000_000, 0)
    assert result["no_request"] == (1, 9_000_000, 0)
