# SPDX-License-Identifier: Apache-2.0
"""Tests for the research summary's send-order check and overlap bound."""

import json
from pathlib import Path

from benchmarks.research.summarize import (
    arm_row,
    decode_rows,
    live_clock_rows,
    order_rows,
    overlap_bound_rows,
    request_hit_rows,
    vllm_tick_rows,
)


def _arm(root: Path, name: str, send_offsets: list[float]) -> Path:
    """Write a requests.jsonl whose trace index i was sent at send_offsets[i]."""
    arm_dir = root / name
    arm_dir.mkdir()
    with (arm_dir / "requests.jsonl").open("w") as f:
        for index, offset in enumerate(send_offsets):
            f.write(json.dumps({"trace_index": index, "send_offset_s": offset}) + "\n")
    return arm_dir


def test_order_check_flags_only_reordered_replays(tmp_path: Path) -> None:
    """In-order replays pass; a swapped pair is counted and flagged."""
    in_order = _arm(tmp_path, "in", [0.0, 1.0, 2.0, 3.0])
    swapped = _arm(tmp_path, "swapped", [0.0, 2.0, 1.0, 3.0])

    good, bad = order_rows([in_order, swapped])

    assert "out_of_order=0/4" in good and "ANOMALY" not in good
    assert "out_of_order=2/4" in bad and "ANOMALY" in bad


def _ttft_arm(root: Path, name: str, ttft_ms: list[float]) -> Path:
    """Write a completed requests.jsonl whose trace index i had ttft_ms[i]."""
    arm_dir = root / name / "art" / "arm"
    arm_dir.mkdir(parents=True)
    with (arm_dir / "requests.jsonl").open("w") as f:
        for index, ttft in enumerate(ttft_ms):
            record = {
                "trace_index": index,
                "status": "completed",
                "ttft_ms": ttft,
                "latency_ms": 2 * ttft,
            }
            f.write(json.dumps(record) + "\n")
    return arm_dir


def test_overlap_bound_hides_at_most_the_oracle_work(tmp_path: Path) -> None:
    """Each request keeps max(oracle, extra wait), never more than its TTFT."""
    oracle = _ttft_arm(tmp_path, "oracle", [1000.0, 1000.0, 1000.0])
    # Extra waits 500 (fully hidden), 3000 (only 1000 hidden), and a request
    # faster than the oracle (noise, kept as measured).
    arm = _ttft_arm(tmp_path, "arm", [1500.0, 4000.0, 900.0])

    header, row = overlap_bound_rows([arm], oracle, first=3)

    assert "on 3 requests" in header and "oracle=3" in header
    assert "ttft=6 " in row
    assert "extra=3.5 " in row
    assert "bound=5 " in row  # 1000 + 3000 + 900
    assert "hidden<=1.5" in row


def test_live_clock_check_flags_poll_counting(tmp_path: Path) -> None:
    """One tick per request passes; a clock counting polls is flagged."""
    arms = []
    for name, ticks in (("ok", 21), ("polls", 80), ("off", None)):
        arm_dir = _arm(tmp_path, name, [float(i) for i in range(20)])
        if ticks is not None:
            (arm_dir / "daser_metrics_before.prom").write_text(
                "daser_arena_live_tick 1.0\n"
            )
            (arm_dir / "daser_metrics_after.prom").write_text(
                f"daser_arena_live_tick {ticks}.0\n"
            )
        arms.append(arm_dir)

    good, bad = live_clock_rows(arms)

    assert "live_ticks=20 requests=20" in good and "ANOMALY" not in good
    assert "live_ticks=79" in bad and "ANOMALY" in bad


def test_request_hit_check_reconciles_cached_tokens(tmp_path: Path) -> None:
    """Per-request cached tokens matching the counters pass; a gap is flagged."""
    arms = []
    for name, cached in (("ok", [0, 80, 90]), ("gap", [0, 0, 0]), ("none", None)):
        arm_dir = tmp_path / name
        arm_dir.mkdir()
        with (arm_dir / "requests.jsonl").open("w") as f:
            for i in range(3):
                record = {"trace_index": i, "prompt_tokens": 100}
                if cached is not None:
                    record["cached_tokens"] = cached[i]
                f.write(json.dumps(record) + "\n")
        arms.append(arm_dir)

    good, bad = request_hit_rows(arms, ["0.567", "0.567", "0.567"])

    assert "n=3 request_hit=0.567" in good and "ANOMALY" not in good
    assert "request_hit=0.000" in bad and "ANOMALY" in bad


def test_arm_row_reports_load_share_of_ttft(tmp_path: Path) -> None:
    """io_share divides summed DaseR load latency by summed vLLM TTFT."""
    arm_dir = _ttft_arm(tmp_path, "io", [100.0, 300.0])
    summary = {
        "completed_requests": 2,
        "failed_requests": 0,
        "wall_seconds": 10.0,
        "client_queue_ms": {"p50": 0.0},
        "arrival_to_completion_ms": {"p50": 0.0},
    }
    (arm_dir / "summary.json").write_text(json.dumps(summary))
    (arm_dir / "daser_metrics_after.prom").write_text(
        'daser_transfer_duration_seconds_sum{op="load"} 5.0\n'
    )
    (arm_dir / "vllm_metrics_after.prom").write_text(
        'vllm:time_to_first_token_seconds_sum{engine="0"} 20.0\n'
    )

    row = arm_row(arm_dir)

    assert row["io_share"] == "0.250"
    assert row["sum_ttft_s"] == "20"
    assert row["csum_lat_s"] == "1"  # (200 + 600) ms


def test_arm_row_sums_counters_over_vllm_instances(tmp_path: Path) -> None:
    """A two-instance snapshot repeats each sample; the row sums both."""
    arm_dir = _ttft_arm(tmp_path, "multi", [100.0, 300.0])
    summary = {
        "completed_requests": 2,
        "failed_requests": 0,
        "wall_seconds": 10.0,
        "client_queue_ms": {"p50": 0.0},
        "arrival_to_completion_ms": {"p50": 0.0},
    }
    (arm_dir / "summary.json").write_text(json.dumps(summary))
    sample = 'vllm:time_to_first_token_seconds_sum{engine="0"}'
    (arm_dir / "vllm_metrics_after.prom").write_text(
        f"# instance a\n{sample} 20.0\n# instance b\n{sample} 12.0\n"
    )

    row = arm_row(arm_dir)

    assert row["sum_ttft_s"] == "32"


def test_vllm_ticks_split_waiting_by_kv_capacity(tmp_path: Path) -> None:
    """Waiting ticks at high KV usage count as capacity-bound, others compute."""
    arm_dir = tmp_path / "arm"
    arm_dir.mkdir()
    line = (
        "(APIServer pid=1) INFO [loggers.py:315] Engine 000: Avg prompt "
        "throughput: {pf} tokens/s, Avg generation throughput: 9.0 tokens/s, "
        "Running: {run} reqs, Waiting: {wait} reqs, GPU KV cache usage: {kv}%, "
        "Prefix cache hit rate: 0.0%\n"
    )
    ticks = [(0, 0, 0, 0.0), (100, 1, 2, 90.0), (300, 1, 1, 70.0), (50, 3, 1, 20.0)]
    (arm_dir / "vllm_daser.log").write_text(
        "".join(line.format(pf=pf, run=r, wait=w, kv=kv) for pf, r, w, kv in ticks)
    )

    (row,) = vllm_tick_rows([arm_dir, tmp_path])

    assert "ticks=4 busy=3 waiting=3 kv>=60%=2 compute=1" in row
    assert "running<=1=2 pf_tok_s@kvfull=200" in row


def test_decode_rows_fit_tpot_against_overlapping_context(tmp_path: Path) -> None:
    """TPOT = floor + slope x context is recovered from disjoint decodes."""
    arm_dir = tmp_path / "arm"
    arm_dir.mkdir()
    records = []
    for i, ctx in enumerate((100_000, 200_000, 300_000)):
        tpot = 5.0 + 10.0 * ctx / 1e5
        records.append(
            {
                "ttft_ms": 100.0,
                "latency_ms": 100.0 + 20 * tpot,
                "completion_offset_s": 10.0 * (i + 1),
                "input_length": ctx,
                "completion_tokens": 21,
            }
        )
    (arm_dir / "requests.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )

    (row,) = decode_rows([arm_dir])

    assert "tpot_ms=25.0" in row
    assert "fit n=3 floor_ms=5.0 ms_per_100k_ctx=10.00" in row
