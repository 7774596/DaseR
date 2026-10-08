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
    replay_concurrency,
    request_hit_rows,
    slo_rows,
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


def test_arm_row_marks_daser_columns_absent_without_a_daser_server(
    tmp_path: Path,
) -> None:
    """An arm with an empty DaseR snapshot (LMCache, native) prints "-"."""
    arm_dir = _ttft_arm(tmp_path, "lmcache", [100.0, 300.0])
    summary = {
        "completed_requests": 2,
        "failed_requests": 0,
        "wall_seconds": 10.0,
        "client_queue_ms": {"p50": 0.0},
        "arrival_to_completion_ms": {"p50": 0.0},
    }
    (arm_dir / "summary.json").write_text(json.dumps(summary))
    (arm_dir / "daser_metrics_after.prom").write_text("")
    (arm_dir / "vllm_metrics_after.prom").write_text(
        'vllm:time_to_first_token_seconds_sum{engine="0"} 20.0\n'
    )

    row = arm_row(arm_dir)

    for column in (
        "io_share",
        "l1_hits",
        "load_gib",
        "store_gib",
        "l2w_gib",
        "ssd_r_gib",
        "ra_wait_s",
    ):
        assert row[column] == "-"
    assert row["sum_ttft_s"] == "20"


def test_arm_row_reports_ssd_reads_and_readahead_wait(tmp_path: Path) -> None:
    """ssd_r_gib is readahead bytes read from SSD; ra_wait_s is load wait on it."""
    arm_dir = _ttft_arm(tmp_path, "ssd", [100.0, 300.0])
    summary = {
        "completed_requests": 2,
        "failed_requests": 0,
        "wall_seconds": 10.0,
        "client_queue_ms": {"p50": 0.0},
        "arrival_to_completion_ms": {"p50": 0.0},
    }
    (arm_dir / "summary.json").write_text(json.dumps(summary))
    (arm_dir / "daser_metrics_before.prom").write_text(
        'daser_readahead_bytes_total{outcome="read"} 1073741824.0\n'
        'daser_load_phase_seconds_total{phase="readahead_wait"} 1.0\n'
    )
    (arm_dir / "daser_metrics_after.prom").write_text(
        'daser_readahead_bytes_total{outcome="read"} 4294967296.0\n'
        'daser_readahead_bytes_total{outcome="served"} 9663676416.0\n'
        'daser_load_phase_seconds_total{phase="readahead_wait"} 3.5\n'
    )
    (arm_dir / "vllm_metrics_after.prom").write_text(
        'vllm:time_to_first_token_seconds_sum{engine="0"} 20.0\n'
    )

    row = arm_row(arm_dir)

    assert row["ssd_r_gib"] == "3.0"
    assert row["ra_wait_s"] == "2.5"


def test_slo_goodput_needs_both_ttft_and_tpot(tmp_path: Path) -> None:
    """Goodput counts requests meeting both SLOs; failures count as misses."""
    arm_dir = tmp_path / "load" / "art" / "arm"
    arm_dir.mkdir(parents=True)
    records = [
        # meets both: TTFT 100, TPOT (1100 - 100) / 10 = 100
        {
            "status": "completed",
            "ttft_ms": 100.0,
            "latency_ms": 1100.0,
            "completion_tokens": 11,
        },
        # TPOT too slow: (3100 - 100) / 10 = 300
        {
            "status": "completed",
            "ttft_ms": 100.0,
            "latency_ms": 3100.0,
            "completion_tokens": 11,
        },
        # TTFT too slow
        {
            "status": "completed",
            "ttft_ms": 900.0,
            "latency_ms": 1000.0,
            "completion_tokens": 2,
        },
        # one token: TTFT alone decides
        {
            "status": "completed",
            "ttft_ms": 50.0,
            "latency_ms": 50.0,
            "completion_tokens": 1,
        },
        {"status": "failed", "ttft_ms": 0.0, "latency_ms": 0.0},
    ]
    with (arm_dir / "requests.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    (arm_dir / "summary.json").write_text(json.dumps({"wall_seconds": 4.0}))

    (line,) = slo_rows([arm_dir], ttft_ms=500.0, tpot_ms=150.0)[1:]

    assert "ok=4" in line
    assert "slo=0.400" in line
    assert "goodput=0.500/s" in line


def test_slo_trim_keeps_the_steady_state_window(tmp_path: Path) -> None:
    """Trimming drops ramp-up/drain sends and divides by the kept window."""
    arm_dir = tmp_path / "pop" / "art" / "arm"
    arm_dir.mkdir(parents=True)
    with (arm_dir / "requests.jsonl").open("w") as f:
        for i, send in enumerate((0.5, 3.0, 5.0, 9.5)):
            record = {
                "status": "completed",
                "ttft_ms": 9000.0 if send < 1 else 100.0,
                "latency_ms": 200.0,
                "completion_tokens": 2,
                "session_id": i % 2,
                "send_offset_s": send,
                "completion_offset_s": send + 0.2,
            }
            f.write(json.dumps(record) + "\n")
    (arm_dir / "summary.json").write_text(json.dumps({"wall_seconds": 10.0}))

    (full,) = slo_rows([arm_dir], ttft_ms=500.0, tpot_ms=150.0)[1:]
    (kept,) = slo_rows([arm_dir], ttft_ms=500.0, tpot_ms=150.0, trim_frac=0.1)[1:]

    assert "ok=4" in full and "slo=0.750" in full and "goodput=0.300/s" in full
    assert "ok=2" in kept and "slo=1.000" in kept and "goodput=0.250/s" in kept
    # Window 1-9 s: session 0 (0.5-5.2 s) covers 4.2 s, session 1 (3.0-9.7 s)
    # covers 6.0 s → 10.2 / 8 = 1.3 active.
    assert "active=1.3" in kept


def test_arm_row_reports_speculative_acceptance_length(tmp_path: Path) -> None:
    """acc is generated tokens per engine step; "-" without step counts."""
    arm_dir = tmp_path / "spec" / "art" / "arm"
    arm_dir.mkdir(parents=True)
    records = [
        {"completion_tokens": 30, "token_events": 10},
        {"completion_tokens": 10, "token_events": 10},
    ]
    with (arm_dir / "requests.jsonl").open("w") as f:
        for record in records:
            record.update(status="completed", ttft_ms=100.0, latency_ms=200.0)
            f.write(json.dumps(record) + "\n")
    summary = {
        "completed_requests": 2,
        "failed_requests": 0,
        "wall_seconds": 10.0,
        "client_queue_ms": {"p50": 0.0},
        "arrival_to_completion_ms": {"p50": 0.0},
    }
    (arm_dir / "summary.json").write_text(json.dumps(summary))
    (arm_dir / "vllm_metrics_after.prom").write_text("")

    assert arm_row(arm_dir)["acc"] == "2.00"

    old = _ttft_arm(tmp_path, "old", [100.0])
    (old / "summary.json").write_text(json.dumps(summary))
    (old / "vllm_metrics_after.prom").write_text("")

    assert arm_row(old)["acc"] == "-"


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


def test_slo_trim_flags_a_window_whose_load_drifts(tmp_path: Path) -> None:
    """Growing prompts between the window halves flag NONSTATIONARY."""
    arms = []
    for name, late_ctx in (("flat", 1000), ("growing", 2000)):
        arm_dir = tmp_path / name / "art" / "arm"
        arm_dir.mkdir(parents=True)
        with (arm_dir / "requests.jsonl").open("w") as f:
            for i in range(20):
                send = 0.5 * i
                record = {
                    "status": "completed",
                    "ttft_ms": 100.0,
                    "latency_ms": 200.0,
                    "completion_tokens": 2,
                    "prompt_tokens": 1000 if send < 5 else late_ctx,
                    "session_id": i % 4,
                    "send_offset_s": send,
                    "completion_offset_s": send + 0.2,
                }
                f.write(json.dumps(record) + "\n")
        (arm_dir / "summary.json").write_text(json.dumps({"wall_seconds": 10.0}))
        arms.append(arm_dir)

    flat, growing = slo_rows(arms, ttft_ms=500.0, tpot_ms=150.0, trim_frac=0.1)[1:]

    assert "drift ctx=+0.00 done=+0.00" in flat and "NONSTATIONARY" not in flat
    assert "drift ctx=+1.00" in growing and "NONSTATIONARY" in growing
    # Every session spans 16 × 0.5 s of sends plus 0.2 s → 8.2 s lifetimes.
    assert "life50_s=8" in flat


def test_slo_population_window_starts_after_one_lifetime(tmp_path: Path) -> None:
    """The window runs from the K-th session start plus a lifetime to the last."""
    arm_dir = tmp_path / "pop" / "art" / "arm"
    arm_dir.mkdir(parents=True)
    with (arm_dir / "requests.jsonl").open("w") as f:
        # Six sessions of lifetime 4 s starting at 0, 1, 4, 5, 8, 9 s; with a
        # population of 2 the window is 1 + 4 = 5 s to the last start, 9 s.
        for sid, start in enumerate((0.0, 1.0, 4.0, 5.0, 8.0, 9.0)):
            for turn in range(2):
                record = {
                    "status": "completed",
                    "ttft_ms": 100.0,
                    "latency_ms": 200.0,
                    "completion_tokens": 2,
                    "prompt_tokens": 1000,
                    "session_id": sid,
                    "send_offset_s": start + 2.0 * turn,
                    "completion_offset_s": start + 2.0 * turn + 2.0,
                }
                f.write(json.dumps(record) + "\n")
    (arm_dir / "summary.json").write_text(json.dumps({"wall_seconds": 13.0}))

    header, row = slo_rows([arm_dir], ttft_ms=500.0, tpot_ms=150.0, population=2)

    assert "population 2 window" in header
    assert "life50_s=4 window_s=5-9 " in row
    # Sends in [5, 9): 5.0, 6.0, 7.0, 8.0 → 4 requests over 4 s.
    assert "ok=4" in row and "goodput=1.000/s" in row


def test_replay_concurrency_weights_in_flight_requests_by_time() -> None:
    """Overlapping spans count together; back-to-back spans do not."""
    records = [
        {"send_offset_s": 0.0, "completion_offset_s": 2.0, "input_length": 100},
        {"send_offset_s": 1.0, "completion_offset_s": 3.0, "input_length": 300},
        {"send_offset_s": 3.0, "completion_offset_s": 4.0, "input_length": 50},
        {"send_offset_s": None, "completion_offset_s": None, "input_length": 9},
    ]

    mean, peak, tokens = replay_concurrency(records)

    # In flight: 1 over [0,1), 2 over [1,2), 1 over [2,4) -> 5 / 4.
    assert peak == 2
    assert mean == 1.25
    # Tokens: 100 + 400 + 300 + 50 over 4 s.
    assert tokens == 212.5


def test_replay_concurrency_serial_replay_never_exceeds_one() -> None:
    """A completion and the next send at the same instant do not overlap."""
    records = [
        {"send_offset_s": float(i), "completion_offset_s": i + 1.0, "input_length": 10}
        for i in range(3)
    ]

    assert replay_concurrency(records) == (1.0, 1, 10.0)
    assert replay_concurrency([]) == (0.0, 0, 0.0)
