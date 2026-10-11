# SPDX-License-Identifier: Apache-2.0

"""Print one compact row per arm from ``agent_trace_arms`` artifact directories.

Columns: request counts, wall time, TTFT/queue/latency percentiles, total,
local, and external prefix-hit rates from ``vllm:prompt_tokens_by_source``,
preemptions, summed vLLM queue/prefill/TTFT seconds, and DaseR L1 hits plus
load/store bytes (deltas between the before/after Prometheus snapshots).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

_COLUMNS = (
    "arm",
    "ok",
    "fail",
    "abrt",
    "wall_s",
    "ttft50",
    "ttft90",
    "ttft99",
    "q50",
    "e2e50",
    "hit",
    "loc_hit",
    "ext_hit",
    "preempt",
    "sum_q_s",
    "sum_pf_s",
    "sum_ttft_s",
    "csum_ttft_s",
    "csum_lat_s",
    "io_share",
    "l1_hits",
    "load_gib",
    "store_gib",
    "ssd_r_gib",
    "ra_wait_s",
    "acc",
)


_LOAD = 'daser_transfer_bytes_total{op="load"}'
_STORE = 'daser_transfer_bytes_total{op="store"}'
_SSD_READ = 'daser_readahead_bytes_total{outcome="read"}'
_RA_WAIT = 'daser_load_phase_seconds_total{phase="readahead_wait"}'
_VLLM_TICK = re.compile(
    r"prompt throughput: ([0-9.]+).*?generation throughput: ([0-9.]+)"
    r".*?Running: (\d+) reqs, Waiting: (\d+) reqs(?:, Deferred: (\d+) reqs)?"
    r".*?usage: ([0-9.]+)%"
)
_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([-+eE0-9.]+)")


def _counters(path: Path) -> dict[str, float]:
    """Parse a Prometheus snapshot keyed by ``name{labels}`` (empty if missing).

    A snapshot of several vLLM instances repeats each sample once per
    instance; repeated keys are summed so counters cover every instance.
    """
    if not path.is_file():
        return {}
    samples: dict[str, float] = {}
    for line in path.read_text().splitlines():
        match = _SAMPLE.match(line)
        if match is not None:
            key = match.group(1)
            samples[key] = samples.get(key, 0.0) + float(match.group(2))
    return samples


def _delta(before: dict[str, float], after: dict[str, float], key: str) -> float:
    """Return the counter increase for ``key`` (0 when absent)."""
    return after.get(key, 0.0) - before.get(key, 0.0)


def _source_tokens(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    """Return ``vllm:prompt_tokens_by_source`` deltas keyed by source label.

    vLLM records these once per request at its first prefill, so unlike the
    prefix-cache query counters they are not inflated by preemption.
    """
    prefix = "vllm:prompt_tokens_by_source_total{"
    sources: dict[str, float] = {}
    for key, value in after.items():
        match = re.search(r'source="([^"]+)"', key) if key.startswith(prefix) else None
        if match is not None:
            sources[match.group(1)] = (
                sources.get(match.group(1), 0.0) + value - before.get(key, 0.0)
            )
    return sources


def _histogram_sum(
    before: dict[str, float], after: dict[str, float], name: str
) -> float:
    """Return the ``<name>_sum`` increase summed over all label sets."""
    prefix = f"{name}_sum"
    return sum(
        value - before.get(key, 0.0)
        for key, value in after.items()
        if key == prefix or key.startswith(prefix + "{")
    )


def _bucket_quantile(
    before: dict[str, float],
    after: dict[str, float],
    name: str,
    q: float,
    labels: str = "",
) -> float:
    """Return the smallest ``<name>`` bucket bound holding quantile ``q``.

    ``labels`` selects one label set, rendered as it precedes ``le`` (for
    example ``'op="load",'``). Returns ``inf`` when the quantile falls past
    the largest finite bucket and 0 when the histogram has no observations.
    """
    prefix = f'{name}_bucket{{{labels}le="'
    buckets = sorted(
        (float(key[len(prefix) : -2]), value - before.get(key, 0.0))
        for key, value in after.items()
        if key.startswith(prefix)
    )
    total = buckets[-1][1] if buckets else 0.0
    if total <= 0:
        return 0.0
    return next(bound for bound, count in buckets if count >= q * total)


def _ssd_contention(before: dict[str, float], after: dict[str, float]) -> str:
    """Format L2 read latency and bandwidth by SSD read concurrency.

    Args:
        before: server counters scraped before the run.
        after: server counters scraped after the run.

    Returns:
        `` ssd[<inflight><w if a write was in flight>=jobs/mean ms/GB/s ...]``
        over the nonempty groups, or ``""`` when no L2 read was recorded.
    """
    groups = []
    for bucket in (1, 2, 4, 8, 16, 32):
        for write in ("false", "true"):
            labels = f'{{inflight="{bucket}",write="{write}"}}'
            jobs = _delta(before, after, f"daser_l2_read_jobs_total{labels}")
            if jobs <= 0:
                continue
            io_s = _delta(before, after, f"daser_l2_read_io_seconds_total{labels}")
            nbytes = _delta(before, after, f"daser_l2_read_bytes_total{labels}")
            groups.append(
                f"{bucket}{'w' if write == 'true' else ''}={jobs:.0f}/"
                f"{io_s / jobs * 1000:.1f}ms/{nbytes / io_s / 1e9 if io_s else 0:.2f}"
            )
    return " ssd[" + " ".join(groups) + "]" if groups else ""


def _rate(part: float, total: float) -> str:
    """Format ``part / total`` or ``-`` when vLLM metrics were not captured."""
    return f"{part / total:.3f}" if total else "-"


def _p(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of ``values`` (0 for an empty list)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))]


def _send_key(record: dict) -> tuple[float, int]:
    """Sort key: send offset, trace index for ties."""
    return float(record["send_offset_s"]), int(record["trace_index"])


def arm_row(arm_dir: Path) -> dict[str, str]:
    """Build one table row for an arm artifact directory.

    Args:
        arm_dir: ``<artifact>/<arm>`` directory containing ``summary.json``.

    Returns:
        Mapping from column name to formatted cell text.
    """
    summary = json.loads((arm_dir / "summary.json").read_text())
    completed = [
        json.loads(line)
        for line in (arm_dir / "requests.jsonl").read_text().splitlines()
        if '"status": "completed"' in line
    ]
    ttft = [float(record["ttft_ms"]) for record in completed]
    before = _counters(arm_dir / "daser_metrics_before.prom")
    after = _counters(arm_dir / "daser_metrics_after.prom")
    vllm_before = _counters(arm_dir / "vllm_metrics_before.prom")
    vllm_after = _counters(arm_dir / "vllm_metrics_after.prom")
    sources = _source_tokens(vllm_before, vllm_after)
    total = sum(sources.values())
    preempt = sum(
        vllm_after[key] - vllm_before.get(key, 0.0)
        for key in vllm_after
        if key.startswith("vllm:num_preemptions_total")
    )
    gib = 2**30
    steps = sum(int(r.get("token_events", 0)) for r in completed)
    generated = sum(
        int(r.get("completion_tokens", 0)) for r in completed if r.get("token_events")
    )

    def hist(name: str) -> float:
        return _histogram_sum(vllm_before, vllm_after, f"vllm:{name}_seconds")

    def daser(cell: str) -> str:
        # Arms without a DaseR server (native, LMCache) have no DaseR metrics;
        # print "-" so a missing counter does not read as a measured zero.
        return cell if after else "-"

    return {
        "arm": f"{arm_dir.parent.name}/{arm_dir.name}",
        "ok": str(summary["completed_requests"]),
        "fail": str(summary["failed_requests"]),
        "abrt": "y" if summary.get("aborted") else "n",
        "wall_s": f"{summary['wall_seconds']:.0f}",
        "ttft50": f"{_p(ttft, 50):.0f}",
        "ttft90": f"{_p(ttft, 90):.0f}",
        "ttft99": f"{_p(ttft, 99):.0f}",
        "q50": f"{summary['client_queue_ms']['p50']:.0f}",
        "e2e50": f"{summary['arrival_to_completion_ms']['p50']:.0f}",
        "hit": _rate(
            sources.get("local_cache_hit", 0.0)
            + sources.get("external_kv_transfer", 0.0),
            total,
        ),
        "loc_hit": _rate(sources.get("local_cache_hit", 0.0), total),
        "ext_hit": _rate(sources.get("external_kv_transfer", 0.0), total),
        "preempt": f"{preempt:.0f}" if vllm_after else "-",
        "sum_q_s": f"{hist('request_queue_time'):.0f}",
        "sum_pf_s": f"{hist('request_prefill_time'):.0f}",
        "sum_ttft_s": f"{hist('time_to_first_token'):.0f}",
        # Client-measured TTFT also covers time spent before vLLM receives the
        # request (for example a gateway admission wait), which vLLM's
        # time_to_first_token histogram cannot see.
        "csum_ttft_s": f"{sum(ttft) / 1000:.0f}",
        # Client send-to-completion latency: a scheduler that trades decode
        # progress for earlier first tokens shows up here, not in TTFT.
        "csum_lat_s": f"{sum(float(r['latency_ms']) for r in completed) / 1000:.0f}",
        # Summed external load latency over summed TTFT: the share of TTFT a
        # free storage tier could remove at most (loads run serially here).
        "io_share": daser(
            _rate(
                _delta(before, after, 'daser_transfer_duration_seconds_sum{op="load"}'),
                hist("time_to_first_token"),
            )
        ),
        "l1_hits": daser(f"{_delta(before, after, 'daser_l1_hits_total'):.0f}"),
        "load_gib": daser(f"{_delta(before, after, _LOAD) / gib:.1f}"),
        "store_gib": daser(f"{_delta(before, after, _STORE) / gib:.1f}"),
        "l2w_gib": daser(
            f"{_delta(before, after, 'daser_l2_write_bytes_total') / gib:.1f}"
        ),
        # Bytes readahead pulled from the SSD tier into L1, and the time loads
        # spent waiting on those reads: the SSD share of the load path.
        "ssd_r_gib": daser(f"{_delta(before, after, _SSD_READ) / gib:.1f}"),
        "ra_wait_s": daser(f"{_delta(before, after, _RA_WAIT):.1f}"),
        # Mean tokens per engine step over completed requests: the speculative
        # acceptance length (1.0 without speculation, "-" for old artifacts).
        "acc": f"{generated / steps:.2f}" if steps else "-",
    }


def publish_rows(arm_dirs: list[Path]) -> list[str]:
    """Summarize DaseR store-publication timing per arm.

    Args:
        arm_dirs: Arm artifact directories.

    Returns:
        One line per arm that recorded ``daser_cache_publish_seconds``: lookup
        hit/miss counts, publish count/mean/p50/p90, and (with the publish
        probe on) the count/mean/p50/p90 of how early a lookup arrived before
        the chunk it needed was published. Percentiles are bucket upper bounds.
    """
    lines = []
    for arm_dir in arm_dirs:
        before = _counters(arm_dir / "daser_metrics_before.prom")
        after = _counters(arm_dir / "daser_metrics_after.prom")
        cells = [f"{arm_dir.parent.name}/{arm_dir.name}:"]
        hit = _delta(before, after, 'daser_cache_lookup_total{result="hit"}')
        miss = _delta(before, after, 'daser_cache_lookup_total{result="miss"}')
        cells.append(f"lookup={hit:.0f}/{miss:.0f}")
        for label, name in (
            ("publish", "daser_cache_publish_seconds"),
            ("early", "daser_cache_pending_overlap_seconds"),
        ):
            count = _delta(before, after, f"{name}_count")
            if count <= 0:
                continue
            mean = _delta(before, after, f"{name}_sum") / count
            p50 = _bucket_quantile(before, after, name, 0.5)
            p90 = _bucket_quantile(before, after, name, 0.9)
            cells.append(
                f"{label} n={count:.0f} mean={mean * 1000:.0f}ms "
                f"p50<={p50 * 1000:.0f}ms p90<={p90 * 1000:.0f}ms"
            )
        if len(cells) > 2:
            lines.append("  " + " ".join(cells))
    return lines


def load_rows(arm_dirs: list[Path]) -> list[str]:
    """Summarize DaseR load transfers per arm.

    Args:
        arm_dirs: Arm artifact directories.

    Returns:
        One line per arm that loaded anything: load operation count, mean and
        bucket p90 latency, and effective throughput (bytes over summed
        latency, so concurrent loads count once each), plus the summed
        grouped-load phase times when the server exported them.
    """
    lines = []
    name = "daser_transfer_duration_seconds"
    for arm_dir in arm_dirs:
        before = _counters(arm_dir / "daser_metrics_before.prom")
        after = _counters(arm_dir / "daser_metrics_after.prom")
        count = _delta(before, after, f'{name}_count{{op="load"}}')
        if count <= 0:
            continue
        seconds = _delta(before, after, f'{name}_sum{{op="load"}}')
        nbytes = _delta(before, after, _LOAD)
        p90 = _bucket_quantile(before, after, name, 0.9, labels='op="load",')
        phases = " ".join(
            f"{phase}={value:.1f}s"
            for phase in (
                "readahead_wait",
                "resolve",
                "pending_write",
                "l2",
                "copy_drain",
                "sync",
            )
            if (
                value := _delta(
                    before,
                    after,
                    f'daser_load_phase_seconds_total{{phase="{phase}"}}',
                )
            )
            > 0
        )
        copies, runs, copied = (
            _delta(before, after, f'daser_dst_copy_total{{unit="{unit}"}}')
            for unit in ("copies", "runs", "bytes")
        )
        h2d = (
            f" h2d[copies={copies:.0f} runs={runs:.0f} "
            f"GB={copied / 1e9:.1f} KiB/run={copied / runs / 1024:.0f}]"
            if runs > 0
            else ""
        )
        evicted = {
            kind: _delta(
                before, after, f'daser_l1_evicted_bytes_total{{kind="{kind}"}}'
            )
            for kind in ("all", "pinned", "unread")
        }
        h2d += (
            f" l1_evict[GB={evicted['all'] / 1e9:.1f} "
            f"pinned={evicted['pinned'] / 1e9:.1f} "
            f"unread={evicted['unread'] / 1e9:.1f}]"
            if evicted["all"] > 0
            else ""
        )
        read, contended = (
            _delta(before, after, f'daser_readahead_bytes_total{{outcome="{kind}"}}')
            for kind in ("read", "contended")
        )
        write_contended = _delta(
            before, after, 'daser_readahead_bytes_total{outcome="write_contended"}'
        )
        h2d += (
            f" ra[read_GB={read / 1e9:.1f} contended={contended / read:.2f} "
            f"write_during_GB={write_contended / 1e9:.1f}]"
            if read > 0
            else ""
        )
        waits = {
            cause: _delta(
                before, after, f'daser_promotion_wait_seconds_total{{cause="{cause}"}}'
            )
            for cause in ("l2_write", "promotion", "pool")
        }
        h2d += (
            " promo_wait[" + " ".join(f"{k}={v:.1f}s" for k, v in waits.items()) + "]"
            if any(waits.values())
            else ""
        )
        session = {
            outcome: _delta(
                before,
                after,
                f'daser_session_prefetch_total{{outcome="{outcome}"}}',
            )
            for outcome in ("started", "skipped", "failed", "returned")
        }
        h2d += (
            " sess_pf[" + " ".join(f"{k}={v:.0f}" for k, v in session.items()) + "]"
            if any(session.values())
            else ""
        )
        h2d += _ssd_contention(before, after)
        lines.append(
            f"  {arm_dir.parent.name}/{arm_dir.name}: n={count:.0f} "
            f"mean={seconds / count * 1000:.0f}ms p90<={p90 * 1000:.0f}ms "
            f"per_op_GBps={nbytes / seconds / 1e9 if seconds else 0:.2f}"
            + (f" phases[{phases}]" if phases else "")
            + h2d
        )
    return lines


_LOAD_TIMING = re.compile(
    r"load timing req=\S+ wall_ms=([0-9.]+) batches=\d+ bytes=(\d+)"
)


def load_wall_rows(arm_dirs: list[Path]) -> list[str]:
    """Summarize per-request worker load wall time per arm.

    Args:
        arm_dirs: Arm artifact directories.

    Returns:
        One line per arm whose ``vllm_daser.log`` has DEBUG ``load timing``
        lines: request count, summed and median wall time, and the wall-time
        throughput of requests above and below 15 GB loaded.
    """
    lines = []
    for arm_dir in arm_dirs:
        log = arm_dir / "vllm_daser.log"
        if not log.exists():
            continue
        rows = [
            (float(m.group(1)), int(m.group(2)))
            for m in _LOAD_TIMING.finditer(log.read_text(errors="replace"))
        ]
        if not rows:
            continue
        walls = [wall for wall, _nbytes in rows]
        parts = []
        for label, keep in (
            ("big", lambda n: n > 15e9),
            ("small", lambda n: n <= 15e9),
        ):
            picked = [(wall, nbytes) for wall, nbytes in rows if keep(nbytes)]
            if picked:
                seconds = sum(wall for wall, _ in picked) / 1000
                gbps = sum(nbytes for _, nbytes in picked) / seconds / 1e9
                p50 = _p([wall for wall, _ in picked], 50)
                parts.append(f"{label} n={len(picked)} p50={p50:.0f}ms {gbps:.1f}GB/s")
        lines.append(
            f"  {arm_dir.parent.name}/{arm_dir.name}: n={len(rows)} "
            f"sum_s={sum(walls) / 1000:.1f} p50={_p(walls, 50):.0f}ms "
            + " ".join(parts)
        )
    return lines


def stage_rows(arm_dirs: list[Path], stages_path: Path, slo_ms: float) -> list[str]:
    """Report TTFT and goodput per workload stage.

    Args:
        arm_dirs: Arm artifact directories.
        stages_path: ``stages.json`` from ``longdoc_trace trace``: a list of
            ``{"name", "start", "end", "qps"}`` row ranges (end exclusive).
        slo_ms: TTFT SLO; goodput counts completed requests at or under it.

    Returns:
        One line per (arm, stage): requests sent, completed, TTFT p50/p90/p99,
        SLO attainment, and goodput in requests per second of stage time
        (first to last scheduled arrival of the stage's completed requests).
    """
    stages = json.loads(stages_path.read_text())
    lines = [f"stages (TTFT ms, SLO {slo_ms:.0f} ms)"]
    for arm_dir in arm_dirs:
        records = [json.loads(line) for line in (arm_dir / "requests.jsonl").open()]
        for stage in stages:
            chosen = [
                r
                for r in records
                if stage["start"] <= int(r["trace_index"]) < stage["end"]
            ]
            done = [r for r in chosen if r["status"] == "completed"]
            ttft = [float(r["ttft_ms"]) for r in done]
            good = sum(value <= slo_ms for value in ttft)
            arrivals = [float(r["scheduled_offset_s"]) for r in done]
            span = max(arrivals) - min(arrivals) if len(arrivals) > 1 else 0.0
            lines.append(
                f"  {arm_dir.parent.name}/{arm_dir.name} {stage['name']}: "
                f"n={len(chosen)} ok={len(done)} p50={_p(ttft, 50):.0f} "
                f"p90={_p(ttft, 90):.0f} p99={_p(ttft, 99):.0f} "
                f"slo={_rate(good, len(done))} "
                f"goodput={good / span if span else 0:.3f}/s"
            )
    return lines


def _active_sessions(records: list[dict], begin: float, end: float) -> float:
    """Mean number of sessions active in ``[begin, end)`` seconds.

    A session is active from its first send to its last completion.
    """
    spans: dict = {}
    for r in records:
        if "send_offset_s" not in r or "completion_offset_s" not in r:
            continue
        lo, hi = spans.get(r.get("session_id"), (math.inf, -math.inf))
        spans[r.get("session_id")] = (
            min(lo, float(r["send_offset_s"])),
            max(hi, float(r["completion_offset_s"])),
        )
    covered = sum(max(0.0, min(hi, end) - max(lo, begin)) for lo, hi in spans.values())
    return covered / (end - begin) if end > begin else 0.0


def _session_lifetime_p50(records: list[dict[str, Any]]) -> float:
    """Median first-send to last-completion span over complete sessions."""
    spans: dict[Any, list[float]] = {}
    for record in records:
        if "send_offset_s" not in record or "completion_offset_s" not in record:
            continue
        span = spans.setdefault(record.get("session_id"), [math.inf, -math.inf])
        span[0] = min(span[0], float(record["send_offset_s"]))
        span[1] = max(span[1], float(record["completion_offset_s"]))
    lifetimes = sorted(end - start for start, end in spans.values())
    return lifetimes[len(lifetimes) // 2] if lifetimes else 0.0


def _session_starts(records: list[dict[str, Any]]) -> list[float]:
    """Sorted first-send offset of every session."""
    starts: dict[Any, float] = {}
    for record in records:
        if "send_offset_s" in record:
            sid = record.get("session_id")
            send = float(record["send_offset_s"])
            starts[sid] = min(starts.get(sid, send), send)
    return sorted(starts.values())


def _window_drift(
    records: list[dict[str, Any]], begin: float, end: float
) -> tuple[float, float]:
    """Relative change of median prompt length and completions, half to half."""
    middle = (begin + end) / 2
    halves = ((begin, middle), (middle, end))
    ctx = []
    done = []
    for low, high in halves:
        sent = sorted(
            int(r.get("prompt_tokens", 0))
            for r in records
            if low <= float(r.get("send_offset_s", -1)) < high
        )
        ctx.append(sent[len(sent) // 2] if sent else 0)
        done.append(
            sum(
                low <= float(r.get("completion_offset_s", -1)) < high
                and r.get("status") == "completed"
                for r in records
            )
        )
    return (
        ctx[1] / ctx[0] - 1 if ctx[0] else 0.0,
        done[1] / done[0] - 1 if done[0] else 0.0,
    )


def slo_rows(
    arm_dirs: list[Path],
    ttft_ms: float,
    tpot_ms: float,
    trim_frac: float = 0.0,
    population: int = 0,
) -> list[str]:
    """Report per-request SLO attainment and goodput per arm.

    A request meets the SLO when its TTFT is at most ``ttft_ms`` and its mean
    time per output token after the first is at most ``tpot_ms``; requests with
    one output token are judged on TTFT alone.

    Args:
        arm_dirs: Arm artifact directories.
        ttft_ms: TTFT SLO in milliseconds.
        tpot_ms: TPOT SLO in milliseconds.
        trim_frac: Drop requests sent in the first and last ``trim_frac`` of
            the wall clock (ramp-up and drain), so the row describes the
            steady state; goodput divides by the kept window. A trimmed row
            also checks that the window is stationary: the median prompt
            length and the completion count of its two halves must differ by
            at most 15%, otherwise the row is flagged NONSTATIONARY and its
            SLO numbers are not a steady-state reading.
        population: For a closed session population of this size, use the
            steady window instead of ``trim_frac``: from the start of the
            ``population``-th session plus one median session lifetime (the
            population is full and its ages have mixed) to the start of the
            last session (after it the population drains). The window gets
            the same stationarity check.

    Returns:
        One line per arm: completed requests, TTFT and TPOT p90, summed TTFT,
        mean active sessions, median session lifetime, SLO attainment over the
        kept sent requests (failures miss it), goodput in SLO-meeting requests
        per second, the TTFT p99 of the longest-prompt fifth of completed
        requests (a fairness check: reordering that helps short requests must
        not starve long ones), and with trimming the half-to-half drift.
    """
    lines = [
        f"SLO goodput (TTFT <= {ttft_ms:.0f} ms, TPOT <= {tpot_ms:.1f} ms, "
        + (
            f"population {population} window)"
            if population
            else f"trim {trim_frac:.2f})"
        )
    ]
    for arm_dir in arm_dirs:
        records = [json.loads(line) for line in (arm_dir / "requests.jsonl").open()]
        wall = float(json.loads((arm_dir / "summary.json").read_text())["wall_seconds"])
        lifetime = _session_lifetime_p50(records)
        if population:
            starts = _session_starts(records)
            begin = starts[min(population, len(starts)) - 1] + lifetime
            end = max(starts[-1], begin)
        else:
            begin, end = trim_frac * wall, (1.0 - trim_frac) * wall
        active = _active_sessions(records, begin, end)
        windowed = bool(trim_frac or population)
        drift = _window_drift(records, begin, end) if windowed else None
        if windowed:
            records = [
                r for r in records if begin <= float(r.get("send_offset_s", 0)) < end
            ]
        wall = end - begin
        done = [r for r in records if r["status"] == "completed"]
        ttft = [float(r["ttft_ms"]) for r in done]
        tpot = [
            (float(r["latency_ms"]) - float(r["ttft_ms"]))
            / max(int(r.get("completion_tokens", 1)) - 1, 1)
            for r in done
        ]
        by_ctx = sorted(done, key=lambda r: int(r.get("prompt_tokens", 0)))
        long_ttft = [float(r["ttft_ms"]) for r in by_ctx[len(by_ctx) * 4 // 5 :]]
        good = sum(
            first <= ttft_ms
            and (int(r.get("completion_tokens", 1)) <= 1 or t <= tpot_ms)
            for r, first, t in zip(done, ttft, tpot, strict=True)
        )
        lines.append(
            f"  {arm_dir.parent.parent.name}/{arm_dir.name}: ok={len(done)} "
            f"ttft_p90={_p(ttft, 90):.0f} tpot_p90={_p(tpot, 90):.1f} "
            f"long20_ttft_p99={_p(long_ttft, 99):.0f} "
            f"sum_ttft_s={sum(ttft) / 1000:.0f} active={active:.1f} "
            f"life50_s={lifetime:.0f} window_s={begin:.0f}-{end:.0f} "
            f"slo={_rate(good, len(records))} "
            f"goodput={good / wall if wall else 0:.3f}/s"
            + (
                ""
                if drift is None
                else f" drift ctx={drift[0]:+.2f} done={drift[1]:+.2f}"
                + (" NONSTATIONARY" if max(map(abs, drift)) > 0.15 else "")
            )
        )
    return lines


def paired_rows(arm_dirs: list[Path], first: int, warm_ms: float = 0.0) -> list[str]:
    """Compare per-request TTFT on the first ``first`` shared trace requests.

    Requests are matched by ``trace_index`` across arms, so runs of different
    lengths over the same trace prefix can be compared on identical requests.

    Args:
        arm_dirs: Arm directories; the first one is the reference.
        first: Number of lowest shared trace indices to compare.
        warm_ms: When positive, keep only requests whose reference TTFT is
            below this bound, so cold full prefills do not mask load changes.

    Returns:
        Formatted lines: one per arm with the request count, summed and
        median TTFT, the ratio of summed TTFT to the reference, and the TTFT
        of the first four requests.
    """
    ttft_by_arm = [
        {
            int(record["trace_index"]): float(record["ttft_ms"])
            for record in map(json.loads, (d / "requests.jsonl").open())
            if record["status"] == "completed"
        }
        for d in arm_dirs
    ]
    shared = sorted(set.intersection(*(set(t) for t in ttft_by_arm)))[:first]
    if warm_ms > 0:
        shared = [i for i in shared if ttft_by_arm[0][i] < warm_ms]
    reference = sum(ttft_by_arm[0][i] for i in shared) or 1.0
    warm = f", reference TTFT < {warm_ms:.0f} ms" if warm_ms > 0 else ""
    lines = [f"paired first {len(shared)} shared requests{warm} (TTFT ms)"]
    for arm_dir, ttft in zip(arm_dirs, ttft_by_arm, strict=True):
        values = [ttft[i] for i in shared]
        head = "/".join(f"{v:.0f}" for v in values[:4])
        lines.append(
            f"  {arm_dir.parent.name}/{arm_dir.name}: n={len(values)} "
            f"sum_s={sum(values) / 1e3:.0f} p50={_p(values, 50):.0f} "
            f"ratio={sum(values) / reference:.2f} first4={head}"
        )
    return lines


def overlap_bound_rows(
    arm_dirs: list[Path], oracle_dir: Path, first: int, warm_ms: float = 0.0
) -> list[str]:
    """Bound the TTFT left if each request's extra IO overlapped its work.

    The oracle arm keeps every stored byte in L1, so its TTFT ``o`` per request
    is the work without SSD waits and ``x = t - o`` is the serial extra wait of
    an arm with TTFT ``t``. Streaming the reads into the forward pass hides at
    most ``min(x, o)``, which leaves ``max(o, x)`` per request.

    Args:
        arm_dirs: Arm directories to bound; the first one selects warm requests.
        oracle_dir: Arm directory of the all-L1 oracle run of the same trace.
        first: Number of lowest shared trace indices to compare.
        warm_ms: When positive, keep only requests whose first-arm TTFT is
            below this bound, as in :func:`paired_rows`.

    Returns:
        Formatted lines: one per arm with its summed TTFT, summed extra wait
        and the summed overlap bound, in seconds.
    """
    dirs = [oracle_dir, *arm_dirs]
    ttft_by_arm = [
        {
            int(record["trace_index"]): float(record["ttft_ms"])
            for record in map(json.loads, (d / "requests.jsonl").open())
            if record["status"] == "completed"
        }
        for d in dirs
    ]
    shared = sorted(set.intersection(*(set(t) for t in ttft_by_arm)))[:first]
    if warm_ms > 0:
        shared = [i for i in shared if ttft_by_arm[1][i] < warm_ms]
    oracle = ttft_by_arm[0]
    lines = [
        f"overlap bound vs {oracle_dir.parent.parent.name} on {len(shared)} "
        f"requests (s): oracle={sum(oracle[i] for i in shared) / 1e3:.0f}"
    ]
    for arm_dir, ttft in zip(arm_dirs, ttft_by_arm[1:], strict=True):
        extra = [max(0.0, ttft[i] - oracle[i]) for i in shared]
        pairs = list(zip(shared, extra, strict=True))
        bound = [min(ttft[i], max(oracle[i], x)) for i, x in pairs]
        hidden = [min(x, oracle[i]) for i, x in pairs]
        lines.append(
            f"  {arm_dir.parent.parent.name}/{arm_dir.name}: "
            f"ttft={sum(ttft[i] for i in shared) / 1e3:.0f} "
            f"extra={sum(extra) / 1e3:.1f} bound={sum(bound) / 1e3:.0f} "
            f"hidden<={sum(hidden) / 1e3:.1f}"
        )
    return lines


def order_rows(arm_dirs: list[Path]) -> list[str]:
    """Check that requests were sent in trace order.

    Sorts sent requests by ``send_offset_s`` and counts positions whose
    ``trace_index`` differs from the trace-order position. Replays that do
    not use strict order are expected to deviate (closed-loop prefix
    dependencies alone send ready requests in wake-up order), so any
    deviation means the trace interleaving was not what was replayed.

    Args:
        arm_dirs: Arm directories.

    Returns:
        One line per arm with the mismatch count, tagged ``ANOMALY`` when
        the send order differs from the trace order.
    """
    lines = []
    for arm_dir in arm_dirs:
        records = [
            record
            for record in map(json.loads, (arm_dir / "requests.jsonl").open())
            if record.get("send_offset_s") is not None
        ]
        sent = [int(r["trace_index"]) for r in sorted(records, key=_send_key)]
        mismatches = sum(a != b for a, b in zip(sent, sorted(sent), strict=True))
        tag = " ANOMALY: send order differs from trace order" if mismatches else ""
        lines.append(
            f"  {arm_dir.parent.name}/{arm_dir.name}: "
            f"out_of_order={mismatches}/{len(sent)}{tag}"
        )
    return lines


def replay_concurrency(records: list[dict]) -> tuple[float, int, float]:
    """Measure how many requests were actually in flight during a replay.

    A request is in flight from its send to its completion. Nominal
    ``--max-inflight`` overstates concurrency when session dependencies
    serialize the trace, so this is read from the recorded offsets.

    Args:
        records: ``requests.jsonl`` records.

    Returns:
        ``(time-weighted mean in-flight requests, max in flight,
        time-weighted mean in-flight prompt tokens)``; zeros when no record
        has both offsets.
    """
    events = []
    for r in records:
        if r.get("send_offset_s") is None or r.get("completion_offset_s") is None:
            continue
        tokens = int(r.get("input_length") or r.get("prompt_tokens") or 0)
        events.append((float(r["send_offset_s"]), 1, tokens))
        events.append((float(r["completion_offset_s"]), -1, -tokens))
    if not events:
        return 0.0, 0, 0.0
    # Completions sort before sends at the same instant.
    events.sort(key=lambda e: (e[0], e[1]))
    inflight = peak = tokens = 0
    area = token_area = 0.0
    last = events[0][0]
    for t, step, delta in events:
        area += inflight * (t - last)
        token_area += tokens * (t - last)
        last = t
        inflight += step
        tokens += delta
        peak = max(peak, inflight)
    span = events[-1][0] - events[0][0]
    if span <= 0:
        return float(peak), peak, float(tokens)
    return area / span, peak, token_area / span


def concurrency_rows(arm_dirs: list[Path]) -> list[str]:
    """Report measured in-flight requests and prompt tokens per arm.

    Args:
        arm_dirs: Arm directories with ``requests.jsonl``.

    Returns:
        One line per arm.
    """
    lines = []
    for arm_dir in arm_dirs:
        records = [json.loads(line) for line in (arm_dir / "requests.jsonl").open()]
        mean, peak, tokens = replay_concurrency(records)
        lines.append(
            f"  {arm_dir.parent.name}/{arm_dir.name}: inflight_mean={mean:.2f} "
            f"max={peak} inflight_ctx_mean={tokens / 1e3:.0f}k"
        )
    return lines


def lmcache_rows(arm_dirs: list[Path]) -> list[str]:
    """Report the LMCache server configuration that was actually in effect.

    Reads ``lmcache_status_after.json`` (written by LMCache arms) so a
    baseline can be checked against the intended setup: chunk size equal to
    the vLLM block size, the configured L1 capacity, the L1 eviction
    watermark, and an attached L2 adapter.

    Args:
        arm_dirs: Arm directories; arms without the status file are skipped.

    Returns:
        One line per LMCache arm, tagged ``ANOMALY`` when the chunk size
        differs from the engine's tokens per block, no L2 adapter is attached,
        or the server reported itself unhealthy.
    """
    lines = []
    for arm_dir in arm_dirs:
        path = arm_dir / "lmcache_status_after.json"
        if not path.is_file():
            continue
        status = json.loads(path.read_text())
        storage = status.get("storage_manager", {})
        l1 = storage.get("l1_manager", {})
        eviction = storage.get("l1_eviction_controller", {})
        adapters = storage.get("l2_adapters", [])
        blocks = {
            group.get("tokens_per_block")
            for meta in status.get("cache_context_meta", {}).values()
            for group in meta.get("kv_cache_layout", {}).get("kernel_groups", [])
        }
        chunk = status.get("chunk_size")
        problems = []
        if blocks and blocks != {chunk}:
            problems.append(f"chunk {chunk} != block {sorted(blocks)}")
        if not adapters:
            problems.append("no L2 adapter")
        if not status.get("is_healthy", False):
            problems.append("unhealthy")
        tag = f" ANOMALY: {'; '.join(problems)}" if problems else ""
        l2 = ",".join(
            f"{a.get('type')}(odirect={a.get('use_odirect')})" for a in adapters
        )
        lines.append(
            f"  {arm_dir.parent.name}/{arm_dir.name}: chunk={chunk} "
            f"l1_cfg={l1.get('memory_configured_bytes', 0) / 2**30:.1f}GiB "
            f"l1_used={l1.get('memory_used_bytes', 0) / 2**30:.1f}GiB "
            f"l1_objs={l1.get('total_object_count')} "
            f"wm={eviction.get('trigger_watermark')} l2={l2 or '-'}{tag}"
        )
    return lines


def live_clock_rows(arm_dirs: list[Path]) -> list[str]:
    """Check that the live-victim arena clock counted one tick per request.

    ``daser_arena_live_tick`` advances once per distinct prompt. Its increase
    over the replay should match the number of sent requests; a clock that
    counts scheduler polls instead ages live sessions into the idle class.

    Args:
        arm_dirs: Arm directories; arms without the gauge are skipped.

    Returns:
        One line per arm with the gauge, tagged ``ANOMALY`` when the tick
        increase is more than 10% away from the sent request count.
    """
    lines = []
    for arm_dir in arm_dirs:
        after = _counters(arm_dir / "daser_metrics_after.prom")
        if "daser_arena_live_tick" not in after:
            continue
        before = _counters(arm_dir / "daser_metrics_before.prom")
        ticks = _delta(before, after, "daser_arena_live_tick")
        sent = sum(
            1
            for record in map(json.loads, (arm_dir / "requests.jsonl").open())
            if record.get("send_offset_s") is not None
        )
        off = abs(ticks - sent) > 0.1 * max(sent, 1)
        tag = " ANOMALY: live clock does not count one tick per request" if off else ""
        lines.append(
            f"  {arm_dir.parent.name}/{arm_dir.name}: "
            f"live_ticks={ticks:.0f} requests={sent}{tag}"
        )
    return lines


def vllm_tick_rows(arm_dirs: list[Path], kv_full_pct: float = 60.0) -> list[str]:
    """Classify vLLM periodic stats ticks to attribute queueing.

    Each ``Running: .. Waiting: ..`` stats line is one tick. Ticks with
    waiting requests are split by why admission stalls: KV usage at or above
    ``kv_full_pct`` (GPU KV capacity), otherwise compute/budget bound. Idle
    ticks have nothing running or waiting.

    Args:
        arm_dirs: Arm directories; arms without a ``vllm_*.log`` are skipped.
        kv_full_pct: KV usage percentage counted as capacity-bound.

    Returns:
        One line per arm: tick counts, waiting ticks split into capacity- and
        compute-bound, waiting ticks with a single running request, and the
        mean prompt throughput over capacity-bound ticks.
    """
    lines = []
    for arm_dir in arm_dirs:
        logs = sorted(arm_dir.glob("vllm_*.log"))
        if not logs:
            continue
        ticks = [
            m.groups() for m in map(_VLLM_TICK.search, logs[0].open()) if m is not None
        ]
        busy = [t for t in ticks if int(t[2]) or int(t[3])]
        waiting = [t for t in busy if int(t[3])]
        full = [t for t in waiting if float(t[5]) >= kv_full_pct]
        single = sum(1 for t in waiting if int(t[2]) <= 1)
        full_pf = sum(float(t[0]) for t in full) / max(len(full), 1)
        lines.append(
            f"  {arm_dir.parent.name}/{arm_dir.name}: ticks={len(ticks)} "
            f"busy={len(busy)} waiting={len(waiting)} kv>={kv_full_pct:.0f}%="
            f"{len(full)} compute={len(waiting) - len(full)} running<=1={single} "
            f"pf_tok_s@kvfull={full_pf:.0f}"
        )
    return lines


def decode_rows(arm_dirs: list[Path], min_tokens: int = 20) -> list[str]:
    """Attribute client latency to decode and relate TPOT to concurrent context.

    Decode time is client latency minus TTFT. For requests with at least
    ``min_tokens`` output tokens, TPOT is regressed on the prompt tokens of
    every request whose decode window overlaps this one (itself included):
    the slope is the decode cost of resident context (memory-bandwidth bound
    attention), the intercept the per-step floor (weights, launch).

    Args:
        arm_dirs: Arm directories with ``requests.jsonl``.
        min_tokens: Shortest output counted in the TPOT regression.

    Returns:
        One line per arm: decode seconds and share of client latency, mean
        TPOT, and the fitted intercept (ms) and slope (ms per 100k tokens).
    """
    lines = []
    for arm_dir in arm_dirs:
        records = [
            record
            for record in map(json.loads, (arm_dir / "requests.jsonl").open())
            if not record.get("error") and record.get("ttft_ms") is not None
        ]
        if not records:
            continue
        spans = []
        for record in records:
            end = float(record["completion_offset_s"])
            first = (
                end - (float(record["latency_ms"]) - float(record["ttft_ms"])) / 1000
            )
            spans.append((first, end, int(record["input_length"]), record))
        decode_s = sum(end - first for first, end, _, _ in spans)
        latency_s = sum(float(r["latency_ms"]) for r in records) / 1000
        out = sum(int(r["completion_tokens"]) for r in records)
        points = []
        for first, end, _, record in spans:
            tokens = int(record["completion_tokens"])
            if tokens < min_tokens:
                continue
            ctx = sum(c for f, e, c, _ in spans if f < end and e > first)
            points.append((ctx / 1e5, (end - first) * 1000 / (tokens - 1)))
        slope = intercept = 0.0
        if len(points) >= 2:
            mx = sum(x for x, _ in points) / len(points)
            my = sum(y for _, y in points) / len(points)
            var = sum((x - mx) ** 2 for x, _ in points)
            slope = sum((x - mx) * (y - my) for x, y in points) / max(var, 1e-9)
            intercept = my - slope * mx
        tpot = decode_s * 1000 / max(out - len(records), 1)
        lines.append(
            f"  {arm_dir.parent.name}/{arm_dir.name}: decode_s={decode_s:.0f} "
            f"share={_rate(decode_s, latency_s)} tpot_ms={tpot:.1f} "
            f"fit n={len(points)} floor_ms={intercept:.1f} "
            f"ms_per_100k_ctx={slope:.2f}"
        )
    return lines


def request_hit_rows(arm_dirs: list[Path], counter_hits: list[str]) -> list[str]:
    """Reconcile per-request cached tokens with the vLLM hit counters.

    vLLM reports ``usage.prompt_tokens_details.cached_tokens`` per request
    when served with ``--enable-prompt-tokens-details``. Their token-weighted
    sum over the replay should equal the counter-based ``hit`` column; a gap
    means per-request hit audits (e.g. against an offline cache simulation)
    would read the wrong requests.

    Args:
        arm_dirs: Arm directories; arms without ``cached_tokens`` are skipped.
        counter_hits: Counter-based hit rate per arm, aligned with
            ``arm_dirs``, as printed in the ``hit`` column.

    Returns:
        One line per arm with per-request data, tagged ``ANOMALY`` when the
        two hit rates differ by more than 0.05.
    """
    lines = []
    for arm_dir, counter in zip(arm_dirs, counter_hits, strict=True):
        records = [
            record
            for record in map(json.loads, (arm_dir / "requests.jsonl").open())
            if record.get("cached_tokens") is not None and not record.get("error")
        ]
        if not records:
            continue
        cached = sum(record["cached_tokens"] for record in records)
        prompt = sum(record["prompt_tokens"] for record in records)
        rate = cached / max(prompt, 1)
        label = f"{arm_dir.parent.name}/{arm_dir.name}"
        off = counter not in ("-", "") and abs(rate - float(counter)) > 0.05
        tag = (
            " ANOMALY: per-request cached tokens disagree with counters" if off else ""
        )
        lines.append(
            f"  {label}: n={len(records)} request_hit={rate:.3f} "
            f"counter_hit={counter}{tag}"
        )
    return lines


def main() -> None:
    """CLI entry point: ``summarize.py <artifact_dir> [<artifact_dir> ...]``.

    ``--first N`` adds a paired TTFT comparison over the first ``N`` shared
    trace requests, using the first arm found as the reference. ``--stages``
    adds per-stage TTFT and goodput under ``--slo-ms``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument(
        "--stages", type=Path, default=None, help="stages.json for per-stage rows"
    )
    parser.add_argument("--slo-ms", type=float, default=2000.0, help="TTFT SLO")
    parser.add_argument(
        "--slo-tpot-ms",
        type=float,
        default=None,
        help="TPOT SLO; adds per-arm SLO goodput rows with --slo-ms as TTFT SLO",
    )
    parser.add_argument(
        "--trim-frac",
        type=float,
        default=0.0,
        help="with --slo-tpot-ms, drop requests sent in the first and last "
        "fraction of the wall clock (ramp-up and drain)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=0,
        help="with --slo-tpot-ms, read SLO rows on the steady window of a "
        "closed session population of this size (overrides --trim-frac)",
    )
    parser.add_argument(
        "--warm-ms",
        type=float,
        default=0.0,
        help="with --first, keep requests whose reference TTFT is below this",
    )
    parser.add_argument(
        "--overlap-oracle",
        type=Path,
        default=None,
        help="with --first, all-L1 oracle artifact dir for the overlap bound",
    )
    args = parser.parse_args()
    arm_dirs = [
        summary.parent
        for artifact in args.artifacts
        for summary in sorted(artifact.glob("*/summary.json"))
    ]
    if not arm_dirs:
        parser.error("no <arm>/summary.json under the given artifact directories")
    rows = [arm_row(arm_dir) for arm_dir in arm_dirs]
    widths = {c: max([len(c)] + [len(r[c]) for r in rows]) for c in _COLUMNS}
    print("  ".join(c.rjust(widths[c]) for c in _COLUMNS))
    for row in rows:
        print("  ".join(row[c].rjust(widths[c]) for c in _COLUMNS))
    publish = publish_rows(arm_dirs)
    if publish:
        print("store publication (DaseR)")
        print("\n".join(publish))
    loads = load_rows(arm_dirs)
    if loads:
        print("load transfers (DaseR)")
        print("\n".join(loads))
    walls = load_wall_rows(arm_dirs)
    if walls:
        print("per-request load wall (DEBUG load timing)")
        print("\n".join(walls))
    print("send order vs trace order")
    print("\n".join(order_rows(arm_dirs)))
    print("measured concurrency (send..completion)")
    print("\n".join(concurrency_rows(arm_dirs)))
    lmcache = lmcache_rows(arm_dirs)
    if lmcache:
        print("LMCache server configuration in effect")
        print("\n".join(lmcache))
    clocks = live_clock_rows(arm_dirs)
    if clocks:
        print("live-victim arena clock")
        print("\n".join(clocks))
    ticks = vllm_tick_rows(arm_dirs)
    if ticks:
        print("vLLM scheduler ticks (waiting = queued in vLLM)")
        print("\n".join(ticks))
    decodes = decode_rows(arm_dirs)
    if decodes:
        print("decode attribution (client latency - TTFT; TPOT vs overlapping ctx)")
        print("\n".join(decodes))
    request_hits = request_hit_rows(arm_dirs, [row["hit"] for row in rows])
    if request_hits:
        print("per-request cached tokens vs hit counters")
        print("\n".join(request_hits))
    if args.first > 0:
        print("\n".join(paired_rows(arm_dirs, args.first, args.warm_ms)))
    if args.first > 0 and args.overlap_oracle is not None:
        oracle_dirs = sorted(args.overlap_oracle.glob("*/summary.json"))
        print(
            "\n".join(
                overlap_bound_rows(
                    arm_dirs, oracle_dirs[0].parent, args.first, args.warm_ms
                )
            )
        )
    if args.stages is not None:
        print("\n".join(stage_rows(arm_dirs, args.stages, args.slo_ms)))
    if args.slo_tpot_ms is not None:
        print(
            "\n".join(
                slo_rows(
                    arm_dirs,
                    args.slo_ms,
                    args.slo_tpot_ms,
                    args.trim_frac,
                    args.population,
                )
            )
        )


if __name__ == "__main__":
    main()
