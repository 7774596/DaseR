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
from pathlib import Path
import re

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
    "l1_hits",
    "load_gib",
    "store_gib",
)


_LOAD = 'daser_transfer_bytes_total{op="load"}'
_STORE = 'daser_transfer_bytes_total{op="store"}'
_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([-+eE0-9.]+)")


def _counters(path: Path) -> dict[str, float]:
    """Parse a Prometheus snapshot keyed by ``name{labels}`` (empty if missing)."""
    if not path.is_file():
        return {}
    samples: dict[str, float] = {}
    for line in path.read_text().splitlines():
        match = _SAMPLE.match(line)
        if match is not None:
            samples[match.group(1)] = float(match.group(2))
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


def _rate(part: float, total: float) -> str:
    """Format ``part / total`` or ``-`` when vLLM metrics were not captured."""
    return f"{part / total:.3f}" if total else "-"


def _p(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of ``values`` (0 for an empty list)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))]


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

    def hist(name: str) -> float:
        return _histogram_sum(vllm_before, vllm_after, f"vllm:{name}_seconds")

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
        "l1_hits": f"{_delta(before, after, 'daser_l1_hits_total'):.0f}",
        "load_gib": f"{_delta(before, after, _LOAD) / gib:.1f}",
        "store_gib": f"{_delta(before, after, _STORE) / gib:.1f}",
        "l2w_gib": f"{_delta(before, after, 'daser_l2_write_bytes_total') / gib:.1f}",
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
    parser.add_argument("--slo-ms", type=float, default=2000.0)
    parser.add_argument(
        "--warm-ms",
        type=float,
        default=0.0,
        help="with --first, keep requests whose reference TTFT is below this",
    )
    args = parser.parse_args()
    arm_dirs = [
        summary.parent
        for artifact in args.artifacts
        for summary in sorted(artifact.glob("*/summary.json"))
    ]
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
    if args.first > 0:
        print("\n".join(paired_rows(arm_dirs, args.first, args.warm_ms)))
    if args.stages is not None:
        print("\n".join(stage_rows(arm_dirs, args.stages, args.slo_ms)))


if __name__ == "__main__":
    main()
