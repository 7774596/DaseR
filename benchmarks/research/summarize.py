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
    }


def main() -> None:
    """CLI entry point: ``summarize.py <artifact_dir> [<artifact_dir> ...]``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    rows = [
        arm_row(summary.parent)
        for artifact in args.artifacts
        for summary in sorted(artifact.glob("*/summary.json"))
    ]
    widths = {c: max([len(c)] + [len(r[c]) for r in rows]) for c in _COLUMNS}
    print("  ".join(c.rjust(widths[c]) for c in _COLUMNS))
    for row in rows:
        print("  ".join(row[c].rjust(widths[c]) for c in _COLUMNS))


if __name__ == "__main__":
    main()
