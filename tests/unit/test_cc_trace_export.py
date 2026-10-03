# SPDX-License-Identifier: Apache-2.0
"""Tests for the cc-traces to Mooncake replay exporter."""

import json
from pathlib import Path

from benchmarks.research.cc_trace_export import export


def _session(requests: list[tuple[float, int, int]]) -> dict:
    """Build a cc-traces session whose requests extend one chained prefix."""
    rows = []
    for t, tokens, out in requests:
        rows.append(
            {
                "t": t,
                "model": "m",
                "in": tokens,
                "out": out,
                "hash_ids": list(range(tokens // 64)),
            }
        )
    return {"id": "s", "block_size": 64, "requests": rows}


def test_export_shares_blocks_within_a_session_only(tmp_path: Path) -> None:
    """Extending prefixes reuse block IDs; other sessions never alias them."""
    path = tmp_path / "traces.jsonl"
    session = _session([(0.0, 1024, 10), (1.0, 1536, 10)])
    path.write_text(json.dumps(session) + "\n" + json.dumps(session) + "\n")

    rows = export(str(path), 2, 1.0, 262144, 0, seed=0, step_ms=10, max_output=4)

    assert [row["timestamp"] for row in rows] == [0, 10, 20, 30]
    by_session: dict[int, list[list[int]]] = {}
    for row in rows:
        assert len(row["hash_ids"]) == -(-row["input_length"] // 512)
        assert row["output_length"] == 4
        by_session.setdefault(row["session_id"], []).append(row["hash_ids"])
    first, second = by_session.values()
    assert first[1][:2] == first[0]
    assert not set(sum(first, [])) & set(sum(second, []))


def test_export_drops_requests_over_max_ctx(tmp_path: Path) -> None:
    """Requests whose input plus output exceed ``max_ctx`` are not replayed."""
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(_session([(0.0, 1024, 10), (1.0, 4096, 10)])) + "\n")

    rows = export(str(path), 1, 1.0, 2048, 0, seed=0, step_ms=1)

    assert [row["input_length"] for row in rows] == [1024]
