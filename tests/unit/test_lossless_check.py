# SPDX-License-Identifier: Apache-2.0
"""Tests for the raw-versus-compressed greedy output comparison."""

import json
from pathlib import Path

from benchmarks.research.lossless_check import compare_probes, load_probe


def _row(index: int, name: str, sha: str, external: float = 0.0) -> dict:
    return {
        "index": index,
        "pass": name,
        "input_length": 1000,
        "error": None,
        "text_sha1": sha,
        "external_hit_tokens": external,
    }


def _run(shas: dict[int, str], external: dict[int, float]) -> list[dict]:
    rows = []
    for index, sha in shas.items():
        rows.append(_row(index, "cold", "c"))
        rows.append(_row(index, "gpu", "c"))
        rows.append(_row(index, "external", sha, external.get(index, 900.0)))
    return rows


def test_identical_external_texts_are_lossless() -> None:
    """Matching connector-served texts on every row pass the check."""
    reference = _run({0: "a", 1: "b"}, {})
    result = compare_probes(reference, _run({0: "a", 1: "b"}, {}))
    assert result["lossless"] is True
    assert result["external"]["compared"] == 2
    assert result["external"]["identical"] == 2
    assert result["cold"]["identical"] == 2


def test_divergent_external_text_fails() -> None:
    """One differing connector-served text is reported by index."""
    result = compare_probes(_run({0: "a", 1: "b"}, {}), _run({0: "a", 1: "x"}, {}))
    assert result["lossless"] is False
    assert result["external"]["mismatched"] == [1]


def test_rows_not_served_by_connector_are_invalid_not_identical() -> None:
    """A pass with few external hits in either arm is skipped, not counted."""
    reference = _run({0: "a", 1: "b"}, {})
    candidate = _run({0: "a", 1: "x"}, {1: 100.0})
    result = compare_probes(reference, candidate)
    assert result["external"]["invalid"] == [1]
    assert result["external"]["compared"] == 1
    assert result["lossless"] is True


def test_no_comparable_rows_is_not_lossless() -> None:
    """With nothing served externally the check cannot pass."""
    result = compare_probes(_run({0: "a"}, {0: 0.0}), _run({0: "a"}, {0: 0.0}))
    assert result["external"]["compared"] == 0
    assert result["lossless"] is False


def test_errored_and_unpaired_rows_are_excluded() -> None:
    """Errored passes are invalid; indices missing from one run are unpaired."""
    reference = _run({0: "a", 1: "b"}, {})
    candidate = _run({0: "a"}, {})
    candidate[0]["error"] = "boom"
    result = compare_probes(reference, candidate)
    assert result["cold"] == {
        "paired": 1,
        "compared": 0,
        "identical": 0,
        "mismatched": [],
        "invalid": [0],
    }
    assert result["external"]["paired"] == 1


def test_load_probe_accepts_arm_directory(tmp_path: Path) -> None:
    """An arm directory resolves to its probe.jsonl."""
    rows = _run({0: "a"}, {})
    (tmp_path / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert load_probe(tmp_path) == rows
