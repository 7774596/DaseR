# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from benchmarks.research.prefix_restore_probe import (
    HEADER_TOKENS,
    Cell,
    PromptSource,
    cell_prompts,
    flush_lengths,
    plan_cells,
    summarise,
)


def test_plan_cells_aligns_prefixes_and_keeps_full_prompts() -> None:
    cells = plan_cells([1000], [0.5, 1.0], ["ssd"], repeats=2, block_tokens=128)
    assert [c.prefix_tokens for c in cells] == [384, 384, 1000, 1000]
    assert {c.repeat for c in cells} == {0, 1}
    with pytest.raises(ValueError):
        plan_cells([1000], [0.05], ["l1"], 1, 128)
    with pytest.raises(ValueError):
        plan_cells([1000], [0.5], ["dram"], 1, 128)


def test_cell_prompts_share_prefix_and_diverge_after_it() -> None:
    source = PromptSource(np.arange(10_000, dtype=np.int32), seed=0)
    prompts = cell_prompts(Cell(512, 256, "l1", 0), source)
    assert len(prompts["write"]) == 256
    assert len(prompts["external"]) == len(prompts["gpu"]) == 512
    assert prompts["external"][:256] == prompts["gpu"][:256] == prompts["write"]
    assert prompts["external"][256:] != prompts["gpu"][256:]
    other = cell_prompts(Cell(512, 256, "l1", 1), source)
    assert other["write"][:HEADER_TOKENS] != prompts["write"][:HEADER_TOKENS]


def test_full_prefix_cell_reuses_the_written_prompt() -> None:
    source = PromptSource(np.arange(10_000, dtype=np.int32), seed=0)
    prompts = cell_prompts(Cell(512, 512, "ssd", 0), source)
    assert prompts["write"] == prompts["external"] == prompts["gpu"]


def test_flush_lengths_cover_the_volume() -> None:
    assert flush_lengths(0, 100) == []
    assert flush_lengths(250, 100) == [100, 100, 50]


def test_summarise_reports_restore_time_and_off_state() -> None:
    def row(name: str, ttft: float, gpu: float, ext: float) -> dict:
        return {
            "total_tokens": 512,
            "prefix_tokens": 256,
            "tier": "ssd",
            "pass": name,
            "error": None,
            "ttft_ms": ttft,
            "itl_ms": [10.0, 12.0],
            "gpu_hit_tokens": gpu,
            "external_hit_tokens": ext,
        }

    out = summarise(
        [
            row("external", 300, 0, 256),
            row("external", 500, 256, 0),
            row("gpu", 100, 256, 0),
        ]
    )
    assert len(out) == 1
    assert out[0]["external_ttft_ms_p50"] == 400
    assert out[0]["restore_ms_p50"] == 300
    assert out[0]["external_off_state"] == 1 and out[0]["gpu_off_state"] == 0
    assert out[0]["gpu_itl_ms_p99"] == 12.0


def test_summarise_counts_full_prefix_output_mismatches() -> None:
    def row(name: str, repeat: int, text: str) -> dict:
        return {
            "total_tokens": 256,
            "prefix_tokens": 256,
            "tier": "l1",
            "repeat": repeat,
            "pass": name,
            "error": None,
            "ttft_ms": 1.0,
            "itl_ms": [],
            "text": text,
            "gpu_hit_tokens": 0,
            "external_hit_tokens": 0,
        }

    out = summarise(
        [row("external", 0, "a"), row("gpu", 0, "a")]
        + [row("external", 1, "a"), row("gpu", 1, "b")]
    )
    assert out[0]["output_mismatch"] == 1
