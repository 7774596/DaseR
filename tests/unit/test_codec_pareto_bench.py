# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

pytest.importorskip("daser.ops.tilelang_compressed_kv")
from benchmarks.research.codec_pareto_bench import (  # noqa: E402
    DECODE_PRESETS,
    byte_shuffle,
    byte_unshuffle,
    pareto_front,
    preset_env,
)
from daser.config import LOAD_TILE_GROUPS_ENV, LOAD_WORD_READS_ENV  # noqa: E402


def test_byte_shuffle_splits_low_and_high_bytes_and_inverts() -> None:
    data = torch.arange(10, dtype=torch.uint8)
    assert byte_shuffle(data).tolist() == [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    assert torch.equal(byte_unshuffle(byte_shuffle(data)), data)


def test_pareto_front_keeps_only_non_dominated_arms() -> None:
    rows = [
        {"arm": "memcpy", "ratio": 1.0, "dec_gbs": 900.0},
        {"arm": "fast", "ratio": 0.70, "dec_gbs": 600.0},
        {"arm": "dominated", "ratio": 0.72, "dec_gbs": 500.0},
        {"arm": "dense", "ratio": 0.67, "dec_gbs": 100.0},
        {"arm": "encode_only", "ratio": 0.60, "enc_gbs": 50.0},
        {"arm": "failed", "error": "unsupported"},
    ]
    assert pareto_front(rows) == ["memcpy", "fast", "dense"]


def test_preset_env_clears_other_switches_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LOAD_TILE_GROUPS_ENV, "9")
    monkeypatch.delenv(LOAD_WORD_READS_ENV, raising=False)
    with preset_env(DECODE_PRESETS["uc_vec"]):
        assert os.environ[LOAD_WORD_READS_ENV] == "1"
        assert LOAD_TILE_GROUPS_ENV not in os.environ
    assert os.environ[LOAD_TILE_GROUPS_ENV] == "9"
    assert LOAD_WORD_READS_ENV not in os.environ
