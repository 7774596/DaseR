# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

pytest.importorskip("daser.ops.tilelang_compressed_kv")
from benchmarks.research.codec_pareto_bench import (  # noqa: E402
    byte_shuffle,
    byte_unshuffle,
    call_groups,
    pareto_front,
    scattered_block_ids,
)


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


def test_scattered_block_ids_are_distinct_seeded_and_in_pool() -> None:
    ids = scattered_block_ids(8, 16, seed=3)
    assert len(set(ids)) == 8 and all(0 <= b < 16 for b in ids)
    assert ids == scattered_block_ids(8, 16, seed=3)
    assert ids != list(range(8))
    with pytest.raises(ValueError):
        scattered_block_ids(17, 16, seed=0)


def test_call_groups_cover_every_block_once_in_order() -> None:
    groups = call_groups(10, 4)
    assert [list(range(10))[g] for g in groups] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert call_groups(3, 16) == [slice(0, 3)]
    with pytest.raises(ValueError):
        call_groups(3, 0)
