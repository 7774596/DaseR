# SPDX-License-Identifier: Apache-2.0

# Standard
import json
from pathlib import Path

import pytest

# First Party
from daser.config import (
    BLOCK_TOKENS,
    DEFAULT_IOURING_L1_BYTES,
    DEFAULT_TIER_GRAIN,
    MAX_STAGING_BYTES,
    STORAGE_FORMAT_COMPRESSED_ONLINE,
    STORAGE_FORMAT_FRAGPACK,
    DaserConfig,
    model_geometry_from_path,
)


def _write_model_config(path: Path, payload: dict[str, object]) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_model_geometry_from_path_reads_hf_config(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    _write_model_config(
        model_path,
        {
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 28,
            "torch_dtype": "bfloat16",
        },
    )

    geometry = model_geometry_from_path(str(model_path))

    assert geometry.num_kv_heads == 4
    assert geometry.head_dim == 128
    assert geometry.num_layers == 28
    assert geometry.dtype_bytes == 2
    assert (
        geometry.slot_size_for_block_tokens(BLOCK_TOKENS)
        == 4 * 128 * 2 * 28 * BLOCK_TOKENS * 2
    )


def test_model_geometry_uses_explicit_head_dim_and_float32_dtype(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    _write_model_config(
        model_path,
        {
            "head_dim": 64,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_hidden_layers": 2,
            "torch_dtype": "float32",
        },
    )

    geometry = model_geometry_from_path(str(model_path))

    assert geometry.head_dim == 64
    assert geometry.dtype_bytes == 4
    assert (
        geometry.slot_size_for_block_tokens(BLOCK_TOKENS)
        == 8 * 64 * 2 * 2 * BLOCK_TOKENS * 4
    )


def test_daser_config_derives_paths_and_slot_size(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(
        model_path,
        {
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 28,
            "torch_dtype": "float16",
        },
    )
    cfg = DaserConfig(
        model_path=str(model_path),
        store_dir=str(store_dir),
    )

    assert cfg.store_path == str(store_dir / "daser.store")
    assert cfg.index_path == str(store_dir / "daser.index")
    assert cfg.model_id == str(model_path)
    assert cfg.resolved_slot_size() == 4 * 128 * 2 * 28 * BLOCK_TOKENS * 2


def test_daser_config_uses_configured_block_tokens_for_slot_size(
    tmp_path: Path,
) -> None:
    """Resolved slot size follows the configured vLLM block size."""
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(
        model_path,
        {
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 28,
            "torch_dtype": "float16",
        },
    )
    cfg = DaserConfig(
        model_path=str(model_path),
        store_dir=str(store_dir),
        block_tokens=128,
    )

    assert cfg.resolved_slot_size() == 4 * 128 * 2 * 28 * 128 * 2
    assert cfg.runtime_config()["block_tokens"] == 128


def test_fragpack_slot_holds_the_largest_block_blob(tmp_path: Path) -> None:
    """FragPack slots fit a blob whose records are all raw, plus its head."""
    model_path = tmp_path / "model"
    _write_model_config(
        model_path,
        {
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 28,
            "torch_dtype": "bfloat16",
        },
    )
    cfg = DaserConfig(
        model_path=str(model_path),
        storage_format=STORAGE_FORMAT_FRAGPACK,
        block_tokens=128,
    )

    raw = 4 * 128 * 2 * 28 * 128 * 2
    assert cfg.resolved_local_slot_size() == 4096 + raw
    assert cfg.runtime_config()["slot_size"] == 4096 + raw
    assert cfg.l1_record_bytes == 0


@pytest.mark.parametrize(
    ("storage_format", "accounting", "expected"),
    [
        ("raw", "raw", 0),
        (STORAGE_FORMAT_COMPRESSED_ONLINE, "stored", 0),
        (STORAGE_FORMAT_COMPRESSED_ONLINE, "raw", 4 * 128 * 2 * 28 * 16 * 2),
    ],
)
def test_l1_record_bytes_follow_raw_accounting(
    tmp_path: Path, storage_format: str, accounting: str, expected: int
) -> None:
    """Raw L1 accounting charges compressed records their raw slot size."""
    model_path = tmp_path / "model"
    _write_model_config(
        model_path,
        {
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 28,
            "torch_dtype": "bfloat16",
        },
    )
    cfg = DaserConfig(
        model_path=str(model_path),
        storage_format=storage_format,
        l1_accounting=accounting,
        block_tokens=16,
    )

    assert cfg.l1_record_bytes == expected


def test_runtime_config_reuses_server_parameters(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(
        model_path,
        {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "num_layers": 4,
        },
    )
    cfg = DaserConfig(
        model_path=str(model_path),
        store_dir=str(store_dir),
        l2_size_bytes=512 * 64 * 2 * 4 * BLOCK_TOKENS * 2,
        ipc_socket_path="/tmp/custom.sock",
    )

    runtime_config = cfg.runtime_config()

    assert runtime_config == {
        "socket_path": "/tmp/custom.sock",
        "store_path": str(store_dir / "daser.store"),
        "slot_size": 8 * 64 * 2 * 4 * BLOCK_TOKENS * 2,
        "local_slot_size": 8 * 64 * 2 * 4 * BLOCK_TOKENS * 2,
        "tensor_parallel_size": 1,
        "block_tokens": BLOCK_TOKENS,
        "model_id": str(model_path),
        "cache_reuse_mode": "chunk",
        "transfer_mode": "iouring",
        "l1_size_bytes": DEFAULT_IOURING_L1_BYTES,
        "l2_size_bytes": 512 * 64 * 2 * 4 * BLOCK_TOKENS * 2,
        "l1_enabled": True,
        "l2_enabled": True,
        "tier_grain": DEFAULT_TIER_GRAIN,
        "l1_lane_bytes": DEFAULT_IOURING_L1_BYTES - DEFAULT_IOURING_L1_BYTES // 8,
        "l2_lane_bytes": 512 * 64 * 2 * 4 * BLOCK_TOKENS * 2,
        "staging_bytes": DEFAULT_IOURING_L1_BYTES // 8,
        "skip_l2": False,
        "storage_format": "raw",
        "l1_accounting": "stored",
        "prefetch_lease_budget_bytes": 0,
        "l2_read_stripe_bytes": 0,
    }


def test_runtime_config_omits_store_path_when_l2_is_skipped(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(
        model_path,
        {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "num_hidden_layers": 4,
        },
    )
    cfg = DaserConfig(
        model_path=str(model_path),
        store_dir=str(store_dir),
        skip_l2=True,
    )

    runtime_config = cfg.runtime_config()

    assert runtime_config["store_path"] == ""
    assert runtime_config["skip_l2"] is True
    assert runtime_config["l2_enabled"] is False
    assert runtime_config["l2_size_bytes"] == 0
    assert runtime_config["l2_lane_bytes"] == 0
    # Without L2 there is nothing to stage through: L1 takes the budget.
    assert runtime_config["staging_bytes"] == 0
    assert runtime_config["l1_lane_bytes"] == DEFAULT_IOURING_L1_BYTES
    assert cfg.persists_index is False


def test_tier_lanes_round_down_to_the_grain() -> None:
    """Lanes split each budget per TP rank in whole grains (design 3.6)."""
    grain = DEFAULT_TIER_GRAIN
    cfg = DaserConfig(
        l2_size_bytes=10 * 1000**3,
        l1_size_bytes=3 * 1000**3,
        tensor_parallel_size=2,
    )

    assert cfg.l2_lane_bytes == (10 * 1000**3 // 2) // grain * grain
    assert cfg.store_file_bytes == cfg.l2_lane_bytes * 2
    assert cfg.staging_bytes == MAX_STAGING_BYTES
    assert (
        cfg.l1_lane_bytes == ((3 * 1000**3 - MAX_STAGING_BYTES) // 2) // grain * grain
    )
    assert cfg.l1_arena_bytes == cfg.l1_lane_bytes * 2


def test_small_l1_keeps_one_grain_of_staging() -> None:
    cfg = DaserConfig(l1_size_bytes=4 * DEFAULT_TIER_GRAIN, l2_size_bytes=1 << 30)

    assert cfg.staging_bytes == DEFAULT_TIER_GRAIN
    assert cfg.l1_lane_bytes == 3 * DEFAULT_TIER_GRAIN


@pytest.mark.parametrize(
    ("kwargs", "l1_enabled"),
    [
        ({"transfer_mode": "gds"}, False),
        ({"l1_size_bytes": 0}, False),
        ({}, True),
    ],
)
def test_l1_tier_needs_iouring_and_a_budget(
    kwargs: dict[str, object], l1_enabled: bool
) -> None:
    cfg = DaserConfig(l2_size_bytes=1 << 30, **kwargs)

    assert cfg.l1_enabled is l1_enabled
    if not l1_enabled:
        assert cfg.l1_lane_bytes == 0
        assert cfg.staging_bytes == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tier_grain": 0},
        {"tier_grain": 2048},
        {"tier_grain": 3 * 4096},
        {"l1_size_bytes": -1},
        {"l2_size_bytes": -1},
        {"tensor_parallel_size": 0},
    ],
)
def test_config_rejects_invalid_tier_settings(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        DaserConfig(**kwargs)


def test_model_id_can_differ_from_model_path(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(
        model_path,
        {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "num_hidden_layers": 4,
        },
    )
    cfg = DaserConfig(
        model_path=str(model_path),
        store_dir=str(store_dir),
        vllm_model_id="served-model",
    )

    assert cfg.model_id == "served-model"
    assert cfg.runtime_config()["model_id"] == "served-model"


def test_tensor_parallel_runtime_uses_rank_lanes(tmp_path: Path) -> None:
    """TP geometry follows vLLM per-rank KV-head replication semantics."""
    model_path = tmp_path / "model"
    _write_model_config(
        model_path,
        {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "num_hidden_layers": 4,
        },
    )
    local_slot_size = 64 * 2 * 4 * BLOCK_TOKENS * 2
    cfg = DaserConfig(
        model_path=str(model_path),
        store_dir=str(tmp_path / "store"),
        l2_size_bytes=2 * 1024 * 1024,
        tensor_parallel_size=2,
    )

    runtime = cfg.runtime_config()

    assert cfg.resolved_local_slot_size() == local_slot_size
    assert cfg.resolved_slot_size() == local_slot_size * 2
    assert cfg.model_id == f"{model_path}::tp2"
    assert runtime["tensor_parallel_size"] == 2
    assert runtime["l2_lane_bytes"] == 1024 * 1024
    assert runtime["l2_size_bytes"] == 2 * 1024 * 1024


def test_model_geometry_rejects_incompatible_tp_head_count(tmp_path: Path) -> None:
    """TP geometry follows vLLM's KV-head divisibility requirement."""
    model_path = tmp_path / "model"
    _write_model_config(
        model_path,
        {
            "hidden_size": 768,
            "num_attention_heads": 6,
            "num_key_value_heads": 6,
            "num_hidden_layers": 2,
        },
    )

    with pytest.raises(ValueError, match="tensor_parallel_size"):
        model_geometry_from_path(str(model_path)).slot_size_for_block_tokens(16, 4)


def test_model_geometry_rejects_fp8_without_runtime_kv_spec(tmp_path: Path) -> None:
    """FP8 layouts fail before DaseR silently sizes them as BF16."""
    model_path = tmp_path / "model"
    _write_model_config(
        model_path,
        {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 2,
            "torch_dtype": "float8_e4m3fn",
        },
    )

    with pytest.raises(ValueError, match="FP8"):
        model_geometry_from_path(str(model_path))
