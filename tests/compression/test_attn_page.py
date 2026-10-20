# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from daser.compression import attn_page as ap


def _bf16_like(rng: np.random.Generator, scale: float) -> np.ndarray:
    values = (rng.standard_normal((ap.PAGE_TOKENS, ap.HEAD_DIM)) * scale).astype(
        np.float32
    )
    return (values.view(np.uint32) >> 16).astype(np.uint16)


@pytest.mark.parametrize("kv", [0, 1])
def test_round_trip_with_escapes_and_raw_bytes(kv: int) -> None:
    rng = np.random.default_rng(0)
    cal = _bf16_like(rng, 0.5)
    table = ap.calibrate_table((cal >> 8).astype(np.uint8))
    record = _bf16_like(rng, 0.5)
    # Force a few bytes outside the 22-entry table so raw escapes occur.
    record[3, 5] = 0x7F00 | (record[3, 5] & 0xFF)
    record[100, 127] = 0x0100
    encoded = ap.encode_record(record, table, kv)
    assert len(encoded) % 16 == 0
    assert len(encoded) <= ap.DENSE_BYTES + ap.VAR_CAP
    assert np.array_equal(ap.decode_record(encoded, table, kv), record)


def test_compressible_record_is_smaller_than_raw() -> None:
    rng = np.random.default_rng(1)
    table = ap.calibrate_table((_bf16_like(rng, 0.1) >> 8).astype(np.uint8))
    encoded = ap.encode_record(_bf16_like(rng, 0.1), table, 0)
    assert len(encoded) < 0.75 * ap.RAW_RECORD_BYTES


@pytest.mark.parametrize("kv", [0, 1])
def test_incompressible_record_round_trips_raw(kv: int) -> None:
    rng = np.random.default_rng(2)
    table = np.arange(22, dtype=np.uint8)
    record = rng.integers(
        0, 1 << 16, size=(ap.PAGE_TOKENS, ap.HEAD_DIM), dtype=np.uint16
    )
    encoded = ap.encode_record(record, table, kv)
    assert len(encoded) == ap.RAW_RECORD_BYTES
    assert np.array_equal(ap.decode_record(encoded, table, kv), record)


def test_rejects_wrong_record_size() -> None:
    with pytest.raises(ValueError):
        ap.encode_record(
            np.zeros(10, dtype=np.uint16), np.arange(22, dtype=np.uint8), 0
        )
