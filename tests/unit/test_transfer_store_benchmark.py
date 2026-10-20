# SPDX-License-Identifier: Apache-2.0
"""Protect benchmark geometry and separation of acceptance/completion metrics."""

import numpy
import pytest

from benchmarks.bench_transfer_store import build_requests, summarize


@pytest.mark.parametrize("storage_format", ["raw", "packed"])
@pytest.mark.parametrize("case", ["same", "overlap", "disjoint"])
def test_store_geometry_retains_overlap_and_compact_source(
    storage_format: str, case: str
) -> None:
    """Each request owns compact source bytes and the intended shared extents."""
    requests = build_requests(256 * 1024, 4, case, storage_format, 4)
    keys = [
        {(span["file_offset"], span["nbytes"]) for span in request}
        for request in requests
    ]
    for request in requests:
        cursor = 0
        for span in request:
            assert span["source_offset"] == cursor
            assert span["file_offset"] % 4096 == 0
            assert span["nbytes"] % 4096 == 0
            cursor += span["nbytes"]
        assert cursor == 256 * 1024
        stride = request[1]["file_offset"] - request[0]["file_offset"]
        assert stride == request[0]["nbytes"] * (2 if storage_format == "packed" else 1)
    assert len(keys[0] & keys[1]) == {"same": 4, "overlap": 2, "disjoint": 0}[case]


def test_unaligned_half_overlap_is_rejected_without_rounding() -> None:
    """A 4 KiB request cannot silently turn its half-overlap into a larger IO."""
    with pytest.raises(ValueError, match="even number of aligned spans"):
        build_requests(4096, 4, "overlap", "raw", 4)


def test_completion_throughput_does_not_use_early_acceptance() -> None:
    """The throughput denominator includes asynchronous persistence drain."""
    samples = [
        {
            "acceptance_latency_ms": [1.0, 2.0],
            "acceptance_ms": 2.0,
            "drain_ms": 8.0,
            "completion_ms": 10.0,
            "requested_bytes": 1 << 20,
            "l2_writes": 2,
            "native_write_calls": 2,
            "native_write_bytes": 1 << 20,
            "correctness": True,
        }
    ]
    result = summarize(samples, numpy)
    assert result["acceptance_p50_ms"] == 1.5
    assert result["completion_p50_ms"] == 10.0
    assert result["drain_p50_ms"] == 8.0
    assert result["completion_throughput_gib_s"] == pytest.approx((1 / 1024) / 0.01)
    assert result["native_write_bytes_total"] == 1 << 20
    assert result["correctness"]
