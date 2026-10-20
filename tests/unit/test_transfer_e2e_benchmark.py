# SPDX-License-Identifier: Apache-2.0
"""Protect E2E source sharing and exact per-wave logical coverage checks."""

import argparse
from copy import deepcopy
from typing import Any

import pytest

from benchmarks.bench_transfer_e2e import cycle_summary


def _cycle_fixture(
    storage_format: str = "compressed-online", concurrency: int = 4
) -> tuple[argparse.Namespace, dict[str, Any], list[dict[str, Any]]]:
    """Build two valid synthetic waves with one shared source per family.

    Args:
        storage_format: Format name selecting a synthetic physical prefix size.
        concurrency: Number of independent destinations sharing each source.
    Returns:
        Workload namespace, geometry contract, and independently owned wave dicts.
    Async/thread-safety:
        Pure CPU fixture construction; no shared state, services, or GPU imports.
    """
    args = argparse.Namespace(
        families=2,
        prefix_tokens=8,
        suffix_tokens=4,
        min_cycle_ssd_bytes=32,
        prefetch_max_requests=0,
    )
    prefix_bytes = 64 if storage_format == "raw" else 32
    contract = {
        "storage_format": storage_format,
        "minimum_prefix_bytes": prefix_bytes,
    }
    waves = [
        {
            "ordinal": family,
            "family_id": family,
            "elapsed_ms": 10.0,
            "valid": True,
            "ssd_touched": True,
            "l1_only": False,
            "requests": [
                {
                    "sample_id": family * concurrency + request,
                    "validation_errors": [],
                    "ttft_ms": 1.0,
                    "latency_ms": 2.0,
                    "completion_tokens": 1,
                }
                for request in range(concurrency)
            ],
            "counters": {
                "l2_reads": 1,
                "process_read_bytes": prefix_bytes,
                "prefetch_operations": 0,
                "load_operations": 1,
                "load_bytes": prefix_bytes,
                "external_query_tokens": concurrency * 12,
                "external_hit_tokens": concurrency * 11,
            },
        }
        for family in range(args.families)
    ]
    return args, contract, waves


@pytest.mark.parametrize("storage_format", ["raw", "compressed-online"])
@pytest.mark.parametrize("concurrency", [4, 16])
def test_shared_source_covers_every_destination_without_duplicate_transfers(
    storage_format: str, concurrency: int
) -> None:
    """Accept one physical source per family while covering all logical requests.

    Args:
        storage_format: Storage format whose minimum physical bytes are charged.
        concurrency: Number of requests sharing the source in each wave.
    Returns:
        None; assertions reject per-request physical-byte requirements.
    Async/thread-safety:
        Synchronous CPU-only validation of fresh fixture data.
    """
    args, contract, waves = _cycle_fixture(storage_format, concurrency)
    original = deepcopy(waves)

    result = cycle_summary(args, contract, waves, concurrency, 0)

    assert result["valid"], result["invalid_reasons"]
    assert result["summary"]["server_load_operations"] == args.families
    assert result["minimum_external_load_bytes"] == (
        args.families * contract["minimum_prefix_bytes"]
    )
    assert result["expected_external_query_tokens"] == args.families * concurrency * 12
    assert result["expected_external_hit_tokens"] == args.families * concurrency * 11
    assert waves == original


def test_missing_source_cannot_hide_behind_another_familys_traffic() -> None:
    """Reject a missing source even when cycle traffic meets the old larger bound.

    Args:
        None.
    Returns:
        None; assertions require a failure tied to the deficient family wave.
    Async/thread-safety:
        Synchronous CPU-only mutation of independently created fixture dicts.
    """
    args, contract, waves = _cycle_fixture()
    waves[0]["counters"]["load_bytes"] = 0
    waves[1]["counters"]["load_bytes"] = (
        args.families * 4 * contract["minimum_prefix_bytes"]
    )

    result = cycle_summary(args, contract, waves, 4, 0)

    assert not result["valid"]
    assert result["invalid_reasons"] == [
        "wave 0 (family 0): external load bytes do not cover its unique prefix"
    ]


@pytest.mark.parametrize(
    ("counter", "message"),
    [
        ("external_query_tokens", "external query tokens"),
        ("external_hit_tokens", "external hit tokens"),
    ],
)
@pytest.mark.parametrize("delta", [-1, 1])
def test_lookup_counts_are_exact_per_wave_even_when_cycle_totals_match(
    counter: str, message: str, delta: int
) -> None:
    """Reject undercounts and overcounts that cancel out in aggregate coverage.

    Args:
        counter: Public query or hit metric changed in the fixture.
        message: Diagnostic fragment identifying that metric's coverage failure.
        delta: Positive or negative one-token discrepancy in the first wave.
    Returns:
        None; assertions require both incorrect wave counts to be reported.
    Async/thread-safety:
        Synchronous CPU-only mutation of independently created fixture dicts.
    """
    args, contract, waves = _cycle_fixture()
    waves[0]["counters"][counter] += delta
    waves[1]["counters"][counter] -= delta

    result = cycle_summary(args, contract, waves, 4, 0)

    assert not result["valid"]
    assert result["invalid_reasons"] == [
        f"wave {family} (family {family}): {message} differ from exact expectation"
        for family in range(args.families)
    ]


@pytest.mark.parametrize("change", ["replace", "append", "remove"])
def test_each_planned_family_requires_exactly_one_wave(change: str) -> None:
    """Reject duplicate or missing families, including an extra otherwise valid wave.

    Args:
        change: Replace one family with a duplicate, append one, or remove one.
    Returns:
        None; assertions require exact family coverage instead of set inclusion.
    Async/thread-safety:
        Synchronous CPU-only mutation of independently created fixture dicts.
    """
    args, contract, waves = _cycle_fixture()
    if change == "replace":
        waves[1]["family_id"] = waves[0]["family_id"]
    elif change == "append":
        waves.append(deepcopy(waves[0]))
    else:
        waves.pop()

    result = cycle_summary(args, contract, waves, 4, 0)

    assert not result["valid"]
    assert (
        "cycle must contain exactly one wave per planned family"
        in result["invalid_reasons"]
    )


def test_request_count_cannot_shift_between_family_waves() -> None:
    """Reject uneven concurrency despite a correct cycle-wide number of requests.

    Args:
        None.
    Returns:
        None; assertions require the declared concurrency in both family waves.
    Async/thread-safety:
        Synchronous CPU-only mutation of independently created fixture dicts.
    """
    args, contract, waves = _cycle_fixture()
    waves[1]["requests"].append(waves[0]["requests"].pop())

    result = cycle_summary(args, contract, waves, 4, 0)

    assert not result["valid"]
    assert result["summary"]["request_count"] == args.families * 4
    assert result["invalid_reasons"] == [
        f"wave {family} (family {family}): request count differs from concurrency"
        for family in range(args.families)
    ]
