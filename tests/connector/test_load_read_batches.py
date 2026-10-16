# SPDX-License-Identifier: Apache-2.0
"""Shared-source load planning across staging-capacity boundaries."""

from copy import deepcopy

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("cupy")

from daser.connector.metadata import CompressedLoadSlot, ReqLoadSpec
from daser.connector.worker.load import build_load_read_batches


@pytest.mark.parametrize("capacity", [16, 24, 32, 128])
@pytest.mark.parametrize("packed", [False, True])
def test_shared_source_is_read_once_across_batches(capacity: int, packed: bool) -> None:
    """Every alias receives exact bytes while each shared fragment transfers once."""
    requests: dict[str, ReqLoadSpec] = {}
    expected: dict[int, bytes] = {}
    lengths = [8, 12, 16, 8] if packed else [16] * 4
    # Server-side records by slot id; a span reads its slots back to back.
    store: dict[int, bytes] = {}
    for alias in range(3):
        for source in range(2):
            records = []
            blocks = []
            for index, length in enumerate(lengths):
                block = alias * 8 + source * 4 + index
                payload = bytes([source * 4 + index + 1]) * length
                store[source * 4 + index] = payload
                blocks.append(block)
                expected[block] = payload
                records.append(
                    CompressedLoadSlot(
                        slot_id=source * 4 + index,
                        mode="raw" if length == 16 else "compressed",
                        stored_length=length,
                    )
                )
            requests[f"request{alias}:source{source}"] = ReqLoadSpec(
                chunk_key=f"source{source}",
                start_slot=source * 4,
                num_slots=4,
                block_ids=blocks,
                token_count=512,
                target_token_start=alias * 128,
                pos_offset=alias,
                compressed_slots=records if packed else [],
            )
    original = deepcopy(requests)
    batches = build_load_read_batches(requests, 16, capacity, include_req_ids=True)
    restored: dict[int, bytes] = {}
    assert sum(size for size, _spans, _ranges in batches) == 2 * sum(lengths)
    for size, spans, ranges in batches:
        assert 0 < size <= capacity
        staging = bytearray(size)
        for span in spans:
            first = span["start_slot"]
            data = b"".join(
                store[slot] for slot in range(first, first + span["num_slots"])
            )
            assert len(data) == span["nbytes"]
            start = span["target_offset"]
            staging[start : start + len(data)] = data
        for start, end, req_id, spec in ranges:
            identity = req_id.split("#")[0]
            assert spec.pos_offset == original[identity].pos_offset
            assert spec.target_token_start == original[identity].target_token_start
            assert spec.token_count == original[identity].token_count
            for index, block in enumerate(spec.block_ids):
                length = spec.compressed_slots[index].stored_length if packed else 16
                assert block not in restored
                restored[block] = bytes(staging[start : start + length])
                start += length
            assert start == end
    assert restored == expected
    assert requests == original
