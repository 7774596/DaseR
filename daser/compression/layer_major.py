# SPDX-License-Identifier: Apache-2.0

"""Layer-major staging of packed compressed slot records.

A packed record is a 4 KiB header page followed by its layer/K-or-V plane
records back-to-back. Restoring a layer group therefore needs only the header
and one contiguous byte range per record. This module plans that range for
many records at once and builds a staging image holding, per record, the
header page followed by the group's plane bytes, with the group's plane
descriptors rebased so the unchanged GPU decoder reads them in place.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from daser.compression.format import IO_ALIGNMENT, SLOT_MAGIC

#: Fixed slot header bytes before the plane descriptor table.
_FIXED_HEADER_BYTES = 120
#: Bytes per plane descriptor.
_DESCRIPTOR_BYTES = 48
#: uint32 word index of ``record_offset`` inside one descriptor.
_RECORD_OFFSET_WORD = 3
#: uint32 word indices of the plane-relative offsets a group move rebases:
#: record, low, symbol, prefix and escape offsets.
_REBASED_WORDS = (3, 5, 6, 7, 8)
#: Staged plane bytes start 16-byte aligned for vectorized copies.
_DATA_ALIGNMENT = 16


@dataclass(frozen=True)
class LayerGroupPlan:
    """Byte ranges and staging layout for one plane group of many records.

    Attributes:
        plane_range: ``(first, count)`` planes covered by the group.
        source_begin: Per-record start of the group's bytes in the record.
        source_end: Per-record exclusive end of the group's bytes.
        staging_offsets: Per-record staging offset of its header page; pass
            these to the decoder as slot offsets.
        data_offsets: Per-record staging offset of the group's bytes.
        staging_bytes: Total staging bytes the group needs.
    """

    plane_range: tuple[int, int]
    source_begin: NDArray[np.int64]
    source_end: NDArray[np.int64]
    staging_offsets: NDArray[np.int64]
    data_offsets: NDArray[np.int64]
    staging_bytes: int


def plan_layer_group(
    headers: NDArray[np.uint8], plane_range: tuple[int, int]
) -> LayerGroupPlan:
    """Plan layer-major staging of one plane group for packed records.

    Args:
        headers: ``[records, 4096]`` uint8 header pages of packed records.
        plane_range: ``(first, count)`` contiguous planes to stage.

    Returns:
        Source ranges and staging layout. Each record occupies its header
        page followed by its group bytes, padded to 4 KiB.

    Raises:
        ValueError: If a header is not a packed record header, or the plane
            range does not fit the descriptor table.

    Thread-safety:
        Pure computation on caller-owned arrays.
    """
    first, count = plane_range
    if headers.ndim != 2 or headers.shape[1] != IO_ALIGNMENT or not len(headers):
        raise ValueError("headers must be a nonempty [records, 4096] array")
    if first < 0 or count <= 0:
        raise ValueError("plane range is invalid")
    last = first + count
    if _FIXED_HEADER_BYTES + last * _DESCRIPTOR_BYTES > IO_ALIGNMENT:
        raise ValueError("plane range exceeds the descriptor table")
    magic = np.frombuffer(SLOT_MAGIC, dtype=np.uint8)
    if not (headers[:, : len(magic)] == magic).all():
        raise ValueError("layer-major staging requires packed record headers")
    words = _descriptor_words(headers)
    begin = words[:, first, _RECORD_OFFSET_WORD].astype(np.int64)
    end = words[:, last - 1, _RECORD_OFFSET_WORD].astype(np.int64) + words[
        :, last - 1, _RECORD_OFFSET_WORD + 1
    ].astype(np.int64)
    if (begin < IO_ALIGNMENT).any() or (end <= begin).any():
        raise ValueError("packed record plane offsets are invalid")
    # Keep the source and staged copies of the group congruent modulo 16 so
    # every per-record copy is a 16-byte aligned run.
    lead = begin % _DATA_ALIGNMENT
    footprint = IO_ALIGNMENT + lead + (end - begin)
    footprint = (footprint + IO_ALIGNMENT - 1) // IO_ALIGNMENT * IO_ALIGNMENT
    staging_offsets = np.concatenate(([0], np.cumsum(footprint)[:-1])).astype(np.int64)
    return LayerGroupPlan(
        plane_range=(first, count),
        source_begin=begin,
        source_end=end,
        staging_offsets=staging_offsets,
        data_offsets=staging_offsets + IO_ALIGNMENT + lead,
        staging_bytes=int(footprint.sum()),
    )


def rebased_headers(
    headers: NDArray[np.uint8], plan: LayerGroupPlan
) -> NDArray[np.uint8]:
    """Return header pages whose group descriptors point at staged bytes.

    Args:
        headers: The ``[records, 4096]`` pages passed to ``plan_layer_group``.
        plan: Plan built from those headers.

    Returns:
        A copy of ``headers`` with the record, low, symbol, prefix and escape
        offsets of every in-range plane shifted by the record's move, so
        offsets stay relative to the staged header page. Descriptors outside
        the range are left unchanged and must not be decoded.

    Thread-safety:
        Pure computation; the input is not modified.
    """
    first, count = plan.plane_range
    shifted = headers.copy()
    words = _descriptor_words(shifted)
    delta = (plan.data_offsets - plan.staging_offsets) - plan.source_begin
    group = words[:, first : first + count, :]
    for word in _REBASED_WORDS:
        group[:, :, word] = (
            group[:, :, word].astype(np.int64) + delta[:, None]
        ).astype(np.uint32)
    return shifted


def stage_layer_group(
    records: list[bytes | memoryview], plan: LayerGroupPlan
) -> bytearray:
    """Build the layer-major staging image of one group on the host.

    Args:
        records: Packed record payloads, in plan order.
        plan: Plan built from those records' header pages.

    Returns:
        ``plan.staging_bytes`` bytes: each record's rebased header page and
        its group bytes at the planned offsets. The production path performs
        the same copies straight into GPU staging.

    Thread-safety:
        Pure computation on caller-owned buffers.
    """
    headers = np.stack(
        [
            np.frombuffer(record, dtype=np.uint8, count=IO_ALIGNMENT)
            for record in records
        ]
    )
    pages = rebased_headers(headers, plan)
    staging = bytearray(plan.staging_bytes)
    for index, record in enumerate(records):
        page = int(plan.staging_offsets[index])
        staging[page : page + IO_ALIGNMENT] = pages[index].tobytes()
        begin, end = int(plan.source_begin[index]), int(plan.source_end[index])
        data = int(plan.data_offsets[index])
        staging[data : data + end - begin] = bytes(memoryview(record)[begin:end])
    return staging


def _descriptor_words(headers: NDArray[np.uint8]) -> NDArray[np.uint32]:
    """View each header's descriptor table as ``[records, planes, 12]`` words."""
    table = headers[:, _FIXED_HEADER_BYTES:]
    planes = table.shape[1] // _DESCRIPTOR_BYTES
    table = table[:, : planes * _DESCRIPTOR_BYTES]
    return table.view(np.uint32).reshape(len(headers), planes, _DESCRIPTOR_BYTES // 4)


__all__ = [
    "LayerGroupPlan",
    "plan_layer_group",
    "rebased_headers",
    "stage_layer_group",
]
