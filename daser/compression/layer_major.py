# SPDX-License-Identifier: Apache-2.0

"""Layer-major staging of packed compressed slot records.

A packed record is a header of whole 4 KiB pages (one page holds 41 layers of
descriptors; deeper models use more) followed by its layer/K-or-V plane
records back-to-back. Restoring a layer group therefore needs only the header
and one contiguous byte range per record. This module plans that range for
many records at once and builds a staging image holding, per record, the
header pages followed by the group's plane bytes, with the group's plane
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
#: Byte offset of the little-endian uint32 ``header_bytes`` header field.
_HEADER_BYTES_FIELD = 12


def record_header_bytes(record: bytes | memoryview | NDArray[np.uint8]) -> int:
    """Return the header length declared by a packed record's first page.

    Args:
        record: At least the first header page of a packed record.

    Returns:
        The record's ``header_bytes`` field, a positive multiple of 4 KiB.

    Raises:
        ValueError: If the bytes are not a packed record header.

    Thread-safety:
        Pure computation on caller-owned bytes.
    """
    page = np.frombuffer(memoryview(record), dtype=np.uint8, count=IO_ALIGNMENT)
    if page[: len(SLOT_MAGIC)].tobytes() != SLOT_MAGIC:
        raise ValueError("layer-major staging requires packed record headers")
    field = page[_HEADER_BYTES_FIELD : _HEADER_BYTES_FIELD + 4]
    header_bytes = int(field.view("<u4")[0])
    if header_bytes <= 0 or header_bytes % IO_ALIGNMENT:
        raise ValueError("packed record header length is invalid")
    return header_bytes


def _descriptor_words(headers: NDArray[np.uint8]) -> NDArray[np.uint32]:
    """View each header's descriptor table as ``[records, planes, 12]`` words."""
    table = headers[:, _FIXED_HEADER_BYTES:]
    planes = table.shape[1] // _DESCRIPTOR_BYTES
    table = table[:, : planes * _DESCRIPTOR_BYTES]
    return table.view(np.uint32).reshape(len(headers), planes, _DESCRIPTOR_BYTES // 4)


def _group_bounds(
    headers: NDArray[np.uint8], plane_range: tuple[int, int]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return per-record ``[begin, end)`` of one plane group's bytes."""
    first, count = plane_range
    if (
        headers.ndim != 2
        or not len(headers)
        or headers.shape[1] <= 0
        or headers.shape[1] % IO_ALIGNMENT
    ):
        raise ValueError("headers must be a nonempty [records, 4 KiB pages] array")
    if first < 0 or count <= 0:
        raise ValueError("plane range is invalid")
    last = first + count
    header_bytes = headers.shape[1]
    if _FIXED_HEADER_BYTES + last * _DESCRIPTOR_BYTES > header_bytes:
        raise ValueError("plane range exceeds the descriptor table")
    magic = np.frombuffer(SLOT_MAGIC, dtype=np.uint8)
    if not (headers[:, : len(magic)] == magic).all():
        raise ValueError("layer-major staging requires packed record headers")
    declared = headers[:, _HEADER_BYTES_FIELD : _HEADER_BYTES_FIELD + 4].copy()
    if (declared.view("<u4")[:, 0] != header_bytes).any():
        raise ValueError("packed record header length does not match the headers")
    words = _descriptor_words(headers)
    begin = words[:, first, _RECORD_OFFSET_WORD].astype(np.int64)
    end = words[:, last - 1, _RECORD_OFFSET_WORD].astype(np.int64) + words[
        :, last - 1, _RECORD_OFFSET_WORD + 1
    ].astype(np.int64)
    if (begin < header_bytes).any() or (end <= begin).any():
        raise ValueError("packed record plane offsets are invalid")
    return begin, end


def _footprint(
    header_bytes: int, begin: NDArray[np.int64], end: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Return 4 KiB-rounded per-record staging bytes for one group."""
    footprint = header_bytes + begin % _DATA_ALIGNMENT + (end - begin)
    return (footprint + IO_ALIGNMENT - 1) // IO_ALIGNMENT * IO_ALIGNMENT


def _plan(
    header_bytes: int,
    plane_range: tuple[int, int],
    begin: NDArray[np.int64],
    end: NDArray[np.int64],
    footprint: NDArray[np.int64],
) -> "LayerGroupPlan":
    """Lay records out back-to-back with the given per-record footprints."""
    staging_offsets = np.concatenate(([0], np.cumsum(footprint)[:-1])).astype(np.int64)
    # Keep the source and staged copies of the group congruent modulo 16 so
    # every per-record copy is a 16-byte aligned run.
    return LayerGroupPlan(
        plane_range=(plane_range[0], plane_range[1]),
        source_begin=begin,
        source_end=end,
        staging_offsets=staging_offsets,
        header_bytes=header_bytes,
        data_offsets=staging_offsets + header_bytes + begin % _DATA_ALIGNMENT,
        staging_bytes=int(footprint.sum()),
    )


@dataclass(frozen=True)
class LayerGroupPlan:
    """Byte ranges and staging layout for one plane group of many records.

    Attributes:
        plane_range: ``(first, count)`` planes covered by the group.
        source_begin: Per-record start of the group's bytes in the record.
        source_end: Per-record exclusive end of the group's bytes.
        header_bytes: Header bytes staged before each record's group.
        staging_offsets: Per-record staging offset of its header pages; pass
            these to the decoder as slot offsets.
        data_offsets: Per-record staging offset of the group's bytes.
        staging_bytes: Total staging bytes the group needs.
    """

    plane_range: tuple[int, int]
    source_begin: NDArray[np.int64]
    source_end: NDArray[np.int64]
    header_bytes: int
    staging_offsets: NDArray[np.int64]
    data_offsets: NDArray[np.int64]
    staging_bytes: int


def plan_layer_group(
    headers: NDArray[np.uint8], plane_range: tuple[int, int]
) -> LayerGroupPlan:
    """Plan layer-major staging of one plane group for packed records.

    Args:
        headers: ``[records, header_bytes]`` uint8 header pages of packed
            records, as declared by ``record_header_bytes``.
        plane_range: ``(first, count)`` contiguous planes to stage.

    Returns:
        Source ranges and staging layout. Each record occupies its header
        pages followed by its group bytes, padded to 4 KiB.

    Raises:
        ValueError: If a header is not a packed record header, or the plane
            range does not fit the descriptor table.

    Thread-safety:
        Pure computation on caller-owned arrays.
    """
    begin, end = _group_bounds(headers, plane_range)
    header_bytes = headers.shape[1]
    return _plan(
        header_bytes, plane_range, begin, end, _footprint(header_bytes, begin, end)
    )


def plan_layer_groups(
    headers: NDArray[np.uint8], plane_ranges: list[tuple[int, int]]
) -> list[LayerGroupPlan]:
    """Plan layer-major staging of several plane groups with one slot layout.

    Every group places each record's header pages at the same staging offset,
    sized for the record's largest group, so one decoder ``prepare`` serves
    all groups and each group launches on its own staging buffer through
    ``PreparedKVRestore.submit_planes``.

    Args:
        headers: ``[records, header_bytes]`` uint8 header pages of packed
            records.
        plane_ranges: Nonempty ``(first, count)`` plane groups.

    Returns:
        One plan per group, all with equal ``staging_offsets`` and
        ``staging_bytes``.

    Raises:
        ValueError: As ``plan_layer_group``, or if no group is given.

    Thread-safety:
        Pure computation on caller-owned arrays.
    """
    if not plane_ranges:
        raise ValueError("at least one plane group is required")
    bounds = [_group_bounds(headers, group) for group in plane_ranges]
    header_bytes = headers.shape[1]
    footprint = np.max(
        [_footprint(header_bytes, begin, end) for begin, end in bounds], axis=0
    )
    return [
        _plan(header_bytes, group, begin, end, footprint)
        for group, (begin, end) in zip(plane_ranges, bounds, strict=True)
    ]


def rebased_headers(
    headers: NDArray[np.uint8], plan: LayerGroupPlan
) -> NDArray[np.uint8]:
    """Return header pages whose group descriptors point at staged bytes.

    Args:
        headers: The header pages passed to ``plan_layer_group``.
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


def staged_descriptor_patch(
    headers: NDArray[np.uint8], plan: LayerGroupPlan
) -> tuple[NDArray[np.int64], NDArray[np.int32]]:
    """Return the words that turn staged original headers into rebased ones.

    A staging buffer that already holds every record's original header page
    at ``plan.staging_offsets`` reads like ``rebased_headers`` after writing
    these words, so the production path patches a few bytes per record on
    the GPU instead of uploading whole pages per group.

    Args:
        headers: The header pages passed to ``plan_layer_group``.
        plan: Plan built from those headers.

    Returns:
        ``(indices, values)``: absolute 32-bit word indices into the staging
        buffer and the rebased words, bit-cast to int32 for device tensors.

    Thread-safety:
        Pure computation; the input is not modified.
    """
    first, count = plan.plane_range
    pages = rebased_headers(headers, plan)
    words = _descriptor_words(pages)[:, first : first + count, list(_REBASED_WORDS)]
    # Word index of each patched field relative to its header page.
    plane = np.arange(first, first + count, dtype=np.int64)[:, None]
    relative = (_FIXED_HEADER_BYTES + plane * _DESCRIPTOR_BYTES) // 4 + np.asarray(
        _REBASED_WORDS, dtype=np.int64
    )[None, :]
    indices = plan.staging_offsets[:, None, None] // 4 + relative[None, :, :]
    return indices.reshape(-1), words.reshape(-1).view(np.int32)


def stage_layer_group(
    records: list[bytes | memoryview], plan: LayerGroupPlan
) -> bytearray:
    """Build the layer-major staging image of one group on the host.

    Args:
        records: Packed record payloads, in plan order.
        plan: Plan built from those records' header pages.

    Returns:
        ``plan.staging_bytes`` bytes: each record's rebased header pages and
        its group bytes at the planned offsets. The production path performs
        the same copies straight into GPU staging.

    Thread-safety:
        Pure computation on caller-owned buffers.
    """
    header_bytes = plan.header_bytes
    headers = np.stack(
        [
            np.frombuffer(record, dtype=np.uint8, count=header_bytes)
            for record in records
        ]
    )
    pages = rebased_headers(headers, plan)
    staging = bytearray(plan.staging_bytes)
    for index, record in enumerate(records):
        page = int(plan.staging_offsets[index])
        staging[page : page + header_bytes] = pages[index].tobytes()
        begin, end = int(plan.source_begin[index]), int(plan.source_end[index])
        data = int(plan.data_offsets[index])
        staging[data : data + end - begin] = bytes(memoryview(record)[begin:end])
    return staging


__all__ = [
    "LayerGroupPlan",
    "plan_layer_group",
    "plan_layer_groups",
    "rebased_headers",
    "record_header_bytes",
    "stage_layer_group",
    "staged_descriptor_patch",
]
