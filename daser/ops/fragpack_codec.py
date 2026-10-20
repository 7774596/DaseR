# SPDX-License-Identifier: Apache-2.0
"""Move FragPack blocks between the HBM KV store and DaseR staging buffers.

With FragPack attention (``daser.attention.fragpack_backend``), the engine's KV cache
holds FragPack records rather than BF16 KV. External tiers therefore store each
block as a block blob (``attn_page.BLOB_MAGIC``) and never re-encode:
:class:`FragPackBlockExporter` copies closed blocks' records into store staging
with the same contract as the online packer, and :class:`FragPackBlockImporter`
installs fetched blobs into the destination blocks on the restore stream.
Both resolve the store lazily through :func:`daser.ops.hbm_kv.active_store`,
since attention creates it only after the connector registers its caches.
"""

from __future__ import annotations

import numpy as np
import torch

from daser.compression.format import SlotMode
from daser.ops.compressed_kv import OnlinePackedSlot
from daser.ops.hbm_kv import active_store, blob_failures


def _upload(
    values: list[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Copy host ints to the device on the current stream without blocking."""
    host = torch.tensor(values, dtype=dtype).pin_memory()
    return host.to(device, non_blocking=True)


class FragPackBlockExporter:
    """Copy closed FragPack blocks into fixed-stride store staging regions.

    Args:
        slot_stride: Staging bytes reserved per block; must equal the store's
            ``blob_stride`` (the server's FragPack slot size).

    Async/thread-safety:
        Call ``pack_into`` from one store thread at a time.
    """

    def __init__(self, slot_stride: int) -> None:
        self._slot_stride = int(slot_stride)

    def pack_into(
        self,
        *,
        staging: torch.Tensor,
        block_ids: list[int],
        logical_slots: list[int],
        slot_stride: int,
        stream: torch.cuda.Stream,
    ) -> list[OnlinePackedSlot]:
        """Write one blob per block at ``i * slot_stride`` of ``staging``.

        Args:
            staging: Worker-owned CUDA byte staging buffer.
            block_ids: Closed physical blocks in logical order.
            logical_slots: Prompt-relative slot ids matching ``block_ids``.
            slot_stride: Staging region stride (the FragPack slot size).
            stream: Store stream, already ordered after the producing forward.

        Returns:
            One compressed record per block: its source offset and its blob
            length (a 4 KiB multiple).

        Raises:
            ValueError: On mismatched metadata or staging geometry.

        Async/thread-safety:
            Blocks the calling thread until ``stream`` has copied the blobs.
        """
        if not block_ids or len(block_ids) != len(logical_slots):
            raise ValueError("FragPack export block metadata is invalid")
        store = active_store()
        if slot_stride != self._slot_stride or slot_stride != store.blob_stride:
            raise ValueError(
                f"FragPack export stride {slot_stride} does not match the store's "
                f"{store.blob_stride}"
            )
        if staging.numel() < len(block_ids) * slot_stride:
            raise ValueError("FragPack export staging capacity is insufficient")
        with torch.cuda.stream(stream):
            blocks = _upload(block_ids, torch.int32, staging.device)
            lengths = store.export_blocks(blocks, staging, slot_stride)
            host = torch.empty(len(block_ids), dtype=torch.int64, pin_memory=True)
            host.copy_(lengths, non_blocking=True)
        stream.synchronize()
        return [
            OnlinePackedSlot(
                logical_slot=int(slot),
                mode=SlotMode.COMPRESSED,
                source_offset=i * slot_stride,
                stored_length=int(length),
            )
            for i, (slot, length) in enumerate(
                zip(logical_slots, host.tolist(), strict=True)
            )
        ]


class FragPackRestorePlan:
    """One prepared blob import; mirrors ``PreparedKVRestore``.

    Async/thread-safety:
        Owned by the load thread. Read :meth:`failed_blocks` only after an
        event recorded after :meth:`submit` has completed.
    """

    def __init__(
        self,
        staging: torch.Tensor,
        offsets: torch.Tensor,
        blocks: torch.Tensor,
        block_ids: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        self._staging = staging
        self._offsets = offsets
        self._blocks = blocks
        self._block_ids = block_ids
        self._stream = stream
        self._status = torch.zeros(len(block_ids), dtype=torch.int32, pin_memory=True)
        self._submitted = False

    @property
    def destination_count(self) -> int:
        """Return the number of destination blocks."""
        return len(self._block_ids)

    def submit(self) -> int:
        """Enqueue the import and its status copy on the plan's stream.

        Returns:
            Destination blocks covered, or zero if already submitted.

        Async/thread-safety:
            Non-blocking; the caller orders ``stream`` after the transfer
            and after the destinations' previous owner.
        """
        if self._submitted:
            return 0
        self._submitted = True
        with torch.cuda.stream(self._stream):
            status = active_store().import_blocks(
                self._staging, self._offsets, self._blocks
            )
            self._status.copy_(status, non_blocking=True)
        return self.destination_count

    def failed_blocks(self) -> list[tuple[int, str]]:
        """Return ``(block, reason)`` for every destination not installed.

        Returns:
            Failed destinations; empty when every blob was installed.

        Async/thread-safety:
            Valid only after the submission's completion event.
        """
        status = self._status.numpy()
        return [
            (self._block_ids[i], blob_failures(int(status[i])))
            for i in np.flatnonzero(status)
        ]


class FragPackBlockImporter:
    """Prepare imports of fetched FragPack blobs; mirrors the fused decoder.

    Async/thread-safety:
        Call from the load thread only.
    """

    def prepare(
        self,
        *,
        staging: torch.Tensor,
        staging_offsets: list[int],
        block_ids: list[int],
        modes: list[int],
        buffer_index: int,
        stream: torch.cuda.Stream,
        plane_range: tuple[int, int] | None = None,
    ) -> FragPackRestorePlan | None:
        """Upload import metadata without launching the import.

        Args:
            staging: Leased GPU byte tensor the server fills with blobs.
            staging_offsets: Blob offset per destination.
            block_ids: Distinct destination blocks.
            modes: Record modes; every FragPack blob is compressed (1).
            buffer_index: Staging ring index (unused; metadata is per plan).
            stream: Restore stream ordering the upload and the import.
            plane_range: Must be None: FragPack blobs restore whole blocks.

        Returns:
            Restore plan, or None for empty input.

        Raises:
            ValueError: On mismatched metadata, non-compressed modes,
                duplicate destinations, or a plane range.

        Async/thread-safety:
            Enqueues host-to-device copies on ``stream``; does not block.
        """
        del buffer_index
        if plane_range is not None:
            raise ValueError("FragPack blobs do not support layer-wise restore")
        if not len(staging_offsets) == len(block_ids) == len(modes):
            raise ValueError("FragPack restore metadata lengths do not match")
        if not block_ids:
            return None
        if any(mode != 1 for mode in modes):
            raise ValueError("FragPack restore requires compressed records")
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("FragPack restore destinations must be distinct")
        with torch.cuda.stream(stream):
            offsets = _upload(staging_offsets, torch.int64, staging.device)
            blocks = _upload(block_ids, torch.int32, staging.device)
        return FragPackRestorePlan(staging, offsets, blocks, list(block_ids), stream)


__all__ = ["FragPackBlockExporter", "FragPackBlockImporter", "FragPackRestorePlan"]
