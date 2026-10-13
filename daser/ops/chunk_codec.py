# SPDX-License-Identifier: Apache-2.0

"""GPU codec adapter for engine-owned kv-major KV chunks.

External KV cache engines such as LMCache stage one chunk per object as a
contiguous ``[2, layers, tokens, heads, head_dim]`` BF16 tensor (K/V outermost)
and copy it to fixed-size host objects. This adapter lets such an engine store
DaseR's strict-lossless online record instead of the raw chunk without
touching the DaseR kernels: it permutes chunks into a small private ``BLNHC``
scratch cache, runs ``FusedOnlineKVPacker`` / ``FusedCompressedKVDecoder`` on
that scratch, and writes records back into the first ``capacity_bytes`` of the
engine's own staging buffer. Engines whose staging buffer already fuses K
and V inside each head (``[layers, tokens, heads, 2, head_dim]``, as LMCache
stages vLLM's ``kv_size == 1`` layouts) pass ``fused_kv=True`` and skip the
permute.

Records are variable-length by default: each chunk stores its own 4 KiB
aligned record, and a chunk whose record would not be smaller than raw keeps
its raw bytes, so a stored length equal to ``raw_bytes`` marks a raw chunk.
Engines then size each host object from the length ``encode`` returns.
Engines that can only allocate one object size pass ``capacity_ratio`` to pad
every record to the capacity class ``align4K(raw * capacity_ratio)``; a chunk
that does not fit raises ``ChunkCapacityError`` and callers fail the store
closed, because the class cannot hold a raw fallback.
"""

from __future__ import annotations

import math
from pathlib import Path
import threading

import torch

from daser.compression.calibration import load_calibrated_codebooks
from daser.compression.codec import default_online_codebooks
from daser.compression.format import (
    ONLINE_TILE_SCALARS,
    CompressedStoreGeometry,
    SlotMode,
    align_up,
)
from daser.ops.compressed_kv import (
    FusedCompressedKVDecoder,
    FusedOnlineKVPacker,
    warm_fused_online_kv_packer,
)
from daser.ops.kv_layout import allocate_cross_layer_kv_cache, physical_kv_view

# Decoder metadata rings alternate so one batch can upload metadata while the
# previous batch's decoder may still be queued on the stream.
_DECODE_RING_DEPTH = 2


class ChunkCapacityError(RuntimeError):
    """A chunk's compressed record exceeds the configured capacity class."""


def chunk_capacity_bytes(raw_bytes: int, capacity_ratio: float) -> int:
    """Return the fixed host-object size for compressed chunk records.

    Args:
        raw_bytes (int): Raw chunk bytes; must be 4 KiB aligned.
        capacity_ratio (float): Capacity class as a fraction of raw, in
            ``(0, 1]``.

    Returns:
        ``align4K(ceil(raw_bytes * capacity_ratio))``, never above raw.

    Raises:
        ValueError: If either argument is out of range.

    Async/thread-safety:
        Pure; safe from any thread.
    """
    if raw_bytes <= 0 or raw_bytes % align_up(1):
        raise ValueError("raw chunk bytes must be positive and 4 KiB aligned")
    if not 0.0 < capacity_ratio <= 1.0:
        raise ValueError("capacity ratio must be in (0, 1]")
    return min(raw_bytes, align_up(math.ceil(raw_bytes * capacity_ratio)))


def kv_major_to_physical(chunk: torch.Tensor) -> torch.Tensor:
    """View a ``[2, L, T, H, D]`` chunk in DaseR's ``[L, T, H, 2, D]`` order.

    Args:
        chunk (torch.Tensor): Kv-major chunk tensor.

    Returns:
        A permuted view sharing ``chunk``'s memory.

    Raises:
        ValueError: If ``chunk`` is not five-dimensional with K/V outermost.
    """
    if chunk.dim() != 5 or chunk.shape[0] != 2:
        raise ValueError(f"unsupported kv-major chunk shape {tuple(chunk.shape)}")
    return chunk.permute(1, 2, 3, 0, 4)


def physical_to_kv_major(block: torch.Tensor) -> torch.Tensor:
    """View one ``[L, T, H, 2, D]`` block in kv-major ``[2, L, T, H, D]`` order.

    Args:
        block (torch.Tensor): One physical DaseR ``BLNHC`` block.

    Returns:
        A permuted view sharing ``block``'s memory.

    Raises:
        ValueError: If ``block`` is not five-dimensional with K/V at dim 3.
    """
    if block.dim() != 5 or block.shape[3] != 2:
        raise ValueError(f"unsupported physical block shape {tuple(block.shape)}")
    return block.permute(3, 0, 1, 2, 4)


class KVMajorChunkCodec:
    """Encode and decode kv-major BF16 chunks in their engine staging buffers.

    Args:
        num_layers (int): KV layers in one chunk.
        chunk_tokens (int): Tokens in one chunk.
        num_kv_heads (int): KV heads per layer on this rank.
        head_dim (int): Scalars per head.
        device (torch.device): CUDA device holding the engine staging buffers.
        max_chunks (int): Largest chunk batch passed to one call.
        capacity_ratio (float | None): Fixed capacity class as a fraction of
            raw bytes; None stores variable-length records.
        codebook_dir (str | Path | None): Calibration artifact directory
            for this model; the model-independent generic palette when None.
        fused_kv (bool): Chunks hold K and V side by side in each head,
            ``[layers, tokens, heads, 2, head_dim]``, instead of kv-major.
            This is DaseR's physical block order, so no permute is needed.

    Raises:
        ValueError: If the device is not CUDA, the geometry is unsupported,
            or the calibrated codebooks were fitted for another geometry.
        FileNotFoundError: If ``codebook_dir`` lacks the artifact files.

    Async/thread-safety:
        Encode and decode own separate scratch and may run concurrently on
        different threads; concurrent calls of the same direction serialize
        on an internal lock. Every call enqueues on the caller's stream.
        ``encode`` synchronizes that stream because record lengths are read
        back to the host; ``decode`` does not synchronize.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        chunk_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        max_chunks: int,
        capacity_ratio: float | None = None,
        codebook_dir: str | Path | None = None,
        fused_kv: bool = False,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("kv-major chunk codec requires a CUDA device")
        if max_chunks <= 0:
            raise ValueError("kv-major chunk codec batch must be positive")
        self._geometry = CompressedStoreGeometry(
            num_slots=max_chunks,
            slot_size=num_layers * 2 * chunk_tokens * num_kv_heads * head_dim * 2,
            block_tokens=chunk_tokens,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tile_scalars=ONLINE_TILE_SCALARS,
        )
        self._fixed_capacity = capacity_ratio is not None
        # Variable records are bounded by raw: anything larger stays raw.
        self._capacity = (
            chunk_capacity_bytes(self._geometry.slot_size, capacity_ratio)
            if capacity_ratio is not None
            else self._geometry.slot_size
        )
        self._max_chunks = max_chunks
        self._fused_kv = fused_kv
        self._chunk_shape = (
            (num_layers, chunk_tokens, num_kv_heads, 2, head_dim)
            if fused_kv
            else (2, num_layers, chunk_tokens, num_kv_heads, head_dim)
        )
        codebooks = (
            default_online_codebooks(self._geometry)
            if codebook_dir is None
            else load_calibrated_codebooks(codebook_dir, self._geometry)
        )
        with torch.cuda.device(device), torch.inference_mode(False):
            self._encode_scratch = self._scratch_cache(device)
            self._decode_scratch = self._scratch_cache(device)
            self._pack_staging = torch.empty(
                max_chunks * self._geometry.slot_size, dtype=torch.uint8, device=device
            )
            self._decode_staging = torch.empty(
                max_chunks * self._capacity, dtype=torch.uint8, device=device
            )
            # Compile the kernels here so no store or retrieve pays TileLang JIT.
            warm_fused_online_kv_packer(
                self._encode_scratch, max_chunks, tile_scalars=ONLINE_TILE_SCALARS
            )
            self._packer = FusedOnlineKVPacker(
                kv_cache=self._encode_scratch,
                codebooks=codebooks,
                tile_scalars=ONLINE_TILE_SCALARS,
                max_slots_per_buffer=max_chunks,
            )
            self._decoder = FusedCompressedKVDecoder(
                kv_cache=self._decode_scratch,
                codebooks=codebooks,
                tile_scalars=ONLINE_TILE_SCALARS,
                ring_depth=_DECODE_RING_DEPTH,
                max_slots_per_buffer=max_chunks,
            )
        # Scratch and staging are shared by calls that may use other streams.
        self._encode_done: torch.cuda.Event | None = None
        self._decode_done: torch.cuda.Event | None = None
        self._decode_events: list[torch.cuda.Event | None] = [None] * _DECODE_RING_DEPTH
        self._next_ring = 0
        self._encode_lock = threading.Lock()
        self._decode_lock = threading.Lock()

    @property
    def raw_bytes(self) -> int:
        """Return the raw bytes of one kv-major chunk."""
        return self._geometry.slot_size

    @property
    def capacity_bytes(self) -> int:
        """Return the largest stored length of one chunk.

        The fixed capacity class when ``capacity_ratio`` was given (every
        stored length then equals it), otherwise ``raw_bytes``.
        """
        return self._capacity

    @property
    def fixed_capacity(self) -> bool:
        """Return whether every record is padded to ``capacity_bytes``."""
        return self._fixed_capacity

    def encode(
        self, chunks: list[torch.Tensor], stream: torch.cuda.Stream
    ) -> list[int]:
        """Replace each raw chunk's leading bytes with its compressed record.

        Args:
            chunks (list[torch.Tensor]): Contiguous CUDA uint8 staging buffers
                of ``raw_bytes`` each, holding BF16 chunks in the codec's chunk
                layout (kv-major, or fused when ``fused_kv``). Once
                ``stream`` passes this call, the first ``capacity_bytes`` of
                each hold its record followed by unspecified padding.
            stream (torch.cuda.Stream): Stream ordered after the producer of
                ``chunks``; consumers of the records must also use it.

        Returns:
            Stored length of each chunk: the 4 KiB aligned record length,
            ``raw_bytes`` for a chunk kept raw, or ``capacity_bytes`` for
            every chunk under a fixed capacity class. Engines copy and later
            restore exactly this prefix of each buffer.

        Raises:
            ValueError: If buffers do not match the codec geometry.
            ChunkCapacityError: Under a fixed capacity class, if a record
                does not fit it. The chunks' contents are then unspecified.

        Async/thread-safety:
            Synchronizes ``stream`` to read record lengths.
        """
        self._validate(chunks)
        raw = self._geometry.slot_size
        with self._encode_lock:
            self._order_after(self._encode_done, stream)
            physical = physical_kv_view(self._encode_scratch)
            with torch.cuda.stream(stream):
                for index, chunk in enumerate(chunks):
                    physical[index].copy_(self._physical_view(chunk))
            records = self._packer.pack_into(
                staging=self._pack_staging,
                block_ids=list(range(len(chunks))),
                logical_slots=list(range(len(chunks))),
                slot_stride=raw,
                stream=stream,
            )
            lengths: list[int] = []
            for record in records:
                compressed = (
                    record.mode is SlotMode.COMPRESSED and record.stored_length < raw
                )
                if self._fixed_capacity and (
                    not compressed or record.stored_length > self._capacity
                ):
                    raise ChunkCapacityError(
                        f"chunk record needs {record.stored_length} bytes "
                        f"(mode {record.mode.name}), capacity is {self._capacity}"
                    )
                lengths.append(record.stored_length if compressed else raw)
            with torch.cuda.stream(stream):
                for chunk, record, length in zip(chunks, records, lengths, strict=True):
                    # A raw chunk already holds its stored bytes.
                    if length < raw:
                        start = record.source_offset
                        chunk[:length].copy_(self._pack_staging[start : start + length])
                self._encode_done = torch.cuda.Event()
                self._encode_done.record(stream)
            if self._fixed_capacity:
                return [self._capacity] * len(chunks)
            return lengths

    def decode(
        self,
        chunks: list[torch.Tensor],
        lengths: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """Restore raw chunks from records at their buffer starts.

        Args:
            chunks (list[torch.Tensor]): Contiguous CUDA uint8 staging buffers
                of ``raw_bytes`` each whose first ``lengths[i]`` bytes hold
                what ``encode`` stored. They are overwritten in place.
            lengths (list[int]): Stored length of each chunk as returned by
                ``encode``; ``raw_bytes`` marks a chunk kept raw, which is
                left untouched.
            stream (torch.cuda.Stream): Stream ordered after the copies that
                filled ``chunks``.

        Raises:
            ValueError: If buffers do not match the codec geometry, a length
                is out of range, or a record header is malformed.

        Async/thread-safety:
            Enqueues work on ``stream`` without synchronizing it; the chunks
            are valid once ``stream`` reaches this point.
        """
        self._validate(chunks)
        if len(lengths) != len(chunks) or any(
            not 0 < length <= self._capacity for length in lengths
        ):
            raise ValueError("kv-major chunk stored lengths are out of range")
        raw = self._geometry.slot_size
        compressed = [
            index
            for index, length in enumerate(lengths)
            if length < raw or self._fixed_capacity
        ]
        if not compressed:
            return
        with self._decode_lock:
            ring = self._next_ring
            self._next_ring = (ring + 1) % _DECODE_RING_DEPTH
            pending = self._decode_events[ring]
            if pending is not None:
                # The ring's pinned metadata is rewritten by ``prepare``.
                pending.synchronize()
            self._order_after(self._decode_done, stream)
            capacity = self._capacity
            physical = physical_kv_view(self._decode_scratch)
            with torch.cuda.stream(stream):
                for index in compressed:
                    length = lengths[index]
                    start = index * capacity
                    self._decode_staging[start : start + length].copy_(
                        chunks[index][:length]
                    )
            self._decoder.decode(
                staging=self._decode_staging,
                staging_offsets=[index * capacity for index in compressed],
                block_ids=compressed,
                modes=[int(SlotMode.COMPRESSED)] * len(compressed),
                buffer_index=ring,
                stream=stream,
            )
            with torch.cuda.stream(stream):
                for index in compressed:
                    self._physical_view(chunks[index]).copy_(physical[index])
                self._decode_done = torch.cuda.Event()
                self._decode_done.record(stream)
            self._decode_events[ring] = self._decode_done

    @staticmethod
    def _order_after(
        previous: torch.cuda.Event | None, stream: torch.cuda.Stream
    ) -> None:
        """Make ``stream`` wait for the previous call's use of shared scratch."""
        if previous is not None:
            stream.wait_event(previous)

    def _scratch_cache(self, device: torch.device) -> torch.Tensor:
        """Allocate one private logical ``BLNHC`` cache of ``max_chunks`` blocks."""
        geometry = self._geometry
        return allocate_cross_layer_kv_cache(
            self._max_chunks,
            geometry.num_layers,
            geometry.block_tokens,
            geometry.num_kv_heads,
            geometry.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )

    def _physical_view(self, chunk: torch.Tensor) -> torch.Tensor:
        """View a raw uint8 chunk buffer in DaseR's physical block order."""
        view = chunk.view(torch.bfloat16).view(self._chunk_shape)
        return view if self._fused_kv else kv_major_to_physical(view)

    def _validate(self, chunks: list[torch.Tensor]) -> None:
        """Reject batches that do not match the compiled geometry."""
        if not 0 < len(chunks) <= self._max_chunks:
            raise ValueError("kv-major chunk batch size is out of range")
        for chunk in chunks:
            if (
                chunk.dtype is not torch.uint8
                or chunk.device != self._pack_staging.device
                or chunk.numel() != self._geometry.slot_size
                or not chunk.is_contiguous()
                or chunk.data_ptr() % 4
            ):
                raise ValueError("kv-major chunk buffer does not match the codec")
