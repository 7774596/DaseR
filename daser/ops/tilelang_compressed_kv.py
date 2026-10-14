# SPDX-License-Identifier: Apache-2.0
"""TileLang fused online compressed-KV store and load operators.

The operators in this module deliberately use the existing variable-length
record format. Store is exposed as a small kernel bundle because a global
prefix is required before variable-length records can be emitted; all stages
are launched on the caller's stream and no host round trip occurs between
them. The final store stage reads the live KV cache directly after layout
planning and emits the slot header in the same launch, avoiding a full-sized
intermediate payload copy and a per-slot host-to-device header copy. Load is a
single kernel because its record offsets and lengths are already known when
the server returns a read plan.

Records are plane-major: plane ``2 * layer + kv`` holds one layer's K or V
scalars in ``[tokens, heads, head_dim]`` order. The kernels address vLLM's
block-outermost cache, whose physical block shape is
``[layers, tokens * heads, 2, head_dim]``, so K and V of one head are adjacent
in memory while the record keeps them in separate planes.
"""

# Do not enable ``from __future__ import annotations``. TileLang inspects the
# concrete annotations while constructing the prim_func.

from dataclasses import dataclass
from typing import Any

import tilelang
import tilelang.language as T
from tilelang.transform import PassConfigKey

from daser.compression.format import (
    IO_ALIGNMENT,
    ONLINE_TILE_SCALARS,
    slot_header_bytes,
)

_COUNT_THREADS = 256
_COUNT_WARPS = _COUNT_THREADS // 32
# Compact assigns one warp to each tile.
_COMPACT_THREADS = 256
_COMPACT_WARPS = _COMPACT_THREADS // 32
_SLOT_FIXED_HEADER_BYTES = 120
_DESCRIPTOR_BYTES = 48
_CODEBOOK_ENTRIES = 15
# The decoder pads each seven-entry codebook level to eight
# bytes so a level fits one PRMT byte pair (see ``pack_load_codebooks``).
_PACKED_CODEBOOK_BYTES = 16
# Eight-tile groups decoded per load CTA; 32 measured fastest
# against 16 and 64 on Qwen3-8B planes (512 tiles).
_LOAD_TILE_GROUPS = 32
_ESCAPE_PACKED_FLAG = 0x80
_ESCAPE_3BIT_FLAG = 0x40


@dataclass(frozen=True)
class TileLangStoreKernels:
    """Compiled TileLang stages used by the online store pipeline."""

    encode: Any
    layout: Any
    compact: Any


_STORE_CACHE: dict[tuple[int, int, int, int, int, int], TileLangStoreKernels] = {}
_LOAD_CACHE: dict[tuple[int, int, int, int, int, bool], Any] = {}


def _vllm_kv_shape(
    num_blocks: int, num_planes: int, plane_scalars: int, head_dim: int
) -> tuple[int, int, int, int, int]:
    """Return the physical ``[blocks, layers, rows, 2, head_dim]`` KV shape."""
    return (num_blocks, num_planes // 2, plane_scalars // head_dim, 2, head_dim)


def _validate_head_dim(num_planes: int, plane_scalars: int, head_dim: int) -> None:
    """Reject geometries that do not split into K/V planes of whole heads."""
    if head_dim <= 0 or num_planes % 2 or plane_scalars % head_dim:
        raise ValueError("KV planes must hold whole K or V heads")


def _validate_tile_scalars(tile_scalars: int) -> None:
    """Reject tile sizes other than the online codec's.

    Args:
        tile_scalars: Codec tile size requested by the caller.

    Raises:
        ValueError: If ``tile_scalars`` is not ``ONLINE_TILE_SCALARS``.
    """
    if tile_scalars != ONLINE_TILE_SCALARS:
        raise ValueError(
            f"TileLang codec kernels support tile_scalars={ONLINE_TILE_SCALARS}, "
            f"got {tile_scalars}"
        )


# Device helpers injected into the word-parallel load kernel. ``__byte_perm``
# masks every non-constant selector with 0x7777 and nvcc expands a C-level
# bit spread into one operation per code, so both are written as inline PTX.
_LOAD_DEVICE_SOURCE = r"""
__device__ __forceinline__ unsigned int daser_prmt(
    unsigned int low, unsigned int high, unsigned int selector) {
  unsigned int result;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(low), "r"(high), "r"(selector));
  return result;
}
__device__ __forceinline__ unsigned int daser_spread_codes(unsigned int word) {
  unsigned int spread;
  asm("{\n\t.reg .b32 t;\n\t"
      "shl.b32 t, %1, 4;\n\t"
      "lop3.b32 %0, %1, t, 0x0000FFFF, 0xE4;\n\t"
      "shl.b32 t, %0, 2;\n\t"
      "lop3.b32 %0, %0, t, 0x003F003F, 0xE4;\n\t"
      "shl.b32 t, %0, 1;\n\t"
      "lop3.b32 %0, %0, t, 0x0F0F0F0F, 0xE4;\n\t"
      "and.b32 %0, %0, 0x77777777;\n\t}"
      : "=r"(spread) : "r"(word));
  return spread;
}
"""


def _prmt(low: Any, high: Any, selector: Any) -> Any:
    """Return the PRMT byte permute of the eight bytes ``high:low``.

    Selector nibble ``i`` (bits 16-31 are ignored) picks the byte of the pair
    placed at result byte ``i``. Callers keep bit 3 of each nibble clear, so
    no sign replication occurs.
    """
    return T.call_extern("uint32", "daser_prmt", low, high, T.cast(selector, T.uint32))


def _spread_codes(word: Any) -> Any:
    """Move the eight 3-bit codes at bits ``3 * i`` of ``word`` to nibble ``i``."""
    return T.call_extern("uint32", "daser_spread_codes", word)


def _nibble_flags(codes: Any) -> Any:
    """Return 1 in each nibble of spread ``codes`` that holds the escape code 7."""
    return codes & (codes >> 1) & (codes >> 2) & T.cast(0x11111111, T.uint32)


def _rank_selector(flags: Any) -> Any:
    """Return a PRMT selector that gathers flagged bytes in order.

    Nibble ``i`` is the number of flagged nibbles below ``i`` when nibble
    ``i`` of ``flags`` is set (the byte's rank in a packed escape stream),
    and 7 otherwise. ``flags * 0x11111110`` sums lower flags into each
    nibble; it never carries because at most eight flags exist.
    """
    return (flags * T.cast(0x11111110, T.uint32)) | (
        T.cast(0x77777777, T.uint32) - flags * T.cast(7, T.uint32)
    )


def _clamp(index: Any, last: Any) -> Any:
    """Return ``index`` clamped to ``last``, or unchanged when ``last`` is None."""
    return index if last is None else T.min(index, last)


def _record_word(words: Any, word_base: Any, lead: Any, offset: Any, last: Any) -> Any:
    """Return the little-endian ``uint32`` at a byte offset inside a record.

    ``words`` is the int32 view of the staging bytes; the record starts
    ``lead`` (0-3) bytes into word ``word_base``. The value is assembled from
    the two aligned words covering the byte with one funnel shift (two loads
    instead of four byte loads).

    Offsets are unsigned 32-bit, so a corrupt negative rank wraps to a large
    offset, and both word indices are clamped to ``last``, the final staging
    word relative to ``word_base``. Every read therefore stays inside the
    staging tensor with 32-bit arithmetic; callers only consume bytes inside
    the record, so clamped tail bytes of a record at the extent are unused.
    ``last=None`` skips the clamps for reads the caller proved in range.
    """
    total = lead + T.cast(offset, T.uint32)
    first = _clamp(total >> 2, last)
    second = _clamp(first + 1, last)
    return T.call_extern(
        "uint32",
        "__funnelshift_r",
        T.cast(words[word_base + first], T.uint32),
        T.cast(words[word_base + second], T.uint32),
        (total & 3) * 8,
    )


def compile_store_kernels(
    *,
    num_blocks: int,
    num_planes: int,
    plane_scalars: int,
    head_dim: int,
    tile_scalars: int,
) -> TileLangStoreKernels:
    """Compile the online TileLang store stages for one KV geometry.

    Args:
        num_blocks: Number of physical vLLM KV blocks in the cache.
        num_planes: Number of layer/K-or-V planes.
        plane_scalars: BF16 scalars in one plane.
        head_dim: Scalars in one K or V head.
        tile_scalars: Scalar quantum used by the packed record; must be
            ``ONLINE_TILE_SCALARS``.

    Returns:
        Cached fused encode, layout, and compact kernels.

    Raises:
        ValueError: If the geometry is not positive, the tile size is not
            ``ONLINE_TILE_SCALARS``, or the geometry exceeds the supported
            one-tile shared-memory envelope.

    Async/thread-safety:
        Compilation is synchronous and intended for registration-time
        warmup. The process-local cache is only mutated during that phase.
    """
    if min(num_blocks, num_planes, plane_scalars) <= 0:
        raise ValueError("TileLang store geometry must be positive")
    _validate_tile_scalars(tile_scalars)
    _validate_head_dim(num_planes, plane_scalars, head_dim)
    tiles_per_plane = (plane_scalars + tile_scalars - 1) // tile_scalars
    if tiles_per_plane > 1024:
        raise ValueError("TileLang store supports at most 1024 tiles per plane")
    key = (
        num_blocks,
        num_planes,
        plane_scalars,
        head_dim,
        tile_scalars,
        tiles_per_plane,
    )
    cached = _STORE_CACHE.get(key)
    if cached is not None:
        return cached
    kernels = TileLangStoreKernels(
        encode=tilelang.compile(
            _build_store_count_prefix(
                num_blocks=num_blocks,
                num_planes=num_planes,
                plane_scalars=plane_scalars,
                head_dim=head_dim,
                tile_scalars=tile_scalars,
                tiles_per_plane=tiles_per_plane,
            ),
            target="cuda",
            execution_backend="cython",
        ),
        layout=tilelang.compile(
            _build_store_layout(
                num_planes=num_planes,
                plane_scalars=plane_scalars,
                max_tiles=tiles_per_plane,
            ),
            target="cuda",
            execution_backend="cython",
        ),
        compact=tilelang.compile(
            _build_store_compact(
                num_blocks=num_blocks,
                num_planes=num_planes,
                plane_scalars=plane_scalars,
                head_dim=head_dim,
                tile_scalars=tile_scalars,
                max_tiles=tiles_per_plane,
            ),
            target="cuda",
            execution_backend="cython",
        ),
    )
    _STORE_CACHE[key] = kernels
    return kernels


def pack_load_codebooks(codebooks: bytes, num_planes: int) -> bytes:
    """Lay out per-plane codebooks for the load kernel.

    Args:
        codebooks (bytes): ``num_planes`` rows of ``CODEBOOK_ENTRIES`` bytes:
            seven primary entries followed by the secondary entries.
        num_planes (int): Number of layer/K-or-V planes.

    Returns:
        bytes: ``num_planes`` rows of 16 bytes:
        bytes 0-6 hold the primary entries and bytes 8-14 the first seven
        secondary entries, so each level fits one PRMT byte pair; bytes 7
        and 15 are zero. Symbol 7 is the escape at both levels, so the
        padding is never selected for a decoded scalar.

    Raises:
        ValueError: If ``codebooks`` does not hold ``num_planes`` rows.
    """
    if len(codebooks) != num_planes * _CODEBOOK_ENTRIES:
        raise ValueError(
            f"expected {num_planes * _CODEBOOK_ENTRIES} codebook bytes, "
            f"got {len(codebooks)}"
        )
    packed = bytearray(num_planes * _PACKED_CODEBOOK_BYTES)
    for plane in range(num_planes):
        row = codebooks[plane * _CODEBOOK_ENTRIES : (plane + 1) * _CODEBOOK_ENTRIES]
        base = plane * _PACKED_CODEBOOK_BYTES
        packed[base : base + 7] = row[:7]
        packed[base + 8 : base + 15] = row[7:14]
    return bytes(packed)


def compile_load_kernel(
    *,
    num_blocks: int,
    num_planes: int,
    plane_scalars: int,
    head_dim: int,
    tile_scalars: int,
    staging_bytes: int,
    fanout: bool = False,
) -> Any:
    """Compile the single-launch TileLang fused compressed load kernel.

    Args:
        num_blocks: Number of destination vLLM KV blocks.
        num_planes: Number of layer/K-or-V planes.
        plane_scalars: BF16 scalars in one plane.
        head_dim: Scalars in one K or V head.
        tile_scalars: Scalar quantum encoded in each record; must be
            ``ONLINE_TILE_SCALARS``. The launch takes the staging bytes plus
            their ``int32`` view, so staging must be 4-byte aligned with a
            length that is a multiple of four, and codebooks laid out by
            ``pack_load_codebooks``.
        staging_bytes: Representative staging capacity used for warmup metadata;
            the compiled kernel accepts the actual extent at runtime.
        fanout: Whether sources may have multiple destination blocks. Both
            variants are compiled at startup; fanout counts remain dynamic.

    Returns:
        Cached TileLang JIT kernel. Slot count and staging extent remain runtime
        dimensions, so request metadata does not trigger compilation.

    Raises:
        ValueError: If geometry is not positive, the tile size is not
            ``ONLINE_TILE_SCALARS``, or the tile count is too large.

    Async/thread-safety:
        Compilation is synchronous and should happen before request traffic.
        The returned kernel launches asynchronously on the current PyTorch
        stream.
    """
    if min(num_blocks, num_planes, plane_scalars, staging_bytes) <= 0:
        raise ValueError("TileLang load geometry must be positive")
    _validate_tile_scalars(tile_scalars)
    _validate_head_dim(num_planes, plane_scalars, head_dim)
    tiles_per_plane = (plane_scalars + tile_scalars - 1) // tile_scalars
    if tiles_per_plane > 1024:
        raise ValueError("TileLang load supports at most 1024 tiles per plane")
    # The staging extent is a runtime tensor dimension. Keep the argument in
    # the public API for callers that document their pool capacity, but do not
    # specialize the JIT cache on that capacity.
    key = (num_blocks, num_planes, plane_scalars, head_dim, tile_scalars, fanout)
    cached = _LOAD_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = tilelang.compile(
        _build_load(
            num_blocks=num_blocks,
            num_planes=num_planes,
            plane_scalars=plane_scalars,
            head_dim=head_dim,
            tile_scalars=tile_scalars,
            tiles_per_plane=tiles_per_plane,
            staging_bytes=staging_bytes,
            fanout=fanout,
        ),
        target="cuda",
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True},
    )
    _LOAD_CACHE[key] = kernel
    return kernel


def _build_store_count_prefix(
    *,
    num_blocks: int,
    num_planes: int,
    plane_scalars: int,
    head_dim: int,
    tile_scalars: int,
    tiles_per_plane: int,
) -> Any:
    """Count escapes and emit their plane prefixes within one CTA.

    Each warp counts independent tiles into shared memory. After every tile
    is ready, a block scan emits the unchanged two prefix tables and totals.
    This removes the global tile-count arrays and a separate prefix launch;
    compact still reads KV directly after slot layout is known.
    """
    n_slots = T.dynamic("N", "int64")
    row_items = T.dynamic("Q", "int64")
    scratch_bytes = T.dynamic("R", "int64")

    @T.prim_func
    def main(
        kv_bits: T.Tensor(
            _vllm_kv_shape(num_blocks, num_planes, plane_scalars, head_dim), "uint16"
        ),
        block_ids: T.Tensor((n_slots,), "int32"),
        lookup: T.Tensor((num_planes * 256,), "uint8"),
        totals: T.Tensor((row_items,), "uint32"),
        raw_totals: T.Tensor((row_items,), "uint32"),
        overflow: T.Tensor((n_slots,), "uint32"),
        scratch: T.Tensor((scratch_bytes,), "uint8"),
        block_tokens: T.int32,
        valid_token_count: T.int32,
        fixed_escape_bytes: T.int32,
        scratch_plane_record_bytes: T.int32,
    ):
        with T.Kernel(n_slots, num_planes, threads=_COUNT_THREADS) as (bx, by):
            tile_counts = T.alloc_shared((tiles_per_plane,), "uint32")
            raw_counts = T.alloc_shared((tiles_per_plane,), "uint32")
            warp_sums = T.alloc_shared((_COUNT_WARPS,), "uint32")
            raw_warp_sums = T.alloc_shared((_COUNT_WARPS,), "uint32")
            for tx in T.Parallel(_COUNT_THREADS):
                slot = bx
                plane = by
                lane = tx & 31
                warp = tx // 32
                block_id = block_ids[slot]
                row_scalars = plane_scalars // block_tokens
                # Geometry guarantees whole, positive token rows. Hoist the
                # partial-slot boundary so the scalar loop needs no runtime
                # division; trailing slots still emit canonical zero bits.
                valid_scalars = (
                    T.min(
                        T.max(valid_token_count - slot * block_tokens, 0), block_tokens
                    )
                    * row_scalars
                )
                for tile_group in range(
                    (tiles_per_plane + _COUNT_WARPS - 1) // _COUNT_WARPS
                ):
                    tile = tile_group * _COUNT_WARPS + warp
                    lane_count = T.alloc_var(T.uint32, init=0)
                    raw_lane_count = T.alloc_var(T.uint32, init=0)
                    for scalar_group in range((tile_scalars + 31) // 32):
                        scalar = tile * tile_scalars + scalar_group * 32 + lane
                        # The final warp group of an odd-sized tile must not
                        # count scalars that belong to the following tile.
                        active = (
                            tile < tiles_per_plane
                            and scalar < plane_scalars
                            and scalar_group * 32 + lane < tile_scalars
                        )
                        valid = active and scalar < valid_scalars
                        bits = (
                            T.cast(
                                kv_bits[
                                    block_id,
                                    plane // 2,
                                    scalar // head_dim,
                                    plane % 2,
                                    scalar % head_dim,
                                ],
                                T.uint16,
                            )
                            if valid
                            else 0
                        )
                        high = (bits >> 8) & 255
                        code = (
                            T.cast(lookup[plane * 256 + high], T.uint8) if valid else 15
                        )
                        # Invalid tail scalars count as zero-bit escapes,
                        # matching the format's canonical partial-slot fill.
                        lane_count = lane_count + T.cast(active and code >= 7, T.uint32)
                        # Online mode uses the narrow three-bit escape stream:
                        # entries 7..13 remain secondary codes and entries
                        # 14..15 are emitted as raw high bytes.  Counting the
                        # full raw set here lets the layout stage size that
                        # stream without a host round trip.
                        raw_lane_count = raw_lane_count + T.cast(
                            active and code >= 14, T.uint32
                        )
                    for offset_index in range(5):
                        lane_count = lane_count + T.shfl_down(
                            lane_count, 16 >> offset_index
                        )
                        raw_lane_count = raw_lane_count + T.shfl_down(
                            raw_lane_count, 16 >> offset_index
                        )
                    if lane == 0 and tile < tiles_per_plane:
                        tile_counts[tile] = lane_count
                        raw_counts[tile] = raw_lane_count
                T.sync_threads()
                row = slot * num_planes + plane
                prefix_base = row * scratch_plane_record_bytes
                raw_prefix_base = prefix_base + 4 * (tiles_per_plane + 1)
                if tx < 4:
                    scratch[prefix_base + tx] = 0
                    scratch[raw_prefix_base + tx] = 0
                running = T.alloc_var(T.uint32, init=0)
                raw_running = T.alloc_var(T.uint32, init=0)
                for group in range(
                    (tiles_per_plane + _COUNT_THREADS - 1) // _COUNT_THREADS
                ):
                    tile = group * _COUNT_THREADS + tx
                    inclusive = T.alloc_var(
                        T.uint32,
                        init=tile_counts[tile] if tile < tiles_per_plane else 0,
                    )
                    raw_inclusive = T.alloc_var(
                        T.uint32,
                        init=raw_counts[tile] if tile < tiles_per_plane else 0,
                    )
                    for step in T.unroll(5):
                        preceding = T.shfl_up(inclusive, 1 << step)
                        raw_preceding = T.shfl_up(raw_inclusive, 1 << step)
                        if lane >= (1 << step):
                            inclusive = inclusive + preceding
                            raw_inclusive = raw_inclusive + raw_preceding
                    if lane == 31:
                        warp_sums[warp] = inclusive
                        raw_warp_sums[warp] = raw_inclusive
                    T.sync_threads()
                    if warp == 0:
                        warp_scan = T.alloc_var(
                            T.uint32, init=warp_sums[lane] if lane < _COUNT_WARPS else 0
                        )
                        raw_warp_scan = T.alloc_var(
                            T.uint32,
                            init=raw_warp_sums[lane] if lane < _COUNT_WARPS else 0,
                        )
                        for step in T.unroll(5):
                            preceding = T.shfl_up(warp_scan, 1 << step)
                            raw_preceding = T.shfl_up(raw_warp_scan, 1 << step)
                            if lane >= (1 << step):
                                warp_scan = warp_scan + preceding
                                raw_warp_scan = raw_warp_scan + raw_preceding
                        if lane < _COUNT_WARPS:
                            warp_sums[lane] = warp_scan
                            raw_warp_sums[lane] = raw_warp_scan
                    T.sync_threads()
                    inclusive = (
                        inclusive + running + (warp_sums[warp - 1] if warp > 0 else 0)
                    )
                    raw_inclusive = (
                        raw_inclusive
                        + raw_running
                        + (raw_warp_sums[warp - 1] if warp > 0 else 0)
                    )
                    if tile < tiles_per_plane:
                        for byte in T.unroll(4):
                            scratch[prefix_base + (tile + 1) * 4 + byte] = T.cast(
                                inclusive >> (byte * 8), T.uint8
                            )
                            scratch[raw_prefix_base + (tile + 1) * 4 + byte] = T.cast(
                                raw_inclusive >> (byte * 8), T.uint8
                            )
                    running = running + warp_sums[_COUNT_WARPS - 1]
                    raw_running = raw_running + raw_warp_sums[_COUNT_WARPS - 1]
                    # No warp may replace a partial sum for the next group
                    # while another still reads this group's shared prefix.
                    T.sync_threads()
                if tx == 0:
                    totals[row] = running
                    raw_totals[row] = raw_running
                    escape_bytes_3 = (running * 3 + 7) // 8 + raw_running
                    escape_bytes_4 = (running + 1) // 2 + raw_running
                    if (
                        escape_bytes_3 > fixed_escape_bytes
                        and escape_bytes_4 > fixed_escape_bytes
                    ):
                        # Plane CTAs share one monotonic slot flag. It is
                        # zeroed on this stream before launch; layout consumes
                        # it only after every plane has completed.
                        T.atomic_max(overflow[slot], T.cast(1, T.uint32))

    return main


def _build_store_layout(*, num_planes: int, plane_scalars: int, max_tiles: int) -> Any:
    """Build compact variable-length offsets from device totals."""
    n_slots = T.dynamic("N", "int64")
    row_items = T.dynamic("Q", "int64")
    payload_items = T.dynamic("B", "int64")

    @T.prim_func
    def main(
        totals: T.Tensor((row_items,), "uint32"),
        raw_totals: T.Tensor((row_items,), "uint32"),
        overflow: T.Tensor((n_slots,), "uint32"),
        symbol_bits: T.Tensor((n_slots,), "int32"),
        source_offsets: T.Tensor((row_items,), "int64"),
        destination_offsets: T.Tensor((row_items,), "int64"),
        payload_bytes: T.Tensor((payload_items,), "int64"),
        slot_offsets: T.Tensor((n_slots,), "int64"),
        slot_stride: T.int32,
        scratch_plane_record_bytes: T.int32,
        scratch_slot_stride: T.int32,
    ):
        header_bytes = slot_header_bytes(num_planes)
        prefix_bytes_3 = 8 * (max_tiles + 1)
        payload_base_3 = plane_scalars + (plane_scalars * 3 + 7) // 8 + prefix_bytes_3
        # A single warp preserves the exact slot prefix order while computing
        # each consecutive group of plane lengths in parallel. Every lane
        # carries the same staging cursor; inactive lanes contribute zero.
        with T.Kernel(1, threads=32):
            for lane in T.Parallel(32):
                staging_cursor = T.alloc_var(T.int64, init=0)
                for slot in range(n_slots):
                    row_base = slot * num_planes
                    raw = T.alloc_var(
                        T.int32, init=T.cast(overflow[slot] != 0, T.int32)
                    )
                    record_cursor = T.alloc_var(T.int64, init=header_bytes)
                    if raw == 0:
                        for group in range((num_planes + 31) // 32):
                            plane = group * 32 + lane
                            row = row_base + plane
                            length = T.alloc_var(T.int64, init=0)
                            if plane < num_planes:
                                length = (
                                    payload_base_3
                                    + (totals[row] * 3 + 7) // 8
                                    + raw_totals[row]
                                )
                            inclusive = T.alloc_var(T.int64, init=length)
                            for step in T.unroll(5):
                                preceding = T.shfl_up(inclusive, 1 << step)
                                if lane >= (1 << step):
                                    inclusive = inclusive + preceding
                            if plane < num_planes:
                                source_offsets[row] = (
                                    slot * scratch_slot_stride
                                    + header_bytes
                                    + plane * scratch_plane_record_bytes
                                )
                                destination_offsets[row] = (
                                    staging_cursor + record_cursor + inclusive - length
                                )
                                payload_bytes[row] = length
                            # All lanes read lane 31, including zero-contribution
                            # lanes of a final partial group. This keeps the next
                            # group's base uniform without a shared-memory fence.
                            record_cursor = record_cursor + T.shfl_down(
                                inclusive, 31 - lane
                            )
                        if record_cursor > slot_stride:
                            raw = 1
                    if raw != 0:
                        # Layout may have tentatively emitted packed offsets
                        # before the slot-wide envelope rejected that record.
                        # Clear every descriptor before the next kernel runs.
                        for group in range((num_planes + 31) // 32):
                            plane = group * 32 + lane
                            if plane < num_planes:
                                row = row_base + plane
                                source_offsets[row] = 0
                                destination_offsets[row] = 0
                                payload_bytes[row] = 0
                    if lane == 0:
                        slot_offsets[slot] = staging_cursor
                        overflow[slot] = T.cast(raw, T.uint32)
                        # Selector 5 denotes the unchanged 3+3-bit wire format.
                        symbol_bits[slot] = 0 if raw != 0 else 5
                    staging_cursor = staging_cursor + (
                        slot_stride
                        if raw != 0
                        else (record_cursor + 4095) // 4096 * 4096
                    )

    return main


def _build_store_compact(
    *,
    num_blocks: int,
    num_planes: int,
    plane_scalars: int,
    head_dim: int,
    tile_scalars: int,
    max_tiles: int,
) -> Any:
    """Emit packed records, headers, and raw-overflow slots in one launch.

    The layout planner marks overflow slots with zero payload descriptors and
    assigns them a raw slot base. One block owns a slot/plane row and each warp
    advances through independent tiles, reducing the launch grid from one
    block per tile to one block per plane while preserving the persisted
    record format.  The plane-zero block also serializes the canonical 4 KiB
    header after layout planning.  Header bytes are written by the same CUDA
    launch as payload bytes, so the caller does not need a pageable or pinned
    host metadata copy on the hot path.
    """
    row_items = T.dynamic("Q", "int64")
    slot_items = T.dynamic("N", "int64")
    destination_bytes = T.dynamic("S", "int64")
    scratch_bytes = T.dynamic("R", "int64")
    escape_word_items = T.dynamic("W", "int64")

    @T.prim_func
    def main(
        kv_bits: T.Tensor(
            _vllm_kv_shape(num_blocks, num_planes, plane_scalars, head_dim), "uint16"
        ),
        block_ids: T.Tensor((slot_items,), "int32"),
        logical_slots: T.Tensor((slot_items,), "int64"),
        overflow: T.Tensor((slot_items,), "uint32"),
        destination: T.Tensor((destination_bytes,), "uint8"),
        escape_words: T.Tensor((escape_word_items,), "uint32"),
        destination_offsets: T.Tensor((row_items,), "int64"),
        payload_bytes: T.Tensor((row_items,), "int64"),
        totals: T.Tensor((row_items,), "uint32"),
        raw_totals: T.Tensor((row_items,), "uint32"),
        slot_offsets: T.Tensor((slot_items,), "int64"),
        symbol_bits: T.Tensor((slot_items,), "int32"),
        lookup: T.Tensor((num_planes * 256,), "uint8"),
        codebook_hash: T.Tensor((32,), "uint8"),
        scratch: T.Tensor((scratch_bytes,), "uint8"),
        tile_escape_capacity: T.int32,
        scratch_plane_record_bytes: T.int32,
        block_tokens: T.int32,
        valid_token_count: T.int32,
        slot_stride: T.int32,
    ):
        with T.Kernel(slot_items, num_planes, threads=_COMPACT_THREADS) as (bx, by):
            for tx in T.Parallel(_COMPACT_THREADS):
                slot = bx
                plane = by
                lane = T.cast(tx & 31, T.int32)
                warp = T.cast(tx // 32, T.int32)
                row = slot * num_planes + plane
                block_id = block_ids[slot]
                # Match the count pass's token-tail boundary without dividing
                # every scalar index in the payload emission loops.
                valid_scalars = T.min(
                    T.max(valid_token_count - slot * block_tokens, 0), block_tokens
                ) * (plane_scalars // block_tokens)
                if overflow[slot] != 0:
                    if plane == 0:
                        raw_base = slot_offsets[slot]
                        for chunk in range(
                            (num_planes * plane_scalars + _COMPACT_THREADS - 1)
                            // _COMPACT_THREADS
                        ):
                            index = chunk * _COMPACT_THREADS + tx
                            if index < num_planes * plane_scalars:
                                value = kv_bits[
                                    block_id,
                                    index // plane_scalars // 2,
                                    index % plane_scalars // head_dim,
                                    index // plane_scalars % 2,
                                    index % plane_scalars % head_dim,
                                ]
                                destination[raw_base + index * 2] = T.cast(
                                    value & 255, T.uint8
                                )
                                destination[raw_base + index * 2 + 1] = T.cast(
                                    value >> 8, T.uint8
                                )
                else:
                    # The compact launch owns the header as well as payload
                    # bytes.  Header fields use the little-endian wire
                    # layout from ``compression.format``; clearing the page
                    # first preserves the canonical zero-filled tail emitted
                    # by ``SlotHeader.pack``.
                    # The online layout planner emits only the new packed
                    # format here (``symbol_bits == 5``: 3-bit symbols plus
                    # 3-bit escape tokens).  Keep these byte counts static in
                    # the TileLang builder.  Making them depend on a device
                    # tensor caused the generated kernel to select the legacy
                    # 4-bit symbol width for header offsets even though the
                    # layout planner had reserved the 3-bit payload.
                    packed_escape = True
                    escape_symbol_bits = 3
                    prefix_bytes = 8 * (max_tiles + 1)
                    symbol_bytes = (plane_scalars * 3 + 7) // 8
                    base_bytes = plane_scalars + symbol_bytes + prefix_bytes
                    # Deep models spill descriptors past one 4 KiB page.
                    header_bytes = slot_header_bytes(num_planes)
                    if plane == 0:
                        header_base = slot_offsets[slot]
                        for header_index in range(header_bytes // _COMPACT_THREADS):
                            destination[
                                header_base + header_index * _COMPACT_THREADS + tx
                            ] = 0
                        T.sync_threads()
                        if tx == 0:
                            destination[header_base + 0] = 68
                            destination[header_base + 1] = 75
                            destination[header_base + 2] = 86
                            destination[header_base + 3] = 83
                            destination[header_base + 4] = 76
                            destination[header_base + 5] = 79
                            destination[header_base + 6] = 84
                            destination[header_base + 7] = 49
                            for byte in range(4):
                                destination[header_base + 8 + byte] = T.cast(
                                    (1 >> (byte * 8)) & 255, T.uint8
                                )
                                destination[header_base + 12 + byte] = T.cast(
                                    (header_bytes >> (byte * 8)) & 255, T.uint8
                                )
                                destination[header_base + 16 + byte] = T.cast(
                                    (IO_ALIGNMENT >> (byte * 8)) & 255, T.uint8
                                )
                                destination[header_base + 20 + byte] = T.cast(
                                    (tile_scalars >> (byte * 8)) & 255, T.uint8
                                )
                            for byte in range(8):
                                destination[header_base + 24 + byte] = T.cast(
                                    (logical_slots[slot] >> (byte * 8)) & 255,
                                    T.uint8,
                                )
                                destination[header_base + 40 + byte] = T.cast(
                                    (slot_stride >> (byte * 8)) & 255, T.uint8
                                )
                            for byte in range(4):
                                destination[header_base + 32 + byte] = T.cast(
                                    (num_planes // 2) >> (byte * 8),
                                    T.uint8,
                                )
                                destination[header_base + 36 + byte] = T.cast(
                                    (num_planes >> (byte * 8)) & 255, T.uint8
                                )
                            for byte in range(32):
                                destination[header_base + 56 + byte] = codebook_hash[
                                    byte
                                ]
                            destination[header_base + 88] = 1
                            destination[
                                header_base
                                + _SLOT_FIXED_HEADER_BYTES
                                + num_planes * _DESCRIPTOR_BYTES
                            ] = T.cast(
                                3 | _ESCAPE_PACKED_FLAG | _ESCAPE_3BIT_FLAG,
                                T.uint8,
                            )
                            stored_length = T.alloc_var(T.int64, init=0)
                            for descriptor_plane in range(num_planes):
                                descriptor_row = slot * num_planes + descriptor_plane
                                descriptor_base = (
                                    header_base
                                    + _SLOT_FIXED_HEADER_BYTES
                                    + descriptor_plane * _DESCRIPTOR_BYTES
                                )
                                relative_offset = (
                                    destination_offsets[descriptor_row] - header_base
                                )
                                payload_length = payload_bytes[descriptor_row]
                                record_length = payload_length
                                stored_length = relative_offset + record_length
                                for byte in range(2):
                                    destination[descriptor_base + byte] = T.cast(
                                        (descriptor_plane // 2 >> (byte * 8)) & 255,
                                        T.uint8,
                                    )
                                    destination[descriptor_base + 2 + byte] = T.cast(
                                        (descriptor_plane % 2 >> (byte * 8)) & 255,
                                        T.uint8,
                                    )
                                for byte in range(4):
                                    destination[descriptor_base + 4 + byte] = T.cast(
                                        (plane_scalars >> (byte * 8)) & 255, T.uint8
                                    )
                                    destination[descriptor_base + 8 + byte] = T.cast(
                                        (max_tiles >> (byte * 8)) & 255, T.uint8
                                    )
                                    destination[descriptor_base + 12 + byte] = T.cast(
                                        (relative_offset >> (byte * 8)) & 255, T.uint8
                                    )
                                    destination[descriptor_base + 16 + byte] = T.cast(
                                        (record_length >> (byte * 8)) & 255, T.uint8
                                    )
                                    destination[descriptor_base + 20 + byte] = T.cast(
                                        (relative_offset >> (byte * 8)) & 255, T.uint8
                                    )
                                    destination[descriptor_base + 24 + byte] = T.cast(
                                        (relative_offset + plane_scalars) >> (byte * 8)
                                        & 255,
                                        T.uint8,
                                    )
                                    destination[descriptor_base + 28 + byte] = T.cast(
                                        (relative_offset + plane_scalars + symbol_bytes)
                                        >> (byte * 8)
                                        & 255,
                                        T.uint8,
                                    )
                                    destination[descriptor_base + 32 + byte] = T.cast(
                                        (relative_offset + base_bytes) >> (byte * 8)
                                        & 255,
                                        T.uint8,
                                    )
                                    escape_count = (
                                        totals[descriptor_row]
                                        if packed_escape
                                        else raw_totals[descriptor_row]
                                    )
                                    destination[descriptor_base + 36 + byte] = T.cast(
                                        escape_count >> (byte * 8) & 255, T.uint8
                                    )
                            for byte in range(8):
                                destination[header_base + 48 + byte] = T.cast(
                                    (
                                        (stored_length + 4095) // 4096 * 4096
                                        >> (byte * 8)
                                    )
                                    & 255,
                                    T.uint8,
                                )
                    destination_base = destination_offsets[row]
                    payload = payload_bytes[row]
                    if payload >= base_bytes:
                        # Only the two prefix tables survive between passes.
                        # Their contiguous row layout must match the prefix
                        # producer; output descriptors still address the
                        # unchanged slot/plane wire payload in destination.
                        scratch_prefix_base = row * scratch_plane_record_bytes
                        scratch_raw_prefix_base = scratch_prefix_base + 4 * (
                            max_tiles + 1
                        )
                        # Match the online packer's three-bit scratch stride.
                        max_escape_code_bytes = (plane_scalars * 3 + 7) // 8
                        max_escape_words = (max_escape_code_bytes + 3) // 4
                        escape_word_base = row * max_escape_words
                        escape_code_bytes = (totals[row] * escape_symbol_bits + 7) // 8
                        if packed_escape:
                            # TileLang requires statically typed loop domains.
                            # Iterate over the maximum packed escape extent and
                            # predicate the runtime tail instead of using the
                            # uint32 total as a range bound.
                            for escape_word_group in range(
                                (max_escape_words + _COMPACT_THREADS - 1)
                                // _COMPACT_THREADS
                            ):
                                escape_word = escape_word_group * _COMPACT_THREADS + tx
                                if escape_word < max_escape_words:
                                    escape_words[escape_word_base + escape_word] = 0
                            T.sync_threads()
                            for escape_group in range(
                                (max_escape_code_bytes + _COMPACT_THREADS - 1)
                                // _COMPACT_THREADS
                            ):
                                escape_index = escape_group * _COMPACT_THREADS + tx
                                if escape_index < escape_code_bytes:
                                    destination[
                                        destination_base + base_bytes + escape_index
                                    ] = 0
                            T.sync_threads()
                        for prefix_group in range(
                            (prefix_bytes + _COMPACT_THREADS - 1) // _COMPACT_THREADS
                        ):
                            prefix_index = prefix_group * _COMPACT_THREADS + tx
                            if prefix_index < prefix_bytes:
                                destination[
                                    destination_base
                                    + plane_scalars
                                    + symbol_bytes
                                    + prefix_index
                                ] = scratch[scratch_prefix_base + prefix_index]
                        # A lane emits eight consecutive scalars. Both
                        # bitstreams preserve scalar order, while their
                        # exclusive warp scans run once per tile.
                        lane_bits = T.alloc_local((8,), "uint16")
                        lane_codes = T.alloc_local((8,), "int32")
                        for tile_group in range(
                            (max_tiles + _COMPACT_WARPS - 1) // _COMPACT_WARPS
                        ):
                            tile = tile_group * _COMPACT_WARPS + warp
                            scalar_start = tile * 256 + lane * 8
                            primary_word = T.alloc_var(T.uint32, init=0)
                            secondary_word = T.alloc_var(T.uint32, init=0)
                            lane_escapes = T.alloc_var(T.int32, init=0)
                            lane_raw = T.alloc_var(T.int32, init=0)
                            for i in T.vectorized(8):
                                lane_bits[i] = (
                                    kv_bits[
                                        block_id,
                                        plane // 2,
                                        (scalar_start + i) // head_dim,
                                        plane % 2,
                                        (scalar_start + i) % head_dim,
                                    ]
                                    if scalar_start + i < plane_scalars
                                    and scalar_start + i < valid_scalars
                                    else 0
                                )
                            for i in T.unroll(8):
                                scalar = scalar_start + i
                                active = scalar < plane_scalars
                                valid = active and scalar < valid_scalars
                                code = (
                                    T.cast(
                                        lookup[plane * 256 + (lane_bits[i] >> 8)],
                                        T.int32,
                                    )
                                    if valid
                                    else 15
                                )
                                lane_codes[i] = code
                                primary_word = primary_word | (
                                    T.cast(
                                        (7 if code >= 7 else code) if active else 0,
                                        T.uint32,
                                    )
                                    << (i * 3)
                                )
                                if active and code >= 7:
                                    secondary_word = secondary_word | (
                                        T.cast(7 if code >= 14 else code - 7, T.uint32)
                                        << (lane_escapes * 3)
                                    )
                                    lane_escapes = lane_escapes + 1
                                    lane_raw = lane_raw + T.cast(code >= 14, T.int32)
                            escape_scan = T.alloc_var(T.int32, init=lane_escapes)
                            raw_scan = T.alloc_var(T.int32, init=lane_raw)
                            for step in T.unroll(5):
                                preceding_escape = T.shfl_up(escape_scan, 1 << step)
                                preceding_raw = T.shfl_up(raw_scan, 1 << step)
                                if lane >= (1 << step):
                                    escape_scan = escape_scan + preceding_escape
                                    raw_scan = raw_scan + preceding_raw
                            tile_prefix = T.alloc_var(T.uint32, init=0)
                            tile_raw_prefix = T.alloc_var(T.uint32, init=0)
                            if tile < max_tiles:
                                for byte in T.unroll(4):
                                    tile_prefix = tile_prefix | (
                                        T.cast(
                                            scratch[
                                                scratch_prefix_base + tile * 4 + byte
                                            ],
                                            T.uint32,
                                        )
                                        << (byte * 8)
                                    )
                                    tile_raw_prefix = tile_raw_prefix | (
                                        T.cast(
                                            scratch[
                                                scratch_raw_prefix_base
                                                + tile * 4
                                                + byte
                                            ],
                                            T.uint32,
                                        )
                                        << (byte * 8)
                                    )
                            for i in T.vectorized(8):
                                if scalar_start + i < plane_scalars:
                                    destination[destination_base + scalar_start + i] = (
                                        T.cast(lane_bits[i], T.uint8)
                                    )
                            for byte in T.unroll(3):
                                symbol_index = scalar_start * 3 // 8 + byte
                                if symbol_index < symbol_bytes:
                                    destination[
                                        destination_base + plane_scalars + symbol_index
                                    ] = T.cast(primary_word >> (byte * 8), T.uint8)
                            if lane_escapes > 0:
                                # Atomic additions own disjoint bit ranges
                                # in the original zeroed scratch words.
                                # Splitting the word preserves that property
                                # even when a lane crosses a uint32 boundary.
                                escape_rank = (
                                    T.cast(tile_prefix, T.int32)
                                    + escape_scan
                                    - lane_escapes
                                )
                                bit_offset = escape_rank * 3
                                word_index = bit_offset // 32
                                word_shift = bit_offset & 31
                                T.atomic_add(
                                    escape_words[escape_word_base + word_index],
                                    secondary_word << word_shift,
                                )
                                if word_shift + lane_escapes * 3 > 32:
                                    T.atomic_add(
                                        escape_words[escape_word_base + word_index + 1],
                                        secondary_word >> (32 - word_shift),
                                    )
                            raw_rank = (
                                T.cast(tile_raw_prefix, T.int32) + raw_scan - lane_raw
                            )
                            local_raw = T.alloc_var(T.int32, init=0)
                            for i in T.unroll(8):
                                if (
                                    scalar_start + i < plane_scalars
                                    and lane_codes[i] >= 14
                                ):
                                    destination[
                                        destination_base
                                        + base_bytes
                                        + escape_code_bytes
                                        + raw_rank
                                        + local_raw
                                    ] = T.cast(lane_bits[i] >> 8, T.uint8)
                                    local_raw = local_raw + 1

                        if packed_escape:
                            # Expand the complete words only after every warp
                            # has emitted its ranks.
                            T.sync_threads()
                            for escape_byte_group in range(
                                (max_escape_code_bytes + _COMPACT_THREADS - 1)
                                // _COMPACT_THREADS
                            ):
                                escape_byte = escape_byte_group * _COMPACT_THREADS + tx
                                if escape_byte < escape_code_bytes:
                                    escape_word = escape_byte // 4
                                    byte_shift = (escape_byte & 3) * 8
                                    destination[
                                        destination_base + base_bytes + escape_byte
                                    ] = T.cast(
                                        escape_words[escape_word_base + escape_word]
                                        >> byte_shift,
                                        T.uint8,
                                    )

    return main


def _build_load(
    *,
    num_blocks: int,
    num_planes: int,
    plane_scalars: int,
    head_dim: int,
    tile_scalars: int,
    tiles_per_plane: int,
    staging_bytes: int,
    fanout: bool,
) -> Any:
    """Decode online 256-scalar tiles with lane-local eight-scalar words.

    Each warp owns a tile, and each lane owns eight consecutive scalars. The
    two scans count preceding lanes' primary and raw escapes, so the
    persisted scalar order is recovered without serial ballot rounds.

    Record fields are read through an ``int32`` view of the staging tensor
    with funnel shifts (``_record_word``); raw records keep byte reads. Each
    CTA reads the plane descriptor once and decodes ``_LOAD_TILE_GROUPS``
    consecutive eight-tile groups. The five words of a group that do not
    depend on the scans (primary symbols, two low-byte words, two tile
    prefixes) are fetched one group ahead, overlapping the next group's loads
    with the current group's dependent reads. When the descriptor offsets
    keep every scan-independent read of the CTA inside the staging tensor,
    those reads skip their clamps; reads addressed by scanned ranks are
    always clamped.

    High bytes are resolved for all eight scalars at once, without a
    per-scalar loop. The 3-bit codes are spread to one nibble each, so a
    code doubles as a PRMT selector into the codebooks packed by
    ``pack_load_codebooks``, whose escape entry decodes to zero. Escaped
    scalars take bytes from an in-order stream instead: secondary codebook
    entries, into which the lane's raw escape bytes (one clamped eight-byte
    window) are gathered the same way. Each gather is a PRMT whose selector
    nibbles are the escape ranks (``_rank_selector``), ORed into the
    codebook bytes. Two more permutes per output word interleave low and
    high bytes. When the head dimension is divisible by eight, each lane
    writes its eight scalars as one 128-bit store; otherwise a lane's
    scalars may straddle heads and are stored one by one.

    The kernel is compiled without TileLang's per-access bounds guards.
    Every global write address derives only from host-validated metadata
    (destination blocks, plane range, grid indices), never from record
    bytes, and reads addressed by record bytes are clamped to the staging
    tensor or proven in range. A corrupt record therefore yields wrong
    scalars in its own destination blocks only.
    """
    vector_stores = head_dim % 8 == 0
    head_words = head_dim // 8
    tile_ctas = (
        (tiles_per_plane + 7) // 8 + _LOAD_TILE_GROUPS - 1
    ) // _LOAD_TILE_GROUPS
    # Tiles addressed by the grid, including the idle tail of the last CTA.
    grid_tiles = tile_ctas * _LOAD_TILE_GROUPS * 8
    # A Qwen3-8B slot is 18,874,368 bytes.  Long restores can exceed 2 GiB
    # while still fitting in the configured staging lease, so shape extents
    # must remain 64-bit; the default ``T.dynamic`` dtype is int32.
    n_slots = T.dynamic("N", "int64")
    n_destinations = T.dynamic("D", "int64")
    # The kernel takes the same staging bytes twice: as bytes and as an int32
    # word view. W is the only staging extent variable (bytes are 4 * W); a
    # second unbound extent would become a stray kernel argument.
    word_count = T.dynamic("W", "int64")
    kv_shape = _vllm_kv_shape(num_blocks, num_planes, plane_scalars, head_dim)
    rows = kv_shape[2]
    kv_words = num_blocks * num_planes * plane_scalars // 8

    @T.macro
    def record_pair(words, word_base, lead, offset, last, out, at):
        # ``out[at]`` and ``out[at + 1]`` get two consecutive record words
        # from three loads; ``last`` clamps as in ``_record_word``.
        total = lead + T.cast(offset, T.uint32)
        first = _clamp(total >> 2, last)
        a = T.cast(words[word_base + first], T.uint32)
        b = T.cast(words[word_base + _clamp(first + 1, last)], T.uint32)
        c = T.cast(words[word_base + _clamp(first + 2, last)], T.uint32)
        out[at] = T.call_extern("uint32", "__funnelshift_r", a, b, (total & 3) * 8)
        out[at + 1] = T.call_extern("uint32", "__funnelshift_r", b, c, (total & 3) * 8)

    @T.macro
    def fetch_group(words, word_base, lead, last, descriptor, tile, lane, out):
        # Scan-independent words of one lane in ``tile``: primary symbols, low
        # bytes [0, 8), and the tile's escape and raw prefixes. Reads past
        # the plane are clamped (or proven in range) and unused.
        scalar_start = tile * 256 + lane * 8
        out[0] = _record_word(
            words,
            word_base,
            lead,
            descriptor[1] + T.cast(scalar_start * 3 // 8, T.uint32),
            last,
        )
        record_pair(
            words,
            word_base,
            lead,
            descriptor[0] + T.cast(scalar_start, T.uint32),
            last,
            out,
            1,
        )
        out[3] = _record_word(
            words,
            word_base,
            lead,
            descriptor[2] + T.cast(tile * 4, T.uint32),
            last,
        )
        out[4] = _record_word(
            words,
            word_base,
            lead,
            descriptor[2] + T.cast((tiles_per_plane + 1 + tile) * 4, T.uint32),
            last,
        )

    @T.macro
    def fetch_group_checked(
        words, word_base, lead, last, unclamped, descriptor, tile, lane, out
    ):
        if unclamped != 0:
            fetch_group(words, word_base, lead, None, descriptor, tile, lane, out)
        else:
            fetch_group(words, word_base, lead, last, descriptor, tile, lane, out)

    @T.prim_func
    def main(
        staging: T.Tensor((word_count * 4,), "uint8"),
        words: T.Tensor((word_count,), "int32"),
        staging_offsets: T.Tensor((n_slots,), "int64"),
        block_ids: T.Tensor((n_destinations,), "int32"),
        destination_offsets: T.Tensor((n_slots + 1,), "int32"),
        modes: T.Tensor((n_slots,), "int32"),
        codebooks: T.Tensor((num_planes * _PACKED_CODEBOOK_BYTES,), "uint8"),
        dst: T.Tensor(kv_shape, "uint16"),
        plane_base: T.int32,
        plane_count: T.int32,
    ):
        with T.Kernel(n_slots, plane_count, tile_ctas, threads=256) as (
            bx,
            by,
            bz,
        ):
            T.import_source(_LOAD_DEVICE_SOURCE)
            if vector_stores:
                # Full occupancy: 32 registers per thread. Per-scalar stores
                # need more and would spill.
                T.annotate_min_blocks_per_sm(8)
            shared_descriptor = T.alloc_shared((5,), "uint32")
            values = T.alloc_local((8,), "uint16")
            tx = T.get_thread_binding()
            lane = tx & 31
            slot_base = staging_offsets[bx]
            mode = modes[bx]
            # Word reads address the record relative to its first word.
            word_base = slot_base // 4
            word_lead = T.cast(slot_base % 4, T.uint32)
            word_last = T.cast(
                T.min(word_count - 1 - word_base, T.cast(0xFFFFFFFF, T.int64)),
                T.uint32,
            )
            global_plane = plane_base + by
            destination_begin = destination_offsets[bx] if fanout else bx
            destination_end = destination_offsets[bx + 1] if fanout else bx + 1
            if mode != 0 and tx < 5:
                address = (
                    slot_base
                    + _SLOT_FIXED_HEADER_BYTES
                    + global_plane * _DESCRIPTOR_BYTES
                    + 20
                    + tx * 4
                )
                shared_descriptor[tx] = (
                    T.cast(staging[address], T.uint32)
                    | (T.cast(staging[address + 1], T.uint32) << 8)
                    | (T.cast(staging[address + 2], T.uint32) << 16)
                    | (T.cast(staging[address + 3], T.uint32) << 24)
                )
            T.sync_threads()
            # Low-byte, symbol, prefix and secondary offsets, then the
            # escape count, held in registers for the whole CTA.
            descriptor = T.alloc_local((5,), "uint32")
            for field in T.unroll(5):
                descriptor[field] = shared_descriptor[field]
            raw_base = descriptor[3] + ((descriptor[4] * 3 + 7) >> 3)
            # Bytes readable from the record start. The scan-independent
            # reads of every grid tile end below these bounds, so a CTA whose
            # descriptor satisfies them reads without clamps.
            room = T.cast(word_last, T.int64) * 4 + 4 - T.cast(word_lead, T.int64)
            unclamped = T.alloc_var(T.int32, init=0)
            if (
                mode != 0
                and T.cast(descriptor[1], T.int64) + grid_tiles * 96 + 8 <= room
                and T.cast(descriptor[0], T.int64) + grid_tiles * 256 + 12 <= room
                and T.cast(descriptor[2], T.int64)
                + (tiles_per_plane + 1 + grid_tiles) * 4
                + 8
                <= room
            ):
                unclamped = 1
            book_words = T.alloc_local((4,), "uint32")
            high_words = T.alloc_local((2,), "uint32")
            # The packed output words alias ``values`` for the vector store.
            out_words = T.view(values, (4,), "uint32")
            packed_books = T.view(codebooks, (num_planes * 4,), "uint32")
            for entry in T.unroll(4):
                book_words[entry] = packed_books[global_plane * 4 + entry]
            # ``ahead`` holds the next group's scan-independent words,
            # ``current`` the words of the group being decoded.
            ahead = T.alloc_local((5,), "uint32")
            current = T.alloc_local((5,), "uint32")
            if mode != 0:
                fetch_group_checked(
                    words,
                    word_base,
                    word_lead,
                    word_last,
                    unclamped,
                    descriptor,
                    bz * _LOAD_TILE_GROUPS * 8 + tx // 32,
                    lane,
                    ahead,
                )
            for group in T.serial(_LOAD_TILE_GROUPS):
                tile = (bz * _LOAD_TILE_GROUPS + group) * 8 + tx // 32
                scalar_start = tile * 256 + lane * 8
                valid = T.min(T.max(plane_scalars - scalar_start, 0), 8)
                if mode == 0:
                    for i in T.unroll(8):
                        raw_address = (
                            slot_base
                            + (global_plane * plane_scalars + scalar_start + i) * 2
                        )
                        values[i] = (
                            T.cast(staging[raw_address], T.uint16)
                            | (T.cast(staging[raw_address + 1], T.uint16) << 8)
                            if i < valid
                            else 0
                        )
                else:
                    for entry in T.unroll(5):
                        current[entry] = ahead[entry]
                    if group + 1 < _LOAD_TILE_GROUPS:
                        fetch_group_checked(
                            words,
                            word_base,
                            word_lead,
                            word_last,
                            unclamped,
                            descriptor,
                            tile + 8,
                            lane,
                            ahead,
                        )
                    primary_word = current[0] & (
                        (T.cast(1, T.uint32) << (valid * 3)) - 1
                    )
                    primary_codes = _spread_codes(primary_word)
                    escapes = _nibble_flags(primary_codes)
                    # Scan indices are signed: subtracting this lane's count
                    # forms an exclusive prefix. Keep packed words unsigned for
                    # logical shifts, but do not mix unsigned rank subtraction
                    # into the compiler's bounds analysis.
                    primary_count = T.cast(T.popcount(escapes), T.int32)
                    primary_scan = T.alloc_var(T.int32, init=primary_count)
                    for step in T.unroll(5):
                        prior = T.shfl_up(primary_scan, 1 << step)
                        if lane >= (1 << step):
                            primary_scan = primary_scan + prior
                    prefix = T.alloc_var(T.uint32, init=0)
                    raw_prefix = T.alloc_var(T.uint32, init=0)
                    # All lanes, including an inactive final warp, participate
                    # in the warp scans. Only real tiles use prefix tables.
                    if tile < tiles_per_plane:
                        prefix = current[3]
                        raw_prefix = current[4]
                    secondary_start = (
                        T.cast(prefix, T.int32) + primary_scan - primary_count
                    )
                    secondary_word = T.alloc_var(T.uint32, init=0)
                    if primary_count > 0:
                        secondary_word = _record_word(
                            words,
                            word_base,
                            word_lead,
                            descriptor[3]
                            + (T.cast(secondary_start, T.uint32) * 3 >> 3),
                            word_last,
                        )
                    secondary_word = secondary_word >> ((secondary_start * 3) & 7)
                    secondary_codes = _spread_codes(secondary_word)
                    raw_flags = _nibble_flags(secondary_codes)
                    # Only the first ``primary_count`` secondary symbols are
                    # this lane's; the funnel shift keeps that many nibbles.
                    secondary_mask = raw_flags & T.call_extern(
                        "uint32",
                        "__funnelshift_rc",
                        T.cast(0xFFFFFFFF, T.uint32),
                        T.cast(0, T.uint32),
                        T.cast(32 - primary_count * 4, T.uint32),
                    )
                    secondary_count = T.cast(T.popcount(secondary_mask), T.int32)
                    secondary_scan = T.alloc_var(T.int32, init=secondary_count)
                    for step in T.unroll(5):
                        prior_raw = T.shfl_up(secondary_scan, 1 << step)
                        if lane >= (1 << step):
                            secondary_scan = secondary_scan + prior_raw
                    raw_start = (
                        T.cast(raw_prefix, T.int32) + secondary_scan - secondary_count
                    )
                    low_first = T.alloc_var(T.uint32, init=0)
                    low_second = T.alloc_var(T.uint32, init=0)
                    if valid > 0:
                        low_first = current[1]
                    if valid > 4:
                        low_second = current[2]
                    raw_window = T.alloc_local((2,), "uint32")
                    raw_window[0] = 0
                    raw_window[1] = 0
                    if secondary_count > 0:
                        record_pair(
                            words,
                            word_base,
                            word_lead,
                            raw_base + T.cast(raw_start, T.uint32),
                            word_last,
                            raw_window,
                            0,
                        )
                    # Rank 7 selects byte 7, which belongs to the stream only
                    # when all eight entries are escapes; otherwise clear it so
                    # unflagged nibbles OR in zero.
                    raw_high = T.Select(
                        secondary_count < 8,
                        raw_window[1] & T.cast(0x00FFFFFF, T.uint32),
                        raw_window[1],
                    )
                    raw_select = _rank_selector(raw_flags)
                    # Secondary-level bytes in stream order: the codebook entry,
                    # or for code 7 (which decodes to zero) the raw byte.
                    stream_low = _prmt(
                        book_words[2], book_words[3], secondary_codes
                    ) | _prmt(raw_window[0], raw_high, raw_select)
                    stream_upper = _prmt(
                        book_words[2], book_words[3], secondary_codes >> 16
                    ) | _prmt(raw_window[0], raw_high, raw_select >> 16)
                    # Likewise byte 7 of the stream for the primary ranks.
                    stream_high = T.Select(
                        primary_count < 8,
                        stream_upper & T.cast(0x00FFFFFF, T.uint32),
                        stream_upper,
                    )
                    escape_select = _rank_selector(escapes)
                    # Primary code 7 decodes to zero, so escaped bytes OR in.
                    high_words[0] = _prmt(
                        book_words[0], book_words[1], primary_codes
                    ) | _prmt(stream_low, stream_high, escape_select)
                    high_words[1] = _prmt(
                        book_words[0], book_words[1], primary_codes >> 16
                    ) | _prmt(stream_low, stream_high, escape_select >> 16)
                    # Interleave four low and four high bytes into two
                    # little-endian BF16 pairs per output word.
                    out_words[0] = _prmt(low_first, high_words[0], 0x5140)
                    out_words[1] = _prmt(low_first, high_words[0], 0x7362)
                    out_words[2] = _prmt(low_second, high_words[1], 0x5140)
                    out_words[3] = _prmt(low_second, high_words[1], 0x7362)
                if vector_stores:
                    # Lane word ``word`` holds plane scalars [8 * word, 8 * word
                    # + 8), which never straddle a head as head_dim % 8 == 0.
                    word = tile * 32 + lane
                    # Alias the cache directly: T.view checks the bit count as
                    # an int32 literal, which a production cache exceeds.
                    flat = T.Tensor((T.int64(kv_words),), "uint32x4", dst.data)
                    packed = T.view(values, (1,), "uint32x4")
                    for target in range(destination_begin, destination_end):
                        if scalar_start < plane_scalars:
                            row = (
                                T.cast(block_ids[target], T.int64) * (num_planes // 2)
                                + global_plane // 2
                            ) * rows + (word // head_words)
                            flat[
                                (row * 2 + global_plane % 2) * head_words
                                + word % head_words
                            ] = packed[0]
                else:
                    for target in range(destination_begin, destination_end):
                        for i in T.unroll(8):
                            if i < valid:
                                dst[
                                    block_ids[target],
                                    global_plane // 2,
                                    (scalar_start + i) // head_dim,
                                    global_plane % 2,
                                    (scalar_start + i) % head_dim,
                                ] = values[i]

    return main


def clear_tilelang_codec_cache() -> None:
    """Clear process-local TileLang codec kernels for tests.

    Returns:
        None.

    Async/thread-safety:
        Call only while no worker can launch a codec kernel.
    """
    _STORE_CACHE.clear()
    _LOAD_CACHE.clear()


__all__ = [
    "TileLangStoreKernels",
    "clear_tilelang_codec_cache",
    "compile_load_kernel",
    "compile_store_kernels",
    "pack_load_codebooks",
]
