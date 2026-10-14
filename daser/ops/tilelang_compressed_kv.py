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

from daser.compression.format import IO_ALIGNMENT, slot_header_bytes

_COUNT_THREADS = 256
_COUNT_WARPS = _COUNT_THREADS // 32
# Compact assigns one warp to each 1024-scalar tile.
_COMPACT_THREADS = 256
_COMPACT_WARPS = _COMPACT_THREADS // 32
# Decode uses fewer threads because each escape ballot is warp-local.  Four
# warps cover one tile through multiple scalar groups while reducing block-wide
# barrier and register overhead on the load critical path.
_LOAD_THREADS = 64
_SLOT_FIXED_HEADER_BYTES = 120
_DESCRIPTOR_BYTES = 48
_CODEBOOK_ENTRIES = 15
# The word-parallel decoder pads each seven-entry codebook level to eight
# bytes so a level fits one PRMT byte pair (see ``pack_load_codebooks``).
_PACKED_CODEBOOK_BYTES = 16
_WORD_PARALLEL_TILE_SCALARS = 256
# Eight-tile groups decoded per word-parallel CTA; 32 measured fastest
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


def _byte_perm(low: Any, high: Any, selector: Any) -> Any:
    """Return the CUDA ``__byte_perm`` of the eight bytes ``high:low``.

    Each selector nibble picks one byte of the pair for the result byte at
    the same position (byte indices 0-7); the instruction is one PRMT.
    """
    return T.call_extern("uint32", "__byte_perm", low, high, T.cast(selector, T.uint32))


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
    """
    total = lead + T.cast(offset, T.uint32)
    first = T.min(total >> 2, last)
    second = T.min(first + 1, last)
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
        tile_scalars: Scalar quantum used by the packed record.

    Returns:
        Cached fused encode, layout, and compact kernels.

    Raises:
        ValueError: If the geometry is not positive or exceeds the supported
            one-tile shared-memory envelope.

    Async/thread-safety:
        Compilation is synchronous and intended for registration-time
        warmup. The process-local cache is only mutated during that phase.
    """
    if min(num_blocks, num_planes, plane_scalars, tile_scalars) <= 0:
        raise ValueError("TileLang store geometry must be positive")
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


def load_reads_words(tile_scalars: int) -> bool:
    """Return whether the load kernel for ``tile_scalars`` takes a word view.

    Args:
        tile_scalars (int): Codec tile size of the decoded records.

    Returns:
        bool: True for the 256-scalar word-parallel decoder, which takes the
        staging bytes a second time as an ``int32`` view and therefore needs
        4-byte aligned staging whose length is a multiple of four.
    """
    return tile_scalars == _WORD_PARALLEL_TILE_SCALARS


def pack_load_codebooks(codebooks: bytes, num_planes: int, tile_scalars: int) -> bytes:
    """Lay out per-plane codebooks for the load kernel of ``tile_scalars``.

    Args:
        codebooks (bytes): ``num_planes`` rows of ``CODEBOOK_ENTRIES`` bytes:
            seven primary entries followed by the secondary entries.
        num_planes (int): Number of layer/K-or-V planes.
        tile_scalars (int): Codec tile size of the decoded records.

    Returns:
        bytes: ``codebooks`` unchanged for the generic decoder. For the
        256-scalar word-parallel decoder, ``num_planes`` rows of 16 bytes:
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
    if not load_reads_words(tile_scalars):
        return bytes(codebooks)
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
        tile_scalars: Scalar quantum encoded in each record. 256-scalar tiles
            use the word-parallel decoder, whose launch takes the staging
            bytes plus their ``int32`` view (see ``load_reads_words``) and
            codebooks laid out by ``pack_load_codebooks``.
        staging_bytes: Representative staging capacity used for warmup metadata;
            the compiled kernel accepts the actual extent at runtime.
        fanout: Whether sources may have multiple destination blocks. Both
            variants are compiled at startup; fanout counts remain dynamic.

    Returns:
        Cached TileLang JIT kernel. Slot count and staging extent remain runtime
        dimensions, so request metadata does not trigger compilation.

    Raises:
        ValueError: If geometry is not positive or the tile count is too
            large.

    Async/thread-safety:
        Compilation is synchronous and should happen before request traffic.
        The returned kernel launches asynchronously on the current PyTorch
        stream.
    """
    if min(num_blocks, num_planes, plane_scalars, tile_scalars, staging_bytes) <= 0:
        raise ValueError("TileLang load geometry must be positive")
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
    word_parallel = load_reads_words(tile_scalars)
    builder = _build_wordparallel_load if word_parallel else _build_load
    kernel = tilelang.compile(
        builder(
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
        pass_configs=(
            {PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True}
            if word_parallel
            else None
        ),
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
                        if tile_scalars == 256:
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
                                            T.cast(
                                                7 if code >= 14 else code - 7, T.uint32
                                            )
                                            << (lane_escapes * 3)
                                        )
                                        lane_escapes = lane_escapes + 1
                                        lane_raw = lane_raw + T.cast(
                                            code >= 14, T.int32
                                        )
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
                                                    scratch_prefix_base
                                                    + tile * 4
                                                    + byte
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
                                        destination[
                                            destination_base + scalar_start + i
                                        ] = T.cast(lane_bits[i], T.uint8)
                                for byte in T.unroll(3):
                                    symbol_index = scalar_start * 3 // 8 + byte
                                    if symbol_index < symbol_bytes:
                                        destination[
                                            destination_base
                                            + plane_scalars
                                            + symbol_index
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
                                            escape_words[
                                                escape_word_base + word_index + 1
                                            ],
                                            secondary_word >> (32 - word_shift),
                                        )
                                raw_rank = (
                                    T.cast(tile_raw_prefix, T.int32)
                                    + raw_scan
                                    - lane_raw
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
                        else:
                            for tile_group in range(
                                (max_tiles + _COMPACT_WARPS - 1) // _COMPACT_WARPS
                            ):
                                tile = tile_group * _COMPACT_WARPS + warp
                                tile_begin = tile * tile_scalars
                                # Each lane scans the same tile-local scalar groups
                                # in lockstep. Keeping the preceding-group escape
                                # count in a lane-local accumulator avoids a
                                # block-wide prefix or a shuffle dependency in the
                                # emit pass.
                                escape_before = T.alloc_var(T.uint32, init=0)
                                raw_escape_before = T.alloc_var(T.uint32, init=0)
                                for scalar_group in range((tile_scalars + 31) // 32):
                                    scalar = tile_begin + scalar_group * 32 + lane
                                    active = (
                                        tile < max_tiles
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
                                        T.cast(lookup[plane * 256 + high], T.uint8)
                                        if valid
                                        else 15
                                    )
                                    if active:
                                        destination[destination_base + scalar] = T.cast(
                                            bits & 255, T.uint8
                                        )
                                    if tile_scalars % 32 == 0:
                                        # Eight adjacent 3-bit codes form exactly
                                        # three bytes. Combine the codes in registers
                                        # while the KV scalar is already live, so
                                        # primary emission needs no second KV scan.
                                        primary_code = (
                                            T.cast(7 if code >= 7 else code, T.uint32)
                                            if active
                                            else T.cast(0, T.uint32)
                                        )
                                        word = T.alloc_var(
                                            T.uint32,
                                            init=primary_code << ((lane & 7) * 3),
                                        )
                                        for reduction in range(3):
                                            word = word | T.shfl_xor(
                                                word, 4 >> reduction, width=8
                                            )
                                        symbol_index = (
                                            (scalar - lane) * 3 // 8
                                            + (lane // 8) * 3
                                            + (lane & 7)
                                        )
                                        if (
                                            lane & 7
                                        ) < 3 and symbol_index < symbol_bytes:
                                            destination[
                                                destination_base
                                                + plane_scalars
                                                + symbol_index
                                            ] = T.cast(
                                                word >> ((lane & 7) * 8), T.uint8
                                            )
                                    mask = T.ballot(active and code >= 7)
                                    raw_mask = T.ballot(active and code >= 14)
                                    prefix_value = (
                                        (
                                            T.cast(
                                                scratch[scratch_prefix_base + tile * 4],
                                                T.uint32,
                                            )
                                            | (
                                                T.cast(
                                                    scratch[
                                                        scratch_prefix_base
                                                        + tile * 4
                                                        + 1
                                                    ],
                                                    T.uint32,
                                                )
                                                << 8
                                            )
                                            | (
                                                T.cast(
                                                    scratch[
                                                        scratch_prefix_base
                                                        + tile * 4
                                                        + 2
                                                    ],
                                                    T.uint32,
                                                )
                                                << 16
                                            )
                                            | (
                                                T.cast(
                                                    scratch[
                                                        scratch_prefix_base
                                                        + tile * 4
                                                        + 3
                                                    ],
                                                    T.uint32,
                                                )
                                                << 24
                                            )
                                        )
                                        if tile < max_tiles
                                        else 0
                                    )
                                    raw_prefix_value = (
                                        (
                                            T.cast(
                                                scratch[
                                                    scratch_raw_prefix_base + tile * 4
                                                ],
                                                T.uint32,
                                            )
                                            | (
                                                T.cast(
                                                    scratch[
                                                        scratch_raw_prefix_base
                                                        + tile * 4
                                                        + 1
                                                    ],
                                                    T.uint32,
                                                )
                                                << 8
                                            )
                                            | (
                                                T.cast(
                                                    scratch[
                                                        scratch_raw_prefix_base
                                                        + tile * 4
                                                        + 2
                                                    ],
                                                    T.uint32,
                                                )
                                                << 16
                                            )
                                            | (
                                                T.cast(
                                                    scratch[
                                                        scratch_raw_prefix_base
                                                        + tile * 4
                                                        + 3
                                                    ],
                                                    T.uint32,
                                                )
                                                << 24
                                            )
                                        )
                                        if tile < max_tiles
                                        else 0
                                    )
                                    if active and code >= 7:
                                        lower_mask = mask & (
                                            T.cast(0xFFFFFFFF, T.uint32) >> (lane ^ 31)
                                        )
                                        current_bit = T.cast(
                                            (mask >> lane) & 1, T.uint32
                                        )
                                        rank = (
                                            prefix_value
                                            + escape_before
                                            + T.popcount(lower_mask)
                                            - current_bit
                                        )
                                        if packed_escape:
                                            raw_sentinel = 7
                                            escape_code = (
                                                code - 7 if code < 14 else raw_sentinel
                                            )
                                            escape_bit_offset = (
                                                rank * escape_symbol_bits
                                            )
                                            escape_byte = escape_bit_offset // 8
                                            escape_word = escape_bit_offset // 32
                                            escape_word_shift = escape_bit_offset & 31
                                            # Escape words are cleared before the
                                            # tile scan. Atomic additions are safe
                                            # because adjacent 3-bit tokens occupy
                                            # disjoint bit ranges; the final copy
                                            # emits only the required byte extent.
                                            T.atomic_add(
                                                escape_words[
                                                    escape_word_base + escape_word
                                                ],
                                                T.cast(escape_code, T.uint32)
                                                << escape_word_shift,
                                            )
                                            if escape_word_shift > 29:
                                                T.atomic_add(
                                                    escape_words[
                                                        escape_word_base
                                                        + escape_word
                                                        + 1
                                                    ],
                                                    T.cast(escape_code, T.uint32)
                                                    >> (32 - escape_word_shift),
                                                )
                                            if code >= 14:
                                                raw_lower_mask = raw_mask & (
                                                    T.cast(0xFFFFFFFF, T.uint32)
                                                    >> (lane ^ 31)
                                                )
                                                raw_current_bit = T.cast(
                                                    (raw_mask >> lane) & 1, T.uint32
                                                )
                                                raw_rank = (
                                                    raw_prefix_value
                                                    + raw_escape_before
                                                    + T.popcount(raw_lower_mask)
                                                    - raw_current_bit
                                                )
                                                destination[
                                                    destination_base
                                                    + base_bytes
                                                    + escape_code_bytes
                                                    + raw_rank
                                                ] = T.cast(bits >> 8, T.uint8)
                                        else:
                                            destination[
                                                destination_base + base_bytes + rank
                                            ] = T.cast(bits >> 8, T.uint8)
                                    group_count = T.popcount(mask)
                                    escape_before = escape_before + group_count
                                    raw_escape_before = raw_escape_before + T.popcount(
                                        raw_mask
                                    )

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

                    # Non-aligned tiles can share a symbol byte at their
                    # boundary, so retain one writer per byte for those shapes.
                    if tile_scalars % 32 != 0:
                        symbol_bytes_3 = (plane_scalars * 3 + 7) // 8
                        for byte_group in range(
                            (symbol_bytes_3 + _COMPACT_THREADS - 1) // _COMPACT_THREADS
                        ):
                            symbol_index = byte_group * _COMPACT_THREADS + tx
                            if symbol_index < symbol_bytes_3:
                                first_scalar = symbol_index * 8 // 3
                                byte_begin = symbol_index * 8
                                packed = T.alloc_var(T.uint32, init=0)
                                for code_index in range(4):
                                    scalar = first_scalar + code_index
                                    if scalar < plane_scalars:
                                        valid = scalar < valid_scalars
                                        bits = (
                                            kv_bits[
                                                block_id,
                                                plane // 2,
                                                scalar // head_dim,
                                                plane % 2,
                                                scalar % head_dim,
                                            ]
                                            if valid
                                            else 0
                                        )
                                        high = (bits >> 8) & 255
                                        code = T.alloc_var(T.uint32, init=7)
                                        if valid:
                                            code = T.cast(
                                                lookup[plane * 256 + high], T.uint32
                                            )
                                        code = 7 if code >= 7 else code
                                        for code_bit in range(3):
                                            global_bit = scalar * 3 + code_bit
                                            if (
                                                global_bit >= byte_begin
                                                and global_bit < byte_begin + 8
                                            ):
                                                packed = packed | (
                                                    ((code >> code_bit) & 1)
                                                    << (global_bit - byte_begin)
                                                )
                                destination[
                                    destination_base + plane_scalars + symbol_index
                                ] = T.cast(packed & 255, T.uint8)

    return main


def _build_wordparallel_load(
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
    with the current group's dependent reads.

    High bytes resolve without divergent branches from codebooks packed by
    ``pack_load_codebooks``: each lane loads its (fewer than eight) raw
    escape bytes as one clamped eight-byte window, then per scalar selects
    among three byte permutes (primary codebook, secondary codebook, raw
    window). Two more permutes per output word interleave low and high
    bytes. When the head dimension is divisible by eight, each lane writes
    its eight scalars as one 128-bit store; otherwise a lane's scalars may
    straddle heads and are stored one by one.

    The kernel is compiled without TileLang's per-access bounds guards.
    Every global write address derives only from host-validated metadata
    (destination blocks, plane range, grid indices), never from record
    bytes, and reads addressed by record bytes are clamped to the staging
    tensor. A corrupt record therefore yields wrong scalars in its own
    destination blocks only.
    """
    vector_stores = head_dim % 8 == 0
    head_words = head_dim // 8
    tile_ctas = (
        (tiles_per_plane + 7) // 8 + _LOAD_TILE_GROUPS - 1
    ) // _LOAD_TILE_GROUPS
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
    def fetch_group(words, word_base, lead, last, descriptor, tile, lane, out):
        # Scan-independent words of one lane in ``tile``: primary symbols, low
        # bytes [0, 4) and [4, 8), and the tile's escape and raw prefixes.
        # Reads past the plane are clamped by ``_record_word`` and unused.
        scalar_start = tile * 256 + lane * 8
        out[0] = _record_word(
            words,
            word_base,
            lead,
            descriptor[1] + T.cast(scalar_start * 3 // 8, T.uint32),
            last,
        )
        out[1] = _record_word(
            words, word_base, lead, descriptor[0] + T.cast(scalar_start, T.uint32), last
        )
        out[2] = _record_word(
            words,
            word_base,
            lead,
            descriptor[0] + T.cast(scalar_start + 4, T.uint32),
            last,
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
            T.annotate_min_blocks_per_sm(8)
            descriptor = T.alloc_shared((5,), "uint32")
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
                descriptor[tx] = (
                    T.cast(staging[address], T.uint32)
                    | (T.cast(staging[address + 1], T.uint32) << 8)
                    | (T.cast(staging[address + 2], T.uint32) << 16)
                    | (T.cast(staging[address + 3], T.uint32) << 24)
                )
            T.sync_threads()
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
                fetch_group(
                    words,
                    word_base,
                    word_lead,
                    word_last,
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
                        fetch_group(
                            words,
                            word_base,
                            word_lead,
                            word_last,
                            descriptor,
                            tile + 8,
                            lane,
                            ahead,
                        )
                    high_words[0] = 0
                    high_words[1] = 0
                    primary_word = current[0] & (
                        (T.cast(1, T.uint32) << (valid * 3)) - 1
                    )
                    # In each three-bit symbol, the low bit of 111 survives this
                    # intersection. Spaced bits avoid carries between symbols.
                    primary_mask = (
                        primary_word
                        & (primary_word >> 1)
                        & (primary_word >> 2)
                        & T.cast(0x249249, T.uint32)
                    )
                    # Scan indices are signed: subtracting this lane's count
                    # forms an exclusive prefix. Keep packed words unsigned for
                    # logical shifts, but do not mix unsigned rank subtraction
                    # into the compiler's bounds analysis.
                    primary_count = T.cast(T.popcount(primary_mask), T.int32)
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
                    secondary_mask = (
                        secondary_word
                        & (secondary_word >> 1)
                        & (secondary_word >> 2)
                        & T.cast(0x249249, T.uint32)
                        & ((T.cast(1, T.uint32) << (primary_count * 3)) - 1)
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
                    # Walk the lane's symbols with running counters: ``pending``
                    # shifts to the next secondary symbol after each escape,
                    # and ``raw_rank`` counts earlier raw escapes.
                    raw_window = T.alloc_local((2,), "uint32")
                    raw_window[0] = 0
                    raw_window[1] = 0
                    if secondary_count > 0:
                        raw_offset = (
                            descriptor[3]
                            + ((descriptor[4] * 3 + 7) >> 3)
                            + T.cast(raw_start, T.uint32)
                        )
                        raw_window[0] = _record_word(
                            words, word_base, word_lead, raw_offset, word_last
                        )
                        raw_window[1] = _record_word(
                            words, word_base, word_lead, raw_offset + 4, word_last
                        )
                    pending = T.alloc_var(T.uint32, init=secondary_word)
                    raw_rank = T.alloc_var(T.uint32, init=0)
                    for i in T.unroll(8):
                        code = (primary_word >> (i * 3)) & T.cast(7, T.uint32)
                        secondary = pending & T.cast(7, T.uint32)
                        escaped = code == T.cast(7, T.uint32)
                        raw = T.And(escaped, secondary == T.cast(7, T.uint32))
                        # All three candidates are computed and selected so
                        # the loop has no divergent branches.
                        from_raw = _byte_perm(raw_window[0], raw_window[1], raw_rank)
                        from_secondary = _byte_perm(
                            book_words[2], book_words[3], secondary
                        )
                        from_primary = _byte_perm(book_words[0], book_words[1], code)
                        high = T.Select(
                            escaped,
                            T.Select(raw, from_raw, from_secondary),
                            from_primary,
                        )
                        # Insert byte 0 of ``high`` at byte i % 4.
                        high_words[i // 4] = _byte_perm(
                            high_words[i // 4],
                            high,
                            (0x3210 & ~(0xF << ((i % 4) * 4))) | (0x4 << ((i % 4) * 4)),
                        )
                        pending = T.Select(escaped, pending >> 3, pending)
                        raw_rank = raw_rank + T.cast(raw, T.uint32)
                    # Interleave four low and four high bytes into two
                    # little-endian BF16 pairs per output word.
                    out_words[0] = _byte_perm(low_first, high_words[0], 0x5140)
                    out_words[1] = _byte_perm(low_first, high_words[0], 0x7362)
                    out_words[2] = _byte_perm(low_second, high_words[1], 0x5140)
                    out_words[3] = _byte_perm(low_second, high_words[1], 0x7362)
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
    """Decode each source once and scatter directly to its destination group."""
    # Keep dynamic extents 64-bit: the staging tensor is a byte extent and
    # may exceed 2 GiB for a long Qwen3-8B restore.  Using the default int32
    # shape silently invalidates address bounds after that boundary.
    n_slots = T.dynamic("N", "int64")
    n_destinations = T.dynamic("D", "int64")
    staging_extent = T.dynamic("S", "int64")
    # A fine rank index need not imply a fine CTA grid. Indexed tiles share
    # a CTA while each warp consumes one tile without rescanning ranks.
    warp_tiles = tile_scalars <= 256 and tile_scalars % 32 == 0
    load_threads = 256 if warp_tiles else _LOAD_THREADS
    load_warps = load_threads // 32
    warp_local = warp_tiles or tile_scalars % load_threads == 0
    grid_tiles = (
        (tiles_per_plane + load_warps - 1) // load_warps
        if warp_tiles
        else tiles_per_plane
    )
    group_threads = 32 if warp_tiles else load_threads

    @T.prim_func
    def main(
        staging: T.Tensor((staging_extent,), "uint8"),
        staging_offsets: T.Tensor((n_slots,), "int64"),
        block_ids: T.Tensor((n_destinations,), "int32"),
        destination_offsets: T.Tensor((n_slots + 1,), "int32"),
        modes: T.Tensor((n_slots,), "int32"),
        codebooks: T.Tensor((num_planes * _CODEBOOK_ENTRIES,), "uint8"),
        dst: T.Tensor(
            _vllm_kv_shape(num_blocks, num_planes, plane_scalars, head_dim), "uint16"
        ),
        plane_base: T.int32,
        plane_count: T.int32,
    ):
        with T.Kernel(n_slots, plane_count, grid_tiles, threads=load_threads) as (
            bx,
            by,
            bz,
        ):
            warp_counts = T.alloc_shared((load_warps,), "uint32")
            warp_prefix = T.alloc_shared((load_warps,), "uint32")
            warp_raw_counts = T.alloc_shared((load_warps,), "uint32")
            warp_raw_prefix = T.alloc_shared((load_warps,), "uint32")
            descriptor_offsets = T.alloc_shared((4,), "uint32")
            escape_count_shared = T.alloc_shared((1,), "uint32")
            tx = T.get_thread_binding()
            slot = bx
            plane = plane_base + by
            lane = tx & 31
            warp = tx // 32
            tile = bz * load_warps + warp if warp_tiles else bz
            slot_base = staging_offsets[slot]
            destination_begin = destination_offsets[slot] if fanout else slot
            destination_end = destination_offsets[slot + 1] if fanout else slot + 1
            block_id = block_ids[destination_begin]
            mode = modes[slot]
            if mode != 0 and tx < 4:
                descriptor = (
                    slot_base + _SLOT_FIXED_HEADER_BYTES + plane * _DESCRIPTOR_BYTES
                )
                descriptor_offsets[0] = (
                    T.cast(staging[descriptor + 20], T.uint32)
                    | (T.cast(staging[descriptor + 21], T.uint32) << 8)
                    | (T.cast(staging[descriptor + 22], T.uint32) << 16)
                    | (T.cast(staging[descriptor + 23], T.uint32) << 24)
                )
                descriptor_offsets[1] = (
                    T.cast(staging[descriptor + 24], T.uint32)
                    | (T.cast(staging[descriptor + 25], T.uint32) << 8)
                    | (T.cast(staging[descriptor + 26], T.uint32) << 16)
                    | (T.cast(staging[descriptor + 27], T.uint32) << 24)
                )
                descriptor_offsets[2] = (
                    T.cast(staging[descriptor + 28], T.uint32)
                    | (T.cast(staging[descriptor + 29], T.uint32) << 8)
                    | (T.cast(staging[descriptor + 30], T.uint32) << 16)
                    | (T.cast(staging[descriptor + 31], T.uint32) << 24)
                )
                descriptor_offsets[3] = (
                    T.cast(staging[descriptor + 32], T.uint32)
                    | (T.cast(staging[descriptor + 33], T.uint32) << 8)
                    | (T.cast(staging[descriptor + 34], T.uint32) << 16)
                    | (T.cast(staging[descriptor + 35], T.uint32) << 24)
                )
                if tx == 0:
                    escape_count_shared[0] = (
                        T.cast(staging[descriptor + 36], T.uint32)
                        | (T.cast(staging[descriptor + 37], T.uint32) << 8)
                        | (T.cast(staging[descriptor + 38], T.uint32) << 16)
                        | (T.cast(staging[descriptor + 39], T.uint32) << 24)
                    )
            T.sync_threads()
            low_offset = descriptor_offsets[0] if mode != 0 else 0
            symbol_offset = descriptor_offsets[1] if mode != 0 else 0
            prefix_offset = descriptor_offsets[2] if mode != 0 else 0
            escape_offset = descriptor_offsets[3] if mode != 0 else 0
            prefix_value = T.alloc_var(T.uint32, init=0)
            raw_prefix_value = T.alloc_var(T.uint32, init=0)
            if mode != 0 and tile < tiles_per_plane:
                prefix_base = slot_base + prefix_offset
                prefix_value = (
                    T.cast(staging[prefix_base + tile * 4], T.uint32)
                    | (T.cast(staging[prefix_base + tile * 4 + 1], T.uint32) << 8)
                    | (T.cast(staging[prefix_base + tile * 4 + 2], T.uint32) << 16)
                    | (T.cast(staging[prefix_base + tile * 4 + 3], T.uint32) << 24)
                )
                raw_prefix_base = prefix_base + 4 * (tiles_per_plane + 1)
                raw_prefix_value = (
                    T.cast(staging[raw_prefix_base + tile * 4], T.uint32)
                    | (T.cast(staging[raw_prefix_base + tile * 4 + 1], T.uint32) << 8)
                    | (T.cast(staging[raw_prefix_base + tile * 4 + 2], T.uint32) << 16)
                    | (T.cast(staging[raw_prefix_base + tile * 4 + 3], T.uint32) << 24)
                )
            escape_group_prefix = T.alloc_var(T.uint32, init=0)
            raw_escape_group_prefix = T.alloc_var(T.uint32, init=0)
            if warp_local and not warp_tiles and mode != 0:
                # Assign each warp a contiguous quarter tile. Reconstruct
                # its starting rank from the existing three-bit streams;
                # this trades bounded redundant reads for the four CTA
                # barriers previously required by every scalar group.
                # Eight symbols fit in uint32 even at a seven-bit offset.
                for rank_pass in T.unroll(2):
                    rank_start = tile * tile_scalars if rank_pass == 0 else prefix_value
                    rank_length = (
                        T.min(
                            warp * (tile_scalars // load_warps),
                            plane_scalars - tile * tile_scalars,
                        )
                        if rank_pass == 0
                        else escape_group_prefix
                    )
                    rank_stream = symbol_offset if rank_pass == 0 else escape_offset
                    rank_total = T.alloc_var(T.uint32, init=0)
                    for rank_group in range((tile_scalars * 3 // 4 + 255) // 256):
                        rank_first = (rank_group * 32 + lane) * 8
                        rank_valid = T.min(
                            T.max(T.cast(rank_length, T.int32) - rank_first, 0), 8
                        )
                        rank_bit = (rank_start + rank_first) * 3
                        rank_shift = rank_bit & 7
                        rank_word = T.alloc_var(T.uint32, init=0)
                        for rank_byte in T.unroll(4):
                            if rank_valid > 0 and rank_byte * 8 < (
                                rank_shift + rank_valid * 3
                            ):
                                rank_word = rank_word | (
                                    T.cast(
                                        staging[
                                            slot_base
                                            + rank_stream
                                            + rank_bit // 8
                                            + rank_byte
                                        ],
                                        T.uint32,
                                    )
                                    << (rank_byte * 8)
                                )
                        rank_word = rank_word >> rank_shift
                        rank_mask = T.cast(0x00249249, T.uint32) & (
                            (T.cast(1, T.uint32) << (rank_valid * 3)) - 1
                        )
                        rank_total = rank_total + T.popcount(
                            rank_word & (rank_word >> 1) & (rank_word >> 2) & rank_mask
                        )
                    # All lanes participate, including those past the
                    # scanned prefix. Never shuffle in a partial-lane branch.
                    for rank_delta in T.unroll(5):
                        rank_total = rank_total + T.shfl_xor(
                            rank_total, 1 << rank_delta
                        )
                    if rank_pass == 0:
                        escape_group_prefix = rank_total
                    else:
                        raw_escape_group_prefix = rank_total
            for scalar_group in range(
                (tile_scalars + group_threads - 1) // group_threads
            ):
                scalar = (
                    tile * tile_scalars + scalar_group * 32 + lane
                    if warp_tiles
                    else tile * tile_scalars
                    + warp * (tile_scalars // load_warps)
                    + scalar_group * 32
                    + lane
                    if warp_local
                    else tile * tile_scalars + scalar_group * load_threads + tx
                )
                # Adjacent non-aligned tiles must never write the same scalar,
                # including the raw path where identical writes still race.
                active = scalar < plane_scalars and scalar < (tile + 1) * tile_scalars
                if mode == 0:
                    byte_base = slot_base + plane * plane_scalars * 2 + scalar * 2
                    raw_low = T.cast(staging[byte_base], T.uint16) if active else 0
                    raw_high = T.cast(staging[byte_base + 1], T.uint16) if active else 0
                    if active:
                        if fanout:
                            for target in range(destination_begin, destination_end):
                                dst[
                                    block_ids[target],
                                    plane // 2,
                                    scalar // head_dim,
                                    plane % 2,
                                    scalar % head_dim,
                                ] = raw_low | (raw_high << 8)
                        else:
                            dst[
                                block_id,
                                plane // 2,
                                scalar // head_dim,
                                plane % 2,
                                scalar % head_dim,
                            ] = raw_low | (raw_high << 8)
                    continue
                # Online records always use a three-bit main stream whose
                # all-ones symbol marks an escape. Keep both as uint32 vars:
                # folding them into constants makes TVM emit negative
                # unsigned literals while simplifying the escape predicates.
                code = T.alloc_var(T.uint32, init=0)
                escape_code = T.alloc_var(T.uint32, init=7)
                bit_offset = scalar * 3
                symbol_byte = bit_offset // 8
                symbol_shift = bit_offset & 7
                packed = (
                    T.cast(
                        staging[slot_base + symbol_offset + symbol_byte],
                        T.uint32,
                    )
                    | (
                        T.cast(
                            staging[slot_base + symbol_offset + symbol_byte + 1],
                            T.uint32,
                        )
                        << 8
                    )
                    if active
                    else 0
                )
                code = (packed >> symbol_shift) & 7
                mask = T.ballot(active and code == escape_code)
                if not warp_local:
                    if lane == 0:
                        warp_counts[warp] = T.popcount(mask)
                    T.sync_threads()
                    if tx == 0:
                        running = T.alloc_var(T.uint32, init=0)
                        for index in range(load_warps):
                            warp_prefix[index] = running
                            running = running + warp_counts[index]
                    T.sync_threads()
                token_rank = (
                    prefix_value
                    + escape_group_prefix
                    + (0 if warp_local else warp_prefix[warp])
                    + T.popcount(mask & ((1 << lane) - 1))
                )
                packed_escape_value = T.alloc_var(T.uint32, init=15)
                if code == escape_code and active:
                    escape_bit_offset = token_rank * 3
                    escape_byte = escape_bit_offset // 8
                    escape_shift = escape_bit_offset & 7
                    escape_word = T.alloc_var(T.uint32, init=0)
                    escape_word = T.cast(
                        staging[slot_base + escape_offset + escape_byte],
                        T.uint32,
                    )
                    escape_word = escape_word | (
                        T.cast(
                            staging[slot_base + escape_offset + escape_byte + 1],
                            T.uint32,
                        )
                        << 8
                    )
                    packed_escape_value = (escape_word >> escape_shift) & 7
                raw_mask = T.ballot(
                    code == escape_code and packed_escape_value == 7 and active
                )
                if not warp_local:
                    if lane == 0:
                        warp_raw_counts[warp] = T.popcount(raw_mask)
                    T.sync_threads()
                    if tx == 0:
                        raw_running = T.alloc_var(T.uint32, init=0)
                        for index in range(load_warps):
                            warp_raw_prefix[index] = raw_running
                            raw_running = raw_running + warp_raw_counts[index]
                    T.sync_threads()
                raw_rank = (
                    raw_prefix_value
                    + raw_escape_group_prefix
                    + (0 if warp_local else warp_raw_prefix[warp])
                    + T.popcount(raw_mask & ((1 << lane) - 1))
                )
                escape_code_bytes = (escape_count_shared[0] * 3 + 7) // 8
                decoded_high = T.if_then_else(
                    code == escape_code and packed_escape_value == 7 and active,
                    T.cast(
                        staging[
                            slot_base + escape_offset + escape_code_bytes + raw_rank
                        ],
                        T.uint16,
                    ),
                    T.if_then_else(
                        code == escape_code and active,
                        T.cast(
                            codebooks[
                                plane * _CODEBOOK_ENTRIES + 7 + packed_escape_value
                            ],
                            T.uint16,
                        ),
                        T.cast(
                            codebooks[plane * _CODEBOOK_ENTRIES + code],
                            T.uint16,
                        ),
                    ),
                )
                decoded_low = (
                    T.cast(staging[slot_base + low_offset + scalar], T.uint16)
                    if active
                    else 0
                )
                # The transfer planner already deduplicates immutable
                # sources. Reuse the decoded scalar across all consumers
                # instead of repeating both escape scans for each target.
                # Each destination block belongs to exactly one source;
                # no intermediate raw tensor or cross-block barrier is
                # required, and the caller retains the staging lease.
                if active:
                    if fanout:
                        for target in range(destination_begin, destination_end):
                            dst[
                                block_ids[target],
                                plane // 2,
                                scalar // head_dim,
                                plane % 2,
                                scalar % head_dim,
                            ] = decoded_low | (decoded_high << 8)
                    else:
                        dst[
                            block_id,
                            plane // 2,
                            scalar // head_dim,
                            plane % 2,
                            scalar % head_dim,
                        ] = decoded_low | (decoded_high << 8)
                if warp_local:
                    escape_group_prefix = escape_group_prefix + T.popcount(mask)
                    raw_escape_group_prefix = raw_escape_group_prefix + T.popcount(
                        raw_mask
                    )
                else:
                    # Shared prefixes cover one 128-thread group; carry
                    # the group total into the next group of the tile.
                    escape_group_prefix = (
                        escape_group_prefix
                        + warp_prefix[load_warps - 1]
                        + warp_counts[load_warps - 1]
                    )
                    raw_escape_group_prefix = (
                        raw_escape_group_prefix
                        + warp_raw_prefix[load_warps - 1]
                        + warp_raw_counts[load_warps - 1]
                    )

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
    "load_reads_words",
    "pack_load_codebooks",
]
