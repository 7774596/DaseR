# SPDX-License-Identifier: Apache-2.0
"""Attention-resident lossless KV page records (research prototype).

A record holds one (page, kv, head) slice of ``PAGE_TOKENS`` tokens by
``HEAD_DIM`` BF16 scalars. The layout is shaped for a fused decode-attention
kernel with 256 threads that decodes straight into ``mma.m16n8k16`` B
fragments: the scalars are first permuted into kernel order, where thread
``t`` (warp ``w``, lane ``4 g + c``) owns positions ``[64 t, 64 t + 64)`` and
its scalar ``s = 8 st + 4 j + 2 f + e`` is

- K: token ``16 w + 8 j + g``, dim ``16 st + 8 f + 2 c + e``;
- V: token ``16 st + 8 f + 2 c + e``, dim ``16 w + 8 j + g``;

so word ``st`` (eight scalars) is the ``(b0, b1)`` pair of both n-blocks of
k-step ``st``. Kernel order then defines every stream below:

- dense low bytes: one raw low byte per scalar (16 KiB). Thread ``t``'s 64
  bytes are four 16-byte chunks stored chunk-major (chunk ``j`` at
  ``16 (256 j + t)``) so one warp load covers 512 contiguous bytes;
- dense symbols: per 32-scalar group three little-endian u32 words; thread
  ``t``'s six words are three 8-byte chunks, also chunk-major. Nibble
  ``j`` of word ``k`` holds the 3-bit primary code of scalar ``8 k + j`` in its
  low three bits; its top bit is bit ``k`` of scalar ``24 + j``'s code. The
  kernel masks the words with ``0x77777777`` and feeds them to ``prmt`` as
  selectors over an 8-byte table, and rebuilds the fourth selector word from the
  top bits;
- variable region: a 48-byte header, then one nibble per escape (primary code
  7) in scalar order, then (4-byte aligned) one little-endian u32 per raw
  scalar, ``ordinal | high << 16`` where ``ordinal`` is its escape index.
  Nibble 0-14 selects extended table entry 7-21; nibble 15 marks a raw scalar
  whose expanded escape byte the kernel patches from the raw list before the
  deposit. The header carries per-warp (2048-scalar) escape and raw prefix
  counts so a warp scan alone locates every thread's escapes and each warp its
  raw patches.

A record that does not fit (variable region over ``VAR_CAP`` or a warp with
more than ``MAX_WARP_RAW`` raw scalars) is stored raw: ``RAW_RECORD_BYTES`` of
little-endian BF16 in kernel order, thread ``t``'s word ``st`` at
``16 (256 st + t)``. Its length alone marks it, since every coded record is at
most ``DENSE_BYTES + VAR_CAP`` bytes.

The high-byte code is the online codec's (3-bit primary, 7 direct entries plus
escape), so bits per scalar match it; only the arrangement differs.
"""

import struct

import numpy as np
from numpy.typing import NDArray

PAGE_TOKENS = 128
HEAD_DIM = 128
RECORD_SCALARS = PAGE_TOKENS * HEAD_DIM
GROUP_SCALARS = 32
WARP_SCALARS = 2048
WARPS = RECORD_SCALARS // WARP_SCALARS
PRIMARY_DIRECT = 7
EXTENDED = 15
TABLE_ENTRIES = PRIMARY_DIRECT + EXTENDED
RAW_NIBBLE = 15
HEADER_BYTES = 48
LOW_BYTES = RECORD_SCALARS
SYMBOL_BYTES = RECORD_SCALARS * 3 // 8
DENSE_BYTES = LOW_BYTES + SYMBOL_BYTES
VAR_CAP = 2048
THREADS = 256
LOW_CHUNKS = 4  # 16-byte low-byte chunks per thread
SYMBOL_CHUNKS = 3  # 8-byte symbol chunks per thread
WORDS = 8  # 8-scalar words (mma k-steps) per thread
RAW_RECORD_BYTES = RECORD_SCALARS * 2
MAX_WARP_RAW = 32  # the kernel patches at most one raw scalar per lane
# Block blob (one closed block's records in an external tier, hbm_kv.cu): a
# fixed header (u32 magic, version, L, H; i64 bytes, digest) and the i32
# var-length table [L, 2, H] in a BLOB_ALIGN-padded head, then the records in
# (layer, kv, head) order, each DENSE_BYTES + var_len (raw records whole).
BLOB_MAGIC = 0x42314641  # "AF1B"
BLOB_FIXED_BYTES = 64
BLOB_ALIGN = 4096

_HEADER = struct.Struct("<4I8H8H")


def _kernel_order(kv: int) -> NDArray[np.int64]:
    """Token-major scalar index of each kernel-order position (module docstring)."""
    t = np.arange(THREADS)[:, None]
    s = np.arange(RECORD_SCALARS // THREADS)[None, :]
    w, g, c = t >> 5, (t >> 2) & 7, t & 3
    st, j, f, e = s >> 3, (s >> 2) & 1, (s >> 1) & 1, s & 1
    row, col = 16 * w + 8 * j + g, 16 * st + 8 * f + 2 * c + e
    tok, dim = (row, col) if kv == 0 else (col, row)
    return (tok * HEAD_DIM + dim).reshape(-1)


_ORDER = (_kernel_order(0), _kernel_order(1))


def calibrate_table(high_bytes: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Pick the 22 most frequent high bytes of one (layer, kv, head).

    Args:
        high_bytes: Calibration high bytes of any shape.

    Returns:
        uint8 array of ``TABLE_ENTRIES`` bytes, most frequent first.

    Async/thread-safety:
        Pure function.
    """
    counts = np.bincount(np.asarray(high_bytes, dtype=np.uint8).ravel(), minlength=256)
    order = np.argsort(-counts, kind="stable")
    return order[:TABLE_ENTRIES].astype(np.uint8)


def primary_lut(table: NDArray[np.uint8]) -> tuple[int, int]:
    """Return the two u32 words of the 8-byte primary ``prmt`` table.

    Args:
        table: Head table from :func:`calibrate_table`.

    Returns:
        ``(lo, hi)`` little-endian words; byte 7 (escape) is zero.

    Async/thread-safety:
        Pure function.
    """
    entries = bytes(table[:PRIMARY_DIRECT]) + b"\x00"
    lo, hi = struct.unpack("<II", entries)
    return lo, hi


def secondary_lut(table: NDArray[np.uint8]) -> bytes:
    """Return the 16-byte extended table indexed by escape nibble.

    Args:
        table: Head table from :func:`calibrate_table`.

    Returns:
        Entries 7-21 followed by one zero byte.

    Async/thread-safety:
        Pure function.
    """
    return bytes(table[PRIMARY_DIRECT:TABLE_ENTRIES]) + b"\x00"


def encode_record(raw: NDArray[np.uint16], table: NDArray[np.uint8], kv: int) -> bytes:
    """Encode one token-major ``[PAGE_TOKENS, HEAD_DIM]`` BF16 record.

    Args:
        raw: BF16 bit patterns as uint16.
        table: Head table from :func:`calibrate_table`.
        kv: 0 for a K record, 1 for a V record (selects the kernel order).

    Returns:
        Record bytes (16-byte aligned): coded, or the ``RAW_RECORD_BYTES`` raw
        fallback when the variable region would exceed ``VAR_CAP`` or a warp
        would hold more than ``MAX_WARP_RAW`` raw scalars.

    Raises:
        ValueError: If ``raw`` has the wrong size.

    Async/thread-safety:
        Pure function.
    """
    values = np.ascontiguousarray(raw, dtype=np.uint16).reshape(-1)
    if values.size != RECORD_SCALARS:
        raise ValueError("record must hold PAGE_TOKENS * HEAD_DIM scalars")
    values = values[_ORDER[kv]]
    low = (values & 0xFF).astype(np.uint8)
    high = (values >> 8).astype(np.uint8)
    rank_of = np.full(256, 255, dtype=np.uint8)
    rank_of[table] = np.arange(TABLE_ENTRIES, dtype=np.uint8)
    rank = rank_of[high]
    primary = (
        np.minimum(rank, PRIMARY_DIRECT).astype(np.uint32).reshape(-1, GROUP_SCALARS)
    )
    words = np.zeros((primary.shape[0], 3), dtype=np.uint32)
    for k in range(3):
        for j in range(8):
            words[:, k] |= (primary[:, 8 * k + j] & 7) << (4 * j)
            words[:, k] |= ((primary[:, 24 + j] >> k) & 1) << (4 * j + 3)

    escape = rank >= PRIMARY_DIRECT
    is_raw = rank >= TABLE_ENTRIES
    esc_rank = rank[escape]
    nibbles = np.where(esc_rank < TABLE_ENTRIES, esc_rank - PRIMARY_DIRECT, RAW_NIBBLE)
    nibbles = nibbles.astype(np.uint8)
    raw_ordinal = np.flatnonzero(esc_rank >= TABLE_ENTRIES).astype(np.uint32)
    raw_list = (
        (raw_ordinal | (high[is_raw].astype(np.uint32) << 16)).astype("<u4").tobytes()
    )
    starts = np.arange(WARPS) * WARP_SCALARS
    esc_prefix = np.concatenate([[0], np.cumsum(escape)])[starts]
    raw_prefix = np.concatenate([[0], np.cumsum(is_raw)])[starts]
    if nibbles.size % 2:
        nibbles = np.append(nibbles, np.uint8(0))
    nibble_bytes = (nibbles[0::2] | (nibbles[1::2] << 4)).astype(np.uint8).tobytes()
    raw_offset = (HEADER_BYTES + len(nibble_bytes) + 3) & ~3
    var_len = _align16(raw_offset + len(raw_list))
    warp_raw = is_raw.reshape(WARPS, WARP_SCALARS).sum(axis=1)
    if var_len > VAR_CAP or int(warp_raw.max()) > MAX_WARP_RAW:
        return _raw_words(values).tobytes()
    header = _HEADER.pack(
        int(escape.sum()), int(is_raw.sum()), raw_offset, 0, *esc_prefix, *raw_prefix
    )
    var = header + nibble_bytes
    var += b"\x00" * (raw_offset - len(var)) + raw_list
    var += b"\x00" * (var_len - len(var))
    return _interleave_dense(low, words) + var


def decode_record(
    record: bytes, table: NDArray[np.uint8], kv: int
) -> NDArray[np.uint16]:
    """Reference decoder for :func:`encode_record`.

    Args:
        record: Encoded record.
        table: Head table used to encode it.
        kv: The ``kv`` the record was encoded with.

    Returns:
        uint16 ``[PAGE_TOKENS, HEAD_DIM]`` BF16 bit patterns.

    Async/thread-safety:
        Pure function.
    """
    out = np.empty(RECORD_SCALARS, dtype=np.uint16)
    if len(record) == RAW_RECORD_BYTES:
        words = np.frombuffer(record, dtype="<u2").reshape(WORDS, THREADS, 8)
        out[_ORDER[kv]] = words.transpose(1, 0, 2).reshape(-1)
        return out.reshape(PAGE_TOKENS, HEAD_DIM)
    buf = np.frombuffer(record, dtype=np.uint8)
    low = (
        buf[:LOW_BYTES]
        .reshape(LOW_CHUNKS, THREADS, 16)
        .transpose(1, 0, 2)
        .reshape(-1)
        .astype(np.uint16)
    )
    words = (
        np.frombuffer(record[LOW_BYTES:DENSE_BYTES], dtype="<u4")
        .reshape(SYMBOL_CHUNKS, THREADS, 2)
        .transpose(1, 0, 2)
        .reshape(-1, 3)
    )
    codes = np.zeros((words.shape[0], GROUP_SCALARS), dtype=np.uint32)
    for k in range(3):
        for j in range(8):
            codes[:, 8 * k + j] = (words[:, k] >> (4 * j)) & 7
            codes[:, 24 + j] |= ((words[:, k] >> (4 * j + 3)) & 1) << k
    codes = codes.reshape(-1)
    high = np.zeros(RECORD_SCALARS, dtype=np.uint16)
    direct = codes < PRIMARY_DIRECT
    high[direct] = table[codes[direct]]
    var = record[DENSE_BYTES:]
    n_esc, n_raw, raw_offset, _ = struct.unpack_from("<4I", var)
    packed = np.frombuffer(
        var[HEADER_BYTES : HEADER_BYTES + (n_esc + 1) // 2], np.uint8
    )
    nibbles = np.stack([packed & 15, packed >> 4], axis=1).reshape(-1)[:n_esc]
    raws = np.frombuffer(var[raw_offset : raw_offset + 4 * n_raw], "<u4")
    values = np.zeros(n_esc, dtype=np.uint16)
    ext = nibbles < RAW_NIBBLE
    values[ext] = table[PRIMARY_DIRECT + nibbles[ext]]
    values[raws & 0xFFFF] = raws >> 16
    high[~direct] = values
    out[_ORDER[kv]] = low | (high << 8)
    return out.reshape(PAGE_TOKENS, HEAD_DIM)


def _interleave_dense(low: NDArray[np.uint8], words: NDArray[np.uint32]) -> bytes:
    """Store thread chunks chunk-major so each warp load is 512 B contiguous."""
    low_t = low.reshape(THREADS, LOW_CHUNKS, 16).transpose(1, 0, 2)
    sym_t = words.astype("<u4").reshape(THREADS, SYMBOL_CHUNKS, 2).transpose(1, 0, 2)
    return np.ascontiguousarray(low_t).tobytes() + np.ascontiguousarray(sym_t).tobytes()


def _raw_words(values: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Kernel-order scalars as 16-byte thread words stored word-major."""
    words = values.astype("<u2").reshape(THREADS, WORDS, 8).transpose(1, 0, 2)
    return np.ascontiguousarray(words)


def blob_head_bytes(num_layers: int, kv_heads: int) -> int:
    """Return the padded head (header + length table) of a block blob.

    Args:
        num_layers: Layers ``L``.
        kv_heads: KV heads ``H``.

    Returns:
        ``BLOB_FIXED_BYTES + 4 * L * 2 * H`` rounded up to ``BLOB_ALIGN``.

    Async/thread-safety:
        Pure function.
    """
    head = BLOB_FIXED_BYTES + 8 * num_layers * kv_heads
    return -(-head // BLOB_ALIGN) * BLOB_ALIGN


def blob_slot_bytes(num_layers: int, kv_heads: int) -> int:
    """Return the largest block blob, rounded up to ``BLOB_ALIGN``.

    Every record raw: the external-tier slot size of AF1 storage.

    Args:
        num_layers: Layers ``L``.
        kv_heads: KV heads ``H``.

    Returns:
        ``blob_head_bytes + 2 L H RAW_RECORD_BYTES`` (already aligned).

    Async/thread-safety:
        Pure function.
    """
    return blob_head_bytes(num_layers, kv_heads) + (
        2 * num_layers * kv_heads * RAW_RECORD_BYTES
    )


def _align16(value: int) -> int:
    return (value + 15) & ~15


__all__ = [
    "BLOB_ALIGN",
    "BLOB_FIXED_BYTES",
    "BLOB_MAGIC",
    "DENSE_BYTES",
    "HEAD_DIM",
    "PAGE_TOKENS",
    "RAW_RECORD_BYTES",
    "VAR_CAP",
    "blob_head_bytes",
    "blob_slot_bytes",
    "calibrate_table",
    "decode_record",
    "encode_record",
    "primary_lut",
    "secondary_lut",
]
