# SPDX-License-Identifier: Apache-2.0
"""Split-KV decode attention over FragPack KV records.

One CTA (256 threads) handles one split of one (sequence, KV head) pair for
all ``GQA`` query heads, with an online softmax; a small combine kernel merges
the splits. The MMAs are transposed, ``S^T = K Q^T`` and ``O^T += V^T P^T``,
so K / V^T is the 16-row A operand and the 8 GQA heads fill the n8 side
without padding. Thread 0 bulk-copies (TMA) each K and V record into its own
shared slot, and every thread decodes its 64 kernel-order scalars straight
into A fragments, overlapping the next record's copy.

Records come from :mod:`daser.compression.attn_page` and are located through
per-record tables: ``rec_addr[P, 2, H]`` absolute device addresses and
``rec_len[P, 2, H]`` record bytes minus ``DENSE_BYTES`` (raw records, whose
length exceeds ``VAR_CAP``, included). Each sequence's pages are split into
``splits`` equal ranges on device, so a launch depends only on the batch size
and is CUDA-graph safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from daser.logging import init_logger
from daser.ops.fragpack_ext import extension

logger = init_logger(__name__)

GQA = 8


@dataclass(frozen=True)
class RecordTables:
    """One layer's record tables for :func:`decode_attention`.

    Attributes:
        rec_addr: int64 ``[P, 2, H]`` record device addresses.
        rec_len: int32 ``[P, 2, H]`` record bytes minus ``DENSE_BYTES``.
        plut: int32 ``[2, H, 2]`` primary ``prmt`` tables.
        slut: uint8 ``[2, H, 16]`` extended tables.
    """

    rec_addr: torch.Tensor
    rec_len: torch.Tensor
    plut: torch.Tensor
    slut: torch.Tensor


def decode_splits(batch: int, kv_heads: int, num_sms: int) -> int:
    """Return the per-(sequence, KV head) split count for a decode batch.

    Two CTAs fit per SM; splits fill one wave. Depending on the batch size only
    keeps the launch shape fixed under CUDA graphs.

    Args:
        batch: Number of sequences (padded batch size under CUDA graphs).
        kv_heads: KV heads per sequence.
        num_sms: Streaming multiprocessor count.

    Returns:
        Split count, at least 1.

    Async/thread-safety:
        Pure function.
    """
    return max(1, (2 * num_sms) // max(1, batch * kv_heads))


def workspace_elems(batch: int, kv_heads: int, splits: int) -> tuple[int, int]:
    """Return the float element counts of the ``(opart, ml)`` workspace.

    Args:
        batch: Number of sequences.
        kv_heads: KV heads per sequence.
        splits: Split count.

    Returns:
        Elements needed for ``opart`` and ``ml``.

    Async/thread-safety:
        Pure function.
    """
    rows = batch * kv_heads * splits * GQA
    return rows * 128, rows * 2


def decode_attention(
    q: torch.Tensor,
    tables: RecordTables,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    splits: int,
    out: torch.Tensor,
    workspace: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Run split-KV decode attention for one query token per sequence.

    Args:
        q: BF16 ``[B, H * GQA, 128]`` queries (rows may be strided).
        tables: Record tables of this layer.
        block_table: int32 ``[B, max_pages]`` page ids (rows may be strided).
        seq_lens: int32 ``[B]`` context lengths; 0 yields a zero output row.
        scale: Softmax scale.
        splits: CTAs per (sequence, KV head), e.g. from :func:`decode_splits`.
        out: BF16 ``[B, H * GQA, 128]`` output (rows may be strided).
        workspace: ``(opart, ml)`` float tensors sized by
            :func:`workspace_elems`.

    Returns:
        None; writes ``out``.

    Async/thread-safety:
        Enqueues on the current CUDA stream; not synchronized.
    """
    extension().fused_decode(
        q,
        tables.rec_addr,
        tables.rec_len,
        tables.plut,
        tables.slut,
        block_table,
        seq_lens,
        workspace[0],
        workspace[1],
        out,
        splits,
        scale,
    )


__all__ = [
    "GQA",
    "RecordTables",
    "decode_attention",
    "decode_splits",
    "workspace_elems",
]
