# SPDX-License-Identifier: Apache-2.0
"""HbmKvStore + fused decode attention against the attn_page reference.

Drives four steps of a small batch through the store's public API (prepare,
write, encode, commit) and checks: arena records bit-identical to
``attn_page.encode_record`` and packed disjointly inside their block's page,
raw records (one fallback; with a tight page, records that no longer fit)
holding the written KV, gather output equal to the written KV, staging
released by GC and commit, and decode attention over coded, raw, and
still-open blocks against an fp32 reference. Block blobs exported from one
store and imported into other blocks of a second store must pass the same
checks, free the overflow slots they replace, and reject corrupt or foreign
blobs without touching the destination.
"""

import numpy as np
import pytest
import torch

from daser.compression import attn_page as ap
from daser.compression.attn_tables import device_tables
from daser.compression.format import SlotMode
from daser.ops.fragpack_codec import FragPackBlockExporter, FragPackBlockImporter
from daser.ops.fused_attn_decode import (
    GQA,
    RecordTables,
    decode_attention,
    decode_units,
    workspace_elems,
)
from daser.ops.hbm_kv import HbmKvStore, page_bytes, register_store

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

L, H, D, T = 2, 2, ap.HEAD_DIM, ap.PAGE_TOKENS
HQ = H * GQA
NB = 12
SCALE = D**-0.5
WIDE = (9, 0, 0, 1)  # (block, layer, kv, head) made incompressible


def _bits(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _bf16(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bits.view(np.int16)).cuda().view(torch.bfloat16)


def _overflow_used(store: HbmKvStore, records: int) -> int:
    bits = np.unpackbits(store.ovf_bits.cpu().numpy().view(np.uint8))
    return int(bits.sum()) - (bits.size - records)


class _Harness:
    def __init__(
        self, rng: np.random.Generator, page: int, overflow_records: int = 16
    ) -> None:
        self.rng = rng
        self.page = page
        cal = _bits(rng.standard_normal((L, 2, H, 4 * T, D)) * 0.5)
        self.table = np.empty((L, 2, H, ap.TABLE_ENTRIES), np.uint8)
        for idx in np.ndindex(L, 2, H):
            self.table[idx] = ap.calibrate_table((cal[idx] >> 8).astype(np.uint8))
        # Garbage, like an engine's uninitialized KV cache.
        self.caches = [
            torch.randint(0, 256, (NB, page), dtype=torch.uint8, device="cuda")
            for _ in range(L)
        ]
        self.store = HbmKvStore(
            self.caches,
            H,
            device_tables(self.table, torch.device("cuda")),
            max_tokens=1024,
            staging_slots=3,
            overflow_records=overflow_records,
        )
        self.overflow_records = overflow_records
        # ref[layer, kv, block, offset, head, dim] BF16 bits of written tokens.
        self.ref = np.zeros((L, 2, NB, T, H, D), np.uint16)

    def step(self, rows: list[tuple[list[int], int, int]], live: list[int]) -> None:
        """Run one step; a row is (blocks, first position, token count)."""
        slots = [
            blocks[p // T] * T + p % T
            for blocks, start, n in rows
            for p in range(start, start + n)
        ]
        n = len(slots)
        slot_t = torch.tensor(slots, dtype=torch.int64, device="cuda")
        closes = sum((s + c) // T - s // T for _, s, c in rows)
        self.store.prepare(slot_t, n, np.array(live, np.int32))
        blk, off = np.array(slots) // T, np.array(slots) % T
        for layer in range(L):
            kv = _bits(self.rng.standard_normal((2, n, H, D)) * 0.5)
            hit = blk == WIDE[0]
            if layer == WIDE[1] and hit.any():
                wide = np.exp2(self.rng.uniform(-40, 4, (int(hit.sum()), D)))
                kv[WIDE[2], hit, WIDE[3]] = _bits(wide)
            self.ref[layer, :, blk, off] = kv.transpose(1, 0, 2, 3)
            # K / V as strided slices of one fused qkv row, like vLLM.
            qkv = torch.empty(n, HQ + 2 * H, D, dtype=torch.bfloat16, device="cuda")
            qkv[:, HQ : HQ + H] = _bf16(kv[0])
            qkv[:, HQ + H :] = _bf16(kv[1])
            key, value = qkv[:, HQ : HQ + H], qkv[:, HQ + H :]
            self.store.write(layer, key, value, slot_t)
            self.store.encode(layer, closes)
        self.store.commit()

    def check_arena(self, blocks: list[int]) -> tuple[int, int]:
        """Check closed blocks' records; return (fallbacks, spills)."""
        addr = self.store.rec_addr.cpu().numpy()
        length = self.store.rec_len.cpu().numpy()
        overflow = self.store.overflow.cpu().numpy().reshape(-1)
        ovf_base = self.store.overflow.data_ptr()
        fallbacks = spills = 0
        for b in blocks:
            for layer in range(L):
                arena = self.caches[layer].cpu().numpy().reshape(-1)
                page = self.caches[layer].data_ptr() + b * self.page
                used = []
                for kv in range(2):
                    for h in range(H):
                        ref = self.ref[layer, kv, b, :, h]
                        table = self.table[layer, kv, h]
                        want = ap.encode_record(ref, table, kv)
                        a = int(addr[layer, b, kv, h])
                        if ovf_base <= a < ovf_base + overflow.size:
                            got = overflow[a - ovf_base :][: ap.RAW_RECORD_BYTES]
                            np.testing.assert_array_equal(
                                ap.decode_record(got.tobytes(), table, kv), ref
                            )
                            if len(want) == ap.RAW_RECORD_BYTES:
                                fallbacks += 1
                                assert (b, layer, kv, h) == WIDE
                            else:
                                spills += 1
                            continue
                        n = len(want)
                        assert n < ap.RAW_RECORD_BYTES
                        assert a % 16 == 0 and page <= a <= page + self.page - n
                        assert ap.DENSE_BYTES + int(length[layer, b, kv, h]) == n
                        o = a - self.caches[layer].data_ptr()
                        assert arena[o : o + n].tobytes() == want, (b, layer, kv, h)
                        used.append((a, a + n))
                used.sort()
                ends, starts = [e for _, e in used[:-1]], [s for s, _ in used[1:]]
                assert all(e <= s for e, s in zip(ends, starts, strict=True))
        return fallbacks, spills

    def check_gather(self, filled: dict[int, int]) -> None:
        """Gather blocks of every layer and compare the written tokens."""
        blocks = sorted(filled)
        src = torch.tensor(blocks, dtype=torch.int32, device="cuda")
        wk = torch.empty(len(blocks), T, H, D, dtype=torch.bfloat16, device="cuda")
        wv = torch.empty_like(wk)
        for layer in range(L):
            self.store.gather(layer, src, wk, wv)
            got = [w.view(torch.int16).cpu().numpy().view(np.uint16) for w in (wk, wv)]
            for i, b in enumerate(blocks):
                n = filled[b]
                for kv in range(2):
                    np.testing.assert_array_equal(
                        got[kv][i, :n], self.ref[layer, kv, b, :n], err_msg=f"{b}"
                    )

    def check_decode(self, tables: list[list[int]], seqs: list[int]) -> None:
        """Fused decode vs an fp32 reference over the written KV.

        Runs with the default unit count (one page per chunk here) and with
        fewer units, so chunks span several pages and sequences end mid-chunk.
        """
        B = len(seqs)
        width = max(len(t) for t in tables)
        bt = np.zeros((B, width), np.int32)
        for i, t in enumerate(tables):
            bt[i, : len(t)] = t
        block_table = torch.from_numpy(bt).cuda()
        seq_lens = torch.tensor(seqs, dtype=torch.int32, device="cuda")
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        for units in (decode_units(B, H, sms), B, B + 1):
            opart, ml = workspace_elems(H, units)
            ws = (
                torch.empty(opart, device="cuda"),
                torch.empty(ml, device="cuda"),
            )
            for layer in range(L):
                q = torch.randn(B, HQ, D, dtype=torch.bfloat16, device="cuda")
                out = torch.empty_like(q)
                tabs = RecordTables(*self.store.layer_tables(layer))
                decode_attention(q, tabs, block_table, seq_lens, SCALE, units, out, ws)
                kv = _bf16(self.ref[layer]).float()  # [2, NB, T, H, D]
                ref = torch.empty(B, HQ, D, device="cuda")
                for i, (t, s) in enumerate(zip(tables, seqs, strict=True)):
                    pg = torch.tensor(t, device="cuda")
                    for h in range(H):
                        k = kv[0, pg, :, h].reshape(-1, D)[:s]
                        v = kv[1, pg, :, h].reshape(-1, D)[:s]
                        qh = q[i, GQA * h : GQA * (h + 1)].float()
                        ref[i, GQA * h : GQA * (h + 1)] = (
                            torch.softmax(qh @ k.T * SCALE, -1) @ v
                        )
                torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "page, spills",
    [
        (page_bytes(H), 0),
        # Room for three of a block-layer's four records: the last one encoded
        # spills raw (except where the WIDE record already fell back).
        (2 * H * ap.DENSE_BYTES - 16, 5 * L - 1),
    ],
)
def test_store_steps_match_reference(page: int, spills: int) -> None:
    h = _Harness(np.random.default_rng(0), page)
    a, b, c, d = [0, 1, 4, 5], [8, 9, 10], [2], [3]
    # Prefill: A and B close blocks 0, 1, 8, 9 in the same step (transient
    # staging); A's block 4 and C's block 2 stay open (persistent).
    h.step([(a, 0, 300), (b, 0, 256), (c, 0, 10)], a + b + c)
    # A's chunk closes persistent block 4; B opens block 10.
    h.step([(a, 300, 84), (b, 256, 1), (c, 10, 1)], a + b + c)
    torch.cuda.synchronize()
    assert h.check_arena([0, 1, 4, 8, 9]) == (1, spills)
    # C finished: GC must free block 2's slot, else D / A's claims overflow
    # the 3 persistent slots (10, 3, 5).
    h.step([(a, 384, 1), (b, 257, 1), (d, 0, 10)], a + b + d)
    h.step([(a, 385, 1), (b, 258, 1), (d, 10, 1)], a + b + d)
    torch.cuda.synchronize()
    stats = h.store.stats()
    assert stats.encoded == 5 * L * 2 * H
    assert stats.fallbacks == 1 + spills and stats.spills == spills
    assert stats.err_staging == 0 and stats.err_overflow == 0
    # Block 11 was never written: it must read as zeros, not arena garbage.
    h.check_gather({0: T, 1: T, 4: T, 5: 2, 8: T, 9: T, 10: 3, 3: 11, 11: T})
    h.check_decode([a, b, d, [11]], [386, 259, 11, T])


# Source blocks closed by _prefill, and where the import puts them (the WIDE
# block keeps its id so check_arena can recognise its fallback).
SRC, DST = [0, 1, 4, 8, 9], [6, 7, 2, 3, 9]


def _prefill(h: _Harness) -> None:
    a, b, c = [0, 1, 4, 5], [8, 9, 10], [2]
    h.step([(a, 0, 300), (b, 0, 256), (c, 0, 10)], a + b + c)
    h.step([(a, 300, 84), (b, 256, 1), (c, 10, 1)], a + b + c)


def _export(h: _Harness, blocks: list[int]) -> tuple[torch.Tensor, list[int]]:
    stride = h.store.blob_stride
    out = torch.zeros(len(blocks) * stride, dtype=torch.uint8, device="cuda")
    ids = torch.tensor(blocks, dtype=torch.int32, device="cuda")
    lengths = h.store.export_blocks(ids, out, stride).cpu().tolist()
    return out, lengths


def _import(h: _Harness, src: torch.Tensor, offsets: list[int], blocks: list[int]):
    off = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    ids = torch.tensor(blocks, dtype=torch.int32, device="cuda")
    return h.store.import_blocks(src, off, ids).cpu().tolist()


@pytest.mark.parametrize(
    "page, spills",
    [(page_bytes(H), 0), (2 * H * ap.DENSE_BYTES - 16, 5 * L - 1)],
)
def test_blob_round_trip_matches_reference(page: int, spills: int) -> None:
    src = _Harness(np.random.default_rng(0), page)
    _prefill(src)
    blobs, lengths = _export(src, SRC)
    stride = src.store.blob_stride
    assert stride == ap.blob_slot_bytes(L, H)
    head = ap.blob_head_bytes(L, H)
    rec_len = src.store.rec_len.cpu().numpy()
    for b, n in zip(SRC, lengths, strict=True):
        body = sum(ap.DENSE_BYTES + int(v) for v in rec_len[:, b].ravel())
        assert n == -(-(head + body) // ap.BLOB_ALIGN) * ap.BLOB_ALIGN <= stride

    # Same seed: same codec tables, so the blobs are accepted. An import
    # places its raw records before freeing the ones it replaces, so the
    # re-import below needs room for both copies.
    dst = _Harness(np.random.default_rng(0), page, overflow_records=32)
    for s_b, d_b in zip(SRC, DST, strict=True):
        dst.ref[:, :, d_b] = src.ref[:, :, s_b]
    offsets = [i * stride for i in range(len(SRC))]
    assert _import(dst, blobs, offsets, DST) == [0] * len(DST)
    torch.cuda.synchronize()
    assert dst.check_arena(DST) == (1, spills)
    assert _overflow_used(dst.store, dst.overflow_records) == 1 + spills
    # Re-importing frees the overflow records it replaces.
    assert _import(dst, blobs, offsets, DST) == [0] * len(DST)
    assert _overflow_used(dst.store, dst.overflow_records) == 1 + spills
    dst.check_gather({b: T for b in DST})
    dst.check_decode([[6, 7], [2], [3, 9]], [256, 128, 256])


def test_blob_import_rejects_without_side_effects() -> None:
    src = _Harness(np.random.default_rng(0), page_bytes(H))
    _prefill(src)
    blobs, _ = _export(src, [8, 9])
    stride = src.store.blob_stride
    dst = _Harness(np.random.default_rng(0), page_bytes(H), overflow_records=1)
    before = dst.store.rec_addr.clone(), dst.store.rec_len.clone()

    bad = blobs.clone()
    bad[0] ^= 1  # magic
    bad[stride + ap.BLOB_FIXED_BYTES] = 8  # first var length: not a record size
    assert _import(dst, bad, [0, stride], [5, 6]) == [1, 2]
    # Tables of another calibration have another digest.
    other = _Harness(np.random.default_rng(1), page_bytes(H))
    assert _import(other, blobs, [0], [5]) == [1]
    torch.cuda.synchronize()
    assert torch.equal(dst.store.rec_addr, before[0])
    assert torch.equal(dst.store.rec_len, before[1])

    # Block 9's WIDE record takes the only overflow slot; a second copy finds
    # the pool full in that layer and must release what it took.
    assert _import(dst, blobs, [stride], [9]) == [0]
    assert _import(dst, blobs, [stride], [10]) == [4]
    assert _overflow_used(dst.store, 1) == 1
    torch.cuda.synchronize()
    wide_layer = WIDE[1]
    assert torch.equal(dst.store.rec_addr[wide_layer, 10], before[0][wide_layer, 10])


def test_codec_wrappers_round_trip() -> None:
    src = _Harness(np.random.default_rng(0), page_bytes(H))
    _prefill(src)
    # The wrappers run on their own stream; production orders it with events.
    torch.cuda.synchronize()
    stride = src.store.blob_stride
    staging = torch.zeros(2 * stride, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    register_store(src.store)
    records = FragPackBlockExporter(stride).pack_into(
        staging=staging,
        block_ids=[0, 1],
        logical_slots=[10, 11],
        slot_stride=stride,
        stream=stream,
    )
    assert [r.logical_slot for r in records] == [10, 11]
    assert all(r.mode == SlotMode.COMPRESSED for r in records)
    assert [r.source_offset for r in records] == [0, stride]
    with pytest.raises(ValueError):
        FragPackBlockExporter(stride + 16).pack_into(
            staging=staging,
            block_ids=[0],
            logical_slots=[0],
            slot_stride=stride + 16,
            stream=stream,
        )

    dst = _Harness(np.random.default_rng(0), page_bytes(H))
    torch.cuda.synchronize()
    register_store(dst.store)
    importer = FragPackBlockImporter()
    offsets = [r.source_offset for r in records]
    plan = importer.prepare(
        staging=staging,
        staging_offsets=offsets,
        block_ids=[6, 7],
        modes=[1, 1],
        buffer_index=0,
        stream=stream,
    )
    assert plan is not None and plan.submit() == 2 and plan.submit() == 0
    stream.synchronize()
    assert plan.failed_blocks() == []
    for s_b, d_b in ((0, 6), (1, 7)):
        dst.ref[:, :, d_b] = src.ref[:, :, s_b]
    dst.check_gather({6: T, 7: T})

    staging[stride] ^= 1
    torch.cuda.synchronize()
    plan = importer.prepare(
        staging=staging,
        staging_offsets=offsets,
        block_ids=[2, 3],
        modes=[1, 1],
        buffer_index=0,
        stream=stream,
    )
    assert plan is not None
    plan.submit()
    stream.synchronize()
    assert plan.failed_blocks() == [(3, "bad header")]
    kwargs = dict(staging=staging, buffer_index=0, stream=stream)
    with pytest.raises(ValueError):
        importer.prepare(staging_offsets=[0], block_ids=[2], modes=[0], **kwargs)
    with pytest.raises(ValueError):
        importer.prepare(
            staging_offsets=[0, 0], block_ids=[2, 2], modes=[1, 1], **kwargs
        )
    with pytest.raises(ValueError):
        importer.prepare(
            staging_offsets=[0],
            block_ids=[2],
            modes=[1],
            plane_range=(0, 2),
            **kwargs,
        )
    assert (
        importer.prepare(staging_offsets=[], block_ids=[], modes=[], **kwargs) is None
    )
