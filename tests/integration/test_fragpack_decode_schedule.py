# SPDX-License-Identifier: Apache-2.0
"""Decode scheduling contracts for ragged and CUDA-graph-padded batches.

Three physical pages are repeated in block tables to exercise long contexts
without a model or a large KV allocation. The reference folds identical token
copies into multiplicities before evaluating FP32 attention; it never inspects
the device planner or the split-KV workspace.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from daser.compression import attn_page as ap
from daser.compression.attn_tables import calibrate, device_tables
from daser.ops.fused_attn_decode import (
    GQA,
    RecordTables,
    decode_attention,
    decode_units,
    workspace_elems,
)
from daser.ops.hbm_kv import HbmKvStore, page_bytes

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

LAYERS, HEADS, BLOCKS = 2, 4, 3
TOKENS, DIM = ap.PAGE_TOKENS, ap.HEAD_DIM
MAX_PAGES = 1024
SCALE = DIM**-0.5
StoreFixture = tuple[HbmKvStore, list[torch.Tensor], torch.Tensor]
Workspace = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@pytest.fixture(scope="module")
def encoded_store() -> StoreFixture:
    """Create two layers of encoded KV and retain their arena tensor owners.

    Returns:
        Store, owning arena tensors, and BF16 source KV of shape
        ``[layers, 2, blocks, tokens, heads, dim]`` on the current GPU.

    Async/thread-safety:
        Initializes on the current CUDA stream and synchronizes through stats.
        Tests run sequentially and only read the resulting store.
    """
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FragPack decode requires Hopper or newer")
    generator = torch.Generator().manual_seed(231)
    source = torch.randn(
        LAYERS, 2, BLOCKS, TOKENS, HEADS, DIM, generator=generator
    ).mul_(0.5)
    # Distinct page means make omitted or misassigned pages visible even when
    # attention is nearly uniform over many repeated tokens.
    source[:, 1] += torch.arange(BLOCKS)[None, :, None, None, None] - 1
    source = source.to(torch.bfloat16)
    bits = source.view(torch.int16).numpy().view(np.uint16)
    table = calibrate(bits.transpose(2, 0, 1, 3, 4, 5))
    device = torch.device("cuda", torch.cuda.current_device())
    arenas = [
        torch.empty(BLOCKS, page_bytes(HEADS), dtype=torch.uint8, device=device)
        for _ in range(LAYERS)
    ]
    store = HbmKvStore(
        arenas,
        HEADS,
        device_tables(table, device),
        max_tokens=BLOCKS * TOKENS,
        staging_slots=1,
        overflow_records=LAYERS * BLOCKS * 2 * HEADS,
    )
    source = source.to(device)
    slots = torch.arange(BLOCKS * TOKENS, dtype=torch.int64, device=device)
    store.prepare(slots, slots.numel(), np.arange(BLOCKS, dtype=np.int32))
    for layer in range(LAYERS):
        key, value = source[layer].reshape(2, BLOCKS * TOKENS, HEADS, DIM)
        store.write(layer, key, value, slots)
        store.encode(layer, BLOCKS)
    store.commit()
    stats = store.stats()
    assert stats.err_staging == 0 and stats.err_overflow == 0
    assert stats.encoded > stats.fallbacks
    return store, arenas, source


def _block_table(batch: int) -> NDArray[np.int32]:
    return (
        (np.arange(batch)[:, None] + np.arange(MAX_PAGES)[None, :]) % BLOCKS
    ).astype(np.int32)


def _ragged_lengths(batch: int) -> NDArray[np.int32]:
    lengths = np.resize(
        np.array([1, TOKENS - 1, TOKENS, TOKENS + 1, 8 * TOKENS + 13], np.int32),
        batch,
    )
    lengths[1::7] = 0
    lengths[0] = 63 * TOKENS + 1
    lengths[-1] = MAX_PAGES * TOKENS - 1
    return lengths


def _workspace(units: int, device: torch.device) -> Workspace:
    opart, ml, plan = workspace_elems(HEADS, units)
    return (
        torch.empty(opart, dtype=torch.float32, device=device),
        torch.empty(ml, dtype=torch.float32, device=device),
        torch.empty(plan, dtype=torch.int32, device=device),
    )


def _reference(
    q: torch.Tensor,
    source: torch.Tensor,
    block_table: NDArray[np.int32],
    lengths: NDArray[np.int32],
) -> torch.Tensor:
    counts = np.zeros((len(lengths), BLOCKS, TOKENS), dtype=np.float32)
    for row, length in enumerate(lengths):
        full_pages, tail = divmod(int(length), TOKENS)
        counts[row] = np.bincount(block_table[row, :full_pages], minlength=BLOCKS)[
            :, None
        ]
        if tail:
            counts[row, block_table[row, full_pages], :tail] += 1
    multiplicity = torch.from_numpy(counts.reshape(len(lengths), -1)).to(q.device)
    key, value = (
        source.permute(0, 3, 1, 2, 4).reshape(2, HEADS, BLOCKS * TOKENS, DIM).float()
    )
    query = q.reshape(len(lengths), HEADS, GQA, DIM).float()
    logits = torch.einsum("bhgd,htd->bhgt", query, key) * SCALE
    # Repeating a token n times contributes n * exp(logit) to both numerator
    # and denominator. Adding log(n) gives exactly the same dense attention,
    # with memory bounded by physical pages instead of logical context length.
    logits += multiplicity.log()[:, None, None, :]
    padding = torch.from_numpy(lengths == 0).to(q.device)
    logits[padding] = 0
    reference = torch.einsum("bhgt,htd->bhgd", logits.softmax(-1), value)
    reference[padding] = 0
    return reference.reshape_as(q)


def _decode_layers(
    store: HbmKvStore,
    queries: list[torch.Tensor],
    outputs: list[torch.Tensor],
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    units: int,
    workspace: Workspace,
) -> None:
    for layer in range(LAYERS):
        decode_attention(
            queries[layer],
            RecordTables(*store.layer_tables(layer)),
            block_table,
            lengths,
            SCALE,
            units,
            outputs[layer],
            workspace,
            replan=layer == 0,
        )


def _check_outputs(
    queries: list[torch.Tensor],
    outputs: list[torch.Tensor],
    source: torch.Tensor,
    block_table: NDArray[np.int32],
    lengths: NDArray[np.int32],
) -> None:
    padding = torch.from_numpy(lengths == 0).to(source.device)
    for layer in range(LAYERS):
        reference = _reference(queries[layer], source[layer], block_table, lengths)
        torch.testing.assert_close(
            outputs[layer].float(), reference, atol=2e-3, rtol=1e-2
        )
        # Zero-length rows must be explicitly overwritten, including after a
        # previous replay populated their output and split-KV workspace.
        assert torch.count_nonzero(outputs[layer][padding]).item() == 0


@pytest.mark.parametrize("batch_case", ["below_wave", "at_wave", "above_wave", "tiled"])
def test_ragged_batches_match_fp32_attention(
    encoded_store: StoreFixture, batch_case: str
) -> None:
    """Check uneven contexts at occupancy boundaries and beyond 256 rows.

    Args:
        encoded_store (StoreFixture): Encoded KV and owning tensors.
        batch_case (str): Batch-size boundary to exercise.

    Returns:
        None.

    Async/thread-safety:
        Runs sequentially on the current CUDA stream; assertions synchronize.
    """
    store, _arenas, source = encoded_store
    sms = torch.cuda.get_device_properties(source.device).multi_processor_count
    wave = max(1, 2 * sms // HEADS)
    batch = {
        "below_wave": max(1, wave - 1),
        "at_wave": wave,
        "above_wave": wave + 1,
        "tiled": max(269, wave + 1),
    }[batch_case]
    table = _block_table(batch)
    lengths = _ragged_lengths(batch)
    units = decode_units(batch, HEADS, sms)
    generator = torch.Generator(device=source.device).manual_seed(325 + batch)
    queries = [
        torch.randn(
            batch,
            HEADS * GQA,
            DIM,
            dtype=torch.bfloat16,
            device=source.device,
            generator=generator,
        )
        for _ in range(LAYERS)
    ]
    outputs = [torch.full_like(q, float("nan")) for q in queries]
    _decode_layers(
        store,
        queries,
        outputs,
        torch.from_numpy(table).to(source.device),
        torch.from_numpy(lengths).to(source.device),
        units,
        _workspace(units, source.device),
    )
    _check_outputs(queries, outputs, source, table, lengths)


@pytest.mark.parametrize("units_policy", ["auto", "minimum"])
def test_graph_replans_padding_and_reuses_plan_across_layers(
    encoded_store: StoreFixture, units_policy: str
) -> None:
    """Replay one graph with changing lengths, empty rows, and an empty batch.

    Args:
        encoded_store (StoreFixture): Encoded KV and owning tensors.
        units_policy (str): Automatic split capacity or the public minimum
            of one reserved unit per batch row.

    Returns:
        None.

    Async/thread-safety:
        Warms up on a side stream, captures fixed tensor addresses and shapes,
        then replays and checks on the current CUDA stream sequentially.
    """
    store, _arenas, source = encoded_store
    batch = 269
    table = _block_table(batch)
    initial = _ragged_lengths(batch)
    changed = np.roll(initial, 17)
    changed[0] = changed[256] = 0
    changed[257] = MAX_PAGES * TOKENS - 11
    empty = np.zeros(batch, dtype=np.int32)
    sparse = empty.copy()
    sparse[[0, 256, batch - 1]] = [1, TOKENS + 1, MAX_PAGES * TOKENS - 17]
    device_table = torch.from_numpy(table).to(source.device)
    device_lengths = torch.from_numpy(initial).to(source.device)
    sms = torch.cuda.get_device_properties(source.device).multi_processor_count
    units = decode_units(batch, HEADS, sms) if units_policy == "auto" else batch
    workspace = _workspace(units, source.device)
    generator = torch.Generator(device=source.device).manual_seed(401)
    queries = [
        torch.randn(
            batch,
            HEADS * GQA,
            DIM,
            dtype=torch.bfloat16,
            device=source.device,
            generator=generator,
        )
        for _ in range(LAYERS)
    ]
    outputs = [torch.full_like(q, float("nan")) for q in queries]
    args = store, queries, outputs, device_table, device_lengths, units, workspace
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _decode_layers(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _decode_layers(*args)
    for lengths in (initial, changed, empty, sparse):
        device_lengths.copy_(torch.from_numpy(lengths))
        for output in outputs:
            output.fill_(float("nan"))
        graph.replay()
        _check_outputs(queries, outputs, source, table, lengths)
