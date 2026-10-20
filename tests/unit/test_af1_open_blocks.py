# SPDX-License-Identifier: Apache-2.0
"""OpenBlocks lists exactly the partially written block of each request."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from daser.attention.af1_backend import OpenBlocks  # noqa: E402


def _output(
    new: tuple = (),
    cached: tuple = (),
    finished: frozenset = frozenset(),
    resumed: frozenset = frozenset(),
) -> SimpleNamespace:
    """Build a minimal scheduler output.

    Args:
        new: ``(req_id, block_ids, num_computed_tokens)`` per new request.
        cached: ``(req_id, num_computed_tokens, new_block_ids or None)``.
        finished: Finished request ids.
        resumed: Resumed request ids among ``cached``.

    Returns:
        Object with the fields ``OpenBlocks.update`` reads.
    """
    return SimpleNamespace(
        finished_req_ids=set(finished),
        preempted_req_ids=None,
        scheduled_new_reqs=[
            SimpleNamespace(req_id=r, block_ids=(list(b),), num_computed_tokens=c)
            for r, b, c in new
        ],
        block_table_updates=None,
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[r for r, _, _ in cached],
            num_computed_tokens=[c for _, c, _ in cached],
            new_block_ids=[None if b is None else (list(b),) for _, _, b in cached],
            resumed_req_ids=set(resumed),
        ),
    )


def test_lists_only_partially_written_blocks() -> None:
    blocks = OpenBlocks()
    blocks.update(_output(new=(("a", [10, 11, 12], 0), ("b", [20], 0))))
    assert blocks.live_blocks().tolist() == []
    blocks.update(_output(cached=(("a", 200, None), ("b", 128, None))))
    assert blocks.live_blocks().tolist() == [11]
    blocks.update(_output(cached=(("a", 300, None), ("b", 129, [21]))))
    assert sorted(blocks.live_blocks().tolist()) == [12, 21]


def test_reallocated_open_block_is_not_live_ahead_of_its_writer() -> None:
    blocks = OpenBlocks()
    blocks.update(_output(new=(("a", [10, 11], 0),)))
    blocks.update(_output(cached=(("a", 200, None),)))
    assert blocks.live_blocks().tolist() == [11]
    # ``a`` finishes with block 11 open; ``b`` gets 11 as its second block
    # and has written part of its first.
    blocks.update(_output(new=(("b", [30, 11], 0),), finished=frozenset({"a"})))
    blocks.update(_output(cached=(("b", 50, None),)))
    assert blocks.live_blocks().tolist() == [30]


def test_resumed_request_replaces_its_blocks() -> None:
    blocks = OpenBlocks()
    blocks.update(_output(new=(("a", [10, 11], 0),)))
    blocks.update(_output(cached=(("a", 150, None),)))
    blocks.update(_output(cached=(("a", 130, [40, 41]),), resumed=frozenset({"a"})))
    assert blocks.live_blocks().tolist() == [41]
