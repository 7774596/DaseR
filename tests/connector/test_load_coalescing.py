# SPDX-License-Identifier: Apache-2.0
"""In-flight load coalescing in the scheduler-side request lifecycle."""

from types import SimpleNamespace

import pytest

from daser.connector.scheduler import lifecycle as lifecycle_module
from daser.connector.scheduler.lifecycle import COALESCE_MAX_WAIT_S, RequestLifecycle

PREFIX = list(range(8))


class _PrefixIPC:
    """IPC stub whose store holds an 8-token prefix."""

    def __init__(self) -> None:
        self.lookups: list[int] = []

    def lookup(
        self,
        tokens,
        model_id,
        external_prefix_queries=None,
        num_computed_tokens=0,
        defer_pending=None,
    ):
        del model_id, external_prefix_queries, defer_pending
        self.lookups.append(num_computed_tokens)
        if list(tokens[:8]) != PREFIX:
            return []
        return [
            {
                "chunk_key": "doc",
                "start_slot": 0,
                "num_slots": 2,
                "file_offset": 0,
                "token_count": 8,
                "target_token_start": 0,
                "pos_offset": 0,
            }
        ]


def _lifecycle(ipc: _PrefixIPC, *, coalesce: bool) -> RequestLifecycle:
    return RequestLifecycle(
        ipc_client=ipc,
        block_tokens=4,
        slot_size=32,
        model_id="model",
        cache_reuse_mode="chunk",
        runtime_config_ready=True,
        coalesce_loads=coalesce,
    )


def _request(req_id: str, tail: list[int], prefix: list[int] = PREFIX):
    return SimpleNamespace(
        request_id=req_id,
        prompt_token_ids=prefix + tail,
        kv_transfer_params={"daser_skip_save": True},
    )


def _step(lifecycle: RequestLifecycle, finished_recving: tuple[str, ...] = ()):
    """Report worker completions, then build the next step's metadata."""
    lifecycle.update_connector_output(
        SimpleNamespace(finished_sending=(), finished_recving=finished_recving)
    )
    lifecycle.build_connector_meta(SimpleNamespace(num_scheduled_tokens={}))


def test_disabled_coalescing_loads_the_same_prefix_twice() -> None:
    ipc = _PrefixIPC()
    lifecycle = _lifecycle(ipc, coalesce=False)

    assert lifecycle.get_num_new_matched_tokens(_request("a", [100] * 4), 0) == (
        8,
        True,
    )
    assert lifecycle.get_num_new_matched_tokens(_request("b", [200] * 4), 0) == (
        8,
        True,
    )


def test_follower_waits_for_inflight_prefix_then_uses_local_hit() -> None:
    ipc = _PrefixIPC()
    lifecycle = _lifecycle(ipc, coalesce=True)
    leader = _request("a", [100] * 4)
    follower = _request("b", [200] * 4)

    assert lifecycle.get_num_new_matched_tokens(leader, 0) == (8, True)
    # Same prefix still loading: wait instead of allocating a second copy.
    assert lifecycle.get_num_new_matched_tokens(follower, 0) == (None, False)
    assert ipc.lookups == [0]

    # The leader's worker finished; vLLM caches its blocks on the next
    # schedule(), which may still visit the follower first.
    _step(lifecycle, finished_recving=("a",))
    assert lifecycle.get_num_new_matched_tokens(follower, 0) == (None, False)

    # Once the landed prefix is a local hit, only the tail is looked up.
    assert lifecycle.get_num_new_matched_tokens(follower, 8) == (0, False)
    assert ipc.lookups == [0, 8]


def test_different_prefix_does_not_wait() -> None:
    ipc = _PrefixIPC()
    lifecycle = _lifecycle(ipc, coalesce=True)

    assert lifecycle.get_num_new_matched_tokens(_request("a", [100] * 4), 0) == (
        8,
        True,
    )
    other = _request("c", [100] * 4, prefix=[9] * 8)
    assert lifecycle.get_num_new_matched_tokens(other, 0) == (0, False)


def test_follower_loads_itself_after_landed_load_expires() -> None:
    ipc = _PrefixIPC()
    lifecycle = _lifecycle(ipc, coalesce=True)
    follower = _request("b", [200] * 4)

    lifecycle.get_num_new_matched_tokens(_request("a", [100] * 4), 0)
    assert lifecycle.get_num_new_matched_tokens(follower, 0) == (None, False)
    _step(lifecycle, finished_recving=("a",))
    _step(lifecycle)
    _step(lifecycle)

    # The landed blocks were evicted before the follower ran: load normally.
    assert lifecycle.get_num_new_matched_tokens(follower, 0) == (8, True)


def test_finished_leader_releases_its_followers() -> None:
    ipc = _PrefixIPC()
    lifecycle = _lifecycle(ipc, coalesce=True)
    leader = _request("a", [100] * 4)
    follower = _request("b", [200] * 4)

    lifecycle.get_num_new_matched_tokens(leader, 0)
    assert lifecycle.get_num_new_matched_tokens(follower, 0) == (None, False)
    lifecycle.request_finished(leader, [])

    assert lifecycle.get_num_new_matched_tokens(follower, 0) == (8, True)


def test_follower_stops_waiting_for_a_stalled_leader(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(lifecycle_module.time, "monotonic", lambda: clock[0])
    ipc = _PrefixIPC()
    lifecycle = _lifecycle(ipc, coalesce=True)
    follower = _request("b", [200] * 4)

    lifecycle.get_num_new_matched_tokens(_request("a", [100] * 4), 0)
    assert lifecycle.get_num_new_matched_tokens(follower, 0) == (None, False)
    clock[0] += COALESCE_MAX_WAIT_S / 2
    assert lifecycle.get_num_new_matched_tokens(follower, 0) == (None, False)
    clock[0] += COALESCE_MAX_WAIT_S

    assert lifecycle.get_num_new_matched_tokens(follower, 0) == (8, True)


@pytest.mark.parametrize("computed", [8, 12])
def test_request_already_covering_the_prefix_does_not_wait(computed: int) -> None:
    ipc = _PrefixIPC()
    lifecycle = _lifecycle(ipc, coalesce=True)

    lifecycle.get_num_new_matched_tokens(_request("a", [100] * 4), 0)
    result = lifecycle.get_num_new_matched_tokens(_request("b", [200] * 8), computed)

    assert result[0] is not None
