# SPDX-License-Identifier: Apache-2.0
"""Hold host source pages until delayed destination copies complete."""

# Standard
import asyncio
from pathlib import Path
from typing import Any

# Third Party
import pytest

# First Party
from daser.transfer import Move
from daser.transfer.iouring import TieredIOUringTransferLayer, copy_ops
from tests.transfer.store_files import create_store_file

PAGE = 4096


class DeferredDestination:
    """Model a DMA queue whose source pointers are consumed only at completion."""

    def __init__(self, nbytes: int = PAGE) -> None:
        self.data = bytearray(nbytes)
        self.chunks: list[copy_ops.CopyChunk] = []
        self.submitted = asyncio.Event()
        self.ready = False

    @property
    def nbytes(self) -> int:
        """Return the destination size validated by the transfer layer."""
        return len(self.data)

    @property
    def done(self) -> bool:
        """Return whether the controlled copy has consumed its source pages."""
        return self.ready

    def complete(self) -> None:
        """Consume borrowed source pages, then publish device completion."""
        if self.ready:
            return
        for offset, source, source_offset, size in self.chunks:
            self.data[offset : offset + size] = source.view()[
                source_offset : source_offset + size
            ]
        self.ready = True


def _deferred_copies(
    monkeypatch: pytest.MonkeyPatch,
    event: Any,
    enqueue: Any,
    record: Any,
) -> None:
    """Route copies into ``DeferredDestination`` objects through copy_ops."""
    original_copy = copy_ops.copy_grouped_to_dst

    def copy(dst: Any, chunks: list[copy_ops.CopyChunk]) -> int:
        if isinstance(dst, DeferredDestination):
            return enqueue(dst, chunks)
        return original_copy(dst, chunks)

    monkeypatch.setattr(copy_ops, "destination_copy_event", event)
    monkeypatch.setattr(copy_ops, "copy_grouped_to_dst", copy)
    monkeypatch.setattr(copy_ops, "record_destination_copy_event", record)


def _layer(path: Path, l1: int, staging: int, l2: int) -> TieredIOUringTransferLayer:
    """Build a layer over ``path``, which must already hold the L2 bytes."""
    return TieredIOUringTransferLayer(
        create_store_file(str(path), l2),
        l1_bytes=l1,
        staging_bytes=staging,
        l2_bytes=l2,
    )


@pytest.mark.parametrize("source", ["l1", "l2"])
@pytest.mark.parametrize("outcome", ["success", "cancel", "submit_error"])
def test_delayed_h2d_keeps_host_source_until_copy_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, outcome: str
) -> None:
    """A load returns, and staging is reused, only after the DMA consumed it."""

    def event(dst: Any, chunks: list[copy_ops.CopyChunk]) -> Any | None:
        return dst if isinstance(dst, DeferredDestination) else None

    def enqueue(dst: DeferredDestination, chunks: list[copy_ops.CopyChunk]) -> int:
        dst.chunks.extend(chunks)
        if outcome == "submit_error":
            raise RuntimeError("partial DMA submission")
        return len(chunks)

    def record(dst: DeferredDestination, completion: DeferredDestination) -> None:
        assert dst is completion
        dst.submitted.set()

    _deferred_copies(monkeypatch, event, enqueue, record)

    async def scenario() -> None:
        path = tmp_path / "copy-lifetime.store"
        path.write_bytes(b"a" * PAGE + b"b" * PAGE)
        layer = _layer(path, PAGE, PAGE, 2 * PAGE)
        if source == "l1":
            await layer.promote([Move(0, PAGE, l1_offset=0, l2_offset=0)])
        move = (
            Move(0, PAGE, l1_offset=0) if source == "l1" else Move(0, PAGE, l2_offset=0)
        )
        dst = DeferredDestination()
        loading = asyncio.create_task(layer.load(dst, [move], source))  # type: ignore[arg-type]
        reusing: asyncio.Task[None] | None = None
        try:
            await asyncio.wait_for(dst.submitted.wait(), timeout=5)
            if outcome == "cancel":
                loading.cancel()
            if source == "l2":
                # The single staging slice is the DMA source: a second staging
                # load must wait until the first copy completed.
                reusing = asyncio.create_task(
                    layer.load(bytearray(PAGE), [Move(0, PAGE, l2_offset=PAGE)], "l2")
                )
            await asyncio.sleep(0.01)
            assert not loading.done()
            if reusing is not None:
                assert not reusing.done()
            dst.complete()
            if outcome == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(loading, timeout=5)
            elif outcome == "submit_error":
                with pytest.raises(RuntimeError, match="partial DMA submission"):
                    await asyncio.wait_for(loading, timeout=5)
            else:
                await asyncio.wait_for(loading, timeout=5)
            if reusing is not None:
                await asyncio.wait_for(reusing, timeout=5)
            assert dst.data == b"a" * PAGE
        finally:
            dst.complete()
            await asyncio.gather(
                loading, *([reusing] if reusing else []), return_exceptions=True
            )
            await layer.close()

    asyncio.run(scenario())


def test_failed_copy_drains_slow_sibling_before_load_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial enqueue failure cannot end a load with another DMA pending."""
    events: list[DeferredDestination] = []

    def event(dst: Any, chunks: list[copy_ops.CopyChunk]) -> Any | None:
        if not isinstance(dst, DeferredDestination):
            return None
        completion = DeferredDestination()
        completion.data = dst.data
        events.append(completion)
        return completion

    def enqueue(dst: DeferredDestination, chunks: list[copy_ops.CopyChunk]) -> int:
        events[-1].chunks.extend(chunks)
        if len(events) == 1:
            raise RuntimeError("first copy submission failed")
        return len(chunks)

    def record(dst: DeferredDestination, completion: DeferredDestination) -> None:
        completion.submitted.set()
        if len(events) == 1:
            completion.complete()
        else:
            dst.submitted.set()

    _deferred_copies(monkeypatch, event, enqueue, record)

    async def scenario() -> None:
        path = tmp_path / "sibling-copy.store"
        path.write_bytes(b"a" * PAGE + b"-" * PAGE + b"b" * PAGE)
        layer = _layer(path, 0, 2 * PAGE, 3 * PAGE)
        dst = DeferredDestination(2 * PAGE)
        loading = asyncio.create_task(
            layer.load(
                dst,
                [Move(0, PAGE, l2_offset=0), Move(PAGE, PAGE, l2_offset=2 * PAGE)],
                "l2",
            )
        )
        try:
            await asyncio.wait_for(dst.submitted.wait(), timeout=5)
            await asyncio.sleep(0.01)
            assert len(events) == 2
            assert not loading.done()
            events[1].complete()
            with pytest.raises(RuntimeError, match="first copy submission failed"):
                await asyncio.wait_for(loading, timeout=5)
            assert dst.data == b"a" * PAGE + b"b" * PAGE
            assert layer.debug_state()["staging_used_bytes"] == 0
        finally:
            for completion in events:
                completion.complete()
            await asyncio.gather(loading, return_exceptions=True)
            await layer.close()

    asyncio.run(scenario())
