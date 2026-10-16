# SPDX-License-Identifier: Apache-2.0

"""Cancellation-safe await helpers shared by transfer backends."""

# Standard
import asyncio
from collections.abc import Awaitable, Sequence
from typing import Any


async def settle(awaitables: Sequence[Awaitable[Any]]) -> None:
    """Wait for every awaitable, deferring cancellation until all ended.

    Cancelling an executor await or a copy poll does not stop the underlying
    IO or DMA, so buffers must stay owned until the real work ends.

    Args:
        awaitables: Futures or tasks to wait for; none is cancelled.

    Raises:
        BaseException: The first failure among ``awaitables``, else a
            cancellation of the caller received while waiting.
    """
    if not awaitables:
        return
    pending = asyncio.gather(*awaitables, return_exceptions=True)
    cancellation: asyncio.CancelledError | None = None
    while not pending.done():
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError as exc:
            cancellation = exc
    for result in pending.result():
        if isinstance(result, BaseException):
            raise result
    if cancellation is not None:
        raise cancellation


def resolve(future: asyncio.Future[None], error: BaseException | None) -> None:
    """Complete ``future`` with ``error`` (or success) if still pending.

    Args:
        future: Handle future to complete.
        error: Failure to report; a cancellation cancels the future.
    """
    if future.done():
        return
    if error is None:
        future.set_result(None)
    elif isinstance(error, asyncio.CancelledError):
        future.cancel()
    else:
        future.set_exception(error)


def consume_exception(future: asyncio.Future[None]) -> None:
    """Done callback marking a handle future's exception as retrieved.

    Args:
        future: Completed handle future; ``drain`` reports its error.
    """
    if not future.cancelled():
        future.exception()
