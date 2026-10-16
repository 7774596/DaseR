# SPDX-License-Identifier: Apache-2.0

"""Validation, coalescing and splitting of transfer ``Move`` batches.

Backends validate a whole batch before submitting any IO, so a bad move never
leaves part of a batch in flight. Coalescing only joins moves whose requested
coordinates are all contiguous, so byte order and destination offsets never
change.
"""

# Standard
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Literal

# First Party
from daser.transfer.base import Move

#: O_DIRECT alignment of L2 offsets and lengths, and of arena extents that
#: are read from or written to L2.
DIRECT_IO_ALIGNMENT = 4096

#: Which offsets a batch must carry: store (L1 and/or L2), a load from one
#: tier, or a promotion (both).
MoveKind = Literal["store", "l1", "l2", "promote"]

#: Move coordinate a coalescing pass keeps contiguous.
Coordinate = Literal["buffer", "l1", "l2"]


def _coordinate(key: Coordinate) -> Callable[[Move], int | None]:
    """Return the accessor of one move coordinate."""
    if key == "buffer":
        return lambda move: move.buffer_offset
    if key == "l1":
        return lambda move: move.l1_offset
    return lambda move: move.l2_offset


def _continues(before: int | None, after: int | None, nbytes: int) -> bool:
    """Return whether ``after`` continues a run that starts at ``before``."""
    if before is None or after is None:
        return before is None and after is None
    return after == before + nbytes


def buffer_nbytes(buffer: Any) -> int | None:
    """Return a staging buffer's byte length, or None if it is unknown.

    Args:
        buffer: CuPy-like array (``nbytes``) or bytes-like object.

    Returns:
        Byte length, or None for objects exposing neither interface.
    """
    nbytes = getattr(buffer, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    try:
        return memoryview(buffer).nbytes
    except TypeError:
        return None


def validate_moves(
    moves: Sequence[Move],
    kind: MoveKind,
    *,
    buffer_bytes: int | None,
    l1_bytes: int,
    l2_bytes: int,
) -> None:
    """Check every move of a batch before any IO is submitted.

    Args:
        moves: Batch to check.
        kind: Required offsets: ``store`` needs ``l1_offset`` or
            ``l2_offset``, ``l1``/``l2`` need that tier's offset, ``promote``
            needs both.
        buffer_bytes: Staging buffer length, None to skip the buffer bound
            (unknown length, or a promotion without a buffer).
        l1_bytes: Arena size; 0 means the backend has no L1 tier.
        l2_bytes: L2 size; 0 means the backend has no L2 tier.

    Raises:
        ValueError: On a non-positive length, an offset outside its region,
            a tier the backend lacks, a missing required offset, or an L2
            offset, length or paired arena offset that is not 4 KiB aligned.
    """
    for move in moves:
        nbytes = move.nbytes
        if nbytes <= 0:
            raise ValueError(f"move length must be positive: {move}")
        l1, l2 = move.l1_offset, move.l2_offset
        if kind == "store" and l1 is None and l2 is None:
            raise ValueError(f"store move names no tier: {move}")
        if kind in ("l1", "promote") and l1 is None:
            raise ValueError(f"{kind} move needs l1_offset: {move}")
        if kind in ("l2", "promote") and l2 is None:
            raise ValueError(f"{kind} move needs l2_offset: {move}")
        if buffer_bytes is not None and not (
            0 <= move.buffer_offset and move.buffer_offset + nbytes <= buffer_bytes
        ):
            raise ValueError(f"move exceeds the {buffer_bytes}-byte buffer: {move}")
        if l1 is not None:
            if not l1_bytes:
                raise ValueError("transfer backend has no L1 arena")
            if not (0 <= l1 and l1 + nbytes <= l1_bytes):
                raise ValueError(f"move exceeds the {l1_bytes}-byte arena: {move}")
        if l2 is None:
            continue
        if not l2_bytes:
            raise ValueError("transfer backend has no L2 tier")
        if not (0 <= l2 and l2 + nbytes <= l2_bytes):
            raise ValueError(f"move exceeds the {l2_bytes}-byte L2 store: {move}")
        if (
            l2 % DIRECT_IO_ALIGNMENT
            or nbytes % DIRECT_IO_ALIGNMENT
            or (l1 is not None and l1 % DIRECT_IO_ALIGNMENT)
        ):
            raise ValueError(
                f"L2 move offsets and length must be {DIRECT_IO_ALIGNMENT}-byte "
                f"aligned: {move}"
            )


def coalesce_moves(
    moves: Sequence[Move],
    keys: Sequence[Coordinate],
    *,
    max_bytes: int | None = None,
) -> list[Move]:
    """Merge moves that are contiguous in every coordinate of ``keys``.

    Moves are ordered by ``keys`` (unset offsets first). Two neighbours merge
    when, for each key, both offsets are unset or the second starts where the
    first ends, and the merged length stays within ``max_bytes``.

    Args:
        moves: Moves to merge; not modified.
        keys: Coordinates that must stay contiguous, primary sort key first.
        max_bytes: Optional bound on a merged move's length.

    Returns:
        Merged moves in sort order.
    """
    getters = [_coordinate(key) for key in keys]

    def order(move: Move) -> tuple[tuple[bool, int], ...]:
        return tuple(
            (value is not None, value or 0)
            for value in (getter(move) for getter in getters)
        )

    merged: list[Move] = []
    for move in sorted(moves, key=order):
        if merged:
            prev = merged[-1]
            joinable = (
                max_bytes is None or prev.nbytes + move.nbytes <= max_bytes
            ) and all(
                _continues(getter(prev), getter(move), prev.nbytes)
                for getter in getters
            )
            if joinable:
                merged[-1] = replace(prev, nbytes=prev.nbytes + move.nbytes)
                continue
        merged.append(move)
    return merged


def split_move(move: Move, piece_bytes: int) -> list[Move]:
    """Cut a move into consecutive pieces of at most ``piece_bytes``.

    Args:
        move: Move to cut.
        piece_bytes: Positive piece length; a multiple of 4 KiB keeps L2
            pieces O_DIRECT aligned.

    Returns:
        Pieces covering the move in order, each offset advanced together.
    """
    pieces = []
    for start in range(0, move.nbytes, piece_bytes):
        pieces.append(
            Move(
                buffer_offset=move.buffer_offset + start,
                nbytes=min(piece_bytes, move.nbytes - start),
                l1_offset=None if move.l1_offset is None else move.l1_offset + start,
                l2_offset=None if move.l2_offset is None else move.l2_offset + start,
            )
        )
    return pieces
