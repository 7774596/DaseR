# SPDX-License-Identifier: Apache-2.0

"""Next-session prediction from the scheduler lookup stream."""

import struct

import pytest

from daser.server.session_predictor import SessionPredictor

_MODEL = "m"


def _prompt(session: int, turn: int) -> bytes:
    """Return a prompt that extends the session's previous turn."""
    tokens = [session * 1000 + i for i in range(8 * (turn + 1))]
    return struct.pack(f"{len(tokens)}i", *tokens)


def test_rejects_invalid_parameters() -> None:
    """Fractions, factors and capacity are validated."""
    with pytest.raises(ValueError):
        SessionPredictor(match_fraction=0.0)
    with pytest.raises(ValueError):
        SessionPredictor(stale_factor=1.0)
    with pytest.raises(ValueError):
        SessionPredictor(max_sessions=0)


def test_no_prediction_before_a_return_is_seen() -> None:
    """Without an observed return gap nothing is predicted."""
    predictor = SessionPredictor()
    predictor.observe(_prompt(0, 0), _MODEL)
    predictor.observe(_prompt(1, 0), _MODEL)
    assert predictor.predict() == []


def test_round_robin_sessions_predict_the_next_return() -> None:
    """Interleaved sessions predict the one idle the longest, every time."""
    predictor = SessionPredictor()
    order = [(turn, session) for turn in range(4) for session in range(3)]
    for index, (turn, session) in enumerate(order):
        assert predictor.observe(_prompt(session, turn), _MODEL)
        predicted = predictor.predict()
        if turn == 0:
            # No session has returned yet, so no gap is known.
            assert predicted == []
        elif index + 1 < len(order):
            next_turn, next_session = order[index + 1]
            # The next session's last seen prompt is one turn behind.
            assert predicted == [(_prompt(next_session, next_turn - 1), _MODEL)]
    assert predictor.requests == 12
    # Every return after the first prediction was predicted.
    assert predictor.predicted_returns == 8


def test_repeated_lookup_is_one_request() -> None:
    """Scheduler retries of the same prompt do not count as returns."""
    predictor = SessionPredictor()
    assert predictor.observe(_prompt(0, 0), _MODEL)
    assert not predictor.observe(_prompt(0, 0), _MODEL)
    assert predictor.requests == 1


def test_finished_sessions_stop_being_predicted() -> None:
    """A session idle far beyond the return gap is skipped."""
    predictor = SessionPredictor()
    for turn in range(3):
        for session in (0, 1):
            predictor.observe(_prompt(session, turn), _MODEL)
    # Session 0 ends; sessions 2 and 3 alternate from now on.
    for turn in range(4):
        for session in (2, 3):
            predictor.observe(_prompt(session, turn), _MODEL)
    assert predictor.predict() == [(_prompt(2, 3), _MODEL)]


def test_models_do_not_share_sessions() -> None:
    """Equal prompts under another model open a separate session."""
    predictor = SessionPredictor()
    assert predictor.observe(_prompt(0, 0), _MODEL)
    assert predictor.observe(_prompt(0, 0), "other")
    assert predictor.requests == 2


def test_deeper_prediction_lists_sessions_in_return_order() -> None:
    """Depth k lists the next k sessions to return, soonest first."""
    predictor = SessionPredictor()
    for turn in range(2):
        for session in range(4):
            predictor.observe(_prompt(session, turn), _MODEL)
    assert predictor.predict(depth=2) == [
        (_prompt(0, 1), _MODEL),
        (_prompt(1, 1), _MODEL),
    ]
    # Session 1 returns: it was the second prediction, so it counts.
    returned = predictor.predicted_returns
    predictor.observe(_prompt(1, 2), _MODEL)
    assert predictor.predicted_returns == returned + 1
