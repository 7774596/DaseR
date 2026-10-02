# SPDX-License-Identifier: Apache-2.0
"""Predict which agent session returns next from the lookup stream."""

from collections import OrderedDict

# Weight of a new return gap in the moving average.
_GAP_EWMA_WEIGHT = 0.25


class SessionPredictor:
    """Track sessions by prompt prefix and predict the next one to return.

    A scheduler lookup whose prompt starts with an earlier lookup's prompt
    (its leading ``match_fraction``) continues that session; otherwise it
    opens a new one. Repeated lookups of one prompt are a single request.
    Agent sessions alternate between a request and a tool call, so the
    session idle the longest is the next to return, unless it has been idle
    far longer than a typical return gap and is presumed finished.

    Args:
        match_fraction: Leading share of a remembered prompt that a new
            prompt must repeat to continue its session.
        stale_factor: A session idle for more than this many average return
            gaps is not predicted.
        max_sessions: Remembered sessions; the least recent are forgotten.

    Async/thread-safety:
        Not thread-safe; the owner (the IPC event loop) serializes calls.
    """

    def __init__(
        self,
        match_fraction: float = 0.95,
        stale_factor: float = 1.5,
        max_sessions: int = 64,
    ) -> None:
        if not 0.0 < match_fraction <= 1.0 or stale_factor <= 1.0 or max_sessions <= 0:
            raise ValueError(
                "match_fraction must be in (0, 1], stale_factor > 1, max_sessions > 0"
            )
        self._match_fraction = match_fraction
        self._stale_factor = stale_factor
        self._max_sessions = max_sessions
        # session -> (prompt bytes, model id, request sequence of last use)
        self._sessions: OrderedDict[int, tuple[bytes, str, int]] = OrderedDict()
        self._next_session = 0
        self._seq = 0
        self._gap = 0.0
        self._current: int | None = None
        self._predicted: int | None = None
        self.requests = 0
        self.predicted_returns = 0

    def observe(self, tokens: bytes, model_id: str) -> bool:
        """Record one scheduler lookup.

        Args:
            tokens: Packed prompt token IDs.
            model_id: Model the prompt belongs to.

        Returns:
            True when this lookup is a new request, False when it repeats
            the current request's prompt.
        """
        session = self._match(tokens, model_id)
        if session is not None and self._sessions[session][0] == tokens:
            return False
        self._seq += 1
        self.requests += 1
        if session is None:
            session = self._next_session
            self._next_session += 1
        else:
            gap = float(self._seq - self._sessions[session][2])
            self._gap = (
                gap
                if self._gap <= 0.0
                else self._gap + _GAP_EWMA_WEIGHT * (gap - self._gap)
            )
            if session == self._predicted:
                self.predicted_returns += 1
        self._sessions[session] = (tokens, model_id, self._seq)
        self._sessions.move_to_end(session)
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
        self._current = session
        self._predicted = None
        return True

    def predict(self) -> tuple[bytes, str] | None:
        """Return the prompt and model of the session expected next.

        Returns:
            The least recently used live session other than the current
            one, or None when there is none or no return gap is known yet.
        """
        if self._gap <= 0.0:
            return None
        horizon = self._stale_factor * self._gap
        for session, (tokens, model_id, seq) in self._sessions.items():
            if session == self._current or self._seq - seq > horizon:
                continue
            self._predicted = session
            return tokens, model_id
        return None

    def _match(self, tokens: bytes, model_id: str) -> int | None:
        """Return the most recent session whose prompt ``tokens`` continues."""
        for session in reversed(self._sessions):
            seen, seen_model, _seq = self._sessions[session]
            if seen_model != model_id or len(seen) > len(tokens):
                continue
            prefix = int(len(seen) * self._match_fraction) // 4 * 4
            if prefix and tokens[:prefix] == seen[:prefix]:
                return session
        return None
