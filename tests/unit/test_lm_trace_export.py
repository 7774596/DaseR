# SPDX-License-Identifier: Apache-2.0
"""Tests for the real-text agent session trace exporter."""

from benchmarks.research.lm_trace_export import session_rows, session_span_s


def _encode(text: str) -> list[int]:
    return [ord(c) for c in text]


def test_session_rows_are_prefixes_ending_at_the_assistant_header() -> None:
    """Iteration prompts end after the assistant header of the next reply."""
    session = {
        "msgs": [
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "ok", "tool_calls": [{"f": 1}]},
            {"role": "tool", "content": "done"},
            {"role": "assistant", "content": "bye"},
        ],
        "iters": [[1, 0.0, 9], [3, 2.5, 4]],
    }
    header = _encode("<|assistant|>\n")
    stream, rows = session_rows(session, _encode, header, 10_000, 5, 1.0)

    text = "".join(map(chr, stream))
    first, second = rows
    assert text[: first["input_length"]] == "<|user|>\nfix<|assistant|>\n"
    assert text[: second["input_length"]].endswith("<|tool|>\ndone<|assistant|>\n")
    assert (first["output_length"], second["output_length"]) == (5, 4)
    assert (first["gap_ms"], second["gap_ms"]) == (0, 1000)


def test_session_rows_drop_iterations_past_the_context_limit() -> None:
    """Prompts whose input plus output exceed max_ctx are skipped."""
    session = {
        "msgs": [{"role": "user", "content": "x" * 50}],
        "iters": [[1, 0.0, 10]],
    }
    _, rows = session_rows(session, _encode, _encode("<|assistant|>\n"), 40, 0, 60.0)
    assert rows == []


def test_session_span_counts_gaps_and_decode_time() -> None:
    """Spacing uses idle gaps plus output tokens at the decode step time."""
    rows = [{"gap_ms": 0, "output_length": 100}, {"gap_ms": 2000, "output_length": 50}]
    assert session_span_s(rows, 0.02) == 2.0 + 3.0
