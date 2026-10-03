# SPDX-License-Identifier: Apache-2.0

import numpy as np

from benchmarks.research.span_reuse_bound import (
    build_trace,
    prefix_hits,
    span_coverage,
    summarize,
)

_RNG = np.random.default_rng(0)


def _text(n: int) -> str:
    return "".join(chr(c) for c in _RNG.integers(ord("a"), ord("z") + 1, n))


SYS1, SYS2, DOC, Q, ANS, DOC2 = (_text(n) for n in (300, 300, 400, 50, 200, 200))
SESSIONS = [
    # A: task t1, one request reading DOC as the user turn.
    {
        "sid": "ds__t1__m1",
        "iters": [[2, 0.0, 1]],
        "msgs": [{"role": "system", "content": SYS1}, {"role": "user", "content": DOC}],
    },
    # B: other task and prompt, reads DOC as a tool result (shifted position).
    {
        "sid": "ds__t2__m1",
        "iters": [[2, 0.0, 1]],
        "msgs": [{"role": "system", "content": SYS2}, {"role": "tool", "content": DOC}],
    },
    # C: same task as A, shares its system prompt; second turn after decode.
    {
        "sid": "ds__t1__m2",
        "iters": [[2, 0.0, 1], [4, 0.0, 1]],
        "msgs": [
            {"role": "system", "content": SYS1},
            {"role": "user", "content": Q},
            {"role": "assistant", "content": ANS},
            {"role": "user", "content": DOC2},
        ],
    },
]


def _trace():
    return build_trace(SESSIONS, lambda text: [ord(c) for c in text])


def test_prefix_hits_reuse_only_aligned_stored_prefixes() -> None:
    trace = _trace()
    hit = prefix_hits(trace)
    start = trace.session_start
    # A and B share nothing; C's first request reuses A's system prompt
    # (311 rendered tokens) up to the last whole 128-token block.
    assert list(hit[:3] - start[[0, 1, 2]]) == [0, 0, 256]
    # C's second request resumes at its first request's aligned end.
    first_end = trace.req_end[2] - start[2]
    assert hit[3] - start[2] == first_end // 128 * 128


def test_span_coverage_finds_shifted_spans_per_scope() -> None:
    trace = _trace()
    hit = prefix_hits(trace)
    b = slice(trace.session_start[1], trace.session_start[2])
    covered_all = span_coverage(trace, hit, 128, "all")[b].sum()
    assert len(DOC) <= covered_all <= len(DOC) + 3
    assert span_coverage(trace, hit, 128, "task")[b].sum() == 0
    assert span_coverage(trace, hit, 128, "session").sum() == 0
    # A window longer than the shared span finds nothing.
    assert span_coverage(trace, hit, 512, "all").sum() == 0


def test_summarize_excludes_decoded_assistant_tokens() -> None:
    trace = _trace()
    hit = prefix_hits(trace)
    covered = {("all", 128): span_coverage(trace, hit, 128, "all")}
    res = summarize(trace, hit, covered, attn_break_even=1e9)
    assistant = len("<|assistant|>\n") + len(ANS)
    assert res["prefill"] - res["fresh"] == assistant
    assert 0 < res["all@128"] < 1
    assert abs(res["all@128w"] - res["all@128"]) < 1e-6
