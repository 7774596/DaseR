# SPDX-License-Identifier: Apache-2.0

# Standard
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

# First Party
from benchmarks.research.counter_trace import CounterTrace, classify

_SSD = 'daser_l2_read_bytes_total{inflight="1",write="0"}'
_SSD_W = 'daser_l2_read_bytes_total{inflight="4",write="1"}'


_DROPPED = 'daser_readahead_bytes_total{outcome="dropped"}'


def _sample(
    t: float, hits: float, ssd: float, ssd_w: float = 0.0, dropped: float = 0.0
) -> dict:
    return {
        "t": t,
        "c": {
            "daser_l1_hits_total": hits,
            _SSD: ssd,
            _SSD_W: ssd_w,
            _DROPPED: dropped,
        },
    }


def _request(sample_id: int, sent: float, ttft_s: float) -> dict:
    return {
        "sample_id": sample_id,
        "sent_at_s": sent,
        "first_token_at_s": sent + ttft_s,
    }


def test_classify_names_tier_from_bracketing_samples() -> None:
    samples = [
        _sample(0.0, 0, 0),
        _sample(1.0, 0, 0),
        _sample(2.0, 3, 0),  # request 1: L1 hits only
        _sample(3.0, 3, 100, 50),  # request 2: SSD bytes over two buckets
        _sample(4.0, 5, 200),  # request 3: both
        _sample(5.0, 5, 200),  # request 4: neither
    ]
    requests = [
        _request(1, 1.1, 0.5),
        _request(2, 2.1, 0.5),
        _request(3, 3.1, 0.5),
        _request(4, 4.1, 0.5),
    ]
    rows = classify(requests, samples)
    assert [row["tier"] for row in rows] == ["l1", "l2", "mixed", "none"]
    assert rows[1]["ssd_read_bytes"] == 150
    assert rows[2]["l1_hits"] == 2
    assert all(row["clean"] for row in rows)


def test_classify_flags_overlapping_windows_and_dropped_readahead() -> None:
    samples = [
        _sample(0.0, 0, 0),
        _sample(1.0, 1, 0),
        _sample(2.0, 1, 0, dropped=64),
        _sample(3.0, 1, 0, dropped=64),
        _sample(4.0, 1, 0, dropped=64),
    ]
    # Requests 1 and 2 load in the same window; request 3 dropped readahead;
    # request 4 is alone in its window.
    requests = [
        _request(1, 0.2, 0.5),
        _request(2, 0.4, 0.3),
        _request(3, 1.2, 0.5),
        _request(4, 3.1, 0.5),
    ]
    clean = {row["sample_id"]: row["clean"] for row in classify(requests, samples)}
    assert clean == {1: False, 2: False, 3: False, 4: True}


def test_classify_skips_requests_without_bracketing_samples() -> None:
    samples = [_sample(1.0, 0, 0), _sample(2.0, 1, 0)]
    requests = [_request(1, 0.5, 0.1), _request(2, 1.5, 1.0), _request(3, 1.2, 0.1)]
    assert [row["sample_id"] for row in classify(requests, samples)] == [3]


def test_classify_names_tier_from_ssd_byte_share() -> None:
    """With stored bytes per token, the SSD byte share names the tier."""
    samples = [
        _sample(float(t), 4 * t, 900 * min(t, 2) + 100 * (t >= 3)) for t in range(5)
    ]
    requests = [
        _request(1, 0.1, 0.5),  # 900 SSD bytes of 1000 and L1 hits: l2
        _request(2, 1.1, 0.5),  # 900 of 1000: l2
        _request(3, 2.1, 0.5),  # 100 of 1000 with L1 hits: l1
        _request(4, 3.1, 0.5),  # 0 SSD bytes, L1 hits only: l1
    ]
    for record in requests:
        record["prompt_tokens"] = 10
    rows = classify(requests, samples, l2_bytes_per_token=100.0)
    assert [row["tier"] for row in rows] == ["l2", "l2", "l1", "l1"]
    assert rows[0]["ssd_share"] == 0.9
    assert classify(requests, samples)[0]["tier"] == "mixed"
    half = [dict(requests[0], prompt_tokens=18)]
    assert classify(half, samples, l2_bytes_per_token=100.0)[0]["tier"] == "mixed"


class _Metrics(BaseHTTPRequestHandler):
    hits = 0

    def do_GET(self) -> None:  # noqa: N802
        _Metrics.hits += 1
        body = (
            "# HELP daser_l1_hits_total L1 memory cache hits.\n"
            f"daser_l1_hits_total {_Metrics.hits}\n"
            f"{_SSD} 4096\n"
            "daser_ipc_requests_total 7\n"
        ).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        return


def test_counter_trace_records_only_tier_counters(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Metrics)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    path = tmp_path / "counters.jsonl"
    trace = CounterTrace(f"http://127.0.0.1:{server.server_address[1]}", path, 0.01)
    start = time.perf_counter()
    trace.start()
    time.sleep(0.1)
    trace.stop()
    server.shutdown()
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(samples) >= 2
    assert all(sample["t"] >= start for sample in samples)
    assert set(samples[-1]["c"]) == {"daser_l1_hits_total", _SSD}
    hits = [sample["c"]["daser_l1_hits_total"] for sample in samples]
    assert hits == sorted(hits)
