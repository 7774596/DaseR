# SPDX-License-Identifier: Apache-2.0

"""Watch a running agent-trace arm and stop it on the first anomaly.

Runs next to ``agent_trace_arms`` in the background. It is silent while the
arm is healthy, rewrites a one-line status file every tick, and exits with a
single line when the replay finishes (exit 0) or an anomaly is detected
(exit 1). On an anomaly it sends SIGTERM to the replay client so the runner
collects artifacts, stops services, and deletes the store as usual.

Checks per tick:
- failed requests in ``requests.jsonl``;
- new ``Traceback``/``ERROR`` lines in the arm's service logs;
- a stall: no prompt tokens for ``--stall-minutes`` while requests are
  waiting, or no prompt and no generation tokens while requests are running;
- completions in the trailing window below ``--rate-ratio`` of a baseline
  run's completions in the same window;
- a service process exiting while the replay still runs;
- free disk below the thresholds.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
import urllib.request

_METRIC = re.compile(r"^(vllm:[a-z_]+)(?:\{[^}]*\})? ([0-9.eE+-]+)$")
_LOG_ALARM = re.compile(r"Traceback|\bERROR\b")


def _pid_running(pid: int) -> bool:
    """Return whether ``pid`` is alive and not a zombie."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _group_running(pgid: int) -> bool:
    """Return whether any live, non-zombie process is in group ``pgid``.

    Services lead their own process group, so a wrapper (``nsys profile``)
    exiting while the server it started still runs is not a service exit.
    """
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if fields[0] != "Z" and int(fields[2]) == pgid:
            return True
    return False


def _find_replay_pid(output_dir: Path) -> int | None:
    """Return the PID of the replay client writing to ``output_dir``."""
    needle = str(output_dir.resolve())
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [part.decode(errors="replace") for part in argv]
        if "benchmarks.mooncake.benchmark" in args and needle in args:
            return int(proc.name)
    return None


def _vllm_counters(url: str) -> dict[str, float] | None:
    """Return summed vLLM counters/gauges, or None when unreachable."""
    try:
        with urllib.request.urlopen(f"{url}/metrics", timeout=10) as response:
            text = response.read().decode()
    except OSError:
        return None
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = _METRIC.match(line)
        if match:
            totals[match[1]] = totals.get(match[1], 0.0) + float(match[2])
    return totals


class _Tail:
    """Incrementally read new complete lines from a growing file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.partial = b""

    def new_lines(self) -> list[str]:
        """Return complete lines appended since the previous call."""
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                data = handle.read()
        except FileNotFoundError:
            return []
        self.offset += len(data)
        data = self.partial + data
        lines = data.split(b"\n")
        self.partial = lines.pop()
        return [line.decode(errors="replace") for line in lines]


def _baseline_offsets(path: Path | None) -> list[float]:
    """Return sorted completion offsets of a finished baseline run."""
    if path is None:
        return []
    offsets = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("error") is None:
            offsets.append(float(record["completion_offset_s"]))
    return sorted(offsets)


def _count_window(offsets: list[float], lo: float, hi: float) -> int:
    """Return how many offsets fall inside ``[lo, hi)``."""
    return sum(lo <= value < hi for value in offsets)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse watchdog arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Arm artifact dir that receives requests.jsonl.",
    )
    parser.add_argument(
        "--service-dir",
        required=True,
        type=Path,
        help="Arm scratch dir holding logs/ and pids.json.",
    )
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--baseline", type=Path, help="requests.jsonl of a same-config baseline run."
    )
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--stall-minutes", type=float, default=5.0)
    parser.add_argument("--rate-window", type=float, default=300.0)
    parser.add_argument("--rate-ratio", type=float, default=0.7)
    parser.add_argument(
        "--rate-min-baseline",
        type=int,
        default=30,
        help="Skip the rate check when the baseline window is smaller.",
    )
    parser.add_argument(
        "--rate-strikes",
        type=int,
        default=2,
        help="Consecutive low-rate ticks before an anomaly.",
    )
    parser.add_argument("--data-path", default="/data")
    parser.add_argument("--min-free-data-gib", type=float, default=500.0)
    parser.add_argument("--min-free-root-gib", type=float, default=20.0)
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=1800.0,
        help="Seconds to wait for the replay client to appear.",
    )
    parser.add_argument(
        "--no-stop",
        action="store_true",
        help="Report anomalies without stopping the replay.",
    )
    return parser.parse_args(argv)


def watch(args: argparse.Namespace) -> tuple[bool, str]:
    """Run the watch loop.

    Args:
        args: Parsed watchdog arguments.

    Returns:
        ``(ok, message)`` where ``ok`` is False for an anomaly.
    """
    deadline = time.monotonic() + args.startup_timeout
    replay_pid = _find_replay_pid(args.output_dir)
    while replay_pid is None:
        if time.monotonic() > deadline:
            return False, "replay client never started"
        time.sleep(args.interval / 4)
        replay_pid = _find_replay_pid(args.output_dir)

    requests_tail = _Tail(args.output_dir / "requests.jsonl")
    log_tails: dict[Path, _Tail] = {}
    baseline = _baseline_offsets(args.baseline)
    completions: list[float] = []
    failures = 0
    start: float | None = None
    stall_ticks = zero_ticks = rate_strikes = 0
    stall_limit = max(1, round(args.stall_minutes * 60 / args.interval))
    history: deque[str] = deque(maxlen=1)
    prev = _vllm_counters(args.vllm_url)

    def status() -> str:
        elapsed = time.monotonic() - start if start is not None else 0.0
        windows = [
            _count_window(completions, lo, lo + args.rate_window)
            for lo in range(0, int(elapsed), int(args.rate_window))
        ]
        return (
            f"pid={replay_pid} elapsed={elapsed:.0f}s done={len(completions)} "
            f"failed={failures} per{args.rate_window:.0f}s={windows} "
            f"zero_prompt_ticks={zero_ticks} stall_run={stall_ticks}"
        )

    while True:
        time.sleep(args.interval)
        for line in requests_tail.new_lines():
            if not line:
                continue
            record = json.loads(line)
            if start is None:
                start = float(record["sent_at_s"]) - float(record["send_offset_s"])
            if record.get("error") is not None:
                failures += 1
            else:
                completions.append(float(record["completion_offset_s"]))
        running = _pid_running(replay_pid)
        history.append(status())
        (args.output_dir / "watchdog.status").write_text(history[0] + "\n")
        if not running:
            return True, f"replay exited; {history[0]}"
        if failures:
            return False, f"{failures} failed request(s); {history[0]}"

        for log in (args.service_dir / "logs").glob("*.log"):
            tail = log_tails.setdefault(log, _Tail(log))
            for line in tail.new_lines():
                if _LOG_ALARM.search(line):
                    return False, f"{log.name}: {line.strip()[:300]}"

        pid_file = args.service_dir / "pids.json"
        if pid_file.is_file():
            for item in json.loads(pid_file.read_text()):
                pid = int(item["pid"])
                if not _pid_running(pid) and not _group_running(pid):
                    name = " ".join(item.get("cmd", [])[:2]) or item["pid"]
                    return False, f"service {name} exited; {history[0]}"

        free_data = shutil.disk_usage(args.data_path).free / 2**30
        free_root = shutil.disk_usage("/").free / 2**30
        if free_data < args.min_free_data_gib or free_root < args.min_free_root_gib:
            return False, f"low disk: data={free_data:.0f}G root={free_root:.0f}G"

        cur = _vllm_counters(args.vllm_url)
        if cur is not None and prev is not None:
            prompt = cur.get("vllm:prompt_tokens_total", 0) - prev.get(
                "vllm:prompt_tokens_total", 0
            )
            gen = cur.get("vllm:generation_tokens_total", 0) - prev.get(
                "vllm:generation_tokens_total", 0
            )
            live = cur.get("vllm:num_requests_running", 0)
            waiting = cur.get("vllm:num_requests_waiting", 0)
            zero_ticks += prompt == 0 and live > 0
            stalled = prompt == 0 and (waiting > 0 or (live > 0 and gen == 0))
            stall_ticks = stall_ticks + 1 if stalled else 0
            if stall_ticks >= stall_limit:
                return False, (
                    f"stall: no prompt tokens for {stall_ticks} ticks "
                    f"(running={live:.0f} waiting={waiting:.0f}); {history[0]}"
                )
        prev = cur

        if baseline and start is not None:
            hi = time.monotonic() - start
            lo = hi - args.rate_window
            expected = _count_window(baseline, lo, hi)
            if lo > 0 and expected >= args.rate_min_baseline:
                actual = _count_window(completions, lo, hi)
                low = actual < args.rate_ratio * expected
                rate_strikes = rate_strikes + 1 if low else 0
                if rate_strikes >= args.rate_strikes:
                    return False, (
                        f"slow: {actual} completions in last "
                        f"{args.rate_window:.0f}s vs baseline {expected}; "
                        f"{history[0]}"
                    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    ok, message = watch(args)
    print(f"watchdog: {'done' if ok else 'ANOMALY'}: {message}", flush=True)
    if not ok and not args.no_stop:
        pid = _find_replay_pid(args.output_dir)
        if pid is not None:
            os.kill(pid, signal.SIGTERM)
            print(f"watchdog: sent SIGTERM to replay pid {pid}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
