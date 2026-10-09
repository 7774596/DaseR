# SPDX-License-Identifier: Apache-2.0
"""Stop every recorded service process group and check a GPU is clean.

Chain scripts call this from their exit trap and after each arm. Services
start in their own sessions and may start more sessions below them (engine
cores, profiler wrappers), so signalling the chain's own process group does
not reach them; ``ServerManager`` records each group in a ``pgids`` file next
to ``pids.json``. This tool stops the groups in every ``pgids`` file found
under the given directories, then asks ``nvidia-smi`` whether processes owned
by the current user are still on the GPU.

Exit status: 0 when the GPU holds no process of this user, 1 otherwise.
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time

from benchmarks.utils.servers import stop_process_groups


def _owner(pid: int) -> str | None:
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except (FileNotFoundError, KeyError):
        return None


def gpu_pids(gpu: str) -> list[int]:
    """Return PIDs of compute processes on ``gpu`` as reported by nvidia-smi.

    Args:
        gpu: nvidia-smi device index.

    Returns:
        Compute-app PIDs on that device.
    """
    out = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [int(line) for line in out.split() if line.strip().isdigit()]


def reap(dirs: list[Path], gpu: str | None, settle_s: float) -> list[int]:
    """Stop recorded groups under ``dirs``; return this user's leftover GPU PIDs.

    Args:
        dirs: Directories searched recursively for ``pgids`` files.
        gpu: nvidia-smi device index to check, or None to skip the check.
        settle_s: Seconds to wait for the driver to release exited processes.

    Returns:
        PIDs owned by the current user still on ``gpu`` (empty when clean).
    """
    for root in dirs:
        if root.is_dir():
            for pgid_file in sorted(root.rglob("pgids")):
                killed = stop_process_groups(pgid_file)
                if killed:
                    print(f"reap: SIGKILL {killed} from {pgid_file}", flush=True)
    if gpu is None:
        return []
    user = getpass.getuser()
    deadline = time.monotonic() + settle_s
    while True:
        mine = [pid for pid in gpu_pids(gpu) if _owner(pid) == user]
        if not mine or time.monotonic() >= deadline:
            return mine
        time.sleep(1.0)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--gpu", help="nvidia-smi index to check for leftovers.")
    parser.add_argument("--settle-s", type=float, default=20.0)
    args = parser.parse_args(argv)
    left = reap(args.dirs, args.gpu, args.settle_s)
    if left:
        print(f"reap: ERROR user processes left on GPU {args.gpu}: {left}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
