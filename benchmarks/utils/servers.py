# SPDX-License-Identifier: Apache-2.0
"""Service lifecycle and manifest helpers for benchmark runs."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any

import httpx

from benchmarks.utils.constants import BLOCK_TOKENS, BYTES_PER_GIB
from benchmarks.utils.sizing import (
    LMCACHE_EVICTION_TRIGGER_WATERMARK,
    bytes_to_lmcache_gb,
)

LMCACHE_MP_HOST = "tcp://localhost"
LMCACHE_MP_PORT = 5555
LMCACHE_HTTP_PORT = 8080
LMCACHE_MP_CONNECTOR_NAME = "DaseRBenchLMCacheMPConnector"
LMCACHE_MP_CONNECTOR_MODULE = "benchmarks.utils.lmcache_connector_shim"
LMCACHE_CLI_MAIN = "from lmcache.cli.main import main; main()"
DEFAULT_DASER_PREFETCH_MAX_REQUESTS = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
LMCACHE_REPO_ROOT = REPO_ROOT.parent / "LMCache"
# The connector targets the vLLM 0.30 KV cache layouts; older releases fail
# only after model load with an unknown cache layout error.
MIN_VLLM_VERSION = (0, 30)


def _proc_table() -> dict[int, tuple[int, int, str]]:
    """Map every PID to ``(ppid, pgid, state)`` from ``/proc``.

    Returns:
        PID -> (parent PID, process group ID, state letter, ``Z`` = zombie);
        processes that exit while the table is read are skipped.
    """
    table: dict[int, tuple[int, int, str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # The command name may contain spaces or parentheses; fields after the
        # last ')' are: state ppid pgrp ...
        fields = stat.rsplit(")", 1)[1].split()
        table[int(entry.name)] = (int(fields[1]), int(fields[2]), fields[0])
    return table


def descendant_process_groups(roots: list[int]) -> list[int]:
    """Return the process groups of ``roots`` and all their descendants.

    A service may start children in their own sessions (a profiler wrapper,
    an engine core under ``setsid``); signalling the root's group alone then
    misses them. The caller's own group is never returned.

    Args:
        roots: PIDs whose process trees are scanned.

    Returns:
        Sorted distinct PGIDs, excluding the calling process's group.
    """
    table = _proc_table()
    children: dict[int, list[int]] = {}
    for pid, (ppid, _pgid, _state) in table.items():
        children.setdefault(ppid, []).append(pid)
    groups: set[int] = set()
    stack = [pid for pid in roots if pid in table]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        groups.add(table[pid][1])
        stack.extend(children.get(pid, []))
    groups.discard(os.getpgrp())
    return sorted(groups)


def read_process_groups(path: str | Path) -> list[int]:
    """Read a PGID file written by ``ServerManager`` (one PGID per line).

    Args:
        path: PGID file path; a missing file yields an empty list.

    Returns:
        Recorded PGIDs in file order.
    """
    pgid_path = Path(path)
    if not pgid_path.is_file():
        return []
    return [int(line) for line in pgid_path.read_text().split() if line.isdigit()]


def pgid_file_for(pid_file: str | Path) -> Path:
    """Return the PGID file that sits next to a benchmark pid file.

    Args:
        pid_file: ``pids.json`` path.

    Returns:
        ``pgids`` path in the same directory.
    """
    return Path(pid_file).with_name("pgids")


def record_process_groups(path: str | Path, roots: list[int]) -> list[int]:
    """Add the process groups under ``roots`` to a PGID file.

    Recorded groups are kept: a group whose leader already exited (a
    profiler wrapper) may still hold live members that are no longer
    descendants of any root.

    Args:
        path: PGID file path (one PGID per line), rewritten in place.
        roots: Service PIDs whose process trees are scanned.

    Returns:
        All recorded PGIDs after the merge.
    """
    groups = set(read_process_groups(path))
    groups.update(descendant_process_groups(roots))
    ordered = sorted(groups)
    Path(path).write_text("".join(f"{g}\n" for g in ordered))
    return ordered


def stop_process_groups(path: str | Path, grace_s: float = 15.0) -> list[int]:
    """SIGTERM every group in a PGID file, then SIGKILL the ones still alive.

    Args:
        path: PGID file path.
        grace_s: Seconds to wait between SIGTERM and SIGKILL.

    Returns:
        PGIDs that needed SIGKILL.

    Thread-safety:
        Sends process signals and should be called by one owner.
    """
    groups = [g for g in read_process_groups(path) if g != os.getpgrp()]
    alive = [g for g in groups if _signal_group(g, signal.SIGTERM)]
    deadline = time.monotonic() + grace_s
    while alive and time.monotonic() < deadline:
        time.sleep(0.2)
        alive = [g for g in alive if group_running(g)]
    for group in alive:
        _signal_group(group, signal.SIGKILL)
    return alive


def group_running(pgid: int) -> bool:
    """Return whether group ``pgid`` has a live, non-zombie member.

    A zombie still answers ``killpg(pgid, 0)`` until its parent reaps it, so
    signal probes overstate liveness after a SIGTERM.

    Args:
        pgid: Process group ID.

    Returns:
        True if any member is alive and not a zombie.
    """
    return any(g == pgid and st != "Z" for _p, g, st in _proc_table().values())


def _signal_group(pgid: int, sig: int) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _service_running(pid: int) -> bool:
    """Return whether a service's group, or the PID alone, is alive (not zombie)."""
    if group_running(pid):
        return True
    state = _proc_table().get(pid)
    return state is not None and state[2] != "Z"


def _signal_service(pid: int, sig: int) -> bool:
    """Send ``sig`` to a service's process group, or to the PID alone.

    Args:
        pid: Service PID; services lead their own process group.
        sig: Signal number; ``0`` only probes whether anything is alive.

    Returns:
        True if the group or process existed and was signalled.
    """
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        pass
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False


def check_runtime_environment() -> None:
    """Fail fast when benchmark services would start from a stale environment.

    Checks that the running interpreter has vLLM >= ``MIN_VLLM_VERSION``, that
    the ``vllm`` executable on ``PATH`` belongs to the same environment, and
    that ``import daser`` resolves to this checkout ahead of any other
    editable install.

    Raises:
        RuntimeError: If any check fails; the message names the offending
            version or path.

    Thread-safety:
        Reads interpreter metadata only; safe to call from any thread.
    """
    try:
        installed = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"vLLM is not installed for {sys.executable}") from exc
    release = tuple(int(part) for part in re.findall(r"\d+", installed)[:2])
    if release < MIN_VLLM_VERSION:
        required = ".".join(str(part) for part in MIN_VLLM_VERSION)
        raise RuntimeError(
            f"vLLM {installed} at {sys.executable} is older than {required}; "
            "activate the vLLM 0.30 environment"
        )
    vllm_bin = shutil.which("vllm")
    expected_bin = Path(sys.executable).parent / "vllm"
    if vllm_bin is None or Path(vllm_bin).resolve() != expected_bin.resolve():
        raise RuntimeError(
            f"vllm on PATH ({vllm_bin}) is not {expected_bin}; put the running "
            "environment's bin directory first on PATH"
        )
    # daser is a namespace package, so every checkout on sys.path contributes
    # to its search path; the first entry decides which files are imported.
    spec = importlib.util.find_spec("daser")
    locations = list(spec.submodule_search_locations or []) if spec else []
    package = Path(locations[0]).resolve() if locations else None
    if package != REPO_ROOT / "daser":
        raise RuntimeError(
            f"import daser resolves to {package}, not {REPO_ROOT / 'daser'}; "
            "set PYTHONPATH to this checkout"
        )


def _lmcache_l1_total_bytes(status: Any) -> int | None:
    """Return the currently committed LMCache L1 address-space size."""
    if not isinstance(status, dict):
        return None
    storage = status.get("storage_manager")
    if not isinstance(storage, dict):
        return None
    l1 = storage.get("l1_manager")
    if not isinstance(l1, dict):
        return None
    total = l1.get("memory_total_bytes")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return None
    return int(total)


def resolve_daser_prefetch_max_requests(
    prefetch_enabled: bool,
    override: int | None,
) -> int:
    """Resolve the effective scheduler prefetch worker limit.

    Args:
        prefetch_enabled: Whether the explicit boolean opt-in was supplied.
        override: Optional numeric compatibility override. Zero explicitly
            disables prefetch even when the boolean flag is present.

    Returns:
        Effective non-negative worker limit. The opt-in default is two.

    Raises:
        ValueError: If ``override`` is negative.

    Thread-safety:
        Pure function; safe to call from CLI parsing or service setup.
    """
    if override is not None:
        if override < 0:
            raise ValueError("daser prefetch worker limit must be non-negative")
        return override
    return DEFAULT_DASER_PREFETCH_MAX_REQUESTS if prefetch_enabled else 0


@dataclass(frozen=True)
class ServiceEndpoint:
    """HTTP endpoint for a benchmark service.

    Args:
        url: Base URL.

    Thread-safety:
        Immutable value object.
    """

    url: str


@dataclass(frozen=True)
class BenchmarkManifest:
    """Service run manifest shared between start and load scripts.

    Args:
        run_id: Run identifier.
        backend: Backend name.
        reuse_mode: DaseR reuse mode, or ``none``.
        model: Model path.
        store_dir: Scratch directory.
        l1_size_bytes: L1 size in bytes.
        l2_size_bytes: L2 size in bytes.
        skip_l2: Whether the service should run with no L2 tier.
        endpoints: Named service endpoints.
        log_dir: Log directory.
        pid_file: JSON file containing subprocess PIDs.
        block_size: vLLM KV block size in tokens.
        prefetch_enabled: Whether scheduler prefetch is effectively enabled.
        prefetch_max_requests: Effective scheduler prefetch worker limit.
        storage_format: DaseR physical KV storage format.
        online_pack_batch_slots: Compressed-online codec batch override, or
            ``None`` for the connector default.

    Thread-safety:
        Immutable value object.
    """

    run_id: str
    backend: str
    reuse_mode: str
    model: str
    store_dir: str
    l1_size_bytes: int
    l2_size_bytes: int
    skip_l2: bool
    endpoints: dict[str, ServiceEndpoint]
    log_dir: str
    pid_file: str
    block_size: int = BLOCK_TOKENS
    prefetch_enabled: bool = False
    prefetch_max_requests: int = 0
    storage_format: str = "raw"
    online_pack_batch_slots: int | None = None

    def write(self, path: str | Path) -> None:
        """Write manifest JSON atomically enough for local benchmark use."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "BenchmarkManifest":
        """Read a manifest JSON file.

        Args:
            path: Manifest path.

        Returns:
            BenchmarkManifest instance.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        endpoints = {
            name: ServiceEndpoint(**endpoint)
            for name, endpoint in payload["endpoints"].items()
        }
        payload["endpoints"] = endpoints
        payload.setdefault("block_size", 16)  # Legacy manifests predate this field.
        payload.setdefault(
            "prefetch_enabled", bool(payload.get("prefetch_max_requests", 0))
        )
        payload.setdefault("prefetch_max_requests", 0)
        payload.setdefault("storage_format", "raw")
        return cls(**payload)


class ServerManager:
    """Start and stop benchmark service subprocesses."""

    def __init__(
        self,
        run_id: str,
        backend: str,
        model: str,
        store_dir: str | Path,
        gpu_id: str,
        gpu_util: float | None,
        max_num_seqs: int | None,
        l1_size_bytes: int,
        l2_size_bytes: int,
        max_num_batched_tokens: int | None = None,
        block_size: int = BLOCK_TOKENS,
        reuse_mode: str = "chunk",
        transfer_mode: str = "iouring",
        vllm_port: int = 8001,
        daser_port: int = 2026,
        startup_timeout: float = 240.0,
        max_model_len: int | None = None,
        skip_l2: bool = False,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = False,
        daser_prefetch_max_requests: int = 0,
        daser_prefetch_lease_budget_bytes: int = 0,
        daser_l2_read_stripe_bytes: int = 0,
        daser_extra_kv_layers: int = 0,
        daser_l1_accounting: str = "stored",
        daser_codebooks: Path | None = None,
        daser_coalesce_loads: bool = False,
        daser_layerwise_load: bool = False,
        storage_format: str = "raw",
        online_pack_batch_slots: int | None = None,
        async_scheduling: bool = True,
        extra_vllm_gpu_ids: tuple[str, ...] = (),
        lmcache_l2_odirect: bool = False,
        lmcache_eviction_watermark: float = LMCACHE_EVICTION_TRIGGER_WATERMARK,
        lmcache_l2_native_workers: int = 0,
        lmcache_l2_bounded: bool = False,
        lmcache_eager_prefetch: bool = False,
        lmcache_pythonpath: tuple[Path, ...] = (LMCACHE_REPO_ROOT,),
        lmcache_daser_codec: bool = False,
        lmcache_codec_capacity: float | None = None,
        lmcache_codec_codebooks: Path | None = None,
        extra_vllm_args: tuple[str, ...] = (),
    ) -> None:
        """Initialize the service manager.

        Args:
            run_id: Run identifier.
            backend: Backend name.
            model: Model path.
            store_dir: Scratch directory.
            gpu_id: GPU ID exposed through CUDA_VISIBLE_DEVICES.
            gpu_util: Optional explicit vLLM GPU memory utilization. ``None``
                leaves vLLM's own default unchanged.
            max_num_seqs: vLLM max_num_seqs; ``None`` keeps the engine default.
            max_num_batched_tokens: Optional vLLM scheduler token budget.
            block_size: vLLM KV block size in tokens.
            l1_size_bytes: L1 size.
            l2_size_bytes: L2 size.
            reuse_mode: DaseR cache reuse mode.
            transfer_mode: DaseR transfer backend.
            vllm_port: vLLM HTTP port.
            daser_port: DaseR HTTP port.
            startup_timeout: Health-check timeout.
            max_model_len: Optional vLLM max model length.
            skip_l2: Disable L2 persistence/adapters for L1-only no-evict runs.
            tensor_parallel_size: vLLM tensor-parallel rank count.
            trust_remote_code: allow model/tokenizer repository Python code.
            daser_prefetch_max_requests: maximum concurrent scheduler prefetches.
            daser_prefetch_lease_budget_bytes: DaseR ``--prefetch-lease-budget``
                (0 leaves the server default of no cap).
            daser_l2_read_stripe_bytes: DaseR ``--l2-read-stripe`` (0 leaves
                the server default of one serial read per extent).
            daser_extra_kv_layers: DaseR ``--extra-kv-layers``, the KV layers
                vLLM registers beyond the target model's (an EAGLE drafter).
            daser_l1_accounting: DaseR ``--l1-accounting`` capacity unit.
            daser_codebooks: calibrated codebook directory passed as DaseR
                ``--compressed-codebooks``; ``None`` uses the generic
                codebooks. Only valid with ``compressed-online`` storage.
            daser_coalesce_loads: let the connector hold back requests whose
                prefix another request is already loading.
            daser_layerwise_load: restore fully leased packed or raw hits
                layer by layer inside the forward pass.
            storage_format: DaseR physical KV storage format.
            lmcache_l2_odirect: open LMCache fs L2 files with ``O_DIRECT`` so
                its reads bypass the OS page cache, as DaseR's L2 reads do.
            lmcache_eviction_watermark: LMCache L1 usage fraction that
                triggers eviction. LMCache reserves L1 for a whole L2
                prefetch at once and falls back to recompute when the
                reservation fails, so long contexts need a lower watermark.
            lmcache_l2_native_workers: when positive, use LMCache's C++
                ``fs_native`` L2 adapter with this many IO threads instead of
                the Python ``fs`` adapter.
            lmcache_l2_bounded: cap the ``fs_native`` L2 at ``l2_size_bytes``
                with LRU eviction so LMCache gets the same SSD capacity as a
                DaseR arm; without it LMCache's fs L2 grows without bound.
            lmcache_eager_prefetch: submit LMCache's lookup (and its L2 to
                L1 prefetch) when a request enters vLLM's waiting queue
                instead of when the scheduler first considers it.
            lmcache_pythonpath: entries put first on ``PYTHONPATH`` of the
                LMCache server and its vLLM: the LMCache checkout to run, then
                any directories holding its extra pure-Python dependencies.
            lmcache_daser_codec: enable the DaseR lossless chunk codec of a
                codec-patched LMCache checkout (``LMCACHE_DASER_CODEC=1``).
            lmcache_codec_capacity: stored capacity class as a fraction of the
                raw chunk (``LMCACHE_DASER_CODEC_CAPACITY``); ``None`` stores
                variable-length records.
            lmcache_codec_codebooks: calibrated codebook directory for the
                codec; ``None`` uses the generic codebooks.
            online_pack_batch_slots: Optional compressed-online codec batch
                size in raw slots; ``None`` keeps the connector default.
            async_scheduling: vLLM async scheduling, on by default as in vLLM;
                ``False`` is an ablation.
            extra_vllm_gpu_ids: GPUs for additional vLLM instances that share
                this service's DaseR server (DaseR backend only). Instance
                ``i`` (1-based) listens on ``vllm_port + i``; the DaseR server
                sees every instance's GPU.
            extra_vllm_args: Extra ``vllm serve`` arguments appended verbatim
                (scheduler knobs such as ``--scheduling-policy``), applied to
                every arm alike.
        """
        if tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if daser_prefetch_max_requests < 0:
            raise ValueError("daser_prefetch_max_requests must be non-negative")
        if daser_prefetch_lease_budget_bytes < 0:
            raise ValueError("daser_prefetch_lease_budget_bytes must be non-negative")
        if daser_l2_read_stripe_bytes < 0:
            raise ValueError("daser_l2_read_stripe_bytes must be non-negative")
        if online_pack_batch_slots is not None and online_pack_batch_slots <= 0:
            raise ValueError("online_pack_batch_slots must be positive")
        if lmcache_l2_bounded and lmcache_l2_native_workers <= 0:
            raise ValueError("lmcache_l2_bounded requires the fs_native L2 adapter")
        if not lmcache_daser_codec and (
            lmcache_codec_capacity is not None or lmcache_codec_codebooks is not None
        ):
            raise ValueError("LMCache codec options require lmcache_daser_codec")
        if daser_codebooks is not None and storage_format != "compressed-online":
            raise ValueError("daser_codebooks require compressed-online storage")
        if lmcache_codec_capacity is not None and not 0 < lmcache_codec_capacity <= 1:
            raise ValueError("lmcache_codec_capacity must be in (0, 1]")
        if extra_vllm_gpu_ids and backend != "daser":
            raise ValueError("extra vLLM instances require the daser backend")
        self.run_id = run_id
        self.backend = backend
        self.model = model
        self.store_dir = Path(store_dir)
        self.gpu_id = gpu_id
        self.gpu_util = gpu_util
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = block_size
        self.l1_size_bytes = l1_size_bytes
        self.l2_size_bytes = l2_size_bytes
        self.reuse_mode = reuse_mode
        self.transfer_mode = transfer_mode
        self.vllm_port = vllm_port
        self.daser_port = daser_port
        self.startup_timeout = startup_timeout
        self.max_model_len = max_model_len
        self.skip_l2 = skip_l2
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.daser_prefetch_max_requests = daser_prefetch_max_requests
        self.daser_prefetch_lease_budget_bytes = daser_prefetch_lease_budget_bytes
        self.daser_l2_read_stripe_bytes = daser_l2_read_stripe_bytes
        self.daser_extra_kv_layers = daser_extra_kv_layers
        self.lmcache_l2_odirect = lmcache_l2_odirect
        self.lmcache_eviction_watermark = lmcache_eviction_watermark
        self.lmcache_l2_native_workers = lmcache_l2_native_workers
        self.lmcache_l2_bounded = lmcache_l2_bounded
        self.lmcache_eager_prefetch = lmcache_eager_prefetch
        self.lmcache_pythonpath = lmcache_pythonpath
        self.lmcache_daser_codec = lmcache_daser_codec
        self.lmcache_codec_capacity = lmcache_codec_capacity
        self.lmcache_codec_codebooks = lmcache_codec_codebooks
        self.daser_l1_accounting = daser_l1_accounting
        self.daser_codebooks = daser_codebooks
        self.daser_coalesce_loads = daser_coalesce_loads
        self.daser_layerwise_load = daser_layerwise_load
        self.storage_format = storage_format
        self.online_pack_batch_slots = online_pack_batch_slots
        self.async_scheduling = async_scheduling
        self.extra_vllm_gpu_ids = tuple(str(gpu) for gpu in extra_vllm_gpu_ids)
        self.extra_vllm_args = tuple(extra_vllm_args)
        self.log_dir = self.store_dir / "logs"
        self.pid_file = self.store_dir / "pids.json"
        # Every process group under the services, for teardown by a parent
        # that outlives a killed runner (chain traps, watchdog).
        self.pgid_file = pgid_file_for(self.pid_file)
        self.socket_path = self.store_dir / "daser.sock"
        self._procs: list[tuple[subprocess.Popen[bytes], str]] = []

    @property
    def vllm_url(self) -> str:
        """Return the vLLM base URL."""
        return f"http://127.0.0.1:{self.vllm_port}"

    @property
    def vllm_urls(self) -> list[str]:
        """Return every vLLM instance base URL, the primary instance first."""
        return [
            f"http://127.0.0.1:{self.vllm_port + offset}"
            for offset in range(1 + len(self.extra_vllm_gpu_ids))
        ]

    @property
    def daser_url(self) -> str:
        """Return the DaseR base URL."""
        return f"http://127.0.0.1:{self.daser_port}"

    async def start(self) -> BenchmarkManifest:
        """Start services for the configured backend and return a manifest."""
        check_runtime_environment()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_ports_available()
        if self.backend == "vllm":
            await self.start_vllm_only()
        elif self.backend == "lmcache":
            await self.start_lmcache_mp_server()
            await self.start_vllm_lmcache()
        elif self.backend == "daser":
            await self.start_vllm_daser()
            await self.start_daser_server()
        else:
            raise ValueError(f"unsupported backend: {self.backend}")
        self._write_pids()
        manifest = self.manifest()
        manifest.write(self.store_dir / "manifest.json")
        return manifest

    def manifest(self) -> BenchmarkManifest:
        """Return the current service manifest."""
        endpoints = {"vllm": ServiceEndpoint(self.vllm_url)}
        if self.backend == "daser":
            endpoints["daser"] = ServiceEndpoint(self.daser_url)
        return BenchmarkManifest(
            run_id=self.run_id,
            backend=self.backend,
            reuse_mode=self.reuse_mode if self.backend == "daser" else "none",
            model=self.model,
            store_dir=str(self.store_dir),
            l1_size_bytes=self.l1_size_bytes,
            l2_size_bytes=self.l2_size_bytes,
            skip_l2=self.skip_l2,
            endpoints=endpoints,
            log_dir=str(self.log_dir),
            pid_file=str(self.pid_file),
            block_size=self.block_size,
            prefetch_enabled=self.daser_prefetch_max_requests > 0,
            prefetch_max_requests=self.daser_prefetch_max_requests,
            storage_format=self.storage_format,
            online_pack_batch_slots=self.online_pack_batch_slots,
        )

    async def start_lmcache_mp_server(self) -> None:
        """Start the LMCache MP server."""
        cmd = self._lmcache_mp_server_command()
        proc = self._start(
            cmd,
            "lmcache_mp_server.log",
            extra_env=self.lmcache_process_env(),
        )
        await self._wait_healthy(
            f"http://127.0.0.1:{LMCACHE_HTTP_PORT}",
            "/healthcheck",
            self.startup_timeout,
            proc,
        )
        await self._wait_lmcache_l1_ready(proc)

    def _lmcache_mp_server_command(self) -> list[str]:
        """Build the LMCache MP server command.

        Returns:
            Command argv. When ``skip_l2`` is true, no ``--l2-adapter`` is
            emitted, so LMCache keeps data in its L1 manager only.

        Thread-safety:
            Pure helper except for creating the L2 scratch directory when the
            adapter is enabled.
        """
        l1_gb = bytes_to_lmcache_gb(self.l1_size_bytes)
        # Run the CLI of the checkout on PYTHONPATH, the same LMCache the vLLM
        # connector imports, rather than whatever is on PATH. Not every
        # release makes lmcache.cli.main runnable with -m, so call main().
        cmd = [
            sys.executable,
            "-c",
            LMCACHE_CLI_MAIN,
            "server",
            "--host",
            "localhost",
            "--port",
            str(LMCACHE_MP_PORT),
            "--chunk-size",
            str(self.block_size),
            "--max-workers",
            "4",
            "--l1-size-gb",
            str(l1_gb),
            "--eviction-policy",
            "LRU",
            "--eviction-trigger-watermark",
            str(self.lmcache_eviction_watermark),
            "--http-port",
            str(LMCACHE_HTTP_PORT),
            # Benchmarks read /status; OTel metrics need an exporter that
            # not every LMCache install ships.
            "--disable-metrics",
        ]
        if not self.skip_l2:
            scratch = self.store_dir / "lmcache_mp_disk"
            scratch.mkdir(parents=True, exist_ok=True)
            cmd.extend(
                [
                    "--l2-adapter",
                    json.dumps(
                        {
                            "base_path": str(scratch),
                            "use_odirect": self.lmcache_l2_odirect,
                            **(
                                {
                                    "type": "fs_native",
                                    "num_workers": self.lmcache_l2_native_workers,
                                    **(
                                        {
                                            "max_capacity_gb": self.l2_size_bytes
                                            / BYTES_PER_GIB,
                                            # Evict late and little so the
                                            # usable L2 matches DaseR's full
                                            # arena (defaults .8/.2 keep less).
                                            "eviction": {
                                                "eviction_policy": "LRU",
                                                "trigger_watermark": 0.95,
                                                "eviction_ratio": 0.05,
                                            },
                                        }
                                        if self.lmcache_l2_bounded
                                        else {}
                                    ),
                                }
                                if self.lmcache_l2_native_workers
                                else {"type": "fs"}
                            ),
                        }
                    ),
                ]
            )
        return cmd

    async def start_vllm_only(self) -> None:
        """Start vanilla vLLM."""
        await self._start_vllm("vllm_vanilla.log", None)

    async def start_vllm_lmcache(self) -> None:
        """Start vLLM with LMCache MP connector."""
        await self._start_vllm(
            "vllm_lmcache.log",
            self.lmcache_kv_transfer_config(),
            extra_env={
                **self.lmcache_process_env(),
                "PYTHONHASHSEED": "42",
            },
        )

    def lmcache_kv_transfer_config(self) -> dict[str, Any]:
        """Return vLLM KV transfer config for the sibling LMCache checkout.

        Returns:
            JSON-serialisable vLLM ``--kv-transfer-config`` payload.

        Thread-safety:
            Pure calculation over module constants and port configuration.
        """
        kv_config = {
            "kv_connector": LMCACHE_MP_CONNECTOR_NAME,
            "kv_connector_module_path": LMCACHE_MP_CONNECTOR_MODULE,
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": LMCACHE_MP_HOST,
                "lmcache.mp.port": LMCACHE_MP_PORT,
                **(
                    {"lmcache.mp.eager_prefetch": True}
                    if self.lmcache_eager_prefetch
                    else {}
                ),
            },
        }
        return kv_config

    def lmcache_process_env(self) -> dict[str, str]:
        """Return environment overrides for LMCache benchmark subprocesses.

        Returns:
            Environment entries that make the configured LMCache checkout
            (by default the sibling ``../LMCache``) take precedence over
            installed LMCache packages and vLLM built-ins, plus the codec
            switches when the DaseR codec is enabled.

        Thread-safety:
            Reads the current process environment without mutating it.
        """
        pythonpath = os.environ.get("PYTHONPATH")
        entries = [*map(str, self.lmcache_pythonpath), str(REPO_ROOT)]
        if pythonpath:
            entries.append(pythonpath)
        env = {"PYTHONPATH": os.pathsep.join(entries)}
        if self.lmcache_daser_codec:
            env["LMCACHE_DASER_CODEC"] = "1"
            if self.lmcache_codec_capacity is not None:
                env["LMCACHE_DASER_CODEC_CAPACITY"] = str(self.lmcache_codec_capacity)
            if self.lmcache_codec_codebooks is not None:
                env["LMCACHE_DASER_CODEC_CODEBOOKS"] = str(self.lmcache_codec_codebooks)
        return env

    async def start_vllm_daser(self) -> None:
        """Start every vLLM instance with the DaseR connector.

        All instances share one DaseR socket; the primary one runs on
        ``gpu_id`` and ``extra_vllm_gpu_ids[i - 1]`` on ``vllm_port + i``.
        """
        kv_config = self.daser_kv_transfer_config()
        await self._start_vllm("vllm_daser.log", kv_config)
        for offset, gpu in enumerate(self.extra_vllm_gpu_ids, start=1):
            await self._start_vllm(
                f"vllm_daser_{offset}.log",
                kv_config,
                port=self.vllm_port + offset,
                gpu_ids=gpu,
            )

    def daser_kv_transfer_config(self) -> dict[str, Any]:
        """Return the vLLM KV transfer payload for a DaseR service.

        Returns:
            JSON-serialisable vLLM connector configuration, including the
            effective prefetch state used by the scheduler role. Lookups do
            not pin chunks, so a hit can be evicted before its load runs;
            ``kv_load_failure_policy`` makes vLLM recompute those blocks
            instead of failing the request.

        Thread-safety:
            Pure calculation over immutable service configuration.
        """
        extra_config: dict[str, Any] = {
            "socket_path": str(self.socket_path),
            "cache_reuse_mode": self.reuse_mode,
            "prefetch_max_requests": self.daser_prefetch_max_requests,
            "prefetch_enabled": self.daser_prefetch_max_requests > 0,
            "storage_format": self.storage_format,
        }
        if self.online_pack_batch_slots is not None:
            extra_config["online_pack_batch_slots"] = self.online_pack_batch_slots
        if self.daser_coalesce_loads:
            extra_config["coalesce_loads"] = True
        if self.daser_layerwise_load:
            extra_config["layerwise_load"] = True
        return {
            "kv_connector": "DaserConnector",
            "kv_connector_module_path": "daser.connector.daser_connector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": extra_config,
        }

    async def start_daser_server(self) -> None:
        """Start DaseR HTTP + IPC server."""
        cmd = self._daser_server_command()
        proc = self._start(
            cmd,
            "daser.log",
            # Instances may share a GPU; list each device once.
            gpu_ids=",".join(
                dict.fromkeys((str(self.gpu_id), *self.extra_vllm_gpu_ids))
            ),
            profile_env="DASER_BENCH_PROFILE_DASER",
        )
        await self._wait_healthy(self.daser_url, "/health", self.startup_timeout, proc)

    def _daser_server_command(self) -> list[str]:
        """Build the DaseR server command.

        Returns:
            Command argv. When ``skip_l2`` is true, no ``--l2-size`` is
            emitted and DaseR runs an L1-only tier of ``--l1-size`` bytes.

        Thread-safety:
            Not thread-safe with respect to concurrent store directory cleanup.
        """
        store = self.store_dir / "daser"
        store.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "daser.server",
            "--vllm-base-url",
            self.vllm_url,
            "--model-path",
            self.model,
            "--store-dir",
            str(store),
            "--l1-size",
            str(self.l1_size_bytes),
            "--transfer-mode",
            self.transfer_mode,
            "--cache-reuse-mode",
            self.reuse_mode,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.daser_port),
            "--socket-path",
            str(self.socket_path),
            "--block-tokens",
            str(self.block_size),
            "--storage-format",
            self.storage_format,
        ]
        if self.skip_l2:
            cmd.append("--skip-l2")
        else:
            cmd.extend(["--l2-size", str(self.l2_size_bytes)])
        if self.tensor_parallel_size > 1:
            cmd.extend(["--tensor-parallel-size", str(self.tensor_parallel_size)])
        if self.daser_prefetch_lease_budget_bytes:
            cmd.extend(
                ["--prefetch-lease-budget", str(self.daser_prefetch_lease_budget_bytes)]
            )
        if self.daser_l2_read_stripe_bytes:
            cmd.extend(["--l2-read-stripe", str(self.daser_l2_read_stripe_bytes)])
        if self.daser_extra_kv_layers:
            cmd.extend(["--extra-kv-layers", str(self.daser_extra_kv_layers)])
        if self.daser_l1_accounting != "stored":
            cmd.extend(["--l1-accounting", self.daser_l1_accounting])
        if self.daser_codebooks is not None:
            cmd.extend(["--compressed-codebooks", str(self.daser_codebooks)])
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        return cmd

    async def _start_vllm(
        self,
        log_name: str,
        kv_transfer_config: dict[str, Any] | None,
        extra_env: dict[str, str] | None = None,
        port: int | None = None,
        gpu_ids: str | None = None,
    ) -> None:
        port = self.vllm_port if port is None else port
        proc = self._start(
            self.vllm_command(kv_transfer_config, port=port),
            log_name,
            extra_env=extra_env,
            gpu_ids=gpu_ids,
            profile_env="DASER_BENCH_PROFILE_VLLM",
        )
        await self._wait_healthy(
            f"http://127.0.0.1:{port}", "/health", self.startup_timeout, proc
        )

    def vllm_command(
        self,
        kv_transfer_config: dict[str, Any] | None,
        port: int | None = None,
    ) -> list[str]:
        """Build the vLLM serve command for benchmark services.

        Args:
            kv_transfer_config: Optional KV transfer configuration.
            port: HTTP port; defaults to the primary ``vllm_port``.

        Returns:
            Command list suitable for subprocess.Popen.

        Thread-safety:
            Pure calculation over immutable instance configuration.
        """
        cmd = [
            "vllm",
            "serve",
            self.model,
            "--served-model-name",
            Path(self.model).name,
            self.model,
            "--port",
            str(self.vllm_port if port is None else port),
            "--no-enable-prefix-caching",
            "--generation-config",
            "vllm",
            "--block-size",
            str(self.block_size),
            "--async-scheduling" if self.async_scheduling else "--no-async-scheduling",
        ]
        if self.max_num_seqs is not None:
            cmd.extend(["--max-num-seqs", str(self.max_num_seqs)])
        if os.environ.get("DASER_BENCH_SKIP_OPTIONAL_VLLM_FLAGS", "0") != "1":
            # Prompt token details carry per-request cached tokens (GPU plus
            # connector), so replays can audit hits request by request.
            cmd.extend(
                [
                    "--stream-interval",
                    "1",
                    "--attention-backend",
                    "FLASH_ATTN",
                    "--enable-prompt-tokens-details",
                ]
            )
        if self.gpu_util is not None:
            cmd.extend(["--gpu-memory-utilization", str(self.gpu_util)])
        if self.max_model_len is not None and self.max_model_len > 0:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens > 0:
            cmd.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])
        if self.tensor_parallel_size > 1:
            cmd.extend(["--tensor-parallel-size", str(self.tensor_parallel_size)])
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if kv_transfer_config is not None:
            cmd.extend(["--kv-transfer-config", json.dumps(kv_transfer_config)])
        cmd.extend(self.extra_vllm_args)
        return cmd

    def _start(
        self,
        cmd: list[str],
        log_name: str,
        extra_env: dict[str, str] | None = None,
        gpu_ids: str | None = None,
        profile_env: str | None = None,
    ) -> subprocess.Popen[bytes]:
        visible = str(self.gpu_id) if gpu_ids is None else gpu_ids
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / log_name
        if profile_env is not None:
            cmd = profile_prefix(profile_env, log_path) + cmd
        fh = log_path.open("wb")
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = visible
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{pythonpath}" if pythonpath else str(REPO_ROOT)
        )
        if extra_env:
            env.update(extra_env)
        # Each service leads its own process group so stop_from_pid_file can
        # reach every descendant, even after a wrapper (profiler, launcher)
        # exited and left the real server re-parented.
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self._procs.append((proc, visible))
        # Record every spawn at once so a later startup failure can still stop
        # the services that already came up.
        self._write_pids()
        return proc

    async def _wait_healthy(
        self,
        base_url: str,
        path: str,
        timeout: float,
        proc: subprocess.Popen[bytes],
    ) -> None:
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"{base_url}{path} startup process exited with "
                        f"code {proc.returncode}"
                    )
                try:
                    response = await client.get(f"{base_url}{path}", timeout=5.0)
                    if response.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(2.0)
        raise RuntimeError(f"{base_url}{path} not healthy after {timeout:.0f}s")

    async def _wait_lmcache_l1_ready(
        self,
        proc: subprocess.Popen[bytes],
    ) -> None:
        """Wait until LMCache has pinned and exposed its complete L1 capacity.

        LMCache's health endpoint becomes ready while ``LazyMemoryAllocator``
        is still registering pinned host memory in the background. Transfers
        during that expansion contend with ``cudaHostRegister`` and can add
        seconds to TTFT, so benchmark startup must wait for the address space
        reported by ``/status`` to reach the configured integer-GiB capacity.

        Args:
            proc: LMCache MP server process being monitored.

        Returns:
            None after the configured L1 capacity is available.

        Raises:
            RuntimeError: If the server exits or L1 is not ready before the
                startup timeout.

        Asyncio/thread-safety:
            Asynchronous polling only; intended for the startup task that owns
            the LMCache process.
        """
        expected_bytes = bytes_to_lmcache_gb(self.l1_size_bytes) * BYTES_PER_GIB
        deadline = time.monotonic() + self.startup_timeout
        last_total_bytes: int | None = None
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        "LMCache L1 initialization process exited with "
                        f"code {proc.returncode}"
                    )
                try:
                    response = await client.get(
                        f"http://127.0.0.1:{LMCACHE_HTTP_PORT}/status",
                        timeout=5.0,
                    )
                    if response.status_code == 200:
                        status = response.json()
                        last_total_bytes = _lmcache_l1_total_bytes(status)
                        if (
                            last_total_bytes is not None
                            and last_total_bytes >= expected_bytes
                        ):
                            return
                except (httpx.HTTPError, ValueError):
                    pass
                await asyncio.sleep(1.0)
        observed = "unknown" if last_total_bytes is None else str(last_total_bytes)
        raise RuntimeError(
            "LMCache L1 did not reach configured capacity before startup timeout: "
            f"observed={observed} expected={expected_bytes} bytes"
        )

    def _write_pids(self) -> None:
        payload = [
            {
                "pid": proc.pid,
                "cmd": proc.args,
                "cuda_visible_devices": visible,
            }
            for proc, visible in self._procs
        ]
        self.pid_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Merge with groups seen earlier: a group whose leader already exited
        # (a profiler wrapper) may still hold live members.
        record_process_groups(self.pgid_file, [proc.pid for proc, _ in self._procs])

    def _ensure_ports_available(self) -> None:
        ports = [int(url.rsplit(":", 1)[1]) for url in self.vllm_urls]
        if self.backend == "lmcache":
            ports.extend([LMCACHE_MP_PORT, LMCACHE_HTTP_PORT])
        if self.backend == "daser":
            ports.append(self.daser_port)
        busy = [port for port in ports if _is_port_open("127.0.0.1", port)]
        if busy:
            joined = ", ".join(str(port) for port in busy)
            raise RuntimeError(f"benchmark service port(s) already in use: {joined}")


def profile_prefix(env_var: str, log_path: Path) -> list[str]:
    """Return a profiler command prefix configured through ``env_var``.

    The variable holds a shell-style command (for example an ``nsys profile``
    invocation) prepended to the service argv; ``{log}`` expands to the
    service log path without its suffix and ``{name}`` to its stem, so each
    instance gets its own report or profiler session.

    Args:
        env_var: ``DASER_BENCH_PROFILE_VLLM`` or ``DASER_BENCH_PROFILE_DASER``.
        log_path: Service log path.

    Returns:
        The prefix argv, empty when the variable is unset (default).

    Thread-safety:
        Reads the process environment only.
    """
    template = os.environ.get(env_var, "").strip()
    if not template:
        return []
    stem = log_path.with_suffix("")
    return shlex.split(
        template.replace("{log}", str(stem)).replace("{name}", stem.name)
    )


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def stop_from_pid_file(path: str | Path) -> None:
    """Terminate processes listed in a benchmark pid file.

    Each recorded PID leads its own process group (services start in a new
    session), so the whole group is signalled: descendants left behind by an
    exited wrapper are stopped too. A PID that is not a group leader is
    signalled alone. Finally every group recorded in the sibling ``pgids``
    file (children that started their own session) is stopped.

    Args:
        path: PID JSON path.

    Thread-safety:
        Sends process signals and should be called by one owner.
    """
    pid_path = Path(path)
    if not pid_path.is_file():
        return
    pgid_path = pgid_file_for(pid_path)
    payload = json.loads(pid_path.read_text(encoding="utf-8"))
    # Catch groups started after the last record (e.g. engine workers).
    record_process_groups(pgid_path, [int(item["pid"]) for item in payload])
    for item in reversed(payload):
        _signal_service(int(item["pid"]), signal.SIGTERM)
    deadline = time.monotonic() + 15.0
    for item in reversed(payload):
        pid = int(item["pid"])
        while time.monotonic() < deadline:
            if not _service_running(pid):
                break
            time.sleep(0.2)
        else:
            _signal_service(pid, signal.SIGKILL)
    # Groups outside the services' own (setsid children) are stopped last.
    stop_process_groups(pgid_path)
