# SPDX-License-Identifier: Apache-2.0
"""Check engine environment isolation and deterministic-run provenance."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from benchmarks.utils.servers import BenchmarkManifest, ServerManager


@pytest.mark.parametrize(
    ("inherited", "requested", "expected"),
    [(None, False, False), ("0", False, False), ("1", False, True), ("0", True, True)],
)
@pytest.mark.asyncio
async def test_engine_determinism_matches_manifest_without_mutating_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    inherited: str | None,
    requested: bool,
    expected: bool,
) -> None:
    """Opt-in and inherited modes reach the engine and its persisted manifest.

    Args:
        monkeypatch: Pytest environment and subprocess isolation fixture.
        tmp_path: Managed scratch directory for manifest and log output.
        inherited: Parent environment value, or None if unset.
        requested: Explicit batch-invariant startup option.
        expected: Effective engine mode recorded in the manifest.
    Returns:
        None; verifies engine environment, provenance and parent isolation.
    Async/thread-safety:
        Runs public service startup with a fake process and in-memory HTTP
        transport; never starts a GPU service or modifies global env directly.
    """
    if inherited is None:
        monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    else:
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", inherited)
    engine_environments: list[dict[str, str]] = []

    def fake_popen(command: list[str], **kwargs: Any) -> SimpleNamespace:
        engine_environments.append(kwargs["env"])
        kwargs["stdout"].close()
        return SimpleNamespace(args=command, pid=123, poll=lambda: None)

    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    monkeypatch.setattr("benchmarks.utils.servers.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "benchmarks.utils.servers.httpx.AsyncClient",
        lambda: client_class(transport=transport),
    )
    manager = ServerManager(
        run_id="determinism",
        backend="vllm",
        model="model",
        store_dir=tmp_path,
        gpu_id="0",
        gpu_util=0.85,
        max_num_seqs=4,
        l1_size_bytes=1024,
        l2_size_bytes=1024,
        batch_invariant=requested,
    )
    await manager.start_vllm_only()
    assert engine_environments[0]["VLLM_BATCH_INVARIANT"] == str(int(expected))
    assert os.environ.get("VLLM_BATCH_INVARIANT") == inherited
    manifest_path = tmp_path / "manifest.json"
    manager.manifest().write(manifest_path)
    assert BenchmarkManifest.read(manifest_path).batch_invariant == expected
