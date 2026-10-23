"""Static setup completes before the unchanged fresh-sensor window begins."""
import asyncio
import json
from types import SimpleNamespace

import pytest

import scripts.runtime_depth_safety_worker as worker
from dronedream_agent_core import native_preflight


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def prepare(root, *, state="initializing", pid=111):
    write(root / "native-runtime-preflight-ready.json", {
        "depth_safety_worker_pid": pid,
        "static_map_binding": {"route_sha256": "a" * 64, "semantic_sha256": "b" * 64}})
    source = {"path": "worker.py", "sha256": "c" * 64, "static_initialization": {
        "pid": pid, "state": state, "route_sha256": "a" * 64,
        "semantic_sha256": "b" * 64, "elapsed_seconds": 76.}}
    write(root / "depth-worker-source.json", source)
    return source


def test_worker_publishes_ready_only_after_seed_and_preserves_source(tmp_path, monkeypatch):
    path = tmp_path / "depth-worker-source.json"
    write(path, {"path": "worker.py", "sha256": "c" * 64})
    def seed(**inputs):
        assert json.loads(path.read_text())["static_initialization"]["state"] == "initializing"
        return 100, 7
    monkeypatch.setattr(worker, "_seed_qualified_static_map", seed)
    assert worker._initialize_qualified_static_map(
        source_path=path, expected_semantic_sha256="b" * 64,
        qualified_clearance=SimpleNamespace(route_sha256="a" * 64)) == (100, 7)
    record = json.loads(path.read_text())
    assert record["sha256"] == "c" * 64
    assert record["static_initialization"]["state"] == "ready"
    assert record["static_initialization"]["occupied_voxels"] == 7


def test_worker_failure_is_published_and_original_error_propagates(tmp_path, monkeypatch):
    path = tmp_path / "depth-worker-source.json"
    write(path, {"path": "worker.py", "sha256": "c" * 64})
    def seed(**inputs):
        raise ValueError("true voxel budget exceeded")
    monkeypatch.setattr(worker, "_seed_qualified_static_map", seed)
    with pytest.raises(ValueError, match="true voxel budget"):
        worker._initialize_qualified_static_map(
            source_path=path, expected_semantic_sha256="b" * 64,
            qualified_clearance=SimpleNamespace(route_sha256="a" * 64))
    state = json.loads(path.read_text())["static_initialization"]
    assert state["state"] == "failed"
    assert "true voxel budget" in state["issue"]


def test_slow_static_seed_does_not_consume_live_health_window(tmp_path, monkeypatch):
    source = prepare(tmp_path)
    clock, health_reads = [0.], []
    monkeypatch.setattr(native_preflight, "time", SimpleNamespace(
        monotonic=lambda: clock[0], time=lambda: 1000 + clock[0]))
    monkeypatch.setattr(native_preflight, "_native_worker_alive", lambda pid: True)
    async def sleep(delay):
        clock[0] += 76. if clock[0] == 0 else delay
        source["static_initialization"]["state"] = "ready"
        write(tmp_path / "depth-worker-source.json", source)
    monkeypatch.setattr(native_preflight.asyncio, "sleep", sleep)
    def read_health():
        health_reads.append(clock[0])
        return {"schema_version": "dronedream.perception-fusion-health.v1",
                "stream_healthy": True, "identity_accepted": True,
                "truth_correction_applied": False,
                "pose_source": "native-estimator-fixed-deployment-binding",
                "latest_sequence": len(health_reads),
                "updated_at_unix_ms": int((1000 + clock[0]) * 1000)}
    evidence, health_evidence = {}, {}
    result = asyncio.run(native_preflight.wait_for_initialized_native_perception(
        tmp_path / "depth-perception-health.json", initialization_deadline=180.,
        initialization_evidence=evidence, evidence=health_evidence,
        receiver=SimpleNamespace(read_latest=read_health), stable_window_ms=0,
        require_realtime_features=False))
    assert result["ready"] is True
    assert not (tmp_path / "clearance.json").exists()
    assert evidence["elapsed_seconds"] == 76.
    assert health_reads and min(health_reads) == 76.
    assert health_evidence["elapsed_seconds"] < 1.


@pytest.mark.parametrize("old_state", ["ready", "failed"])
def test_supervisor_restart_ignores_old_pid_and_uses_new_worker(tmp_path, monkeypatch, old_state):
    source = prepare(tmp_path, state=old_state)
    marker = json.loads((tmp_path / "native-runtime-preflight-ready.json").read_text())
    marker["depth_safety_worker_pid"] = 222
    write(tmp_path / "native-runtime-preflight-ready.json", marker)
    clock, checks = [0.], []
    monkeypatch.setattr(native_preflight, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(native_preflight, "_native_worker_alive", lambda pid: pid == 222)
    async def sleep(delay):
        clock[0] += delay
        source["static_initialization"].update(pid=222, state="ready")
        write(tmp_path / "depth-worker-source.json", source)
    async def health(*args, **kwargs):
        checks.append(clock[0])
        return {"ready": True}
    monkeypatch.setattr(native_preflight.asyncio, "sleep", sleep)
    monkeypatch.setattr(native_preflight, "wait_for_native_perception", health)
    evidence = {}
    asyncio.run(native_preflight.wait_for_initialized_native_perception(
        tmp_path / "health.json", initialization_deadline=1., initialization_evidence=evidence))
    assert checks == [.1]
    assert evidence["worker_pid"] == 222


@pytest.mark.parametrize("failure", ["failed", "binding", "dead"])
def test_failure_or_dead_worker_cannot_reach_sensor_gate(tmp_path, monkeypatch, failure):
    source = prepare(tmp_path, state="failed" if failure == "failed" else "ready")
    if failure == "binding":
        source["static_initialization"]["route_sha256"] = "d" * 64
        write(tmp_path / "depth-worker-source.json", source)
    clock = [9.95]
    monkeypatch.setattr(native_preflight, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(native_preflight, "_native_worker_alive", lambda pid: failure != "dead")
    async def sleep(delay):
        clock[0] += delay
    async def health(*args, **kwargs):
        pytest.fail("initialization did not grant permission to inspect live health")
    monkeypatch.setattr(native_preflight.asyncio, "sleep", sleep)
    monkeypatch.setattr(native_preflight, "wait_for_native_perception", health)
    with pytest.raises(RuntimeError, match="STATIC_MAP_INITIALIZATION"):
        asyncio.run(native_preflight.wait_for_initialized_native_perception(
            tmp_path / "health.json", initialization_deadline=10., initialization_evidence={}))
    # The already-spent startup time is not reset by this wait or by a dead worker.
    assert clock[0] <= 10.05


def test_supervisor_restart_during_health_returns_to_static_initialization(tmp_path, monkeypatch):
    prepare(tmp_path, state="ready")
    clock, calls, cancelled = [0.], [], []
    monkeypatch.setattr(native_preflight, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(native_preflight, "_native_worker_alive", lambda pid: True)

    async def sleep(delay):
        # The replacement has to finish its own static initialization.
        assert cancelled == [111]
        clock[0] += 76.
        prepare(tmp_path, state="ready", pid=222)

    async def health(*args, **kwargs):
        calls.append((clock[0], kwargs["timeout_seconds"]))
        if len(calls) == 1:
            prepare(tmp_path, state="initializing", pid=222)
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(111)
        return {"ready": True, "replacement": True}

    monkeypatch.setattr(native_preflight.asyncio, "sleep", sleep)
    monkeypatch.setattr(native_preflight, "wait_for_native_perception", health)
    evidence = {}
    result = asyncio.run(native_preflight.wait_for_initialized_native_perception(
        tmp_path / "health.json", initialization_deadline=100., initialization_evidence=evidence))
    assert result == {"ready": True, "replacement": True}
    assert calls == [(0., 30.), (76., 24.)]
    assert cancelled == [111]
    assert evidence["worker_pid"] == 222


def test_health_completion_after_absolute_startup_deadline_cannot_allow_arm(tmp_path, monkeypatch):
    prepare(tmp_path, state="ready")
    clock, timeouts = [9.9], []
    monkeypatch.setattr(native_preflight, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(native_preflight, "_native_worker_alive", lambda pid: True)

    async def health(*args, **kwargs):
        timeouts.append(kwargs["timeout_seconds"])
        clock[0] = 12.
        return {"ready": True}

    monkeypatch.setattr(native_preflight, "wait_for_native_perception", health)
    evidence = {}
    with pytest.raises(RuntimeError, match="STATIC_MAP_INITIALIZATION_STARTUP_BUDGET_EXCEEDED"):
        asyncio.run(native_preflight.wait_for_initialized_native_perception(
            tmp_path / "health.json", initialization_deadline=10.,
            initialization_evidence=evidence))
    assert timeouts == [pytest.approx(.1)]
    assert evidence["status"] == "timeout"
