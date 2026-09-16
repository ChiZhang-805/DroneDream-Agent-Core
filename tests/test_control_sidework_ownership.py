"""Stable asynchronous inputs and mandatory evidence after model scheduling."""

import ast
import inspect
import json
import threading
from types import SimpleNamespace

import pytest
from control_fixtures import complete_feature_snapshot
from test_perception_runtime import _frame, _world

import dronedream_agent_core.perception_runtime as perception_runtime
import scripts.runtime_depth_safety_worker as worker
from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.perception_runtime import (
    EventDrivenIndoorNavigationCoordinator,
    RuntimePerceptionFusion,
)


@pytest.mark.parametrize("mode", ["legacy-candidate-selection", "normalized-body-velocity"])
def test_async_compiler_and_pending_goal_own_their_input_graph(monkeypatch, mode):
    fusion = RuntimePerceptionFusion(world=_world(), accepted_sensor_ids={"front-lidar"})
    fusion.ingest(_frame(), now_unix_ms=1020, now_monotonic_seconds=1.02)
    entered, release = threading.Event(), threading.Event()
    recorded = []

    def capture(**kwargs):
        entered.set()
        assert release.wait(2.)
        recorded.append(kwargs)

    monkeypatch.setattr(perception_runtime, "_compile_and_request_navigation_decision", capture)
    coordinator = EventDrivenIndoorNavigationCoordinator(fusion=fusion, port=SimpleNamespace(),
        required_clearance_m=0., control_output_mode=mode)
    goal = Vector3(x=3.75, y=.25, z=.25)
    features = complete_feature_snapshot().model_dump(mode="json")
    original_feature_hash = sha256_json(features)
    context = {"task": {"phase": "TRANSIT", "local_navigation_output_mode": mode}}
    try:
        assert coordinator.schedule(goal_position_m=goal, now_unix_ms=1020, trigger="initial",
            strategic_context=context, realtime_feature_snapshot=features) is None
        assert entered.wait(1.)
        pending = coordinator._pending
        goal.x = 999.
        features["encodings"][0]["features"][0] = 999.
        features["fused_features"][0] = 999.
        features["captured_at_unix_ms"] = 999999
        context["task"]["phase"] = "FAILED"
        assert pending.goal_position_m.x == 3.75
        assert pending.submitted_at_unix_ms == 1020
        release.set()
        pending.future.result(timeout=2.)
        assert recorded[0]["goal_position_m"].x == 3.75
        assert sha256_json(recorded[0]["realtime_feature_snapshot"]) == original_feature_hash
        assert recorded[0]["strategic_context"]["task"]["phase"] == "TRANSIT"
        assert recorded[0]["frame"].observed_at_unix_ms == 1000
    finally:
        release.set()
        coordinator.close()


def test_diagnostic_json_is_owned_before_background_write(monkeypatch, tmp_path):
    started, release = threading.Event(), threading.Event()
    original = worker._atomic_bytes
    path = tmp_path / "health.json"
    writer = worker._LatestRuntimeSnapshotWriter(tmp_path / "writer.json")

    def delayed(destination, payload, **kwargs):
        if destination == path:
            started.set()
            assert release.wait(2.)
        return original(destination, payload, **kwargs)

    monkeypatch.setattr(worker, "_atomic_bytes", delayed)
    payload = {"observed_at_unix_ms": 1000, "nested": {"values": [1., 2.]}}
    try:
        assert writer.submit_json(path, payload)
        assert started.wait(1.)
        payload["nested"]["values"][0] = 999.
        payload["observed_at_unix_ms"] = 999999
    finally:
        release.set()
        summary = writer.close(timeout_seconds=2.)
    assert summary["complete"]
    assert json.loads(path.read_bytes()) == {
        "observed_at_unix_ms": 1000, "nested": {"values": [1., 2.]}}
    assert b"\n  " not in path.read_bytes()


@pytest.mark.parametrize("payload", [{"bad": float("nan")}, {"bad": object()}])
def test_unserializable_snapshot_is_explicitly_rejected_without_creating_a_file(tmp_path, payload):
    writer = worker._LatestRuntimeSnapshotWriter(tmp_path / "writer.json")
    path = tmp_path / "invalid.json"
    try:
        assert not writer.submit_json(path, payload)
        assert writer.issue.startswith("RUNTIME_SNAPSHOT_SERIALIZATION_FAILED:")
    finally:
        summary = writer.close(timeout_seconds=2.)
    assert not summary["complete"]
    assert summary["rejected_count"] == 1
    assert not path.exists()


def test_mutable_binary_payload_is_not_borrowed(tmp_path):
    writer = worker._LatestRuntimeSnapshotWriter(tmp_path / "writer.json")
    try:
        assert not writer.submit_bytes(tmp_path / "frame.png", bytearray(b"mutable"))
    finally:
        summary = writer.close(timeout_seconds=2.)
    assert summary["issue_code"] == "RUNTIME_SNAPSHOT_BYTES_REQUIRED"


def test_worker_keeps_safety_first_and_evidence_in_model_error_finalizer():
    # This is a source-order integration invariant, not a timing or flight
    # qualification test. The actual runtime must separately measure latency.
    tree = ast.parse(inspect.getsource(worker._run_worker))

    def calls(nodes, text):
        return any(isinstance(node, ast.Call) and ast.unparse(node.func) == text
                   for parent in nodes for node in ast.walk(parent))

    stages = [node for node in ast.walk(tree) if isinstance(node, ast.Try)
              and calls(node.body, "model_navigation.schedule")
              and calls(node.finalbody, "learning_recorder.submit")]
    assert len(stages) == 1
    stage = stages[0]
    # Optional learning/media failures must not bypass the control record
    # either; merely placing both in one finally would not guarantee this.
    sidework = [node for node in stage.finalbody if isinstance(node, ast.Try)]
    assert len(sidework) == 1
    assert calls(sidework[0].body, "learning_recorder.submit")
    assert calls(sidework[0].finalbody, "runtime_evidence_writer.submit")
    history_calls = [node for parent in stage.finalbody for node in ast.walk(parent)
                     if isinstance(node, ast.Call) and node.args
                     and ast.unparse(node.func) == "runtime_evidence_writer.submit"
                     and ast.unparse(node.args[0]) == "local_safety_history"]
    assert len(history_calls) == 1
    publishers = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and ast.unparse(node.func) == "safety_publisher.send"]
    assert publishers and max(node.lineno for node in publishers) < stage.lineno
    assert not calls(stage.body, "learning_recorder.submit")
