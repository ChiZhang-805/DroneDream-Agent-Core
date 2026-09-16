"""Exercise the actual nested witness branch without starting Gazebo."""

import ast
import json
from collections import deque
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest
from test_training_runtime_evidence import pose

from dronedream_agent_core import gazebo_adapter
from dronedream_agent_core.training.runtime_evidence import (
    IndependentPoseMonitor,
    OutcomeWindowError,
)


# 功能：
#   执行实际原生观测分支，核对只发布见证后返回，不落入真值导航指令生成。
# 输入：
#   tmp_path：临时观测输出路径。
# 输出：
#   None：不返回业务数据。
def test_native_pose_worker_publishes_observation_without_truth_navigation_commands(tmp_path):
    tree = ast.parse(Path(gazebo_adapter.__file__).read_text(encoding="utf-8"))
    process = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "process_pose"
    )
    branch = next(
        node
        for node in ast.walk(process)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "depth_safety_supported"
    )
    # Execute the exact production branch in a minimal scope. Reaching the
    # following marker represents erroneously continuing to truth planning.
    function = ast.parse("def witness():\n    raise RuntimeError('truth planner reached')").body[0]
    function.body.insert(0, branch)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    writes = []
    scope = {
        "depth_safety_supported": True,
        "observation": pose(1, 0),
        "elapsed": 1.0,
        "local_safety_observation_path": tmp_path / "observation.json",
        "_write_bytes_atomic": lambda path, content: writes.append((path, content)),
    }
    exec(compile(module, "native-witness-branch", "exec"), scope)
    scope["witness"]()
    assert len(writes) == 1
    assert json.loads(writes[0][1]) == scope["observation"].model_dump(mode="json")
    assert "command_position_m" not in json.loads(writes[0][1])
    scope["depth_safety_supported"] = False
    with pytest.raises(RuntimeError, match="truth planner reached"):
        scope["witness"]()
    assert len(writes) == 1  # Other explicitly supported modes keep their own controller.


# 功能：
#   核对缺失窗口保留真实首尾间隙与独立诊断快照，不放宽阈值以伪造通过。
# 输入：
#   无；使用两帧合成独立见证。
# 输出：
#   None：不返回业务数据。
def test_rejected_witness_retains_real_gap_without_relaxing_thresholds():
    monitor = object.__new__(IndependentPoseMonitor)
    monitor._lock, monitor._rows, monitor._error = Lock(), deque([pose(1, 0), pose(2, 0.1)]), None
    with pytest.raises(OutcomeWindowError, match="UNVERIFIED_GAP") as failed:
        monitor.window(1050, 1301)
    details = failed.value.diagnostics
    assert details["start_gap_ms"] == 0
    assert details["end_gap_ms"] == 201
    assert details["maximum_internal_gap_ms"] == 50
    assert details["selected_count"] == 2 and not details["qualification_granted"]
    monitor._rows[0].current_position_m.x = 5.0
    assert details["selected_observations"][0]["current_position_m"]["x"] == 0.0


# 功能：
#   核对未来已缓存帧不能被改写为缺失的过去帧，同时保留拒绝原因。
# 输入：
#   无；使用一个晚于请求窗口的合成见证。
# 输出：
#   None：不返回业务数据。
def test_no_future_witness_is_relabelled_as_a_missing_past_sample():
    monitor = object.__new__(IndependentPoseMonitor)
    monitor._lock, monitor._rows, monitor._error = Lock(), deque([pose(2, 0)]), None
    with pytest.raises(OutcomeWindowError, match="NOT_YET_OBSERVED") as failed:
        monitor.window(1050, 1099)
    assert failed.value.diagnostics["selected_count"] == 0
    assert (
        failed.value.diagnostics["latest_available_observations"][0]["observed_at_unix_ms"] == 1100
    )


# 功能：
#   执行实际观测构造表达式，确认排队时间仍计入年龄，不刷新延迟帧的源时间。
# 输入：
#   无；显式提供接收和处理时刻及最小作用域。
# 输出：
#   None：不返回业务数据。
def test_delayed_simulation_witness_preserves_receive_clock_and_age():
    tree = ast.parse(Path(gazebo_adapter.__file__).read_text(encoding="utf-8"))
    process = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "process_pose"
    )
    assignment = next(
        node
        for node in ast.walk(process)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "RuntimeLocalSafetyObservation"
    )
    scope = {
        "RuntimeLocalSafetyObservation": gazebo_adapter.RuntimeLocalSafetyObservation,
        "Vector3": gazebo_adapter.Vector3,
        "local_safety_sequence": 1,
        "received_at_unix_ms": 1000,
        "received_at": 1.0,
        "time": SimpleNamespace(monotonic=lambda: 1.2),
        "observation_age_seconds": 0.2,
        "previous": (0, (0.0, 0.0, 1.0)),
        "current": gazebo_adapter.Vector3(x=0.0, y=0.0, z=1.0),
        "velocity": (0.0, 0.0, 0.0),
        "target": gazebo_adapter.Vector3(x=1.0, y=0.0, z=1.0),
        "obstacles": [],
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
            "source-bound-witness",
            "exec",
        ),
        scope,
    )
    observation = scope["observation"]
    assert observation.observed_at_unix_ms == 1000
    assert observation.stream_age_seconds == pytest.approx(0.2)
    assert not observation.stream_healthy
