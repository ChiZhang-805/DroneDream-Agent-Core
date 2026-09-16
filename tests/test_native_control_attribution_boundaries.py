"""Synthetic control receipts and capture mutations; no aircraft is connected."""

import json
import math
from dataclasses import replace

import pytest
from test_executed_control_training import teacher_evidence
from test_native_corrections import fixture

from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand, RuntimeLocalSafetyObservation
from dronedream_agent_core.control_execution_evidence import ControlApplicationRecord
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.pilot_control_mapping import PilotControlLimits
from dronedream_agent_core.training.executed_control import (
    executed_pilot_control,
    transport_command_change_squared,
)
from dronedream_agent_core.training.flight_environment import FlightObservation, FlightStep
from dronedream_agent_core.training.native_transition import (
    CapturedNativeTransition,
    evaluate_native_transition,
)
from dronedream_agent_core.training.outcome_verifier import SimulationOutcomeVerifier
from dronedream_agent_core.training.policy_exchange import TrainingProposal


# 功能：
#   建立绑定同一任务快照的原生捕获，使用合成执行回执和独立几何见证。
# 输入：
#   tmp_path：测试独占临时目录。
# 输出：
#   capture：合法的单步捕获。
#   verifier：不访问飞机的独立结果评估器。
def native_capture(tmp_path):
    oracle, _, path, _ = fixture(tmp_path)
    row = json.loads(path.read_bytes())
    snapshot = row["source_snapshot"]
    snapshot["strategic_context"]["task"]["navigation_goal_id"] = "office"
    snapshot.pop("snapshot_sha256")
    digest = snapshot["snapshot_sha256"] = sha256_json(snapshot)
    row["source_observation"]["sample"]["source_snapshot_sha256"] = digest
    row["command"]["model_navigation_snapshot_sha256"] = digest
    row["command"]["requested_control_intent"]["navigation_snapshot_sha256"] = digest
    command = RuntimeLocalSafetyCommand.model_validate(row["command"])
    row["application"]["command_sha256"] = sha256_json(command)
    row["application"]["intent"]["navigation_snapshot_sha256"] = digest
    step = FlightStep.model_validate(row["step"])
    capture = CapturedNativeTransition(
        FlightObservation.model_validate(row["source_observation"]), step.observation, snapshot,
        TrainingProposal.model_validate(row["proposal"]), command,
        ControlApplicationRecord.model_validate(row["application"]), None, step.applied_action,
        False, tuple(RuntimeLocalSafetyObservation.model_validate(value)
                     for value in row["outcome_receipt"]["observations"]), False,
    )
    verifier = SimulationOutcomeVerifier(
        oracle.teacher.geometry.primitives, oracle.teacher.envelope)
    return capture, verifier


# 功能：
#   验证首个回执也必须合法，不能因只建立基准就把 NaN、布尔值或非法长度判为零代价。
# 输入：
#   previous_present：是否已有上一回执。
#   velocity：故意损坏的当前 NED 速度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("previous_present", [False, True])
@pytest.mark.parametrize("velocity", [(math.nan, 0., 0.), (True, 0., 0.), (0., 0.)])
def test_change_cost_revalidates_even_initial_receipt(previous_present, velocity):
    _, _, application, limits = teacher_evidence()
    current = application.model_copy(update={"velocity_ned_mps": velocity})
    with pytest.raises(ValueError):
        transport_command_change_squared(application if previous_present else None,
                                         current, limits=limits)


# 功能：
#   验证尺度被绕过冻结保护篡改后仍拒绝，避免零除或 NaN 动作变化代价进入奖励。
# 输入：
#   bad_scale：非法的水平速度尺度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad_scale", [0., math.nan, True])
def test_change_cost_revalidates_physical_scales(bad_scale):
    _, _, application, limits = teacher_evidence()
    object.__setattr__(limits, "horizontal_speed_mps", bad_scale)
    with pytest.raises(ValueError, match="limits"):
        transport_command_change_squared(application, application, limits=limits)


# 功能：
#   验证有限但极端的回执速度或归一化尺度不能让平方计算返回无穷或抛出未分类溢出。
# 输入：
#   scale：合法正值尺度；极小值专门触发归一化溢出。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("scale", [.8, 1e-300])
def test_change_cost_rejects_derived_overflow(scale):
    _, _, application, _ = teacher_evidence()
    first = application.model_copy(update={"velocity_ned_mps": (-1e308, 0., 0.)})
    second = application.model_copy(update={"velocity_ned_mps": (1e308, 0., 0.)})
    with pytest.raises(ValueError, match="CHANGE_COST"):
        transport_command_change_squared(first, second, limits=PilotControlLimits(scale, .75, 20.))


# 功能：
#   验证极大但有限的等价航向先各自取模，避免先相减溢出导致虚构的转向代价。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_change_cost_wraps_each_heading_before_subtracting():
    _, _, application, limits = teacher_evidence()
    first = application.model_copy(update={"yaw_heading_deg": -1e308, "yaw_rate_application": None})
    second = application.model_copy(update={"yaw_heading_deg": 1e308, "yaw_rate_application": None})
    expected = ((1e308 % 360 - (-1e308 % 360) + 180) % 360 - 180) ** 2 / 180**2
    assert transport_command_change_squared(first, second, limits=limits) == pytest.approx(expected)


# 功能：
#   验证控制坐标系只能明确选择布尔值，禁止非空字符串静默选择执行姿态。
# 输入：
#   frame：并非布尔类型的坐标系选择参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("frame", ["false", 1, None])
def test_execution_frame_requires_explicit_boolean(frame):
    snapshot, command, application, limits = teacher_evidence()
    with pytest.raises(ValueError, match="FRAME"):
        executed_pilot_control(snapshot, command, application, limits=limits, execution_frame=frame)


# 功能：
#   验证非字符串键不能利用旧摘要的键规范化规则伪装成合法 JSON 快照。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_teacher_snapshot_rejects_non_json_keys_before_hashing():
    snapshot, command, application, limits = teacher_evidence()
    snapshot[1] = "not a JSON key"
    snapshot.pop("snapshot_sha256")
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    with pytest.raises(ValueError):
        executed_pilot_control(snapshot, command, application, limits=limits)


# 功能：
#   验证归档内宣称的实际动作和过期干预必须由回执独立恢复，不能任意替换。
# 输入：
#   tmp_path：合成捕获目录。
#   mutation：实际轴值或过期标记的篡改方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["axes", "deadline"])
def test_native_transition_recomputes_actual_action(tmp_path, mutation):
    capture, verifier = native_capture(tmp_path)
    evaluate_native_transition(capture, verifier)
    if mutation == "axes":
        capture.applied.axes[0] += .1
    else:
        capture = replace(capture, deadline_intervened=True)
    with pytest.raises(ValueError, match="APPLIED|DEADLINE"):
        evaluate_native_transition(capture, verifier)


# 功能：
#   验证返回的完整训练记录独立持有快照，调用方后续修改捕获不改变已经评估的记录。
# 输入：
#   tmp_path：合成捕获目录。
# 输出：
#   None：不返回业务数据。
def test_native_transition_record_owns_evaluated_snapshot(tmp_path):
    capture, verifier = native_capture(tmp_path)
    step, record = evaluate_native_transition(capture, verifier)
    digest = sha256_json(record)
    capture.snapshot["goal_position_m"]["x"] += 1
    capture.observation.sample.state_features[0] += 1
    assert sha256_json(record) == digest
    assert step.observation.sample.state_features != capture.observation.sample.state_features
