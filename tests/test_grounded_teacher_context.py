"""Privileged teacher context tests; synthetic truth never becomes actor input."""

import json

import pytest
from test_native_corrections import fixture

from dronedream_agent_core.contracts import RuntimeLocalSafetyObservation
from dronedream_agent_core.control_execution_evidence import ControlApplicationRecord
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.grounded_teacher_context import grounded_teacher_context


# 功能：
#   从合成原生回合取得共享上下文编译器的全部输入，不连接仿真进程或飞机。
# 输入：
#   tmp_path：测试回合目录。
# 输出：
#   inputs：依次为学生观测、源快照、独立见证、实际回执、教师的元组。
def context_inputs(tmp_path):
    oracle, observation, path, _ = fixture(tmp_path)
    row = json.loads(path.read_bytes())
    inputs = (observation, row["source_snapshot"],
              RuntimeLocalSafetyObservation.model_validate(row["outcome_receipt"]["observations"][0]),
              ControlApplicationRecord.model_validate(row["application"]), oracle.teacher)
    return inputs


# 功能：
#   验证 NaN 时效和非法见证速度不能借助模型实例跳过字段验证进入离线教师状态。
# 输入：
#   tmp_path：测试回合目录。
#   field：故意绕过验证修改的见证字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["age", "velocity"])
def test_grounded_context_revalidates_witness(tmp_path, field):
    observation, snapshot, witness, application, teacher = context_inputs(tmp_path)
    if field == "age":
        witness = witness.model_copy(update={"stream_age_seconds": float("nan")})
    else:
        witness = witness.model_copy(update={"current_velocity_mps":
            witness.current_velocity_mps.model_copy(update={"x": float("nan")})})
    with pytest.raises(ValueError):
        grounded_teacher_context(observation, snapshot, witness, application, teacher, binding={})


# 功能：
#   验证上下文同时绑定快照原始内容和学生记录中的身份，重算摘要也不能换成另一帧。
# 输入：
#   tmp_path：测试回合目录。
#   rehash：是否给改动后的快照重新计算摘要。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("rehash", [False, True])
def test_grounded_context_requires_original_snapshot(tmp_path, rehash):
    observation, snapshot, witness, application, teacher = context_inputs(tmp_path)
    snapshot["goal_position_m"]["x"] += .1
    if rehash:
        snapshot.pop("snapshot_sha256")
        snapshot["snapshot_sha256"] = sha256_json(snapshot)
    with pytest.raises(ValueError, match="SNAPSHOT"):
        grounded_teacher_context(observation, snapshot, witness, application, teacher, binding={})


# 功能：
#   验证缺失飞行姿态编码时明确拒绝，不能泄漏迭代终止异常或继续编造姿态。
# 输入：
#   tmp_path：测试回合目录。
# 输出：
#   None：不返回业务数据。
def test_grounded_context_rejects_absent_orientation(tmp_path):
    observation, snapshot, witness, application, teacher = context_inputs(tmp_path)
    features = snapshot["realtime_feature_snapshot"]
    features["encodings"] = [r for r in features["encodings"]
                             if r["encoder_role"] != "flight-state-encoder"]
    features["required_roles"] = [r for r in features["required_roles"]
                                 if r != "flight-state-encoder"]
    snapshot.pop("snapshot_sha256")
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    observation.sample.source_snapshot_sha256 = snapshot["snapshot_sha256"]
    with pytest.raises(ValueError):
        grounded_teacher_context(observation, snapshot, witness, application, teacher, binding={})


# 功能：
#   验证返回状态与回执不再共享调用方的见证速度和绑定字典，摘要在返回后保持稳定。
# 输入：
#   tmp_path：测试回合目录。
# 输出：
#   None：不返回业务数据。
def test_grounded_context_owns_witness_and_binding(tmp_path):
    observation, snapshot, witness, application, teacher = context_inputs(tmp_path)
    binding = {"sample": {"values": [1, 2]}}
    state, context = grounded_teacher_context(observation, snapshot, witness, application,
                                              teacher, binding=binding)
    expected = state.velocity.x
    assert sha256_json(context) == state.context_sha256
    witness.current_velocity_mps.x += 1
    binding["sample"]["values"].append(3)
    assert state.velocity.x == expected
    assert sha256_json(context) == state.context_sha256


# 功能：
#   验证见证时间先外推到源观测时刻，并明确累积位置不确定度及实测执行延迟。
# 输入：
#   tmp_path：测试回合目录。
# 输出：
#   None：不返回业务数据。
def test_grounded_context_predicts_only_bounded_time_offset(tmp_path):
    observation, snapshot, witness, application, teacher = context_inputs(tmp_path)
    witness = witness.model_copy(update={"observed_at_unix_ms": 950})
    state, context = grounded_teacher_context(observation, snapshot, witness, application,
                                              teacher, binding={})
    assert context["pose_prediction_seconds"] == .05
    assert state.position.x == pytest.approx(
        witness.current_position_m.x + .05 * witness.current_velocity_mps.x)
    assert state.additional_position_uncertainty_m == pytest.approx(
        .5 * teacher.config.acceleration_mps2 * .05**2)
    assert state.observed_action_latency_seconds == .1
    assert observation.sample.temporal_evidence.observed_at_unix_ms == 1000
