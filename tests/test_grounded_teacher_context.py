"""Privileged teacher context tests; synthetic truth never becomes actor input."""

import json

import pytest
from test_native_corrections import fixture

from dronedream_agent_core.contracts import RuntimeLocalSafetyObservation
from dronedream_agent_core.control_execution_evidence import ControlApplicationRecord
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.realtime_feature_encoders import (
    RealtimeFeatureEncoding,
    fuse_realtime_features,
)
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
#   固定合成学生观测中的定位方差编码，重算同一测试快照身份，不改动生产来源。
# 输入：
#   tmp_path：独立测试目录；encoded：归一化方差；valid：来源有效掩码。
# 输出：
#   inputs：已绑定测试观测和独立教师上下文输入。
def covariance_inputs(tmp_path, encoded, valid=1.0):
    inputs = context_inputs(tmp_path)
    observation, snapshot, *_ = inputs
    flight = next(row for row in snapshot["realtime_feature_snapshot"]["encodings"]
                  if row["encoder_role"] == "flight-state-encoder")
    flight["features"][7], flight["valid_mask"][7] = encoded, valid
    feature_snapshot = snapshot["realtime_feature_snapshot"]
    snapshot["realtime_feature_snapshot"] = fuse_realtime_features(
        [RealtimeFeatureEncoding.model_validate(row) for row in feature_snapshot["encodings"]],
        captured_at_unix_ms=feature_snapshot["captured_at_unix_ms"],
        required_roles=feature_snapshot["required_roles"],
    ).model_dump(mode="json")
    snapshot.pop("snapshot_sha256")
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    observation.sample.source_snapshot_sha256 = snapshot["snapshot_sha256"]
    return inputs


# 功能：
#   检查独立真值不能抹掉学生定位误差，且已有固定位置余量不被重复相加。
# 输入：
#   tmp_path：独立测试目录；variance：学生当时已观测的方差。
# 输出：
#   None：断言不符合时测试失败。
@pytest.mark.parametrize("variance", [0.0, 0.001, 0.04, 0.25])
def test_grounded_context_retains_onboard_localization_margin(tmp_path, variance):
    inputs = covariance_inputs(tmp_path, variance / 0.25)
    state, context = grounded_teacher_context(*inputs, binding={})
    observation, _, witness, _, teacher = inputs
    dt = (observation.sample.temporal_evidence.observed_at_unix_ms
          - witness.observed_at_unix_ms) / 1000
    prediction = 0.5 * teacher.config.acceleration_mps2 * dt * dt
    total = teacher.config.position_uncertainty_m + state.additional_position_uncertainty_m
    assert total == pytest.approx(max(teacher.config.position_uncertainty_m,
                                     3 * variance ** 0.5) + prediction)
    assert context["localization_covariance_m2"] == pytest.approx(variance)
    assert state.context_sha256 == sha256_json(context)


# 功能：
#   拒绝缺失或饱和的方差编码，不能从截断输入假造可信不确定度上界。
# 输入：
#   tmp_path：独立测试目录；encoded、valid：不可用的定位编码组合。
# 输出：
#   None：断言不符合时测试失败。
@pytest.mark.parametrize("encoded,valid", [(0.2, 0.0), (4.0, 1.0), (3.9, 1.0)])
def test_grounded_context_rejects_unfunded_localization(tmp_path, encoded, valid):
    with pytest.raises(ValueError, match="LOCALIZATION_UNCERTAINTY"):
        grounded_teacher_context(*covariance_inputs(tmp_path, encoded, valid), binding={})


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


# 功能：
#   控制快照可以晚于原始传感器，但不能晚于实际执行或重新起算传感器和见证年龄。
# 输入：
#   tmp_path：测试目录；reference：组装快照时间；valid：是否符合原始时间边界。
# 输出：
#   None：合法处理延迟保留完整延迟；回退、超时和执行后快照被拒绝。
@pytest.mark.parametrize("reference,valid", [(1059, True), (999, False), (1101, False),
                                           (1251, False), (True, False)])
def test_reference_clock_is_not_sensor_clock(tmp_path, reference, valid):
    observation, snapshot, witness, application, teacher = context_inputs(tmp_path)
    snapshot["control_reference_observed_at_unix_ms"] = reference
    snapshot.pop("snapshot_sha256")
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    observation.sample.source_snapshot_sha256 = snapshot["snapshot_sha256"]
    if not valid:
        with pytest.raises(ValueError, match="SOURCE_TIME_MISMATCH"):
            grounded_teacher_context(observation, snapshot, witness, application,
                                     teacher, binding={})
        return
    state, context = grounded_teacher_context(observation, snapshot, witness, application,
                                              teacher, binding={})
    assert context["source_ms"] == 1000
    assert context["control_reference_ms"] == reference
    assert state.observed_action_latency_seconds == .1
    assert context["pose_prediction_seconds"] == 0.


# 功能：
#   拒绝用另一个编码来源或时钟给合法快照补造训练时间证据。
# 输入：
#   tmp_path：隔离目录；field：待破坏的时序来源字段。
# 输出：
#   None：重新绑定快照摘要也不能跨过编码来源一致性检查。
@pytest.mark.parametrize("field", ["sample_sha256", "observed_at_unix_ms", "history_slot_revision"])
def test_context_binds_temporal_source_to_actual_encoding(tmp_path, field):
    observation, snapshot, witness, application, teacher = context_inputs(tmp_path)
    witness = witness.model_copy(update={"observed_at_unix_ms": 950})
    temporal = observation.sample.temporal_evidence
    setattr(temporal, field, "f" * 64 if field == "sample_sha256" else
            getattr(temporal, field) - 1 if field == "observed_at_unix_ms" else 1)
    with pytest.raises(ValueError, match="CONTEXT_SOURCE_ENCODING_MISMATCH"):
        grounded_teacher_context(observation, snapshot, witness, application, teacher, binding={})
