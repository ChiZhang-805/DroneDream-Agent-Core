"""Coordinate/provenance tests for deployment observations, never truth substitution."""

import math

import pytest

from dronedream_agent_core.decision_runtime_adapter import (
    enu_to_frd,
    state_from_navigation_snapshot,
)
from dronedream_agent_core.decision_state_adapter import decision_digest


# 功能：生成最小可核验导航记录；输入：无；输出：明确合成的快照，不纳入训练。
def snapshot():
    value = {"schema_version": "dronedream.text-navigation-snapshot.v1",
             "source_of_truth": "metric-range-and-localization-evidence-not-rendered-image",
             "control_reference_observed_at_unix_ms": 10000,
             "strategic_context": {"task": {"navigation_goal_id": "test-goal"},
                                   "vehicle": {"body_radius_m": .25}},
             "current_position_m": {"x": 1., "y": 2., "z": 3.},
             "goal_position_m": {"x": 2., "y": 4., "z": 6.},
             "current_velocity_mps": {"x": .1, "y": .2, "z": .3},
             "known_static_map": {"qualified_route_sha256": "b"*64, "source_sha256": "c"*64},
             "perception_health": {"localization_covariance_m2": .0025},
             "egocentric_sector_reference": "front points toward current semantic goal",
             "realtime_feature_snapshot": {"encodings": [{
                 "encoder_role": "flight-state-encoder", "source_sha256": "a"*64,
                 "feature_contract_sha256":
                 "48f9536199319c760afc8e208826a2d349eea2d5b49f2de1437d481b4f831ad9",
                 "observed_at_unix_ms": 9900, "maximum_age_milliseconds": 350,
                 "features": [1., 0., 0., 0.], "valid_mask": [1., 1., 1., 1.]}]}}
    value["snapshot_sha256"] = decision_digest(value)
    return value


# 功能：用显式测试标定调用适配器；输入：快照；输出：状态及问题列表。
def adapt(value):
    return state_from_navigation_snapshot(value, mission_id="test", sequence=1,
        calibration_sha256="d"*64, braking_acceleration_mps2=.8, scene="corridor")


# 功能：确认 FLU→FRD 符号及 yaw 旋转；输入：四元数/预期向量；输出：数值断言。
@pytest.mark.parametrize(("q", "expected"), [
    ([1., 0., 0., 0.], {"x": 1., "y": -2., "z": -3.}),
    ([math.sqrt(.5), 0., 0., math.sqrt(.5)], {"x": 2., "y": 1., "z": -3.}),
])
def test_body_transform(q, expected):
    assert enu_to_frd({"x": 1., "y": 2., "z": 3.}, q) == pytest.approx(expected)


# 功能：拒绝 NaN 和非单位姿态；输入：非法姿态；输出：明确失败。
@pytest.mark.parametrize("q", [[float("nan"), 0., 0., 0.], [2., 0., 0., 0.], [True, 0., 0., 0.]])
def test_invalid_orientation(q):
    with pytest.raises(ValueError, match="ORIENTATION"):
        enu_to_frd({"x": 1., "y": 0., "z": 0.}, q)


# 功能：目标对齐扇区不能冒充机体净空；输入：完整快照；输出：保留未知及原采样年龄。
def test_no_invented_clearance():
    state, issues = adapt(snapshot())
    assert state.frame.goal_relative_body_frd_m.model_dump() == {"x": 1., "y": -2., "z": -3.}
    assert state.frame.pose_source.observed_at_ms == 9900
    assert state.frame.position_uncertainty_m == .05
    assert state.frame.clearance_front_m is None
    assert state.frame.local_route_verified is None
    assert "GOAL_ALIGNED_SECTORS_NOT_BODY_CLEARANCES" in issues


# 功能：损坏内容、不同特征语义或仿真真值不能静默适配；输入：突变；输出：拒绝。
@pytest.mark.parametrize("mutation", ["hash", "encoder", "truth"])
def test_invalid_provenance(mutation):
    value = snapshot()
    if mutation == "hash":
        value["current_position_m"]["x"] += 1
    else:
        value.pop("snapshot_sha256")
        if mutation == "encoder":
            value["realtime_feature_snapshot"]["encodings"][0]["feature_contract_sha256"] = "e"*64
        else:
            value["source_of_truth"] = "simulation-ground-truth"
        value["snapshot_sha256"] = decision_digest(value)
    with pytest.raises(ValueError):
        adapt(value)


# 功能：丢失编码器不等于程序崩溃或虚构位置；输入：空编码；输出：unknown。
def test_missing_encoder():
    value = snapshot()
    value.pop("snapshot_sha256")
    value["realtime_feature_snapshot"] = None
    value["snapshot_sha256"] = decision_digest(value)
    state, issues = adapt(value)
    assert state.frame.position_world_enu_m is None
    assert state.frame.goal_relative_body_frd_m is None
    assert "BODY_ORIENTATION_UNKNOWN" in issues
