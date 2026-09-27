"""Causal native collection teacher tests; fixtures never enter formal datasets."""

import copy
import json

import pytest
from test_decision_runtime_adapter import snapshot

from dronedream_agent_core.decision_collection_teacher import CollectionTeacher
from dronedream_agent_core.decision_state_adapter import DecisionStateV2, decision_digest
from dronedream_agent_core.pilot_control_mapping import PilotControlLimits


# 功能：将双帧动态证据接入真实教师接口，而非只测试孤立几何函数。
# 输入：合成横穿及离开记录；输出：等待之后在同一任务内恢复路线提案。
def test_teacher_waits_for_observed_crossing_and_resumes():
    from test_decision_dynamic_context import pair, seal

    teacher, template = fixture()
    old, new = pair()
    parameters = dict(
        mission_id="fixture",
        calibration_sha256="a" * 64,
        braking_acceleration_mps2=0.8,
        scene="corridor",
    )
    for index, value in enumerate((old, new)):
        value["realtime_feature_snapshot"]["encodings"].append(
            template["realtime_feature_snapshot"]["encodings"][1]
        )
        seal(value)
        state, _, _ = teacher.observe(value, sequence=index, **parameters)
    action, pilot, _ = teacher.propose(state, new, PilotControlLimits(0.6, 0.6, 20.0))
    assert action == "wait" and pilot.mode == "hold"
    # 连续保留真实格式观测，不通过删除目标假称“已离开”。
    for index in range(1, 20):
        value = copy.deepcopy(new)
        value["control_reference_observed_at_unix_ms"] += 100 * index
        for encoder in value["realtime_feature_snapshot"]["encodings"]:
            encoder["observed_at_unix_ms"] += 100 * index
        value["dynamic_obstacles"][0]["position_m"]["y"] -= 0.1 * index
        seal(value)
        state, _, _ = teacher.observe(value, sequence=index + 1, **parameters)
    assert state.frame.crossing_obstacle is False
    action, pilot, _ = teacher.propose(state, value, PilotControlLimits(0.6, 0.6, 20.0))
    assert action == "follow_route" and pilot.mode == "pilot-control"


# 功能：建立无飞行权限的测试教师与带来源的观测；输入：无；输出：合成夹具。
def fixture():
    raw = snapshot()
    raw["realtime_feature_snapshot"]["encodings"].append(
        {
            "encoder_role": "metric-geometry-encoder",
            "source_sha256": "e" * 64,
            "observed_at_unix_ms": 9900,
            "maximum_age_milliseconds": 350,
        }
    )
    raw.pop("snapshot_sha256")
    raw["snapshot_sha256"] = decision_digest(raw)
    teacher = CollectionTeacher(
        map_sha256="c" * 64,
        primitives=[
            {
                "shape": "box",
                "center_x": 20.0,
                "center_y": 0.0,
                "center_z": 0.0,
                "size_x": 1.0,
                "size_y": 1.0,
                "size_z": 1.0,
            }
        ],
        body_radius_m=0.25,
        body_height_m=0.3,
        nominal_speed_mps=0.4,
        clearance_m=0.2,
    )
    return teacher, raw


# 功能：静态核验只声明局部路线，不伪造六向距离、动态清空或制动证据；输出：严格断言。
def test_static_prefix_does_not_invent_sensor_coverage():
    teacher, raw = fixture()
    state, proof, issues = teacher.observe(
        raw,
        mission_id="fixture",
        sequence=1,
        calibration_sha256="a" * 64,
        braking_acceleration_mps2=0.8,
        scene="corridor",
    )
    assert state.frame.local_route_verified
    assert state.frame.route_source.observed_at_ms == 9900
    assert state.frame.clearance_front_m is None
    assert state.frame.crossing_obstacle is None and state.frame.can_brake is None
    assert state.frame.route_source.evidence_sha256 == decision_digest(proof)
    action, pilot, reason = teacher.propose(state, raw, PilotControlLimits(0.6, 0.6, 20.0))
    assert action == "follow_route" and pilot.mode == "pilot-control"
    assert sum((v * 0.6) ** 2 for v in pilot.axes[:3]) <= 0.4**2 + 1e-9


# 功能：缺观测不能复用静态地图冒充全传感器有效；输入：过期深度；输出：补观测提案。
def test_stale_geometry_requests_observation_without_abort():
    teacher, raw = fixture()
    state, _, _ = teacher.observe(
        raw,
        mission_id="fixture",
        sequence=1,
        calibration_sha256="a" * 64,
        braking_acceleration_mps2=0.8,
        scene="corridor",
    )
    document = state.model_dump(mode="json")
    document["frame"]["geometry_source"]["observed_at_ms"] = 9000
    state = DecisionStateV2.model_validate_json(json.dumps(document))
    action, pilot, _ = teacher.propose(state, raw, PilotControlLimits(0.6, 0.6, 20.0))
    assert action == "request_observation" and pilot.mode == "request-new-scan"


# 功能：地图内容身份不匹配时拒绝跨图计算，而非复用旧地图合格结论；输出：明确错误。
def test_map_identity_is_required():
    teacher, raw = fixture()
    teacher.map_sha256 = "f" * 64
    with pytest.raises(ValueError, match="MAP_CHANGED"):
        teacher.observe(
            raw,
            mission_id="fixture",
            sequence=1,
            calibration_sha256="a" * 64,
            braking_acceleration_mps2=0.8,
            scene="corridor",
        )


# 功能：低名义速度下“降速”仍须真的降低上限；输入：0.1米/秒；输出：0.05米/秒。
def test_slowdown_remains_distinct_at_low_nominal_speed():
    teacher, raw = fixture()
    teacher.nominal_speed_mps = 0.1
    state, _, _ = teacher.observe(
        raw,
        mission_id="fixture",
        sequence=1,
        calibration_sha256="a" * 64,
        braking_acceleration_mps2=0.8,
        scene="corridor",
    )
    document = state.model_dump(mode="json")
    document["frame"]["caution_required"] = True
    state = DecisionStateV2.model_validate_json(json.dumps(document))
    action, _, reason = teacher.propose(state, raw, PilotControlLimits(0.6, 0.6, 20.0))
    assert action == "slow_down" and reason["requested_speed_limit_mps"] == 0.05


# 功能：定位不连续时必须清空堵塞计时；输入：新参考时刻与过期定位；输出：不续计。
def test_missing_pose_breaks_persistent_blockage_evidence():
    teacher, raw = fixture()
    teacher.previous_blocked = ("old", 7000)
    raw["control_reference_observed_at_unix_ms"] = 10300
    raw.pop("snapshot_sha256")
    raw["snapshot_sha256"] = decision_digest(raw)
    state, _, _ = teacher.observe(
        raw,
        mission_id="fixture",
        sequence=1,
        calibration_sha256="a" * 64,
        braking_acceleration_mps2=0.8,
        scene="corridor",
    )
    assert teacher.previous_blocked is None
    assert state.frame.persistent_blockage is None


# 功能：机型上限小于教师期望速度时保持方向整体缩放；输入：低控制上限；输出：合法四轴。
def test_vehicle_limits_scale_proposal_without_changing_direction():
    teacher, raw = fixture()
    state, _, _ = teacher.observe(
        raw,
        mission_id="fixture",
        sequence=1,
        calibration_sha256="a" * 64,
        braking_acceleration_mps2=0.8,
        scene="corridor",
    )
    _, high, _ = teacher.propose(state, raw, PilotControlLimits(0.6, 0.6, 20.0))
    _, low, _ = teacher.propose(state, raw, PilotControlLimits(0.05, 0.05, 20.0))
    assert max(abs(v) for v in low.axes) <= 1
    factor = max(abs(v) for v in high.axes)
    assert low.axes == pytest.approx([v / factor for v in high.axes])


# 功能：两个新鲜但间隔很大的帧不能证明持续堵塞；输入：旧计时与当前有效帧；输出：重置。
def test_gap_between_fresh_frames_breaks_blockage_duration():
    teacher, raw = fixture()
    teacher.previous_observed_ms = 7000
    teacher.previous_blocked = ("old", 7000)
    # 故意放到碰撞附近，排除“因路线通过而重置”的另一条分支。
    teacher.clearance_m = 100
    state, _, _ = teacher.observe(
        raw,
        mission_id="fixture",
        sequence=1,
        calibration_sha256="a" * 64,
        braking_acceleration_mps2=0.8,
        scene="corridor",
    )
    assert state.frame.local_route_verified is False
    assert state.frame.persistent_blockage is False
    assert teacher.previous_blocked[1] == state.frame.observed_at_ms
