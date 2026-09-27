"""Reject self-declared success certificates and missing domain reviews."""

import copy
import json

import pytest

from dronedream_agent_core.decision_dataset import audit_records, file_digest
from dronedream_agent_core.decision_label_evidence import verify_label_evidence, verify_native_label
from dronedream_agent_core.decision_state_adapter import decision_digest
from scripts.prepare_decision_contract_smoke import smoke_rows


# 功能：写入测试 JSON 并返回真实摘要；输入：临时目录/名称/内容；输出：证据引用。
def reference(root, name, data):
    path = root / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return {"path": name, "sha256": file_digest(path)}


# 功能：证书和哈希都正确也不能把一个 success 字段当成飞行结果；输入：临时目录；输出：拒绝。
def test_boolean_certificate_is_not_physical_evidence(tmp_path):
    row = smoke_rows()[0]
    row.update(label_source="verified-simulation-outcome", formal_training_eligible=True)
    result = reference(tmp_path, "outcome.json", {"success": True})
    certificate = {"schema_version": "dronedream.decision-label-certificate.v1",
                   "state_sha256": decision_digest(row["state"]),
                   "parent_group": row["parent_group"], "episode_id": row["episode_id"],
                   "label_source": row["label_source"],
                   "acceptable_actions": row["acceptable_actions"],
                   "verified": True, "verifier_version": "test", "outcome_references": [result]}
    row["evidence"] = [reference(tmp_path, "certificate.json", certificate)]
    with pytest.raises(ValueError, match="NATIVE_OUTCOME_REQUIRED"):
        audit_records([row], tmp_path)


# 功能：伪称人工审核但没有审核记录不能准入；输入：临时目录；输出：拒绝。
def test_missing_expert_review(tmp_path):
    row = smoke_rows()[0]
    row["label_source"] = "reviewed-expert"
    evidence = reference(tmp_path, "review.json", {"verified": True})
    with pytest.raises(ValueError, match="EXPERT_REVIEW_REQUIRED"):
        verify_label_evidence(row, {"outcome_references": [evidence]}, tmp_path)


# 功能：独立结果必须绑定同一回合/状态；输入：突变；输出：在几何计算之前拒绝。
@pytest.mark.parametrize("fault", ["episode", "state", "application"])
def test_native_label_binding(fault):
    row = smoke_rows()[0]
    doc = {"schema_version": "dronedream.decision-native-outcome.v1",
           "parent_group": row["parent_group"], "episode_id": row["episode_id"],
           "state_sha256": decision_digest(row["state"]),
           "behavior_application": {"action": "follow_route"}}
    if fault == "episode":
        doc["episode_id"] = "another"
    elif fault == "state":
        doc["state_sha256"] = "0"*64
    with pytest.raises(ValueError):
        verify_native_label(row, doc)


# 功能：构造带原始命令/传输回执/连续独立见证的测试结果；输出：合成夹具而非正式数据。
def native_fixture():
    from dronedream_agent_core.contracts import (
        BodyFrameControlIntent,
        PredictiveSafetyDecision,
        QuaternionWxyz,
        RuntimeLocalSafetyCommand,
        RuntimeLocalSafetyObservation,
        Vector3,
    )
    from dronedream_agent_core.control_execution_evidence import control_application_record

    row = smoke_rows()[0]
    commands, applications, observations = [], [], []
    for index in range(11):
        stamp = 10000 + 200*index
        intent = BodyFrameControlIntent(source_expert="local-navigation-policy",
            control_origin="continuous-model-output", model_call_id="model-" + "a"*24,
            navigation_snapshot_sha256="b"*64, task_reference_sha256="c"*64,
            generated_at_unix_ms=stamp, valid_until_unix_ms=stamp+200,
            forward_velocity_mps=.25, right_velocity_mps=0., up_velocity_mps=0.,
            yaw_rate_dps=0., maximum_acceleration_mps2=1., maximum_jerk_mps3=2.)
        command = RuntimeLocalSafetyCommand(observation_sha256="d"*64,
            observation_sequence=index+1, generated_at_unix_ms=stamp,
            valid_until_unix_ms=stamp+200, source="onboard",
            evaluated_body_orientation_world_from_body=QuaternionWxyz(w=1., x=0., y=0., z=0.),
            evaluated_target_position_m=Vector3(x=2., y=0., z=1.5),
            navigation_goal_id=row["state"]["goal_id"],
            navigation_control_authority="model-required",
            model_navigation_authorized=True, model_call_id=intent.model_call_id,
            model_path_sha256=intent.task_reference_sha256,
            model_navigation_snapshot_sha256=intent.navigation_snapshot_sha256,
            requested_control_intent=intent, command_position_m=Vector3(x=.05*index, y=0., z=1.5),
            decision=PredictiveSafetyDecision(action="continue",
                selected_velocity_mps=Vector3(x=.25, y=0., z=0.), selected_yaw_rate_dps=0.,
                control_source="local-model-body-control",
                predicted_path_m=[Vector3(x=.05*index, y=0., z=1.5)],
                minimum_predicted_clearance_m=1., time_to_minimum_clearance_seconds=.1,
                evaluated_candidate_count=1))
        commands.append(command.model_dump(mode="json"))
        applications.append(control_application_record(command, sequence=index+1,
            accepted_at_unix_ms=stamp, transport="velocity-ned", velocity_ned_mps=(0., .25, 0.),
            yaw_heading_deg=0.).model_dump(mode="json"))
        observations.append(RuntimeLocalSafetyObservation(sequence=index+1,
            observed_at_unix_ms=stamp, source="simulation-ground-truth", stream_healthy=True,
            stream_age_seconds=0., localization_covariance_m2=0.,
            current_position_m=Vector3(x=.05*index, y=0., z=1.5),
            current_velocity_mps=Vector3(x=.25, y=0., z=0.),
            target_position_m=Vector3(x=2., y=0., z=1.5)).model_dump(mode="json"))
    primitives = [{"shape": "box", "name": "far-wall", "center_x": 5.,
                   "center_y": 0., "center_z": 1.5, "size_x": .5, "size_y": 2., "size_z": 3.}]
    state_sha = decision_digest(row["state"])
    return row, {"schema_version": "dronedream.decision-native-outcome.v1",
        "parent_group": row["parent_group"], "episode_id": row["episode_id"],
        "state_sha256": state_sha, "map_sha256": row["state"]["map_sha256"],
        "static_primitives": primitives, "geometry_sha256": decision_digest(primitives),
        "envelope": {"body_radius_m": .25, "body_height_m": .3,
                     "minimum_enu_m": [-10., -10., -10.], "maximum_enu_m": [10., 10., 10.]},
        "goal_world_enu_m": {"x": 2., "y": 0., "z": 1.5}, "commanded_change_squared": 0.,
        "premature_landing": False, "commands": commands, "control_applications": applications,
        "observations": observations, "behavior_application": {
            "purpose": "native-behavior-application", "action": "follow_route", "applied": True,
            "state_sha256": state_sha, "source": "executor", "overridden": False,
            "start_ms": 10000, "end_ms": 12000,
            "control_application_hashes": [decision_digest(a) for a in applications]}}


# 功能：实际重算独立几何，拒绝重签摘要后的撞墙/缺帧/错控制；输入：突变；输出：预期准入。
@pytest.mark.parametrize("fault", [None, "collision", "gap", "velocity"])
def test_recomputes_physical_outcome(fault):
    row, doc = native_fixture()
    if fault == "collision":
        doc["static_primitives"][0]["center_x"] = .2
        doc["geometry_sha256"] = decision_digest(doc["static_primitives"])
    elif fault == "gap":
        doc["observations"].pop(5)
    elif fault == "velocity":
        doc["control_applications"][2]["velocity_ned_mps"] = [0., 10., 0.]
        doc["behavior_application"]["control_application_hashes"] = [
            decision_digest(a) for a in doc["control_applications"]]
    if fault is None:
        assert verify_native_label(row, doc) == "follow_route"
    else:
        with pytest.raises(ValueError):
            verify_native_label(row, doc)


# 功能：普通向前飞不能被随意改名为减速示范；输入：缺少降速证据；输出：必须拒绝。
def test_slowdown_needs_actual_speed_bound():
    row, doc = native_fixture()
    doc["behavior_application"]["action"] = "slow_down"
    with pytest.raises(ValueError, match="SLOWDOWN_EXECUTION"):
        verify_native_label(row, doc)


# 功能：原始结果 JSON 同样不能用重复键隐藏错误；输入：重复键证据；输出：严格拒绝。
def test_outcome_duplicate_keys(tmp_path):
    from dronedream_agent_core.decision_label_evidence import read_bound

    path = tmp_path / "duplicate.json"
    path.write_text('{"collision":true,"collision":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="DUPLICATE_KEY"):
        read_bound(tmp_path, {"path": path.name, "sha256": file_digest(path)})


# 功能：补观测必须获得新内容，不能只刷新心跳或沿用旧摘要；输入：恢复变体；输出：真假区分。
@pytest.mark.parametrize("fault", [None, "empty", "replayed"])
def test_recovery_requires_new_observed_content(fault):
    row, doc = stationary_fixture()
    row["state"]["frame"]["geometry_source"]["observed_at_ms"] = 9000
    state_sha = decision_digest(row["state"])
    doc["state_sha256"] = state_sha
    doc["behavior_application"].update(action="request_observation", state_sha256=state_sha)
    after = copy.deepcopy(row["state"]["frame"])
    after["observed_at_ms"] = 12000
    for name in ("pose_source", "geometry_source", "route_source"):
        after[name]["observed_at_ms"] = 11900
        after[name]["evidence_sha256"] = "e"*64
    if fault == "empty":
        for direction in ("front", "back", "left", "right", "up", "down"):
            after[f"clearance_{direction}_m"] = None
    elif fault == "replayed":
        after["geometry_source"]["evidence_sha256"] = "a"*64
    doc["end_decision_frame"] = after
    # 即使末帧看似新鲜，没有原始传感器输入仍不得获得正式恢复标签。
    with pytest.raises(ValueError, match="OBSERVATION_RECOVERY_INPUT_REQUIRED"):
        verify_native_label(row, doc)


# 功能：构造因果绑定的实际悬停回执；PX4局部NED原点故意不同于地图ENU原点。
# 输入：无；输出：仅用于测试的静止见证和请求观测行为，绝不纳入正式训练数据。
def stationary_fixture():
    from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand
    from dronedream_agent_core.control_execution_evidence import control_application_record

    row, doc = native_fixture()
    state = row["state"]
    snapshot = {
        "schema_version": "dronedream.text-navigation-snapshot.v1",
        "source_of_truth": "metric-range-and-localization-evidence-not-rendered-image",
        "strategic_context": {"task": {"navigation_goal_id": state["goal_id"]}},
        "known_static_map": {"source_sha256": state["map_sha256"],
                             "qualified_route_sha256": state["route_sha256"]},
        "goal_position_m": doc["goal_world_enu_m"],
    }
    digest = decision_digest(snapshot)
    snapshot["snapshot_sha256"] = digest
    doc["navigation_snapshots"] = {digest: snapshot}
    for i, command in enumerate(doc["commands"]):
        command.update(model_navigation_authorized=False, requested_control_intent=None,
                       model_authority_reason="model-requested-hold",
                       model_navigation_snapshot_sha256=digest)
        command["decision"].update(action="hold", control_source="deterministic-brake",
                                   selected_velocity_mps={"x": 0., "y": 0., "z": 0.})
        command["command_position_m"] = {"x": 0., "y": 0., "z": 1.5}
        original = RuntimeLocalSafetyCommand.model_validate(command)
        stamp = command["generated_at_unix_ms"]
        doc["control_applications"][i] = control_application_record(
            original, sequence=i+1, accepted_at_unix_ms=stamp,
            transport="position-velocity-ned", velocity_ned_mps=(0., 0., 0.),
            position_ned_m=(7., 4., -1.3), yaw_heading_deg=0.,
            measured_hold_anchor={"source": "px4-measured-latched-position",
                                  "selected_at_unix_ms": 10000,
                                  "north_m": 7., "east_m": 4., "down_m": -1.3},
        ).model_dump(mode="json")
        doc["observations"][i].update(current_position_m={"x": 0., "y": 0., "z": 1.5},
                                      current_velocity_mps={"x": 0., "y": 0., "z": 0.})
    doc["behavior_application"].update(action="request_observation",
        control_application_hashes=[decision_digest(a) for a in doc["control_applications"]])
    return row, doc


# 功能：等待/补观测必须来自有效模型悬停，不能把超时刹车、前进或其他输入冒充该行为。
# 输入：篡改后的合成回执；输出：在物理验收之前明确拒绝。
@pytest.mark.parametrize("fault", ["moving", "expired", "missing-anchor", "missing-snapshot"])
def test_nonmotion_behavior_cannot_claim_unrelated_control(fault):
    from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand
    from dronedream_agent_core.control_execution_evidence import ControlApplicationRecord
    from dronedream_agent_core.decision_label_evidence import behavior_application_matches

    _, doc = stationary_fixture()
    command = doc["commands"][0]
    actual = doc["control_applications"][0]
    if fault == "moving":
        actual.pop("measured_hold_anchor")
        actual["velocity_ned_mps"][0] = .1
    elif fault == "expired":
        command["model_authority_reason"] = "model-lease-expired"
    elif fault == "missing-anchor":
        actual.pop("measured_hold_anchor")
    else:
        command["model_navigation_snapshot_sha256"] = None
    assert not behavior_application_matches(RuntimeLocalSafetyCommand.model_validate(command),
        ControlApplicationRecord.model_validate(actual), "request_observation")


# 功能：密集窗口必须逐帧检查，尤其不能漏掉分块接缝处碰撞/序列丢失。
# 输入：600帧合成见证及接缝故障；输出：正常通过，任何一处碰撞或缺口均拒绝。
@pytest.mark.parametrize("fault", [None, "boundary-collision", "sequence-gap"])
def test_dense_behavior_checks_every_witness(fault):
    row, doc = native_fixture()
    template = doc["observations"][0]
    observations = []
    for i in range(601):
        frame = copy.deepcopy(template)
        frame.update(sequence=i+1, observed_at_unix_ms=10000+i*3)
        frame["current_position_m"]["x"] = i*.00075
        observations.append(frame)
    # 600个间隔覆盖完整两秒，整数UNIX毫秒仍严格递增。
    for i, frame in enumerate(observations):
        frame["observed_at_unix_ms"] = 10000+i*2000//600
    if fault == "boundary-collision":
        observations[511]["current_position_m"]["x"] = 5.
    elif fault == "sequence-gap":
        observations[511]["sequence"] += 1
    doc["observations"] = observations
    if fault is None:
        assert verify_native_label(row, doc) == "follow_route"
    else:
        with pytest.raises(ValueError):
            verify_native_label(row, doc)
