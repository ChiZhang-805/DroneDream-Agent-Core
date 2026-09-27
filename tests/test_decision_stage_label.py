"""Explicitly synthetic stage labels; never add these fixtures to a training corpus."""

import copy

import pytest
from decision_goal_fixture import local_target_fixture
from test_decision_stage_control import stage_fixture

from dronedream_agent_core.decision_label_evidence import verify_native_label
from dronedream_agent_core.decision_state_adapter import decision_digest


# 功能：给合成短制动窗口补齐阶段、原输入与提前决策证据；输出：非真实训练夹具。
def fixture():
    row, doc = stage_fixture()
    _, _, snapshot, _ = local_target_fixture()
    digest = snapshot["snapshot_sha256"]
    for command, actual in zip(doc["commands"], doc["control_applications"], strict=True):
        if actual["model_authorized"]:
            command["model_navigation_snapshot_sha256"] = digest
            command["requested_control_intent"]["navigation_snapshot_sha256"] = digest
            actual["intent"]["navigation_snapshot_sha256"] = digest
        actual["command_sha256"] = decision_digest(command)
    doc["schema_version"] = "dronedream.decision-stage-outcome.v1"
    doc["behavior_application"].update(
        purpose="native-stage-application",
        control_application_hashes=[decision_digest(a) for a in doc["control_applications"]],
    )
    doc["behavior_selections"] = [
        dict(
            state=row["state"],
            action="follow_route",
            selected_at_ms=10000,
            snapshot=snapshot,
            submission=dict(
                call_id=doc["commands"][0]["model_call_id"], source_snapshot_sha256=digest
            ),
        )
    ]
    doc["executor_phase_summary"] = dict(
        complete=True,
        phase_history_complete=True,
        phase_events=[
            dict(sequence=1, at_unix_ms=9999, phase="TRACK", executor_phase="TRACK"),
            dict(sequence=2, at_unix_ms=12001, phase="LANDING", executor_phase="LANDING"),
        ],
    )
    return row, doc


# 功能：允许明确混合实现而保持原始制动来源不变；输入：合成成功物理窗口；输出：阶段标签。
def test_stage_label_is_not_a_pure_model_label():
    row, doc = fixture()
    original = copy.deepcopy(doc)
    assert verify_native_label(row, doc) == "follow_route"
    assert doc == original and doc["control_applications"][4]["model_authorized"] is False


# 功能：阶段标签包含所有独立制动的内容摘要及零速度变化，不接受漏签或改写回执。
# 输入：合成窗口和制动附件；输出：仅完整绑定且独立物理验收通过者有效。
@pytest.mark.parametrize("fault", [None, "hash", "collision"])
def test_stage_label_accounts_for_transport_brakes(fault):
    from dronedream_agent_core.executor_brake_evidence import ExecutorBrakeApplication
    row, doc = fixture()
    brake = ExecutorBrakeApplication(sequence=1, after_command_application_sequence=3,
        accepted_at_unix_ms=10500, reason="command-stale",
        position_ned_m=(0., .1, -1.2), yaw_heading_deg=0.).model_dump(mode="json")
    doc["executor_brake_applications"] = [brake]
    doc["behavior_application"]["executor_brake_hashes"] = [decision_digest(brake)]
    if fault == "hash":
        brake["position_ned_m"][0] += .1
    elif fault == "collision":
        doc["static_primitives"][0]["center_x"] = .1
        doc["geometry_sha256"] = decision_digest(doc["static_primitives"])
    if fault:
        with pytest.raises(ValueError):
            verify_native_label(row, doc)
    else:
        assert verify_native_label(row, doc) == "follow_route"


# 功能：用非零起飞原点验证实测悬停不误当地图ENU；输入：合成NED锚点；输出：来源完整才可验收。
@pytest.mark.parametrize("fault", [None, "missing", "position", "future", "moving"])
def test_measured_ned_hold_requires_its_actual_anchor(fault):
    row, doc = fixture()
    actual = doc["control_applications"][4]
    actual.update(
        transport="position-velocity-ned",
        position_ned_m=[0.01, 0.3, -1.163],
        measured_hold_anchor=dict(
            source="px4-measured-latched-position", selected_at_unix_ms=10799,
            north_m=0.01, east_m=0.3, down_m=-1.163,
        ),
    )
    if fault == "missing":
        actual.pop("measured_hold_anchor")
    elif fault == "position":
        actual["position_ned_m"][2] += 0.437
    elif fault == "future":
        actual["measured_hold_anchor"]["selected_at_unix_ms"] = 10801
    elif fault == "moving":
        actual["velocity_ned_mps"][0] = 0.1
    doc["behavior_application"]["control_application_hashes"] = [
        decision_digest(a) for a in doc["control_applications"]
    ]
    if fault is None:
        assert verify_native_label(row, doc) == "follow_route"
    else:
        with pytest.raises(ValueError):
            verify_native_label(row, doc)


# 功能：来源重签不能掩盖错行为、事后选项、无进展、碰撞或提前结束；输出：拒绝正标签。
@pytest.mark.parametrize(
    "fault", ["future", "action", "state", "snapshot", "landing", "gap", "progress", "collision"]
)
def test_stage_does_not_relax_input_or_physical_outcome_checks(fault):
    row, doc = fixture()
    selection = doc["behavior_selections"][0]
    if fault == "future":
        selection["selected_at_ms"] = 13000
    elif fault == "action":
        selection["action"] = "wait"
    elif fault == "state":
        selection["state"] = copy.deepcopy(row["state"])
        selection["state"]["sequence"] += 1
    elif fault == "snapshot":
        selection["snapshot"]["goal_position_m"]["x"] = 50
    elif fault == "landing":
        doc["executor_phase_summary"]["phase_events"].insert(
            1, dict(sequence=2, at_unix_ms=11000, phase="LANDING", executor_phase="LANDING")
        )
        doc["executor_phase_summary"]["phase_events"][-1]["sequence"] = 3
    elif fault == "gap":
        doc["observations"].pop(3)
    elif fault == "progress":
        for observation in doc["observations"]:
            observation["current_position_m"]["x"] = 0
    else:
        doc["static_primitives"][0]["center_x"] = 0.2
        doc["geometry_sha256"] = decision_digest(doc["static_primitives"])
    with pytest.raises(ValueError):
        verify_native_label(row, doc)
