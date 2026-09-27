"""Synthetic stage wait/recovery verification; never formal training samples."""

import copy

import pytest
from test_decision_label_evidence import native_fixture, stationary_fixture
from test_decision_stage_label import fixture

from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand
from dronedream_agent_core.control_execution_evidence import control_application_record
from dronedream_agent_core.decision_label_evidence import verify_native_label
from dronedream_agent_core.decision_stage_control import verify_stage_control
from dronedream_agent_core.decision_state_adapter import decision_digest


# 功能：等待与恢复不能跨训练/测试集合拼接；输入：合成且换集合的恢复段；输出：拒绝。
def test_wait_continuation_preserves_split():
    row, document = wait_fixture()
    document["continuation"]["row"]["split"] = "test"
    with pytest.raises(ValueError):
        verify_native_label(row, document)


# 功能：生成有因果等待身份及后续恢复的合成夹具；输入：无；输出：不进入正式语料的测试数据。
def wait_fixture():
    row, doc = stationary_fixture()
    row["state"]["frame"]["crossing_obstacle"] = True
    digest = decision_digest(row["state"])
    doc.update(schema_version="dronedream.decision-stage-outcome.v1", state_sha256=digest)
    doc["behavior_application"].update(
        action="wait",
        state_sha256=digest,
        purpose="native-stage-application",
        controller_mode="model-with-bounded-hybrid",
    )
    doc["executor_phase_summary"] = fixture()[1]["executor_phase_summary"]
    snapshot = next(iter(doc["navigation_snapshots"].values()))
    doc["behavior_selections"] = [
        dict(
            state=row["state"],
            action="wait",
            selected_at_ms=10000,
            snapshot=snapshot,
            submission=dict(
                call_id=doc["commands"][0]["model_call_id"],
                source_snapshot_sha256=snapshot["snapshot_sha256"],
            ),
        )
    ]
    # 一条真实来源语义为独立刹车，不能将其model-requested-hold身份保留成等待决策。
    raw = doc["commands"][4]
    raw.update(
        model_authority_reason="model-lease-expired",
        model_call_id=None,
        model_path_sha256=None,
        model_navigation_snapshot_sha256=None,
    )
    doc["control_applications"][4] = control_application_record(
        RuntimeLocalSafetyCommand.model_validate(raw),
        sequence=5,
        accepted_at_unix_ms=10800,
        transport="velocity-ned",
        velocity_ned_mps=(0.0, 0.0, 0.0),
        yaw_heading_deg=0.0,
    ).model_dump(mode="json")
    doc["behavior_application"]["control_application_hashes"] = [
        decision_digest(a) for a in doc["control_applications"]
    ]
    resumed, outcome = native_fixture()
    # 测试恢复片段移到等待结束之后；重建命令回执，不能只改样本开始时间。
    resumed["state"]["frame"]["observed_at_ms"] = 13000
    resumed["state"]["frame"]["crossing_obstacle"] = False
    for name in ("pose_source", "geometry_source", "route_source"):
        if resumed["state"]["frame"][name]:
            resumed["state"]["frame"][name]["observed_at_ms"] = 13000
    for command, actual, observed in zip(
        outcome["commands"], outcome["control_applications"], outcome["observations"], strict=True
    ):
        for target in (command, command["requested_control_intent"]):
            target["generated_at_unix_ms"] += 3000
            target["valid_until_unix_ms"] += 3000
        actual.update(
            control_application_record(
                RuntimeLocalSafetyCommand.model_validate(command),
                sequence=actual["sequence"],
                accepted_at_unix_ms=actual["accepted_at_unix_ms"] + 3000,
                transport="velocity-ned",
                velocity_ned_mps=(0.0, 0.25, 0.0),
                yaw_heading_deg=0.0,
            ).model_dump(mode="json")
        )
        observed["observed_at_unix_ms"] += 3000
    outcome["state_sha256"] = decision_digest(resumed["state"])
    outcome["behavior_application"].update(
        start_ms=13000,
        end_ms=15000,
        state_sha256=outcome["state_sha256"],
        control_application_hashes=[decision_digest(a) for a in outcome["control_applications"]],
    )
    doc["continuation"] = dict(row=resumed, outcome=outcome)
    return row, doc


# 功能：确认等待与保护性刹车分别记账，并要求独立后续真实推进；输出：仅等待标签通过。
def test_stage_wait_keeps_control_sources_separate():
    row, doc = wait_fixture()
    original = copy.deepcopy(doc)
    assert verify_native_label(row, doc) == "wait"
    counts = verify_stage_control(row, doc)
    assert counts["model"] == 0 and counts["selected_hold"] == 10 and counts["measured_brake"] == 1
    assert original == doc


# 功能：覆盖借用其他回合、未恢复、位置漂移、地图替换等假阳性；输入：故障夹具；输出：拒绝。
@pytest.mark.parametrize(
    "fault",
    [
        "no-recovery",
        "other-episode",
        "other-route",
        "still-crossing",
        "late",
        "no-progress",
        "drift",
        "cause",
        "geometry",
    ],
)
def test_wait_recovery_is_not_inferred_from_zero_velocity(fault):
    row, doc = wait_fixture()
    if fault == "no-recovery":
        doc.pop("continuation")
    elif fault == "other-episode":
        doc["continuation"]["row"]["episode_id"] = "another"
    elif fault == "other-route":
        doc["continuation"]["row"]["state"]["route_sha256"] = "e" * 64
    elif fault == "still-crossing":
        doc["continuation"]["row"]["state"]["frame"]["crossing_obstacle"] = True
    elif fault == "late":
        doc["continuation"]["row"]["state"]["frame"]["observed_at_ms"] = 50000
    elif fault == "no-progress":
        for observed in doc["continuation"]["outcome"]["observations"]:
            observed["current_position_m"]["x"] = 0.0
    elif fault == "drift":
        doc["observations"][5]["current_position_m"]["x"] = 0.4
    elif fault == "cause":
        row["state"]["frame"]["crossing_obstacle"] = False
    else:
        doc["geometry_sha256"] = "e" * 64
    with pytest.raises(ValueError):
        verify_native_label(row, doc)
