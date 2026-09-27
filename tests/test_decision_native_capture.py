"""Native recorder contract tests; these generated fixtures are not training data."""

import json

import pytest
from test_decision_label_evidence import native_fixture

from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand, RuntimeLocalSafetyObservation
from dronedream_agent_core.control_execution_evidence import ControlApplicationRecord
from dronedream_agent_core.decision_native_capture import NativeDecisionCapture
from dronedream_agent_core.decision_state_adapter import DecisionStateV2


# 功能：创建仅用于单元测试的记录器；输入：无；输出：记录器和合成物理夹具。
def capture():
    row, outcome = native_fixture()
    row["state"]["clock_domain"] = "unix-ms"
    state = DecisionStateV2.model_validate_json(json.dumps(row["state"]))
    recorder = NativeDecisionCapture(state=state, action="follow_route", selected_at_ms=10000,
        episode_id=row["episode_id"], parent_group=row["parent_group"],
        label_context={k: outcome[k] for k in ("map_sha256", "static_primitives", "geometry_sha256",
                                              "envelope", "goal_world_enu_m")})
    return recorder, outcome


# 功能：模拟实际回执/见证回调而非教师建议；输入：夹具；输出：采集后的记录器。
def feed(recorder, outcome):
    for command, application, observation in zip(outcome["commands"],
            outcome["control_applications"], outcome["observations"], strict=True):
        assert recorder.accepted(RuntimeLocalSafetyCommand.model_validate(command),
                                 ControlApplicationRecord.model_validate(application))
        assert recorder.witness(RuntimeLocalSafetyObservation.model_validate(observation))


# 功能：完整独立证据可复核且真值不进入模型输入；输入：夹具；输出：通过。
def test_capture_rechecks_complete_result():
    recorder, outcome = capture()
    before = recorder.state.model_dump_json()
    feed(recorder, outcome)
    document, report = recorder.finish(end_ms=12000, overridden=False, premature_landing=False)
    assert report["verified"] and not report["execution_authority"]
    assert document["observations"] == outcome["observations"]
    assert recorder.state.model_dump_json() == before
    with pytest.raises(ValueError, match="ALREADY_FINISHED"):
        recorder.finish(end_ms=12000, overridden=False, premature_landing=False)


# 功能：未真正应用、缺见证和安全接管均不可伪装成正例；输入：不同缺口；输出：隔离。
@pytest.mark.parametrize("fault", ["no-application", "no-witness", "overridden", "injection"])
def test_capture_missing_evidence_quarantined(fault):
    recorder, outcome = capture()
    if fault != "no-application":
        feed(recorder, outcome)
    if fault == "no-witness":
        recorder.observations.clear()
    _, report = recorder.finish(end_ms=12000, overridden=fault == "overridden",
        premature_landing=False, action_evidence={"commands": []} if fault == "injection" else None)
    assert not report["verified"]


# 功能：倒序/重复实际回执使窗口永久隔离但不抛出控制线程异常；输出：False。
def test_duplicate_receipt_does_not_throw_in_control_callback():
    recorder, outcome = capture()
    command = RuntimeLocalSafetyCommand.model_validate(outcome["commands"][0])
    actual = ControlApplicationRecord.model_validate(outcome["control_applications"][0])
    assert recorder.accepted(command, actual)
    assert not recorder.accepted(command, actual)
    assert recorder.failure == "DECISION_CAPTURE_APPLICATION_ORDER_OR_IDENTITY"


# 功能：允许独立采样稍早于实际行为起点，不要求两个进程在同一毫秒采样。
# 输入：早50毫秒的真值起点；输出：完整结果仍可复核，不修改原时间。
def test_witness_may_precede_selection_within_existing_tolerance():
    recorder, outcome = capture()
    outcome["observations"][0]["observed_at_unix_ms"] -= 50
    feed(recorder, outcome)
    _, report = recorder.finish(end_ms=12000, overridden=False, premature_landing=False)
    assert report["verified"]
