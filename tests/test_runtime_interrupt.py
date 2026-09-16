from datetime import UTC, datetime

import pytest

from dronedream_agent_core.contracts import (
    RuntimeHoldAcknowledgement,
    RuntimeMessageClassification,
    RuntimeUserMessage,
    Vector3,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.runtime_interrupt import (
    RuntimeMessageRejected,
    _runtime_model_attempt_policy,
    authorize_runtime_action,
    close_runtime_control_session,
    create_runtime_control_session,
    submit_runtime_message,
)


# 功能：
#   验证分类与一次修复共用总预算，保留本地处理余量，不能每次尝试都花完整截止时间。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_runtime_model_attempts_fit_inside_end_to_end_deadline() -> None:
    attempts, per_attempt_seconds = _runtime_model_attempt_policy(180.0)

    assert attempts == 2
    assert per_attempt_seconds == pytest.approx(89.0)
    assert attempts * per_attempt_seconds < 180.0


# 功能：
#   构造身份完整的运行消息，不写入账户、远端接口或真实执行器。
# 输入：
#   text：测试使用的自然语言消息。
# 输出：
#   message：携带任务、计划、对话及执行身份的类型化消息。
def _message(text: str) -> RuntimeUserMessage:
    message = RuntimeUserMessage(
        message_id="runtime-msg-" + "a" * 32,
        conversation_id="conversation-a",
        mission_id="mission-" + "b" * 32,
        plan_revision_id="plan-" + "c" * 32,
        contract_id="mission-contract-a",
        execution_id="execution-" + "d" * 32,
        text=text,
        submitted_at=datetime.now(UTC),
    )
    return message


# 功能：
#   构造摘要绑定当前消息的稳定悬停回执，并允许单独破坏确定性门控。
# 输入：
#   message：回执应绑定的消息。
#   gate_updates：测试需要覆盖的悬停门控值。
# 输出：
#   acknowledgement：包含冻结位置、遥测及延迟的类型化回执。
def _ack(message: RuntimeUserMessage, **gate_updates: bool) -> RuntimeHoldAcknowledgement:
    gates = {
        "telemetry_finite": True,
        "position_error_within_0_50_m": True,
        "speed_within_0_35_mps": True,
        "old_plan_advancement_inhibited": True,
        "semantic_side_effects_inhibited": True,
    }
    gates.update(gate_updates)
    now = datetime.now(UTC)
    acknowledgement = RuntimeHoldAcknowledgement(
        message_sha256=sha256_json(message),
        message_id=message.message_id,
        execution_id=message.execution_id,
        interrupted_phase="TRACK",
        schedule_index=17,
        detected_at=now,
        detection_latency_ms=25,
        stable_at=now,
        stabilization_latency_ms=1100,
        frozen_command_ned_m=Vector3(x=1.0, y=2.0, z=-3.0),
        hold_command_ned_m=Vector3(x=1.1, y=2.1, z=-3.0),
        observed_position_ned_m=Vector3(x=1.1, y=2.1, z=-3.0),
        observed_velocity_ned_mps=Vector3(x=0.0, y=0.0, z=0.0),
        position_error_m=0.0,
        speed_mps=0.0,
        deterministic_gates=gates,
    )
    return acknowledgement


# 功能：
#   验证明确立即停止并降落的用户指令，不会被模型“只是信息”的分类覆盖。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_emergency_wording_forces_land_even_if_model_says_informational() -> None:
    message = _message("请立即停止并降落")
    action, gates, _ = authorize_runtime_action(
        message=message,
        acknowledgement=_ack(message),
        classification=RuntimeMessageClassification(
            message_kind="informational",
            requested_action="resume",
            requires_plan_revision=False,
            summary="No change.",
        ),
    )
    assert all(gates.values())
    assert action == "land"


# 功能：
#   验证修改目的地后返回降落的未来要求仍进入重规划，不能误作当前位置立即落地。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_future_landing_after_destination_change_does_not_force_emergency_land() -> None:
    message = _message(
        "不要继续去原来的外卖取餐点了，改道去教学楼入口完成取件，"
        "再返回办公室无人机起降坪并降落。"
    )
    action, gates, _ = authorize_runtime_action(
        message=message,
        acknowledgement=_ack(message),
        classification=RuntimeMessageClassification(
            message_kind="mission_amendment",
            requested_action="redirect",
            target_entity="教学楼入口",
            requires_plan_revision=True,
            summary="Redirect, then return and land at the end of the mission.",
        ),
    )

    assert all(gates.values())
    assert action == "hold_for_replan"


# 功能：
#   验证文本明确改目的地时，即使模型要求继续，也必须悬停等待新计划。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_destination_change_can_never_resume_the_old_plan() -> None:
    message = _message("不是原来的外卖点，改到保安亭")
    action, _, _ = authorize_runtime_action(
        message=message,
        acknowledgement=_ack(message),
        classification=RuntimeMessageClassification(
            message_kind="informational",
            requested_action="resume",
            requires_plan_revision=False,
            summary="No change.",
        ),
    )
    assert action == "hold_for_replan"


# 功能：
#   验证“先悬停再改道”仍保留后续改道意图，不被简单暂停分类变成永久等待。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_destination_change_with_safe_hover_wording_cannot_become_persistent_pause() -> None:
    message = _message(
        "不要继续去原来的外卖点了，请先安全悬停，再改道去校园门口的保安亭"
    )
    action, _, _ = authorize_runtime_action(
        message=message,
        acknowledgement=_ack(message),
        classification=RuntimeMessageClassification(
            message_kind="mission_amendment",
            requested_action="pause",
            target_entity="校园门口的保安亭",
            requires_plan_revision=False,
            summary="Change destination after reaching stable hold.",
        ),
    )

    assert action == "hold_for_replan"


# 功能：
#   验证悬停确定性门控失败时不能恢复飞行，核心选择受控降落。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_hold_gate_forces_landing() -> None:
    message = _message("只是告诉你天气不错")
    action, gates, _ = authorize_runtime_action(
        message=message,
        acknowledgement=_ack(message, telemetry_finite=False),
        classification=RuntimeMessageClassification(
            message_kind="informational",
            requested_action="resume",
            requires_plan_revision=False,
            summary="No task change.",
        ),
    )
    assert not gates["hold_deterministic_gates_passed"]
    assert action == "land"


# 功能：
#   对照每类已支持改令的初步安全状态，区分悬停、重新规划、外围控制、继续和降落。
# 输入：
#   requested_action：模型请求的动作类别。
#   message_kind：模型识别的消息类别。
#   expected：核心应给出的安全状态。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    ("requested_action", "message_kind", "expected"),
    [
        ("redirect", "mission_amendment", "hold_for_replan"),
        ("set_speed", "motion_adjustment", "hold_for_replan"),
        ("pause", "motion_adjustment", "hold"),
        ("resume", "informational", "resume_original"),
        ("return_home", "mission_amendment", "hold_for_replan"),
        ("safe_land", "mission_amendment", "land"),
        ("set_return_point", "mission_amendment", "hold_for_replan"),
        ("set_coverage", "mission_amendment", "hold_for_replan"),
        ("camera_control", "motion_adjustment", "apply_command"),
        ("payload_control", "mission_amendment", "apply_command"),
        ("set_avoidance", "motion_adjustment", "apply_command"),
        ("follow_target", "mission_amendment", "hold_for_replan"),
        ("operator_takeover", "mission_amendment", "hold"),
        ("operator_release", "informational", "resume_original"),
    ],
)
def test_every_runtime_amendment_first_enters_a_code_authorized_safe_state(
    requested_action: str,
    message_kind: str,
    expected: str,
) -> None:
    message = _message("用户运行时改令")
    action, gates, _ = authorize_runtime_action(
        message=message,
        acknowledgement=_ack(message),
        classification=RuntimeMessageClassification(
            message_kind=message_kind,  # type: ignore[arg-type]
            requested_action=requested_action,  # type: ignore[arg-type]
            target_entity="guard-house",
            requires_plan_revision=requested_action
            not in {"pause", "resume", "safe_land", "operator_release"},
            summary="A typed runtime amendment.",
            parameters={"maximum_speed_mps": 0.3} if requested_action == "set_speed" else {},
        ),
    )

    assert all(gates.values())
    assert action == expected


# 功能：
#   实际创建会话和消息文件，验证执行身份绑定，以及关闭会话后拒绝再投递。
# 输入：
#   tmp_path：隔离的运行目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_ingress_binds_message_to_exact_session_and_closes(tmp_path) -> None:
    control_dir = tmp_path / "run" / "runtime-control"
    create_runtime_control_session(
        control_dir=control_dir,
        conversation_id="conversation-a",
        mission_id="mission-" + "b" * 32,
        plan_revision_id="plan-" + "c" * 32,
        contract_id="mission-contract-a",
        execution_id="execution-" + "d" * 32,
        prepared_mission_sha256="e" * 64,
    )
    message = submit_runtime_message(control_dir=control_dir, text="请先悬停")
    assert message.execution_id == "execution-" + "d" * 32
    assert (control_dir / "inbox" / f"{message.message_id}.json").is_file()

    close_runtime_control_session(control_dir)
    with pytest.raises(RuntimeMessageRejected, match="RUNTIME_CONTROL_SESSION_CLOSED"):
        submit_runtime_message(control_dir=control_dir, text="第二条消息")


# 功能：
#   验证很小的截止预算也不被重试倍增，所有调用尝试的合计预算严格小于总值。
# 输入：
#   budget：总超时秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("budget", [.01, .5, 1., 3., 4., 180.])
def test_attempt_policy_never_expands_small_total_budget(budget):
    attempts, timeout = _runtime_model_attempt_policy(budget)
    assert 0 < attempts * timeout < budget


# 功能：
#   验证非有限、错误类型或无法转换的超时在构造模型端口前被拒绝。
# 输入：
#   budget：非法超时输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("budget", [True, None, float("nan"), float("inf"), 10**400])
def test_invalid_runtime_budget_fails_before_provider_construction(budget):
    with pytest.raises(ValueError, match="timeout"):
        _runtime_model_attempt_policy(budget)


# 功能：
#   验证绕过模型构造后损坏的门控也不能继续飞行，不接受空集合或可强制转换的真值。
# 输入：
#   gates：缺失或错误类型的确定性门控。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("gates", [{}, {"stable": 1}, {"stable": "true"}])
def test_unchecked_hold_gate_objects_cannot_resume(gates):
    message = _message("任务继续")
    acknowledgement = _ack(message).model_copy(update={"deterministic_gates": gates})
    action, checks, _ = authorize_runtime_action(
        message=message, acknowledgement=acknowledgement,
        classification=RuntimeMessageClassification(message_kind="informational",
            requested_action="resume", requires_plan_revision=False, summary="Continue"),
    )
    assert action == "land" and not checks["hold_deterministic_gates_passed"]


# 功能：
#   验证参数拒绝会阻止新动作，但不得降低已经要求的安全降落。
# 输入：
#   action：核心初步授权动作。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("action", ["land", "resume_original", "apply_command", "hold_for_replan"])
def test_bad_directive_cannot_weaken_a_safety_landing(action):
    from dronedream_agent_core.contracts import RuntimeAmendmentDirective
    from dronedream_agent_core.runtime_interrupt import _apply_directive_validation

    directive = RuntimeAmendmentDirective(action="camera_control", issue_codes=["BAD_PARAMETERS"])
    result, reason = _apply_directive_validation(action, "safety landing required", directive)
    assert result == ("land" if action == "land" else "hold")
    if action == "land":
        assert reason == "safety landing required"


# 功能：
#   验证协调器后到的中止原因不能覆盖操作员记录，并只回收自己创建的暂存。
# 输入：
#   tmp_path：已有操作员中止记录的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_abort_preserves_operator_evidence_and_cleans_staging(tmp_path):
    from dronedream_agent_core.runtime_interrupt import RuntimeInterruptionCoordinator, _atomic_json

    coordinator = object.__new__(RuntimeInterruptionCoordinator)
    coordinator.abort_file = tmp_path / "abort.json"
    _atomic_json(coordinator.abort_file, {"reason": "operator"})
    before = coordinator.abort_file.read_bytes()
    coordinator._request_abort("sidecar error")
    assert coordinator.abort_file.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))
