"""Core-owned validation for bounded runtime commands that do not replace a track."""

from __future__ import annotations

import copy
import re
from datetime import UTC, datetime

from .contracts import (
    PreparedMission,
    RuntimeAuthorizedCommand,
    RuntimeHoldAcknowledgement,
    RuntimeInterruptionDecision,
    RuntimeUserMessage,
)
from .hashing import sha256_json
from .runtime_payload_binding import resolve_runtime_payload


class RuntimeCommandError(RuntimeError):
    """A runtime command failed deterministic construction or adoption."""


_GAZEBO_TOPIC = re.compile(r"^/[A-Za-z0-9_./-]{1,240}$")


# 功能：
#   将当前稳定悬停后的核心决策转换成有界设备命令，绑定消息、执行实例和批准回执。
#   模型分类不是授权；需要计划修订的指令不能绕过计划流程直接发送给设备。
#   执行器仍须验证命令时效、身份及实际设备读回。
# 输入：
#   message：本次运行时自然语言消息。
#   acknowledgement：执行器确认的稳定悬停回执。
#   decision：模型分类及核心授权决策。
#   prepared：载荷控制必须使用的已冻结任务及设备绑定。
# 输出：
#   command：带参数、门控结果和来源摘要的设备命令。
def build_runtime_command(
    *,
    message: RuntimeUserMessage,
    acknowledgement: RuntimeHoldAcknowledgement,
    decision: RuntimeInterruptionDecision,
    prepared: PreparedMission | None = None,
) -> RuntimeAuthorizedCommand:
    # 只在本函数自己的快照上校验和计算摘要，防止产物继续共享插件回执的可变对象。
    message, acknowledgement, decision = copy.deepcopy((message, acknowledgement, decision))
    action = decision.classification.requested_action
    if decision.authorized_action != "apply_command" or action not in {
        "camera_control",
        "payload_control",
        "set_avoidance",
    }:
        raise RuntimeCommandError("RUNTIME_COMMAND_NOT_AUTHORIZED")
    parameters = copy.deepcopy(
        decision.amendment_directive.parameters
        if decision.amendment_directive is not None
        else decision.classification.parameters
    )
    parameter_gate = False
    payload_binding_gate = action != "payload_control"
    if action == "camera_control":
        parameter_gate = (
            isinstance(parameters.get("command"), str)
            and parameters.get("command")
            in {
                "take_photo",
                "start_video",
                "stop_video",
            }
            and type(parameters.get("component_id")) is int
            and 1 <= parameters["component_id"] <= 255
        )
    elif action == "payload_control":
        # Syntax-valid topics are not permission to address another vehicle.
        # Recheck the asset binding here even if the policy plugin is replaced.
        try:
            bound = resolve_runtime_payload(prepared, parameters)
            payload_binding_gate = (
                prepared is not None and prepared.contract.contract_id == message.contract_id
            )
            parameters = bound
        except ValueError:
            payload_binding_gate = False
        protocol = parameters.get("protocol")
        if protocol == "gazebo-transport":
            parameter_gate = (
                isinstance(parameters.get("operation"), str)
                and parameters.get("operation") in {"attach", "detach"}
                and isinstance(parameters.get("topic"), str)
                and _GAZEBO_TOPIC.fullmatch(str(parameters["topic"])) is not None
                and isinstance(parameters.get("output_topic"), str)
                and _GAZEBO_TOPIC.fullmatch(str(parameters["output_topic"])) is not None
            )
        elif protocol == "mavsdk-actuator":
            parameter_gate = (
                type(parameters.get("actuator_index")) is int
                and 1 <= parameters["actuator_index"] <= 16
                # bool is an int subclass, not a physical actuator amplitude.
                and type(parameters.get("actuator_value")) in (int, float)
                and -1.0 <= parameters["actuator_value"] <= 1.0
            )
    elif action == "set_avoidance":
        parameter_gate = isinstance(parameters.get("enabled"), bool)
    message_sha256 = sha256_json(message)
    gates = {
        "message_matches_decision": decision.message_sha256 == message_sha256,
        "hold_matches_decision": decision.hold_ack_sha256 == sha256_json(acknowledgement),
        "hold_matches_message": (
            acknowledgement.message_sha256 == message_sha256
            and acknowledgement.message_id == message.message_id
            and acknowledgement.execution_id == message.execution_id
        ),
        "stable_hold_gates_passed": bool(acknowledgement.deterministic_gates)
        and all(accepted is True for accepted in acknowledgement.deterministic_gates.values()),
        "decision_gates_passed": bool(decision.authorization_gates)
        and all(accepted is True for accepted in decision.authorization_gates.values()),
        "amendment_matches_classification": (
            decision.amendment_directive is None or decision.amendment_directive.action == action
        ),
        "semantic_side_effects_inhibited": acknowledgement.side_effects_inhibited is True,
        "command_parameters_valid": parameter_gate,
        "payload_transport_bound": payload_binding_gate,
        "plan_revision_not_required": (
            decision.classification.requires_plan_revision is False
            and (
                decision.amendment_directive is None
                or decision.amendment_directive.requires_plan_revision is False
            )
        ),
        "directive_verdict_passed": (
            decision.amendment_directive is None
            or (
                not decision.amendment_directive.issue_codes
                and decision.amendment_directive.requires_stable_hold is True
                and decision.amendment_directive.requires_core_authorization is True
            )
        ),
        "core_authorized_command": decision.authorized_action == "apply_command",
    }
    if not all(gates.values()):
        failed = ",".join(name for name, accepted in gates.items() if not accepted)
        raise RuntimeCommandError(f"RUNTIME_COMMAND_GATE_FAILED:{failed}")
    command = RuntimeAuthorizedCommand(
        message_id=message.message_id,
        execution_id=message.execution_id,
        action=action,
        parameters=parameters,
        message_sha256=message_sha256,
        hold_ack_sha256=sha256_json(acknowledgement),
        decision_sha256=sha256_json(decision),
        deterministic_gates=gates,
        plugin_hook_receipts=decision.plugin_hook_receipts,
        generated_at=datetime.now(UTC),
    )
    return command
