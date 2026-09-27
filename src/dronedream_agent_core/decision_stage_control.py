"""Audit implementation of a stage intent separately from pure model control labels.

A brief measured brake/leased bridge may implement the same high-level route
intent. It remains a different control source; no receipt or authorization flag
is rewritten. This check alone is not a positive decision label.
"""

import math

from .contracts import RuntimeLocalSafetyCommand
from .control_execution_evidence import ControlApplicationRecord, validate_application_binding
from .decision_hybrid_evidence import audit_hybrid_displacement
from .decision_label_evidence import behavior_application_matches
from .decision_state_adapter import decision_digest
from .executor_brake_evidence import ExecutorBrakeApplication


# 功能：按执行器原序号合并模型命令回执与独立制动；禁止仅按时间排序掩盖错序或删记录。
# 输入：窗口内连续命令及完整制动子序列，UNIX毫秒；输出：真实传输顺序及其原始来源类型。
def stage_transport_sequence(document):
    actuals = [ControlApplicationRecord.model_validate(a) for a in document["control_applications"]]
    brakes = [
        ExecutorBrakeApplication.model_validate(a)
        for a in document.get("executor_brake_applications", [])
    ]
    if not actuals:
        return []
    if any(b.sequence != a.sequence + 1 for a, b in zip(brakes, brakes[1:], strict=False)):
        raise ValueError("DECISION_STAGE_BRAKE_LEDGER_GAP")
    groups = {}
    prior_anchor = -1
    for brake in brakes:
        anchor = brake.after_command_application_sequence
        if not actuals[0].sequence <= anchor < actuals[-1].sequence or anchor < prior_anchor:
            raise ValueError("DECISION_STAGE_BRAKE_ORDER_INVALID")
        groups.setdefault(anchor, []).append(brake)
        prior_anchor = anchor
    merged = []
    for actual in actuals:
        merged.append(actual)
        merged.extend(groups.get(actual.sequence, []))
    if len(merged) != len(actuals) + len(brakes):
        raise ValueError("DECISION_STAGE_BRAKE_ORIGIN_MISSING")
    return merged


# 功能：核对阶段意图下的完整控制序列，区分模型动作、短时制动和显式有界衔接。
# 输入：同一状态、2～30秒实际回执及原命令/独立见证；UNIX毫秒、ENU/NED米制。
# 输出：控制来源统计；不授予标签，安全/进展和完整输入来源仍由调用方独立验证。
def verify_stage_control(row, document):
    application = document["behavior_application"]
    if application.get("controller_mode") != "model-with-bounded-hybrid":
        raise ValueError("DECISION_STAGE_CONTROLLER_MODE_REQUIRED")
    action = application["action"]
    stationary = action in {"wait", "request_observation", "replan"}
    if action not in {"follow_route", "slow_down"} and not stationary:
        raise ValueError("DECISION_STAGE_ACTION_NOT_SUPPORTED")
    start, end = application["start_ms"], application["end_ms"]
    if type(start) is not int or type(end) is not int or not 2000 <= end - start <= 30000:
        raise ValueError("DECISION_STAGE_DURATION_INVALID")
    commands = {
        decision_digest(c): RuntimeLocalSafetyCommand.model_validate(c)
        for c in document["commands"]
    }
    actuals = [ControlApplicationRecord.model_validate(a) for a in document["control_applications"]]
    if (
        not actuals
        or not behavior_application_matches(commands[actuals[0].command_sha256], actuals[0], action)
        or not behavior_application_matches(
            commands[actuals[-1].command_sha256], actuals[-1], action
        )
        or actuals[0].accepted_at_unix_ms != start
        or actuals[-1].accepted_at_unix_ms != end
    ):
        raise ValueError("DECISION_STAGE_MODEL_BOOKENDS_REQUIRED")
    # 回执序号来自同一执行器的连续传输记录，不能删除不利制动、换目标或不明控制。
    if any(b.sequence != a.sequence + 1 for a, b in zip(actuals, actuals[1:], strict=False)):
        raise ValueError("DECISION_STAGE_CONTROL_LEDGER_GAP")
    transports = stage_transport_sequence(document)
    if any(
        not 0 < b.accepted_at_unix_ms - a.accepted_at_unix_ms <= 250
        for a, b in zip(transports, transports[1:], strict=False)
    ):
        raise ValueError("DECISION_STAGE_CONTROL_TIME_GAP")
    counts = {"model": 0, "measured_brake": 0, "bounded_bridge": 0}
    if stationary:
        # 等待是有身份的模型行为，但不是模型运动授权；二者不能合并计数。
        counts["selected_hold"] = 0
    gap_start = None
    last_model = None
    last_model_accepted_ms = None
    bridge_origins = set()
    bridges = []
    for actual in transports:
        if isinstance(actual, ExecutorBrakeApplication):
            stamp = actual.accepted_at_unix_ms
            gap_start = stamp if gap_start is None else gap_start
            if stamp - gap_start > 1500:
                raise ValueError("DECISION_STAGE_HANDOFF_TOO_LONG")
            counts["measured_brake"] += 1
            continue
        command = commands.get(actual.command_sha256)
        if command is None:
            raise ValueError("DECISION_STAGE_COMMAND_MISSING")
        validate_application_binding(command, actual)
        velocity = command.decision.selected_velocity_mps
        if any(
            abs(a - b) > 1e-6
            for a, b in zip(
                actual.velocity_ned_mps, (velocity.y, velocity.x, -velocity.z), strict=True
            )
        ):
            raise ValueError("DECISION_STAGE_ACTUAL_VELOCITY_MISMATCH")
        stamp = actual.accepted_at_unix_ms
        if (
            command.navigation_goal_id != row["state"]["goal_id"]
            or not start <= stamp <= end
            or not command.generated_at_unix_ms <= stamp <= command.valid_until_unix_ms
        ):
            raise ValueError("DECISION_STAGE_COMMAND_IDENTITY_OR_TIME_INVALID")
        if behavior_application_matches(command, actual, action):
            if stationary:
                if gap_start is not None and stamp - gap_start > 1500:
                    raise ValueError("DECISION_STAGE_HANDBACK_TOO_LATE")
                gap_start = None
                counts["selected_hold"] += 1
                continue
            if actual.transport != "velocity-ned" or actual.safety_action not in {
                "continue",
                "slow",
            }:
                raise ValueError("DECISION_STAGE_MODEL_ACTION_REPLACED")
            if gap_start is not None and stamp - gap_start > 1500:
                raise ValueError("DECISION_STAGE_HANDBACK_TOO_LATE")
            gap_start = None
            last_model = command
            last_model_accepted_ms = stamp
            counts["model"] += 1
            continue
        if actual.model_authorized or (stationary and actual.safety_action != "hold"):
            raise ValueError("DECISION_STAGE_BEHAVIOR_REPLACED")
        gap_start = stamp if gap_start is None else gap_start
        if stamp - gap_start > 1500:
            raise ValueError("DECISION_STAGE_HANDOFF_TOO_LONG")
        if command.navigation_control_authority == "bounded-hybrid" and actual.safety_action in {
            "continue",
            "slow",
        }:
            lease = command.hybrid_lease
            if (
                lease is None
                or lease.route_sha256 != row["state"]["route_sha256"]
                or last_model is None
                or lease.navigation_goal_id != last_model.navigation_goal_id
            ):
                raise ValueError("DECISION_STAGE_BRIDGE_ORIGIN_INVALID")
            # 独立路线衔接不是过期模型轨迹的续租，合同可以没有model_call_id/path。
            # 每个衔接事件首次出现时，必须有窗口内近期真实模型执行作为前序；
            # 路线/目标、租期、速度、累计位移及真实交还仍由独立审计逐项检查。
            origin = (lease.route_sha256, lease.navigation_goal_id, lease.episode)
            if origin not in bridge_origins:
                if not 0 <= lease.started_at_unix_ms - last_model_accepted_ms <= 1500:
                    raise ValueError("DECISION_STAGE_BRIDGE_ORIGIN_INVALID")
                bridge_origins.add(origin)
            counts["bounded_bridge"] += 1
            bridges.append(actual)
        elif (
            actual.safety_action == "hold"
            and actual.control_source == "deterministic-brake"
            and actual.transport in {"velocity-ned", "position-velocity-ned"}
            and math.hypot(*actual.velocity_ned_mps) <= 1e-6
            and command.requested_control_intent is None
        ):
            # 是低层短时制动，不重命名为模型输出，也不据此宣称“等待行人”发生。
            counts["measured_brake"] += 1
        else:
            raise ValueError("DECISION_STAGE_UNBOUNDED_OR_UNKNOWN_OVERRIDE")
    if bridges:
        witnesses = document.get("hybrid_displacement_observations", document["observations"])
        by_sequence = {o["sequence"]: o for o in witnesses}
        if len(by_sequence) != len(witnesses) or any(
            by_sequence.get(o["sequence"]) != o for o in document["observations"]
        ):
            raise ValueError("DECISION_STAGE_INDEPENDENT_WITNESS_CONFLICT")
        # 独立真值按实际累计路径核验短租期，不用指令积分或首尾净位移替代。
        audit_hybrid_displacement(
            [{"command": c} for c in document["commands"]],
            document["control_applications"],
            witnesses,
            require_model_handback=True,
        )
    return {**counts, "formal_training_additions": 0, "motion_permission_granted": False}
