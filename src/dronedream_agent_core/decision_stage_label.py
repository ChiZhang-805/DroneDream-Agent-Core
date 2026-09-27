"""Explicit high-level stage outcomes, distinct from pure-model demonstrations."""

import math

from .contracts import RuntimeLocalSafetyCommand, RuntimeLocalSafetyObservation
from .control_execution_evidence import ControlApplicationRecord
from .decision_label_evidence import (
    behavior_application_matches,
    evaluate_behavior_window,
    verify_command_goal,
)
from .decision_phase_evidence import verify_window_phases
from .decision_stage_control import stage_transport_sequence, verify_stage_control
from .decision_state_adapter import decision_digest
from .training.outcome_verifier import OutcomeEnvelope, SimulationOutcomeVerifier


# 功能：阶段行为可由模型和有界衔接共同实现，但每种来源、真实后果和输入身份分别重验。
# 输入：原因果状态及显式stage结果包；输出：已执行行为；不证明未执行选项更优或整程成功。
def verify_stage_label(row, document):
    if document.get("schema_version") != "dronedream.decision-stage-outcome.v1":
        raise ValueError("DECISION_STAGE_OUTCOME_REQUIRED")
    state = row["state"]
    digest = decision_digest(state)
    if (
        any(document.get(k) != row[k] for k in ("parent_group", "episode_id"))
        or document.get("state_sha256") != digest
    ):
        raise ValueError("DECISION_STAGE_IDENTITY_MISMATCH")
    app = document["behavior_application"]
    if "executor_brake_applications" in document and app.get("executor_brake_hashes") != [
        decision_digest(a) for a in document["executor_brake_applications"]
    ]:
        raise ValueError("DECISION_STAGE_BRAKE_BINDING_INVALID")
    if (
        app.get("purpose") != "native-stage-application"
        or app.get("applied") is not True
        or app.get("source") != "executor"
        or app.get("state_sha256") != digest
        or app.get("control_application_hashes")
        != [decision_digest(a) for a in document["control_applications"]]
    ):
        raise ValueError("DECISION_STAGE_APPLICATION_BINDING_INVALID")
    start, end = app["start_ms"], app["end_ms"]
    control = verify_stage_control(row, document)
    if not 0 <= start - state["frame"]["observed_at_ms"] <= 750:
        raise ValueError("DECISION_STAGE_INPUT_TOO_OLD")
    phases = verify_window_phases(document["executor_phase_summary"], start, end)
    if phases["premature_landing"]:
        raise ValueError("DECISION_STAGE_ENDED_PREMATURELY")
    commands = {
        decision_digest(c): RuntimeLocalSafetyCommand.model_validate(c)
        for c in document["commands"]
    }
    selections = {
        r["submission"]["call_id"]: r
        for r in document["behavior_selections"]
        if r.get("submission")
    }
    first_model = None
    target = document["goal_world_enu_m"]
    for actual in document["control_applications"]:
        command = commands[actual["command_sha256"]]
        if behavior_application_matches(
            command, ControlApplicationRecord.model_validate(actual), app["action"]
        ):
            selection = selections.get(command.model_call_id)
            if (
                selection is None
                or selection["action"] != app["action"]
                or selection["selected_at_ms"] > actual["accepted_at_unix_ms"]
                or selection["submission"]["source_snapshot_sha256"]
                != command.model_navigation_snapshot_sha256
            ):
                raise ValueError("DECISION_STAGE_PROSPECTIVE_SELECTION_REQUIRED")
            verify_command_goal(command, state, target, selection["snapshot"])
            first_model = first_model or selection
        elif actual["transport"] == "position-velocity-ned":
            # 未授权模型期间，执行器锁定PX4实测NED位置，而不是把地图ENU命令直接转轴。
            # 实际锚点在ControlApplicationRecord入口核对；旧回执缺来源时不能补造该字段。
            if actual.get("measured_hold_anchor") is None:
                raise ValueError("DECISION_STAGE_MEASURED_HOLD_ANCHOR_REQUIRED")
    if first_model is None or decision_digest(first_model["state"]) != digest:
        raise ValueError("DECISION_STAGE_INITIAL_STATE_MISMATCH")
    observations = [
        RuntimeLocalSafetyObservation.model_validate(o) for o in document["observations"]
    ]
    envelope = OutcomeEnvelope(**document["envelope"])
    verifier = SimulationOutcomeVerifier(document["static_primitives"], envelope)
    if (
        envelope.body_radius_m != state["body_radius_m"]
        or document["map_sha256"] != state["map_sha256"]
        or document["geometry_sha256"] != verifier.geometry_sha256
    ):
        raise ValueError("DECISION_STAGE_PHYSICAL_BINDING_INVALID")
    velocities = [a.velocity_ned_mps for a in stage_transport_sequence(document)]
    variation = sum(math.dist(a, b) ** 2 for a, b in zip(velocities, velocities[1:], strict=False))
    goal = tuple(target[k] for k in "xyz")
    reward, _ = evaluate_behavior_window(
        verifier,
        observations,
        start_ms=start,
        end_ms=end,
        stage_id=state["goal_id"],
        goal_revision=decision_digest(list(goal)),
        goal_position_m=goal,
        # 上面逐条核验的是明确允许的阶段实现，不把其低层制动当作行为被其他任务替换。
        # 碰撞、围栏、提前降落及缺帧仍独立失败，原控制来源完全保留。
        safety_intervened=False,
        commanded_change_squared=variation,
        power_joules=None,
        premature_landing=phases["premature_landing"],
    )
    first = tuple(getattr(observations[0].current_position_m, k) for k in "xyz")
    if app["action"] in {"follow_route", "slow_down"} and (
        math.dist(first, goal) - reward.goal_distance_m < 0.05
        or state["frame"]["local_route_verified"] is not True
    ):
        raise ValueError("DECISION_STAGE_PROGRESS_NOT_WITNESSED")
    if app["action"] == "wait":
        verify_wait_result(row, document, observations, first, end)
    if app["action"] in {"request_observation", "replan"}:
        if any(
            math.dist(tuple(getattr(o.current_position_m, k) for k in "xyz"), first) > 0.25
            for o in observations
        ):
            raise ValueError("DECISION_HOLD_NOT_WITNESSED")
        if app["action"] == "request_observation":
            from .decision_observation_recovery import verify_observation_recovery

            verify_observation_recovery(
                row,
                document,
                start_ms=start,
                end_ms=end,
                last_position=tuple(getattr(observations[-1].current_position_m, k) for k in "xyz"),
            )
        else:
            from .decision_replan_evidence import verify_replan_result

            verify_replan_result(
                row,
                document,
                verifier=verifier,
                goal=goal,
                start_ms=start,
                end_ms=end,
                last_observation=observations[-1],
            )
    if app["action"] == "slow_down":
        requested, nominal = (
            app.get("requested_speed_limit_mps"),
            app.get("nominal_speed_limit_mps"),
        )
        if (
            type(requested) not in (int, float)
            or type(nominal) not in (int, float)
            or not math.isfinite(requested)
            or not math.isfinite(nominal)
            or not 0 < requested < nominal
            or nominal - requested < 0.01
            or state["frame"]["caution_required"] is not True
            or any(math.hypot(*v) > requested + 0.001 for v in velocities)
        ):
            raise ValueError("DECISION_STAGE_SLOWDOWN_NOT_WITNESSED")
    if sum(
        control.get(k, 0) for k in ("model", "selected_hold", "measured_brake", "bounded_bridge")
    ) != len(velocities):
        raise ValueError("DECISION_STAGE_CONTROL_ACCOUNTING_INVALID")
    return app["action"]


# 功能：等待必须确由横穿触发、保持位置并在同一回合恢复前进；不能把一直不动当成功。
# 输入：事先状态、完整阶段回执和独立ENU米制见证，结束UNIX毫秒；输出：无，失败拒绝。
def verify_wait_result(row, document, observations, first, end):
    from .decision_label_evidence import verify_native_label

    if row["state"]["frame"]["crossing_obstacle"] is not True:
        raise ValueError("DECISION_WAIT_CAUSE_NOT_WITNESSED")
    if any(
        math.dist(tuple(getattr(o.current_position_m, k) for k in "xyz"), first) > 0.25
        for o in observations
    ):
        raise ValueError("DECISION_HOLD_NOT_WITNESSED")
    continuation = document.get("continuation")
    if not isinstance(continuation, dict):
        raise ValueError("DECISION_WAIT_RECOVERY_REQUIRED")
    resumed = continuation["row"]
    if (
        any(resumed[k] != row[k] for k in ("episode_id", "parent_group", "split"))
        or any(
            resumed["state"][k] != row["state"][k]
            for k in ("route_sha256", "mission_id", "map_sha256", "goal_id")
        )
        or not end <= resumed["state"]["frame"]["observed_at_ms"] <= end + 30000
        or resumed["state"]["frame"]["crossing_obstacle"] is not False
        or continuation["outcome"]["behavior_application"]["action"]
        not in {"follow_route", "slow_down"}
    ):
        raise ValueError("DECISION_WAIT_RECOVERY_REQUIRED")
    verify_native_label(resumed, continuation["outcome"])
