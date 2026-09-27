"""Recompute native outcome evidence before accepting a decision demonstration.

Privileged simulation witnesses remain label-side and never enter model_state.
An outcome alone cannot say which unexecuted alternative was best.
"""

import math

from .decision_dataset import decode_object, evidence_path, file_digest
from .decision_state_adapter import decision_digest


# 功能：区分真正选中并执行的非运动行为与输入失效后的保护性刹车。
# 输入：原始命令、实际回执及事先选择的行为；输出：是否具有该行为的执行身份。
# 此判断不证明安全或结果成功；后续仍检查原输入、时限、连续性与独立物理见证。
def behavior_application_matches(command, actual, action):
    if action in {"follow_route", "slow_down"}:
        return actual.model_authorized and actual.transport == "velocity-ned"
    return (
        action in {"wait", "request_observation", "replan"}
        and not actual.model_authorized
        and actual.safety_action == "hold"
        and command.model_authority_reason == "model-requested-hold"
        and command.model_call_id is not None
        and command.model_navigation_snapshot_sha256 is not None
        and not any(actual.velocity_ned_mps)
        and actual.measured_hold_anchor is not None
    )


# 功能：区分语义目的地与滚动控制目标，使用原始导航输入绑定真实任务身份。
# 输入：执行命令、因果状态、ENU目的地和可选原快照；输出：无，错图/错路线/错目标抛错。
# 没有快照的旧证据只接受完全相同的目的地，不能借兼容路径绕过提供但无效的新证据。
def verify_command_goal(command, state, goal, snapshot=None):
    if snapshot is None:
        target = command.evaluated_target_position_m
        if target is None or any(abs(getattr(target, k) - goal[k]) > 1e-6 for k in "xyz"):
            raise ValueError("DECISION_EXECUTED_TARGET_CHANGED")
        return
    digest = decision_digest({k: v for k, v in snapshot.items() if k != "snapshot_sha256"})
    if (
        snapshot.get("schema_version") != "dronedream.text-navigation-snapshot.v1"
        or snapshot.get("source_of_truth")
        != "metric-range-and-localization-evidence-not-rendered-image"
        or snapshot.get("snapshot_sha256") != digest
        or command.model_navigation_snapshot_sha256 != digest
    ):
        raise ValueError("DECISION_COMMAND_SOURCE_SNAPSHOT_INVALID")
    task = snapshot["strategic_context"]["task"]
    static = snapshot["known_static_map"]
    if (
        task["navigation_goal_id"] != state["goal_id"]
        or static["source_sha256"] != state["map_sha256"]
        or static["qualified_route_sha256"] != state["route_sha256"]
        or any(abs(snapshot["goal_position_m"][k] - goal[k]) > 1e-6 for k in "xyz")
    ):
        raise ValueError("DECISION_COMMAND_SOURCE_GOAL_CHANGED")


# 功能：将长行为的全部连续真值按重叠端点分块复核，不降采样、不漏掉块间碰撞。
# 输入：独立评分器、最多1501帧UNIX毫秒见证及结果参数；输出：末帧奖励和全程最小净空。
def evaluate_behavior_window(verifier, observations, **arguments):
    if not 2 <= len(observations) <= 1501:
        raise ValueError("DECISION_WITNESS_COUNT_INVALID")
    minimum = math.inf
    last_reward = None
    for first in range(0, len(observations) - 1, 511):
        block = observations[first : first + 512]
        start = arguments["start_ms"] if first == 0 else block[0].observed_at_unix_ms
        end = (
            arguments["end_ms"]
            if first + len(block) == len(observations)
            else block[-1].observed_at_unix_ms
        )
        reward, receipt = verifier.evaluate(
            observations=block, **{**arguments, "start_ms": start, "end_ms": end}
        )
        minimum = min(minimum, receipt["minimum_clearance_lower_bound_m"])
        if (
            reward.collision
            or reward.geofence_violation
            or reward.premature_landing
            or reward.loss_of_control
            or reward.safety_intervened
            or minimum < 0.05
        ):
            raise ValueError("DECISION_UNSAFE_OUTCOME_NOT_POSITIVE_LABEL")
        last_reward = reward
    return last_reward, minimum


# 功能：读取已绑定原文件，不相信证书里单独声明的成功；输入：根目录/引用；输出：JSON。
def read_bound(root, reference):
    if type(reference) is not dict or set(reference) != {"path", "sha256"}:
        raise ValueError("DECISION_LABEL_REFERENCE_INVALID")
    path = evidence_path(root, reference["path"])
    if path.stat().st_size > 16 * 1024**2 or file_digest(path) != reference["sha256"]:
        raise ValueError("DECISION_LABEL_SOURCE_INVALID")
    return decode_object(path.read_text("utf-8"))


# 功能：用独立几何/连续遥测重新算结果，并核对实际行为回执；不把建议当执行。
# 输入：单条领域样本和原始结果包；输出：经过复核的已执行行为，不评价未执行替代项。
def verify_native_label(row, document):
    if document.get("schema_version") == "dronedream.decision-stage-outcome.v1":
        from .decision_stage_label import verify_stage_label

        return verify_stage_label(row, document)
    from .contracts import RuntimeLocalSafetyCommand, RuntimeLocalSafetyObservation
    from .training.outcome_verifier import OutcomeEnvelope, SimulationOutcomeVerifier

    if document.get("schema_version") != "dronedream.decision-native-outcome.v1":
        raise ValueError("DECISION_NATIVE_OUTCOME_REQUIRED")
    if any(document.get(key) != row[key] for key in ("parent_group", "episode_id")):
        raise ValueError("DECISION_OUTCOME_EPISODE_MISMATCH")
    state_hash = decision_digest(row["state"])
    if document.get("state_sha256") != state_hash:
        raise ValueError("DECISION_OUTCOME_STATE_MISMATCH")
    application = document["behavior_application"]
    action = application["action"]
    if (
        application.get("purpose") != "native-behavior-application"
        or application.get("applied") is not True
        or application.get("state_sha256") != state_hash
        or application.get("source") != "executor"
        or not application.get("control_application_hashes")
        or application.get("overridden") is not False
    ):
        raise ValueError("DECISION_APPLIED_BEHAVIOR_REQUIRED")
    applications = document["control_applications"]
    if [decision_digest(a) for a in applications] != application["control_application_hashes"]:
        raise ValueError("DECISION_CONTROL_APPLICATION_BINDING_INVALID")
    # 单有自定义行为回执不够：同时通过现有执行器实际传输记录契约。
    from .control_execution_evidence import ControlApplicationRecord, validate_application_binding

    applications = [ControlApplicationRecord.model_validate(a) for a in applications]
    if not applications:
        raise ValueError("DECISION_EMPTY_CONTROL_APPLICATIONS")
    source_frame = row["state"]["frame"]
    start, end = application["start_ms"], application["end_ms"]
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start - source_frame["observed_at_ms"] <= 750
        or not 2000 <= end - start <= 30000
    ):
        raise ValueError("DECISION_OUTCOME_TIME_INVALID")
    commands = {
        decision_digest(c): RuntimeLocalSafetyCommand.model_validate(c)
        for c in document["commands"]
    }
    target = document["goal_world_enu_m"]
    snapshots = document.get("navigation_snapshots")
    for actual in applications:
        command = commands.get(actual.command_sha256)
        if command is None:
            raise ValueError("DECISION_EXECUTED_COMMAND_MISSING")
        validate_application_binding(command, actual)
        if not behavior_application_matches(command, actual, action):
            raise ValueError("DECISION_BEHAVIOR_CONTROL_MISMATCH")
        snapshot = None
        if action in {"wait", "request_observation", "replan"} and snapshots is None:
            raise ValueError("DECISION_COMMAND_SOURCE_SNAPSHOT_MISSING")
        if snapshots is not None:
            snapshot = snapshots.get(command.model_navigation_snapshot_sha256)
            if snapshot is None:
                raise ValueError("DECISION_COMMAND_SOURCE_SNAPSHOT_MISSING")
        verify_command_goal(command, row["state"], target, snapshot)
        stamp = actual.accepted_at_unix_ms
        if (
            not start <= stamp <= end
            or not command.generated_at_unix_ms <= stamp <= command.valid_until_unix_ms
            or command.navigation_goal_id != row["state"]["goal_id"]
        ):
            raise ValueError("DECISION_EXECUTED_COMMAND_TIME_OR_GOAL_INVALID")
        selected = command.decision.selected_velocity_mps
        if any(
            abs(a - b) > 1e-6
            for a, b in zip(
                actual.velocity_ned_mps, (selected.y, selected.x, -selected.z), strict=True
            )
        ):
            raise ValueError("DECISION_ACTUAL_VELOCITY_MISMATCH")
        if actual.transport == "position-velocity-ned":
            # PX4估计器原点不等于地图原点；悬停以执行器实际测得的NED锚点绑定。
            # 回执契约已逐轴核对实际设定值。不能只交换ENU轴就冒充坐标标定。
            if actual.measured_hold_anchor is None:
                raise ValueError("DECISION_MEASURED_HOLD_ANCHOR_REQUIRED")
    times = [a.accepted_at_unix_ms for a in applications]
    if (
        times[0] - start > 100
        or end - times[-1] > 100
        or any(not 0 < b - a <= 250 for a, b in zip(times, times[1:], strict=False))
    ):
        raise ValueError("DECISION_EXECUTION_GAP")
    observations = [
        RuntimeLocalSafetyObservation.model_validate(o) for o in document["observations"]
    ]
    envelope = OutcomeEnvelope(**document["envelope"])
    if envelope.body_radius_m != row["state"]["body_radius_m"]:
        raise ValueError("DECISION_OUTCOME_VEHICLE_MISMATCH")
    verifier = SimulationOutcomeVerifier(document["static_primitives"], envelope)
    if (
        document["map_sha256"] != row["state"]["map_sha256"]
        or document["geometry_sha256"] != verifier.geometry_sha256
    ):
        raise ValueError("DECISION_OUTCOME_MAP_MISMATCH")
    goal = tuple(document["goal_world_enu_m"][k] for k in "xyz")
    reward, _minimum = evaluate_behavior_window(
        verifier,
        observations,
        start_ms=start,
        end_ms=end,
        stage_id=row["state"]["goal_id"],
        goal_revision=decision_digest(list(goal)),
        goal_position_m=goal,
        safety_intervened=application["overridden"],
        commanded_change_squared=document["commanded_change_squared"],
        power_joules=None,
        premature_landing=document["premature_landing"],
    )
    first = observations[0].current_position_m
    progress = math.dist(tuple(getattr(first, k) for k in "xyz"), goal) - reward.goal_distance_m
    if action in {"follow_route", "slow_down"}:
        if progress < 0.05:
            raise ValueError("DECISION_PROGRESS_NOT_WITNESSED")
        if source_frame["local_route_verified"] is not True:
            raise ValueError("DECISION_MOTION_ROUTE_UNKNOWN")
        if action == "slow_down":
            requested = application.get("requested_speed_limit_mps")
            nominal = application.get("nominal_speed_limit_mps")
            if (
                type(requested) not in (int, float)
                or type(nominal) not in (int, float)
                or not math.isfinite(requested)
                or not math.isfinite(nominal)
                or not 0 < requested < nominal
                or nominal - requested < 0.01
                or source_frame["caution_required"] is not True
                or any(math.hypot(*a.velocity_ned_mps) > requested + 0.001 for a in applications)
            ):
                raise ValueError("DECISION_SLOWDOWN_EXECUTION_NOT_WITNESSED")
    elif action == "wait":
        if source_frame["crossing_obstacle"] is not True:
            raise ValueError("DECISION_WAIT_CAUSE_NOT_WITNESSED")
        if any(
            math.dist(
                tuple(getattr(o.current_position_m, k) for k in "xyz"),
                tuple(getattr(first, k) for k in "xyz"),
            )
            > 0.25
            for o in observations
        ):
            raise ValueError("DECISION_HOLD_NOT_WITNESSED")
        # 短时没碰撞不等于等待有效；必须另有障碍消失后继续推进的已验证结果。
        continuation = document.get("continuation")
        if not isinstance(continuation, dict):
            raise ValueError("DECISION_WAIT_RECOVERY_REQUIRED")
        resumed = continuation["row"]
        if (
            resumed["episode_id"] != row["episode_id"]
            or resumed["split"] != row["split"]
            or resumed["parent_group"] != row["parent_group"]
            or resumed["state"]["route_sha256"] != row["state"]["route_sha256"]
            or resumed["state"]["mission_id"] != row["state"]["mission_id"]
            or resumed["state"]["map_sha256"] != row["state"]["map_sha256"]
            or resumed["state"]["goal_id"] != row["state"]["goal_id"]
            or not end <= resumed["state"]["frame"]["observed_at_ms"] <= end + 30000
            or resumed["state"]["frame"]["crossing_obstacle"] is not False
            or continuation["outcome"]["behavior_application"]["action"]
            not in {"follow_route", "slow_down"}
        ):
            raise ValueError("DECISION_WAIT_RECOVERY_REQUIRED")
        verify_native_label(resumed, continuation["outcome"])
    elif action == "request_observation":
        from .decision_observation_recovery import verify_observation_recovery

        verify_observation_recovery(
            row,
            document,
            start_ms=start,
            end_ms=end,
            last_position=tuple(getattr(observations[-1].current_position_m, k) for k in "xyz"),
        )
    elif action == "replan":
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
    else:
        raise ValueError("DECISION_OUTCOME_ACTION_INVALID")
    return action


# 功能：要求标签有独立审核/结果来源；输入：样本、证书、根目录；输出：无，失败抛错。
def verify_label_evidence(row, certificate, root):
    documents = [read_bound(root, r) for r in certificate["outcome_references"]]
    if row["label_source"] == "verified-simulation-outcome":
        verified = {verify_native_label(row, d) for d in documents}
        if set(row["acceptable_actions"]) != verified:
            raise ValueError("DECISION_UNEXECUTED_LABEL_NOT_VERIFIED")
        from .decision_input_evidence import verify_input_evidence, verify_input_outcome_geometry

        if "input_reference" not in certificate:
            raise ValueError("DECISION_CANDIDATE_INPUT_EVIDENCE_REQUIRED")
        proof = read_bound(root, certificate["input_reference"])
        verify_input_evidence(row["state"], proof)
        for document in documents:
            verify_input_outcome_geometry(proof, document)
    else:
        reviews = [
            d
            for d in documents
            if d.get("schema_version") == "dronedream.decision-expert-review.v1"
        ]
        if len(reviews) != 1:
            raise ValueError("DECISION_EXPERT_REVIEW_REQUIRED")
        review = reviews[0]
        if (
            review.get("state_sha256") != decision_digest(row["state"])
            or review.get("decision") != "approved"
            or review.get("reviewer_kind") != "human-domain-reviewer"
            or not review.get("reviewer_id")
            or not review.get("rationale_summary")
            or set(review.get("acceptable_actions", [])) != set(row["acceptable_actions"])
            or review.get("episode_id") != row["episode_id"]
            or not review.get("source_references")
        ):
            raise ValueError("DECISION_EXPERT_REVIEW_INVALID")
        for reference in review["source_references"]:
            read_bound(root, reference)
