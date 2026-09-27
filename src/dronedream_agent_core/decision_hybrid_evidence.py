"""Independent native displacement audit for bounded bridging; not a label grant."""

import bisect
import math

from .contracts import RuntimeLocalSafetyCommand, RuntimeLocalSafetyObservation
from .control_execution_evidence import ControlApplicationRecord, validate_application_binding
from .hashing import sha256_json


# 功能：用独立仿真位置重算每次衔接的累计路径长度，绝不以指令速度积分冒充实际位移。
# 输入：完整命令/执行回执/标签侧真值；UNIX 毫秒及世界 ENU 米。
# 输出：通过的短时衔接统计；不授予飞行权限，不自动给阶段行为添加正确标签。
def audit_hybrid_displacement(
    command_rows, application_rows, witness_rows, *, require_model_handback=False
):
    commands = {}
    for row in command_rows:
        if row.get("command") is not None:
            command = RuntimeLocalSafetyCommand.model_validate(row["command"])
            commands[sha256_json(command)] = command
    actuals = [ControlApplicationRecord.model_validate(row) for row in application_rows]
    witnesses = [RuntimeLocalSafetyObservation.model_validate(row) for row in witness_rows]
    times = [row.observed_at_unix_ms for row in witnesses]
    if (
        not witnesses
        or any(row.source != "simulation-ground-truth" for row in witnesses)
        or any(b <= a for a, b in zip(times, times[1:], strict=False))
    ):
        raise ValueError("HYBRID_AUDIT_INDEPENDENT_WITNESS_REQUIRED")
    if any(
        b.accepted_at_unix_ms <= a.accepted_at_unix_ms
        for a, b in zip(actuals, actuals[1:], strict=False)
    ):
        raise ValueError("HYBRID_AUDIT_APPLICATION_ORDER_INVALID")
    episodes = {}
    for actual in actuals:
        command = commands.get(actual.command_sha256)
        if command is None:
            raise ValueError("HYBRID_AUDIT_COMMAND_MISSING")
        validate_application_binding(command, actual)
        if command.navigation_control_authority != "bounded-hybrid" or actual.safety_action not in {
            "continue",
            "slow",
        }:
            continue
        lease, budget = command.hybrid_lease, command.observation_budget
        stamp = actual.accepted_at_unix_ms
        velocity = command.decision.selected_velocity_mps
        if (
            lease is None
            or budget is None
            or actual.model_authorized
            or actual.transport != "velocity-ned"
            or not command.generated_at_unix_ms <= stamp <= command.valid_until_unix_ms
            or any(
                abs(a - b) > 1e-6
                for a, b in zip(
                    actual.velocity_ned_mps, (velocity.y, velocity.x, -velocity.z), strict=True
                )
            )
            or not lease.started_at_unix_ms
            <= stamp
            <= min(lease.expires_at_unix_ms, budget.control_deadline_unix_ms)
            or lease.navigation_goal_id != command.navigation_goal_id
            or math.hypot(*actual.velocity_ned_mps) > lease.maximum_speed_mps + 1e-6
        ):
            raise ValueError("HYBRID_AUDIT_AUTHORITY_INVALID")
        identity = (lease.route_sha256, lease.navigation_goal_id, lease.episode)
        record = episodes.setdefault(identity, {"lease": lease, "last_ms": stamp, "count": 0})
        if record["lease"] != lease:
            raise ValueError("HYBRID_AUDIT_LEASE_RENEWED")
        record["last_ms"], record["count"] = stamp, record["count"] + 1
    results = []
    for (route, goal, episode), record in episodes.items():
        lease = record["lease"]
        end_ms = record["last_ms"]
        if require_model_handback:
            # 最后一次桥接发送不等于桥接运动结束；阶段标签必须覆盖到模型真正接回。
            handback = next(
                (a for a in actuals if a.accepted_at_unix_ms > end_ms and a.model_authorized), None
            )
            if handback is None or handback.accepted_at_unix_ms > lease.expires_at_unix_ms:
                raise ValueError("HYBRID_AUDIT_MODEL_HANDBACK_MISSING_OR_LATE")
            if commands[handback.command_sha256].navigation_goal_id != goal:
                raise ValueError("HYBRID_AUDIT_MODEL_HANDBACK_GOAL_CHANGED")
            end_ms = handback.accepted_at_unix_ms
        # 起点取真实开始前最近一帧，终点取末次发送之后一帧，保守包含两端运动。
        left = bisect.bisect_right(times, lease.started_at_unix_ms) - 1
        right = bisect.bisect_left(times, end_ms)
        if (
            left < 0
            or right >= len(times)
            or lease.started_at_unix_ms - times[left] > 100
            or times[right] - end_ms > 100
            or any(
                b - a > 100
                for a, b in zip(times[left:right], times[left + 1 : right + 1], strict=True)
            )
        ):
            raise ValueError("HYBRID_AUDIT_DISPLACEMENT_COVERAGE_GAP")
        positions = [
            tuple(getattr(row.current_position_m, key) for key in "xyz")
            for row in witnesses[left : right + 1]
        ]
        distance = sum(math.dist(a, b) for a, b in zip(positions, positions[1:], strict=False))
        if distance > lease.maximum_distance_m + 1e-6:
            raise ValueError("HYBRID_AUDIT_DISPLACEMENT_BUDGET_EXCEEDED")
        results.append(
            {
                "route_sha256": route,
                "goal_id": goal,
                "episode": episode,
                "motion_applications": record["count"],
                "sampled_path_length_m": distance,
                "start_ms": lease.started_at_unix_ms,
                "last_accepted_ms": record["last_ms"],
                "audited_until_ms": end_ms,
            }
        )
    return {
        "schema_version": "dronedream.hybrid-displacement-audit.v1",
        "episodes": results,
        "formal_training_additions": 0,
        "qualification_granted": False,
        "model_handback_required": require_model_handback,
        "limitation": "Sampled path length; sub-frame motion and stage behavior require audit.",
    }
