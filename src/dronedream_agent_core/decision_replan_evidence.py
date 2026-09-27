"""A planned detour is not a successful replan until its bound continuation executes."""

import math

from .contracts import GraphRoute
from .decision_state_adapter import decision_digest


# 功能：独立校验替代路径的来源、静态扫掠净空和真正执行接续，不能只凭重规划请求入库。
# 输入：旧状态/结果、完整静态几何验证器、原目标和末帧真值；ENU米、UNIX毫秒。
# 输出：无；动态避让路径尚无时空路径证据时明确拒收，保持原有0.05米最低净空。
def verify_replan_result(row, document, *, verifier, goal, start_ms, end_ms, last_observation):
    event = document["replan_event"]
    points = document["replanned_path_world_enu_m"]
    if (
        row["state"]["frame"]["persistent_blockage"] is not True
        or event.get("source") != "local-planner"
        or event.get("old_route_sha256") != row["state"]["route_sha256"]
        or event.get("path_sha256") != decision_digest(points)
        or not start_ms <= event["generated_at_ms"] <= end_ms
        or not 2 <= len(points) <= 256
    ):
        raise ValueError("DECISION_REPLAN_EVENT_INVALID")
    vectors = [tuple(p[k] for k in "xyz") for p in points]
    last = tuple(getattr(last_observation.current_position_m, k) for k in "xyz")
    if (
        math.dist(vectors[0], last) > 0.25
        or math.dist(vectors[-1], goal) > 0.25
        or verifier.geometry.clearance(vectors) < 0.05
    ):
        raise ValueError("DECISION_REPLAN_PATH_NOT_VERIFIED")
    if last_observation.dynamic_obstacles:
        raise ValueError("DECISION_REPLAN_DYNAMIC_VERIFIER_REQUIRED")
    verify_replan_continuation(row, document, end_ms=end_ms, last_position=last)


# 功能：要求替代路线被同一任务实际采用并恢复推进，不把规划器返回路径当成执行成功。
# 输入：旧状态、旧等待结果、已生成路线及后续真实执行；ENU米、UNIX毫秒。
# 输出：无；校验路由版本、时间、接续位置和独立物理结果，失败拒收正式重规划标签。
def verify_replan_continuation(row, document, *, end_ms, last_position):
    from .decision_label_evidence import verify_native_label

    continuation = document.get("continuation")
    if not isinstance(continuation, dict):
        raise ValueError("DECISION_REPLAN_EXECUTED_CONTINUATION_REQUIRED")
    event = document["replan_event"]
    route = GraphRoute.model_validate(document.get("replacement_route"))
    digest = decision_digest(route.model_dump(mode="json"))
    resumed, outcome = continuation["row"], continuation["outcome"]
    state, next_state = row["state"], resumed["state"]
    if (
        digest != event.get("new_route_sha256")
        or digest != next_state["route_sha256"]
        or digest == state["route_sha256"]
        or [p.model_dump(mode="json") for p in route.positions_m]
        != document["replanned_path_world_enu_m"]
        or any(resumed[k] != row[k] for k in ("parent_group", "episode_id", "split"))
        or any(next_state[k] != state[k] for k in ("mission_id", "map_sha256", "goal_id"))
        or not end_ms <= next_state["frame"]["observed_at_ms"] <= end_ms + 30000
        or outcome["behavior_application"]["action"] not in {"follow_route", "slow_down"}
        or outcome.get("continuation") is not None
    ):
        raise ValueError("DECISION_REPLAN_CONTINUATION_BINDING_INVALID")
    # 移动命令必须显式保留采用新路线时的导航快照，不接受旧版仅比对目标点的弱绑定。
    if outcome.get("schema_version") == "dronedream.decision-native-outcome.v1":
        if not outcome.get("navigation_snapshots"):
            raise ValueError("DECISION_REPLAN_ROUTE_ADOPTION_NOT_WITNESSED")
    elif not outcome.get("behavior_selections"):
        raise ValueError("DECISION_REPLAN_ROUTE_ADOPTION_NOT_WITNESSED")
    observations = outcome["observations"]
    if (
        not observations
        or math.dist(last_position, tuple(observations[0]["current_position_m"][k] for k in "xyz"))
        > 0.25
    ):
        raise ValueError("DECISION_REPLAN_CONTINUATION_POSITION_JUMP")
    verify_native_label(resumed, outcome)
