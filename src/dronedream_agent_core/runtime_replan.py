"""Deterministic online rerouting from a telemetry-confirmed stable hold."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from .collision import build_tracking_corridor_budget
from .contracts import (
    CoveragePattern,
    CoveragePlanRequest,
    GraphRoute,
    MapAsset,
    MapCatalog,
    Px4Track,
    RouteClearanceReport,
    RouteQuery,
    RuntimeActionExecutionContract,
    RuntimeActionExecutionStep,
    RuntimeCheckpoint,
    RuntimeCheckpointContract,
    RuntimeHoldAcknowledgement,
    RuntimeInterruptionDecision,
    RuntimeReplacementTrack,
    RuntimeToolReceipt,
    RuntimeTrackRequest,
    RuntimeUserMessage,
    TaskGraph,
    Vector3,
    VehicleAsset,
)
from .extensions import ExtensionExecutionError
from .hashing import sha256_json
from .control_uncertainty import finite_positive_number
from .localization_evidence import native_localization_evidence
from .plugin_api import (
    ToolEnvironment,
    build_discovered_extension_registry,
    build_discovered_tool_registry,
    build_snapshot_tool_registry,
)
from .plugin_contracts import PluginSnapshot
from .runtime_plugins import all_required_gates_passed
from .runtime_progress import bound_resume_point_index
from .tools import ToolExecutionError, ToolReceipt, ToolRegistry


class RuntimeReplanError(RuntimeError):
    """No safe, fully bound replacement route could be produced."""


# 功能：
#   为目的地寻找可触发检查点的实际重合航点，不能把任意远处的最近点当成已到目的地。
# 输入：
#   track：替换后轨迹，首点是当前悬停点而非新到达事件。
#   target：动作目的地世界 ENU 坐标。
# 输出：
#   checkpoint_index：与目的地重合且至少为 1 的航点下标。
def _target_checkpoint_index(track: Px4Track, target: Vector3) -> int:
    for index, point in enumerate(track.source_world_points[1:], start=1):
        if math.dist(
            (point.east_m, point.north_m, point.up_m), (target.x, target.y, target.z)
        ) <= 1e-6:
            checkpoint_index = index
            return checkpoint_index
    raise RuntimeReplanError("RUNTIME_REPLAN_ACTION_TARGET_NOT_ON_TRACK")


# 功能：
#   1. 保留未完成动作，重绑改令涉及的任务目标、检查点与外设请求。
#   2. 分配新动作标识并记录被取代标识；不将生成制品视为执行完成。
# 输入：
#   task_graph：改令前活动任务图，独立路线构建时可为空。
#   prior_actions：活动动作合同，须与任务图同时存在或同时为空。
#   completed_step_ids：已有成功证据的动作标识。
#   replacement_sequence：本次替换序号。
#   graph：当前地图节点位置。
#   track：替换后轨迹。
#   target_node：本次任务目标。
#   return_node：本次返回目标。
#   prior_target_node：原任务目标。
#   prior_return_node：原返回目标。
#   amendment_action：用户获准修改的动作类别。
# 输出：
#   artifacts：任务图、检查点合同、动作合同及被替代标识列表的四元组。
def _revised_execution_artifacts(
    *,
    task_graph: TaskGraph | None,
    prior_actions: RuntimeActionExecutionContract | None,
    completed_step_ids: set[str],
    replacement_sequence: int,
    graph: MapAsset,
    track: Px4Track,
    target_node: str,
    return_node: str,
    prior_target_node: str,
    prior_return_node: str,
    amendment_action: str,
) -> tuple[
    TaskGraph | None,
    RuntimeCheckpointContract | None,
    RuntimeActionExecutionContract | None,
    list[str],
]:
    if prior_actions is None and task_graph is None:
        # 独立几何构建仍可无执行合同；实际采纳入口要求完整执行修订，不能靠此授权飞行。
        artifacts = None, None, None, []
        return artifacts
    if prior_actions is None or task_graph is None:
        raise RuntimeReplanError("RUNTIME_REPLAN_EXECUTION_SOURCE_INCOMPLETE")
    original_tasks = {task.task_id: task for task in task_graph.nodes}
    source_bound = (
        prior_actions.task_graph_sha256 == sha256_json(task_graph)
        and len(original_tasks) == len(task_graph.nodes)
        and len({step.step_id for step in prior_actions.steps}) == len(prior_actions.steps)
        and all(
            step.task_id in original_tasks
            and step.target_node == original_tasks[step.task_id].target_node
            and step.action == original_tasks[step.task_id].action
            for step in prior_actions.steps
        )
    )
    if not source_bound:
        raise RuntimeReplanError("RUNTIME_REPLAN_EXECUTION_SOURCE_MISMATCH")
    remaining = [
        step for step in prior_actions.steps if step.step_id not in completed_step_ids
    ]

    node_targets: dict[str, str] = {}
    if amendment_action in {"replan", "redirect", "return_home", "follow_target"}:
        node_targets[prior_target_node] = target_node
    if amendment_action == "set_return_point":
        node_targets[prior_return_node] = return_node
    revised_graph = TaskGraph.model_validate(
        {
            **task_graph.model_dump(mode="json"),
            "nodes": [
                {
                    **task.model_dump(mode="json"),
                    "target_node": node_targets.get(task.target_node, task.target_node),
                }
                for task in task_graph.nodes
            ],
            "revision": task_graph.revision + 1,
        }
    )
    tasks = {task.task_id: task for task in revised_graph.nodes}
    graph_nodes = {node.node_id: node.position_m for node in graph.nodes}
    checkpoint_by_target: dict[str, RuntimeCheckpoint] = {}
    for step in remaining:
        if step.trigger != "checkpoint":
            continue
        revised_target = node_targets.get(step.target_node, step.target_node)
        if revised_target in checkpoint_by_target:
            continue
        position = graph_nodes.get(revised_target)
        if position is None:
            raise RuntimeReplanError(
                f"RUNTIME_REPLAN_ACTION_TARGET_UNKNOWN:{revised_target}"
            )
        checkpoint_index = 500 + replacement_sequence * 128 + len(checkpoint_by_target) + 1
        checkpoint_by_target[revised_target] = RuntimeCheckpoint(
            checkpoint_id=f"checkpoint-{checkpoint_index:03d}",
            segment_id=f"segment-{checkpoint_index:03d}",
            task_id=step.task_id,
            track_point_index=_target_checkpoint_index(track, position),
            target_node=revised_target,
        )
    if not checkpoint_by_target:
        checkpoint_index = 500 + replacement_sequence * 128 + 1
        task = next(
            (
                item
                for item in revised_graph.nodes
                if item.target_node == target_node
            ),
            revised_graph.nodes[-1],
        )
        position = graph_nodes.get(target_node)
        if position is None:
            final_world = track.source_world_points[-1]
            position = Vector3(
                x=final_world.east_m,
                y=final_world.north_m,
                z=final_world.up_m,
            )
        checkpoint_by_target[target_node] = RuntimeCheckpoint(
            checkpoint_id=f"checkpoint-{checkpoint_index:03d}",
            segment_id=f"segment-{checkpoint_index:03d}",
            task_id=task.task_id,
            track_point_index=_target_checkpoint_index(track, position),
            target_node=target_node,
        )
    checkpoints = RuntimeCheckpointContract(
        contract_id=prior_actions.contract_id,
        checkpoints=list(checkpoint_by_target.values()),
    )
    revised_steps: list[RuntimeActionExecutionStep] = []
    first_action_index = 500 + replacement_sequence * 128 + 1
    for index, step in enumerate(remaining, start=first_action_index):
        revised_target = node_targets.get(step.target_node, step.target_node)
        task = tasks.get(step.task_id)
        if task is None:
            raise RuntimeReplanError(f"RUNTIME_REPLAN_ACTION_TASK_UNKNOWN:{step.task_id}")
        parameters = dict(step.parameters)
        if step.driver == "ros2-service" and isinstance(parameters.get("request"), dict):
            parameters["request"] = {
                **parameters["request"],
                "target_node": revised_target,
                "arguments_json": json.dumps(
                    task.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        revised_steps.append(
            step.model_copy(
                update={
                    "step_id": f"action-{index:03d}",
                    "target_node": revised_target,
                    "trigger": (
                        "checkpoint" if step.trigger == "checkpoint" else "post-replan"
                    ),
                    "checkpoint_id": (
                        None
                        if step.trigger != "checkpoint"
                        else checkpoint_by_target[revised_target].checkpoint_id
                    ),
                    "parameters": parameters,
                },
                deep=True,
            )
        )
    revised_actions = RuntimeActionExecutionContract(
        contract_id=prior_actions.contract_id,
        task_graph_sha256=sha256_json(revised_graph),
        domain_action_catalog_sha256=prior_actions.domain_action_catalog_sha256,
        adapter_catalog_sha256=prior_actions.adapter_catalog_sha256,
        steps=revised_steps,
    )
    artifacts = (
        revised_graph,
        checkpoints,
        revised_actions,
        [step.step_id for step in remaining],
    )
    return artifacts


# 功能：
#   把重建的任务、检查点与动作作为一套完整执行修订绑定到替换轨迹，并重新校验制品。
# 输入：
#   replacement：已通过几何门控的替换候选。
#   task_graph：当前活动任务图。
#   prior_actions：当前活动动作合同。
#   completed_step_ids：已完成动作标识，无记录时为空。
#   graph：当前地图。
#   prior_target_node：改令前目标。
#   prior_return_node：改令前返回点。
# 输出：
#   bound_replacement：带完整修订绑定的替换制品。
def _bind_execution_revision(
    replacement: RuntimeReplacementTrack,
    *,
    task_graph: TaskGraph | None,
    prior_actions: RuntimeActionExecutionContract | None,
    completed_step_ids: set[str] | None,
    graph: MapAsset,
    prior_target_node: str,
    prior_return_node: str,
) -> RuntimeReplacementTrack:
    revised_graph, checkpoints, actions, superseded = _revised_execution_artifacts(
        task_graph=task_graph,
        prior_actions=prior_actions,
        completed_step_ids=set(completed_step_ids or set()),
        replacement_sequence=replacement.replacement_sequence,
        graph=graph,
        track=replacement.track,
        target_node=replacement.target_node,
        return_node=replacement.return_node,
        prior_target_node=prior_target_node,
        prior_return_node=prior_return_node,
        amendment_action=replacement.amendment_action,
    )
    gates = dict(replacement.deterministic_gates)
    if revised_graph is not None:
        gates["runtime_execution_revision_complete"] = all(
            item is not None for item in (revised_graph, checkpoints, actions)
        )
        gates["runtime_action_supersession_bound"] = len(superseded) == len(set(superseded))
    bound_replacement = RuntimeReplacementTrack.model_validate(
        {
            **replacement.model_dump(mode="json"),
            "revised_task_graph": (
                revised_graph.model_dump(mode="json") if revised_graph is not None else None
            ),
            "runtime_checkpoints": (
                checkpoints.model_dump(mode="json") if checkpoints is not None else None
            ),
            "runtime_actions": actions.model_dump(mode="json") if actions is not None else None,
            "superseded_runtime_action_step_ids": superseded,
            "deterministic_gates": gates,
        }
    )
    return bound_replacement


# 功能：
#   将明确节点、图别名或唯一目录别名解析为现存节点；多个候选或失效引用均拒绝猜测。
# 输入：
#   entity：用户目标短语。
#   catalog：当前语义目录。
#   graph：当前拓扑图。
# 输出：
#   target_node：唯一解析得到的图节点标识。
def _resolve_target(entity: str, catalog: MapCatalog, graph: MapAsset) -> str:
    node_ids = {node.node_id for node in graph.nodes}
    if entity in node_ids:
        target_node = entity
        return target_node
    direct = graph.named_entities.get(entity)
    if direct is not None:
        if direct not in node_ids:
            raise RuntimeReplanError(f"RUNTIME_TARGET_UNRESOLVED:{entity}")
        target_node = direct
        return target_node
    normalized = entity.casefold().strip()
    if not normalized:
        raise RuntimeReplanError("RUNTIME_TARGET_UNRESOLVED:empty")
    exact_matches: set[str] = set()
    partial_matches: set[str] = set()
    for item in catalog.entities:
        candidates = {
            value.casefold().strip() for value in (item.entity_id, *item.aliases) if value.strip()
        }
        resolved = graph.named_entities.get(item.entity_id)
        if resolved is None and item.entity_id in node_ids:
            resolved = item.entity_id
        if resolved not in node_ids:
            continue
        if normalized in candidates:
            exact_matches.add(resolved)
        # 先遍历完所有同名别名，再决定唯一性，不能让目录排列顺序决定飞往哪里。
        if any(len(candidate) >= 2 and candidate in normalized for candidate in candidates):
            partial_matches.add(resolved)
    matches = exact_matches or partial_matches
    if len(matches) == 1:
        target_node = matches.pop()
        return target_node
    if matches:
        raise RuntimeReplanError(f"RUNTIME_TARGET_AMBIGUOUS:{entity}")
    raise RuntimeReplanError(f"RUNTIME_TARGET_UNRESOLVED:{entity}")


# 功能：
#   按当前坐标合同，把 PX4 局部 NED 观测换成碰撞包络中心的世界 ENU 坐标。
# 输入：
#   ack：稳定悬停时的局部位置观测。
#   prior_track：提供出生原点及碰撞中心偏移的当前轨迹。
# 输出：
#   current_world：世界 ENU 位置，单位米。
def _current_world_position(ack: RuntimeHoldAcknowledgement, prior_track: Px4Track) -> Vector3:
    root_east, root_north, root_up = prior_track.coordinate_contract.model_root_world_enu_m
    offset_east, offset_north, offset_up = (
        prior_track.coordinate_contract.resolved_collision_center_offset_model_m()
    )
    observed = ack.observed_position_ned_m
    current_world = Vector3(
        x=root_east + offset_east + observed.y,
        y=root_north + offset_north + observed.x,
        z=root_up + offset_up - observed.z,
    )
    return current_world


# 功能：
#   优先调用度量规划；失败时记录回执并取拓扑候选，候选仍须通过连续净空和跟踪预算。
# 输入：
#   registry：绑定工具的注册器。
#   query：路线两端与规划约束。
#   prefer_metric：是否先尝试度量规划。
# 输出：
#   result：路线候选、调用回执列表及是否使用度量规划的三元组。
def _call_runtime_route(
    registry: ToolRegistry, query: RouteQuery, *, prefer_metric: bool = True,
) -> tuple[GraphRoute, list[ToolReceipt], bool]:

    metric_tool_id = "planning.candidate-metric-geometry.candidate"
    available = {str(item["tool_id"]) for item in registry.catalog()}
    receipts: list[ToolReceipt] = []
    if prefer_metric and metric_tool_id in available:
        try:
            value, receipt = registry.call(metric_tool_id, query)
        except ToolExecutionError as error:
            receipts.append(error.receipt)
        else:
            result = GraphRoute.model_validate(value), [receipt], True
            return result
    value, receipt = registry.call_slot("planning.route-strategy", query)
    receipts.append(receipt)
    result = GraphRoute.model_validate(value), receipts, False
    return result


# 功能：
#   核对净空报告绑定且内部一致，并要求每个替换路段留足定位、安全及跟踪预算。
# 输入：
#   route：待执行路线。
#   clearance：该路线的连续净空报告。
# 输出：
#   accepted：全部路段预算和报告一致性通过时为 True。
def _route_tracking_budget_accepted(
    route: GraphRoute, clearance: RouteClearanceReport, localization_variance_m2: float,
) -> bool:
    segment_clearances = list(clearance.segment_minimum_clearances_m)
    accepted = False
    if (
        not finite_positive_number(localization_variance_m2)
        or clearance.accepted is not True
        or type(clearance.collision_count) is not int
        or clearance.collision_count != 0
        or clearance.collisions
        or clearance.route_sha256 != sha256_json(route)
        or not math.isfinite(clearance.minimum_clearance_m)
        or clearance.minimum_clearance_m < 0
        or len(segment_clearances) != max(0, len(route.positions_m) - 1)
    ):
        return accepted
    try:
        for measured_m in segment_clearances:
            build_tracking_corridor_budget(
                measured_m, localization_covariance_m2=localization_variance_m2)
    except ValueError:
        return accepted
    accepted = bool(segment_clearances)
    return accepted


# 功能：读取与本次稳定悬停绑定的真实定位方差，拒绝缺失、过期或未来测量，不退回规划默认值。
# 输入：ack：稳定悬停回执；测量在 stable_at 时须不超过 250 毫秒，路线采纳仍须再检当前状态。
# 输出：variance_m2：可供候选搜索及连续路线验收使用的方差上界。
def _hold_localization_variance(ack: RuntimeHoldAcknowledgement) -> float:
    variance = ack.localization_variance_m2
    observed = ack.localization_observed_at_unix_ms
    if (type(variance) not in (int, float) or not math.isfinite(variance)
            or not 0 < variance <= 10_000 or type(observed) is not int
            or ack.stable_at.tzinfo is None):
        raise RuntimeReplanError("RUNTIME_REPLAN_LOCALIZATION_EVIDENCE_REQUIRED")
    age_ms = round(ack.stable_at.timestamp() * 1000) - observed
    if not 0 <= age_ms <= 250:
        raise RuntimeReplanError("RUNTIME_REPLAN_LOCALIZATION_EVIDENCE_EXPIRED")
    return float(variance)


# 功能：在执行器实际接收替换轨迹时重新核对定位，不能把几秒前悬停时的误差当作当前证据。
# 输入：replacement：绑定路线与连续净空；dynamics：原生遥测快照；now_unix_ms：当前消费时刻。
# 输出：accepted：来源新鲜且整条替换路线能覆盖当前误差时为 True；未知测量保守返回 False。
def replacement_localization_budget_accepted(
    replacement: RuntimeReplacementTrack, dynamics: dict, *, now_unix_ms: int,
) -> bool:
    try:
        variance, _ = native_localization_evidence(
            {"dynamics": dynamics}, now_unix_ms=now_unix_ms, maximum_age_ms=250)
    except ValueError:
        return False
    return _route_tracking_budget_accepted(replacement.route, replacement.clearance, variance)


# 功能：
#   在调用规划工具前确认消息、悬停、决定和活动轨迹属于同一改令，拒绝空门控或旧摘要。
# 输入：
#   message：当前用户消息。
#   acknowledgement：本消息的稳定悬停回执。
#   decision：获准重规划的决定。
#   prior_track：活动轨迹。
#   prior_track_sha256：活动轨迹的预期摘要。
# 输出：
#   None：不返回业务数据。
def _validate_replan_input_binding(
    message: RuntimeUserMessage, acknowledgement: RuntimeHoldAcknowledgement,
    decision: RuntimeInterruptionDecision, prior_track: Px4Track, prior_track_sha256: str,
) -> None:
    gates = {
        "prior_track": sha256_json(prior_track) == prior_track_sha256,
        "ack_message": acknowledgement.message_sha256 == sha256_json(message),
        "ack_identity": (
            acknowledgement.message_id == message.message_id
            and acknowledgement.execution_id == message.execution_id
        ),
        "decision_message": decision.message_sha256 == sha256_json(message),
        "decision_ack": decision.hold_ack_sha256 == sha256_json(acknowledgement),
        "stable": all_required_gates_passed(acknowledgement.deterministic_gates),
        "authorized": (
            decision.authorized_action == "hold_for_replan"
            and all_required_gates_passed(decision.authorization_gates)
        ),
    }
    if not all_required_gates_passed(gates):
        failed = ",".join(name for name, accepted in gates.items() if not accepted)
        raise RuntimeReplanError(f"RUNTIME_REPLAN_INPUT_BINDING_FAILED:{failed}")


# 功能：
#   1. 逐点核对已检查路线、世界点与局部飞控坐标，不能仅用首点正确替代整条轨迹验证。
#   2. 保持活动坐标合同、悬停设置及车辆速度边界，并绑定当前语义资产的净空报告。
# 输入：
#   route：本次已检查的完整路线。
#   clearance：本次净空报告。
#   track：插件导出的执行轨迹。
#   prior_track：当前活动轨迹及其坐标和悬停约束。
#   vehicle：当前车辆速度边界。
#   semantic_sha256：预期语义资产摘要。
# 输出：
#   gates：几何、设置与证据一致性的逐项门控。
def _replacement_geometry_gates(
    route: GraphRoute, clearance: RouteClearanceReport, track: Px4Track,
    prior_track: Px4Track, vehicle: VehicleAsset, semantic_sha256: str,
    localization_variance_m2: float,
) -> dict[str, bool]:
    counts_match = (
        len(route.node_ids) == len(route.positions_m) == len(track.points)
        == len(track.source_world_points)
        and len(route.edge_ids) == len(route.positions_m) - 1
    )
    root = prior_track.coordinate_contract.model_root_world_enu_m
    offset = prior_track.coordinate_contract.resolved_collision_center_offset_model_m()
    world_matches = counts_match
    local_matches = counts_match
    for position, world, point in zip(
        route.positions_m, track.source_world_points, track.points, strict=False
    ):
        world_matches &= math.dist(
            (position.x, position.y, position.z), (world.east_m, world.north_m, world.up_m)
        ) <= 1e-6
        # 执行轨迹 z 是局部向上；只有发送到 NED 接口时才取反，不能在此再翻转一次。
        local_matches &= math.dist(
            (position.y - root[1] - offset[1], position.x - root[0] - offset[0],
             position.z - root[2] - offset[2]),
            (point.x, point.y, point.z),
        ) <= 1e-6
    settings = (
        "waypoint_hold_seconds", "waypoint_position_tolerance_m", "waypoint_speed_tolerance_mps",
        "waypoint_stable_window_seconds", "waypoint_settle_timeout_seconds",
    )
    gates = {
        "replacement_geometry_counts_match": counts_match,
        "replacement_world_points_match_route": world_matches,
        "replacement_local_points_match_route": local_matches,
        "replacement_coordinates_unchanged": (
            sha256_json(track.coordinate_contract) == sha256_json(prior_track.coordinate_contract)
        ),
        "replacement_hold_settings_preserved": (
            track.stop_at_waypoints is True
            and all(getattr(track, field) == getattr(prior_track, field) for field in settings)
        ),
        "replacement_speed_within_vehicle_envelope": all(
            0 < point.speed_limit_mps <= vehicle.max_speed_mps for point in track.points
        ),
        "replacement_clearance_semantic_bound": clearance.semantic_sha256 == semantic_sha256,
        "replacement_tracking_corridor_budget_accepted": (
            _route_tracking_budget_accepted(route, clearance, localization_variance_m2)
        ),
    }
    return gates


# 功能：
#   统一地点与覆盖改令的锚点约束，拒绝非有限或布尔距离，并落实已验证边和最大接入距离。
# 输入：
#   policy：锚点策略插件的输出。
#   graph：当前地图及边验收状态。
#   current_world：稳定悬停的世界位置。
# 输出：
#   selection：锚点标识、实测接入距离、允许距离上界的三元组。
def _validated_anchor_policy(policy: object, graph: MapAsset, current_world: Vector3):
    if not isinstance(policy, dict):
        raise RuntimeReplanError("RUNTIME_REPLAN_POLICY_INVALID")
    anchor = policy.get("anchor_node")
    maximum_join_distance = policy.get("maximum_join_distance_m")
    require_verified = policy.get("requires_flight_verified_anchor")
    if (
        not isinstance(anchor, str)
        or type(maximum_join_distance) not in (int, float)
        or not math.isfinite(maximum_join_distance)
        or maximum_join_distance <= 0
        or type(require_verified) is not bool
    ):
        raise RuntimeReplanError("RUNTIME_REPLAN_POLICY_INVALID")
    anchor_position = next(
        (node.position_m for node in graph.nodes if node.node_id == anchor), None
    )
    if anchor_position is None:
        raise RuntimeReplanError("RUNTIME_REPLAN_ANCHOR_UNKNOWN")
    join_distance = math.dist(
        (current_world.x, current_world.y, current_world.z),
        (anchor_position.x, anchor_position.y, anchor_position.z),
    )
    if not math.isfinite(join_distance) or join_distance > maximum_join_distance:
        raise RuntimeReplanError("RUNTIME_REPLAN_ANCHOR_TOO_FAR")
    if require_verified and not any(
        edge.qualification == "flight-verified" and anchor in {edge.from_node, edge.to_node}
        for edge in graph.edges
    ):
        raise RuntimeReplanError("RUNTIME_REPLAN_ANCHOR_NOT_FLIGHT_VERIFIED")
    selection = anchor, join_distance, float(maximum_join_distance)
    return selection


# 功能：
#   从当前悬停点接入锚点，组合去程及必要的返程；连接段仍须由后续整段净空检查验收。
# 输入：
#   graph：地图节点位置。
#   registry：本次使用的路径工具注册器。
#   current_world：当前悬停的世界 ENU 位置。
#   anchor：接入拓扑的节点。
#   target_node：任务目的地。
#   return_node：最后返回点。
# 输出：
#   result：完整路线候选及规划工具回执。
def _replacement_route(
    graph: MapAsset,
    registry: ToolRegistry,
    *,
    current_world: Vector3,
    anchor: str,
    target_node: str,
    return_node: str,
) -> tuple[GraphRoute, list[ToolReceipt]]:
    outbound, outbound_receipts, outbound_metric = _call_runtime_route(
        registry, RouteQuery(start_node=anchor, goal_node=target_node)
    )
    anchor_position = next(node.position_m for node in graph.nodes if node.node_id == anchor)
    join_distance = math.dist(
        (current_world.x, current_world.y, current_world.z),
        (anchor_position.x, anchor_position.y, anchor_position.z),
    )
    receipts = list(outbound_receipts)
    if target_node == return_node:
        node_ids = ["runtime-current", *outbound.node_ids]
        edge_ids = ["runtime-safe-join", *outbound.edge_ids]
        positions = [current_world, *outbound.positions_m]
        route_length = join_distance + outbound.route_length_m
    else:
        inbound, inbound_receipts, _inbound_metric = _call_runtime_route(
            registry,
            RouteQuery(start_node=target_node, goal_node=return_node),
            # If the outbound metric search could not traverse this corridor,
            # do not spend the same deadline again on its reverse leg.
            prefer_metric=outbound_metric,
        )
        receipts.extend(inbound_receipts)
        node_ids = ["runtime-current", *outbound.node_ids, *inbound.node_ids[1:]]
        edge_ids = ["runtime-safe-join", *outbound.edge_ids, *inbound.edge_ids]
        positions = [current_world, *outbound.positions_m, *inbound.positions_m[1:]]
        route_length = join_distance + outbound.route_length_m + inbound.route_length_m
    result = (
        GraphRoute(
            start_node="runtime-current",
            goal_node=return_node,
            node_ids=node_ids,
            edge_ids=edge_ids,
            positions_m=positions,
            route_length_m=route_length,
            all_edges_flight_verified=False,
        ),
        receipts,
    )
    return result


# 功能：
#   1. 根据获准改令解析目标，选择可接入锚点并规划去程和返程。
#   2. 验证悬停、资产、净空、逐点导出和执行修订绑定；不直接发送飞控指令。
# 输入：
#   message：用户改令消息。
#   acknowledgement：本消息的稳定悬停回执。
#   decision：已获准的改令分类与决定。
#   replacement_sequence：当前替换序号。
#   prior_track_sha256：活动轨迹摘要。
#   prior_track：活动执行轨迹。
#   graph：当前拓扑地图。
#   catalog：当前语义目录。
#   semantic_path：几何语义文件路径。
#   vehicle：当前车辆参数。
#   expected_map_sha256：合同绑定的地图摘要。
#   expected_semantic_sha256：合同绑定的几何语义摘要。
#   expected_vehicle_asset_id：合同绑定的车辆标识。
#   return_node：原返回点。
#   active_target_node：当前任务目标。
#   active_task_graph：当前活动任务图。
#   active_runtime_actions：当前动作合同。
#   completed_runtime_action_step_ids：已有成功证据的动作标识。
#   plugin_snapshot：任务冻结的插件快照，独立构建未提供时才发现本地插件。
# 输出：
#   bound_replacement：通过门控且绑定执行修订的替换轨迹。
def build_runtime_replacement(
    *,
    message: RuntimeUserMessage,
    acknowledgement: RuntimeHoldAcknowledgement,
    decision: RuntimeInterruptionDecision,
    replacement_sequence: int,
    prior_track_sha256: str,
    prior_track: Px4Track,
    graph: MapAsset,
    catalog: MapCatalog,
    semantic_path: Path,
    vehicle: VehicleAsset,
    expected_map_sha256: str,
    expected_semantic_sha256: str,
    expected_vehicle_asset_id: str,
    return_node: str,
    active_target_node: str | None = None,
    active_task_graph: TaskGraph | None = None,
    active_runtime_actions: RuntimeActionExecutionContract | None = None,
    completed_runtime_action_step_ids: set[str] | None = None,
    plugin_snapshot: PluginSnapshot | None = None,
) -> RuntimeReplacementTrack:
    _validate_replan_input_binding(
        message, acknowledgement, decision, prior_track, prior_track_sha256
    )
    localization_variance_m2 = _hold_localization_variance(acknowledgement)
    prior_return_node = return_node
    target_entity = decision.classification.target_entity
    amendment_action = decision.classification.requested_action
    if amendment_action not in {
        "replan", "redirect", "return_home", "set_return_point", "follow_target"
    }:
        raise RuntimeReplanError("RUNTIME_REPLAN_TARGET_REQUIRED")
    if amendment_action == "return_home" and not target_entity:
        target_node = return_node
    elif target_entity:
        target_node = _resolve_target(target_entity, catalog, graph)
    else:
        raise RuntimeReplanError("RUNTIME_REPLAN_TARGET_REQUIRED")
    requested_return_node: str | None = None
    if decision.classification.requested_action == "set_return_point":
        requested_return_node = target_node
        return_node = target_node
    current_world = _current_world_position(acknowledgement, prior_track)
    if requested_return_node is not None:
        try:
            next_index = bound_resume_point_index(acknowledgement, prior_track)
        except ValueError as error:
            raise RuntimeReplanError(str(error)) from error
        pickup_indices = [
            index for index, point in enumerate(prior_track.points) if point.phase == "pickup"
        ]
        target_already_reached = bool(pickup_indices and next_index > pickup_indices[0])
        target_node = (
            requested_return_node
            if target_already_reached
            else active_target_node or requested_return_node
        )
    environment = ToolEnvironment(
        map_graph=graph,
        semantic_path=semantic_path,
        vehicle_diameter_m=vehicle.body_radius_m * 2.0,
        vehicle_height_m=vehicle.body_height_m,
        waypoint_hold_seconds=prior_track.waypoint_hold_seconds,
        vehicle=vehicle,
        planning_localization_variance_m2=localization_variance_m2,
    )
    if plugin_snapshot is None:
        registry, _snapshot = build_discovered_tool_registry(environment)
    else:
        registry = build_snapshot_tool_registry(environment, plugin_snapshot)
    extensions = build_discovered_extension_registry(plugin_snapshot)
    try:
        replan_policy, hook_receipts = extensions.invoke_single(
            "runtime.replan-policy",
            "select_anchor",
            required=True,
            current_world=current_world,
            graph=graph,
            target_node=target_node,
            return_node=return_node,
            acknowledgement=acknowledgement,
        )
    except ExtensionExecutionError as error:
        raise RuntimeReplanError(str(error)) from error
    anchor, _join_distance, _max_join = _validated_anchor_policy(
        replan_policy, graph, current_world
    )
    active_return_node = target_node if amendment_action == "follow_target" else return_node
    try:
        route, receipts = _replacement_route(
            graph,
            registry,
            current_world=current_world,
            anchor=anchor,
            target_node=target_node,
            return_node=active_return_node,
        )
        clearance_value, clearance_receipt = registry.call_slot(
            "safety.route-clearance", route
        )
        clearance = RouteClearanceReport.model_validate(clearance_value)
        track_value, track_receipt = registry.call_slot(
            "runtime.track-export",
            RuntimeTrackRequest(
                route=route,
                prior_track=prior_track,
                target_node=target_node,
                vehicle=vehicle,
            ),
        )
    except ToolExecutionError as error:
        issue = error.receipt.issue_codes[0] if error.receipt.issue_codes else "UNKNOWN"
        raise RuntimeReplanError(
            f"RUNTIME_REPLAN_TOOL_FAILED:{error.receipt.tool_id}:{issue}"
        ) from error
    track = Px4Track.model_validate(track_value)
    receipts.extend([clearance_receipt, track_receipt])
    gates = {
        **_replacement_geometry_gates(
            route, clearance, track, prior_track, vehicle, expected_semantic_sha256,
            localization_variance_m2,
        ),
        "message_matches_decision": decision.message_sha256 == sha256_json(message),
        "hold_matches_decision": decision.hold_ack_sha256 == sha256_json(acknowledgement),
        "decision_authorized_replan": decision.authorized_action == "hold_for_replan",
        "stable_hold_gates_passed": all(acknowledgement.deterministic_gates.values()),
        "map_hash_matches_contract": sha256_json(graph) == expected_map_sha256,
        "semantic_hash_matches_contract": (
            hashlib.sha256(semantic_path.read_bytes()).hexdigest() == expected_semantic_sha256
        ),
        "vehicle_identity_matches_contract": vehicle.asset_id == expected_vehicle_asset_id,
        "replacement_clearance_accepted": clearance.accepted,
        "replacement_tracking_corridor_budget_accepted": (
            _route_tracking_budget_accepted(route, clearance, localization_variance_m2)
        ),
        "replacement_begins_at_stable_hold": (
            math.dist(
                (track.points[0].x, track.points[0].y, -track.points[0].z),
                (
                    acknowledgement.observed_position_ned_m.x,
                    acknowledgement.observed_position_ned_m.y,
                    acknowledgement.observed_position_ned_m.z,
                ),
            )
            <= 0.05
        ),
    }
    if not all(gates.values()):
        failed = ",".join(name for name, accepted in gates.items() if not accepted)
        raise RuntimeReplanError(f"RUNTIME_REPLAN_GATE_FAILED:{failed}")
    replacement = RuntimeReplacementTrack(
        message_id=message.message_id,
        execution_id=message.execution_id,
        replacement_sequence=replacement_sequence,
        message_sha256=sha256_json(message),
        hold_ack_sha256=sha256_json(acknowledgement),
        decision_sha256=sha256_json(decision),
        prior_track_sha256=prior_track_sha256,
        amendment_action=(
            amendment_action
            if amendment_action
            in {
                "replan",
                "redirect",
                "return_home",
                "set_return_point",
                "set_coverage",
                "follow_target",
            }
            else "replan"
        ),
        amendment_parameters=dict(
            {
                **(
                    decision.amendment_directive.parameters
                    if decision.amendment_directive is not None
                    else decision.classification.parameters
                ),
                **(
                    {"new_return_node": requested_return_node}
                    if requested_return_node is not None
                    else {}
                ),
            }
        ),
        target_node=target_node,
        return_node=return_node,
        route=route,
        clearance=clearance,
        track=track,
        plugin_tool_receipts=[
            RuntimeToolReceipt(
                tool_id=receipt.tool_id,
                plugin_id=receipt.plugin_id,
                plugin_package_sha256=receipt.plugin_package_sha256,
                outcome=receipt.outcome,
                input_sha256=receipt.input_sha256,
                output_sha256=receipt.output_sha256,
            )
            for receipt in receipts
        ],
        plugin_hook_receipts=hook_receipts,
        deterministic_gates=gates,
        generated_at=datetime.now(UTC),
    )
    bound_replacement = _bind_execution_revision(
        replacement,
        task_graph=active_task_graph,
        prior_actions=active_runtime_actions,
        completed_step_ids=completed_runtime_action_step_ids,
        graph=graph,
        prior_target_node=active_target_node or target_node,
        prior_return_node=prior_return_node,
    )
    return bound_replacement


# 功能：
#   从稳定悬停位置重建剩余轨迹，对导出速度仅作收紧，并复核每段净空和执行动作绑定。
# 输入：
#   message：用户限速消息。
#   acknowledgement：稳定悬停回执。
#   decision：获准的 set_speed 决定。
#   replacement_sequence：替换序号。
#   prior_track_sha256：活动轨迹摘要。
#   prior_track：提供剩余世界点及坐标合同的活动轨迹。
#   graph：当前地图。
#   semantic_path：几何语义文件。
#   vehicle：车辆速度边界。
#   expected_map_sha256：预期地图摘要。
#   expected_semantic_sha256：预期几何语义摘要。
#   expected_vehicle_asset_id：预期车辆标识。
#   active_target_node：当前任务目标。
#   active_return_node：当前返回点。
#   active_task_graph：活动任务图。
#   active_runtime_actions：活动动作合同。
#   completed_runtime_action_step_ids：已完成动作标识。
#   plugin_snapshot：任务冻结插件快照，独立构建时可不提供。
# 输出：
#   bound_replacement：带速度上限和完整绑定的替换轨迹。
def build_runtime_speed_replacement(
    *,
    message: RuntimeUserMessage,
    acknowledgement: RuntimeHoldAcknowledgement,
    decision: RuntimeInterruptionDecision,
    replacement_sequence: int,
    prior_track_sha256: str,
    prior_track: Px4Track,
    graph: MapAsset,
    semantic_path: Path,
    vehicle: VehicleAsset,
    expected_map_sha256: str,
    expected_semantic_sha256: str,
    expected_vehicle_asset_id: str,
    active_target_node: str | None = None,
    active_return_node: str | None = None,
    active_task_graph: TaskGraph | None = None,
    active_runtime_actions: RuntimeActionExecutionContract | None = None,
    completed_runtime_action_step_ids: set[str] | None = None,
    plugin_snapshot: PluginSnapshot | None = None,
) -> RuntimeReplacementTrack:
    _validate_replan_input_binding(
        message, acknowledgement, decision, prior_track, prior_track_sha256
    )
    localization_variance_m2 = _hold_localization_variance(acknowledgement)
    if (
        decision.authorized_action != "hold_for_replan"
        or decision.classification.requested_action != "set_speed"
    ):
        raise RuntimeReplanError("RUNTIME_SPEED_REPLACEMENT_NOT_AUTHORIZED")
    raw_speed = decision.classification.parameters.get("maximum_speed_mps")
    if not isinstance(raw_speed, int | float) or isinstance(raw_speed, bool):
        raise RuntimeReplanError("RUNTIME_SPEED_VALUE_REQUIRED")
    speed_limit = float(raw_speed)
    if not 0.1 <= speed_limit <= vehicle.max_speed_mps:
        raise RuntimeReplanError("RUNTIME_SPEED_OUTSIDE_VEHICLE_ENVELOPE")
    current_world = _current_world_position(acknowledgement, prior_track)
    prior_world = prior_track.source_world_points
    try:
        next_index = bound_resume_point_index(acknowledgement, prior_track)
    except ValueError as error:
        raise RuntimeReplanError(str(error)) from error
    # 保留仍未到达的转弯点；位置最近不代表已经经过，尤其不能据此跳段或重启环路。
    suffix = prior_world[next_index:]
    positions = [
        current_world,
        *[Vector3(x=point.east_m, y=point.north_m, z=point.up_m) for point in suffix],
    ]
    node_ids = [
        "runtime-current",
        *[f"runtime-speed-{index:04d}" for index in range(1, len(positions))],
    ]
    edge_ids = [f"runtime-speed-edge-{index:04d}" for index in range(len(positions) - 1)]
    route = GraphRoute(
        start_node=node_ids[0],
        goal_node=node_ids[-1],
        node_ids=node_ids,
        edge_ids=edge_ids,
        positions_m=positions,
        route_length_m=sum(
            math.dist(
                (start.x, start.y, start.z),
                (end.x, end.y, end.z),
            )
            for start, end in zip(positions, positions[1:], strict=False)
        ),
        all_edges_flight_verified=False,
    )
    environment = ToolEnvironment(
        map_graph=graph,
        semantic_path=semantic_path,
        vehicle_diameter_m=vehicle.body_radius_m * 2.0,
        vehicle_height_m=vehicle.body_height_m,
        waypoint_hold_seconds=prior_track.waypoint_hold_seconds,
        vehicle=vehicle,
        planning_localization_variance_m2=localization_variance_m2,
    )
    registry = (
        build_discovered_tool_registry(environment)[0]
        if plugin_snapshot is None
        else build_snapshot_tool_registry(environment, plugin_snapshot)
    )
    clearance_value, clearance_receipt = registry.call_slot("safety.route-clearance", route)
    clearance = RouteClearanceReport.model_validate(clearance_value)
    track_value, track_receipt = registry.call_slot(
        "runtime.track-export",
        RuntimeTrackRequest(
            route=route,
            prior_track=prior_track,
            target_node=node_ids[-1],
            vehicle=vehicle,
        ),
    )
    exported = Px4Track.model_validate(track_value)
    track = exported.model_copy(
        update={
            "points": [
                point.model_copy(
                    update={"speed_limit_mps": min(point.speed_limit_mps, speed_limit)}
                )
                for point in exported.points
            ]
        }
    )
    gates = {
        **_replacement_geometry_gates(
            route, clearance, track, prior_track, vehicle, expected_semantic_sha256,
            localization_variance_m2,
        ),
        "message_matches_decision": decision.message_sha256 == sha256_json(message),
        "hold_matches_decision": decision.hold_ack_sha256 == sha256_json(acknowledgement),
        "decision_authorized_replan": decision.authorized_action == "hold_for_replan",
        "stable_hold_gates_passed": all(acknowledgement.deterministic_gates.values()),
        "map_hash_matches_contract": sha256_json(graph) == expected_map_sha256,
        "semantic_hash_matches_contract": (
            hashlib.sha256(semantic_path.read_bytes()).hexdigest() == expected_semantic_sha256
        ),
        "vehicle_identity_matches_contract": vehicle.asset_id == expected_vehicle_asset_id,
        "replacement_clearance_accepted": clearance.accepted,
        "replacement_begins_at_stable_hold": (
            math.dist(
                (track.points[0].x, track.points[0].y, -track.points[0].z),
                (
                    acknowledgement.observed_position_ned_m.x,
                    acknowledgement.observed_position_ned_m.y,
                    acknowledgement.observed_position_ned_m.z,
                ),
            )
            <= 0.05
        ),
        "all_points_respect_speed_ceiling": all(
            point.speed_limit_mps <= speed_limit for point in track.points
        ),
    }
    if not all(gates.values()):
        failed = ",".join(name for name, accepted in gates.items() if not accepted)
        raise RuntimeReplanError(f"RUNTIME_SPEED_REPLACEMENT_GATE_FAILED:{failed}")
    replacement = RuntimeReplacementTrack(
        message_id=message.message_id,
        execution_id=message.execution_id,
        replacement_sequence=replacement_sequence,
        message_sha256=sha256_json(message),
        hold_ack_sha256=sha256_json(acknowledgement),
        decision_sha256=sha256_json(decision),
        prior_track_sha256=prior_track_sha256,
        amendment_action="set_speed",
        amendment_parameters={"maximum_speed_mps": speed_limit},
        target_node=active_target_node or node_ids[-1],
        return_node=active_return_node or node_ids[-1],
        route=route,
        clearance=clearance,
        track=track,
        plugin_tool_receipts=[
            RuntimeToolReceipt(
                tool_id=receipt.tool_id,
                plugin_id=receipt.plugin_id,
                plugin_package_sha256=receipt.plugin_package_sha256,
                outcome=receipt.outcome,
                input_sha256=receipt.input_sha256,
                output_sha256=receipt.output_sha256,
            )
            for receipt in (clearance_receipt, track_receipt)
        ],
        deterministic_gates=gates,
        generated_at=datetime.now(UTC),
    )
    bound_replacement = _bind_execution_revision(
        replacement,
        task_graph=active_task_graph,
        prior_actions=active_runtime_actions,
        completed_step_ids=completed_runtime_action_step_ids,
        graph=graph,
        prior_target_node=active_target_node or replacement.target_node,
        prior_return_node=active_return_node or replacement.return_node,
    )
    return bound_replacement


# 功能：
#   按选定覆盖插件生成区域扫描形状，再组合接入、覆盖和返回路线并校验整条执行轨迹。
# 输入：
#   message：用户区域覆盖改令。
#   acknowledgement：稳定悬停回执。
#   decision：获准的 set_coverage 决定。
#   replacement_sequence：替换序号。
#   prior_track_sha256：活动轨迹摘要。
#   prior_track：活动轨迹和坐标合同。
#   graph：当前拓扑图。
#   catalog：语义目标目录。
#   semantic_path：几何语义文件。
#   vehicle：车辆包络与速度参数。
#   expected_map_sha256：预期地图摘要。
#   expected_semantic_sha256：预期几何语义摘要。
#   expected_vehicle_asset_id：预期车辆标识。
#   return_node：覆盖后的返回点。
#   active_target_node：原活动任务目标。
#   active_task_graph：活动任务图。
#   active_runtime_actions：活动动作合同。
#   completed_runtime_action_step_ids：已完成动作标识。
#   plugin_snapshot：任务冻结插件快照，独立构建时可不提供。
# 输出：
#   bound_replacement：整条路线门控通过并绑定执行修订的覆盖轨迹。
def build_runtime_coverage_replacement(
    *,
    message: RuntimeUserMessage,
    acknowledgement: RuntimeHoldAcknowledgement,
    decision: RuntimeInterruptionDecision,
    replacement_sequence: int,
    prior_track_sha256: str,
    prior_track: Px4Track,
    graph: MapAsset,
    catalog: MapCatalog,
    semantic_path: Path,
    vehicle: VehicleAsset,
    expected_map_sha256: str,
    expected_semantic_sha256: str,
    expected_vehicle_asset_id: str,
    return_node: str,
    active_target_node: str | None = None,
    active_task_graph: TaskGraph | None = None,
    active_runtime_actions: RuntimeActionExecutionContract | None = None,
    completed_runtime_action_step_ids: set[str] | None = None,
    plugin_snapshot: PluginSnapshot | None = None,
) -> RuntimeReplacementTrack:
    _validate_replan_input_binding(
        message, acknowledgement, decision, prior_track, prior_track_sha256
    )
    localization_variance_m2 = _hold_localization_variance(acknowledgement)
    if (
        decision.authorized_action != "hold_for_replan"
        or decision.classification.requested_action != "set_coverage"
    ):
        raise RuntimeReplanError("RUNTIME_COVERAGE_NOT_AUTHORIZED")
    parameters = dict(
        decision.amendment_directive.parameters
        if decision.amendment_directive is not None
        else decision.classification.parameters
    )
    target_entity = decision.classification.target_entity
    raw_polygon = parameters.get("polygon_enu_m", [])
    if not isinstance(raw_polygon, list):
        raise RuntimeReplanError("RUNTIME_COVERAGE_POLYGON_INVALID")
    try:
        polygon = [Vector3.model_validate(point) for point in raw_polygon]
    except ValueError as exc:
        raise RuntimeReplanError("RUNTIME_COVERAGE_POLYGON_INVALID") from exc
    if target_entity:
        target_node = _resolve_target(target_entity, catalog, graph)
        center = next(node.position_m for node in graph.nodes if node.node_id == target_node)
    elif len(polygon) >= 3:
        center = Vector3(
            x=sum(point.x for point in polygon) / len(polygon),
            y=sum(point.y for point in polygon) / len(polygon),
            z=sum(point.z for point in polygon) / len(polygon),
        )
        target_node = min(
            graph.nodes,
            key=lambda node: math.dist(
                (node.position_m.x, node.position_m.y, node.position_m.z),
                (center.x, center.y, center.z),
            ),
        ).node_id
    else:
        raise RuntimeReplanError("RUNTIME_COVERAGE_TARGET_OR_POLYGON_REQUIRED")
    lane_spacing = float(parameters.get("lane_spacing_m", max(vehicle.body_radius_m * 2.5, 0.5)))
    boundary_margin = float(parameters.get("boundary_margin_m", vehicle.body_radius_m + 0.15))
    request = CoveragePlanRequest(
        center_enu_m=center,
        polygon_enu_m=polygon,
        width_m=float(parameters.get("width_m", 6.0)),
        height_m=float(parameters.get("height_m", 6.0)),
        lane_spacing_m=lane_spacing,
        boundary_margin_m=boundary_margin,
        altitude_m=float(parameters.get("altitude_m", center.z)),
    )
    environment = ToolEnvironment(
        map_graph=graph,
        semantic_path=semantic_path,
        vehicle_diameter_m=vehicle.body_radius_m * 2.0,
        vehicle_height_m=vehicle.body_height_m,
        waypoint_hold_seconds=prior_track.waypoint_hold_seconds,
        vehicle=vehicle,
        planning_localization_variance_m2=localization_variance_m2,
    )
    registry = (
        build_discovered_tool_registry(environment)[0]
        if plugin_snapshot is None
        else build_snapshot_tool_registry(environment, plugin_snapshot)
    )
    extensions = build_discovered_extension_registry(plugin_snapshot)
    current_world = _current_world_position(acknowledgement, prior_track)
    try:
        anchor_value, anchor_receipts = extensions.invoke_single(
            "runtime.replan-policy",
            "select_anchor",
            required=True,
            current_world=current_world,
            graph=graph,
            target_node=target_node,
            return_node=return_node,
            acknowledgement=acknowledgement,
        )
        pattern_value, coverage_receipts = extensions.invoke_single(
            "runtime.coverage-planner",
            "plan_coverage",
            required=True,
            request=request,
        )
    except ExtensionExecutionError as error:
        raise RuntimeReplanError(str(error)) from error
    anchor, join_distance, max_join = _validated_anchor_policy(anchor_value, graph, current_world)
    pattern = CoveragePattern.model_validate(pattern_value)
    if not all_required_gates_passed(pattern.deterministic_gates):
        raise RuntimeReplanError("RUNTIME_COVERAGE_PATTERN_GATE_FAILED")
    outbound, outbound_receipts, outbound_metric = _call_runtime_route(
        registry, RouteQuery(start_node=anchor, goal_node=target_node)
    )
    inbound, inbound_receipts, _inbound_metric = _call_runtime_route(
        registry,
        RouteQuery(start_node=target_node, goal_node=return_node),
        prefer_metric=outbound_metric,
    )
    coverage_nodes = [f"runtime-coverage-{index:04d}" for index in range(len(pattern.points_enu_m))]
    positions = [
        current_world,
        *outbound.positions_m,
        *pattern.points_enu_m,
        *inbound.positions_m,
    ]
    node_ids = [
        "runtime-current",
        *outbound.node_ids,
        *coverage_nodes,
        *inbound.node_ids,
    ]
    edge_ids = [f"runtime-coverage-edge-{index:04d}" for index in range(len(positions) - 1)]
    route = GraphRoute(
        start_node="runtime-current",
        goal_node=return_node,
        node_ids=node_ids,
        edge_ids=edge_ids,
        positions_m=positions,
        route_length_m=sum(
            math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
            for a, b in zip(positions, positions[1:], strict=False)
        ),
        all_edges_flight_verified=False,
    )
    clearance_value, clearance_receipt = registry.call_slot("safety.route-clearance", route)
    clearance = RouteClearanceReport.model_validate(clearance_value)
    track_value, track_receipt = registry.call_slot(
        "runtime.track-export",
        RuntimeTrackRequest(
            route=route,
            prior_track=prior_track,
            target_node=target_node,
            vehicle=vehicle,
        ),
    )
    track = Px4Track.model_validate(track_value)
    gates = {
        **_replacement_geometry_gates(
            route, clearance, track, prior_track, vehicle, expected_semantic_sha256,
            localization_variance_m2,
        ),
        "message_matches_decision": decision.message_sha256 == sha256_json(message),
        "hold_matches_decision": decision.hold_ack_sha256 == sha256_json(acknowledgement),
        "decision_authorized_coverage": decision.authorized_action == "hold_for_replan",
        "stable_hold_gates_passed": all(acknowledgement.deterministic_gates.values()),
        "map_hash_matches_contract": sha256_json(graph) == expected_map_sha256,
        "semantic_hash_matches_contract": (
            hashlib.sha256(semantic_path.read_bytes()).hexdigest() == expected_semantic_sha256
        ),
        "vehicle_identity_matches_contract": vehicle.asset_id == expected_vehicle_asset_id,
        "coverage_pattern_gates_passed": all(pattern.deterministic_gates.values()),
        "coverage_has_multiple_points": len(pattern.points_enu_m) >= 2,
        "safe_anchor_join": max_join > 0 and join_distance <= max_join,
        "replacement_clearance_accepted": clearance.accepted,
        "replacement_tracking_corridor_budget_accepted": (
            _route_tracking_budget_accepted(route, clearance, localization_variance_m2)
        ),
        "replacement_begins_at_stable_hold": (
            math.dist(
                (track.points[0].x, track.points[0].y, -track.points[0].z),
                (
                    acknowledgement.observed_position_ned_m.x,
                    acknowledgement.observed_position_ned_m.y,
                    acknowledgement.observed_position_ned_m.z,
                ),
            )
            <= 0.05
        ),
    }
    if not all(gates.values()):
        failed = ",".join(name for name, accepted in gates.items() if not accepted)
        raise RuntimeReplanError(f"RUNTIME_COVERAGE_GATE_FAILED:{failed}")
    replacement = RuntimeReplacementTrack(
        message_id=message.message_id,
        execution_id=message.execution_id,
        replacement_sequence=replacement_sequence,
        message_sha256=sha256_json(message),
        hold_ack_sha256=sha256_json(acknowledgement),
        decision_sha256=sha256_json(decision),
        prior_track_sha256=prior_track_sha256,
        amendment_action="set_coverage",
        amendment_parameters={
            **parameters,
            "pattern": pattern.pattern,
            "lane_count": pattern.lane_count,
            "estimated_area_m2": pattern.estimated_area_m2,
        },
        target_node=target_node,
        return_node=return_node,
        route=route,
        clearance=clearance,
        track=track,
        plugin_tool_receipts=[
            RuntimeToolReceipt(
                tool_id=receipt.tool_id,
                plugin_id=receipt.plugin_id,
                plugin_package_sha256=receipt.plugin_package_sha256,
                outcome=receipt.outcome,
                input_sha256=receipt.input_sha256,
                output_sha256=receipt.output_sha256,
            )
            for receipt in (
                *outbound_receipts,
                *inbound_receipts,
                clearance_receipt,
                track_receipt,
            )
        ],
        plugin_hook_receipts=[*anchor_receipts, *coverage_receipts],
        deterministic_gates=gates,
        generated_at=datetime.now(UTC),
    )
    bound_replacement = _bind_execution_revision(
        replacement,
        task_graph=active_task_graph,
        prior_actions=active_runtime_actions,
        completed_step_ids=completed_runtime_action_step_ids,
        graph=graph,
        prior_target_node=active_target_node or target_node,
        prior_return_node=return_node,
    )
    return bound_replacement
