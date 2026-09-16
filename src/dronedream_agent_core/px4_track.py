"""Convert validated ENU graph routes into the real PX4 executor contract."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .assets import _load_object
from .contracts import (
    GraphRoute,
    MapAsset,
    Px4CoordinateContract,
    Px4Track,
    Px4TrackPoint,
    RuntimeTrackRequest,
    VehicleAsset,
    WorldTrackPoint,
)
from .runtime_bindings import (
    load_map_runtime_bindings,
    resolve_vehicle_collision_center_offset,
)


# 功能：
#   复用资产层的单次有界语义读取，拒绝链接替换、重复字段、非有限值及过深结构。
# 输入：
#   path：当前地图语义文件路径。
# 输出：
#   value：读取并验证的语义字典。
def _load(path: Path) -> dict[str, Any]:
    value, _raw = _load_object(path)
    return value


# 功能：
#   检查路线数组和首末节点一致且至少有一段，避免静默丢弃多余数据或使用虚构边。
# 输入：
#   route：将被导出为飞控轨迹的路线。
# 输出：
#   None：不返回业务数据。
def _validate_route_structure(route: GraphRoute) -> None:
    if (
        len(route.positions_m) < 2
        or len(route.node_ids) != len(route.positions_m)
        or len(route.edge_ids) != len(route.positions_m) - 1
        or route.node_ids[0] != route.start_node
        or route.node_ids[-1] != route.goal_node
    ):
        raise ValueError("PX4_ROUTE_STRUCTURE_INVALID")


# 功能：
#   1. 用地图出生点和所选车辆碰撞中心，把世界 ENU 路线转换为 PX4 局部轨迹。
#   2. 要求起点匹配已绑定起飞点，保留楼梯等阶段并把所有速度限制在车辆能力内。
# 输入：
#   route：已规划路线，仍须由上游完成连续碰撞验收。
#   graph：当前地图节点、边与语义。
#   semantic_path：含显式运行绑定的语义文件。
#   vehicle：所选车辆几何和速度边界。
#   waypoint_hold_seconds：各航点的稳定保持时长。
# 输出：
#   track：独立的世界点、局部飞控点及坐标合同。
def route_to_px4_track(
    route: GraphRoute, graph: MapAsset, semantic_path: Path, *, vehicle: VehicleAsset,
    waypoint_hold_seconds: float = 0.4,
) -> Px4Track:
    _validate_route_structure(route)
    semantic = _load(semantic_path)
    bindings = load_map_runtime_bindings(semantic)
    spawn = bindings.vehicle_spawn
    offset = resolve_vehicle_collision_center_offset(vehicle)
    launch = bindings.mission_launch_waypoint
    model_root = (spawn.x, spawn.y, spawn.z)
    center_offset = (offset.x, offset.y, offset.z)
    expected_launch = (launch.x, launch.y, launch.z)
    first = route.positions_m[0]
    if math.dist((first.x, first.y, first.z), expected_launch) > 0.02:
        raise ValueError("route does not begin at the qualified PX4 launch waypoint")

    edge_by_id = {edge.edge_id: edge for edge in graph.edges}
    node_by_id = {node.node_id: node for node in graph.nodes}
    synthesized_speed_limit_mps = min(
        vehicle.max_speed_mps,
        0.6,
    )
    speed_limits = []
    for edge_id in route.edge_ids:
        edge = edge_by_id.get(edge_id)
        if edge is None:
            if not edge_id.startswith("metric-edge-"):
                raise ValueError(f"route references an unknown edge: {edge_id}")
            speed_limits.append(synthesized_speed_limit_mps)
        else:
            speed_limits.append(min(edge.speed_limit_mps, vehicle.max_speed_mps))
    if len(speed_limits) != len(route.positions_m) - 1:
        raise ValueError("route point and edge counts do not align")

    points: list[Px4TrackPoint] = []
    world_points: list[WorldTrackPoint] = []
    for index, world in enumerate(route.positions_m):
        phase = (
            "launch" if index == 0 else "land" if index == len(route.positions_m) - 1 else "transit"
        )
        node_id = route.node_ids[index]
        node = node_by_id.get(node_id)
        if node is None and node_id.startswith("metric-"):
            node = min(
                graph.nodes,
                key=lambda candidate: math.dist(
                    (world.x, world.y, world.z),
                    (
                        candidate.position_m.x,
                        candidate.position_m.y,
                        candidate.position_m.z,
                    ),
                ),
            )
        if node is None:
            raise ValueError(f"route references an unknown node: {node_id}")
        if node.semantic == "stairs" and phase == "transit":
            phase = "stairs"
        speed = speed_limits[index - 1] if index > 0 else speed_limits[0]
        if phase == "stairs":
            speed = min(speed, 0.45)
        points.append(
            Px4TrackPoint(
                # 此处 x 为局部北、y 为局部东、z 向上；执行器发送 NED 时再对高度取反。
                x=world.y - model_root[1] - center_offset[1],
                y=world.x - model_root[0] - center_offset[0],
                z=world.z - model_root[2] - center_offset[2],
                phase=phase,
                speed_limit_mps=speed,
            )
        )
        world_points.append(WorldTrackPoint(east_m=world.x, north_m=world.y, up_m=world.z))
    track = Px4Track(
        coordinate_contract=Px4CoordinateContract(
            model_root_world_enu_m=list(model_root),
            collision_center_offset_model_m=list(center_offset),
        ),
        points=points,
        source_world_points=world_points,
        waypoint_hold_seconds=waypoint_hold_seconds,
    )
    return track


# 功能：
#   从稳定悬停后的新路线导出局部飞控点，沿用但不共享原坐标合同，保留原稳定控制设置。
# 输入：
#   request：新路线、当前轨迹、目标与车辆参数。
#   graph：当前地图的边速限制及楼梯语义。
# 输出：
#   track：导出后的独立轨迹，尚需重规划消费端验证净空及逐点绑定。
def runtime_route_to_px4_track(request: RuntimeTrackRequest, graph: MapAsset) -> Px4Track:
    route = request.route
    _validate_route_structure(route)
    prior_track = request.prior_track
    vehicle = request.vehicle
    root_east, root_north, root_up = prior_track.coordinate_contract.model_root_world_enu_m
    offset_east, offset_north, offset_up = (
        prior_track.coordinate_contract.resolved_collision_center_offset_model_m()
    )
    edges = {edge.edge_id: edge for edge in graph.edges}
    nodes = {node.node_id: node for node in graph.nodes}
    join_speed = min(0.6, vehicle.max_speed_mps)
    points: list[Px4TrackPoint] = []
    world_points: list[WorldTrackPoint] = []
    target_seen = False
    for index, (node_id, world) in enumerate(zip(route.node_ids, route.positions_m, strict=True)):
        if index == len(route.positions_m) - 1:
            phase = "land"
        elif node_id == request.target_node and not target_seen:
            phase = "pickup"
            target_seen = True
        elif target_seen:
            phase = "return"
        else:
            semantic = nodes[node_id].semantic if node_id in nodes else "outdoor"
            phase = "stairs" if semantic == "stairs" else "transit"
        # 结构检查保证真实存在对应边；合成边可使用保守速度，但缺边不能冒充安全接入段。
        edge_id = route.edge_ids[max(0, index - 1)]
        edge_speed = edges[edge_id].speed_limit_mps if edge_id in edges else join_speed
        points.append(
            Px4TrackPoint(
                x=world.y - root_north - offset_north,
                y=world.x - root_east - offset_east,
                z=world.z - root_up - offset_up,
                phase=phase,
                speed_limit_mps=min(edge_speed, vehicle.max_speed_mps),
            )
        )
        world_points.append(WorldTrackPoint(east_m=world.x, north_m=world.y, up_m=world.z))
    track = Px4Track(
        coordinate_contract=prior_track.coordinate_contract.model_copy(deep=True),
        points=points,
        source_world_points=world_points,
        stop_at_waypoints=True,
        waypoint_hold_seconds=prior_track.waypoint_hold_seconds,
        waypoint_position_tolerance_m=prior_track.waypoint_position_tolerance_m,
        waypoint_speed_tolerance_mps=prior_track.waypoint_speed_tolerance_mps,
        waypoint_stable_window_seconds=prior_track.waypoint_stable_window_seconds,
        waypoint_settle_timeout_seconds=prior_track.waypoint_settle_timeout_seconds,
    )
    return track
