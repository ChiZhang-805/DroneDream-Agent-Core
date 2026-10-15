from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_runtime_replan import _replan_inputs

from dronedream_agent_core.contracts import (
    GraphRoute,
    MapAsset,
    RuntimeTrackRequest,
    Vector3,
    VehicleAsset,
)
from dronedream_agent_core.px4_track import _load as load_track_semantic
from dronedream_agent_core.px4_track import route_to_px4_track, runtime_route_to_px4_track
from dronedream_agent_core.runtime_bindings import (
    MapRuntimeBindingsError,
    load_map_runtime_bindings,
    resolve_grounded_vehicle_model_root,
)


# 功能：
#   创建非零世界出生点的仓库拓扑，检测坐标换算是否误用固定学校地图原点。
# 输入：
#   无。
# 输出：
#   graph：测试用仓库地图。
def _graph() -> MapAsset:
    graph = MapAsset.model_validate(
        {
            "asset_id": "warehouse-map",
            "name": "Warehouse Map",
            "nodes": [
                {
                    "node_id": "roof-launch",
                    "label": "Roof launch",
                    "position_m": {"x": 12.0, "y": 22.0, "z": 8.0},
                    "semantic": "launch",
                },
                {
                    "node_id": "dock-a",
                    "label": "Dock A",
                    "position_m": {"x": 15.0, "y": 24.0, "z": 3.0},
                    "semantic": "pickup",
                },
            ],
            "edges": [
                {
                    "edge_id": "launch-to-dock",
                    "from_node": "roof-launch",
                    "to_node": "dock-a",
                    "distance_m": 6.164414,
                    "minimum_clearance_m": 2.0,
                    "speed_limit_mps": 1.2,
                }
            ],
            "named_entities": {
                "roof-launch": "roof-launch",
                "dock-a": "dock-a",
            },
        }
    )
    return graph


# 功能：
#   提供当前语义结构和显式仿真绑定，车辆几何不存入地图运行绑定。
# 输入：
#   无。
# 输出：
#   semantic：测试用地图语义字典。
def _semantic() -> dict[str, object]:
    semantic = {
        "schema_version": "dronedream.map-semantic.v1",
        "coordinate_frame": "ENU",
        "scene_id": "warehouse-a",
        "entities": [
            {
                "entity_id": "roof-launch",
                "aliases": ["roof launch pad"],
                "position_m": [12.0, 22.0, 8.0],
                "semantic": "launch",
            },
            {
                "entity_id": "dock-a",
                "aliases": ["loading dock"],
                "position_m": [15.0, 24.0, 3.0],
                "semantic": "pickup",
            },
        ],
        "runtime_bindings": {
            "schema_version": "dronedream.map-runtime-bindings.v1",
            "simulator": "gazebo-harmonic",
            "coordinate_frame": "ENU",
            "vehicle_spawn": {"x": 10.0, "y": 20.0, "z": 5.0},
            "mission_launch_waypoint": {"x": 12.0, "y": 22.0, "z": 8.0},
        },
        "collision_primitives": [
            {
                "name": "roof-support",
                "semantic": "terrain",
                "center_x": 10.0,
                "center_y": 20.0,
                "center_z": 4.9,
                "size_x": 4.0,
                "size_y": 4.0,
                "size_z": 0.2,
                "yaw_rad": 0.0,
            }
        ],
    }
    return semantic


# 功能：
#   创建带非零水平碰撞中心偏移的车辆，检验完整三轴转换而非仅高度补偿。
# 输入：
#   无。
# 输出：
#   vehicle：测试车辆几何和性能参数。
def _vehicle() -> VehicleAsset:
    vehicle = VehicleAsset(
        asset_id="warehouse-drone",
        name="Warehouse Drone",
        dry_mass_kg=1.2,
        max_takeoff_mass_kg=2.5,
        body_radius_m=0.3,
        body_height_m=0.5,
        collision_center_offset_model_m={"x": 0.1, "y": 0.0, "z": 0.25},
        max_speed_mps=4.0,
        max_acceleration_mps2=3.0,
        qualified_range_m=1_000.0,
        reserve_battery_percent=20.0,
        max_pickup_payload_kg=0.8,
        sensors=["camera", "lidar"],
    )
    return vehicle


# 功能：
#   核对出生点和所选车辆几何共同决定 ENU 到 PX4 局部坐标的三轴换算。
# 输入：
#   tmp_path：临时语义文件目录。
# 输出：
#   None：不返回业务数据。
def test_map_spawn_and_selected_vehicle_drive_complete_enu_to_px4_transform(
    tmp_path: Path,
) -> None:
    graph = _graph()
    route = GraphRoute(
        start_node="roof-launch",
        goal_node="dock-a",
        node_ids=["roof-launch", "dock-a"],
        edge_ids=["launch-to-dock"],
        positions_m=[graph.nodes[0].position_m, graph.nodes[1].position_m],
        route_length_m=graph.edges[0].distance_m,
        all_edges_flight_verified=False,
    )
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(json.dumps(_semantic()), encoding="utf-8")

    track = route_to_px4_track(route, graph, semantic_path, vehicle=_vehicle())

    assert track.schema_version == "dronedream.px4-track.v2"
    assert track.coordinate_contract.collision_center_offset_model_m == [0.1, 0.0, 0.25]
    assert track.points[0].x == pytest.approx(2.0)
    assert track.points[0].y == pytest.approx(2.0)
    assert track.points[0].z == pytest.approx(2.75)


# 功能：
#   回归真实 X500 深度机型曾把地图通用出生高度和碰撞中心重复相加的错误。
# 输入：
#   无；用地面顶面、真实机身高度和碰撞中心偏移构造最小语义图。
# 输出：
#   None：模型根落在 -0.013 米且碰撞中心在机身半高时通过。
def test_grounded_model_root_uses_support_surface_and_vehicle_envelope() -> None:
    semantic = _semantic()
    semantic["runtime_bindings"]["vehicle_spawn"] = {"x": 10.0, "y": 20.0, "z": 0.25}
    semantic["runtime_bindings"]["mission_launch_waypoint"] = {
        "x": 10.0,
        "y": 20.0,
        "z": 1.4,
    }
    semantic["collision_primitives"] = [
        {
            "name": "floor",
            "semantic": "terrain",
            "center_x": 10.0,
            "center_y": 20.0,
            "center_z": -0.06,
            "size_x": 4.0,
            "size_y": 4.0,
            "size_z": 0.12,
            "yaw_rad": 0.0,
        }
    ]
    vehicle = VehicleAsset.model_validate(
        {
            **_vehicle().model_dump(mode="python"),
            "body_height_m": 0.43,
            "collision_center_offset_model_m": {"x": 0.0, "y": 0.0, "z": 0.228},
        }
    )

    root = resolve_grounded_vehicle_model_root(semantic, vehicle)

    assert (root.x, root.y, root.z) == pytest.approx((10.0, 20.0, -0.013))


# 功能：
#   回归多层地图只把 terrain 当支撑面、导致三楼出生点被移到零层的错误。
# 输入：
#   无；构造地面、楼板、圆形停机坪和高于起飞点的屋顶。
# 输出：
#   None：模型根必须停在地图声明的三楼停机坪高度。
def test_grounded_model_root_selects_declared_storey_launch_pad() -> None:
    semantic = _semantic()
    semantic["runtime_bindings"]["vehicle_spawn"] = {"x": -42.25, "y": 15.3, "z": 7.487}
    semantic["runtime_bindings"]["mission_launch_waypoint"] = {
        "x": -42.25,
        "y": 15.3,
        "z": 8.15,
    }
    semantic["runtime_collision_primitives"] = [
        {
            "name": "ground", "semantic": "terrain",
            "center_x": -42.25, "center_y": 15.3, "center_z": -0.05,
            "size_x": 100.0, "size_y": 100.0, "size_z": 0.1,
        },
        {
            "name": "third-floor", "semantic": "floor",
            "center_x": -42.25, "center_y": 15.3, "center_z": 7.31,
            "size_x": 20.0, "size_y": 20.0, "size_z": 0.22,
        },
        {
            "name": "office-drone-launch-pad", "semantic": "launch-pad",
            "center_x": -42.25, "center_y": 15.3, "center_z": 7.46,
            "radius_m": 0.85, "height_m": 0.08,
        },
        {
            "name": "roof", "semantic": "roof",
            "center_x": -42.25, "center_y": 15.3, "center_z": 10.95,
            "size_x": 20.0, "size_y": 20.0, "size_z": 0.2,
        },
    ]
    vehicle = VehicleAsset.model_validate(
        {
            **_vehicle().model_dump(mode="python"),
            "body_height_m": 0.43,
            "collision_center_offset_model_m": {"x": 0.0, "y": 0.0, "z": 0.228},
        }
    )

    root = resolve_grounded_vehicle_model_root(semantic, vehicle)

    assert (root.x, root.y, root.z) == pytest.approx((-42.25, 15.3, 7.487))


# 功能：
#   确认已废弃的地图内车辆几何字段不能进入当前运行绑定，避免覆盖所选车辆参数。
# 输入：
#   tmp_path：临时语义文件目录。
# 输出：
#   None：不返回业务数据。
def test_current_map_contract_rejects_retired_vehicle_geometry(tmp_path: Path) -> None:
    graph = _graph()
    route = GraphRoute(
        start_node="roof-launch",
        goal_node="dock-a",
        node_ids=["roof-launch", "dock-a"],
        edge_ids=["launch-to-dock"],
        positions_m=[graph.nodes[0].position_m, graph.nodes[1].position_m],
        route_length_m=graph.edges[0].distance_m,
        all_edges_flight_verified=False,
    )
    semantic = _semantic()
    runtime = semantic["runtime_bindings"]
    assert isinstance(runtime, dict)
    runtime["vehicle_collision_center_offset"] = {"x": 0.4, "y": -0.2, "z": 0.5}
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")

    with pytest.raises(ValueError, match="MAP_RUNTIME_BINDINGS_INVALID"):
        route_to_px4_track(route, graph, semantic_path, vehicle=_vehicle())


# 功能：
#   验证几何规划生成的中间点可导出，并按路线策略连续巡航、转弯减速。
# 输入：
#   tmp_path：临时语义文件目录。
# 输出：
#   None：不返回业务数据。
def test_metric_geometry_route_exports_with_continuous_policy_driven_speed(
    tmp_path: Path,
) -> None:
    graph = _graph()
    route = GraphRoute(
        start_node="roof-launch",
        goal_node="dock-a",
        node_ids=["roof-launch", "metric-safe-0001", "dock-a"],
        edge_ids=["metric-edge-safe-0000", "metric-edge-safe-0001"],
        positions_m=[
            graph.nodes[0].position_m,
            {"x": 13.5, "y": 23.0, "z": 5.5},
            graph.nodes[1].position_m,
        ],
        route_length_m=7.0,
        all_edges_flight_verified=False,
    )
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(json.dumps(_semantic()), encoding="utf-8")

    vehicle = _vehicle().model_copy(update={"max_speed_mps": 6.0})
    track = route_to_px4_track(route, graph, semantic_path, vehicle=vehicle)
    urgent = route_to_px4_track(
        route,
        graph,
        semantic_path,
        vehicle=vehicle,
        route_policy="minimum-time",
    )
    corner_route = route.model_copy(
        update={
            "positions_m": [
                graph.nodes[0].position_m,
                Vector3(x=16.0, y=22.0, z=8.0),
                graph.nodes[1].position_m,
            ]
        }
    )
    corner = route_to_px4_track(
        corner_route,
        graph,
        semantic_path,
        vehicle=vehicle,
    )

    assert [point.speed_limit_mps for point in track.points] == [5.0, 5.0, 5.0]
    assert track.stop_at_waypoints is False
    assert [point.target_speed_mps for point in track.points] == [1.5, 4.0, 0.5]
    assert [point.target_speed_mps for point in urgent.points] == [1.5, 5.0, 0.5]
    assert [point.target_speed_mps for point in corner.points] == [1.5, 1.8, 0.5]


def test_return_route_must_end_at_qualified_launch_waypoint(tmp_path: Path) -> None:
    graph = _graph()
    route = GraphRoute(
        start_node="roof-launch",
        goal_node="roof-launch",
        node_ids=["roof-launch", "dock-a", "roof-launch"],
        edge_ids=["launch-to-dock", "launch-to-dock"],
        positions_m=[
            graph.nodes[0].position_m,
            graph.nodes[1].position_m,
            {"x": 12.0, "y": 21.75, "z": 8.0},
        ],
        route_length_m=12.0,
        all_edges_flight_verified=False,
    )
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(json.dumps(_semantic()), encoding="utf-8")

    with pytest.raises(ValueError, match="does not end at the qualified PX4 launch"):
        route_to_px4_track(route, graph, semantic_path, vehicle=_vehicle())


# 功能：
#   普通未知边不因几何规划支持而获得默认速度或被默认为可执行。
# 输入：
#   tmp_path：临时语义文件目录。
# 输出：
#   None：不返回业务数据。
def test_non_metric_unknown_edge_still_fails_closed(tmp_path: Path) -> None:
    graph = _graph()
    route = GraphRoute(
        start_node="roof-launch",
        goal_node="dock-a",
        node_ids=["roof-launch", "dock-a"],
        edge_ids=["untrusted-generated-edge"],
        positions_m=[graph.nodes[0].position_m, graph.nodes[1].position_m],
        route_length_m=graph.edges[0].distance_m,
        all_edges_flight_verified=False,
    )
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(json.dumps(_semantic()), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown edge"):
        route_to_px4_track(route, graph, semantic_path, vehicle=_vehicle())


# 功能：
#   地图缺少显式运行绑定时拒绝使用，不自动填入历史学校地图坐标。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_generic_map_without_runtime_bindings_fails_closed() -> None:
    semantic = _semantic()
    semantic.pop("runtime_bindings")

    with pytest.raises(MapRuntimeBindingsError, match="^MAP_RUNTIME_BINDINGS_MISSING$"):
        load_map_runtime_bindings(semantic)


# 功能：
#   确认旧学校地图结构不能被当前绑定解析器当作可执行资产。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_retired_map_schema_cannot_enter_runtime_binding_resolution() -> None:
    with pytest.raises(
        MapRuntimeBindingsError,
        match="^MAP_SEMANTIC_SCHEMA_OBSOLETE$",
    ):
        load_map_runtime_bindings(
            {
                "schema_version": "dronedream.autonomy.school-map-semantic.v1",
                "simulation_bindings": {
                    "px4_recommended_spawn": {"x": 0, "y": 0, "z": 1},
                    "mission_launch_waypoint": {"x": 0, "y": 0, "z": 2},
                },
            }
        )


# 功能：
#   验证通用语义目录不能代替地图的实际仿真运行绑定。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_map_qualification_rejects_non_executable_generic_semantics() -> None:
    semantic = _semantic()
    semantic.pop("runtime_bindings")

    with pytest.raises(MapRuntimeBindingsError, match="MAP_RUNTIME_BINDINGS_MISSING"):
        load_map_runtime_bindings(semantic)


# 功能：
#   已知地图边限速高于本机能力时仍须受所选车辆限制，不能只约束几何生成的边。
# 输入：
#   tmp_path：临时语义文件目录。
# 输出：
#   None：不返回业务数据。
def test_named_route_edge_speed_is_capped_by_vehicle(tmp_path: Path) -> None:
    graph = _graph()
    route = GraphRoute(
        start_node="roof-launch", goal_node="dock-a", node_ids=["roof-launch", "dock-a"],
        edge_ids=["launch-to-dock"], positions_m=[node.position_m for node in graph.nodes],
        route_length_m=graph.edges[0].distance_m, all_edges_flight_verified=False,
    )
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(json.dumps(_semantic()), encoding="utf-8")
    vehicle = _vehicle().model_copy(update={"max_speed_mps": 0.7})
    track = route_to_px4_track(route, graph, semantic_path, vehicle=vehicle)
    assert all(point.speed_limit_mps <= 0.7 for point in track.points)


# 功能：
#   原始语义文件的重复字段和非有限数必须在投影前拒绝，不能被 JSON 默认解析掩盖。
# 输入：
#   tmp_path：独立文件目录。
#   raw：存在歧义或非有限值的原始 JSON。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'])
def test_track_semantic_reader_rejects_ambiguous_values(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "semantic.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        load_track_semantic(path)


# 功能：
#   构造标准运行期导出请求，直接检查坐标对象所有权和路线结构，不执行任何飞行。
# 输入：
#   tmp_path：独立测试资产目录。
# 输出：
#   fixture：运行请求与地图的二元组。
def _runtime_request(tmp_path: Path) -> tuple[RuntimeTrackRequest, MapAsset]:
    inputs = _replan_inputs(tmp_path)
    graph = inputs["graph"]
    route = GraphRoute(
        start_node="office", goal_node="guard-house", node_ids=["office", "guard-house"],
        edge_ids=["office-guard"], positions_m=[node.position_m for node in graph.nodes],
        route_length_m=5.0, all_edges_flight_verified=True,
    )
    request = RuntimeTrackRequest(
        route=route, prior_track=inputs["prior_track"], vehicle=inputs["vehicle"],
        target_node="guard-house",
    )
    fixture = request, graph
    return fixture


# 功能：
#   运行期导出结果的坐标合同必须独立，修改返回制品不能反向改写正在使用的原轨迹。
# 输入：
#   tmp_path：测试请求目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_track_export_detaches_coordinate_contract(tmp_path: Path) -> None:
    request, graph = _runtime_request(tmp_path)
    exported = runtime_route_to_px4_track(request, graph)
    exported.coordinate_contract.model_root_world_enu_m[0] = 123.0
    assert request.prior_track.coordinate_contract.model_root_world_enu_m[0] == 0.0


# 功能：
#   运行路线缺少边或有多余边时必须拒绝，不能悄悄使用默认边速或丢弃末尾数据。
# 输入：
#   tmp_path：测试请求目录。
#   edges：故意损坏的边列表。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("edges", [[], ["office-guard", "unused-edge"]])
def test_runtime_track_export_rejects_misaligned_edges(tmp_path: Path, edges: list[str]) -> None:
    request, graph = _runtime_request(tmp_path)
    request = request.model_copy(update={
        "route": request.route.model_copy(update={"edge_ids": edges})
    })
    with pytest.raises(ValueError, match="ROUTE_STRUCTURE"):
        runtime_route_to_px4_track(request, graph)
