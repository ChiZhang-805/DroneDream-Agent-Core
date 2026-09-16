from __future__ import annotations

import pytest

from dronedream_agent_core.contracts import (
    FlightPlan,
    GraphRoute,
    MapAsset,
    MapEdge,
    MapNode,
    MissionContract,
    PlannerContribution,
    PlannerValidation,
    PlanSegment,
    Px4CoordinateContract,
    Px4Track,
    Px4TrackPoint,
    RouteClearanceReport,
    RouteCollision,
    RoutePoint,
    SemanticPlan,
    TaskGraph,
    TaskNode,
    Vector3,
    VehicleAsset,
    WorldTrackPoint,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_api import build_discovered_extension_registry
from dronedream_agent_plugins.planning_specialists import _reachable, _validation


# 功能：
#   构造往返取件、机体、任务图和摘要绑定净空夹具，不执行实际几何搜索或飞行。
# 输入：
#   无。
# 输出：
#   fixtures：合同、地图、机体、任务图、路线、净空报告及轨迹的有序七元组。
def _fixtures() -> tuple[
    MissionContract,
    MapAsset,
    VehicleAsset,
    TaskGraph,
    GraphRoute,
    RouteClearanceReport,
    Px4Track,
]:
    graph = MapAsset(
        asset_id="map-a",
        name="Map",
        nodes=[
            MapNode(
                node_id="start",
                label="Start",
                position_m=Vector3(x=0, y=0, z=1),
                semantic="office",
            ),
            MapNode(
                node_id="target",
                label="Target",
                position_m=Vector3(x=2, y=0, z=1),
                semantic="pickup",
            ),
        ],
        edges=[
            MapEdge(
                edge_id="start-target",
                from_node="start",
                to_node="target",
                distance_m=2,
                minimum_clearance_m=1,
                speed_limit_mps=1,
                qualification="flight-verified",
                evidence_sha256="1" * 64,
            )
        ],
        named_entities={"start": "start", "target": "target"},
    )
    contract = MissionContract(
        contract_id="mission-" + "a" * 24,
        conversation_id="conversation-a",
        goal="Pick up and return",
        start_node="start",
        target_node="target",
        return_node="start",
        payload_action="pickup",
        map_asset_id=graph.asset_id,
        map_sha256="2" * 64,
        map_semantic_sha256="3" * 64,
        vehicle_asset_id="vehicle-a",
        vehicle_sha256="4" * 64,
        constraints=["simulation", "safety_priority"],
        immutable_safety_rules=["Unknown telemetry causes hold or abort."],
    )
    vehicle = VehicleAsset(
        asset_id="vehicle-a",
        name="Vehicle",
        dry_mass_kg=1,
        max_takeoff_mass_kg=2,
        body_radius_m=0.2,
        body_height_m=0.2,
        max_speed_mps=2,
        max_acceleration_mps2=2,
        reserve_battery_percent=20,
        qualified_range_m=100,
        max_pickup_payload_kg=0.5,
        sensors=["camera"],
    )
    task_graph = TaskGraph(
        nodes=[
            TaskNode(
                task_id="pickup",
                action="pickup",
                target_node="target",
                success_evidence=["payload attached"],
                fallback="hold",
            )
        ]
    )
    route = GraphRoute(
        start_node="start",
        goal_node="start",
        node_ids=["start", "target", "start"],
        edge_ids=["start-target", "start-target"],
        positions_m=[
            Vector3(x=0, y=0, z=1),
            Vector3(x=2, y=0, z=1),
            Vector3(x=0, y=0, z=1),
        ],
        route_length_m=4,
        all_edges_flight_verified=True,
    )
    clearance = RouteClearanceReport(
        accepted=True,
        route_sha256=sha256_json(route),
        semantic_sha256="6" * 64,
        sample_interval_m=0.1,
        sample_count=41,
        primitive_count=1,
        collision_count=0,
        minimum_clearance_m=1,
        minimum_clearance_point=Vector3(x=1, y=0, z=1),
        minimum_clearance_primitive="wall",
    )
    track = Px4Track(
        coordinate_contract=Px4CoordinateContract(
            model_root_world_enu_m=[0, 0, 0],
            collision_center_offset_model_m=[0.0, 0.0, 0.2],
        ),
        points=[
            Px4TrackPoint(x=0, y=0, z=0.8, phase="launch", speed_limit_mps=1),
            Px4TrackPoint(x=0, y=2, z=0.8, phase="pickup", speed_limit_mps=1),
            Px4TrackPoint(x=0, y=0, z=0.8, phase="land", speed_limit_mps=1),
        ],
        source_world_points=[
            WorldTrackPoint(east_m=0, north_m=0, up_m=1),
            WorldTrackPoint(east_m=2, north_m=0, up_m=1),
            WorldTrackPoint(east_m=0, north_m=0, up_m=1),
        ],
        waypoint_hold_seconds=0.2,
    )
    fixtures = (contract, graph, vehicle, task_graph, route, clearance, track)
    return fixtures


# 功能：
#   验证发现的十一类规划层均能产生类型化贡献及校验结果，不以此替代飞行验收。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_all_planning_layers_contribute_and_validate() -> None:
    registry = build_discovered_extension_registry()
    contract, graph, vehicle, task_graph, route, clearance, track = _fixtures()
    contributions, _ = registry.invoke_multiple(
        "planning.specialists",
        "contribute_planning",
        contract=contract,
        map_graph=graph,
        vehicle=vehicle,
    )
    validations, _ = registry.invoke_multiple(
        "planning.specialists",
        "validate_planning",
        contract=contract,
        map_graph=graph,
        vehicle=vehicle,
        task_graph=task_graph,
        semantic_plan=SemanticPlan(
            ordered_targets=["target", "start"], rationale_summary="Bound route"
        ),
        flight_plan=FlightPlan(
            revision=1,
            contract_id=contract.contract_id,
            semantic_plan_sha256="7" * 64,
            segments=[
                PlanSegment(
                    segment_id="segment-001",
                    task_id="pickup",
                    from_node="start",
                    to_node="target",
                    path=[
                        RoutePoint(node_id="start", position_m=Vector3(x=0, y=0, z=1)),
                        RoutePoint(node_id="target", position_m=Vector3(x=2, y=0, z=1)),
                    ],
                    speed_limit_mps=1,
                    minimum_clearance_m=1,
                    success_evidence=["target reached"],
                )
            ],
        ),
        route=route,
        clearance=clearance,
        px4_track=track,
    )

    parsed_contributions = [PlannerContribution.model_validate(value) for value in contributions]
    parsed_validations = [PlannerValidation.model_validate(value) for value in validations]
    assert {value.layer for value in parsed_contributions} == {
        "semantic",
        "temporal",
        "global",
        "local",
        "indoor",
        "outdoor",
        "dynamic-obstacle",
        "energy",
        "link",
        "payload",
        "regulatory",
    }
    assert len(parsed_validations) == 11
    assert all(value.accepted for value in parsed_validations)


# 功能：
#   验证航程边界接受微小浮点表示差异，但拒绝实际距离超限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_energy_specialist_tolerates_only_serialization_noise() -> None:
    contract, graph, vehicle, task_graph, route, clearance, track = _fixtures()

    serialized_equal = _validation(
        "energy",
        contract=contract,
        map_graph=graph,
        vehicle=vehicle.model_copy(update={"qualified_range_m": 4.0}),
        task_graph=task_graph,
        route=route.model_copy(update={"route_length_m": 4.000000000000001}),
        clearance=clearance,
        px4_track=track,
    )
    real_overrun = _validation(
        "energy",
        contract=contract,
        map_graph=graph,
        vehicle=vehicle.model_copy(update={"qualified_range_m": 4.0}),
        task_graph=task_graph,
        route=route.model_copy(update={"route_length_m": 4.0001}),
        clearance=clearance,
        px4_track=track,
    )

    assert serialized_equal.accepted is True
    assert real_overrun.accepted is False


# 功能：
#   在完整默认夹具上替换指定字段，单独测试某个边界而保留其他制品。
# 输入：
#   layer：需要校验的规划层。
#   overrides：替代默认制品或配置的关键字参数。
# 输出：
#   validation：当前规划层返回的实际校验结果。
def _validate_fixture(layer, **overrides):
    artifacts = dict(
        zip(
            ("contract", "map_graph", "vehicle", "task_graph", "route", "clearance", "px4_track"),
            _fixtures(),
            strict=True,
        )
    )
    validation = _validation(layer, **(artifacts | overrides))
    return validation


# 功能：
#   验证局部规划不能借用另一条路线的净空摘要作为当前候选的证明。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_local_specialist_rejects_clearance_from_another_route():
    report = _fixtures()[5]
    report.route_sha256 = "5" * 64
    result = _validate_fixture("local", clearance=report)
    assert not result.accepted
    assert not result.deterministic_gates["clearance_bound_to_route"]


# 功能：
#   验证直接调用钩子也不能绕过航程配置的有限性、范围和数值类型检查。
# 输入：
#   value：待拒绝的配置值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "100"])
def test_specialists_reject_bypassed_range_configuration(value):
    with pytest.raises(ValueError):
        _validate_fixture("energy", configuration={"qualified_range_m": value})


# 功能：
#   验证嵌套轨迹被改坏后仍重新校验，同时未知规划层不得落入默认成功分支。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_specialists_reject_mutated_nested_track_and_unknown_layer():
    track = _fixtures()[-1]
    track.points[0] = track.points[0].model_copy(update={"speed_limit_mps": float("nan")})
    with pytest.raises(ValueError):
        _validate_fixture("indoor", px4_track=track)
    with pytest.raises(ValueError, match="LAYER_UNKNOWN"):
        _validate_fixture("unknown")


# 功能：
#   验证取件动作必须发生在合同目标，而非仅存在任意一个 pickup 动作就通过。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_payload_specialist_requires_pickup_at_the_contract_target():
    graph = _fixtures()[3]
    graph.nodes[0].target_node = "start"
    assert not _validate_fixture("payload", task_graph=graph).accepted


# 功能：
#   验证两个同名但不存在的端点不能被当作零距离可达路线。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_graph_reachability_does_not_accept_same_unknown_endpoint():
    graph = _fixtures()[1]
    assert not _reachable(graph, "unknown", "unknown")


# 功能：
#   验证依赖净空的规划层拒绝自相矛盾或属于另一条路线的报告，不能只相信 accepted。
# 输入：
#   layer：使用净空判断的规划层。
#   mutation：本例损坏的报告内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("layer", ["local", "indoor", "outdoor", "dynamic-obstacle"])
@pytest.mark.parametrize("mutation", ["digest", "count", "distance", "collisions"])
def test_clearance_dependent_layers_require_consistent_evidence(layer, mutation):
    contract, graph, vehicle, tasks, route, report, track = _fixtures()
    graph.nodes[0].semantic = "outdoor" if layer == "outdoor" else "door"
    if mutation == "digest":
        report.route_sha256 = "9" * 64
    elif mutation == "count":
        report.collision_count = 1
    elif mutation == "distance":
        report.minimum_clearance_m = -0.1
    else:
        report.collisions = [
            RouteCollision(
                sample_index=0,
                position_m=Vector3(x=0, y=0, z=1),
                primitive_name="wall",
                clearance_m=-0.1,
            )
        ]
    result = _validation(
        layer,
        contract=contract,
        map_graph=graph,
        vehicle=vehicle,
        task_graph=tasks,
        route=route,
        clearance=report,
        px4_track=track,
    )
    assert not result.accepted
