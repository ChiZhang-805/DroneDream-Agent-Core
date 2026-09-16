from __future__ import annotations

from pathlib import Path

import pytest

from dronedream_agent_core.contracts import (
    FlightPlan,
    MapAsset,
    MissionContract,
    PlanSegment,
    Px4Track,
    RoutePoint,
    RouteQuery,
    Vector3,
)
from dronedream_agent_core.plugin_api import ToolEnvironment
from dronedream_agent_plugins.checkpoint_policies import (
    _mission_boundary_checkpoints,
    _risk_adaptive_checkpoints,
    _segment_checkpoints,
    _turn_angle_degrees,
)
from dronedream_agent_plugins.harness_transform_plugins import (
    _corner_speed_envelope,
    _telemetry_waypoint_settle,
)
from dronedream_agent_plugins.planning_quality_plugins import _clearance_speed_gate
from dronedream_agent_plugins.route_alternative_plugins import plugin_definitions as route_plugins
from dronedream_agent_plugins.runtime_replan_policies import (
    _mission_continuity_anchor,
    _nearest_anchor,
    _verified_anchor,
)


# 功能：
#   为策略测试构造合成任务合同，不使用生产地图、账户或飞行授权。
# 输入：
#   无。
# 输出：
#   contract：包含出发、目标和返程节点的测试合同。
def _contract() -> MissionContract:
    digest = "a" * 64
    contract = MissionContract(
        contract_id="mission-0123456789abcdef01234567",
        conversation_id="differentiation-test",
        goal="fly through the school map and return",
        start_node="start",
        target_node="target",
        return_node="return",
        payload_action="navigate",
        map_asset_id="school-map",
        map_sha256=digest,
        map_semantic_sha256=digest,
        vehicle_asset_id="my-drone",
        vehicle_sha256=digest,
        constraints=[],
        immutable_safety_rules=["continuous collision clearance"],
    )
    return contract


# 功能：
#   创建带死端、短窄路线和较长宽裕路线的地图，使不同规划策略可产生可区分结果。
# 输入：
#   无。
# 输出：
#   graph：仅用于本地策略测试的拓扑地图。
def _graph() -> MapAsset:
    graph = MapAsset.model_validate(
        {
            "asset_id": "test.school-map",
            "name": "Differentiation graph",
            "nodes": [
                {
                    "node_id": "start",
                    "label": "Start",
                    "position_m": {"x": 0, "y": 0, "z": 1},
                    "semantic": "launch",
                },
                {
                    "node_id": "short",
                    "label": "Short",
                    "position_m": {"x": 1, "y": 0, "z": 1},
                    "semantic": "corridor",
                },
                {
                    "node_id": "wide-a",
                    "label": "Wide A",
                    "position_m": {"x": 0, "y": 2, "z": 1},
                    "semantic": "corridor",
                },
                {
                    "node_id": "wide-b",
                    "label": "Wide B",
                    "position_m": {"x": 2, "y": 2, "z": 1},
                    "semantic": "corridor",
                },
                {
                    "node_id": "target",
                    "label": "Target",
                    "position_m": {"x": 2, "y": 0, "z": 1},
                    "semantic": "pickup",
                },
                {
                    "node_id": "return",
                    "label": "Return",
                    "position_m": {"x": 4, "y": 0, "z": 1},
                    "semantic": "office",
                },
                {
                    "node_id": "dead",
                    "label": "Dead end",
                    "position_m": {"x": -1, "y": 0, "z": 1},
                    "semantic": "outdoor",
                },
            ],
            "edges": [
                {
                    "edge_id": "short-1",
                    "from_node": "start",
                    "to_node": "short",
                    "distance_m": 1,
                    "minimum_clearance_m": 0.35,
                    "speed_limit_mps": 1,
                },
                {
                    "edge_id": "short-2",
                    "from_node": "short",
                    "to_node": "target",
                    "distance_m": 1,
                    "minimum_clearance_m": 0.35,
                    "speed_limit_mps": 1,
                },
                {
                    "edge_id": "wide-1",
                    "from_node": "start",
                    "to_node": "wide-a",
                    "distance_m": 2,
                    "minimum_clearance_m": 2.5,
                    "speed_limit_mps": 1,
                    "qualification": "flight-verified",
                },
                {
                    "edge_id": "wide-2",
                    "from_node": "wide-a",
                    "to_node": "wide-b",
                    "distance_m": 2,
                    "minimum_clearance_m": 2.5,
                    "speed_limit_mps": 1,
                    "qualification": "flight-verified",
                },
                {
                    "edge_id": "wide-3",
                    "from_node": "wide-b",
                    "to_node": "target",
                    "distance_m": 2,
                    "minimum_clearance_m": 2.5,
                    "speed_limit_mps": 1,
                    "qualification": "flight-verified",
                },
                {
                    "edge_id": "target-return",
                    "from_node": "target",
                    "to_node": "return",
                    "distance_m": 2,
                    "minimum_clearance_m": 2,
                    "speed_limit_mps": 1,
                },
            ],
            "named_entities": {"start": "start", "target": "target", "return": "return"},
        }
    )
    return graph


# 功能：
#   构造含直角转弯的单段计划，使风险检查点和限速整形的效果可直接断言。
# 输入：
#   clearance_m：本例声明的航段净空。
#   speed_mps：本例声明的航段速度上限。
# 输出：
#   plan：绑定测试合同的直角路线计划。
def _flight_plan(*, clearance_m: float = 0.8, speed_mps: float = 1.4) -> FlightPlan:
    path = [
        RoutePoint(node_id="start", position_m=Vector3(x=0, y=0, z=1)),
        RoutePoint(node_id="corner-a", position_m=Vector3(x=1, y=0, z=1)),
        RoutePoint(node_id="corner-b", position_m=Vector3(x=1, y=1, z=1)),
        RoutePoint(node_id="target", position_m=Vector3(x=2, y=1, z=1)),
    ]
    plan = FlightPlan(
        revision=1,
        contract_id=_contract().contract_id,
        segments=[
            PlanSegment(
                segment_id="segment-001",
                task_id="navigate-target",
                from_node="start",
                to_node="target",
                path=path,
                speed_limit_mps=speed_mps,
                minimum_clearance_m=clearance_m,
                success_evidence=["target reached"],
            )
        ],
        semantic_plan_sha256="b" * 64,
    )
    return plan


# 功能：
#   验证距离策略确实选择短路，并与净空优先策略产生不同的路线长度及说明。
# 输入：
#   tmp_path：pytest 独立临时目录，用于构造未读取的语义路径。
# 输出：
#   None：不返回业务数据。
def test_distance_candidate_is_a_distinct_shortest_path_baseline(tmp_path: Path) -> None:
    definitions = {value.manifest.plugin_id: value for value in route_plugins()}
    distance = definitions["planning.candidate-distance"]
    clearance = definitions["planning.candidate-clearance"]
    environment = ToolEnvironment(
        map_graph=_graph(),
        semantic_path=tmp_path / "semantic.json",
        vehicle_diameter_m=0.6,
        vehicle_height_m=0.4,
        waypoint_hold_seconds=0.4,
    )
    query = RouteQuery(start_node="start", goal_node="target")

    assert distance.tool_factory is not None and clearance.tool_factory is not None
    shortest = distance.tool_factory(environment)[0].handler(query)
    widest = clearance.tool_factory(environment)[0].handler(query)

    assert shortest.node_ids == ["start", "short", "target"]
    assert shortest.route_length_m < widest.route_length_m
    assert distance.manifest.description != clearance.manifest.description


# 功能：
#   验证风险策略覆盖转弯和狭窄内部位置，同时仍保留任务终点检查。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_risk_adaptive_checkpoints_cover_turns_and_tight_mid_segment_geometry() -> None:
    checkpoints = _risk_adaptive_checkpoints(
        contract=_contract(),
        flight_plan=_flight_plan(),
        configuration={
            "minimum_turn_degrees": 45,
            "tight_clearance_m": 1.25,
            "maximum_internal_checkpoints_per_segment": 2,
        },
    )

    assert [value.track_point_index for value in checkpoints.checkpoints] == [1, 2, 3]
    assert checkpoints.checkpoints[-1].target_node == "target"


# 功能：
#   验证狭窄且过快的组合被拒绝，而相同净空下符合限速的计划通过。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_clearance_speed_gate_rejects_only_the_tight_and_fast_combination() -> None:
    rejected = _clearance_speed_gate(
        flight_plan=_flight_plan(clearance_m=0.8, speed_mps=1.4),
        configuration={"tight_clearance_m": 1.25, "maximum_tight_speed_mps": 1.0},
    )
    accepted = _clearance_speed_gate(
        flight_plan=_flight_plan(clearance_m=0.8, speed_mps=0.8),
        configuration={"tight_clearance_m": 1.25, "maximum_tight_speed_mps": 1.0},
    )

    assert rejected["accepted"] is False
    assert rejected["violations"] == [
        {"segment_id": "segment-001", "minimum_clearance_m": 0.8, "speed_limit_mps": 1.4}
    ]
    assert accepted["accepted"] is True


# 功能：
#   验证任务连续性策略跳过最近死端，选择能继续到目标并返程的节点。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_continuity_anchor_ignores_the_nearest_unreachable_dead_end() -> None:
    selected = _mission_continuity_anchor(
        current_world=Vector3(x=-1, y=0, z=1),
        graph=_graph(),
        target_node="target",
        return_node="return",
        configuration={"maximum_join_distance_m": 3.0, "target_route_weight": 0.35},
    )

    assert selected["anchor_node"] == "start"
    assert selected["target_and_return_reachable"] is True


# 功能：
#   验证转弯限速会在直角处进一步降低速度，但不无故修改两端的速度上限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_corner_speed_envelope_brakes_below_qualified_indoor_edge_speed() -> None:
    track = Px4Track.model_validate(
        {
            "coordinate_contract": {
                "model_root_world_enu_m": [0, 0, 0],
                "collision_center_offset_model_m": [0.0, 0.0, 0.228],
            },
            "points": [
                {"x": 0, "y": 0, "z": 1, "phase": "launch", "speed_limit_mps": 0.55},
                {"x": 1, "y": 0, "z": 1, "phase": "transit", "speed_limit_mps": 0.55},
                {"x": 1, "y": 1, "z": 1, "phase": "pickup", "speed_limit_mps": 0.55},
            ],
            "source_world_points": [
                {"east_m": 0, "north_m": 0, "up_m": 1},
                {"east_m": 0, "north_m": 1, "up_m": 1},
                {"east_m": 1, "north_m": 1, "up_m": 1},
            ],
            "stop_at_waypoints": True,
            "waypoint_hold_seconds": 0.4,
        }
    )

    optimized = _corner_speed_envelope(value=track)

    assert optimized.points[1].speed_limit_mps == 0.3
    assert optimized.points[0].speed_limit_mps == 0.55
    assert optimized.points[2].speed_limit_mps == 0.55


# 功能：
#   验证遥测稳定契约修改停稳要求及容差，不把稳定时间设置混为路径限速修改。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_telemetry_settle_contract_is_distinct_from_speed_shaping() -> None:
    track = Px4Track.model_validate(
        {
            "coordinate_contract": {
                "model_root_world_enu_m": [0, 0, 0],
                "collision_center_offset_model_m": [0.0, 0.0, 0.228],
            },
            "points": [
                {"x": 0, "y": 0, "z": 1, "phase": "launch", "speed_limit_mps": 0.55},
                {"x": 1, "y": 0, "z": 1, "phase": "pickup", "speed_limit_mps": 0.55},
            ],
            "source_world_points": [
                {"east_m": 0, "north_m": 0, "up_m": 1},
                {"east_m": 0, "north_m": 1, "up_m": 1},
            ],
            "stop_at_waypoints": False,
            "waypoint_hold_seconds": 0.0,
        }
    )

    optimized = _telemetry_waypoint_settle(
        value=track,
        configuration={
            "position_tolerance_m": 0.18,
            "speed_tolerance_mps": 0.12,
            "stable_window_seconds": 0.7,
            "settle_timeout_seconds": 15.0,
        },
    )

    assert optimized.stop_at_waypoints is True
    assert optimized.waypoint_position_tolerance_m == 0.18
    assert optimized.waypoint_speed_tolerance_mps == 0.12
    assert optimized.waypoint_stable_window_seconds == 0.7
    assert optimized.waypoint_settle_timeout_seconds == 15.0
    assert [point.speed_limit_mps for point in optimized.points] == [0.55, 0.55]


# 功能：
#   构造先通行、再到目标、最后返程的三段连续计划，覆盖目标不在第一段的情况。
# 输入：
#   无。
# 输出：
#   plan：具有共享端点的多航段测试计划。
def _multi_segment_plan() -> FlightPlan:
    plan = _flight_plan()
    nodes = ["start", "corridor", "target", "return"]
    points = [
        RoutePoint(node_id=node, position_m=Vector3(x=i, y=0, z=1)) for i, node in enumerate(nodes)
    ]
    plan.segments = [
        PlanSegment(
            segment_id=f"segment-{i + 1:03d}",
            task_id=f"task-{i}",
            from_node=nodes[i],
            to_node=nodes[i + 1],
            path=points[i : i + 2],
            speed_limit_mps=0.5,
            minimum_clearance_m=1,
            success_evidence=["arrival"],
        )
        for i in range(3)
    ]
    return plan


# 功能：
#   验证任务边界策略识别实际目标与返程节点，并给出与整条轨迹一致的索引。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_boundary_checkpoints_include_actual_target_after_transit_segment():
    result = _mission_boundary_checkpoints(contract=_contract(), flight_plan=_multi_segment_plan())
    assert [item.target_node for item in result.checkpoints] == ["target", "return"]
    assert [item.track_point_index for item in result.checkpoints] == [2, 3]


# 功能：
#   验证三种检查点策略均拒绝错误合同及同名却不同位置的共享端点。
# 输入：
#   hook：本例检查的检查点策略函数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "hook", [_mission_boundary_checkpoints, _segment_checkpoints, _risk_adaptive_checkpoints]
)
def test_checkpoint_policies_reject_wrong_contract_or_discontinuous_shared_point(hook):
    plan = _multi_segment_plan()
    plan.contract_id = "another-mission"
    with pytest.raises(ValueError, match="CONTRACT_MISMATCH"):
        hook(contract=_contract(), flight_plan=plan)
    plan = _multi_segment_plan()
    plan.segments[1].path[0] = plan.segments[1].path[0].model_copy(deep=True)
    plan.segments[1].path[0].position_m.x += 1
    with pytest.raises(ValueError, match="PATH_DISCONTINUITY"):
        hook(contract=_contract(), flight_plan=plan)


# 功能：
#   验证绕过设置界面直接调用时，风险策略仍拒绝错误类型、非有限或越界配置。
# 输入：
#   configuration：待拒绝的风险检查点配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "configuration",
    [
        {"long_segment_points": True},
        {"long_segment_points": 3.5},
        {"maximum_internal_checkpoints_per_segment": "3"},
        {"tight_clearance_m": float("nan")},
        {"minimum_turn_degrees": -1},
        [],
    ],
)
def test_risk_checkpoint_configuration_is_strict_even_without_settings_ui(configuration):
    with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
        _risk_adaptive_checkpoints(
            contract=_contract(), flight_plan=_flight_plan(), configuration=configuration
        )


# 功能：
#   验证大但有限的坐标仍能用归一化向量得到直角，不因中间乘方溢出而失真。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_turn_angle_avoids_overflow_for_large_finite_coordinates():
    points = [
        RoutePoint(node_id=str(i), position_m=Vector3(x=x, y=y, z=0))
        for i, (x, y) in enumerate([(0, 0), (1e200, 0), (1e200, 1e200)])
    ]
    assert _turn_angle_degrees(*points) == pytest.approx(90.0)


# 功能：
#   验证近邻及历史验证锚点策略自身执行半径硬限制，不只把配置值留给后续模块。
# 输入：
#   hook：本例需要检查的锚点选择函数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("hook", [_nearest_anchor, _verified_anchor])
def test_anchor_selector_itself_enforces_join_radius(hook):
    with pytest.raises(ValueError, match="OUTSIDE_JOIN_RADIUS"):
        hook(current_world=Vector3(x=100, y=0, z=1), graph=_graph(), configuration={})


# 功能：
#   验证在线连续性策略拒绝错误数值或文本布尔值，不通过强制转换改变用户限制。
# 输入：
#   configuration：待拒绝的在线换路配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "configuration",
    [
        {"maximum_join_distance_m": True},
        {"maximum_join_distance_m": 0},
        {"maximum_join_distance_m": float("inf")},
        {"target_route_weight": "0.35"},
        {"require_flight_verified_edges": "false"},
    ],
)
def test_continuity_selector_rejects_coerced_policy(configuration):
    with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
        _mission_continuity_anchor(
            current_world=Vector3(x=0, y=0, z=1),
            graph=_graph(),
            target_node="target",
            return_node="return",
            configuration=configuration,
        )


# 功能：
#   验证所有候选共享的目标至返程搜索只做一次，同时仍执行真实本地拓扑搜索。
# 输入：
#   monkeypatch：pytest 属性替换工具。
# 输出：
#   None：不返回业务数据。
def test_continuity_checks_fixed_return_leg_only_once(monkeypatch):
    from dronedream_agent_plugins import runtime_replan_policies as policies

    original = policies.shortest_route
    calls = []

    # 功能：
    #   记录查询端点后转发给真实搜索器，不用伪造可行路线替代算法。
    # 输入：
    #   graph：测试拓扑地图。
    #   query：本次有向路线查询。
    # 输出：
    #   route：实际本地搜索返回的路线。
    def record(graph, query):
        calls.append((query.start_node, query.goal_node))
        route = original(graph, query)
        return route

    monkeypatch.setattr(policies, "shortest_route", record)
    policies._mission_continuity_anchor(
        current_world=Vector3(x=0, y=0, z=1),
        graph=_graph(),
        target_node="target",
        return_node="return",
        configuration={},
    )
    assert calls.count(("target", "return")) == 1
