import hashlib
import json
from pathlib import Path

import pytest

from dronedream_agent_core.collision import validate_route_clearance
from dronedream_agent_core.contracts import MapAsset, MapEdge, MapNode, RouteQuery, Vector3
from dronedream_agent_core.known_map_planner import KnownMapMetricPlanner, MetricPlannerPolicy


# 功能：
#   构造只有直连拓扑但需由碰撞规划另行绕障的最小命名图。
# 输入：
#   无。
# 输出：
#   graph：具有起点和取件目标的地图夹具。
def _graph() -> MapAsset:
    return MapAsset(
        asset_id="metric-planner-test",
        name="metric planner test",
        nodes=[
            MapNode(
                node_id="start", label="start", position_m=Vector3(x=0, y=0, z=1), semantic="launch"
            ),
            MapNode(
                node_id="goal", label="goal", position_m=Vector3(x=6, y=0, z=1), semantic="pickup"
            ),
        ],
        edges=[
            MapEdge(
                edge_id="fixed-direct",
                from_node="start",
                to_node="goal",
                distance_m=6,
                minimum_clearance_m=0,
                speed_limit_mps=0.5,
            )
        ],
        named_entities={"start": "start", "goal": "goal"},
    )


# 功能：
#   写入地面与挡路墙体，供测试区分真正几何绕障和盲从拓扑直线。
# 输入：
#   path：独立临时目录内的语义文件路径。
# 输出：
#   path：已写入的语义文件。
def _semantic(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "collision_primitives": [
                    {
                        "name": "ground",
                        "semantic": "floor",
                        "center_x": 3,
                        "center_y": 0,
                        "center_z": -0.1,
                        "size_x": 12,
                        "size_y": 12,
                        "size_z": 0.2,
                    },
                    {
                        "name": "blocking-wall",
                        "semantic": "wall",
                        "center_x": 3,
                        "center_y": 0,
                        "center_z": 1,
                        "size_x": 0.3,
                        "size_y": 3,
                        "size_z": 2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


# 功能：
#   用独立净空验收器验证路线实际绕过墙体，并保留未经飞行验证的标记。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_metric_planner_uses_collision_geometry_instead_of_fixed_graph(tmp_path: Path):
    semantic = _semantic(tmp_path / "semantic.json")
    planner = KnownMapMetricPlanner(
        graph=_graph(),
        semantic_path=semantic,
        vehicle_diameter_m=0.4,
        vehicle_height_m=0.3,
        policy=MetricPlannerPolicy(
            resolution_m=0.4,
            required_clearance_m=0.25,
            search_padding_m=2.5,
        ),
    )

    route = planner.plan(RouteQuery(start_node="start", goal_node="goal"))
    report = validate_route_clearance(
        route,
        semantic,
        vehicle_diameter_m=0.4,
        vehicle_height_m=0.3,
        sample_interval_m=0.05,
    )

    assert report.accepted
    assert report.minimum_clearance_m >= 0.25
    assert len(report.segment_minimum_clearances_m) == len(route.positions_m) - 1
    assert report.minimum_clearance_m == pytest.approx(
        min(report.segment_minimum_clearances_m)
    )
    assert route.route_length_m > 6
    assert any(abs(point.y) > 1.8 for point in route.positions_m)
    assert route.all_edges_flight_verified is False


# 功能：
#   验证起点落在墙内时明确拒绝，而不是将其吸附到安全的旧坐标。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_metric_planner_fails_closed_when_endpoint_has_no_operational_clearance(
    tmp_path: Path,
):
    semantic = _semantic(tmp_path / "semantic.json")
    graph = _graph()
    graph = graph.model_copy(
        update={
            "nodes": [
                graph.nodes[0].model_copy(update={"position_m": Vector3(x=3, y=0, z=1)}),
                graph.nodes[1],
            ]
        }
    )
    planner = KnownMapMetricPlanner(
        graph=graph,
        semantic_path=semantic,
        vehicle_diameter_m=0.4,
        vehicle_height_m=0.3,
        policy=MetricPlannerPolicy(required_clearance_m=0.25),
    )

    with pytest.raises(ValueError, match="start violates"):
        planner.plan(RouteQuery(start_node="start", goal_node="goal"))


# 功能：
#   验证实际起终点偏离命名节点时，路线仍保留权威位置及语义身份。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_metric_planner_uses_authoritative_runtime_positions(tmp_path: Path):
    semantic = _semantic(tmp_path / "semantic.json")
    planner = KnownMapMetricPlanner(
        graph=_graph(),
        semantic_path=semantic,
        vehicle_diameter_m=0.4,
        vehicle_height_m=0.3,
        policy=MetricPlannerPolicy(
            resolution_m=0.4,
            required_clearance_m=0.25,
            search_padding_m=2.5,
        ),
    )
    start = Vector3(x=0.0, y=-0.5, z=1.0)
    goal = Vector3(x=6.0, y=-0.5, z=1.0)

    route = planner.plan_positions(
        RouteQuery(start_node="start", goal_node="goal"),
        start_position_m=start,
        goal_position_m=goal,
    )

    assert route.positions_m[0] == start
    assert route.positions_m[-1] == goal
    assert route.node_ids[0] == "start"
    assert route.node_ids[-1] == "goal"


# 功能：
#   验证无额外走廊惩罚时可复用精确反向路线，且不重新发起搜索。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：阻止第二次搜索以核对缓存行为。
# 输出：
#   None：不返回业务数据。
def test_metric_planner_reuses_exact_reverse_route(tmp_path: Path, monkeypatch):
    semantic = _semantic(tmp_path / "semantic.json")
    planner = KnownMapMetricPlanner(
        graph=_graph(),
        semantic_path=semantic,
        vehicle_diameter_m=0.4,
        vehicle_height_m=0.3,
        policy=MetricPlannerPolicy(
            resolution_m=0.4,
            required_clearance_m=0.25,
            search_padding_m=2.5,
        ),
    )
    outbound = planner.plan(RouteQuery(start_node="start", goal_node="goal"))
    monkeypatch.setattr(
        planner,
        "_search_bounds",
        lambda *_args, **_kwargs: pytest.fail("reverse route should come from cache"),
    )

    inbound = planner.plan(RouteQuery(start_node="goal", goal_node="start"))

    assert inbound.positions_m == list(reversed(outbound.positions_m))
    assert inbound.route_length_m == outbound.route_length_m
    assert inbound.start_node == "goal"
    assert inbound.goal_node == "start"


# 功能：
#   验证碰撞图元与地图摘要来自同一次读取，文件随后变化不能使摘要绑定到另一版几何。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：在字节读取完成后改写文件，模拟并发更新。
# 输出：
#   None：不返回业务数据。
def test_geometry_and_hash_use_one_snapshot(tmp_path, monkeypatch):
    import dronedream_agent_core.known_map_planner as module

    path = _semantic(tmp_path / "semantic.json")
    original = path.read_bytes()
    real_read = module.read_plugin_file
    calls = []

    # 功能：
    #   返回完整旧快照后改变目标文件，暴露再次读取导致的身份错配。
    # 输入：
    #   source、limit：读取路径和字节预算。
    # 输出：
    #   data：原始文件内容。
    def change_after_read(source, *, limit):
        data = real_read(source, limit=limit)
        source.write_text('{"collision_primitives": []}', encoding="utf-8")
        calls.append(source)
        return data

    monkeypatch.setattr(module, "read_plugin_file", change_after_read)
    planner = KnownMapMetricPlanner(graph=None, semantic_path=path,
        vehicle_diameter_m=.4, vehicle_height_m=.3,
        expected_semantic_sha256=hashlib.sha256(original).hexdigest())
    assert calls == [path]
    assert len(planner.primitives) == 2
    assert planner.semantic_sha256 == hashlib.sha256(original).hexdigest()
    assert planner.semantic_sha256 != hashlib.sha256(path.read_bytes()).hexdigest()


# 功能：
#   验证损坏 JSON、重复键和与已批准摘要不符的地图在构建空间索引前拒绝。
# 输入：
#   tmp_path：独立测试目录。
#   content：无效或重复键的地图文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("content", ['{"collision_primitives": NaN}',
    '{"collision_primitives": [], "collision_primitives": []}', '[]'])
def test_invalid_semantic_snapshot_is_rejected(tmp_path, content):
    path = tmp_path / "semantic.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        KnownMapMetricPlanner(graph=None, semantic_path=path,
            vehicle_diameter_m=.4, vehicle_height_m=.3)
    _semantic(path)
    with pytest.raises(ValueError, match="hash mismatch"):
        KnownMapMetricPlanner(graph=None, semantic_path=path,
            vehicle_diameter_m=.4, vehicle_height_m=.3, expected_semantic_sha256="0" * 64)


# 功能：
#   验证非有限值、布尔和负代价不能进入 A* 配置，避免放松净空或破坏代价下界。
# 输入：
#   field、value：配置字段与非法数值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["required_clearance_m", "search_padding_m",
    "vertical_cost_multiplier", "clearance_cost_weight", "corridor_penalty_weight",
    "maximum_search_seconds"])
@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf"), -1.,
                                  pytest.param(10**1000, id="oversized-policy-number")])
def test_policy_rejects_invalid_real_values(field, value):
    with pytest.raises(ValueError):
        MetricPlannerPolicy(**{field: value})


# 功能：
#   验证有历史走廊惩罚时重新求解，不直接返回无惩罚的路线缓存。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：截获搜索入口以确认缓存未短路新的求解请求。
# 输出：
#   None：不返回业务数据。
def test_prior_corridor_request_bypasses_unpenalized_cache(tmp_path, monkeypatch):
    planner = KnownMapMetricPlanner(graph=_graph(), semantic_path=_semantic(tmp_path / "map.json"),
        vehicle_diameter_m=.4, vehicle_height_m=.3, policy=MetricPlannerPolicy(resolution_m=.4))
    query = RouteQuery(start_node="start", goal_node="goal")
    route = planner.plan(query)

    # 功能：
    #   用稳定异常证明本次调用已经穿过缓存进入搜索。
    # 输入：
    #   args、kwargs：搜索入口参数。
    # 输出：
    #   None：不返回业务数据。
    def observe_search(*args, **kwargs):
        raise RuntimeError("fresh search requested")

    monkeypatch.setattr(planner, "_search_bounds", observe_search)
    with pytest.raises(RuntimeError, match="fresh search requested"):
        planner.plan(query, prior_corridors=[[(point.x, point.y, point.z)
                                             for point in route.positions_m]])
    assert planner.plan(query) == route


# 功能：
#   验证仅两个端点的路线也检查中途障碍，不能因缺少中间点而跳过连接净空。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_two_point_simplification_checks_obstructed_connection(tmp_path):
    planner = KnownMapMetricPlanner(graph=None, semantic_path=_semantic(tmp_path / "map.json"),
        vehicle_diameter_m=.4, vehicle_height_m=.3)
    with pytest.raises(ValueError, match="endpoint connection"):
        planner._simplify([(0., 0., 1.), (6., 0., 1.)])
