import hashlib
import json
from pathlib import Path

import pytest

from dronedream_agent_core.collision import planning_collision_primitives, validate_route_clearance
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
#   验证软高度偏好确实影响求解路线，且简化不会抹掉爬升，取件端点仍保持原位置。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：路线进入室外偏好高度后回到低空目标，独立净空复核通过。
def test_preferred_airspace_guides_ascent_without_replacing_pickup_endpoint(tmp_path):
    graph = _graph()
    graph.nodes[1].position_m.x = 20.
    graph.edges[0].distance_m = 20.
    semantic = tmp_path / "outdoor.json"
    semantic.write_text(json.dumps({"coordinate_frame": "ENU", "collision_primitives": [{
        "center_x": 10., "center_y": 0., "center_z": -.1,
        "size_x": 40., "size_y": 20., "size_z": .2, "semantic": "terrain"}]}))
    planner = KnownMapMetricPlanner(graph=graph, semantic_path=semantic,
        vehicle_diameter_m=.4, vehicle_height_m=.4,
        policy=MetricPlannerPolicy(resolution_m=1., preferred_airspace_weight=3.,
                                   clearance_cost_weight=0., maximum_expansions=30000))
    route = planner.plan(RouteQuery(start_node="start", goal_node="goal"))
    assert max(p.z for p in route.positions_m) >= 5.
    assert route.positions_m[-1] == Vector3(x=20., y=0., z=1.)
    report = validate_route_clearance(route, semantic, vehicle_diameter_m=.4, vehicle_height_m=.4)
    assert report.minimum_clearance_m > .3


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


# 功能：
#   验证 Runtime 独有墙体既阻止直线净空放行，也改变真实 A* 路线；静态层仍保留。
# 输入：
#   tmp_path：隔离语义文件目录。
# 输出：
#   None：墙心被拒绝，求解绕行后独立验证通过。
def test_runtime_only_wall_is_hard_constraint_for_planner_and_clearance(tmp_path):
    path = _semantic(tmp_path / "runtime-wall.json")
    data = json.loads(path.read_text())
    data["runtime_collision_primitives"] = [data["collision_primitives"].pop()]
    path.write_text(json.dumps(data))
    planner = KnownMapMetricPlanner(graph=_graph(), semantic_path=path,
        vehicle_diameter_m=.4, vehicle_height_m=.3,
        policy=MetricPlannerPolicy(resolution_m=.4, required_clearance_m=.25))
    assert len(planner.primitives) == 2
    assert planner.clearance((3., 0., 1.)) < 0
    with pytest.raises(ValueError, match="endpoint connection"):
        planner._simplify([(0., 0., 1.), (6., 0., 1.)])
    route = planner.plan(RouteQuery(start_node="start", goal_node="goal"))
    assert len(route.positions_m) > 2
    report = validate_route_clearance(route, path, vehicle_diameter_m=.4, vehicle_height_m=.3)
    assert report.accepted and report.primitive_count == 2


# 功能：
#   验证损坏 Runtime 层不能被当成不存在而回退到静态层。
# 输入：
#   tmp_path、runtime：隔离文件目录和非法运行碰撞数据。
# 输出：
#   None：合并入口拒绝损坏的运行几何。
@pytest.mark.parametrize("runtime", [None, {}, [None], [True], [{"center_x": float("nan")}]])
def test_invalid_runtime_geometry_is_not_silently_ignored(tmp_path, runtime):
    data = json.loads(_semantic(tmp_path / "invalid-runtime.json").read_text())
    data["runtime_collision_primitives"] = runtime
    with pytest.raises(ValueError):
        planning_collision_primitives(data)


# 功能：
#   对随机坐标逐图元独立计算净空，检查空间桶饱和查询不会漏掉球形、倾斜或细长障碍。
# 输入：
#   tmp_path：隔离地图目录。
# 输出：
#   None：粗筛后的下界与完整几何的饱和值一致。
def test_indexed_clearance_matches_saturated_full_geometry(tmp_path):
    import random

    from dronedream_agent_core.collision import vehicle_clearance

    data = json.loads(_semantic(tmp_path / "geometry.json").read_text())
    data["runtime_collision_primitives"] = [
        {"center_x": 2., "center_y": 1., "center_z": 4., "radius_m": .4},
        {"center_x": -2., "center_y": 0., "center_z": 2., "radius_m": .2,
         "height_m": 3., "pitch_rad": .8},
        {"center_x": 0., "center_y": -2., "center_z": 3., "radius_m": .1,
         "length_m": 4., "roll_rad": 1.},
        {"center_x": 0., "center_y": 1., "center_z": 3.,
         "size_x": 3., "size_y": .2, "size_z": 1., "yaw_rad": .5, "roll_rad": .3}]
    path = tmp_path / "all-geometry.json"
    path.write_text(json.dumps(data))
    planner = KnownMapMetricPlanner(graph=None, semantic_path=path,
        vehicle_diameter_m=1.4, vehicle_height_m=1.4)
    shapes = planning_collision_primitives(data)
    randomizer = random.Random(531)
    for _ in range(500):
        point = tuple(randomizer.uniform(-6, 8) for _ in range(3))
        full = min(vehicle_clearance(point, shape, radius_m=.7, half_height_m=.7)
                   for shape in shapes)
        assert planner.clearance(point) == pytest.approx(min(full, planner._clearance_cap_m))
    with pytest.raises(TypeError):
        planner.primitives[0]["center_x"] = 99


# 功能：
#   检查连续净空证明能阻止位于旧固定采样间隔中央的薄墙，不以提速为由漏掉中途碰撞。
# 输入：
#   tmp_path：隔离语义目录。
# 输出：
#   None：两端安全但中间碰撞的线段被拒绝。
def test_continuous_segment_proof_rejects_thin_between_sample_wall(tmp_path):
    path = tmp_path / "thin-wall.json"
    path.write_text(json.dumps({"collision_primitives": [{"center_x": .025,
        "center_y": 0., "center_z": 10., "size_x": .001, "size_y": 1., "size_z": 1.}]}))
    planner = KnownMapMetricPlanner(graph=None, semantic_path=path,
        vehicle_diameter_m=.004, vehicle_height_m=.004,
        policy=MetricPlannerPolicy(required_clearance_m=0.))
    assert planner.clearance((0., 0., 10.)) > 0
    assert planner.clearance((.05, 0., 10.)) > 0
    assert not planner._segment_is_free((0., 0., 10.), (.05, 0., 10.))


# 功能：
#   确认开阔空间只需两次端点查询即可证明整段，而不是仍然密集采样。
# 输入：
#   tmp_path、monkeypatch：隔离地图及查询计数工具。
# 输出：
#   None：安全线段通过且只计算两次净空。
def test_open_segment_proof_avoids_redundant_samples(tmp_path, monkeypatch):
    planner = KnownMapMetricPlanner(graph=None, semantic_path=_semantic(tmp_path / "open.json"),
        vehicle_diameter_m=.4, vehicle_height_m=.3)
    original = planner.clearance
    calls = []

    # 功能：
    #   记录实际净空计算次数，仍返回真实算法结果。
    # 输入：
    #   point：本次待检查坐标。
    # 输出：
    #   result：真实净空下界。
    def counted(point):
        calls.append(point)
        result = original(point)
        return result

    monkeypatch.setattr(planner, "clearance", counted)
    assert planner._segment_is_free((0., 0., 10.), (.45, .45, 10.))
    assert len(calls) == 2


# 功能：
#   拒绝非有限、布尔及超大坐标，即使该位置没有空间索引候选也不能默认为安全。
# 输入：
#   tmp_path、point：隔离地图及无效坐标。
# 输出：
#   None：查询抛出明确的坐标校验异常。
@pytest.mark.parametrize("point", [(True, 0., 1.), (float("nan"), 0., 1.),
    (float("inf"), 0., 1.), pytest.param((10**1000, 0., 1.), id="oversized-coordinate")])
def test_clearance_rejects_invalid_points_before_index_lookup(tmp_path, point):
    path = _semantic(tmp_path / "bad-point.json")
    planner = KnownMapMetricPlanner(graph=None, semantic_path=path,
        vehicle_diameter_m=.4, vehicle_height_m=.3)
    with pytest.raises(ValueError, match="POINT_INVALID"):
        planner.clearance(point)


# 功能：
#   验证任意偏移的窄通道保留真实端点，不因网格相位或负 ENU 高度误判无路。
# 输入：
#   tmp_path、height_offset：隔离目录及地图相对 ENU 原点的高度偏移。
# 输出：
#   None：路线保持真实端点并在完整机身与相同安全余量下通过独立复核。
@pytest.mark.parametrize("height_offset", [0., -10.])
def test_exact_endpoint_axes_resolve_narrow_offset_passage(tmp_path, height_offset):
    graph = _graph()
    graph.nodes[0].position_m = Vector3(x=.13, y=0., z=1. + height_offset)
    graph.nodes[1].position_m = Vector3(x=.13, y=6., z=1. + height_offset)
    primitives = [{"center_x": .13, "center_y": 3., "center_z": height_offset - .1,
                   "size_x": 12., "size_y": 20., "size_z": .2}]
    primitives.extend({"center_x": .13 + side * .5, "center_y": 3.,
                       "center_z": height_offset + 4.,
                       "size_x": .2, "size_y": 20., "size_z": 8.} for side in (-1, 1))
    path = tmp_path / "narrow.json"
    path.write_text(json.dumps({"collision_primitives": primitives}))
    planner = KnownMapMetricPlanner(graph=graph, semantic_path=path,
        vehicle_diameter_m=.4, vehicle_height_m=.3,
        policy=MetricPlannerPolicy(required_clearance_m=.18, maximum_expansions=20000))
    route = planner.plan(RouteQuery(start_node="start", goal_node="goal"))
    assert route.positions_m[0] == graph.nodes[0].position_m
    assert route.positions_m[-1] == graph.nodes[1].position_m
    assert all(abs(p.x - .13) < .02 for p in route.positions_m)
    report = validate_route_clearance(route, path,
        vehicle_diameter_m=.4, vehicle_height_m=.3, sample_interval_m=.01)
    assert report.accepted and report.minimum_clearance_m > .18
