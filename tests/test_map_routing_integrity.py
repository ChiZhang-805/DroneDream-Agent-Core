"""Map-routing boundary fixtures; these are not flight-qualification evidence."""

import hashlib

import pytest
from test_map_semantic_boundaries import _graph, _semantic, _write

import dronedream_agent_core.map_reasoning as reasoning
from dronedream_agent_core import assets, navigation
from dronedream_agent_core.contracts import MapAsset, MapEdge, MapNode, RouteQuery, Vector3


# 功能：
#   确认非法权重在寻路前被拒绝，不能破坏非负 Dijkstra 的终止和最优性前提。
# 输入：
#   cost：无穷、非数、负数、布尔、文本或极大整数。
# 输出：
#   无：错误权重未被拒绝时断言失败。
@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -1, True, "1", 10**400])
def test_invalid_routing_costs_fail_before_search(cost):
    with pytest.raises(ValueError, match="ROUTE_EDGE_COST_INVALID"):
        navigation._weighted_route(
            _graph(), RouteQuery(start_node="a", goal_node="b"), edge_cost=lambda edge: cost
        )


# 功能：
#   在代价回调中改写外部图和回调副本，确认本次寻路仍只使用冻结的输入。
# 输入：
#   无：函数创建独立临时图和查询。
# 输出：
#   无：返回位置、边及终点必须保持调用开始时的值。
def test_routing_callback_cannot_rewrite_frozen_inputs():
    graph = _graph()
    query = RouteQuery(start_node="a", goal_node="b")

    # 功能：
    #   模拟拥有回调的模块在执行期间误改图、查询和传入边。
    # 输入：
    #   edge：寻路器传入的边副本。
    # 输出：
    #   cost：合法代价，修改行为不应污染寻路元数据。
    def cost(edge):
        graph.nodes[1].position_m.x = 500
        query.goal_node = "a"
        edge.distance_m = 100
        edge.to_node = "missing"
        result = 1.0
        return result

    route = navigation._weighted_route(graph, query, edge_cost=cost)
    assert route.node_ids == ["a", "b"]
    assert route.positions_m[-1].x == 3
    assert route.route_length_m == 3
    route.positions_m[0].x = 999
    assert graph.nodes[0].position_m.x == 0


# 功能：
#   确认构造后写入的字符串坐标和安全开关不能被重新校验时自动纠正。
# 输入：
#   tmp_path：测试语义文件目录。
# 输出：
#   无：寻路、目录绑定、实体解析均应拒绝改坏的模型。
def test_mutated_map_values_are_not_coerced(tmp_path):
    graph = _graph()
    catalog = assets.load_map_catalog(_write(tmp_path, _semantic()))
    # 常规赋值会先被 Pydantic 转换；直接破坏内部值才覆盖外部反序列化绕过验证的情况。
    graph.nodes[0].position_m.__dict__["x"] = "0"
    with pytest.raises(ValueError):
        navigation.shortest_route(graph, RouteQuery(start_node="a", goal_node="b"))
    with pytest.raises(ValueError):
        assets.resolve_map_entity("start", catalog, graph)
    with pytest.raises(ValueError):
        assets.load_map_catalog(_write(tmp_path, _semantic()), qualified_graph=graph)
    query = RouteQuery(start_node="a", goal_node="b")
    query.__dict__["require_flight_verified_edges"] = "false"
    with pytest.raises(ValueError):
        navigation.shortest_route(_graph(), query)


# 功能：
#   验证只有 flight-verified 标签但没有证据散列的边不能被声称已验证。
# 输入：
#   无：本地未带证据的图夹具。
# 输出：
#   无：普通候选降为未验证，只用已验证边时不可达。
def test_verified_label_without_evidence_is_not_qualification():
    graph = _graph()
    graph.edges[0].qualification = "flight-verified"
    assert not navigation.shortest_route(
        graph, RouteQuery(start_node="a", goal_node="b")
    ).all_edges_flight_verified
    with pytest.raises(ValueError, match="qualification"):
        navigation.shortest_route(
            graph, RouteQuery(start_node="a", goal_node="b", require_flight_verified_edges=True)
        )


# 功能：
#   覆盖原地零移动候选，不能对空边集合求最小值或编造无限净空。
# 输入：
#   无：本地拓扑夹具。
# 输出：
#   无：零长候选应返回未知净空且不声称已验证飞行。
def test_zero_movement_has_no_fabricated_clearance():
    candidates = reasoning._route_alternatives(_graph(), start_node="a", goal_node="a")
    assert len(candidates) == 1
    assert candidates[0]["node_ids"] == ["a"]
    assert candidates[0]["minimum_clearance_m"] is None
    assert candidates[0]["minimum_speed_limit_mps"] is None
    assert candidates[0]["all_edges_flight_verified"] is False


# 功能：
#   验证当前目录绑定后文件被替换时，环境摘要不接受另一个文件的内容。
# 输入：
#   tmp_path：测试文件目录。
# 输出：
#   无：摘要应明确报告散列不匹配，不能输出伪环境字段。
def test_semantic_context_is_bound_to_catalog_bytes(tmp_path):
    path = _write(tmp_path, _semantic())
    catalog = assets.load_map_catalog(path)
    changed = _semantic()
    changed["dynamic_people"] = {"runtime_spawn_required": True}
    _write(tmp_path, changed)
    context = reasoning.build_map_reasoning_context(_graph(), catalog, path)
    assert context["semantic_environment"] == {
        "available": False,
        "issue_codes": ["MAP_SEMANTIC_CONTEXT_HASH_MISMATCH"],
    }


# 功能：
#   拒绝会被 Python 真值转换伪装成有效环境的错误类型。
# 输入：
#   tmp_path：语义夹具目录。
#   field：待破坏的环境字段。
#   value：该字段的无效值。
# 输出：
#   无：摘要不可用且不输出碰撞或准备就绪假结论。
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("navigation_layers", {"occupancy_ready": "false"}),
        ("dynamic_people", {"runtime_spawn_required": 1}),
        ("collision_primitives", None),
        ("runtime_collision_primitives", "walls"),
        ("known_export_limits", [None]),
    ],
)
def test_semantic_context_rejects_coerced_environment_fields(tmp_path, field, value):
    semantic = _semantic()
    semantic[field] = value
    path = _write(tmp_path, semantic)
    summary = reasoning._semantic_summary(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert summary == {"available": False, "issue_codes": ["MAP_SEMANTIC_CONTEXT_INVALID"]}


# 功能：
#   缺少传感器层声明时保持未知，不能把没有读到字段等同于已经测量为假。
# 输入：
#   tmp_path：测试文件目录。
# 输出：
#   无：未声明的环境状态必须保持 None。
def test_absent_environment_fields_remain_unknown(tmp_path):
    path = _write(tmp_path, _semantic())
    summary = reasoning._semantic_summary(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert summary["available"] is True
    assert summary["occupancy_ready"] is None
    assert summary["dynamic_obstacles_runtime_required"] is None


# 功能：
#   构造长链以验证预算限制，不引入学校地图、真实任务或外部模型。
# 输入：
#   count：节点数。
# 输出：
#   graph：显式单链测试图。
def _chain(count):
    graph = MapAsset(
        asset_id="chain",
        name="Bounded routing fixture",
        nodes=[
            MapNode(
                node_id=f"n{i}",
                label=f"Node {i}",
                position_m=Vector3(x=i, y=0, z=2),
                semantic="corridor",
            )
            for i in range(count)
        ],
        edges=[
            MapEdge(
                edge_id=f"e{i}",
                from_node=f"n{i}",
                to_node=f"n{i + 1}",
                distance_m=1,
                minimum_clearance_m=1,
                speed_limit_mps=1,
            )
            for i in range(count - 1)
        ],
        named_entities={"start": "n0", "target": f"n{count - 1}"},
    )
    return graph


# 功能：
#   验证尝试上限也作用于单条长路线的内部循环，不等到整个循环结束才检查。
# 输入：
#   monkeypatch：替换寻路调用以精确计数。
# 输出：
#   无：初次寻路加候选调用不得超过一个加固定预算。
def test_alternative_budget_applies_inside_long_route(monkeypatch):
    graph = _chain(180)
    first = navigation.shortest_route(graph, RouteQuery(start_node="n0", goal_node="n179"))
    calls = []

    # 功能：
    #   返回首条真实算出的路线，再模拟移除任意边后不连通。
    # 输入：
    #   graph：该次候选图。
    #   query：候选查询。
    # 输出：
    #   first：首轮路线，后续明确抛出不可达异常。
    def counted(graph, query):
        calls.append((len(graph.edges), query.goal_node))
        if len(calls) > 1:
            raise ValueError("disconnected fixture")
        return first

    monkeypatch.setattr(reasoning, "shortest_route", counted)
    candidates = reasoning._route_alternatives(graph, start_node="n0", goal_node="n179")
    assert len(calls) == 1 + reasoning.MAX_ROUTE_ATTEMPTS
    assert candidates[0]["route_details_truncated"] is True
    assert len(candidates[0]["node_ids"]) == reasoning.MAX_INCLUDED_NODES
    assert candidates[0]["complete_route_node_count"] == 180
    assert candidates[0]["unverified_edge_count"] == 179


# 功能：
#   验证上下文容量受限时任务尾部节点仍优先保留，未知目的地不能被静默忽略。
# 输入：
#   tmp_path：目录文件位置。
# 输出：
#   无：尾部任务节点必须在摘要中，非法目的地应拒绝。
def test_task_nodes_have_priority_over_graph_storage_order(tmp_path):
    path = _write(tmp_path, _semantic())
    catalog = assets.load_map_catalog(path)
    context = reasoning.build_map_reasoning_context(
        _chain(180), catalog, path, focus_nodes=["n178", "n179"]
    )
    assert {"n178", "n179"}.issubset({node["node_id"] for node in context["nodes"]})
    with pytest.raises(ValueError, match="MAP_FOCUS_NODES_INVALID"):
        reasoning.build_map_reasoning_context(
            _graph(), catalog, path, focus_nodes=["retired-place"]
        )


# 功能：
#   核对用户限制数量的上界，不静默丢掉最后一条限制以腾出默认告警。
# 输入：
#   tmp_path：临时语义文件目录。
# 输出：
#   无：有图时保留全部限制，无图且容量不足时明确拒绝。
def test_topology_warning_does_not_silently_drop_declared_limits(tmp_path):
    semantic = _semantic()
    semantic["known_limits"] = [f"limit {i}" for i in range(64)]
    path = _write(tmp_path, semantic)
    assert len(assets.load_map_catalog(path, qualified_graph=_graph()).known_limits) == 64
    with pytest.raises(ValueError, match="LIMITS_CAPACITY_EXCEEDED"):
        assets.load_map_catalog(path)
