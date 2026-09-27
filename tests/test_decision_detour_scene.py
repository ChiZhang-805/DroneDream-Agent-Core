"""Static detour preparation tests are not physical replan demonstrations."""

import pytest

from dronedream_agent_core.contracts import GraphRoute
from dronedream_agent_core.decision_detour_scene import (
    detour_documents,
    graph_family,
    shortest_path,
    verify_detour_geometry,
)
from dronedream_agent_core.decision_scene_topology import generate_edges


# 功能：给树增加一个此前不存在的相邻通道；输入：种子；输出：有真实环路的材料。
def fixture(seed=0):
    recipe = {"size": 4, "edges": generate_edges(4, seed), "cell_m": 4.0, "flight_z_m": 1.6}
    existing = {tuple(e) for e in recipe["edges"]}
    extra = next(
        [a, b]
        for a in range(16)
        for b in range(a + 1, 16)
        if abs(a % 4 - b % 4) + abs(a // 4 - b // 4) == 1 and (a, b) not in existing
    )
    return recipe, extra, detour_documents(recipe, [extra], {})


# 功能：多个布局都必须存在不穿挡板的真实替代路径；输出：旧路碰撞、新路净空且正式计数0。
@pytest.mark.parametrize("seed", range(20))
def test_detour_geometry_and_routes(seed):
    recipe, extra, doc = fixture(seed)
    before = GraphRoute.model_validate(doc["initial_route"])
    after = GraphRoute.model_validate(doc["replacement_route"])
    assert after.goal_node == before.goal_node
    assert after.route_length_m > before.route_length_m
    assert doc["blocked_edge"] in doc["edges"]
    report = verify_detour_geometry(doc, radius_m=0.3, height_m=0.3, clearance_m=0.25)
    assert report["initial_route_clearance_m"] <= 0
    assert report["replacement_route_clearance_m"] >= 0.25
    assert not report["physically_executed"] and report["formal_training_additions"] == 0
    assert doc["blocked"]["collision_primitives"] == doc["blocked"]["runtime_collision_primitives"]
    assert not any(p["name"] == "closed-passage" for p in doc["before"]["collision_primitives"])


# 功能：树边阻断没有替代路径，不能假装每个地图都能重规划；输出：None。
def test_tree_cannot_detour_without_a_loop():
    recipe, _, _ = fixture()
    edge = recipe["edges"][0]
    assert shortest_path(4, recipe["edges"], *edge, blocked=[edge]) is None


# 功能：变换房间编号不能把相同布局放进不同集合；输出：保守图桶一致。
def test_graph_family_is_invariant_under_rotation():
    _, _, doc = fixture()

    def rotate(n):
        return (n % 4) * 4 + 3 - n // 4

    rotated = [[rotate(a), rotate(b)] for a, b in doc["edges"]]
    assert graph_family(4, rotated) == doc["family"]


# 功能：非法开放边不能修改几何；输入：重复、跨格、布尔等；输出：拒绝。
@pytest.mark.parametrize("fault", ["existing", "repeat", "long", "bool", "empty"])
def test_bad_extra_edges_rejected(fault):
    recipe, extra, _ = fixture()
    values = {
        "existing": [recipe["edges"][0]],
        "repeat": [extra, extra],
        "long": [[0, 15]],
        "bool": [[False, 1]],
        "empty": [],
    }
    with pytest.raises(ValueError, match="EXTRA_EDGES_INVALID"):
        detour_documents(recipe, values[fault], {})
