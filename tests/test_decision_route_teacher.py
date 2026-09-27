"""Route guidance is a reference, not a localization measurement or a flight certificate."""

import copy
import json

import pytest
from test_decision_input_evidence import input_fixture

from dronedream_agent_core.contracts import GraphRoute
from dronedream_agent_core.decision_input_evidence import (
    input_algorithm_identity,
    verify_input_evidence,
)
from dronedream_agent_core.decision_route_teacher import RouteCollectionTeacher
from dronedream_agent_core.decision_state_adapter import decision_digest


# 功能：生成带折线的v2输入证明，原位置1/2/3保持为测量值；输出：纯合成测试夹具。
def fixture():
    _, proof = input_fixture()
    snapshot = proof["snapshots"][0]
    route = GraphRoute.model_validate(
        {
            "schema_version": "dronedream.graph-route.v1",
            "start_node": "start",
            "goal_node": "end",
            "node_ids": ["start", "turn", "end"],
            "edge_ids": ["a", "b"],
            "positions_m": [
                snapshot["current_position_m"],
                {"x": 5.0, "y": 2.0, "z": 3.0},
                {"x": 5.0, "y": 6.0, "z": 3.0},
            ],
            "route_length_m": 8.0,
            "all_edges_flight_verified": False,
        }
    ).model_dump(mode="json")
    snapshot["goal_position_m"] = route["positions_m"][-1]
    snapshot["known_static_map"]["qualified_route_sha256"] = decision_digest(route)
    snapshot.pop("snapshot_sha256")
    snapshot["snapshot_sha256"] = decision_digest(snapshot)
    proof.update(
        schema_version="dronedream.decision-input-evidence.v2",
        algorithm=input_algorithm_identity(2),
        qualified_route=route,
    )
    teacher = RouteCollectionTeacher(
        map_sha256=snapshot["known_static_map"]["source_sha256"],
        primitives=json.loads(proof["semantic_text"])["runtime_collision_primitives"],
        qualified_route=route,
        **proof["teacher"],
    )
    return teacher, proof


# 功能：直指终点会切角，沿折线应先向东；输入：折线路线；输出：方向正确且不改变原快照。
def test_guidance_uses_route_without_faking_pose():
    teacher, proof = fixture()
    raw = proof["snapshots"][0]
    original = copy.deepcopy(raw)
    state, check, _ = teacher.observe(raw, sequence=0, **proof["identity"])
    assert state.frame.goal_relative_body_frd_m.model_dump() == {"x": 0.75, "y": 0.0, "z": 0.0}
    assert state.frame.position_world_enu_m.model_dump() == original["current_position_m"]
    assert check["remaining_route_m"] == 8.0
    assert check["goal_world_enu_m"] == [5.0, 6.0, 3.0]
    assert raw == original
    rebuilt, _ = verify_input_evidence(state.model_dump(mode="json"), proof)
    assert rebuilt == state


# 功能：接近转角能继续转入下一段，不以无限接近顶点造成停止；偏离路线仍保留误差。
def test_turn_and_off_route_measurements_are_preserved():
    teacher, _ = fixture()
    target, deviation, remaining = teacher.route_target((4.9, 2.0, 3.0), (5.0, 6.0, 3.0))
    assert target == pytest.approx((5.0, 2.65, 3.0))
    assert remaining == pytest.approx(4.1)
    assert deviation == 0
    target, deviation, remaining = teacher.route_target((2.0, 1.0, 3.0), (5.0, 6.0, 3.0))
    assert deviation == 1.0
    assert target == pytest.approx((2.75, 2.0, 3.0))
    assert remaining == 7.0


# 功能：错路线、错目标、输入版本或变更配方均不可重建；输出：拒绝且不改变旧v1兼容。
@pytest.mark.parametrize("fault", ["route", "goal", "algorithm", "schema", "geometry"])
def test_reconstruction_rejects_contract_changes(fault):
    teacher, proof = fixture()
    raw = proof["snapshots"][0]
    state, _, _ = teacher.observe(raw, sequence=0, **proof["identity"])
    if fault == "route":
        proof["qualified_route"]["positions_m"][1]["x"] += 1
    elif fault == "goal":
        raw["goal_position_m"]["y"] += 1
        raw.pop("snapshot_sha256")
        raw["snapshot_sha256"] = decision_digest(raw)
    elif fault == "algorithm":
        proof["algorithm"] = input_algorithm_identity(1)
    elif fault == "schema":
        proof["schema_version"] = "dronedream.decision-input-evidence.v1"
    else:
        proof["semantic_text"] += " "
    with pytest.raises(ValueError):
        verify_input_evidence(state.model_dump(mode="json"), proof)


# 功能：新的输入版本必须保留旧正式数据的原算法身份；输出：v1仍独立重放通过。
def test_legacy_input_reconstruction_unchanged():
    state, proof = input_fixture()
    assert verify_input_evidence(state, proof)[0].model_dump(mode="json") == state
    assert set(input_algorithm_identity(2)) - set(input_algorithm_identity(1)) == {
        "decision_route_teacher.py"
    }


# 功能：同一物理边在往返时方向相反，不能因坐标相同选错任务段；输入：往返索引；输出：方向。
def test_roundtrip_binds_executor_waypoint_not_nearest_duplicate():
    teacher, _ = fixture()
    teacher.points = [(1., 2., 3.), (5., 2., 3.), (1., 2., 3.)]
    outbound = teacher.route_target((3., 2., 3.), (5., 2., 3.), waypoint_index=1)[0]
    inbound = teacher.route_target((3., 2., 3.), (1., 2., 3.), waypoint_index=2)[0]
    assert outbound == (3.75, 2., 3.)
    assert inbound == (2.25, 2., 3.)
    with pytest.raises(ValueError, match="WAYPOINT_BINDING"):
        teacher.route_target((3., 2., 3.), (1., 2., 3.), waypoint_index=1)
    with pytest.raises(ValueError, match="GOAL_NOT_ON_BOUND"):
        teacher.route_target((3., 2., 3.), (1., 2., 3.))
