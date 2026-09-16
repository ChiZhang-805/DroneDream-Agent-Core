"""Planning scores cannot manufacture physical validity or bypass independent vetoes."""

from types import SimpleNamespace

import pytest
from test_planning_energy_gates import _route, _vehicle
from test_planning_specialists import _fixtures

from dronedream_agent_core.contracts import (
    FlightPlan,
    PlanSegment,
    RoutePoint,
    RuntimeCheckpoint,
    RuntimeCheckpointContract,
    SemanticPlan,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_plugins.planning_quality_plugins import (
    _clearance_speed_gate,
    _energy_reserve_gate,
    _payload_gate,
    _readiness_evaluation,
    _route_binding_gate,
    _stability_gate,
    _stability_score,
    _within_qualified_range,
)


# 功能：
#   验证负数、非有限值及错误类型不能通过航程比较或浮点容差获得通行。
# 输入：
#   value：分别作为所需航程和合格航程的非法值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True, "1", 10**400])
def test_invalid_range_never_passes_by_comparison_or_tolerance(value) -> None:
    assert not _within_qualified_range(value, 400)
    assert not _within_qualified_range(1, value)


# 功能：
#   验证非法能耗配置在收紧包络前被拒绝，不能被 min/max 合并掩盖。
# 输入：
#   policy：含非法航程或储备比例的配置字典。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "policy",
    [
        {"qualified_range_m": "400"},
        {"qualified_range_m": float("inf")},
        {"qualified_range_m": 1},
        {"reserve_fraction": True},
        {"reserve_fraction": 0.99},
    ],
)
def test_invalid_energy_policy_cannot_disappear_into_min_max(policy) -> None:
    with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
        _energy_reserve_gate(route=_route(1), vehicle=_vehicle(), configuration=policy)


# 功能：
#   验证绕过资产验证构造的百分之百储备被明确拒绝，避免航程换算除零。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_invalid_asset_reserve_cannot_divide_by_zero() -> None:
    vehicle = _vehicle().model_copy(update={"reserve_battery_percent": 100})
    with pytest.raises(ValueError, match="ENERGY_ASSET_ENVELOPE_INVALID"):
        _energy_reserve_gate(route=_route(1), vehicle=vehicle)


# 功能：
#   验证取件动作必须同时符合合同目标和载荷意图，不能夹带另一个地点的取件。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_additional_pickup_outside_contract_target_is_rejected() -> None:
    contract = SimpleNamespace(payload_action="pickup", target_node="cabinet")
    graph = SimpleNamespace(nodes=[SimpleNamespace(action="pickup", target_node="cabinet")])
    assert _payload_gate(contract=contract, task_graph=graph)["accepted"]
    graph.nodes.append(SimpleNamespace(action="pickup", target_node="other"))
    assert not _payload_gate(contract=contract, task_graph=graph)["accepted"]
    contract.payload_action = "none"
    assert not _payload_gate(contract=contract, task_graph=graph)["accepted"]


# 功能：
#   验证航点停车策略不能使空轨迹、非法速度或超限速度变得合格。
# 输入：
#   speeds：需要拒绝的航点速度列表。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("speeds", [[], [-1], [float("nan")], [True], [4]])
def test_stop_policy_does_not_make_bad_track_values_safe(speeds) -> None:
    track = SimpleNamespace(
        points=[SimpleNamespace(speed_limit_mps=s) for s in speeds], stop_at_waypoints=True
    )
    assert not _stability_gate(px4_track=track)["accepted"]


# 功能：
#   验证匀速差为零、大幅速度差不因平方溢出失真，且空轨迹不产生稳定性分数。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_stability_score_has_defined_constant_speed_and_overflow_behavior() -> None:
    track = SimpleNamespace(points=[SimpleNamespace(speed_limit_mps=s) for s in [1, 1]])
    assert _stability_score(px4_track=track)["value"] == 0
    track.points = [SimpleNamespace(speed_limit_mps=s) for s in [0, 1e200]]
    assert _stability_score(px4_track=track)["value"] == 1e200
    track.points = []
    with pytest.raises(ValueError, match="TRACK_SPEED_VALUES_INVALID"):
        _stability_score(px4_track=track)


# 功能：
#   验证空计划、非有限净空和非数值策略阈值不能通过净空速度门控。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_clearance_gate_rejects_empty_and_nonfinite_segments() -> None:
    plan = SimpleNamespace(segments=[])
    assert not _clearance_speed_gate(flight_plan=plan)["accepted"]
    plan.segments = [
        SimpleNamespace(segment_id="one", minimum_clearance_m=float("nan"), speed_limit_mps=0.5)
    ]
    assert not _clearance_speed_gate(flight_plan=plan)["accepted"]
    with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
        _clearance_speed_gate(flight_plan=plan, configuration={"tight_clearance_m": "2"})


# 功能：
#   验证空的错误类型或未知配置键不能被缺省合并吞掉，直接调用同样遵守设置契约。
# 输入：
#   policy：需要拒绝的配置。
#   gate：当前测试的能耗或净空速度门控。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("policy", [False, [], 0, "", {"misspelled_limit": 1}])
@pytest.mark.parametrize("gate", ["energy", "clearance"])
def test_falsey_or_unknown_policy_is_not_an_omitted_configuration(policy, gate):
    with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
        if gate == "energy":
            _energy_reserve_gate(route=_route(1), vehicle=_vehicle(), configuration=policy)
        else:
            _clearance_speed_gate(flight_plan=SimpleNamespace(segments=[]), configuration=policy)


# 功能：
#   验证准备门控不能接受其他任务的计划或自相矛盾的净空报告。
# 输入：
#   mutation：本例损坏的合同标识或净空内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["contract", "clearance", "accepted_type", "count_type"])
def test_route_binding_requires_plan_identity_and_consistent_clearance(mutation):
    contract, _, _, _, route, clearance, _ = _fixtures()
    semantic = SemanticPlan(ordered_targets=["target", "start"], rationale_summary="test")
    plan = FlightPlan(
        revision=1,
        contract_id=contract.contract_id,
        semantic_plan_sha256=sha256_json(semantic),
        segments=[
            PlanSegment(
                segment_id="segment-001",
                task_id="pickup",
                from_node=route.start_node,
                to_node=route.goal_node,
                path=[
                    RoutePoint(node_id=node, position_m=point)
                    for node, point in zip(route.node_ids, route.positions_m, strict=True)
                ],
                speed_limit_mps=1,
                minimum_clearance_m=1,
                success_evidence=["arrived"],
            )
        ],
    )
    if mutation == "contract":
        plan.contract_id = "other-contract"
    elif mutation == "clearance":
        clearance.collision_count = 1
    elif mutation == "accepted_type":
        clearance = clearance.model_copy(update={"accepted": "true"})
    else:
        clearance = clearance.model_copy(update={"collision_count": False})
    result = _route_binding_gate(
        contract=contract,
        semantic_plan=semantic,
        flight_plan=plan,
        route=route,
        clearance=clearance,
    )
    assert not result["accepted"]


# 功能：
#   验证就绪显示不能借用另一个任务的检查点，或忽略报告中已存在的碰撞。
# 输入：
#   mutation：本例损坏的检查点任务身份或净空报告。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["contract", "clearance", "accepted_type", "count_type"])
def test_readiness_does_not_mislabel_foreign_or_unsafe_artifacts(mutation):
    contract, _, _, _, route, clearance, _ = _fixtures()
    checkpoints = RuntimeCheckpointContract(
        contract_id=contract.contract_id,
        checkpoints=[
            RuntimeCheckpoint(
                checkpoint_id="checkpoint-001",
                segment_id="segment-001",
                task_id="pickup",
                track_point_index=1,
                target_node="target",
            )
        ],
    )
    if mutation == "contract":
        checkpoints.contract_id = "other-contract"
    elif mutation == "clearance":
        clearance.collision_count = 1
    elif mutation == "accepted_type":
        clearance = clearance.model_copy(update={"accepted": "true"})
    else:
        clearance = clearance.model_copy(update={"collision_count": False})
    result = _readiness_evaluation(
        contract=contract,
        route=route,
        clearance=clearance,
        runtime_checkpoints=checkpoints,
    )
    assert not result["ready"]
