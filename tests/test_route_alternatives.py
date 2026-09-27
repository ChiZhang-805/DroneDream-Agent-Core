from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dronedream_agent_core.contracts import (
    GraphRoute,
    PlanCritique,
    RouteAlternativeCandidate,
    RouteAlternativeSet,
    RouteClearanceReport,
    ToolReceipt,
    Vector3,
)
from dronedream_agent_core.hashing import canonical_json, sha256_json
from dronedream_agent_core.orchestrator import (
    _collect_failed_exploratory_tool_ids,
    _discard_unsupported_metric_edge_history_gate,
    _normalize_plan_critique,
)
from dronedream_agent_plugins import route_alternative_plugins
from dronedream_agent_plugins.route_alternative_plugins import (
    _planning_clearance_requirement_m,
    plugin_definitions,
)


# 功能：
#   创建仅用于排序的合成路线；声明长度与历史验证标记不表示实际碰撞几何已验收。
# 输入：
#   length：本例声明的路线长度。
#   verified：本例声明的历史边验证状态。
# 输出：
#   route：三节点往返路线夹具。
def _route(*, length: float, verified: bool) -> GraphRoute:
    route = GraphRoute(
        start_node="start",
        goal_node="return",
        node_ids=["start", "target", "return"],
        edge_ids=["out", "back"],
        positions_m=[
            Vector3(x=0, y=0, z=1),
            Vector3(x=1, y=0, z=1),
            Vector3(x=0, y=0, z=1),
        ],
        route_length_m=length,
        all_edges_flight_verified=verified,
    )
    return route


# 功能：
#   将合成净空回执绑定到本例路线，并准备一致的指标、硬门控与失败原因。
# 输入：
#   candidate_id：候选标识。
#   length：声明路线长度。
#   clearance：声明最小净空。
#   energy：能耗代理分数。
#   verified：历史边验证标记。
#   feasible：本例候选是否通过合成硬门控。
# 输出：
#   candidate：带路线及摘要绑定回执的排序输入夹具。
def _candidate(
    candidate_id: str,
    *,
    length: float,
    clearance: float,
    energy: float,
    verified: bool = True,
    feasible: bool = True,
) -> RouteAlternativeCandidate:
    route = _route(length=length, verified=verified)
    report = RouteClearanceReport(
        accepted=feasible,
        route_sha256=sha256_json(route),
        semantic_sha256="b" * 64,
        sample_interval_m=0.1,
        sample_count=10,
        primitive_count=1,
        collision_count=0 if feasible else 1,
        minimum_clearance_m=clearance,
        minimum_clearance_point=Vector3(x=0, y=0, z=1),
        minimum_clearance_primitive="wall",
    )
    candidate = RouteAlternativeCandidate(
        alternative_id=candidate_id,
        strategy_tool_id=f"strategy.{candidate_id}",
        route=route,
        clearance=report,
        objectives={
            "distance_m": length,
            "minimum_clearance_m": clearance,
            "energy_proxy": energy,
            "transition_count": 2.0,
            "qualification_penalty": 0.0 if verified else 1.0,
        },
        hard_gates={"continuous_clearance": feasible},
        feasible=feasible,
        issue_codes=[] if feasible else ["CONTINUOUS_CLEARANCE_REJECTED"],
    )
    return candidate


# 功能：
#   构造探索工具历史判定所需的回执身份，不调用任何真实工具。
# 输入：
#   call_hex：组成测试调用标识的十六进制字符。
#   tool_id：工具标识。
#   outcome：合成执行结果状态。
# 输出：
#   receipt：本例工具回执。
def _receipt(call_hex: str, tool_id: str, outcome: str) -> ToolReceipt:
    receipt = ToolReceipt(
        call_id=f"tool-{call_hex * 24}",
        tool_id=tool_id,
        tool_version="1.0.0",
        outcome=outcome,
        input_sha256="1" * 64,
        output_sha256="2" * 64,
        output={},
    )
    return receipt


# 功能：
#   验证探索失败工具列表去重并稳定排序，不误收已选择成功的调用且可直接编码为 JSON。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_exploratory_tool_ids_are_stable_and_json_safe() -> None:
    receipts = [
        _receipt("a", "planning.zeta", "failed"),
        _receipt("b", "planning.alpha", "rejected"),
        _receipt("c", "planning.selected", "accepted"),
        _receipt("d", "planning.zeta", "failed"),
    ]

    result = _collect_failed_exploratory_tool_ids(receipts, {receipts[2].call_id})

    assert result == ["planning.alpha", "planning.zeta"]
    assert canonical_json({"failed_tool_ids": result})


# 功能：
#   验证更短、更省能但碰撞的候选仍被排除，并为真正选中的候选给出排序解释。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_multi_objective_ranker_excludes_infeasible_route_and_explains_choice():
    definition = next(
        value
        for value in plugin_definitions()
        if value.manifest.plugin_id == "planning.multi-objective-ranker"
    )
    assert definition.tool_factory is not None
    tool = definition.tool_factory(
        SimpleNamespace(plugin_configuration=None)  # type: ignore[arg-type]
    )[0]
    values = RouteAlternativeSet(
        contract_id="contract-1",
        candidates=[
            _candidate("short-narrow", length=10, clearance=0.4, energy=11),
            _candidate("wide-safe", length=12, clearance=1.8, energy=12.5),
            _candidate("collision", length=5, clearance=-0.1, energy=5, feasible=False),
        ],
        objective_weights={
            "distance_m": 0.20,
            "minimum_clearance_m": 0.46,
            "energy_proxy": 0.14,
            "transition_count": 0.10,
            "qualification_penalty": 0.10,
        },
    )
    decision = tool.handler(values)
    assert decision.selected_alternative_id == "wide-safe"
    assert "collision" not in decision.ranked_alternative_ids
    assert decision.rejected_alternatives == {"collision": ["CONTINUOUS_CLEARANCE_REJECTED"]}
    assert any("综合得分" in reason for reason in decision.selection_reasons)


# 功能：
#   验证小机体使用最低净空，大机体使用随半径增长的余量。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_metric_candidate_matches_vehicle_scaled_operational_gate():
    assert _planning_clearance_requirement_m(0.76) == pytest.approx(0.35)
    assert _planning_clearance_requirement_m(1.20) == pytest.approx(0.45)


# 功能：
#   验证规划器构造时使用与保守验证一致的扩张机体包络，而非只扩大一端。
# 输入：
#   monkeypatch：pytest 属性替换工具。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("phase,budget", [("initial", 60.), ("runtime", 8.)])
def test_metric_candidate_searches_with_the_conservative_validation_envelope(
    monkeypatch: pytest.MonkeyPatch, phase: str, budget: float,
):
    captured: dict[str, object] = {}

    class PlannerProbe:
        # 功能：
        #   记录规划器构造参数，不加载地图或运行搜索。
        # 输入：
        #   kwargs：插件传给规划器的全部构造参数。
        # 输出：
        #   None：不返回业务数据。
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        # 功能：
        #   提供符合工具接口的合成路线回调。
        # 输入：
        #   _query：本夹具不使用的路线查询。
        # 输出：
        #   route：声明为未历史验证的测试路线。
        def plan(self, _query: object) -> GraphRoute:
            route = _route(length=2.0, verified=False)
            return route

    monkeypatch.setattr(
        route_alternative_plugins,
        "KnownMapMetricPlanner",
        PlannerProbe,
    )
    definition = next(
        value
        for value in plugin_definitions()
        if value.manifest.plugin_id == "planning.candidate-metric-geometry"
    )
    assert definition.tool_factory is not None
    tools = definition.tool_factory(
        SimpleNamespace(
            map_graph=object(),
            semantic_path=Path("semantic.json"),
            vehicle_diameter_m=0.76,
            vehicle_height_m=0.43,
            planning_phase=phase,
        )  # type: ignore[arg-type]
    )

    assert len(tools) == 1
    assert captured["vehicle_diameter_m"] == pytest.approx(0.912)
    assert captured["vehicle_height_m"] == pytest.approx(0.516)
    policy = captured["policy"]
    assert policy.required_clearance_m == pytest.approx(0.35)  # type: ignore[union-attr]
    assert policy.maximum_search_seconds == budget


# 功能：
#   验证未使用历史图边本身不能否决已经具有独立净空证据的度量候选。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_metric_route_history_label_is_not_promoted_to_a_hard_gate():
    candidate = _candidate(
        "metric-safe",
        length=12,
        clearance=0.5,
        energy=13,
        verified=False,
    ).model_copy(
        update={
            "strategy_tool_id": "planning.candidate-metric-geometry.candidate",
        }
    )
    review = PlanCritique(
        accepted=False,
        issue_codes=["EXEC_ROUTE_NOT_ALL_EDGES_FLIGHT_VERIFIED"],
        repair_instructions=["Use a historical graph edge."],
    )

    normalized = _discard_unsupported_metric_edge_history_gate(review, candidate)

    assert normalized.accepted is True
    assert normalized.issue_codes == []


# 功能：
#   验证未选探索策略失败和未来证据尚未产生，不会在前置条件满足时误否决已选计划。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_exploratory_route_does_not_reject_selected_plan():
    candidate = _candidate(
        "selected-safe",
        length=12,
        clearance=0.5,
        energy=13,
    )
    review = PlanCritique(
        accepted=False,
        issue_codes=[
            "ALL_TOOL_RECEIPTS_NOT_ACCEPTED",
            "METRIC_GEOMETRY_CANDIDATE_FAILED",
            "FUTURE_RUNTIME_EVIDENCE_NOT_HASH_BOUND",
        ],
        repair_instructions=["Make every explored route strategy succeed."],
    )

    normalized = _normalize_plan_critique(
        review,
        candidate,
        selected_plan_receipts_accepted=True,
        future_runtime_evidence_declared=True,
        failed_exploratory_tool_ids={
            "planning.candidate-metric-geometry.candidate",
        },
    )

    assert normalized.accepted is True
    assert normalized.issue_codes == []
    assert normalized.repair_instructions == []


# 功能：
#   验证清除不成立的探索门控时仍保留真实目标顺序问题及其修复要求。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_real_plan_issue_survives_unsupported_gate_normalization():
    candidate = _candidate(
        "selected-safe",
        length=12,
        clearance=0.5,
        energy=13,
    )
    review = PlanCritique(
        accepted=False,
        issue_codes=[
            "ALL_TOOL_RECEIPTS_NOT_ACCEPTED",
            "SEMANTIC_TARGET_ORDER_INVALID",
        ],
        repair_instructions=["Restore the required target order."],
    )

    normalized = _normalize_plan_critique(
        review,
        candidate,
        selected_plan_receipts_accepted=True,
        future_runtime_evidence_declared=True,
        failed_exploratory_tool_ids={
            "planning.candidate-metric-geometry.candidate",
        },
    )

    assert normalized.accepted is False
    assert normalized.issue_codes == ["SEMANTIC_TARGET_ORDER_INVALID"]
    assert normalized.repair_instructions == ["Restore the required target order."]


# 功能：
#   验证选中路线自身的生成失败不能以“探索失败不影响结果”为由被抹除。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_selected_metric_route_is_not_normalized_away():
    candidate = _candidate(
        "selected-metric",
        length=12,
        clearance=0.5,
        energy=13,
    ).model_copy(update={"strategy_tool_id": "planning.candidate-metric-geometry.candidate"})
    review = PlanCritique(
        accepted=False,
        issue_codes=["CANDIDATE_METRIC_GEOMETRY_TOOL_FAILED"],
        repair_instructions=["Repair the selected route generator."],
    )

    normalized = _normalize_plan_critique(
        review,
        candidate,
        selected_plan_receipts_accepted=False,
        future_runtime_evidence_declared=False,
        failed_exploratory_tool_ids={
            "planning.candidate-metric-geometry.candidate",
        },
    )

    assert normalized == review


# 功能：
#   直接调用纯排序函数，不启动几何进程或模型提供方。
# 输入：
#   candidates：合成候选列表。
#   configuration：可选插件级权重配置。
#   objective_weights：可选请求级目标权重。
# 输出：
#   decision：实际排序函数返回的选择结果。
def _rank(candidates, configuration=None, objective_weights=None):
    tool = route_alternative_plugins._ranker_tools(
        SimpleNamespace(plugin_configuration=configuration)
    )[0]
    decision = tool.handler(
        RouteAlternativeSet(
            contract_id="test", candidates=candidates, objective_weights=objective_weights or {}
        )
    )
    return decision


# 功能：
#   验证伪造可行标记无法绕过路线摘要、硬门控、碰撞计数或净空接受状态检查。
# 输入：
#   mutation：本例损坏的可行性证据字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["digest", "hard_gate", "collisions", "accepted"])
def test_ranker_rejects_false_feasibility_claim(mutation):
    candidate = _candidate("safe", length=10, clearance=1, energy=11)
    if mutation == "digest":
        candidate.clearance.route_sha256 = "0" * 64
    elif mutation == "hard_gate":
        candidate.hard_gates["energy"] = False
    elif mutation == "collisions":
        candidate.clearance.collision_count = 1
    else:
        candidate.clearance.accepted = False
    with pytest.raises(ValueError, match="FEASIBILITY_EVIDENCE_INVALID"):
        _rank([candidate])


# 功能：
#   验证权重拒绝越界、布尔、文本和非有限值，不靠宽松转换改变实际评分。
# 输入：
#   value：待拒绝的插件配置权重。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [-1, 2, True, "0.5", float("nan")])
def test_ranker_rejects_invalid_configuration_weights(value):
    with pytest.raises(ValueError):
        _rank(
            [_candidate("safe", length=10, clearance=1, energy=11)],
            configuration={"distance_weight": value},
        )


# 功能：
#   验证权重总和被规范化，同时重复候选身份必须拒绝而非覆盖分数字典。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_ranker_normalizes_weight_sum_and_rejects_duplicate_ids():
    candidate = _candidate("safe", length=10, clearance=1, energy=11)
    result = _rank([candidate], objective_weights={metric: 1.0 for metric in candidate.objectives})
    assert result.normalized_scores["safe"] == 1
    with pytest.raises(ValueError, match="IDENTIFIERS_DUPLICATE"):
        _rank([candidate, candidate])


# 功能：
#   验证缺少必需指标在评分前明确拒绝，不在生成部分结果后出现索引异常。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_ranker_rejects_missing_metric_instead_of_crashing_mid_ranking():
    candidate = _candidate("safe", length=10, clearance=1, energy=11)
    del candidate.objectives["energy_proxy"]
    with pytest.raises(ValueError, match="METRICS_INVALID"):
        _rank([candidate])


# 功能：
#   验证排序指标必须对应绑定的路线及净空报告，不能独立改高分数误导路线选择。
# 输入：
#   metric：本例故意改写的可由路线或报告直接核对的指标。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "metric", ["distance_m", "minimum_clearance_m", "transition_count", "qualification_penalty"]
)
def test_ranker_rejects_metrics_detached_from_route_evidence(metric):
    candidate = _candidate("safe", length=10, clearance=1, energy=11)
    candidate.objectives[metric] += 2.0
    with pytest.raises(ValueError, match="ROUTE_ALTERNATIVE_METRIC_EVIDENCE_MISMATCH"):
        _rank([candidate])
