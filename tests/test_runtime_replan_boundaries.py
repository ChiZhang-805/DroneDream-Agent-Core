from __future__ import annotations

from pathlib import Path

import pytest
from test_runtime_replan import _execution_inputs, _replan_inputs

from dronedream_agent_core import runtime_replan
from dronedream_agent_core.contracts import CatalogEntity, RuntimeTrackRequest
from dronedream_agent_core.extensions import ExtensionRegistry
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.tools import ToolRegistry


# 功能：
#   为地点、限速和覆盖三个入口提供同源合法输入，分别检验共享门控确实接入各分支。
# 输入：
#   tmp_path：独立语义资产目录。
#   mode：待测重规划入口类别。
# 输出：
#   case：构建函数与对应关键字参数的二元组。
def _mode_inputs(tmp_path: Path, mode: str):
    inputs = _replan_inputs(tmp_path)
    builder = runtime_replan.build_runtime_replacement
    if mode != "destination":
        decision = inputs["decision"]
        classification = decision.classification.model_copy(update={
            "requested_action": "set_speed" if mode == "speed" else "set_coverage",
            "parameters": {"maximum_speed_mps": 0.3} if mode == "speed" else {
                "width_m": 2.0, "height_m": 2.0, "lane_spacing_m": 0.5,
                "boundary_margin_m": 0.2, "altitude_m": 1.0,
            },
        })
        inputs["decision"] = decision.model_copy(update={"classification": classification})
        if mode == "speed":
            builder = runtime_replan.build_runtime_speed_replacement
            inputs.pop("catalog")
            inputs["active_return_node"] = inputs.pop("return_node")
        else:
            builder = runtime_replan.build_runtime_coverage_replacement
    case = builder, inputs
    return case


# 功能：
#   在真实插件返回后注入错配证据，单独检验重规划消费端，而非声称注册器会生成坏数据。
# 输入：
#   tmp_path：独立测试资产目录。
#   monkeypatch：临时替换插件结果的夹具。
#   fault：错配净空或轨迹的具体位置。
#   mode：地点、限速或覆盖构建分支。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["route_hash", "semantic_hash", "collision", "world", "local"])
@pytest.mark.parametrize("mode", ["destination", "speed", "coverage"])
def test_replan_rejects_mismatched_tool_evidence(tmp_path, monkeypatch, fault, mode) -> None:
    builder, inputs = _mode_inputs(tmp_path, mode)
    original = ToolRegistry.call_slot

    # 功能：
    #   先执行正常工具再改写返回值，保留原回执，模拟消费边界收到不一致制品。
    # 输入：
    #   registry：当前工具注册器。
    #   slot_id：工具槽标识。
    #   request：原始类型化工具请求。
    # 输出：
    #   result：注入故障后的制品与原回执。
    def corrupt(registry, slot_id, request):
        value, receipt = original(registry, slot_id, request)
        if slot_id == "safety.route-clearance":
            field = {"route_hash": "route_sha256", "semantic_hash": "semantic_sha256"}.get(fault)
            if field:
                value = value.model_copy(update={field: "0" * 64})
            elif fault == "collision":
                value = value.model_copy(update={"collision_count": 1})
        elif slot_id == "runtime.track-export" and fault in {"world", "local"}:
            assert isinstance(request, RuntimeTrackRequest)
            value = value.model_copy(deep=True)
            # 不改第一点，以防只验证起点的实现意外通过这些对照。
            if fault == "world":
                value.source_world_points[1].east_m += 1.0
            else:
                value.points[1].y += 1.0
        result = value, receipt
        return result

    monkeypatch.setattr(ToolRegistry, "call_slot", corrupt)
    with pytest.raises(runtime_replan.RuntimeReplanError, match="GATE_FAILED"):
        builder(**inputs)


# 功能：
#   验证错误的原轨迹摘要或悬停身份不能通过重新计算决定摘要变成合法改令。
# 输入：
#   tmp_path：独立测试目录。
#   fault：本次破坏的输入绑定。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["prior_hash", "ack_message", "ack_execution", "empty_gates"])
def test_replan_rejects_input_binding_mismatch(tmp_path: Path, fault: str) -> None:
    inputs = _replan_inputs(tmp_path)
    ack = inputs["acknowledgement"]
    if fault == "prior_hash":
        inputs["prior_track_sha256"] = "0" * 64
    else:
        updates = {
            "ack_message": {"message_sha256": "0" * 64},
            "ack_execution": {"execution_id": "execution-" + "0" * 32},
            "empty_gates": {"deterministic_gates": {}},
        }
        ack = ack.model_copy(update=updates[fault])
        inputs["acknowledgement"] = ack
        inputs["decision"] = inputs["decision"].model_copy(
            update={"hold_ack_sha256": sha256_json(ack)}
        )
    with pytest.raises(runtime_replan.RuntimeReplanError, match="INPUT_BINDING"):
        runtime_replan.build_runtime_replacement(**inputs)


# 功能：
#   重名别名应拒绝猜测，字面图节点仍可精确指定；空白输入不能匹配空别名。
# 输入：
#   tmp_path：共享合法地图夹具的目录。
# 输出：
#   None：不返回业务数据。
def test_replan_alias_resolution_is_unambiguous(tmp_path: Path) -> None:
    inputs = _replan_inputs(tmp_path)
    graph, catalog = inputs["graph"], inputs["catalog"]
    first = catalog.entities[0].model_copy(update={"aliases": ["front desk", ""]})
    second = CatalogEntity(
        entity_id="office", aliases=["Front Desk"], position_m=graph.nodes[0].position_m,
        semantic="office", source_pointer="/office",
    )
    catalog = catalog.model_copy(update={"entities": [first, second]})
    with pytest.raises(runtime_replan.RuntimeReplanError, match="AMBIGUOUS"):
        runtime_replan._resolve_target("front desk", catalog, graph)
    with pytest.raises(runtime_replan.RuntimeReplanError, match="UNRESOLVED"):
        runtime_replan._resolve_target("   ", catalog, graph)
    assert runtime_replan._resolve_target("office", catalog, graph) == "office"


# 功能：
#   图别名引用已不存在的节点时明确拒绝，不能把失效地图引用继续交给路径工具。
# 输入：
#   tmp_path：测试图目录。
# 输出：
#   None：不返回业务数据。
def test_replan_rejects_dangling_named_entity(tmp_path: Path) -> None:
    inputs = _replan_inputs(tmp_path)
    graph = inputs["graph"].model_copy(update={"named_entities": {"lost": "removed-node"}})
    with pytest.raises(runtime_replan.RuntimeReplanError, match="UNRESOLVED"):
        runtime_replan._resolve_target("lost", inputs["catalog"], graph)


# 功能：
#   检验不能把部分缺失或摘要错配的活动执行合同重新绑定成看似完整的新合同。
# 输入：
#   fault：缺失或破坏的活动制品字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["graph_missing", "actions_missing", "graph_hash", "task_target"])
def test_execution_revision_rejects_inconsistent_source(fault: str) -> None:
    inputs = _execution_inputs()
    if fault == "graph_missing":
        inputs["task_graph"] = None
    elif fault == "actions_missing":
        inputs["prior_actions"] = None
    elif fault == "graph_hash":
        inputs["prior_actions"].task_graph_sha256 = "0" * 64
    else:
        inputs["prior_actions"].steps[0].target_node = "office"
    with pytest.raises(runtime_replan.RuntimeReplanError, match="SOURCE"):
        runtime_replan._revised_execution_artifacts(**inputs)


# 功能：
#   连续两次改令时，尚未执行的 post-replan 外设动作继续在采纳后触发，不移到目的地。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_execution_revision_preserves_post_replan_trigger() -> None:
    inputs = _execution_inputs()
    step = inputs["prior_actions"].steps[0].model_copy(
        update={"trigger": "post-replan", "checkpoint_id": None}
    )
    inputs["prior_actions"] = inputs["prior_actions"].model_copy(update={"steps": [step]})
    _, _, actions, superseded = runtime_replan._revised_execution_artifacts(**inputs)
    assert actions.steps[0].trigger == "post-replan"
    assert actions.steps[0].checkpoint_id is None
    assert superseded == [step.step_id]


# 功能：
#   替换轨迹不经过未完成动作的目的地时拒绝挂靠远处最近点，避免异地执行取件。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_execution_revision_rejects_action_target_outside_track() -> None:
    inputs = _execution_inputs()
    # 只把整条轨迹沿北向平移，不改地图目标，制造真实的“最近但未到达”情况。
    for world, local in zip(
        inputs["track"].source_world_points, inputs["track"].points, strict=True
    ):
        world.north_m += 20.0
        local.x += 20.0
    with pytest.raises(runtime_replan.RuntimeReplanError, match="TARGET_NOT_ON_TRACK"):
        runtime_replan._revised_execution_artifacts(**inputs)


# 功能：
#   地点和覆盖入口均须严格执行锚点策略，不能让布尔距离或未满足的已验证边要求通过。
# 输入：
#   tmp_path：独立地图目录。
#   monkeypatch：在合法策略输出后注入故障。
#   mode：地点或覆盖入口。
#   fault：非法距离或被忽略的锚点约束。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mode", ["destination", "coverage"])
@pytest.mark.parametrize("fault", ["boolean_distance", "unverified_anchor"])
def test_replan_anchor_policy_is_consistent(tmp_path, monkeypatch, mode, fault) -> None:
    builder, inputs = _mode_inputs(tmp_path, mode)
    if fault == "unverified_anchor":
        # 修改图后同步摘要，避免在别的绑定门控处失败而掩盖锚点检查。
        edge = inputs["graph"].edges[0].model_copy(update={"qualification": "geometry-derived"})
        inputs["graph"] = inputs["graph"].model_copy(update={"edges": [edge]})
        inputs["expected_map_sha256"] = sha256_json(inputs["graph"])
    original = ExtensionRegistry.invoke_single

    # 功能：
    #   保留其他扩展实际调用，只在锚点输出处替换策略字段。
    # 输入：
    #   registry：扩展注册器。
    #   slot_id：扩展槽。
    #   hook：钩子名称。
    #   kwargs：原始钩子上下文。
    # 输出：
    #   result：策略输出及原回执。
    def policy(registry, slot_id, hook, **kwargs):
        value, receipts = original(registry, slot_id, hook, **kwargs)
        if slot_id == "runtime.replan-policy":
            value = {**value, "requires_flight_verified_anchor": fault == "unverified_anchor"}
            if fault == "boolean_distance":
                value["maximum_join_distance_m"] = True
        result = value, receipts
        return result

    monkeypatch.setattr(ExtensionRegistry, "invoke_single", policy)
    with pytest.raises(
        runtime_replan.RuntimeReplanError, match="POLICY_INVALID|NOT_FLIGHT_VERIFIED"
    ):
        builder(**inputs)
