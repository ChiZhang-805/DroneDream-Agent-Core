"""Compiler-only tests: visual configuration must not claim unsupported execution semantics."""

import pytest

from dronedream_agent_core.model_harness.design import (
    HarnessBoundedLoop,
    HarnessEdgeBinding,
    HarnessEdgeEndpoint,
    HarnessNodePosition,
    HarnessPortDescriptor,
    HarnessTopologyCandidate,
    HarnessVisualNode,
    validate_and_compile_harness,
)


# 功能：
#   创建带类型端口的最小控制依赖，不绑定或运行真实模型和插件。
# 输入：
#   无。
# 输出：
#   candidate：两个节点及单条控制连线组成的候选。
def _candidate():
    candidate = HarnessTopologyCandidate(
        topology_id="test.topology",
        name="Compiler fixture",
        profile_id="test",
        base_revision=0,
        nodes=[
            HarnessVisualNode(
                node_id="source",
                descriptor_id="source",
                title="Source",
                title_zh="源",
                node_kind="stage",
                handler_id="test.source",
                runtime_node_kind="plugin",
                output_ports=[HarnessPortDescriptor(port_id="out", schema_ref="test.value.v1")],
            ),
            HarnessVisualNode(
                node_id="target",
                descriptor_id="target",
                title="Target",
                title_zh="目标",
                node_kind="stage",
                handler_id="test.target",
                runtime_node_kind="plugin",
                input_ports=[HarnessPortDescriptor(port_id="input", schema_ref="test.value.v1")],
            ),
        ],
        edges=[
            HarnessEdgeBinding(
                edge_id="source-to-target",
                source=HarnessEdgeEndpoint(node_id="source", port_id="out"),
                target=HarnessEdgeEndpoint(node_id="target", port_id="input"),
                schema_ref="test.value.v1",
                binding_mode="control",
            )
        ],
    )
    return candidate


# 功能：
#   验证控制连线真实进入执行依赖，而不是只存在于可视图中。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_control_edge_compiles_into_an_executable_dependency():
    result = validate_and_compile_harness(_candidate())
    assert result.valid
    assert result.compiled_topology.nodes[1].depends_on == ["source"]


# 功能：
#   拒绝尚无执行实现的边转换，避免激活后转换逻辑被悄悄丢弃。
# 输入：
#   mode：连线绑定模式。
#   plugin：可选的转换插件标识。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mode,plugin", [("transform", None), ("control", "missing.transform")])
def test_unimplemented_edge_transform_cannot_be_silently_discarded(mode, plugin):
    candidate = _candidate()
    candidate.edges[0].binding_mode = mode
    candidate.edges[0].transform_plugin_id = plugin
    result = validate_and_compile_harness(candidate)
    assert not result.valid
    assert any(issue.code == "HARNESS_EDGE_TRANSFORM_UNSUPPORTED" for issue in result.issues)


# 功能：
#   检查同名端口不能覆盖另一条保密契约，碰撞必须产生编译诊断。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_duplicate_port_identity_cannot_shadow_a_confidentiality_contract():
    candidate = _candidate()
    candidate.nodes[0].output_ports.append(
        candidate.nodes[0]
        .output_ports[0]
        .model_copy(
            update={"confidentiality": "secret"},
        )
    )
    result = validate_and_compile_harness(candidate)
    assert not result.valid
    assert any(issue.code == "HARNESS_DUPLICATE_PORT_ID" for issue in result.issues)


# 功能：
#   确认仅在视觉配置中声明循环不能声称拥有执行循环的能力。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_visual_loop_requires_an_executable_loop_handler():
    candidate = _candidate()
    candidate.loops.append(HarnessBoundedLoop(loop_id="loop.fixture", node_ids=["source"]))
    result = validate_and_compile_harness(candidate)
    assert not result.valid
    assert any(issue.code == "HARNESS_LOOP_RUNTIME_UNSUPPORTED" for issue in result.issues)


# 功能：
#   检查运行时尚未落实的逐节点调用预算被明确拒绝，不冒充已经生效的限制。
# 输入：
#   field：待验证的逐节点预算字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["maximum_model_calls", "maximum_tool_calls"])
def test_unenforced_per_node_budget_is_not_presented_as_compiled(field):
    candidate = _candidate()
    setattr(candidate.nodes[0].policy, field, 1)
    result = validate_and_compile_harness(candidate)
    assert not result.valid
    assert any(issue.code == "HARNESS_NODE_CALL_BUDGET_UNSUPPORTED" for issue in result.issues)


# 功能：
#   大量错误连线仍返回有界失败诊断，不能超出结果契约或被截断后误判通过。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_many_invalid_edges_return_bounded_diagnostics_instead_of_raising():
    candidate = _candidate()
    candidate.edges = [
        candidate.edges[0].model_copy(
            update={
                "edge_id": f"missing-{index}",
                "source": HarnessEdgeEndpoint(node_id="absent", port_id="out"),
            }
        )
        for index in range(600)
    ]
    result = validate_and_compile_harness(candidate)
    assert not result.valid and result.compiled_topology is None
    assert len(result.issues) <= 512
    assert result.issues[-1].code == "HARNESS_DIAGNOSTICS_TRUNCATED"


# 功能：
#   非有限画布坐标在模型构造时就被拒绝，不能流入摘要或界面布局。
# 输入：
#   position：注入的 NaN 或无穷坐标。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("position", [float("nan"), float("inf")])
def test_non_finite_canvas_positions_are_rejected_before_hashing(position):
    with pytest.raises(ValueError):
        HarnessNodePosition(x=position, y=0)
