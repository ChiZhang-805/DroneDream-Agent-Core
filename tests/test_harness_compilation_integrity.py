"""可视候选不能通过可变对象或未实现的执行语义伪装成有效配置。"""

import pytest

from dronedream_agent_core.model_harness.design import (
    HarnessEdgeBinding,
    HarnessEdgeEndpoint,
    HarnessPluginBinding,
    HarnessPortDescriptor,
    HarnessTopologyCandidate,
    HarnessVisualNode,
    validate_and_compile_harness,
)


# 功能：
#   构造两节点控制依赖，节点处理器仅作声明，不会被测试调用。
# 输入：
#   无。
# 输出：
#   candidate：用于独立检验编译边界的可视候选。
def _candidate():
    candidate = HarnessTopologyCandidate(
        topology_id="test.integrity",
        name="Integrity",
        profile_id="test",
        base_revision=0,
        nodes=[
            HarnessVisualNode(
                node_id="source",
                descriptor_id="source",
                title="Source",
                title_zh="来源",
                node_kind="stage",
                handler_id="test.source",
                runtime_node_kind="plugin",
                output_ports=[HarnessPortDescriptor(port_id="out", schema_ref="test.data.v1")],
            ),
            HarnessVisualNode(
                node_id="target",
                descriptor_id="target",
                title="Target",
                title_zh="目标",
                node_kind="stage",
                handler_id="test.target",
                runtime_node_kind="plugin",
                input_ports=[HarnessPortDescriptor(port_id="input", schema_ref="test.data.v1")],
            ),
        ],
        edges=[
            HarnessEdgeBinding(
                edge_id="source-to-target",
                binding_mode="control",
                source=HarnessEdgeEndpoint(node_id="source", port_id="out"),
                target=HarnessEdgeEndpoint(node_id="target", port_id="input"),
                schema_ref="test.data.v1",
            )
        ],
    )
    return candidate


# 功能：
#   验证候选构造后的非法改写仍被拒绝，不因首次模型验证通过而跳过编译入口验证。
# 输入：
#   mutation：需要注入的字段变化。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "mutation", ["confidentiality", "duplicate_node", "boolean_budget", "foreign_node"]
)
def test_mutated_candidate_is_revalidated(mutation):
    candidate = _candidate()
    if mutation == "confidentiality":
        candidate.nodes[0].output_ports[0].confidentiality = "unknown"
    elif mutation == "duplicate_node":
        candidate.nodes.append(candidate.nodes[0])
    elif mutation == "boolean_budget":
        candidate.maximum_parallelism = True
    else:
        candidate.nodes.append(object())
    result = validate_and_compile_harness(candidate)
    assert not result.valid and result.compiled_topology is None
    assert result.issues[0].code == "HARNESS_CANDIDATE_INVALID"


# 功能：
#   拒绝元数据中的非 JSON 类型，不能经序列化静默改写键或集合后获得有效摘要。
# 输入：
#   metadata：带错误键或值的自由元数据。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("metadata", [{1: "numeric key"}, {"set": {1, 2}}, {"tuple": (1, 2)}])
def test_metadata_is_checked_before_json_coercion(metadata):
    candidate = _candidate()
    candidate.metadata = metadata
    result = validate_and_compile_harness(candidate)
    assert not result.valid and result.compiled_topology is None


# 功能：
#   验证编译产物与候选不共享嵌套元数据，调用后修改来源不会改变已绑定摘要的产物。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_compiled_metadata_does_not_borrow_mutable_candidate():
    candidate = _candidate()
    candidate.metadata = {"nested": {"safety": "original"}}
    result = validate_and_compile_harness(candidate)
    assert result.valid
    checksum = result.semantic_sha256
    candidate.metadata["nested"]["safety"] = "changed"
    assert result.compiled_topology.metadata["nested"]["safety"] == "original"
    assert result.semantic_sha256 == checksum


# 功能：
#   验证输出端连接数限制同样生效，不能只检查每个接收端而忽略来源的总扇出。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_output_connection_budget_is_enforced():
    candidate = _candidate()
    candidate.nodes.append(candidate.nodes[1].model_copy(deep=True, update={"node_id": "third"}))
    candidate.edges.append(
        candidate.edges[0].model_copy(
            deep=True,
            update={
                "edge_id": "source-to-third",
                "target": HarnessEdgeEndpoint(node_id="third", port_id="input"),
            },
        )
    )
    result = validate_and_compile_harness(candidate)
    assert not result.valid
    assert any(issue.code == "HARNESS_PORT_CONNECTION_LIMIT" for issue in result.issues)


# 功能：
#   拒绝尚未实现的直接端口数据映射，不能把它静默编译成仅有先后顺序的依赖。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_direct_data_binding_requires_runtime_mapping():
    candidate = _candidate()
    candidate.edges[0].binding_mode = "direct"
    result = validate_and_compile_harness(candidate)
    assert not result.valid
    assert any(issue.code == "HARNESS_EDGE_DATA_BINDING_UNSUPPORTED" for issue in result.issues)


# 功能：
#   检查精确插件绑定不能只留在视觉配置中，尚未传入执行拓扑时必须拒绝激活。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_uncompiled_exact_plugin_binding_is_rejected():
    candidate = _candidate()
    candidate.nodes[0].plugin = HarnessPluginBinding(
        plugin_id="test.plugin",
        version="1.0.0",
        package_sha256="a" * 64,
        trust_status="verified",
        health="healthy",
    )
    result = validate_and_compile_harness(candidate)
    assert not result.valid
    assert any(issue.code == "HARNESS_EXACT_PLUGIN_BINDING_UNSUPPORTED" for issue in result.issues)


# 功能：
#   未实现的人工批准、条件分支和循环节点不能仅靠更换视觉类型就被当成可执行机制。
# 输入：
#   kind：需要专门执行语义的节点类别。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["human_approval", "branch", "bounded_loop"])
def test_unimplemented_control_nodes_are_not_ordinary_stages(kind):
    candidate = _candidate()
    candidate.nodes[0].node_kind = kind
    result = validate_and_compile_harness(candidate)
    assert not result.valid
    assert any(issue.code == "HARNESS_CONTROL_NODE_UNSUPPORTED" for issue in result.issues)
