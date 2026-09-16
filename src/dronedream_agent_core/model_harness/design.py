"""Contracts and compiler for the visual DroneDream Harness editor.

The visual graph is deliberately separate from the executable graph.  Layout
changes affect only ``layout_sha256`` while ports, edges, policies and exact
plugin bindings determine ``semantic_sha256`` and the compiled runtime graph.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dronedream_plugin_sdk.protocol import copy_json

from ..hashing import sha256_json
from .graph import HarnessNodeSpec, HarnessTopology, validate_protected_mission_topology

HarnessVisualNodeKind = Literal[
    "input",
    "output",
    "model_call",
    "tool_call",
    "transform",
    "branch",
    "join",
    "safety_barrier",
    "bounded_loop",
    "human_approval",
    "composite",
    "stage",
]
HarnessCardinality = Literal["one", "many", "stream", "event"]
HarnessConfidentiality = Literal["public", "task", "sensitive", "secret"]
HarnessBindingMode = Literal["direct", "control", "transform"]
HarnessRevisionState = Literal[
    "candidate",
    "active",
    "applies_next_run",
    "rejected",
    "frozen",
]
HarnessOperationKind = Literal[
    "add_node",
    "remove_node",
    "connect",
    "disconnect",
    "move_node",
    "update_layout",
    "update_node",
    "replace_node",
    "apply_profile",
    "apply_template",
]

_CONFIDENTIALITY_RANK = {"public": 0, "task": 1, "sensitive": 2, "secret": 3}
_CANDIDATE_BYTES = 8 * 1024 * 1024


class HarnessDesignModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        strict=True,
    )


class HarnessPortDescriptor(HarnessDesignModel):
    port_id: str = Field(pattern=r"^[a-z][a-z0-9._:-]{1,119}$")
    schema_ref: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,159}\.v[1-9][0-9]*$")
    required: bool = True
    cardinality: HarnessCardinality = "one"
    confidentiality: HarnessConfidentiality = "task"
    maximum_connections: int = Field(default=1, ge=1, le=64)


class HarnessNodePolicy(HarnessDesignModel):
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    retry_limit: int = Field(default=0, ge=0, le=5)
    failure_mode: Literal["fail-closed", "isolate", "fallback"] = "fail-closed"
    fallback_handler_id: str | None = Field(default=None, max_length=160)
    cacheable: bool = False
    authority: Literal["read", "plan", "simulate"] = "plan"
    maximum_model_calls: int | None = Field(default=None, ge=1, le=48)
    maximum_tool_calls: int | None = Field(default=None, ge=0, le=64)

    # 功能：
    #   回退策略必须声明具体处理器，界面选择回退不能自动创造可执行实现。
    # 输入：
    #   self：待验证的节点策略。
    # 输出：
    #   self：满足回退绑定要求的策略对象。
    @model_validator(mode="after")
    def validate_fallback(self) -> HarnessNodePolicy:
        if self.failure_mode == "fallback" and not self.fallback_handler_id:
            raise ValueError("fallback policy requires fallback_handler_id")
        return self


class HarnessNodeCapabilities(HarnessDesignModel):
    removable: bool = True
    replaceable: bool = True
    branchable: bool = False
    wrappable_in_loop: bool = True
    protected: bool = False
    allowed_operations: list[str] = Field(default_factory=list, max_length=32)


class HarnessPluginBinding(HarnessDesignModel):
    plugin_id: str
    version: str
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trust_status: str
    health: str
    permissions: list[str] = Field(default_factory=list, max_length=64)


class HarnessCompositionItem(HarnessDesignModel):
    """One navigable puzzle in the three-level visual composition catalog.

    The hierarchy is descriptive: every item points back to executable nodes or
    real plugin slots.  It never creates a second, presentation-only runtime
    graph.
    """

    schema_version: Literal["dronedream.harness-composition-item.v1"] = (
        "dronedream.harness-composition-item.v1"
    )
    item_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,159}$")
    level: Literal[1, 2, 3]
    parent_item_id: str | None = Field(default=None, max_length=160)
    kind: Literal["phase", "stage", "plugin-slot", "policy"]
    granularity: Literal["large", "medium", "small"]
    title: str = Field(min_length=1, max_length=120)
    title_zh: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=320)
    description_zh: str = Field(default="", max_length=320)
    category_id: str = Field(
        default="orchestration",
        pattern=r"^[a-z][a-z0-9-]{1,39}$",
    )
    color_token: str = Field(pattern=r"^[a-z][a-z0-9-]{1,39}$")
    visual_kind: Literal["puzzle", "model"] = "puzzle"
    aspect_ratio: Literal["1:1", "1.5:1", "model-bar"] = "1:1"
    icon: str = Field(default="blocks", max_length=48)
    order: int = Field(ge=0, le=999)
    member_node_ids: list[str] = Field(default_factory=list, max_length=128)
    plugin_slot_ids: list[str] = Field(default_factory=list, max_length=32)
    child_item_ids: list[str] = Field(default_factory=list, max_length=128)
    enterable: bool = False
    replaceable: bool = False
    protected: bool = False
    scope: Literal["workflow", "phase", "node"] = "workflow"


class HarnessNodeDescriptor(HarnessDesignModel):
    schema_version: Literal["dronedream.harness-node-descriptor.v2"] = (
        "dronedream.harness-node-descriptor.v2"
    )
    descriptor_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,159}$")
    title: str = Field(min_length=1, max_length=120)
    title_zh: str = Field(min_length=1, max_length=120)
    node_kind: HarnessVisualNodeKind
    handler_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,159}$")
    runtime_node_kind: Literal["core", "plugin", "barrier"]
    required_inputs: list[str] = Field(default_factory=list, max_length=64)
    output_key: str | None = Field(default=None, max_length=120)
    input_ports: list[HarnessPortDescriptor] = Field(default_factory=list, max_length=64)
    output_ports: list[HarnessPortDescriptor] = Field(default_factory=list, max_length=64)
    policy: HarnessNodePolicy = Field(default_factory=HarnessNodePolicy)
    capabilities: HarnessNodeCapabilities = Field(default_factory=HarnessNodeCapabilities)
    plugin: HarnessPluginBinding | None = None
    category: str = "stage"
    icon: str = "blocks"


class HarnessVisualNode(HarnessDesignModel):
    node_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,119}$")
    descriptor_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,159}$")
    title: str = Field(min_length=1, max_length=120)
    title_zh: str = Field(min_length=1, max_length=120)
    node_kind: HarnessVisualNodeKind
    handler_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,159}$")
    runtime_node_kind: Literal["core", "plugin", "barrier"]
    required_inputs: list[str] = Field(default_factory=list, max_length=64)
    output_key: str | None = Field(default=None, max_length=120)
    input_ports: list[HarnessPortDescriptor] = Field(default_factory=list, max_length=64)
    output_ports: list[HarnessPortDescriptor] = Field(default_factory=list, max_length=64)
    policy: HarnessNodePolicy = Field(default_factory=HarnessNodePolicy)
    capabilities: HarnessNodeCapabilities = Field(default_factory=HarnessNodeCapabilities)
    plugin: HarnessPluginBinding | None = None
    category: str = "stage"
    icon: str = "blocks"

    # 功能：
    #   从目录描述构造独立可编辑节点，保留目录身份及原始执行策略和端口。
    # 输入：
    #   cls：要构造的节点类型。
    #   descriptor：节点目录描述。
    #   node_id：可选实例标识；未提供时使用目录标识。
    # 输出：
    #   node：与描述的可变容器分离的节点实例。
    @classmethod
    def from_descriptor(
        cls,
        descriptor: HarnessNodeDescriptor,
        *,
        node_id: str | None = None,
    ) -> HarnessVisualNode:
        value = descriptor.model_dump(mode="python")
        value.pop("schema_version")
        value.pop("descriptor_id")
        node = cls(
            node_id=descriptor.descriptor_id if node_id is None else node_id,
            descriptor_id=descriptor.descriptor_id,
            **value,
        )
        return node


class HarnessEdgeEndpoint(HarnessDesignModel):
    node_id: str
    port_id: str


class HarnessEdgeBinding(HarnessDesignModel):
    schema_version: Literal["dronedream.harness-edge-binding.v1"] = (
        "dronedream.harness-edge-binding.v1"
    )
    edge_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._:>-]{2,359}$")
    source: HarnessEdgeEndpoint
    target: HarnessEdgeEndpoint
    schema_ref: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,159}\.v[1-9][0-9]*$")
    transform_plugin_id: str | None = None
    binding_mode: HarnessBindingMode = "direct"


class HarnessNodePosition(HarnessDesignModel):
    x: float
    y: float
    pinned: bool = False


class HarnessViewport(HarnessDesignModel):
    x: float = 0.0
    y: float = 0.0
    zoom: float = Field(default=1.0, ge=0.2, le=2.0)


class HarnessCanvasLayout(HarnessDesignModel):
    positions: dict[str, HarnessNodePosition] = Field(default_factory=dict)
    viewport: HarnessViewport = Field(default_factory=HarnessViewport)
    collapsed_node_ids: list[str] = Field(default_factory=list, max_length=256)
    selected_node_id: str | None = None


class HarnessBoundedLoop(HarnessDesignModel):
    schema_version: Literal["dronedream.harness-loop-group.v1"] = "dronedream.harness-loop-group.v1"
    loop_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,119}$")
    node_ids: list[str] = Field(min_length=1, max_length=64)
    max_iterations: int = Field(default=3, ge=1, le=20)
    exit_condition_schema: str = Field(
        default="dronedream.harness-loop-qualified.v1",
        pattern=r"^[a-z][a-z0-9._-]{2,159}\.v[1-9][0-9]*$",
    )
    progress_metric: str = Field(default="blocking_issue_count", max_length=120)
    no_progress_limit: int = Field(default=2, ge=1, le=10)
    time_budget_seconds: int = Field(default=120, ge=1, le=3600)
    model_call_budget: int = Field(default=12, ge=1, le=48)
    tool_call_budget: int = Field(default=20, ge=0, le=128)
    on_exhausted: Literal["safe_exit", "fail_closed"] = "safe_exit"


class HarnessTopologyCandidate(HarnessDesignModel):
    schema_version: Literal["dronedream.harness-topology-candidate.v2"] = (
        "dronedream.harness-topology-candidate.v2"
    )
    topology_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,119}$")
    name: str = Field(min_length=1, max_length=120)
    profile_id: str
    base_revision: int = Field(ge=0)
    nodes: list[HarnessVisualNode] = Field(min_length=1, max_length=256)
    edges: list[HarnessEdgeBinding] = Field(default_factory=list, max_length=2048)
    loops: list[HarnessBoundedLoop] = Field(default_factory=list, max_length=32)
    maximum_parallelism: int = Field(default=4, ge=1, le=16)
    layout: HarnessCanvasLayout = Field(default_factory=HarnessCanvasLayout)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # 功能：
    #   拒绝节点、连线或循环身份碰撞，避免后续字典索引悄悄覆盖另一项。
    # 输入：
    #   self：候选配置。
    # 输出：
    #   self：三类身份均唯一的候选配置。
    @model_validator(mode="after")
    def validate_identity(self) -> HarnessTopologyCandidate:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("HARNESS_DUPLICATE_NODE_ID")
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("HARNESS_DUPLICATE_EDGE_ID")
        loop_ids = [loop.loop_id for loop in self.loops]
        if len(loop_ids) != len(set(loop_ids)):
            raise ValueError("HARNESS_DUPLICATE_LOOP_ID")
        return self


class HarnessEditOperation(HarnessDesignModel):
    schema_version: Literal["dronedream.harness-edit-operation.v1"] = (
        "dronedream.harness-edit-operation.v1"
    )
    client_operation_id: str = Field(min_length=8, max_length=120)
    base_revision: int = Field(ge=0)
    operation: HarnessOperationKind
    payload: dict[str, Any] = Field(default_factory=dict)


class HarnessValidationIssue(HarnessDesignModel):
    code: str
    message: str
    node_id: str | None = None
    port_id: str | None = None
    edge_id: str | None = None
    severity: Literal["error", "warning"] = "error"


class HarnessValidationResult(HarnessDesignModel):
    valid: bool
    issues: list[HarnessValidationIssue] = Field(default_factory=list, max_length=512)
    semantic_sha256: str | None = None
    layout_sha256: str | None = None
    compiled_topology: HarnessTopology | None = None


class HarnessRevision(HarnessDesignModel):
    schema_version: Literal["dronedream.harness-revision.v1"] = "dronedream.harness-revision.v1"
    revision: int = Field(ge=1)
    parent_revision: int | None = Field(default=None, ge=1)
    state: HarnessRevisionState
    candidate: HarnessTopologyCandidate
    validation: HarnessValidationResult
    operations: list[HarnessEditOperation] = Field(default_factory=list, max_length=512)
    created_at: str
    activated_at: str | None = None
    applies_next_run: bool = False


# 功能：
#   按连线两端的节点和端口生成稳定显示标识；是否允许连接仍需图编译校验。
# 输入：
#   source_node_id：来源节点标识。
#   source_port_id：来源端口标识。
#   target_node_id：目标节点标识。
#   target_port_id：目标端口标识。
# 输出：
#   identity：包含两端身份的连线标识。
def edge_identity(
    source_node_id: str,
    source_port_id: str,
    target_node_id: str,
    target_port_id: str,
) -> str:
    identity = f"{source_node_id}:{source_port_id}->{target_node_id}:{target_port_id}"
    return identity


# 功能：
#   以严格有界 JSON 复制并重新验证候选，拒绝构造后的非法改写和非标准元数据。
#   编译产物只使用此独立快照，不借用调用者后续仍可修改的嵌套容器。
# 输入：
#   candidate：来自编辑器或内部调用的候选模型。
# 输出：
#   checked：类型、身份和预算检查通过的独立候选。
def _checked_candidate(candidate: HarnessTopologyCandidate) -> HarnessTopologyCandidate:
    if not isinstance(candidate, HarnessTopologyCandidate):
        raise ValueError("HARNESS_CANDIDATE_INVALID")
    value = copy_json(candidate.model_dump(mode="python", warnings="error"), limit=_CANDIDATE_BYTES)
    checked = HarnessTopologyCandidate.model_validate(value)
    return checked


# 功能：
#   从重新校验的候选提取语义摘要载荷，排除画布布局和编辑历史基准编号。
# 输入：
#   candidate：待绑定执行配置身份的候选。
# 输出：
#   payload：独立的语义 JSON 载荷，不包含画布布局或基准编号。
def semantic_payload(candidate: HarnessTopologyCandidate) -> dict[str, Any]:
    payload = _checked_candidate(candidate).model_dump(mode="json")
    payload.pop("layout", None)
    payload.pop("base_revision", None)
    return payload


# 功能：
#   1. 重新验证独立候选，检查端口模式、基数、保密等级、双向连接预算和必要安全节点。
#   2. 仅将已实现的控制依赖编译成执行图；不支持的数据映射、精确插件绑定和可视循环明确拒绝。
#   3. 生成有界诊断与语义／布局摘要，不执行模型、插件或飞行器，不代替运行时权限及预算校验。
# 输入：
#   candidate：待检查的可视候选配置。
# 输出：
#   result：是否通过、问题列表、摘要以及通过时的独立执行拓扑。
def validate_and_compile_harness(candidate: HarnessTopologyCandidate) -> HarnessValidationResult:
    try:
        candidate = _checked_candidate(candidate)
    except (ValueError, TypeError, RecursionError):
        result = HarnessValidationResult(
            valid=False,
            issues=[
                HarnessValidationIssue(
                    code="HARNESS_CANDIDATE_INVALID",
                    message="Candidate must contain bounded JSON and valid typed node identities.",
                )
            ],
        )
        return result
    issues: list[HarnessValidationIssue] = []
    nodes = {node.node_id: node for node in candidate.nodes}
    incoming: dict[tuple[str, str], list[HarnessEdgeBinding]] = defaultdict(list)
    outgoing: dict[tuple[str, str], list[HarnessEdgeBinding]] = defaultdict(list)
    dependencies: dict[str, set[str]] = defaultdict(set)

    for edge in candidate.edges:
        # 当前执行图只保存节点依赖；端口值映射尚无执行实现，不能默默丢弃其语义。
        if edge.binding_mode == "direct":
            issues.append(
                HarnessValidationIssue(
                    code="HARNESS_EDGE_DATA_BINDING_UNSUPPORTED",
                    message="Direct port data bindings require an executable value mapping.",
                    edge_id=edge.edge_id,
                )
            )
        if edge.binding_mode == "transform" or edge.transform_plugin_id is not None:
            issues.append(
                HarnessValidationIssue(
                    code="HARNESS_EDGE_TRANSFORM_UNSUPPORTED",
                    message="Edge transforms need an executable handler before activation.",
                    edge_id=edge.edge_id,
                )
            )
        source = nodes.get(edge.source.node_id)
        target = nodes.get(edge.target.node_id)
        if source is None or target is None:
            issues.append(
                HarnessValidationIssue(
                    code="HARNESS_EDGE_NODE_MISSING",
                    message="The connection references a missing node.",
                    edge_id=edge.edge_id,
                )
            )
            continue
        source_port = next(
            (port for port in source.output_ports if port.port_id == edge.source.port_id), None
        )
        target_port = next(
            (port for port in target.input_ports if port.port_id == edge.target.port_id), None
        )
        if source_port is None or target_port is None:
            issues.append(
                HarnessValidationIssue(
                    code="HARNESS_EDGE_PORT_MISSING",
                    message="The connection references a missing port.",
                    edge_id=edge.edge_id,
                )
            )
            continue
        if source_port.schema_ref != edge.schema_ref or target_port.schema_ref != edge.schema_ref:
            issues.append(
                HarnessValidationIssue(
                    code="HARNESS_SCHEMA_INCOMPATIBLE",
                    message="The connected ports use incompatible Schema versions.",
                    edge_id=edge.edge_id,
                )
            )
        if source_port.cardinality != target_port.cardinality:
            issues.append(
                HarnessValidationIssue(
                    code="HARNESS_CARDINALITY_INCOMPATIBLE",
                    message="The connected ports use incompatible cardinality.",
                    edge_id=edge.edge_id,
                )
            )
        if (
            _CONFIDENTIALITY_RANK[source_port.confidentiality]
            > _CONFIDENTIALITY_RANK[target_port.confidentiality]
        ):
            issues.append(
                HarnessValidationIssue(
                    code="HARNESS_CONFIDENTIALITY_DOWNGRADE",
                    message="The connection would lower the data confidentiality boundary.",
                    edge_id=edge.edge_id,
                )
            )
        incoming[(target.node_id, target_port.port_id)].append(edge)
        outgoing[(source.node_id, source_port.port_id)].append(edge)
        dependencies[target.node_id].add(source.node_id)

    for node in candidate.nodes:
        if node.node_kind in {"human_approval", "branch", "bounded_loop"}:
            # 当前运行图没有批准等待、条件选择或逐次循环字段，不能仅换图标就视为实现。
            issues.append(
                HarnessValidationIssue(
                    code="HARNESS_CONTROL_NODE_UNSUPPORTED",
                    message="This visual control node requires dedicated execution semantics.",
                    node_id=node.node_id,
                )
            )
        # 不采用最后一个同名端口覆盖前一个的行为，保密级别与连接容量必须保持唯一。
        for ports in (node.input_ports, node.output_ports):
            if len({port.port_id for port in ports}) != len(ports):
                issues.append(
                    HarnessValidationIssue(
                        code="HARNESS_DUPLICATE_PORT_ID",
                        message=(
                            "Port identities must be unique within each input/output direction."
                        ),
                        node_id=node.node_id,
                    )
                )
        if (
            node.policy.maximum_model_calls is not None
            or node.policy.maximum_tool_calls is not None
        ):
            issues.append(
                HarnessValidationIssue(
                    code="HARNESS_NODE_CALL_BUDGET_UNSUPPORTED",
                    message="Per-node call budgets are not implemented by the compiled runtime.",
                    node_id=node.node_id,
                )
            )
        for port in node.input_ports:
            count = len(incoming[(node.node_id, port.port_id)])
            if port.required and count == 0:
                issues.append(
                    HarnessValidationIssue(
                        code="HARNESS_REQUIRED_INPUT_MISSING",
                        message="A required structured input is not connected.",
                        node_id=node.node_id,
                        port_id=port.port_id,
                    )
                )
            if count > port.maximum_connections:
                issues.append(
                    HarnessValidationIssue(
                        code="HARNESS_PORT_CONNECTION_LIMIT",
                        message="The input port has too many connections.",
                        node_id=node.node_id,
                        port_id=port.port_id,
                    )
                )
        for port in node.output_ports:
            if len(outgoing[(node.node_id, port.port_id)]) > port.maximum_connections:
                issues.append(
                    HarnessValidationIssue(
                        code="HARNESS_PORT_CONNECTION_LIMIT",
                        message="The output port has too many connections.",
                        node_id=node.node_id,
                        port_id=port.port_id,
                    )
                )
        if node.capabilities.protected and node.policy.failure_mode != "fail-closed":
            issues.append(
                HarnessValidationIssue(
                    code="HARNESS_PROTECTED_NODE_MUST_FAIL_CLOSED",
                    message="A protected node must fail closed.",
                    node_id=node.node_id,
                )
            )
        if node.plugin is not None:
            # 运行图尚无此节点精确包绑定字段；目录的 healthy 标签不能代替实际绑定。
            issues.append(
                HarnessValidationIssue(
                    code="HARNESS_EXACT_PLUGIN_BINDING_UNSUPPORTED",
                    message=(
                        "Exact per-node plugin bindings must be preserved "
                        "by the runtime before activation."
                    ),
                    node_id=node.node_id,
                )
            )
        if node.plugin is not None and (
            node.plugin.health != "healthy"
            or node.plugin.trust_status not in {"verified", "local-approved", "builtin"}
        ):
            issues.append(
                HarnessValidationIssue(
                    code="HARNESS_PLUGIN_UNAVAILABLE",
                    message="The bound plugin is not healthy and trusted.",
                    node_id=node.node_id,
                )
            )

    loop_members: set[str] = set()
    if candidate.loops:
        # 将循环仅放进元数据不会执行迭代、退出或预算，因此不能声称已经支持循环。
        issues.append(
            HarnessValidationIssue(
                code="HARNESS_LOOP_RUNTIME_UNSUPPORTED",
                message="Visual loop groups require an executable bounded-loop runtime.",
            )
        )
    for loop in candidate.loops:
        missing = sorted(set(loop.node_ids) - set(nodes))
        overlap = sorted(set(loop.node_ids) & loop_members)
        if missing:
            issues.append(
                HarnessValidationIssue(
                    code="HARNESS_LOOP_NODE_MISSING",
                    message="The bounded loop references a missing node.",
                    node_id=missing[0],
                )
            )
        if overlap:
            issues.append(
                HarnessValidationIssue(
                    code="HARNESS_LOOP_OVERLAP",
                    message="A node cannot belong to more than one bounded loop.",
                    node_id=overlap[0],
                )
            )
        loop_members.update(loop.node_ids)

    compiled: HarnessTopology | None = None
    if not issues:
        try:
            # 保留运行时实际支持的策略；模型、工具和飞行调用由后续执行器承担。
            runtime_nodes = [
                HarnessNodeSpec(
                    node_id=node.node_id,
                    node_kind=node.runtime_node_kind,
                    handler_id=node.handler_id,
                    depends_on=sorted(dependencies[node.node_id]),
                    required_inputs=node.required_inputs,
                    output_key=node.output_key,
                    timeout_seconds=node.policy.timeout_seconds,
                    retry_limit=node.policy.retry_limit,
                    failure_mode=node.policy.failure_mode,
                    fallback_handler_id=node.policy.fallback_handler_id,
                    cacheable=node.policy.cacheable,
                    authority=node.policy.authority,
                )
                for node in candidate.nodes
            ]
            compiled = HarnessTopology(
                topology_id=candidate.topology_id,
                name=candidate.name,
                nodes=runtime_nodes,
                maximum_parallelism=candidate.maximum_parallelism,
                metadata={
                    **candidate.metadata,
                    "profile_id": candidate.profile_id,
                    "bounded_loops": [loop.model_dump(mode="json") for loop in candidate.loops],
                },
            )
            if any(node.node_id.startswith("mission.") for node in runtime_nodes):
                validate_protected_mission_topology(compiled)
        except ValueError as error:
            issues.append(
                HarnessValidationIssue(
                    code=str(error).split(":", maxsplit=1)[0],
                    message=str(error),
                )
            )
            compiled = None

    try:
        semantic_sha256 = sha256_json(semantic_payload(candidate))
        layout_sha256 = sha256_json(candidate.layout.model_dump(mode="json"))
    except (ValueError, TypeError, RecursionError) as error:
        # 摘要无法生成时必须拒绝，不能输出没有身份绑定的执行图。
        issues.append(
            HarnessValidationIssue(
                code="HARNESS_UNHASHABLE_CONFIGURATION",
                message=f"Configuration must contain finite JSON data ({type(error).__name__}).",
            )
        )
        semantic_sha256 = layout_sha256 = None
        compiled = None
    if len(issues) > 512:
        issues = [
            *issues[:511],
            HarnessValidationIssue(
                code="HARNESS_DIAGNOSTICS_TRUNCATED",
                message="Additional validation issues omitted; the candidate remains rejected.",
            ),
        ]
    result = HarnessValidationResult(
        valid=not issues and compiled is not None,
        issues=issues,
        semantic_sha256=semantic_sha256,
        layout_sha256=layout_sha256,
        compiled_topology=compiled,
    )
    return result


__all__ = [
    "HarnessBoundedLoop",
    "HarnessCanvasLayout",
    "HarnessEditOperation",
    "HarnessEdgeBinding",
    "HarnessEdgeEndpoint",
    "HarnessNodeCapabilities",
    "HarnessCompositionItem",
    "HarnessNodeDescriptor",
    "HarnessNodePolicy",
    "HarnessNodePosition",
    "HarnessPluginBinding",
    "HarnessPortDescriptor",
    "HarnessRevision",
    "HarnessTopologyCandidate",
    "HarnessValidationIssue",
    "HarnessValidationResult",
    "HarnessVisualNode",
    "HarnessViewport",
    "edge_identity",
    "semantic_payload",
    "validate_and_compile_harness",
]
