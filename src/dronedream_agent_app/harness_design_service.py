"""Persistent, fail-closed service behind the visual Harness editor."""

from __future__ import annotations

import os
import tempfile
import threading
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dronedream_agent_core.model_harness.design import (
    HarnessCanvasLayout,
    HarnessCompositionItem,
    HarnessEdgeBinding,
    HarnessEdgeEndpoint,
    HarnessEditOperation,
    HarnessNodeCapabilities,
    HarnessNodeDescriptor,
    HarnessNodePolicy,
    HarnessNodePosition,
    HarnessPortDescriptor,
    HarnessRevision,
    HarnessTopologyCandidate,
    HarnessValidationResult,
    HarnessVisualNode,
    edge_identity,
    validate_and_compile_harness,
)
from dronedream_agent_core.model_harness.graph import HarnessNodeSpec, HarnessTopology
from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file
from dronedream_agent_plugins.workflow_topologies import official_topology_templates
from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .plugin_manager import PluginManager
from .storage import AppStore

_CONTROL_SCHEMA = "dronedream.harness-control.v1"
_DOCUMENT_BYTES = 8 * 1024 * 1024
_OPTIONAL_STAGE_IDS = {"mission.tool-advice", "mission.plan-evaluation"}
_MODEL_HANDLERS = {
    "core.intent-parse",
    "core.intent-review",
    "core.intent-consensus",
    "core.task-decompose",
    "core.semantic-plan",
    "core.plan-review",
}
_TITLE_ZH = {
    "mission.request-ingest": "接收任务",
    "mission.context-prepare": "准备上下文",
    "mission.intent-parse": "解析意图",
    "mission.intent-review-1": "意图审查 1",
    "mission.intent-review-2": "意图审查 2",
    "mission.intent-review-3": "意图审查 3",
    "mission.intent-consensus": "意图共识",
    "mission.contract-freeze": "冻结任务合同",
    "mission.tool-advice": "工具顾问",
    "mission.task-decompose": "拆分任务",
    "mission.semantic-plan": "生成语义计划",
    "mission.route-resolve": "求解路线",
    "mission.clearance-gate": "净空安全门",
    "mission.track-export": "导出航迹",
    "mission.plan-evaluation": "评估计划",
    "mission.plan-review": "审查计划",
    "mission.runtime-checkpoints": "生成运行检查点",
    "mission.verification-plan": "编译验证要求",
    "mission.evidence-finalize": "固化验收证据",
}
_TITLE_EN = {
    "mission.request-ingest": "Task input",
    "mission.context-prepare": "Context preparation",
    "mission.intent-parse": "Intent parsing",
    "mission.intent-review-1": "Intent review 1",
    "mission.intent-review-2": "Intent review 2",
    "mission.intent-review-3": "Intent review 3",
    "mission.intent-consensus": "Intent consensus",
    "mission.contract-freeze": "Mission contract",
    "mission.tool-advice": "Tool advisor",
    "mission.task-decompose": "Task decomposition",
    "mission.semantic-plan": "Semantic planning",
    "mission.route-resolve": "Route resolution",
    "mission.clearance-gate": "Clearance gate",
    "mission.track-export": "Track export",
    "mission.plan-evaluation": "Plan evaluation",
    "mission.plan-review": "Plan review",
    "mission.runtime-checkpoints": "Runtime checkpoints",
    "mission.verification-plan": "Verification plan",
    "mission.evidence-finalize": "Evidence finalization",
}

_COMPOSITION_PHASES = (
    {
        "item_id": "composition.phase.intake",
        "title": "Task intake",
        "title_zh": "任务接入",
        "description": "Receive the request and assemble task-scoped context.",
        "description_zh": "接收用户任务并准备任务级上下文。",
        "color_token": "amber",
        "icon": "inbox",
        "nodes": ("mission.request-ingest", "mission.context-prepare"),
    },
    {
        "item_id": "composition.phase.intent",
        "title": "Intent understanding",
        "title_zh": "意图理解",
        "description": "Parse, independently review, and converge the mission intent.",
        "description_zh": "解析、独立审查并收敛任务意图。",
        "color_token": "green",
        "icon": "brain",
        "nodes": (
            "mission.intent-parse",
            "mission.intent-review-1",
            "mission.intent-review-2",
            "mission.intent-review-3",
            "mission.intent-consensus",
        ),
    },
    {
        "item_id": "composition.phase.contract",
        "title": "Contract and tools",
        "title_zh": "合同与工具",
        "description": "Freeze the structured contract, consult tools, and decompose work.",
        "description_zh": "冻结结构化合同、咨询工具并拆分任务。",
        "color_token": "blue",
        "icon": "file-check",
        "nodes": (
            "mission.contract-freeze",
            "mission.tool-advice",
            "mission.task-decompose",
        ),
    },
    {
        "item_id": "composition.phase.planning",
        "title": "Planning and route",
        "title_zh": "规划与路线",
        "description": "Produce a semantic plan and resolve a flyable route.",
        "description_zh": "生成语义计划并求解可飞行路线。",
        "color_token": "red",
        "icon": "route",
        "nodes": ("mission.semantic-plan", "mission.route-resolve"),
    },
    {
        "item_id": "composition.phase.safety",
        "title": "Safety and export",
        "title_zh": "安全与导出",
        "description": "Enforce clearance boundaries before exporting the track.",
        "description_zh": "通过净空安全边界后才允许导出航迹。",
        "color_token": "violet",
        "icon": "shield",
        "nodes": ("mission.clearance-gate", "mission.track-export"),
    },
    {
        "item_id": "composition.phase.assurance",
        "title": "Review and evidence",
        "title_zh": "审查与证据",
        "description": "Evaluate, review, checkpoint, and finalize acceptance evidence.",
        "description_zh": "评估、审查、生成检查点并固化验收证据。",
        "color_token": "cyan",
        "icon": "badge-check",
        "nodes": (
            "mission.plan-evaluation",
            "mission.plan-review",
            "mission.runtime-checkpoints",
            "mission.verification-plan",
            "mission.evidence-finalize",
        ),
    },
)

_STAGE_PLUGIN_SLOTS = {
    "mission.request-ingest": ("harness.event-bus", "harness.observers"),
    "mission.context-prepare": ("harness.cache-policy", "harness.observers"),
    "mission.intent-parse": (
        "harness.retry-policy",
        "harness.timeout-policy",
        "harness.budget-policy",
        "harness.fallback-policy",
        "harness.cache-policy",
    ),
    "mission.intent-review-1": (
        "harness.retry-policy",
        "harness.timeout-policy",
        "harness.budget-policy",
    ),
    "mission.intent-review-2": (
        "harness.retry-policy",
        "harness.timeout-policy",
        "harness.budget-policy",
    ),
    "mission.intent-review-3": (
        "harness.retry-policy",
        "harness.timeout-policy",
        "harness.budget-policy",
    ),
    "mission.intent-consensus": ("harness.scheduler", "harness.fallback-policy"),
    "mission.contract-freeze": ("harness.event-bus", "harness.observers"),
    "mission.tool-advice": (
        "harness.retry-policy",
        "harness.timeout-policy",
        "harness.budget-policy",
        "harness.fallback-policy",
    ),
    "mission.task-decompose": ("harness.scheduler", "harness.budget-policy"),
    "mission.semantic-plan": (
        "harness.retry-policy",
        "harness.timeout-policy",
        "harness.budget-policy",
        "harness.cache-policy",
    ),
    "mission.route-resolve": (
        "harness.retry-policy",
        "harness.timeout-policy",
        "harness.fallback-policy",
    ),
    "mission.clearance-gate": ("harness.timeout-policy", "harness.observers"),
    "mission.track-export": ("harness.event-bus", "harness.observers"),
    "mission.plan-evaluation": (
        "harness.retry-policy",
        "harness.timeout-policy",
        "harness.cache-policy",
    ),
    "mission.plan-review": (
        "harness.retry-policy",
        "harness.timeout-policy",
        "harness.fallback-policy",
    ),
    "mission.runtime-checkpoints": ("harness.event-bus", "harness.observers"),
    "mission.verification-plan": ("harness.event-bus", "harness.observers"),
    "mission.evidence-finalize": ("harness.event-bus", "harness.observers"),
}

_SLOT_TITLES = {
    "harness.scheduler": ("Scheduler", "调度器"),
    "harness.retry-policy": ("Retry policy", "重试策略"),
    "harness.timeout-policy": ("Timeout policy", "超时策略"),
    "harness.budget-policy": ("Call budget", "调用预算"),
    "harness.fallback-policy": ("Fallback policy", "回退策略"),
    "harness.cache-policy": ("Cache policy", "缓存策略"),
    "harness.event-bus": ("Event bus", "事件总线"),
    "harness.observers": ("Observers", "观测器"),
}

_PHASE_CATEGORIES = {
    "composition.phase.intake": ("input", "amber"),
    "composition.phase.intent": ("reasoning", "violet"),
    "composition.phase.contract": ("structure", "cyan"),
    "composition.phase.planning": ("planning", "blue"),
    "composition.phase.safety": ("safety", "red"),
    "composition.phase.assurance": ("assurance", "green"),
}

_NODE_CATEGORIES = {
    "input": ("input", "amber"),
    "output": ("output", "green"),
    "model_call": ("model", "magenta"),
    "tool_call": ("tooling", "blue"),
    "transform": ("structure", "cyan"),
    "branch": ("control", "violet"),
    "join": ("control", "violet"),
    "safety_barrier": ("safety", "red"),
    "bounded_loop": ("control", "violet"),
    "human_approval": ("safety", "red"),
    "composite": ("orchestration", "indigo"),
    "stage": ("orchestration", "indigo"),
}

_POLICY_CATEGORIES = {
    "timeout": ("orchestration", "violet"),
    "retry": ("orchestration", "violet"),
    "failure": ("safety", "red"),
    "cache": ("memory", "teal"),
}

_SLOT_CATEGORIES = {
    "harness.scheduler": ("orchestration", "violet"),
    "harness.retry-policy": ("orchestration", "violet"),
    "harness.timeout-policy": ("orchestration", "violet"),
    "harness.budget-policy": ("control", "indigo"),
    "harness.fallback-policy": ("safety", "red"),
    "harness.cache-policy": ("memory", "teal"),
    "harness.event-bus": ("integration", "blue"),
    "harness.observers": ("assurance", "green"),
}


# 功能：
#   按双语标题长度选择两种允许的拼图比例，模型条形外观由调用方另行选择。
# 输入：
#   title：英文标题。
#   title_zh：中文标题。
# 输出：
#   ratio：方形或横向拼图的比例标识。
def _aspect_ratio(title: str, title_zh: str) -> str:
    ratio = "1.5:1" if len(title) > 15 or len(title_zh) > 6 else "1:1"
    return ratio


class HarnessDesignServiceError(RuntimeError):
    """Stable issue code returned to the desktop client."""


# 功能：
#   生成带 UTC 时区的记录时间；不用于实时控制延迟或单调计时。
# 输入：
#   无。
# 输出：
#   timestamp：ISO 格式的当前墙钟时间。
def _now() -> str:
    timestamp = datetime.now(UTC).isoformat()
    return timestamp


# 功能：
#   将执行节点映射成编辑器分类，固定任务端点和安全门优先于普通处理器分类。
# 输入：
#   node：执行拓扑中的节点。
# 输出：
#   kind：界面使用的节点类别，不授予新的执行权限。
def _node_kind(node: HarnessNodeSpec) -> str:
    if node.node_id == "mission.request-ingest":
        kind = "input"
    elif node.node_id == "mission.evidence-finalize":
        kind = "output"
    elif node.node_kind == "barrier":
        kind = "safety_barrier"
    elif node.handler_id in _MODEL_HANDLERS:
        kind = "model_call"
    elif node.handler_id in {"core.route-resolve", "core.tool-advice"}:
        kind = "tool_call"
    elif node.handler_id in {"core.context-prepare", "core.track-export"}:
        kind = "transform"
    else:
        kind = "stage"
    return kind


# 功能：
#   从执行节点生成可视描述，依赖关系形成输入端口，策略和权限保留执行层语义。
#   只有明确列入可选集合的阶段可删除或替换，不能通过编辑器绕过安全节点。
# 输入：
#   node：官方执行拓扑中的节点定义。
# 输出：
#   descriptor：包含双语标题、端口、策略及编辑能力的描述对象。
def _descriptor(node: HarnessNodeSpec) -> HarnessNodeDescriptor:
    protected = node.node_id not in _OPTIONAL_STAGE_IDS
    output_ports = []
    if node.node_id != "mission.evidence-finalize":
        output_ports.append(
            HarnessPortDescriptor(
                port_id="control.out",
                schema_ref=_CONTROL_SCHEMA,
                required=False,
                cardinality="event",
                maximum_connections=64,
            )
        )
    input_ports = [
        HarnessPortDescriptor(
            port_id=f"control.in.{dependency.replace('.', '-')}"[:120],
            schema_ref=_CONTROL_SCHEMA,
            cardinality="event",
        )
        for dependency in node.depends_on
    ]
    allowed = ["move_node", "inspect", "dry_run", "update_node"]
    if not protected:
        allowed.extend(["remove_node", "replace_node"])
    descriptor = HarnessNodeDescriptor(
        descriptor_id=node.node_id,
        title=_TITLE_EN.get(node.node_id, node.node_id),
        title_zh=_TITLE_ZH.get(node.node_id, node.node_id),
        node_kind=_node_kind(node),
        handler_id=node.handler_id,
        runtime_node_kind=node.node_kind,
        required_inputs=list(node.required_inputs),
        output_key=node.output_key,
        input_ports=input_ports,
        output_ports=output_ports,
        policy=HarnessNodePolicy(
            timeout_seconds=node.timeout_seconds,
            retry_limit=node.retry_limit,
            failure_mode=node.failure_mode,
            fallback_handler_id=node.fallback_handler_id,
            cacheable=node.cacheable,
            authority=node.authority,
        ),
        capabilities=HarnessNodeCapabilities(
            removable=not protected,
            replaceable=not protected,
            branchable=node.node_id.startswith("mission.intent-review-"),
            wrappable_in_loop=not protected,
            protected=protected,
            allowed_operations=allowed,
        ),
        category="optional" if not protected else "mission-stage",
        icon=_node_kind(node),
    )
    return descriptor


# 功能：
#   生成阶段、节点及内部策略／槽位的层次目录，并绑定实际节点和已安装插件槽位。
#   第三级用于内部配置，不代表额外执行阶段；目录中的可替换标记不跳过编译校验。
# 输入：
#   descriptors：已知节点标识到描述对象的映射。
#   available_plugin_slots：当前插件目录中存在的槽位集合。
# 输出：
#   output：按阶段及其子项排序的组合目录。
def _composition_items(
    descriptors: dict[str, HarnessNodeDescriptor],
    available_plugin_slots: set[str],
) -> list[HarnessCompositionItem]:
    output: list[HarnessCompositionItem] = []
    for phase_order, phase in enumerate(_COMPOSITION_PHASES):
        # 只为实际存在的节点建目录，避免模板未包含的审查阶段出现空拼图。
        member_ids = [node_id for node_id in phase["nodes"] if node_id in descriptors]
        if not member_ids:
            continue
        stage_ids = [f"composition.stage.{node_id}" for node_id in member_ids]
        phase_slots = sorted(
            {
                slot_id
                for node_id in member_ids
                for slot_id in _STAGE_PLUGIN_SLOTS.get(node_id, ())
                if slot_id in available_plugin_slots
            }
        )
        phase_category, phase_color = _PHASE_CATEGORIES[str(phase["item_id"])]
        output.append(
            HarnessCompositionItem(
                item_id=str(phase["item_id"]),
                level=1,
                kind="phase",
                granularity="large",
                title=str(phase["title"]),
                title_zh=str(phase["title_zh"]),
                description=str(phase["description"]),
                description_zh=str(phase["description_zh"]),
                category_id=phase_category,
                color_token=phase_color,
                visual_kind="puzzle",
                aspect_ratio=_aspect_ratio(str(phase["title"]), str(phase["title_zh"])),
                icon=str(phase["icon"]),
                order=phase_order * 10,
                member_node_ids=member_ids,
                plugin_slot_ids=phase_slots,
                child_item_ids=stage_ids,
                enterable=True,
                replaceable=False,
                protected=all(
                    descriptors[node_id].capabilities.protected for node_id in member_ids
                ),
                scope="phase",
            )
        )
        for stage_order, node_id in enumerate(member_ids):
            descriptor = descriptors[node_id]
            stage_category, stage_color = _NODE_CATEGORIES[descriptor.node_kind]
            stage_item_id = f"composition.stage.{node_id}"
            policy_items = [
                ("timeout", "Timeout", "超时", "clock"),
                ("retry", "Retry", "重试", "refresh"),
                ("failure", "Failure handling", "失败处理", "shield"),
                ("cache", "Node cache", "节点缓存", "database"),
            ]
            slot_ids = [
                slot_id
                for slot_id in _STAGE_PLUGIN_SLOTS.get(node_id, ())
                if slot_id in available_plugin_slots
            ]
            child_ids = [
                f"composition.policy.{node_id}.{policy_id}" for policy_id, *_ in policy_items
            ] + [f"composition.slot.{node_id}.{slot_id}" for slot_id in slot_ids]
            output.append(
                HarnessCompositionItem(
                    item_id=stage_item_id,
                    level=2,
                    parent_item_id=str(phase["item_id"]),
                    kind="stage",
                    granularity="medium",
                    title=descriptor.title,
                    title_zh=descriptor.title_zh,
                    description=f"Executable stage backed by {descriptor.handler_id}.",
                    description_zh=f"由 {descriptor.handler_id} 执行的真实阶段。",
                    category_id=stage_category,
                    color_token=stage_color,
                    visual_kind="model" if descriptor.node_kind == "model_call" else "puzzle",
                    aspect_ratio=(
                        "model-bar"
                        if descriptor.node_kind == "model_call"
                        else _aspect_ratio(descriptor.title, descriptor.title_zh)
                    ),
                    icon=descriptor.icon,
                    order=stage_order * 10,
                    member_node_ids=[node_id],
                    plugin_slot_ids=slot_ids,
                    child_item_ids=child_ids,
                    enterable=True,
                    replaceable=descriptor.capabilities.replaceable,
                    protected=descriptor.capabilities.protected,
                    scope="node",
                )
            )
            for policy_order, (policy_id, title, title_zh, icon) in enumerate(policy_items):
                policy_category, policy_color = _POLICY_CATEGORIES[policy_id]
                output.append(
                    HarnessCompositionItem(
                        item_id=f"composition.policy.{node_id}.{policy_id}",
                        level=3,
                        parent_item_id=stage_item_id,
                        kind="policy",
                        granularity="small",
                        title=title,
                        title_zh=title_zh,
                        category_id=policy_category,
                        color_token=policy_color,
                        visual_kind="puzzle",
                        aspect_ratio=_aspect_ratio(title, title_zh),
                        icon=icon,
                        order=policy_order * 10,
                        member_node_ids=[node_id],
                        enterable=False,
                        replaceable=policy_id in {"timeout", "retry", "cache"}
                        or not descriptor.capabilities.protected,
                        protected=descriptor.capabilities.protected and policy_id == "failure",
                        scope="node",
                    )
                )
            for slot_order, slot_id in enumerate(slot_ids, start=len(policy_items)):
                title, title_zh = _SLOT_TITLES[slot_id]
                slot_category, slot_color = _SLOT_CATEGORIES[slot_id]
                output.append(
                    HarnessCompositionItem(
                        item_id=f"composition.slot.{node_id}.{slot_id}",
                        level=3,
                        parent_item_id=stage_item_id,
                        kind="plugin-slot",
                        granularity="small",
                        title=title,
                        title_zh=title_zh,
                        description="Shared runtime plugin slot used by this executable stage.",
                        description_zh="该执行阶段使用的共享运行时插件槽位。",
                        category_id=slot_category,
                        color_token=slot_color,
                        visual_kind="puzzle",
                        aspect_ratio=_aspect_ratio(title, title_zh),
                        icon="plug",
                        order=slot_order * 10,
                        member_node_ids=[node_id],
                        plugin_slot_ids=[slot_id],
                        enterable=False,
                        replaceable=True,
                        protected=False,
                        scope="workflow",
                    )
                )
    return output


# 功能：
#   对已验证无环的执行拓扑分层，同层节点仅表示依赖已满足，不表示已实际运行。
# 输入：
#   topology：依赖关系经过执行拓扑契约验证的图。
# 输出：
#   output：层内按节点标识稳定排序的节点列表。
def _layers(topology: HarnessTopology) -> list[list[str]]:
    dependencies = {node.node_id: set(node.depends_on) for node in topology.nodes}
    dependants: dict[str, set[str]] = defaultdict(set)
    for node_id, values in dependencies.items():
        for dependency in values:
            dependants[dependency].add(node_id)
    ready = deque(sorted(node_id for node_id, values in dependencies.items() if not values))
    output: list[list[str]] = []
    completed: set[str] = set()
    while ready:
        layer = list(ready)
        ready.clear()
        output.append(layer)
        completed.update(layer)
        candidates = sorted({child for node_id in layer for child in dependants[node_id]})
        for candidate in candidates:
            if candidate not in completed and dependencies[candidate] <= completed:
                ready.append(candidate)
    return output


# 功能：
#   将官方执行拓扑转换为独立编辑候选，保持控制依赖并生成初始分层布局。
# 输入：
#   topology：要呈现的执行拓扑。
#   profile_id：候选所属配置档标识。
#   base_revision：客户端基准编号，用于后续编辑冲突检查。
# 输出：
#   candidate：带可视端口、连线及初始位置的候选配置。
def _candidate_from_topology(
    topology: HarnessTopology,
    *,
    profile_id: str,
    base_revision: int,
) -> HarnessTopologyCandidate:
    descriptors = {node.node_id: _descriptor(node) for node in topology.nodes}
    nodes = [
        HarnessVisualNode.from_descriptor(descriptors[node.node_id]) for node in topology.nodes
    ]
    edges: list[HarnessEdgeBinding] = []
    for node in topology.nodes:
        for dependency in node.depends_on:
            target_port = f"control.in.{dependency.replace('.', '-')}"[:120]
            edges.append(
                HarnessEdgeBinding(
                    edge_id=edge_identity(dependency, "control.out", node.node_id, target_port),
                    source=HarnessEdgeEndpoint(node_id=dependency, port_id="control.out"),
                    target=HarnessEdgeEndpoint(node_id=node.node_id, port_id=target_port),
                    schema_ref=_CONTROL_SCHEMA,
                    binding_mode="control",
                )
            )
    positions: dict[str, HarnessNodePosition] = {}
    for column, layer in enumerate(_layers(topology)):
        for row, node_id in enumerate(layer):
            positions[node_id] = HarnessNodePosition(x=column * 300.0, y=row * 154.0)
    candidate = HarnessTopologyCandidate(
        topology_id=topology.topology_id,
        name=topology.name,
        profile_id=profile_id,
        base_revision=base_revision,
        nodes=nodes,
        edges=edges,
        maximum_parallelism=topology.maximum_parallelism,
        layout=HarnessCanvasLayout(positions=positions),
        metadata={**topology.metadata, "visual_editor": "dronedream.harness-designer.v1"},
    )
    return candidate


class HarnessDesignService:
    """Owns visual revisions and compiles the active one into the mission runtime."""

    # 功能：
    #   打开本机设计历史，首次使用时生成经过校验的默认配置；拒绝链接目录。
    #   锁只协调当前服务实例，不宣称隔离同用户恶意进程或提供跨进程事务。
    # 输入：
    #   self：待初始化实例。
    #   store：应用存储及其规范根目录。
    #   plugin_manager：负责实际插件启停与信任校验的管理器。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, store: AppStore, plugin_manager: PluginManager) -> None:
        self.store = store
        self.plugin_manager = plugin_manager
        self.root = store.root / "harness-designer"
        self.revisions_root = self.root / "revisions"
        self.receipts_root = self.root / "receipts"
        self.index_path = self.root / "index.json"
        self.transition_path = self.root / "transition.json"
        self._lock = threading.RLock()
        for directory in (self.revisions_root, self.receipts_root):
            check_plain_plugin_path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            check_plain_plugin_path(directory)
        with self._lock:
            check_plain_plugin_path(self.index_path)
            if not self.index_path.exists():
                self._initialize()
            else:
                self._upgrade_known_templates()

    # 功能：
    #   识别缺少验证计划阶段的官方旧模板，生成当前模板；不覆盖自定义执行语义。
    # 输入：
    #   record：待核对的旧历史记录。
    # 输出：
    #   candidate：可验证的升级候选；不符合已知迁移条件时为 None。
    def _upgrade_candidate(self, record: HarnessRevision) -> HarnessTopologyCandidate | None:
        template = official_topology_templates().get(record.candidate.topology_id)
        if template is None:
            return None
        legacy_nodes = [
            node.model_copy(update={"depends_on": ["mission.runtime-checkpoints"],
                                    "required_inputs": ["checkpoints"]})
            if node.node_id == "mission.evidence-finalize" else node
            for node in template.nodes if node.node_id != "mission.verification-plan"
        ]
        legacy = _candidate_from_topology(template.model_copy(update={"nodes": legacy_nodes}),
            profile_id=record.candidate.profile_id, base_revision=record.candidate.base_revision)
        # 仅忽略画布与历史编号；节点、端口、连线、策略等执行语义必须完全匹配。
        excluded = {"layout", "base_revision"}
        if legacy.model_dump(exclude=excluded) != record.candidate.model_dump(exclude=excluded):
            return None
        checked = validate_and_compile_harness(record.candidate)
        if (not record.validation.valid
                or checked.semantic_sha256 != record.validation.semantic_sha256
                or checked.layout_sha256 != record.validation.layout_sha256):
            return None
        candidate = _candidate_from_topology(template, profile_id=record.candidate.profile_id,
                                             base_revision=record.revision)
        positions = dict(candidate.layout.positions)
        positions.update(record.candidate.layout.positions)
        candidate.layout = record.candidate.layout.model_copy(update={"positions": positions})
        return candidate

    # 功能：
    #   将已知旧模板发布为新历史版本，全部写入成功后原子切换索引；原文件永久保留。
    #   重启重入不会重复迁移，未识别的配置仍按原有严格校验处理。
    # 输入：
    #   self：持有设计器锁的服务。
    # 输出：
    #   None：不返回业务数据。
    def _upgrade_known_templates(self) -> None:
        if self._transition_pending():
            return
        index = self._read_index()
        replacements = {}
        for number in dict.fromkeys((index["active_revision"], index["current_revision"])):
            try:
                self.get_revision(number)
                continue
            except HarnessDesignServiceError:
                pass
            try:
                record = HarnessRevision.model_validate(self._read_json(self._revision_path(number)), strict=True)
            except (OSError, ValueError):
                continue
            if record.revision != number or (record.parent_revision is not None
                    and not 1 <= record.parent_revision < number):
                continue
            candidate = self._upgrade_candidate(record)
            if candidate is None:
                continue
            validation = validate_and_compile_harness(candidate)
            if not validation.valid:
                raise HarnessDesignServiceError("HARNESS_MIGRATION_COMPILATION_FAILED")
            replacement = HarnessRevision(revision=self._next_revision(), parent_revision=number,
                state=record.state, candidate=candidate, validation=validation,
                created_at=_now(), activated_at=_now() if number == index["active_revision"] else None,
                applies_next_run=True)
            self._save_revision(replacement)
            replacements[number] = replacement.revision
        if not replacements:
            return
        # 回执先于索引发布；断电留下的孤立历史不会替代仍在使用的配置。
        self._record("harness.schema-upgraded", {"migration": "verification-plan-v1",
            "revisions": {str(old): new for old, new in replacements.items()}})
        updated = dict(index)
        for key in ("active_revision", "current_revision"):
            updated[key] = replacements.get(index[key], index[key])
        updated["redo_stack"] = []
        self._save_index(updated)

    # 功能：
    #   将有界严格 JSON 写入独占暂存后发布，历史文件只能新建，索引可原子替换。
    #   清理只移除仍归本次写入所有的暂存，保留失败根因与被替换的外来文件。
    # 输入：
    #   path：经过路径检查的本机目标文件。
    #   value：标准 JSON 数据。
    #   replace：是否允许替换既有目标；历史记录必须为 False。
    # 输出：
    #   None：不返回业务数据。
    @staticmethod
    def _write_json(path: Path, value: Any, *, replace: bool = True) -> None:
        rendered = encode_json(value, limit=_DOCUMENT_BYTES).encode("utf-8")
        check_plain_plugin_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        check_plain_plugin_path(path)
        temporary = None
        owned = None
        error = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                owned = os.fstat(stream.fileno())
                stream.write(rendered)
                stream.flush()
                os.fsync(stream.fileno())
            check_plain_plugin_path(temporary)
            check_plain_plugin_path(path)
            if not os.path.samestat(owned, temporary.stat()):
                raise HarnessDesignServiceError("HARNESS_TEMPORARY_CHANGED")
            if replace:
                os.replace(temporary, path)
            else:
                # 同目录硬链接原子创建目标且拒绝覆盖；随后只回收临时名称。
                os.link(temporary, path)
        except BaseException as failure:
            error = failure
            raise
        finally:
            if temporary is not None and owned is not None:
                try:
                    if temporary.exists():
                        check_plain_plugin_path(temporary)
                        if not os.path.samestat(owned, temporary.stat()):
                            raise HarnessDesignServiceError("HARNESS_TEMPORARY_CHANGED")
                        temporary.unlink()
                except Exception:
                    if error is None:
                        raise
                    error.add_note("HARNESS_TEMPORARY_CLEANUP_FAILED")

    # 功能：
    #   有界稳定读取设计文档，拒绝链接、重复 JSON 键、非有限数和过深结构。
    # 输入：
    #   path：本机设计文档路径。
    # 输出：
    #   value：通过字节、节点及结构预算检查的 JSON 数据。
    @staticmethod
    def _read_json(path: Path) -> Any:
        value = decode_json(read_plugin_file(path, limit=_DOCUMENT_BYTES), limit=_DOCUMENT_BYTES)
        return value

    # 功能：
    #   读取并验证设计索引，禁止布尔、文本或越界编号被静默转为合法历史标识。
    # 输入：
    #   self：当前设计服务。
    # 输出：
    #   value：含当前编号、活动编号和有效重做栈的索引。
    def _read_index(self) -> dict[str, Any]:
        try:
            value = self._read_json(self.index_path)
        except (OSError, ValueError) as error:
            raise HarnessDesignServiceError("HARNESS_INDEX_INVALID") from error
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != "dronedream.harness-revision-index.v1"
            or any(
                type(value.get(key)) is not int or not 1 <= value[key] < 2**63
                for key in ("current_revision", "active_revision")
            )
            or not isinstance(value.get("redo_stack", []), list)
        ):
            raise HarnessDesignServiceError("HARNESS_INDEX_INVALID")
        redo = value.get("redo_stack", [])
        if (
            len(redo) > 100
            or any(type(item) is not int or not 1 <= item < 2**63 for item in redo)
            or len(set(redo)) != len(redo)
        ):
            raise HarnessDesignServiceError("HARNESS_INDEX_INVALID")
        return value

    # 功能：
    #   原子发布索引，活动指针只在候选记录已经保存后更新。
    # 输入：
    #   self：当前设计服务。
    #   value：准备发布的索引快照。
    # 输出：
    #   None：不返回业务数据。
    def _save_index(self, value: dict[str, Any]) -> None:
        self._write_json(self.index_path, value)

    # 功能：
    #   将严格正整数编号映射为规范文件名，拒绝宽松类型转换及超大编号。
    # 输入：
    #   self：当前设计服务。
    #   revision：历史记录编号。
    # 输出：
    #   path：该编号在历史目录中的唯一规范路径。
    def _revision_path(self, revision: int) -> Path:
        if type(revision) is not int or not 1 <= revision < 2**63:
            raise HarnessDesignServiceError("HARNESS_REVISION_INVALID")
        path = self.revisions_root / f"revision-{revision:08d}.json"
        return path

    # 功能：
    #   独占发布新的历史文件，已有编号不可覆盖，失败遗留历史仍保留供核对。
    # 输入：
    #   self：当前设计服务。
    #   revision：已构造的完整历史记录。
    # 输出：
    #   None：不返回业务数据。
    def _save_revision(self, revision: HarnessRevision) -> None:
        self._write_json(
            self._revision_path(revision.revision), revision.model_dump(mode="json"), replace=False
        )

    # 功能：
    #   分配高于所有已保存历史的编号，撤销、拒绝和失败后留下的文件都不能被复用。
    # 输入：
    #   self：当前设计服务。
    # 输出：
    #   revision：下一可用编号；并发占用仍由独占发布拒绝。
    def _next_revision(self) -> int:
        highest = 0
        for path in self.revisions_root.glob("revision-*.json"):
            suffix = path.stem.removeprefix("revision-")
            if not suffix.isascii() or not suffix.isdecimal() or len(suffix) > 19:
                raise HarnessDesignServiceError("HARNESS_REVISION_FILENAME_INVALID")
            number = int(suffix)
            if path != self._revision_path(number):
                raise HarnessDesignServiceError("HARNESS_REVISION_FILENAME_INVALID")
            highest = max(highest, number)
        revision = highest + 1
        self._revision_path(revision)
        return revision

    # 功能：
    #   核对文件身份、递减父链及重新编译结果，拒绝旧摘要或被改写的缓存拓扑。
    #   重编译只验证结构和语义，不运行模型，也不等同于飞行验收。
    # 输入：
    #   self：当前设计服务。
    #   revision：请求读取的严格正整数编号。
    # 输出：
    #   record：与当前编译器和文件身份一致的历史记录。
    def get_revision(self, revision: int) -> HarnessRevision:
        try:
            value = self._read_json(self._revision_path(revision))
            if not isinstance(value, dict) or type(value.get("revision")) is not int:
                raise ValueError("revision identity")
            parent = value.get("parent_revision")
            if parent is not None and (type(parent) is not int or not 1 <= parent < revision):
                raise ValueError("revision ancestry")
            record = HarnessRevision.model_validate(value, strict=True)
            compiled = validate_and_compile_harness(record.candidate)
            if record.revision != revision or record.validation != compiled:
                raise ValueError("revision compilation")
            return record
        except FileNotFoundError as error:
            raise HarnessDesignServiceError("HARNESS_REVISION_NOT_FOUND") from error
        except (OSError, ValueError) as error:
            raise HarnessDesignServiceError("HARNESS_REVISION_INVALID") from error

    # 功能：
    #   保存带唯一身份和 UTC 时间的设计操作回执，既有回执不能覆盖。
    # 输入：
    #   self：当前设计服务。
    #   event：操作结果事件名。
    #   payload：不含长期凭证的结构化操作资料。
    # 输出：
    #   value：本次已经发布的回执数据。
    def _record(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        value = {
            "receipt_id": f"harness-receipt-{uuid4().hex[:24]}",
            "event": event,
            "created_at": _now(),
            "payload": payload,
        }
        self._write_json(self.receipts_root / f"{value['receipt_id']}.json", value, replace=False)
        return value

    # 功能：
    #   首次建立经过编译验证的默认配置，先存历史，再发布活动索引与初始化回执。
    # 输入：
    #   self：尚无索引的设计服务。
    # 输出：
    #   None：不返回业务数据。
    def _initialize(self) -> None:
        topology = official_topology_templates()["topology.balanced-closed-loop"]
        candidate = _candidate_from_topology(
            topology,
            profile_id="harness.profile-balanced",
            base_revision=0,
        )
        validation = validate_and_compile_harness(candidate)
        if not validation.valid:
            raise HarnessDesignServiceError("HARNESS_DEFAULT_TOPOLOGY_INVALID")
        revision = HarnessRevision(
            revision=1,
            state="active",
            candidate=candidate,
            validation=validation,
            created_at=_now(),
            activated_at=_now(),
            applies_next_run=True,
        )
        self._save_revision(revision)
        self._save_index(
            {
                "schema_version": "dronedream.harness-revision-index.v1",
                "current_revision": 1,
                "active_revision": 1,
                "redo_stack": [],
            }
        )
        self._record("harness.initialized", {"revision": 1})

    # 功能：
    #   从真实插件目录筛选配置档槽位，不凭插件名称推断类型。
    # 输入：
    #   self：当前设计服务。
    # 输出：
    #   records：包含活动、健康及信任元数据的配置档记录。
    def _profile_records(self) -> list[dict[str, Any]]:
        records = [
            value
            for value in self.plugin_manager.list_plugins()
            if isinstance(value.get("placement"), dict)
            and value["placement"].get("slot_id") == "harness.profile"
        ]
        return records

    # 功能：
    #   投影配置档公开字段，界面状态来自插件管理器，不在这里执行启停操作。
    # 输入：
    #   self：当前设计服务。
    # 输出：
    #   profiles：适合界面显示的配置档列表。
    def profiles(self) -> list[dict[str, Any]]:
        profiles = [
            {
                "profile_id": str(value["plugin_id"]),
                "name": str(value["name"]),
                "description": str(value["description"]),
                "enabled": bool(value["enabled"]),
                "health": str(value["health"]),
                "trust_status": str(value["trust_status"]),
            }
            for value in self._profile_records()
        ]
        return profiles

    # 功能：
    #   联合官方拓扑模板和当前插件目录构造编辑器目录，绑定组合层次及可用命令。
    #   健康和信任标签只是当前快照，实际启用及运行仍必须由管理器重新校验。
    # 输入：
    #   self：当前设计服务。
    # 输出：
    #   catalog：节点、模板、插件、配置档及组合目录的结构化视图。
    def catalog(self) -> dict[str, Any]:
        templates = official_topology_templates()
        descriptors: dict[str, HarnessNodeDescriptor] = {}
        for topology in templates.values():
            for node in topology.nodes:
                descriptors.setdefault(node.node_id, _descriptor(node))
        harness_plugins: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for value in self.plugin_manager.list_plugins():
            placement = value.get("placement")
            if not isinstance(placement, dict):
                continue
            slot_id = str(placement.get("slot_id", ""))
            if not slot_id.startswith("harness."):
                continue
            harness_plugins.append((value, placement))
        available_slots = {str(placement.get("slot_id", "")) for _, placement in harness_plugins}
        composition_items = _composition_items(descriptors, available_slots)
        owners_by_slot: dict[str, list[str]] = defaultdict(list)
        for item in composition_items:
            for slot_id in item.plugin_slot_ids:
                owners_by_slot[slot_id].append(item.item_id)
        plugins: list[dict[str, Any]] = []
        for value, placement in harness_plugins:
            slot_id = str(placement.get("slot_id", ""))
            large = slot_id in {"harness.profile", "harness.workflow-topology"}
            plugins.append(
                {
                    "plugin_id": str(value["plugin_id"]),
                    "name": str(value["name"]),
                    "description": str(value["description"]),
                    "slot_id": slot_id,
                    "slot_label": str(placement.get("slot_label", slot_id)),
                    "activation_mode": str(placement.get("activation_mode", "single")),
                    "enabled": bool(value["enabled"]),
                    "health": str(value["health"]),
                    "trust_status": str(value["trust_status"]),
                    "version": str(value["version"]),
                    "package_sha256": str(value["package_sha256"]),
                    "granularity": "large" if large else "small",
                    "composition_level": 1 if large else 3,
                    "owner_item_ids": sorted(owners_by_slot.get(slot_id, [])),
                }
            )
        catalog = {
            "schema_version": "dronedream.harness-catalog.v1",
            "node_descriptors": [value.model_dump(mode="json") for value in descriptors.values()],
            "topology_templates": [
                {
                    "topology_id": value.topology_id,
                    "name": value.name,
                    "node_count": len(value.nodes),
                    "maximum_parallelism": value.maximum_parallelism,
                    "metadata": value.metadata,
                }
                for value in templates.values()
            ],
            "plugins": plugins,
            "profiles": self.profiles(),
            "composition_items": [value.model_dump(mode="json") for value in composition_items],
            "context_commands": {
                "canvas": ["auto_layout", "apply_template", "dry_run"],
                "phase": ["enter", "inspect", "dry_run"],
                "stage": ["enter", "inspect", "move_node", "update_node", "dry_run"],
                "plugin_slot": ["inspect", "replace_plugin", "manage_switches"],
                "policy": ["inspect", "update_node"],
                "protected_node": ["inspect", "move_node", "update_node", "dry_run"],
                "optional_node": [
                    "inspect",
                    "move_node",
                    "update_node",
                    "remove_node",
                ],
                "edge": ["inspect", "disconnect"],
            },
        }
        return catalog

    # 功能：
    #   在同一实例锁内读取活动和当前编辑记录，分别显示生效配置与可能被拒绝的草稿。
    # 输入：
    #   self：当前设计服务。
    # 输出：
    #   state：两条配置记录及当前撤销、重做可用性。
    def current(self) -> dict[str, Any]:
        with self._lock:
            index = self._read_index()
            active = self.get_revision(int(index["active_revision"]))
            current = self.get_revision(int(index["current_revision"]))
            state = {
                "active": active.model_dump(mode="json"),
                "current": current.model_dump(mode="json"),
                "can_undo": current.revision != active.revision
                or active.parent_revision is not None,
                "can_redo": bool(index.get("redo_stack")),
                "transition_pending": self._transition_pending(),
            }
            return state

    # 功能：
    #   对候选执行结构及策略校验，不发布历史或触发任何外部执行。
    # 输入：
    #   self：当前设计服务。
    #   candidate：待验证的可视候选配置。
    # 输出：
    #   validation：问题列表、摘要及校验成功时的编译拓扑。
    def validate(self, candidate: HarnessTopologyCandidate) -> HarnessValidationResult:
        validation = validate_and_compile_harness(candidate)
        return validation

    # 功能：
    #   在已验证的候选节点列表中定位明确身份，找不到时返回稳定错误。
    # 输入：
    #   value：候选的独立 JSON 快照。
    #   node_id：目标节点标识。
    # 输出：
    #   located：节点在列表中的索引及节点字典。
    @staticmethod
    def _payload_node(value: dict[str, Any], node_id: str) -> tuple[int, dict[str, Any]]:
        nodes = value["nodes"]
        for index, node in enumerate(nodes):
            if node["node_id"] == node_id:
                located = index, node
                return located
        raise HarnessDesignServiceError("HARNESS_NODE_NOT_FOUND")

    # 功能：
    #   严格解析画布位置，不把字符串或布尔坐标、文本开关转换成合法值。
    # 输入：
    #   payload：含 x、y 及可选 pinned 的布局资料。
    # 输出：
    #   position：坐标有限、开关类型明确的布局字典。
    @staticmethod
    def _position(payload: dict[str, Any]) -> dict[str, Any]:
        position = HarnessNodePosition.model_validate(
            {"x": payload.get("x"), "y": payload.get("y"), "pinned": payload.get("pinned", False)},
            strict=True,
        ).model_dump(mode="json")
        return position

    # 功能：
    #   在独立候选上应用单个编辑，保护必要节点、端口契约和失败关闭策略。
    #   此处只构造候选；是否激活必须经过后续全图编译，不能用局部编辑成功代替验收。
    # 输入：
    #   self：当前设计服务。
    #   candidate：基准历史的候选配置。
    #   operation：包含基准编号和操作资料的编辑请求。
    # 输出：
    #   updated：应用编辑后的新候选，不与原候选共享可变布局。
    def _apply_to_candidate(
        self,
        candidate: HarnessTopologyCandidate,
        operation: HarnessEditOperation,
    ) -> HarnessTopologyCandidate:
        value = candidate.model_dump(mode="json")
        value["base_revision"] = operation.base_revision
        payload = operation.payload
        if operation.operation == "move_node":
            node_id = str(payload.get("node_id", ""))
            self._payload_node(value, node_id)
            value.setdefault("layout", {}).setdefault("positions", {})[node_id] = self._position(
                payload
            )
        elif operation.operation == "update_layout":
            positions = payload.get("positions")
            if not isinstance(positions, dict):
                raise HarnessDesignServiceError("HARNESS_LAYOUT_POSITIONS_REQUIRED")
            known = {node["node_id"] for node in value["nodes"]}
            if not set(positions) <= known:
                raise HarnessDesignServiceError("HARNESS_LAYOUT_NODE_NOT_FOUND")
            for node_id, position in positions.items():
                if not isinstance(position, dict):
                    raise HarnessDesignServiceError("HARNESS_LAYOUT_POSITION_INVALID")
                value.setdefault("layout", {}).setdefault("positions", {})[node_id] = (
                    self._position(position)
                )
        elif operation.operation == "update_node":
            index, node = self._payload_node(value, str(payload.get("node_id", "")))
            policy = payload.get("policy")
            if not isinstance(policy, dict):
                raise HarnessDesignServiceError("HARNESS_NODE_POLICY_REQUIRED")
            merged = {**node["policy"], **policy}
            if node["capabilities"]["protected"]:
                merged["failure_mode"] = "fail-closed"
                merged["fallback_handler_id"] = None
            value["nodes"][index]["policy"] = HarnessNodePolicy.model_validate(
                merged, strict=True
            ).model_dump(mode="json")
        elif operation.operation == "remove_node":
            node_id = str(payload.get("node_id", ""))
            _, node = self._payload_node(value, node_id)
            if not node["capabilities"]["removable"]:
                raise HarnessDesignServiceError("HARNESS_PROTECTED_NODE")
            value["nodes"] = [item for item in value["nodes"] if item["node_id"] != node_id]
            value["edges"] = [
                edge
                for edge in value["edges"]
                if edge["source"]["node_id"] != node_id and edge["target"]["node_id"] != node_id
            ]
            value["layout"]["positions"].pop(node_id, None)
        elif operation.operation == "replace_node":
            node_id = str(payload.get("node_id", ""))
            descriptor_id = str(payload.get("descriptor_id", ""))
            node_index, node = self._payload_node(value, node_id)
            if not node["capabilities"]["replaceable"]:
                raise HarnessDesignServiceError("HARNESS_PROTECTED_NODE")
            descriptor = next(
                (
                    HarnessNodeDescriptor.model_validate(item)
                    for item in self.catalog()["node_descriptors"]
                    if item["descriptor_id"] == descriptor_id
                ),
                None,
            )
            if descriptor is None:
                raise HarnessDesignServiceError("HARNESS_DESCRIPTOR_NOT_FOUND")
            if descriptor.capabilities.protected:
                raise HarnessDesignServiceError("HARNESS_PROTECTED_NODE_CANNOT_BE_ADDED")
            replacement = HarnessVisualNode.from_descriptor(descriptor, node_id=node_id)
            for edge in value["edges"]:
                if edge["source"]["node_id"] == node_id:
                    matching = next(
                        (
                            port
                            for port in replacement.output_ports
                            if port.schema_ref == edge["schema_ref"]
                        ),
                        None,
                    )
                    if matching is None:
                        raise HarnessDesignServiceError("HARNESS_REPLACEMENT_PORT_INCOMPATIBLE")
                    edge["source"]["port_id"] = matching.port_id
                if edge["target"]["node_id"] == node_id:
                    matching = next(
                        (
                            port
                            for port in replacement.input_ports
                            if port.schema_ref == edge["schema_ref"]
                        ),
                        None,
                    )
                    if matching is None:
                        raise HarnessDesignServiceError("HARNESS_REPLACEMENT_PORT_INCOMPATIBLE")
                    edge["target"]["port_id"] = matching.port_id
            value["nodes"][node_index] = replacement.model_dump(mode="json")
        elif operation.operation == "add_node":
            descriptor_id = str(payload.get("descriptor_id", ""))
            descriptor = next(
                (
                    HarnessNodeDescriptor.model_validate(item)
                    for item in self.catalog()["node_descriptors"]
                    if item["descriptor_id"] == descriptor_id
                ),
                None,
            )
            if descriptor is None:
                raise HarnessDesignServiceError("HARNESS_DESCRIPTOR_NOT_FOUND")
            if descriptor.capabilities.protected:
                raise HarnessDesignServiceError("HARNESS_PROTECTED_NODE_CANNOT_BE_ADDED")
            node_id = str(payload.get("node_id") or descriptor_id)
            if any(item["node_id"] == node_id for item in value["nodes"]):
                raise HarnessDesignServiceError("HARNESS_NODE_ALREADY_EXISTS")
            visual_node = HarnessVisualNode.from_descriptor(descriptor, node_id=node_id)
            value["nodes"].append(visual_node.model_dump(mode="json"))
            value["layout"]["positions"][node_id] = self._position(
                {"x": payload.get("x", 0), "y": payload.get("y", 0), "pinned": False}
            )
        elif operation.operation in {"connect", "disconnect"}:
            edge = HarnessEdgeBinding.model_validate(payload.get("edge"))
            if operation.operation == "connect":
                if any(item["edge_id"] == edge.edge_id for item in value["edges"]):
                    raise HarnessDesignServiceError("HARNESS_EDGE_ALREADY_EXISTS")
                value["edges"].append(edge.model_dump(mode="json"))
            else:
                original_count = len(value["edges"])
                value["edges"] = [
                    item for item in value["edges"] if item["edge_id"] != edge.edge_id
                ]
                if len(value["edges"]) == original_count:
                    raise HarnessDesignServiceError("HARNESS_EDGE_NOT_FOUND")
        elif operation.operation == "apply_profile":
            profile_id = str(payload.get("profile_id", ""))
            if profile_id in {
                "harness.profile-evaluation-lab",
                "harness.profile-field-readiness",
            }:
                template_id = "topology.committee-closed-loop"
            elif profile_id == "harness.profile-emergency-response":
                template_id = "topology.rapid-safe"
            else:
                template_id = "topology.balanced-closed-loop"
            updated = _candidate_from_topology(
                official_topology_templates()[template_id],
                profile_id=profile_id,
                base_revision=operation.base_revision,
            )
            return updated
        elif operation.operation == "apply_template":
            template_id = str(payload.get("topology_id", ""))
            templates = official_topology_templates()
            if template_id not in templates:
                raise HarnessDesignServiceError("HARNESS_TEMPLATE_NOT_FOUND")
            updated = _candidate_from_topology(
                templates[template_id],
                profile_id=candidate.profile_id,
                base_revision=operation.base_revision,
            )
            return updated
        else:
            raise HarnessDesignServiceError("HARNESS_OPERATION_NOT_SUPPORTED")
        updated = HarnessTopologyCandidate.model_validate(value)
        return updated

    # 功能：
    #   检查是否存在未完成的跨文件／插件切换，标记损坏或链接也不能当作已完成。
    # 输入：
    #   self：当前设计服务。
    # 输出：
    #   pending：是否必须先通过显式激活完成恢复。
    def _transition_pending(self) -> bool:
        check_plain_plugin_path(self.transition_path)
        pending = self.transition_path.exists()
        return pending

    # 功能：
    #   协调活动指针与插件选择；跨存储切换前持久化标记，全部完成后才解除运行阻断。
    #   中断不会伪称多存储原子提交；显式激活历史配置会重选其配置档和拓扑完成恢复。
    # 输入：
    #   self：已持有实例锁的设计服务。
    #   index：待发布的索引。
    #   target：目标活动记录，必须已重新验证且保存。
    #   event：完成后记录的事件名。
    #   payload：事件资料。
    #   force_profile：是否明确要求重新应用配置档。
    #   force_topology：是否明确要求重新选择拓扑插件。
    # 输出：
    #   receipt：索引和插件选择均完成后的回执。
    def _publish_index(
        self,
        index: dict[str, Any],
        target: HarnessRevision,
        event: str,
        payload: dict[str, Any],
        *,
        force_profile: bool = False,
        force_topology: bool = False,
    ) -> dict[str, Any]:
        previous = self.get_revision(self._read_index()["active_revision"])
        recovering = self._transition_pending()
        select_profile = (
            recovering
            or force_profile
            or previous.candidate.profile_id != target.candidate.profile_id
        )
        select_topology = (
            recovering
            or force_topology
            or select_profile
            or previous.candidate.topology_id != target.candidate.topology_id
        )
        transition = None
        if select_profile or select_topology:
            checksum = target.validation.semantic_sha256
            plugin_by_topology = {
                "topology.balanced-closed-loop": "harness.topology-balanced",
                "topology.committee-closed-loop": "harness.topology-committee",
                "topology.rapid-safe": "harness.topology-rapid-safe",
            }
            topology_plugin = plugin_by_topology.get(target.candidate.topology_id)
            if not target.validation.valid or checksum is None or topology_plugin is None:
                raise HarnessDesignServiceError("HARNESS_ACTIVE_REVISION_INVALID")
            transition = {
                "transition_id": uuid4().hex,
                "from_revision": previous.revision,
                "to_revision": target.revision,
                "created_at": _now(),
            }
            # 标记不作为执行配置；它只禁止部分切换期间创建新的任务快照。
            self._write_json(self.transition_path, transition, replace=recovering)
            if select_profile:
                self.plugin_manager.apply_profile(
                    target.candidate.profile_id,
                    selected_by="agent_harness_designer",
                    harness_revision_sha256=checksum,
                )
            if select_topology:
                self.plugin_manager.enable(
                    topology_plugin,
                    selected_by="agent_harness_designer",
                    harness_revision_sha256=checksum,
                )
        self._save_index(index)
        receipt = self._record(event, payload)
        if transition is not None:
            if self._read_json(self.transition_path) != transition:
                raise HarnessDesignServiceError("HARNESS_TRANSITION_CHANGED")
            self.transition_path.unlink()
        return receipt

    # 功能：
    #   校验客户端基准并保存不可变编辑历史，拒绝的候选保留诊断但不替换活动配置。
    #   合法配置从下一任务起使用，不修改已经冻结的执行任务。
    # 输入：
    #   self：当前设计服务。
    #   operation：待提交的编辑请求。
    # 输出：
    #   result：新历史记录和对应操作回执。
    def apply_operation(self, operation: HarnessEditOperation) -> dict[str, Any]:
        with self._lock:
            if self._transition_pending():
                raise HarnessDesignServiceError("HARNESS_TRANSITION_INCOMPLETE")
            # 请求模型可被 Python 调用者原地改写，持久化前再次走严格 JSON 边界。
            operation = HarnessEditOperation.model_validate(
                decode_json(
                    encode_json(operation.model_dump(mode="python"), limit=_DOCUMENT_BYTES),
                    limit=_DOCUMENT_BYTES,
                ),
                strict=True,
            )
            index = self._read_index()
            current_revision = int(index["current_revision"])
            if operation.base_revision != current_revision:
                raise HarnessDesignServiceError(f"HARNESS_REVISION_CONFLICT:{current_revision}")
            base = self.get_revision(current_revision)
            candidate = self._apply_to_candidate(base.candidate, operation)
            validation = validate_and_compile_harness(candidate)
            # 当前指针可因撤销后退，编号必须来自整个历史目录而不是当前指针加一。
            next_revision = self._next_revision()
            revision = HarnessRevision(
                revision=next_revision,
                parent_revision=current_revision,
                state="active" if validation.valid else "rejected",
                candidate=candidate,
                validation=validation,
                operations=[operation],
                created_at=_now(),
                activated_at=_now() if validation.valid else None,
                applies_next_run=validation.valid,
            )
            self._save_revision(revision)
            index["current_revision"] = next_revision
            if validation.valid:
                index["active_revision"] = next_revision
            index["redo_stack"] = []
            target = revision if validation.valid else self.get_revision(index["active_revision"])
            receipt = self._publish_index(
                index,
                target,
                "harness.edit.accepted" if validation.valid else "harness.edit.rejected",
                {
                    "revision": next_revision,
                    "base_revision": current_revision,
                    "operation": operation.model_dump(mode="json"),
                    "semantic_sha256": validation.semantic_sha256,
                    "layout_sha256": validation.layout_sha256,
                    "issue_codes": [issue.code for issue in validation.issues],
                },
                force_profile=validation.valid and operation.operation == "apply_profile",
                force_topology=validation.valid and operation.operation == "apply_template",
            )
            result = {"revision": revision.model_dump(mode="json"), "receipt": receipt}
            return result

    # 功能：
    #   激活经过重新校验的历史配置，清空重做栈并记录活动指针变化。
    # 输入：
    #   self：当前设计服务。
    #   revision_number：要激活的历史编号。
    # 输出：
    #   result：选中的历史记录及激活回执。
    def activate(self, revision_number: int) -> dict[str, Any]:
        with self._lock:
            revision = self.get_revision(revision_number)
            if not revision.validation.valid:
                raise HarnessDesignServiceError("HARNESS_REVISION_INVALID")
            index = self._read_index()
            index["active_revision"] = revision_number
            # 编号分配已独立于当前指针；激活旧配置后，后续编辑必须以它为基准。
            index["current_revision"] = revision_number
            index["redo_stack"] = []
            receipt = self._publish_index(
                index, revision, "harness.revision.activated", {"revision": revision_number}
            )
            result = {"revision": revision.model_dump(mode="json"), "receipt": receipt}
            return result

    # 功能：
    #   校验基准后撤销当前编辑；跳过被拒绝的父记录，历史文件始终保留。
    #   丢弃拒绝草稿只恢复当前指针，不改变已生效配置。
    # 输入：
    #   self：当前设计服务。
    #   base_revision：客户端预期的当前编号。
    # 输出：
    #   result：恢复后的记录及撤销回执。
    def undo(self, base_revision: int) -> dict[str, Any]:
        self._revision_path(base_revision)
        with self._lock:
            if self._transition_pending():
                raise HarnessDesignServiceError("HARNESS_TRANSITION_INCOMPLETE")
            index = self._read_index()
            current = self.get_revision(int(index["current_revision"]))
            active = self.get_revision(int(index["active_revision"]))
            if current.revision != base_revision:
                raise HarnessDesignServiceError(f"HARNESS_REVISION_CONFLICT:{current.revision}")
            if current.revision != active.revision:
                index["current_revision"] = active.revision
                self._save_index(index)
                receipt = self._record(
                    "harness.revision.discard-rejected",
                    {"from_revision": current.revision, "to_revision": active.revision},
                )
                result = {"revision": active.model_dump(mode="json"), "receipt": receipt}
                return result
            if active.parent_revision is None:
                raise HarnessDesignServiceError("HARNESS_NOTHING_TO_UNDO")
            target = self.get_revision(active.parent_revision)
            while not target.validation.valid and target.parent_revision is not None:
                target = self.get_revision(target.parent_revision)
            if not target.validation.valid:
                raise HarnessDesignServiceError("HARNESS_UNDO_TARGET_INVALID")
            redo = list(index.get("redo_stack", []))
            redo.append(active.revision)
            index["active_revision"] = target.revision
            index["current_revision"] = target.revision
            index["redo_stack"] = redo[-100:]
            receipt = self._publish_index(
                index,
                target,
                "harness.revision.undo",
                {"from_revision": active.revision, "to_revision": target.revision},
            )
            result = {"revision": target.model_dump(mode="json"), "receipt": receipt}
            return result

    # 功能：
    #   校验基准后取出最近一次撤销的有效记录，恢复活动和当前编辑指针。
    # 输入：
    #   self：当前设计服务。
    #   base_revision：客户端预期的当前编号。
    # 输出：
    #   result：恢复的历史配置及重做回执。
    def redo(self, base_revision: int) -> dict[str, Any]:
        self._revision_path(base_revision)
        with self._lock:
            if self._transition_pending():
                raise HarnessDesignServiceError("HARNESS_TRANSITION_INCOMPLETE")
            index = self._read_index()
            if int(index["current_revision"]) != base_revision:
                raise HarnessDesignServiceError(
                    f"HARNESS_REVISION_CONFLICT:{index['current_revision']}"
                )
            redo = list(index.get("redo_stack", []))
            if not redo:
                raise HarnessDesignServiceError("HARNESS_NOTHING_TO_REDO")
            target = self.get_revision(int(redo.pop()))
            if not target.validation.valid:
                raise HarnessDesignServiceError("HARNESS_REDO_TARGET_INVALID")
            index["active_revision"] = target.revision
            index["current_revision"] = target.revision
            index["redo_stack"] = redo
            receipt = self._publish_index(
                index,
                target,
                "harness.revision.redo",
                {"from_revision": base_revision, "to_revision": target.revision},
            )
            result = {"revision": target.model_dump(mode="json"), "receipt": receipt}
            return result

    # 功能：
    #   结构预演候选或当前活动配置，展示依赖层次与预计调用节点，不实际调用外部系统。
    # 输入：
    #   self：当前设计服务。
    #   candidate：可选候选；省略时读取重新校验过的活动历史。
    # 输出：
    #   result：结构验证结果和外部调用次数为零的预演报告。
    def dry_run(self, candidate: HarnessTopologyCandidate | None = None) -> dict[str, Any]:
        if candidate is None:
            with self._lock:
                candidate = self.get_revision(self._read_index()["active_revision"]).candidate
        validation = validate_and_compile_harness(candidate)
        topology = validation.compiled_topology
        layers = _layers(topology) if topology is not None else []
        model_nodes = [node.node_id for node in candidate.nodes if node.node_kind == "model_call"]
        tool_nodes = [node.node_id for node in candidate.nodes if node.node_kind == "tool_call"]
        result = {
            "schema_version": "dronedream.harness-dry-run.v1",
            "valid": validation.valid,
            "validation": validation.model_dump(mode="json"),
            "layers": layers,
            "node_count": len(candidate.nodes),
            "edge_count": len(candidate.edges),
            "protected_node_count": sum(node.capabilities.protected for node in candidate.nodes),
            "projected_model_nodes": model_nodes,
            "projected_tool_nodes": tool_nodes,
            "external_calls_executed": 0,
            "note": "Structural dry run only; no model, tool, simulator, or vehicle call was made.",
        }
        return result

    # 功能：
    #   读取有界回执并按真实 UTC 时间返回最近记录，损坏条目不冒充有效回执。
    # 输入：
    #   self：当前设计服务。
    #   limit：请求条数，严格整数并裁剪至 1 到 500。
    # 输出：
    #   values：最新回执在前的列表。
    def receipts(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if type(limit) is not int:
            raise HarnessDesignServiceError("HARNESS_RECEIPT_LIMIT_INVALID")
        bounded_limit = max(1, min(limit, 500))
        ranked: list[tuple[datetime, str, dict[str, Any]]] = []
        for path in self.receipts_root.glob("harness-receipt-*.json"):
            try:
                value = self._read_json(path)
                if not isinstance(value, dict) or value.get("receipt_id") != path.stem:
                    continue
                created_at = datetime.fromisoformat(value["created_at"])
                if created_at.utcoffset() is None:
                    continue
            except (OSError, ValueError, TypeError, KeyError):
                continue
            ranked.append((created_at, path.stem, value))
            # 回执编号是随机数，不能据其字典序判断时间；只保留请求需要的最近条目。
            ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
            del ranked[bounded_limit:]
        values = [entry[2] for entry in ranked]
        return values

    # 功能：
    #   为新任务冻结已经重新编译核对的活动配置，返回独立拓扑及精确摘要。
    #   冻结不授予插件调用或飞行权限，运行阶段仍需执行各自的权限与安全校验。
    # 输入：
    #   self：当前设计服务。
    # 输出：
    #   frozen：历史编号、语义与布局摘要、配置档、执行拓扑及冻结时间。
    def freeze_active_for_task(self) -> dict[str, Any]:
        with self._lock:
            if self._transition_pending():
                raise HarnessDesignServiceError("HARNESS_TRANSITION_INCOMPLETE")
            revision = self.get_revision(int(self._read_index()["active_revision"]))
            topology = revision.validation.compiled_topology
            if topology is None or not revision.validation.valid:
                raise HarnessDesignServiceError("HARNESS_ACTIVE_REVISION_INVALID")
            frozen = {
                "revision": revision.revision,
                "semantic_sha256": revision.validation.semantic_sha256,
                "layout_sha256": revision.validation.layout_sha256,
                "profile_id": revision.candidate.profile_id,
                "topology": topology.model_dump(mode="json"),
                "frozen_at": _now(),
            }
            return frozen


__all__ = ["HarnessDesignService", "HarnessDesignServiceError"]
