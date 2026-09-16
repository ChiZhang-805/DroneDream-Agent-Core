"""Preparation DAG templates, distinct from the continuous onboard/local control loop."""

from __future__ import annotations

from typing import Any

from dronedream_agent_core.model_harness.graph import HarnessNodeSpec, HarnessTopology
from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_plugin_sdk.protocol import copy_json

from ._helpers import hook_plugin


# 功能：
#   构建从请求到任务准备完成的依赖图，按指定数量安排意图审查，并保留合同、净空和证据门。
# 输入：
#   review_count：并列意图审查节点数，严格限制为 1 至 3 的整数。
#   optional_advisors：是否加入可隔离失败的工具顾问阶段。
# 输出：
#   nodes：按依赖关系描述准备流程的节点列表，不是实时飞控执行循环。
def _core_nodes(*, review_count: int, optional_advisors: bool) -> list[HarnessNodeSpec]:
    if type(review_count) is not int or not 1 <= review_count <= 3:
        raise ValueError("WORKFLOW_REVIEW_COUNT_INVALID")
    if type(optional_advisors) is not bool:
        raise ValueError("WORKFLOW_ADVISOR_POLICY_INVALID")
    nodes = [
        HarnessNodeSpec(
            node_id="mission.request-ingest",
            node_kind="core",
            handler_id="core.request-ingest",
            required_inputs=["request"],
            output_key="request_context",
        ),
        HarnessNodeSpec(
            node_id="mission.context-prepare",
            node_kind="plugin",
            handler_id="core.context-prepare",
            depends_on=["mission.request-ingest"],
            required_inputs=["request_context"],
            output_key="context",
            cacheable=True,
        ),
        HarnessNodeSpec(
            node_id="mission.intent-parse",
            node_kind="core",
            handler_id="core.intent-parse",
            depends_on=["mission.context-prepare"],
            required_inputs=["context"],
            output_key="intent",
            retry_limit=2,
        ),
    ]
    reviews: list[str] = []
    # 多次审查分别产出，合同冻结只能依赖随后的汇合门，不能直接消费单次审查。
    for index in range(1, review_count + 1):
        node_id = f"mission.intent-review-{index}"
        reviews.append(node_id)
        nodes.append(
            HarnessNodeSpec(
                node_id=node_id,
                node_kind="core",
                handler_id="core.intent-review",
                depends_on=["mission.intent-parse"],
                required_inputs=["intent"],
                output_key=f"intent_review_{index}",
                retry_limit=1,
            )
        )
    nodes.extend(
        [
            HarnessNodeSpec(
                node_id="mission.intent-consensus",
                node_kind="barrier",
                handler_id="core.intent-consensus",
                depends_on=reviews,
                output_key="accepted_intent",
            ),
            HarnessNodeSpec(
                node_id="mission.contract-freeze",
                node_kind="barrier",
                handler_id="core.contract-freeze",
                depends_on=["mission.intent-consensus"],
                required_inputs=["accepted_intent"],
                output_key="contract",
            ),
        ]
    )
    task_dependencies = ["mission.contract-freeze"]
    if optional_advisors:
        nodes.append(
            HarnessNodeSpec(
                node_id="mission.tool-advice",
                node_kind="plugin",
                handler_id="core.tool-advice",
                depends_on=["mission.contract-freeze"],
                required_inputs=["contract"],
                output_key="tool_advice",
                failure_mode="isolate",
                retry_limit=1,
            )
        )
        task_dependencies.append("mission.tool-advice")
    nodes.extend(
        [
            HarnessNodeSpec(
                node_id="mission.task-decompose",
                node_kind="core",
                handler_id="core.task-decompose",
                depends_on=task_dependencies,
                required_inputs=["contract"],
                output_key="task_graph",
                retry_limit=2,
            ),
            HarnessNodeSpec(
                node_id="mission.semantic-plan",
                node_kind="core",
                handler_id="core.semantic-plan",
                depends_on=["mission.task-decompose"],
                required_inputs=["task_graph"],
                output_key="semantic_plan",
                retry_limit=3,
            ),
            HarnessNodeSpec(
                node_id="mission.route-resolve",
                node_kind="core",
                handler_id="core.route-resolve",
                depends_on=["mission.semantic-plan"],
                required_inputs=["semantic_plan"],
                output_key="route",
            ),
            HarnessNodeSpec(
                node_id="mission.clearance-gate",
                node_kind="barrier",
                handler_id="core.clearance-gate",
                depends_on=["mission.route-resolve"],
                required_inputs=["route"],
                output_key="clearance",
            ),
            HarnessNodeSpec(
                node_id="mission.track-export",
                node_kind="core",
                handler_id="core.track-export",
                depends_on=["mission.clearance-gate"],
                required_inputs=["clearance"],
                output_key="track",
            ),
            HarnessNodeSpec(
                node_id="mission.plan-evaluation",
                node_kind="plugin",
                handler_id="core.plan-evaluation",
                depends_on=["mission.track-export"],
                required_inputs=["track"],
                output_key="evaluation",
                failure_mode="isolate",
            ),
            HarnessNodeSpec(
                node_id="mission.plan-review",
                node_kind="core",
                handler_id="core.plan-review",
                depends_on=["mission.plan-evaluation"],
                required_inputs=["track"],
                output_key="plan_review",
                retry_limit=2,
            ),
            HarnessNodeSpec(
                node_id="mission.runtime-checkpoints",
                node_kind="core",
                handler_id="core.runtime-checkpoints",
                depends_on=["mission.plan-review"],
                required_inputs=["plan_review"],
                output_key="checkpoints",
            ),
            HarnessNodeSpec(
                node_id="mission.verification-plan",
                node_kind="barrier",
                handler_id="core.verification-plan",
                depends_on=["mission.runtime-checkpoints"],
                required_inputs=["checkpoints", "runtime_actions"],
                output_key="verification_plan",
            ),
            HarnessNodeSpec(
                node_id="mission.evidence-finalize",
                node_kind="barrier",
                handler_id="core.evidence-finalize",
                depends_on=["mission.verification-plan"],
                required_inputs=["verification_plan"],
                output_key="prepared_mission",
            ),
        ]
    )
    return nodes


# 功能：
#   严格验证并冻结拓扑说明，避免编辑器之后的修改改变已安装的工作流选择。
# 输入：
#   topology：准备注册的类型化拓扑模板。
# 输出：
#   resolve：每次返回独立拓扑描述的解析钩子。
def _resolve(topology: HarnessTopology):
    frozen = copy_json(
        HarnessTopology.model_validate(topology.model_dump(mode="python"), strict=True).model_dump(
            mode="json"
        )
    )

    # 功能：
    #   返回冻结拓扑的独立副本，由宿主调度器负责执行其节点。
    # 输入：
    #   _：此静态模板不使用的调度参数。
    # 输出：
    #   description：本次读取的独立 DAG 描述。
    def resolve(**_: Any) -> dict[str, object]:
        description = copy_json(frozen)
        return description

    return resolve


# 功能：
#   为一个拓扑注册互斥、下次任务生效的插件，附带保护门与并行度说明。
# 输入：
#   plugin_id：拓扑插件标识。
#   name：展示名称。
#   description：用途说明。
#   topology：需要冻结的工作流拓扑。
#   order：同插槽内的展示顺序。
#   enabled：是否默认启用本模板。
# 输出：
#   definition：含冻结解析钩子和拓扑元数据的插件定义。
def _definition(
    *,
    plugin_id: str,
    name: str,
    description: str,
    topology: HarnessTopology,
    order: int,
    enabled: bool,
) -> PluginDefinition:
    definition = hook_plugin(
        module_name=__name__,
        plugin_id=plugin_id,
        name=name,
        description=description,
        capability_id=f"{plugin_id}.resolve",
        capability_kind="workflow-topology",
        capability_name=name,
        capability_description=description,
        category_id="harness",
        category_label="Harness 与智能体",
        slot_id="harness.workflow-topology",
        slot_label="工作流拓扑",
        activation_mode="single",
        category_order=10,
        slot_order=20,
        plugin_order=order,
        hooks={"resolve_topology": _resolve(topology)},
        default_enabled=enabled,
        failure_mode="fail-closed",
        swap_policy="next-mission",
        metadata={
            "topology_id": topology.topology_id,
            "node_count": len(topology.nodes),
            "maximum_parallelism": topology.maximum_parallelism,
            "protected_barriers": [
                item.node_id for item in topology.nodes if item.node_kind == "barrier"
            ],
        },
    )
    return definition


# 功能：
#   创建均衡、委员会和快速安全三种独立模板，改变审查密度而不移除受保护门禁。
# 输入：
#   无。
# 输出：
#   templates：拓扑标识到独立模型对象的映射。
def official_topology_templates() -> dict[str, HarnessTopology]:
    values = [
        HarnessTopology(
            topology_id="topology.balanced-closed-loop",
            name="Balanced closed loop",
            nodes=_core_nodes(review_count=1, optional_advisors=True),
            maximum_parallelism=4,
            metadata={"review_strategy": "single-specialist"},
        ),
        HarnessTopology(
            topology_id="topology.committee-closed-loop",
            name="Committee closed loop",
            nodes=_core_nodes(review_count=3, optional_advisors=True),
            maximum_parallelism=6,
            metadata={"review_strategy": "three-way-consensus"},
        ),
        HarnessTopology(
            topology_id="topology.rapid-safe",
            name="Rapid safe closed loop",
            nodes=_core_nodes(review_count=1, optional_advisors=False),
            maximum_parallelism=3,
            metadata={"review_strategy": "latency-bounded"},
        ),
    ]
    templates = {value.topology_id: value.model_copy(deep=True) for value in values}
    return templates


# 功能：
#   将三种官方拓扑登记为可选择插件，仅默认启用均衡方案。
# 输入：
#   无。
# 输出：
#   definitions：均衡、委员会和快速安全拓扑的插件定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    templates = official_topology_templates()
    definitions = [
        _definition(
            plugin_id="harness.topology-balanced",
            name="均衡闭环拓扑",
            description="完整执行意图、合同、任务、路线、净空、轨迹、审查和证据门。",
            topology=templates["topology.balanced-closed-loop"],
            order=10,
            enabled=True,
        ),
        _definition(
            plugin_id="harness.topology-committee",
            name="委员会并行审查拓扑",
            description="并行执行三次独立意图审查，并在合同冻结前形成一致结论。",
            topology=templates["topology.committee-closed-loop"],
            order=20,
            enabled=False,
        ),
        _definition(
            plugin_id="harness.topology-rapid-safe",
            name="快速安全拓扑",
            description="省略非关键顾问阶段，但保留合同、净空、计划审查和证据安全门。",
            topology=templates["topology.rapid-safe"],
            order=30,
            enabled=False,
        ),
    ]
    return definitions


__all__ = ["official_topology_templates", "plugin_definitions"]
