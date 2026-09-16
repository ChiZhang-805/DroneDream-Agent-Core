"""Declarative execution policies consumed and enforced by the core Harness.

The event hooks return envelopes/receipts. Durable recording is performed by the
caller; a successful hook is neither a delivered actuator command nor a disk commit.
"""

from __future__ import annotations

from typing import Any

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_plugin_sdk.protocol import copy_json

from ._helpers import hook_plugin


# 功能：
#   注册时冻结策略内容，并为每次任务读取提供独立配置，避免编辑器或使用者反向修改。
# 输入：
#   payload：准备登记的有界 JSON 策略对象。
# 输出：
#   resolve：读取冻结策略副本的钩子。
def _constant(payload: dict[str, object]):
    frozen = copy_json(payload)

    # 功能：
    #   返回策略数据副本，不与原配置或其他调用共享可变字段。
    # 输入：
    #   _：常量策略不使用的调度参数。
    # 输出：
    #   policy：本次调用独立的策略值。
    def resolve(**_: Any) -> dict[str, object]:
        policy = copy_json(frozen)
        return policy

    return resolve


# 功能：
#   构造带内容摘要的进程内事件封装，不宣称已远端投递、落盘或触发执行器。
# 输入：
#   event：生命周期事件名称。
#   payload：事件的业务内容。
#   _：事件封装器不使用的扩展参数。
# 输出：
#   envelope：内容独立且与摘要绑定的事件封装。
def _event_bus(*, event: str, payload: dict[str, object], **_: Any) -> dict[str, object]:
    detached = copy_json(payload)
    envelope = {
        "accepted": True,
        "transport": "in-process",
        "event": event,
        "payload": detached,
        "payload_sha256": sha256_json(detached),
    }
    return envelope


# 功能：
#   为有限事件载荷生成摘要绑定的观察回执，由调用方负责持久化，空载荷不标为接受。
# 输入：
#   event：被观察事件的名称。
#   payload：用于绑定摘要的事件内容。
#   _：观察器不使用的扩展参数。
# 输出：
#   receipt：含事件名、内容摘要及接受标记的观察回执。
def _observer(*, event: str, payload: dict[str, object], **_: Any) -> dict[str, object]:
    detached = copy_json(payload)
    receipt = {
        "observer": "execution-ledger",
        "event": event,
        "payload_sha256_required": True,
        "payload_sha256": sha256_json(detached),
        "accepted": bool(detached),
    }
    return receipt


# 功能：
#   注册可替换的独占策略插槽，冻结钩子返回值与目录元数据，不绕过宿主强制边界。
# 输入：
#   plugin_id：策略插件标识。
#   name：展示名称。
#   description：用途说明。
#   kind：策略能力类别。
#   slot_id：目标插槽标识。
#   slot_label：插槽展示名称。
#   hook_name：宿主读取策略时调用的钩子。
#   payload：策略参数对象。
#   order：插槽与插件排序值。
#   enabled：是否默认启用。
# 输出：
#   definition：下次任务生效、失败关闭的策略插件定义。
def _policy(
    *,
    plugin_id: str,
    name: str,
    description: str,
    kind: str,
    slot_id: str,
    slot_label: str,
    hook_name: str,
    payload: dict[str, object],
    order: int,
    enabled: bool,
) -> PluginDefinition:
    definition = hook_plugin(
        module_name=__name__,
        plugin_id=plugin_id,
        name=name,
        description=description,
        capability_id=f"{plugin_id}.resolve",
        capability_kind=kind,
        capability_name=name,
        capability_description=description,
        category_id="harness",
        category_label="Harness 与智能体",
        slot_id=slot_id,
        slot_label=slot_label,
        activation_mode="single",
        category_order=10,
        slot_order=order,
        plugin_order=order,
        hooks={hook_name: _constant(payload)},
        default_enabled=enabled,
        failure_mode="fail-closed",
        swap_policy="next-mission",
        metadata=copy_json(payload),
    )
    return definition


# 功能：
#   声明调度、重试、超时、调用预算、降级、缓存以及事件与观察策略，供宿主读取和执行。
# 输入：
#   无。
# 输出：
#   definitions：各策略及进程内事件钩子的插件定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions = [
        _policy(
            plugin_id="harness.scheduler-parallel-ready",
            name="就绪节点并行调度",
            description="并行运行依赖已满足的独立节点，核心安全门仍保持顺序。",
            kind="harness-scheduler",
            slot_id="harness.scheduler",
            slot_label="Harness 调度器",
            hook_name="resolve_schedule",
            payload={"strategy": "parallel-ready", "maximum_parallelism": 4},
            order=30,
            enabled=True,
        ),
        _policy(
            plugin_id="harness.scheduler-sequential",
            name="完全顺序调度",
            description="适用于调试与严格复现，每次只运行一个 Harness 节点。",
            kind="harness-scheduler",
            slot_id="harness.scheduler",
            slot_label="Harness 调度器",
            hook_name="resolve_schedule",
            payload={"strategy": "sequential", "maximum_parallelism": 1},
            order=31,
            enabled=False,
        ),
        _policy(
            plugin_id="harness.retry-bounded-exponential",
            name="有界指数重试",
            description="仅对可恢复故障执行有界重试，并记录每一次尝试。",
            kind="retry-policy",
            slot_id="harness.retry-policy",
            slot_label="重试策略",
            hook_name="resolve_retry",
            payload={
                "maximum_retries": 24,
                "provider_attempts": 3,
                "backoff": "exponential",
                "jitter": True,
            },
            order=40,
            enabled=True,
        ),
        _policy(
            plugin_id="harness.retry-immediate-once",
            name="快速单次重试",
            description="低延迟任务只进行一次立即重试。",
            kind="retry-policy",
            slot_id="harness.retry-policy",
            slot_label="重试策略",
            hook_name="resolve_retry",
            payload={
                "maximum_retries": 1,
                "provider_attempts": 2,
                "backoff": "none",
                "jitter": False,
            },
            order=41,
            enabled=False,
        ),
        _policy(
            plugin_id="harness.timeout-adaptive",
            name="阶段自适应超时",
            description="为模型、工具和本地验证分别设置边界明确的超时时间。",
            kind="timeout-policy",
            slot_id="harness.timeout-policy",
            slot_label="超时策略",
            hook_name="resolve_timeout",
            payload={
                "model_seconds": 180.0,
                "tool_seconds": 180.0,
                "local_stage_seconds": 30.0,
                "mission_seconds": 900.0,
            },
            order=50,
            enabled=True,
        ),
        _policy(
            plugin_id="harness.timeout-low-latency",
            name="低延迟超时",
            description="面向应急预览缩短模型和工具等待时间，超时后进入安全回退。",
            kind="timeout-policy",
            slot_id="harness.timeout-policy",
            slot_label="超时策略",
            hook_name="resolve_timeout",
            payload={
                "model_seconds": 45.0,
                "tool_seconds": 20.0,
                "local_stage_seconds": 10.0,
                "mission_seconds": 300.0,
            },
            order=51,
            enabled=False,
        ),
        _policy(
            plugin_id="harness.budget-balanced",
            name="均衡调用预算",
            description="限制模型、工具、节点、重试和并行度，避免无限循环。",
            kind="budget-policy",
            slot_id="harness.budget-policy",
            slot_label="调用预算",
            hook_name="resolve_budget",
            payload={
                "maximum_model_calls": 48,
                "maximum_tool_calls": 16,
                "maximum_nodes": 128,
                "maximum_retries": 24,
                "maximum_parallelism": 4,
            },
            order=60,
            enabled=True,
        ),
        _policy(
            plugin_id="harness.budget-cost-capped",
            name="费用封顶预算",
            description="减少模型审查和顾问工具调用，同时保留所有安全门。",
            kind="budget-policy",
            slot_id="harness.budget-policy",
            slot_label="调用预算",
            hook_name="resolve_budget",
            payload={
                "maximum_model_calls": 16,
                "maximum_tool_calls": 4,
                "maximum_nodes": 96,
                "maximum_retries": 8,
                "maximum_parallelism": 2,
            },
            order=61,
            enabled=False,
        ),
        _policy(
            plugin_id="harness.fallback-safe-degrade",
            name="安全降级回退",
            description="可选顾问失败时隔离，模型或安全节点失败时悬停并停止准备。",
            kind="fallback-policy",
            slot_id="harness.fallback-policy",
            slot_label="回退策略",
            hook_name="resolve_fallback",
            payload={
                "optional_failure": "isolate",
                "planning_failure": "stop",
                "runtime_failure": "safe-hold",
                "may_bypass_core_gate": False,
            },
            order=70,
            enabled=True,
        ),
        _policy(
            plugin_id="harness.cache-mission-hash",
            name="任务哈希缓存",
            description="只复用输入、插件快照和资产哈希完全一致的只读结果。",
            kind="cache-policy",
            slot_id="harness.cache-policy",
            slot_label="缓存策略",
            hook_name="resolve_cache",
            payload={
                "strategy": "mission-hash",
                "cache_read_only": True,
                "cache_model_outputs": False,
                "cache_safety_authorization": False,
            },
            order=80,
            enabled=True,
        ),
    ]
    definitions.extend(
        [
            hook_plugin(
                module_name=__name__,
                plugin_id="harness.event-bus-in-process",
                name="进程内任务事件总线",
                description="在任务准备链中传递哈希绑定事件，不跨越进程权限边界。",
                capability_id="harness.event-bus-in-process.transport",
                capability_kind="event-bus",
                capability_name="进程内事件传输",
                capability_description="传输结构化 Harness 生命周期事件。",
                category_id="harness",
                category_label="Harness 与智能体",
                slot_id="harness.event-bus",
                slot_label="事件总线",
                activation_mode="single",
                category_order=10,
                slot_order=90,
                plugin_order=10,
                hooks={"transport_message": _event_bus},
                default_enabled=True,
                failure_mode="fail-closed",
                swap_policy="next-mission",
            ),
            hook_plugin(
                module_name=__name__,
                plugin_id="harness.observer-execution-ledger",
                name="Harness 执行账本",
                description="记录拓扑、阶段与策略事件，用于回放和审计。",
                capability_id="harness.observer-execution-ledger.observe",
                capability_kind="observer",
                capability_name="执行账本观测器",
                capability_description="为 Harness 事件生成可审计观测记录。",
                category_id="harness",
                category_label="Harness 与智能体",
                slot_id="harness.observers",
                slot_label="Harness 观测器",
                activation_mode="multiple",
                category_order=10,
                slot_order=100,
                plugin_order=10,
                hooks={"observe_harness": _observer},
                default_enabled=True,
                failure_mode="isolate",
                swap_policy="anytime",
            ),
        ]
    )
    return definitions
