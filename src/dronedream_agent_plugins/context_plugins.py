"""Bounded conversation views and policies, distinct from account-memory consent.

Extractive summaries are intentionally lossy and are never flight authority.
The core store owns persistence; these hooks must not mutate its event payloads.
"""

from __future__ import annotations

from typing import Any

from dronedream_agent_core.contracts import ConversationWindow, MapAsset, MapCatalog, MissionRequest
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_plugin_sdk.protocol import copy_json

from ._helpers import hook_plugin


# 功能：
#   重新严格校验事件窗口，在摘要和哈希之前拒绝混入其他对话、乱序游标或非法 JSON 载荷。
# 输入：
#   window：准备供上下文处理使用的对话窗口。
# 输出：
#   validated：身份一致、序号递增且载荷可有界编码的窗口副本。
def _validated_window(window: ConversationWindow) -> ConversationWindow:
    validated = ConversationWindow.model_validate(window.model_dump(mode="python"), strict=True)
    previous = 0
    for event in validated.recent_events:
        if event.conversation_id != validated.conversation_id or event.sequence <= previous:
            raise ValueError("CONTEXT_WINDOW_IDENTITY_OR_ORDER_INVALID")
        copy_json(event.payload)
        previous = event.sequence
    return validated


# 功能：
#   保留近期结构化事件，将工具大结果缩减为状态和摘要引用，不修改存储中的原始载荷。
# 输入：
#   window：待压缩对话窗口。
# 输出：
#   context：摘要与近期事件组成的独立上下文对象。
def _structured_window(*, window: ConversationWindow) -> dict[str, object]:
    window = _validated_window(window)
    recent: list[dict[str, object]] = []
    for event in window.recent_events:
        payload = copy_json(event.payload)
        if event.role == "tool":
            payload = {
                key: payload.get(key)
                for key in (
                    "tool_id",
                    "outcome",
                    "input_sha256",
                    "output_sha256",
                    "issue_codes",
                )
            }
        recent.append(
            {
                "sequence": event.sequence,
                "role": event.role,
                "event_type": event.event_type,
                "payload": payload,
            }
        )
    context = {
        "strategy": "structured-window",
        "conversation_id": window.conversation_id,
        "summary": window.summary,
        "recent_events": recent,
    }
    return context


# 功能：
#   导出有序事件身份和载荷摘要，不把摘要引用误当成已经读入的完整制品。
# 输入：
#   window：待组织的对话事件窗口。
# 输出：
#   ledger：包含对话摘要与事件哈希列表的账本视图。
def _event_ledger(*, window: ConversationWindow) -> dict[str, object]:
    window = _validated_window(window)
    ledger = {
        "strategy": "event-ledger",
        "conversation_id": window.conversation_id,
        "summary": window.summary,
        "events": [
            {
                "sequence": event.sequence,
                "role": event.role,
                "event_type": event.event_type,
                "artifact_sha256": sha256_json(event.payload),
            }
            for event in window.recent_events
        ],
    }
    return ledger


# 功能：
#   加入选定地图的规模及实体别名索引；语义别名不是当前障碍物观测或几何重建结果。
# 输入：
#   value：已有上下文。
#   map_catalog：上游选定并绑定的地图语义目录。
#   map_graph：对应任务地图。
#   _：本增强器不使用的扩展参数。
# 输出：
#   context：包含独立实体别名列表的增强上下文。
def _map_context(
    *, value: dict[str, object], map_catalog: MapCatalog, map_graph: MapAsset, **_: Any
) -> dict[str, object]:
    context = {
        **copy_json(value),
        "map_context": {
            "asset_id": map_graph.asset_id,
            "node_count": len(map_graph.nodes),
            "edge_count": len(map_graph.edges),
            "entities": [
                {"entity_id": item.entity_id, "aliases": list(item.aliases)}
                for item in map_catalog.entities
            ],
        },
    }
    return context


# 功能：
#   加入本轮请求身份与记忆政策元数据；读取这些元数据不等于取得账户记忆访问许可。
# 输入：
#   value：已有上下文。
#   request：本次任务请求。
#   _：本增强器不使用的扩展参数。
# 输出：
#   context：包含对话标识、语言、起点和消息规模的独立上下文。
def _request_context(
    *, value: dict[str, object], request: MissionRequest, **_: Any
) -> dict[str, object]:
    memory_policy = request.input_metadata.get("memory_policy")
    context = {
        **copy_json(value),
        "request_context": {
            "conversation_id": request.conversation_id,
            "locale": request.locale,
            "start_entity": request.start_entity,
            "message_length": len(request.message),
            "memory_policy": copy_json(memory_policy) if isinstance(memory_policy, dict) else {},
        },
    }
    return context


# 功能：
#   在模型上下文中重申执行权限和未知状态处理边界；真正执行约束仍由核心代码实现。
# 输入：
#   value：已有上下文。
#   _：本增强器不使用的扩展参数。
# 输出：
#   context：附加不可放宽安全说明的独立上下文。
def _safety_context(*, value: dict[str, object], **_: Any) -> dict[str, object]:
    context = {
        **copy_json(value),
        "non_relaxable_context": {
            "model_has_actuator_authority": False,
            "unknown_critical_state": "hold-or-abort",
            "runtime_change_requires_safe_hold": True,
        },
    }
    return context


# 功能：
#   声明持久化上下文存储策略，不在此钩子中创建数据库或把供应商上下文当作记忆库。
# 输入：
#   _：固定存储策略不使用的扩展参数。
# 输出：
#   policy：SQLite WAL 后端及持久化要求。
def _sqlite_wal_store(**_: Any) -> dict[str, object]:
    policy = {
        "backend": "sqlite-wal",
        "durable": True,
        "provider_context_is_not_memory": True,
    }
    return policy


# 功能：
#   选择短近期窗口及摘要，实际按对话隔离的读取由上下文存储执行。
# 输入：
#   _：固定检索策略不使用的扩展参数。
# 输出：
#   policy：最多二十四条近期事件并包含摘要的检索配置。
def _balanced_retrieval(**_: Any) -> dict[str, object]:
    policy = {"maximum_recent_events": 24, "include_summary": True}
    return policy


# 功能：
#   为人工检查请求较长的近期证据窗口，仍保持在存储分页上限内。
# 输入：
#   _：固定检索策略不使用的扩展参数。
# 输出：
#   policy：最多一百二十条近期事件并包含摘要的检索配置。
def _audit_retrieval(**_: Any) -> dict[str, object]:
    policy = {"maximum_recent_events": 120, "include_summary": True}
    return policy


# 功能：
#   1. 合并旧摘要及当前最早待处理页中的用户请求与解析目标，保留最近十二条独立内容。
#   2. 先去重再截断，避免重复消息挤掉其他要求；按内容最后出现的顺序排列。
#   3. 摘要有损且不替代原始合同；空页返回零游标，存储层不得以此发布新摘要。
# 输入：
#   window：包含旧摘要及最早待摘要事件的窗口。
#   _：本提取器不使用的扩展参数。
# 输出：
#   summary_result：有界摘要文本及实际处理到的事件序号。
def _extractive_summary(*, window: ConversationWindow, **_: Any) -> dict[str, object]:
    window = _validated_window(window)
    statements = [line[:520] for line in (window.summary or "").splitlines()[-12:] if line.strip()]
    for event in window.recent_events:
        if event.event_type == "mission.request":
            message = event.payload.get("message")
            if isinstance(message, str) and message.strip():
                statements.append(f"User request: {' '.join(message.split())[:500]}")
        elif event.event_type.startswith("model.intent_parser"):
            artifact = event.payload.get("artifact")
            if isinstance(artifact, dict):
                goal = artifact.get("goal")
                if isinstance(goal, str) and goal.strip():
                    statements.append(f"Resolved goal: {' '.join(goal.split())[:500]}")
    through = max((event.sequence for event in window.recent_events), default=0)
    # 倒序去重保留最后一次出现，再恢复正序；不能先截最后十二次事件再丢掉重复项。
    distinct = list(dict.fromkeys(reversed(statements)))[:12]
    summary_result = {
        "summary": "\n".join(reversed(distinct)),
        "through_sequence": through,
    }
    return summary_result


# 功能：
#   声明每个对话的标准事件保留上限，钩子自身不删除事件。
# 输入：
#   _：固定保留策略不使用的扩展参数。
# 输出：
#   policy：最多一万条事件的保留配置。
def _standard_retention(**_: Any) -> dict[str, object]:
    policy = {"maximum_events": 10_000, "policy": "bounded-event-ledger"}
    return policy


# 功能：
#   提供明确的较短留存选项，不把保留下来的摘要冒称为完整历史归档。
# 输入：
#   _：固定保留策略不使用的扩展参数。
# 输出：
#   policy：最多五百条事件的最小留存配置。
def _minimal_retention(**_: Any) -> dict[str, object]:
    policy = {"maximum_events": 500, "policy": "privacy-minimal"}
    return policy


# 功能：
#   注册上下文压缩、增强、存储、检索、摘要及留存策略，账户记忆同意仍由独立链路控制。
# 输入：
#   无。
# 输出：
#   definitions：包含互斥策略和有序增强管线的插件定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    strategies = [
        (
            "context.structured-window",
            "结构化上下文窗口",
            "保留近期事件并压缩大型工具结果，适合日常任务对话。",
            _structured_window,
            True,
        ),
        (
            "context.event-ledger",
            "事件账本上下文",
            "以有序事件和哈希摘要组织上下文，适合审计与长任务。",
            _event_ledger,
            False,
        ),
    ]
    enrichers = [
        (
            "context.map-ontology",
            "地图本体上下文",
            "加入地图实体、别名、节点和边界规模。",
            _map_context,
            10,
        ),
        (
            "context.request-identity",
            "任务身份上下文",
            "加入稳定对话 ID、语言、起点和本轮消息信息。",
            _request_context,
            20,
        ),
        (
            "context.safety-boundary",
            "安全边界上下文",
            "在每次意图提取前加入不可放宽的执行权限边界。",
            _safety_context,
            30,
        ),
    ]
    definitions = [
        hook_plugin(
            module_name=__name__,
            plugin_id=plugin_id,
            name=name,
            description=description,
            capability_id=f"{plugin_id}.compact",
            capability_kind="context-strategy",
            capability_name=name,
            capability_description=description,
            category_id="context",
            category_label="上下文与记忆",
            slot_id="context.compaction-strategy",
            slot_label="上下文策略",
            activation_mode="single",
            category_order=30,
            slot_order=10,
            plugin_order=index * 10,
            hooks={"compact_context": handler},
            default_enabled=enabled,
            failure_mode="fail-closed",
        )
        for index, (plugin_id, name, description, handler, enabled) in enumerate(
            strategies, start=1
        )
    ]
    definitions.extend(
        hook_plugin(
            module_name=__name__,
            plugin_id=plugin_id,
            name=name,
            description=description,
            capability_id=f"{plugin_id}.enrich",
            capability_kind="context-enricher",
            capability_name=name,
            capability_description=description,
            category_id="context",
            category_label="上下文与记忆",
            slot_id="context.enrichment",
            slot_label="上下文增强管线",
            activation_mode="pipeline",
            category_order=30,
            slot_order=20,
            plugin_order=order,
            pipeline_order=order,
            hooks={"enrich_context": handler},
            default_enabled=True,
            failure_mode="isolate",
        )
        for plugin_id, name, description, handler, order in enrichers
    )
    for plugin_id, name, description, slot_id, slot_label, kind, hook, handler, enabled, order in [
        (
            "context.store-sqlite-wal",
            "SQLite WAL 上下文库",
            "使用事务化事件账本、摘要和供应商上下文指针。",
            "context.store",
            "上下文存储",
            "context-store",
            "resolve_context_store",
            _sqlite_wal_store,
            True,
            10,
        ),
        (
            "context.retrieve-balanced",
            "平衡检索",
            "读取摘要和最近二十四条结构化事件。",
            "context.retrieval-policy",
            "上下文检索",
            "context-retriever",
            "retrieve_context",
            _balanced_retrieval,
            True,
            10,
        ),
        (
            "context.retrieve-audit",
            "审计检索",
            "为评估任务读取更长的结构化事件窗口。",
            "context.retrieval-policy",
            "上下文检索",
            "context-retriever",
            "retrieve_context",
            _audit_retrieval,
            False,
            20,
        ),
        (
            "context.summary-extractive",
            "结构化摘要",
            "从任务请求和意图产物生成有界提取式摘要，保留近期请求与目标。",
            "context.summarization-policy",
            "上下文摘要",
            "context-summarizer",
            "summarize_context",
            _extractive_summary,
            True,
            10,
        ),
        (
            "context.retention-standard",
            "标准留存",
            "每个任务保留一万条事件并持续保留摘要。",
            "context.retention-policy",
            "上下文留存",
            "retention-policy",
            "resolve_retention",
            _standard_retention,
            True,
            10,
        ),
        (
            "context.retention-minimal",
            "最小留存",
            "每个任务只保留五百条事件，适合隐私敏感部署。",
            "context.retention-policy",
            "上下文留存",
            "retention-policy",
            "resolve_retention",
            _minimal_retention,
            False,
            20,
        ),
    ]:
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.{hook}",
                capability_kind=kind,
                capability_name=name,
                capability_description=description,
                category_id="context",
                category_label="上下文与记忆",
                slot_id=slot_id,
                slot_label=slot_label,
                activation_mode="single",
                category_order=30,
                slot_order=order + 30,
                plugin_order=order,
                hooks={hook: handler},
                default_enabled=enabled,
                failure_mode="fail-closed",
                permissions=["context.read", "context.write-summary"],
            )
        )
    return definitions
