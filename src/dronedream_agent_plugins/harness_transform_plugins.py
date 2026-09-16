"""Pure preparation transforms; all changed control contracts are revalidated before return."""

from __future__ import annotations

import math
from typing import Any, TypeVar

from pydantic import BaseModel

from dronedream_agent_core.contracts import (
    IntentArtifact,
    Px4Track,
    SemanticPlan,
    TaskGraph,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_plugin_sdk.protocol import copy_json

from ._helpers import bounded_number, hook_plugin, policy_integer, policy_number

_Model = TypeVar("_Model", bound=BaseModel)


# 功能：
#   严格重建嵌套模型契约，拒绝修改后不合法的字段，不以深复制代替校验。
# 输入：
#   value：待复核的任务、意图、计划或轨迹模型。
# 输出：
#   validated：通过完整契约校验且不共享原数据的模型。
def _validated_copy(value: _Model) -> _Model:
    validated = type(value).model_validate(value.model_dump(mode="python"), strict=True)
    return validated


# 功能：
#   有界复制请求并检查消息类型与长度，保留原文和元数据，不渲染任意 Python 对象。
# 输入：
#   value：包含 message 的用户请求对象。
# 输出：
#   result：消息合法、嵌套数据独立的请求副本。
def _request_value(value: dict[str, object]) -> dict[str, object]:
    result = copy_json(value)
    message = result.get("message") if isinstance(result, dict) else None
    if not isinstance(message, str) or len(message) > 4000:
        raise ValueError("INPUT_REQUEST_FEATURES_INVALID")
    return result


# 功能：
#   统计常用汉字与 ASCII 字母作为路由线索，没有匹配字符时保留未知语言。
# 输入：
#   value：含自然语言消息的请求对象。
#   _：本特征提取器不使用的扩展参数。
# 输出：
#   enriched：保留原文并附加字符统计与主导语言提示的请求。
def _language_features(*, value: dict[str, object], **_: Any) -> dict[str, object]:
    value = _request_value(value)
    message = value["message"]
    chinese_count = sum("\u4e00" <= character <= "\u9fff" for character in message)
    latin_count = sum(character.isascii() and character.isalpha() for character in message)
    dominant = (
        "unknown"
        if not chinese_count and not latin_count
        else ("zh" if chinese_count >= latin_count else "en")
    )
    enriched = {
        **value,
        "language_features": {
            "dominant_language": dominant,
            "chinese_character_count": chinese_count,
            "latin_character_count": latin_count,
        },
    }
    return enriched


# 功能：
#   提取紧急、速度、载荷、巡检和隐私的话题线索，不以关键词匹配解释否定或发出运动指令。
# 输入：
#   value：包含用户原始消息的请求对象。
#   _：本特征提取器不使用的扩展参数。
# 输出：
#   enriched：保留原请求并附加命中话题类别的对象。
def _directive_features(*, value: dict[str, object], **_: Any) -> dict[str, object]:
    value = _request_value(value)
    message = value["message"].casefold()
    groups = {
        "urgency": ("尽快", "马上", "立即", "紧急", "asap", "immediately"),
        "speed_change": ("快点", "慢点", "加速", "减速", "faster", "slower"),
        "payload": ("外卖", "包裹", "取件", "拿回", "pickup", "parcel", "delivery"),
        "inspection": ("检查", "巡检", "拍摄", "inspect", "survey", "photograph"),
        "privacy": ("隐私", "不要拍", "安静", "privacy", "do not record", "quiet"),
    }
    detected = [
        name for name, tokens in groups.items() if any(token in message for token in tokens)
    ]
    enriched = {**value, "directive_classes": detected}
    return enriched


# 功能：
#   规范约束空白并去除完全重复项，保留大小写差异和首次出现顺序，不解释或放宽约束。
# 输入：
#   value：模型生成的结构化意图。
#   _：本规范化器不使用的扩展参数。
# 输出：
#   normalized_intent：约束清理后重新通过契约校验的独立意图。
def _canonicalize_intent(*, value: IntentArtifact, **_: Any) -> IntentArtifact:
    value = _validated_copy(value)
    constraints: list[str] = []
    seen: set[str] = set()
    for item in value.constraints:
        normalized = " ".join(item.split())
        # 地图实体或变量名可能区分大小写，不能把 zone A 与 zone a 的限制合并。
        if normalized and normalized not in seen:
            seen.add(normalized)
            constraints.append(normalized)
    normalized_intent = _validated_copy(value.model_copy(update={"constraints": constraints}))
    return normalized_intent


# 功能：
#   清理意图起点、目标与返程实体的空白，实体是否存在仍由后续当前地图解析验证。
# 输入：
#   value：准备规范实体名称的意图模型。
#   _：本规范化器不使用的扩展参数。
# 输出：
#   normalized_intent：实体空白清理并通过契约校验的独立意图。
def _normalize_entities(*, value: IntentArtifact, **_: Any) -> IntentArtifact:
    value = _validated_copy(value)
    normalized_intent = _validated_copy(
        value.model_copy(
            update={
                "start_entity": " ".join(value.start_entity.split()),
                "target_entity": " ".join(value.target_entity.split()),
                "return_entity": " ".join(value.return_entity.split()),
            }
        )
    )
    return normalized_intent


_ACTION_EVIDENCE = {
    "takeoff": "vehicle airborne state confirmed",
    "traverse": "target node pose reached within tolerance",
    "navigate": "target node pose reached within tolerance",
    "return": "return node pose reached within tolerance",
    "pickup": "payload attachment and custody state confirmed",
    "land": "on-ground state confirmed",
    "inspect": "inspection observation bound to target node",
}


# 功能：
#   去重各任务成功条件，补齐已知动作的必需证据要求；满额无法补齐时拒绝，不伪造成功证据。
# 输入：
#   value：包含任务节点和成功条件的任务图。
#   _：本增强器不使用的扩展参数。
# 输出：
#   enriched_graph：成功条件补齐后重新校验的独立任务图。
def _enrich_task_evidence(*, value: TaskGraph, **_: Any) -> TaskGraph:
    value = _validated_copy(value)
    nodes = []
    for node in value.nodes:
        evidence = list(dict.fromkeys(" ".join(item.split()) for item in node.success_evidence))
        required = _ACTION_EVIDENCE.get(node.action)
        if required and required not in evidence:
            # TaskNode 最多容纳 16 项；不能因满额就假装本动作的必需要求已补齐。
            if len(evidence) >= 16:
                raise ValueError(f"TASK_EVIDENCE_CAPACITY_EXCEEDED:{node.task_id}")
            evidence.append(required)
        nodes.append(node.model_copy(update={"success_evidence": evidence}))
    enriched_graph = _validated_copy(value.model_copy(update={"nodes": nodes}))
    return enriched_graph


# 功能：
#   在任务契约上限内，按移动与交互动作设置最小重试次数，不降低原节点已有预算。
# 输入：
#   value：需要配置重试预算的任务图。
#   configuration：可选的 movement_retries 与 interaction_retries 整数配置。
#   _：本变换不使用的扩展参数。
# 输出：
#   budgeted_graph：重试次数变更后重新校验的独立任务图。
def _apply_retry_budget(
    *, value: TaskGraph, configuration: dict[str, object] | None = None, **_: Any
) -> TaskGraph:
    value = _validated_copy(value)
    configured = {} if configuration is None else configuration
    movement_retries = policy_integer(configured, "movement_retries", 2, 0, 8)
    interaction_retries = policy_integer(configured, "interaction_retries", 1, 0, 8)
    nodes = []
    for node in value.nodes:
        budget = (
            movement_retries
            if node.action in {"traverse", "navigate", "return"}
            else interaction_retries
        )
        nodes.append(node.model_copy(update={"max_retries": max(node.max_retries, budget)}))
    budgeted_graph = _validated_copy(value.model_copy(update={"nodes": nodes}))
    return budgeted_graph


# 功能：
#   仅删除连续重复的目标，保留非相邻重访以免丢掉返程或重复到访任务。
# 输入：
#   value：含有序目标的语义计划。
#   _：本优化器不使用的扩展参数。
# 输出：
#   optimized_plan：目标序列去重并重新校验的独立计划。
def _deduplicate_targets(*, value: SemanticPlan, **_: Any) -> SemanticPlan:
    value = _validated_copy(value)
    targets: list[str] = []
    for target in value.ordered_targets:
        if not targets or targets[-1] != target:
            targets.append(target)
    optimized_plan = _validated_copy(value.model_copy(update={"ordered_targets": targets}))
    return optimized_plan


# 功能：
#   按六种飞行阶段收紧速度上限，拒绝未知阶段、非有限或非正配置，不提升原轨迹限速。
# 输入：
#   value：具有阶段与限速的 PX4 轨迹契约。
#   configuration：可选的 phase_speed_caps_mps 阶段限速，单位米每秒。
#   _：本优化器不使用的扩展参数。
# 输出：
#   limited_track：阶段限速收紧并重新校验的独立轨迹。
def _phase_speed_envelope(
    *, value: Px4Track, configuration: dict[str, object] | None = None, **_: Any
) -> Px4Track:
    value = _validated_copy(value)
    configured = {} if configuration is None else configuration
    if not isinstance(configured, dict):
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    default_caps = {
        "launch": 1.0,
        "transit": 2.0,
        "stairs": 0.8,
        "pickup": 0.5,
        "return": 2.0,
        "land": 0.5,
    }
    raw_caps = configured.get("phase_speed_caps_mps", {})
    if (
        not isinstance(raw_caps, dict)
        or set(raw_caps) - set(default_caps)
        or any(not bounded_number(item, 0, 20) or item <= 0 for item in raw_caps.values())
    ):
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    caps = {**default_caps, **raw_caps}
    points = [
        point.model_copy(update={"speed_limit_mps": min(point.speed_limit_mps, caps[point.phase])})
        for point in value.points
    ]
    limited_track = _validated_copy(value.model_copy(update={"points": points}))
    return limited_track


# 功能：
#   根据三维航迹转角收紧拐点速度，跳过零长度方向，不把计划几何当成实时避障感知。
# 输入：
#   value：含各点坐标与速度上限的 PX4 轨迹。
#   configuration：可选转角阈值（度）和拐点限速（米每秒）。
#   _：本优化器不使用的扩展参数。
# 输出：
#   limited_track：拐点限速收紧后重新校验的独立轨迹。
def _corner_speed_envelope(
    *, value: Px4Track, configuration: dict[str, object] | None = None, **_: Any
) -> Px4Track:
    value = _validated_copy(value)
    configured = {} if configuration is None else configuration
    threshold_deg = policy_number(configured, "minimum_turn_angle_deg", 35.0, 5.0, 175.0)
    # 拐点到达速度应足够低，并且不得覆盖原轨迹中已经更严格的速度上限。
    corner_speed_mps = policy_number(configured, "corner_speed_limit_mps", 0.3, 0, 5.0)
    if corner_speed_mps <= 0:
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    points = list(value.points)
    for index in range(1, len(points) - 1):
        previous, current, following = points[index - 1], points[index], points[index + 1]
        incoming = (current.x - previous.x, current.y - previous.y, current.z - previous.z)
        outgoing = (following.x - current.x, following.y - current.y, following.z - current.z)
        incoming_norm = math.hypot(*incoming)
        outgoing_norm = math.hypot(*outgoing)
        if not math.isfinite(incoming_norm) or not math.isfinite(outgoing_norm):
            raise ValueError("TRACK_CORNER_GEOMETRY_INVALID")
        if incoming_norm <= 1e-9 or outgoing_norm <= 1e-9:
            continue
        cosine = sum(
            (a / incoming_norm) * (b / outgoing_norm)
            for a, b in zip(incoming, outgoing, strict=True)
        )
        turn_angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if turn_angle_deg >= threshold_deg:
            points[index] = current.model_copy(
                update={"speed_limit_mps": min(current.speed_limit_mps, corner_speed_mps)}
            )
    limited_track = _validated_copy(value.model_copy(update={"points": points}))
    return limited_track


# 功能：
#   配置航点位置、速度及连续稳定时间要求，使执行器以实测收敛而非单纯停顿决定推进。
# 输入：
#   value：需要附加航点收敛约束的 PX4 轨迹。
#   configuration：位置容差（米）、速度容差（米每秒）、稳定窗口和等待上限（秒）。
#   _：本优化器不使用的扩展参数。
# 输出：
#   settled_track：启用航点稳定检查且通过时间约束校验的独立轨迹。
def _telemetry_waypoint_settle(
    *, value: Px4Track, configuration: dict[str, object] | None = None, **_: Any
) -> Px4Track:
    value = _validated_copy(value)
    configured = {} if configuration is None else configuration
    updates: dict[str, object] = {"stop_at_waypoints": True}
    for key, default, maximum in (
        ("position_tolerance_m", 0.2, 2),
        ("speed_tolerance_mps", 0.15, 2),
        ("stable_window_seconds", 0.5, 10),
        ("settle_timeout_seconds", 12.0, 120),
    ):
        number = policy_number(configured, key, default, 0, maximum)
        if number <= 0:
            raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
        updates[f"waypoint_{key}"] = number
    if updates["waypoint_stable_window_seconds"] > updates["waypoint_settle_timeout_seconds"]:
        raise ValueError("WAYPOINT_SETTLE_WINDOW_EXCEEDS_TIMEOUT")
    settled_track = _validated_copy(value.model_copy(update=updates))
    return settled_track


# 功能：
#   对工具结果按完整内容去重并稳定排序，保留不同调用及相互矛盾的证据。
# 输入：
#   value：可能含重复项和相反结论的工具结果列表。
#   _：本融合器不使用的扩展参数。
# 输出：
#   fused：不共享原数据、未删除独特证据的结果列表。
def _fuse_tool_advice(*, value: list[dict[str, object]], **_: Any) -> list[dict[str, object]]:
    detached = copy_json(value)
    if not isinstance(detached, list) or any(not isinstance(item, dict) for item in detached):
        raise ValueError("TOOL_ADVICE_LIST_INVALID")
    by_content: dict[str, dict[str, object]] = {}
    for item in detached:
        if "accepted" in item and type(item["accepted"]) is not bool:
            raise ValueError("TOOL_ADVICE_VERDICT_INVALID")
        by_content.setdefault(sha256_json(item), item)
    # 工具名称不是调用身份；同一工具检查不同位置或不同时间的结果不能互相覆盖。
    fused = sorted(
        by_content.values(),
        key=lambda item: (item.get("accepted") is not True, str(item.get("tool_id", ""))),
    )
    return fused


# 功能：
#   注册请求、意图、任务、目标、轨迹和工具结果的准备期变换，不替代核心身份及安全复核。
# 输入：
#   无。
# 输出：
#   definitions：含独立配置边界、顺序及失败策略的纯变换插件列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions: list[PluginDefinition] = []
    request_plugins = [
        (
            "input.language-features",
            "语言特征提取",
            "从原始自然语言中提取中英文字符分布，不改变用户原文。",
            _language_features,
        ),
        (
            "input.directive-features",
            "任务指令特征",
            "提取紧急、速度、载荷、巡检和隐私类话题线索，不代替语义判断。",
            _directive_features,
        ),
    ]
    for index, (plugin_id, name, description, handler) in enumerate(request_plugins, start=1):
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.extract",
                capability_kind="structured-decoder",
                capability_name=name,
                capability_description=description,
                category_id="input",
                category_label="输入与结构化",
                slot_id="input.request-features",
                slot_label="请求特征管线",
                activation_mode="pipeline",
                category_order=10,
                slot_order=10,
                plugin_order=index * 10,
                pipeline_order=index * 10,
                hooks={"enrich_request": handler},
                default_enabled=True,
                failure_mode="isolate",
            )
        )
    intent_plugins = [
        (
            "input.intent-constraint-normalizer",
            "约束规范化",
            "清理并去重结构化意图中的约束，同时保留首次出现顺序。",
            _canonicalize_intent,
        ),
        (
            "input.intent-entity-normalizer",
            "实体规范化",
            "清理起点、目标和返程实体中的重复空白。",
            _normalize_entities,
        ),
    ]
    for index, (plugin_id, name, description, handler) in enumerate(intent_plugins, start=1):
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.normalize",
                capability_kind="structured-decoder",
                capability_name=name,
                capability_description=description,
                category_id="input",
                category_label="输入与结构化",
                slot_id="input.intent-normalizers",
                slot_label="意图规范化管线",
                activation_mode="pipeline",
                category_order=10,
                slot_order=20,
                plugin_order=index * 10,
                pipeline_order=index * 10,
                hooks={"normalize_intent": handler},
                default_enabled=True,
                failure_mode="fail-closed",
            )
        )
    task_plugins = [
        (
            "planning.task-evidence-enricher",
            "任务证据增强",
            "为每类任务补充可检查的成功证据，不替换模型已有证据。",
            _enrich_task_evidence,
            {},
        ),
        (
            "planning.task-retry-budget",
            "任务重试预算",
            "按移动和交互任务配置最小重试预算。",
            _apply_retry_budget,
            {
                "type": "object",
                "properties": {
                    "movement_retries": {"type": "integer", "minimum": 0, "maximum": 8},
                    "interaction_retries": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 8,
                    },
                },
                "additionalProperties": False,
            },
        ),
    ]
    for index, (plugin_id, name, description, handler, schema) in enumerate(task_plugins, start=1):
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.transform",
                capability_kind="task-decomposer",
                capability_name=name,
                capability_description=description,
                category_id="planning",
                category_label="任务规划",
                slot_id="planning.task-transformers",
                slot_label="任务图增强管线",
                activation_mode="pipeline",
                category_order=40,
                slot_order=10,
                plugin_order=index * 10,
                pipeline_order=index * 10,
                hooks={"transform_task_graph": handler},
                default_enabled=True,
                failure_mode="fail-closed",
                configuration_schema=schema,
            )
        )
    definitions.append(
        hook_plugin(
            module_name=__name__,
            plugin_id="planning.semantic-target-deduplicator",
            name="语义目标去重",
            description="删除连续重复目标，减少无意义的零长度路线段。",
            capability_id="planning.semantic-target-deduplicator.optimize",
            capability_kind="plan-optimizer",
            capability_name="语义目标去重",
            capability_description="删除连续重复目标，减少无意义的零长度路线段。",
            category_id="planning",
            category_label="任务规划",
            slot_id="planning.semantic-optimizers",
            slot_label="语义计划优化管线",
            activation_mode="pipeline",
            category_order=40,
            slot_order=20,
            plugin_order=10,
            pipeline_order=10,
            hooks={"optimize_semantic_plan": _deduplicate_targets},
            default_enabled=True,
            failure_mode="fail-closed",
        )
    )
    definitions.append(
        hook_plugin(
            module_name=__name__,
            plugin_id="planning.corner-speed-envelope",
            name="转弯速度包线",
            description="识别三维航迹急转点并限制转弯速度，避免按直线速度穿越门洞和拐角。",
            capability_id="planning.corner-speed-envelope.optimize",
            capability_kind="plan-optimizer",
            capability_name="转弯速度包线",
            capability_description="按三维转角限制局部航迹速度。",
            category_id="planning",
            category_label="任务规划",
            slot_id="planning.track-optimizers",
            slot_label="航迹优化管线",
            activation_mode="pipeline",
            category_order=40,
            slot_order=40,
            plugin_order=20,
            pipeline_order=20,
            runs_after=["planning.phase-speed-envelope"],
            hooks={"optimize_track": _corner_speed_envelope},
            default_enabled=False,
            failure_mode="fail-closed",
            configuration_schema={
                "type": "object",
                "properties": {
                    "minimum_turn_angle_deg": {
                        "type": "number",
                        "minimum": 5,
                        "maximum": 175,
                    },
                    "corner_speed_limit_mps": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 5,
                        "default": 0.3,
                    },
                },
                "additionalProperties": False,
            },
        )
    )
    definitions.append(
        hook_plugin(
            module_name=__name__,
            plugin_id="planning.telemetry-waypoint-settle",
            name="遥测航点收敛",
            description="每个航点必须实测到位并稳定后才允许进入下一航段，避免定时停顿结束时仍带偏差转弯。",
            capability_id="planning.telemetry-waypoint-settle.optimize",
            capability_kind="plan-optimizer",
            capability_name="遥测航点收敛",
            capability_description="为 PX4 航迹绑定位置、速度、稳定窗口和超时收敛契约。",
            category_id="planning",
            category_label="任务规划",
            slot_id="planning.track-optimizers",
            slot_label="航迹优化管线",
            activation_mode="pipeline",
            category_order=40,
            slot_order=40,
            plugin_order=30,
            pipeline_order=30,
            runs_after=["planning.corner-speed-envelope"],
            hooks={"optimize_track": _telemetry_waypoint_settle},
            default_enabled=False,
            failure_mode="fail-closed",
            configuration_schema={
                "type": "object",
                "properties": {
                    "position_tolerance_m": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 2,
                        "default": 0.2,
                    },
                    "speed_tolerance_mps": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 2,
                        "default": 0.15,
                    },
                    "stable_window_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 10,
                        "default": 0.5,
                    },
                    "settle_timeout_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 120,
                        "default": 12,
                    },
                },
                "additionalProperties": False,
            },
            metadata={
                "differentiation": "measured-convergence-gate-not-command-speed-shaping",
                "failure_origin": "school-map-office-corner-tracking-overshoot",
            },
        )
    )
    definitions.append(
        hook_plugin(
            module_name=__name__,
            plugin_id="planning.phase-speed-envelope",
            name="分阶段速度包线",
            description="按起飞、通行、楼梯、取件、返程和降落阶段限制航迹速度。",
            capability_id="planning.phase-speed-envelope.optimize",
            capability_kind="plan-optimizer",
            capability_name="分阶段速度包线",
            capability_description="按飞行阶段限制航迹速度。",
            category_id="planning",
            category_label="任务规划",
            slot_id="planning.track-optimizers",
            slot_label="航迹优化管线",
            activation_mode="pipeline",
            category_order=40,
            slot_order=40,
            plugin_order=10,
            pipeline_order=10,
            hooks={"optimize_track": _phase_speed_envelope},
            default_enabled=True,
            failure_mode="fail-closed",
            configuration_schema={
                "type": "object",
                "properties": {
                    "phase_speed_caps_mps": {
                        "type": "object",
                        "propertyNames": {
                            "enum": ["launch", "transit", "stairs", "pickup", "return", "land"]
                        },
                        "additionalProperties": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 20,
                        },
                    }
                },
                "additionalProperties": False,
            },
        )
    )
    definitions.append(
        hook_plugin(
            module_name=__name__,
            plugin_id="tools.result-fusion-deduplicator",
            name="工具结果融合",
            description="去除完全重复的工具结果，保留不同调用以及相互矛盾的证据。",
            capability_id="tools.result-fusion-deduplicator.fuse",
            capability_kind="result-fusion",
            capability_name="工具结果融合",
            capability_description="去重并稳定排序工具建议。",
            category_id="tools",
            category_label="工具与服务",
            slot_id="tools.result-fusion",
            slot_label="工具结果融合管线",
            activation_mode="pipeline",
            category_order=50,
            slot_order=40,
            plugin_order=10,
            pipeline_order=10,
            hooks={"fuse_results": _fuse_tool_advice},
            default_enabled=True,
            failure_mode="isolate",
        )
    )
    return definitions
