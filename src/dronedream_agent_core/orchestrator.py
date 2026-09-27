"""Multi-call mission preparation bound to real model APIs and qualified geometry."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar
from uuid import uuid4

import jsonschema
from pydantic import BaseModel

from dronedream_plugin_sdk.protocol import encode_json

from .assets import AssetQualificationError, read_map_semantic_object, resolve_map_entity
from .collision import (
    PREFERRED_TRANSIT_CLEARANCE_M,
    build_tracking_corridor_budget,
)
from .context import ContextStore
from .contracts import (
    ConversationWindow,
    DomainActionCatalog,
    FlightPlan,
    GraphRoute,
    IntentArtifact,
    IntentCritique,
    MapAsset,
    MapCatalog,
    MissionAssetPairQualificationBinding,
    MissionContract,
    MissionRequest,
    MissionVerificationPlan,
    ModelCallRecord,
    PlanCritique,
    PlannerContribution,
    PlannerValidation,
    PlanSegment,
    PluginInvocationPlan,
    PreparedMission,
    Px4Track,
    Px4TrackRequest,
    RouteAlternativeCandidate,
    RouteAlternativeDecision,
    RouteAlternativeSet,
    RouteClearanceReport,
    RoutePoint,
    RouteQuery,
    RuntimeActionExecutionContract,
    RuntimeCheckpoint,
    RuntimeCheckpointContract,
    SemanticPlan,
    TaskGraph,
    TaskGraphArtifact,
    ToolReceipt,
    VehicleAsset,
)
from .domain_actions import action_by_id, action_ids, merge_action_packs, movement_action_ids
from .evidence import EvidenceChain
from .extensions import ExtensionExecutionError, ExtensionRegistry
from .hashing import sha256_json
from .lifecycle import LifecycleTransitionError
from .map_reasoning import build_map_reasoning_context
from .model_harness.boundary import HarnessInputEnvelope
from .model_harness.graph import (
    HarnessGraphError,
    HarnessStageRuntime,
    HarnessTopology,
    resolve_harness_runtime_policy,
)
from .model_harness.memory import (
    empty_account_memory_model_context,
    validate_account_memory_model_context,
)
from .model_harness.model_port import (
    ModelInvocationError,
    ProviderName,
    StructuredCallResult,
    StructuredModelPort,
)
from .navigation_readiness import assess_navigation_readiness, enforce_environment_readiness
from .plugin_api import (
    ToolEnvironment,
    build_discovered_extension_registry,
    build_discovered_tool_registry,
)
from .plugin_contracts import PluginHookReceipt, PluginSnapshot
from .prompts import (
    GLOBAL_PLANNER,
    INTENT_CRITIC,
    INTENT_PARSER,
    PLAN_CRITIC,
    PLUGIN_ROUTER,
    TASK_DECOMPOSER,
)
from .runtime_actions import (
    RuntimeActionContractError,
    build_runtime_action_execution_contract,
    merge_runtime_action_adapters,
)
from .runtime_bindings import load_map_runtime_bindings, resolve_vehicle_collision_center_offset
from .tools import ToolExecutionError, ToolRegistry
from .verification import MissionVerificationPlanError, build_mission_verification_plan

MOVEMENT_ACTIONS = frozenset({"traverse", "navigate", "return"})
CORE_PLANNING_SLOTS = frozenset(
    {
        "planning.route-strategy",
        "planning.route-candidates",
        "planning.alternative-ranker",
        "safety.route-clearance",
        "flight-control.track-export",
        "runtime.track-export",
    }
)
IMMUTABLE_SAFETY_RULES = [
    "Model output never has direct actuator authority.",
    "Unknown or unavailable critical telemetry causes hold or abort.",
    "Every executable route must pass continuous vehicle-envelope collision checks.",
    "Execution must remain inside the qualified map and vehicle asset hashes.",
    "Landing or abort remains available at every runtime phase.",
]
REQUIRED_MISSION_CONSTRAINTS = [
    "safety_priority",
    "bounded_execution_speed",
    "runtime_hold_return_or_abort_available",
    "explicit_execution_and_landing_confirmation",
]
EXPLICIT_CONSTRAINT_PHRASES = {
    "safety_priority": ("安全优先", "safety first", "prioritize safety"),
    "plan_only": (
        "先给我看计划",
        "先生成计划",
        "先只生成",
        "plan only",
        "show me the plan",
    ),
    "do_not_execute": (
        "不要立刻执行",
        "计划先不要执行",
        "不要开始执行",
        "do not execute",
    ),
    "safe_return_path": ("沿安全路径返回", "安全返航路径", "safe return path"),
    "requires_school_map": ("学校地图", "school map"),
    "requires_forward_vision": ("前向视觉", "forward vision", "forward camera"),
    "requires_depth_sensing": ("深度感知", "depth sensing", "depth camera"),
    "requires_localization": ("定位", "localization"),
    "requires_flight_telemetry": ("飞行遥测", "flight telemetry"),
}

_Artifact = TypeVar("_Artifact", bound=BaseModel)
MAX_PREPARATION_ARTIFACT_BYTES = 64 * 1024 * 1024


# 功能：
#   冻结并严格重验跨插件和编排边界的材料，拒绝绕过赋值验证的模型实例。
# 输入：
#   value：模型实例或 JSON 数据。
#   output_type：此边界要求的合同类型。
# 输出：
#   snapshot：与原对象不共享可变成员的已验证合同。
def _artifact_snapshot(value: Any, output_type: type[_Artifact]) -> _Artifact:
    raw = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    rendered = encode_json(raw, limit=MAX_PREPARATION_ARTIFACT_BYTES, node_limit=1_000_000)
    snapshot = output_type.model_validate_json(rendered, strict=True)
    return snapshot


# 功能：
#   给工具路由保留有限的附件内容视图；不把任意对象或歧义键强转成文字。
# 输入：
#   value：已验证请求中的 JSON 值。
#   depth：当前嵌套层数。
# 输出：
#   bounded：裁剪后的标量、列表或字典，深层结构用明确占位符表示。
def _bounded_plugin_context_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise MissionPreparationBlocked("PLUGIN_CONTEXT_NONFINITE_NUMBER")
        return value
    if isinstance(value, str):
        return value[:2_000]
    if depth >= 3:
        return "<nested-value-omitted>"
    if isinstance(value, dict):
        bounded = {}
        for index, (key, item) in enumerate(value.items()):
            if index == 32:
                break
            if not isinstance(key, str) or not key or len(key) > 160:
                raise MissionPreparationBlocked("PLUGIN_CONTEXT_KEY_INVALID")
            bounded[key] = _bounded_plugin_context_value(item, depth=depth + 1)
        return bounded
    if isinstance(value, list):
        return [_bounded_plugin_context_value(item, depth=depth + 1) for item in value[:32]]
    raise MissionPreparationBlocked("PLUGIN_CONTEXT_VALUE_INVALID")


# 功能：
#   为路由模型构造用户请求及附件摘要，摘要不赋予附件执行权限。
# 输入：
#   request：本次任务请求。
# 输出：
#   context：包含消息、输入渠道与有界附件视图的字典。
def _plugin_request_context(request: MissionRequest) -> dict[str, object]:
    context = {
        "message": request.message,
        "locale": request.locale,
        "input_channel": request.input_channel,
        "attachments": [
            {
                "attachment_id": item.attachment_id,
                "display_name": item.display_name,
                "content_type": item.content_type,
                "decoded_kind": item.decoded_kind,
                "text_excerpt": item.text[:2_000] if item.text else None,
                "structured_data": _bounded_plugin_context_value(item.structured_data),
                "issue_codes": item.issue_codes,
            }
            for item in request.attachments[:8]
        ],
        "attachment_view_limits": {"count": 8, "depth": 3, "members": 32, "text_chars": 2_000},
        "attachment_view_is_lossy": True,
    }
    return context


# 功能：
#   按工具显式路由条件筛选建议项；建议不等于调用授权。
# 输入：
#   catalog：当前可路由工具目录。
#   contract：已冻结的目标、负载动作与约束。
# 输出：
#   tool_ids：排序后的匹配工具标识列表。
def _recommended_plugin_tools(
    catalog: list[dict[str, object]], contract: MissionContract
) -> list[str]:
    recommended: list[str] = []
    constraints = " ".join(contract.constraints).casefold()
    goal = contract.goal.casefold()
    for item in catalog:
        metadata = item.get("routing_metadata")
        if not isinstance(metadata, dict):
            continue
        condition = metadata.get("recommended_when")
        if not isinstance(condition, dict) or not condition:
            continue
        if set(condition) - {
            "always",
            "payload_action_in",
            "constraints_any",
            "goal_terms_any",
        }:
            continue
        matches = True
        always = condition.get("always")
        if always is not None:
            matches = always is True
        payload_actions = condition.get("payload_action_in")
        if matches and payload_actions is not None:
            matches = (
                isinstance(payload_actions, list) and contract.payload_action in payload_actions
            )
        constraint_values = condition.get("constraints_any")
        if matches and constraint_values is not None:
            matches = isinstance(constraint_values, list) and any(
                isinstance(value, str) and bool(value.strip()) and value.casefold() in constraints
                for value in constraint_values
            )
        goal_terms = condition.get("goal_terms_any")
        if matches and goal_terms is not None:
            matches = isinstance(goal_terms, list) and any(
                isinstance(value, str) and bool(value.strip()) and value.casefold() in goal
                for value in goal_terms
            )
        if matches:
            recommended.append(str(item["tool_id"]))
    tool_ids = sorted(set(recommended))
    return tool_ids


# 功能：
#   把核心发现的缺失约束或未解析字段交回意图修缮轮，不误传评审的接受标志。
# 输入：
#   intent：当前意图草案。
#   critique：模型评审结果。
#   missing_explicit_constraints：用户明确要求但尚未绑定的约束。
# 输出：
#   feedback：下一轮解析器可读取的结构化反馈。
def _intent_repair_feedback(
    intent: IntentArtifact,
    critique: IntentCritique,
    missing_explicit_constraints: list[str],
) -> dict[str, object]:
    """Return actionable feedback whenever deterministic intent gates reject a round.

    A model critic may accept the semantic intent while the fixed Core still sees
    unresolved critical fields. Returning that accepted critique unchanged gives
    the parser no instruction to revisit those fields and can exhaust every repair
    round. Keep the fail-closed gate, but make its reason explicit and bounded.
    """

    if missing_explicit_constraints:
        return {
            "schema_version": "dronedream.intent-critique.v1",
            "accepted": False,
            "issue_codes": ["MISSING_EXPLICIT_CONSTRAINT"],
            "repair_instructions": [
                "Add these canonical values to constraints: "
                + ", ".join(missing_explicit_constraints)
            ],
        }
    if intent.missing_critical_fields:
        fields = ", ".join(intent.missing_critical_fields)
        return {
            "schema_version": "dronedream.intent-critique.v1",
            "accepted": False,
            "issue_codes": ["UNRESOLVED_CRITICAL_FIELDS"],
            "repair_instructions": [
                "Re-evaluate these claimed missing fields against the supplied "
                f"map_catalog and workflow inputs: {fields}. Remove a field when "
                "the supplied catalog already provides the required planning evidence "
                "or when it is not required for safe route preparation; otherwise keep "
                "it so the workflow remains fail-closed."
            ],
        }
    return critique.model_dump(mode="json")


class MissionPreparationBlocked(RuntimeError):
    """A bounded planning stage could not produce a safe executable package."""


class MissionClarificationRequired(MissionPreparationBlocked):
    # 功能：
    #   把仍缺少的关键用户信息交回对话层，不生成执行合同或赋予飞行权限。
    # 输入：
    #   fields：经过结构化意图验证的疑问。
    # 输出：
    #   self：带有限长度追问信息的准备中断。
    def __init__(self, fields: list[str]) -> None:
        super().__init__("MISSION_CLARIFICATION_REQUIRED")
        self.fields = list(dict.fromkeys(field.strip()[:240] for field in fields if field.strip()))[
            :16
        ]


# 功能：
#   在检索前限制历史窗口大小，拒绝布尔值和越界配置。
# 输入：
#   policy：检索策略插件输出。
# 输出：
#   value：一至二百之间的历史事件数量。
def _retrieval_event_limit(policy: dict[str, object]) -> int:
    """Reject malformed plugin limits instead of silently changing historical context scope."""
    value = policy.get("maximum_recent_events", 24)
    if type(value) is not int or not 1 <= value <= 200:
        raise MissionPreparationBlocked("CONTEXT_RETRIEVAL_POLICY_INVALID")
    return value


@dataclass(frozen=True)
class PreparationConfig:
    """Per-mission bounded preparation policy, separate from high-rate flight control."""

    provider: ProviderName
    critic_provider: ProviderName
    max_provider_attempts: int = 3
    max_intent_rounds: int = 3
    max_planning_rounds: int = 5
    plugin_router_rounds: int = 2
    maximum_plugin_calls: int = 8
    intent_reviews_per_round: int = 1
    plan_reviews_per_round: int = 1
    maximum_model_calls: int = 48
    maximum_optional_tool_calls: int = 16
    model_timeout_seconds: float = 180.0
    vehicle_diameter_m: float = 0.76
    vehicle_height_m: float = 0.43
    waypoint_hold_seconds: float = 0.4
    persisted_task_context: bool = True

    # 功能：
    #   在创建模型连接前校验调用次数、物理单位和上下文开关。
    # 输入：
    #   self：准备阶段配置。
    # 输出：
    #   None。
    def __post_init__(self) -> None:
        """Reject invalid budgets/units before creating transports or planning side effects."""
        bounds = {
            "max_provider_attempts": (1, 5),
            "max_intent_rounds": (1, 5),
            "max_planning_rounds": (1, 5),
            "plugin_router_rounds": (1, 5),
            "maximum_plugin_calls": (0, 16),
            "intent_reviews_per_round": (1, 3),
            "plan_reviews_per_round": (1, 3),
            "maximum_model_calls": (8, 48),
            "maximum_optional_tool_calls": (0, 64),
        }
        for name, (minimum, maximum) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
        for name in (
            "model_timeout_seconds",
            "vehicle_diameter_m",
            "vehicle_height_m",
            "waypoint_hold_seconds",
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (name != "waypoint_hold_seconds" and value == 0)
            ):
                raise ValueError(f"{name} must be finite and within its physical range")
        if type(self.persisted_task_context) is not bool:
            raise ValueError("persisted_task_context must be a boolean")
        if self.waypoint_hold_seconds > 30:
            raise ValueError("waypoint_hold_seconds exceeds the track contract")


# 功能：
#   分块计算当前资产文件摘要，读取期间的变动导致拒绝而非混用两版内容。
# 输入：
#   path：语义地图或车辆 SDF 文件。
# 输出：
#   digest：本次读取内容的 SHA-256 十六进制摘要。
def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_PREPARATION_ARTIFACT_BYTES:
            raise MissionPreparationBlocked("PREPARATION_ASSET_FILE_INVALID")
        byte_count = 0
        while chunk := source.read(1024 * 1024):
            byte_count += len(chunk)
            if byte_count > MAX_PREPARATION_ARTIFACT_BYTES:
                raise MissionPreparationBlocked("PREPARATION_ASSET_FILE_INVALID")
            hasher.update(chunk)
        after = os.fstat(source.fileno())
    current = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if any(
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns) != identity_before
        for item in (after, current)
    ):
        raise MissionPreparationBlocked("PREPARATION_ASSET_CHANGED")
    digest = hasher.hexdigest()
    return digest


# 功能：
#   将本次准备目录中的阶段快照写成严格 JSON；各轮历史另由证据链保存。
# 输入：
#   path：由编排器选定的阶段文件路径。
#   artifact：合同模型或 JSON 材料。
# 输出：
#   None。
def _write_artifact(path: Path, artifact: Any) -> None:
    raw = artifact.model_dump(mode="json") if isinstance(artifact, BaseModel) else artifact
    encode_json(raw, limit=MAX_PREPARATION_ARTIFACT_BYTES, node_limit=1_000_000)
    rendered = json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(rendered + "\n", encoding="utf-8")


# 功能：
#   压缩历史事件的工具载荷，同时保留结果身份、模型材料与对话顺序。
# 输入：
#   window：持久化线程的历史窗口。
# 输出：
#   context：用于模型提示词的历史视图。
def _compact_context(window: Any) -> dict[str, object]:
    """Keep durable history while excluding bulky tool payloads from model prompts."""

    recent: list[dict[str, object]] = []
    for event in window.recent_events:
        payload = event.payload
        if event.role == "tool":
            payload = {
                key: payload.get(key)
                for key in (
                    "tool_id",
                    "tool_version",
                    "outcome",
                    "input_sha256",
                    "output_sha256",
                    "issue_codes",
                )
            }
        elif event.event_type.startswith("model."):
            record = payload.get("record", {})
            payload = {
                "artifact": payload.get("artifact"),
                "record": {
                    key: record.get(key)
                    for key in (
                        "role",
                        "provider",
                        "model",
                        "input_sha256",
                        "output_sha256",
                    )
                }
                if isinstance(record, dict)
                else {},
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
        "conversation_id": window.conversation_id,
        "summary": window.summary,
        "recent_events": recent,
    }
    return context


# 功能：
#   按当前用户、线程历史、账户记忆的顺序组装参考上下文，重验记忆治理边界。
# 输入：
#   request：当前用户任务及账户记忆投影。
#   thread_session：经过插件增强的线程历史。
# 输出：
#   context：带固定优先级和记忆权限说明的模型上下文。
def _assemble_model_context(
    request: MissionRequest, thread_session: dict[str, object]
) -> dict[str, object]:
    """Apply the fixed precedence and safety wrapper around pluggable context."""

    raw_memory = request.input_metadata.get("long_term_memory")
    consolidated = validate_account_memory_model_context(
        empty_account_memory_model_context() if raw_memory is None else raw_memory
    )
    context = {
        "precedence": [
            "current_user",
            "thread_session",
            "consolidated_account_memory",
        ],
        "current_user": {
            "message": request.message,
            "locale": request.locale,
            "input_channel": request.input_channel,
        },
        "thread_session": thread_session,
        "consolidated_account_memory": consolidated,
    }
    return context


# 功能：
#   核对唯一模型输入信封的请求、线程、资产和控制面绑定，阻止新旧双路径漂移。
# 输入：
#   request：当前准备请求及其输入信封。
# 输出：
#   context：通过绑定检查的模型输入字典。
def _model_request_context(request: MissionRequest) -> dict[str, object]:
    """Use one canonical model input and reject dual-path field drift."""

    raw_envelope = request.input_metadata.get("model_harness_input")
    if raw_envelope is not None:
        try:
            envelope = _artifact_snapshot(raw_envelope, HarnessInputEnvelope)
        except ValueError as error:
            raise ValueError("HARNESS_INPUT_ENVELOPE_INVALID") from error
        current = envelope.current_request
        expected_fields: dict[str, object] = {
            "message": request.message,
            "locale": request.locale,
            "input_channel": request.input_channel,
            "start_entity": request.start_entity,
            "attachment_ids": [item.attachment_id for item in request.attachments],
        }
        for field_name, expected in expected_fields.items():
            if current.get(field_name) != expected:
                raise ValueError(f"HARNESS_INPUT_MISSION_REQUEST_MISMATCH:{field_name}")
        if envelope.task_id != request.conversation_id or envelope.thread_id != (
            request.conversation_id
        ):
            raise ValueError("HARNESS_INPUT_MISSION_REQUEST_MISMATCH:thread_id")
        raw_assets = request.input_metadata.get("asset_versions")
        if raw_assets is not None and not isinstance(raw_assets, dict):
            raise ValueError("HARNESS_INPUT_MISSION_REQUEST_MISMATCH:asset_versions")
        if isinstance(raw_assets, dict):
            for kind in ("map", "vehicle"):
                binding = raw_assets.get(kind)
                if not isinstance(binding, dict):
                    raise ValueError("HARNESS_INPUT_MISSION_REQUEST_MISMATCH:asset_versions")
                if current.get(f"{kind}_asset_id") != binding.get("asset_id") or current.get(
                    f"{kind}_content_sha256"
                ) != binding.get("content_sha256"):
                    raise ValueError(f"HARNESS_INPUT_MISSION_REQUEST_MISMATCH:{kind}_asset")
        raw_control_plane = request.input_metadata.get("model_harness_control_plane")
        raw_runtime = request.input_metadata.get("model_harness_runtime")
        if not isinstance(raw_control_plane, dict) or not isinstance(raw_runtime, dict):
            raise ValueError("HARNESS_INPUT_CONTROL_PLANE_BINDING_REQUIRED")
        if (
            envelope.control_plane_selection_sha256 != raw_control_plane.get("selection_sha256")
            or envelope.control_plane_selection_sha256
            != raw_runtime.get("control_plane_selection_sha256")
            or envelope.owner_binding_sha256 != raw_runtime.get("owner_binding_sha256")
            or envelope.tenant_binding_sha256 != raw_runtime.get("tenant_binding_sha256")
        ):
            raise ValueError("HARNESS_INPUT_CONTROL_PLANE_BINDING_MISMATCH")
        context = envelope.model_dump(mode="json")
        return context

    raise ValueError("HARNESS_INPUT_ENVELOPE_REQUIRED")


# 功能：
#   给意图模型展示可识别动作，不暴露由核心掌管的执行证据定义。
# 输入：
#   catalog：已安装动作目录。
# 输出：
#   view：用于意图分类的动作视图。
def _intent_action_catalog_view(catalog: DomainActionCatalog) -> dict[str, object]:
    """Expose only the action facts required to classify a user request."""

    view = {
        "catalog_id": catalog.catalog_id,
        "domain_ids": list(catalog.domain_ids),
        "catalog_view": "intent-selection",
        "actions": [
            {
                "action_id": action.action_id,
                "domain_id": action.domain_id,
                "label": action.label,
                "description": action.description,
                "payload": action.payload,
                "flight_boundary": action.flight_boundary,
            }
            for action in catalog.actions
        ],
    }
    return view


# 功能：
#   提供实体名称、别名及已知限制，供模型将自然语言落到当前地图实体。
# 输入：
#   catalog：与当前语义文件绑定的地图目录。
# 输出：
#   view：不包含坐标生成权的实体视图。
def _intent_map_catalog_view(catalog: MapCatalog) -> dict[str, object]:
    """Expose entity identity for intent binding without irrelevant route geometry."""

    view = {
        "schema_version": catalog.schema_version,
        "scene_id": catalog.scene_id,
        "coordinate_frame": catalog.coordinate_frame,
        "semantic_sha256": catalog.semantic_sha256,
        "topology_available": catalog.topology_available,
        "known_limits": list(catalog.known_limits),
        "entity_count": len(catalog.entities),
        "road_segment_count": len(catalog.road_segment_ids),
        # 二维地标仅提供语义环境；可用于动作绑定的实体仍必须来自下方三维 entities。
        "non_actionable_planar_landmarks": [
            landmark.model_dump(mode="json") for landmark in catalog.planar_landmarks
        ],
        "entities": [
            {
                "entity_id": entity.entity_id,
                "aliases": list(entity.aliases),
                "semantic": entity.semantic,
            }
            for entity in catalog.entities
        ],
    }
    return view


# 功能：
#   声明此阶段只获准准备仿真计划；真实执行仍须独立确认，不限制任务只能单程。
# 输入：
#   无。
# 输出：
#   scope：计划权限和返程含义的字典。
def _intent_workflow_scope() -> dict[str, object]:
    """Declare workflow authority without implying a one-way mission shape."""

    scope = {
        "environment": "simulation-only",
        "operator_authorized": True,
        "authorization_scope": "planning-only",
        "execution_authorized": False,
        "execution_confirmation_required": True,
        "physical_hardware_authority": False,
        "round_trip_supported": True,
        "return_entity_semantics": (
            "final landing entity for both one-way and round-trip missions"
        ),
    }
    return scope


# 功能：
#   将用户明确约束绑定到意图，防止模型漏抄要求，并重新验证完整结果。
# 输入：
#   intent：已解析意图。
#   explicit_constraints：从当前原始请求提取的明确要求。
# 输出：
#   bound_intent：保留原顺序并去重后的意图。
def _bind_explicit_intent_constraints(
    intent: IntentArtifact, explicit_constraints: list[str]
) -> IntentArtifact:
    """Keep deterministic user constraints from depending on model echo fidelity."""

    values = list(dict.fromkeys([*intent.constraints, *explicit_constraints]))
    bound_intent = _artifact_snapshot(
        {**intent.model_dump(mode="json"), "constraints": values}, IntentArtifact
    )
    return bound_intent


# 功能：
#   为任务分解提供动作参数、后备动作与执行器信息，成功证据仍由核心绑定。
# 输入：
#   catalog：当前动作包合并后的目录。
# 输出：
#   view：任务分解可引用的动作接口。
def _decomposition_action_catalog_view(catalog: DomainActionCatalog) -> dict[str, object]:
    """Expose model-selectable task fields while keeping evidence core-owned."""

    view = {
        "catalog_id": catalog.catalog_id,
        "domain_ids": list(catalog.domain_ids),
        "catalog_view": "task-decomposition",
        "actions": [
            {
                "action_id": action.action_id,
                "domain_id": action.domain_id,
                "label": action.label,
                "description": action.description,
                "movement": action.movement,
                "payload": action.payload,
                "flight_boundary": action.flight_boundary,
                "input_schema": action.input_schema,
                "allowed_fallbacks": list(action.allowed_fallbacks),
                "runtime_executor": action.runtime_executor,
            }
            for action in catalog.actions
        ],
    }
    return view


# 功能：
#   压缩地图给宏观规划使用，保留完整风险数量、返程顺序和截断标记。
# 输入：
#   context：由当前合格地图计算的推理上下文。
#   contract：本次起点、任务点和返程点。
# 输出：
#   view：有内容摘要绑定的语义地图视图。
def _semantic_planning_map_view(
    context: dict[str, object], contract: MissionContract
) -> dict[str, object]:
    """Keep semantic risk facts while leaving route geometry to qualified tools."""

    topology = context.get("topology")
    topology = topology if isinstance(topology, dict) else {}
    focus_routes = topology.get("focus_routes")
    focus_routes = focus_routes if isinstance(focus_routes, list) else []
    route_summaries: list[dict[str, object]] = []
    for raw_route in focus_routes[:8]:
        if not isinstance(raw_route, dict):
            continue
        raw_candidates = raw_route.get("candidate_routes")
        raw_candidates = raw_candidates if isinstance(raw_candidates, list) else []
        semantic_sequence = raw_route.get("semantic_sequence")
        semantic_sequence = semantic_sequence if isinstance(semantic_sequence, list) else []
        unverified_edges = raw_route.get("unverified_edge_ids")
        unverified_edges = unverified_edges if isinstance(unverified_edges, list) else []
        candidate_summaries = [
            {
                key: candidate.get(key)
                for key in (
                    "route_length_m",
                    "minimum_speed_limit_mps",
                    "route_sha256",
                    "node_count",
                    "truncated_for_model_context",
                )
            }
            for candidate in raw_candidates[:5]
            if isinstance(candidate, dict)
        ]
        route_summaries.append(
            {
                key: raw_route.get(key)
                for key in (
                    "start_node",
                    "goal_node",
                    "route_length_m",
                    "minimum_speed_limit_mps",
                    "route_sha256",
                    "candidate_route_count",
                    "node_count",
                    "edge_count",
                    "truncated_for_model_context",
                    "issue_codes",
                )
                if key in raw_route
            }
            | {
                # 返程可以再次经过同一走廊，不能全局去重而抹去任务顺序。
                "semantic_sequence": semantic_sequence[:24],
                "semantic_sequence_truncated": (
                    len(semantic_sequence) > 24
                    or raw_route.get("truncated_for_model_context") is True
                ),
                "unverified_edge_count": raw_route.get(
                    "unverified_edge_count", len(unverified_edges)
                ),
                "candidate_summaries": candidate_summaries,
                # Imported graph metadata may use zero for an as-yet-unfilled
                # clearance summary.  It is not execution evidence.  The
                # selected route's content-bound continuous collision tool is
                # the only authority for physical vehicle-envelope clearance.
                "geometry_authority": "deterministic-selected-route-tools",
                "continuous_clearance_evaluated_here": False,
            }
        )

    relevant_nodes = {contract.start_node, contract.target_node, contract.return_node}
    named_entities = context.get("named_entities")
    named_entities = named_entities if isinstance(named_entities, list) else []
    view = {
        "schema_version": "dronedream.semantic-planning-map-view.v1",
        "source_context_sha256": sha256_json(context),
        "source_of_truth": context.get("source_of_truth"),
        "graph_sha256": context.get("graph_sha256"),
        "semantic_sha256": context.get("semantic_sha256"),
        "coordinate_frame": context.get("coordinate_frame"),
        "bounds_m": context.get("bounds_m"),
        "topology": {
            key: topology.get(key)
            for key in (
                "complete_graph_node_count",
                "complete_graph_edge_count",
                "truncated_for_model_context",
                "focus_nodes",
            )
        }
        | {"focus_route_summaries": route_summaries},
        "mission_entities": [
            item
            for item in named_entities
            if isinstance(item, dict) and item.get("node_id") in relevant_nodes
        ],
        "semantic_environment": context.get("semantic_environment"),
        "catalog_known_limits": context.get("catalog_known_limits"),
        "model_authority": context.get("model_authority"),
    }
    return view


# 功能：
#   展开一次逻辑结果背后的主响应和支持响应，保留每次实际成功调用记录。
# 输入：
#   result：结构化调用或共识结果。
# 输出：
#   records：按选中响应优先排列的物理调用记录。
def _model_records(result: StructuredCallResult[Any]) -> list[ModelCallRecord]:
    """Return every physical successful call behind one logical Harness result."""

    records = [result.record, *result.supporting_records]
    return records


# 功能：
#   按策略选择多响应结果：严格一致时拒绝异议，评审采用保守拒绝，不伪造新投票。
# 输入：
#   results：各独立端口的实际响应。
#   require_identical：是否要求所有材料摘要相同。
#   record_dissent：非评审材料不同时是否必须拒绝；关闭时显式采用首响应策略。
# 输出：
#   consensus_result：选中材料及所有调用记录、决策方式组成的二元组。
def _resolve_consensus_result(
    results: list[StructuredCallResult[Any]],
    *,
    require_identical: bool,
    record_dissent: bool,
) -> tuple[StructuredCallResult[Any], str]:
    if not results:
        raise MissionPreparationBlocked("MODEL_CONSENSUS_NO_RESPONSES")
    hashes = [sha256_json(item.artifact) for item in results]
    if len(set(hashes)) == 1:
        selected = results[0]
        return (
            StructuredCallResult(
                artifact=selected.artifact,
                record=selected.record,
                supporting_records=(
                    *selected.supporting_records,
                    *(record for item in results[1:] for record in _model_records(item)),
                ),
                attempt_failures=tuple(
                    failure for item in results for failure in item.attempt_failures
                ),
            ),
            "identical",
        )
    if require_identical:
        raise MissionPreparationBlocked("MODEL_CONSENSUS_DISSENT")
    accepted_votes = [getattr(item.artifact, "accepted", None) for item in results]
    if all(isinstance(value, bool) for value in accepted_votes):
        rejected_index = next(
            (index for index, value in enumerate(accepted_votes) if value is False),
            0,
        )
        selected = results[rejected_index]
        supporting = tuple(
            [*selected.supporting_records]
            + [
                record
                for index, item in enumerate(results)
                if index != rejected_index
                for record in _model_records(item)
            ]
        )
        return (
            StructuredCallResult(
                artifact=selected.artifact,
                record=selected.record,
                supporting_records=supporting,
                attempt_failures=tuple(
                    failure for item in results for failure in item.attempt_failures
                ),
            ),
            "conservative-rejection" if False in accepted_votes else "all-accepted",
        )
    if record_dissent:
        raise MissionPreparationBlocked("MODEL_CONSENSUS_DISSENT_UNRESOLVED")
    selected = results[0]
    return (
        StructuredCallResult(
            artifact=selected.artifact,
            record=selected.record,
            supporting_records=(
                *selected.supporting_records,
                *(record for item in results[1:] for record in _model_records(item)),
            ),
            attempt_failures=tuple(
                failure for item in results for failure in item.attempt_failures
            ),
        ),
        "first-response-policy",
    )


# 功能：
#   提取明确措辞、悬停时长和载荷数值，作为模型解析的约束补充而非通用语义理解器。
# 输入：
#   request：本次原始自然语言请求。
# 输出：
#   constraints：按出现顺序去重的规范约束字符串。
def _explicit_constraint_hints(request: MissionRequest) -> list[str]:
    normalized = request.message.casefold()
    hints = [
        canonical
        for canonical, phrases in EXPLICIT_CONSTRAINT_PHRASES.items()
        if any(phrase.casefold() in normalized for phrase in phrases)
    ]
    hover_match = re.search(
        r"(?:悬停|hover)[^0-9+\-.]{0,16}(?P<seconds>[+-]?[0-9]+(?:\.[0-9]+)?)\s*(?:秒|seconds?|s\b)",
        normalized,
    )
    if hover_match is not None:
        seconds = float(hover_match.group("seconds"))
        if not math.isfinite(seconds) or seconds <= 0:
            raise MissionPreparationBlocked("EXPLICIT_HOVER_DURATION_INVALID")
        hints.append(f"pickup_hover_seconds={seconds:g}")
    mass_match = re.search(
        r"(?<![0-9.])(?P<mass>[+-]?[0-9]+(?:\.[0-9]+)?)\s*(?:公斤|千克|kg\b)", normalized
    )
    if mass_match is not None:
        mass = float(mass_match.group("mass"))
        if not math.isfinite(mass) or mass < 0:
            raise MissionPreparationBlocked("EXPLICIT_PAYLOAD_MASS_INVALID")
        hints.append(f"payload_mass_kg={mass:g}")
    constraints = list(dict.fromkeys(hints))
    return constraints


# 功能：
#   在付费模型调用之前检查用户明确载荷是否超出当前机型能力。
# 输入：
#   explicit_constraint_hints：当前请求的规范约束。
#   vehicle：当前已选定机型的质量上限。
# 输出：
#   None。
def _validate_explicit_payload_capacity(
    explicit_constraint_hints: list[str], vehicle: VehicleAsset
) -> None:
    """Reject a physically impossible explicit payload before any model call.

    The value is owned by the user's request, not by plan generation, so asking
    the model to retry cannot repair it.  Failing during deterministic preflight
    avoids repeated paid calls and prevents an old or mismatched payload setup
    from leaking into planning.
    """

    payload_hints = [
        item for item in explicit_constraint_hints if item.startswith("payload_mass_kg=")
    ]
    if not payload_hints:
        return
    payload_mass_kg = float(payload_hints[-1].split("=", 1)[1])
    if not math.isfinite(payload_mass_kg) or payload_mass_kg < 0:
        raise MissionPreparationBlocked("EXPLICIT_PAYLOAD_MASS_INVALID")
    permitted_payload_kg = min(
        vehicle.max_pickup_payload_kg,
        max(0.0, vehicle.max_takeoff_mass_kg - vehicle.dry_mass_kg),
    )
    if payload_mass_kg > permitted_payload_kg + 1e-9:
        raise MissionPreparationBlocked("EXPLICIT_PAYLOAD_EXCEEDS_VEHICLE_CAPACITY")


# 功能：
#   查找意图中缺失的用户明确要求，保持反馈顺序与原要求一致。
# 输入：
#   intent：本轮模型意图。
#   explicit_constraint_hints：必须保留的规范要求。
# 输出：
#   missing：未出现在意图中的要求列表。
def _missing_explicit_constraints(
    intent: IntentArtifact, explicit_constraint_hints: list[str]
) -> list[str]:
    present = {constraint.casefold() for constraint in intent.constraints}
    missing = [hint for hint in explicit_constraint_hints if hint.casefold() not in present]
    return missing


# 功能：
#   只使用当前地图目录解析实体名称，将资格或歧义错误转为准备阻断。
# 输入：
#   entity：用户或模型提供的实体名。
#   catalog：当前实体目录。
#   graph：对应的地图拓扑。
# 输出：
#   node_id：唯一有效的地图节点标识。
def _resolve_entity(entity: str, catalog: MapCatalog, graph: MapAsset) -> str:
    try:
        node_id = resolve_map_entity(entity, catalog, graph)
    except AssetQualificationError as error:
        raise MissionPreparationBlocked(str(error)) from error
    return node_id


# 功能：
#   遍历已验证无环任务图，求指定动作的全部前置任务。
# 输入：
#   graph：通过依赖完整性验证的任务图。
#   task_id：需要检查执行顺序的任务标识。
# 输出：
#   ancestors：去重后的祖先任务标识集合。
def _task_ancestors(graph: TaskGraph, task_id: str) -> set[str]:
    by_id = {task.task_id: task for task in graph.nodes}
    ancestors: set[str] = set()
    pending = list(by_id[task_id].depends_on)
    while pending:
        dependency = pending.pop()
        if dependency in ancestors:
            continue
        ancestors.add(dependency)
        pending.extend(by_id[dependency].depends_on)
    return ancestors


# 功能：
#   以当前动作包的成功证据接口替换模型措辞，保留模型分解和依赖选择。
# 输入：
#   graph：模型或插件产生的任务图。
#   domain_actions：当前已验证动作目录。
# 输出：
#   bound_graph：绑定规范证据后重新验证的任务图。
def _bind_task_graph_action_contracts(
    graph: TaskGraph, domain_actions: DomainActionCatalog
) -> TaskGraph:
    """Replace model-authored evidence labels with the canonical action contract.

    The model owns decomposition and dependency choices. Evidence vocabulary is an
    executable runtime interface and therefore comes only from the installed action
    pack; a paraphrase must neither weaken the gate nor waste another model round.
    """

    definitions = {action.action_id: action for action in domain_actions.actions}
    bound_graph = graph.model_copy(
        update={
            "nodes": [
                task.model_copy(
                    update={
                        "success_evidence": list(definitions[task.action].required_success_evidence)
                    }
                )
                if task.action in definitions
                else task
                for task in graph.nodes
            ]
        }
    )
    return _artifact_snapshot(bound_graph, TaskGraph)


# 功能：
#   核查地图、授权动作、参数、飞行边界及稳定悬停与挂载前置条件；取件不强制扫码。
#   外部插件若明确提供身份验证动作，仍检查它的地点与顺序；目录存在不代表获得授权。
# 输入：
#   graph：待执行任务图。
#   contract：本次任务授予的动作和目标范围。
#   map_graph：当前地图节点集合。
#   domain_actions：可选的当前动作目录，进一步限制可用动作及证据接口。
# 输出：
#   None。
def _validate_task_graph(
    graph: TaskGraph,
    contract: MissionContract,
    map_graph: MapAsset,
    domain_actions: DomainActionCatalog | None = None,
) -> None:
    graph = _artifact_snapshot(graph, TaskGraph)
    contract = _artifact_snapshot(contract, MissionContract)
    known_nodes = {node.node_id for node in map_graph.nodes}
    if any(task.target_node not in known_nodes for task in graph.nodes):
        raise MissionPreparationBlocked("TASK_GRAPH_UNKNOWN_NODE")
    authorized_actions = set(contract.authorized_actions)
    if domain_actions is not None:
        authorized_actions &= action_ids(domain_actions)
    unknown_actions = sorted({task.action for task in graph.nodes} - authorized_actions)
    if unknown_actions:
        raise MissionPreparationBlocked(
            "TASK_GRAPH_UNAUTHORIZED_ACTION:" + ",".join(unknown_actions)
        )
    movement_actions = (
        movement_action_ids(domain_actions) if domain_actions is not None else set(MOVEMENT_ACTIONS)
    )
    contract_movement_targets = {contract.target_node, contract.return_node}
    if any(
        task.action in movement_actions and task.target_node not in contract_movement_targets
        for task in graph.nodes
    ):
        raise MissionPreparationBlocked("TASK_GRAPH_UNAUTHORIZED_MOVEMENT_TARGET")
    if domain_actions is not None:
        for task in graph.nodes:
            definition = action_by_id(domain_actions, task.action)
            if task.fallback not in definition.allowed_fallbacks:
                raise MissionPreparationBlocked(
                    f"TASK_GRAPH_ACTION_FALLBACK_UNAUTHORIZED:{task.action}:{task.fallback}"
                )
            required = {item.casefold().strip() for item in definition.required_success_evidence}
            provided = {item.casefold().strip() for item in task.success_evidence}
            if required != provided:
                raise MissionPreparationBlocked(f"TASK_GRAPH_ACTION_EVIDENCE_MISSING:{task.action}")
            try:
                if definition.input_schema:
                    jsonschema.validate(task.arguments, definition.input_schema)
            except jsonschema.ValidationError as error:
                raise MissionPreparationBlocked(
                    f"TASK_GRAPH_ACTION_ARGUMENTS_INVALID:{task.action}:{error.validator}"
                ) from error
    actions = [task.action for task in graph.nodes]
    if "takeoff" not in actions or "land" not in actions:
        raise MissionPreparationBlocked("TASK_GRAPH_MISSING_FLIGHT_BOUNDARY")
    if actions.count("takeoff") != 1 or actions.count("land") != 1:
        raise MissionPreparationBlocked("TASK_GRAPH_DUPLICATE_FLIGHT_BOUNDARY")
    if not any(
        task.action == "takeoff" and task.target_node == contract.start_node for task in graph.nodes
    ):
        raise MissionPreparationBlocked("TASK_GRAPH_WRONG_TAKEOFF_NODE")
    if not any(
        task.action in movement_actions and task.target_node == contract.target_node
        for task in graph.nodes
    ):
        raise MissionPreparationBlocked("TASK_GRAPH_MISSING_TARGET_MOVEMENT")
    if contract.payload_action == "pickup" and not any(
        task.action == "pickup" and task.target_node == contract.target_node for task in graph.nodes
    ):
        raise MissionPreparationBlocked("TASK_GRAPH_MISSING_PICKUP")
    if contract.payload_action == "pickup":
        by_id = {task.task_id: task for task in graph.nodes}
        if actions.count("pickup") != 1:
            raise MissionPreparationBlocked("TASK_GRAPH_DUPLICATE_PICKUP")
        pickup = next(task for task in graph.nodes if task.action == "pickup")
        ancestors: set[str] = set()
        pending = list(pickup.depends_on)
        while pending:
            dependency = pending.pop()
            if dependency in ancestors:
                continue
            ancestors.add(dependency)
            pending.extend(by_id[dependency].depends_on)
        verification_actions = {"delivery.scan-code", "delivery.verify-recipient"}
        # 内置取件是悬停挂载，不要求身份动作。显式扩展仍不得把异地或事后验证充当前置。
        verification_task_ids = {
            task.task_id for task in graph.nodes if task.action in verification_actions
        }
        if any(task_id not in ancestors or by_id[task_id].target_node != pickup.target_node
               for task_id in verification_task_ids):
            raise MissionPreparationBlocked(
                "TASK_GRAPH_PICKUP_VERIFICATION_LOCATION_OR_ORDER_INVALID"
            )
        if "delivery.precontact-hold" in authorized_actions:
            precontact_task_ids = {
                task_id
                for task_id in ancestors
                if by_id[task_id].action == "delivery.precontact-hold"
                and by_id[task_id].target_node == pickup.target_node
            }
            if not precontact_task_ids:
                raise MissionPreparationBlocked("TASK_GRAPH_MISSING_PRECONTACT_HOLD")
            if any(not any(precontact_id in _task_ancestors(graph, verification_id)
                           for precontact_id in precontact_task_ids)
                   for verification_id in verification_task_ids):
                raise MissionPreparationBlocked("TASK_GRAPH_PRECONTACT_ORDER_INVALID")
        if "delivery.confirm-custody" in authorized_actions:
            custody_tasks = [
                task
                for task in graph.nodes
                if task.action == "delivery.confirm-custody"
                and task.target_node == pickup.target_node
                and pickup.task_id in _task_ancestors(graph, task.task_id)
            ]
            if not custody_tasks:
                raise MissionPreparationBlocked("TASK_GRAPH_MISSING_CUSTODY_CONFIRMATION")
            if not any(
                pickup.task_id in _task_ancestors(graph, task.task_id) for task in custody_tasks
            ):
                raise MissionPreparationBlocked("TASK_GRAPH_CUSTODY_ORDER_INVALID")
            returns = [
                task
                for task in graph.nodes
                if task.action in movement_actions and task.target_node == contract.return_node
            ]
            if "delivery.verify-loaded-stability" in authorized_actions:
                stability_tasks = [
                    task
                    for task in graph.nodes
                    if task.action == "delivery.verify-loaded-stability"
                    and task.target_node == contract.target_node
                    and any(
                        custody.task_id in _task_ancestors(graph, task.task_id)
                        for custody in custody_tasks
                    )
                ]
                if not stability_tasks:
                    raise MissionPreparationBlocked(
                        "TASK_GRAPH_MISSING_LOADED_STABILITY_VERIFICATION"
                    )
                if not any(
                    custody.task_id in _task_ancestors(graph, stability.task_id)
                    for custody in custody_tasks
                    for stability in stability_tasks
                ):
                    raise MissionPreparationBlocked("TASK_GRAPH_LOADED_STABILITY_ORDER_INVALID")
                if not returns or not all(
                    any(
                        stability.task_id in _task_ancestors(graph, return_task.task_id)
                        for stability in stability_tasks
                    )
                    for return_task in returns
                ):
                    raise MissionPreparationBlocked(
                        "TASK_GRAPH_RETURN_WITHOUT_LOADED_STABILITY_AUTHORIZATION"
                    )
            elif not returns or not all(
                any(
                    custody.task_id in _task_ancestors(graph, return_task.task_id)
                    for custody in custody_tasks
                )
                for return_task in returns
            ):
                raise MissionPreparationBlocked("TASK_GRAPH_RETURN_WITHOUT_CUSTODY_AUTHORIZATION")
    by_id = {task.task_id: task for task in graph.nodes}
    prerequisite_actions = {
        "delivery.release-payload": "delivery.verify-release-area",
        "emergency.drop-kit": "emergency.verify-drop-zone",
    }
    for task in graph.nodes:
        required_action = prerequisite_actions.get(task.action)
        if required_action is None:
            continue
        ancestors: set[str] = set()
        pending = list(task.depends_on)
        while pending:
            dependency = pending.pop()
            if dependency in ancestors:
                continue
            ancestors.add(dependency)
            pending.extend(by_id[dependency].depends_on)
        if not any(
            by_id[task_id].action == required_action
            and by_id[task_id].target_node == task.target_node
            for task_id in ancestors
        ):
            raise MissionPreparationBlocked(
                f"TASK_GRAPH_MISSING_ACTION_PREREQUISITE:{task.action}:{required_action}"
            )
    if not any(
        task.action == "land" and task.target_node == contract.return_node for task in graph.nodes
    ):
        raise MissionPreparationBlocked("TASK_GRAPH_WRONG_LANDING_NODE")
    takeoff = next(task for task in graph.nodes if task.action == "takeoff")
    landing = next(task for task in graph.nodes if task.action == "land")
    landing_ancestors = _task_ancestors(graph, landing.task_id)
    # 一份计划只描述一次起降周期，所有移动都必须在起飞之后、最终落地之前。
    for task in graph.nodes:
        if task.action in movement_actions and (
            takeoff.task_id not in _task_ancestors(graph, task.task_id)
            or task.task_id not in landing_ancestors
        ):
            raise MissionPreparationBlocked("TASK_GRAPH_FLIGHT_ORDER_INVALID")


# 功能：
#   限定宏观目标序列只能使用本次合同允许的已知节点，并以约定返程点结束。
# 输入：
#   plan：模型语义路线。
#   contract：本次任务合同。
#   graph：当前地图拓扑。
# 输出：
#   None。
def _validate_semantic_plan(plan: SemanticPlan, contract: MissionContract, graph: MapAsset) -> None:
    known = {node.node_id for node in graph.nodes}
    if any(target not in known for target in plan.ordered_targets):
        raise MissionPreparationBlocked("SEMANTIC_PLAN_UNKNOWN_NODE")
    if any(
        target not in {contract.target_node, contract.return_node}
        for target in plan.ordered_targets
    ):
        raise MissionPreparationBlocked("SEMANTIC_PLAN_UNAUTHORIZED_TARGET")
    if plan.ordered_targets[0] == contract.start_node:
        raise MissionPreparationBlocked("SEMANTIC_PLAN_REPEATS_START")
    if contract.target_node not in plan.ordered_targets:
        raise MissionPreparationBlocked("SEMANTIC_PLAN_MISSING_TARGET")
    if plan.ordered_targets[-1] != contract.return_node:
        raise MissionPreparationBlocked("SEMANTIC_PLAN_WRONG_FINAL_NODE")
    if any(
        first == second
        for first, second in zip(plan.ordered_targets, plan.ordered_targets[1:], strict=False)
    ):
        raise MissionPreparationBlocked("SEMANTIC_PLAN_CONSECUTIVE_DUPLICATE")


# 功能：
#   拼接各段工具路线，在公共节点检查实际坐标连续性，零边路线不能凭空拥有验证记录。
# 输入：
#   routes：按实际访问顺序排列的分段路线。
# 输出：
#   combined：保留逐点几何和全部边的完整路线。
def _combine_routes(routes: list[GraphRoute]) -> GraphRoute:
    if not routes:
        raise MissionPreparationBlocked("NO_MOVEMENT_ROUTE")
    routes = [_artifact_snapshot(route, GraphRoute) for route in routes]
    for route in routes:
        if (
            len(route.node_ids) != len(route.positions_m)
            or len(route.edge_ids) != len(route.node_ids) - 1
            or route.node_ids[0] != route.start_node
            or route.node_ids[-1] != route.goal_node
        ):
            raise MissionPreparationBlocked("ROUTE_TOOL_TOPOLOGY_INVALID")
    node_ids = list(routes[0].node_ids)
    edge_ids = list(routes[0].edge_ids)
    positions = list(routes[0].positions_m)
    for previous, route in zip(routes, routes[1:], strict=False):
        if (
            previous.goal_node != route.start_node
            or math.dist(
                (
                    previous.positions_m[-1].x,
                    previous.positions_m[-1].y,
                    previous.positions_m[-1].z,
                ),
                (route.positions_m[0].x, route.positions_m[0].y, route.positions_m[0].z),
            )
            > 1e-6
        ):
            raise MissionPreparationBlocked("ROUTE_TOOL_DISCONTINUITY")
        node_ids.extend(route.node_ids[1:])
        edge_ids.extend(route.edge_ids)
        positions.extend(route.positions_m[1:])
    combined = GraphRoute(
        start_node=routes[0].start_node,
        goal_node=routes[-1].goal_node,
        node_ids=node_ids,
        edge_ids=edge_ids,
        positions_m=positions,
        route_length_m=sum(route.route_length_m for route in routes),
        all_edges_flight_verified=bool(edge_ids)
        and all(route.all_edges_flight_verified for route in routes if route.edge_ids),
    )
    return combined


# 功能：
#   计算候选路线距离、净空和爬升启发式成本；能量代理并非电池实测消耗。
# 输入：
#   route：候选几何路线。
#   clearance：该路线对应的连续碰撞检查结果。
# 输出：
#   objectives：候选排序所需的数值指标。
def _route_objectives(route: GraphRoute, clearance: RouteClearanceReport) -> dict[str, float]:
    climb_m = sum(
        max(0.0, second.z - first.z)
        for first, second in zip(route.positions_m, route.positions_m[1:], strict=False)
    )
    objectives = {
        "distance_m": route.route_length_m,
        "minimum_clearance_m": clearance.minimum_clearance_m,
        "energy_proxy": route.route_length_m + climb_m * 2.5 + len(route.edge_ids) * 0.05,
        "transition_count": float(len(route.edge_ids)),
        "qualification_penalty": 0.0 if route.all_edges_flight_verified else 1.0,
    }
    return objectives


# 功能：
#   依据机体包络和定位、跟踪、避障保留量确定常规通行净空目标。
# 输入：
#   vehicle：当前机型尺寸。
# 输出：
#   required_m：常规通行的目标净空，单位米。
def _required_operational_clearance_m(vehicle: VehicleAsset) -> float:
    """Reserve space for localization, tracking and local avoidance corrections."""

    localization_reserve_m = 0.08
    tracking_reserve_m = 0.12
    local_avoidance_reserve_m = 0.15
    envelope_scaled_floor_m = vehicle.body_radius_m * 0.75
    required_m = max(
        PREFERRED_TRANSIT_CLEARANCE_M,
        envelope_scaled_floor_m,
        localization_reserve_m + tracking_reserve_m + local_avoidance_reserve_m,
    )
    return required_m


# 功能：
#   核查各段净空是否容纳闭环误差，并标记需要精细控制的路段。
# 输入：
#   route：待检查的连续路线。
#   clearance：逐段净空；旧摘要缺失逐段数据时使用全路线保守最小值。
#   preferred_transit_clearance_m：常规通行目标净空。
# 输出：
#   assessment：净空预算、精细控制段及拒绝段的有界摘要。
def _route_operational_clearance_assessment(
    route: GraphRoute,
    clearance: RouteClearanceReport,
    *,
    preferred_transit_clearance_m: float,
) -> dict[str, object]:
    """Prove every segment can fund closed-loop motion and identify precision work."""

    segment_clearances = list(clearance.segment_minimum_clearances_m)
    expected_count = max(0, len(route.positions_m) - 1)
    topology_matches = not segment_clearances or len(segment_clearances) == expected_count
    assessed_clearances = segment_clearances or [clearance.minimum_clearance_m]
    rejected_indexes: list[int] = []
    budgets: list[dict[str, float]] = []
    for index, measured_m in enumerate(assessed_clearances):
        try:
            budgets.append(build_tracking_corridor_budget(measured_m))
        except ValueError:
            rejected_indexes.append(index)
    precision_indexes = [
        index
        for index, measured_m in enumerate(assessed_clearances)
        if measured_m < preferred_transit_clearance_m
    ]
    accepted = topology_matches and not rejected_indexes
    assessment = {
        "accepted": accepted,
        "topology_matches": topology_matches,
        "preferred_transit_clearance_m": preferred_transit_clearance_m,
        "measured_minimum_clearance_m": clearance.minimum_clearance_m,
        "segment_budget_count": len(budgets),
        "precision_control_required": bool(precision_indexes),
        "precision_segment_count": len(precision_indexes),
        "precision_segment_indexes": precision_indexes[:256],
        "rejected_segment_count": len(rejected_indexes),
        "rejected_segment_indexes": rejected_indexes[:256],
    }
    # 几何通过不代表实际定位达标；将具体定位要求交给后续规划审查，而不是隐含假设。
    from .collision import assess_tracking_corridor_budget

    if clearance.minimum_clearance_m > 0 and math.isfinite(clearance.minimum_clearance_m):
        assessment["localization_requirement"] = assess_tracking_corridor_budget(
            clearance.minimum_clearance_m)
    return assessment


# 功能：
#   仅移除已通过连续几何检查的度量路线不适用的历史拓扑边验证要求。
# 输入：
#   review：模型提出的评审结果。
#   selected_alternative：实际选中的候选及其净空证据。
# 输出：
#   normalized：保留其他拒绝条件的评审结果。
def _discard_unsupported_metric_edge_history_gate(
    review: PlanCritique,
    selected_alternative: RouteAlternativeCandidate,
) -> PlanCritique:
    """Do not let a model invent historical-edge provenance as a safety gate."""

    if (
        selected_alternative.strategy_tool_id == "planning.candidate-metric-geometry.candidate"
        and selected_alternative.feasible
        and selected_alternative.clearance.accepted
        and review.issue_codes == ["EXEC_ROUTE_NOT_ALL_EDGES_FLIGHT_VERIFIED"]
    ):
        return PlanCritique(accepted=True, issue_codes=[], repair_instructions=[])
    return review


# 功能：
#   按权威阶段事实纠正特定评审误判，不把探索候选失败或尚未发生的飞行证据当作当前失败。
# 输入：
#   review：原始评审。
#   selected_alternative：最终选中路线。
#   selected_plan_receipts_accepted：选中计划的必要工具回执是否通过。
#   future_runtime_evidence_declared：是否已声明确认执行之后需要的证据。
#   failed_exploratory_tool_ids：未选中且失败的探索工具集合。
# 输出：
#   normalized：移除不适用错误码后仍保留有效拒绝理由的结果。
def _normalize_plan_critique(
    review: PlanCritique,
    selected_alternative: RouteAlternativeCandidate,
    *,
    selected_plan_receipts_accepted: bool,
    future_runtime_evidence_declared: bool,
    failed_exploratory_tool_ids: set[str] | None = None,
) -> PlanCritique:
    """Discard critic gates that contradict authoritative planning-stage facts."""

    unsupported_codes: set[str] = set()
    metric_history_normalized = _discard_unsupported_metric_edge_history_gate(
        review,
        selected_alternative,
    )
    if metric_history_normalized != review:
        unsupported_codes.add("EXEC_ROUTE_NOT_ALL_EDGES_FLIGHT_VERIFIED")
    if selected_plan_receipts_accepted:
        unsupported_codes.update(
            {
                "ALL_TOOL_RECEIPTS_NOT_ACCEPTED",
                "DETERMINISTIC_GATE_ALL_TOOL_RECEIPTS_ACCEPTED_FALSE",
            }
        )
    if future_runtime_evidence_declared:
        unsupported_codes.add("FUTURE_RUNTIME_EVIDENCE_NOT_HASH_BOUND")
    if (
        failed_exploratory_tool_ids
        and "planning.candidate-metric-geometry.candidate" in failed_exploratory_tool_ids
        and selected_alternative.strategy_tool_id != "planning.candidate-metric-geometry.candidate"
    ):
        unsupported_codes.update(
            {
                "METRIC_GEOMETRY_CANDIDATE_FAILED",
                "CANDIDATE_METRIC_GEOMETRY_TOOL_FAILED",
            }
        )

    remaining_codes = [code for code in review.issue_codes if code not in unsupported_codes]
    if len(remaining_codes) == len(review.issue_codes):
        return review
    return PlanCritique(
        accepted=review.accepted if remaining_codes else True,
        issue_codes=remaining_codes,
        repair_instructions=review.repair_instructions if remaining_codes else [],
    )


# 功能：
#   区分未选中探索失败和最终计划依赖失败，生成可序列化的工具集合。
# 输入：
#   receipts：本轮全部工具回执。
#   selected_receipt_ids：最终计划依赖的回执标识。
# 输出：
#   tool_ids：未选中且失败的工具标识列表。
def _collect_failed_exploratory_tool_ids(
    receipts: list[ToolReceipt], selected_receipt_ids: set[str]
) -> list[str]:
    """Return stable JSON-safe identifiers for failed, non-selected tool attempts."""

    tool_ids = sorted(
        {
            receipt.tool_id
            for receipt in receipts
            if receipt.outcome != "accepted" and receipt.call_id not in selected_receipt_ids
        }
    )
    return tool_ids


# 功能：
#   将模型选定的有限路线偏好转为明确排序权重，拒绝未知策略。
# 输入：
#   route_policy：经过合同验证的路线偏好名称。
# 输出：
#   weights：距离、净空、能量代理、转移次数及资格风险的权重。
def _route_objective_weights(route_policy: str) -> dict[str, float]:
    """Translate a model's bounded semantic policy into deterministic weights."""

    policies = {
        "balanced": (0.20, 0.46, 0.14, 0.10, 0.10),
        "clearance-first": (0.10, 0.65, 0.05, 0.10, 0.10),
        "minimum-time": (0.50, 0.20, 0.15, 0.05, 0.10),
        "energy-conserving": (0.15, 0.25, 0.40, 0.10, 0.10),
        "few-transitions": (0.15, 0.25, 0.10, 0.40, 0.10),
    }
    values = policies.get(route_policy)
    if values is None:
        raise MissionPreparationBlocked("SEMANTIC_ROUTE_POLICY_INVALID")
    weights = dict(
        zip(
            (
                "distance_m",
                "minimum_clearance_m",
                "energy_proxy",
                "transition_count",
                "qualification_penalty",
            ),
            values,
            strict=True,
        )
    )
    return weights


# 功能：
#   限制轨迹优化只收紧速度和到达容差、增加停留，不能改坐标基准、阶段或放宽稳定要求。
# 输入：
#   track：优化后的轨迹。
#   route：已通过净空检查的世界坐标路线。
#   vehicle：当前机型速度上限。
#   baseline：优化前冻结的轨迹；为空时只验证初始导出几何及机型边界。
# 输出：
#   None。
def _validate_plugin_track_tightening(
    track: Px4Track, route: GraphRoute, vehicle: VehicleAsset, *, baseline: Px4Track | None = None
) -> None:
    track = _artifact_snapshot(track, Px4Track)
    if len(track.source_world_points) != len(route.positions_m):
        raise MissionPreparationBlocked("PLUGIN_TRACK_GEOMETRY_CHANGED")
    root_east, root_north, root_up = track.coordinate_contract.model_root_world_enu_m
    offset_east, offset_north, offset_up = (
        track.coordinate_contract.resolved_collision_center_offset_model_m()
    )
    for source, point, expected in zip(
        track.source_world_points, track.points, route.positions_m, strict=True
    ):
        gates = (
            math.dist(
                (source.east_m, source.north_m, source.up_m),
                (expected.x, expected.y, expected.z),
            )
            <= 1e-6,
            math.dist(
                (
                    point.y + root_east + offset_east,
                    point.x + root_north + offset_north,
                    point.z + root_up + offset_up,
                ),
                (expected.x, expected.y, expected.z),
            )
            <= 1e-6,
            point.speed_limit_mps <= vehicle.max_speed_mps,
        )
        if not all(gates):
            raise MissionPreparationBlocked("PLUGIN_TRACK_SAFETY_ENVELOPE_RELAXED")
    if not track.stop_at_waypoints:
        raise MissionPreparationBlocked("PLUGIN_TRACK_STOP_POLICY_RELAXED")
    if baseline is not None:
        baseline = _artifact_snapshot(baseline, Px4Track)
        if track.coordinate_contract != baseline.coordinate_contract:
            raise MissionPreparationBlocked("PLUGIN_TRACK_COORDINATE_BINDING_CHANGED")
        if len(track.points) != len(baseline.points) or any(
            point.phase != prior.phase or point.speed_limit_mps > prior.speed_limit_mps
            for point, prior in zip(track.points, baseline.points, strict=True)
        ):
            raise MissionPreparationBlocked("PLUGIN_TRACK_SAFETY_ENVELOPE_RELAXED")
        if (
            track.waypoint_hold_seconds < baseline.waypoint_hold_seconds
            or track.waypoint_position_tolerance_m > baseline.waypoint_position_tolerance_m
            or track.waypoint_speed_tolerance_mps > baseline.waypoint_speed_tolerance_mps
            or track.waypoint_stable_window_seconds < baseline.waypoint_stable_window_seconds
            or track.waypoint_settle_timeout_seconds != baseline.waypoint_settle_timeout_seconds
            or track.waypoint_stable_window_seconds > track.waypoint_settle_timeout_seconds
        ):
            raise MissionPreparationBlocked("PLUGIN_TRACK_SETTLE_POLICY_RELAXED")


# 功能：
#   将模型任务顺序绑定到工具求解路线和规范成功证据，形成移动段而非伪造实时操纵输出。
# 输入：
#   contract：冻结任务合同。
#   task_graph：已验证动作依赖图。
#   semantic_plan：模型选择的目标序列。
#   routes：工具求出的逐段几何。
#   map_graph：当前拓扑及边速度限制。
#   validated_clearance_m：连续碰撞工具测得的全路线最小净空。
#   domain_actions：可选的当前动作目录。
# 输出：
#   flight_plan：绑定语义计划摘要、移动任务与路线的飞行计划。
def _flight_plan(
    contract: MissionContract,
    task_graph: TaskGraph,
    semantic_plan: SemanticPlan,
    routes: list[GraphRoute],
    map_graph: MapAsset,
    validated_clearance_m: float,
    domain_actions: DomainActionCatalog | None = None,
) -> FlightPlan:
    edges = {edge.edge_id: edge for edge in map_graph.edges}
    allowed_movement_actions = (
        movement_action_ids(domain_actions) if domain_actions is not None else set(MOVEMENT_ACTIONS)
    )
    movement_tasks = [task for task in task_graph.nodes if task.action in allowed_movement_actions]
    used_task_ids: set[str] = set()
    segments: list[PlanSegment] = []
    for index, (target, route) in enumerate(
        zip(semantic_plan.ordered_targets, routes, strict=True), start=1
    ):
        task = next(
            (
                item
                for item in movement_tasks
                if item.target_node == target and item.task_id not in used_task_ids
            ),
            None,
        )
        if task is None:
            raise MissionPreparationBlocked(f"NO_MOVEMENT_TASK_FOR_TARGET: {target}")
        used_task_ids.add(task.task_id)
        route_edges = [edges[edge_id] for edge_id in route.edge_ids if edge_id in edges]
        metric_geometry_route = len(route_edges) != len(route.edge_ids)
        speed_limit_mps = (
            min(edge.speed_limit_mps for edge in route_edges)
            if route_edges and not metric_geometry_route
            else 0.5
        )
        segments.append(
            PlanSegment(
                segment_id=f"segment-{index:03d}",
                task_id=task.task_id,
                from_node=route.start_node,
                to_node=route.goal_node,
                path=[
                    RoutePoint(node_id=node_id, position_m=position)
                    for node_id, position in zip(route.node_ids, route.positions_m, strict=True)
                ],
                speed_limit_mps=speed_limit_mps,
                # The conservative continuous collision tool is authoritative.
                # Graph-edge clearance metadata can be zero when not yet qualified.
                minimum_clearance_m=validated_clearance_m,
                success_evidence=task.success_evidence,
            )
        )
    if used_task_ids != {task.task_id for task in movement_tasks}:
        raise MissionPreparationBlocked("TASK_GRAPH_UNSCHEDULED_MOVEMENT")
    flight_plan = FlightPlan(
        revision=1,
        contract_id=contract.contract_id,
        segments=segments,
        semantic_plan_sha256=sha256_json(semantic_plan),
    )
    return flight_plan


# 功能：
#   为每段到达节点生成默认检查点，连续路段的公共端点只占一个轨迹索引。
# 输入：
#   contract：当前任务合同。
#   flight_plan：已绑定的移动段计划。
# 输出：
#   checkpoints_contract：任务、段、目标及轨迹索引对应的检查点合同。
def _runtime_checkpoints(
    contract: MissionContract, flight_plan: FlightPlan
) -> RuntimeCheckpointContract:
    checkpoints: list[RuntimeCheckpoint] = []
    track_point_index = 0
    for index, segment in enumerate(flight_plan.segments, start=1):
        track_point_index += len(segment.path) - 1
        checkpoints.append(
            RuntimeCheckpoint(
                checkpoint_id=f"checkpoint-{index:03d}",
                segment_id=segment.segment_id,
                task_id=segment.task_id,
                track_point_index=track_point_index,
                target_node=segment.to_node,
            )
        )
    checkpoints_contract = RuntimeCheckpointContract(
        contract_id=contract.contract_id, checkpoints=checkpoints
    )
    return checkpoints_contract


# 功能：
#   重验计划门控的布尔值和逐项判据，防止缺失结果或字符串 false 被当成通过。
# 输入：
#   results：插件门控及专家验证结果。
# 输出：
#   rejected：明确拒绝的结果列表；结构错误或自相矛盾时阻断准备。
def _rejected_plan_validators(results: list[Any]) -> list[dict[str, Any]]:
    rejected = []
    for value in results:
        if not isinstance(value, dict) or type(value.get("accepted")) is not bool:
            raise MissionPreparationBlocked("PLUGIN_PLAN_GATE_RESULT_INVALID")
        gates = value.get("deterministic_gates")
        if gates is not None and (
            not isinstance(gates, dict)
            or not gates
            or any(type(gate) is not bool for gate in gates.values())
            or (value["accepted"] and not all(gates.values()))
        ):
            raise MissionPreparationBlocked("PLUGIN_PLAN_GATE_RESULT_INVALID")
        for field in ("issue_codes", "repair_instructions"):
            entries = value.get(field, [])
            if (
                not isinstance(entries, list)
                or len(entries) > 32
                or any(
                    not isinstance(item, str) or not item.strip() or len(item) > 2_000
                    for item in entries
                )
            ):
                raise MissionPreparationBlocked("PLUGIN_PLAN_GATE_RESULT_INVALID")
        if value["accepted"] is False:
            rejected.append(value)
    return rejected


class MissionOrchestrator:
    """Prepare, but never silently actuate, a hash-bound simulated mission."""

    # 功能：
    #   冻结任务准备依赖，连接插件和云端模型；此对象不产生实时飞控命令。
    # 输入：
    #   config：有界规划和模型调用策略。
    #   map_catalog：当前地图实体目录。
    #   map_graph：当前地图拓扑。
    #   semantic_path：当前地图语义文件。
    #   vehicle_sdf：当前机型 SDF 文件。
    #   vehicle_asset_id：被选中机型的身份。
    #   vehicle：该机型的尺寸、质量和运动限制。
    #   context_store：由调用方管理生命周期的会话存储。
    #   primary_port：可选的借用主模型端口。
    #   critic_port：可选的借用评审端口。
    #   model_ports：额外借用端口及明确角色覆盖。
    #   tool_registry：可选的已构建工具注册表。
    #   extension_registry：可选的已构建扩展注册表。
    #   plugin_snapshot：与工具注册表成对提供的冻结插件选择。
    #   initial_hook_receipts：附件与账户上下文阶段已有回执。
    #   harness_topology_override：本次采用的拓扑覆盖。
    #   harness_revision_binding：与该覆盖对应的内容绑定。
    # 输出：
    #   None。
    def __init__(
        self,
        *,
        config: PreparationConfig,
        map_catalog: MapCatalog,
        map_graph: MapAsset,
        semantic_path: Path,
        vehicle_sdf: Path,
        vehicle_asset_id: str,
        vehicle: VehicleAsset,
        context_store: ContextStore,
        primary_port: StructuredModelPort | None = None,
        critic_port: StructuredModelPort | None = None,
        model_ports: dict[str, StructuredModelPort] | None = None,
        tool_registry: ToolRegistry | None = None,
        extension_registry: ExtensionRegistry | None = None,
        plugin_snapshot: PluginSnapshot | None = None,
        initial_hook_receipts: list[PluginHookReceipt] | None = None,
        harness_topology_override: HarnessTopology | None = None,
        harness_revision_binding: dict[str, object] | None = None,
    ) -> None:
        self.config = config
        config.__post_init__()
        self.map_catalog = _artifact_snapshot(map_catalog, MapCatalog)
        self.map_graph = _artifact_snapshot(map_graph, MapAsset)
        self.semantic_path = semantic_path
        self.vehicle_sdf = vehicle_sdf
        self.vehicle_asset_id = vehicle_asset_id
        if vehicle.asset_id != vehicle_asset_id:
            raise ValueError("vehicle metadata does not match vehicle_asset_id")
        self.vehicle = _artifact_snapshot(vehicle, VehicleAsset)
        self.context_store = context_store
        if (tool_registry is None) != (plugin_snapshot is None):
            raise ValueError("tool_registry and plugin_snapshot must be supplied together")
        if tool_registry is None or plugin_snapshot is None:
            tool_registry, plugin_snapshot = build_discovered_tool_registry(
                ToolEnvironment(
                    map_graph=self.map_graph,
                    semantic_path=semantic_path,
                    vehicle_diameter_m=vehicle.body_radius_m * 2.0,
                    vehicle_height_m=vehicle.body_height_m,
                    waypoint_hold_seconds=config.waypoint_hold_seconds,
                    vehicle=self.vehicle,
                    planning_phase="initial",
                )
            )
        self.tool_registry = tool_registry
        self.plugin_snapshot = _artifact_snapshot(plugin_snapshot, PluginSnapshot)
        self.extension_registry = extension_registry or build_discovered_extension_registry(
            plugin_snapshot
        )
        self._hook_receipts: list[PluginHookReceipt] = []
        self._initial_hook_receipts = [
            _artifact_snapshot(receipt, PluginHookReceipt)
            for receipt in (initial_hook_receipts or [])
        ]
        self.harness_topology_override = (
            _artifact_snapshot(harness_topology_override, HarnessTopology)
            if harness_topology_override is not None
            else None
        )
        self.harness_revision_binding = json.loads(encode_json(harness_revision_binding or {}))
        self._model_call_count = 0
        self._model_call_budget = config.maximum_model_calls
        self._model_media: list[dict[str, object]] = []
        self._preparation_lock = Lock()
        self._closed = False
        self.model_ports = dict(model_ports or {})
        # 先完成可能失败的依赖检查，再创建连接；第二个连接失败也必须回收第一个。
        with ExitStack() as acquired:
            for name, supplied, provider in (
                ("primary", primary_port, config.provider),
                ("critic", critic_port, config.critic_provider),
            ):
                effective = self.model_ports.get(name, supplied)
                if effective is None:
                    effective = StructuredModelPort(
                        provider,
                        max_attempts=config.max_provider_attempts,
                        timeout_seconds=config.model_timeout_seconds,
                    )
                    acquired.callback(effective.close)
                self.model_ports[name] = effective
            self.primary = self.model_ports["primary"]
            self.critic = self.model_ports["critic"]
            self._owned_ports = acquired.pop_all()

    # 功能：
    #   关闭本对象创建的模型连接；借用端口和会话数据库仍由调用方负责，执行中禁止关闭。
    # 输入：
    #   self：待回收的编排器。
    # 输出：
    #   None。
    def close(self) -> None:
        if not self._preparation_lock.acquire(blocking=False):
            raise MissionPreparationBlocked("PREPARATION_ALREADY_RUNNING")
        try:
            if not self._closed:
                self._closed = True
                self._owned_ports.close()
        finally:
            self._preparation_lock.release()

    # 功能：
    #   冻结插件调用回执并记录到本轮证据链，失败回执与成功回执同样保留。
    # 输入：
    #   receipts：扩展注册表返回的回执。
    #   evidence：本次准备独占的证据接收器。
    # 输出：
    #   None。
    def _record_hook_receipts(
        self, receipts: list[PluginHookReceipt], evidence: EvidenceChain
    ) -> None:
        for receipt in receipts:
            receipt = _artifact_snapshot(receipt, PluginHookReceipt)
            self._hook_receipts.append(receipt)
            evidence.append(
                f"plugin-hook.{receipt.slot_id}.{receipt.hook}",
                receipt.model_dump(mode="json"),
            )

    # 功能：
    #   调用单选扩展并保存回执，必要插件缺失或执行错误均阻断准备。
    # 输入：
    #   slot_id：扩展插槽。
    #   hook：该插槽的操作名称。
    #   evidence：本轮证据链。
    #   required：是否必须存在选中的实现。
    #   kwargs：按钩子合同传入的参数。
    # 输出：
    #   output：扩展输出，非必要且无实现时为空。
    def _invoke_single_extension(
        self,
        slot_id: str,
        hook: str,
        *,
        evidence: EvidenceChain,
        required: bool = False,
        **kwargs: Any,
    ) -> Any | None:
        try:
            output, receipts = self.extension_registry.invoke_single(
                slot_id, hook, required=required, **kwargs
            )
        except ExtensionExecutionError as error:
            self._record_hook_receipts([error.receipt], evidence)
            raise MissionPreparationBlocked(str(error)) from error
        self._record_hook_receipts(receipts, evidence)
        return output

    # 功能：
    #   调用该插槽选择的全部扩展，收集输出并保留完整失败或成功回执。
    # 输入：
    #   slot_id：多选扩展插槽。
    #   hook：调用操作。
    #   evidence：本轮证据链。
    #   kwargs：各扩展读取的合同参数。
    # 输出：
    #   outputs：按注册顺序排列的扩展结果。
    def _invoke_multiple_extensions(
        self,
        slot_id: str,
        hook: str,
        *,
        evidence: EvidenceChain,
        **kwargs: Any,
    ) -> list[Any]:
        try:
            outputs, receipts = self.extension_registry.invoke_multiple(slot_id, hook, **kwargs)
        except ExtensionExecutionError as error:
            self._record_hook_receipts([error.receipt], evidence)
            raise MissionPreparationBlocked(str(error)) from error
        self._record_hook_receipts(receipts, evidence)
        return outputs

    # 功能：
    #   依序运行材料变换链，链内失败立即保留回执并停止后续准备。
    # 输入：
    #   slot_id：流水线插槽。
    #   hook：材料变换操作。
    #   value：流水线初始材料。
    #   evidence：本轮证据链。
    #   kwargs：各步骤共享的只读任务背景。
    # 输出：
    #   output：流水线最终输出，调用方仍须验证固定核心安全条件。
    def _invoke_extension_pipeline(
        self,
        slot_id: str,
        hook: str,
        value: Any,
        *,
        evidence: EvidenceChain,
        **kwargs: Any,
    ) -> Any:
        try:
            output, receipts = self.extension_registry.invoke_pipeline(
                slot_id, hook, value, **kwargs
            )
        except ExtensionExecutionError as error:
            self._record_hook_receipts([error.receipt], evidence)
            raise MissionPreparationBlocked(str(error)) from error
        self._record_hook_receipts(receipts, evidence)
        return output

    # 功能：
    #   按角色路由和物理次数预算调用模型，保留每个响应的用量，再执行共识和输出验证。
    # 输入：
    #   port：此角色建议使用的端口。
    #   role：计划解析、分解、评审等有限职责。
    #   output_type：该职责的结构输出合同。
    #   instructions：宿主角色指令。
    #   input_artifact：本次调用任务材料。
    #   conversation_id：所属线程。
    #   evidence：本轮证据链。
    # 输出：
    #   result：选中材料及主响应、支持响应和失败尝试记录。
    def _call(
        self,
        *,
        port: StructuredModelPort,
        role: str,
        output_type: Any,
        instructions: str,
        input_artifact: dict[str, object],
        conversation_id: str,
        evidence: EvidenceChain,
    ) -> StructuredCallResult[Any]:
        """Run bounded role calls and retain every returned response's usage before consensus.

        Routing can propose peers, not amplify a duplicated port into agreement.
        Failed peers may fall through within the physical-attempt budget; even
        dissenting responses consume provider resources and require a receipt.
        """
        requested_port = "critic" if port is self.critic else "primary"
        role_policy = self._invoke_single_extension(
            "models.role-policy",
            "select_port",
            evidence=evidence,
            role=role,
            requested_port=requested_port,
        )
        if isinstance(role_policy, dict):
            selected = role_policy.get("port")
            if isinstance(selected, str) and selected in self.model_ports:
                requested_port = selected
                port = self.model_ports[selected]
        route_policy = self._invoke_single_extension(
            "models.runtime-router",
            "route_model",
            evidence=evidence,
            required=True,
            role=role,
            requested_port=requested_port,
            available_ports=sorted(self.model_ports),
        )
        if not isinstance(route_policy, dict):
            raise MissionPreparationBlocked("MODEL_ROUTER_INVALID")
        candidates = route_policy.get("candidates")
        if not isinstance(candidates, list) or len(candidates) > 64:
            raise MissionPreparationBlocked("MODEL_ROUTER_CANDIDATES_INVALID")
        candidate_ports: list[str] = []
        seen_ports: set[int] = set()
        for name in candidates:
            if not isinstance(name, str) or name not in self.model_ports:
                continue
            identity = id(self.model_ports[name])
            if identity not in seen_ports:
                candidate_ports.append(name)
                seen_ports.add(identity)
        if not candidate_ports:
            raise MissionPreparationBlocked("MODEL_ROUTER_NO_AVAILABLE_PORT")
        model_media = self._model_media if role == "intent_parser" else []
        if model_media:
            candidate_ports = [
                port_name
                for port_name in candidate_ports
                if self.model_ports[port_name].supports_image_input
            ]
            if not candidate_ports:
                raise MissionPreparationBlocked("MODEL_ROUTER_NO_MULTIMODAL_PORT")
        consensus = self._invoke_single_extension(
            "models.consensus-policy",
            "select_consensus",
            evidence=evidence,
            required=True,
            role=role,
            candidates=candidate_ports,
        )
        if not isinstance(consensus, dict):
            raise MissionPreparationBlocked("MODEL_CONSENSUS_INVALID")
        minimum_responses = consensus.get("minimum_responses", 1)
        maximum_responses = consensus.get("maximum_responses", 1)
        if (
            type(minimum_responses) is not int
            or type(maximum_responses) is not int
            or not 1 <= minimum_responses <= maximum_responses <= 3
            or type(consensus.get("require_identical", False)) is not bool
            or type(consensus.get("record_dissent", False)) is not bool
        ):
            raise MissionPreparationBlocked("MODEL_CONSENSUS_BOUNDS_INVALID")
        if len(candidate_ports) < minimum_responses:
            raise MissionPreparationBlocked("MODEL_CONSENSUS_INSUFFICIENT_RESPONSES:distinct ports")
        instructions = self._invoke_extension_pipeline(
            "models.prompt-packs",
            "augment_prompt",
            instructions,
            evidence=evidence,
            role=role,
        )
        if not isinstance(instructions, str) or not instructions.strip():
            raise MissionPreparationBlocked("PLUGIN_PROMPT_PIPELINE_INVALID")
        results: list[StructuredCallResult[Any]] = []
        response_metering: list[Any] = []
        failures: list[str] = []
        attempted_ports: list[str] = []
        for port_name in candidate_ports:
            if len(results) >= maximum_responses:
                break
            remaining_attempts = self._model_call_budget - self._model_call_count
            if remaining_attempts <= 0:
                raise MissionPreparationBlocked("HARNESS_MODEL_CALL_BUDGET_EXCEEDED")
            candidate_port = self.model_ports[port_name]
            attempted_ports.append(port_name)
            role_context = f"{conversation_id}::{role}::{port_name}"
            provider_context_key = f"{candidate_port.settings.name}:{candidate_port.settings.model}"
            if self.config.persisted_task_context:
                stored = self.context_store.window(role_context)
                previous = stored.previous_response_ids.get(provider_context_key)
                if previous and candidate_port.supports_provider_context:
                    candidate_port.restore_provider_context(role_context, previous)
            try:
                candidate_result = candidate_port.call(
                    role=role,
                    output_type=output_type,
                    instructions=instructions,
                    input_artifact=input_artifact,
                    context_id=role_context,
                    multimodal=model_media,
                    maximum_physical_attempts=remaining_attempts,
                )
            except ModelInvocationError as error:
                self._model_call_count += error.attempts_used
                failure = f"{port_name}:{type(error).__name__}:{str(error)[:360]}"
                failures.append(failure)
                evidence.append(
                    f"model.{role}.provider-attempts-failed",
                    {
                        "port": port_name,
                        "attempts_used": error.attempts_used,
                        "issue": str(error)[:360],
                        "physical_attempt_count": self._model_call_count,
                        "physical_attempt_budget": self._model_call_budget,
                    },
                )
                continue
            self._model_call_count += candidate_result.record.attempt
            # 插件计量也可能失败。原供应商用量先落证据，失败不能抹去已消耗的调用。
            evidence.append(
                f"model.{role}.provider-response",
                {
                    "port": port_name,
                    "record": candidate_result.record.model_dump(mode="json"),
                    "attempt_failures": list(candidate_result.attempt_failures),
                },
            )
            metering = self._invoke_multiple_extensions(
                "models.token-meters",
                "measure_tokens",
                evidence=evidence,
                role=role,
                record=candidate_result.record,
            )
            response_metering.append(metering)
            # Persist before quorum/output guards: rejected answers cost tokens
            # too. This local receipt is not a second cloud billing debit.
            evidence.append(
                f"model.{role}.response-usage",
                {
                    "port": port_name,
                    "record": candidate_result.record.model_dump(mode="json"),
                    "metering": metering,
                },
            )
            if (
                self.config.persisted_task_context
                and candidate_result.record.response_id
                and candidate_port.supports_provider_context
            ):
                self.context_store.set_response_id(
                    role_context, provider_context_key, candidate_result.record.response_id
                )
            results.append(candidate_result)
        if len(results) < minimum_responses:
            raise MissionPreparationBlocked(
                "MODEL_CONSENSUS_INSUFFICIENT_RESPONSES:" + ",".join(failures)
            )
        output_hashes = [sha256_json(item.artifact) for item in results]
        result, consensus_resolution = _resolve_consensus_result(
            results,
            require_identical=consensus.get("require_identical") is True,
            record_dissent=consensus.get("record_dissent") is True,
        )
        evidence.append(
            f"model.{role}.consensus",
            {
                "policy": consensus,
                "candidate_ports": attempted_ports,
                "response_hashes": output_hashes,
                "response_records": [
                    {
                        "call_id": item.record.call_id,
                        "provider": item.record.provider,
                        "model": item.record.model,
                        "input_sha256": item.record.input_sha256,
                        "output_sha256": item.record.output_sha256,
                        "input_tokens": item.record.input_tokens,
                        "output_tokens": item.record.output_tokens,
                        "latency_ms": item.record.latency_ms,
                        "attempt_failures": list(item.attempt_failures),
                    }
                    for item in results
                ],
                "selected_call_id": result.record.call_id,
                "resolution": consensus_resolution,
                "failures": failures,
                "selected_attempt_failures": list(result.attempt_failures),
                "physical_attempt_count": self._model_call_count,
                "physical_attempt_budget": self._model_call_budget,
            },
        )
        output_envelope = {
            "artifact": result.artifact.model_dump(mode="json"),
            "record": result.record.model_dump(mode="json"),
        }
        guarded_output = self._invoke_extension_pipeline(
            "models.structured-output-guards",
            "validate_output",
            output_envelope,
            evidence=evidence,
            role=role,
            expected_schema=output_type.__name__,
        )
        if sha256_json(guarded_output) != sha256_json(output_envelope):
            raise MissionPreparationBlocked("PLUGIN_MODEL_OUTPUT_MUTATION_FORBIDDEN")
        metering = next(
            usage
            for item, usage in zip(results, response_metering, strict=True)
            if item.record is result.record
        )
        self.context_store.append(
            conversation_id,
            role="assistant",
            event_type=f"model.{role}",
            payload={
                "artifact": result.artifact.model_dump(mode="json"),
                "record": result.record.model_dump(mode="json"),
                "attempt_failures": list(result.attempt_failures),
                "metering": metering,
            },
        )
        evidence.append(
            f"model.{role}",
            {
                "artifact": result.artifact.model_dump(mode="json"),
                "record": result.record.model_dump(mode="json"),
                "attempt_failures": list(result.attempt_failures),
                "metering": metering,
            },
        )
        return result

    # 功能：
    #   将同一工具回执写入线程历史和本轮证据，不把探索失败改成成功。
    # 输入：
    #   receipt：已验证工具回执。
    #   conversation_id：所属线程。
    #   context_store：线程历史存储。
    #   evidence：本轮证据链。
    # 输出：
    #   None。
    @staticmethod
    def _record_tool(
        receipt: ToolReceipt,
        *,
        conversation_id: str,
        context_store: ContextStore,
        evidence: EvidenceChain,
    ) -> None:
        from .model_harness.progress import report_progress
        report_progress("tools", f"工具 {receipt.tool_id} 已返回回执，正在记录结果并核对任务约束。", f"Tool {receipt.tool_id} returned a receipt; recording the result and checking task constraints.")
        payload = receipt.model_dump(mode="json")
        context_store.append(
            conversation_id, role="tool", event_type=f"tool.{receipt.tool_id}", payload=payload
        )
        evidence.append(f"tool.{receipt.tool_id}", payload)

    # 功能：
    #   调用最终计划依赖的工具，成功或失败都先保留回执，再返回结果或阻断准备。
    # 输入：
    #   slot_id：候选排序或轨迹导出等必要工具插槽。
    #   value：固定合同输入。
    #   conversation_id：所属任务线程。
    #   evidence：本轮证据链。
    # 输出：
    #   result：工具结果与实际回执的二元组。
    def _call_required_slot(
        self, slot_id: str, value: BaseModel, *, conversation_id: str, evidence: EvidenceChain
    ) -> tuple[Any, ToolReceipt]:
        try:
            output, receipt = self.tool_registry.call_slot(slot_id, value)
        except ToolExecutionError as error:
            self._record_tool(
                error.receipt,
                conversation_id=conversation_id,
                context_store=self.context_store,
                evidence=evidence,
            )
            raise MissionPreparationBlocked(
                "REQUIRED_TOOL_FAILED:" + ",".join(error.receipt.issue_codes)
            ) from error
        self._record_tool(
            receipt,
            conversation_id=conversation_id,
            context_store=self.context_store,
            evidence=evidence,
        )
        result = output, receipt
        return result

    # 功能：
    #   串行准备一个真实请求，隔离调用方可变输入，拒绝并发复用或关闭后的调用。
    # 输入：
    #   request：当前自然语言任务及资产、账户、控制面绑定。
    #   output_dir：本次准备专用的空目录。
    # 输出：
    #   prepared：等待用户确认的计划包，不代表飞行已经发生。
    def prepare(self, request: MissionRequest, output_dir: Path) -> PreparedMission:
        if not self._preparation_lock.acquire(blocking=False):
            raise MissionPreparationBlocked("PREPARATION_ALREADY_RUNNING")
        try:
            if self._closed:
                raise MissionPreparationBlocked("PREPARATION_CLOSED")
            frozen_request = _artifact_snapshot(request, MissionRequest)
            prepared = self._prepare(frozen_request, output_dir)
            return prepared
        finally:
            self._preparation_lock.release()

    # 功能：
    #   执行意图理解、工具查询、分解、规划、几何验证、修缮评审和证据冻结的完整准备链。
    # 输入：
    #   request：经入口冻结的任务请求。
    #   output_dir：本轮阶段快照与最终合同的目录。
    # 输出：
    #   prepared：绑定当前地图、机型、插件、调用记录与执行前置条件的计划包。
    def _prepare(self, request: MissionRequest, output_dir: Path) -> PreparedMission:
        try:
            model_request_context = _model_request_context(request)
        except ValueError as error:
            raise MissionPreparationBlocked(str(error)) from error
        explicit_constraint_hints = _explicit_constraint_hints(request)
        _validate_explicit_payload_capacity(explicit_constraint_hints, self.vehicle)
        semantic_sha256 = _file_sha256(self.semantic_path)
        vehicle_sha256 = _file_sha256(self.vehicle_sdf)
        if semantic_sha256 != self.map_catalog.semantic_sha256:
            raise MissionPreparationBlocked("PREPARATION_SEMANTIC_CATALOG_MISMATCH")
        # 坐标基准或碰撞中心缺失不能等到多轮付费规划后才在导出步骤发现。
        try:
            load_map_runtime_bindings(
                read_map_semantic_object(self.semantic_path)
            )
            resolve_vehicle_collision_center_offset(self.vehicle)
        except ValueError as error:
            raise MissionPreparationBlocked(str(error)) from error
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"preparation directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        # 同一对象由互斥锁保护，另一个对象/进程抢占相同输出目录则由独占标记拒绝。
        with (output_dir / ".preparation-claim").open("x", encoding="utf-8") as claim:
            claim.write(request.conversation_id + "\n")
            claim.flush()
            os.fsync(claim.fileno())
        thread = self.context_store.lifecycle.ensure_thread(request.conversation_id)
        if thread.state in {"executing", "holding", "landing"}:
            raise LifecycleTransitionError("ACTIVE_EXECUTION_REJECTS_PREFLIGHT_REPLAN")
        evidence = EvidenceChain(output_dir / "evidence.jsonl")
        raw_pair_binding = request.input_metadata.get("asset_pair_qualification")
        if raw_pair_binding is not None:
            try:
                pair_binding = _artifact_snapshot(
                    raw_pair_binding, MissionAssetPairQualificationBinding
                )
            except ValueError as error:
                raise MissionPreparationBlocked(
                    "ASSET_PAIR_QUALIFICATION_BINDING_INVALID"
                ) from error
            if (
                pair_binding.map_asset_id != self.map_graph.asset_id
                or pair_binding.vehicle_asset_id != self.vehicle.asset_id
            ):
                raise MissionPreparationBlocked("ASSET_PAIR_QUALIFICATION_IDENTITY_MISMATCH")
            evidence.append(
                "mission.asset-pair-qualification",
                pair_binding.model_dump(mode="json"),
            )
        self._hook_receipts = list(self._initial_hook_receipts)
        for receipt in self._initial_hook_receipts:
            evidence.append(
                f"plugin-hook.{receipt.slot_id}.{receipt.hook}",
                receipt.model_dump(mode="json"),
            )
        self._model_call_count = 0
        self.tool_registry.configure_extensions(
            self.extension_registry,
            receipt_sink=lambda receipts: self._record_hook_receipts(receipts, evidence),
        )
        harness_profile = self._invoke_single_extension(
            "harness.profile",
            "resolve_profile",
            evidence=evidence,
            required=True,
            request=request,
            map_catalog=self.map_catalog,
            map_graph=self.map_graph,
        )
        if not isinstance(harness_profile, dict):
            raise MissionPreparationBlocked("HARNESS_PROFILE_INVALID")
        evidence.append("harness.profile", harness_profile)
        if self.harness_topology_override is None:
            topology_value = self._invoke_single_extension(
                "harness.workflow-topology",
                "resolve_topology",
                evidence=evidence,
                required=True,
                request=request,
                harness_profile=harness_profile,
                map_catalog=self.map_catalog,
                map_graph=self.map_graph,
            )
            try:
                harness_topology = _artifact_snapshot(topology_value, HarnessTopology)
            except ValueError as error:
                raise MissionPreparationBlocked("HARNESS_TOPOLOGY_INVALID") from error
        else:
            harness_topology = _artifact_snapshot(self.harness_topology_override, HarnessTopology)
            evidence.append("harness.visual-revision", self.harness_revision_binding)
        policy_hooks = {
            "scheduler": ("harness.scheduler", "resolve_schedule"),
            "retry": ("harness.retry-policy", "resolve_retry"),
            "timeout": ("harness.timeout-policy", "resolve_timeout"),
            "budget": ("harness.budget-policy", "resolve_budget"),
            "fallback": ("harness.fallback-policy", "resolve_fallback"),
            "cache": ("harness.cache-policy", "resolve_cache"),
        }
        harness_policies: dict[str, dict[str, object]] = {}
        for policy_name, (slot_id, hook_name) in policy_hooks.items():
            policy_value = self._invoke_single_extension(
                slot_id,
                hook_name,
                evidence=evidence,
                required=True,
                request=request,
                harness_profile=harness_profile,
                harness_topology=harness_topology,
            )
            if not isinstance(policy_value, dict):
                raise MissionPreparationBlocked(f"HARNESS_POLICY_INVALID:{policy_name}")
            harness_policies[policy_name] = policy_value
        try:
            runtime_policy = resolve_harness_runtime_policy(
                harness_topology,
                harness_policies,
                maximum_model_calls=self.config.maximum_model_calls,
                maximum_tool_calls=self.config.maximum_optional_tool_calls,
                model_timeout_seconds=self.config.model_timeout_seconds,
            )
        except (TypeError, ValueError) as error:
            raise MissionPreparationBlocked("HARNESS_RUNTIME_POLICY_INVALID") from error
        harness_topology = runtime_policy.topology
        stage_runtime = HarnessStageRuntime(harness_topology)
        self._model_call_budget = runtime_policy.maximum_model_calls
        for port in self.model_ports.values():
            configure = getattr(port, "configure_execution_policy", None)
            if callable(configure):
                configure(
                    maximum_attempts=runtime_policy.provider_attempts,
                    timeout_seconds=runtime_policy.model_timeout_seconds,
                )
        self.tool_registry.configure_runtime_limits(
            maximum_calls=runtime_policy.maximum_tool_calls,
            timeout_seconds=runtime_policy.tool_timeout_seconds,
            maximum_total_calls=256,
            budget_exempt_slot_ids=CORE_PLANNING_SLOTS,
        )
        evidence.append("harness.topology", harness_topology.model_dump(mode="json"))
        evidence.append("harness.policies", harness_policies)
        evidence.append("harness.runtime-policy", runtime_policy.model_dump(mode="json"))
        simulator_descriptor = self._invoke_single_extension(
            "simulation.simulator-descriptor",
            "describe_simulator",
            evidence=evidence,
            required=True,
            request=request,
        )
        simulation_capabilities = {
            "simulator": simulator_descriptor,
            "physics": self._invoke_multiple_extensions(
                "simulation.physics-models",
                "describe_physics",
                evidence=evidence,
                request=request,
            ),
            "sensors": self._invoke_multiple_extensions(
                "simulation.sensor-models",
                "describe_sensor",
                evidence=evidence,
                request=request,
            ),
            "environment": self._invoke_multiple_extensions(
                "simulation.environment-models",
                "describe_environment",
                evidence=evidence,
                request=request,
            ),
            "clock": self._invoke_single_extension(
                "simulation.clock-policy",
                "resolve_clock",
                evidence=evidence,
                required=True,
                request=request,
            ),
            "monte_carlo": self._invoke_single_extension(
                "simulation.monte-carlo-policy",
                "resolve_monte_carlo",
                evidence=evidence,
                required=True,
                request=request,
            ),
        }
        simulation_capabilities["native_runtime"] = {
            "transport": self._invoke_single_extension(
                "native.transport",
                "transport_message",
                evidence=evidence,
                required=True,
                request=request,
            ),
            "state_estimator": self._invoke_single_extension(
                "native.state-estimator",
                "estimate_state",
                evidence=evidence,
                required=True,
                request=request,
            ),
            "localization": self._invoke_single_extension(
                "native.localization",
                "localize",
                evidence=evidence,
                required=True,
                request=request,
            ),
            "controller": self._invoke_single_extension(
                "native.controller",
                "control_policy",
                evidence=evidence,
                required=True,
                request=request,
            ),
            "watchdog": self._invoke_single_extension(
                "native.watchdog",
                "resolve_watchdog",
                evidence=evidence,
                required=True,
                request=request,
            ),
            "telemetry": self._invoke_multiple_extensions(
                "native.telemetry",
                "normalize_telemetry",
                evidence=evidence,
                request=request,
            ),
            "perception": self._invoke_multiple_extensions(
                "native.perception",
                "normalize_telemetry",
                evidence=evidence,
                request=request,
            ),
            "payload": self._invoke_multiple_extensions(
                "native.payload-drivers",
                "payload_command",
                evidence=evidence,
                request=request,
            ),
            "blackbox": self._invoke_multiple_extensions(
                "native.blackbox",
                "normalize_telemetry",
                evidence=evidence,
                request=request,
            ),
        }
        evidence.append("simulation.capabilities", simulation_capabilities)
        event_output = self._invoke_single_extension(
            "harness.event-bus",
            "transport_message",
            evidence=evidence,
            required=True,
            event="mission.preparation.started",
            payload={
                "conversation_id": request.conversation_id,
                "topology_id": harness_topology.topology_id,
                "profile_id": harness_profile.get("profile_id"),
            },
        )
        observer_outputs = self._invoke_multiple_extensions(
            "harness.observers",
            "observe_harness",
            evidence=evidence,
            event="mission.preparation.started",
            payload={
                "conversation_id": request.conversation_id,
                "topology_id": harness_topology.topology_id,
            },
        )
        evidence.append(
            "harness.event",
            {"transport": event_output, "observers": observer_outputs},
        )
        action_pack_outputs = self._invoke_multiple_extensions(
            "mission.action-packs",
            "declare_actions",
            evidence=evidence,
            request=request,
            harness_profile=harness_profile,
            map_catalog=self.map_catalog,
            map_graph=self.map_graph,
        )
        try:
            domain_actions = merge_action_packs(action_pack_outputs)
        except (ValueError, RuntimeError) as error:
            raise MissionPreparationBlocked("DOMAIN_ACTION_CATALOG_INVALID") from error
        evidence.append("mission.domain-actions", domain_actions.model_dump(mode="json"))
        runtime_adapter_outputs = self._invoke_multiple_extensions(
            "runtime.action-adapters",
            "declare_runtime_action_adapters",
            evidence=evidence,
            request=request,
            harness_profile=harness_profile,
            map_graph=self.map_graph,
            vehicle=self.vehicle,
        )
        try:
            runtime_action_adapters = merge_runtime_action_adapters(runtime_adapter_outputs)
        except (ValueError, RuntimeError) as error:
            raise MissionPreparationBlocked("RUNTIME_ACTION_ADAPTER_CATALOG_INVALID") from error
        evidence.append(
            "mission.runtime-action-adapters",
            runtime_action_adapters.model_dump(mode="json"),
        )
        self.context_store.append(
            request.conversation_id,
            role="user",
            event_type="mission.request",
            payload=request.model_dump(mode="json"),
        )
        evidence.append("mission.request", request.model_dump(mode="json"))
        try:
            request_stage = stage_runtime.complete(
                "mission.request-ingest",
                inputs={"request": request.model_dump(mode="json")},
                output={"conversation_id": request.conversation_id},
            )
        except HarnessGraphError as error:
            raise MissionPreparationBlocked(error.code) from error
        evidence.append("harness.stage", request_stage.model_dump(mode="json"))
        channel_outputs = self._invoke_multiple_extensions(
            "input.channels",
            "ingest_input",
            evidence=evidence,
            request=request,
        )
        accepted_channels = [
            output
            for output in channel_outputs
            if isinstance(output, dict) and output.get("accepted") is True
        ]
        if len(accepted_channels) != 1:
            raise MissionPreparationBlocked("INPUT_CHANNEL_NOT_ACCEPTED")
        language_features = self._invoke_extension_pipeline(
            "input.locale-pipeline",
            "resolve_locale",
            {
                "message": request.message,
                "locale": request.locale,
            },
            evidence=evidence,
            request=request,
        )
        entity_features = self._invoke_extension_pipeline(
            "input.entity-pipeline",
            "resolve_entity",
            language_features,
            evidence=evidence,
            request=request,
            map_catalog=self.map_catalog,
            map_graph=self.map_graph,
        )
        if not isinstance(entity_features, dict):
            raise MissionPreparationBlocked("INPUT_ENTITY_PIPELINE_INVALID")
        request_features = self._invoke_extension_pipeline(
            "input.request-features",
            "enrich_request",
            {
                **entity_features,
                **_plugin_request_context(request),
                "start_entity": request.start_entity,
                "accepted_input_channel": accepted_channels[0],
            },
            evidence=evidence,
            request=request,
        )
        if not isinstance(request_features, dict):
            raise MissionPreparationBlocked("PLUGIN_REQUEST_FEATURES_INVALID")
        evidence.append("mission.request-features", request_features)
        multimodal = self._invoke_extension_pipeline(
            "models.multimodal-preprocessors",
            "preprocess_multimodal",
            {"media": []},
            evidence=evidence,
            attachments=request.attachments,
        )
        if not isinstance(multimodal, dict) or not isinstance(multimodal.get("media"), list):
            raise MissionPreparationBlocked("MODEL_MULTIMODAL_PREPROCESSOR_INVALID")
        if len(multimodal["media"]) > 4 or any(
            not isinstance(item, dict) for item in multimodal["media"]
        ):
            raise MissionPreparationBlocked("MODEL_MULTIMODAL_PREPROCESSOR_INVALID")
        self._model_media = multimodal["media"]
        evidence.append(
            "mission.multimodal-input",
            {
                "count": len(self._model_media),
                "sha256": sha256_json(
                    [
                        {key: value for key, value in item.items() if key != "path"}
                        for item in self._model_media
                    ]
                ),
            },
        )
        context_store_policy = self._invoke_single_extension(
            "context.store",
            "resolve_context_store",
            evidence=evidence,
            required=True,
            conversation_id=request.conversation_id,
        )
        if (
            not isinstance(context_store_policy, dict)
            or context_store_policy.get("backend") != "sqlite-wal"
        ):
            raise MissionPreparationBlocked("CONTEXT_STORE_UNAVAILABLE")
        retrieval_policy = self._invoke_single_extension(
            "context.retrieval-policy",
            "retrieve_context",
            evidence=evidence,
            required=True,
            conversation_id=request.conversation_id,
        )
        if not isinstance(retrieval_policy, dict):
            raise MissionPreparationBlocked("CONTEXT_RETRIEVAL_POLICY_INVALID")
        # Plugin policy is a contract, not user text: coercing True, fractions,
        # strings or clamping negatives would silently change retrieval scope.
        maximum_recent_events = _retrieval_event_limit(retrieval_policy)
        context_window = self.context_store.window(
            request.conversation_id,
            max_recent_events=maximum_recent_events,
        )
        if not self.config.persisted_task_context:
            context_window = ConversationWindow(
                conversation_id=request.conversation_id,
                summary=None,
                recent_events=context_window.recent_events[-1:],
                previous_response_ids={},
            )
        compact_context = self._invoke_single_extension(
            "context.compaction-strategy",
            "compact_context",
            evidence=evidence,
            window=context_window,
        )
        if compact_context is None:
            compact_context = _compact_context(context_window)
        compact_context = self._invoke_extension_pipeline(
            "context.enrichment",
            "enrich_context",
            compact_context,
            evidence=evidence,
            request=request,
            map_catalog=self.map_catalog,
            map_graph=self.map_graph,
        )
        if not isinstance(compact_context, dict):
            raise MissionPreparationBlocked("PLUGIN_CONTEXT_PIPELINE_INVALID")
        try:
            compact_context = _assemble_model_context(request, compact_context)
        except ValueError as error:
            # Retrieval/ranking may be plugin-assisted, but the fixed Core
            # wrapper owns identity, schema, budget and authority filtering.
            raise MissionPreparationBlocked(str(error)) from error
        try:
            context_stage = stage_runtime.complete(
                "mission.context-prepare",
                inputs={"request_context": request_features},
                output=compact_context,
            )
        except HarnessGraphError as error:
            raise MissionPreparationBlocked(error.code) from error
        evidence.append("harness.stage", context_stage.model_dump(mode="json"))
        general_map_context = build_map_reasoning_context(
            self.map_graph,
            self.map_catalog,
            self.semantic_path,
        )
        navigation_readiness = assess_navigation_readiness(
            self.map_graph,
            self.map_catalog,
            self.semantic_path,
            self.vehicle,
        )
        _write_artifact(output_dir / "00-model-map-context.json", general_map_context)
        _write_artifact(output_dir / "00-navigation-readiness.json", navigation_readiness)
        evidence.append("mission.model-map-context", general_map_context)
        evidence.append(
            "mission.navigation-readiness",
            navigation_readiness.model_dump(mode="json"),
        )
        model_calls: list[ModelCallRecord] = []
        tool_receipts: list[ToolReceipt] = []
        intent_action_catalog = _intent_action_catalog_view(domain_actions)
        intent_map_catalog = _intent_map_catalog_view(self.map_catalog)
        intent_workflow_scope = _intent_workflow_scope()
        decomposition_action_catalog = _decomposition_action_catalog_view(domain_actions)
        runtime_action_availability = {
            "catalog_id": runtime_action_adapters.catalog_id,
            "runtime_executors": sorted(
                executor
                for adapter in runtime_action_adapters.adapters
                for executor in adapter.runtime_executors
            ),
        }

        intent: IntentArtifact | None = None
        intent_critique: IntentCritique | None = None
        prior_intent: dict[str, object] | None = None
        prior_critique: dict[str, object] | None = None
        for intent_round in range(1, self.config.max_intent_rounds + 1):
            parsed = self._call(
                port=self.primary,
                role="intent_parser",
                output_type=IntentArtifact,
                instructions=INTENT_PARSER,
                input_artifact={
                    "round": intent_round,
                    "mission_request": model_request_context,
                    "explicit_constraint_hints": explicit_constraint_hints,
                    "workflow_scope": intent_workflow_scope,
                    "harness_profile": harness_profile,
                    "request_features": request_features,
                    "conversation_window": compact_context,
                    "map_catalog": intent_map_catalog,
                    "navigation_readiness": navigation_readiness.model_dump(mode="json"),
                    "domain_action_catalog": intent_action_catalog,
                    "previous_candidate": prior_intent,
                    "critic_feedback": prior_critique,
                },
                conversation_id=request.conversation_id,
                evidence=evidence,
            )
            model_calls.extend(_model_records(parsed))
            intent = self._invoke_extension_pipeline(
                "input.intent-normalizers",
                "normalize_intent",
                parsed.artifact,
                evidence=evidence,
                request=request,
                map_catalog=self.map_catalog,
            )
            intent = _artifact_snapshot(intent, IntentArtifact)
            model_constraints = list(intent.constraints)
            intent = _bind_explicit_intent_constraints(intent, explicit_constraint_hints)
            if intent.constraints != model_constraints:
                evidence.append(
                    "intent.explicit-constraints-core-bound",
                    {
                        "model_constraints": model_constraints,
                        "bound_constraints": intent.constraints,
                        "binding_authority": "deterministic-core",
                    },
                )
            if intent.payload_action not in action_ids(domain_actions):
                prior_intent = intent.model_dump(mode="json")
                prior_critique = {
                    "schema_version": "dronedream.intent-critique.v1",
                    "accepted": False,
                    "issue_codes": ["PAYLOAD_ACTION_NOT_REGISTERED"],
                    "repair_instructions": ["Select payload_action from domain_action_catalog."],
                }
                evidence.append("intent.validation-rejected", prior_critique)
                continue
            intent_reviews: list[IntentCritique] = []
            topology_review_count = len(
                [
                    item
                    for item in harness_topology.nodes
                    if item.node_id.startswith("mission.intent-review-")
                ]
            )
            intent_review_count = max(self.config.intent_reviews_per_round, topology_review_count)
            for review_index in range(1, intent_review_count + 1):
                reviewed = self._call(
                    port=self.critic,
                    role="intent_critic",
                    output_type=IntentCritique,
                    instructions=INTENT_CRITIC,
                    input_artifact={
                        "review_index": review_index,
                        "review_count": intent_review_count,
                        "independent_review": intent_review_count > 1,
                        "mission_request": model_request_context,
                        "workflow_scope": intent_workflow_scope,
                        "harness_profile": harness_profile,
                        "request_features": request_features,
                        "candidate_intent": intent.model_dump(mode="json"),
                        "explicit_constraint_hints": explicit_constraint_hints,
                        "map_catalog": intent_map_catalog,
                        "navigation_readiness": navigation_readiness.model_dump(mode="json"),
                        "domain_action_catalog": intent_action_catalog,
                    },
                    conversation_id=request.conversation_id,
                    evidence=evidence,
                )
                model_calls.extend(_model_records(reviewed))
                intent_reviews.append(reviewed.artifact)
            intent_critique = IntentCritique(
                accepted=all(review.accepted for review in intent_reviews),
                issue_codes=list(
                    dict.fromkeys(code for review in intent_reviews for code in review.issue_codes)
                )[:16],
                repair_instructions=list(
                    dict.fromkeys(
                        instruction
                        for review in intent_reviews
                        for instruction in review.repair_instructions
                    )
                )[:16],
            )
            missing_explicit_constraints = _missing_explicit_constraints(
                intent, explicit_constraint_hints
            )
            if (
                intent_critique.accepted
                and not intent.missing_critical_fields
                and not missing_explicit_constraints
            ):
                break
            prior_intent = intent.model_dump(mode="json")
            prior_critique = _intent_repair_feedback(
                intent,
                intent_critique,
                missing_explicit_constraints,
            )
            if missing_explicit_constraints or intent.missing_critical_fields:
                evidence.append("intent.validation-rejected", prior_critique)
        else:
            if intent is not None and intent.missing_critical_fields:
                evidence.append(
                    "intent.clarification-required",
                    {
                        "fields": intent.missing_critical_fields,
                        "actuator_authority": False,
                    },
                )
                raise MissionClarificationRequired(intent.missing_critical_fields)
            raise MissionPreparationBlocked("INTENT_REVIEW_EXHAUSTED")
        assert intent is not None and intent_critique is not None
        try:
            intent_stage = stage_runtime.complete(
                "mission.intent-parse",
                inputs={"context": compact_context, "request_features": request_features},
                output=intent.model_dump(mode="json"),
            )
            review_stage_receipts = []
            for review_index, review in enumerate(intent_reviews, start=1):
                stage_id = f"mission.intent-review-{review_index}"
                if stage_runtime.contains(stage_id):
                    review_stage_receipts.append(
                        stage_runtime.complete(
                            stage_id,
                            inputs={"intent": intent.model_dump(mode="json")},
                            output=review.model_dump(mode="json"),
                        )
                    )
            consensus_stage = stage_runtime.complete(
                "mission.intent-consensus",
                inputs={"reviews": [item.model_dump(mode="json") for item in intent_reviews]},
                output=intent_critique.model_dump(mode="json"),
            )
        except HarnessGraphError as error:
            raise MissionPreparationBlocked(error.code) from error
        for stage_receipt in [intent_stage, *review_stage_receipts, consensus_stage]:
            evidence.append("harness.stage", stage_receipt.model_dump(mode="json"))
        _write_artifact(output_dir / "01-intent.json", intent)
        _write_artifact(output_dir / "02-intent-critique.json", intent_critique)
        try:
            enforce_environment_readiness(intent.environment_mode, navigation_readiness)
        except ValueError as error:
            evidence.append(
                "mission.navigation-readiness-rejected",
                {
                    "environment_mode": intent.environment_mode,
                    "issue_code": str(error),
                    "readiness": navigation_readiness.model_dump(mode="json"),
                },
            )
            raise MissionPreparationBlocked(str(error)) from error

        contract = MissionContract(
            contract_id=f"mission-{uuid4().hex[:24]}",
            conversation_id=request.conversation_id,
            goal=intent.goal,
            start_node=_resolve_entity(intent.start_entity, self.map_catalog, self.map_graph),
            target_node=_resolve_entity(intent.target_entity, self.map_catalog, self.map_graph),
            return_node=_resolve_entity(intent.return_entity, self.map_catalog, self.map_graph),
            payload_action=intent.payload_action,
            domain_ids=domain_actions.domain_ids,
            authorized_actions=sorted(action_ids(domain_actions)),
            action_catalog_sha256=sha256_json(domain_actions),
            map_asset_id=self.map_graph.asset_id,
            map_sha256=sha256_json(self.map_graph),
            map_semantic_sha256=semantic_sha256,
            vehicle_asset_id=self.vehicle_asset_id,
            vehicle_sha256=vehicle_sha256,
            constraints=list(dict.fromkeys([*intent.constraints, *REQUIRED_MISSION_CONSTRAINTS])),
            immutable_safety_rules=IMMUTABLE_SAFETY_RULES,
        )
        _write_artifact(output_dir / "03-mission-contract.json", contract)
        evidence.append("mission.contract", contract.model_dump(mode="json"))
        focused_map_context = build_map_reasoning_context(
            self.map_graph,
            self.map_catalog,
            self.semantic_path,
            focus_nodes=[contract.start_node, contract.target_node, contract.return_node],
        )
        semantic_planning_map_context = _semantic_planning_map_view(
            focused_map_context,
            contract,
        )
        _write_artifact(output_dir / "03d-focused-map-context.json", focused_map_context)
        evidence.append("mission.focused-map-context", focused_map_context)
        evidence.append(
            "mission.semantic-planning-map-view",
            semantic_planning_map_context,
        )
        try:
            contract_stage = stage_runtime.complete(
                "mission.contract-freeze",
                inputs={"accepted_intent": intent.model_dump(mode="json")},
                output=contract.model_dump(mode="json"),
            )
        except HarnessGraphError as error:
            raise MissionPreparationBlocked(error.code) from error
        evidence.append("harness.stage", contract_stage.model_dump(mode="json"))

        registry = self.tool_registry
        plugin_advice: list[dict[str, object]] = []
        optional_catalog = (
            [
                item
                for item in registry.catalog()
                if item.get("slot_id") not in CORE_PLANNING_SLOTS
                and item["authority"] in {"read", "plan", "simulate"}
            ][:32]
            if stage_runtime.contains("mission.tool-advice")
            else []
        )
        if optional_catalog:
            recommended_tool_ids = _recommended_plugin_tools(optional_catalog, contract)
            router_policy = self._invoke_single_extension(
                "tools.router-policy",
                "recommend_tools",
                evidence=evidence,
                contract=contract,
                catalog=optional_catalog,
                recommended_tool_ids=recommended_tool_ids,
            )
            if isinstance(router_policy, dict):
                candidate_ids = router_policy.get("recommended_tool_ids")
                if isinstance(candidate_ids, list):
                    available_ids = {str(item["tool_id"]) for item in optional_catalog}
                    recommended_tool_ids = sorted(
                        {str(tool_id) for tool_id in candidate_ids if str(tool_id) in available_ids}
                    )
            router_feedback: dict[str, object] | None = None
            routed: StructuredCallResult[PluginInvocationPlan]
            missing_recommended: list[str] = []
            for router_round in range(1, self.config.plugin_router_rounds + 1):
                routed = self._call(
                    port=self.primary,
                    role="plugin_router",
                    output_type=PluginInvocationPlan,
                    instructions=PLUGIN_ROUTER,
                    input_artifact={
                        "round": router_round,
                        "mission_contract": contract.model_dump(mode="json"),
                        "mission_request_context": _plugin_request_context(request),
                        "request_features": request_features,
                        "optional_tool_catalog": optional_catalog,
                        "recommended_tool_ids": recommended_tool_ids,
                        "maximum_calls": min(
                            self.config.maximum_plugin_calls,
                            self.config.maximum_optional_tool_calls,
                            int(
                                harness_policies["budget"].get(
                                    "maximum_tool_calls",
                                    self.config.maximum_optional_tool_calls,
                                )
                            ),
                        ),
                        "advisory_only": True,
                        "harness_profile": harness_profile,
                        "router_feedback": router_feedback,
                    },
                    conversation_id=request.conversation_id,
                    evidence=evidence,
                )
                model_calls.extend(_model_records(routed))
                selected_tools = {call.tool_id for call in routed.artifact.calls}
                missing_recommended = sorted(set(recommended_tool_ids) - selected_tools)
                if not missing_recommended:
                    break
                router_feedback = {
                    "issue_code": "RECOMMENDED_PLUGIN_NOT_SELECTED",
                    "missing_tool_ids": missing_recommended,
                    "repair": "Select each matching recommended tool with schema-valid JSON.",
                }
            if missing_recommended:
                evidence.append(
                    "plugin.routing-recommendation-unmet",
                    {"tool_ids": missing_recommended},
                )
            available_optional_tools = {str(item["tool_id"]): item for item in optional_catalog}
            for invocation in routed.artifact.calls:
                if invocation.tool_id not in available_optional_tools:
                    evidence.append(
                        "plugin.invocation-rejected",
                        {
                            "tool_id": invocation.tool_id,
                            "issue_code": "PLUGIN_TOOL_NOT_IN_ROUTING_CATALOG",
                        },
                    )
                    continue
                try:
                    arguments = invocation.parsed_arguments()
                except ValueError:
                    evidence.append(
                        "plugin.invocation-rejected",
                        {
                            "tool_id": invocation.tool_id,
                            "issue_code": "PLUGIN_ARGUMENTS_JSON_INVALID",
                        },
                    )
                    continue
                try:
                    output, receipt = registry.call(invocation.tool_id, arguments)
                except ToolExecutionError as error:
                    receipt = error.receipt
                    self._record_tool(
                        receipt,
                        conversation_id=request.conversation_id,
                        context_store=self.context_store,
                        evidence=evidence,
                    )
                    tool_receipts.append(receipt)
                    plugin_advice.append(
                        {
                            "tool_id": invocation.tool_id,
                            "purpose": invocation.purpose,
                            "accepted": False,
                            "issue_codes": receipt.issue_codes,
                        }
                    )
                    continue
                self._record_tool(
                    receipt,
                    conversation_id=request.conversation_id,
                    context_store=self.context_store,
                    evidence=evidence,
                )
                tool_receipts.append(receipt)
                plugin_advice.append(
                    {
                        "tool_id": invocation.tool_id,
                        "purpose": invocation.purpose,
                        "accepted": True,
                        "output": (
                            output.model_dump(mode="json")
                            if hasattr(output, "model_dump")
                            else output
                        ),
                        "output_sha256": receipt.output_sha256,
                    }
                )
            _write_artifact(output_dir / "03a-plugin-advice.json", plugin_advice)
            evidence.append("plugin.advice", {"results": plugin_advice})
        plugin_advice = self._invoke_extension_pipeline(
            "tools.result-fusion",
            "fuse_results",
            plugin_advice,
            evidence=evidence,
            contract=contract,
            domain_actions=domain_actions,
        )
        if not isinstance(plugin_advice, list) or any(
            not isinstance(item, dict) for item in plugin_advice
        ):
            raise MissionPreparationBlocked("PLUGIN_RESULT_FUSION_INVALID")
        _write_artifact(output_dir / "03b-plugin-advice-fused.json", plugin_advice)
        if stage_runtime.contains("mission.tool-advice"):
            try:
                advice_stage = stage_runtime.complete(
                    "mission.tool-advice",
                    inputs={"contract": contract.model_dump(mode="json")},
                    output=plugin_advice,
                )
            except HarnessGraphError as error:
                raise MissionPreparationBlocked(error.code) from error
            evidence.append("harness.stage", advice_stage.model_dump(mode="json"))
        planning_contributions = [
            _artifact_snapshot(value, PlannerContribution)
            for value in self._invoke_multiple_extensions(
                "planning.specialists",
                "contribute_planning",
                evidence=evidence,
                contract=contract,
                map_graph=self.map_graph,
                vehicle=self.vehicle,
                simulator_descriptor=simulator_descriptor,
                simulation_components=simulation_capabilities,
            )
        ]
        if not planning_contributions or any(
            not all(contribution.deterministic_gates.values())
            for contribution in planning_contributions
        ):
            raise MissionPreparationBlocked("PLANNING_SPECIALIST_CONTRIBUTION_REJECTED")
        _write_artifact(
            output_dir / "03c-planning-contributions.json",
            [value.model_dump(mode="json") for value in planning_contributions],
        )
        evidence.append(
            "planning.specialists",
            {"contributions": [value.model_dump(mode="json") for value in planning_contributions]},
        )
        plan_feedback: dict[str, object] | None = None
        accepted_values: (
            tuple[
                TaskGraph,
                SemanticPlan,
                FlightPlan,
                PlanCritique,
                GraphRoute,
                RouteClearanceReport,
                Px4Track,
                RuntimeCheckpointContract,
                RuntimeActionExecutionContract,
                MissionVerificationPlan,
            ]
            | None
        ) = None
        planning_attempts = 0
        for planning_attempts in range(1, self.config.max_planning_rounds + 1):
            decomposed = self._call(
                port=self.primary,
                role="task_decomposer",
                output_type=TaskGraphArtifact,
                instructions=TASK_DECOMPOSER,
                input_artifact={
                    "round": planning_attempts,
                    "mission_contract": contract.model_dump(mode="json"),
                    "available_node_ids": [node.node_id for node in self.map_graph.nodes],
                    "domain_action_catalog": decomposition_action_catalog,
                    "runtime_action_adapter_catalog": runtime_action_availability,
                    "harness_profile": harness_profile,
                    "planning_hard_constraints": [
                        constraint
                        for contribution in planning_contributions
                        if contribution.applicable
                        for constraint in contribution.hard_constraints
                    ],
                    "previous_plan_critique": plan_feedback,
                },
                conversation_id=request.conversation_id,
                evidence=evidence,
            )
            model_calls.extend(_model_records(decomposed))
            task_graph = self._invoke_extension_pipeline(
                "planning.task-transformers",
                "transform_task_graph",
                decomposed.artifact.graph,
                evidence=evidence,
                contract=contract,
                map_graph=self.map_graph,
                planning_round=planning_attempts,
            )
            task_graph = _artifact_snapshot(task_graph, TaskGraph)
            unbound_task_graph_sha256 = sha256_json(task_graph)
            task_graph = _bind_task_graph_action_contracts(task_graph, domain_actions)
            evidence.append(
                "planning.task-action-contracts-bound",
                {
                    "model_task_graph_sha256": unbound_task_graph_sha256,
                    "bound_task_graph_sha256": sha256_json(task_graph),
                    "domain_action_catalog_sha256": sha256_json(domain_actions),
                    "bound_task_ids": [task.task_id for task in task_graph.nodes],
                },
            )
            try:
                _validate_task_graph(task_graph, contract, self.map_graph, domain_actions)
            except MissionPreparationBlocked as exc:
                plan_feedback = {
                    "accepted": False,
                    "issue_codes": [str(exc)],
                    "repair_instructions": [
                        "Repair the task graph using only the grounded contract nodes."
                    ],
                }
                evidence.append("planning.validation-rejected", plan_feedback)
                continue

            planned = self._call(
                port=self.primary,
                role="global_planner",
                output_type=SemanticPlan,
                instructions=GLOBAL_PLANNER,
                input_artifact={
                    "round": planning_attempts,
                    "mission_contract": contract.model_dump(mode="json"),
                    "task_graph": task_graph.model_dump(mode="json"),
                    "available_node_ids": [node.node_id for node in self.map_graph.nodes],
                    "structured_map_context": semantic_planning_map_context,
                    "navigation_readiness": navigation_readiness.model_dump(mode="json"),
                    "hard_output_rules": {
                        "first_target_must_not_equal": contract.start_node,
                        "must_include_target": contract.target_node,
                        "final_target_must_equal": contract.return_node,
                        "only_allowed_targets": list(
                            dict.fromkeys([contract.target_node, contract.return_node])
                        ),
                    },
                    "previous_plan_critique": plan_feedback,
                    "harness_profile": harness_profile,
                    "planning_hard_constraints": [
                        constraint
                        for contribution in planning_contributions
                        if contribution.applicable
                        for constraint in contribution.hard_constraints
                    ],
                },
                conversation_id=request.conversation_id,
                evidence=evidence,
            )
            model_calls.extend(_model_records(planned))
            semantic_plan = self._invoke_extension_pipeline(
                "planning.semantic-optimizers",
                "optimize_semantic_plan",
                planned.artifact,
                evidence=evidence,
                contract=contract,
                task_graph=task_graph,
                map_graph=self.map_graph,
            )
            semantic_plan = _artifact_snapshot(semantic_plan, SemanticPlan)
            try:
                _validate_semantic_plan(semantic_plan, contract, self.map_graph)
            except MissionPreparationBlocked as exc:
                plan_feedback = {
                    "accepted": False,
                    "issue_codes": [str(exc)],
                    "repair_instructions": [
                        "Repair the ordered targets; geometry remains tool-owned."
                    ],
                }
                evidence.append("planning.validation-rejected", plan_feedback)
                continue

            attempt_receipts: list[ToolReceipt] = []
            primary_strategy = registry.tool_for_slot("planning.route-strategy")
            strategy_tools = list(
                dict.fromkeys(
                    [
                        primary_strategy,
                        *registry.tool_ids_for_slot("planning.route-candidates"),
                    ]
                )
            )
            route_alternatives: list[RouteAlternativeCandidate] = []
            alternative_segments: dict[str, list[GraphRoute]] = {}
            alternative_receipts: dict[str, list[ToolReceipt]] = {}
            route_fingerprint_indexes: dict[str, int] = {}
            for strategy_tool_id in strategy_tools:
                candidate_routes: list[GraphRoute] = []
                candidate_receipts: list[ToolReceipt] = []
                current = contract.start_node
                strategy_failed = False
                for target in semantic_plan.ordered_targets:
                    try:
                        route_value, receipt = registry.call(
                            strategy_tool_id,
                            RouteQuery(start_node=current, goal_node=target),
                        )
                    except ToolExecutionError as error:
                        attempt_receipts.append(error.receipt)
                        self._record_tool(
                            error.receipt,
                            conversation_id=request.conversation_id,
                            context_store=self.context_store,
                            evidence=evidence,
                        )
                        evidence.append(
                            "planning.route-candidate-failed",
                            {
                                "strategy_tool_id": strategy_tool_id,
                                "target": target,
                                "issue_codes": error.receipt.issue_codes,
                            },
                        )
                        strategy_failed = True
                        break
                    route = _artifact_snapshot(route_value, GraphRoute)
                    if route.start_node != current or route.goal_node != target:
                        raise MissionPreparationBlocked("ROUTE_TOOL_QUERY_MISMATCH")
                    candidate_routes.append(route)
                    attempt_receipts.append(receipt)
                    candidate_receipts.append(receipt)
                    self._record_tool(
                        receipt,
                        conversation_id=request.conversation_id,
                        context_store=self.context_store,
                        evidence=evidence,
                    )
                    current = target
                if strategy_failed:
                    continue
                candidate_route = _combine_routes(candidate_routes)
                route_fingerprint = sha256_json(candidate_route)
                duplicate_index = route_fingerprint_indexes.get(route_fingerprint)
                if duplicate_index is not None:
                    existing = route_alternatives[duplicate_index]
                    route_alternatives[duplicate_index] = existing.model_copy(
                        update={
                            "equivalent_strategy_tool_ids": list(
                                dict.fromkeys(
                                    [
                                        *existing.equivalent_strategy_tool_ids,
                                        strategy_tool_id,
                                    ]
                                )
                            )
                        }
                    )
                    evidence.append(
                        "planning.route-candidate-deduplicated",
                        {
                            "route_sha256": route_fingerprint,
                            "retained_strategy_tool_id": existing.strategy_tool_id,
                            "equivalent_strategy_tool_id": strategy_tool_id,
                        },
                    )
                    continue
                try:
                    clearance_value, clearance_receipt = registry.call_slot(
                        "safety.route-clearance", candidate_route
                    )
                except ToolExecutionError as error:
                    attempt_receipts.append(error.receipt)
                    self._record_tool(
                        error.receipt,
                        conversation_id=request.conversation_id,
                        context_store=self.context_store,
                        evidence=evidence,
                    )
                    evidence.append(
                        "planning.route-candidate-clearance-failed",
                        {
                            "strategy_tool_id": strategy_tool_id,
                            "issue_codes": error.receipt.issue_codes,
                        },
                    )
                    continue
                candidate_clearance = _artifact_snapshot(clearance_value, RouteClearanceReport)
                attempt_receipts.append(clearance_receipt)
                candidate_receipts.append(clearance_receipt)
                self._record_tool(
                    clearance_receipt,
                    conversation_id=request.conversation_id,
                    context_store=self.context_store,
                    evidence=evidence,
                )
                required_operational_clearance_m = _required_operational_clearance_m(self.vehicle)
                clearance_assessment = _route_operational_clearance_assessment(
                    candidate_route,
                    candidate_clearance,
                    preferred_transit_clearance_m=required_operational_clearance_m,
                )
                gates = {
                    "continuous_clearance": candidate_clearance.accepted,
                    "clearance_evidence_binding": (
                        candidate_clearance.route_sha256 == sha256_json(candidate_route)
                        and candidate_clearance.semantic_sha256 == semantic_sha256
                        and candidate_clearance.collision_count == 0
                        and not candidate_clearance.collisions
                    ),
                    "operational_clearance": bool(clearance_assessment["accepted"]),
                    "contract_start": candidate_route.start_node == contract.start_node,
                    "contract_return": candidate_route.goal_node == contract.return_node,
                    "all_targets_present": all(
                        target in candidate_route.node_ids
                        for target in semantic_plan.ordered_targets
                    ),
                }
                issue_codes = [
                    f"ROUTE_ALTERNATIVE_{name.upper()}_REJECTED"
                    for name, accepted in gates.items()
                    if not accepted
                ]
                alternative_id = (
                    "route-alternative-"
                    + sha256_json(
                        {
                            "route": candidate_route,
                            "clearance": candidate_clearance,
                        }
                    )[:20]
                )
                candidate = RouteAlternativeCandidate(
                    alternative_id=alternative_id,
                    strategy_tool_id=strategy_tool_id,
                    equivalent_strategy_tool_ids=[],
                    route=candidate_route,
                    clearance=candidate_clearance,
                    objectives=_route_objectives(candidate_route, candidate_clearance),
                    hard_gates=gates,
                    feasible=all(gates.values()),
                    issue_codes=issue_codes,
                )
                route_alternatives.append(candidate)
                evidence.append(
                    "planning.operational-clearance",
                    {
                        "alternative_id": alternative_id,
                        "required_m": required_operational_clearance_m,
                        "measured_m": candidate_clearance.minimum_clearance_m,
                        "accepted": gates["operational_clearance"],
                        "clearance_budget": clearance_assessment,
                        "reserved_for": [
                            "localization-error",
                            "closed-loop-tracking-error",
                            "local-avoidance-correction",
                        ],
                    },
                )
                route_fingerprint_indexes[route_fingerprint] = len(route_alternatives) - 1
                alternative_segments[alternative_id] = candidate_routes
                alternative_receipts[alternative_id] = candidate_receipts
            if not route_alternatives or not any(
                candidate.feasible for candidate in route_alternatives
            ):
                raise MissionPreparationBlocked("ROUTE_ALTERNATIVE_NO_FEASIBLE_CANDIDATE")
            alternative_set = RouteAlternativeSet(
                contract_id=contract.contract_id,
                candidates=route_alternatives,
                objective_weights=_route_objective_weights(semantic_plan.route_policy),
            )
            decision_value, decision_receipt = self._call_required_slot(
                "planning.alternative-ranker",
                alternative_set,
                conversation_id=request.conversation_id,
                evidence=evidence,
            )
            alternative_decision = _artifact_snapshot(decision_value, RouteAlternativeDecision)
            attempt_receipts.append(decision_receipt)
            feasible_ids = {
                candidate.alternative_id for candidate in route_alternatives if candidate.feasible
            }
            if (
                alternative_decision.selected_alternative_id not in feasible_ids
                or alternative_decision.ranked_alternative_ids[0]
                != alternative_decision.selected_alternative_id
                or set(alternative_decision.ranked_alternative_ids) != feasible_ids
                or set(alternative_decision.normalized_scores) != feasible_ids
                or len(alternative_decision.ranked_alternative_ids) != len(feasible_ids)
            ):
                raise MissionPreparationBlocked("ROUTE_ALTERNATIVE_DECISION_INVALID")
            selected_alternative = next(
                candidate
                for candidate in route_alternatives
                if candidate.alternative_id == alternative_decision.selected_alternative_id
            )
            routes = alternative_segments[selected_alternative.alternative_id]
            execution_route = selected_alternative.route
            clearance = selected_alternative.clearance
            alternative_artifact = {
                "candidate_set": alternative_set.model_dump(mode="json"),
                "decision": alternative_decision.model_dump(mode="json"),
            }
            _write_artifact(output_dir / "05a-route-alternatives.json", alternative_artifact)
            evidence.append("planning.route-alternatives", alternative_artifact)
            flight_plan = _flight_plan(
                contract,
                task_graph,
                semantic_plan,
                routes,
                self.map_graph,
                clearance.minimum_clearance_m,
                domain_actions,
            )
            track_value, track_receipt = self._call_required_slot(
                "flight-control.track-export",
                Px4TrackRequest(
                    route=execution_route,
                    waypoint_hold_seconds=self.config.waypoint_hold_seconds,
                ),
                conversation_id=request.conversation_id,
                evidence=evidence,
            )
            px4_track = _artifact_snapshot(track_value, Px4Track)
            _validate_plugin_track_tightening(px4_track, execution_route, self.vehicle)
            baseline_track = _artifact_snapshot(px4_track, Px4Track)
            px4_track = self._invoke_extension_pipeline(
                "planning.track-optimizers",
                "optimize_track",
                px4_track,
                evidence=evidence,
                contract=contract,
                task_graph=task_graph,
                semantic_plan=semantic_plan,
                route=execution_route,
                clearance=clearance,
            )
            px4_track = _artifact_snapshot(px4_track, Px4Track)
            _validate_plugin_track_tightening(
                px4_track, execution_route, self.vehicle, baseline=baseline_track
            )
            attempt_receipts.append(track_receipt)
            selected_plan_receipts = [
                *alternative_receipts[selected_alternative.alternative_id],
                decision_receipt,
                track_receipt,
            ]
            selected_plan_receipt_ids = {receipt.call_id for receipt in selected_plan_receipts}
            selected_plan_receipts_accepted = all(
                receipt.outcome == "accepted" for receipt in selected_plan_receipts
            )
            failed_exploratory_receipt_count = sum(
                receipt.outcome != "accepted" and receipt.call_id not in selected_plan_receipt_ids
                for receipt in attempt_receipts
            )
            failed_exploratory_tool_ids = _collect_failed_exploratory_tool_ids(
                attempt_receipts,
                selected_plan_receipt_ids,
            )
            tool_receipts.extend(attempt_receipts)

            plan_scores = self._invoke_multiple_extensions(
                "planning.plan-scorers",
                "score_plan",
                evidence=evidence,
                contract=contract,
                task_graph=task_graph,
                semantic_plan=semantic_plan,
                flight_plan=flight_plan,
                vehicle=self.vehicle,
                route=execution_route,
                clearance=clearance,
                px4_track=px4_track,
            )
            validation_results = self._invoke_multiple_extensions(
                "validation.plan-gates",
                "validate_plan",
                evidence=evidence,
                contract=contract,
                task_graph=task_graph,
                semantic_plan=semantic_plan,
                flight_plan=flight_plan,
                vehicle=self.vehicle,
                route=execution_route,
                clearance=clearance,
                px4_track=px4_track,
            )
            specialist_validations = [
                _artifact_snapshot(value, PlannerValidation)
                for value in self._invoke_multiple_extensions(
                    "planning.specialists",
                    "validate_planning",
                    evidence=evidence,
                    contract=contract,
                    map_graph=self.map_graph,
                    vehicle=self.vehicle,
                    task_graph=task_graph,
                    semantic_plan=semantic_plan,
                    flight_plan=flight_plan,
                    route=execution_route,
                    clearance=clearance,
                    px4_track=px4_track,
                )
            ]
            validation_results.extend(
                value.model_dump(mode="json") for value in specialist_validations
            )
            rejected_validators = _rejected_plan_validators(validation_results)
            if rejected_validators:
                plan_feedback = {
                    "accepted": False,
                    "issue_codes": [
                        str(code)
                        for value in rejected_validators
                        for code in value.get("issue_codes", ["PLUGIN_PLAN_GATE_REJECTED"])
                    ],
                    "repair_instructions": [
                        str(item)
                        for value in rejected_validators
                        for item in value.get("repair_instructions", [])
                    ],
                }
                evidence.append("planning.plugin-validation-rejected", plan_feedback)
                continue

            runtime_checkpoints_value = self._invoke_single_extension(
                "runtime.checkpoint-policy",
                "build_checkpoints",
                evidence=evidence,
                contract=contract,
                domain_actions=domain_actions,
                flight_plan=flight_plan,
            )
            if runtime_checkpoints_value is None:
                runtime_checkpoints_value = _runtime_checkpoints(contract, flight_plan)
            runtime_checkpoints = _artifact_snapshot(
                runtime_checkpoints_value, RuntimeCheckpointContract
            )
            try:
                runtime_actions = build_runtime_action_execution_contract(
                    mission_contract=contract,
                    task_graph=task_graph,
                    domain_actions=domain_actions,
                    adapter_catalog=runtime_action_adapters,
                    checkpoints=runtime_checkpoints,
                    vehicle=self.vehicle,
                    vehicle_sdf=self.vehicle_sdf,
                )
                verification_plan = build_mission_verification_plan(
                    contract=contract,
                    domain_actions=domain_actions,
                    task_graph=task_graph,
                    semantic_plan=semantic_plan,
                    flight_plan=flight_plan,
                    execution_route=execution_route,
                    route_clearance=clearance,
                    px4_track=px4_track,
                    runtime_checkpoints=runtime_checkpoints,
                    runtime_actions=runtime_actions,
                )
            except (RuntimeActionContractError, MissionVerificationPlanError) as error:
                plan_feedback = {
                    "accepted": False,
                    "issue_codes": [str(error).split(":", 1)[0]],
                    "repair_instructions": [
                        "Repair the task graph and evidence requirements using only the "
                        "registered action, checkpoint, and runtime-adapter contracts."
                    ],
                }
                evidence.append("planning.verification-plan-rejected", plan_feedback)
                continue

            plan_reviews: list[PlanCritique] = []
            plan_review_input = {
                "mission_contract": contract.model_dump(mode="json"),
                "task_graph": task_graph.model_dump(mode="json"),
                "semantic_plan": semantic_plan.model_dump(mode="json"),
                "flight_plan_scope": (
                    "movement segments only; takeoff, pickup, and land are explicit "
                    "TaskGraph actions executed by dedicated runtime adapters"
                ),
                "flight_plan_summary": {
                    "contract_id": flight_plan.contract_id,
                    "semantic_plan_sha256": flight_plan.semantic_plan_sha256,
                    "segments": [
                        {
                            "segment_id": segment.segment_id,
                            "task_id": segment.task_id,
                            "from_node": segment.from_node,
                            "to_node": segment.to_node,
                            "path_point_count": len(segment.path),
                            "path_sha256": sha256_json(segment.path),
                            "speed_limit_mps": segment.speed_limit_mps,
                            "minimum_clearance_m": segment.minimum_clearance_m,
                            "success_evidence": segment.success_evidence,
                        }
                        for segment in flight_plan.segments
                    ],
                },
                "semantic_plan_binding": {
                    "semantic_plan_sha256": sha256_json(semantic_plan),
                    "flight_plan_semantic_plan_sha256": flight_plan.semantic_plan_sha256,
                    "hash_matches": (
                        flight_plan.semantic_plan_sha256 == sha256_json(semantic_plan)
                    ),
                },
                "execution_route_summary": {
                    "start_node": execution_route.start_node,
                    "goal_node": execution_route.goal_node,
                    "point_count": len(execution_route.node_ids),
                    "route_length_m": execution_route.route_length_m,
                    "all_edges_flight_verified": (execution_route.all_edges_flight_verified),
                    "route_sha256": sha256_json(execution_route),
                    "geometry_authority": (
                        "qualified-metric-collision-semantics"
                        if selected_alternative.strategy_tool_id
                        == "planning.candidate-metric-geometry.candidate"
                        else "qualified-map-topology"
                    ),
                },
                "route_clearance_summary": {
                    "accepted": clearance.accepted,
                    "route_sha256": clearance.route_sha256,
                    "semantic_sha256": clearance.semantic_sha256,
                    "sample_count": clearance.sample_count,
                    "primitive_count": clearance.primitive_count,
                    "collision_count": clearance.collision_count,
                    "minimum_clearance_m": clearance.minimum_clearance_m,
                },
                "deterministic_gates": {
                    "semantic_plan_hash_matches_flight_plan": (
                        flight_plan.semantic_plan_sha256 == sha256_json(semantic_plan)
                    ),
                    "clearance_route_hash_matches": (
                        clearance.route_sha256 == sha256_json(execution_route)
                    ),
                    "clearance_semantic_hash_matches_contract": (
                        clearance.semantic_sha256 == contract.map_semantic_sha256
                    ),
                    "route_starts_at_contract_start": (
                        execution_route.start_node == contract.start_node
                    ),
                    "route_ends_at_contract_return": (
                        execution_route.goal_node == contract.return_node
                    ),
                    "all_required_tool_receipts_accepted": (selected_plan_receipts_accepted),
                    "failed_exploratory_tool_receipt_count": (failed_exploratory_receipt_count),
                    "selected_alternative_is_feasible": selected_alternative.feasible,
                    "selected_route_hash_matches_execution": (
                        sha256_json(selected_alternative.route) == sha256_json(execution_route)
                    ),
                    "selected_clearance_hash_matches_execution": (
                        selected_alternative.clearance.route_sha256 == sha256_json(execution_route)
                    ),
                    "qualified_geometry_route_accepted": (
                        selected_alternative.feasible
                        and selected_alternative.clearance.accepted
                        and selected_alternative.clearance.semantic_sha256
                        == contract.map_semantic_sha256
                    ),
                    "route_alternative_candidates_are_unique": (
                        len({sha256_json(item.route) for item in route_alternatives})
                        == len(route_alternatives)
                    ),
                },
                "selected_plan_tool_receipts": [
                    {
                        "tool_id": item.tool_id,
                        "outcome": item.outcome,
                        "input_sha256": item.input_sha256,
                        "output_sha256": item.output_sha256,
                        "required_for_selected_plan": (item.call_id in selected_plan_receipt_ids),
                    }
                    for item in attempt_receipts
                    if item.call_id in selected_plan_receipt_ids
                ],
                "exploratory_tool_receipt_summary": {
                    "attempted_count": len(attempt_receipts),
                    "failed_count": failed_exploratory_receipt_count,
                    "failed_tool_ids": failed_exploratory_tool_ids,
                    "receipt_set_sha256": sha256_json(attempt_receipts),
                },
                "planning_evidence_scope": {
                    "phase": "pre_execution_planning",
                    "present_evidence_must_be_hash_bound": True,
                    "future_runtime_evidence_is_expected_after_confirmation": True,
                    "future_runtime_evidence": [
                        "telemetry-continuity",
                        "payload-identity",
                        "payload-mass-and-attachment",
                        "runtime-checkpoints",
                        "execution-confirmation",
                        "landing-confirmation",
                        "completion-assessment",
                    ],
                    "future_runtime_evidence_declared_by_accepted_tool_receipts": all(
                        item.outcome == "accepted" for item in selected_plan_receipts
                    ),
                },
                "verification_plan": verification_plan.model_dump(mode="json"),
                "plugin_validation_results": validation_results,
                "route_alternative_decision": alternative_decision.model_dump(mode="json"),
                "route_alternative_evidence": {
                    "unique_candidate_count": len(route_alternatives),
                    "selected_route_sha256": sha256_json(selected_alternative.route),
                    "execution_route_sha256": sha256_json(execution_route),
                    "selected_clearance_route_sha256": (
                        selected_alternative.clearance.route_sha256
                    ),
                    "selected_strategy_tool_id": selected_alternative.strategy_tool_id,
                    "equivalent_strategy_tool_ids": (
                        selected_alternative.equivalent_strategy_tool_ids
                    ),
                },
                "structured_map_context": semantic_planning_map_context,
                "navigation_readiness": navigation_readiness.model_dump(mode="json"),
                "harness_profile": harness_profile,
            }
            for review_index in range(1, self.config.plan_reviews_per_round + 1):
                critique = self._call(
                    port=self.critic,
                    role="plan_critic",
                    output_type=PlanCritique,
                    instructions=PLAN_CRITIC,
                    input_artifact={
                        **plan_review_input,
                        "review_index": review_index,
                        "review_count": self.config.plan_reviews_per_round,
                        "independent_review": self.config.plan_reviews_per_round > 1,
                    },
                    conversation_id=request.conversation_id,
                    evidence=evidence,
                )
                model_calls.extend(_model_records(critique))
                normalized_review = _normalize_plan_critique(
                    critique.artifact,
                    selected_alternative,
                    selected_plan_receipts_accepted=selected_plan_receipts_accepted,
                    future_runtime_evidence_declared=(selected_plan_receipts_accepted),
                    failed_exploratory_tool_ids=failed_exploratory_tool_ids,
                )
                if normalized_review != critique.artifact:
                    discarded_codes = [
                        code
                        for code in critique.artifact.issue_codes
                        if code not in normalized_review.issue_codes
                    ]
                    evidence.append(
                        "planning.critic-unsupported-gate-discarded",
                        {
                            "issue_codes": discarded_codes,
                            "selected_strategy_tool_id": (selected_alternative.strategy_tool_id),
                            "continuous_clearance_accepted": (
                                selected_alternative.clearance.accepted
                            ),
                            "selected_candidate_feasible": (selected_alternative.feasible),
                        },
                    )
                plan_reviews.append(normalized_review)
            plan_critique = PlanCritique(
                accepted=all(review.accepted for review in plan_reviews),
                issue_codes=list(
                    dict.fromkeys(code for review in plan_reviews for code in review.issue_codes)
                )[:32],
                repair_instructions=list(
                    dict.fromkeys(
                        instruction
                        for review in plan_reviews
                        for instruction in review.repair_instructions
                    )
                )[:32],
            )
            if plan_critique.accepted:
                accepted_values = (
                    task_graph,
                    semantic_plan,
                    flight_plan,
                    plan_critique,
                    execution_route,
                    clearance,
                    px4_track,
                    runtime_checkpoints,
                    runtime_actions,
                    verification_plan,
                )
                break
            plan_feedback = plan_critique.model_dump(mode="json")
        if accepted_values is None:
            raise MissionPreparationBlocked("PLAN_REVIEW_EXHAUSTED")
        (
            task_graph,
            semantic_plan,
            flight_plan,
            plan_critique,
            execution_route,
            clearance,
            px4_track,
            runtime_checkpoints,
            runtime_actions,
            verification_plan,
        ) = accepted_values
        try:
            preparation_stages = [
                stage_runtime.complete(
                    "mission.task-decompose",
                    inputs={
                        "contract": contract.model_dump(mode="json"),
                        "tool_advice": plugin_advice,
                    },
                    output=task_graph.model_dump(mode="json"),
                ),
                stage_runtime.complete(
                    "mission.semantic-plan",
                    inputs={"task_graph": task_graph.model_dump(mode="json")},
                    output=semantic_plan.model_dump(mode="json"),
                ),
                stage_runtime.complete(
                    "mission.route-resolve",
                    inputs={"semantic_plan": semantic_plan.model_dump(mode="json")},
                    output=execution_route.model_dump(mode="json"),
                ),
                stage_runtime.complete(
                    "mission.clearance-gate",
                    inputs={"route": execution_route.model_dump(mode="json")},
                    output=clearance.model_dump(mode="json"),
                ),
                stage_runtime.complete(
                    "mission.track-export",
                    inputs={"clearance": clearance.model_dump(mode="json")},
                    output=px4_track.model_dump(mode="json"),
                ),
                stage_runtime.complete(
                    "mission.plan-evaluation",
                    inputs={"track": px4_track.model_dump(mode="json")},
                    output={
                        "scores": plan_scores,
                        "validation_results": validation_results,
                    },
                ),
                stage_runtime.complete(
                    "mission.plan-review",
                    inputs={"track": px4_track.model_dump(mode="json")},
                    output=plan_critique.model_dump(mode="json"),
                ),
            ]
        except HarnessGraphError as error:
            raise MissionPreparationBlocked(error.code) from error
        for stage_receipt in preparation_stages:
            evidence.append("harness.stage", stage_receipt.model_dump(mode="json"))
        evidence.append("mission.runtime-actions", runtime_actions.model_dump(mode="json"))
        evidence.append(
            "mission.verification-plan",
            verification_plan.model_dump(mode="json"),
        )
        try:
            checkpoints_stage = stage_runtime.complete(
                "mission.runtime-checkpoints",
                inputs={"plan_review": plan_critique.model_dump(mode="json")},
                output=runtime_checkpoints.model_dump(mode="json"),
            )
            verification_stage = stage_runtime.complete(
                "mission.verification-plan",
                inputs={
                    "checkpoints": runtime_checkpoints.model_dump(mode="json"),
                    "runtime_actions": runtime_actions.model_dump(mode="json"),
                },
                output=verification_plan.model_dump(mode="json"),
            )
            finalize_stage = stage_runtime.complete(
                "mission.evidence-finalize",
                inputs={
                    "verification_plan": verification_plan.model_dump(mode="json"),
                },
                output={
                    "contract_id": contract.contract_id,
                    "route_sha256": sha256_json(execution_route),
                    "track_sha256": sha256_json(px4_track),
                },
            )
            harness_stage_receipts = stage_runtime.finish()
        except HarnessGraphError as error:
            raise MissionPreparationBlocked(error.code) from error
        for stage_receipt in [checkpoints_stage, verification_stage, finalize_stage]:
            evidence.append("harness.stage", stage_receipt.model_dump(mode="json"))
        completed_event_payload = {
            "conversation_id": request.conversation_id,
            "topology_id": harness_topology.topology_id,
            "stage_receipts": [item.model_dump(mode="json") for item in harness_stage_receipts],
        }
        completed_transport = self._invoke_single_extension(
            "harness.event-bus",
            "transport_message",
            evidence=evidence,
            required=True,
            event="mission.preparation.completed",
            payload=completed_event_payload,
        )
        completed_observers = self._invoke_multiple_extensions(
            "harness.observers",
            "observe_harness",
            evidence=evidence,
            event="mission.preparation.completed",
            payload=completed_event_payload,
        )
        evidence.append(
            "harness.event",
            {"transport": completed_transport, "observers": completed_observers},
        )

        evaluations = self._invoke_multiple_extensions(
            "evaluation.preflight",
            "evaluate_preflight",
            evidence=evidence,
            contract=contract,
            task_graph=task_graph,
            semantic_plan=semantic_plan,
            flight_plan=flight_plan,
            route=execution_route,
            clearance=clearance,
            px4_track=px4_track,
            runtime_checkpoints=runtime_checkpoints,
        )
        if evaluations:
            _write_artifact(output_dir / "11a-plugin-evaluations.json", evaluations)
            evidence.append("evaluation.preflight", {"results": evaluations})
        simulation_campaign = self._invoke_single_extension(
            "simulation.campaign-generator",
            "generate_campaign",
            evidence=evidence,
            contract=contract,
            task_graph=task_graph,
            semantic_plan=semantic_plan,
            flight_plan=flight_plan,
            route=execution_route,
            clearance=clearance,
            px4_track=px4_track,
        )
        fault_library = self._invoke_multiple_extensions(
            "simulation.fault-library",
            "describe_fault",
            evidence=evidence,
            contract=contract,
            task_graph=task_graph,
            semantic_plan=semantic_plan,
            flight_plan=flight_plan,
            route=execution_route,
            clearance=clearance,
            px4_track=px4_track,
        )
        if simulation_campaign is not None or fault_library:
            campaign_artifact = {
                "schema_version": "dronedream.simulation-campaign.v1",
                "generator": simulation_campaign,
                "fault_library": fault_library,
                "execution_contract": (
                    "Fault entries are typed simulator-adapter inputs. They do not alter the "
                    "confirmed mission unless an explicit evaluation run selects them."
                ),
            }
            _write_artifact(output_dir / "11b-simulation-campaign.json", campaign_artifact)
            evidence.append("simulation.campaign", campaign_artifact)
        for name, artifact in (
            ("04-task-graph.json", task_graph),
            ("05-semantic-plan.json", semantic_plan),
            ("06-flight-plan.json", flight_plan),
            ("07-plan-critique.json", plan_critique),
            ("08-execution-route.json", execution_route),
            ("09-route-clearance.json", clearance),
            ("10-px4-track.json", px4_track),
            ("11-runtime-checkpoints.json", runtime_checkpoints),
            ("12-runtime-actions.json", runtime_actions),
            ("13-verification-plan.json", verification_plan),
        ):
            _write_artifact(output_dir / name, artifact)

        summary_policy = self._invoke_single_extension(
            "context.summarization-policy",
            "summarize_context",
            evidence=evidence,
            required=True,
            window=self.context_store.summary_window(
                request.conversation_id, max_recent_events=200
            ),
        )
        if not isinstance(summary_policy, dict):
            raise MissionPreparationBlocked("CONTEXT_SUMMARY_POLICY_INVALID")
        summary_text = summary_policy.get("summary")
        through_sequence = summary_policy.get("through_sequence")
        # Empty extractive summaries still consume a real page of tool events.
        # An empty page (cursor 0) must not publish or roll back an existing summary.
        if isinstance(summary_text, str) and type(through_sequence) is int and through_sequence > 0:
            self.context_store.set_summary(request.conversation_id, summary_text, through_sequence)
        retention_policy = self._invoke_single_extension(
            "context.retention-policy",
            "resolve_retention",
            evidence=evidence,
            required=True,
            conversation_id=request.conversation_id,
        )
        if not isinstance(retention_policy, dict):
            raise MissionPreparationBlocked("CONTEXT_RETENTION_POLICY_INVALID")
        removed_events = self.context_store.apply_retention(
            request.conversation_id,
            maximum_events=retention_policy.get("maximum_events", 10_000),
        )
        evidence.append(
            "context.maintenance",
            {
                "store": context_store_policy,
                "retrieval": retrieval_policy,
                "summary_through_sequence": through_sequence,
                "retention": retention_policy,
                "removed_events": removed_events,
            },
        )

        # 规划可能持续数分钟；最终冻结不能換用期间被替换的地图或机型内容。
        if (
            _file_sha256(self.semantic_path) != semantic_sha256
            or _file_sha256(self.vehicle_sdf) != vehicle_sha256
        ):
            raise MissionPreparationBlocked("PREPARATION_ASSET_CHANGED")
        prepared = PreparedMission(
            schema_version="dronedream.prepared-mission.v4",
            intent=intent,
            intent_critique=intent_critique,
            contract=contract,
            domain_actions=domain_actions,
            simulation_capabilities=simulation_capabilities,
            task_graph=task_graph,
            semantic_plan=semantic_plan,
            plan=flight_plan,
            plan_critique=plan_critique,
            execution_route=execution_route,
            route_clearance=clearance,
            px4_track=px4_track,
            runtime_checkpoints=runtime_checkpoints,
            runtime_actions=runtime_actions,
            verification_plan=verification_plan,
            planning_attempts=planning_attempts,
            model_attempt_count=self._model_call_count,
            model_calls=model_calls,
            plugin_snapshot=self.plugin_snapshot,
            harness_topology=harness_topology,
            harness_stage_receipts=harness_stage_receipts,
            plugin_hook_receipts=list(self._hook_receipts),
            tool_receipts=tool_receipts,
            evidence=evidence.read(),
        )
        _write_artifact(output_dir / "prepared-mission.json", prepared)
        lifecycle_binding = self.context_store.lifecycle.record_plan_revision(
            conversation_id=request.conversation_id,
            contract_id=prepared.contract.contract_id,
            prepared_mission_sha256=sha256_json(prepared),
            source_message_sha256=sha256_json({"message": request.message}),
        )
        _write_artifact(output_dir / "mission-lifecycle.json", lifecycle_binding)
        return prepared
