"""Canonical Model + Harness boundary shared with the public control plane.

The public receipt models in this module intentionally have the exact same
fields as the public product repository. Private runtime identity, edition,
and plugin-snapshot bindings use a separately versioned envelope so the public
schema name never describes two different objects.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Final, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dronedream_plugin_sdk.protocol import copy_json, encode_json

from ..plugin_api import verify_plugin_snapshot
from ..plugin_contracts import PluginSnapshot
from .memory import AUTONOMY_MISSION_NAMESPACE, MemoryOwnerScope

CONTROL_PLANE_SCHEMA_VERSION: Final = "dronedream.model-harness-control-plane.v1"
STRUCTURED_INPUT_SCHEMA_VERSION: Final = "dronedream.model-harness-input.v1"
STRUCTURED_OUTPUT_SCHEMA_VERSION: Final = "dronedream.model-harness-output.v1"
RUNTIME_ENVELOPE_SCHEMA_VERSION: Final = "dronedream.model-harness-runtime-envelope.v1"
EXECUTION_AUTHORITY_SCHEMA_VERSION: Final = "dronedream.model-harness-execution-authority.v1"
MEMORY_RETRIEVAL_POLICY_VERSION: Final = "dronedream.memory-retrieval-policy.v1"
LEARNING_PROMOTION_POLICY_VERSION: Final = "dronedream.learning-promotion-policy.v1"
HARD_MAXIMUM_MODEL_CALLS: Final = 48
HARD_MAXIMUM_REPAIR_CYCLES: Final = 6

SourceEdition: TypeAlias = Literal["universal", "sim", "lab", "field", "autonomy"]
PluginTrust: TypeAlias = Literal["managed", "signed", "local_development"]
PluginSelectionSource: TypeAlias = Literal["explicit", "product_managed_default"]
PluginSelectionAuthority: TypeAlias = Literal[
    "product_managed",
    "account_configurable",
    "agent_harness_designer",
]
PluginCapability: TypeAlias = Literal[
    "model_provider",
    "intent_extractor",
    "context_enricher",
    "prompt_pack",
    "tool_provider",
    "planner",
    "optimizer",
    "critic",
    "validator",
    "memory_extractor",
    "memory_consolidator",
    "memory_retriever",
    "semantic_retriever",
    "simulator_adapter",
    "asset_adapter",
    "telemetry_adapter",
    "recovery_strategy",
    "evidence_exporter",
    "notification_adapter",
]

FIXED_KERNEL_RESPONSIBILITIES: Final = (
    "identity_and_tenant_boundary",
    "structured_io_validation",
    "safety_policy",
    "budget_enforcement",
    "acceptance_and_evidence",
    "memory_governance",
    "plugin_trust_and_lifecycle",
)


# 功能：
#   检查标准 JSON 与 2 MiB 预算后计算公共契约使用的规范摘要；摘要相同不授予执行权限。
# 输入：
#   value：由已验证边界导出的 JSON 值。
# 输出：
#   digest：按键排序、紧凑 UTF-8 编码的 SHA-256 摘要。
def _sha256(value: object) -> str:
    encode_json(value)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return digest


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    # 功能：
    #   权限标志只能由显式布尔值表达，不能把数字零或一当作禁用或启用声明。
    # 输入：
    #   cls：正在解析的边界模型类型。
    #   value：显式提供的权限标志。
    # 输出：
    #   value：类型正确的布尔值，具体允许值仍由各字段契约限制。
    @field_validator(
        "online_policy_updates_allowed",
        "grants_execution_authority",
        "tenant_membership_fail_closed",
        "authority_reuse_allowed",
        "single_use",
        "reusable",
        "physical_action_performed",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def validate_boolean_flag(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("HARNESS_BOOLEAN_FLAG_INVALID")
        return value

    # 功能：
    #   固定预算字面值也必须是整数，避免浮点或文本混入公共摘要。
    # 输入：
    #   cls：正在解析的边界模型类型。
    #   value：显式提供的固定预算。
    # 输出：
    #   value：类型正确的整数，是否等于固定上限仍由字段契约限制。
    @field_validator(
        "hard_maximum_model_calls", "hard_maximum_repair_cycles", mode="before", check_fields=False
    )
    @classmethod
    def validate_fixed_integer(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("HARNESS_FIXED_BUDGET_INVALID")
        return value


class PluginSelection(_StrictModel):
    """Exact public content-bound plugin selection schema."""

    slot: PluginCapability
    plugin_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[a-z0-9](?:[a-z0-9._-]{1,126}[a-z0-9])?$",
    )
    version: str = Field(min_length=1, max_length=64)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trust: PluginTrust
    source: PluginSelectionSource = "explicit"
    selected_by: PluginSelectionAuthority = "product_managed"


class HarnessControlPlaneReceipt(_StrictModel):
    """Exact public receipt schema; never add private runtime fields here."""

    schema_version: Literal["dronedream.model-harness-control-plane.v1"] = (
        CONTROL_PLANE_SCHEMA_VERSION
    )
    structured_input_schema_version: Literal["dronedream.model-harness-input.v1"] = (
        STRUCTURED_INPUT_SCHEMA_VERSION
    )
    structured_output_schema_version: Literal["dronedream.model-harness-output.v1"] = (
        STRUCTURED_OUTPUT_SCHEMA_VERSION
    )
    domain: Literal["autonomy.mission"] = AUTONOMY_MISSION_NAMESPACE
    loop_kind: Literal["observe_repair"] = "observe_repair"
    hard_maximum_model_calls: Literal[48] = HARD_MAXIMUM_MODEL_CALLS
    hard_maximum_repair_cycles: Literal[6] = HARD_MAXIMUM_REPAIR_CYCLES
    effective_maximum_model_calls: int = Field(ge=1, le=48, strict=True)
    effective_maximum_repair_cycles: int = Field(ge=0, le=6, strict=True)
    fixed_kernel_responsibilities: tuple[str, ...] = FIXED_KERNEL_RESPONSIBILITIES
    readable_memory_domains: tuple[Literal["account.shared"], Literal["autonomy.mission"]] = (
        "account.shared",
        AUTONOMY_MISSION_NAMESPACE,
    )
    writable_memory_domain: Literal["autonomy.mission"] = AUTONOMY_MISSION_NAMESPACE
    memory_retrieval_policy_version: Literal["dronedream.memory-retrieval-policy.v1"] = (
        MEMORY_RETRIEVAL_POLICY_VERSION
    )
    learning_promotion_policy_version: Literal["dronedream.learning-promotion-policy.v1"] = (
        LEARNING_PROMOTION_POLICY_VERSION
    )
    semantic_memory_authority: Literal["advisory_only"] = "advisory_only"
    online_policy_updates_allowed: Literal[False] = False
    execution_authority_enforcement: Literal["not_integrated"] = "not_integrated"
    grants_execution_authority: Literal[False] = False
    selected_plugins: tuple[PluginSelection, ...]
    selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    # 功能：
    #   核对固定内核、插件身份唯一性、有效上限及完整选择摘要；重算摘要不能移除安全职责。
    # 输入：
    #   self：待接收的公共控制平面回执。
    # 输出：
    #   self：满足固定契约且摘要与内容一致的回执。
    @model_validator(mode="after")
    def effective_caps_are_hard_capped(self) -> HarnessControlPlaneReceipt:
        if self.fixed_kernel_responsibilities != FIXED_KERNEL_RESPONSIBILITIES:
            raise ValueError("HARNESS_FIXED_KERNEL_MISMATCH")
        identifiers = [item.plugin_id for item in self.selected_plugins]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("HARNESS_DUPLICATE_PLUGIN_SELECTION")
        if self.effective_maximum_model_calls > self.hard_maximum_model_calls:
            raise ValueError("effective model-call cap cannot exceed the immutable hard cap")
        if self.effective_maximum_repair_cycles > self.hard_maximum_repair_cycles:
            raise ValueError("effective repair-cycle cap cannot exceed the immutable hard cap")
        if self.selection_sha256 != _sha256(self.selection_payload()):
            raise ValueError("selection_sha256 does not bind the effective control plane")
        return self

    # 功能：
    #   生成排序后的标准 JSON 摘要内容，不把摘要字段本身或私有运行身份混入公共契约。
    # 输入：
    #   self：当前公共回执。
    # 输出：
    #   payload：保持公共 JSON 字节含义的独立选择资料。
    def selection_payload(self) -> dict[str, object]:
        ordered = sorted(
            self.selected_plugins,
            key=lambda item: (item.slot, item.plugin_id, item.version),
        )
        payload = {
            "schema_version": self.schema_version,
            "structured_input_schema_version": self.structured_input_schema_version,
            "structured_output_schema_version": self.structured_output_schema_version,
            "domain": self.domain,
            "loop_kind": self.loop_kind,
            "hard_maximum_model_calls": self.hard_maximum_model_calls,
            "hard_maximum_repair_cycles": self.hard_maximum_repair_cycles,
            "effective_maximum_model_calls": self.effective_maximum_model_calls,
            "effective_maximum_repair_cycles": self.effective_maximum_repair_cycles,
            "fixed_kernel_responsibilities": list(self.fixed_kernel_responsibilities),
            "readable_memory_domains": list(self.readable_memory_domains),
            "writable_memory_domain": self.writable_memory_domain,
            "memory_retrieval_policy_version": self.memory_retrieval_policy_version,
            "learning_promotion_policy_version": self.learning_promotion_policy_version,
            "semantic_memory_authority": self.semantic_memory_authority,
            "online_policy_updates_allowed": self.online_policy_updates_allowed,
            "execution_authority_enforcement": self.execution_authority_enforcement,
            "grants_execution_authority": self.grants_execution_authority,
            "selected_plugins": [item.model_dump(mode="json") for item in ordered],
        }
        return payload


class HarnessRuntimeEnvelope(_StrictModel):
    """Private verified-identity and frozen-snapshot binding."""

    schema_version: Literal["dronedream.model-harness-runtime-envelope.v1"] = (
        RUNTIME_ENVELOPE_SCHEMA_VERSION
    )
    domain: Literal["autonomy.mission"] = AUTONOMY_MISSION_NAMESPACE
    owner_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tenant_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_edition: SourceEdition
    control_plane_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plugin_snapshot_id: str = Field(pattern=r"^plugin-snapshot-[0-9a-f]{24}$")
    plugin_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hard_maximum_model_calls: Literal[48] = HARD_MAXIMUM_MODEL_CALLS
    selected_maximum_model_calls: int = Field(ge=1, le=48, strict=True)
    effective_maximum_model_calls: int = Field(ge=1, le=48, strict=True)
    maximum_intent_rounds: int = Field(ge=1, le=5, strict=True)
    maximum_planning_rounds: int = Field(ge=1, le=5, strict=True)
    budget_sources: dict[str, str]
    identity_source: Literal["verified_supabase_jwt_app_metadata"] = (
        "verified_supabase_jwt_app_metadata"
    )
    tenant_membership_fail_closed: Literal[True] = True
    authority_reuse_allowed: Literal[False] = False

    # 功能：
    #   检查私有运行预算等于受硬上限约束的选中预算，不能独立扩展执行次数。
    # 输入：
    #   self：待验证的私有运行边界。
    # 输出：
    #   self：选中预算与有效预算一致的运行边界。
    @model_validator(mode="after")
    def selected_budget_is_effective_budget(self) -> HarnessRuntimeEnvelope:
        if self.effective_maximum_model_calls != min(
            self.hard_maximum_model_calls, self.selected_maximum_model_calls
        ):
            raise ValueError("runtime model-call budget is not hard capped")
        return self


class ModelHarnessExecutionAuthority(_StrictModel):
    """Content-bound single-use execution record for one validated proposal.

    The public control-plane and structured-output contracts intentionally do
    not grant execution. This separate artifact is issued only after fixed
    kernel validation and is atomically consumed by the authenticated desktop
    runtime before any process is launched.
    """

    schema_version: Literal["dronedream.model-harness-execution-authority.v1"] = (
        EXECUTION_AUTHORITY_SCHEMA_VERSION
    )
    authority_id: str = Field(pattern=r"^execution-authority-[0-9a-f]{32}$")
    thread_id: str = Field(min_length=8, max_length=128)
    plan_revision_id: str = Field(pattern=r"^plan-[0-9a-f]{32}$")
    contract_id: str = Field(min_length=8, max_length=160)
    prepared_mission_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    owner_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tenant_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_plane_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plugin_snapshot_id: str = Field(pattern=r"^plugin-snapshot-[0-9a-f]{24}$")
    plugin_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    issued_at: datetime
    single_use: Literal[True] = True
    reusable: Literal[False] = False
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    # 功能：
    #   核对带时区的签发时间及所有执行维度的摘要；一次性消费由持久化层另行执行。
    # 输入：
    #   self：待接收的一次性执行绑定。
    # 输出：
    #   self：内容和摘要一致的绑定，不代表已经消费或启动飞行。
    @model_validator(mode="after")
    def hash_binds_every_execution_dimension(self) -> ModelHarnessExecutionAuthority:
        if self.issued_at.utcoffset() is None:
            raise ValueError("HARNESS_AUTHORITY_TIMEZONE_REQUIRED")
        if self.authority_sha256 != _sha256(self.binding_payload()):
            raise ValueError("execution authority hash does not bind its payload")
        return self

    # 功能：
    #   导出摘要覆盖的账户、提案、插件和签发信息，排除摘要自身。
    # 输入：
    #   self：当前一次性执行绑定。
    # 输出：
    #   payload：保持签发时间原始时区表示的绑定资料。
    def binding_payload(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "authority_id": self.authority_id,
            "thread_id": self.thread_id,
            "plan_revision_id": self.plan_revision_id,
            "contract_id": self.contract_id,
            "prepared_mission_sha256": self.prepared_mission_sha256,
            "owner_binding_sha256": self.owner_binding_sha256,
            "tenant_binding_sha256": self.tenant_binding_sha256,
            "control_plane_selection_sha256": self.control_plane_selection_sha256,
            "plugin_snapshot_id": self.plugin_snapshot_id,
            "plugin_catalog_sha256": self.plugin_catalog_sha256,
            "issued_at": self.issued_at.isoformat(),
            "single_use": self.single_use,
            "reusable": self.reusable,
        }
        return payload


class HarnessInputEnvelope(_StrictModel):
    """Exact public structured Harness input schema."""

    schema_version: Literal["dronedream.model-harness-input.v1"] = STRUCTURED_INPUT_SCHEMA_VERSION
    request_id: str = Field(min_length=8, max_length=128)
    task_id: str = Field(min_length=8, max_length=128)
    thread_id: str = Field(min_length=8, max_length=128)
    owner_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tenant_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_edition: SourceEdition
    domain: Literal["autonomy.mission"] = AUTONOMY_MISSION_NAMESPACE
    control_plane_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_request: dict[str, object]
    session_context: dict[str, object] = Field(default_factory=dict)
    memory_record_ids: tuple[str, ...] = Field(default=(), max_length=32)

    # 功能：
    #   在 Pydantic 转换前检查并复制输入 JSON，拒绝非有限值、非字符串键和非 JSON 对象。
    # 输入：
    #   cls：结构化输入模型类型。
    #   value：请求或会话上下文的原始映射。
    # 输出：
    #   payload：在 64 KiB、深度和节点预算内的独立 JSON 映射。
    @field_validator("current_request", "session_context", mode="before")
    @classmethod
    def validate_input_payload(cls, value: object) -> dict[str, object]:
        if type(value) is not dict:
            raise ValueError("HARNESS_INPUT_PAYLOAD_INVALID")
        payload = copy_json(value, limit=65_536)
        return payload

    # 功能：
    #   以合并后的 UTF-8 字节检查请求与上下文的共同预算，不分别放行两份满额输入。
    # 输入：
    #   self：字段级 JSON 检查通过后的输入。
    # 输出：
    #   self：合并输入不超过 64 KiB 的封装。
    @model_validator(mode="after")
    def context_is_bounded(self) -> HarnessInputEnvelope:
        encode_json(
            {"current_request": self.current_request, "session_context": self.session_context},
            limit=65_536,
        )
        return self


class HarnessOutputEnvelope(_StrictModel):
    """Exact public structured Harness output schema."""

    schema_version: Literal["dronedream.model-harness-output.v1"] = STRUCTURED_OUTPUT_SCHEMA_VERSION
    request_id: str = Field(min_length=8, max_length=128)
    task_id: str = Field(min_length=8, max_length=128)
    domain: Literal["autonomy.mission"] = AUTONOMY_MISSION_NAMESPACE
    control_plane_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["draft", "needs_input", "blocked", "validated_proposal", "closed"]
    structured_result: dict[str, object]
    model_call_count: int = Field(ge=0, strict=True)
    repair_cycle_count: int = Field(ge=0, strict=True)
    tool_receipt_ids: tuple[str, ...] = Field(default=(), max_length=128)
    validation_receipt_ids: tuple[str, ...] = Field(default=(), max_length=128)
    evidence_receipt_ids: tuple[str, ...] = Field(default=(), max_length=128)
    memory_candidate_ids: tuple[str, ...] = Field(default=(), max_length=32)
    execution_authority_enforcement: Literal["not_integrated"] = "not_integrated"
    grants_execution_authority: Literal[False] = False
    physical_action_performed: Literal[False] = False

    # 功能：
    #   为输出结果应用标准 JSON、大小和复杂度检查，复制内容后再交给消费方验证。
    # 输入：
    #   cls：结构化输出模型类型。
    #   value：模型或任务编排器提供的结构化结果。
    # 输出：
    #   payload：不超过 2 MiB 且不借用原容器的 JSON 结果。
    @field_validator("structured_result", mode="before")
    @classmethod
    def validate_output_payload(cls, value: object) -> dict[str, object]:
        if type(value) is not dict:
            raise ValueError("HARNESS_OUTPUT_PAYLOAD_INVALID")
        payload = copy_json(value)
        return payload


# 功能：
#   按既有优先顺序把内部槽位投影到公共能力分类；这是目录标签，不是授权。
# 输入：
#   slot_id：注册表中已选择插件的槽位标识。
# 输出：
#   capability：公共契约使用的能力分类。
def _capability_for_slot(slot_id: str) -> PluginCapability:
    capability: PluginCapability
    if slot_id == "models.providers" or slot_id in {
        "models.role-policy",
        "models.runtime-router",
        "models.consensus-policy",
    }:
        capability = "model_provider"
    elif slot_id == "input.intent-normalizers":
        capability = "intent_extractor"
    elif slot_id == "models.prompt-packs":
        capability = "prompt_pack"
    elif "memory" in slot_id and "retriev" in slot_id:
        capability = "memory_retriever"
    elif "memory" in slot_id and ("summary" in slot_id or "consolid" in slot_id):
        capability = "memory_consolidator"
    elif "memory" in slot_id and "extract" in slot_id:
        capability = "memory_extractor"
    elif slot_id == "models.structured-output-guards":
        capability = "validator"
    elif slot_id.startswith("context.") or slot_id.startswith("harness."):
        capability = "context_enricher"
    elif slot_id.startswith("assets."):
        capability = "asset_adapter"
    elif slot_id.startswith("simulation."):
        capability = "simulator_adapter"
    elif slot_id.startswith("evidence."):
        capability = "evidence_exporter"
    elif slot_id.startswith("notifications."):
        capability = "notification_adapter"
    elif slot_id.startswith("connectors.") or slot_id.startswith("tools."):
        capability = "tool_provider"
    elif slot_id.startswith("models."):
        capability = "critic"
    elif "optimizer" in slot_id or "ranker" in slot_id:
        capability = "optimizer"
    elif "scorer" in slot_id or "advisor" in slot_id:
        capability = "critic"
    elif slot_id.startswith("validation.") or slot_id.startswith("evaluation."):
        capability = "validator"
    elif slot_id.startswith("safety.") or "replan" in slot_id or "fallback" in slot_id:
        capability = "recovery_strategy"
    elif "anomaly" in slot_id or "watchdog" in slot_id:
        capability = "telemetry_adapter"
    elif slot_id.startswith("planning.") or "coverage-planner" in slot_id:
        capability = "planner"
    elif slot_id.startswith("runtime.") or slot_id.startswith("flight-control."):
        capability = "tool_provider"
    else:
        capability = "context_enricher"
    return capability


# 功能：
#   复用统一快照校验核对配置、清单和目录摘要，再投影为公共回执；标签不替代插件准入验签。
# 输入：
#   snapshot：插件管理器提供的已冻结快照。
# 输出：
#   ordered：按公共能力、插件标识与版本排序的选择元组。
def selections_from_plugin_snapshot(snapshot: PluginSnapshot) -> tuple[PluginSelection, ...]:
    if snapshot.schema_version != "dronedream.plugin-snapshot.v2":
        raise ValueError("PLUGIN_SNAPSHOT_SELECTION_PROVENANCE_REQUIRED")
    snapshot = verify_plugin_snapshot(snapshot)
    selections: list[PluginSelection] = []
    for entry in snapshot.plugins:
        manifest = entry.manifest
        if manifest is None:
            raise ValueError(f"PLUGIN_SNAPSHOT_MANIFEST_REQUIRED:{entry.plugin_id}")
        trust: PluginTrust
        source: PluginSelectionSource
        if entry.bundle_root is None:
            trust = "managed"
        elif manifest.signature is not None:
            # This labels the frozen registry snapshot. Signature verification belongs
            # to plugin admission; presence of this field alone cannot admit a bundle.
            trust = "signed"
        else:
            trust = "local_development"
        source = entry.selection_source
        selections.append(
            PluginSelection(
                slot=_capability_for_slot(manifest.placement.slot_id),
                plugin_id=entry.plugin_id,
                version=entry.version,
                content_sha256=_sha256(
                    {
                        "package_sha256": entry.package_sha256,
                        "manifest_sha256": entry.manifest_sha256,
                        "configuration_sha256": entry.configuration_sha256,
                        "capability_ids": sorted(entry.capability_ids),
                    }
                ),
                trust=trust,
                source=source,
                selected_by=entry.selected_by,
            )
        )
    ordered = tuple(
        sorted(
            selections,
            key=lambda item: (item.slot, item.plugin_id, item.version),
        )
    )
    return ordered


# 功能：
#   冻结独立的插件选择，校验严格整数预算后生成与公共契约字节兼容的控制回执。
# 输入：
#   selections：已选插件的公共投影元组。
#   effective_maximum_model_calls：本次最多允许的模型调用次数。
#   effective_maximum_repair_cycles：本次最多允许的修正轮数。
# 输出：
#   receipt：不与调用方选择对象共享状态的公共控制回执。
def compile_autonomy_control_plane_receipt(
    selections: tuple[PluginSelection, ...],
    *,
    effective_maximum_model_calls: int,
    effective_maximum_repair_cycles: int = HARD_MAXIMUM_REPAIR_CYCLES,
) -> HarnessControlPlaneReceipt:
    if type(effective_maximum_model_calls) is not int or not (
        1 <= effective_maximum_model_calls <= HARD_MAXIMUM_MODEL_CALLS
    ):
        raise ValueError("effective model-call cap must be within the immutable hard cap")
    if type(effective_maximum_repair_cycles) is not int or not (
        0 <= effective_maximum_repair_cycles <= HARD_MAXIMUM_REPAIR_CYCLES
    ):
        raise ValueError("effective repair-cycle cap must be within the immutable hard cap")
    selected = [PluginSelection.model_validate(item) for item in selections]
    ordered = tuple(sorted(selected, key=lambda item: (item.slot, item.plugin_id, item.version)))
    canonical: dict[str, Any] = {
        "schema_version": CONTROL_PLANE_SCHEMA_VERSION,
        "structured_input_schema_version": STRUCTURED_INPUT_SCHEMA_VERSION,
        "structured_output_schema_version": STRUCTURED_OUTPUT_SCHEMA_VERSION,
        "domain": AUTONOMY_MISSION_NAMESPACE,
        "loop_kind": "observe_repair",
        "hard_maximum_model_calls": HARD_MAXIMUM_MODEL_CALLS,
        "hard_maximum_repair_cycles": HARD_MAXIMUM_REPAIR_CYCLES,
        "effective_maximum_model_calls": effective_maximum_model_calls,
        "effective_maximum_repair_cycles": effective_maximum_repair_cycles,
        "fixed_kernel_responsibilities": list(FIXED_KERNEL_RESPONSIBILITIES),
        "readable_memory_domains": ["account.shared", AUTONOMY_MISSION_NAMESPACE],
        "writable_memory_domain": AUTONOMY_MISSION_NAMESPACE,
        "memory_retrieval_policy_version": MEMORY_RETRIEVAL_POLICY_VERSION,
        "learning_promotion_policy_version": LEARNING_PROMOTION_POLICY_VERSION,
        "semantic_memory_authority": "advisory_only",
        "online_policy_updates_allowed": False,
        "execution_authority_enforcement": "not_integrated",
        "grants_execution_authority": False,
        "selected_plugins": [item.model_dump(mode="json") for item in ordered],
    }
    receipt = HarnessControlPlaneReceipt(
        effective_maximum_model_calls=effective_maximum_model_calls,
        effective_maximum_repair_cycles=effective_maximum_repair_cycles,
        selected_plugins=ordered,
        selection_sha256=_sha256(canonical),
    )
    return receipt


# 功能：
#   将已验证账户、同一插件快照和公共回执绑定为运行边界，拒绝预算或选择错配。
# 输入：
#   scope：由身份层确认的账户与租户范围；此函数不自行验证登录令牌。
#   receipt：本任务的公共控制回执。
#   snapshot：生成该回执时使用的冻结插件快照。
#   selected_maximum_model_calls：与回执一致的模型调用预算。
#   maximum_intent_rounds：意图理解最多轮数。
#   maximum_planning_rounds：计划生成最多轮数。
# 输出：
#   runtime：身份、插件与选中预算一致的私有运行封装。
def autonomy_runtime_envelope(
    scope: MemoryOwnerScope,
    receipt: HarnessControlPlaneReceipt,
    snapshot: PluginSnapshot,
    *,
    selected_maximum_model_calls: int,
    maximum_intent_rounds: int,
    maximum_planning_rounds: int,
) -> HarnessRuntimeEnvelope:
    scope = MemoryOwnerScope.model_validate(scope.model_dump(mode="python", warnings="error"))
    receipt = HarnessControlPlaneReceipt.model_validate(receipt)
    snapshot = verify_plugin_snapshot(snapshot)
    if scope.source_edition is None:
        raise ValueError("source edition is required for a runtime envelope")
    if selected_maximum_model_calls != receipt.effective_maximum_model_calls:
        raise ValueError("HARNESS_RUNTIME_RECEIPT_BUDGET_MISMATCH")
    if selections_from_plugin_snapshot(snapshot) != tuple(
        sorted(receipt.selected_plugins, key=lambda item: (item.slot, item.plugin_id, item.version))
    ):
        raise ValueError("HARNESS_RUNTIME_SNAPSHOT_SELECTION_MISMATCH")
    owner_binding, tenant_binding = owner_scope_bindings(scope)
    runtime = HarnessRuntimeEnvelope(
        owner_binding_sha256=owner_binding,
        tenant_binding_sha256=tenant_binding,
        source_edition=scope.source_edition,
        control_plane_selection_sha256=receipt.selection_sha256,
        plugin_snapshot_id=snapshot.snapshot_id,
        plugin_catalog_sha256=snapshot.catalog_sha256,
        selected_maximum_model_calls=selected_maximum_model_calls,
        effective_maximum_model_calls=min(HARD_MAXIMUM_MODEL_CALLS, selected_maximum_model_calls),
        maximum_intent_rounds=maximum_intent_rounds,
        maximum_planning_rounds=maximum_planning_rounds,
        budget_sources={
            "model_calls": "harness.budget-policy",
            "repair_rounds": "planning.workflow-policy",
        },
    )
    return runtime


# 功能：
#   生成自主任务账户与租户的稳定摘要；摘要是绑定标识，不是登录凭证或匿名化保证。
# 输入：
#   scope：由身份层确定的账户、租户与组织范围。
# 输出：
#   bindings：账户摘要与租户摘要组成的元组。
def owner_scope_bindings(scope: MemoryOwnerScope) -> tuple[str, str]:
    scope = MemoryOwnerScope.model_validate(scope.model_dump(mode="python", warnings="error"))
    owner_binding = _sha256(
        {
            "owner_account_id": scope.owner_account_id,
            "namespace": AUTONOMY_MISSION_NAMESPACE,
        }
    )
    tenant_binding = _sha256(
        {
            "owner_binding_sha256": owner_binding,
            "tenant_id": scope.tenant_id or "personal",
            "organization_id": scope.organization_id or "personal",
        }
    )
    bindings = owner_binding, tenant_binding
    return bindings


# 功能：
#   重新验证运行边界后生成单次执行绑定；批准、持久化和原子消费仍由固定内核负责。
# 输入：
#   thread_id：绑定的任务会话。
#   plan_revision_id：已验证计划的精确修订标识。
#   contract_id：任务契约标识。
#   prepared_mission_sha256：已准备任务内容摘要。
#   runtime：该计划使用的私有运行边界。
#   issued_at：可选带时区签发时间，省略时使用当前 UTC。
# 输出：
#   authority：涵盖身份、任务和插件快照的一次性执行资料。
def compile_execution_authority(
    *,
    thread_id: str,
    plan_revision_id: str,
    contract_id: str,
    prepared_mission_sha256: str,
    runtime: HarnessRuntimeEnvelope,
    issued_at: datetime | None = None,
) -> ModelHarnessExecutionAuthority:
    runtime = HarnessRuntimeEnvelope.model_validate(runtime)
    observed_at = issued_at if issued_at is not None else datetime.now(UTC)
    if not isinstance(observed_at, datetime) or observed_at.utcoffset() is None:
        raise ValueError("HARNESS_AUTHORITY_TIMEZONE_REQUIRED")
    payload: dict[str, object] = {
        "schema_version": EXECUTION_AUTHORITY_SCHEMA_VERSION,
        "authority_id": f"execution-authority-{uuid4().hex}",
        "thread_id": thread_id,
        "plan_revision_id": plan_revision_id,
        "contract_id": contract_id,
        "prepared_mission_sha256": prepared_mission_sha256,
        "owner_binding_sha256": runtime.owner_binding_sha256,
        "tenant_binding_sha256": runtime.tenant_binding_sha256,
        "control_plane_selection_sha256": runtime.control_plane_selection_sha256,
        "plugin_snapshot_id": runtime.plugin_snapshot_id,
        "plugin_catalog_sha256": runtime.plugin_catalog_sha256,
        "issued_at": observed_at.isoformat(),
        "single_use": True,
        "reusable": False,
    }
    authority = ModelHarnessExecutionAuthority(
        **payload,
        authority_sha256=_sha256(payload),
    )
    return authority


# 功能：
#   重新验证输入后计算规范摘要，不直接为被绕过校验而修改的对象签下有效输入标识。
# 输入：
#   input_envelope：待绑定的结构化输入。
# 输出：
#   digest：当前有效输入的完整 JSON 摘要。
def harness_input_sha256(input_envelope: HarnessInputEnvelope) -> str:
    input_envelope = HarnessInputEnvelope.model_validate(input_envelope)
    digest = _sha256(input_envelope.model_dump(mode="json"))
    return digest


# 功能：
#   1. 重新验证全部模型，拒绝构造后改写导致的权限、数据或预算绕过。
#   2. 核对账户、租户、版本来源、请求与选择摘要，拒绝跨运行重放及超预算提案。
#   3. 完成态必须提供对应回执标识；固定内核仍需核查回执内容，标识列表不授权物理动作。
# 输入：
#   receipt：本次选择的公共控制回执。
#   runtime：已绑定账户和插件的私有运行封装。
#   output：准备接收的结构化输出。
#   input_envelope：可选原始结构化输入，存在时执行精确身份与内容比对。
# 输出：
#   None：不返回业务数据。
def validate_output_against_boundaries(
    receipt: HarnessControlPlaneReceipt,
    runtime: HarnessRuntimeEnvelope,
    output: HarnessOutputEnvelope,
    *,
    input_envelope: HarnessInputEnvelope | None = None,
) -> None:
    receipt = HarnessControlPlaneReceipt.model_validate(receipt)
    runtime = HarnessRuntimeEnvelope.model_validate(runtime)
    output = HarnessOutputEnvelope.model_validate(output)
    if input_envelope is not None:
        input_envelope = HarnessInputEnvelope.model_validate(input_envelope)
    if output.domain != receipt.domain or output.domain != runtime.domain:
        raise ValueError("Harness output domain does not match its boundary")
    if runtime.control_plane_selection_sha256 != receipt.selection_sha256:
        raise ValueError("Harness runtime is not bound to its control-plane selection")
    if runtime.effective_maximum_model_calls != receipt.effective_maximum_model_calls:
        raise ValueError("HARNESS_RUNTIME_RECEIPT_BUDGET_MISMATCH")
    if output.control_plane_selection_sha256 != receipt.selection_sha256:
        raise ValueError("Harness output is not bound to its control-plane selection")
    if output.model_call_count > runtime.effective_maximum_model_calls:
        raise ValueError("Harness output exceeds the selected model-call budget")
    if output.model_call_count > receipt.effective_maximum_model_calls:
        raise ValueError("Harness output exceeds the canonical model-call budget")
    if output.repair_cycle_count > receipt.effective_maximum_repair_cycles:
        raise ValueError("Harness output exceeds the fixed repair-cycle budget")
    if input_envelope is not None:
        # Digest equality binds bytes, not the account entitled to use those bytes.
        # Compare the authenticated runtime bindings separately from the input hash.
        if input_envelope.owner_binding_sha256 != runtime.owner_binding_sha256:
            raise ValueError("Harness input owner does not match its runtime")
        if input_envelope.tenant_binding_sha256 != runtime.tenant_binding_sha256:
            raise ValueError("Harness input tenant does not match its runtime")
        if input_envelope.source_edition != runtime.source_edition:
            raise ValueError("Harness input edition does not match its runtime")
        if input_envelope.control_plane_selection_sha256 != receipt.selection_sha256:
            raise ValueError("Harness input is not bound to its control-plane selection")
        if input_envelope.domain != receipt.domain:
            raise ValueError("Harness input domain does not match its control-plane receipt")
        if output.request_id != input_envelope.request_id:
            raise ValueError("Harness output request does not match its input envelope")
        if output.task_id != input_envelope.task_id:
            raise ValueError("Harness output task does not match its input envelope")
        if output.input_envelope_sha256 != harness_input_sha256(input_envelope):
            raise ValueError("Harness output is not bound to its validated input envelope")
    if output.status in {"validated_proposal", "closed"} and not output.validation_receipt_ids:
        raise ValueError("validated Harness output requires a validation receipt")
    if output.status == "closed" and not output.evidence_receipt_ids:
        raise ValueError("closed Harness output requires an evidence receipt")
