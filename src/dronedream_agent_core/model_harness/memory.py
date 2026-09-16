"""Governed Model + Harness account memory shared by every DroneDream edition.

Raw conversation events remain in :mod:`dronedream_agent_core.context` and are
always isolated by conversation/thread.  This store contains only deliberately
promoted summaries, preferences, and constraints.  It never stores execution
authority or one-time credentials.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dronedream_plugin_sdk.protocol import decode_json, encode_json

AUTONOMY_MISSION_NAMESPACE = "autonomy.mission"
ACCOUNT_SHARED_NAMESPACE = "account.shared"
_EMPTY_BOUNDARY = ""
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{1,159}$")
_MEMORY_KEY = re.compile(r"^(summary|preference|constraint)\.[a-z][a-z0-9._-]{1,95}$")
SourceEdition = Literal["universal", "sim", "lab", "field", "autonomy"]
MemorySourceKind = Literal[
    "explicit_user_update",
    "verified_product_receipt",
    "model_inference",
    "plugin_inference",
]
ForgetMode = Literal["soft", "permanent"]
_PROMOTABLE_SOURCE_KINDS = {"explicit_user_update", "verified_product_receipt"}
_RECEIPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$")

# These fields would turn memory into a reusable authorization channel.  They
# are rejected both before persistence and again when records are read.
_FORBIDDEN_AUTHORITY_FIELDS = {
    "access_token",
    "actuator_authority",
    "arm_authority",
    "authorization",
    "bearer",
    "confirmation",
    "confirmation_token",
    "execute_authority",
    "execution_id",
    "execution_token",
    "flight_authority",
    "grant",
    "grant_token",
    "model_grant",
    "one_time_authorization",
    "operator_approval",
    "operator_takeover",
    "plan_confirmation",
    "refresh_token",
    "runtime_token",
    "secret",
    "session_token",
    "write_authority",
}

_ALLOWED_PAYLOAD_FIELDS = {
    "constraints",
    "input_channel",
    "locale",
    "map_asset_id",
    "payload_action",
    "return_entity",
    "target_entity",
    "vehicle_asset_id",
}

_SENSITIVE_TEXT = re.compile(
    r"(?:api[ _-]?key|password|refresh[ _-]?token|access[ _-]?token|"
    r"bearer\s+[a-z0-9._-]+|\bsk-[a-z0-9_-]{8,}|"
    r"eyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.)",
    re.IGNORECASE,
)
_INSTRUCTION_TEXT = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?|"
    r"system\s+prompt|developer\s+message|"
    r"(?:^|\n)\s*(?:system|assistant|tool)\s*:|"
    r"<\|(?:system|assistant|developer)|"
    r"operator\s+approval|plan\s+confirmation|"
    r"(?:arm|flight|write|execute|actuator)\s+authority|"
    r"one[ _-]?time\s+(?:grant|authorization)|execution\s+token)"
    r"|(?:忽略|无视)(?:之前|以上|先前|前面)(?:的)?(?:所有)?(?:指令|要求)|"
    r"系统提示词|开发者消息|(?:操作员|用户)(?:批准|授权)|计划确认|"
    r"(?:执行|飞行|写入|武装|解锁|控制)(?:权限|授权)|一次性授权|执行令牌",
    re.IGNORECASE,
)


# 功能：
#   为记忆持久化与过期判断取得 UTC 时间，不承担飞行控制的单调时钟职责。
# 输入：
#   无。
# 输出：
#   now：带 UTC 时区的当前时间。
def _now() -> datetime:
    now = datetime.now(UTC)
    return now


# 功能：
#   统一字段大小写与分隔符，随后同时检查规范化碰撞和禁止授权字段。
# 输入：
#   value：原始记忆字段名。
# 输出：
#   normalized：规范化字段名。
def _normalize_field(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return normalized


# 功能：
#   对已验证记忆载荷生成确定性 JSON，用于相同观察去重而非生成执行授权。
# 输入：
#   value：已通过内容限制的字典。
# 输出：
#   canonical：按键排序的无非有限数值 JSON 文本。
def _canonical_payload(value: dict[str, Any]) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return canonical


# 功能：
#   提取拉丁词及中文连续片段/双字词用于本地词汇相关性排序，不声称这是语义神经模型。
# 输入：
#   value：有界查询文本或已治理记忆 JSON。
# 输出：
#   terms：只影响检索顺序、不会新增记忆或权限的词集合。
def _semantic_terms(value: object) -> set[str]:
    text = (value if isinstance(value, str) else encode_json(value, limit=64 * 1024)).casefold()[
        :16_000
    ]
    terms = {token for token in re.findall(r"[a-z0-9][a-z0-9._-]{1,63}", text) if len(token) >= 2}
    for sequence in re.findall(r"[\u3400-\u9fff]{2,64}", text):
        terms.add(sequence)
        terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return terms


# 功能：
#   1. 只接受受治理字段，拒绝授权、秘密、可疑指令、重复规范化字段及超限内容。
#   2. 保留结构化约束列表，其余标识与偏好必须是文本或空值，不能把布尔值解释为资产。
# 输入：
#   value：候选或数据库中读取的载荷字典。
# 输出：
#   decoded：独立且不超过 32 KiB 的安全结构化载荷。
def _validated_payload(value: dict[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or len(value) > len(_ALLOWED_PAYLOAD_FIELDS):
        raise ValueError("ACCOUNT_MEMORY_PAYLOAD_INVALID")
    normalized_payload: dict[str, Any] = {}
    for key, item in value.items():
        if type(key) is not str or len(key) > 100:
            raise ValueError("ACCOUNT_MEMORY_PAYLOAD_KEY_INVALID")
        normalized = _normalize_field(key)
        if normalized in normalized_payload:
            raise ValueError("ACCOUNT_MEMORY_PAYLOAD_FIELD_COLLISION")
        normalized_payload[normalized] = item
    unknown = set(normalized_payload).difference(_ALLOWED_PAYLOAD_FIELDS)
    forbidden = set(normalized_payload).intersection(_FORBIDDEN_AUTHORITY_FIELDS)
    if forbidden:
        raise ValueError("ACCOUNT_MEMORY_AUTHORITY_FIELD_FORBIDDEN")
    if unknown:
        raise ValueError("ACCOUNT_MEMORY_PAYLOAD_FIELD_NOT_GOVERNED")
    for normalized, item in normalized_payload.items():
        if normalized == "constraints":
            if (
                not isinstance(item, list)
                or len(item) > 32
                or any(
                    not isinstance(constraint, str) or not constraint.strip() for constraint in item
                )
            ):
                raise ValueError("ACCOUNT_MEMORY_CONSTRAINTS_INVALID")
            text_values = item
        elif not isinstance(item, str) and item is not None:
            raise ValueError("ACCOUNT_MEMORY_PAYLOAD_NOT_STRUCTURED")
        else:
            text_values = [item] if isinstance(item, str) else []
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("ACCOUNT_MEMORY_NUMBER_INVALID")
        for text in text_values:
            if len(text) > 500 or "\x00" in text:
                raise ValueError("ACCOUNT_MEMORY_TEXT_INVALID")
            if _SENSITIVE_TEXT.search(text):
                raise ValueError("ACCOUNT_MEMORY_SENSITIVE_TEXT_FORBIDDEN")
            if _INSTRUCTION_TEXT.search(text):
                raise ValueError("ACCOUNT_MEMORY_INSTRUCTION_TEXT_FORBIDDEN")
    encoded = json.dumps(
        normalized_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(encoded.encode("utf-8")) > 32 * 1024:
        raise ValueError("ACCOUNT_MEMORY_PAYLOAD_TOO_LARGE")
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("ACCOUNT_MEMORY_PAYLOAD_INVALID")
    return decoded


# 功能：
#   1. 插件检索/排序后重验固定记忆封装，剔除身份、凭证及执行授权通道。
#   2. 核对条目类型、唯一键、有效期和实际估算预算；该预算估算不等于供应商计费 token。
# 输入：
#   value：插件或本地检索输出的记忆封装。
# 输出：
#   context：只含受治理内容且 authority_reuse_allowed 为 False 的独立封装。
def validate_account_memory_model_context(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("ACCOUNT_MEMORY_CONTEXT_INVALID")
    if set(value) != {
        "namespace",
        "items",
        "top_k",
        "token_budget",
        "estimated_tokens",
        "authority_reuse_allowed",
    }:
        raise ValueError("ACCOUNT_MEMORY_CONTEXT_FIELDS_INVALID")
    if value.get("namespace") != AUTONOMY_MISSION_NAMESPACE:
        raise ValueError("ACCOUNT_MEMORY_NAMESPACE_INVALID")
    if value.get("authority_reuse_allowed") is not False:
        raise ValueError("ACCOUNT_MEMORY_AUTHORITY_REUSE_FORBIDDEN")
    top_k = value.get("top_k")
    token_budget = value.get("token_budget")
    estimated_tokens = value.get("estimated_tokens")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 16:
        raise ValueError("ACCOUNT_MEMORY_TOP_K_INVALID")
    if (
        not isinstance(token_budget, int)
        or isinstance(token_budget, bool)
        or not 256 <= token_budget <= 4_096
    ):
        raise ValueError("ACCOUNT_MEMORY_TOKEN_BUDGET_INVALID")
    if (
        not isinstance(estimated_tokens, int)
        or isinstance(estimated_tokens, bool)
        or not 0 <= estimated_tokens <= token_budget
    ):
        raise ValueError("ACCOUNT_MEMORY_ESTIMATE_INVALID")
    raw_items = value.get("items")
    if not isinstance(raw_items, list) or len(raw_items) > top_k:
        raise ValueError("ACCOUNT_MEMORY_ITEMS_INVALID")
    items: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    actual_estimated_tokens = 0
    for raw_item in raw_items:
        if not isinstance(raw_item, dict) or set(raw_item) != {
            "kind",
            "memory_key",
            "payload",
            "evidence_count",
            "confidence",
            "last_observed_at",
            "expires_at",
            "status",
            "provenance_count",
        }:
            raise ValueError("ACCOUNT_MEMORY_ITEM_FIELDS_INVALID")
        kind = raw_item.get("kind")
        memory_key = raw_item.get("memory_key")
        status = raw_item.get("status")
        if not isinstance(kind, str) or kind not in {"summary", "preference", "constraint"}:
            raise ValueError("ACCOUNT_MEMORY_KIND_INVALID")
        if not isinstance(memory_key, str) or not _MEMORY_KEY.fullmatch(memory_key):
            raise ValueError("ACCOUNT_MEMORY_KEY_INVALID")
        if memory_key in seen_keys:
            raise ValueError("ACCOUNT_MEMORY_DUPLICATE_KEY")
        seen_keys.add(memory_key)
        if not memory_key.startswith(f"{kind}.") or status != "active":
            raise ValueError("ACCOUNT_MEMORY_ITEM_STATE_INVALID")
        payload = raw_item.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("ACCOUNT_MEMORY_PAYLOAD_INVALID")
        clean_payload = _validated_payload(payload)
        evidence_count = raw_item.get("evidence_count")
        confidence = raw_item.get("confidence")
        provenance_count = raw_item.get("provenance_count")
        if (
            not isinstance(evidence_count, int)
            or isinstance(evidence_count, bool)
            or not 1 <= evidence_count <= 1_000_000
        ):
            raise ValueError("ACCOUNT_MEMORY_EVIDENCE_INVALID")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("ACCOUNT_MEMORY_CONFIDENCE_INVALID")
        if (
            not isinstance(provenance_count, int)
            or isinstance(provenance_count, bool)
            or not 0 <= provenance_count <= 16
        ):
            raise ValueError("ACCOUNT_MEMORY_PROVENANCE_INVALID")
        try:
            if any(
                type(raw_item[key]) is not str or len(raw_item[key]) > 80
                for key in ("last_observed_at", "expires_at")
            ):
                raise ValueError("ACCOUNT_MEMORY_TIME_INVALID")
            last_observed_at = datetime.fromisoformat(raw_item["last_observed_at"])
            expires_at = datetime.fromisoformat(raw_item["expires_at"])
        except (TypeError, ValueError) as error:
            raise ValueError("ACCOUNT_MEMORY_TIME_INVALID") from error
        if (
            last_observed_at.tzinfo is None
            or expires_at.tzinfo is None
            or expires_at <= _now()
            or last_observed_at > expires_at
        ):
            raise ValueError("ACCOUNT_MEMORY_TTL_INVALID")
        clean_item = {**raw_item, "payload": clean_payload, "confidence": float(confidence)}
        actual_estimated_tokens += max(
            1,
            len(json.dumps(clean_item, ensure_ascii=False, sort_keys=True).encode("utf-8")) // 4,
        )
        items.append(clean_item)
    if actual_estimated_tokens > token_budget or estimated_tokens != actual_estimated_tokens:
        raise ValueError("ACCOUNT_MEMORY_ESTIMATE_MISMATCH")
    context = {
        "namespace": AUTONOMY_MISSION_NAMESPACE,
        "items": items,
        "top_k": top_k,
        "token_budget": token_budget,
        "estimated_tokens": actual_estimated_tokens,
        "authority_reuse_allowed": False,
    }
    return context


# 功能：
#   为匿名调用或关闭记忆的链路构造空封装，仍执行与非空上下文相同的预算校验。
# 输入：
#   top_k：条目上限；token_budget：估算上下文预算。
# 输出：
#   context：没有条目且不授予任何权限的封装。
def empty_account_memory_model_context(
    *, top_k: int = 8, token_budget: int = 1_200
) -> dict[str, Any]:
    context = validate_account_memory_model_context(
        {
            "namespace": AUTONOMY_MISSION_NAMESPACE,
            "items": [],
            "top_k": top_k,
            "token_budget": token_budget,
            "estimated_tokens": 0,
            "authority_reuse_allowed": False,
        }
    )
    return context


class MemoryOwnerScope(BaseModel):
    """Stable account boundary for one Model + Harness responsibility domain.

    ``source_edition`` is audit metadata only.  It is intentionally excluded
    from :meth:`boundary_key`, so SIM/LAB/FIELD/AUTONOMY/Universal share the same
    AUTONOMY mission memory for the same account and tenant boundary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    owner_account_id: str = Field(min_length=2, max_length=160)
    namespace: Literal["autonomy.mission", "account.shared"] = AUTONOMY_MISSION_NAMESPACE
    tenant_id: str | None = Field(default=None, min_length=2, max_length=160)
    organization_id: str | None = Field(default=None, min_length=2, max_length=160)
    source_edition: SourceEdition | None = None

    # 功能：
    #   只允许稳定、不透明的账户和租户标识，显示名称或任意文本不能成为隔离边界。
    # 输入：
    #   cls：范围合同类型；value：账户/租户/组织标识或 None。
    # 输出：
    #   value：验证通过的标识。
    @field_validator("owner_account_id", "tenant_id", "organization_id")
    @classmethod
    def validate_identity(cls, value: str | None) -> str | None:
        if value is not None and not _IDENTITY.fullmatch(value):
            raise ValueError("memory identity must be a stable opaque identifier")
        return value

    # 功能：
    #   重新验证可能被绕过冻结规则修改的范围，并生成数据库隔离四元组；软件版本不是租户。
    # 输入：
    #   self：待使用的账户范围。
    # 输出：
    #   boundary：账户、租户、组织及职责域；缺省可选范围用空串表示。
    def boundary_key(self) -> tuple[str, str, str, str]:
        checked = MemoryOwnerScope.model_validate(self.model_dump(mode="python"), strict=True)
        boundary = (
            checked.owner_account_id,
            checked.tenant_id or _EMPTY_BOUNDARY,
            checked.organization_id or _EMPTY_BOUNDARY,
            checked.namespace,
        )
        return boundary


class GovernedMemoryEntry(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True, hide_input_in_errors=True
    )

    owner_account_id: str
    namespace: Literal["autonomy.mission", "account.shared"] = AUTONOMY_MISSION_NAMESPACE
    tenant_id: str | None = None
    organization_id: str | None = None
    kind: Literal["summary", "preference", "constraint"]
    memory_key: str = Field(min_length=3, max_length=104)
    payload: dict[str, Any]
    source_conversation_id: str = Field(min_length=1, max_length=160)
    source_edition: SourceEdition | None = None
    provenance: list[dict[str, str]] = Field(default_factory=list, max_length=16)
    evidence_count: int = Field(ge=1, le=1_000_000, strict=True)
    confidence: float = Field(ge=0.0, le=1.0, strict=True)
    last_observed_at: datetime
    expires_at: datetime
    status: Literal["active", "superseded", "expired", "quarantined", "revoked"] = "active"
    created_at: datetime
    updated_at: datetime

    # 功能：
    #   验证记忆键属于摘要、偏好或约束类型，不能当作任意存储路径。
    # 输入：
    #   cls：条目合同类型；value：待验证记忆键。
    # 输出：
    #   value：合法记忆键。
    @field_validator("memory_key")
    @classmethod
    def validate_memory_key(cls, value: str) -> str:
        if not _MEMORY_KEY.fullmatch(value):
            raise ValueError("account memory key is invalid")
        return value

    # 功能：
    #   对新建或读出的活动条目应用相同内容限制并断开可变引用。
    # 输入：
    #   cls：条目合同类型；value：结构化记忆内容。
    # 输出：
    #   payload：安全、独立的载荷。
    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        payload = _validated_payload(value)
        return payload

    # 功能：
    #   统一持久化时间为 UTC，拒绝不能确定时区的记录，防止跨时区字符串比较错误。
    # 输入：
    #   cls：条目合同类型；value：已解析的观测、过期或写入时间。
    # 输出：
    #   timestamp：带 UTC 时区的时间。
    @field_validator("last_observed_at", "expires_at", "created_at", "updated_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ACCOUNT_MEMORY_TIME_INVALID")
        timestamp = value.astimezone(UTC)
        return timestamp

    # 功能：
    #   联合核对记忆键类型、完整所属范围和生命周期顺序。
    # 输入：
    #   self：各字段已验证的条目。
    # 输出：
    #   self：范围和有效期关系一致的条目。
    @model_validator(mode="after")
    def kind_matches_key(self) -> GovernedMemoryEntry:
        if not self.memory_key.startswith(f"{self.kind}."):
            raise ValueError("account memory kind and key disagree")
        MemoryOwnerScope(
            owner_account_id=self.owner_account_id,
            namespace=self.namespace,
            tenant_id=self.tenant_id,
            organization_id=self.organization_id,
            source_edition=self.source_edition,
        )
        if self.last_observed_at > self.expires_at:
            raise ValueError("ACCOUNT_MEMORY_TTL_INVALID")
        return self

    # 功能：
    #   仅投影模型需要的受治理内容，不输出账户、来源凭证或可复用授权。
    # 输入：
    #   self：验证过的记忆条目。
    # 输出：
    #   context：包含独立载荷和有界证据统计的模型条目。
    def model_context(self) -> dict[str, Any]:
        context = {
            "kind": self.kind,
            "memory_key": self.memory_key,
            "payload": deepcopy(self.payload),
            "evidence_count": self.evidence_count,
            "confidence": self.confidence,
            "last_observed_at": self.last_observed_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "status": self.status,
            "provenance_count": len(self.provenance),
        }
        return context


class SessionMemoryCandidate(BaseModel):
    """One structured observation awaiting account-level consolidation."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True, hide_input_in_errors=True
    )

    candidate_id: str = Field(pattern=r"^memory-candidate-[0-9a-f]{32}$")
    owner_account_id: str
    namespace: Literal["autonomy.mission"] = AUTONOMY_MISSION_NAMESPACE
    tenant_id: str | None = None
    organization_id: str | None = None
    kind: Literal["summary", "preference", "constraint"]
    memory_key: str
    payload: dict[str, Any]
    source_conversation_id: str = Field(min_length=1, max_length=160)
    source_edition: SourceEdition | None = None
    source_kind: MemorySourceKind = "model_inference"
    source_receipt_id: str | None = Field(default=None, min_length=8, max_length=160)
    source_receipt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_verified: bool = Field(default=False, strict=True)
    explicit_reconsent: bool = Field(default=False, strict=True)
    confidence: float = Field(ge=0.0, le=1.0, strict=True)
    observed_at: datetime
    ttl_days: int = Field(ge=1, le=3_650, strict=True)
    status: Literal["pending", "consolidated", "rejected"] = "pending"

    # 功能：
    #   候选入库前检查受治理记忆键，拒绝任意路径或未知类型。
    # 输入：
    #   cls：候选合同类型；value：候选记忆键。
    # 输出：
    #   value：检查通过的键。
    @field_validator("memory_key")
    @classmethod
    def validate_memory_key(cls, value: str) -> str:
        if not _MEMORY_KEY.fullmatch(value):
            raise ValueError("account memory key is invalid")
        return value

    # 功能：
    #   待批准观察也执行完整内容限制，不能因尚未活动而保存秘密或执行指令。
    # 输入：
    #   cls：候选合同类型；value：结构化观察。
    # 输出：
    #   payload：与调用方容器隔离的安全载荷。
    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        payload = _validated_payload(value)
        return payload

    # 功能：
    #   规范化候选采集时间，禁止无时区时间改变证据排序与 TTL。
    # 输入：
    #   cls：候选合同类型；value：候选观察时间。
    # 输出：
    #   timestamp：UTC 时间。
    @field_validator("observed_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ACCOUNT_MEMORY_TIME_INVALID")
        timestamp = value.astimezone(UTC)
        return timestamp

    # 功能：
    #   联合检查范围、类型和来源回执，推断不能声称已核验或携带用户重新同意。
    # 输入：
    #   self：已通过字段验证的候选。
    # 输出：
    #   self：来源声明相互一致的候选；这本身仍不授予提升权限。
    @model_validator(mode="after")
    def validate_scope_and_kind(self) -> SessionMemoryCandidate:
        if not self.memory_key.startswith(f"{self.kind}."):
            raise ValueError("account memory kind and key disagree")
        MemoryOwnerScope(
            owner_account_id=self.owner_account_id,
            namespace=self.namespace,
            tenant_id=self.tenant_id,
            organization_id=self.organization_id,
            source_edition=self.source_edition,
        )
        if self.source_kind == "verified_product_receipt":
            if (
                not self.source_verified
                or self.source_receipt_id is None
                or self.source_receipt_sha256 is None
                or _RECEIPT_ID.fullmatch(self.source_receipt_id) is None
            ):
                raise ValueError("verified product memory requires a bound source receipt")
        elif self.source_receipt_id is not None or self.source_receipt_sha256 is not None:
            raise ValueError("memory source receipt is only valid for verified product evidence")
        if self.source_kind in {"model_inference", "plugin_inference"} and (
            self.source_verified or self.explicit_reconsent
        ):
            raise ValueError("inference memory cannot claim verified user authority")
        if self.explicit_reconsent and self.source_kind != "explicit_user_update":
            raise ValueError("only an explicit user update can restore forgotten memory")
        return self


class PluginMemoryCandidate(BaseModel):
    """Owner-free candidate contract accepted from a memory extraction plugin."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True, hide_input_in_errors=True
    )

    kind: Literal["summary", "preference", "constraint"]
    memory_key: str = Field(min_length=3, max_length=104)
    payload: dict[str, Any]
    confidence: float = Field(default=0.75, ge=0.0, le=0.95, strict=True)
    ttl_days: int = Field(default=90, ge=1, le=365, strict=True)

    # 功能：
    #   将插件提取结果限制在受治理键内，插件不能指定账户或直接写活动记忆。
    # 输入：
    #   cls：插件候选类型；value：插件输出键。
    # 输出：
    #   value：通过约束的键。
    @field_validator("memory_key")
    @classmethod
    def validate_memory_key(cls, value: str) -> str:
        if not _MEMORY_KEY.fullmatch(value):
            raise ValueError("account memory key is invalid")
        return value

    # 功能：
    #   在插件输出边界拒绝秘密、执行指令和不受治理内容。
    # 输入：
    #   cls：插件候选类型；value：插件提取载荷。
    # 输出：
    #   payload：独立的安全载荷。
    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        payload = _validated_payload(value)
        return payload

    # 功能：
    #   验证插件声明类型与记忆键前缀一致。
    # 输入：
    #   self：不携带账户身份的插件候选。
    # 输出：
    #   self：类型一致的候选。
    @model_validator(mode="after")
    def validate_kind(self) -> PluginMemoryCandidate:
        if not self.memory_key.startswith(f"{self.kind}."):
            raise ValueError("account memory kind and key disagree")
        return self


class MemoryCandidateResult(BaseModel):
    """Governance decision for one candidate; it carries no reusable authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(pattern=r"^memory-candidate-[0-9a-f]{32}$")
    memory_key: str
    status: Literal["pending", "consolidated", "rejected"]
    promoted: bool
    conflict_with_active: bool
    independent_evidence_count: int = Field(ge=1)
    decision_reason: str


# 功能：
#   为当前操作系统用户选择跨软件款式共享的记忆库位置，允许明确配置存储根目录。
# 输入：
#   无；读取 DRONEDREAM_SHARED_MEMORY_ROOT 及系统数据目录环境变量。
# 输出：
#   path：规范化的账户记忆 SQLite 文件路径。
def default_account_memory_path() -> Path:
    override = os.environ.get("DRONEDREAM_SHARED_MEMORY_ROOT", "").strip()
    if override:
        root = Path(override).expanduser()
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "DroneDream" / "shared" / "memory"
    else:
        data_home = os.environ.get("XDG_DATA_HOME", "").strip()
        root = (
            Path(data_home).expanduser() / "DroneDream" / "shared" / "memory"
            if data_home
            else Path.home() / ".local" / "share" / "DroneDream" / "shared" / "memory"
        )
    path = root.resolve() / "account-memory.sqlite3"
    return path


class AccountMemoryStore:
    """Cross-process SQLite WAL store with mandatory owner/domain predicates."""

    # 功能：
    #   初始化账户隔离的 WAL 存储，并迁移旧候选字段及重复证据；旧推断不自动获得可信来源。
    # 输入：
    #   self：记忆存储；path：明确归属此软件的数据库路径。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, path: Path) -> None:
        self.path = path
        self._promotion_authority = object()
        self._projection_authority = object()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS governed_account_memory (
                  owner_account_id TEXT NOT NULL,
                  tenant_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  namespace TEXT NOT NULL,
                  memory_key TEXT NOT NULL,
                  kind TEXT NOT NULL CHECK(kind IN ('summary','preference','constraint')),
                  payload_json TEXT NOT NULL,
                  source_conversation_id TEXT NOT NULL,
                  source_edition TEXT,
                  provenance_json TEXT NOT NULL,
                  evidence_count INTEGER NOT NULL,
                  confidence REAL NOT NULL,
                  last_observed_at TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY (
                    owner_account_id, tenant_id, organization_id, namespace, memory_key
                  )
                );
                CREATE INDEX IF NOT EXISTS governed_account_memory_scope_updated_idx
                  ON governed_account_memory(
                    owner_account_id, tenant_id, organization_id, namespace, updated_at DESC
                  );
                CREATE TABLE IF NOT EXISTS account_memory_candidates (
                  candidate_id TEXT PRIMARY KEY,
                  owner_account_id TEXT NOT NULL,
                  tenant_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  namespace TEXT NOT NULL,
                  memory_key TEXT NOT NULL,
                  kind TEXT NOT NULL CHECK(kind IN ('summary','preference','constraint')),
                  payload_json TEXT NOT NULL,
                  source_conversation_id TEXT NOT NULL,
                  source_edition TEXT,
                  confidence REAL NOT NULL,
                  observed_at TEXT NOT NULL,
                  ttl_days INTEGER NOT NULL,
                  trusted_product_evidence INTEGER NOT NULL DEFAULT 0,
                  source_kind TEXT NOT NULL DEFAULT 'model_inference',
                  source_receipt_id TEXT,
                  source_receipt_sha256 TEXT,
                  source_verified INTEGER NOT NULL DEFAULT 0,
                  explicit_reconsent INTEGER NOT NULL DEFAULT 0,
                  conflict_with_active INTEGER NOT NULL DEFAULT 0,
                  decision_reason TEXT NOT NULL DEFAULT 'awaiting_independent_evidence',
                  status TEXT NOT NULL CHECK(status IN ('pending','consolidated','rejected'))
                );
                CREATE INDEX IF NOT EXISTS account_memory_candidates_scope_observed_idx
                  ON account_memory_candidates(
                    owner_account_id, tenant_id, organization_id, namespace, observed_at DESC
                  );
                CREATE TABLE IF NOT EXISTS account_memory_thread_bindings (
                  conversation_id TEXT PRIMARY KEY,
                  owner_account_id TEXT NOT NULL,
                  tenant_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  namespace TEXT NOT NULL,
                  bound_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS account_memory_tombstones (
                  owner_account_id TEXT NOT NULL,
                  tenant_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  namespace TEXT NOT NULL,
                  memory_key TEXT NOT NULL,
                  mode TEXT NOT NULL CHECK(mode IN ('soft','permanent')),
                  reason TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY(
                    owner_account_id,tenant_id,organization_id,namespace,memory_key
                  )
                );
                """
            )
            candidate_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(account_memory_candidates)")
            }
            if "trusted_product_evidence" not in candidate_columns:
                connection.execute(
                    "ALTER TABLE account_memory_candidates "
                    "ADD COLUMN trusted_product_evidence INTEGER NOT NULL DEFAULT 0"
                )
            for column_name, declaration in {
                "source_kind": "TEXT NOT NULL DEFAULT 'model_inference'",
                "source_receipt_id": "TEXT",
                "source_receipt_sha256": "TEXT",
                "source_verified": "INTEGER NOT NULL DEFAULT 0",
                "explicit_reconsent": "INTEGER NOT NULL DEFAULT 0",
            }.items():
                if column_name not in candidate_columns:
                    connection.execute(
                        "ALTER TABLE account_memory_candidates "
                        f"ADD COLUMN {column_name} {declaration}"
                    )
            if "conflict_with_active" not in candidate_columns:
                connection.execute(
                    "ALTER TABLE account_memory_candidates "
                    "ADD COLUMN conflict_with_active INTEGER NOT NULL DEFAULT 0"
                )
            if "decision_reason" not in candidate_columns:
                connection.execute(
                    "ALTER TABLE account_memory_candidates "
                    "ADD COLUMN decision_reason TEXT NOT NULL "
                    "DEFAULT 'awaiting_independent_evidence'"
                )
            # Candidate evidence is conversation-scoped.  Older development
            # databases may contain retries of the same canonical observation;
            # keep the strongest audit row before enforcing write idempotency.
            duplicate_rows = connection.execute(
                """SELECT candidate_id,owner_account_id,tenant_id,organization_id,
                          namespace,memory_key,payload_json,source_conversation_id,status,
                          observed_at
                   FROM account_memory_candidates
                   ORDER BY
                     CASE status
                       WHEN 'consolidated' THEN 0
                       WHEN 'pending' THEN 1
                       ELSE 2
                     END,
                     observed_at ASC,candidate_id ASC"""
            ).fetchall()
            seen_evidence: set[tuple[str, str, str, str, str, str, str]] = set()
            for row in duplicate_rows:
                evidence_key = (
                    str(row["owner_account_id"]),
                    str(row["tenant_id"]),
                    str(row["organization_id"]),
                    str(row["namespace"]),
                    str(row["memory_key"]),
                    str(row["payload_json"]),
                    str(row["source_conversation_id"]),
                )
                if evidence_key in seen_evidence:
                    connection.execute(
                        "DELETE FROM account_memory_candidates WHERE candidate_id=?",
                        (str(row["candidate_id"]),),
                    )
                else:
                    seen_evidence.add(evidence_key)
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS
                     account_memory_candidates_evidence_unique_idx
                   ON account_memory_candidates(
                     owner_account_id,tenant_id,organization_id,namespace,
                     memory_key,payload_json,source_conversation_id
                   )"""
            )

    # 功能：
    #   持有一次短事务，异常回滚，正常退出提交，并关闭 SQLite 而非依赖垃圾回收。
    # 输入：
    #   self：保存数据库路径的存储实例。
    # 输出：
    #   connection：仅在 with 范围有效的连接。
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=15000")
            with connection:
                yield connection
        finally:
            connection.close()

    # 功能：
    #   为 SQL 生成验证过的账户绑定值，禁止将账户文本拼接成查询语句。
    # 输入：
    #   scope：当前操作的完整账户边界。
    # 输出：
    #   boundary：账户、租户、组织及职责域四元组。
    @staticmethod
    def _scope_values(scope: MemoryOwnerScope) -> tuple[str, str, str, str]:
        boundary = scope.boundary_key()
        return boundary

    # 功能：
    #   1. 保存产品明确核验的用户更新；普通模型推断必须走候选入口。
    #   2. 不覆盖不同的活动内容，冲突必须由显式候选解决操作处理。
    # 输入：
    #   self：存储；scope：完整账户范围；kind：记忆类型；memory_key：记忆键。
    #   payload：结构化内容；source_conversation_id：来源会话；confidence：核验置信度。
    #   ttl_days：有效天数；explicit_reconsent：是否明确允许重学曾删除内容。
    # 输出：
    #   entry：成功提升后的活动记忆条目。
    def upsert(
        self,
        scope: MemoryOwnerScope,
        *,
        kind: Literal["summary", "preference", "constraint"],
        memory_key: str,
        payload: dict[str, Any],
        source_conversation_id: str,
        confidence: float = 0.99,
        ttl_days: int = 365,
        explicit_reconsent: bool = False,
    ) -> GovernedMemoryEntry:
        result = self._record_candidate(
            scope,
            kind=kind,
            memory_key=memory_key,
            payload=payload,
            source_conversation_id=source_conversation_id,
            confidence=confidence,
            ttl_days=ttl_days,
            source_kind="explicit_user_update",
            source_verified=True,
            explicit_reconsent=explicit_reconsent,
            promotion_authority=self._promotion_authority,
        )
        if not result.promoted:
            raise ValueError(f"ACCOUNT_MEMORY_CONFLICT_REQUIRES_RESOLUTION:{result.candidate_id}")
        entry = self.get(scope, memory_key)
        return entry

    # 功能：
    #   原始会话只能绑定一个完整账户范围；重复绑定同范围幂等，其他范围明确拒绝。
    # 输入：
    #   self：存储；scope：待绑定范围；conversation_id：原始会话标识。
    # 输出：
    #   None：不返回业务数据。
    def bind_thread(self, scope: MemoryOwnerScope, conversation_id: str) -> None:
        if (
            not isinstance(conversation_id, str)
            or not conversation_id.strip()
            or len(conversation_id) > 160
            or "\x00" in conversation_id
        ):
            raise ValueError("ACCOUNT_MEMORY_CONVERSATION_ID_INVALID")
        owner, tenant, organization, namespace = self._scope_values(scope)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT OR IGNORE INTO account_memory_thread_bindings(
                     conversation_id,owner_account_id,tenant_id,organization_id,namespace,bound_at
                   ) VALUES(?,?,?,?,?,?)""",
                (conversation_id, owner, tenant, organization, namespace, _now().isoformat()),
            )
            row = connection.execute(
                """SELECT owner_account_id,tenant_id,organization_id,namespace
                   FROM account_memory_thread_bindings WHERE conversation_id=?""",
                (conversation_id,),
            ).fetchone()
        if row is None or (
            str(row["owner_account_id"]),
            str(row["tenant_id"]),
            str(row["organization_id"]),
            str(row["namespace"]),
        ) != (owner, tenant, organization, namespace):
            raise ValueError("ACCOUNT_MEMORY_THREAD_OWNER_MISMATCH")

    # 功能：
    #   只核验已有会话归属，不因一次读取要求而创建新绑定。
    # 输入：
    #   self：存储；scope：要求的范围；conversation_id：已有会话标识。
    # 输出：
    #   None：匹配时继续，缺失或跨账户时抛错。
    def require_thread_binding(self, scope: MemoryOwnerScope, conversation_id: str) -> None:
        if (
            not isinstance(conversation_id, str)
            or not conversation_id.strip()
            or len(conversation_id) > 160
            or "\x00" in conversation_id
        ):
            raise ValueError("ACCOUNT_MEMORY_CONVERSATION_ID_INVALID")
        owner, tenant, organization, namespace = self._scope_values(scope)
        with self._connect() as connection:
            row = connection.execute(
                """SELECT owner_account_id,tenant_id,organization_id,namespace
                   FROM account_memory_thread_bindings WHERE conversation_id=?""",
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        if (
            str(row["owner_account_id"]),
            str(row["tenant_id"]),
            str(row["organization_id"]),
            str(row["namespace"]),
        ) != (owner, tenant, organization, namespace):
            raise ValueError("ACCOUNT_MEMORY_THREAD_OWNER_MISMATCH")

    # 功能：
    #   保存模型或插件推断为待批准候选，无论重复多少次都不能自行提升为活动记忆。
    # 输入：
    #   self：存储；scope：账户范围；kind、memory_key：受治理记忆类型与键。
    #   payload：结构化推断；source_conversation_id：来源会话。
    #   confidence：推断置信度；ttl_days：有效天数；source_kind：模型或插件来源。
    # 输出：
    #   result：候选标识、状态、独立证据数及治理原因。
    def record_candidate(
        self,
        scope: MemoryOwnerScope,
        *,
        kind: Literal["summary", "preference", "constraint"],
        memory_key: str,
        payload: dict[str, Any],
        source_conversation_id: str,
        confidence: float,
        ttl_days: int,
        source_kind: Literal["model_inference", "plugin_inference"] = "model_inference",
    ) -> MemoryCandidateResult:
        result = self._record_candidate(
            scope,
            kind=kind,
            memory_key=memory_key,
            payload=payload,
            source_conversation_id=source_conversation_id,
            confidence=confidence,
            ttl_days=ttl_days,
            source_kind=source_kind,
            source_verified=False,
            explicit_reconsent=False,
            promotion_authority=None,
        )
        return result

    # 功能：
    #   保存由固定产品边界事先验证的回执，不能仅凭模型声称可信就调用此入口。
    # 输入：
    #   self：存储；scope：账户范围；kind、memory_key：记忆类型与键。
    #   payload：已核验内容；source_conversation_id：来源会话；confidence：核验置信度。
    #   ttl_days：有效期；source_receipt_id、source_receipt_sha256：绑定的产品回执。
    # 输出：
    #   result：候选提升或冲突的治理结果。
    def record_verified_product_candidate(
        self,
        scope: MemoryOwnerScope,
        *,
        kind: Literal["summary", "preference", "constraint"],
        memory_key: str,
        payload: dict[str, Any],
        source_conversation_id: str,
        confidence: float,
        ttl_days: int,
        source_receipt_id: str,
        source_receipt_sha256: str,
    ) -> MemoryCandidateResult:
        result = self._record_candidate(
            scope,
            kind=kind,
            memory_key=memory_key,
            payload=payload,
            source_conversation_id=source_conversation_id,
            confidence=confidence,
            ttl_days=ttl_days,
            source_kind="verified_product_receipt",
            source_receipt_id=source_receipt_id,
            source_receipt_sha256=source_receipt_sha256,
            source_verified=True,
            explicit_reconsent=False,
            promotion_authority=self._promotion_authority,
        )
        return result

    # 功能：
    #   1. 验证来源、载荷和会话归属后按独立会话去重，删除标记只能由明确重新同意释放。
    #   2. 只有固定包装器提供的用户更新或核验回执可提升，且不自动覆盖活动冲突。
    #   3. 同会话先推断后批准可更新原候选来源，不重复计数；推断本身不能走此提升分支。
    # 输入：
    #   self：存储；scope：账户范围；kind、memory_key：记忆类型及键；payload：内容。
    #   source_conversation_id：来源会话；confidence：置信度；ttl_days：有效天数。
    #   source_kind、source_verified：来源类别及核验结果；explicit_reconsent：明确重新同意。
    #   promotion_authority：固定包装器的私有提升能力对象。
    #   source_receipt_id、source_receipt_sha256：可选产品回执绑定。
    # 输出：
    #   result：候选状态、是否提升、冲突与独立证据数。
    def _record_candidate(
        self,
        scope: MemoryOwnerScope,
        *,
        kind: Literal["summary", "preference", "constraint"],
        memory_key: str,
        payload: dict[str, Any],
        source_conversation_id: str,
        confidence: float,
        ttl_days: int,
        source_kind: MemorySourceKind,
        source_verified: bool,
        explicit_reconsent: bool,
        promotion_authority: object | None,
        source_receipt_id: str | None = None,
        source_receipt_sha256: str | None = None,
    ) -> MemoryCandidateResult:
        self._scope_values(scope)
        if type(confidence) not in (int, float) or not 0.0 <= confidence <= 1.0:
            raise ValueError("ACCOUNT_MEMORY_CONFIDENCE_INVALID")
        if type(source_verified) is not bool or type(explicit_reconsent) is not bool:
            raise ValueError("ACCOUNT_MEMORY_SOURCE_FLAGS_INVALID")
        if scope.namespace != AUTONOMY_MISSION_NAMESPACE:
            raise ValueError("ACCOUNT_MEMORY_WRITABLE_NAMESPACE_FORBIDDEN")
        if source_kind in _PROMOTABLE_SOURCE_KINDS:
            if promotion_authority is not self._promotion_authority or not source_verified:
                raise ValueError("ACCOUNT_MEMORY_PROMOTION_SOURCE_NOT_VERIFIED")
            if confidence < 0.95:
                raise ValueError("ACCOUNT_MEMORY_VERIFIED_SOURCE_CONFIDENCE_TOO_LOW")
        elif promotion_authority is not None or source_verified:
            raise ValueError("ACCOUNT_MEMORY_INFERENCE_AUTHORITY_FORBIDDEN")
        observed_at = _now()
        candidate = SessionMemoryCandidate(
            candidate_id=f"memory-candidate-{uuid4().hex}",
            owner_account_id=scope.owner_account_id,
            namespace=scope.namespace,
            tenant_id=scope.tenant_id,
            organization_id=scope.organization_id,
            kind=kind,
            memory_key=memory_key,
            payload=payload,
            source_conversation_id=source_conversation_id,
            source_edition=scope.source_edition,
            source_kind=source_kind,
            source_receipt_id=source_receipt_id,
            source_receipt_sha256=source_receipt_sha256,
            source_verified=source_verified,
            explicit_reconsent=explicit_reconsent,
            confidence=confidence,
            observed_at=observed_at,
            ttl_days=ttl_days,
        )
        # Invalid extraction must not reserve a conversation for an account.
        # Validate content and provenance before any durable thread binding.
        self.bind_thread(scope, candidate.source_conversation_id)
        owner, tenant, organization, namespace = self._scope_values(scope)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            canonical_payload = _canonical_payload(candidate.payload)
            tombstone = connection.execute(
                """SELECT mode FROM account_memory_tombstones
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                     AND namespace=? AND memory_key=?""",
                (owner, tenant, organization, namespace, candidate.memory_key),
            ).fetchone()
            if tombstone is not None:
                if not (
                    candidate.source_kind == "explicit_user_update"
                    and candidate.source_verified
                    and candidate.explicit_reconsent
                ):
                    raise ValueError("ACCOUNT_MEMORY_TOMBSTONE_BLOCKS_RELEARNING")
                connection.execute(
                    """DELETE FROM account_memory_tombstones
                       WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                         AND namespace=? AND memory_key=?""",
                    (owner, tenant, organization, namespace, candidate.memory_key),
                )
                # A new explicit consent starts a new evidence epoch. Old
                # candidates must not satisfy the new epoch's unique evidence key.
                connection.execute(
                    """DELETE FROM account_memory_candidates
                       WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                         AND namespace=? AND memory_key=?""",
                    (owner, tenant, organization, namespace, candidate.memory_key),
                )
            duplicate = connection.execute(
                """SELECT * FROM account_memory_candidates
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                     AND namespace=? AND memory_key=? AND payload_json=?
                     AND source_conversation_id=?""",
                (
                    owner,
                    tenant,
                    organization,
                    namespace,
                    candidate.memory_key,
                    canonical_payload,
                    candidate.source_conversation_id,
                ),
            ).fetchone()
            reuse_pending = bool(
                duplicate is not None
                and duplicate["status"] == "pending"
                and duplicate["source_verified"] == 0
                and candidate.source_verified
                and candidate.source_kind in _PROMOTABLE_SOURCE_KINDS
            )
            if reuse_pending:
                # 这是固定用户/产品入口带来的新核验，不是重复模型推断带来的“投票”。
                candidate = candidate.model_copy(update={"candidate_id": duplicate["candidate_id"]})
            if duplicate is not None and not reuse_pending:
                candidate_rows = connection.execute(
                    """SELECT * FROM account_memory_candidates
                       WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                         AND namespace=? AND memory_key=?
                         AND status IN ('pending','consolidated')
                       ORDER BY observed_at ASC,candidate_id ASC""",
                    (owner, tenant, organization, namespace, candidate.memory_key),
                ).fetchall()
                matching_rows = self._matching_candidate_rows(candidate_rows, candidate.payload)
                status_value = str(duplicate["status"])
                return MemoryCandidateResult(
                    candidate_id=str(duplicate["candidate_id"]),
                    memory_key=candidate.memory_key,
                    status=status_value,
                    promoted=status_value == "consolidated",
                    conflict_with_active=bool(duplicate["conflict_with_active"]),
                    independent_evidence_count=max(
                        1, len(self._independent_candidate_rows(matching_rows))
                    ),
                    decision_reason=str(duplicate["decision_reason"]),
                )
            connection.execute(
                """INSERT INTO account_memory_candidates(
                     candidate_id,owner_account_id,tenant_id,organization_id,namespace,
                     memory_key,kind,payload_json,source_conversation_id,source_edition,
                     confidence,observed_at,ttl_days,trusted_product_evidence,
                     source_kind,source_receipt_id,source_receipt_sha256,
                     source_verified,explicit_reconsent,
                     conflict_with_active,decision_reason,status
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(candidate_id) DO UPDATE SET
                     confidence=excluded.confidence,observed_at=excluded.observed_at,
                     ttl_days=excluded.ttl_days,source_edition=excluded.source_edition,
                     source_kind=excluded.source_kind,source_receipt_id=excluded.source_receipt_id,
                     source_receipt_sha256=excluded.source_receipt_sha256,
                     source_verified=excluded.source_verified,
                     explicit_reconsent=excluded.explicit_reconsent""",
                (
                    candidate.candidate_id,
                    owner,
                    tenant,
                    organization,
                    namespace,
                    candidate.memory_key,
                    candidate.kind,
                    canonical_payload,
                    candidate.source_conversation_id,
                    candidate.source_edition,
                    candidate.confidence,
                    candidate.observed_at.isoformat(),
                    candidate.ttl_days,
                    0,
                    candidate.source_kind,
                    candidate.source_receipt_id,
                    candidate.source_receipt_sha256,
                    int(candidate.source_verified),
                    int(candidate.explicit_reconsent),
                    0,
                    "awaiting_independent_evidence",
                    "pending",
                ),
            )
            existing = connection.execute(
                """SELECT * FROM governed_account_memory
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                      AND namespace=? AND memory_key=? AND status='active'
                      AND expires_at>?""",
                (
                    owner,
                    tenant,
                    organization,
                    namespace,
                    candidate.memory_key,
                    observed_at.isoformat(),
                ),
            ).fetchone()
            candidate_rows = connection.execute(
                """SELECT * FROM account_memory_candidates
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                     AND namespace=? AND memory_key=?
                     AND status IN ('pending','consolidated')
                   ORDER BY observed_at ASC, candidate_id ASC""",
                (owner, tenant, organization, namespace, candidate.memory_key),
            ).fetchall()
            matching_rows = self._matching_candidate_rows(candidate_rows, candidate.payload)
            independent_rows = self._independent_candidate_rows(matching_rows)
            independent_evidence_count = len(independent_rows)
            existing_payload = (
                _validated_payload(json.loads(str(existing["payload_json"])))
                if existing is not None
                else None
            )
            if existing_payload is not None and existing_payload != candidate.payload:
                connection.execute(
                    """UPDATE account_memory_candidates
                       SET conflict_with_active=1,
                           decision_reason='active_payload_conflict'
                       WHERE candidate_id=?""",
                    (candidate.candidate_id,),
                )
                return MemoryCandidateResult(
                    candidate_id=candidate.candidate_id,
                    memory_key=candidate.memory_key,
                    status="pending",
                    promoted=False,
                    conflict_with_active=True,
                    independent_evidence_count=independent_evidence_count,
                    decision_reason="active_payload_conflict",
                )

            if existing_payload == candidate.payload and existing is not None:
                if not (
                    candidate.source_verified and candidate.source_kind in _PROMOTABLE_SOURCE_KINDS
                ):
                    connection.execute(
                        """UPDATE account_memory_candidates
                           SET decision_reason='inference_cannot_reinforce_active'
                           WHERE candidate_id=?""",
                        (candidate.candidate_id,),
                    )
                    return MemoryCandidateResult(
                        candidate_id=candidate.candidate_id,
                        memory_key=candidate.memory_key,
                        status="pending",
                        promoted=False,
                        conflict_with_active=False,
                        independent_evidence_count=independent_evidence_count,
                        decision_reason="inference_cannot_reinforce_active",
                    )
                provenance = json.loads(str(existing["provenance_json"]))
                if not isinstance(provenance, list):
                    provenance = []
                seen_conversations = {
                    str(item.get("source_conversation_id"))
                    for item in provenance
                    if isinstance(item, dict) and item.get("source_conversation_id")
                }
                if candidate.source_conversation_id in seen_conversations:
                    connection.execute(
                        """UPDATE account_memory_candidates
                           SET status='rejected',
                               decision_reason='duplicate_conversation_evidence'
                           WHERE candidate_id=?""",
                        (candidate.candidate_id,),
                    )
                    return MemoryCandidateResult(
                        candidate_id=candidate.candidate_id,
                        memory_key=candidate.memory_key,
                        status="rejected",
                        promoted=False,
                        conflict_with_active=False,
                        independent_evidence_count=max(
                            int(existing["evidence_count"]), independent_evidence_count
                        ),
                        decision_reason="duplicate_conversation_evidence",
                    )
                provenance.append(self._provenance(candidate))
                evidence_count = max(
                    int(existing["evidence_count"]) + 1,
                    independent_evidence_count,
                )
                previous_confidence = float(existing["confidence"])
                consolidated_confidence = min(
                    0.99,
                    previous_confidence + (1.0 - previous_confidence) * candidate.confidence * 0.25,
                )
                expires_at = max(
                    datetime.fromisoformat(str(existing["expires_at"])),
                    candidate.observed_at + timedelta(days=candidate.ttl_days),
                )
                connection.execute(
                    """UPDATE governed_account_memory SET
                         source_conversation_id=?,source_edition=?,provenance_json=?,
                         evidence_count=?,confidence=?,last_observed_at=?,expires_at=?,
                         updated_at=?
                       WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                         AND namespace=? AND memory_key=?""",
                    (
                        candidate.source_conversation_id,
                        candidate.source_edition,
                        json.dumps(provenance[-16:], ensure_ascii=False, sort_keys=True),
                        evidence_count,
                        consolidated_confidence,
                        candidate.observed_at.isoformat(),
                        expires_at.isoformat(),
                        candidate.observed_at.isoformat(),
                        owner,
                        tenant,
                        organization,
                        namespace,
                        candidate.memory_key,
                    ),
                )
                connection.execute(
                    """UPDATE account_memory_candidates
                       SET status='consolidated',decision_reason='active_payload_reinforced'
                       WHERE candidate_id=?""",
                    (candidate.candidate_id,),
                )
                return MemoryCandidateResult(
                    candidate_id=candidate.candidate_id,
                    memory_key=candidate.memory_key,
                    status="consolidated",
                    promoted=True,
                    conflict_with_active=False,
                    independent_evidence_count=evidence_count,
                    decision_reason="active_payload_reinforced",
                )

            verified_rows = [
                row
                for row in independent_rows
                if bool(row["source_verified"])
                and str(row["source_kind"]) in _PROMOTABLE_SOURCE_KINDS
            ]
            if not verified_rows:
                connection.execute(
                    """UPDATE account_memory_candidates
                       SET decision_reason='inference_requires_explicit_promotion'
                       WHERE candidate_id=?""",
                    (candidate.candidate_id,),
                )
                return MemoryCandidateResult(
                    candidate_id=candidate.candidate_id,
                    memory_key=candidate.memory_key,
                    status="pending",
                    promoted=False,
                    conflict_with_active=False,
                    independent_evidence_count=independent_evidence_count,
                    decision_reason="inference_requires_explicit_promotion",
                )
            self._promote_rows(
                connection,
                scope,
                candidate.memory_key,
                candidate.kind,
                candidate.payload,
                verified_rows,
                decision_reason="verified_source_promoted",
            )
            return MemoryCandidateResult(
                candidate_id=candidate.candidate_id,
                memory_key=candidate.memory_key,
                status="consolidated",
                promoted=True,
                conflict_with_active=False,
                independent_evidence_count=len(verified_rows),
                decision_reason="verified_source_promoted",
            )

    # 功能：
    #   在单事务中显式批准或拒绝待处理候选；批准冲突是用户决定，不是模型自动覆盖。
    # 输入：
    #   self：存储；scope：账户范围；candidate_id：待处理候选标识。
    #   approve：明确批准/拒绝；explicit_reconsent：是否允许释放同键删除标记。
    # 输出：
    #   entry：批准后的活动记忆；拒绝时为 None。
    def resolve_candidate(
        self,
        scope: MemoryOwnerScope,
        candidate_id: str,
        *,
        approve: bool,
        explicit_reconsent: bool = False,
    ) -> GovernedMemoryEntry | None:
        if not re.fullmatch(r"memory-candidate-[0-9a-f]{32}", candidate_id):
            raise ValueError("ACCOUNT_MEMORY_CANDIDATE_ID_INVALID")
        if not isinstance(approve, bool):
            raise ValueError("ACCOUNT_MEMORY_CANDIDATE_DECISION_INVALID")
        if not isinstance(explicit_reconsent, bool):
            raise ValueError("ACCOUNT_MEMORY_RECONSENT_INVALID")
        owner, tenant, organization, namespace = self._scope_values(scope)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT * FROM account_memory_candidates
                   WHERE candidate_id=? AND owner_account_id=? AND tenant_id=?
                     AND organization_id=? AND namespace=?""",
                (candidate_id, owner, tenant, organization, namespace),
            ).fetchone()
            if row is None:
                raise KeyError(candidate_id)
            if str(row["status"]) != "pending":
                raise ValueError("ACCOUNT_MEMORY_CANDIDATE_NOT_PENDING")
            if not approve:
                connection.execute(
                    """UPDATE account_memory_candidates
                       SET status='rejected',decision_reason='explicitly_rejected'
                       WHERE candidate_id=?""",
                    (candidate_id,),
                )
                return None
            tombstone = connection.execute(
                """SELECT mode FROM account_memory_tombstones
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                     AND namespace=? AND memory_key=?""",
                (owner, tenant, organization, namespace, str(row["memory_key"])),
            ).fetchone()
            if tombstone is not None:
                if not explicit_reconsent:
                    raise ValueError("ACCOUNT_MEMORY_TOMBSTONE_REQUIRES_RECONSENT")
                connection.execute(
                    """DELETE FROM account_memory_tombstones
                       WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                         AND namespace=? AND memory_key=?""",
                    (owner, tenant, organization, namespace, str(row["memory_key"])),
                )
            payload = _validated_payload(json.loads(str(row["payload_json"])))
            matching = self._matching_candidate_rows(
                connection.execute(
                    """SELECT * FROM account_memory_candidates
                       WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                         AND namespace=? AND memory_key=?
                         AND status IN ('pending','consolidated')
                       ORDER BY observed_at ASC,candidate_id ASC""",
                    (owner, tenant, organization, namespace, str(row["memory_key"])),
                ).fetchall(),
                payload,
            )
            independent = self._independent_candidate_rows(matching)
            self._promote_rows(
                connection,
                scope,
                str(row["memory_key"]),
                str(row["kind"]),
                payload,
                independent,
                decision_reason="explicitly_approved",
            )
            matching_ids = {str(item["candidate_id"]) for item in matching}
            pending_rows = connection.execute(
                """SELECT candidate_id FROM account_memory_candidates
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                     AND namespace=? AND memory_key=? AND status='pending'""",
                (owner, tenant, organization, namespace, str(row["memory_key"])),
            ).fetchall()
            for pending in pending_rows:
                pending_id = str(pending["candidate_id"])
                if pending_id not in matching_ids:
                    connection.execute(
                        """UPDATE account_memory_candidates
                           SET conflict_with_active=1,
                               decision_reason='active_payload_conflict'
                           WHERE candidate_id=?""",
                        (pending_id,),
                    )
        return self.get(scope, str(row["memory_key"]))

    # 功能：
    #   通过统一的范围隔离事务拒绝候选，保留拒绝记录而非直接删除。
    # 输入：
    #   self：存储；scope：账户范围；candidate_id：候选标识。
    # 输出：
    #   None：不返回业务数据。
    def reject_candidate(self, scope: MemoryOwnerScope, candidate_id: str) -> None:
        self.resolve_candidate(scope, candidate_id, approve=False)

    # 功能：
    #   列出指定范围的安全候选，活动冲突以显示状态 conflict 表达但保留原存储状态。
    # 输入：
    #   self：存储；scope：账户范围；memory_key：可选记忆键筛选。
    #   limit：列表上限；candidate_id：可选精确标识，不受最近条目窗口影响。
    # 输出：
    #   result：包含来源回执和治理状态的候选资料列表。
    def list_candidates(
        self,
        scope: MemoryOwnerScope,
        *,
        memory_key: str | None = None,
        limit: int = 100,
        candidate_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if memory_key is not None and not _MEMORY_KEY.fullmatch(memory_key):
            raise ValueError("ACCOUNT_MEMORY_KEY_INVALID")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
            raise ValueError("ACCOUNT_MEMORY_CANDIDATE_LIMIT_INVALID")
        owner, tenant, organization, namespace = self._scope_values(scope)
        predicate = "owner_account_id=? AND tenant_id=? AND organization_id=? AND namespace=?"
        values: list[Any] = [owner, tenant, organization, namespace]
        if memory_key is not None:
            predicate += " AND memory_key=?"
            values.append(memory_key)
        if candidate_id is not None:
            if (
                not isinstance(candidate_id, str)
                or re.fullmatch(r"memory-candidate-[0-9a-f]{32}", candidate_id) is None
            ):
                raise ValueError("ACCOUNT_MEMORY_CANDIDATE_ID_INVALID")
            predicate += " AND candidate_id=?"
            values.append(candidate_id)
        values.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT * FROM account_memory_candidates
                    WHERE {predicate}
                    ORDER BY observed_at DESC,candidate_id ASC LIMIT ?""",  # noqa: S608
                values,
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                try:
                    candidate = self._candidate_row(row)
                    payload = candidate.payload
                except (json.JSONDecodeError, TypeError, ValueError):
                    # Legacy/corrupt unsafe rows are never surfaced to product
                    # code and therefore cannot be injected into model context.
                    continue
                evidence_count = connection.execute(
                    """SELECT COUNT(DISTINCT source_conversation_id)
                       FROM account_memory_candidates
                       WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                         AND namespace=? AND memory_key=? AND payload_json=?
                         AND status IN ('pending','consolidated')""",
                    (
                        owner,
                        tenant,
                        organization,
                        namespace,
                        str(row["memory_key"]),
                        str(row["payload_json"]),
                    ),
                ).fetchone()[0]
                storage_status = str(row["status"])
                conflict = bool(row["conflict_with_active"])
                result.append(
                    {
                        "candidate_id": str(row["candidate_id"]),
                        "kind": str(row["kind"]),
                        "memory_key": str(row["memory_key"]),
                        "payload": payload,
                        "source_conversation_id": str(row["source_conversation_id"]),
                        "source_edition": (
                            str(row["source_edition"])
                            if row["source_edition"] is not None
                            else None
                        ),
                        "source_kind": str(row["source_kind"]),
                        "source_receipt_id": (
                            str(row["source_receipt_id"])
                            if row["source_receipt_id"] is not None
                            else None
                        ),
                        "source_receipt_sha256": (
                            str(row["source_receipt_sha256"])
                            if row["source_receipt_sha256"] is not None
                            else None
                        ),
                        "source_verified": bool(row["source_verified"]),
                        "explicit_reconsent": bool(row["explicit_reconsent"]),
                        "confidence": float(row["confidence"]),
                        "observed_at": str(row["observed_at"]),
                        "ttl_days": int(row["ttl_days"]),
                        "status": (
                            "conflict"
                            if storage_status == "pending" and conflict
                            else storage_status
                        ),
                        "storage_status": storage_status,
                        "conflict_with_active": conflict,
                        "independent_evidence_count": max(1, int(evidence_count)),
                        "decision_reason": str(row["decision_reason"]),
                    }
                )
        return result

    # 功能：
    #   为固定同步桥查找一个经过净化的精确候选，不能因候选较旧就落出最近 500 条窗口。
    # 输入：
    #   self：存储；scope：完整账户边界；candidate_id：目标候选标识。
    # 输出：
    #   candidate：净化后的候选资料，不携带提升或执行权限。
    def projection_candidate(
        self,
        scope: MemoryOwnerScope,
        candidate_id: str,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"memory-candidate-[0-9a-f]{32}", candidate_id):
            raise ValueError("ACCOUNT_MEMORY_CANDIDATE_ID_INVALID")
        candidates = self.list_candidates(scope, limit=1, candidate_id=candidate_id)
        if candidates:
            candidate = candidates[0]
            return candidate
        raise KeyError(candidate_id)

    # 功能：
    #   合并活动条目、候选与删除标记中的键，使远端删除能同时覆盖三类本地状态。
    # 输入：
    #   self：存储；scope：完整账户范围。
    # 输出：
    #   keys：去重排序的记忆键，不包含任何记忆载荷。
    def projection_keys(self, scope: MemoryOwnerScope) -> list[str]:
        owner, tenant, organization, namespace = self._scope_values(scope)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT memory_key FROM governed_account_memory
                     WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                       AND namespace=?
                   UNION
                   SELECT memory_key FROM account_memory_candidates
                     WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                       AND namespace=?
                   UNION
                   SELECT memory_key FROM account_memory_tombstones
                     WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                       AND namespace=?
                   ORDER BY memory_key""",
                (
                    owner,
                    tenant,
                    organization,
                    namespace,
                    owner,
                    tenant,
                    organization,
                    namespace,
                    owner,
                    tenant,
                    organization,
                    namespace,
                ),
            ).fetchall()
        return [str(row["memory_key"]) for row in rows]

    # 功能：
    #   1. 仅允许固定云端同步桥导入，远端内容也必须通过本地记忆安全合同。
    #   2. 不覆盖不同的活动记忆、不释放本地删除标记；已过期本地条目不阻止新的有效投影。
    # 输入：
    #   self：存储；scope：账户范围；memory_key、payload：远端键与结构化内容。
    #   source_conversation_id、source_edition：来源；evidence_count、confidence：证据统计。
    #   last_observed_at、expires_at：带时区的观察时间与过期时间。
    #   remote_memory_id、remote_payload_sha256、remote_revision：远端标识、服务端摘要及修订。
    #   projection_authority：固定桥提供的私有能力对象。
    # 输出：
    #   outcome：imported、unchanged、conflict 或 tombstoned。
    def import_projected_entry(
        self,
        scope: MemoryOwnerScope,
        *,
        memory_key: str,
        payload: dict[str, Any],
        source_conversation_id: str,
        source_edition: SourceEdition | None,
        evidence_count: int,
        confidence: float,
        last_observed_at: datetime,
        expires_at: datetime,
        remote_memory_id: str,
        remote_payload_sha256: str,
        remote_revision: int,
        projection_authority: object,
    ) -> Literal["imported", "unchanged", "conflict", "tombstoned"]:
        if projection_authority is not self._projection_authority:
            raise ValueError("ACCOUNT_MEMORY_PROJECTION_AUTHORITY_REQUIRED")
        kind = memory_key.split(".", 1)[0]
        if kind not in {"summary", "preference", "constraint"}:
            raise ValueError("ACCOUNT_MEMORY_PROJECTED_KIND_INVALID")
        if not re.fullmatch(r"[0-9a-f]{64}", remote_payload_sha256):
            raise ValueError("ACCOUNT_MEMORY_PROJECTED_HASH_INVALID")
        if (
            type(remote_revision) is not int
            or not 1 <= remote_revision <= 9_223_372_036_854_775_807
        ):
            raise ValueError("ACCOUNT_MEMORY_PROJECTED_REVISION_INVALID")
        if (
            not isinstance(remote_memory_id, str)
            or len(remote_memory_id) != 36
            or str(UUID(remote_memory_id)) != remote_memory_id
        ):
            raise ValueError("ACCOUNT_MEMORY_PROJECTED_ID_INVALID")
        now = _now()
        entry = GovernedMemoryEntry(
            owner_account_id=scope.owner_account_id,
            namespace=scope.namespace,
            tenant_id=scope.tenant_id,
            organization_id=scope.organization_id,
            kind=kind,
            memory_key=memory_key,
            payload=payload,
            source_conversation_id=source_conversation_id,
            source_edition=source_edition,
            provenance=[
                {
                    "source_kind": "supabase_projection",
                    "remote_memory_id": remote_memory_id,
                    "remote_payload_sha256": remote_payload_sha256,
                    "remote_revision": str(remote_revision),
                }
            ],
            evidence_count=evidence_count,
            confidence=confidence,
            last_observed_at=last_observed_at,
            expires_at=expires_at,
            status="active",
            created_at=now,
            updated_at=now,
        )
        if entry.expires_at <= now:
            raise ValueError("ACCOUNT_MEMORY_PROJECTED_ENTRY_EXPIRED")
        owner, tenant, organization, namespace = self._scope_values(scope)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            tombstone = connection.execute(
                """SELECT 1 FROM account_memory_tombstones
                     WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                       AND namespace=? AND memory_key=?""",
                (owner, tenant, organization, namespace, memory_key),
            ).fetchone()
            if tombstone is not None:
                return "tombstoned"
            existing = connection.execute(
                """SELECT payload_json,status,expires_at FROM governed_account_memory
                     WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                       AND namespace=? AND memory_key=?""",
                (owner, tenant, organization, namespace, memory_key),
            ).fetchone()
            canonical_payload = _canonical_payload(entry.payload)
            if existing is not None and str(existing["status"]) == "active":
                try:
                    existing_expiry = datetime.fromisoformat(existing["expires_at"])
                    if existing_expiry.tzinfo is None:
                        raise ValueError("ACCOUNT_MEMORY_TIME_INVALID")
                    existing_payload = _canonical_payload(
                        _validated_payload(decode_json(existing["payload_json"], limit=32 * 1024))
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    connection.execute(
                        """UPDATE governed_account_memory SET status='quarantined'
                             WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                               AND namespace=? AND memory_key=?""",
                        (owner, tenant, organization, namespace, memory_key),
                    )
                    return "conflict"
                if existing_expiry > now:
                    if existing_payload != canonical_payload:
                        return "conflict"
                    return "unchanged"
            connection.execute(
                """INSERT INTO governed_account_memory(
                     owner_account_id,tenant_id,organization_id,namespace,memory_key,kind,
                     payload_json,source_conversation_id,source_edition,provenance_json,
                     evidence_count,confidence,last_observed_at,expires_at,status,
                     created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(owner_account_id,tenant_id,organization_id,namespace,memory_key)
                   DO UPDATE SET
                     kind=excluded.kind,payload_json=excluded.payload_json,
                     source_conversation_id=excluded.source_conversation_id,
                     source_edition=excluded.source_edition,
                     provenance_json=excluded.provenance_json,
                     evidence_count=excluded.evidence_count,confidence=excluded.confidence,
                     last_observed_at=excluded.last_observed_at,
                     expires_at=excluded.expires_at,status='active',
                     updated_at=excluded.updated_at""",
                (
                    owner,
                    tenant,
                    organization,
                    namespace,
                    memory_key,
                    entry.kind,
                    canonical_payload,
                    source_conversation_id,
                    source_edition,
                    json.dumps(entry.provenance, ensure_ascii=False, sort_keys=True),
                    evidence_count,
                    confidence,
                    entry.last_observed_at.isoformat(),
                    entry.expires_at.isoformat(),
                    "active",
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        return "imported"

    # 功能：
    #   撤回或永久删除一个键的内容与候选，同时保留无载荷标记阻止自动重学。
    # 输入：
    #   self：存储；scope：账户范围；memory_key：目标键。
    #   mode：软撤回或永久删除；reason：不含秘密的操作原因标识。
    # 输出：
    #   result：确实撤回/删除的数量和保留删除标记的状态。
    def forget(
        self,
        scope: MemoryOwnerScope,
        memory_key: str,
        *,
        mode: ForgetMode = "soft",
        reason: str = "user_requested",
    ) -> dict[str, int | str]:
        if not _MEMORY_KEY.fullmatch(memory_key):
            raise ValueError("ACCOUNT_MEMORY_KEY_INVALID")
        if mode not in {"soft", "permanent"}:
            raise ValueError("ACCOUNT_MEMORY_FORGET_MODE_INVALID")
        if not reason.strip() or len(reason) > 160 or "\x00" in reason:
            raise ValueError("ACCOUNT_MEMORY_FORGET_REASON_INVALID")
        owner, tenant, organization, namespace = self._scope_values(scope)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if mode == "soft":
                active_deleted = 0
                active_revoked = connection.execute(
                    """UPDATE governed_account_memory SET status='revoked',updated_at=?
                       WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                         AND namespace=? AND memory_key=?""",
                    (_now().isoformat(), owner, tenant, organization, namespace, memory_key),
                ).rowcount
                candidates_deleted = 0
                candidates_revoked = connection.execute(
                    """UPDATE account_memory_candidates
                       SET status='rejected',decision_reason='memory_soft_revoked'
                       WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                         AND namespace=? AND memory_key=? AND status!='rejected'""",
                    (owner, tenant, organization, namespace, memory_key),
                ).rowcount
            else:
                active_revoked = 0
                candidates_revoked = 0
                active_deleted = connection.execute(
                    """DELETE FROM governed_account_memory
                       WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                         AND namespace=? AND memory_key=?""",
                    (owner, tenant, organization, namespace, memory_key),
                ).rowcount
                candidates_deleted = connection.execute(
                    """DELETE FROM account_memory_candidates
                       WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                         AND namespace=? AND memory_key=?""",
                    (owner, tenant, organization, namespace, memory_key),
                ).rowcount
            connection.execute(
                """INSERT INTO account_memory_tombstones(
                     owner_account_id,tenant_id,organization_id,namespace,memory_key,
                     mode,reason,created_at
                   ) VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(owner_account_id,tenant_id,organization_id,namespace,memory_key)
                   DO UPDATE SET mode=excluded.mode,reason=excluded.reason,
                                 created_at=excluded.created_at""",
                (
                    owner,
                    tenant,
                    organization,
                    namespace,
                    memory_key,
                    mode,
                    reason.strip(),
                    _now().isoformat(),
                ),
            )
        return {
            "memory_key": memory_key,
            "mode": mode,
            "active_revoked": int(active_revoked),
            "active_deleted": int(active_deleted),
            "candidates_revoked": int(candidates_revoked),
            "candidates_deleted": int(candidates_deleted),
            "tombstone": "created",
        }

    # 功能：
    #   比对安全规范化载荷，仅相同且仍有效的观察可参与证据归并。
    # 输入：
    #   rows：当前账户同键候选；payload：目标安全载荷。
    # 输出：
    #   matching：未过期且内容一致的候选行。
    @staticmethod
    def _matching_candidate_rows(
        rows: list[sqlite3.Row], payload: dict[str, Any]
    ) -> list[sqlite3.Row]:
        matching: list[sqlite3.Row] = []
        now = _now()
        for row in rows:
            candidate = AccountMemoryStore._candidate_row(row)
            if (
                candidate.observed_at + timedelta(days=candidate.ttl_days) > now
                and candidate.payload == payload
            ):
                matching.append(row)
        return matching

    # 功能：
    #   重验数据库候选的内容、来源、时间和数值类型，存储损坏不能变成有效核验声明。
    # 输入：
    #   row：已按账户过滤的候选行。
    # 输出：
    #   candidate：独立且通过完整候选合同的对象。
    @staticmethod
    def _candidate_row(row: sqlite3.Row) -> SessionMemoryCandidate:
        if any(
            type(row[key]) is not int or row[key] not in (0, 1)
            for key in ("source_verified", "explicit_reconsent", "conflict_with_active")
        ):
            raise ValueError("ACCOUNT_MEMORY_CANDIDATE_FLAGS_INVALID")
        candidate = SessionMemoryCandidate(
            candidate_id=row["candidate_id"],
            owner_account_id=row["owner_account_id"],
            namespace=row["namespace"],
            tenant_id=row["tenant_id"] or None,
            organization_id=row["organization_id"] or None,
            kind=row["kind"],
            memory_key=row["memory_key"],
            payload=decode_json(row["payload_json"], limit=32 * 1024),
            source_conversation_id=row["source_conversation_id"],
            source_edition=row["source_edition"],
            source_kind=row["source_kind"],
            source_receipt_id=row["source_receipt_id"],
            source_receipt_sha256=row["source_receipt_sha256"],
            source_verified=row["source_verified"] == 1,
            explicit_reconsent=row["explicit_reconsent"] == 1,
            confidence=row["confidence"],
            observed_at=datetime.fromisoformat(row["observed_at"]),
            ttl_days=row["ttl_days"],
            status=row["status"],
        )
        return candidate

    # 功能：
    #   同一会话只保留一次观察，模型多次调用不能伪装成多个独立见证。
    # 输入：
    #   rows：已筛选的同内容候选行。
    # 输出：
    #   independent：按观察时间及候选标识稳定排序的独立会话候选。
    @staticmethod
    def _independent_candidate_rows(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
        by_conversation: dict[str, sqlite3.Row] = {}
        for row in rows:
            by_conversation[str(row["source_conversation_id"])] = row
        independent = sorted(
            by_conversation.values(),
            key=lambda row: (str(row["observed_at"]), str(row["candidate_id"])),
        )
        return independent

    # 功能：
    #   保存来源及核验回执的有界元数据，不复制访问凭证或执行权限。
    # 输入：
    #   candidate：已经验证的候选。
    # 输出：
    #   provenance：用于追踪记忆来源的字符串字典。
    @staticmethod
    def _provenance(candidate: SessionMemoryCandidate) -> dict[str, str]:
        provenance = {
            "candidate_id": candidate.candidate_id,
            "source_conversation_id": candidate.source_conversation_id,
            "source_edition": candidate.source_edition or "unspecified",
            "source_kind": candidate.source_kind,
            "observed_at": candidate.observed_at.isoformat(),
        }
        if candidate.source_receipt_id is not None:
            provenance["source_receipt_id"] = candidate.source_receipt_id
        if candidate.source_receipt_sha256 is not None:
            provenance["source_receipt_sha256"] = candidate.source_receipt_sha256
        if candidate.explicit_reconsent:
            provenance["explicit_reconsent"] = "true"
        return provenance

    # 功能：
    #   在调用方已经授权的事务内写入提升结果、合并有界来源并更新候选状态。
    # 输入：
    #   self：存储；connection：调用方事务；scope：完整账户范围。
    #   memory_key、kind、payload：受治理键、类型和内容。
    #   independent_rows：有效独立证据；decision_reason：显式批准或产品核验原因。
    # 输出：
    #   None：不返回业务数据。
    def _promote_rows(
        self,
        connection: sqlite3.Connection,
        scope: MemoryOwnerScope,
        memory_key: str,
        kind: str,
        payload: dict[str, Any],
        independent_rows: list[sqlite3.Row],
        *,
        decision_reason: str,
    ) -> None:
        if not independent_rows:
            raise ValueError("ACCOUNT_MEMORY_PROMOTION_EVIDENCE_REQUIRED")
        owner, tenant, organization, namespace = self._scope_values(scope)
        latest = independent_rows[-1]
        confidence = float(independent_rows[0]["confidence"])
        for row in independent_rows[1:]:
            observed_confidence = float(row["confidence"])
            confidence = min(0.99, confidence + (1.0 - confidence) * observed_confidence * 0.25)
        provenance = [
            {
                key: value
                for key, value in {
                    "candidate_id": str(row["candidate_id"]),
                    "source_conversation_id": str(row["source_conversation_id"]),
                    "source_edition": str(row["source_edition"] or "unspecified"),
                    "source_kind": str(row["source_kind"]),
                    "source_receipt_id": (
                        str(row["source_receipt_id"])
                        if row["source_receipt_id"] is not None
                        else None
                    ),
                    "source_receipt_sha256": (
                        str(row["source_receipt_sha256"])
                        if row["source_receipt_sha256"] is not None
                        else None
                    ),
                    "explicit_reconsent": ("true" if bool(row["explicit_reconsent"]) else None),
                    "observed_at": str(row["observed_at"]),
                }.items()
                if value is not None
            }
            for row in independent_rows[-16:]
        ]
        expires_at = max(
            datetime.fromisoformat(str(row["observed_at"])) + timedelta(days=int(row["ttl_days"]))
            for row in independent_rows
        )
        if expires_at <= _now() or len(independent_rows) > 1_000_000:
            raise ValueError("ACCOUNT_MEMORY_PROMOTION_EVIDENCE_INVALID")
        now = _now().isoformat()
        existing = connection.execute(
            """SELECT created_at FROM governed_account_memory
               WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                 AND namespace=? AND memory_key=?""",
            (owner, tenant, organization, namespace, memory_key),
        ).fetchone()
        created_at = str(existing["created_at"]) if existing is not None else now
        # 提交前用活动条目合同再验一次，不能等下一次读取才发现已写入坏数据。
        validated = GovernedMemoryEntry(
            owner_account_id=scope.owner_account_id,
            namespace=scope.namespace,
            tenant_id=scope.tenant_id,
            organization_id=scope.organization_id,
            kind=kind,
            memory_key=memory_key,
            payload=payload,
            source_conversation_id=latest["source_conversation_id"],
            source_edition=latest["source_edition"],
            provenance=provenance,
            evidence_count=len(independent_rows),
            confidence=confidence,
            last_observed_at=datetime.fromisoformat(latest["observed_at"]),
            expires_at=expires_at,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(now),
        )
        connection.execute(
            """INSERT INTO governed_account_memory(
                 owner_account_id,tenant_id,organization_id,namespace,memory_key,kind,
                 payload_json,source_conversation_id,source_edition,provenance_json,
                 evidence_count,confidence,last_observed_at,expires_at,status,
                 created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(owner_account_id,tenant_id,organization_id,namespace,memory_key)
               DO UPDATE SET
                 kind=excluded.kind,payload_json=excluded.payload_json,
                 source_conversation_id=excluded.source_conversation_id,
                 source_edition=excluded.source_edition,
                 provenance_json=excluded.provenance_json,
                 evidence_count=excluded.evidence_count,confidence=excluded.confidence,
                 last_observed_at=excluded.last_observed_at,expires_at=excluded.expires_at,
                 status='active',updated_at=excluded.updated_at""",
            (
                owner,
                tenant,
                organization,
                namespace,
                memory_key,
                kind,
                _canonical_payload(payload),
                str(latest["source_conversation_id"]),
                latest["source_edition"],
                json.dumps(provenance, ensure_ascii=False, sort_keys=True),
                len(independent_rows),
                confidence,
                validated.last_observed_at.isoformat(),
                validated.expires_at.isoformat(),
                "active",
                validated.created_at.isoformat(),
                now,
            ),
        )
        for row in independent_rows:
            connection.execute(
                """UPDATE account_memory_candidates
                   SET status='consolidated',conflict_with_active=0,decision_reason=?
                   WHERE candidate_id=?""",
                (decision_reason, str(row["candidate_id"])),
            )

    # 功能：
    #   只返回当前账户范围中仍活动且未过期的条目，不把候选、撤回或损坏数据视为可用记忆。
    # 输入：
    #   self：存储；scope：完整账户范围；memory_key：目标键。
    # 输出：
    #   entry：重新验证后的独立记忆条目。
    def get(self, scope: MemoryOwnerScope, memory_key: str) -> GovernedMemoryEntry:
        owner, tenant, organization, namespace = self._scope_values(scope)
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM governed_account_memory
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                     AND namespace=? AND memory_key=?""",
                (owner, tenant, organization, namespace, memory_key),
            ).fetchone()
        if row is None:
            raise KeyError(memory_key)
        entry = self._row(row)
        if entry.expires_at <= _now():
            self._expire(scope, memory_key)
            raise KeyError(memory_key)
        if entry.status != "active":
            raise KeyError(memory_key)
        return entry

    # 功能：
    #   列出任务职责域的活动记忆；跨域共享候选由模型上下文组装入口另行合并。
    # 输入：
    #   self：存储；scope：账户范围；limit：条目数上限。
    # 输出：
    #   entries：验证过的任务记忆条目。
    def list(self, scope: MemoryOwnerScope, *, limit: int = 64) -> list[GovernedMemoryEntry]:
        entries = self._list_namespace(scope, AUTONOMY_MISSION_NAMESPACE, limit=limit)
        return entries

    # 功能：
    #   处理到期条目并有界读取活动记录，损坏载荷按原代次隔离，不能覆盖并发修正。
    # 输入：
    #   self：存储；scope：账户边界；namespace：固定允许的职责域；limit：条目上限。
    # 输出：
    #   result：内容与时间均验证通过的独立条目。
    def _list_namespace(
        self,
        scope: MemoryOwnerScope,
        namespace: Literal["autonomy.mission", "account.shared"],
        *,
        limit: int,
    ) -> list[GovernedMemoryEntry]:
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("account memory limit must be between 1 and 256")
        owner, tenant, organization, _ = self._scope_values(scope)
        now = _now().isoformat()
        with self._connect() as connection:
            connection.execute(
                """UPDATE governed_account_memory SET status='expired'
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=? AND namespace=?
                     AND status='active' AND julianday(expires_at) <= julianday(?)""",
                (owner, tenant, organization, namespace, now),
            )
            rows = connection.execute(
                """SELECT * FROM governed_account_memory
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=? AND namespace=?
                     AND status='active' AND (
                       julianday(expires_at) > julianday(?) OR julianday(expires_at) IS NULL)
                   ORDER BY
                     CASE kind WHEN 'constraint' THEN 0 WHEN 'preference' THEN 1 ELSE 2 END,
                     confidence DESC, evidence_count DESC, last_observed_at DESC, memory_key ASC
                   LIMIT ?""",
                (owner, tenant, organization, namespace, now, limit),
            ).fetchall()
        result: list[GovernedMemoryEntry] = []
        for row in rows:
            try:
                result.append(self._row(row))
            except (ValueError, TypeError, json.JSONDecodeError):
                # A corrupt or legacy-unsafe row never crosses the model-context boundary.
                with self._connect() as connection:
                    connection.execute(
                        """UPDATE governed_account_memory SET status='quarantined'
                           WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                             AND namespace=? AND memory_key=?
                             AND payload_json=? AND updated_at=?""",
                        (
                            owner,
                            tenant,
                            organization,
                            namespace,
                            str(row["memory_key"]),
                            str(row["payload_json"]),
                            str(row["updated_at"]),
                        ),
                    )
                continue
        return result

    # 功能：
    #   合并任务与共享记忆，按词汇相关性及治理证据排序，在条目与估算预算内提供模型参考。
    # 输入：
    #   self：存储；scope：账户范围；query：可选任务查询文本。
    #   limit：最多条目数；token_budget：基于 UTF-8 长度的估算预算，不是计费 token。
    # 输出：
    #   context：最终重验的建议性记忆封装，不授予飞行控制权。
    def model_context(
        self,
        scope: MemoryOwnerScope,
        *,
        query: str | None = None,
        limit: int = 8,
        token_budget: int = 1_200,
    ) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 16:
            raise ValueError("account memory model-context limit must be between 1 and 16")
        if type(token_budget) is not int or not 256 <= token_budget <= 4_096:
            raise ValueError("account memory token budget must be between 256 and 4096")
        if query is not None and (
            not isinstance(query, str) or len(query) > 4_000 or "\x00" in query
        ):
            raise ValueError("account memory semantic query is invalid")
        items: list[dict[str, Any]] = []
        used_tokens = 0
        primary_entries = self._list_namespace(
            scope, AUTONOMY_MISSION_NAMESPACE, limit=min(64, limit * 4)
        )
        shared_entries = self._list_namespace(
            scope, ACCOUNT_SHARED_NAMESPACE, limit=min(64, limit * 4)
        )
        seen_keys: set[str] = set()
        candidates: list[tuple[int, int, GovernedMemoryEntry]] = []
        # Domain-specific memory deterministically overrides the lower-priority
        # account.shared view.  Both queries expose active records only.
        for source_priority, entry in [
            *((0, item) for item in primary_entries),
            *((1, item) for item in shared_entries),
        ]:
            if entry.memory_key in seen_keys:
                continue
            seen_keys.add(entry.memory_key)
            candidates.append((source_priority, len(candidates), entry))
        query_terms = _semantic_terms(query or "")

        # 功能：
        #   依次比较词汇重叠、内容种类、来源域和证据，保留稳定并列次序。
        # 输入：
        #   candidate：来源优先级、稳定索引及记忆条目。
        # 输出：
        #   ranking：供 sorted 使用的排序元组。
        def rank(candidate: tuple[int, int, GovernedMemoryEntry]) -> tuple[object, ...]:
            source_priority, stable_index, entry = candidate
            entry_terms = _semantic_terms(
                {"memory_key": entry.memory_key, "payload": entry.payload}
            )
            overlap = len(query_terms.intersection(entry_terms)) if query_terms else 0
            kind_priority = {"constraint": 0, "preference": 1, "summary": 2}[entry.kind]
            ranking = (
                0 if overlap else 1,
                -overlap,
                kind_priority,
                source_priority,
                -entry.confidence,
                -entry.evidence_count,
                stable_index,
            )
            return ranking

        for _source_priority, _stable_index, entry in sorted(candidates, key=rank):
            item = entry.model_context()
            estimated_tokens = max(
                1,
                len(json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")) // 4,
            )
            if used_tokens + estimated_tokens > token_budget:
                continue
            items.append(item)
            used_tokens += estimated_tokens
            if len(items) >= limit:
                break
        context = validate_account_memory_model_context(
            {
                "namespace": scope.namespace,
                "items": items,
                "top_k": limit,
                "token_budget": token_budget,
                "estimated_tokens": used_tokens,
                "authority_reuse_allowed": False,
            }
        )
        return context

    # 功能：
    #   按本账户职责域的年龄与数量保留规则清理内容及旧候选，但不删除防止重学的标记。
    # 输入：
    #   self：存储；scope：账户范围；maximum_entries：保留上限。
    #   maximum_age_days：条目最长年龄，候选最多保留 90 天。
    # 输出：
    #   removed：被删除的记忆条目数，不包括候选数量。
    def apply_retention(
        self,
        scope: MemoryOwnerScope,
        *,
        maximum_entries: int = 128,
        maximum_age_days: int = 365,
    ) -> int:
        if type(maximum_entries) is not int or not 3 <= maximum_entries <= 10_000:
            raise ValueError("maximum_entries must be between 3 and 10000")
        if type(maximum_age_days) is not int or not 1 <= maximum_age_days <= 3_650:
            raise ValueError("maximum_age_days must be between 1 and 3650")
        owner, tenant, organization, namespace = self._scope_values(scope)
        cutoff = (_now() - timedelta(days=maximum_age_days)).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            expired = connection.execute(
                """DELETE FROM governed_account_memory
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                     AND namespace=? AND updated_at < ?""",
                (owner, tenant, organization, namespace, cutoff),
            ).rowcount
            overflow = connection.execute(
                """DELETE FROM governed_account_memory
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=? AND namespace=?
                     AND memory_key IN (
                       SELECT memory_key FROM governed_account_memory
                       WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                         AND namespace=?
                       ORDER BY updated_at DESC, memory_key ASC LIMIT -1 OFFSET ?
                     )""",
                (
                    owner,
                    tenant,
                    organization,
                    namespace,
                    owner,
                    tenant,
                    organization,
                    namespace,
                    maximum_entries,
                ),
            ).rowcount
            candidate_cutoff = (_now() - timedelta(days=min(maximum_age_days, 90))).isoformat()
            connection.execute(
                """DELETE FROM account_memory_candidates
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                     AND namespace=? AND observed_at < ?""",
                (owner, tenant, organization, namespace, candidate_cutoff),
            )
        removed = int(expired) + int(overflow)
        return removed

    # 功能：
    #   清空当前职责域的内容与候选，不更改授权开关，也不释放删除标记。
    # 输入：
    #   self：存储；scope：要清理的明确账户范围。
    # 输出：
    #   removed：实际删除的活动表条目数。
    def clear(self, scope: MemoryOwnerScope) -> int:
        owner, tenant, organization, namespace = self._scope_values(scope)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """DELETE FROM governed_account_memory
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                     AND namespace=?""",
                (owner, tenant, organization, namespace),
            )
            connection.execute(
                """DELETE FROM account_memory_candidates
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                     AND namespace=?""",
                (owner, tenant, organization, namespace),
            )
        removed = int(cursor.rowcount)
        return removed

    # 功能：
    #   只过期数据库当前仍已到期的行，读取后并发续期的记录不能被旧请求撤销。
    # 输入：
    #   self：存储；scope：账户范围；memory_key：待检查键。
    # 输出：
    #   None：不返回业务数据。
    def _expire(self, scope: MemoryOwnerScope, memory_key: str) -> None:
        owner, tenant, organization, namespace = self._scope_values(scope)
        with self._connect() as connection:
            connection.execute(
                """UPDATE governed_account_memory SET status='expired'
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                     AND namespace=? AND memory_key=? AND status='active'
                     AND julianday(expires_at)<=julianday(?)""",
                (owner, tenant, organization, namespace, memory_key, _now().isoformat()),
            )

    # 功能：
    #   主动隔离指定账户的不安全记忆键，保留内容供故障追溯而不再提供给模型。
    # 输入：
    #   self：存储；scope：账户范围；memory_key：要隔离的键。
    # 输出：
    #   None：不返回业务数据。
    def _quarantine(self, scope: MemoryOwnerScope, memory_key: str) -> None:
        owner, tenant, organization, namespace = self._scope_values(scope)
        with self._connect() as connection:
            connection.execute(
                """UPDATE governed_account_memory SET status='quarantined'
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                     AND namespace=? AND memory_key=?""",
                (owner, tenant, organization, namespace, memory_key),
            )

    # 功能：
    #   通过与写入相同的验证器重建数据库条目，不强转损坏的证据类型。
    # 输入：
    #   row：账户过滤后读出的 SQLite 行。
    # 输出：
    #   entry：独立且满足内容、时间和身份约束的条目。
    @staticmethod
    def _row(row: sqlite3.Row) -> GovernedMemoryEntry:
        tenant = str(row["tenant_id"])
        organization = str(row["organization_id"])
        entry = GovernedMemoryEntry(
            owner_account_id=str(row["owner_account_id"]),
            namespace=str(row["namespace"]),
            tenant_id=tenant or None,
            organization_id=organization or None,
            kind=str(row["kind"]),
            memory_key=str(row["memory_key"]),
            payload=decode_json(row["payload_json"], limit=32 * 1024),
            source_conversation_id=str(row["source_conversation_id"]),
            source_edition=(str(row["source_edition"]) if row["source_edition"] else None),
            provenance=decode_json(row["provenance_json"], limit=64 * 1024),
            evidence_count=row["evidence_count"],
            confidence=row["confidence"],
            last_observed_at=datetime.fromisoformat(str(row["last_observed_at"])),
            expires_at=datetime.fromisoformat(str(row["expires_at"])),
            status=str(row["status"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )
        return entry
