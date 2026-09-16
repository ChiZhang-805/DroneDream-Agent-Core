"""Strict HTTP contracts for the local desktop application."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, model_validator

from dronedream_plugin_sdk.protocol import encode_json

# Match the installed plugin contract, including supported prerelease/build
# suffixes. A prefix match must not turn another string into a version identity.
_PLUGIN_VERSION = (
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_METADATA_BYTES_LIMIT = 4 * 1024 * 1024


class AppModel(BaseModel):
    """JSON requests are typed values, not coercible control or consent strings."""

    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, strict=True, allow_inf_nan=False
    )

    # 功能：
    #   1. 在字段转换前限制结构深度、节点及非有限数，保护 object 类型的嵌套值。
    #   2. 任意元数据只允许标准 JSON，单个字段不超过 4 MiB；实际 HTTP 请求总量由入口控制。
    # 输入：
    #   cls：当前请求模型类型。
    #   value：字段校验前的请求对象。
    # 输出：
    #   value：通过边界检查的原请求对象。
    @model_validator(mode="before")
    @classmethod
    def _bounded_finite_payload(cls, value: object) -> object:
        if isinstance(value, dict):
            for name in ("metadata", "input_metadata", "configuration"):
                if name in cls.model_fields and name in value:
                    # 不把整个请求强制序列化：内部调用仍可以传入已验证的子模型、HttpUrl。
                    # 自由元数据没有这种豁免，否则 object 字段会保留集合、元组等非 JSON 值。
                    encode_json(value[name], limit=_METADATA_BYTES_LIMIT, node_limit=65_536)
        pending = [(value, 0)]
        visited = 0
        while pending:
            item, depth = pending.pop()
            visited += 1
            if visited > 65_536 or depth > 32:
                raise ValueError("APP_REQUEST_STRUCTURE_LIMIT")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("APP_REQUEST_NONFINITE_NUMBER")
            if isinstance(item, (dict, list, tuple)):
                children = item.values() if isinstance(item, dict) else item
                if visited + len(pending) + len(item) > 65_536:
                    raise ValueError("APP_REQUEST_STRUCTURE_LIMIT")
                pending.extend((child, depth + 1) for child in children)
        return value


class ThreadCreate(AppModel):
    """Create a local conversation in planning state, without execution authority."""

    title: str | None = Field(default=None, min_length=1, max_length=120)
    selected_model: str = Field(default="gpt-5.4", min_length=1, max_length=80)
    locale: Literal["zh-CN", "en-US"] = "zh-CN"

    # 功能：
    #   只在未提供标题时按已验证语言生成默认标题，不覆盖用户自己的标题。
    # 输入：
    #   self：完成字段校验的任务创建请求。
    # 输出：
    #   self：已补齐标题的请求。
    @model_validator(mode="after")
    def _localized_default_title(self) -> ThreadCreate:
        if self.title is None:
            self.title = "New task" if self.locale == "en-US" else "新任务"
        return self


class ThreadPatch(AppModel):
    """Omission preserves state; null clears only optional asset selections."""

    title: str | None = Field(default=None, min_length=1, max_length=120)
    selected_model: str | None = Field(default=None, min_length=1, max_length=80)
    selected_map_id: str | None = Field(default=None, max_length=120)
    selected_map_content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    selected_vehicle_id: str | None = Field(default=None, max_length=120)
    selected_vehicle_content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    locale: Literal["zh-CN", "en-US"] | None = None
    pinned: bool | None = None
    archived: bool | None = None

    # 功能：
    #   在 HTTP 校验阶段拒绝显式清空不可为空的任务字段，未传入的字段保持不变。
    # 输入：
    #   self：任务更新请求及实际传入字段集合。
    # 输出：
    #   self：不会清空数据库必填列的请求。
    @model_validator(mode="after")
    def _nonnull_stored_fields(self) -> ThreadPatch:
        for name in ("title", "selected_model", "locale", "pinned", "archived"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError("THREAD_FIELD_CANNOT_BE_NULL")
        return self


class MessageCreate(AppModel):
    """Store a bounded conversation item; plan-shaped content is not executable consent."""

    content: str = Field(min_length=1, max_length=4_000)
    role: Literal["user", "assistant", "system"] = "user"
    kind: Literal["text", "status", "plan", "error"] = "text"
    metadata: dict[str, object] = Field(default_factory=dict)


class AccountMemoryScopeRequest(AppModel):
    """Authenticated product scope for account-memory governance operations."""

    expected_owner_account_id: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]{1,159}$",
    )
    expected_tenant_id: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]{1,159}$",
    )
    expected_organization_id: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]{1,159}$",
    )
    source_edition: Literal["universal", "sim", "lab", "field", "autonomy"]
    thread_id: str = Field(pattern=r"^thread-[0-9a-f]{32}$")


class AccountMemoryCandidateListRequest(AccountMemoryScopeRequest):
    """List candidates within the authenticated scope, not arbitrary account memory."""

    memory_key: str | None = Field(
        default=None,
        min_length=3,
        max_length=104,
        pattern=r"^(summary|preference|constraint)\.[a-z][a-z0-9._-]{1,95}$",
    )
    limit: int = Field(default=100, ge=1, le=500)


class AccountMemoryCandidateDecisionRequest(AccountMemoryScopeRequest):
    """An explicit boolean records renewed consent; strings cannot grant it."""

    explicit_reconsent: bool = False


class AccountMemoryForgetRequest(AccountMemoryScopeRequest):
    """Target one validated memory key; permanent forgetting has different retention semantics."""

    memory_key: str = Field(
        min_length=3,
        max_length=104,
        pattern=r"^(summary|preference|constraint)\.[a-z][a-z0-9._-]{1,95}$",
    )
    mode: Literal["soft", "permanent"] = "soft"
    reason: str = Field(default="user_requested", min_length=1, max_length=160)


class ModelRoleConnectionRequest(AppModel):
    """Bind one auxiliary model role to its own short-lived model grant."""

    role_port: Literal["critic", "safety", "perception", "local"]
    model_id: str = Field(min_length=1, max_length=80)
    model_grant: str = Field(pattern=r"^dd[gc]_[A-Za-z0-9_-]{20,124}$")
    gateway_base_url: HttpUrl | None = None


class MissionPrepareRequest(AppModel):
    """Request planning for an exact account scope, asset selection and model connection.

    Optional asset hashes are resolved and qualified by the service, not invented
    by the model. Expected account identifiers must be checked against identity;
    placing them in a request does not authenticate their owner.
    """

    expected_owner_account_id: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]{1,159}$",
    )
    expected_tenant_id: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]{1,159}$",
    )
    expected_organization_id: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]{1,159}$",
    )
    source_edition: Literal["universal", "sim", "lab", "field", "autonomy"]
    message: str = Field(min_length=3, max_length=4_000)
    map_id: str = Field(min_length=1, max_length=120)
    map_content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    vehicle_id: str = Field(min_length=1, max_length=120)
    vehicle_content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_id: str = Field(min_length=1, max_length=80)
    model_grant: str = Field(pattern=r"^dd[gc]_[A-Za-z0-9_-]{20,124}$")
    gateway_base_url: HttpUrl | None = None
    locale: Literal["zh-CN", "en-US"] = "zh-CN"
    start_entity: str = Field(default="__auto__", min_length=1, max_length=160)
    attachment_ids: list[str] = Field(default_factory=list, max_length=12)
    role_models: list[ModelRoleConnectionRequest] = Field(default_factory=list, max_length=4)
    input_channel: Literal["text", "voice", "camera", "api", "webhook", "scheduled"] = "text"
    input_metadata: dict[str, object] = Field(default_factory=dict)

    # 功能：
    #   拒绝重复的辅助模型角色，避免后续转为映射时由列表顺序决定保留哪一项。
    # 输入：
    #   self：准备任务请求。
    # 输出：
    #   self：辅助模型角色唯一的请求。
    @model_validator(mode="after")
    def _unique_model_roles(self) -> MissionPrepareRequest:
        roles = [connection.role_port for connection in self.role_models]
        if len(set(roles)) != len(roles):
            raise ValueError("MISSION_MODEL_ROLE_DUPLICATE")
        return self


class AssetInterpretationRequest(AppModel):
    """Interpret a selected immutable asset without planning or actuator authority."""

    expected_owner_account_id: str = Field(min_length=2, max_length=160)
    expected_tenant_id: str | None = Field(default=None, min_length=2, max_length=160)
    expected_organization_id: str | None = Field(default=None, min_length=2, max_length=160)
    source_edition: Literal["universal", "sim", "lab", "field", "autonomy"]
    kind: Literal["map", "vehicle"]
    asset_id: str = Field(min_length=1, max_length=120)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str = Field(min_length=1, max_length=80)
    model_grant: str = Field(pattern=r"^dd[gc]_[A-Za-z0-9_-]{20,124}$", repr=False)
    gateway_base_url: HttpUrl | None = None
    locale: Literal["zh-CN", "en-US"] = "zh-CN"
    force: bool = Field(default=False, strict=True)


class MissionExecuteRequest(AppModel):
    """Confirm one prepared plan revision; the service still validates ownership and grants."""

    expected_owner_account_id: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]{1,159}$",
    )
    expected_tenant_id: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]{1,159}$",
    )
    expected_organization_id: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]{1,159}$",
    )
    source_edition: Literal["universal", "sim", "lab", "field", "autonomy"]
    plan_revision_id: str = Field(pattern=r"^plan-[0-9a-f]{32}$")
    model_id: str = Field(min_length=1, max_length=80)
    model_grant: str = Field(pattern=r"^dd[gc]_[A-Za-z0-9_-]{20,124}$")
    gateway_base_url: HttpUrl | None = None


class CustomModelDiscoverRequest(AppModel):
    """Sensitive discovery input; never persist or log this request's API key."""

    base_url: str = Field(min_length=8, max_length=500)
    # 秘密不能复用展示文本的自动去空格规则；也不出现在模型 repr 中。
    api_key: Annotated[str, StringConstraints(strip_whitespace=False)] = Field(
        min_length=8, max_length=8_192, repr=False
    )


class CustomModelCreateRequest(CustomModelDiscoverRequest):
    """Save connection metadata separately from vault-managed secret material."""

    display_name: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=160)
    provider: str | None = Field(default=None, min_length=1, max_length=80)
    api_style: Literal["responses", "chat-completions"] = "chat-completions"


class SettingsPatch(AppModel):
    """Optional fields mean unchanged; explicit memory choices must be booleans."""

    locale: Literal["zh-CN", "en-US"] | None = None
    theme: Literal["system", "light", "dark"] | None = None
    update_channel: Literal["stable", "preview"] | None = None
    default_model_id: str | None = Field(default=None, min_length=1, max_length=160)
    memory_enabled: bool | None = None
    remember_task_preferences: bool | None = None
    remember_asset_choices: bool | None = None
    plugin_update_ring: Literal["stable", "preview", "canary", "pinned"] | None = None

    # 功能：
    #   区分未修改开关与显式空值；空值不是记忆同意，不能替换已保存开关。
    # 输入：
    #   self：设置更新请求及实际传入字段集合。
    # 输出：
    #   self：所有显式记忆开关均为布尔值的请求。
    @model_validator(mode="after")
    def _nonnull_consent(self) -> SettingsPatch:
        for name in ("memory_enabled", "remember_task_preferences", "remember_asset_choices"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError("MEMORY_CONSENT_BOOLEAN_REQUIRED")
        return self


class AssetPairQualificationCreateRequest(AppModel):
    """Qualify exactly one immutable map/vehicle pair, not mutable display names."""

    map_asset_id: str = Field(min_length=1, max_length=160)
    map_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vehicle_asset_id: str = Field(min_length=1, max_length=160)
    vehicle_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AssetRemoteImportRequest(AppModel):
    """Describe a remote source; network/path validation remains the importer's responsibility."""

    source_type: Literal["direct_url", "git"]
    location: HttpUrl
    source_format: str = Field(
        default="auto",
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    expected_kind: Literal["map", "world", "vehicle"] | None = None
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    git_ref: str | None = Field(default=None, min_length=1, max_length=160)
    subpath: str | None = Field(default=None, min_length=1, max_length=500)

    # 功能：
    #   非 Git 导入不能携带 Git 引用或子目录选择，避免入口静默忽略用户要求。
    # 输入：
    #   self：远程资产导入请求。
    # 输出：
    #   self：选择参数与来源类型一致的请求。
    @model_validator(mode="after")
    def _git_fields_match_source_type(self) -> AssetRemoteImportRequest:
        if self.source_type != "git" and (self.git_ref is not None or self.subpath is not None):
            raise ValueError("git_ref and subpath require source_type=git")
        return self


class RuntimeMessageRequest(AppModel):
    """Submit bounded natural language for runtime interpretation, not raw actuator access."""

    text: str = Field(min_length=1, max_length=1_000)


class OperatorTakeoverGrantRequest(AppModel):
    """Request a time-limited operator lease bound to an existing runtime message."""

    message_id: str = Field(pattern=r"^runtime-msg-[0-9a-f]{32}$")
    operator_id: str = Field(min_length=1, max_length=160)
    duration_seconds: int = Field(default=300, ge=10, le=600)


class OperatorControlRequest(AppModel):
    """Short-lived NED velocity (m/s) and yaw-rate (degrees/s) request, not a waypoint.

    Positive down is descent. The lease, current authority and transport expiry
    are validated downstream before the flight controller sees a command.
    """

    message_id: str = Field(pattern=r"^runtime-msg-[0-9a-f]{32}$")
    grant_token: str = Field(min_length=32, max_length=256)
    action: Literal["velocity", "release"] = "velocity"
    north_mps: float = Field(default=0.0, ge=-3.0, le=3.0)
    east_mps: float = Field(default=0.0, ge=-3.0, le=3.0)
    down_mps: float = Field(default=0.0, ge=-2.0, le=2.0)
    yaw_rate_dps: float = Field(default=0.0, ge=-180.0, le=180.0)
    duration_seconds: float = Field(default=0.25, gt=0.0, le=0.5)


class PluginConfigurationRequest(AppModel):
    """Finite JSON configuration, subsequently checked against the plugin's own schema."""

    configuration: dict[str, object] = Field(default_factory=dict)


class HarnessRevisionActionRequest(AppModel):
    """Apply a change only against the expected current Harness revision."""

    base_revision: int = Field(ge=1)


class ConnectorCredentialCreateRequest(AppModel):
    """Associate a vault secret with a narrow plugin allowlist; never return the secret."""

    display_name: str = Field(min_length=1, max_length=80)
    secret: Annotated[str, StringConstraints(strip_whitespace=False)] = Field(
        min_length=1, max_length=16_384, repr=False
    )
    allowed_plugin_ids: list[str] = Field(min_length=1, max_length=32)


class PluginRollbackRequest(AppModel):
    """Select an exact installed version; staging and enabling are separate actions."""

    version: str = Field(pattern=_PLUGIN_VERSION, max_length=160)


class TrustedPublisherRequest(AppModel):
    """Request a publisher key binding; decoding and trust approval occur downstream."""

    key_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,119}$")
    publisher: str = Field(min_length=1, max_length=120)
    public_key_base64: str = Field(min_length=40, max_length=80)


class PluginMarketplaceInstallRequest(AppModel):
    """Bind installation to the selected source, plugin and complete version identity."""

    source_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,119}$")
    plugin_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,119}$")
    version: str = Field(pattern=_PLUGIN_VERSION, max_length=160)
