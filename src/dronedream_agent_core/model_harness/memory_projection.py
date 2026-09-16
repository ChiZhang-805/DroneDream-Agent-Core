"""RLS-bound Supabase projection for governed account memory.

The bridge is deliberately not a second memory authority.  Local memory remains
governed by :mod:`memory`; this module only moves sanitized candidates and
accepted records through Supabase using a real user JWT plus the public project
key.  Network failure leaves an idempotent local outbox and grants no capability.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dronedream_plugin_sdk.protocol import copy_json, decode_json, encode_json

from .memory import (
    ACCOUNT_SHARED_NAMESPACE,
    AUTONOMY_MISSION_NAMESPACE,
    AccountMemoryStore,
    MemoryOwnerScope,
)

_ZERO_UUID = "00000000-0000-0000-0000-000000000000"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_READ_NAMESPACES = (ACCOUNT_SHARED_NAMESPACE, AUTONOMY_MISSION_NAMESPACE)
_SCOPES = (
    "chat_preferences",
    "device_vehicle",
    "reports_delivery",
    "safety_approvals",
)
ProjectionStatus = Literal[
    "synced",
    "pending",
    "local_disabled",
    "consent_required",
    "remote_disabled",
    "configuration_missing",
    "network_unavailable",
    "conflict",
]


# 功能：
#   为持久化同步记录取得 UTC 时间；网络耗时另用单调时钟计算。
# 输入：
#   无。
# 输出：
#   now：带时区的当前时间。
def _now() -> datetime:
    now = datetime.now(UTC)
    return now


# 功能：
#   限制 JSON 的类型、深度及大小，再按键排序生成供存储和摘要共用的文本。
# 输入：
#   value：待保存的标准 JSON 值。
# 输出：
#   canonical：不含非有限数值、最大 2 MiB 的确定性 JSON 文本。
def _canonical_json(value: object) -> str:
    encode_json(value, limit=_MAX_RESPONSE_BYTES)
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return canonical


# 功能：
#   为本地规范化载荷计算完整性摘要；摘要不代表签名或远端授权。
# 输入：
#   value：待绑定的 JSON 载荷。
# 输出：
#   digest：64 位十六进制 SHA-256。
def _sha256(value: object) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return digest


# 功能：
#   将受治理记忆键映射到云端同意类别，未知键不能默认取得聊天记忆权限。
# 输入：
#   memory_key：包含类型前缀的本地记忆键。
# 输出：
#   category：摘要、偏好或约束对应的授权类别。
def _memory_scope(memory_key: str) -> str:
    if (
        not isinstance(memory_key, str)
        or re.fullmatch(r"(summary|preference|constraint)\.[a-z][a-z0-9._-]{1,95}", memory_key)
        is None
    ):
        raise ValueError("MEMORY_PROJECTION_MEMORY_KEY_INVALID")
    kind = memory_key.split(".", 1)[0]
    category = {
        "summary": "reports_delivery",
        "preference": "chat_preferences",
        "constraint": "safety_approvals",
    }[kind]
    return category


# 功能：
#   拒绝缺少时区的远端时间，统一过期比较使用的 UTC 表示。
# 输入：
#   value：ISO 8601 时间字符串。
# 输出：
#   timestamp：已验证的 UTC 时间。
def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or len(value) > 80 or "\x00" in value:
        raise ValueError("MEMORY_PROJECTION_TIMESTAMP_INVALID")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("MEMORY_PROJECTION_TIMESTAMP_INVALID")
    timestamp = parsed.astimezone(UTC)
    return timestamp


@dataclass(frozen=True, slots=True)
class RemoteBoundary:
    user_id: str
    tenant_id: str
    organization_id: str


# 功能：
#   从经过本地验证的账户范围生成个人或组织租户边界，不能由请求任意指定其他租户。
# 输入：
#   scope：账户、组织及职责域。
# 输出：
#   boundary：供 RLS 查询和当前用户 RPC 使用的 UUID 边界。
def remote_boundary(scope: MemoryOwnerScope) -> RemoteBoundary:
    scope = MemoryOwnerScope.model_validate(scope.model_dump(mode="python"), strict=True)
    try:
        owner = str(UUID(scope.owner_account_id))
    except ValueError as error:
        raise ValueError("MEMORY_PROJECTION_OWNER_UUID_REQUIRED") from error
    if scope.organization_id in {None, "", _ZERO_UUID}:
        if scope.tenant_id not in {None, "", owner}:
            raise ValueError("MEMORY_PROJECTION_PERSONAL_TENANT_MISMATCH")
        return RemoteBoundary(owner, owner, _ZERO_UUID)
    try:
        organization = str(UUID(scope.organization_id))
    except ValueError as error:
        raise ValueError("MEMORY_PROJECTION_ORGANIZATION_UUID_REQUIRED") from error
    if scope.tenant_id != organization:
        raise ValueError("MEMORY_PROJECTION_ORGANIZATION_TENANT_MISMATCH")
    return RemoteBoundary(owner, organization, organization)


class MemoryProjectionCredentials(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)

    project_url: str = Field(min_length=20, max_length=300)
    publishable_key: str = Field(min_length=20, max_length=4_096, repr=False)
    access_token: str = Field(min_length=40, max_length=16_384, repr=False)
    timeout_seconds: float = Field(default=4.0, ge=0.5, le=10.0)

    # 功能：
    #   将凭证发送目标限制为 Supabase 项目的标准 HTTPS 源，拒绝用户信息和其他端口。
    # 输入：
    #   cls：凭证合同类型。
    #   value：项目 URL，不接受 REST 路径。
    # 输出：
    #   origin：去掉末尾斜线的项目源。
    @field_validator("project_url")
    @classmethod
    def validate_project_url(cls, value: str) -> str:
        if any(ord(character) <= 32 or ord(character) >= 127 for character in value):
            raise ValueError("MEMORY_PROJECTION_SUPABASE_URL_INVALID")
        parsed = urlparse(value)
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or not parsed.hostname
            or re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.supabase\.co", parsed.hostname)
            is None
            or parsed.port not in (None, 443)
            or parsed.path.rstrip("/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("MEMORY_PROJECTION_SUPABASE_URL_INVALID")
        origin = value.rstrip("/")
        return origin

    # 功能：
    #   只接受公开项目密钥，拒绝服务密钥；读取旧 JWT 的角色仅用于拒绝，不用于认证。
    # 输入：
    #   cls：凭证合同类型。
    #   value：公开密钥或旧式 anon JWT。
    # 输出：
    #   value：检查通过且未静默裁剪的密钥。
    @field_validator("publishable_key")
    @classmethod
    def validate_publishable_key(cls, value: str) -> str:
        if any(ord(character) <= 32 or ord(character) >= 127 for character in value):
            raise ValueError("MEMORY_PROJECTION_PUBLISHABLE_KEY_INVALID")
        if value.startswith("sb_secret_"):
            raise ValueError("MEMORY_PROJECTION_SECRET_KEY_FORBIDDEN")
        if value.startswith("sb_publishable_"):
            return value
        segments = value.split(".")
        if len(segments) != 3:
            raise ValueError("MEMORY_PROJECTION_PUBLIC_KEY_REQUIRED")
        try:
            raw = base64.b64decode(
                segments[1] + "=" * (-len(segments[1]) % 4), altchars=b"-_", validate=True
            )
            claims = decode_json(raw, limit=4_096)
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("MEMORY_PROJECTION_PUBLIC_KEY_REQUIRED") from error
        # This unverified payload is used only to reject authority-bearing or
        # unknown keys. It is never an authentication or authorization input.
        if not isinstance(claims, dict) or claims.get("role") != "anon":
            raise ValueError("MEMORY_PROJECTION_SERVICE_ROLE_KEY_FORBIDDEN")
        return value

    # 功能：
    #   验证 Bearer 头部可安全承载的三段 JWT 外形，真正身份验证仍由服务端执行。
    # 输入：
    #   cls：凭证合同类型。
    #   value：当前用户访问令牌。
    # 输出：
    #   value：不含空白、控制字符或空分段的令牌。
    @field_validator("access_token")
    @classmethod
    def validate_access_token(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", value) is None:
            raise ValueError("MEMORY_PROJECTION_ACCESS_TOKEN_INVALID")
        return value


class MemoryProjectionSyncStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ProjectionStatus
    pulled: int = Field(default=0, ge=0)
    pushed: int = Field(default=0, ge=0)
    tombstones_applied: int = Field(default=0, ge=0)
    pending_operations: int = Field(default=0, ge=0)
    conflicts: int = Field(default=0, ge=0)
    issues: tuple[str, ...] = ()


class ProjectionTransport(Protocol):
    # 功能：
    #   定义固定源的 JSON 请求接口，允许离线测试替换传输而不改变记忆治理。
    # 输入：
    #   self：传输实例。
    #   method：HTTP 方法；path：固定 REST 路径；query：查询参数。
    #   body：JSON 请求体；prefer：PostgREST 响应偏好。
    # 输出：
    #   response：远端 JSON 值或无响应体时的 None。
    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        prefer: str | None = None,
    ) -> object: ...


class ProjectionHttpError(RuntimeError):
    # 功能：
    #   保存经过清理的 HTTP 失败类别，禁止把远端自由文本回显进本地日志。
    # 输入：
    #   self：异常实例；status：HTTP 状态；code：受限错误标识。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, status: int, code: str) -> None:
        if not isinstance(code, str) or re.fullmatch(r"[A-Za-z0-9_]{1,80}", code) is None:
            code = "REMOTE_REJECTED"
        super().__init__(f"MEMORY_PROJECTION_HTTP_{status}:{code}")
        self.status = status
        self.code = code


class SupabaseProjectionTransport:
    # 功能：
    #   冻结并复核凭证，创建不继承系统代理、不跟随重定向的独立连接器。
    # 输入：
    #   self：传输实例；credentials：项目公开密钥与当前用户 JWT。
    #   deadline_monotonic：本次同步的单调截止时间，None 表示仅使用单请求预算。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        credentials: MemoryProjectionCredentials,
        *,
        deadline_monotonic: float | None = None,
    ) -> None:
        self.credentials = MemoryProjectionCredentials.model_validate(
            credentials.model_dump(mode="python"), strict=True
        )
        if deadline_monotonic is not None and (
            type(deadline_monotonic) not in (int, float) or not math.isfinite(deadline_monotonic)
        ):
            raise ValueError("MEMORY_PROJECTION_DEADLINE_INVALID")
        self.deadline_monotonic = deadline_monotonic
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _RejectProjectionRedirects()
        )

    # 功能：
    #   发送有界 JSON 并拒绝重定向、歧义 JSON 和截止后才完成的响应。
    # 输入：
    #   self：绑定当前同步预算的传输实例。
    #   method：GET 或 POST；path：项目内 REST 路径；query：短字符串查询。
    #   body：待发送的 JSON 对象；prefer：明确支持的 PostgREST 偏好。
    # 输出：
    #   response_value：预算内完成并严格解析的 JSON 值，空响应为 None。
    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        prefer: str | None = None,
    ) -> object:
        if not isinstance(path, str) or re.fullmatch(r"/rest/v1/[a-zA-Z0-9_/]+", path) is None:
            raise ValueError("MEMORY_PROJECTION_PATH_INVALID")
        if method not in ("GET", "POST"):
            raise ValueError("MEMORY_PROJECTION_METHOD_INVALID")
        if prefer not in (None, "resolution=merge-duplicates,return=representation"):
            raise ValueError("MEMORY_PROJECTION_PREFER_INVALID")
        if query is not None and (
            type(query) is not dict
            or len(query) > 24
            or any(
                type(key) is not str
                or type(value) is not str
                or len(key) > 80
                or len(value) > 4_096
                for key, value in query.items()
            )
        ):
            raise ValueError("MEMORY_PROJECTION_QUERY_INVALID")
        if body is not None and type(body) is not dict:
            raise ValueError("MEMORY_PROJECTION_BODY_INVALID")
        query_string = urllib.parse.urlencode(query or {}, safe="(),.*")
        url = f"{self.credentials.project_url}{path}"
        if query_string:
            url = f"{url}?{query_string}"
        payload = None if body is None else _canonical_json(body).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.credentials.access_token}",
            "apikey": self.credentials.publishable_key,
            "User-Agent": "DroneDream-AGENT/1",
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if prefer:
            headers["Prefer"] = prefer
        request = urllib.request.Request(  # noqa: S310
            url,
            data=payload,
            headers=headers,
            method=method,
        )
        deadline = min(
            time.monotonic() + self.credentials.timeout_seconds,
            self.deadline_monotonic if self.deadline_monotonic is not None else math.inf,
        )
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise TimeoutError("MEMORY_PROJECTION_SYNC_BUDGET_EXHAUSTED")
        try:
            with self._opener.open(request, timeout=timeout) as response:
                chunks: list[bytes] = []
                total = 0
                # read1 不等待凑满整帧；每个网络块后重新检查整体期限。
                # urllib 的 DNS/套接字调用不能被此循环硬抢占，因此超时结果必定丢弃，
                # 但此后台同步接口不承诺实时飞行线程的硬截止时间。
                while total <= _MAX_RESPONSE_BYTES:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("MEMORY_PROJECTION_SYNC_BUDGET_EXHAUSTED")
                    chunk = response.read1(min(65_536, _MAX_RESPONSE_BYTES + 1 - total))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                raw = b"".join(chunks)
        except urllib.error.HTTPError as error:
            # 不读取可能回显令牌的错误正文，也不让错误响应拖延同步预算。
            error.close()
            raise ProjectionHttpError(error.code, "REMOTE_REJECTED") from error
        except TimeoutError:
            raise
        except OSError as error:
            raise ConnectionError("MEMORY_PROJECTION_NETWORK_UNAVAILABLE") from error
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("MEMORY_PROJECTION_RESPONSE_TOO_LARGE")
        if time.monotonic() >= deadline:
            raise TimeoutError("MEMORY_PROJECTION_SYNC_BUDGET_EXHAUSTED")
        if not raw:
            return None
        try:
            response_value = decode_json(raw, limit=_MAX_RESPONSE_BYTES)
            return response_value
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("MEMORY_PROJECTION_RESPONSE_INVALID") from error


class _RejectProjectionRedirects(urllib.request.HTTPRedirectHandler):
    """A REST redirect is a failed request, never permission to forward a user JWT."""

    # 功能：
    #   阻止 urllib 对重定向重新发送当前用户凭证，包括同源重定向。
    # 输入：
    #   self：处理器；req、fp：原请求及响应流。
    #   code、msg、headers：重定向响应；newurl：建议跳转地址。
    # 输出：
    #   None：交给 urllib 报错，不生成后续请求。
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Returning None lets urllib turn every redirect into HTTPError. Even a
        # same-origin redirect is unnecessary for the fixed PostgREST endpoints.
        return None


class _ProjectionConsentChanged(RuntimeError):
    """A newer local consent generation invalidates the in-flight sync view."""


class _ConsentCheckedTransport:
    # 功能：
    #   给已完成授权前置操作的传输附加请求前后授权代次检查。
    # 输入：
    #   self：包装器；client：原传输；check：当前同步的授权及期限检查函数。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, client: ProjectionTransport, check: Callable[[], None]) -> None:
        self.client = client
        self.check = check

    # 功能：
    #   本地授权发生改变时中止后续读取/上传，已经在途的远端写入仍由远端 RLS 约束。
    # 输入：
    #   self：带代次保护的传输；method、path：固定操作与路径。
    #   query、body、prefer：原请求参数。
    # 输出：
    #   response：代次未变时的独立 JSON 响应。
    def request(self, method: str, path: str, *, query=None, body=None, prefer=None) -> object:
        self.check()
        response = self.client.request(method, path, query=query, body=body, prefer=prefer)
        self.check()
        response = copy_json(response, limit=_MAX_RESPONSE_BYTES)
        return response


class AccountMemoryProjectionBridge:
    """Durable outbox plus bounded pull for one shared account-memory database."""

    # 功能：
    #   在受治理记忆旁建立持久待发箱、远端映射与同步状态，不创建第二套授权来源。
    # 输入：
    #   self：同步桥；memory：本地受治理存储；path：可选独立状态库路径。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, memory: AccountMemoryStore, path: Path | None = None) -> None:
        self.memory = memory
        self.path = path or memory.path
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS account_memory_projection_outbox (
                  operation_id TEXT PRIMARY KEY,
                  owner_account_id TEXT NOT NULL,
                  tenant_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  namespace TEXT NOT NULL,
                  operation_kind TEXT NOT NULL CHECK(operation_kind IN (
                    'stage_candidate','resolve_candidate','forget','permanent_delete','consent'
                  )),
                  local_reference TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  payload_sha256 TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('pending','delivered','conflict')),
                  attempts INTEGER NOT NULL DEFAULT 0,
                  next_attempt_at TEXT NOT NULL,
                  last_error TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS account_memory_projection_outbox_pending_idx
                  ON account_memory_projection_outbox(status,next_attempt_at,created_at);
                CREATE TABLE IF NOT EXISTS account_memory_projection_mapping (
                  owner_account_id TEXT NOT NULL,
                  tenant_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  namespace TEXT NOT NULL,
                  local_reference TEXT NOT NULL,
                  remote_reference TEXT NOT NULL,
                  remote_payload_sha256 TEXT,
                  remote_revision INTEGER,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(
                    owner_account_id,tenant_id,organization_id,namespace,local_reference
                  )
                );
                CREATE TABLE IF NOT EXISTS account_memory_projection_state (
                  owner_account_id TEXT NOT NULL,
                  tenant_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  namespace TEXT NOT NULL,
                  remote_memory_id TEXT NOT NULL,
                  remote_payload_sha256 TEXT NOT NULL,
                  remote_revision INTEGER NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('accepted','conflict','tombstoned')),
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(
                    owner_account_id,tenant_id,organization_id,namespace,remote_memory_id
                  )
                );
                CREATE TABLE IF NOT EXISTS account_memory_projection_consent (
                  owner_account_id TEXT NOT NULL,
                  tenant_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  payload_sha256 TEXT NOT NULL,
                  observed_at TEXT NOT NULL,
                  PRIMARY KEY(owner_account_id,tenant_id,organization_id)
                );
                """
            )

    # 功能：
    #   为短数据库操作持有事务，成功提交、异常回滚，并始终关闭连接。
    # 输入：
    #   self：保存状态库路径的同步桥。
    # 输出：
    #   connection：仅在 with 作用域内有效的 SQLite 连接。
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=15000")
            with connection:
                yield connection
        finally:
            # sqlite3.Connection.__exit__ handles transactions, not connection lifetime.
            connection.close()

    # 功能：
    #   得到完整本地范围，所有待发和映射查询必须包含此四元组。
    # 输入：
    #   scope：账户、租户、组织及职责域。
    # 输出：
    #   boundary：严格验证的 SQL 绑定参数。
    @staticmethod
    def _local_boundary(scope: MemoryOwnerScope) -> tuple[str, str, str, str]:
        boundary = scope.boundary_key()
        return boundary

    # 功能：
    #   网络调用前持久保存操作，重复意图幂等；替换仅限同范围、同类型的既有操作。
    # 输入：
    #   self：同步桥；scope：完整账户范围；operation_id：幂等操作标识。
    #   operation_kind：操作类型；local_reference：本地目标；payload：待发 JSON。
    #   replace：是否以新授权代次取代同一操作。
    # 输出：
    #   operation_id：持久化的操作标识。
    def _queue(
        self,
        scope: MemoryOwnerScope,
        *,
        operation_id: str,
        operation_kind: str,
        local_reference: str,
        payload: dict[str, Any],
        replace: bool = False,
    ) -> str:
        owner, tenant, organization, namespace = self._local_boundary(scope)
        now = _now().isoformat()
        canonical = _canonical_json(payload)
        values = (
            operation_id,
            owner,
            tenant,
            organization,
            namespace,
            operation_kind,
            local_reference,
            canonical,
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "pending",
            now,
            now,
            now,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM account_memory_projection_outbox WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if existing is not None and (
                tuple(
                    existing[key]
                    for key in (
                        "owner_account_id",
                        "tenant_id",
                        "organization_id",
                        "namespace",
                        "operation_kind",
                        "local_reference",
                    )
                )
                != (owner, tenant, organization, namespace, operation_kind, local_reference)
                or (not replace and existing["payload_sha256"] != values[8])
            ):
                raise ValueError("MEMORY_PROJECTION_OPERATION_ID_CONFLICT")
            if replace:
                connection.execute(
                    """INSERT INTO account_memory_projection_outbox(
                         operation_id,owner_account_id,tenant_id,organization_id,namespace,
                         operation_kind,local_reference,payload_json,payload_sha256,status,
                         next_attempt_at,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(operation_id) DO UPDATE SET
                         payload_json=excluded.payload_json,
                         payload_sha256=excluded.payload_sha256,status='pending',attempts=0,
                         next_attempt_at=excluded.next_attempt_at,last_error=NULL,
                         updated_at=excluded.updated_at""",
                    values,
                )
            else:
                connection.execute(
                    """INSERT OR IGNORE INTO account_memory_projection_outbox(
                         operation_id,owner_account_id,tenant_id,organization_id,namespace,
                         operation_kind,local_reference,payload_json,payload_sha256,status,
                         next_attempt_at,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    values,
                )
        return operation_id

    # 功能：
    #   为已有安全候选构造稳定来源回执并排队，不把候选直接提升为有效记忆。
    # 输入：
    #   self：同步桥；scope：候选所属账户范围；candidate_id：本地候选标识。
    # 输出：
    #   operation_id：候选上传操作标识。
    def queue_candidate(self, scope: MemoryOwnerScope, candidate_id: str) -> str:
        candidate = self.memory.projection_candidate(scope, candidate_id)
        run_id = str(uuid5(NAMESPACE_URL, f"dronedream:memory-run:{candidate_id}"))
        conversation_id = str(
            uuid5(
                NAMESPACE_URL,
                "dronedream:conversation:"
                f"{scope.owner_account_id}:{candidate['source_conversation_id']}",
            )
        )
        source = {
            "candidate_id": candidate_id,
            "memory_key": candidate["memory_key"],
            "payload": candidate["payload"],
            "conversation_id": conversation_id,
            "run_id": run_id,
        }
        source_hash = _sha256(source)
        payload = {
            "p_responsibility_namespace": scope.namespace,
            "p_scope": _memory_scope(str(candidate["memory_key"])),
            "p_memory_key": candidate["memory_key"],
            "p_memory_kind": "structured_state",
            "p_payload": candidate["payload"],
            "p_source_edition": candidate.get("source_edition") or "autonomy",
            "p_source_workspace_id": "console-autonomy",
            "p_conversation_id": conversation_id,
            "p_run_id": run_id,
            "p_source_receipt_id": f"projection.{source_hash[:48]}",
            "p_source_receipt_sha256": source_hash,
            "p_source_metadata": {
                "local_candidate_id": candidate_id,
                "local_source_kind": candidate.get("source_kind") or "model_inference",
            },
            "p_retrieval_metadata": {"projection_contract": "account-memory.v1"},
            "p_evidence_sha256": _sha256(
                {"payload": candidate["payload"], "conversation_id": conversation_id}
            ),
            "p_confidence": candidate["confidence"],
        }
        return self._queue(
            scope,
            operation_id=f"projection-stage-{candidate_id.removeprefix('memory-candidate-')}",
            operation_kind="stage_candidate",
            local_reference=candidate_id,
            payload=payload,
        )

    # 功能：
    #   为当前账户确有的候选排队显式批准或拒绝，实际发送须先有远端候选映射。
    # 输入：
    #   self：同步桥；scope：账户范围；candidate_id：候选标识；approve：明确布尔决定。
    # 输出：
    #   operation_id：决策同步操作标识。
    def queue_resolution(self, scope: MemoryOwnerScope, candidate_id: str, *, approve: bool) -> str:
        if type(approve) is not bool:
            raise ValueError("MEMORY_PROJECTION_RESOLUTION_INVALID")
        try:
            self.memory.projection_candidate(scope, candidate_id)
        except KeyError as error:
            raise ValueError("MEMORY_PROJECTION_CANDIDATE_UNAVAILABLE") from error
        suffix = "promote" if approve else "reject"
        return self._queue(
            scope,
            operation_id=f"projection-resolve-{candidate_id[-32:]}-{suffix}",
            operation_kind="resolve_candidate",
            local_reference=candidate_id,
            payload={"p_resolution": suffix},
        )

    # 功能：
    #   按用户选择保存软撤回或永久删除意图，非法模式不能升级为永久删除。
    # 输入：
    #   self：同步桥；scope：账户范围；memory_key：受治理键；mode：明确删除方式。
    # 输出：
    #   operation_id：删除操作标识。
    def queue_forget(
        self,
        scope: MemoryOwnerScope,
        memory_key: str,
        *,
        mode: Literal["soft", "permanent"],
    ) -> str:
        if mode not in ("soft", "permanent"):
            raise ValueError("MEMORY_PROJECTION_FORGET_MODE_INVALID")
        operation_kind = "forget" if mode == "soft" else "permanent_delete"
        return self._queue(
            scope,
            operation_id=f"projection-forget-{uuid4().hex}",
            operation_kind=operation_kind,
            local_reference=memory_key,
            payload={
                "p_responsibility_namespace": scope.namespace,
                "p_scope": _memory_scope(memory_key),
                "p_memory_key": memory_key,
            },
        )

    # 功能：
    #   用稳定操作标识合并最新记忆同意设置，不生成飞行或执行权限。
    # 输入：
    #   self：同步桥；scope：账户范围；enabled：用户明确选择的布尔开关。
    # 输出：
    #   operation_id：本地保存的授权设置代次所属操作。
    def queue_consent(self, scope: MemoryOwnerScope, *, enabled: bool) -> str:
        if type(enabled) is not bool:
            raise ValueError("MEMORY_PROJECTION_CONSENT_INVALID")
        boundary = remote_boundary(scope)
        namespaces = list(_READ_NAMESPACES) if enabled else []
        scopes = {name: enabled for name in _SCOPES}
        local_scope_hash = _sha256(list(self._local_boundary(scope)))[:32]
        return self._queue(
            scope,
            operation_id=f"projection-consent-{local_scope_hash}",
            operation_kind="consent",
            local_reference="account-consent",
            payload={
                "user_id": boundary.user_id,
                "tenant_id": boundary.tenant_id,
                "organization_id": boundary.organization_id,
                "memory_enabled": enabled,
                "read_namespaces": namespaces,
                "write_namespaces": namespaces,
                "memory_scopes": scopes,
            },
            replace=True,
        )

    # 功能：
    #   统计指定账户职责域尚待发送的操作，不计入别的账户。
    # 输入：
    #   self：同步桥；scope：待查询的完整范围。
    # 输出：
    #   count：待发操作数量。
    def pending_count(self, scope: MemoryOwnerScope) -> int:
        owner, tenant, organization, namespace = self._local_boundary(scope)
        with self._connect() as connection:
            return int(
                connection.execute(
                    """SELECT COUNT(*) FROM account_memory_projection_outbox
                         WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                           AND namespace=? AND status='pending'""",
                    (owner, tenant, organization, namespace),
                ).fetchone()[0]
            )

    # 功能：
    #   统计需要处理的非重试失败，失败操作不能假装已送达。
    # 输入：
    #   self：同步桥；scope：待查询的完整范围。
    # 输出：
    #   count：冲突操作数量。
    def conflict_count(self, scope: MemoryOwnerScope) -> int:
        owner, tenant, organization, namespace = self._local_boundary(scope)
        with self._connect() as connection:
            return int(
                connection.execute(
                    """SELECT COUNT(*) FROM account_memory_projection_outbox
                         WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                           AND namespace=? AND status='conflict'""",
                    (owner, tenant, organization, namespace),
                ).fetchone()[0]
            )

    # 功能：
    #   1. 优先发送撤回、删除和同意变更，再依据新读取的云端同意同步候选与条目。
    #   2. 任一失败、未完整读取删除记录或并发同意变更都不能报告完整同步成功。
    # 输入：
    #   self：同步桥；scope：账户范围；credentials：公开项目凭证和用户 JWT，可缺省。
    #   local_memory_enabled：本地开关；operation_limit：本次最多发送的操作数。
    #   pull_limit：每个命名空间最多读取条目数；time_budget_seconds：总同步软截止预算。
    #   transport：可替换的传输实现，仅用于受控集成或离线测试。
    # 输出：
    #   status：同步结果、真实处理数量及受限错误标识。
    def sync(
        self,
        scope: MemoryOwnerScope,
        credentials: MemoryProjectionCredentials | None,
        *,
        local_memory_enabled: bool,
        operation_limit: int = 8,
        pull_limit: int = 64,
        time_budget_seconds: float = 3.0,
        transport: ProjectionTransport | None = None,
    ) -> MemoryProjectionSyncStatus:
        if type(local_memory_enabled) is not bool:
            raise ValueError("MEMORY_PROJECTION_LOCAL_CONSENT_INVALID")
        if (
            type(operation_limit) is not int
            or not 1 <= operation_limit <= 32
            or type(pull_limit) is not int
            or not 1 <= pull_limit <= 256
            or type(time_budget_seconds) not in (int, float)
            or not 0.5 <= time_budget_seconds <= 10.0
        ):
            raise ValueError("MEMORY_PROJECTION_BUDGET_INVALID")
        if credentials is None:
            return MemoryProjectionSyncStatus(
                status=("configuration_missing" if local_memory_enabled else "local_disabled"),
                pending_operations=self.pending_count(scope),
                conflicts=self.conflict_count(scope),
            )
        boundary = remote_boundary(scope)
        deadline = time.monotonic() + time_budget_seconds
        client = transport or SupabaseProjectionTransport(
            credentials,
            deadline_monotonic=deadline,
        )
        issues: list[str] = []
        pulled = pushed = applied = conflicts = 0
        consent: dict[str, Any] | None = None
        try:
            delivered, attempted = self._flush_preconsent_operations(
                scope, boundary, client, operation_limit, issues
            )
            pushed += delivered
            if not local_memory_enabled:
                return MemoryProjectionSyncStatus(
                    status="local_disabled",
                    pushed=pushed,
                    pending_operations=self.pending_count(scope),
                    conflicts=self.conflict_count(scope),
                    issues=tuple(issues[:32]),
                )
            # A user may change consent while the preceding network call is in flight.
            # Do not read or upload memory using its obsolete enable acknowledgement.
            if self._has_unsettled_consent(scope):
                outstanding_conflicts = self.conflict_count(scope)
                return MemoryProjectionSyncStatus(
                    status="conflict" if outstanding_conflicts else "pending",
                    pushed=pushed,
                    pending_operations=self.pending_count(scope),
                    conflicts=outstanding_conflicts,
                    issues=tuple(issues[:32]),
                )
            generation = self._consent_generation(scope)

            # 功能：
            #   每次网络请求前后复核账户全部职责域的授权代次和同步期限。
            # 输入：
            #   无；闭包绑定本次 scope、generation 和 deadline。
            # 输出：
            #   None：未变更且预算有效时继续，否则抛出中止信号。
            def check_consent() -> None:
                if self._consent_generation(scope) != generation:
                    raise _ProjectionConsentChanged("MEMORY_PROJECTION_CONSENT_CHANGED")
                if time.monotonic() >= deadline:
                    raise TimeoutError("MEMORY_PROJECTION_SYNC_BUDGET_EXHAUSTED")

            client = _ConsentCheckedTransport(client, check_consent)
            consent = self._pull_consent(scope, boundary, client)
            applied += self._pull_tombstones(scope, boundary, client)
            read_allowed = bool(
                consent
                and consent.get("memory_enabled") is True
                and any(
                    namespace in consent.get("read_namespaces", [])
                    for namespace in _READ_NAMESPACES
                )
            )
            write_allowed = bool(
                consent
                and consent.get("memory_enabled") is True
                and scope.namespace in consent.get("write_namespaces", [])
            )
            enabled_scopes = consent.get("memory_scopes", {}) if consent else {}
            if read_allowed:
                pulled, pull_conflicts = self._pull_records(
                    scope,
                    boundary,
                    client,
                    pull_limit,
                    enabled_scopes,
                    issues,
                    read_namespaces=consent["read_namespaces"],
                )
                conflicts += pull_conflicts
            pushed += self._flush_memory_operations(
                scope,
                boundary,
                client,
                operation_limit - attempted,
                write_allowed=write_allowed,
                enabled_scopes=enabled_scopes,
                issues=issues,
            )
        except _ProjectionConsentChanged:
            return MemoryProjectionSyncStatus(
                status="pending",
                pulled=pulled,
                pushed=pushed,
                tombstones_applied=applied,
                pending_operations=self.pending_count(scope),
                conflicts=conflicts,
                issues=tuple([*issues, "MEMORY_PROJECTION_CONSENT_CHANGED"][:32]),
            )
        except (ConnectionError, TimeoutError):
            return MemoryProjectionSyncStatus(
                status="network_unavailable",
                pulled=pulled,
                pushed=pushed,
                tombstones_applied=applied,
                pending_operations=self.pending_count(scope),
                conflicts=conflicts,
                issues=tuple([*issues, "MEMORY_PROJECTION_NETWORK_UNAVAILABLE"][:32]),
            )
        except (ProjectionHttpError, ValueError, TypeError, sqlite3.Error):
            issues.append("MEMORY_PROJECTION_SYNC_REJECTED")
            conflicts += 1
        pending = self.pending_count(scope)
        conflicts += self.conflict_count(scope)
        if conflicts:
            status: ProjectionStatus = "conflict"
        elif consent is None:
            status = "consent_required"
        elif consent.get("memory_enabled") is not True:
            status = "remote_disabled"
        elif pending:
            status = "pending"
        else:
            status = "synced"
        return MemoryProjectionSyncStatus(
            status=status,
            pulled=pulled,
            pushed=pushed,
            tombstones_applied=applied,
            pending_operations=pending,
            conflicts=conflicts,
            issues=tuple(issues[:32]),
        )

    # 功能：
    #   为 REST 查询附加不可覆盖的账户过滤；服务器仍必须独立执行 RLS。
    # 输入：
    #   self：同步桥；boundary：远端 UUID 边界；extra：条目筛选及分页条件。
    # 输出：
    #   query：包含完整账户条件的查询字典。
    def _query(self, boundary: RemoteBoundary, **extra: str) -> dict[str, str]:
        if set(extra).intersection({"user_id", "tenant_id", "organization_id"}):
            raise ValueError("MEMORY_PROJECTION_QUERY_BOUNDARY_OVERRIDE")
        query = {
            "user_id": f"eq.{boundary.user_id}",
            "tenant_id": f"eq.{boundary.tenant_id}",
            "organization_id": f"eq.{boundary.organization_id}",
            **extra,
        }
        return query

    # 功能：
    #   读取当前云端同意并校验布尔和命名空间；本地缓存只用于诊断，不能离线授予权限。
    # 输入：
    #   self：同步桥；scope：本地账户范围；boundary：远端边界；client：受控传输。
    # 输出：
    #   consent：独立授权字典；远端无配置时为 None。
    def _pull_consent(
        self,
        scope: MemoryOwnerScope,
        boundary: RemoteBoundary,
        client: ProjectionTransport,
    ) -> dict[str, Any] | None:
        response = client.request(
            "GET",
            "/rest/v1/console_memory_consents",
            query=self._query(
                boundary,
                select=("memory_enabled,read_namespaces,write_namespaces,memory_scopes,updated_at"),
                limit="1",
            ),
        )
        if not isinstance(response, list) or len(response) > 1:
            raise ValueError("MEMORY_PROJECTION_CONSENT_RESPONSE_INVALID")
        if not response:
            return None
        consent = copy_json(response[0], limit=_MAX_RESPONSE_BYTES)
        if (
            not isinstance(consent, dict)
            or not isinstance(consent.get("memory_enabled"), bool)
            or not isinstance(consent.get("read_namespaces"), list)
            or not isinstance(consent.get("write_namespaces"), list)
            or not isinstance(consent.get("memory_scopes"), dict)
            or any(
                type(item) is not str
                for key in ("read_namespaces", "write_namespaces")
                for item in consent.get(key, [])
            )
            or any(type(flag) is not bool for flag in consent.get("memory_scopes", {}).values())
        ):
            raise ValueError("MEMORY_PROJECTION_CONSENT_RESPONSE_INVALID")
        if any(item not in _READ_NAMESPACES for item in consent["read_namespaces"]):
            # Other product domains may be present; retain only the local bridge domains.
            consent["read_namespaces"] = [
                item for item in consent["read_namespaces"] if item in _READ_NAMESPACES
            ]
        consent["write_namespaces"] = [
            item for item in consent["write_namespaces"] if item in _READ_NAMESPACES
        ]
        owner, tenant, organization, _ = self._local_boundary(scope)
        canonical = _canonical_json(consent)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO account_memory_projection_consent(
                     owner_account_id,tenant_id,organization_id,payload_json,
                     payload_sha256,observed_at
                   ) VALUES(?,?,?,?,?,?)
                   ON CONFLICT(owner_account_id,tenant_id,organization_id) DO UPDATE SET
                     payload_json=excluded.payload_json,
                     payload_sha256=excluded.payload_sha256,
                     observed_at=excluded.observed_at""",
                (
                    owner,
                    tenant,
                    organization,
                    canonical,
                    hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                    _now().isoformat(),
                ),
            )
        return consent

    # 功能：
    #   完整分页读取当前删除标记后再应用本地删除，不能用被截断的首批结果继续导入记忆。
    # 输入：
    #   self：同步桥；scope：本地账户；boundary：远端边界；client：带截止预算的传输。
    # 输出：
    #   applied：确实删除本地内容或候选的记忆键数量。
    def _pull_tombstones(
        self,
        scope: MemoryOwnerScope,
        boundary: RemoteBoundary,
        client: ProjectionTransport,
    ) -> int:
        response: list[dict[str, Any]] = []
        cursor: str | None = None
        # 主键游标不受前页条目被释放/删除的偏移变化影响。最多 16 页；未读完即中止。
        # 多次 HTTP 请求不是数据库快照，并发新删除仍须由后续同步及远端事务门控处理。
        for _page in range(16):
            page = client.request(
                "GET",
                "/rest/v1/console_memory_deletion_tombstones",
                query=self._query(
                    boundary,
                    select=(
                        "tombstone_id,responsibility_namespace,scope,memory_key_sha256,"
                        "released_at,deleted_at"
                    ),
                    released_at="is.null",
                    order="tombstone_id.asc",
                    limit="256",
                    **({"tombstone_id": f"gt.{cursor}"} if cursor is not None else {}),
                ),
            )
            if not isinstance(page, list) or len(page) > 256:
                raise ValueError("MEMORY_PROJECTION_TOMBSTONE_RESPONSE_INVALID")
            for raw in page:
                if not isinstance(raw, dict) or not isinstance(raw.get("tombstone_id"), str):
                    raise ValueError("MEMORY_PROJECTION_TOMBSTONE_RESPONSE_INVALID")
                identifier = str(UUID(raw["tombstone_id"]))
                if cursor is not None and identifier <= cursor:
                    raise ValueError("MEMORY_PROJECTION_TOMBSTONE_CURSOR_INVALID")
                if "released_at" not in raw or raw["released_at"] is not None:
                    raise ValueError("MEMORY_PROJECTION_TOMBSTONE_RELEASED")
                _parse_timestamp(raw.get("deleted_at"))
                cursor = identifier
                response.append(raw)
            if len(page) < 256:
                break
        else:
            raise ValueError("MEMORY_PROJECTION_TOMBSTONE_SCAN_INCOMPLETE")
        applied = 0
        namespaces = _READ_NAMESPACES
        for raw in response:
            if not isinstance(raw, dict):
                raise ValueError("MEMORY_PROJECTION_TOMBSTONE_RESPONSE_INVALID")
            remote_namespace = raw.get("responsibility_namespace")
            if remote_namespace == "account.all":
                selected_namespaces = namespaces
            elif remote_namespace in namespaces:
                selected_namespaces = (str(remote_namespace),)
            else:
                continue
            remote_scope = raw.get("scope")
            key_hash = raw.get("memory_key_sha256")
            if key_hash is not None and (
                not isinstance(key_hash, str) or _SHA256.fullmatch(key_hash) is None
            ):
                raise ValueError("MEMORY_PROJECTION_TOMBSTONE_RESPONSE_INVALID")
            for namespace in selected_namespaces:
                projected_scope = MemoryOwnerScope(
                    owner_account_id=scope.owner_account_id,
                    tenant_id=scope.tenant_id,
                    organization_id=scope.organization_id,
                    namespace=namespace,
                    source_edition=scope.source_edition,
                )
                for memory_key in self.memory.projection_keys(projected_scope):
                    if remote_scope is not None and _memory_scope(memory_key) != remote_scope:
                        continue
                    if (
                        key_hash is not None
                        and hashlib.sha256(memory_key.encode("utf-8")).hexdigest() != key_hash
                    ):
                        continue
                    result = self.memory.forget(
                        projected_scope,
                        memory_key,
                        mode="permanent",
                        reason="remote_deletion_tombstone",
                    )
                    if int(result["active_deleted"]) or int(result["candidates_deleted"]):
                        applied += 1
        return applied

    # 功能：
    #   只拉取同意读取的职责域和类别，逐条验证后导入；单条异常不隐藏其他条目的结果。
    # 输入：
    #   self：同步桥；scope、boundary：本地与远端账户边界；client：受控传输。
    #   limit：每个职责域条目上限；enabled_scopes：类别开关；issues：诊断收集列表。
    #   read_namespaces：云端明确允许读取的职责域。
    # 输出：
    #   result：成功新增条目数和拒绝/冲突条目数。
    def _pull_records(
        self,
        scope: MemoryOwnerScope,
        boundary: RemoteBoundary,
        client: ProjectionTransport,
        limit: int,
        enabled_scopes: object,
        issues: list[str],
        *,
        read_namespaces: list[str],
    ) -> tuple[int, int]:
        if not isinstance(enabled_scopes, dict):
            raise ValueError("MEMORY_PROJECTION_CONSENT_SCOPES_INVALID")
        imported = conflicts = 0
        for namespace in _READ_NAMESPACES:
            if namespace not in read_namespaces:
                continue
            response = client.request(
                "GET",
                "/rest/v1/console_memory_records",
                query=self._query(
                    boundary,
                    responsibility_namespace=f"eq.{namespace}",
                    status="eq.active",
                    select=(
                        "memory_id,responsibility_namespace,scope,memory_key,payload,"
                        "payload_sha256,source_version,projection_revision,edition,"
                        "conversation_id,evidence_count,confidence,last_seen,expires_at"
                    ),
                    order="updated_at.desc",
                    limit=str(limit),
                ),
            )
            if not isinstance(response, list) or len(response) > limit:
                raise ValueError("MEMORY_PROJECTION_RECORD_RESPONSE_INVALID")
            for raw in response:
                if not isinstance(raw, dict):
                    raise ValueError("MEMORY_PROJECTION_RECORD_RESPONSE_INVALID")
                try:
                    remote_scope = raw.get("scope")
                    if not isinstance(remote_scope, str):
                        raise ValueError("MEMORY_PROJECTION_RECORD_SCOPE_INVALID")
                    if raw.get("responsibility_namespace") != namespace:
                        raise ValueError("MEMORY_PROJECTION_RECORD_NAMESPACE_MISMATCH")
                    if enabled_scopes.get(remote_scope) is not True:
                        continue
                    if _memory_scope(raw.get("memory_key")) != remote_scope:
                        raise ValueError("MEMORY_PROJECTION_RECORD_CATEGORY_MISMATCH")
                    outcome = self._import_record(scope, namespace, raw)
                except (ValueError, TypeError, sqlite3.Error):
                    if len(issues) < 32:
                        issues.append("MEMORY_PROJECTION_RECORD_REJECTED")
                    conflicts += 1
                    continue
                if outcome == "imported":
                    imported += 1
                elif outcome in {"conflict", "tombstoned"}:
                    conflicts += 1
        return imported, conflicts

    # 功能：
    #   拒绝远端修订回退和同代摘要冲突，再将内容交给本地安全合同与删除标记检查。
    # 输入：
    #   self：同步桥；scope：本地账户；namespace：已获同意的职责域；raw：远端条目。
    # 输出：
    #   outcome：新增、不变、冲突或删除标记阻止。
    def _import_record(
        self,
        scope: MemoryOwnerScope,
        namespace: str,
        raw: dict[str, Any],
    ) -> str:
        memory_id = str(UUID(str(raw.get("memory_id"))))
        memory_key = raw.get("memory_key")
        payload = raw.get("payload")
        remote_hash = raw.get("payload_sha256")
        revision = raw.get("projection_revision")
        if (
            not isinstance(memory_key, str)
            or not isinstance(payload, dict)
            or not isinstance(remote_hash, str)
            or _SHA256.fullmatch(remote_hash) is None
            or type(revision) is not int
            or revision < 1
            or revision > 9_223_372_036_854_775_807
            or type(raw.get("evidence_count")) is not int
            or type(raw.get("confidence")) not in (int, float)
            or raw.get("edition") not in (None, "universal", "sim", "lab", "field", "autonomy")
        ):
            raise ValueError("MEMORY_PROJECTION_RECORD_RESPONSE_INVALID")
        projected_scope = MemoryOwnerScope(
            owner_account_id=scope.owner_account_id,
            tenant_id=scope.tenant_id,
            organization_id=scope.organization_id,
            namespace=namespace,
            source_edition=scope.source_edition,
        )
        owner, tenant, organization, _ = self._local_boundary(projected_scope)
        with self._connect() as connection:
            state = connection.execute(
                """SELECT remote_payload_sha256,remote_revision
                     FROM account_memory_projection_state
                    WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                      AND namespace=? AND remote_memory_id=?""",
                (owner, tenant, organization, namespace, memory_id),
            ).fetchone()
        if state is not None and (
            revision < int(state["remote_revision"])
            or (
                revision == int(state["remote_revision"])
                and remote_hash != str(state["remote_payload_sha256"])
            )
        ):
            self._record_projection_state(
                projected_scope, memory_id, remote_hash, revision, "conflict"
            )
            return "conflict"
        conversation_id = raw.get("conversation_id") or memory_id
        outcome = self.memory.import_projected_entry(
            projected_scope,
            memory_key=memory_key,
            payload=payload,
            source_conversation_id=str(UUID(str(conversation_id))),
            source_edition=raw.get("edition"),
            evidence_count=raw["evidence_count"],
            confidence=raw["confidence"],
            last_observed_at=_parse_timestamp(raw.get("last_seen")),
            expires_at=_parse_timestamp(raw.get("expires_at")),
            remote_memory_id=memory_id,
            remote_payload_sha256=remote_hash,
            remote_revision=revision,
            projection_authority=self.memory._projection_authority,
        )
        self._record_projection_state(
            projected_scope,
            memory_id,
            remote_hash,
            revision,
            "accepted" if outcome in {"imported", "unchanged"} else outcome,
        )
        return outcome

    # 功能：
    #   保存远端修订高水位；后续旧响应即使标记冲突，也不能降低已见修订或替换其摘要。
    # 输入：
    #   self：同步桥；scope：账户范围；remote_memory_id：远端条目标识。
    #   remote_hash：服务端 JSONB 载荷摘要；revision：远端修订号；status：治理结果。
    # 输出：
    #   None：不返回业务数据。
    def _record_projection_state(
        self,
        scope: MemoryOwnerScope,
        remote_memory_id: str,
        remote_hash: str,
        revision: int,
        status: str,
    ) -> None:
        owner, tenant, organization, namespace = self._local_boundary(scope)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO account_memory_projection_state(
                     owner_account_id,tenant_id,organization_id,namespace,
                     remote_memory_id,remote_payload_sha256,remote_revision,status,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(
                     owner_account_id,tenant_id,organization_id,namespace,remote_memory_id
                   )
                   DO UPDATE SET remote_payload_sha256=CASE
                       WHEN excluded.remote_revision >
                            account_memory_projection_state.remote_revision
                       THEN excluded.remote_payload_sha256
                       ELSE account_memory_projection_state.remote_payload_sha256 END,
                     remote_revision=MAX(account_memory_projection_state.remote_revision,
                                         excluded.remote_revision),status=excluded.status,
                     updated_at=excluded.updated_at""",
                (
                    owner,
                    tenant,
                    organization,
                    namespace,
                    remote_memory_id,
                    remote_hash,
                    revision,
                    status,
                    _now().isoformat(),
                ),
            )

    # 功能：
    #   有界读取已到重试时间的待发快照，后续确认必须再次比对快照代次。
    # 输入：
    #   self：同步桥；scope：账户范围；kinds：指定操作种类；limit：剩余操作预算。
    # 输出：
    #   rows：按创建时间和操作标识排序的待发快照。
    def _pending_rows(
        self, scope: MemoryOwnerScope, kinds: tuple[str, ...], limit: int
    ) -> list[sqlite3.Row]:
        if limit <= 0:
            return []
        owner, tenant, organization, namespace = self._local_boundary(scope)
        placeholders = ",".join("?" for _ in kinds)
        with self._connect() as connection:
            return connection.execute(
                f"""SELECT * FROM account_memory_projection_outbox
                      WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                        AND namespace=? AND status='pending' AND next_attempt_at<=?
                        AND operation_kind IN ({placeholders})
                      ORDER BY created_at ASC,operation_id ASC LIMIT ?""",  # noqa: S608
                (
                    owner,
                    tenant,
                    organization,
                    namespace,
                    _now().isoformat(),
                    *kinds,
                    limit,
                ),
            ).fetchall()

    # 功能：
    #   优先投递同意设置和删除意图；失败或等待映射的尝试仍计入发送预算。
    # 输入：
    #   self：同步桥；scope、boundary：账户边界；client：传输；limit：操作上限。
    #   issues：诊断收集列表。
    # 输出：
    #   result：成功确认数与实际尝试数。
    def _flush_preconsent_operations(
        self,
        scope: MemoryOwnerScope,
        boundary: RemoteBoundary,
        client: ProjectionTransport,
        limit: int,
        issues: list[str],
    ) -> tuple[int, int]:
        delivered = 0
        rows = self._pending_rows(scope, ("consent", "forget", "permanent_delete"), limit)
        for row in rows:
            if self._deliver_row(scope, boundary, client, row, issues):
                delivered += 1
        return delivered, len(rows)

    # 功能：
    #   写入同意有效时发送候选及显式决策，损坏或无法确定授权类别的操作不能默认放行。
    # 输入：
    #   self：同步桥；scope、boundary：账户边界；client：传输；limit：剩余预算。
    #   write_allowed：职责域写权限；enabled_scopes：类别开关；issues：诊断列表。
    # 输出：
    #   delivered：实际确认成功的操作数。
    def _flush_memory_operations(
        self,
        scope: MemoryOwnerScope,
        boundary: RemoteBoundary,
        client: ProjectionTransport,
        limit: int,
        *,
        write_allowed: bool,
        enabled_scopes: object,
        issues: list[str],
    ) -> int:
        if limit <= 0 or not write_allowed or not isinstance(enabled_scopes, dict):
            return 0
        delivered = 0
        for row in self._pending_rows(scope, ("stage_candidate", "resolve_candidate"), limit):
            try:
                payload = self._outbox_payload(row)
                remote_scope = payload.get("p_scope")
                if str(row["operation_kind"]) == "resolve_candidate":
                    remote_scope = self._candidate_remote_scope(scope, str(row["local_reference"]))
                if not isinstance(remote_scope, str):
                    raise ValueError("MEMORY_PROJECTION_OPERATION_SCOPE_INVALID")
            except (KeyError, TypeError, ValueError):
                self._defer(row, "MEMORY_PROJECTION_OUTBOX_CORRUPT", retry=False)
                if len(issues) < 32:
                    issues.append("MEMORY_PROJECTION_OUTBOX_CORRUPT")
                continue
            if enabled_scopes.get(remote_scope) is not True:
                self._defer(row, "MEMORY_PROJECTION_SCOPE_CONSENT_REQUIRED", retry=False)
                continue
            if self._deliver_row(scope, boundary, client, row, issues):
                delivered += 1
        return delivered

    # 功能：
    #   从原候选恢复决策所需的授权类别，候选已失效或不属于账户时拒绝猜测。
    # 输入：
    #   self：同步桥；scope：账户范围；local_reference：原候选标识。
    # 输出：
    #   category：该候选的云端同意类别。
    def _candidate_remote_scope(self, scope: MemoryOwnerScope, local_reference: str) -> str:
        candidate = self.memory.projection_candidate(scope, local_reference)
        category = _memory_scope(str(candidate["memory_key"]))
        return category

    # 功能：
    #   在发出任何待发请求前验证存储载荷与其摘要一致，拒绝重复 JSON 键和非有限数字。
    # 输入：
    #   row：当前操作的数据库快照。
    # 输出：
    #   payload：验证通过的独立 JSON 对象。
    @staticmethod
    def _outbox_payload(row: sqlite3.Row) -> dict[str, Any]:
        payload = decode_json(row["payload_json"], limit=_MAX_RESPONSE_BYTES)
        if not isinstance(payload, dict) or _sha256(payload) != row["payload_sha256"]:
            raise ValueError("MEMORY_PROJECTION_OUTBOX_CORRUPT")
        return payload

    # 功能：
    #   发送单个快照并按原代次确认或退避，不能覆盖请求期间出现的更新意图。
    # 输入：
    #   self：同步桥；scope、boundary：账户边界；client：受控传输。
    #   row：待发快照；issues：诊断收集列表。
    # 输出：
    #   delivered：该代次是否已得到远端响应并成功确认。
    def _deliver_row(
        self,
        scope: MemoryOwnerScope,
        boundary: RemoteBoundary,
        client: ProjectionTransport,
        row: sqlite3.Row,
        issues: list[str],
    ) -> bool:
        kind = str(row["operation_kind"])
        try:
            payload = self._outbox_payload(row)
            if kind == "consent":
                response = client.request(
                    "POST",
                    "/rest/v1/console_memory_consents",
                    query={"on_conflict": "user_id,tenant_id,organization_id"},
                    body=payload,
                    prefer="resolution=merge-duplicates,return=representation",
                )
            elif kind == "stage_candidate":
                payload = {
                    **payload,
                    "p_tenant_id": boundary.tenant_id,
                    "p_organization_id": boundary.organization_id,
                }
                response = client.request(
                    "POST",
                    "/rest/v1/rpc/console_memory_stage_current_user",
                    body=payload,
                )
                if isinstance(response, list):
                    if len(response) != 1 or not isinstance(response[0], dict):
                        raise ValueError("MEMORY_PROJECTION_STAGE_RESPONSE_INVALID")
                    response = response[0]
                if not isinstance(response, dict) or not isinstance(
                    response.get("candidate_id"), str
                ):
                    raise ValueError("MEMORY_PROJECTION_STAGE_RESPONSE_INVALID")
                self._map_candidate(
                    scope,
                    str(row["local_reference"]),
                    str(UUID(response["candidate_id"])),
                    str(response.get("payload_sha256") or ""),
                )
            elif kind == "resolve_candidate":
                remote_candidate = self._mapped_candidate(scope, str(row["local_reference"]))
                if remote_candidate is None:
                    self._defer(row, "MEMORY_PROJECTION_CANDIDATE_MAPPING_PENDING")
                    return False
                response = client.request(
                    "POST",
                    "/rest/v1/rpc/console_memory_resolve_current_user",
                    body={
                        "p_tenant_id": boundary.tenant_id,
                        "p_organization_id": boundary.organization_id,
                        "p_candidate_id": remote_candidate,
                        "p_resolution": payload["p_resolution"],
                    },
                )
            elif kind == "forget":
                response = client.request(
                    "POST",
                    "/rest/v1/rpc/console_memory_forget_current_user",
                    body={
                        **payload,
                        "p_tenant_id": boundary.tenant_id,
                        "p_organization_id": boundary.organization_id,
                    },
                )
            elif kind == "permanent_delete":
                response = client.request(
                    "POST",
                    "/rest/v1/rpc/console_memory_permanently_delete_current_user",
                    body={
                        **payload,
                        "p_tenant_id": boundary.tenant_id,
                        "p_organization_id": boundary.organization_id,
                    },
                )
            else:
                raise ValueError("MEMORY_PROJECTION_OPERATION_INVALID")
            _ = response
        except ProjectionHttpError as error:
            permanent = error.status in {400, 401, 403, 404, 409, 422}
            self._defer(row, error.code, retry=not permanent)
            issues.append(str(error)[:240])
            return False
        except (ConnectionError, TimeoutError):
            # Keep the operation durable and add bounded backoff before the
            # outer sync reports the transport failure. Network loss must
            # never make an operation look delivered or discard its intent.
            self._defer(row, "MEMORY_PROJECTION_NETWORK_UNAVAILABLE", retry=True)
            raise
        except (KeyError, TypeError, ValueError) as error:
            code = str(error)
            if re.fullmatch(r"MEMORY_PROJECTION_[A-Z_]{1,100}", code) is None:
                code = "MEMORY_PROJECTION_OPERATION_REJECTED"
            self._defer(row, code, retry=False)
            issues.append(code)
            return False
        return self._mark_delivered(row)

    # 功能：
    #   保存服务端分配的候选标识与摘要，映射隔离在原账户和职责域内。
    # 输入：
    #   self：同步桥；scope：账户范围；local_reference：本地候选标识。
    #   remote_reference：远端候选 UUID；remote_hash：可选服务端载荷摘要。
    # 输出：
    #   None：不返回业务数据。
    def _map_candidate(
        self,
        scope: MemoryOwnerScope,
        local_reference: str,
        remote_reference: str,
        remote_hash: str,
    ) -> None:
        owner, tenant, organization, namespace = self._local_boundary(scope)
        if remote_hash and _SHA256.fullmatch(remote_hash) is None:
            raise ValueError("MEMORY_PROJECTION_STAGE_RESPONSE_INVALID")
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO account_memory_projection_mapping(
                     owner_account_id,tenant_id,organization_id,namespace,local_reference,
                     remote_reference,remote_payload_sha256,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(owner_account_id,tenant_id,organization_id,namespace,local_reference)
                   DO UPDATE SET remote_reference=excluded.remote_reference,
                     remote_payload_sha256=excluded.remote_payload_sha256,
                     updated_at=excluded.updated_at""",
                (
                    owner,
                    tenant,
                    organization,
                    namespace,
                    local_reference,
                    remote_reference,
                    remote_hash or None,
                    _now().isoformat(),
                ),
            )

    # 功能：
    #   查找指定范围的远端候选映射，不借用其他账户的目标标识。
    # 输入：
    #   self：同步桥；scope：完整账户范围；local_reference：本地候选标识。
    # 输出：
    #   remote_id：远端 UUID，未映射时为 None。
    def _mapped_candidate(self, scope: MemoryOwnerScope, local_reference: str) -> str | None:
        owner, tenant, organization, namespace = self._local_boundary(scope)
        with self._connect() as connection:
            row = connection.execute(
                """SELECT remote_reference FROM account_memory_projection_mapping
                     WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                       AND namespace=? AND local_reference=?""",
                (owner, tenant, organization, namespace, local_reference),
            ).fetchone()
        return str(row["remote_reference"]) if row is not None else None

    # 功能：
    #   取得同账户全部职责域的同意代次，捕捉另一同步线程已送达的关闭操作。
    # 输入：
    #   self：同步桥；scope：账户、租户及组织边界。
    # 输出：
    #   generation：稳定排序的同意快照；没有本地同意记录时为空元组。
    def _consent_generation(self, scope: MemoryOwnerScope) -> tuple[tuple[str, ...], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT operation_id,payload_sha256,status,updated_at
                   FROM account_memory_projection_outbox
                   WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                     AND operation_kind='consent' ORDER BY operation_id""",
                self._local_boundary(scope)[:3],
            ).fetchall()
        generation = tuple(tuple(row) for row in rows)
        return generation

    # 功能：
    #   任一共享职责域存在待发或冲突同意时，阻止按旧授权进行普通记忆投递。
    # 输入：
    #   self：同步桥；scope：账户、租户及组织边界。
    # 输出：
    #   unsettled：是否存在尚未解决的同意设置。
    def _has_unsettled_consent(self, scope: MemoryOwnerScope) -> bool:
        with self._connect() as connection:
            return (
                connection.execute(
                    """SELECT 1 FROM account_memory_projection_outbox
                     WHERE owner_account_id=? AND tenant_id=? AND organization_id=?
                       AND operation_kind='consent' AND status!='delivered'
                     LIMIT 1""",
                    self._local_boundary(scope)[:3],
                ).fetchone()
                is not None
            )

    # 功能：
    #   只确认已发送的摘要和更新时间，较新的同 ID 操作仍保持待发送。
    # 输入：
    #   self：同步桥；row：实际发送时的快照。
    # 输出：
    #   delivered：数据库是否成功确认这一代次。
    def _mark_delivered(self, row: sqlite3.Row) -> bool:
        with self._connect() as connection:
            result = connection.execute(
                """UPDATE account_memory_projection_outbox
                     SET status='delivered',last_error=NULL,updated_at=?
                   WHERE operation_id=? AND payload_sha256=? AND updated_at=?
                     AND status='pending'""",
                (_now().isoformat(), row["operation_id"], row["payload_sha256"], row["updated_at"]),
            )
            return result.rowcount == 1

    # 功能：
    #   对原代次应用有界指数退避或永久冲突，不能延迟请求期间新替换的用户意图。
    # 输入：
    #   self：同步桥；row：失败快照；error：安全错误标识；retry：是否允许自动重试。
    # 输出：
    #   None：不返回业务数据。
    def _defer(self, row: sqlite3.Row, error: str, *, retry: bool = True) -> None:
        attempts = int(row["attempts"]) + 1
        status = "pending" if retry else "conflict"
        delay = min(300, 2 ** min(attempts, 8)) if retry else 0
        next_attempt = (_now() + timedelta(seconds=delay)).isoformat()
        with self._connect() as connection:
            connection.execute(
                """UPDATE account_memory_projection_outbox
                     SET status=?,attempts=?,next_attempt_at=?,last_error=?,updated_at=?
                   WHERE operation_id=? AND payload_sha256=? AND updated_at=?
                     AND status='pending'""",
                (
                    status,
                    attempts,
                    next_attempt,
                    error[:240],
                    _now().isoformat(),
                    str(row["operation_id"]),
                    row["payload_sha256"],
                    row["updated_at"],
                ),
            )
