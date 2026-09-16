from __future__ import annotations

import base64
import hashlib
import io
import json
import sqlite3
import urllib.error
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from dronedream_agent_core.model_harness.memory import (
    AUTONOMY_MISSION_NAMESPACE,
    AccountMemoryStore,
    MemoryOwnerScope,
)
from dronedream_agent_core.model_harness.memory_projection import (
    AccountMemoryProjectionBridge,
    MemoryProjectionCredentials,
    ProjectionHttpError,
    SupabaseProjectionTransport,
    remote_boundary,
)


class ScriptedProjectionTransport:
    """Small in-process PostgREST contract peer used only by unit tests."""

    # 功能：
    #   初始化仅用于测试的 PostgREST 内存对端，可控制断网和复合响应形状。
    # 输入：
    #   self：测试对端。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.consent: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []
        self.tombstones: list[dict[str, Any]] = []
        self.network_available = True
        self.stage_list_size: int | None = None

    # 功能：
    #   记录请求并模拟固定同意、候选、删除与条目端点；未知请求立即使测试失败。
    # 输入：
    #   self：测试对端；method、path：请求；query、body、prefer：PostgREST 参数。
    # 输出：
    #   response：按当前测试设置返回的 JSON 值。
    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        prefer: str | None = None,
    ) -> object:
        if not self.network_available:
            raise ConnectionError("offline")
        self.calls.append(
            {
                "method": method,
                "path": path,
                "query": query,
                "body": body,
                "prefer": prefer,
            }
        )
        if path == "/rest/v1/console_memory_consents" and method == "POST":
            assert body is not None
            self.consent = dict(body)
            return [dict(body)]
        if path == "/rest/v1/console_memory_consents":
            return [] if self.consent is None else [dict(self.consent)]
        if path == "/rest/v1/console_memory_deletion_tombstones":
            return list(self.tombstones)
        if path == "/rest/v1/console_memory_records":
            namespace = (query or {}).get("responsibility_namespace", "").removeprefix("eq.")
            return [
                dict(record)
                for record in self.records
                if record["responsibility_namespace"] == namespace
            ]
        if path == "/rest/v1/rpc/console_memory_stage_current_user":
            result = {"candidate_id": str(uuid4()), "payload_sha256": "a" * 64}
            if self.stage_list_size is not None:
                return [dict(result) for _ in range(self.stage_list_size)]
            return result
        if path.startswith("/rest/v1/rpc/console_memory_"):
            return {"ok": True}
        raise AssertionError(f"unexpected projection request: {method} {path}")


# 功能：
#   建立各用例共用的虚构账户边界，不从真实登录状态读取身份。
# 输入：
#   无。
# 输出：
#   scope：测试账户的任务记忆范围。
def _scope() -> MemoryOwnerScope:
    return MemoryOwnerScope(
        owner_account_id="9e443860-b848-4a9f-bb52-5df9d06d80a8",
        source_edition="autonomy",
    )


# 功能：
#   构造格式合法但不可用于真实登录的测试项目凭证。
# 输入：
#   无。
# 输出：
#   credentials：仅用于离线及回环测试的凭证。
def _credentials() -> MemoryProjectionCredentials:
    return MemoryProjectionCredentials(
        project_url="https://project-ref.supabase.co",
        publishable_key="sb_publishable_unit_contract_key",
        access_token=f"{'a' * 24}.{'b' * 24}.{'c' * 24}",
        timeout_seconds=1.0,
    )


# 功能：
#   生成声明指定角色的未签名旧式密钥，用于验证角色拒绝规则而非身份认证。
# 输入：
#   role：测试角色声明。
# 输出：
#   key：三段形状的虚构 JWT。
def _legacy_key(role: str) -> str:
    encoded = (
        base64.urlsafe_b64encode(json.dumps({"role": role}).encode("utf-8"))
        .decode("ascii")
        .rstrip("=")
    )
    return f"{'a' * 24}.{encoded}.{'c' * 24}"


# 功能：
#   按开关构造测试职责域及类别读写同意。
# 输入：
#   enabled：是否启用本用例的测试记忆。
# 输出：
#   consent：带更新时间的授权资料。
def _consent(*, enabled: bool = True) -> dict[str, Any]:
    namespaces = ["account.shared", AUTONOMY_MISSION_NAMESPACE] if enabled else []
    return {
        "memory_enabled": enabled,
        "read_namespaces": namespaces,
        "write_namespaces": namespaces,
        "memory_scopes": {
            "chat_preferences": enabled,
            "device_vehicle": enabled,
            "reports_delivery": enabled,
            "safety_approvals": enabled,
        },
        "updated_at": datetime.now(UTC).isoformat(),
    }


# 功能：
#   在独立测试存储中创建一条未获提升的语言偏好推断。
# 输入：
#   memory：测试存储；scope：虚构账户范围。
# 输出：
#   candidate：待批准的候选结果。
def _candidate(memory: AccountMemoryStore, scope: MemoryOwnerScope):
    return memory.record_candidate(
        scope,
        kind="preference",
        memory_key="preference.task_locale",
        payload={"locale": "zh-CN"},
        source_conversation_id="thread-projection-contract",
        confidence=0.84,
        ttl_days=180,
    )


# 功能：
#   为拉取验证创建远端条目夹具，摘要是测试修订标识而非真实性证据。
# 输入：
#   memory_id：可复用远端 ID；payload：测试内容；payload_hash：测试摘要；revision：修订。
# 输出：
#   record：有明确来源和有效期的虚构远端条目。
def _record(
    *,
    memory_id: str | None = None,
    payload: dict[str, Any] | None = None,
    payload_hash: str = "b" * 64,
    revision: int = 1,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "memory_id": memory_id or str(uuid4()),
        "responsibility_namespace": AUTONOMY_MISSION_NAMESPACE,
        "scope": "chat_preferences",
        "memory_key": "preference.task_locale",
        "payload": payload or {"locale": "zh-CN"},
        "payload_sha256": payload_hash,
        "source_version": 1,
        "projection_revision": revision,
        "edition": "autonomy",
        "conversation_id": str(uuid4()),
        "evidence_count": 1,
        "confidence": 0.92,
        "last_seen": now.isoformat(),
        "expires_at": (now + timedelta(days=90)).isoformat(),
    }


# 功能：
#   个人租户必须从用户 UUID 推导，不能使用任意组织标识。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_personal_remote_boundary_is_derived_from_account_uuid() -> None:
    boundary = remote_boundary(_scope())

    assert boundary.user_id == _scope().owner_account_id
    assert boundary.tenant_id == _scope().owner_account_id
    assert boundary.organization_id == "00000000-0000-0000-0000-000000000000"


# 功能：
#   服务角色与秘密项目密钥不能进入普通用户投影传输。
# 输入：
#   unsafe_key：不允许的测试密钥。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "unsafe_key",
    ["sb_secret_unit_contract_key", _legacy_key("service_role")],
)
def test_projection_credentials_reject_authority_bearing_keys(unsafe_key: str) -> None:
    with pytest.raises(ValueError, match="FORBIDDEN"):
        MemoryProjectionCredentials(
            project_url="https://project-ref.supabase.co",
            publishable_key=unsafe_key,
            access_token=f"{'a' * 24}.{'b' * 24}.{'c' * 24}",
        )


# 功能：
#   保留旧公开 anon 项目密钥的必要兼容性，不因此接受服务角色。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_projection_credentials_accept_legacy_anon_key() -> None:
    credentials = MemoryProjectionCredentials(
        project_url="https://project-ref.supabase.co",
        publishable_key=_legacy_key("anon"),
        access_token=f"{'a' * 24}.{'b' * 24}.{'c' * 24}",
    )

    assert credentials.publishable_key == _legacy_key("anon")


# 功能：
#   缺少云端配置时保存候选上传意图，重复排队不会新增一份操作。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_configuration_missing_keeps_idempotent_candidate_outbox(tmp_path) -> None:
    memory = AccountMemoryStore(tmp_path / "memory.sqlite3")
    bridge = AccountMemoryProjectionBridge(memory)
    candidate = _candidate(memory, _scope())

    first_operation = bridge.queue_candidate(_scope(), candidate.candidate_id)
    second_operation = bridge.queue_candidate(_scope(), candidate.candidate_id)
    status = bridge.sync(_scope(), None, local_memory_enabled=True)

    assert first_operation == second_operation
    assert status.status == "configuration_missing"
    assert status.pending_operations == 1


# 功能：
#   同意读取前断网不会丢弃候选，也不能产生模型可见活动记忆。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_network_failure_retains_pending_outbox_and_grants_no_state(tmp_path) -> None:
    memory = AccountMemoryStore(tmp_path / "memory.sqlite3")
    bridge = AccountMemoryProjectionBridge(memory)
    candidate = _candidate(memory, _scope())
    bridge.queue_candidate(_scope(), candidate.candidate_id)
    transport = ScriptedProjectionTransport()
    transport.network_available = False

    status = bridge.sync(
        _scope(),
        _credentials(),
        local_memory_enabled=True,
        transport=transport,
    )

    assert status.status == "network_unavailable"
    assert status.pending_operations == 1
    assert bridge.pending_count(_scope()) == 1
    assert memory.model_context(_scope(), query="locale")["items"] == []
    with sqlite3.connect(memory.path) as connection:
        retained = connection.execute(
            """SELECT status, attempts, last_error
                 FROM account_memory_projection_outbox"""
        ).fetchone()
    assert retained is not None
    assert retained[0] == "pending"
    assert retained[1] == 0
    assert retained[2] is None


# 功能：
#   实际发送待发操作时断网，保留操作并增加一次受限退避尝试。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_network_failure_during_outbox_delivery_adds_retry_backoff(tmp_path) -> None:
    memory = AccountMemoryStore(tmp_path / "memory.sqlite3")
    bridge = AccountMemoryProjectionBridge(memory)
    bridge.queue_consent(_scope(), enabled=True)
    transport = ScriptedProjectionTransport()
    transport.network_available = False

    status = bridge.sync(
        _scope(),
        _credentials(),
        local_memory_enabled=True,
        transport=transport,
    )

    assert status.status == "network_unavailable"
    with sqlite3.connect(memory.path) as connection:
        retained = connection.execute(
            """SELECT status, attempts, last_error
                 FROM account_memory_projection_outbox"""
        ).fetchone()
    assert retained is not None
    assert retained[0] == "pending"
    assert retained[1] == 1
    assert retained[2] == "MEMORY_PROJECTION_NETWORK_UNAVAILABLE"


# 功能：
#   候选使用由用户 JWT 决定身份的当前用户 RPC，请求不得自行指定 p_user_id。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_consent_and_candidate_use_rls_current_user_rpc_without_user_id(tmp_path) -> None:
    memory = AccountMemoryStore(tmp_path / "memory.sqlite3")
    bridge = AccountMemoryProjectionBridge(memory)
    candidate = _candidate(memory, _scope())
    bridge.queue_candidate(_scope(), candidate.candidate_id)
    bridge.queue_consent(_scope(), enabled=True)
    transport = ScriptedProjectionTransport()
    transport.stage_list_size = 1

    status = bridge.sync(
        _scope(),
        _credentials(),
        local_memory_enabled=True,
        transport=transport,
    )

    stage_call = next(
        call
        for call in transport.calls
        if call["path"] == "/rest/v1/rpc/console_memory_stage_current_user"
    )
    assert status.status == "synced"
    assert status.pushed == 2
    assert stage_call["method"] == "POST"
    assert "p_user_id" not in stage_call["body"]
    assert stage_call["body"]["p_tenant_id"] == _scope().owner_account_id
    assert stage_call["body"]["p_organization_id"] == ("00000000-0000-0000-0000-000000000000")


# 功能：
#   单候选 RPC 返回多行时拒绝选择任意一行，操作必须呈现冲突。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_stage_rpc_rejects_multirow_composite_response(tmp_path) -> None:
    memory = AccountMemoryStore(tmp_path / "memory.sqlite3")
    bridge = AccountMemoryProjectionBridge(memory)
    candidate = _candidate(memory, _scope())
    bridge.queue_candidate(_scope(), candidate.candidate_id)
    transport = ScriptedProjectionTransport()
    transport.consent = _consent()
    transport.stage_list_size = 2

    status = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=transport)

    assert status.status == "conflict"
    assert status.conflicts == 1
    assert status.pending_operations == 0
    assert any("STAGE_RESPONSE_INVALID" in issue for issue in status.issues)


# 功能：
#   显式批准使用上传后得到的同账户远端候选映射，而非直接发送本地标识。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_explicit_local_resolution_is_queued_after_remote_stage_mapping(tmp_path) -> None:
    memory = AccountMemoryStore(tmp_path / "memory.sqlite3")
    bridge = AccountMemoryProjectionBridge(memory)
    candidate = _candidate(memory, _scope())
    bridge.queue_candidate(_scope(), candidate.candidate_id)
    transport = ScriptedProjectionTransport()
    transport.consent = _consent()
    first = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=transport)
    assert first.status == "synced"

    bridge.queue_resolution(_scope(), candidate.candidate_id, approve=True)
    second = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=transport)

    resolve_call = next(
        call
        for call in transport.calls
        if call["path"] == "/rest/v1/rpc/console_memory_resolve_current_user"
    )
    assert second.status == "synced"
    assert resolve_call["body"]["p_resolution"] == "promote"
    assert "p_user_id" not in resolve_call["body"]


# 功能：
#   即使远端返回记录，含执行授权字段的内容仍不能进入本地模型上下文。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_pull_revalidates_remote_payload_before_local_import(tmp_path) -> None:
    memory = AccountMemoryStore(tmp_path / "memory.sqlite3")
    bridge = AccountMemoryProjectionBridge(memory)
    transport = ScriptedProjectionTransport()
    transport.consent = _consent()
    transport.records = [_record(payload={"authorization": "execute-now"})]

    status = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=transport)

    assert status.status == "conflict"
    assert status.conflicts == 1
    assert memory.model_context(_scope(), query="authorization")["items"] == []
    assert any("RECORD_REJECTED" in issue for issue in status.issues)


# 功能：
#   修订回退及摘要变化不能覆盖已有内容或降低记忆投影的高水位。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_revision_or_hash_rollback_is_conflict_and_does_not_replace_local(tmp_path) -> None:
    memory = AccountMemoryStore(tmp_path / "memory.sqlite3")
    bridge = AccountMemoryProjectionBridge(memory)
    transport = ScriptedProjectionTransport()
    transport.consent = _consent()
    remote_id = str(uuid4())
    transport.records = [_record(memory_id=remote_id, payload={"locale": "zh-CN"}, revision=2)]

    first = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=transport)
    assert first.status == "synced"
    assert first.pulled == 1

    transport.records = [
        _record(
            memory_id=remote_id,
            payload={"locale": "en-US"},
            payload_hash="c" * 64,
            revision=1,
        )
    ]
    second = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=transport)

    context = memory.model_context(_scope(), query="locale")
    assert second.status == "conflict"
    assert second.conflicts == 1
    assert context["items"][0]["payload"] == {"locale": "zh-CN"}
    with sqlite3.connect(memory.path) as connection:
        high_water = connection.execute(
            "SELECT remote_revision, remote_payload_sha256 FROM account_memory_projection_state"
        ).fetchone()
    assert high_water == (2, "b" * 64)


# 功能：
#   云端删除确实清理本地对应内容，并且不反向再生成一份删除请求。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_remote_tombstone_removes_local_value_without_requeue(tmp_path) -> None:
    memory = AccountMemoryStore(tmp_path / "memory.sqlite3")
    bridge = AccountMemoryProjectionBridge(memory)
    memory.upsert(
        _scope(),
        kind="preference",
        memory_key="preference.task_locale",
        payload={"locale": "zh-CN"},
        source_conversation_id="thread-local",
    )
    transport = ScriptedProjectionTransport()
    transport.consent = _consent()
    transport.tombstones = [
        {
            "tombstone_id": str(uuid4()),
            "responsibility_namespace": AUTONOMY_MISSION_NAMESPACE,
            "scope": "chat_preferences",
            "memory_key_sha256": hashlib.sha256(b"preference.task_locale").hexdigest(),
            "released_at": None,
            "deleted_at": datetime.now(UTC).isoformat(),
        }
    ]

    status = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=transport)

    assert status.tombstones_applied == 1
    assert bridge.pending_count(_scope()) == 0
    assert memory.model_context(_scope(), query="locale")["items"] == []


# 功能：
#   本地关闭记忆后仍优先发送关闭意图，再停止普通记忆拉取与上传。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_disabling_memory_still_propagates_consent_before_local_stop(tmp_path) -> None:
    memory = AccountMemoryStore(tmp_path / "memory.sqlite3")
    bridge = AccountMemoryProjectionBridge(memory)
    bridge.queue_consent(_scope(), enabled=False)
    transport = ScriptedProjectionTransport()

    status = bridge.sync(
        _scope(),
        _credentials(),
        local_memory_enabled=False,
        transport=transport,
    )

    assert status.status == "local_disabled"
    assert status.pushed == 1
    assert status.pending_operations == 0
    assert transport.consent is not None
    assert transport.consent["memory_enabled"] is False
    assert transport.consent["read_namespaces"] == []


# 功能：
#   较早的开启请求无论成功、断网或被拒绝，都不能确认请求期间新产生的关闭代次。
# 输入：
#   tmp_path：独立数据库目录；delivery_error：旧请求的结束方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("delivery_error", [None, "offline", "rejected"])
def test_inflight_consent_cannot_acknowledge_a_newer_revoke(tmp_path, delivery_error) -> None:
    memory = AccountMemoryStore(tmp_path / "memory.sqlite3")
    bridge = AccountMemoryProjectionBridge(memory)
    bridge.queue_consent(_scope(), enabled=True)

    class ReplacingTransport(ScriptedProjectionTransport):
        # 功能：
        #   在旧开启请求尚未完成时插入新关闭请求，并模拟旧请求的三种响应。
        # 输入：
        #   self：测试传输；method、path、kwargs：原请求参数。
        # 输出：
        #   response：旧请求响应或指定异常。
        def request(self, method, path, **kwargs):
            if method == "POST" and path == "/rest/v1/console_memory_consents":
                # Reproduce a user disabling memory while an earlier enable is in flight.
                bridge.queue_consent(_scope(), enabled=False)
                if delivery_error == "offline":
                    raise ConnectionError("offline")
                if delivery_error == "rejected":
                    raise ProjectionHttpError(403, "denied")
            return super().request(method, path, **kwargs)

    transport = ReplacingTransport()
    status = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=transport)
    assert status.pending_operations == 1
    with sqlite3.connect(memory.path) as connection:
        row = connection.execute(
            "SELECT status, attempts, payload_json FROM account_memory_projection_outbox"
        ).fetchone()
    assert row[:2] == ("pending", 0)
    assert json.loads(row[2])["memory_enabled"] is False
    assert not any(call["method"] == "GET" for call in transport.calls)

    # The latest intent remains immediately deliverable, even if the older call failed.
    peer = ScriptedProjectionTransport()
    final = bridge.sync(_scope(), _credentials(), local_memory_enabled=False, transport=peer)
    assert final.pending_operations == 0
    assert peer.consent["memory_enabled"] is False


# 功能：
#   未获读取同意的职责域不产生条目查询。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_pull_queries_only_namespaces_with_read_consent(tmp_path) -> None:
    bridge = AccountMemoryProjectionBridge(AccountMemoryStore(tmp_path / "memory.sqlite3"))
    transport = ScriptedProjectionTransport()
    transport.consent = _consent()
    transport.consent["read_namespaces"] = [AUTONOMY_MISSION_NAMESPACE]
    status = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=transport)
    assert status.status == "synced"
    assert [
        call["query"]["responsibility_namespace"]
        for call in transport.calls
        if call["path"] == "/rest/v1/console_memory_records"
    ] == [f"eq.{AUTONOMY_MISSION_NAMESPACE}"]


# 功能：
#   错误形状的记忆类别应成为可报告冲突，而不是使整个同步抛出未处理异常。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_malformed_record_scope_is_rejected_without_crashing_sync(tmp_path) -> None:
    bridge = AccountMemoryProjectionBridge(AccountMemoryStore(tmp_path / "memory.sqlite3"))
    transport = ScriptedProjectionTransport()
    transport.consent = _consent()
    transport.records = [{**_record(), "scope": ["chat_preferences"]}]
    status = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=transport)
    assert status.status == "conflict"
    assert status.conflicts == 1


# 功能：
#   错误删除模式不得被真值分支解释为永久删除。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_invalid_forget_mode_never_becomes_permanent_delete(tmp_path) -> None:
    bridge = AccountMemoryProjectionBridge(AccountMemoryStore(tmp_path / "memory.sqlite3"))
    with pytest.raises(ValueError, match="MODE_INVALID"):
        bridge.queue_forget(_scope(), "preference.task_locale", mode="typo")
    assert bridge.pending_count(_scope()) == 0


# 功能：
#   删除请求失败仍消耗本次发送额度，不能借失败额外发送候选。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_failed_preconsent_delivery_still_consumes_operation_budget(tmp_path) -> None:
    memory = AccountMemoryStore(tmp_path / "memory.sqlite3")
    bridge = AccountMemoryProjectionBridge(memory)
    bridge.queue_forget(_scope(), "preference.task_locale", mode="soft")
    candidate = _candidate(memory, _scope())
    bridge.queue_candidate(_scope(), candidate.candidate_id)

    class FailedDeleteTransport(ScriptedProjectionTransport):
        # 功能：
        #   记录删除尝试后返回服务暂不可用，其他操作保留普通测试行为。
        # 输入：
        #   self：测试传输；method、path、kwargs：原请求参数。
        # 输出：
        #   result：非删除请求的测试响应。
        def request(self, method, path, **kwargs):
            result = super().request(method, path, **kwargs)
            if path == "/rest/v1/rpc/console_memory_forget_current_user":
                raise ProjectionHttpError(503, "try_later")
            return result

    transport = FailedDeleteTransport()
    transport.consent = _consent()
    bridge.sync(
        _scope(), _credentials(), local_memory_enabled=True, operation_limit=1, transport=transport
    )
    assert len([call for call in transport.calls if call["method"] == "POST"]) == 1
    assert bridge.pending_count(_scope()) == 2


# 功能：
#   不可信 HTTP 错误正文的形状不影响稳定错误类别，并确保关闭错误流。
# 输入：
#   monkeypatch：替换网络连接器；error_body：不同形状的错误正文。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("error_body", [b"[]", b"null", b'"rejected"', b"not-json"])
def test_http_error_shape_does_not_crash_or_leak_response(monkeypatch, error_body) -> None:
    payload = io.BytesIO(error_body)
    error = urllib.error.HTTPError(
        "https://project-ref.supabase.co/rest/v1/test", 403, "rejected", {}, payload
    )

    class RejectingOpener:
        # 功能：
        #   抛出指定 HTTP 错误而不打开真实网络。
        # 输入：
        #   self：测试连接器；request、data、timeout：原网络参数。
        # 输出：
        #   None：始终抛出指定测试异常。
        def open(self, request, data=None, timeout=None):
            raise error

    monkeypatch.setattr("urllib.request._opener", None)
    monkeypatch.setattr("urllib.request.build_opener", lambda *handlers: RejectingOpener())
    transport = SupabaseProjectionTransport(_credentials())
    with pytest.raises(ProjectionHttpError, match="HTTP_403:REMOTE_REJECTED"):
        transport.request("GET", "/rest/v1/test")
    assert payload.closed


# 功能：
#   通过真实本机 HTTP 重定向验证凭证不会被再次发往跳转地址。
# 输入：
#   monkeypatch：仅将测试请求重定向到回环服务的工具。
# 输出：
#   None：不返回业务数据。
def test_projection_rejects_redirect_instead_of_forwarding_user_credentials(monkeypatch) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    paths = []
    monkeypatch.setattr("urllib.request._opener", None)

    class RedirectPeer(BaseHTTPRequestHandler):
        # 功能：
        #   记录访问路径并给出指向凭证接收测试路径的重定向，不保存实际凭证。
        # 输入：
        #   self：本机 HTTP 测试处理器。
        # 输出：
        #   None：写入 HTTP 响应，不返回业务数据。
        def do_GET(self):
            paths.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/credential-sink")
            self.end_headers()

        # 功能：
        #   静默本机测试服务日志，避免把传输细节混入测试输出。
        # 输入：
        #   self：处理器；args：HTTP 日志参数。
        # 输出：
        #   None：不输出日志。
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectPeer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Keep credential validation intact; redirect only this test's network peer to loopback.
        transport = SupabaseProjectionTransport(_credentials())
        original_open = transport._opener.open

        # 功能：
        #   将虚构项目请求送到本机测试 HTTP 服务，同时保留真实重定向处理器。
        # 输入：
        #   request：仅含虚构凭证的请求；kwargs：网络超时参数。
        # 输出：
        #   response：本机 HTTP 响应或原重定向异常。
        def open_loopback(request, **kwargs):
            local = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/rest/v1/test",
                headers=dict(request.headers),
                method=request.method,
            )
            response = original_open(local, **kwargs)
            return response

        monkeypatch.setattr(transport._opener, "open", open_loopback)
        with pytest.raises(ProjectionHttpError, match="HTTP_302"):
            transport.request("GET", "/rest/v1/test")
        assert paths == ["/rest/v1/test"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
