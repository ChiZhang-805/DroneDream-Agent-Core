"""Synthetic, offline checks for account-memory transport and synchronization integrity."""

import io
import json
import sqlite3
import urllib.error
from uuid import UUID

import pytest
from test_memory_projection import (
    ScriptedProjectionTransport,
    _candidate,
    _consent,
    _credentials,
    _record,
    _scope,
)

from dronedream_agent_core.model_harness.memory import AccountMemoryStore
from dronedream_agent_core.model_harness.memory_projection import (
    AccountMemoryProjectionBridge,
    MemoryProjectionCredentials,
    ProjectionHttpError,
    SupabaseProjectionTransport,
)


# 功能：
#   为每个测试建立虚构账户的独立记忆库和离线传输，不访问真实 Supabase。
# 输入：
#   tmp_path：测试专属目录。
# 输出：
#   fixture：本地存储、同步桥和已同意记忆的测试传输。
@pytest.fixture
def projection(tmp_path):
    memory = AccountMemoryStore(tmp_path / "memory.sqlite3")
    bridge = AccountMemoryProjectionBridge(memory)
    peer = ScriptedProjectionTransport()
    peer.consent = _consent()
    fixture = memory, bridge, peer
    return fixture


# 功能：
#   证明拉取阶段失败不会在待发箱为空时被误报为 synced。
# 输入：
#   projection：独立存储和离线传输。
# 输出：
#   None：不返回业务数据。
def test_failed_pull_never_reports_synced(projection):
    _, bridge, peer = projection
    peer.tombstones = "invalid"
    status = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=peer)
    assert status.status == "conflict"
    assert status.issues


# 功能：
#   合法的共享记忆读取同意不应依赖同时开启任务域读取。
# 输入：
#   projection：独立存储和离线传输。
# 输出：
#   None：不返回业务数据。
def test_shared_read_consent_is_independent_of_mission_read(projection):
    _, bridge, peer = projection
    peer.consent["read_namespaces"] = ["account.shared"]
    status = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=peer)
    assert status.status == "synced"
    queries = [
        call["query"]["responsibility_namespace"]
        for call in peer.calls
        if call["path"] == "/rest/v1/console_memory_records"
    ]
    assert queries == ["eq.account.shared"]


# 功能：
#   同步读取在途时新增关闭意图，旧响应不得继续进入本地活动记忆。
# 输入：
#   projection：独立存储、同步桥和离线传输；monkeypatch：替换该测试传输方法。
# 输出：
#   None：不返回业务数据。
def test_consent_change_while_pulling_discards_old_response(projection, monkeypatch):
    memory, bridge, peer = projection
    peer.records = [_record()]
    original = peer.request

    # 功能：
    #   模拟真实请求发出后用户关闭记忆，返回的旧响应仍包含原先的条目。
    # 输入：
    #   method、path：原请求；kwargs：传输参数。
    # 输出：
    #   response：关闭前生成的测试响应。
    def revoke_during_response(method, path, **kwargs):
        response = original(method, path, **kwargs)
        if path == "/rest/v1/console_memory_records":
            bridge.queue_consent(_scope(), enabled=False)
        return response

    monkeypatch.setattr(peer, "request", revoke_during_response)
    status = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=peer)
    assert status.status == "pending"
    assert "MEMORY_PROJECTION_CONSENT_CHANGED" in status.issues
    assert memory.model_context(_scope())["items"] == []


# 功能：
#   修改已入队载荷但不更新摘要时，不能将损坏数据发送到远端。
# 输入：
#   projection：独立存储和离线传输。
# 输出：
#   None：不返回业务数据。
def test_outbox_payload_is_verified_before_delivery(projection):
    memory, bridge, peer = projection
    candidate = _candidate(memory, _scope())
    bridge.queue_candidate(_scope(), candidate.candidate_id)
    with sqlite3.connect(memory.path) as connection:
        connection.execute("UPDATE account_memory_projection_outbox SET payload_json='{}'")
    status = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=peer)
    assert status.status == "conflict"
    assert not any(call["method"] == "POST" for call in peer.calls)


# 功能：
#   拒绝把字符串或整数开关当作用户批准，错误请求不能先写入待发箱。
# 输入：
#   projection：独立存储；value：错误开关值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", ["false", 1, None])
def test_consent_requires_boolean(projection, value):
    _, bridge, _ = projection
    with pytest.raises(ValueError):
        bridge.queue_consent(_scope(), enabled=value)
    with pytest.raises(ValueError):
        bridge.sync(_scope(), None, local_memory_enabled=value)
    assert bridge.pending_count(_scope()) == 0


# 功能：
#   即使没有配置云端凭证，也必须验证操作预算而非接受布尔或字符串上限。
# 输入：
#   projection：独立存储；kwargs：非法预算参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "kwargs", [{"operation_limit": True}, {"pull_limit": 1.5}, {"time_budget_seconds": "1"}]
)
def test_budget_types_are_not_coerced(projection, kwargs):
    _, bridge, _ = projection
    with pytest.raises(ValueError):
        bridge.sync(_scope(), None, local_memory_enabled=True, **kwargs)


# 功能：
#   远端证据数量、置信度和内容类别必须保持正确类型及对应关系。
# 输入：
#   projection：独立存储和传输；update：被破坏的远端字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "update",
    [
        {"evidence_count": True},
        {"confidence": "0.99"},
        {"scope": "safety_approvals"},
        {"edition": "unknown"},
    ],
)
def test_remote_record_preserves_typed_evidence(projection, update):
    memory, bridge, peer = projection
    peer.records = [{**_record(), **update}]
    status = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=peer)
    assert status.status == "conflict"
    assert memory.model_context(_scope())["items"] == []


# 功能：
#   构造有序的无载荷删除回执，供超过首批大小的分页测试使用。
# 输入：
#   number：单调增长的测试 UUID 整数。
# 输出：
#   tombstone：不涉及真实账户的删除回执。
def tombstone(number):
    tombstone = {
        "tombstone_id": str(UUID(int=number)),
        "responsibility_namespace": "autonomy.mission",
        "scope": "chat_preferences",
        "memory_key_sha256": None,
        "released_at": None,
        "deleted_at": "2026-01-01T00:00:00+00:00",
    }
    return tombstone


# 功能：
#   第 257 条删除记录必须被读取，不能读完首批 256 条就开始导入记忆。
# 输入：
#   projection：独立存储；monkeypatch：替换分页响应。
# 输出：
#   None：不返回业务数据。
def test_tombstone_scan_continues_after_first_page(projection, monkeypatch):
    _, bridge, peer = projection
    original = peer.request
    cursors = []

    # 功能：
    #   模拟两页按 UUID 排序的删除结果，检查第二页使用最后主键而非偏移。
    # 输入：
    #   method、path：请求；kwargs：分页查询及原传输参数。
    # 输出：
    #   response：本页删除记录或普通离线响应。
    def paged(method, path, **kwargs):
        if path.endswith("console_memory_deletion_tombstones"):
            cursor = kwargs["query"].get("tombstone_id")
            cursors.append(cursor)
            return (
                [tombstone(index) for index in range(1, 257)]
                if cursor is None
                else [tombstone(257)]
            )
        return original(method, path, **kwargs)

    monkeypatch.setattr(peer, "request", paged)
    status = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=peer)
    assert status.status == "synced"
    assert cursors == [None, f"gt.{UUID(int=256)}"]


# 功能：
#   拒绝重复首批、无法前进的游标以及已经释放的删除记录。
# 输入：
#   projection：独立存储；released：是否返回已释放记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("released", [False, True])
def test_invalid_tombstones_block_following_memory_import(projection, released):
    memory, bridge, peer = projection
    peer.records = [_record()]
    peer.tombstones = (
        [{**tombstone(1), "released_at": "2026-01-02T00:00:00Z"}]
        if released
        else [tombstone(index) for index in range(1, 257)]
    )
    status = bridge.sync(_scope(), _credentials(), local_memory_enabled=True, transport=peer)
    assert status.status == "conflict"
    assert memory.model_context(_scope())["items"] == []


# 功能：
#   凭证必须只指向标准项目源，不能用 URL 中控制字符、空用户信息或其他端口绕过检查。
# 输入：
#   origin：不合规项目 URL。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "origin",
    [
        "https://@project-ref.supabase.co",
        "https://project-ref.supabase.co:8443",
        "https://project-ref.supabase.co\n",
        "https://evil.child.supabase.co",
    ],
)
def test_credentials_reject_noncanonical_origin(origin):
    data = _credentials().model_dump()
    data["project_url"] = origin
    with pytest.raises(ValueError):
        MemoryProjectionCredentials.model_validate(data)
    with pytest.raises(ValueError):
        SupabaseProjectionTransport(_credentials().model_copy(update={"project_url": origin}))


# 功能：
#   HTTP 错误正文即使回显令牌，也不得进入异常文本或待发箱诊断。
# 输入：
#   monkeypatch：替换本测试的网络打开方法。
# 输出：
#   None：不返回业务数据。
def test_http_errors_do_not_echo_sensitive_remote_messages(monkeypatch):
    credentials = _credentials()
    body = io.BytesIO(json.dumps({"message": credentials.access_token}).encode())
    error = urllib.error.HTTPError(credentials.project_url, 403, "rejected", {}, body)
    transport = SupabaseProjectionTransport(credentials)

    # 功能：
    #   返回含测试令牌回显的 HTTP 异常，不产生真实网络连接。
    # 输入：
    #   args、kwargs：原网络调用参数。
    # 输出：
    #   None：始终抛出测试错误。
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(transport._opener, "open", fail)
    with pytest.raises(ProjectionHttpError) as caught:
        transport.request("GET", "/rest/v1/test")
    assert credentials.access_token not in str(caught.value)
    assert credentials.access_token not in repr(credentials)
    assert body.closed
