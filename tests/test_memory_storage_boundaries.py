"""Database lifetime and value-ownership checks using isolated synthetic accounts."""

import sqlite3

import pytest

from dronedream_agent_core.model_harness import memory


# 功能：
#   通过固定显式更新入口创建虚构记忆，测试不访问真实账户。
# 输入：
#   store：当前测试独占的记忆数据库。
# 输出：
#   result：测试账户范围和已经保存的记忆条目。
def _entry(store):
    scope = memory.MemoryOwnerScope(owner_account_id="synthetic-owner")
    entry = store.upsert(
        scope,
        kind="preference",
        memory_key="preference.locale",
        payload={"constraints": ["keep clearance"], "locale": "zh-CN"},
        source_conversation_id="synthetic-thread",
    )
    result = scope, entry
    return result


# 功能：
#   保持连接强引用以排除垃圾回收的帮助，确认初始化、写入和读取后数据库连接立即关闭。
# 输入：
#   tmp_path：测试独占的目录。
#   monkeypatch：仅替换本测试的 SQLite 连接入口。
# 输出：
#   None：不返回业务数据。
def test_memory_operation_closes_connection_without_waiting_for_garbage_collection(
    tmp_path,
    monkeypatch,
):
    connections, original = [], sqlite3.connect

    # 功能：
    #   记录真实测试连接并保留强引用，不改变连接的事务或关闭行为。
    # 输入：
    #   args：SQLite 连接的位置参数。
    #   kwargs：SQLite 连接的命名参数。
    # 输出：
    #   connection：由原始 SQLite 工厂创建的连接。
    def tracked_connect(*args, **kwargs):
        connection = original(*args, **kwargs)
        connections.append(connection)  # Keep strong references to expose missing close().
        return connection

    monkeypatch.setattr(memory.sqlite3, "connect", tracked_connect)
    try:
        store = memory.AccountMemoryStore(tmp_path / "memory.sqlite3")
        scope, _ = _entry(store)
        store.list(scope)
        for connection in connections:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
    finally:
        for connection in connections:
            connection.close()


# 功能：
#   修改对模型提供的嵌套约束列表，不能反向清空保留的记忆条目。
# 输入：
#   tmp_path：测试独占的数据库目录。
# 输出：
#   None：不返回业务数据。
def test_model_context_cannot_mutate_a_retained_memory_entry(tmp_path):
    _, entry = _entry(memory.AccountMemoryStore(tmp_path / "memory.sqlite3"))
    entry.model_context()["payload"]["constraints"].clear()
    assert entry.payload["constraints"] == ["keep clearance"]


# 功能：
#   对仍在有效期的条目提交过期处理请求时保持活动状态，不能只按键无条件过期。
# 输入：
#   tmp_path：测试独占的数据库目录。
# 输出：
#   None：不返回业务数据。
def test_stale_expiration_request_does_not_expire_a_refreshed_entry(tmp_path):
    store = memory.AccountMemoryStore(tmp_path / "memory.sqlite3")
    scope, entry = _entry(store)
    store._expire(scope, entry.memory_key)
    assert store.get(scope, entry.memory_key).status == "active"


# 功能：
#   非法记忆候选在绑定会话前拒绝，不能用失败请求占用一个尚未绑定的会话。
# 输入：
#   tmp_path：测试独占的数据库目录。
# 输出：
#   None：不返回业务数据。
def test_invalid_candidate_does_not_bind_a_thread_before_validation(tmp_path):
    store = memory.AccountMemoryStore(tmp_path / "memory.sqlite3")
    scope = memory.MemoryOwnerScope(owner_account_id="synthetic-owner")
    with pytest.raises(ValueError):
        store.record_candidate(
            scope,
            kind="preference",
            memory_key="preference.locale",
            payload={"password": "fake"},
            source_conversation_id="invalid-thread",
            confidence=0.8,
            ttl_days=30,
        )
    with pytest.raises(KeyError):
        store.require_thread_binding(scope, "invalid-thread")


# 功能：
#   主动在事务中抛错，确认未提交的会话绑定回滚且连接关闭，下一次连接看不到残留数据。
# 输入：
#   tmp_path：测试独占的数据库目录。
# 输出：
#   None：不返回业务数据。
def test_transaction_exception_rolls_back_and_closes_connection(tmp_path):
    store = memory.AccountMemoryStore(tmp_path / "memory.sqlite3")
    with pytest.raises(RuntimeError, match="fixture"), store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO account_memory_thread_bindings VALUES(?,?,?,?,?,?)",
            ("rollback", "owner", "", "", "autonomy.mission", "2026-01-01"),
        )
        raise RuntimeError("fixture failure")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with store._connect() as fresh:
        assert (
            fresh.execute("SELECT COUNT(*) FROM account_memory_thread_bindings").fetchone()[0] == 0
        )
