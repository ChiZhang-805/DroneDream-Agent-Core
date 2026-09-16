"""Offline regressions for governed memory promotion, retention and typed context."""

import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest
from test_memory_projection import _record, _scope

from dronedream_agent_core.model_harness.memory import (
    AccountMemoryStore,
    MemoryOwnerScope,
    validate_account_memory_model_context,
)


# 功能：
#   创建虚构账户的任务语言偏好候选，不使用真实账户或外部模型。
# 输入：
#   store：测试存储；kwargs：明确覆盖的候选字段。
# 输出：
#   result：待批准候选。
def candidate(store, **kwargs):
    values = dict(
        kind="preference",
        memory_key="preference.task_locale",
        payload={"locale": "zh-CN"},
        source_conversation_id="synthetic-thread",
        confidence=0.8,
        ttl_days=30,
    )
    values.update(kwargs)
    result = store.record_candidate(_scope(), **values)
    return result


# 功能：
#   同会话推断之后明确批准应能生效，同时只保存一份独立会话证据。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_explicit_update_can_promote_same_conversation_pending_inference(tmp_path):
    store = AccountMemoryStore(tmp_path / "memory.sqlite3")
    pending = candidate(store)
    entry = store.upsert(
        _scope(),
        kind="preference",
        memory_key="preference.task_locale",
        payload={"locale": "zh-CN"},
        source_conversation_id="synthetic-thread",
    )
    assert entry.evidence_count == 1
    candidates = store.list_candidates(_scope())
    assert len(candidates) == 1
    assert candidates[0]["candidate_id"] == pending.candidate_id
    assert candidates[0]["source_kind"] == "explicit_user_update"
    assert candidates[0]["storage_status"] == "consolidated"


# 功能：
#   老候选仍可按精确 ID 找到，不受最近 500 条窗口限制，且不能跨账户查找。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_projection_candidate_is_not_limited_to_recent_window(tmp_path):
    store = AccountMemoryStore(tmp_path / "memory.sqlite3")
    pending = candidate(store)
    with store._connect() as connection:
        row = dict(connection.execute("SELECT * FROM account_memory_candidates").fetchone())
        keys = list(row)
        values = []
        for index in range(501):
            newer = {
                **row,
                "candidate_id": f"memory-candidate-{UUID(int=index + 1).hex}",
                "source_conversation_id": f"newer-thread-{index}",
                "observed_at": (datetime.now(UTC) + timedelta(seconds=1)).isoformat(),
            }
            values.append(tuple(newer[key] for key in keys))
        connection.executemany(
            f"INSERT INTO account_memory_candidates({','.join(keys)}) "
            f"VALUES({','.join('?' for _ in keys)})",
            values,
        )
    assert pending.candidate_id not in {
        row["candidate_id"] for row in store.list_candidates(_scope(), limit=500)
    }
    assert (
        store.projection_candidate(_scope(), pending.candidate_id)["candidate_id"]
        == pending.candidate_id
    )
    with pytest.raises(KeyError):
        store.projection_candidate(
            MemoryOwnerScope(owner_account_id="another-owner"), pending.candidate_id
        )


# 功能：
#   字符串、布尔证据和错误 TTL 不能被 Pydantic 隐式变为合法候选，也不能先绑定会话。
# 输入：
#   tmp_path：独立数据库目录；values：非法候选字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "values",
    [
        {"confidence": True},
        {"confidence": "0.9"},
        {"ttl_days": True},
        {"payload": {"map_asset_id": False}},
    ],
)
def test_candidate_values_keep_exact_semantics(tmp_path, values):
    store = AccountMemoryStore(tmp_path / "memory.sqlite3")
    with pytest.raises(ValueError):
        candidate(store, **values)
    with pytest.raises(KeyError):
        store.require_thread_binding(_scope(), "synthetic-thread")


# 功能：
#   插件不得重复输出同一个条目来制造多份证据或放大对模型的影响。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_context_rejects_duplicate_keys(tmp_path):
    store = AccountMemoryStore(tmp_path / "memory.sqlite3")
    pending = candidate(store)
    store.resolve_candidate(_scope(), pending.candidate_id, approve=True)
    context = store.model_context(_scope())
    context["items"] *= 2
    context["estimated_tokens"] *= 2
    with pytest.raises(ValueError, match="DUPLICATE_KEY"):
        validate_account_memory_model_context(context)


# 功能：
#   模拟持久化时间缺少时区或载荷包含歧义键，检索必须隔离而非提供给模型。
# 输入：
#   tmp_path：独立数据库；corruption：对虚构条目的损坏方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("corruption", ["naive-time", "duplicate-json"])
def test_corrupt_stored_record_is_quarantined(tmp_path, corruption):
    store = AccountMemoryStore(tmp_path / "memory.sqlite3")
    pending = candidate(store)
    store.resolve_candidate(_scope(), pending.candidate_id, approve=True)
    with sqlite3.connect(store.path) as connection:
        if corruption == "naive-time":
            connection.execute(
                "UPDATE governed_account_memory SET last_observed_at='2026-01-01T00:00:00'"
            )
        else:
            connection.execute(
                "UPDATE governed_account_memory SET payload_json=?",
                ('{"locale":"zh-CN","locale":"en-US"}',),
            )
    assert store.model_context(_scope())["items"] == []
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute("SELECT status FROM governed_account_memory").fetchone()[0]
            == "quarantined"
        )


# 功能：
#   时区偏移不能让真正已过期的记录继续活动，也不能让仍有效的记录提前被 SQL 判为过期。
# 输入：
#   tmp_path：独立数据库；future：构造未来或过去的绝对过期时间。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("future", [False, True])
def test_expiry_compares_absolute_time_not_iso_text(tmp_path, future):
    store = AccountMemoryStore(tmp_path / "memory.sqlite3")
    pending = candidate(store)
    store.resolve_candidate(_scope(), pending.candidate_id, approve=True)
    now = datetime.now(UTC)
    expiry = now + timedelta(minutes=5 if future else -5)
    offset = timezone(timedelta(hours=-12 if future else 14))
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE governed_account_memory SET expires_at=?,last_observed_at=?",
            (expiry.astimezone(offset).isoformat(), (now - timedelta(days=1)).isoformat()),
        )
    assert bool(store.model_context(_scope())["items"]) is future


# 功能：
#   本地过期但尚未被标记的活动行，不能永久挡住远端同键的新有效记录。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_expired_local_entry_does_not_block_valid_remote_refresh(tmp_path):
    store = AccountMemoryStore(tmp_path / "memory.sqlite3")
    pending = candidate(store)
    store.resolve_candidate(_scope(), pending.candidate_id, approve=True)
    now = datetime.now(UTC)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE governed_account_memory SET expires_at=?",
            ((now - timedelta(days=1)).isoformat(),),
        )
    record = _record(payload={"locale": "en-US"})
    outcome = store.import_projected_entry(
        _scope(),
        memory_key=record["memory_key"],
        payload=record["payload"],
        source_conversation_id=record["conversation_id"],
        source_edition="autonomy",
        evidence_count=1,
        confidence=0.9,
        last_observed_at=now,
        expires_at=now + timedelta(days=1),
        remote_memory_id=record["memory_id"],
        remote_payload_sha256="a" * 64,
        remote_revision=1,
        projection_authority=store._projection_authority,
    )
    assert outcome == "imported"
    assert store.get(_scope(), record["memory_key"]).payload == {"locale": "en-US"}


# 功能：
#   早已过期的候选不能仅因用户点击批准就被当作新证据写入活动表。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_expired_candidate_cannot_be_promoted(tmp_path):
    store = AccountMemoryStore(tmp_path / "memory.sqlite3")
    pending = candidate(store)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE account_memory_candidates SET observed_at=?",
            ((datetime.now(UTC) - timedelta(days=60)).isoformat(),),
        )
    with pytest.raises(ValueError, match="PROMOTION_EVIDENCE_REQUIRED"):
        store.resolve_candidate(_scope(), pending.candidate_id, approve=True)
    assert store.model_context(_scope())["items"] == []


# 功能：
#   损坏候选的核验位不能被 bool(2) 转换成已验证来源。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_corrupt_candidate_flags_are_not_presented_or_promoted(tmp_path):
    store = AccountMemoryStore(tmp_path / "memory.sqlite3")
    pending = candidate(store)
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE account_memory_candidates SET source_verified=2")
    assert store.list_candidates(_scope()) == []
    with pytest.raises(ValueError, match="FLAGS_INVALID"):
        store.resolve_candidate(_scope(), pending.candidate_id, approve=True)
    assert store.model_context(_scope())["items"] == []


# 功能：
#   验证错误文本不能回显用户误提交到记忆字段中的秘密字符串。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_candidate_validation_message_does_not_echo_payload(tmp_path):
    store = AccountMemoryStore(tmp_path / "memory.sqlite3")
    secret = "password synthetic-private-value"
    with pytest.raises(ValueError) as error:
        candidate(store, payload={"locale": secret})
    assert secret not in str(error.value)
