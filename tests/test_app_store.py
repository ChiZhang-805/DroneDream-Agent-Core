"""Isolated stores/threads, never installed user data, credentials or cloud quota."""

from __future__ import annotations

import sqlite3

import pytest

from dronedream_agent_app.plugin_manager import PluginManager
from dronedream_agent_app.storage import AppStore


# 功能：
#   收集每个测试创建的存储，在断言失败后也释放连接，不遗留数据库文件锁。
# 输入：
#   无。
# 输出：
#   open_store：创建并登记待关闭存储的函数。
@pytest.fixture
def open_store():
    stores = []

    # 功能：
    #   打开独立测试存储并交由外层夹具负责最终关闭。
    # 输入：
    #   root：独立测试目录。
    # 输出：
    #   store：已打开的应用存储。
    def open_store(root):
        store = AppStore(root)
        stores.append(store)
        return store

    try:
        yield open_store
    finally:
        for store in reversed(stores):
            store.close()


# 功能：
#   验证旧任务表添加资产摘要及语言列时保留既有数据，不靠空表测试冒充迁移兼容。
# 输入：
#   tmp_path：独立测试根目录。
#   open_store：负责关闭连接的存储工厂。
# 输出：
#   None：不返回业务数据。
def test_existing_thread_database_gains_immutable_asset_version_columns(tmp_path, open_store):
    root = tmp_path / "store"
    root.mkdir()
    connection = sqlite3.connect(root / "autonomy.sqlite3")
    connection.execute(
        "CREATE TABLE threads ("
        "thread_id TEXT PRIMARY KEY,title TEXT NOT NULL,state TEXT NOT NULL,"
        "selected_model TEXT NOT NULL,selected_map_id TEXT,selected_vehicle_id TEXT,"
        "pinned INTEGER NOT NULL DEFAULT 0,archived INTEGER NOT NULL DEFAULT 0,"
        "created_at TEXT NOT NULL,updated_at TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO threads VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            "old-thread",
            "keep this title",
            "planning",
            "fixture",
            "old-map",
            "old-vehicle",
            0,
            0,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    connection.commit()
    connection.close()

    store = open_store(root)
    columns = {
        str(row[1])
        for row in store._connection.execute("PRAGMA table_info(threads)")  # noqa: SLF001
    }
    assert "selected_map_content_sha256" in columns
    assert "selected_vehicle_content_sha256" in columns
    assert "locale" in columns
    migrated = store._connection.execute(  # noqa: SLF001
        "SELECT locale FROM threads"
    ).fetchall()
    assert all(str(row[0]) == "zh-CN" for row in migrated)
    restored = store.get_thread("old-thread")
    assert restored["title"] == "keep this title"
    assert restored["selected_map_id"] == "old-map"
    assert restored["selected_map_content_sha256"] is None


# 功能：
#   验证本机偏好的默认值和持久化更新，避免混淆本地选择与云端账户权益。
# 输入：
#   tmp_path：独立测试目录。
#   open_store：负责关闭连接的存储工厂。
# 输出：
#   None：不返回业务数据。
def test_startup_preferences_are_durable_and_safe_by_default(tmp_path, open_store):
    store = open_store(tmp_path)

    defaults = store.get_settings()
    assert defaults["default_model_id"] == "gpt-5.4"
    assert defaults["memory_enabled"] is True
    assert defaults["last_map_id"] is None

    updated = store.patch_settings(
        {
            "locale": "en-US",
            "default_model_id": "deepseek-v4-flash",
            "remember_asset_choices": False,
        }
    )
    assert updated["locale"] == "en-US"
    assert updated["default_model_id"] == "deepseek-v4-flash"
    assert updated["remember_asset_choices"] is False


# 功能：
#   验证资产记忆绑定 ID 与摘要，关闭本机资产记忆时同时清除两类身份。
# 输入：
#   tmp_path：独立测试目录。
#   open_store：负责关闭连接的存储工厂。
# 输出：
#   None：不返回业务数据。
def test_asset_memory_tracks_valid_thread_choices_and_clears_when_disabled(tmp_path, open_store):
    store = open_store(tmp_path)
    thread = store.create_thread("巡检", "gpt-5.4")

    store.patch_thread(
        str(thread["thread_id"]),
        {
            "selected_map_id": "school-map",
            "selected_map_content_sha256": "a" * 64,
            "selected_vehicle_id": "dd-x4",
            "selected_vehicle_content_sha256": "b" * 64,
        },
    )
    remembered = store.get_settings()
    assert remembered["last_map_id"] == "school-map"
    assert remembered["last_map_content_sha256"] == "a" * 64
    assert remembered["last_vehicle_id"] == "dd-x4"
    assert remembered["last_vehicle_content_sha256"] == "b" * 64

    cleared = store.patch_settings({"remember_asset_choices": False})
    assert cleared["last_map_id"] is None
    assert cleared["last_map_content_sha256"] is None
    assert cleared["last_vehicle_id"] is None
    assert cleared["last_vehicle_content_sha256"] is None


# 功能：
#   验证消息顺序与归档开关在重读时保持，未明确请求时不返回归档任务。
# 输入：
#   tmp_path：独立测试目录。
#   open_store：负责关闭连接的存储工厂。
# 输出：
#   None：不返回业务数据。
def test_thread_messages_and_archive_are_durable(tmp_path, open_store):
    store = open_store(tmp_path)
    thread = store.create_thread("Pickup", "gpt-5.4", "en-US")
    assert thread["locale"] == "en-US"
    message = store.append_message(
        str(thread["thread_id"]), role="user", kind="text", content="去保安亭取外卖"
    )

    loaded = store.get_thread(str(thread["thread_id"]))
    assert loaded["messages"] == [message]
    assert len(store.list_threads()) == 1

    store.patch_thread(str(thread["thread_id"]), {"archived": True})
    assert store.list_threads() == []
    assert len(store.list_threads(include_archived=True)) == 1


# 功能：
#   检查真实内置目录包含核心能力标识，而非只验证界面模拟列表；测试后释放管理器。
# 输入：
#   tmp_path：独立测试目录。
#   open_store：负责关闭连接的存储工厂。
# 输出：
#   None：不返回业务数据。
def test_builtin_plugins_are_real_core_capabilities(tmp_path, open_store):
    store = open_store(tmp_path)
    manager = PluginManager(store)
    try:
        plugin_ids = {item["plugin_id"] for item in store.list_plugins()}
    finally:
        manager.close()
    assert "navigation.shortest-route" in plugin_ids
    assert "safety.route-clearance" in plugin_ids
    assert "px4.export-track" in plugin_ids
    assert "simulation.gazebo-px4" in plugin_ids
    assert "model.openai" in plugin_ids
