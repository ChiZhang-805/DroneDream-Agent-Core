"""Exercise durable state boundaries using isolated SQLite stores, not user data."""

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from dronedream_agent_app.storage import AppStore, _qualification_payload


# 功能：
#   为每条反例创建独立存储，即使断言失败也关闭连接。
# 输入：
#   tmp_path：pytest 分配的临时目录。
# 输出：
#   value：独立应用存储实例。
@pytest.fixture
def store(tmp_path):
    value = AppStore(tmp_path)
    try:
        yield value
    finally:
        value.close()


# 功能：
#   验证被拒绝的普通嵌套事务不会回滚仍在使用的父事务。
# 输入：
#   store：独立测试存储。
# 输出：
#   None：不返回业务数据。
def test_nested_transaction_failure_cannot_rollback_parent(store):
    with store.transaction() as connection:
        connection.execute("INSERT INTO settings VALUES('theme', '\"dark\"')")
        with pytest.raises(RuntimeError, match="NESTED_TRANSACTION"), store.transaction():
            pytest.fail("Nested transaction unexpectedly entered")
        assert connection.in_transaction
    assert store.get_settings()["theme"] == "dark"


# 功能：
#   用真实线程验证共享连接的读取锁，未提交变更不能靠 WAL 名义泄露给其他线程。
# 输入：
#   store：独立测试存储。
# 输出：
#   None：不返回业务数据。
def test_reader_never_observes_another_threads_uncommitted_state(store):
    entered = Event()

    # 功能：
    #   通知主线程读取任务已开始，然后通过存储入口读取主题。
    # 输入：
    #   无；使用外层 entered 事件和 store。
    # 输出：
    #   theme：取得锁后读取到的已提交主题值。
    def read():
        entered.set()
        theme = store.get_settings()["theme"]
        return theme

    with ThreadPoolExecutor(max_workers=1) as executor:
        with store.transaction() as connection:
            connection.execute("INSERT INTO settings VALUES('theme', '\"dark\"')")
            pending = executor.submit(read)
            assert entered.wait(2)
            # The read cannot complete until the owning transaction releases its lock.
            from concurrent.futures import TimeoutError

            with pytest.raises(TimeoutError):
                pending.result(timeout=0.05)
            connection.rollback()
        assert pending.result(timeout=2) == "system"


# 功能：
#   验证记忆开关严格要求布尔值，非法更新不能先写入同一请求中的其他字段。
# 输入：
#   store：独立测试存储。
#   key：待验证的记忆开关名。
#   bad：非法开关值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "key", ["memory_enabled", "remember_asset_choices", "remember_task_preferences"]
)
@pytest.mark.parametrize("bad", ["false", 0, 1, None, [], {}])
def test_memory_flags_require_actual_booleans_without_partial_write(store, key, bad):
    with pytest.raises(ValueError, match="BOOLEAN"):
        store.patch_settings({"theme": "dark", key: bad})
    assert store.get_settings()["theme"] == "system"
    assert store.get_settings()[key] is True


# 功能：
#   验证模型启用值不能由文本、整数或空值的真值性推断，拒绝时不创建资料。
# 输入：
#   store：独立测试存储。
#   bad：非法启用值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", ["false", 1, None])
def test_custom_model_enablement_is_not_truthiness(store, bad):
    profile = dict(
        profile_id="profile",
        display_name="test",
        provider="custom",
        icon="custom",
        base_url="https://example.invalid/v1",
        api_style="openai-chat",
        model_id="test",
        enabled=bad,
    )
    with pytest.raises(ValueError, match="BOOLEAN"):
        store.save_custom_model(profile)
    assert store.list_custom_models() == []


# 功能：
#   验证非有限 JSON 数字使整个设置更新失败，前面字段也不能部分提交。
# 输入：
#   store：独立测试存储。
#   bad：非有限浮点数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_settings_reject_non_json_numbers_atomically(store, bad):
    with pytest.raises(ValueError):
        store.patch_settings({"theme": "dark", "extra": {"number": bad}})
    assert store.get_settings()["theme"] == "system"


# 功能：
#   验证清除地图选择后其记忆摘要也同步清空，不把旧摘要附到新的空选择上。
# 输入：
#   store：独立测试存储。
# 输出：
#   None：不返回业务数据。
def test_clearing_asset_selection_also_clears_remembered_identity(store):
    thread = store.create_thread("test", "test")
    store.patch_thread(
        thread["thread_id"], {"selected_map_id": "map", "selected_map_content_sha256": "a" * 64}
    )
    store.patch_thread(thread["thread_id"], {"selected_map_id": None})
    assert store.get_settings()["last_map_id"] is None
    assert store.get_settings()["last_map_content_sha256"] is None


# 功能：
#   验证任务标记写入拒绝文本 false，原布尔状态不变。
# 输入：
#   store：独立测试存储。
#   key：待更新的置顶或归档标记。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("key", ["pinned", "archived"])
def test_thread_flags_are_not_sqlite_text_truthiness(store, key):
    thread = store.create_thread("test", "test")
    with pytest.raises(ValueError, match="BOOLEAN"):
        store.patch_thread(thread["thread_id"], {key: "false"})
    assert store.get_thread(thread["thread_id"])[key] is False


# 功能：
#   用增长流验证回执读取始终带声明长度上限，超限后拒绝并关闭流。
# 输入：
#   monkeypatch：替换 Path.open 的 pytest 夹具。
# 输出：
#   None：不返回业务数据。
def test_qualification_read_stops_at_declared_bytes_even_after_file_growth(monkeypatch):
    limits = []

    class GrowingStream(io.BytesIO):
        # 功能：
        #   记录实际传入的读取上限后返回模拟流字节，用于区分限长读与整流读。
        # 输入：
        #   self：模拟增长流。
        #   size：调用方申请的最大字节数。
        # 输出：
        #   payload：流实际返回的字节。
        def read(self, size=-1):
            limits.append(size)
            payload = super().read(size)
            return payload

    stream = GrowingStream(b"x" * 1024)
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: stream)
    with pytest.raises(ValueError, match="size changed"):
        _qualification_payload(Path("unused"), 10)
    assert limits == [11]
    assert stream.closed


# 功能：
#   验证非法声明大小在访问文件系统前拒绝，不为坏元数据打开文件。
# 输入：
#   monkeypatch：替换文件打开函数的 pytest 夹具。
#   bad：零、负数、布尔或过大长度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", [0, -1, True, 8 * 1024 * 1024 + 1])
def test_invalid_receipt_size_is_rejected_before_open(monkeypatch, bad):
    # 功能：
    #   捕获本不应发生的文件访问并让测试立即失败。
    # 输入：
    #   _args：文件打开的位置参数。
    #   _kwargs：文件打开的关键字参数。
    # 输出：
    #   None：不返回业务数据。
    def unexpected_open(*_args, **_kwargs):
        pytest.fail("Invalid receipt size reached the filesystem")

    monkeypatch.setattr(Path, "open", unexpected_open)
    with pytest.raises(ValueError, match="size invalid"):
        _qualification_payload(Path("unused"), bad)
