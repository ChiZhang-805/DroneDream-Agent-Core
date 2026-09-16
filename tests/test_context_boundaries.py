"""Local SQLite regressions for append ordering and summary publication; no account data."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from dronedream_agent_core.context import ContextStore


# 功能：
#   写入合成用户事件，让摘要游标测试必须对应真实已落库的行。
# 输入：
#   store：隔离的测试上下文存储。
#   count：需要写入的事件数量。
# 输出：
#   None：不返回业务数据。
def _populate(store, count=4):
    for index in range(count):
        store.append("thread", role="user", event_type="message", payload={"index": index})


# 功能：
#   验证摘要游标拒绝类型错误及超出已保存事件的序号，不隐藏尚未观察到的事件。
# 输入：
#   tmp_path：测试 SQLite 文件所在目录。
#   through：非法摘要游标。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("through", [True, 1.5, 999])
def test_summary_cannot_hide_unobserved_events_or_use_untyped_watermark(tmp_path, through):
    store = ContextStore(tmp_path / "context.sqlite3")
    try:
        _populate(store)
        with pytest.raises(ValueError):
            store.set_summary("thread", "untrusted summary", through)
        assert store.window("thread").summary is None
        assert len(store.window("thread").recent_events) == 4
    finally:
        store.close()


# 功能：
#   验证较旧游标的迟到摘要不能覆盖已经发布的新摘要。
# 输入：
#   tmp_path：隔离数据库目录。
# 输出：
#   None：不返回业务数据。
def test_old_summary_cannot_overwrite_newer_summary(tmp_path):
    store = ContextStore(tmp_path / "context.sqlite3")
    try:
        _populate(store)
        store.set_summary("thread", "current", 3)
        with pytest.raises(ValueError, match="older"):
            store.set_summary("thread", "stale", 1)
        assert store.window("thread").summary == "current"
    finally:
        store.close()


# 功能：
#   验证四个独立连接并发追加同一对话时，所有事件获得唯一且连续的持久化序号。
# 输入：
#   tmp_path：多连接共享的测试数据库目录。
# 输出：
#   None：不返回业务数据。
def test_independent_connections_allocate_unique_ordered_sequences(tmp_path):
    path = tmp_path / "context.sqlite3"
    ContextStore(path).close()
    ready = Barrier(4)

    # 功能：
    #   在线程自身创建和关闭连接，同步起跑后连续写入事件以制造序号分配竞争。
    # 输入：
    #   worker：用于区分事件来源的测试工作线程编号。
    # 输出：
    #   sequences：该连接成功写入的事件序号列表。
    def write_events(worker):
        store = ContextStore(path)
        try:
            ready.wait(timeout=20)
            sequences = [
                store.append(
                    "thread",
                    role="user",
                    event_type="message",
                    payload={"worker": worker, "index": index},
                ).sequence
                for index in range(20)
            ]
            return sequences
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        sequences = [sequence for batch in pool.map(write_events, range(4)) for sequence in batch]
    assert sorted(sequences) == list(range(1, 81))
    store = ContextStore(path)
    try:
        assert len(store.window("thread", max_recent_events=100).recent_events) == 80
    finally:
        store.close()


# 功能：
#   验证非法事件导致事务回滚后，不占用序号或遗留写锁，后续合法写入仍从首号开始。
# 输入：
#   tmp_path：隔离数据库目录。
# 输出：
#   None：不返回业务数据。
def test_rejected_append_rolls_back_and_releases_writer_lock(tmp_path):
    store = ContextStore(tmp_path / "context.sqlite3")
    try:
        with pytest.raises(ValueError):
            store.append("thread", role="unknown", event_type="message", payload={})
        assert store.append("thread", role="user", event_type="message", payload={}).sequence == 1
    finally:
        store.close()


# 功能：
#   验证非法嵌套载荷被拒绝，不把 NaN、无穷值或未知对象静默转成空值后存入上下文。
# 输入：
#   tmp_path：隔离数据库目录。
#   value：嵌入载荷的非法值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_context_append_cannot_silently_replace_invalid_payload_with_null(tmp_path, value):
    store = ContextStore(tmp_path / "context.sqlite3")
    try:
        with pytest.raises(ValueError):
            store.append("thread", role="user", event_type="message", payload={"nested": [value]})
        assert store.window("thread").recent_events == []
    finally:
        store.close()


# 功能：
#   验证长对话检索可偏向最新事件，而摘要仍从最早未处理页推进，不因分页丢失中间指令。
# 输入：
#   tmp_path：包含两百五十条合成指令的测试数据库目录。
# 输出：
#   None：不返回业务数据。
def test_summary_pages_advance_oldest_first_without_skipping_long_conversations(tmp_path):
    from dronedream_agent_plugins.context_plugins import _extractive_summary

    store = ContextStore(tmp_path / "context.sqlite3")
    try:
        for i in range(250):
            store.append(
                "thread",
                role="user",
                event_type="mission.request",
                payload={"message": f"instruction {i}"},
            )
        assert store.window("thread").recent_events[0].sequence == 227
        page = store.summary_window("thread")
        assert [event.sequence for event in page.recent_events] == list(range(1, 201))
        result = _extractive_summary(window=page)
        store.set_summary("thread", result["summary"], result["through_sequence"])
        page = store.summary_window("thread")
        assert [event.sequence for event in page.recent_events] == list(range(201, 251))
        assert page.summary == result["summary"]
    finally:
        store.close()


# 功能：
#   验证仅包含工具结果的新页推进游标时仍保留已有用户意图摘要，空页不伪造进度。
# 输入：
#   tmp_path：隔离数据库目录。
# 输出：
#   None：不返回业务数据。
def test_extractive_summary_carries_prior_summary_through_tool_only_pages(tmp_path):
    from dronedream_agent_plugins.context_plugins import _extractive_summary

    store = ContextStore(tmp_path / "context.sqlite3")
    try:
        store.append(
            "thread",
            role="user",
            event_type="mission.request",
            payload={"message": "return to the office"},
        )
        store.set_summary("thread", "User request: return to the office", 1)
        store.append("thread", role="tool", event_type="tool.result", payload={"ok": True})
        result = _extractive_summary(window=store.summary_window("thread"))
        assert result == {"summary": "User request: return to the office", "through_sequence": 2}
        store.set_summary("thread", result["summary"], result["through_sequence"])
        assert _extractive_summary(window=store.summary_window("thread"))["through_sequence"] == 0
    finally:
        store.close()


# 功能：
#   验证对模型提供的上下文副本不会反向修改原始窗口，且不允许混入其他对话的事件。
# 输入：
#   tmp_path：仅包含本次合成对话的数据库目录。
# 输出：
#   None：不返回业务数据。
def test_context_projection_detaches_nested_payload_and_rejects_foreign_events(tmp_path):
    from dronedream_agent_plugins.context_plugins import _extractive_summary, _structured_window

    store = ContextStore(tmp_path / "context.sqlite3")
    try:
        store.append(
            "thread",
            role="user",
            event_type="mission.request",
            payload={"message": "go", "nested": {"value": 1}},
        )
        window = store.window("thread")
        result = _structured_window(window=window)
        result["recent_events"][0]["payload"]["nested"]["value"] = 9
        assert window.recent_events[0].payload["nested"]["value"] == 1
        window.recent_events[0].conversation_id = "foreign-thread"
        with pytest.raises(ValueError, match="IDENTITY_OR_ORDER_INVALID"):
            _extractive_summary(window=window)
    finally:
        store.close()


# 功能：
#   验证重复请求不会挤掉十二条独立摘要的容量，空目标也不会作为新的有效信息入摘要。
# 输入：
#   tmp_path：隔离测试数据库目录。
# 输出：
#   None：不返回业务数据。
def test_extractive_summary_keeps_distinct_requests_despite_repetition(tmp_path):
    from dronedream_agent_plugins.context_plugins import _extractive_summary

    store = ContextStore(tmp_path / "context.sqlite3")
    try:
        for message in ["return to office", "avoid stairwell", *(["carry sample"] * 16)]:
            store.append(
                "thread", role="user", event_type="mission.request", payload={"message": message}
            )
        store.append(
            "thread",
            role="assistant",
            event_type="model.intent_parser.result",
            payload={"artifact": {"goal": "   "}},
        )
        result = _extractive_summary(window=store.summary_window("thread"))
        assert result["summary"].splitlines() == [
            "User request: return to office",
            "User request: avoid stairwell",
            "User request: carry sample",
        ]
        assert result["through_sequence"] == 19
    finally:
        store.close()
