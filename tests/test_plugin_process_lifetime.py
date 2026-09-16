"""插件进程和会话退出边界；使用自有替身，不寻找或终止产品进程。"""

import io
import queue
import threading
from types import SimpleNamespace

import pytest

import dronedream_agent_core.plugin_process as module
from dronedream_agent_core.plugin_contracts import PluginResourcePolicy
from dronedream_agent_core.plugin_process import McpSessionPool, McpStdioClient, PluginProcessError


# 功能：
#   构造具备客户端关闭状态的惰性进程替身，并记录每一步资源回收。
# 输入：
#   failed_step：需要注入异常的步骤名称。
# 输出：
#   fixture：客户端、清理事件和故障位置的元组。
def lifetime_fixture(failed_step=None):
    events = []
    fault = {"step": failed_step}

    # 功能：
    #   记录清理步骤，在指定步骤模拟系统错误。
    # 输入：
    #   step：被测代码当前执行的步骤。
    # 输出：
    #   None：不返回业务数据。
    def record(step):
        events.append(step)
        if step == fault["step"]:
            raise OSError(f"fixture {step}")

    class Pipe(io.StringIO):
        # 功能：
        #   为内存管道记录其通道名，保持真实流关闭语义。
        # 输入：
        #   name：管道名称。
        # 输出：
        #   None：不返回业务数据。
        def __init__(self, name):
            super().__init__()
            self.name = name

        # 功能：
        #   记录每个管道的关闭动作，允许检查首个失败后后续是否仍被清理。
        # 输入：
        #   无。
        # 输出：
        #   None：不返回业务数据。
        def close(self):
            record(self.name)
            super().close()

    process = SimpleNamespace(
        pid=123,
        poll=lambda: None,
        terminate=lambda: record("terminate"),
        kill=lambda: record("kill"),
        wait=lambda timeout: record("wait"),
        stdin=Pipe("stdin"),
        stdout=Pipe("stdout"),
        stderr=Pipe("stderr"),
    )
    client = object.__new__(McpStdioClient)
    client._process = process
    client._job = SimpleNamespace(close=lambda: record("job"))
    client._closed = False
    client._cleanup_complete = False
    client._close_lock = threading.Lock()
    client._threads = []
    client._pending = {"request": queue.Queue(maxsize=1)}
    client._pending_lock = threading.Lock()
    client._read_failure = None
    fixture = client, events, fault
    return fixture


# 功能：
#   验证单项退出失败不跳过其他资源，错误可见，且第二次关闭能重试未完成清理。
# 输入：
#   failed_step：本次清理的故障位置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failed_step", ["terminate", "job", "wait", "stdin"])
def test_client_cleanup_continues_and_can_be_retried(failed_step):
    client, events, fault = lifetime_fixture(failed_step)
    try:
        with pytest.raises(PluginProcessError, match="CLEANUP_FAILED"):
            client.close()
        assert "job" in events and "wait" in events
        assert "stdout" in events and "stderr" in events
        assert client._closed
        assert not client._cleanup_complete
        assert isinstance(client._pending["request"].get_nowait(), PluginProcessError)
        fault["step"] = None
        client.close()
        assert client._cleanup_complete
    finally:
        fault["step"] = None
        for stream in (client._process.stdin, client._process.stdout, client._process.stderr):
            io.StringIO.close(stream)


# 功能：
#   验证通信线程仍可能持有管道锁时，不在关闭线程上强行关闭缓冲管道。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_client_live_channel_defers_pipe_close():
    client, events, _fault = lifetime_fixture()
    client._threads = [
        SimpleNamespace(join=lambda timeout: events.append("join"), is_alive=lambda: True)
    ]
    try:
        with pytest.raises(PluginProcessError, match="CLEANUP_FAILED"):
            client.close()
        assert "join" in events
        assert not any(name in events for name in ("stdin", "stdout", "stderr"))
    finally:
        client._threads = []
        client.close()


# 功能：
#   验证会话池关闭或失效处理中的首个客户端失败，不阻断后续客户端回收。
# 输入：
#   operation：关闭全池或使同一插件的多个会话失效。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("operation", ["close", "invalidate"])
def test_pool_attempts_all_client_closes(operation):
    pool = McpSessionPool(heartbeat_interval_seconds=3600)
    events = []

    # 功能：
    #   模拟第一个会话关闭失败，检查其余会话是否仍被访问。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def reject_close():
        events.append("first")
        raise OSError("fixture close failed")

    pool._sessions = {
        ("fixture", "a", "b", "c"): SimpleNamespace(close=reject_close),
        ("fixture", "a", "b", "d"): SimpleNamespace(close=lambda: events.append("second")),
    }
    try:
        with pytest.raises(PluginProcessError, match="CLEANUP_FAILED"):
            if operation == "close":
                pool.close()
            else:
                pool.invalidate("fixture")
        assert events == ["first", "second"]
        assert not pool._sessions
    finally:
        pool.close()
    assert not pool._heartbeat_thread.is_alive()


# 功能：
#   验证启动通信线程失败时回收已经创建的子进程和 Job，不等到协议初始化才进入清理。
# 输入：
#   tmp_path：惰性插件的工作目录。
#   monkeypatch：替换进程创建、隔离包装与线程启动，禁止启动真实插件。
# 输出：
#   None：不返回业务数据。
def test_client_thread_start_failure_cleans_started_process(tmp_path, monkeypatch):
    fixture, events, _fault = lifetime_fixture()
    monkeypatch.setattr(module, "_isolated_command", lambda **kwargs: ["fixture"])
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: fixture._process)
    monkeypatch.setattr(module, "_WindowsJob", lambda *args: fixture._job)

    # 功能：
    #   在通信线程启动入口模拟资源不足，不让任何后台线程逃离本测试。
    # 输入：
    #   thread：被测代码准备启动的线程对象。
    # 输出：
    #   None：不返回业务数据。
    def reject_start(thread):
        raise RuntimeError("fixture thread start failed")

    monkeypatch.setattr(threading.Thread, "start", reject_start)
    try:
        with pytest.raises(RuntimeError, match="fixture thread start failed"):
            McpStdioClient(
                plugin_root=tmp_path,
                command=["fixture"],
                protocol_version="dronedream.plugin.v1",
                startup_timeout_seconds=1,
                call_timeout_seconds=1,
                resource_policy=PluginResourcePolicy(),
            )
        assert "job" in events and "wait" in events
        assert all(
            pipe.closed
            for pipe in (fixture._process.stdin, fixture._process.stdout, fixture._process.stderr)
        )
    finally:
        for pipe in (fixture._process.stdin, fixture._process.stdout, fixture._process.stderr):
            io.StringIO.close(pipe)


# 功能：
#   验证过大的超时在转换为浮点或送入操作系统等待前被拒绝，不产生溢出或后台异常。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_pool_huge_integer_interval_is_a_configuration_error():
    with pytest.raises(ValueError, match="INTERVAL_INVALID"):
        McpSessionPool(heartbeat_interval_seconds=1 << 10_000)


# 功能：
#   验证无效超时与非对象空配置在任何进程解析或创建前拒绝，不能被默认值掩盖。
# 输入：
#   tmp_path：客户端工作目录。
#   monkeypatch：阻止校验错误的参数进入启动阶段。
#   overrides：覆盖正常启动参数的非法输入。
#   issue：预期的配置错误代码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "overrides, issue",
    [
        ({"startup_timeout_seconds": 1 << 10_000}, "TIMEOUT_INVALID"),
        ({"call_timeout_seconds": threading.TIMEOUT_MAX * 2}, "TIMEOUT_INVALID"),
        ({"configuration": []}, "CONFIGURATION_INVALID"),
        ({"configuration": False}, "CONFIGURATION_INVALID"),
    ],
    ids=["huge-integer", "os-wait-overflow", "empty-list", "false-config"],
)
def test_client_rejects_invalid_startup_before_launch(tmp_path, monkeypatch, overrides, issue):
    monkeypatch.setattr(module, "_isolated_command", lambda **kwargs: pytest.fail("not launched"))
    options = {
        "plugin_root": tmp_path,
        "command": ["fixture"],
        "protocol_version": "dronedream.plugin.v1",
        "startup_timeout_seconds": 1,
        "call_timeout_seconds": 1,
    }
    options.update(overrides)
    with pytest.raises(PluginProcessError, match=issue):
        McpStdioClient(**options)


# 功能：
#   验证会话握手期间池已停止时，不把刚创建的会话发布给调用方。
# 输入：
#   tmp_path：惰性客户端根目录。
# 输出：
#   None：不返回业务数据。
def test_pool_stop_during_startup_closes_unpublished_client(tmp_path):
    pool = McpSessionPool(heartbeat_interval_seconds=3600)
    closed = []

    # 功能：
    #   模拟创建客户端期间收到关闭请求，返回一个可核对回收动作的惰性实例。
    # 输入：
    #   kwargs：原客户端构造参数。
    # 输出：
    #   client：本测试创建的客户端替身。
    def stop_during_creation(**kwargs):
        pool._stop.set()
        client = SimpleNamespace(close=lambda: closed.append(True))
        return client

    try:
        with pytest.raises(PluginProcessError, match="POOL_CLOSED"):
            pool.get(
                plugin_id="fixture",
                package_sha256="a" * 64,
                plugin_root=tmp_path,
                command=["fixture"],
                protocol_version="dronedream.plugin.v1",
                startup_timeout_seconds=1,
                call_timeout_seconds=1,
                configuration={},
                permissions=[],
                resource_policy=PluginResourcePolicy(),
                client_factory=stop_during_creation,
            )
        assert closed == [True]
        assert not pool._sessions
    finally:
        pool.close()
