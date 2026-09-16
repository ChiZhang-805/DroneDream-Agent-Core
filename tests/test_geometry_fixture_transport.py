"""Exercise calibration RPC ownership without starting Gazebo or an aircraft."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

import dronedream_agent_core.geometry_fixture_transport as transport

PARTITION = "dronedream-geometry-fixture-" + "a" * 32
POSE = {"position_m": [1., 2., 3.], "orientation_wxyz": [1., 0., 0., 0.]}


class FakePipe:
    # 功能：
    #   建立可控制应答与超时顺序的内存管道，不建立原生通信连接。
    # 输入：
    #   self：当前测试管道。
    #   replies：依次接收的字节帧。
    #   polls：依次等待时的就绪结果。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, replies=(), polls=()):
        self.replies, self.polls = deque(replies), deque(polls)
        self.sent = []
        self.received = 0
        self.closed = False

    # 功能：
    #   记录父进程发送的完整帧，关闭后模拟真实管道错误。
    # 输入：
    #   self：当前测试管道。
    #   value：发送字节。
    # 输出：
    #   None：不返回业务数据。
    def send_bytes(self, value):
        if self.closed:
            raise OSError("pipe closed")
        self.sent.append(value)

    # 功能：
    #   返回下一次受控就绪结果，不进行实际等待。
    # 输入：
    #   self：当前测试管道。
    #   timeout：调用方声明的等待秒数。
    # 输出：
    #   ready：有预设结果时使用该结果，否则依据应答队列判断。
    def poll(self, timeout):
        ready = self.polls.popleft() if self.polls else bool(self.replies)
        return ready

    # 功能：
    #   接收一帧并核对调用方使用了字节预算，记录实际读取次数。
    # 输入：
    #   self：当前测试管道。
    #   limit：允许的帧字节数。
    # 输出：
    #   value：下一帧应答字节。
    def recv_bytes(self, limit):
        self.received += 1
        value = self.replies.popleft()
        assert len(value) <= limit
        return value

    # 功能：
    #   记录管道已关闭，供失败清理断言使用。
    # 输入：
    #   self：当前测试管道。
    # 输出：
    #   None：不返回业务数据。
    def close(self):
        self.closed = True


class FakeProcess:
    # 功能：
    #   建立受控子进程状态，支持启动失败与强制终止路径。
    # 输入：
    #   self：当前测试子进程。
    #   start_error：是否在启动时抛出异常。
    #   graceful：等待时是否正常结束。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, *, start_error=False, graceful=True):
        self.start_error, self.graceful = start_error, graceful
        self.alive, self.started, self.terminated = False, False, False
        self.exitcode = None

    # 功能：
    #   模拟进程启动，故障时不伪造已经启动的状态。
    # 输入：
    #   self：当前测试子进程。
    # 输出：
    #   None：不返回业务数据。
    def start(self):
        if self.start_error:
            raise OSError("start failed")
        self.alive = self.started = True

    # 功能：
    #   读取受控存活状态。
    # 输入：
    #   self：当前测试子进程。
    # 输出：
    #   alive：当前是否存活。
    def is_alive(self):
        alive = self.alive
        return alive

    # 功能：
    #   验证不会等待尚未启动的进程，并模拟正常退出。
    # 输入：
    #   self：当前测试子进程。
    #   timeout：允许的等待秒数。
    # 输出：
    #   None：不返回业务数据。
    def join(self, timeout):
        assert self.started
        if self.graceful:
            self.alive, self.exitcode = False, 0

    # 功能：
    #   模拟强制终止，保留非正常退出码。
    # 输入：
    #   self：当前测试子进程。
    # 输出：
    #   None：不返回业务数据。
    def terminate(self):
        self.terminated, self.alive, self.exitcode = True, False, -15


# 功能：
#   替换隔离夹具的进程工厂与平台视图，仅当前模块看到 Linux，不改全局平台。
# 输入：
#   monkeypatch：pytest 替换工具。
#   replies：握手后的命令应答帧。
#   polls：等待结果序列。
#   process：可选的受控进程。
#   ready：握手应答帧。
# 输出：
#   state：父管道、子管道及进程对象。
def install_worker(monkeypatch, replies=(), polls=(), process=None, ready=b'{"ready":true}'):
    parent = FakePipe((ready, *replies), polls)
    child, process = FakePipe(), process or FakeProcess()

    # 功能：
    #   返回预先建立的管道对，检查请求的是双工通信。
    # 输入：
    #   duplex：是否为双工管道。
    # 输出：
    #   pair：父端与子端。
    def pipe(*, duplex):
        assert duplex is True
        pair = parent, child
        return pair

    # 功能：
    #   核对进程目标与隔离参数后返回测试进程。
    # 输入：
    #   options：进程构造参数。
    # 输出：
    #   process：受控进程对象。
    def factory(**options):
        assert options["target"] is transport._request_worker
        assert options["args"] == (child, PARTITION, "fixture")
        return process

    # 功能：
    #   提供 spawn 模式的假上下文，阻止测试启动原生服务。
    # 输入：
    #   method：所请求的多进程启动模式。
    # 输出：
    #   context：管道与进程工厂。
    def context_factory(method):
        assert method == "spawn"
        context = SimpleNamespace(Pipe=pipe, Process=factory)
        return context

    monkeypatch.setattr(transport, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(transport, "multiprocessing", SimpleNamespace(get_context=context_factory))
    state = parent, child, process
    return state


# 功能：
#   复现超时后迟到回执，要求后续姿态或停止请求均不能读取该回执。
# 输入：
#   monkeypatch：pytest 替换工具。
# 输出：
#   None：不返回业务数据。
def test_timeout_invalidates_worker_before_next_command(monkeypatch):
    parent, _, _ = install_worker(monkeypatch,
        replies=(b'{"acknowledged":true,"service_latency_ms":900}',), polls=(True, False, True))
    client = transport.FixturePoseRequests(PARTITION, "fixture")
    with pytest.raises(RuntimeError, match="TIMED_OUT"):
        client.request(POSE)
    sends = len(parent.sent)
    with pytest.raises(RuntimeError):
        client.request(POSE)
    assert client.stop_server() is False
    assert parent.received == 1 and len(parent.sent) == sends
    assert parent.closed


# 功能：
#   核对启动失败时两端管道均关闭，不等待未启动的子进程。
# 输入：
#   monkeypatch：pytest 替换工具。
# 输出：
#   None：不返回业务数据。
def test_failed_start_closes_both_pipe_ends(monkeypatch):
    parent, child, _ = install_worker(monkeypatch, process=FakeProcess(start_error=True))
    with pytest.raises(OSError, match="start failed"):
        transport.FixturePoseRequests(PARTITION, "fixture")
    assert parent.closed and child.closed


# 功能：
#   拒绝类型混淆、重复键、非有限耗时及额外字段，防止非法回执被当作控制证据。
# 输入：
#   monkeypatch：pytest 替换工具。
#   reply：畸形应答字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("reply", [b'{"acknowledged":"true","service_latency_ms":1}',
    b'{"acknowledged":true,"acknowledged":false,"service_latency_ms":1}',
    b'{"acknowledged":true,"service_latency_ms":NaN}',
    b'{"acknowledged":true,"service_latency_ms":-1}',
    b'{"acknowledged":true,"service_latency_ms":true}',
    b'{"acknowledged":true,"service_latency_ms":1,"extra":2}', b'[]'])
def test_malformed_reply_closes_worker(monkeypatch, reply):
    parent, _, _ = install_worker(monkeypatch, replies=(reply,))
    client = transport.FixturePoseRequests(PARTITION, "fixture")
    with pytest.raises(RuntimeError):
        client.request(POSE)
    assert parent.closed


# 功能：
#   正常拒绝回执仍是已完成的请求，后续请求可以继续；重复关闭不增加通信。
# 输入：
#   monkeypatch：pytest 替换工具。
# 输出：
#   None：不返回业务数据。
def test_valid_negative_reply_and_idempotent_close(monkeypatch):
    parent, _, _ = install_worker(monkeypatch, replies=(
        b'{"acknowledged":false,"service_latency_ms":1}',
        b'{"acknowledged":true,"service_latency_ms":2}', b'{"acknowledged":true}'))
    client = transport.FixturePoseRequests(PARTITION, "fixture")
    assert client.request(POSE)["acknowledged"] is False
    assert client.request(POSE)["service_latency_ms"] == 2
    assert client.stop_server() is True
    assert client.close() is True
    count = len(parent.sent)
    assert client.close() is True and len(parent.sent) == count


# 功能：
#   握手 ready 必须是真布尔值，不能以整数一或重复字段冒充。
# 输入：
#   monkeypatch：pytest 替换工具。
#   ready：候选握手帧。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("ready", [b'{"ready":1}', b'{"ready":false,"ready":true}'])
def test_ready_is_strict_and_unique(monkeypatch, ready):
    parent, _, _ = install_worker(monkeypatch, ready=ready)
    with pytest.raises(RuntimeError, match="NOT_READY"):
        transport.FixturePoseRequests(PARTITION, "fixture")
    assert parent.closed


# 功能：
#   强制终止属于清理失败而非正常退出，重复关闭保留同一失败结论。
# 输入：
#   monkeypatch：pytest 替换工具。
# 输出：
#   None：不返回业务数据。
def test_forced_cleanup_is_not_reported_as_graceful(monkeypatch):
    parent, child, process = install_worker(monkeypatch, process=FakeProcess(graceful=False))
    client = transport.FixturePoseRequests(PARTITION, "fixture")
    assert client.close() is False and client.close() is False
    assert parent.closed and child.closed and process.terminated


# 功能：
#   验证姿态请求等待期间停止请求不能插入同一管道，两个回执分别归属原请求。
# 输入：
#   monkeypatch：pytest 替换工具。
# 输出：
#   None：不返回业务数据。
def test_pose_and_stop_requests_are_serialized(monkeypatch):
    parent, _, _ = install_worker(monkeypatch, replies=(
        b'{"acknowledged":true,"service_latency_ms":1}', b'{"acknowledged":true}'))
    client = transport.FixturePoseRequests(PARTITION, "fixture")
    waiting, release, stop_started = Event(), Event(), Event()
    original_poll = parent.poll

    # 功能：
    #   只阻塞姿态请求的第一次接收，为第二线程提供可复现的交错窗口。
    # 输入：
    #   timeout：声明的等待秒数。
    # 输出：
    #   ready：原管道的就绪结果。
    def blocked_poll(timeout):
        if not waiting.is_set():
            waiting.set()
            assert release.wait(3)
        ready = original_poll(timeout)
        return ready

    # 功能：
    #   标记第二线程已进入停止调用，返回其独立回执结果。
    # 输入：
    #   无。
    # 输出：
    #   accepted：当前停止请求是否得到确认。
    def stop():
        stop_started.set()
        accepted = client.stop_server()
        return accepted

    monkeypatch.setattr(parent, "poll", blocked_poll)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pose_result = pool.submit(client.request, POSE)
        try:
            assert waiting.wait(3)
            stop_result = pool.submit(stop)
            assert stop_started.wait(3)
            assert len(parent.sent) == 1
        finally:
            release.set()
        assert pose_result.result(timeout=3)["service_latency_ms"] == 1
        assert stop_result.result(timeout=3) is True
    assert client.close() is True
