"""Local transport fixtures: no downloaded plugin, network broker or flight is executed."""

from __future__ import annotations

import io
import math
import os
import queue
import subprocess
import sys
import threading
from types import SimpleNamespace

import jsonschema
import pytest

from dronedream_agent_core.plugin_contracts import PluginResourcePolicy
from dronedream_agent_core.plugin_process import (
    McpSessionPool,
    McpStdioClient,
    PluginProcessError,
    _WindowsJob,
)
from dronedream_plugin_sdk import McpPluginServer, ToolContext, ToolSpec
from dronedream_plugin_sdk.protocol import (
    copy_json,
    decode_json,
    encode_json,
    read_frame,
    valid_request_id,
    validate_local_schema,
)


# 功能：
#   验证出站编码拒绝非有限数值、非字符串键和未声明的容器或对象类型。
# 输入：
#   value：不可直接上线路传输的候选值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "key"}, (1, 2), object()])
def test_wire_encoding_rejects_non_json_values(value):
    with pytest.raises(ValueError):
        encode_json(value)


# 功能：
#   验证入站解析拒绝重复键、非有限数值和非法 UTF-8，不用宽松解码掩盖损坏输入。
# 输入：
#   value：含歧义或非法数值的 JSON 原文。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', b"\xff"])
def test_wire_decoder_rejects_ambiguous_or_non_finite_values(value):
    with pytest.raises(ValueError):
        decode_json(value)


# 功能：
#   验证复制隔离可变引用，同时拒绝循环、大节点集合及 UTF-8 字节超限内容。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_wire_copy_detaches_aliases_and_bounds_cycles_depth_nodes_and_utf8():
    original = {"shared": [1]}
    detached = copy_json(original)
    detached["shared"].append(2)
    assert original == {"shared": [1]}
    cycle = []
    cycle.append(cycle)
    for value in (cycle, [None] * 100_001):
        with pytest.raises(ValueError, match="COMPLEXITY"):
            encode_json(value)
    with pytest.raises(ValueError, match="SIZE"):
        encode_json({"text": "中" * 8}, limit=20)
    with pytest.raises(ValueError, match="SIZE"):
        decode_json('"中中中"', limit=8)


# 功能：
#   验证超长帧最多读取预算加一个探测字符，并区别正常帧与流结束。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_frame_reader_does_not_read_an_entire_unbounded_line():
    stream = io.StringIO("x" * 100 + "\n{}\n")
    with pytest.raises(ValueError, match="TOO_LARGE"):
        read_frame(stream, limit=10)
    assert stream.tell() == 11
    assert read_frame(io.StringIO("{}\n"), limit=10) == "{}\n"
    assert read_frame(io.StringIO(""), limit=10) is None


# 功能：
#   验证请求标识不隐式转换类型，也不接纳超出协议精度或长度的值。
# 输入：
#   value：非法 JSON-RPC 标识候选值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, False, 1.5, None, [], {}, "", "a" * 161, 2**53])
def test_rpc_ids_do_not_coerce_unsupported_types(value):
    assert not valid_request_id(value)


class RecordingServer(McpPluginServer):
    """Capture complete messages without sharing the process-global stdout."""

    # 功能：
    #   建立只向内存记录回复的工具服务，避免测试争用进程全局标准输出。
    # 输入：
    #   handler：测试工具的处理函数。
    #   schema：可选输入 Schema；省略时使用对象类型。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, handler, *, schema=None):
        self.messages = []
        self.condition = threading.Condition()
        super().__init__(
            name="fixture",
            version="1.0.0",
            tools=[
                ToolSpec(
                    "fixture",
                    "Local test only",
                    schema or {"type": "object"},
                    {"type": "object"},
                    handler,
                ),
            ],
        )

    # 功能：
    #   先检查回复能否按正式协议编码，再持锁记录并唤醒等待回复的测试线程。
    # 输入：
    #   value：服务准备发送的完整消息。
    # 输出：
    #   None：不返回业务数据。
    def _write(self, value):
        encode_json(value)
        with self.condition:
            self.messages.append(value)
            self.condition.notify_all()

    # 功能：
    #   在短期限内等待指定请求的回复，超时使测试失败而非无限等待。
    # 输入：
    #   request_id：需要等待的请求标识。
    # 输出：
    #   reply：与指定标识匹配的完整回复。
    def wait_for_reply(self, request_id):
        with self.condition:
            assert self.condition.wait_for(
                lambda: any(message.get("id") == request_id for message in self.messages),
                timeout=1.0,
            )
            reply = next(message for message in self.messages if message.get("id") == request_id)
            return reply


# 功能：
#   创建本地测试工具的标准调用消息，不执行工具本身。
# 输入：
#   request_id：调用标识。
#   arguments：工具输入参数；省略时使用空对象。
# 输出：
#   message：符合 JSON-RPC 外层结构的请求字典。
def call(request_id=1, arguments=None):
    message = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": "fixture", "arguments": arguments or {}},
    }
    return message


# 功能：
#   验证服务保留整数请求标识，并将工具返回的非法浮点值转换为错误回复。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_server_preserves_integer_request_id_and_rejects_non_finite_output():
    for result, expected in (({"ok": True}, "result"), ({"bad": math.nan}, "error")):
        server = RecordingServer(lambda _args, _context, current=result: current)
        try:
            server.dispatch(call(9))
            reply = server.wait_for_reply(9)
            assert type(reply["id"]) is int
            assert expected in reply
        finally:
            server.close()


# 功能：
#   验证重复活动请求不会用忙碌错误提前完成原调用，原工具完成后仍能正常回复。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_duplicate_active_id_cannot_complete_the_original_call_with_busy_error():
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   阻塞本地工具直到测试释放，以构造同一标识仍在执行的时间窗口。
    # 输入：
    #   _arguments：未使用的工具参数。
    #   _context：未使用的工具上下文。
    # 输出：
    #   result：释放后返回的完成标记。
    def handler(_arguments, _context):
        entered.set()
        assert release.wait(2.0)
        result = {"done": True}
        return result

    server = RecordingServer(handler)
    try:
        server.dispatch(call(7))
        assert entered.wait(1.0)
        server.dispatch(call(7))
        assert not any(message.get("id") == 7 for message in server.messages)
        release.set()
        assert "result" in server.wait_for_reply(7)
    finally:
        release.set()
        server.close()


# 功能：
#   验证读取线程可在工具执行时处理取消与 ping，取消后不发布该工具的迟到结果或进度。
# 输入：
#   monkeypatch：将标准输入替换为本测试拥有的队列流。
# 输出：
#   None：不返回业务数据。
def test_server_reader_cancels_running_work_and_remains_responsive(monkeypatch):
    entered, cancelled = threading.Event(), threading.Event()
    lines = queue.Queue()

    class InputStream:
        # 功能：
        #   用有界等待从测试队列取回预先构造的输入帧，不读取真实终端。
        # 输入：
        #   _limit：正式读取器传入的预算，夹具帧大小由测试代码限定。
        # 输出：
        #   line：下一条测试输入或表示结束的空字符串。
        def readline(self, _limit):
            line = lines.get(timeout=2.0)
            return line

    # 功能：
    #   等待取消后故意尝试发送进度和结果，用于验证迟到输出被抑制。
    # 输入：
    #   _arguments：未使用的工具参数。
    #   context：含取消事件和进度发布接口的工具上下文。
    # 输出：
    #   result：不应被取消后的调用发布的结果。
    def handler(_arguments, context):
        entered.set()
        assert context.cancelled.wait(1.0)
        context.progress(0.9, "must not be published after cancellation")
        cancelled.set()
        result = {"stale": True}
        return result

    server = RecordingServer(handler)
    monkeypatch.setattr("dronedream_plugin_sdk.server.sys.stdin", InputStream())
    runner = threading.Thread(target=server.run)
    runner.start()
    try:
        lines.put(encode_json(call(1)) + "\n")
        assert entered.wait(1.0)
        lines.put(
            encode_json(
                {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}}
            )
            + "\n"
        )
        lines.put('{"jsonrpc":"2.0","id":2,"method":"ping"}\n')
        assert server.wait_for_reply(2)["result"] == {}
        assert cancelled.wait(1.0)
    finally:
        lines.put("")
        runner.join(timeout=2.0)
        server.close()
    assert not runner.is_alive()
    assert not any(message.get("id") == 1 or "method" in message for message in server.messages)


# 功能：
#   验证取消不提前释放未退出工具的执行槽，且字符串标识不会误取消整数标识的请求。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cancel_does_not_release_an_uncooperative_callback_slot():
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   忽略取消事件并等待测试显式释放，以模拟暂不配合取消的工具。
    # 输入：
    #   _arguments：未使用的工具参数。
    #   _context：故意不响应其取消事件的上下文。
    # 输出：
    #   result：释放后的完成标记。
    def handler(_arguments, _context):
        entered.set()
        assert release.wait(2.0)
        result = {"done": True}
        return result

    server = RecordingServer(handler)
    try:
        server.dispatch(call(1))
        assert entered.wait(1.0)
        server.dispatch(
            {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": "1"}}
        )
        assert not server._cancelled[1].is_set()
        server.dispatch(
            {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}}
        )
        server.dispatch(call(2))
        assert server.wait_for_reply(2)["error"]["message"] == "TOOL_SERVER_BUSY"
        assert server._cancelled[1].is_set()
    finally:
        release.set()
        server.close()
    assert not any(message.get("id") == 1 for message in server.messages)


# 功能：
#   验证工具启动后外部修改 Schema、配置或参数，不会改变该调用已经接受的内容。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_server_snapshots_schema_configuration_and_arguments():
    entered, release = threading.Event(), threading.Event()
    configuration = {"nested": [1]}
    arguments = {"input": [2]}
    schema = {"type": "object", "required": ["input"]}

    # 功能：
    #   等待测试修改外部对象后回传已接受的输入快照，用于检查引用隔离。
    # 输入：
    #   value：当前调用收到的参数。
    #   context：当前调用收到的配置上下文。
    # 输出：
    #   result：工具实际读取的配置与参数。
    def handler(value, context):
        entered.set()
        assert release.wait(1.0)
        result = {"configuration": context.configuration, "arguments": value}
        return result

    server = RecordingServer(handler, schema=schema)
    try:
        schema["required"] = ["wrong"]
        server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {"initializationOptions": {"configuration": configuration}},
            }
        )
        server.dispatch(call("tool", arguments))
        assert entered.wait(1.0)
        configuration["nested"].append(3)
        arguments["input"].append(4)
        release.set()
        output = server.wait_for_reply("tool")["result"]["structuredContent"]
        assert output == {"configuration": {"nested": [1]}, "arguments": {"input": [2]}}
    finally:
        release.set()
        server.close()


# 功能：
#   验证远程 Schema 引用在服务构造阶段被拒绝，不触发网络取回。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_server_schema_cannot_fetch_remote_references():
    with pytest.raises(ValueError, match="EXTERNAL_REFERENCE"):
        RecordingServer(lambda _a, _c: {}, schema={"$ref": "https://example.test/schema"})


# 功能：
#   验证取消发生在输出校验期间时，校验结束后不会继续发布已经失效的工具结果。
# 输入：
#   monkeypatch：在输出 Schema 校验处加入测试可控的等待窗口。
# 输出：
#   None：不返回业务数据。
def test_cancellation_during_output_validation_suppresses_late_result(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original_validate = jsonschema.validate

    # 功能：
    #   仅阻塞工具输出的校验，保留原有输入校验和真实 Schema 规则。
    # 输入：
    #   instance：待验证的输入或输出值。
    #   schema：对应的 Schema。
    # 输出：
    #   result：原校验器的返回值。
    def validate_after_release(instance, schema):
        if instance == {"ready": True}:
            entered.set()
            assert release.wait(2.0)
        result = original_validate(instance, schema)
        return result

    server = RecordingServer(lambda _arguments, _context: {"ready": True})
    monkeypatch.setattr(jsonschema, "validate", validate_after_release)
    try:
        server.dispatch(call(31))
        assert entered.wait(1.0)
        server.dispatch(
            {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 31}}
        )
        assert server._cancelled[31].is_set()
    finally:
        release.set()
        server.close()
    assert not any(message.get("id") == 31 for message in server.messages)


# 功能：
#   验证引用检查允许示例数据和名为 $ref 的属性，但拒绝嵌套属性 Schema 中的外部引用。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_schema_reference_guards_distinguish_data_from_executable_schema_keywords():
    schema = {
        "type": "object",
        "properties": {"$ref": {"type": "string"}},
        "default": {"$ref": "https://example.test/data-only"},
    }
    assert validate_local_schema(schema) == schema
    with pytest.raises(ValueError, match="EXTERNAL_REFERENCE"):
        validate_local_schema({"properties": {"value": {"$ref": "https://example.test/schema"}}})


# 功能：
#   验证进度接口不把布尔值、字符串或非有限数值强制转换成可发布进度。
# 输入：
#   progress：非法进度候选值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("progress", [True, "0.1", math.nan, math.inf])
def test_progress_refuses_coercion_and_non_finite_numbers(progress):
    context = ToolContext("id", threading.Event(), {}, lambda _: None)
    with pytest.raises(ValueError, match="PROGRESS"):
        context.progress(progress)


# 功能：
#   验证有限但极大的整数进度先截到合法范围，不在转浮点时溢出；正常小数保持原值。
# 输入：
#   progress：合法数值类型的候选进度。
#   expected：应发布的截断后进度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "progress, expected",
    [(1 << 10_000, 1.0), (-(1 << 10_000), 0.0), (0.25, 0.25)],
    ids=["large-positive", "large-negative", "fraction"],
)
def test_finite_integer_progress_is_clamped_before_float_conversion(progress, expected):
    messages = []
    context = ToolContext("id", threading.Event(), {}, messages.append)
    context.progress(progress)
    assert messages[0]["params"]["progress"] == expected


# 功能：
#   仅建立协议读取器需要的内存状态，不通过构造函数启动外部程序。
# 输入：
#   text：模拟子进程输出的帧文本。
# 输出：
#   client：使用内存流与小型队列的测试客户端。
def reader_fixture(text):
    client = object.__new__(McpStdioClient)
    client._process = SimpleNamespace(stdout=io.StringIO(text))
    client._pending = {"request": queue.Queue(maxsize=1)}
    client._pending_lock = threading.Lock()
    client._notifications = queue.Queue(maxsize=2)
    client._maximum_message_bytes = 4096
    client._read_failure = None
    return client


# 功能：
#   验证入站帧格式、标识、数值或大小非法时，读取器记录失败并唤醒等待中的请求。
# 输入：
#   frame：故意违反协议的完整输出帧。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "frame",
    [
        '{"jsonrpc":"2.0","id":"request","result":{},"error":{}}\n',
        '{"jsonrpc":"wrong","id":"request","result":{}}\n',
        '{"jsonrpc":"2.0","id":true,"result":{}}\n',
        '{"jsonrpc":"2.0","id":"request","result":{"n":NaN}}\n',
        "x" * 5000 + "\n",
    ],
)
def test_client_reader_fails_closed_on_invalid_frames(frame):
    client = reader_fixture(frame)
    client._read_stdout()
    assert isinstance(client._pending["request"].get_nowait(), BaseException)
    assert client._read_failure is not None


# 功能：
#   验证重复回复不会阻塞容量为一的待回复队列，也不会被解释成宿主待执行工作。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_duplicate_reply_cannot_block_the_reader_or_become_host_work():
    frame = '{"jsonrpc":"2.0","id":"request","result":{}}\n'
    client = reader_fixture(frame + frame)
    client._read_stdout()
    assert str(client._read_failure) == "PLUGIN_DUPLICATE_RESPONSE"
    assert client._pending["request"].qsize() == 1


# 功能：
#   验证子进程管道写入阻塞时触发超时与回收；这里使用假管道，不终止真实产品进程。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_blocked_child_pipe_write_times_out_and_terminates_child():
    released, killed = threading.Event(), threading.Event()

    class BlockedPipe:
        # 功能：
        #   模拟写管道阻塞，直到假进程被回收或测试清理阶段释放。
        # 输入：
        #   _value：不实际写出的消息。
        # 输出：
        #   None：不返回业务数据。
        def write(self, _value):
            assert released.wait(2.0)

        # 功能：
        #   提供假管道所需的刷新接口，不产生文件或进程副作用。
        # 输入：
        #   无。
        # 输出：
        #   None：不返回业务数据。
        def flush(self):
            pass

    # 功能：
    #   记录假进程被回收并解除假管道阻塞，不调用操作系统终止接口。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def kill():
        killed.set()
        released.set()

    client = reader_fixture("")
    client._process = SimpleNamespace(stdin=BlockedPipe(), poll=lambda: None, kill=kill)
    client._closed = False
    client._write_lock = threading.Lock()
    client._writes = queue.Queue(maxsize=32)
    writer = threading.Thread(target=client._write_stdin)
    writer.start()
    try:
        with pytest.raises(PluginProcessError, match="WRITE_TIMEOUT"):
            client._send({"jsonrpc": "2.0", "method": "ping"}, timeout=0.02)
        assert killed.is_set()
    finally:
        client._closed = True
        released.set()
        writer.join(timeout=1.0)
    assert not writer.is_alive()


# 功能：
#   验证会话池拒绝非法存活探测周期，防止忙循环或无限等待。
# 输入：
#   interval：非法周期值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("interval", [True, math.nan, math.inf, 0, -1])
def test_session_pool_rejects_invalid_intervals(interval):
    with pytest.raises(ValueError, match="INTERVAL"):
        McpSessionPool(heartbeat_interval_seconds=interval)


# 功能：
#   验证真实 Windows Job 句柄能关闭本测试新建的子进程，失败清理仍只针对该进程对象。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
@pytest.mark.skipif(os.name != "nt", reason="Actual pointer-sized Windows Job handles")
def test_windows_job_closes_only_its_owned_test_process():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    job = None
    try:
        job = _WindowsJob(child.pid, PluginResourcePolicy())
        assert child.poll() is None
        job.close()
        child.wait(timeout=3.0)
        assert job._handle is None
    finally:
        if job is not None:
            job.close()
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3.0)


# 功能：
#   通过自有 Python 夹具验证真实管道、UTF-8 往返、超时取消和服务响应，不运行第三方插件。
# 输入：
#   tmp_path：存放本次夹具脚本的独立目录。
#   monkeypatch：仅替换可执行文件解析，其余通信和资源限制使用正式实现。
# 输出：
#   None：不返回业务数据。
def test_actual_sdk_stdio_pair_executes_locally_and_cancellation_frees_worker(
    tmp_path, monkeypatch
):
    from pathlib import Path

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = tmp_path / "fixture.py"
    script.write_text(
        f"import sys\nsys.path.insert(0, {str(source_root)!r})\n"
        "from dronedream_plugin_sdk import McpPluginServer, ToolSpec\n"
        "# 功能：\n#   回传参数，并在指定时配合测试取消。\n"
        "# 输入：\n#   value：工具参数。\n#   context：取消上下文。\n"
        "# 输出：\n#   result：带 received 字段的回复。\n"
        "def handler(value, context):\n"
        "    if value.get('wait'): context.cancelled.wait(2.0)\n"
        "    result = {'received': value}\n"
        "    return result\n"
        "McpPluginServer(name='local-fixture', version='1.0.0', tools=[\n"
        "    ToolSpec('fixture', 'Inert fixture', {'type':'object'},\n"
        "             {'type':'object'}, handler)]).run()\n",
        encoding="utf-8",
    )
    # Only executable resolution is replaced. Wire parsing, bounded writes,
    # cancellation, schema checks and the Windows Job are the production code.
    monkeypatch.setattr(
        "dronedream_agent_core.plugin_process._isolated_command",
        lambda **_kwargs: [sys.executable, str(script)],
    )
    with McpStdioClient(
        plugin_root=tmp_path,
        command=["fixture.py"],
        protocol_version="dronedream.plugin.v1",
        startup_timeout_seconds=10,
        call_timeout_seconds=2,
    ) as client:
        assert client.list_tools()[0]["name"] == "fixture"
        assert client.call_tool("fixture", {"text": "传感器"}) == {"received": {"text": "传感器"}}
        with pytest.raises(PluginProcessError, match="TIMEOUT"):
            client._request(
                "tools/call", {"name": "fixture", "arguments": {"wait": True}}, timeout=0.05
            )
        assert client.ping()
