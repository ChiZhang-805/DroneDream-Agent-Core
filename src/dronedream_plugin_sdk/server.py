"""Small dependency-light MCP server with schema validation and cancellation."""

from __future__ import annotations

import math
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from .protocol import (
    MAX_MESSAGE_BYTES,
    copy_json,
    decode_json,
    encode_json,
    read_frame,
    valid_request_id,
    validate_local_schema,
)


@dataclass(frozen=True)
class ToolContext:
    """Per-call values and cooperative cancellation; handlers must check the event."""

    request_id: str | int
    cancelled: threading.Event
    configuration: dict[str, Any]
    _notify: Callable[[dict[str, Any]], None] = field(repr=False)

    # 功能：
    #   校验进度数值并限制其范围及文案长度；已观察到取消时不再发送新的进度通知。
    # 输入：
    #   progress：有限进度数值，超出零至一范围时截到边界。
    #   message：附带进度文案，最多发送前三百个字符。
    # 输出：
    #   None：不返回业务数据。
    def progress(self, progress: float, message: str = "") -> None:
        if type(progress) not in (int, float) or (
            type(progress) is float and not math.isfinite(progress)
        ):
            raise ValueError("TOOL_PROGRESS_INVALID")
        if not isinstance(message, str):
            raise ValueError("TOOL_PROGRESS_MESSAGE_INVALID")
        if self.cancelled.is_set():
            return
        self._notify(
            {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {
                    "requestId": self.request_id,
                    # 先截范围再转浮点，避免有限的大整数在 isfinite 或 float 中溢出。
                    "progress": float(max(0, min(1, progress))),
                    "message": message[:300],
                },
            }
        )


@dataclass(frozen=True)
class ToolSpec:
    """A declared input/output contract and its local implementation callback."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], dict[str, Any]]


class McpPluginServer:
    """Serve the DroneDream MCP profile with one tool worker and a live reader.

    Tool callbacks remain serial, but cancellation and ping never wait for the
    callback to return. Cancellation is cooperative, not forced thread killing;
    an uncooperative callback keeps the tool slot occupied until it exits.
    """

    # 功能：
    #   在启动工具前校验并复制 Schema，拒绝重复工具名，建立独立的状态锁、写锁和取消登记。
    # 输入：
    #   name：插件服务名称。
    #   version：插件服务版本。
    #   tools：输入输出契约及其本地实现的工具集合。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, *, name: str, version: str, tools: list[ToolSpec]) -> None:
        self.name = name
        self.version = version
        self.tools = {
            tool.name: ToolSpec(
                tool.name,
                tool.description,
                validate_local_schema(tool.input_schema),
                validate_local_schema(tool.output_schema),
                tool.handler,
            )
            for tool in tools
        }
        if len(self.tools) != len(tools):
            raise ValueError("duplicate tool name")
        self.configuration: dict[str, Any] = {}
        self._cancelled: dict[str | int, threading.Event] = {}
        self._workers: set[threading.Thread] = set()
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._closed = False

    # 功能：
    #   为换行符预留预算，并通过单一写锁发送完整 UTF-8 帧，避免多个线程的消息交错。
    # 输入：
    #   value：准备发送的完整 JSON-RPC 消息；标准输出仅用于协议，不写业务日志。
    # 输出：
    #   None：不返回业务数据。
    def _write(self, value: dict[str, Any]) -> None:
        rendered = encode_json(value, limit=MAX_MESSAGE_BYTES - 1)
        with self._write_lock:
            binary = getattr(sys.stdout, "buffer", None)
            if binary is not None:
                binary.write((rendered + "\n").encode("utf-8"))
                binary.flush()
            else:
                sys.stdout.write(rendered + "\n")
                sys.stdout.flush()

    # 功能：
    #   发送成功回复，保留请求标识的原始 JSON 类型，不把整数标识变成字符串。
    # 输入：
    #   request_id：已验证的请求标识。
    #   result：准备回传的结果对象。
    # 输出：
    #   None：不返回业务数据。
    def _result(self, request_id: object, result: object) -> None:
        self._write({"jsonrpc": "2.0", "id": request_id, "result": result})

    # 功能：
    #   发送稳定的协议错误信息，不透传工具异常文本或回调中的敏感载荷。
    # 输入：
    #   request_id：错误关联的请求标识；无法可靠关联时为 None。
    #   code：协议错误码。
    #   message：调用方选择的稳定问题标识。
    # 输出：
    #   None：不返回业务数据。
    def _error(self, request_id: object, code: int, message: str) -> None:
        self._write(
            {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
        )

    # 功能：
    #   1. 检查工具名和参数，复制参数及配置后为唯一工作槽登记取消事件。
    #   2. 启动失败时撤销登记；繁忙或重复标识不能提前结束另一条仍在执行的调用。
    # 输入：
    #   request_id：本次工具调用的请求标识。
    #   params：包含工具名和 arguments 的参数字典。
    # 输出：
    #   None：不返回业务数据。
    def _call_tool(self, request_id: str | int, params: dict[str, Any]) -> None:
        name = params.get("name")
        arguments = params.get("arguments", {})
        tool = self.tools.get(name) if isinstance(name, str) else None
        if tool is None or not isinstance(arguments, dict):
            return self._error(request_id, -32602, "TOOL_CALL_INVALID")
        arguments = copy_json(arguments)
        cancellation = threading.Event()
        with self._state_lock:
            duplicate = request_id in self._cancelled
            busy = self._closed or bool(self._cancelled)
            if not busy:
                configuration = copy_json(self.configuration)
                self._cancelled[request_id] = cancellation
                worker = threading.Thread(
                    target=self._execute_tool,
                    args=(request_id, tool, arguments, configuration, cancellation),
                    name="dronedream-plugin-tool",
                    daemon=True,
                )
                self._workers.add(worker)
                # 登记和启动使用同一状态锁，避免取消发生在登记已出现但线程尚未启动的间隙。
                try:
                    worker.start()
                except BaseException:
                    self._workers.remove(worker)
                    self._cancelled.pop(request_id, None)
                    raise
        if busy:
            # 重复标识的错误不能携带原标识，否则宿主可能误把原请求记为已完成。
            self._error(None if duplicate else request_id, -32000, "TOOL_SERVER_BUSY")

    # 功能：
    #   1. 在工具执行前后按 Schema 和 JSON 预算验证数据，错误仅发布稳定问题标识。
    #   2. 在构造回复后再次检查取消，避免输出校验期间收到的取消被漏掉；已发出的消息不能撤回。
    #   3. 无论成功、失败或取消，都在回调退出后释放该调用的执行登记。
    # 输入：
    #   request_id：本次调用标识。
    #   tool：已注册的工具契约及实现。
    #   arguments：隔离后的调用参数。
    #   configuration：调用启动时冻结的配置副本。
    #   cancellation：本次调用独立的协作取消事件。
    # 输出：
    #   None：不返回业务数据。
    def _execute_tool(
        self,
        request_id: str | int,
        tool: ToolSpec,
        arguments: dict[str, Any],
        configuration: dict[str, Any],
        cancellation: threading.Event,
    ) -> None:
        try:
            if cancellation.is_set():
                return
            jsonschema.validate(arguments, tool.input_schema)
            output = tool.handler(
                arguments,
                ToolContext(request_id, cancellation, configuration, self._write),
            )
            if cancellation.is_set():
                return
            output = copy_json(output)
            if not isinstance(output, dict):
                raise ValueError("TOOL_OUTPUT_NOT_OBJECT")
            jsonschema.validate(output, tool.output_schema)
            result = {
                "content": [{"type": "text", "text": encode_json(output)}],
                "structuredContent": output,
                "isError": False,
            }
            # 校验与序列化都可能耗时；取消检查必须位于它们之后，而不只在工具返回时检查。
            if cancellation.is_set():
                return
            self._result(request_id, result)
        except jsonschema.ValidationError:
            if not cancellation.is_set():
                self._error(request_id, -32602, "TOOL_SCHEMA_VALIDATION_FAILED")
        except Exception as error:
            if not cancellation.is_set():
                self._error(request_id, -32000, f"TOOL_EXECUTION_FAILED:{type(error).__name__}")
        finally:
            with self._state_lock:
                self._cancelled.pop(request_id, None)
                self._workers.discard(threading.current_thread())

    # 功能：
    #   1. 校验请求外层结构与标识，同步处理初始化、探测和取消，工具交给独立工作线程。
    #   2. 初始化配置只影响后续调用，不改变运行中工具已经持有的快照。
    # 输入：
    #   message：解析得到的候选 JSON-RPC 消息。
    # 输出：
    #   None：不返回业务数据。
    def dispatch(self, message: dict[str, Any]) -> None:
        message = copy_json(message)
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return self._error(None, -32600, "INVALID_REQUEST")
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params", {})
        if method == "notifications/cancelled" and isinstance(params, dict):
            # 标识必须保持原类型；错误类型或已经结束的请求不能取消其他调用。
            cancelled_id = params.get("requestId")
            if valid_request_id(cancelled_id):
                with self._state_lock:
                    cancelled = self._cancelled.get(cancelled_id)
                    if cancelled is not None:
                        cancelled.set()
            return
        if method == "notifications/initialized":
            return
        if request_id is None:
            return
        if not valid_request_id(request_id) or not isinstance(method, str):
            return self._error(None, -32600, "INVALID_REQUEST")
        if not isinstance(params, dict):
            return self._error(request_id, -32602, "INVALID_PARAMS")
        if method == "initialize":
            options = params.get("initializationOptions", {}) if isinstance(params, dict) else {}
            configuration = options.get("configuration", {}) if isinstance(options, dict) else {}
            with self._state_lock:
                self.configuration = (
                    copy_json(configuration) if isinstance(configuration, dict) else {}
                )
            return self._result(
                request_id,
                {
                    "protocolVersion": "dronedream.plugin.v1",
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {"name": self.name, "version": self.version},
                },
            )
        if method == "ping":
            return self._result(request_id, {})
        if method == "tools/list":
            return self._result(
                request_id,
                {
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.input_schema,
                            "outputSchema": tool.output_schema,
                        }
                        for tool in self.tools.values()
                    ]
                },
            )
        if method == "tools/call" and isinstance(params, dict):
            return self._call_tool(request_id, params)
        if method == "resources/list":
            return self._result(request_id, {"resources": []})
        self._error(request_id, -32601, "METHOD_NOT_FOUND")

    # 功能：
    #   逐帧读取并分派请求；帧读取超限时结束通道，不能把剩余碎片作为下一条命令继续执行。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def run(self) -> None:
        try:
            while True:
                try:
                    line = read_frame(sys.stdin)
                except ValueError:
                    self._error(None, -32700, "FRAME_TOO_LARGE")
                    break
                if line is None:
                    break
                try:
                    self.dispatch(decode_json(line))
                except ValueError:
                    self._error(None, -32700, "PARSE_ERROR")
        finally:
            self.close()

    # 功能：
    #   停止接受新工具调用，通知现存工具取消并有限等待；不强杀线程，最终进程回收由宿主负责。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        with self._state_lock:
            self._closed = True
            workers = list(self._workers)
            for cancellation in self._cancelled.values():
                cancellation.set()
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join(timeout=1.0)
