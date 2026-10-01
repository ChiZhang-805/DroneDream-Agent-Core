"""Protocol regression tests for the bundled mission-evidence MCP plugin."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType


# 功能：
#   从正式插件源文件加载模块，使测试覆盖发布构建实际打包的入口。
# 输入：
#   无。
# 输出：
#   module：已执行的插件服务器模块。
def _load_server() -> ModuleType:
    path = Path(__file__).parents[1] / "official_plugins" / "mission_evidence_gate" / "server.py"
    spec = importlib.util.spec_from_file_location("mission_evidence_gate_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 功能：
#   验证初始化后的宿主心跳获得成功回复，不会因 method-not-found 被会话池隔离。
# 输入：
#   monkeypatch：将标准输入输出替换为内存协议流。
# 输出：
#   None：断言初始化与 ping 两个响应均符合 JSON-RPC 契约。
def test_server_accepts_persistent_session_ping(monkeypatch) -> None:
    server = _load_server()
    requests = [
        {
            "jsonrpc": "2.0",
            "id": "initialize",
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
        {"jsonrpc": "2.0", "id": "heartbeat", "method": "ping", "params": {}},
    ]
    stdin = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)

    assert server.serve() == 0

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses[0]["result"]["protocolVersion"] == "2025-06-18"
    assert responses[1] == {"jsonrpc": "2.0", "id": "heartbeat", "result": {}}
