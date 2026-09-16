from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from dronedream_agent_core.plugin_contracts import PluginResourcePolicy
from dronedream_agent_core.plugin_process import (
    McpSessionFailure,
    McpSessionPool,
    McpStdioClient,
    _isolated_command,
    _safe_environment,
)


class _HeartbeatClient:
    instances: list[_HeartbeatClient] = []

    # 功能：
    #   创建默认无响应的惰性客户端，记录实例供会话淘汰测试核对。
    # 输入：
    #   _kwargs：会话池传入但本替身不执行的启动参数。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, **_kwargs) -> None:
        self.closed = False
        self.healthy = False
        self.instances.append(self)

    # 功能：
    #   返回测试设置的响应状态，不发起网络或进程请求。
    # 输入：
    #   无。
    # 输出：
    #   healthy：当前模拟健康状态。
    def ping(self) -> bool:
        healthy = self.healthy
        return healthy

    # 功能：
    #   记录客户端已被回收，不操作真实进程。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        self.closed = True


# 功能：
#   验证外网权限不会移除子进程代理限制，获准网络访问仍应经过宿主代理。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_external_network_permission_still_forces_host_broker_proxy() -> None:
    environment = _safe_environment(
        permissions=["network.external"],
        resource_policy=PluginResourcePolicy(allowed_network_hosts=["api.example.test"]),
    )
    assert environment["DRONEDREAM_PLUGIN_NETWORK_BROKER_ONLY"] == "1"
    assert environment["HTTPS_PROXY"] == "http://127.0.0.1:9"
    assert environment["NO_PROXY"] == ""


# 功能：
#   核对 Windows 启动参数确实包含隔离器、根目录和资源配额，不运行夹具可执行文件。
# 输入：
#   tmp_path：惰性插件和隔离器占位文件所在目录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.skipif(os.name != "nt", reason="Windows AppContainer wrapper")
def test_windows_external_process_is_wrapped_by_appcontainer_isolator(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin.exe"
    isolator = tmp_path / "dronedream-plugin-isolator.exe"
    plugin.write_bytes(b"plugin")
    isolator.write_bytes(b"isolator")
    command = _isolated_command(
        plugin_root=tmp_path,
        command=["plugin.exe", "--serve"],
        require_os_isolation=True,
        isolator_path=isolator,
    )
    assert command[0] == str(isolator.resolve())
    assert command[1:3] == ["--profile", command[2]]
    assert command[3:5] == ["--root", str(tmp_path.resolve())]
    assert command[5:11] == [
        "--memory-mb",
        "256",
        "--cpu-seconds",
        "120",
        "--process-limit",
        "4",
    ]
    assert command[11] == "--"
    assert command[-2:] == [str(plugin.resolve()), "--serve"]


# 功能：
#   验证存活探测淘汰并报告精确失败会话，下一次获取会创建新实例。
# 输入：
#   tmp_path：会话身份中使用的独立根目录。
# 输出：
#   None：不返回业务数据。
def test_persistent_session_heartbeat_evicts_and_reports_unhealthy_plugin(tmp_path: Path):
    _HeartbeatClient.instances.clear()
    unhealthy = threading.Event()
    reports: list[tuple[str, str]] = []

    # 功能：
    #   核对报告绑定原安装包并通知测试线程，不执行产品遥测写入。
    # 输入：
    #   failure：池产生的精确失败报告。
    # 输出：
    #   None：不返回业务数据。
    def report(failure: McpSessionFailure) -> None:
        assert failure.package_sha256 == "a" * 64
        reports.append((failure.plugin_id, failure.issue_code))
        unhealthy.set()

    pool = McpSessionPool(heartbeat_interval_seconds=0.01, on_unhealthy=report)
    arguments = {
        "plugin_id": "test.heartbeat",
        "package_sha256": "a" * 64,
        "plugin_root": tmp_path,
        "command": ["plugin.exe"],
        "protocol_version": "dronedream.plugin.v1",
        "startup_timeout_seconds": 1.0,
        "call_timeout_seconds": 1.0,
        "configuration": {},
        "permissions": [],
        "resource_policy": PluginResourcePolicy(),
        "client_factory": _HeartbeatClient,
    }
    try:
        first = pool.get(**arguments)  # type: ignore[arg-type]
        assert pool.get(**arguments) is first  # type: ignore[arg-type]
        assert unhealthy.wait(0.5)
        assert first.closed is True
        assert reports == [("test.heartbeat", "PLUGIN_HEARTBEAT_INVALID")]
        second = pool.get(**arguments)  # type: ignore[arg-type]
        assert second is not first
    finally:
        pool.close()
    assert all(client.closed for client in _HeartbeatClient.instances)


# 功能：
#   验证仅转发 DroneDream 命名空间的宿主方法，其他方法得到权限拒绝回复。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_reverse_rpc_dispatches_only_dronedream_host_methods() -> None:
    client = object.__new__(McpStdioClient)
    sent: list[dict[str, object]] = []
    client._host_services = lambda method, params: {  # type: ignore[attr-defined]
        "method": method,
        "value": params["value"],
    }
    client._send = sent.append  # type: ignore[method-assign]

    client._serve_host_request(  # type: ignore[attr-defined]
        {
            "jsonrpc": "2.0",
            "id": "host-1",
            "method": "dronedream/filesystem/read",
            "params": {"value": 7},
        }
    )
    client._serve_host_request(  # type: ignore[attr-defined]
        {
            "jsonrpc": "2.0",
            "id": "host-2",
            "method": "arbitrary/unsafe",
            "params": {},
        }
    )

    assert sent[0]["result"] == {
        "method": "dronedream/filesystem/read",
        "value": 7,
    }
    assert sent[1]["error"] == {
        "code": -32601,
        "message": "HOST_METHOD_NOT_ALLOWED",
    }
