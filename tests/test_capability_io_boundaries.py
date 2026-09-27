"""Offline broker authority, path ownership and transport-boundary regressions."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from clock_fixtures import isolate_monotonic

import dronedream_agent_core.capability_broker as module
import dronedream_agent_core.pinned_https as https
from dronedream_agent_core.capability_broker import (
    CapabilityBrokerError,
    CapabilityBrokerHostServices,
    CoreCapabilityBroker,
)
from dronedream_agent_core.plugin_contracts import PluginManifest
from dronedream_agent_core.plugin_process import PluginProcessError


# 功能：
#   验证进程隔离失败会写入失败回执，不被记为成功或丢失执行痕迹。
# 输入：
#   monkeypatch：替换进程隔离入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_failed_process_isolation_is_receipted(monkeypatch):
    import sys

    receipts = []
    scoped = CoreCapabilityBroker(
        allowed_executables={"fixture": Path(sys.executable)}, receipt_sink=receipts.append
    ).scope(_manifest(["process.spawn", "network.external"]))

    # 功能：
    #   模拟 Windows Job 分配失败，不实际启动缺少隔离限制的程序。
    # 输入：
    #   args：进程命令位置参数。
    #   kwargs：进程隔离配置。
    # 输出：
    #   None：不返回业务数据。
    def failed(*args, **kwargs):
        raise PluginProcessError("TEST_JOB_ASSIGNMENT_FAILED")

    monkeypatch.setattr(module, "capture_process", failed)
    with pytest.raises(CapabilityBrokerError, match="BROKER_PROCESS_FAILED"):
        scoped.spawn("fixture", [])
    assert receipts[-1].outcome == "failed"
    assert receipts[-1].issue_codes == ["BROKER_PROCESS_FAILED"]


# 功能：
#   构造有界插件权限夹具，不读取实际账户或模型供应商配置。
# 输入：
#   permissions：本次测试授予的权限列表。
# 输出：
#   manifest：包含测试能力、网络目标与消息预算的插件声明。
def _manifest(permissions):
    manifest = PluginManifest.model_validate(
        {
            "plugin_id": "test.boundaries",
            "name": "Fixture",
            "version": "1.0.0",
            "description": "Offline boundary fixture",
            "publisher": "DroneDream",
            "runtime": {"kind": "builtin-python", "entrypoint": "fixture:definition"},
            "resource_policy": {
                "maximum_message_bytes": 4096,
                "allowed_network_hosts": ["api.example.com"],
            },
            "capabilities": [
                {
                    "capability_id": "test.boundaries.invoke",
                    "kind": "tool",
                    "name": "Fixture",
                    "description": "Offline fixture",
                    "authority": "read",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                }
            ],
            "permissions": permissions,
        }
    )
    return manifest


# 功能：
#   验证独占创建失败后不会清理碰巧同名、但属于其他写入方的文件。
# 输入：
#   tmp_path：隔离输出目录。
#   monkeypatch：固定临时文件名称的测试工具。
# 输出：
#   None：不返回业务数据。
def test_write_collision_does_not_delete_existing_file(tmp_path, monkeypatch):
    scoped = CoreCapabilityBroker(write_roots={"output": tmp_path}).scope(
        _manifest(["mission.write-output", "network.external"])
    )
    existing = tmp_path / "result.broker-fixed.tmp"
    existing.write_bytes(b"preserve")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="fixed"))
    with pytest.raises(CapabilityBrokerError, match="WRITE_FAILED"):
        scoped.write_bytes("output", "result", b"new")
    assert existing.read_bytes() == b"preserve"
    assert not (tmp_path / "result").exists()


# 功能：
#   重复验证并发写入只发布某一个完整结果，不混合字节或遗留共享临时文件。
# 输入：
#   tmp_path：隔离输出目录。
#   round_number：独立测试重复序号。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("round_number", range(10))
def test_concurrent_writes_publish_one_complete_value(tmp_path, round_number):
    scopes = [
        CoreCapabilityBroker(write_roots={"output": tmp_path}).scope(
            _manifest(["mission.write-output", "network.external"])
        )
        for _ in range(4)
    ]
    values = [bytes([65 + i]) * 4000 for i in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda item: scopes[item[0] % 4].write_bytes("output", "result", item[1]),
                enumerate(values),
            )
        )
    assert (tmp_path / "result").read_bytes() in values
    assert list(tmp_path.glob("*.tmp")) == []


# 功能：
#   验证父级穿越、Windows 设备名和备用数据流等不可移植路径在读取前被拒绝。
# 输入：
#   tmp_path：隔离资产目录。
#   path：待拒绝的路径表达式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("path", ["../escape", "name:stream", "CON", "name.", "a//b", ""])
def test_broker_rejects_path_aliases_before_file_access(tmp_path, path):
    scoped = CoreCapabilityBroker(read_roots={"assets": tmp_path}).scope(
        _manifest(["asset.read", "network.external"])
    )
    with pytest.raises(CapabilityBrokerError, match="PATH_INVALID"):
        scoped.read_bytes("assets", path)


# 功能：
#   验证代理读取经过有界文件描述符检查，不退回一次性无上限读取。
# 输入：
#   tmp_path：隔离资产目录。
#   monkeypatch：禁用无界读取入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_broker_does_not_use_unbounded_read_bytes(tmp_path, monkeypatch):
    (tmp_path / "input").write_bytes(b"data")
    scoped = CoreCapabilityBroker(read_roots={"assets": tmp_path}).scope(
        _manifest(["asset.read", "network.external"])
    )

    # 功能：
    #   捕获退回无界读取的实现，不需要创建或消耗真正的大文件。
    # 输入：
    #   args：文件读取的位置参数。
    # 输出：
    #   None：不返回业务数据。
    def forbidden(*args):
        raise AssertionError("unbounded read")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert scoped.read_bytes("assets", "input") == b"data"


# 功能：
#   验证直接调用 RPC 时也不能强制转换非法字段，或忽略未经声明的额外字段。
# 输入：
#   tmp_path：隔离资产目录。
#   params：类型错误或带额外字段的请求参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "params",
    [
        {"root": True, "path": "input"},
        {"root": "assets", "path": None},
        {"root": "assets", "path": "input", "ignored_permission": True},
        [],
    ],
)
def test_rpc_does_not_coerce_or_ignore_invalid_fields(tmp_path, params):
    scoped = CoreCapabilityBroker(read_roots={"assets": tmp_path}).scope(
        _manifest(["asset.read", "network.external"])
    )
    with pytest.raises(CapabilityBrokerError, match="PARAMETERS_INVALID"):
        CapabilityBrokerHostServices(scoped)("dronedream/filesystem/read", params)


# 功能：
#   验证插件不能注入 HTTP 分帧头或非法头值，且拒绝发生在任何 DNS 查询之前。
# 输入：
#   monkeypatch：拦截域名解析的测试工具。
#   headers：待拒绝的请求头。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "headers",
    [{"Bad\rName": "x"}, {"Content-Length": "0"}, {"X-Test": "x\r\ny"}, {"X-Test": True}, []],
)
def test_network_rejects_bad_headers_before_dns(monkeypatch, headers):
    scoped = CoreCapabilityBroker().scope(_manifest(["network.external"]))

    # 功能：
    #   确保非法请求头不能使请求进入域名解析阶段。
    # 输入：
    #   args：解析器位置参数。
    #   kwargs：解析器关键字参数。
    # 输出：
    #   None：不返回业务数据。
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected DNS")

    monkeypatch.setattr(module.socket, "getaddrinfo", forbidden)
    with pytest.raises(CapabilityBrokerError, match="HEADER_DENIED"):
        scoped.request("GET", "https://api.example.com/test", headers=headers)


# 功能：
#   验证套接字使用已检查的数值 IP，而 TLS 仍以原始主机名验证证书。
# 输入：
#   monkeypatch：用记录器替换套接字和解析器的测试工具。
#   port：本次明确批准的 HTTPS 端口。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("port", [443, 8443])
def test_pinned_connection_keeps_tls_hostname_without_dns_lookup(monkeypatch, port):
    calls = []

    class Socket:
        """Record socket operations without opening an actual network connection."""

        # 功能：
        #   验证连接阶段传给套接字的剩余时间为正。
        # 输入：
        #   self：套接字替身。
        #   value：剩余秒数。
        # 输出：
        #   None：不返回业务数据。
        def settimeout(self, value):
            assert value > 0

        # 功能：
        #   记录实际连接目标，不打开网络连接。
        # 输入：
        #   self：套接字替身。
        #   target：数值地址与端口。
        # 输出：
        #   None：不返回业务数据。
        def connect(self, target):
            calls.append(("connect", target))

        # 功能：
        #   记录连接释放动作。
        # 输入：
        #   self：套接字替身。
        # 输出：
        #   None：不返回业务数据。
        def close(self):
            calls.append(("close",))

    class Context:
        """Preserve the hostname argument through the fake handshake boundary."""

        # 功能：
        #   记录 TLS 使用的服务主机名，以替身完成握手边界检查。
        # 输入：
        #   self：TLS 上下文替身。
        #   raw：已记录数值连接目标的套接字。
        #   server_hostname：TLS 验证与 SNI 使用的主机名。
        # 输出：
        #   raw：未包装的套接字替身。
        def wrap_socket(self, raw, *, server_hostname):
            calls.append(("tls", server_hostname))
            return raw

    # 功能：
    #   阻止连接时重新解析域名，避免已校验地址被第二次解析替换。
    # 输入：
    #   args：解析器位置参数。
    #   kwargs：解析器关键字参数。
    # 输出：
    #   None：不返回业务数据。
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected DNS lookup")

    connection = https.PinnedHTTPSConnection(
        f"api.example.com:{port}",
        allowed_host="api.example.com",
        addresses=("93.184.216.34",),
        timeout=1,
        allowed_port=port,
    )
    assert connection._context.check_hostname is True
    connection._context = Context()
    monkeypatch.setattr(https.socket, "socket", lambda *_: Socket())
    monkeypatch.setattr(https.socket, "getaddrinfo", forbidden)
    connection.connect()
    assert calls == [("connect", ("93.184.216.34", port)), ("tls", "api.example.com")]
    connection.close()


# 功能：
#   验证 urllib 提供的主机或端口偏离授权目标时不能创建连接。
# 输入：
#   无输入参数。
# 输出：
#   None：不返回业务数据。
def test_pinned_connection_refuses_host_or_port_switch():
    with pytest.raises(OSError, match="TARGET_DENIED"):
        https.PinnedHTTPSConnection(
            "other.example.com",
            allowed_host="api.example.com",
            addresses=("93.184.216.34",),
            timeout=1,
        )
    with pytest.raises(OSError, match="TARGET_DENIED"):
        https.PinnedHTTPSConnection(
            "api.example.com:444",
            allowed_host="api.example.com",
            addresses=("93.184.216.34",),
            timeout=1,
        )


# 功能：
#   验证响应体读取超过整体时限后被拒绝，正常 HTTP 状态不会掩盖超时。
# 输入：
#   monkeypatch：替换时钟和网络请求的测试工具。
# 输出：
#   None：不返回业务数据。
def test_broker_rejects_late_body_without_returning_data(monkeypatch):
    now = [0.0]
    scoped = CoreCapabilityBroker().scope(_manifest(["network.external"]))
    isolate_monotonic(monkeypatch, module, lambda: now[0])
    monkeypatch.setattr(
        module.socket, "getaddrinfo", lambda *_, **__: [(2, 1, 6, "", ("93.184.216.34", 443))]
    )

    class Response:
        """Advance a fake clock during read; no sleeps or network are involved."""

        status = 200
        headers = {}
        closed = False

        # 功能：
        #   将响应替身交给请求上下文。
        # 输入：
        #   self：可推进测试时钟的响应。
        # 输出：
        #   self：当前响应替身。
        def __enter__(self):
            return self

        # 功能：
        #   记录请求退出后的资源释放，不吞掉上下文异常。
        # 输入：
        #   self：当前响应替身。
        #   exception：上下文传入的异常信息。
        # 输出：
        #   None：不返回业务数据。
        def __exit__(self, *exception):
            self.closed = True

        # 功能：
        #   返回少量数据前将时钟推进到截止时间之后，模拟迟到的完整响应。
        # 输入：
        #   self：当前响应替身。
        #   limit：调用方请求读取的字节上限。
        # 输出：
        #   body：内容合法但超过使用时限的字节。
        def read(self, limit):
            now[0] = scoped._timeout + 1
            body = b"late"
            return body

    response = Response()
    monkeypatch.setattr(
        module, "build_opener", lambda *_: SimpleNamespace(open=lambda *_, **__: response)
    )
    with pytest.raises(CapabilityBrokerError, match="DEADLINE_EXCEEDED"):
        scoped.request("GET", "https://api.example.com/test")
    assert response.closed is True


# 功能：
#   验证非法凭据头前缀在解析实际凭据前被拒绝，避免原始凭据进入后续传输错误。
# 输入：
#   无输入参数。
# 输出：
#   None：不返回业务数据。
def test_invalid_credential_prefix_is_rejected_before_resolving():
    # 功能：
    #   拦截本不应该发生的凭据解析，不读取任何真实密钥。
    # 输入：
    #   args：凭据解析位置参数。
    #   kwargs：凭据解析关键字参数。
    # 输出：
    #   None：不返回业务数据。
    def forbidden(*args, **kwargs):
        raise AssertionError("credential must not be resolved")

    scoped = CoreCapabilityBroker(credential_resolver=SimpleNamespace(resolve=forbidden)).scope(
        _manifest(["network.external", "credential.reference"])
    )
    with pytest.raises(CapabilityBrokerError, match="CREDENTIAL_HEADER_DENIED"):
        scoped.request(
            "GET",
            "https://api.example.com/test",
            credential_reference="ref",
            credential_prefix="Bearer \r\nX-Other: ",
        )
