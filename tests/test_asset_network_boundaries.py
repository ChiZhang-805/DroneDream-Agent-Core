"""Remote asset transport and temporary ownership checks with no external connections."""

import io
import os
import urllib.request
from types import SimpleNamespace

import pytest

from dronedream_agent_app import asset_remote_sources as remote


# 功能：
#   为网络边界测试提供固定公网解析结果，不访问实际 DNS。
# 输入：
#   monkeypatch：替换系统解析器的测试工具。
# 输出：
#   None：不返回业务数据。
@pytest.fixture
def public_dns(monkeypatch):
    monkeypatch.setattr(
        remote.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )


class _Body(io.BytesIO):
    headers = {}

    # 功能：
    #   返回受控下载来源，供文件名提取和跳转后校验使用。
    # 输入：
    #   self：内存响应体。
    # 输出：
    #   url：测试资源地址。
    def geturl(self):
        url = "https://example.invalid/map.sdf"
        return url


# 功能：
#   验证资产下载禁用继承代理并使用逐跳固定目标的 HTTPS 处理器。
# 输入：
#   public_dns：受控公网解析夹具。
#   monkeypatch：替换请求处理器创建的测试工具。
#   tmp_path：本次下载独占的测试暂存目录。
# 输出：
#   None：不返回业务数据。
def test_download_uses_pinned_transport_without_proxy(public_dns, monkeypatch, tmp_path):
    handlers = []
    body = _Body(b"asset")

    # 功能：
    #   记录下载器选择的处理器，返回内存响应而不联网。
    # 输入：
    #   configured：准备使用的 urllib 处理器。
    # 输出：
    #   opener：只返回测试响应的请求入口。
    def build(*configured):
        handlers.extend(configured)
        opener = SimpleNamespace(open=lambda *args, **kwargs: body)
        return opener

    monkeypatch.setattr(remote.urllib.request, "build_opener", build)
    with remote.RemoteAssetSourceService(tmp_path).acquire(
        source_type="direct_url", location=body.geturl()
    ):
        assert any(
            isinstance(handler, urllib.request.ProxyHandler) and handler.proxies == {}
            for handler in handlers
        )
        assert any(type(handler).__name__ == "_AssetHTTPSHandler" for handler in handlers)


# 功能：
#   验证导入方替换暂存路径后，下载器清理时保留不属于自己的新文件。
# 输入：
#   public_dns：受控公网解析夹具。
#   tmp_path：隔离测试目录。
#   monkeypatch：替换网络入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_download_preserves_replaced_temporary(public_dns, tmp_path, monkeypatch):
    body = _Body(b"original")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"another writer")
    monkeypatch.setattr(
        remote.urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=lambda *a, **k: body),
    )
    with remote.RemoteAssetSourceService(tmp_path).acquire(
        source_type="direct_url", location=body.geturl()
    ) as (path, _name):
        os.replace(replacement, path)
    assert path.read_bytes() == b"another writer"


# 功能：
#   验证每一跳 HTTPS 请求将本次公网解析及端口传给固定连接处理器。
# 输入：
#   public_dns：受控公网解析夹具。
#   monkeypatch：替换 TLS 处理器的测试工具。
# 输出：
#   None：不返回业务数据。
def test_https_hop_pins_checked_addresses_and_port(public_dns, monkeypatch):
    observed = []
    response = object()

    # 功能：
    #   捕获固定连接的目标，不实际创建套接字。
    # 输入：
    #   host：允许的服务主机名。
    #   addresses：已验证的数值地址。
    #   port：允许的端口。
    # 输出：
    #   handler：返回固定响应的处理器替身。
    def pinned(host, addresses, *, port):
        observed.append((host, addresses, port))
        handler = SimpleNamespace(https_open=lambda request: response)
        return handler

    monkeypatch.setattr(remote, "PinnedHTTPSHandler", pinned, raising=False)
    handler = remote._AssetHTTPSHandler()
    assert (
        handler.https_open(urllib.request.Request("https://example.invalid:8443/map")) is response
    )
    assert observed == [("example.invalid", ("93.184.216.34",), 8443)]


# 功能：
#   验证 Git 获取固定已验证地址且不继承环境代理，退出时由有界进程捕获器管理。
# 输入：
#   public_dns：受控公网解析夹具。
#   tmp_path：隔离测试目录。
#   monkeypatch：替换 Git 进程和环境的测试工具。
# 输出：
#   None：不返回业务数据。
def test_git_pins_dns_and_drops_ambient_proxy(public_dns, tmp_path, monkeypatch):
    workspace = tmp_path / "stage"
    workspace.mkdir()
    monkeypatch.setattr(remote.tempfile, "mkdtemp", lambda **kwargs: str(workspace))
    monkeypatch.setattr(remote.shutil, "which", lambda name: "git.exe")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    monkeypatch.setenv("SSH_ASKPASS", "unexpected-interactive-program")
    calls = []

    # 功能：
    #   创建无害工作树并记录 Git 参数和隔离环境。
    # 输入：
    #   command：Git 命令行。
    #   kwargs：进程配置。
    # 输出：
    #   completed：模拟成功的退出结果。
    def clone(command, **kwargs):
        calls.append((command, kwargs))
        checkout = workspace / "checkout"
        checkout.mkdir()
        (checkout / "asset.sdf").write_bytes(b"asset")
        completed = SimpleNamespace(returncode=0)
        return completed

    monkeypatch.setattr(remote, "capture_process", clone)
    with remote.RemoteAssetSourceService().acquire(
        source_type="git", location="https://example.invalid/repo.git"
    ):
        command, kwargs = calls[0]
        assert "http.curloptResolve=example.invalid:443:93.184.216.34" in command
        assert "http.proxy=" in command
        assert "http.sslVerify=true" in command
        assert "HTTPS_PROXY" not in kwargs["environment"]
        assert kwargs["environment"]["SSH_ASKPASS"] == ""
        assert kwargs["environment"]["GIT_ASKPASS"] == ""
        assert kwargs["environment"]["GIT_CEILING_DIRECTORIES"] == str(workspace.parent)
        assert kwargs["resource_policy"].process_limit >= 4


# 功能：
#   验证请求构造失败不会留下尚未进入下载清理范围的暂存文件。
# 输入：
#   public_dns：受控公网解析夹具。
#   tmp_path：隔离暂存目录。
#   monkeypatch：注入请求初始化失败的测试工具。
# 输出：
#   None：不返回业务数据。
def test_request_creation_failure_leaves_no_staging_file(public_dns, tmp_path, monkeypatch):
    # 功能：
    #   模拟 urllib 在创建请求时发现非法请求内容。
    # 输入：
    #   args：请求构造的位置参数。
    #   kwargs：请求头等关键字参数。
    # 输出：
    #   None：不返回业务数据。
    def rejected_request(*args, **kwargs):
        raise ValueError("invalid request")

    monkeypatch.setattr(remote.urllib.request, "Request", rejected_request)
    with (
        pytest.raises(ValueError, match="invalid request"),
        remote.RemoteAssetSourceService(tmp_path).acquire(
            source_type="direct_url", location="https://example.invalid/map.sdf"
        ),
    ):
        pytest.fail("invalid request reached importer")
    assert list(tmp_path.iterdir()) == []


# 功能：
#   验证数值公网地址无需再次解析，避免 IPv6 主机被拼成有歧义的 DNS 固定配置。
# 输入：
#   address：公网 IPv4 或 IPv6 数值地址。
#   monkeypatch：阻止数值地址意外调用 DNS 的测试工具。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("address", ["93.184.216.34", "2606:4700:4700::1111"])
def test_literal_address_uses_no_dns(address, monkeypatch):
    # 功能：
    #   使不必要的域名解析立即失败，而非访问实际网络。
    # 输入：
    #   args：域名解析的位置参数。
    #   kwargs：域名解析的关键字参数。
    # 输出：
    #   None：不返回业务数据。
    def forbidden(*args, **kwargs):
        raise AssertionError("literal target must not be re-resolved")

    monkeypatch.setattr(remote.socket, "getaddrinfo", forbidden)
    hostname = f"[{address}]" if ":" in address else address
    parsed, addresses = remote._validated_https_target(f"https://{hostname}:8443/map.sdf")
    assert parsed.port == 8443
    assert addresses == (address,)


# 功能：
#   验证下载方式不能静默忽略仅适用于 Git 的选择参数。
# 输入：
#   options：不适用于普通下载的仓库选择参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("options", [{"git_ref": "main"}, {"subpath": "map"}])
def test_direct_source_rejects_git_selection(options):
    with (
        pytest.raises(remote.AssetImportError, match="GIT_OPTIONS_REQUIRE_GIT"),
        remote.RemoteAssetSourceService().acquire(
            source_type="direct_url", location="https://example.invalid/map", **options
        ),
    ):
        pytest.fail("Git-only options were ignored")
