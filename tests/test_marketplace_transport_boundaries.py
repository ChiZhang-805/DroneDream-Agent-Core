"""Exercise remote-catalog failure modes without making network connections."""

import io
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from dronedream_agent_app import plugin_marketplace as marketplace


class Response(io.BytesIO):
    """In-memory HTTP body that records the largest permitted read."""

    # 功能：
    #   创建内存响应并保留长度声明，记录下载器每次请求的读取大小。
    # 输入：
    #   self：响应替身。
    #   payload：实际响应字节。
    #   length：可选的 Content-Length 字符串。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, payload=b"{}", length=None):
        super().__init__(payload)
        self.headers = {} if length is None else {"Content-Length": length}
        self.requests = []

    # 功能：
    #   提供响应对应的固定 HTTPS 来源。
    # 输入：
    #   self：响应替身。
    # 输出：
    #   url：测试目录地址。
    def geturl(self):
        url = "https://example.invalid/catalog"
        return url

    # 功能：
    #   记录读取预算，再使用真实内存流语义返回字节。
    # 输入：
    #   self：响应替身。
    #   size：调用方本次请求的最大字节数。
    # 输出：
    #   payload：实际读取的响应字节。
    def read(self, size=-1):
        self.requests.append(size)
        payload = super().read(size)
        return payload


# 功能：
#   仅替换网络打开入口，不绕开正在验证的 URL、跳转与读取策略。
# 输入：
#   monkeypatch：测试替换工具。
#   response：本次调用应取得的响应替身。
# 输出：
#   None：不返回业务数据。
def install_response(monkeypatch, response):
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_handlers: SimpleNamespace(open=lambda *_args, **_kwargs: response),
    )


# 功能：
#   验证不安全或歧义地址在打开网络入口之前被拒绝。
# 输入：
#   monkeypatch：拦截请求入口的测试工具。
#   url：待拒绝的来源地址。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "url",
    [
        "http://example.invalid",
        "https:///missing-host",
        "https://name:secret@example.invalid",
        "https://example.invalid:0",
        "https://example.invalid/#fragment",
        "https://example.invalid/\n",
        "https://example.invalid\\other/path",
        "https://example.invalid/a\u2003b",
    ],
)
def test_bad_transport_is_rejected_before_any_connection(monkeypatch, url):
    # 功能：
    #   使非法 URL 意外进入网络阶段时立即失败。
    # 输入：
    #   args：请求入口的位置参数。
    #   kwargs：请求入口的配置参数。
    # 输出：
    #   None：不返回业务数据。
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid URL reached the opener")

    monkeypatch.setattr(urllib.request, "build_opener", unexpected)
    with pytest.raises(marketplace.PluginMarketplaceError, match="HTTPS_REQUIRED"):
        marketplace.PluginMarketplaceService._download(url, limit=100, timeout_seconds=1)


# 功能：
#   验证 HTTPS 降级到 HTTP 的跳转被拒绝，并关闭尚未消费的原响应。
# 输入：
#   无输入参数。
# 输出：
#   None：不返回业务数据。
def test_downgrade_redirect_is_rejected_and_its_body_closed():
    body = io.BytesIO(b"redirect")
    request = urllib.request.Request("https://example.invalid/catalog")
    with pytest.raises(marketplace.PluginMarketplaceError, match="REDIRECT_DENIED"):
        marketplace._HttpsRedirect().redirect_request(
            request, body, 302, "", {}, "http://unsafe.invalid"
        )
    assert body.closed


# 功能：
#   验证安全的 HTTPS CDN 跳转仍可生成目标请求。
# 输入：
#   无输入参数。
# 输出：
#   None：不返回业务数据。
def test_https_cdn_redirect_remains_supported():
    request = urllib.request.Request("https://example.invalid/catalog")
    target = marketplace._HttpsRedirect().redirect_request(
        request, None, 302, "", {}, "https://cdn.example.invalid/catalog"
    )
    assert target.full_url == "https://cdn.example.invalid/catalog"


# 功能：
#   验证非法长度声明在读取响应体前被拒绝，且响应会正常关闭。
# 输入：
#   monkeypatch：提供响应替身的测试工具。
#   length：非法长度声明。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("length", ["-1", "bad", "+2", "٢"])
def test_bad_declared_length_is_rejected_and_response_closed(monkeypatch, length):
    response = Response(length=length)
    install_response(monkeypatch, response)
    with pytest.raises(marketplace.PluginMarketplaceError, match="LENGTH_INVALID"):
        marketplace.PluginMarketplaceService._download(
            response.geturl(), limit=100, timeout_seconds=1
        )
    assert response.closed
    assert response.requests == []


# 功能：
#   验证未声明长度的响应仍受实际字节预算限制，并只多读一个探测字节。
# 输入：
#   monkeypatch：替换响应入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_missing_length_cannot_bypass_actual_byte_limit(monkeypatch):
    response = Response(payload=b"a" * 1000)
    install_response(monkeypatch, response)
    with pytest.raises(marketplace.PluginMarketplaceError, match="RESPONSE_TOO_LARGE"):
        marketplace.PluginMarketplaceService._download(
            response.geturl(), limit=10, timeout_seconds=1
        )
    assert response.requests == [11]
    assert response.closed


# 功能：
#   验证实际字节少于声明长度时不能作为完整目录交付。
# 输入：
#   monkeypatch：注入不完整响应的测试工具。
# 输出：
#   None：不返回业务数据。
def test_truncated_declared_body_is_not_accepted(monkeypatch):
    response = Response(length="3")
    install_response(monkeypatch, response)
    with pytest.raises(marketplace.PluginMarketplaceError, match="LENGTH_MISMATCH"):
        marketplace.PluginMarketplaceService._download(
            response.geturl(), limit=10, timeout_seconds=1
        )


# 功能：
#   验证最后一块数据读取后超过总预算时仍被拒绝，不因已读到数据而忽略超时。
# 输入：
#   monkeypatch：替换响应和单调时钟的测试工具。
# 输出：
#   None：不返回业务数据。
def test_late_final_chunk_cannot_return_a_successful_download(monkeypatch):
    response = Response()
    install_response(monkeypatch, response)
    times = iter([0.0, 0.1, 1.1])
    monkeypatch.setattr(marketplace.time, "monotonic", lambda: next(times))
    with pytest.raises(marketplace.PluginMarketplaceError, match="TIMEOUT"):
        marketplace.PluginMarketplaceService._download(
            response.geturl(), limit=100, timeout_seconds=1
        )
    assert response.closed


# 功能：
#   验证布尔、非正数、非有限数及过大时限在连接前被拒绝。
# 输入：
#   monkeypatch：阻止网络入口执行的测试工具。
#   budget：非法时间预算。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("budget", [True, 0, -1, float("inf"), float("nan"), 10**300])
def test_bad_timeout_is_rejected_before_connection(monkeypatch, budget):
    monkeypatch.setattr(
        urllib.request, "build_opener", lambda *_: pytest.fail("Unexpected connection")
    )
    with pytest.raises(marketplace.PluginMarketplaceError, match="TIMEOUT_INVALID"):
        marketplace.PluginMarketplaceService._download(
            "https://example.invalid", limit=100, timeout_seconds=budget
        )


# 功能：
#   验证 HTTP 失败关闭响应体，对外错误不泄露可能含敏感查询参数的 URL。
# 输入：
#   monkeypatch：注入 HTTP 错误的测试工具。
# 输出：
#   None：不返回业务数据。
def test_http_error_response_is_closed_without_exposing_the_url(monkeypatch):
    body = io.BytesIO(b"error")
    error = urllib.error.HTTPError(
        "https://example.invalid/?secret=private", 403, "private", {}, body
    )

    # 功能：
    #   在打开阶段抛出含测试敏感标记的 HTTP 错误，不访问实际地址。
    # 输入：
    #   args：网络打开位置参数。
    #   kwargs：网络打开配置参数。
    # 输出：
    #   None：不返回业务数据。
    def fail_open(*args, **kwargs):
        raise error

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: SimpleNamespace(open=fail_open))
    with pytest.raises(marketplace.PluginMarketplaceError, match="FETCH_FAILED") as caught:
        marketplace.PluginMarketplaceService._download(
            "https://example.invalid", limit=100, timeout_seconds=1
        )
    assert "private" not in str(caught.value)
    assert body.closed
