"""HTTPS transport bound to prevalidated literal addresses while preserving TLS hostname checks."""

from __future__ import annotations

import ipaddress
import socket
import ssl
import time
from http.client import HTTPSConnection
from urllib.request import HTTPSHandler


class PinnedHTTPSConnection(HTTPSConnection):
    """Connect only to the checked DNS result, not a second resolver answer or a proxy."""

    # 功能：
    #   固定已批准的主机、端口与解析地址，保留 TLS 证书及主机名校验。
    # 输入：
    #   self：当前 HTTPS 连接。
    #   host：请求使用的主机及可选端口。
    #   allowed_host：调用方批准的规范主机名。
    #   addresses：调用方已检查的数值 IP 地址元组。
    #   timeout：连接和 TLS 握手共用的秒数预算。
    #   allowed_port：批准的目标端口，默认仍为 443。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        host: str,
        *,
        allowed_host: str,
        addresses: tuple[str, ...],
        timeout: float,
        allowed_port: int = 443,
    ):
        super().__init__(host, timeout=timeout, context=ssl.create_default_context())
        if (
            self.host.lower().rstrip(".") != allowed_host
            or type(allowed_port) is not int
            or not 1 <= allowed_port <= 65535
            or self.port != allowed_port
            or not addresses
        ):
            raise OSError("BROKER_NETWORK_TARGET_DENIED")
        self._addresses = addresses

    # 功能：
    #   在共享截止时间内连接已检查的数值地址，不使用代理或重新查询 DNS。
    # 输入：
    #   self：包含目标地址与时间预算的连接。
    # 输出：
    #   None：不返回业务数据。
    def connect(self) -> None:
        if self._tunnel_host:
            raise OSError("BROKER_NETWORK_PROXY_DENIED")
        deadline = time.monotonic() + self.timeout
        last_error: OSError | None = None
        for address in self._addresses:
            ip = ipaddress.ip_address(address)
            raw = socket.socket(
                socket.AF_INET6 if ip.version == 6 else socket.AF_INET, socket.SOCK_STREAM
            )
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("BROKER_NETWORK_CONNECT_TIMEOUT")
                raw.settimeout(remaining)
                # 数值 sockaddr 固定实际连接地址，TLS 仍验证最初的服务主机名。
                raw.connect((str(ip), self.port))
                raw.settimeout(max(0.001, deadline - time.monotonic()))
                self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
                if time.monotonic() >= deadline:
                    self.sock.close()
                    self.sock = None
                    raise TimeoutError("BROKER_NETWORK_CONNECT_TIMEOUT")
                return
            except OSError as error:
                last_error = error
                raw.close()
        raise last_error or OSError("BROKER_NETWORK_DNS_FAILED")


class PinnedHTTPSHandler(HTTPSHandler):
    """urllib integration retaining normal response/error handling and hostname identity."""

    # 功能：
    #   为单个请求保存批准的网络目标，不共享可被其他请求替换的解析缓存。
    # 输入：
    #   self：当前 urllib HTTPS 处理器。
    #   host：已检查的规范主机名。
    #   addresses：该主机已检查的数值地址。
    #   port：批准的目标端口，默认 443。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, host: str, addresses: tuple[str, ...], *, port: int = 443):
        super().__init__()
        self._host = host
        self._addresses = addresses
        self._port = port

    # 功能：
    #   将 urllib 请求交给固定目标的连接工厂，沿用正常响应与错误处理。
    # 输入：
    #   self：当前请求的 HTTPS 处理器。
    #   request：准备发送的请求。
    # 输出：
    #   response：固定目标连接返回的 HTTP 响应。
    def https_open(self, request):
        response = self.do_open(self._connection, request)
        return response

    # 功能：
    #   创建独立 TLS 连接，再次核对 urllib 交来的主机和端口未偏离批准目标。
    # 输入：
    #   self：包含批准目标的处理器。
    #   host：urllib 提供的主机及端口。
    #   timeout：当前请求的连接预算。
    # 输出：
    #   connection：固定数值地址的 HTTPS 连接。
    def _connection(self, host, *, timeout):
        connection = PinnedHTTPSConnection(
            host,
            allowed_host=self._host,
            addresses=self._addresses,
            timeout=timeout,
            allowed_port=self._port,
        )
        return connection
