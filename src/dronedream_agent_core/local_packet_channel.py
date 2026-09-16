"""Bounded run-scoped authenticated datagrams, shared by telemetry and target context.

This transport owns no sensor freshness or motion authority. Domain validators
and consumers retain original source clocks and enforce their own deadlines.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import os
import secrets
import socket
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from dronedream_plugin_sdk.protocol import copy_json, decode_json, encode_json

from .plugin_files import check_plain_plugin_path, read_plugin_file

MAX_PACKET_BYTES = 60_000
TRANSPORT = "abstract-unix-datagram" if sys.platform.startswith("linux") else "loopback-udp"
FAMILY = socket.AF_UNIX if TRANSPORT == "abstract-unix-datagram" else socket.AF_INET


@dataclass(frozen=True)
class PacketContract:
    """Bind transport framing to one payload domain and its semantic validator."""
    protocol: str
    prefix: str
    payload_schema: str
    validate: Callable[[dict], None] | None = None

    # 功能：
    #   在创建套接字前验证协议身份和地址前缀，避免非法配置进入运行通信链路。
    # 输入：
    #   self：数据域、协议和可选验证函数。
    # 输出：
    #   None：不返回业务数据。
    def __post_init__(self):
        if (any(type(value) is not str or not 1 <= len(value) <= 64
                or any(not (char.isascii() and (char.isalnum() or char in ".-_")) for char in value)
                for value in (self.protocol, self.prefix, self.payload_schema))
                or self.validate is not None and not callable(self.validate)):
            raise ValueError("LOCAL_PACKET_CONTRACT_INVALID")


# 功能：
#   将已验证描述符映射成本机抽象 Unix 地址或固定回环 UDP 地址，不连接外部网络。
# 输入：
#   binding：当前运行的已校验绑定信息。
# 输出：
#   address：当前操作系统支持的本机套接字地址。
def _address(binding):
    address = ("\0" + binding["address"] if TRANSPORT == "abstract-unix-datagram"
               else ("127.0.0.1", binding["address"]))
    return address


# 功能：
#   读取最多 2 KiB 的普通端点文件，严格验证协议域、地址和本次运行的随机认证材料。
# 输入：
#   path：本次运行的端点描述符路径。
#   contract：调用方预期的通信域。
# 输出：
#   value：已验证的独立绑定字典。
def read_descriptor(path: Path, contract: PacketContract) -> dict:
    if not isinstance(contract, PacketContract):
        raise ValueError("LOCAL_PACKET_CONTRACT_INVALID")
    raw = read_plugin_file(path, limit=2048)
    value = decode_json(raw, limit=2048)
    if (not isinstance(value, dict)
            or set(value) != {"protocol", "transport", "address", "secret"}
            or value["protocol"] != contract.protocol or value["transport"] != TRANSPORT
            or type(value["secret"]) is not str or len(value["secret"]) != 64
            or any(char not in "0123456789abcdef" for char in value["secret"])):
        raise ValueError("LOCAL_PACKET_DESCRIPTOR_INVALID")
    address = value["address"]
    if TRANSPORT == "abstract-unix-datagram":
        if (not isinstance(address, str) or not address.startswith(contract.prefix)
                or len(address) != len(contract.prefix) + 32
                or any(c not in "0123456789abcdef" for c in address[len(contract.prefix):])):
            raise ValueError("LOCAL_PACKET_ADDRESS_INVALID")
    elif type(address) is not int or not 1024 <= address <= 65535:
        raise ValueError("LOCAL_PACKET_ADDRESS_INVALID")
    return value


# 功能：
#   验证通信帧的业务域和精确整数时间；来源新鲜度与动作许可仍由业务消费者负责。
# 输入：
#   payload：准备接入或发送的状态对象。
#   contract：业务域验证契约。
# 输出：
#   None：不返回业务数据。
def _validate_payload(payload, contract):
    if (not isinstance(payload, dict)
            or payload.get("schema_version") != contract.payload_schema
            or type(payload.get("updated_at_unix_ms")) is not int
            or not 0 <= payload["updated_at_unix_ms"] <= 2**63 - 1):
        raise ValueError("LOCAL_PACKET_PAYLOAD_INVALID")
    if contract.validate is not None:
        contract.validate(payload)


# 功能：
#   以私有独占暂存文件完整写入并无覆盖发布端点，清理只操作本次仍持有的文件身份。
# 输入：
#   path：尚不存在的端点文件。
#   value：本次运行的地址和随机认证绑定。
# 输出：
#   identity：成功发布文件的系统身份，用于关闭时核对归属。
def _publish_descriptor(path: Path, value: dict):
    raw = encode_json(value, limit=2048).encode("utf-8")
    check_plain_plugin_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = identity = None
    try:
        # NamedTemporaryFile 使用独占创建；POSIX 权限为 0600，Windows 继承运行目录 ACL。
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as stream:
            temporary = Path(stream.name)
            identity = os.fstat(stream.fileno())
            if stream.write(raw) != len(raw):
                raise OSError("LOCAL_PACKET_DESCRIPTOR_INCOMPLETE_WRITE")
            stream.flush()
            os.fsync(stream.fileno())
        check_plain_plugin_path(temporary)
        check_plain_plugin_path(path)
        if not os.path.samestat(identity, temporary.stat()):
            raise ValueError("LOCAL_PACKET_DESCRIPTOR_STAGING_REPLACED")
        os.link(temporary, path)
        return identity
    finally:
        if temporary is not None and identity is not None:
            with suppress(OSError, ValueError):
                check_plain_plugin_path(temporary)
                if os.path.samestat(identity, temporary.stat()):
                    temporary.unlink()


class LatestPacketReceiver:
    """Own a fresh endpoint and reject tampering, replay and regressed stamps."""

    # 功能：
    #   创建本次运行独占端点，实际探测最大报文容量后发布完整描述符，失败时关闭套接字。
    # 输入：
    #   self：当前接收器。
    #   descriptor：尚不存在的运行端点文件。
    #   contract：业务域契约。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, descriptor: Path, *, contract: PacketContract):
        if not isinstance(contract, PacketContract):
            raise ValueError("LOCAL_PACKET_CONTRACT_INVALID")
        descriptor = descriptor.absolute()
        check_plain_plugin_path(descriptor)
        if descriptor.exists():
            raise FileExistsError("LOCAL_PACKET_DESCRIPTOR_EXISTS")
        self._contract = contract
        self.path = descriptor
        self._socket = socket.socket(FAMILY, socket.SOCK_DGRAM)
        self._sequence, self._stamp = 0, -1
        self.accepted = self.rejected = 0
        self._closed = False
        try:
            address = (contract.prefix + secrets.token_hex(16)
                       if TRANSPORT == "abstract-unix-datagram" else 0)
            self._socket.bind(_address({"address": address}))
            if FAMILY == socket.AF_INET:
                address = self._socket.getsockname()[1]
            self._value = {"protocol": contract.protocol, "transport": TRANSPORT,
                           "address": address, "secret": secrets.token_hex(32)}
            self._secret = bytes.fromhex(self._value["secret"])
            self._socket.settimeout(.5)
            with socket.socket(FAMILY, socket.SOCK_DGRAM) as sender:
                sender.settimeout(.5)
                probe = secrets.token_bytes(MAX_PACKET_BYTES)
                sender.sendto(probe, _address(self._value))
                if self._socket.recvfrom(MAX_PACKET_BYTES + 1)[0] != probe:
                    raise RuntimeError("LOCAL_PACKET_CAPACITY_PROBE_FAILED")
            self._socket.setblocking(False)
            self._descriptor_identity = _publish_descriptor(self.path, self._value)
        except BaseException:
            self._socket.close()
            raise

    # 功能：
    #   处理至多 64 个数据报，只返回最新有效状态；坏帧不推进序号和时间，不回放历史。
    # 输入：
    #   self：当前运行唯一接收器，由单个消费者调用。
    # 输出：
    #   latest：本轮最新合法状态，无新状态时为 None。
    def read_latest(self) -> dict | None:
        if self._closed:
            return None
        latest = None
        # Drain is bounded. Only the latest complete authenticated packet is
        # encoded; intermediate telemetry is not fabricated as history.
        for _ in range(64):
            try:
                packet, peer = self._socket.recvfrom(MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                break
            except OSError as error:
                # Windows 丢弃被截断数据报并报 WSAEMSGSIZE，不能因此中断后续原生状态采样。
                if error.errno == errno.EMSGSIZE or getattr(error, "winerror", None) == 10040:
                    self.rejected += 1
                    continue
                raise
            try:
                if (not 33 < len(packet) <= MAX_PACKET_BYTES
                        or FAMILY == socket.AF_INET and peer[0] != "127.0.0.1"):
                    raise ValueError("LOCAL_PACKET_PACKET_SIZE")
                signature, body = packet[:32], packet[32:]
                if not hmac.compare_digest(signature, hmac.digest(self._secret, body, "sha256")):
                    raise ValueError("LOCAL_PACKET_AUTHENTICATION_FAILED")
                envelope = decode_json(body, limit=MAX_PACKET_BYTES - 32)
                if (not isinstance(envelope, dict)
                        or set(envelope) != {"sequence", "payload"}
                        or type(envelope["sequence"]) is not int
                        or not self._sequence < envelope["sequence"] <= 2**63 - 1):
                    raise ValueError("LOCAL_PACKET_REPLAY")
                payload = envelope["payload"]
                _validate_payload(payload, self._contract)
                if payload["updated_at_unix_ms"] < self._stamp:
                    raise ValueError("LOCAL_PACKET_CLOCK_REGRESSED")
                self._sequence, self._stamp = envelope["sequence"], payload["updated_at_unix_ms"]
                latest = payload
                self.accepted += 1
            except (ValueError, TypeError, KeyError, RecursionError):
                # Even authenticated producers can emit malformed nested JSON.
                # Reject one bad frame without killing the native sampler.
                self.rejected += 1
        return latest

    # 功能：
    #   关闭接收套接字，只删除文件身份和完整绑定仍属于本次运行的描述符。
    # 输入：
    #   self：当前接收器。
    # 输出：
    #   None：不返回业务数据。
    def close(self):
        if not self._closed:
            self._closed = True
            self._socket.close()
            try:
                check_plain_plugin_path(self.path)
                before = self.path.stat()
                if (os.path.samestat(self._descriptor_identity, before)
                        and read_descriptor(self.path, self._contract) == self._value):
                    check_plain_plugin_path(self.path)
                    current = self.path.stat()
                    if (os.path.samestat(before, current) and before.st_size == current.st_size
                            and before.st_mtime_ns == current.st_mtime_ns):
                        self.path.unlink()
            except (OSError, ValueError):
                pass  # Do not delete an endpoint replaced by another owner.


class LatestPacketPublisher:
    """Publish once per sample without queues, replay or implicit run rebinding."""

    # 功能：
    #   初始化非阻塞发布套接字，允许接收端稍后创建，但首次绑定后不跨运行重新连接。
    # 输入：
    #   self：当前发布器。
    #   descriptor：预期的运行端点文件。
    #   contract：业务域契约。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, descriptor: Path, *, contract: PacketContract):
        if not isinstance(contract, PacketContract):
            raise ValueError("LOCAL_PACKET_CONTRACT_INVALID")
        self._contract = contract
        self.path = descriptor.absolute()
        check_plain_plugin_path(self.path)
        self._socket = socket.socket(FAMILY, socket.SOCK_DGRAM)
        try:
            self._socket.setblocking(False)
        except BaseException:
            self._socket.close()
            raise
        self._binding = None
        self._sequence = 0
        self._closed = False
        self.sent = self.dropped = 0

    # 功能：
    #   发送单个有界认证状态，不排队、不重试、不重写采集时间；发送成功不代表消费或控制成功。
    # 输入：
    #   self：由单个生产者使用的本次运行发布器。
    #   payload：当前状态对象。
    # 输出：
    #   sent：完整数据报交给本机传输为 True，端点未就绪或拥塞丢弃时为 False。
    def send(self, payload: dict) -> bool:
        if self._closed:
            raise RuntimeError("LOCAL_PACKET_CLOSED")
        if type(payload) is not dict:
            raise ValueError("LOCAL_PACKET_PAYLOAD_INVALID")
        if self._binding is None:
            try:
                self._binding = read_descriptor(self.path, self._contract)
            except FileNotFoundError:
                self.dropped += 1
                return False
        sequence = self._sequence + 1
        if sequence > 2**63 - 1:
            raise ValueError("LOCAL_PACKET_SEQUENCE_EXHAUSTED")
        try:
            frozen = copy_json(payload, limit=MAX_PACKET_BYTES - 32)
            _validate_payload(frozen, self._contract)
            body = encode_json({"sequence": sequence, "payload": frozen},
                               limit=MAX_PACKET_BYTES - 32).encode("utf-8")
        except ValueError as error:
            raise ValueError("LOCAL_PACKET_PAYLOAD_OR_PACKET_SIZE_INVALID") from error
        packet = hmac.digest(bytes.fromhex(self._binding["secret"]), body, hashlib.sha256) + body
        if len(packet) > MAX_PACKET_BYTES:
            raise ValueError("LOCAL_PACKET_PACKET_SIZE")
        self._sequence = sequence
        try:
            written = self._socket.sendto(packet, _address(self._binding))
        except (BlockingIOError, ConnectionRefusedError):
            self.dropped += 1
            return False  # Never queue, rebind, or resend under a newer timestamp.
        if written != len(packet):
            raise RuntimeError("LOCAL_PACKET_INCOMPLETE_SEND")
        self.sent += 1
        sent = True
        return sent

    # 功能：
    #   幂等释放发布套接字，不删除接收端拥有的端点描述符。
    # 输入：
    #   self：当前发布器。
    # 输出：
    #   None：不返回业务数据。
    def close(self):
        if not self._closed:
            self._closed = True
            self._socket.close()
