"""Authenticated, run-bound simulation witnesses, never policy input or authority.

Every received witness is retained in order, unlike latest-value telemetry.
There is no disk fallback, timestamp renewal, replay or interpolation. Loss is
an invalid outcome window, not permission to fabricate missing labels.
"""

from __future__ import annotations

import hmac
import os
import secrets
import socket
import sys
from pathlib import Path

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from ..contracts import RuntimeLocalSafetyObservation
from ..plugin_files import check_plain_plugin_path, read_plugin_file

PROTOCOL = "dronedream-simulation-outcome"
TRANSPORT = "abstract-unix-datagram" if sys.platform.startswith("linux") else "loopback-udp"
FAMILY = socket.AF_UNIX if TRANSPORT == "abstract-unix-datagram" else socket.AF_INET
PREFIX = "dronedream-outcome-"
MAX_PACKET_BYTES = 60_000


# 功能：
#   在明确学习通道旁派生独立真值描述路径，不与策略消息共用端点。
# 输入：
#   policy_channel：本次运行的学习通道路径。
# 输出：
#   path：独立真值通道描述文件路径。
def descriptor_path(policy_channel: Path) -> Path:
    path = policy_channel.with_name(policy_channel.name + ".outcome.json")
    return path


# 功能：
#   将已经验证的本机绑定转换为抽象 Unix 地址或 IPv4 回环端口。
# 输入：
#   binding：已校验或由本进程生成的地址字段。
# 输出：
#   address：当前平台的 socket 地址。
def _address(binding):
    address = ("\0" + binding["address"] if TRANSPORT == "abstract-unix-datagram"
               else ("127.0.0.1", binding["address"]))
    return address


# 功能：
#   有界读取普通描述文件，严格检查 JSON、协议、本机地址和每次运行的独立密钥。
# 输入：
#   path：明确选中的本次运行描述路径。
# 输出：
#   value：校验通过的连接描述。
def _binding(path: Path) -> dict:
    raw = read_plugin_file(path, limit=2048)
    try:
        value = decode_json(raw, limit=2048)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("OUTCOME_DESCRIPTOR_INVALID") from error
    if (not isinstance(value, dict)
            or set(value) != {"protocol", "transport", "address", "secret"}
            or value["protocol"] != PROTOCOL or value["transport"] != TRANSPORT
            or not isinstance(value["secret"], str) or len(value["secret"]) != 64
            or any(c not in "0123456789abcdef" for c in value["secret"])):
        raise ValueError("OUTCOME_DESCRIPTOR_INVALID")
    address = value["address"]
    if TRANSPORT == "abstract-unix-datagram":
        if (not isinstance(address, str) or not address.startswith(PREFIX)
                or len(address) != len(PREFIX) + 32
                or any(c not in "0123456789abcdef" for c in address[len(PREFIX):])):
            raise ValueError("OUTCOME_ADDRESS_INVALID")
    elif type(address) is not int or not 1024 <= address <= 65535:
        raise ValueError("OUTCOME_ADDRESS_INVALID")
    return value


class OutcomeReceiver:
    """Receive every independent witness in sequence; loss invalidates the window."""

    # 功能：
    #   建立非阻塞本机接收端并独占创建描述文件，初始化失败时关闭资源并尝试清理自有前缀。
    # 输入：
    #   self：待初始化接收器。
    #   path：必须尚不存在的描述文件路径。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, path: Path):
        self.path = Path(path).absolute()
        check_plain_plugin_path(self.path)
        self._closed = False
        self._failed = False
        self._descriptor_identity = None
        self._sequence = 0
        self._socket = socket.socket(FAMILY, socket.SOCK_DGRAM)
        try:
            address = (PREFIX + secrets.token_hex(16)
                       if TRANSPORT == "abstract-unix-datagram" else 0)
            self._socket.bind(_address({"address": address}))
            if FAMILY == socket.AF_INET:
                address = self._socket.getsockname()[1]
            self._value = {"protocol": PROTOCOL, "transport": TRANSPORT,
                           "address": address, "secret": secrets.token_hex(32)}
            self._secret = bytes.fromhex(self._value["secret"])
            self._descriptor_content = encode_json(self._value, limit=2048).encode("utf-8")
            self._socket.setblocking(False)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            check_plain_plugin_path(self.path)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                self._descriptor_identity = os.fstat(fd)
                stream = os.fdopen(fd, "wb")
            except BaseException:
                os.close(fd)
                raise
            with stream:
                if stream.write(self._descriptor_content) != len(self._descriptor_content):
                    raise OSError("OUTCOME_DESCRIPTOR_SHORT_WRITE")
        except BaseException:
            try:
                self._socket.close()
            finally:
                self._remove_descriptor(incomplete=True)
            raise

    # 功能：
    #   接收一个有界完整批次；任何认证、结构或序号错误都会锁存失败，后续不能恢复旧通道。
    # 输入：
    #   self：本次运行的非阻塞接收器。
    # 输出：
    #   rows：本批次通过检查的原始独立见证，空队列时为空列表。
    def read(self) -> list[RuntimeLocalSafetyObservation]:
        if self._closed:
            raise RuntimeError("OUTCOME_CHANNEL_CLOSED")
        if self._failed:
            raise ValueError("OUTCOME_CHANNEL_FAILED")
        try:
            rows = self._read_batch()
        except BaseException:
            self._failed = True
            raise
        return rows

    # 功能：
    #   最多读取六十四个原始数据报，逐包验证来源、大小、HMAC、严格 JSON 和连续序号。
    # 输入：
    #   self：已检查状态的接收器。
    # 输出：
    #   rows：全部验证通过的有序批次。
    def _read_batch(self) -> list[RuntimeLocalSafetyObservation]:
        rows = []
        for _ in range(64):
            try:
                packet, peer = self._socket.recvfrom(MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                break
            if (not 33 < len(packet) <= MAX_PACKET_BYTES
                    or FAMILY == socket.AF_INET and peer[0] != "127.0.0.1"):
                raise ValueError("OUTCOME_PACKET_SIZE_OR_PEER")
            signature, body = packet[:32], packet[32:]
            if not hmac.compare_digest(signature, hmac.digest(self._secret, body, "sha256")):
                raise ValueError("OUTCOME_AUTHENTICATION_FAILED")
            payload = decode_json(body, limit=MAX_PACKET_BYTES - 32)
            row = RuntimeLocalSafetyObservation.model_validate(payload, strict=True)
            if row.source != "simulation-ground-truth":
                raise ValueError("OUTCOME_REQUIRES_INDEPENDENT_SIMULATOR_WITNESS")
            if row.sequence != self._sequence + 1:
                raise ValueError("OUTCOME_CHANNEL_WITNESS_LOST_OR_REPLAYED")
            # Sequence is never repaired or renumbered. The monitor treats any
            # batch error as a failed outcome stream, not a successful prefix.
            self._sequence = row.sequence
            rows.append(row)
        return rows

    # 功能：
    #   只清理文件身份与内容仍属于本次创建的描述，无法证明归属时保留现场。
    # 输入：
    #   self：保留原始描述身份与字节的接收器。
    #   incomplete：初始化失败时是否接受本次内容的未完成前缀。
    # 输出：
    #   None：不返回业务数据。
    def _remove_descriptor(self, *, incomplete=False):
        if self._descriptor_identity is None:
            return
        try:
            check_plain_plugin_path(self.path)
            if not os.path.samestat(self._descriptor_identity, self.path.stat()):
                return
            content = read_plugin_file(self.path, limit=2048)
            matches = (self._descriptor_content.startswith(content) if incomplete
                       else content == self._descriptor_content)
            if matches and os.path.samestat(self._descriptor_identity, self.path.stat()):
                self.path.unlink()
        except (ValueError, OSError):
            pass  # 不能证明归属时不删；这不替代针对恶意本机进程竞态的系统隔离。

    # 功能：
    #   幂等关闭接收套接字并尝试清理仍属于本次运行的描述。
    # 输入：
    #   self：待关闭的接收器。
    # 输出：
    #   None：不返回业务数据。
    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._socket.close()
        finally:
            self._remove_descriptor()


class OutcomePublisher:
    """Load the explicit descriptor before callbacks start; send without disk I/O."""

    # 功能：
    #   在回调启动前固定运行描述及认证密钥，发送热路径不重新读取磁盘或寻找替代端点。
    # 输入：
    #   self：待初始化发布器。
    #   path：接收器明确创建的描述文件。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, path: Path):
        self._value = _binding(path)
        self._secret = bytes.fromhex(self._value["secret"])
        self._socket = socket.socket(FAMILY, socket.SOCK_DGRAM)
        try:
            self._socket.setblocking(False)
        except BaseException:
            self._socket.close()
            raise
        self._closed = False
        self.sent = self.dropped = 0

    # 功能：
    #   重验独立见证并以非阻塞数据报发送；丢包明确计数，不重试、改序号或刷新源时刻。
    # 输入：
    #   self：已固定运行身份的发布器。
    #   row：原始独立仿真见证。
    # 输出：
    #   sent：完整发送时为 True，已知非阻塞丢包时为 False。
    def send(self, row: RuntimeLocalSafetyObservation) -> bool:
        if self._closed:
            raise RuntimeError("OUTCOME_CHANNEL_CLOSED")
        if not isinstance(row, RuntimeLocalSafetyObservation):
            raise ValueError("OUTCOME_WITNESS_TYPE_INVALID")
        if row.source != "simulation-ground-truth":
            raise ValueError("OUTCOME_REQUIRES_INDEPENDENT_SIMULATOR_WITNESS")
        row = RuntimeLocalSafetyObservation.model_validate(row.model_dump(), strict=True)
        body = encode_json(row.model_dump(mode="json"), limit=MAX_PACKET_BYTES - 32).encode("utf-8")
        packet = hmac.digest(self._secret, body, "sha256") + body
        if len(packet) > MAX_PACKET_BYTES:
            raise ValueError("OUTCOME_PACKET_TOO_LARGE")
        try:
            size = self._socket.sendto(packet, _address(self._value))
        except (BlockingIOError, ConnectionRefusedError):
            self.dropped += 1
            sent = False
            return sent
        if size != len(packet):
            raise RuntimeError("OUTCOME_INCOMPLETE_SEND")
        self.sent += 1
        sent = True
        return sent

    # 功能：
    #   幂等关闭发布器套接字，不删除由接收器拥有的描述文件。
    # 输入：
    #   self：待关闭发布器。
    # 输出：
    #   None：不返回业务数据。
    def close(self):
        if not self._closed:
            self._closed = True
            self._socket.close()
