"""Run-scoped, bounded local transport for one atomic safety/observation pair.

The channel is a transport, not an authority: the executor still checks the
original source deadline, task binding and physical limits. Files remain the
durable evidence path but DrvFS rename latency is not in the actuation path.
Loss, replay or an absent producer never renews the cached command's lease.
Linux uses abstract Unix sockets, avoiding both filesystem socket artifacts and
WSL IP fragmentation/filtering. Other supported desktop hosts use loopback UDP.
Every receiver verifies the full maximum datagram size before advertising itself.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import os
import secrets
import socket
import sys
from pathlib import Path

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .contracts import RuntimeLocalSafetyCommand, RuntimeLocalSafetyObservation
from .hashing import sha256_json
from .local_packet_channel import PacketContract, _publish_descriptor, read_descriptor
from .plugin_files import check_plain_plugin_path

_MAX_PACKET = 60_000
_PROTOCOL = "dronedream-local-safety-pair"
_TRANSPORT = "abstract-unix-datagram" if sys.platform.startswith("linux") else "loopback-udp"
_FAMILY = socket.AF_UNIX if _TRANSPORT == "abstract-unix-datagram" else socket.AF_INET
_CONTRACT = PacketContract(
    _PROTOCOL, "dronedream-safety-", "dronedream.runtime-local-safety-observation.v1"
)


# 功能：
#   将已验证的运行描述符转换为固定本机地址，不接受远端主机。
# 输入：
#   binding：已验证的当前平台端点绑定。
# 输出：
#   address：抽象 Unix 地址或 IPv4 回环地址与端口。
def _address(binding: dict):
    address = (
        "\0" + binding["address"]
        if binding["transport"] == "abstract-unix-datagram"
        else ("127.0.0.1", binding["address"])
    )
    return address


# 功能：
#   使用公共普通文件边界读取最多 2 KiB 的安全通道描述符，拒绝重复键、链接及身份漂移。
# 输入：
#   path：当前运行的端点文件。
# 输出：
#   value：包含安全通道协议、地址与随机认证材料的已验证字典。
def _descriptor(path: Path) -> dict:
    value = read_descriptor(path, _CONTRACT)
    return value


# 功能：
#   验证已重验模型的机载来源、观测摘要与序号、生成时间及最多两秒的短租期；不授予飞行权限。
# 输入：
#   observation：已通过字段校验的机载观测。
#   command：已通过字段校验的短时控制指令。
# 输出：
#   None：不返回业务数据。
def validate_pair(
    observation: RuntimeLocalSafetyObservation, command: RuntimeLocalSafetyCommand
) -> None:
    if observation.source != "onboard" or command.source != "onboard":
        raise ValueError("LOCAL_SAFETY_CHANNEL_REQUIRES_ONBOARD_EVIDENCE")
    if (
        command.observation_sequence != observation.sequence
        or command.observation_sha256 != sha256_json(observation)
        or observation.observed_at_unix_ms > command.generated_at_unix_ms
    ):
        raise ValueError("LOCAL_SAFETY_CHANNEL_PAIR_MISMATCH")
    if not 0 < command.valid_until_unix_ms - command.generated_at_unix_ms <= 2_000:
        raise ValueError("LOCAL_SAFETY_CHANNEL_COMMAND_LEASE_INVALID")


class LocalSafetyReceiver:
    """Executor-owned endpoint; no command writes and no network listen beyond loopback."""

    # 功能：
    #   创建本次执行独占的非阻塞端点，容量探测成功后无覆盖发布完整描述符，失败释放套接字。
    # 输入：
    #   self：执行器持有的接收端。
    #   descriptor_path：当前运行尚不存在的端点文件。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, descriptor_path: Path):
        self.path = descriptor_path.absolute()
        check_plain_plugin_path(self.path)
        if self.path.exists():
            raise FileExistsError("LOCAL_SAFETY_CHANNEL_DESCRIPTOR_EXISTS")
        self._socket = socket.socket(_FAMILY, socket.SOCK_DGRAM)
        self._latest: tuple[RuntimeLocalSafetyObservation, RuntimeLocalSafetyCommand] | None = None
        self._order: tuple[int, int] = (-1, -1)
        self.accepted = self.rejected = 0
        self._closed = False
        try:
            address = (
                "dronedream-safety-" + secrets.token_hex(16)
                if _TRANSPORT == "abstract-unix-datagram"
                else 0
            )
            self._socket.bind(_address({"transport": _TRANSPORT, "address": address}))
            if _TRANSPORT == "loopback-udp":
                address = self._socket.getsockname()[1]
            self._socket.setblocking(False)
            self._secret = secrets.token_bytes(32)
            self._value = {
                "protocol": _PROTOCOL,
                "transport": _TRANSPORT,
                "address": address,
                "secret": self._secret.hex(),
            }
            self._verify_datagram_capacity()
            # 复用完整写出后无覆盖发布的实现，不暴露只写到一半的端点。
            self._descriptor_identity = _publish_descriptor(self.path, self._value)
        except BaseException:
            self._socket.close()
            raise

    # 功能：
    #   在发布端点之前实际回传最大长度数据报，防止平台容量不足却宣称通道可用。
    # 输入：
    #   self：已绑定但未发布的接收端。
    # 输出：
    #   None：不返回业务数据。
    def _verify_datagram_capacity(self) -> None:
        probe = secrets.token_bytes(_MAX_PACKET)
        with socket.socket(_FAMILY, socket.SOCK_DGRAM) as sender:
            sender.settimeout(0.5)
            self._socket.settimeout(0.5)
            try:
                sender.sendto(probe, _address(self._value))
                observed, _ = self._socket.recvfrom(_MAX_PACKET + 1)
                if observed != probe:
                    raise ValueError("LOCAL_SAFETY_CHANNEL_CAPACITY_PROBE_MISMATCH")
            except OSError as error:
                raise RuntimeError("LOCAL_SAFETY_CHANNEL_CAPACITY_PROBE_FAILED") from error
            finally:
                self._socket.setblocking(False)

    # 功能：
    #   提供容量探测与接收计数，不披露密钥，也不将接收成功等同于动作已执行。
    # 输入：
    #   self：已成功建立的接收端。
    # 输出：
    #   summary：本次端点的传输类型、报文预算和收拒计数。
    def summary(self) -> dict:
        summary = {
            "transport": _TRANSPORT,
            "maximum_packet_bytes": _MAX_PACKET,
            "capacity_probe_passed": True,
            "accepted_pairs": self.accepted,
            "rejected_packets": self.rejected,
        }
        return summary

    # 功能：
    #   1. 最多读取六十四帧，拒绝认证错误、重放、时钟倒退和畸形结构，坏帧不替换缓存。
    #   2. 返回私有缓存的深副本，不更新原观测或租期；消费者仍须检查时效与任务授权。
    # 输入：
    #   self：单消费者使用的当前接收端。
    # 输出：
    #   latest：最近合法的观测指令对；关闭或尚无合法数据时为 None。
    def read_latest(self) -> tuple[RuntimeLocalSafetyObservation, RuntimeLocalSafetyCommand] | None:
        latest = None
        if self._closed:
            return latest
        # A finite drain budget prevents a noisy local process starving control.
        for _ in range(64):
            try:
                packet, peer = self._socket.recvfrom(_MAX_PACKET + 1)
            except BlockingIOError:
                break
            except OSError as error:
                # Windows 丢弃过大 UDP 帧并报告截断，不能让一帧终止控制循环。
                if error.errno == errno.EMSGSIZE or getattr(error, "winerror", None) == 10040:
                    self.rejected += 1
                    continue
                raise
            try:
                if (_TRANSPORT == "loopback-udp" and peer[0] != "127.0.0.1") or not 65 < len(
                    packet
                ) <= _MAX_PACKET:
                    raise ValueError("LOCAL_SAFETY_CHANNEL_PACKET_SIZE")
                signature, body = packet[:64], packet[64:]
                expected = hmac.new(self._secret, body, hashlib.sha256).hexdigest().encode()
                if not hmac.compare_digest(signature, expected):
                    raise ValueError("LOCAL_SAFETY_CHANNEL_AUTHENTICATION_FAILED")
                value = decode_json(body, limit=_MAX_PACKET - 64)
                if not isinstance(value, dict) or set(value) != {"observation", "command"}:
                    raise ValueError("LOCAL_SAFETY_CHANNEL_PACKET_INVALID")
                observation = RuntimeLocalSafetyObservation.model_validate(
                    value["observation"], strict=True
                )
                command = RuntimeLocalSafetyCommand.model_validate(value["command"], strict=True)
                validate_pair(observation, command)
                order = command.observation_sequence, command.generated_at_unix_ms
                if (
                    order <= self._order
                    or command.generated_at_unix_ms < self._order[1]
                    or self._latest is not None
                    and observation.observed_at_unix_ms < self._latest[0].observed_at_unix_ms
                ):
                    raise ValueError("LOCAL_SAFETY_CHANNEL_REPLAY")
                self._order = order
                self._latest = observation, command
                self.accepted += 1
            except (ValueError, TypeError, KeyError, RecursionError):
                self.rejected += 1
        if self._latest is None:
            return latest
        latest = tuple(item.model_copy(deep=True) for item in self._latest)
        return latest

    # 功能：
    #   幂等关闭套接字，只移除身份、内容和最终状态仍属于本次运行的端点，不删除替换文件。
    # 输入：
    #   self：当前接收端。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close()
        try:
            check_plain_plugin_path(self.path)
            before = self.path.stat()
            if (
                os.path.samestat(self._descriptor_identity, before)
                and _descriptor(self.path) == self._value
            ):
                check_plain_plugin_path(self.path)
                current = self.path.stat()
                if (
                    os.path.samestat(before, current)
                    and before.st_size == current.st_size
                    and before.st_mtime_ns == current.st_mtime_ns
                ):
                    self.path.unlink()
        except (OSError, ValueError):
            pass  # Never remove a replaced or unknown descriptor.


class LocalSafetyPublisher:
    # 功能：
    #   建立非阻塞发送端，首次发送时才读取接收器绑定；不创建或占有接收器的文件。
    # 输入：
    #   self：由单个生产者使用的发送端。
    #   descriptor_path：本次执行的端点文件。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, descriptor_path: Path):
        self.path = descriptor_path.absolute()
        check_plain_plugin_path(self.path)
        self._socket = socket.socket(_FAMILY, socket.SOCK_DGRAM)
        try:
            self._socket.setblocking(False)
        except BaseException:
            self._socket.close()
            raise
        self._binding: dict | None = None
        self._closed = False
        self.sent = self.not_ready = 0

    # 功能：
    #   1. 重验并私有复制观测和指令，绑定摘要后在单帧中认证发送；不排队、不重试、不续租。
    #   2. 首次绑定后不会改连另一次执行；本机传输接受报文不表示接收器已验证或执行。
    # 输入：
    #   self：当前运行的发送端。
    #   observation：本次机载观测。
    #   command：该观测对应的短时指令。
    # 输出：
    #   delivered：整帧交给本机传输为 True，未就绪或拥塞时为 False。
    def send(
        self, observation: RuntimeLocalSafetyObservation, command: RuntimeLocalSafetyCommand
    ) -> bool:
        if self._closed:
            raise RuntimeError("LOCAL_SAFETY_CHANNEL_CLOSED")
        observation = RuntimeLocalSafetyObservation.model_validate(
            observation.model_dump(mode="python"), strict=True
        )
        command = RuntimeLocalSafetyCommand.model_validate(
            command.model_dump(mode="python"), strict=True
        )
        validate_pair(observation, command)
        delivered = False
        if self._binding is None:
            try:
                self._binding = _descriptor(self.path)
            except FileNotFoundError:
                self.not_ready += 1
                return delivered
        body = encode_json(
            {
                "observation": observation.model_dump(mode="json"),
                "command": command.model_dump(mode="json"),
            },
            limit=_MAX_PACKET - 64,
        ).encode("utf-8")
        signature = (
            hmac.new(bytes.fromhex(self._binding["secret"]), body, hashlib.sha256)
            .hexdigest()
            .encode()
        )
        packet = signature + body
        if len(packet) > _MAX_PACKET:
            raise ValueError("LOCAL_SAFETY_CHANNEL_PACKET_SIZE")
        try:
            sent = self._socket.sendto(packet, _address(self._binding))
        except (BlockingIOError, ConnectionRefusedError):
            self.not_ready += 1
            # The executor may already be down during ordered shutdown. No
            # receiver means no accepted command; never queue or rebind it.
            return delivered
        if sent != len(packet):
            raise RuntimeError("LOCAL_SAFETY_CHANNEL_INCOMPLETE_SEND")
        self.sent += 1
        delivered = True
        return delivered

    # 功能：
    #   幂等释放发送套接字，保留接收端的描述符和已产生的证据文件。
    # 输入：
    #   self：当前发送端。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._socket.close()
