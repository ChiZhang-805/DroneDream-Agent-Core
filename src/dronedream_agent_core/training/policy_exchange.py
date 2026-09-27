"""Run-bound local policy exchange for a simulation-only offline learner.

Only proposals cross this channel. The existing safety arbiter and flight
controller remain the sole actuation path. Expired or unanswered requests do
not yield actions; a timeout is never retried against a newer observation.
No training framework is imported into the sensor/control process.
"""

from __future__ import annotations

import hmac
import os
import secrets
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from dronedream_plugin_sdk.protocol import copy_json, decode_json, encode_json

from ..contracts import StrictModel
from ..hashing import sha256_json
from ..local_expert_harness import NavigationExpertRole
from ..plugin_files import check_plain_plugin_path, read_plugin_file
from .flight_environment import PilotAction

MAX_PACKET_BYTES = 2 * 1024 * 1024
PROTOCOL = "dronedream-simulation-policy-exchange"
DEFER_REASONS = frozenset({"insufficient-input-budget", "proposal-budget-exhausted",
                          "stream-native-state-not-new", "stream-input-expired-at-adoption"})


class TrainingProposalDeferred(TimeoutError):
    """经当前运行鉴权、请求摘要绑定的未执行声明；不是动作或新租期。"""

    reason_code = "TRAINING_POLICY_PROPOSAL_DEFERRED"


@dataclass(frozen=True, slots=True)
class _PendingIdentity:
    """只保留认证消息的不可变绑定；不持有交给学习器的可变观测树。"""

    request_sha256: str
    valid_until_unix_ms: int


class TrainingProposal(StrictModel):
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expert_role: NavigationExpertRole
    action: PilotAction


# 功能：
#   检查时钟秒值的类型和有限范围，避免 NaN、布尔值或巨大整数进入系统超时接口。
# 输入：
#   value：时钟读数或截止秒值。
# 输出：
#   seconds：可安全计算的有限非负秒值。
def _clock_seconds(value) -> float:
    if type(value) not in (float, int) or not 0 <= value <= 2**53:
        raise ValueError("TRAINING_POLICY_CLOCK_INVALID")
    seconds = float(value)
    return seconds


# 功能：
#   计算同一单调截止时刻的剩余时间，过期即失败，不因分片或处理步骤重开预算。
# 输入：
#   deadline：本次交换的单调时钟截止秒值。
# 输出：
#   remaining：至多六十秒的正剩余时长。
def _time_remaining(deadline: float) -> float:
    remaining = _clock_seconds(deadline) - _clock_seconds(time.monotonic())
    if remaining <= 0:
        raise TimeoutError("TRAINING_POLICY_EXCHANGE_EXPIRED")
    if remaining > 60:
        raise ValueError("TRAINING_POLICY_CLOCK_INVALID")
    return remaining


# 功能：
#   在统一截止时间内读取指定字节数，拒绝非法长度和提前断开的连接。
# 输入：
#   stream：本地交换套接字。
#   count：需要读取的字节数，上限为最大消息帧。
#   deadline：本次交换的单调时钟截止秒值。
# 输出：
#   content：完整接收的指定长度字节。
def _read_exact(stream: socket.socket, count: int, *, deadline: float) -> bytes:
    if type(count) is not int or not 1 <= count <= MAX_PACKET_BYTES:
        raise ValueError("TRAINING_POLICY_READ_SIZE_INVALID")
    parts = bytearray()
    while len(parts) < count:
        remaining = _time_remaining(deadline)
        stream.settimeout(remaining)
        chunk = stream.recv(count - len(parts))
        if not chunk:
            raise ConnectionError("TRAINING_POLICY_PEER_CLOSED")
        parts.extend(chunk)
    content = bytes(parts)
    return content


# 功能：
#   核对帧大小和 HMAC 后严格解析 JSON，认证通过不代表消息内容或动作合法。
# 输入：
#   stream：本地交换套接字。
#   secret：当前训练运行的认证密钥。
#   deadline：本次交换统一的单调时钟截止秒值。
# 输出：
#   value：认证和结构检查通过的消息对象。
def _receive(stream: socket.socket, secret: bytes, *, deadline: float) -> dict:
    length = struct.unpack("!I", _read_exact(stream, 4, deadline=deadline))[0]
    if not 33 <= length <= MAX_PACKET_BYTES:
        raise ValueError("TRAINING_POLICY_PACKET_SIZE_INVALID")
    packet = _read_exact(stream, length, deadline=deadline)
    signature, body = packet[:32], packet[32:]
    if not hmac.compare_digest(signature, hmac.digest(secret, body, "sha256")):
        raise ValueError("TRAINING_POLICY_AUTHENTICATION_FAILED")
    try:
        value = decode_json(body, limit=MAX_PACKET_BYTES - 32)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("TRAINING_POLICY_PACKET_INVALID") from error
    if not isinstance(value, dict):
        raise ValueError("TRAINING_POLICY_PACKET_NOT_OBJECT")
    return value


# 功能：
#   对有界严格 JSON 对象加签并发送，不延长观测的原始控制期限。
# 输入：
#   stream：本地交换套接字。
#   value：待发送的消息对象。
#   secret：当前训练运行的认证密钥。
#   deadline：本次交换统一的单调时钟截止秒值。
# 输出：
#   None：不返回业务数据。
def _send(stream: socket.socket, value: dict, secret: bytes, *, deadline: float) -> None:
    if not isinstance(value, dict):
        raise ValueError("TRAINING_POLICY_PACKET_NOT_OBJECT")
    body = encode_json(value, limit=MAX_PACKET_BYTES - 32).encode("utf-8")
    packet = hmac.digest(secret, body, "sha256") + body
    if len(packet) > MAX_PACKET_BYTES:
        raise ValueError("TRAINING_POLICY_PACKET_SIZE_INVALID")
    remaining = _time_remaining(deadline)
    stream.settimeout(remaining)
    stream.sendall(struct.pack("!I", len(packet)) + packet)


# 功能：
#   有界读取明确选中的本地描述文件，验证回环地址、端口、运行标识和密钥，不自动发现端点。
# 输入：
#   path：当前训练运行的描述文件路径。
# 输出：
#   value：通过严格校验的连接配置。
def read_descriptor(path: Path) -> dict:
    raw = read_plugin_file(path, limit=2048)
    try:
        value = decode_json(raw, limit=2048)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("TRAINING_POLICY_DESCRIPTOR_INVALID") from error
    if (not isinstance(value, dict)
            or set(value) != {"protocol", "host", "port", "secret", "run_id"}
            or value["protocol"] != PROTOCOL or value["host"] != "127.0.0.1"
            or type(value["port"]) is not int or not 1024 <= value["port"] <= 65535
            or not isinstance(value["run_id"], str) or len(value["run_id"]) != 32
            or any(c not in "0123456789abcdef" for c in value["run_id"])
            or not isinstance(value["secret"], str) or len(value["secret"]) != 64
            or any(c not in "0123456789abcdef" for c in value["secret"])):
        raise ValueError("TRAINING_POLICY_DESCRIPTOR_INVALID")
    try:
        bytes.fromhex(value["secret"])
    except ValueError as error:
        raise ValueError("TRAINING_POLICY_DESCRIPTOR_INVALID") from error
    return value


# 功能：
#   验证请求的 Unix 毫秒截止值，要求当前剩余有效期在零至二百五十毫秒之间。
# 输入：
#   request：包含源观测有效期的请求对象。
# 输出：
#   seconds：按当前墙钟计算的正剩余秒数。
def _remaining_lease(request: dict) -> float:
    if not isinstance(request, dict):
        raise ValueError("TRAINING_POLICY_REQUEST_INVALID")
    stamp = request.get("valid_until_unix_ms")
    if type(stamp) is not int or not 0 <= stamp < 2**63:
        raise ValueError("TRAINING_POLICY_REQUEST_INVALID")
    remaining_ms = stamp - int(_clock_seconds(time.time()) * 1000)
    if remaining_ms <= 0:
        raise ValueError("TRAINING_POLICY_INPUT_EXPIRED")
    if remaining_ms > 250:
        raise ValueError("TRAINING_POLICY_INPUT_LEASE_EXCEEDS_BOUND")
    seconds = remaining_ms / 1000
    return seconds


class TrainingPolicyClient:
    """Called by the existing local model worker, never by a setpoint publisher."""

    # 功能：
    #   固定一个已验证的训练运行连接，不因之后描述文件被替换而重新选取学习器。
    # 输入：
    #   self：待初始化的客户端。
    #   descriptor：用户明确选择的本地运行描述文件。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, descriptor: Path):
        self.binding = read_descriptor(descriptor)

    # 功能：
    #   冻结请求后执行一次本地交换，仅返回及时且绑定同一观测摘要的提案，不重试过期动作。
    # 输入：
    #   self：绑定当前训练运行的客户端。
    #   request：带原始有效期的传感器请求。
    # 输出：
    #   proposal：通过消息、结构、摘要和双时钟期限检查的提案。
    def propose(self, request: dict) -> TrainingProposal:
        request = copy_json(request, limit=MAX_PACKET_BYTES - 32)
        remaining = _remaining_lease(request)
        deadline_ms = request["valid_until_unix_ms"]
        deadline = _clock_seconds(time.monotonic()) + remaining
        secret = bytes.fromhex(self.binding["secret"])
        envelope = {"run_id": self.binding["run_id"], "request": request}
        with socket.create_connection(("127.0.0.1", self.binding["port"]),
                                      timeout=remaining) as stream:
            stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            _send(stream, envelope, secret, deadline=deadline)
            response = _receive(stream, secret, deadline=deadline)
        if response.get("status") == "deferred":
            if (set(response) != {"status", "request_sha256", "reason"}
                    or response["request_sha256"] != sha256_json(request)
                    or response["reason"] not in DEFER_REASONS):
                raise ValueError("TRAINING_POLICY_DEFERRED_RESPONSE_INVALID")
            raise TrainingProposalDeferred(response["reason"])
        proposal = TrainingProposal.model_validate(response)
        if proposal.request_sha256 != sha256_json(request):
            raise ValueError("TRAINING_POLICY_RESPONSE_OBSERVATION_MISMATCH")
        if (_clock_seconds(time.monotonic()) >= deadline
                or int(_clock_seconds(time.time()) * 1000) >= deadline_ms):
            raise TimeoutError("TRAINING_POLICY_EXCHANGE_EXPIRED")
        return proposal


class TrainingPolicyExchange:
    """Learner-owned, one outstanding request; no replay queue of stale actions."""

    # 功能：
    #   创建当前训练运行的独占描述文件与单连接回环监听器，初始化失败时释放监听资源。
    # 输入：
    #   self：待初始化的服务端。
    #   descriptor：尚不存在的运行描述文件路径。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, descriptor: Path):
        self.path = descriptor.absolute()
        check_plain_plugin_path(self.path)
        self._descriptor_identity = None
        self._stream: socket.socket | None = None
        self._request: _PendingIdentity | None = None
        self._deadline = 0.
        self._closed = False
        self._seen: set[str] = set()
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._listener.bind(("127.0.0.1", 0))
            self._listener.listen(1)
            self._value = {"protocol": PROTOCOL, "host": "127.0.0.1",
                           "port": self._listener.getsockname()[1],
                           "secret": secrets.token_hex(32), "run_id": secrets.token_hex(16)}
            self._secret = bytes.fromhex(self._value["secret"])
            self._descriptor_content = encode_json(self._value, limit=2048).encode("utf-8")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            check_plain_plugin_path(self.path)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                self._descriptor_identity = os.fstat(fd)
                handle = os.fdopen(fd, "w", encoding="utf-8")
            except BaseException:
                os.close(fd)
                raise
            with handle:
                content = self._descriptor_content.decode("utf-8")
                if handle.write(content) != len(content):
                    raise OSError("TRAINING_POLICY_DESCRIPTOR_SHORT_WRITE")
        except BaseException:
            self._listener.close()
            self._remove_descriptor(incomplete=True)
            raise

    # 功能：
    #   接纳未过期且未重放的请求，将解析树移交学习器，只留存不可变摘要和原始截止时刻。
    # 输入：
    #   self：当前运行的交换服务端。
    #   timeout_seconds：至多六十秒的等待连接时间，不替代观测自己的期限。
    # 输出：
    #   request：已通过运行身份、消息、时效和容量检查的请求对象。
    def wait_request(self, *, timeout_seconds: float) -> dict:
        if self._closed or self._stream is not None:
            raise RuntimeError("TRAINING_POLICY_EXCHANGE_STATE_INVALID")
        if type(timeout_seconds) not in (float, int) or not 0 < timeout_seconds <= 60:
            raise ValueError("TRAINING_POLICY_WAIT_INVALID")
        self._listener.settimeout(timeout_seconds)
        stream, peer = self._listener.accept()
        try:
            if peer[0] != "127.0.0.1":
                raise ValueError("TRAINING_POLICY_REMOTE_PEER_FORBIDDEN")
            stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            envelope = _receive(stream, self._secret,
                                deadline=_clock_seconds(time.monotonic()) + .25)
            if (set(envelope) != {"run_id", "request"}
                    or envelope["run_id"] != self._value["run_id"]):
                raise ValueError("TRAINING_POLICY_RUN_MISMATCH")
            request = envelope["request"]
            if not isinstance(request, dict) or type(request.get("valid_until_unix_ms")) is not int:
                raise ValueError("TRAINING_POLICY_REQUEST_INVALID")
            remaining = _remaining_lease(request)
            deadline = _clock_seconds(time.monotonic()) + remaining
            digest = sha256_json(request)
            # _receive 的严格解析已创建独占 JSON 树；服务器后续只需要摘要和期限。
            # 不再复制整棵树或在回复时重新散列，学习器改写内容也不能改写原始绑定。
            identity = _PendingIdentity(digest, request["valid_until_unix_ms"])
            # Serialization and hashing consume the same lease as transport.
            # Reject before storing pending state or exposing expired data.
            if (_clock_seconds(time.monotonic()) >= deadline
                    or int(_clock_seconds(time.time()) * 1000) >= request["valid_until_unix_ms"]
                    or digest in self._seen):
                raise ValueError("TRAINING_POLICY_REQUEST_EXPIRED_OR_REPLAYED")
            if len(self._seen) >= 100_000:
                raise ValueError("TRAINING_POLICY_EPISODE_CAPACITY_EXCEEDED")
            self._seen.add(digest)
            self._stream, self._request = stream, identity
            # Hashing/parsing may consume time; never add it back to the lease.
            self._deadline = deadline
            return request
        except BaseException:
            stream.close()
            raise

    # 功能：
    #   重验提案并回复当前请求，无论成功、身份不符或过期都消耗该请求并关闭连接。
    # 输入：
    #   self：持有唯一待回复请求的服务端。
    #   proposal：学习器生成的结构化操纵提案。
    # 输出：
    #   None：不返回业务数据。
    def reply(self, proposal: TrainingProposal) -> None:
        if self._stream is None or self._request is None or self._closed:
            raise RuntimeError("TRAINING_POLICY_NO_PENDING_REQUEST")
        stream = self._stream
        try:
            if not isinstance(proposal, TrainingProposal):
                raise ValueError("TRAINING_POLICY_PROPOSAL_INVALID")
            proposal = TrainingProposal.model_validate(proposal.model_dump(mode="json"))
            if proposal.request_sha256 != self._request.request_sha256:
                raise ValueError("TRAINING_POLICY_PROPOSAL_OBSERVATION_MISMATCH")
            # 墙钟前跳时单调期限可能尚未到期；服务端也必须拒绝原始有效期已过的回复。
            if int(_clock_seconds(time.time()) * 1000) >= self._request.valid_until_unix_ms:
                raise TimeoutError("TRAINING_POLICY_EXCHANGE_EXPIRED")
            _send(stream, proposal.model_dump(mode="json"), self._secret, deadline=self._deadline)
        finally:
            stream.close()
            self._stream = self._request = None

    # 功能：
    #   尝试删除本次独占创建且身份和内容仍匹配的描述文件，不删除同内容的新文件。
    # 输入：
    #   self：保存初始文件身份和内容的服务端。
    #   incomplete：初始化失败时是否允许清理本次写入的内容前缀。
    # 输出：
    #   None：不返回业务数据。
    def _remove_descriptor(self, *, incomplete: bool = False) -> None:
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
        except (OSError, ValueError):
            # 文件已变化或无法证明归属时保留现场；这不是对恶意进程竞态的系统隔离。
            pass

    # 功能：
    #   幂等关闭连接和监听器，最后只清理仍属于本次运行的描述文件。
    # 输入：
    #   self：需要关闭的交换服务端。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.discard_pending()
        finally:
            try:
                self._listener.close()
            finally:
                self._remove_descriptor()

    # 功能：
    #   丢弃已被替代的待处理请求并关闭连接，不伪造悬停或其他学习器动作。
    # 输入：
    #   self：当前交换服务端。
    # 输出：
    #   None：不返回业务数据。
    def discard_pending(self, *, reason: str | None = None) -> None:
        # 功能：只对已鉴权请求发送明确的时效拒绝；其他取消/故障仍为断开。
        # 输入：固定原因或 None；输出：无动作，保持请求的原截止时间并关闭连接。
        if reason is not None and reason not in DEFER_REASONS:
            raise ValueError("TRAINING_POLICY_DEFER_REASON_INVALID")
        if self._stream is not None:
            try:
                if reason is not None and self._request is not None:
                    try:
                        _send(self._stream, {"status": "deferred",
                            "request_sha256": self._request.request_sha256, "reason": reason},
                            self._secret, deadline=self._deadline)
                    except (OSError, TimeoutError):
                        pass  # 超期仍关闭，不续租，也不伪造已送达的拒绝回执。
            finally:
                self._stream.close()
                self._stream = self._request = None
