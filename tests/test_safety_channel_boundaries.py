"""Local datagram and descriptor fault injection; no aircraft is controlled."""

import errno
import hashlib
import hmac
import json
import socket

import pytest
from test_local_safety_channel import pair

from dronedream_agent_core import local_safety_channel as channel


# 功能：
#   将测试指定的原始正文按本次真实端点密钥签名，直接发送以覆盖非法协议内容。
# 输入：
#   path：测试接收器创建的端点。
#   body：测试自行构造的原始正文。
# 输出：
#   None：不返回业务数据。
def send_raw(path, body):
    descriptor = json.loads(path.read_text())
    signature = hmac.new(bytes.fromhex(descriptor["secret"]), body, hashlib.sha256).hexdigest()
    with socket.socket(channel._FAMILY, socket.SOCK_DGRAM) as sender:
        sender.sendto(signature.encode() + body, channel._address(descriptor))


# 功能：
#   重复 JSON 键即使具有合法认证，也不能被接收为确定含义的安全指令。
# 输入：
#   tmp_path：测试端点目录。
# 输出：
#   None：不返回业务数据。
def test_authenticated_duplicate_json_is_rejected(tmp_path):
    path = tmp_path / "endpoint.json"
    receiver = channel.LocalSafetyReceiver(path)
    try:
        observation, command = pair()
        obs = observation.model_dump_json()
        body = (
            '{"observation":'
            + obs
            + ',"observation":'
            + obs
            + ',"command":'
            + command.model_dump_json()
            + "}"
        ).encode()
        send_raw(path, body)
        assert receiver.read_latest() is None
        assert receiver.rejected == 1
    finally:
        receiver.close()


# 功能：
#   端点被替换成内容完全相同但身份不同的文件，旧接收器关闭时仍须保留替换文件。
# 输入：
#   tmp_path：测试端点目录。
# 输出：
#   None：不返回业务数据。
def test_close_keeps_identical_replacement_file(tmp_path):
    path = tmp_path / "endpoint.json"
    receiver = channel.LocalSafetyReceiver(path)
    replacement = tmp_path / "replacement.json"
    original = path.read_bytes()
    replacement.write_bytes(original)
    replacement.replace(path)
    receiver.close()
    assert path.read_bytes() == original


# 功能：
#   端点读取拒绝重复字段，不允许依靠 JSON 后值覆盖前值选择运行身份。
# 输入：
#   tmp_path：测试端点目录。
# 输出：
#   None：不返回业务数据。
def test_descriptor_rejects_duplicate_identity_fields(tmp_path):
    path = tmp_path / "endpoint.json"
    receiver = channel.LocalSafetyReceiver(path)
    try:
        body = path.read_text()
        path.write_text('{"protocol":"wrong-domain",' + body[1:])
        with pytest.raises(ValueError):
            channel._descriptor(path)
    finally:
        receiver.close()


# 功能：
#   新观测序号不能掩盖生成时钟倒退，缓存仍保留较新的一对观测和指令。
# 输入：
#   tmp_path：测试端点目录。
# 输出：
#   None：不返回业务数据。
def test_increasing_sequence_does_not_allow_regressed_generation_clock(tmp_path):
    path = tmp_path / "endpoint.json"
    receiver, publisher = channel.LocalSafetyReceiver(path), channel.LocalSafetyPublisher(path)
    try:
        publisher.send(*pair(1, 1020))
        assert receiver.read_latest()[0].sequence == 1
        publisher.send(*pair(2, 1010))
        assert receiver.read_latest()[0].sequence == 1
        assert receiver.rejected == 1
    finally:
        publisher.close()
        receiver.close()


# 功能：
#   认证帧内的非法布尔序号、空租期或过长租期必须拒绝，不进入安全指令缓存。
# 输入：
#   tmp_path：测试端点目录。
#   change：只针对序号或租期构造的非法字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "change",
    [
        {"observation_sequence": True},
        {"valid_until_unix_ms": 1020},
        {"valid_until_unix_ms": 4000},
    ],
)
def test_authenticated_command_fields_are_strict_and_bounded(tmp_path, change):
    path = tmp_path / "endpoint.json"
    receiver = channel.LocalSafetyReceiver(path)
    try:
        observation, command = pair()
        payload = command.model_dump(mode="json")
        payload.update(change)
        body = json.dumps(
            {"observation": observation.model_dump(mode="json"), "command": payload}
        ).encode()
        send_raw(path, body)
        assert receiver.read_latest() is None
        assert receiver.rejected == 1
    finally:
        receiver.close()


# 功能：
#   超大 UDP 帧在 Windows 返回截断错误时仅计为坏帧，继续读取后续有效状态。
# 输入：
#   tmp_path：实际端点目录。
# 输出：
#   None：不返回业务数据。
def test_truncated_datagram_does_not_terminate_reader(tmp_path):
    path = tmp_path / "endpoint.json"
    receiver = channel.LocalSafetyReceiver(path)
    original = receiver._socket

    class TruncatedSocket:
        # 功能：
        #   首次模拟操作系统截断错误，此后转交真实套接字。
        # 输入：
        #   self：测试套接字代理。
        #   size：接收器的字节预算。
        # 输出：
        #   packet：真实套接字返回的数据与来源地址。
        def recvfrom(self, size):
            if not getattr(self, "failed", False):
                self.failed = True
                raise OSError(errno.EMSGSIZE, "truncated datagram")
            packet = original.recvfrom(size)
            return packet

    try:
        receiver._socket = TruncatedSocket()
        assert receiver.read_latest() is None
        assert receiver.rejected == 1
    finally:
        receiver._socket = original
        receiver.close()


# 功能：
#   接收器绑定失败必须关闭刚创建的套接字，不能留下未发布但占用资源的端点。
# 输入：
#   tmp_path：测试描述符目录。
#   monkeypatch：只替换本模块套接字构造的测试工具。
# 输出：
#   None：不返回业务数据。
def test_bind_failure_releases_receiver_socket(tmp_path, monkeypatch):
    class FailedSocket:
        closed = False

        # 功能：
        #   模拟端点绑定被操作系统拒绝。
        # 输入：
        #   self：测试套接字。
        #   address：准备绑定的本机地址。
        # 输出：
        #   None：不返回业务数据。
        def bind(self, address):
            raise OSError("bind rejected")

        # 功能：
        #   记录释放请求，不操作实际网络资源。
        # 输入：
        #   self：测试套接字。
        # 输出：
        #   None：不返回业务数据。
        def close(self):
            self.closed = True

    sock = FailedSocket()
    monkeypatch.setattr(channel.socket, "socket", lambda *_: sock)
    with pytest.raises(OSError, match="bind rejected"):
        channel.LocalSafetyReceiver(tmp_path / "endpoint.json")
    assert sock.closed
