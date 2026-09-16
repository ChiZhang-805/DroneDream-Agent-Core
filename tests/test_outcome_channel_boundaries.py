"""Bounded local witness transport and receiver-owned descriptor cleanup."""

import hmac
import os
from contextlib import suppress
from pathlib import Path

import pytest
from test_training_outcome_channel import endpoints as endpoints
from test_training_runtime_evidence import pose

from dronedream_agent_core.training import outcome_channel as channel


# 功能：
#   验证重复字段的描述文件即使最后字段正确，也不能决定发布器连接目标。
# 输入：
#   endpoints：本机测试收发通道。
# 输出：
#   None：不返回业务数据。
def test_descriptor_rejects_duplicate_keys(endpoints):
    receiver, _ = endpoints
    content = receiver.path.read_bytes()
    receiver.path.write_bytes(b'{"protocol":"other",' + content[1:])
    with pytest.raises(ValueError):
        value = channel.OutcomePublisher(receiver.path)
        value.close()


# 功能：
#   验证合法认证签名不能使重复序号字段通过消息结构校验。
# 输入：
#   endpoints：实际本机通道。
# 输出：
#   None：不返回业务数据。
def test_signed_packet_still_requires_unique_json_keys(endpoints):
    receiver, publisher = endpoints
    body = b'{"sequence":999,' + pose(1, 0).model_dump_json().encode()[1:]
    packet = hmac.digest(publisher._secret, body, "sha256") + body
    publisher._socket.sendto(packet, channel._address(publisher._value))
    with pytest.raises(ValueError):
        receiver.read()


# 功能：
#   验证收到认证失败包后通道永久失败，不因下一条合法帧出现而恢复可用证据。
# 输入：
#   endpoints：实际本机通道。
# 输出：
#   None：不返回业务数据。
def test_receiver_failure_is_sticky(endpoints):
    receiver, publisher = endpoints
    secret = publisher._secret
    publisher._secret = b"different-key"
    publisher.send(pose(1, 0))
    with pytest.raises(ValueError, match="AUTHENTICATION"):
        receiver.read()
    publisher._secret = secret
    publisher.send(pose(1, 0))
    with pytest.raises(ValueError, match="FAILED"):
        receiver.read()


# 功能：
#   验证发布器拒绝绕过字段校验生成的非法见证，不把错误数据发送后才依赖接收端发现。
# 输入：
#   endpoints：本机收发通道。
#   change：越界序号或可能被宽松类型转换掩盖的错误值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("change", [{"sequence": -1}, {"stream_healthy": "yes"},
                                    {"observed_at_unix_ms": False}])
def test_publisher_revalidates_preconstructed_witness(endpoints, change):
    receiver, publisher = endpoints
    invalid = pose(1, 0).model_copy(update=change)
    with pytest.raises(ValueError):
        publisher.send(invalid)
    assert receiver.read() == []
    assert publisher.sent == 0


# 功能：
#   验证同内容但文件身份不同的替代描述，不能被原接收器关闭时删除。
# 输入：
#   tmp_path：独占描述目录。
# 输出：
#   None：不返回业务数据。
def test_close_preserves_same_content_replacement(tmp_path):
    path = tmp_path / "outcome.json"
    receiver = channel.OutcomeReceiver(path)
    original = path.read_bytes()
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(original)
    os.replace(replacement, path)
    receiver.close()
    assert path.read_bytes() == original


# 功能：
#   验证描述路径初始化后固定，切换工作目录不能遗留原描述或清理另一个相对路径。
# 输入：
#   tmp_path：独占测试根。
#   monkeypatch：负责恢复当前目录。
# 输出：
#   None：不返回业务数据。
def test_receiver_freezes_descriptor_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    receiver = channel.OutcomeReceiver(Path("outcome.json"))
    destination = tmp_path / "other"
    destination.mkdir()
    try:
        monkeypatch.chdir(destination)
    finally:
        receiver.close()
    assert not (tmp_path / "outcome.json").exists()


# 功能：
#   验证从文件描述符构造流失败时，同时关闭原始描述符并清理本次未完成的描述文件。
# 输入：
#   tmp_path：独占描述目录。
#   monkeypatch：只注入 fdopen 失败，不影响操作系统关闭行为。
# 输出：
#   None：不返回业务数据。
def test_descriptor_initialization_failure_closes_raw_fd(tmp_path, monkeypatch):
    descriptor = []

    # 功能：
    #   捕获本次新建描述符后模拟流包装失败。
    # 输入：
    #   fd：接收器刚创建的原始文件描述符。
    #   args、kwargs：流包装参数。
    # 输出：
    #   None：不返回业务数据。
    def fail_open(fd, *args, **kwargs):
        descriptor.append(fd)
        raise OSError("TEST_FDOPEN_FAILURE")

    with monkeypatch.context() as patch:
        patch.setattr(channel.os, "fdopen", fail_open)
        with pytest.raises(OSError, match="TEST_FDOPEN_FAILURE"):
            channel.OutcomeReceiver(tmp_path / "outcome.json")
    assert len(descriptor) == 1
    try:
        with pytest.raises(OSError):
            os.fstat(descriptor[0])
    finally:
        # 旧实现会泄漏 fd；测试自身必须清理，不能把复现副作用留在后续用例中。
        with suppress(OSError):
            os.close(descriptor[0])
    assert not (tmp_path / "outcome.json").exists()
