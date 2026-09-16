"""Real loopback transport checks, isolated from accounts and flight hardware."""

import hmac
import json

import pytest

from dronedream_agent_core import local_packet_channel as channel

CONTRACT = channel.PacketContract("unit-state", "unit-state-", "unit-state-schema")


# 功能：
#   构造带原始更新时间的单个测试状态，不包含账户或外部密钥。
# 输入：
#   stamp：本次状态的毫秒时间。
# 输出：
#   payload：协议测试状态。
def _payload(stamp=1000):
    payload = {"schema_version": CONTRACT.payload_schema, "updated_at_unix_ms": stamp}
    return payload


# 功能：
#   已认证报文仍需严格解析，不接受重复键、非有限数或重复嵌套身份。
# 输入：
#   tmp_path：运行专属描述符目录。
#   kind：坏报文内容类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["duplicate", "nonfinite", "nested-duplicate"])
def test_authenticated_json_is_unambiguous(tmp_path, kind):
    receiver = channel.LatestPacketReceiver(tmp_path / "channel.json", contract=CONTRACT)
    publisher = channel.LatestPacketPublisher(receiver.path, contract=CONTRACT)
    try:
        body = json.dumps({"sequence": 1, "payload": _payload()}).encode()
        if kind == "duplicate":
            body = body.replace(b'"sequence": 1', b'"sequence": 2, "sequence": 1')
        elif kind == "nonfinite":
            body = body.replace(b'"payload": {', b'"payload": {"value":NaN,')
        else:
            body = body.replace(b'"payload": {', b'"payload": {"state":0,"state":1,')
        packet = hmac.digest(receiver._secret, body, "sha256") + body
        publisher._socket.sendto(packet, channel._address(receiver._value))
        assert receiver.read_latest() is None
        assert receiver.rejected == 1
        assert publisher.send(_payload())
        assert receiver.read_latest() == _payload()
    finally:
        publisher.close()
        receiver.close()


# 功能：
#   描述符重复身份字段不能按最后一项静默覆盖。
# 输入：
#   tmp_path：运行专属描述符目录。
# 输出：
#   None：不返回业务数据。
def test_descriptor_duplicate_keys_rejected(tmp_path):
    receiver = channel.LatestPacketReceiver(tmp_path / "channel.json", contract=CONTRACT)
    try:
        raw = receiver.path.read_text()
        receiver.path.write_text(raw.replace('{', '{"protocol":"wrong",', 1))
        with pytest.raises(ValueError):
            channel.read_descriptor(receiver.path, CONTRACT)
    finally:
        receiver.close()
    assert receiver.path.exists()


# 功能：
#   相同内容但不同文件身份的描述符归新所有者，旧接收器关闭时不能删除。
# 输入：
#   tmp_path：运行专属描述符目录。
# 输出：
#   None：不返回业务数据。
def test_close_preserves_replaced_identical_descriptor(tmp_path):
    receiver = channel.LatestPacketReceiver(tmp_path / "channel.json", contract=CONTRACT)
    raw = receiver.path.read_bytes()
    receiver.path.rename(tmp_path / "original.json")
    receiver.path.write_bytes(raw)
    receiver.close()
    assert receiver.path.read_bytes() == raw


# 功能：
#   超大数据报只增加拒绝计数，不令 Windows 原生 recvfrom 的截断错误终止接收循环。
# 输入：
#   tmp_path：运行专属描述符目录。
# 输出：
#   None：不返回业务数据。
def test_oversized_datagram_does_not_break_receiver(tmp_path):
    receiver = channel.LatestPacketReceiver(tmp_path / "channel.json", contract=CONTRACT)
    publisher = channel.LatestPacketPublisher(receiver.path, contract=CONTRACT)
    try:
        publisher._socket.sendto(b"x" * (channel.MAX_PACKET_BYTES + 100),
                                 channel._address(receiver._value))
        assert receiver.read_latest() is None
        assert receiver.rejected == 1
        assert publisher.send(_payload())
        assert receiver.read_latest() == _payload()
    finally:
        publisher.close()
        receiver.close()


# 功能：
#   发布端拒绝会在 JSON 字符串化后发生键冲突的对象。
# 输入：
#   tmp_path：运行专属描述符目录。
# 输出：
#   None：不返回业务数据。
def test_publisher_rejects_ambiguous_nested_keys(tmp_path):
    receiver = channel.LatestPacketReceiver(tmp_path / "channel.json", contract=CONTRACT)
    publisher = channel.LatestPacketPublisher(receiver.path, contract=CONTRACT)
    try:
        with pytest.raises(ValueError):
            publisher.send({**_payload(), "nested": {1: "first", "1": "second"}})
        assert receiver.read_latest() is None
    finally:
        publisher.close()
        receiver.close()


# 功能：
#   发布描述符持久化失败必须关闭套接字且不能留下半份可绑定端点。
# 输入：
#   tmp_path、monkeypatch：隔离目录与磁盘故障注入工具。
# 输出：
#   None：不返回业务数据。
def test_descriptor_fsync_failure_does_not_publish(tmp_path, monkeypatch):
    # 功能：
    #   模拟本次暂存描述符刷新磁盘失败。
    # 输入：
    #   descriptor：要刷新的文件描述符。
    # 输出：
    #   None：不返回业务数据。
    def fail_sync(descriptor):
        raise OSError("injected sync failure")

    monkeypatch.setattr(channel.os, "fsync", fail_sync)
    path = tmp_path / "channel.json"
    with pytest.raises(OSError):
        channel.LatestPacketReceiver(path, contract=CONTRACT)
    assert not path.exists()
