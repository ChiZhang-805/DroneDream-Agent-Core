import hmac
import json
import time
from pathlib import Path

import pytest
from test_native_state_stream import packet

from dronedream_agent_core.native_state_channel import (
    MAX_PACKET_BYTES,
    NativeStatePublisher,
    NativeStateReceiver,
    _address,
)
from dronedream_agent_core.native_state_stream import NativeStateSampler


# 功能：
#   本机真实通信只返回最新状态，不改写来源时间，也不因轮询虚构新观测。
# 输入：
#   tmp_path：独立运行目录。
# 输出：
#   None：不返回业务数据。
def test_native_channel_delivers_latest_packet_without_rewriting_source_clocks(tmp_path):
    descriptor = tmp_path / "channel.json"
    receiver, publisher = NativeStateReceiver(descriptor), NativeStatePublisher(descriptor)
    try:
        for stamp in (1000, 1020, 1040):
            assert publisher.send(packet(stamp))
        latest = receiver.read_latest()
        assert latest == packet(1040)
        assert receiver.accepted == 3
        assert receiver.read_latest() is None  # Polling is not a new observation.
        latest["updated_at_unix_ms"] = 9999
        assert publisher.send(packet(1060))
        assert receiver.read_latest() == packet(1060)
    finally:
        publisher.close()
        receiver.close()
    assert not descriptor.exists()


# 功能：
#   拒绝认证、重放、时间倒退和协议错误，之后合法状态仍能被接收。
# 输入：
#   tmp_path：独立运行目录。
#   fault：本次报文故障类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["authentication", "replay", "clock", "schema", "sequence"])
def test_native_channel_rejects_tampering_replay_and_clock_regression(tmp_path, fault):
    receiver = NativeStateReceiver(tmp_path / "channel.json")
    publisher = NativeStatePublisher(receiver.path)
    try:
        publisher.send(packet(1000))
        assert receiver.read_latest() == packet(1000)
        envelope = {"sequence": 2, "payload": packet(1020)}
        if fault == "replay":
            envelope["sequence"] = 1
        elif fault == "clock":
            envelope["payload"] = packet(999)
        elif fault == "schema":
            envelope["payload"]["schema_version"] = "simulation-ground-truth"
        elif fault == "sequence":
            envelope["sequence"] = True
        body = json.dumps(envelope).encode()
        secret = b"wrong-key" if fault == "authentication" else receiver._secret
        raw = hmac.digest(secret, body, "sha256") + body
        publisher._socket.sendto(raw, _address(receiver._value))
        assert receiver.read_latest() is None
        assert receiver.rejected == 1
        # One malformed message cannot permanently suppress later authentic data.
        publisher._sequence = 3
        publisher.send(packet(1040))
        assert receiver.read_latest() == packet(1040)
    finally:
        publisher.close()
        receiver.close()


# 功能：
#   通信体积有界且首次绑定后不跨运行重新连接，端点不存在时直接丢弃。
# 输入：
#   tmp_path：独立运行目录。
# 输出：
#   None：不返回业务数据。
def test_native_channel_is_run_bound_and_bounded(tmp_path):
    descriptor = tmp_path / "channel.json"
    publisher = NativeStatePublisher(descriptor)
    assert publisher.send(packet()) is False
    receiver = NativeStateReceiver(descriptor)
    try:
        with pytest.raises(FileExistsError):
            NativeStateReceiver(descriptor)
        too_large = packet()
        too_large["padding"] = "x" * MAX_PACKET_BYTES
        with pytest.raises(ValueError, match="PACKET_SIZE"):
            publisher.send(too_large)
        receiver.close()
        replacement = NativeStateReceiver(descriptor)
        try:
            # Existing producers must not reconnect to an unrelated receiver.
            publisher.send(packet())  # UDP may accept a datagram with no receiver.
            assert replacement.read_latest() is None
        finally:
            replacement.close()
    finally:
        publisher.close()
        receiver.close()


# 功能：
#   原生状态采样从已绑定通道接收，不回退到磁盘旧快照，也不放宽来源期限。
# 输入：
#   tmp_path：独立运行目录。
#   monkeypatch：禁止采样期间文件读取的局部替换工具。
# 输出：
#   None：不返回业务数据。
def test_native_sampler_uses_channel_not_mutable_telemetry_file(tmp_path, monkeypatch):
    descriptor = tmp_path / "channel.json"
    sampler = NativeStateSampler(tmp_path / "must-not-read.json", channel_path=descriptor)
    publisher = NativeStatePublisher(descriptor)
    # Prime the descriptor before forbidding the telemetry reader's filesystem I/O.
    stamp = int(time.time() * 1000)
    publisher.send(packet(stamp))
    with monkeypatch.context() as patch:
        # 功能：
        #   将任何采样期文件回退转换为明确测试失败。
        # 输入：
        #   args、kwargs：文件打开参数。
        # 输出：
        #   None：不返回业务数据。
        def forbid_file(*args, **kwargs):
            raise AssertionError("Native stream cannot fall back to a disk snapshot")
        patch.setattr(Path, "open", forbid_file)
        sampler.start()
        try:
            deadline = time.monotonic() + 1
            while sampler.buffer._latest is None and time.monotonic() < deadline:
                time.sleep(.005)
            observation = sampler.buffer.latest(now_unix_ms=int(time.time() * 1000))
            assert observation.sample.observed_at_unix_ms == stamp
            assert observation.payload == packet(stamp)
            with pytest.raises(ValueError, match="EXPIRED"):
                sampler.buffer.latest(now_unix_ms=stamp + 1000)
        finally:
            sampler._stop.set()
            sampler._thread.join(timeout=1)
    sampler.close()
    publisher.close()


# 功能：
#   接收器关闭时不删除已被不同内容替换的端点描述符。
# 输入：
#   tmp_path：独立运行目录。
# 输出：
#   None：不返回业务数据。
def test_native_receiver_preserves_a_replaced_descriptor(tmp_path):
    receiver = NativeStateReceiver(tmp_path / "channel.json")
    receiver.path.write_text('{"different-owner":true}', encoding="utf-8")
    receiver.close()
    assert receiver.path.read_text(encoding="utf-8") == '{"different-owner":true}'


# 功能：
#   即使字节数合法且签名正确，过深 JSON 也应被拒绝且不终止后续采样。
# 输入：
#   tmp_path：独立运行目录。
# 输出：
#   None：不返回业务数据。
def test_deeply_nested_authenticated_packet_is_rejected_without_killing_receiver(tmp_path):
    receiver = NativeStateReceiver(tmp_path / "channel.json")
    publisher = NativeStatePublisher(receiver.path)
    try:
        body = b'[' * 2000 + b'0' + b']' * 2000
        raw = hmac.digest(receiver._secret, body, "sha256") + body
        assert len(raw) < MAX_PACKET_BYTES
        publisher._socket.sendto(raw, _address(receiver._value))
        assert receiver.read_latest() is None
        assert receiver.rejected == 1
        assert publisher.send(packet(1000))
        assert receiver.read_latest() == packet(1000)
    finally:
        publisher.close()
        receiver.close()


# 功能：
#   过深的替换描述符不能破坏关闭过程，也不能被旧所有者删除。
# 输入：
#   tmp_path：独立运行目录。
# 输出：
#   None：不返回业务数据。
def test_nested_replacement_descriptor_does_not_break_close_or_get_deleted(tmp_path):
    receiver = NativeStateReceiver(tmp_path / "channel.json")
    # It fits the descriptor byte budget, but exceeds CPython's nesting limit.
    body = '[' * 1010 + '0' + ']' * 1010
    receiver.path.write_text(body, encoding="utf-8")
    receiver.close()
    assert receiver.path.read_text(encoding="utf-8") == body
