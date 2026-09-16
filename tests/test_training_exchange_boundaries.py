"""Loopback learner protocol boundaries; all identities and packets are synthetic."""

import hmac
import io
import json
import struct
import time
from unittest.mock import Mock

import pytest
from test_training_policy_exchange import proposal

from dronedream_agent_core.training import policy_exchange as exchange


# 功能：
#   验证认证通过的消息仍需严格 JSON 校验，签名不能使重复键或非有限数值合法。
# 输入：
#   body：含歧义或非标准数字的消息体。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("body", [b'{"run_id":"old","run_id":"new"}', b'{"value":NaN}'])
def test_authenticated_packet_requires_strict_json(body):
    secret = bytes(32)
    packet = hmac.digest(secret, body, "sha256") + body
    payload = io.BytesIO(struct.pack("!I", len(packet)) + packet)
    stream = Mock(recv=payload.read)
    with pytest.raises(ValueError, match="PACKET_INVALID"):
        exchange._receive(stream, secret, deadline=time.monotonic() + 1)


# 功能：
#   验证发送器不会把非字符串字典键转换成字符串，或接受非对象消息。
# 输入：
#   value：不符合通信 JSON 契约的消息值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [{1: "ambiguous"}, ["not an object"]])
def test_sender_rejects_invalid_json_before_socket(value):
    stream = Mock()
    with pytest.raises(ValueError):
        exchange._send(stream, value, bytes(32), deadline=time.monotonic() + 1)
    stream.sendall.assert_not_called()


# 功能：
#   验证相同内容的新描述文件也不会在关闭旧交换器时被误删，清理必须核对文件身份。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_close_preserves_same_content_replacement(tmp_path):
    path = tmp_path / "learner.json"
    server = exchange.TrainingPolicyExchange(path)
    try:
        content = path.read_bytes()
        replacement = tmp_path / "replacement.json"
        replacement.write_bytes(content)
        replacement.replace(path)
    finally:
        server.close()
    assert path.read_bytes() == content


# 功能：
#   验证监听端口绑定失败后关闭已分配的套接字，不泄漏操作系统资源。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：注入绑定失败的夹具。
# 输出：
#   None：不返回业务数据。
def test_listener_bind_failure_closes_socket(tmp_path, monkeypatch):
    listener = Mock(bind=Mock(side_effect=OSError("synthetic bind failure")))
    monkeypatch.setattr(exchange.socket, "socket", Mock(return_value=listener))
    with pytest.raises(OSError, match="synthetic bind failure"):
        exchange.TrainingPolicyExchange(tmp_path / "not-created.json")
    listener.close.assert_called_once()
    assert not (tmp_path / "not-created.json").exists()


# 功能：
#   验证未校验复制得到的非法提案在发包前拒绝，并消耗当前请求、关闭连接。
# 输入：
#   monkeypatch：监视发送入口的夹具。
# 输出：
#   None：不返回业务数据。
def test_reply_revalidates_proposal_before_sending(monkeypatch):
    request = {"valid_until_unix_ms": int(time.time() * 1000) + 200}
    server = object.__new__(exchange.TrainingPolicyExchange)
    server._closed, server._request = False, request
    server._stream, server._secret = Mock(), bytes(32)
    server._deadline = time.monotonic() + .2
    stream = server._stream
    sender = Mock()
    monkeypatch.setattr(exchange, "_send", sender)
    invalid = proposal(request).model_copy(update={"policy_sha256": "invalid"})
    with pytest.raises(ValueError):
        server.reply(invalid)
    sender.assert_not_called()
    stream.close.assert_called_once()
    assert server._request is None and server._stream is None


# 功能：
#   验证本地描述文件重复的目标端口不能由最后一个有效值掩盖。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_descriptor_rejects_duplicate_endpoint(tmp_path):
    value = {"protocol": exchange.PROTOCOL, "host": "127.0.0.1", "port": 12345,
             "secret": "a" * 64, "run_id": "b" * 32}
    path = tmp_path / "descriptor.json"
    path.write_text('{"port":1,' + json.dumps(value)[1:], encoding="utf-8")
    with pytest.raises(ValueError, match="DESCRIPTOR_INVALID"):
        exchange.read_descriptor(path)


# 功能：
#   验证无效读取长度在访问套接字之前拒绝，不能产生空成功结果或非法 recv 参数。
# 输入：
#   count：非法的目标读取字节数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("count", [-1, True, 1.5, exchange.MAX_PACKET_BYTES + 1])
def test_read_count_rejected_before_socket(count):
    stream = Mock(recv=Mock(side_effect=AssertionError("invalid count reached socket")))
    with pytest.raises(ValueError, match="READ_SIZE"):
        exchange._read_exact(stream, count, deadline=time.monotonic() + 1)
    stream.recv.assert_not_called()


# 功能：
#   验证无效截止时刻不会传入系统超时接口或无限延长读取。
# 输入：
#   deadline：非法单调时钟截止值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf"), 10**400])
def test_nonfinite_deadline_rejected_before_socket(deadline):
    stream = Mock(recv=Mock(return_value=b"x"))
    with pytest.raises(ValueError, match="CLOCK"):
        exchange._read_exact(stream, 1, deadline=deadline)
    stream.settimeout.assert_not_called()


# 功能：
#   验证独占文件创建后文本句柄初始化失败时，关闭原始描述符并清理本次空文件。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：注入文本句柄初始化失败的夹具。
# 输出：
#   None：不返回业务数据。
def test_descriptor_wrapper_failure_releases_owned_resources(tmp_path, monkeypatch):
    path = tmp_path / "failed-descriptor.json"
    original_close = exchange.os.close
    closed = Mock(wraps=original_close)
    with monkeypatch.context() as patch:
        patch.setattr(exchange.os, "close", closed)
        patch.setattr(exchange.os, "fdopen", Mock(side_effect=OSError("synthetic fdopen failure")))
        with pytest.raises(OSError, match="synthetic fdopen failure"):
            exchange.TrainingPolicyExchange(path)
    closed.assert_called_once()
    assert not path.exists()
