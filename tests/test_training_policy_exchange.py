"""Actual loopback protocol tests, not evidence of physical flight success."""

import json
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.flight_environment import PilotAction
from dronedream_agent_core.training.policy_exchange import (
    MAX_PACKET_BYTES,
    TrainingPolicyClient,
    TrainingPolicyExchange,
    TrainingProposal,
    _read_exact,
    read_descriptor,
)


# 功能：
#   创建大于常规 UDP 报文的测试请求，并保留二百四十五毫秒的原始有效期。
# 输入：
#   无：使用当前墙钟和固定合成观测内容。
# 输出：
#   value：带明确截止时间的大消息测试请求。
def request():
    value = {"valid_until_unix_ms": int(time.time() * 1000) + 245,
            "observation": "s" * 60000}  # Larger than a UDP datagram.
    return value


# 功能：
#   构造绑定指定请求摘要的四轴合成操纵提案，不形成实际飞行授权。
# 输入：
#   value：需要回复的测试请求。
# 输出：
#   result：带合成策略摘要和操纵角色的结构化提案。
def proposal(value):
    result = TrainingProposal(request_sha256=sha256_json(value), policy_sha256="a" * 64,
                            expert_role="local-navigation-policy",
                            action=PilotAction(mode="pilot-control", axes=[.1, -.2, .3, -.4]))
    return result


# 功能：
#   验证实际回环 TCP 可以交换大消息和正确提案，并在幂等关闭时清理本次描述文件。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_real_exchange_large_frame_action_identity_and_descriptor_cleanup(tmp_path):
    path = tmp_path / "learner.json"
    server = TrainingPolicyExchange(path)
    try:
        client = TrainingPolicyClient(path)
        value = request()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(client.propose, value)
            received = server.wait_request(timeout_seconds=1)
            assert received == value
            server.reply(proposal(received))
            assert pending.result(timeout=1) == proposal(value)
    finally:
        server.close()
    server.close()
    assert not path.exists()


# 功能：
#   验证学习器不能用属于另一份观测的提案回复当前请求，拒绝后对端连接结束。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_learner_reply_cannot_address_different_observation(tmp_path):
    server = TrainingPolicyExchange(tmp_path / "learner.json")
    try:
        client = TrainingPolicyClient(server.path)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(client.propose, request())
            server.wait_request(timeout_seconds=1)
            wrong = proposal({"different": True})
            with pytest.raises(ValueError, match="PROPOSAL_OBSERVATION_MISMATCH"):
                server.reply(wrong)
            with pytest.raises(ConnectionError):
                pending.result(timeout=1)
    finally:
        server.close()


# 功能：
#   验证学习器回复超过原始时效后双方均报告超时，不因等待结果而续期动作。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_expired_reply_does_not_renew_action_lease(tmp_path):
    server = TrainingPolicyExchange(tmp_path / "learner.json")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            client = TrainingPolicyClient(server.path)
            pending = pool.submit(client.propose, request())
            value = server.wait_request(timeout_seconds=1)
            time.sleep(.27)
            with pytest.raises(TimeoutError):
                server.reply(proposal(value))
            with pytest.raises(TimeoutError):
                pending.result(timeout=1)
    finally:
        server.close()


# 功能：
#   验证超出预算的帧头和伪造认证消息会被服务端拒绝。
# 输入：
#   tmp_path：测试独立目录。
#   packet：非法的原始网络字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("packet", [struct.pack("!I", MAX_PACKET_BYTES + 1),
                                   struct.pack("!I", 34) + b"x" * 34])
def test_oversized_and_unauthenticated_packets_are_rejected(tmp_path, packet):
    server = TrainingPolicyExchange(tmp_path / "learner.json")
    try:
        binding = read_descriptor(server.path)
        with socket.create_connection(("127.0.0.1", binding["port"])) as stream:
            stream.sendall(packet)
            with pytest.raises(ValueError):
                server.wait_request(timeout_seconds=1)
    finally:
        server.close()


# 功能：
#   验证运行描述不能被第二个服务端覆盖，内容已改写的文件也不能被旧服务端清理。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_descriptor_cannot_be_overwritten_or_deleted_after_replacement(tmp_path):
    path = tmp_path / "learner.json"
    server = TrainingPolicyExchange(path)
    try:
        with pytest.raises(FileExistsError):
            TrainingPolicyExchange(path)
        changed = read_descriptor(path)
        changed["run_id"] = "f" * 32
        path.write_text(json.dumps(changed))
    finally:
        server.close()
    assert path.exists()


# 功能：
#   验证逐字节到达的消息遵守整帧总期限，不能靠分片持续刷新单次 socket 超时。
# 输入：
#   无：测试自行创建实际回环连接和慢速发送线程。
# 输出：
#   None：不返回业务数据。
def test_fragmented_read_obeys_total_deadline():
    # Each byte arrives within a socket timeout, but the entire frame is late.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    sender = socket.create_connection(listener.getsockname())
    receiver, _ = listener.accept()
    # 功能：
    #   以固定间隔发送测试字节，连接被接收方关闭后结束发送。
    # 输入：
    #   无：使用外层持有的发送套接字。
    # 输出：
    #   None：不返回业务数据。
    def trickle():
        for _ in range(12):
            try:
                sender.sendall(b"x")
                time.sleep(.015)
            except OSError:
                break
    thread = threading.Thread(target=trickle)
    try:
        thread.start()
        with pytest.raises(TimeoutError):
            _read_exact(receiver, 12, deadline=time.monotonic() + .045)
    finally:
        receiver.close()
        sender.close()
        listener.close()
        thread.join(timeout=1)


# 功能：
#   验证请求散列和复制耗时不会加回有效期，且学习器拿到的副本不能修改服务端留存请求。
# 输入：
#   monkeypatch：提供受控时钟和消息入口的夹具。
# 输出：
#   None：不返回业务数据。
def test_request_hashing_cannot_add_processing_time_back_to_lease(monkeypatch):
    from dronedream_agent_core.training import policy_exchange as module

    now = [10.]
    stream = SimpleNamespace(setsockopt=lambda *_: None, close=lambda: None)
    server = object.__new__(TrainingPolicyExchange)
    server._closed, server._stream, server._request = False, None, None
    server._value, server._secret, server._seen = {"run_id": "run"}, b"secret", set()
    server._listener = SimpleNamespace(settimeout=lambda *_: None,
                                      accept=lambda: (stream, ("127.0.0.1", 8000)))
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: 1., monotonic=lambda: now[0]))
    packet = {"valid_until_unix_ms": 1200, "observation": "original"}
    monkeypatch.setattr(module, "_receive", lambda *_args, **_kwargs:
                        {"run_id": "run", "request": packet})

    # 功能：
    #   在计算真实摘要前推进八十毫秒，模拟消息准备消耗的原始期限。
    # 输入：
    #   value：需要散列的测试请求。
    # 输出：
    #   digest：真实请求内容摘要。
    def delayed_hash(value):
        now[0] += .08
        digest = sha256_json(value)
        return digest

    monkeypatch.setattr(module, "sha256_json", delayed_hash)
    received = server.wait_request(timeout_seconds=1)
    assert received == packet
    assert server._deadline == pytest.approx(10.2)
    received["observation"] = "caller modification"
    assert server._request["observation"] == "original"


# 功能：
#   验证看起来够长但含空白的十六进制文本不能冒充三十二字节运行密钥。
# 输入：
#   tmp_path：测试独立目录。
#   secret：非法测试密钥文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("secret", [" " * 64, "aa " * 21 + " "])
def test_descriptor_requires_32_actual_secret_bytes(tmp_path, secret):
    path = tmp_path / "learner.json"
    path.write_text(json.dumps({
        "protocol": "dronedream-simulation-policy-exchange", "host": "127.0.0.1",
        "port": 12345, "secret": secret, "run_id": "a" * 32,
    }))
    with pytest.raises(ValueError, match="DESCRIPTOR_INVALID"):
        read_descriptor(path)


# 功能：
#   验证等待连接预算在调用套接字前检查，禁止布尔值、非有限值和巨大整数。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：阻止非法值进入系统接口的夹具。
#   seconds：非法等待秒值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("seconds", [True, float("nan"), float("inf"), 10**400])
def test_wait_budget_is_validated_before_accept(tmp_path, monkeypatch, seconds):
    server = TrainingPolicyExchange(tmp_path / "learner.json")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(server, "_listener", SimpleNamespace(
                settimeout=lambda *_: pytest.fail("invalid budget reached socket")))
            with pytest.raises(ValueError, match="WAIT_INVALID"):
                server.wait_request(timeout_seconds=seconds)
    finally:
        server.close()


# 功能：
#   验证处理消息期间已过期的观测不会交给学习器，不保存待回复状态或重放标记。
# 输入：
#   monkeypatch：注入受控时间消耗和连接替身的夹具。
# 输出：
#   None：不返回业务数据。
def test_processing_expiry_cannot_deliver_an_observation_to_learner(monkeypatch):
    from dronedream_agent_core.training import policy_exchange as module

    clock, closed = [10.], []
    stream = SimpleNamespace(setsockopt=lambda *_: None, close=lambda: closed.append(True))
    server = object.__new__(TrainingPolicyExchange)
    server._closed, server._stream, server._request = False, None, None
    server._value, server._secret, server._seen = {"run_id": "run"}, b"secret", set()
    server._listener = SimpleNamespace(settimeout=lambda *_: None,
                                      accept=lambda: (stream, ("127.0.0.1", 8000)))
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: 1.,
                                                        monotonic=lambda: clock[0]))
    monkeypatch.setattr(module, "_receive", lambda *a, **k: {
        "run_id": "run", "request": {"valid_until_unix_ms": 1200}})

    # 功能：
    #   在散列请求时推进三百毫秒，模拟准备时间超过观测期限。
    # 输入：
    #   value：待散列的测试请求。
    # 输出：
    #   digest：实际请求摘要。
    def delayed_hash(value):
        clock[0] += .3
        digest = sha256_json(value)
        return digest

    monkeypatch.setattr(module, "sha256_json", delayed_hash)
    with pytest.raises(ValueError, match="EXPIRED_OR_REPLAYED"):
        server.wait_request(timeout_seconds=1)
    assert closed and server._request is None and not server._seen
