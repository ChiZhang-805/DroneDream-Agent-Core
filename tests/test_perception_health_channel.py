import asyncio
import time

import pytest
from test_native_perception_preflight import health
from test_perception_lifecycle import health as completion_health
from test_perception_lifecycle import timing

from dronedream_agent_core import native_preflight
from dronedream_agent_core.perception_health_channel import (
    PerceptionHealthPublisher,
    PerceptionHealthReceiver,
)
from dronedream_agent_core.perception_lifecycle import (
    capture_control_completion,
    verify_control_completion,
)


# 功能：
#   有界等待实际收包线程交付新包，不以固定睡眠推断网络已经送达。
# 输入：
#   receiver：本次测试的实际健康接收端。
# 输出：
#   packet：唯一消费的新包，超时使测试失败。
def receive(receiver):
    deadline = time.monotonic() + 1.
    while time.monotonic() < deadline:
        packet = receiver.read_latest()
        if packet is not None:
            return packet
        time.sleep(.002)
    pytest.fail("health packet was not received")


# 功能：
#   验证真实本机报文保留来源时间、裁掉无关诊断数组，并且重复轮询不是新观测。
# 输入：
#   tmp_path：独立运行目录。
# 输出：
#   None：不返回业务数据。
def test_health_channel_projects_only_original_readiness_fields(tmp_path):
    receiver = PerceptionHealthReceiver(tmp_path / "endpoint.json")
    publisher = PerceptionHealthPublisher(receiver.path)
    try:
        packet = health()
        assert publisher.send({**packet, "diagnostic_array": ["unused"] * 100_000})
        assert receive(receiver) == packet
        assert receiver.read_latest() is None
        assert publisher.send({**packet, "stream_healthy": False, "updated_at_unix_ms": 1020})
        assert receive(receiver)["stream_healthy"] is False
    finally:
        publisher.close()
        receiver.close()
    assert not receiver.path.exists()


# 功能：
#   在飞行消费端不读取期间持续排空实际通道；终点冻结新状态，不从堵塞队列或磁盘借旧健康。
# 输入：
#   tmp_path：独立证据和通信目录。
# 输出：
#   None：不返回业务数据。
def test_completion_uses_current_received_health_without_queued_preflight_or_disk(tmp_path):
    receiver = PerceptionHealthReceiver(tmp_path / "endpoint.json")
    publisher = PerceptionHealthPublisher(receiver.path)
    try:
        for sequence in range(1, 101):
            packet = {**completion_health(),
                      "schema_version": "dronedream.perception-fusion-health.v1",
                      "latest_sequence": sequence, "updated_at_unix_ms": 1000 + sequence}
            assert publisher.send(packet)
            time.sleep(.004)
        deadline = time.monotonic() + 1.
        while time.monotonic() < deadline:
            latest = receiver.latest_snapshot()
            if latest is not None and latest["latest_sequence"] == 100:
                break
            time.sleep(.002)
        assert latest["latest_sequence"] == 100
        record = capture_control_completion(tmp_path / "missing.json", receiver=receiver,
            track_sha256="a" * 64, completed_at_unix_ms=1110)
        assert verify_control_completion(timing(record))
        assert record["health"]["updated_at_unix_ms"] == 1100
        assert publisher.dropped == 0
        # 后来的不健康状态不能回写此前已冻结的记录，也不能倒退到磁盘好帧。
        receive(receiver)
        publisher.send({**packet, "stream_healthy": False, "updated_at_unix_ms": 1120})
        assert not receive(receiver)["stream_healthy"]
        rejected = capture_control_completion(tmp_path / "missing.json", receiver=receiver,
            track_sha256="a" * 64, completed_at_unix_ms=1130)
        assert not rejected["accepted"]
        assert verify_control_completion(timing(record))
    finally:
        publisher.close()
        receiver.close()


# 功能：
#   未收到、过期或未来报文不能借健康磁盘文件补出结束证明，接收关闭后也不能复用缓存。
# 输入：
#   tmp_path：独立证据目录。
# 输出：
#   None：不返回业务数据。
def test_completion_channel_does_not_fall_back_or_renew_source(tmp_path):
    import json

    path = tmp_path / "health.json"
    path.write_text(json.dumps(completion_health()))
    receiver = PerceptionHealthReceiver(tmp_path / "endpoint.json")
    publisher = PerceptionHealthPublisher(receiver.path)
    try:
        assert not capture_control_completion(path, receiver=receiver,
            track_sha256="a" * 64, completed_at_unix_ms=1010)["accepted"]
        publisher.send({**completion_health(),
                        "schema_version": "dronedream.perception-fusion-health.v1"})
        receive(receiver)
        for completed in (999, 1201, 1500):
            assert not capture_control_completion(path, receiver=receiver,
                track_sha256="a" * 64, completed_at_unix_ms=completed)["accepted"]
        receiver.close()
        assert not capture_control_completion(path, receiver=receiver,
            track_sha256="a" * 64, completed_at_unix_ms=1010)["accepted"]
    finally:
        publisher.close()
        receiver.close()


# 功能：
#   使用真实套接字验证就绪检查不访问磁盘快照，不因报文运输延长原始有效期。
# 输入：
#   tmp_path：独立运行目录。
#   monkeypatch：把磁盘读取替换成明确失败，防止隐式回退。
# 输出：
#   None：不返回业务数据。
def test_preflight_channel_accepts_live_frames_without_disk_reads(tmp_path, monkeypatch):
    # 功能：
    #   将不应发生的磁盘回退转换为测试失败。
    # 输入：
    #   path：不应被读取的健康文件路径。
    # 输出：
    #   无：总是抛出异常。
    def forbidden_read(path):
        raise AssertionError("readiness must not fall back to disk")

    monkeypatch.setattr(native_preflight, "_read_native_health", forbidden_read)

    # 功能：
    #   并发运行真实发布与就绪检查，结束时回收所有通信资源。
    # 输入：
    #   无：使用测试目录与当前读取替身。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        receiver = PerceptionHealthReceiver(tmp_path / "endpoint.json")
        publisher = PerceptionHealthPublisher(receiver.path)

        # 功能：
        #   发布七个来源推进的合成定位样本，仅验证传输与就绪逻辑。
        # 输入：
        #   无：使用当前发布端。
        # 输出：
        #   None：不返回业务数据。
        async def produce():
            for sequence in range(1, 8):
                now = int(time.time() * 1000)
                publisher.send({**health(sequence, now), "localization_covariance_m2": .001,
                                "localization_observed_at_unix_ms": now - 10})
                await asyncio.sleep(.06)

        writer = asyncio.create_task(produce())
        try:
            result = await native_preflight.wait_for_native_perception(
                tmp_path / "absent-health.json", receiver=receiver,
                minimum_route_clearance_m=1., timeout_seconds=1.)
            assert result["transport"] == "run-scoped-datagram"
            assert result["independent_frames"] >= 3
            assert result["valid_until_unix_ms"] == result["source_observed_at_unix_ms"] + 250
            native_preflight.assert_native_preflight_current(
                result, now_unix_ms=int(time.time() * 1000))
            assert receiver.path.exists()  # 所有者在飞行资源回收阶段关闭，不阻塞解锁。
        finally:
            await writer
            publisher.close()
            receiver.close()

    asyncio.run(scenario())


# 功能：
#   单份健康缓存或后续不健康包均不能凑足独立帧，也不能借磁盘健康记录取得许可。
# 输入：
#   tmp_path：独立运行目录。
#   unhealthy：是否在健康包之后发出一份不健康包。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("unhealthy", [False, True])
def test_channel_does_not_promote_a_repeated_or_revoked_sample(tmp_path, unhealthy):
    receiver = PerceptionHealthReceiver(tmp_path / "endpoint.json")
    publisher = PerceptionHealthPublisher(receiver.path)
    try:
        now = int(time.time() * 1000)
        publisher.send(health(1, now))
        if unhealthy:
            publisher.send({**health(2, now), "stream_healthy": False})
        with pytest.raises(RuntimeError, match="NOT_READY_BEFORE_ARM"):
            asyncio.run(native_preflight.wait_for_native_perception(
                tmp_path / "health.json", receiver=receiver, timeout_seconds=.35))
    finally:
        publisher.close()
        receiver.close()
