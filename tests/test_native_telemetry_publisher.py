import asyncio
import copy
import threading
from types import SimpleNamespace

import pytest

from dronedream_agent_core.native_telemetry_publisher import NativeTelemetryPublisher


# 功能：发布频率必须有界且为真实数值，拒绝会造成忙循环或伪造配置的输入。
# 输入：rate：非法请求频率。
# 输出：构造时拒绝，不启动线程或订阅。
@pytest.mark.parametrize("rate", [None, True, "50", 0, -1, 19.9, 100.1, float("nan"),
                                  float("inf"), 10**1000])
def test_publication_rate_is_bounded(rate):
    with pytest.raises(ValueError, match="NATIVE_PUBLICATION_RATE_INVALID"):
        NativeTelemetryPublisher(object(), lambda *_: None, publication_rate_hz=rate)


# 功能：相机路径保留每份 50 Hz 原生里程计，默认路径仍为 20 Hz；重复缓存不续期。
# 输入：monkeypatch：只控制调度钟和线程执行；真实协议解析、来源历史和配对仍执行。
# 输出：慢位置缓存不限制独立里程计推进，选帧保留真实源时间和未知协方差。
@pytest.mark.parametrize("rate", [None, 50.0])
def test_camera_publication_preserves_source_cadence_and_duplicate_time(monkeypatch, rate):
    from test_native_image_pose import payload

    import dronedream_agent_core.native_telemetry_publisher as module
    from dronedream_agent_core.native_image_pose import NativeImagePoseBuffer

    elapsed = 0.0
    received = []
    pose_history = NativeImagePoseBuffer(clock_domain="camera-run")

    class Client:
        # 功能：位置来源比里程计慢，返回同一份真实位置缓存而不重新计时。
        # 输入：timeout：调用者保留的等待上限。
        # 输出：固定位置来源时间。
        async def sample_position_velocity_ned(self, timeout):
            assert timeout == .1
            return SimpleNamespace(received_at_unix_ms=1000)

        # 功能：每 20 ms 推进一份原始里程计，100 ms 后故意停流以检查重复缓存。
        # 输入：max_age：调用者保留的时效上限。
        # 输出：来源时间独立于发布时刻的原始协议包。
        def latest_dynamics_telemetry(self, max_age):
            assert max_age == .25
            offset = min(100, round(elapsed * 1000) // 20 * 20)
            return payload(1_000_000 + offset * 1000, 10.0 + offset / 1000,
                           1000 + offset)["dynamics"]

    # 功能：把发布得到的原始里程计送入正式源时钟配对缓存，不修改其时间或内容。
    # 输入：observed、dynamics：发布器独立复制的快照。
    # 输出：八次受控调度之后停止，保持测试与实际机器负载无关。
    def publish(observed, dynamics):
        assert observed.received_at_unix_ms == 1000
        received.append(copy.deepcopy(dynamics["sources"]["odometry"]))
        bound = payload()
        bound["dynamics"] = dynamics
        pose_history.ingest(bound, available_at_monotonic=10.0 + elapsed)
        if len(received) == 8:
            publisher._stop.set()

    # 功能：消除测试线程调度抖动，仍执行正式发布回调及全部协议校验。
    # 输入：callback、args：原线程调用及实参。
    # 输出：回调结果。
    async def inline(callback, *args):
        return callback(*args)

    # 功能：推进实际请求的等待间隔，不替换传感器时刻或续期旧观测。
    # 输入：awaitable、timeout：发布循环请求的停止等待。
    # 输出：等待超时，驱动下一个确定性发布周期。
    async def advance(awaitable, timeout):
        nonlocal elapsed
        awaitable.close()
        elapsed += timeout
        raise TimeoutError

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: elapsed))
    monkeypatch.setattr(module, "asyncio", SimpleNamespace(
        Event=asyncio.Event, to_thread=inline, wait_for=advance))
    options = {} if rate is None else {"publication_rate_hz": rate}
    publisher = NativeTelemetryPublisher(Client(), publish, **options)
    asyncio.run(publisher._run())
    expected = [1_000_000, 1_040_000, 1_100_000] if rate is None else list(
        range(1_000_000, 1_100_001, 20_000))
    assert sorted({row["timestamp_us"] for row in received}) == expected
    assert publisher.summary["target_rate_hz"] == (20.0 if rate is None else 50.0)
    assert publisher.summary["unique_source_count"] == len(expected)
    assert publisher.summary["published_count"] == 8
    assert publisher.summary["error_count"] == 0
    assert received[-1] == received[-2]
    assert received[-1]["received_at_unix_ms"] == 1100
    assert received[-1]["mavlink_source"]["received_monotonic_seconds"] == 10.1
    if rate is not None:
        for offset in range(10, 100, 20):
            aligned = pose_history.align(image_timestamp_ns=(1_000_000 + offset * 1000) * 1000,
                clock_domain="camera-run", image_received_unix_ms=1000 + offset,
                now_unix_ms=1140, now_monotonic=10.14,
                maximum_range_m=10., maximum_acceleration_mps2=1.)
            assert abs(aligned.source_skew_us) == 10_000
            assert aligned.source_evidence["pose_extrapolated"] is False
            assert aligned.source_evidence["received_at_unix_ms"] == 1000 + offset - 10


# 功能：
#   验证慢速线程写入不占用控制事件循环、不积压多个写入，重复缓存不虚增来源变化。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_slow_publication_does_not_block_control_event_loop_or_queue_writes():
    # 功能：
    #   在隔离事件循环中阻塞写线程，并同时运行独立节拍任务检查循环仍可前进。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        published = []

        class Client:
            # 功能：
            #   始终返回同一接收时间，模拟反复读取同一原生位置样本。
            # 输入：
            #   timeout：接口等待参数，本夹具立即返回。
            # 输出：
            #   observed：固定接收时间的位置替身。
            async def sample_position_velocity_ned(self, timeout):
                observed = SimpleNamespace(received_at_unix_ms=1000)
                return observed

            # 功能：
            #   提供固定来源标识的动力学缓存，不引入真实传感器订阅。
            # 输入：
            #   max_age：接口要求的最大样本年龄，本夹具无需据此刷新数据。
            # 输出：
            #   dynamics：含固定 IMU 时间标识的包。
            def latest_dynamics_telemetry(self, max_age):
                dynamics = {"sources": {"imu": {"timestamp_us": 7}}}
                return dynamics

        # 功能：
        #   阻塞写线程直到释放，并记录真正完成写入的样本时间。
        # 输入：
        #   observed：位置样本。
        #   dynamics：动力学快照。
        # 输出：
        #   None：不返回业务数据。
        def write(observed, dynamics):
            entered.set()
            assert release.wait(timeout=1)
            published.append(observed.received_at_unix_ms)

        publisher = NativeTelemetryPublisher(Client(), write)
        publisher.start()
        try:
            for _ in range(20):
                await asyncio.sleep(.005)
                if entered.is_set():
                    break
            assert entered.is_set()
            ticks = 0
            for _ in range(10):
                await asyncio.sleep(.005)
                ticks += 1
            assert ticks == 10
            assert published == []
            release.set()
            await asyncio.sleep(.06)
        finally:
            release.set()
            summary = await publisher.close()
        assert summary["drained"] is True
        assert summary["published_count"] >= 2
        assert summary["unique_source_count"] == 1
        assert set(published) == {1000}
        with pytest.raises(RuntimeError):
            publisher.start()

    asyncio.run(scenario())


# 功能：
#   验证缺少接收时间的位姿只产生错误计数，不能被当作新鲜状态发布。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_missing_timestamp_is_not_published_as_fresh():
    # 功能：
    #   运行错误样本发布循环并检查停止后的计数与错误原因。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        class Client:
            # 功能：
            #   返回没有时间标识的对象，模拟不完整位置缓存。
            # 输入：
            #   timeout：接口等待参数。
            # 输出：
            #   observed：缺失接收时间的位置替身。
            async def sample_position_velocity_ned(self, timeout):
                observed = SimpleNamespace()
                return observed

            # 功能：
            #   提供空但结构合法的来源字典，以隔离位置时间缺失这一错误。
            # 输入：
            #   max_age：接口要求的最大样本年龄。
            # 输出：
            #   dynamics：没有来源条目的动力学包。
            def latest_dynamics_telemetry(self, max_age):
                dynamics = {"sources": {}}
                return dynamics

        publisher = NativeTelemetryPublisher(Client(), lambda *args: pytest.fail("fake sample"))
        publisher.start()
        await asyncio.sleep(.025)
        summary = await publisher.close()
        assert summary["published_count"] == 0
        assert summary["unique_source_count"] == 0
        assert summary["error_count"] > 0
        assert "RECEIVE_TIME_MISSING" in summary["last_issue"]

    asyncio.run(scenario())


# 功能：验证定位发布器能先于连接启动，缺失期间不伪造状态，连接后自动使用真实缓存。
# 输入：无；异步夹具控制连接何时可用。
# 输出：启动失败不泄漏任务、不续期时间，恢复后只有真实来源被发布。
def test_preflight_publisher_waits_for_connection_without_fake_state():
    # 功能：运行一次先启动发布、后到达连接的有界生命周期。
    # 输入：无。
    # 输出：完整回收及原始来源时间保持断言。
    async def scenario():
        connected = False
        published = []

        class Client:
            # 功能：连接前报告缺失，连接后返回原始缓存时间而不是当前时刻。
            # 输入：timeout：采样等待预算。
            # 输出：真实缓存的替身或明确缺失异常。
            async def sample_position_velocity_ned(self, timeout):
                if not connected:
                    raise RuntimeError('not connected')
                return SimpleNamespace(received_at_unix_ms=1234)

            # 功能：提供固定来源标识，不创建新的订阅。
            # 输入：max_age：消费者期限。
            # 输出：隔离测试来源。
            def latest_dynamics_telemetry(self, max_age):
                return {'sources': {'imu': {'timestamp_us': 99}}}

        publisher = NativeTelemetryPublisher(Client(), lambda sample, _: published.append(
            sample.received_at_unix_ms))
        publisher.start()
        try:
            await asyncio.sleep(.04)
            assert not published
            assert publisher.summary['error_count'] > 0
            connected = True
            for _ in range(50):
                if publisher.summary['published_count']:
                    break
                await asyncio.sleep(.01)
            assert published and set(published) == {1234}
            assert publisher.summary['last_issue'] is None
        finally:
            summary = await publisher.close()
        assert summary['drained']

    asyncio.run(scenario())


# 功能：
#   验证慢速证据写入与快速状态通道相互独立，且待落盘数据只保留最新快照。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_slow_persistence_does_not_throttle_fast_native_state_channel():
    # 功能：
    #   阻塞首次证据写入，同时持续产生不同来源时间并统计快速发布次数。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        fast_samples, persisted_samples = [], []
        sample_number = 0

        class Client:
            # 功能：生成具有单调接收时间的位置样本，模拟持续更新的飞控缓存。
            # 输入：timeout：接口等待参数，本夹具立即返回。
            # 输出：observed：本轮位置样本。
            async def sample_position_velocity_ned(self, timeout):
                nonlocal sample_number
                sample_number += 1
                return SimpleNamespace(received_at_unix_ms=1000 + sample_number)

            # 功能：返回结构合法且时间随位置样本变化的动力学来源。
            # 输入：max_age：消费者允许的最大年龄。
            # 输出：dynamics：本轮动力学快照。
            def latest_dynamics_telemetry(self, max_age):
                return {"sources": {"imu": {"timestamp_us": sample_number}}}

        # 功能：记录控制通道实际取得的每个位置时间。
        # 输入：observed、dynamics：本次对齐后的状态。
        # 输出：None。
        def publish(observed, dynamics):
            fast_samples.append(observed.received_at_unix_ms)

        # 功能：阻塞首次证据写入，释放后记录真正落盘的合并快照。
        # 输入：observed、dynamics：本次待持久化状态。
        # 输出：None。
        def persist(observed, dynamics):
            entered.set()
            assert release.wait(timeout=2)
            persisted_samples.append(observed.received_at_unix_ms)

        publisher = NativeTelemetryPublisher(
            Client(), publish, persist=persist, persist_interval_seconds=.01
        )
        publisher.start()
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            baseline = len(fast_samples)
            for _ in range(100):
                if len(fast_samples) >= baseline + 3:
                    break
                await asyncio.sleep(.01)
            # 证据线程仍被阻塞时，20 Hz 快通道必须继续前进。
            assert len(fast_samples) >= baseline + 3
            assert persisted_samples == []
        finally:
            release.set()
            summary = await publisher.close()
        assert summary["drained"] is True
        assert summary["published_count"] == len(fast_samples)
        assert summary["persisted_count"] >= 1
        assert persisted_samples[-1] == fast_samples[-1]
        assert len(persisted_samples) < len(fast_samples)

    asyncio.run(scenario())
