import asyncio
import threading
from types import SimpleNamespace

import pytest

from dronedream_agent_core.native_telemetry_publisher import NativeTelemetryPublisher


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
