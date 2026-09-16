"""Native publication ownership and shutdown, using only local threads and fake caches."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from dronedream_agent_core import native_telemetry_publisher as publication


# 功能：
#   建立复用可变缓存的假客户端，以验证发布器是否持有自己的输入副本。
# 输入：
#   无。
# 输出：
#   fixture：客户端、位置缓存与动力学缓存。
def client_fixture():
    observed = SimpleNamespace(received_at_unix_ms=1000, north_m=1.)
    dynamics = {"sources": {"imu": {"timestamp_us": 7, "values": [1., 2.]}}}

    # 功能：
    #   返回同一个位置缓存，模拟调用者不转交独占所有权的适配器。
    # 输入：
    #   timeout：采样等待参数，本夹具立即返回。
    # 输出：
    #   observed：共享位置缓存。
    async def sample(timeout):
        return observed

    client = SimpleNamespace(sample_position_velocity_ned=sample,
                             latest_dynamics_telemetry=lambda max_age: dynamics)
    fixture = client, observed, dynamics
    return fixture


# 功能：
#   验证停止后不会重新启动，未启动实例关闭时可明确报告没有待排空写入。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_close_before_start_is_drained_and_cannot_restart():
    # 功能：
    #   在独立事件循环内完成未启动发布器的关闭及重启拒绝检查。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        client, _, _ = client_fixture()
        publisher = publication.NativeTelemetryPublisher(client, lambda *args: None)
        summary = await publisher.close()
        assert summary["drained"] is True
        with pytest.raises(RuntimeError, match="closed"):
            publisher.start()

    asyncio.run(scenario())


# 功能：
#   验证关闭等待超时不取消正在等待后台写入的所有者，写入结束后可以再次排空。
# 输入：
#   monkeypatch：临时缩短发布器关闭等待预算。
# 输出：
#   None：不返回业务数据。
def test_close_timeout_preserves_inflight_write_for_later_drain(monkeypatch):
    monkeypatch.setattr(publication, "CLOSE_TIMEOUT_SECONDS", .01, raising=False)

    # 功能：
    #   用真实阻塞写入覆盖第一次关闭超时与第二次关闭成功两条路径。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        client, _, _ = client_fixture()

        # 功能：
        #   持续占用写入线程直到显式释放，不执行真实文件发布。
        # 输入：
        #   observed：该次位置样本。
        #   dynamics：该次动力学快照。
        # 输出：
        #   None：不返回业务数据。
        def write(observed, dynamics):
            entered.set()
            assert release.wait(3)

        publisher = publication.NativeTelemetryPublisher(client, write)
        publisher.start()
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            first = await publisher.close()
            assert first["drained"] is False
            assert not publisher._task.cancelled()
        finally:
            release.set()
        await asyncio.sleep(.02)
        final = await publisher.close()
        assert final["drained"] is True
        assert final["published_count"] == 1

    asyncio.run(scenario())


# 功能：
#   验证线程处理期间调用方修改原缓存不会改变本次发布内容或反向污染调用方。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_publish_owns_nested_snapshot_before_crossing_thread_boundary():
    # 功能：
    #   在写入开始后修改源对象，并核对写入函数仍收到修改前的完整状态。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        client, observed, dynamics = client_fixture()
        written = []

        # 功能：
        #   等待原缓存已被改变后再读取发布参数，复现跨线程共享可变对象的风险。
        # 输入：
        #   position：发布器传来的位置对象。
        #   packet：发布器传来的动力学对象。
        # 输出：
        #   None：不返回业务数据。
        def write(position, packet):
            entered.set()
            assert release.wait(1)
            written.append((position.north_m, packet["sources"]["imu"]["values"][:]))
            packet["sources"]["imu"]["values"].append(9.)

        publisher = publication.NativeTelemetryPublisher(client, write)
        publisher.start()
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            observed.north_m = 4.
            dynamics["sources"]["imu"]["values"][0] = 5.
            publisher._stop.set()
        finally:
            release.set()
            summary = await publisher.close()
        assert summary["drained"] is True
        assert written == [(1., [1., 2.])]
        assert dynamics["sources"]["imu"]["values"] == [5., 2.]

    asyncio.run(scenario())


# 功能：
#   验证无效接收时间及嵌套来源类型不能被发布为原生观测。
# 输入：
#   mutation：需要污染的位置时间或来源字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["negative_time", "bad_battery", "boolean_clock"])
def test_invalid_source_identity_is_rejected_before_publication(mutation):
    # 功能：
    #   等待发布循环实际处理错误输入，检查失败被计数且没有写出。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        client, observed, dynamics = client_fixture()
        if mutation == "negative_time":
            observed.received_at_unix_ms = -1
        elif mutation == "bad_battery":
            dynamics["sources"]["battery"] = [1]
        else:
            dynamics["sources"]["imu"]["timestamp_us"] = True
        written = []
        publisher = publication.NativeTelemetryPublisher(client, lambda *args: written.append(args))
        publisher.start()
        await asyncio.sleep(.025)
        summary = await publisher.close()
        assert summary["error_count"] > 0
        assert not written and summary["published_count"] == 0

    asyncio.run(scenario())


# 功能：
#   验证调用方取消关闭等待不会取消后台写入，随后仍能取得真实排空结果。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cancelled_close_keeps_owner_of_inflight_write():
    # 功能：
    #   独立取消 close 任务，同时保留并最终完成发布线程与发布器任务。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        client, _, _ = client_fixture()

        # 功能：
        #   模拟已开始但未完成的线程写入。
        # 输入：
        #   observed：位置快照。
        #   dynamics：动力学快照。
        # 输出：
        #   None：不返回业务数据。
        def write(observed, dynamics):
            entered.set()
            assert release.wait(1)

        publisher = publication.NativeTelemetryPublisher(client, write)
        publisher.start()
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            closer = asyncio.create_task(publisher.close())
            await asyncio.sleep(0)
            closer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closer
            assert not publisher._task.done()
        finally:
            release.set()
            summary = await publisher.close()
        assert summary["drained"] is True
        assert summary["published_count"] == 1

    asyncio.run(scenario())


# 功能：
#   验证同步发布回调意外返回协程时不计入成功，协程不会被偷偷执行。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_async_result_is_not_counted_as_published():
    executed = []

    # 功能：
    #   构造本不该由同步发布器接收的异步写入。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def async_write():
        executed.append(True)

    # 功能：
    #   运行返回协程的同步包装函数，核对只有错误计数增长。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        client, _, _ = client_fixture()
        publisher = publication.NativeTelemetryPublisher(client, lambda *args: async_write())
        publisher.start()
        await asyncio.sleep(.025)
        summary = await publisher.close()
        assert summary["published_count"] == 0 and summary["error_count"] > 0
        assert "RETURNED_AWAITABLE" in summary["last_issue"]

    asyncio.run(scenario())
    assert not executed


# 功能：
#   验证未知的普通发布异常只影响本次写入，下一次缓存发布仍可执行。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_unexpected_callback_exception_is_recorded_and_next_write_recovers():
    # 功能：
    #   首次发布抛出字典错误，等待第二次真正写入后检查成功与失败分别计数。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        client, _, _ = client_fixture()
        completed = threading.Event()
        attempts = []

        # 功能：
        #   模拟一次不在旧异常白名单中的发布失败，之后正常完成。
        # 输入：
        #   observed：位置快照。
        #   dynamics：动力学快照。
        # 输出：
        #   None：不返回业务数据。
        def write(observed, dynamics):
            attempts.append(True)
            if len(attempts) == 1:
                raise KeyError("missing metadata")
            completed.set()

        publisher = publication.NativeTelemetryPublisher(client, write)
        publisher.start()
        try:
            assert await asyncio.to_thread(completed.wait, 1)
        finally:
            summary = await publisher.close()
        assert summary["drained"] is True
        assert summary["error_count"] == 1 and summary["published_count"] >= 1

    asyncio.run(scenario())


# 功能：
#   验证电池等非姿态来源的时间更新也进入来源变化统计，不只读取三个预选来源。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_battery_source_changes_are_not_hidden_by_fixed_position_identity():
    # 功能：
    #   在位置和 IMU 时间固定时只推进电池接收时间，核对相邻来源变化数。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        client, _, packet = client_fixture()
        packet["sources"]["battery"] = {"received_at_unix_ms": 1000}
        reads = []
        finished = threading.Event()

        # 功能：
        #   每次读取仅更新电池缓存的原始接收时间，不更改位置或 IMU 身份。
        # 输入：
        #   max_age：接口的样本年龄上限。
        # 输出：
        #   packet：含本次电池时间的共享缓存。
        def latest(max_age):
            reads.append(True)
            packet["sources"]["battery"]["received_at_unix_ms"] = 1000 + len(reads)
            return packet

        # 功能：
        #   至少两次发布完成后通知测试，可据此结束发布循环。
        # 输入：
        #   observed：固定位置快照。
        #   dynamics：本次动力学副本。
        # 输出：
        #   None：不返回业务数据。
        def write(observed, dynamics):
            if dynamics["sources"]["battery"]["received_at_unix_ms"] >= 1002:
                finished.set()

        client.latest_dynamics_telemetry = latest
        publisher = publication.NativeTelemetryPublisher(client, write)
        publisher.start()
        try:
            assert await asyncio.to_thread(finished.wait, 1)
        finally:
            summary = await publisher.close()
        assert summary["published_count"] >= 2
        assert summary["unique_source_count"] == summary["published_count"]

    asyncio.run(scenario())
