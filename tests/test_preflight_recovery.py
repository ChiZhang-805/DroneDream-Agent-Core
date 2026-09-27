"""Fault-injected startup checks; these fixtures do not qualify physical flight."""

import asyncio
from types import SimpleNamespace

import pytest
from test_native_perception_preflight import health
from test_px4_base_telemetry import _base_module

from dronedream_agent_core.native_preflight import (
    NativePerceptionReadiness,
    wait_for_native_perception,
)
from dronedream_agent_core.preflight_recovery import run_preflight_recovery
from dronedream_agent_core.sensor_diagnostics import sensor_issue_code, sensor_issue_codes


# 功能：
#   验证临时失败先清理再重试，实际成功结果和每次失败均可追溯。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_supervisor_recovers_in_order():
    # 功能：
    #   在独立事件循环中检查恢复顺序。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        events, evidence = [], {}

        # 功能：
        #   在首轮模拟传输失败，第二轮返回测试结果。
        # 输入：
        #   remaining、row：剩余预算和本轮证据。
        # 输出：
        #   result：明确的测试就绪标记。
        async def attempt(remaining, row):
            events.append("attempt")
            assert 0 < remaining <= 2
            if row["attempt"] == 1:
                raise ConnectionError()
            result = "ready-fixture"
            return result

        # 功能：
        #   记录测试清理已完成。
        # 输入：
        #   无。
        # 输出：
        #   None：不返回业务数据。
        async def recover():
            events.append("cleanup")

        result = await run_preflight_recovery(attempt=attempt, recover=recover,
            retryable=lambda error, row: isinstance(error, ConnectionError),
            motion_requested=lambda: False, abort_check=lambda: None, evidence=evidence,
            timeout_seconds=2)
        assert result == "ready-fixture"
        assert events == ["attempt", "cleanup", "attempt"]
        assert evidence["attempts"][0]["cleanup"] == "closed_and_reaped"
        assert evidence["status"] == "ready"

    asyncio.run(scenario())


# 功能：
#   真实客户端只接受飞控明确未解锁状态，所有分支都关闭检查订阅。
# 输入：
#   armed：飞控状态夹具，只有 False 可接受。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("armed", [False, True, None, 0, "false"])
def test_actual_disarmed_check_is_strict_and_releases_subscription(armed):
    base = _base_module()

    # 功能：
    #   在独立事件循环内检查物理解锁标志与订阅回收。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        closed = []

        # 功能：
        #   提供单个解锁状态并记录实际关闭。
        # 输入：
        #   无。
        # 输出：
        #   armed：调用方传入的测试值。
        async def stream():
            try:
                yield armed
            finally:
                closed.append(True)

        client = object.__new__(base.MavsdkOffboardClient)
        client._flight_command_requested = False
        client._system = SimpleNamespace(telemetry=SimpleNamespace(armed=stream))
        if armed is False:
            assert (await client.verify_disarmed_before_preflight())["disarmed"] is True
        else:
            with pytest.raises(RuntimeError, match="ALREADY_ARMED_OR_STATE_INVALID"):
                await client.verify_disarmed_before_preflight()
        assert closed == [True]

    asyncio.run(scenario())


# 功能：
#   1. 验证首条解锁遥测晚于旧的两秒限制时，仍可在总准备期限内通过。
#   2. 验证传给真实检查的预算来自剩余准备时间，且通过后没有请求飞行。
# 输入：
#   tmp_path：本次测试日志目录。
# 输出：
#   None：断言冷启动等待和禁止提前解锁的约束。
def test_disarmed_cold_start_uses_remaining_preflight_budget(tmp_path):
    base = _base_module()
    closed = []

    # 功能：
    #   模拟 MAVLink 首条状态延迟到达，不模拟实际飞行或感知结果。
    # 输入：
    #   无。
    # 输出：
    #   armed：明确的未解锁状态 False。
    async def stream():
        try:
            await asyncio.sleep(2.1)
            armed = False
            yield armed
        finally:
            closed.append(True)

    class Client(base.FakeOffboardClient):
        verify_disarmed_before_preflight = (
            base.MavsdkOffboardClient.verify_disarmed_before_preflight)

        # 功能：
        #   为真实检查方法提供本测试的异步遥测接口。
        # 输入：
        #   self：本测试客户端。
        # 输出：
        #   system：仅提供解锁状态的夹具。
        def _require_system(self):
            system = SimpleNamespace(telemetry=SimpleNamespace(armed=stream))
            return system

    client, evidence = Client(), {}
    client._flight_command_requested = False
    health_result = asyncio.run(base.connect_preflight_with_recovery(client,
        connection="test-only", readiness_timeout_seconds=5., abort_check=lambda: None,
        log_path=tmp_path / "log", evidence=evidence))
    assert health_result.armable is True
    assert 2. < evidence["attempts"][0]["disarmed_timeout_seconds"] <= 5.
    assert evidence["attempts"][0]["disarmed_check"]["disarmed"] is True
    assert len(evidence["attempts"]) == 1
    assert client._flight_command_requested is False
    assert closed == [True]


# 功能：
#   首帧永远不到或调用被取消时，检查必须退出并回收订阅，不能生成通过回执。
# 输入：
#   cancel：是否通过外部取消中止，而非等待检查超时。
# 输出：
#   None：断言失败类型、明确超时代码和订阅回收结果。
@pytest.mark.parametrize("cancel", [False, True])
def test_disarmed_wait_timeout_and_cancellation_release_subscription(cancel):
    base = _base_module()

    # 功能：
    #   在隔离事件循环中注入未到达的解锁遥测。
    # 输入：
    #   无。
    # 输出：
    #   None：执行失败与回收断言。
    async def scenario():
        started, closed = asyncio.Event(), []

        # 功能：
        #   阻塞首条状态并记录订阅是否被实际关闭。
        # 输入：
        #   无。
        # 输出：
        #   armed：不会在本测试期限内产生的状态。
        async def stream():
            try:
                started.set()
                await asyncio.Event().wait()
                armed = False
                yield armed
            finally:
                closed.append(True)

        client = object.__new__(base.MavsdkOffboardClient)
        client._flight_command_requested = False
        client._system = SimpleNamespace(telemetry=SimpleNamespace(armed=stream))
        task = asyncio.create_task(client.verify_disarmed_before_preflight(
            timeout_seconds=1. if cancel else .03))
        await started.wait()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(TimeoutError, match="PREFLIGHT_DISARMED_TELEMETRY_TIMEOUT"):
                await task
        assert closed == [True]
        assert client._flight_command_requested is False

    asyncio.run(scenario())


# 功能：
#   无效等待参数必须在订阅遥测之前失败，不能产生无界等待。
# 输入：
#   timeout：非法的准备预算。
# 输出：
#   None：断言参数被拒绝。
@pytest.mark.parametrize("timeout", [0., -1., float("nan"), float("inf"), True])
def test_disarmed_check_rejects_invalid_budget_before_subscribing(timeout):
    base = _base_module()
    client = object.__new__(base.MavsdkOffboardClient)
    client._flight_command_requested = False
    with pytest.raises((ValueError, TypeError)):
        asyncio.run(client.verify_disarmed_before_preflight(timeout_seconds=timeout))


# 功能：
#   解锁回执丢失后保留命令尝试锁，不能再次运行起飞前连接恢复。
# 输入：
#   tmp_path：私有测试日志目录。
# 输出：
#   None：不返回业务数据。
def test_lost_arm_ack_never_reopens_preflight_recovery(tmp_path):
    base = _base_module()

    # 功能：
    #   模拟解锁命令被发送但回执超时。
    # 输入：
    #   无。
    # 输出：
    #   None：本夹具只抛超时。
    async def lose_ack():
        raise TimeoutError("ack fixture")

    # 功能：
    #   验证命令尝试先于回执记录，恢复入口立即拒绝。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        client = object.__new__(base.MavsdkOffboardClient)
        client._flight_command_requested = False
        client._system = SimpleNamespace(action=SimpleNamespace(arm=lose_ack))
        with pytest.raises(TimeoutError):
            await client.arm()
        assert client._flight_command_requested is True
        evidence = {}
        with pytest.raises(RuntimeError, match="FORBIDDEN_AFTER_MOTION_REQUEST"):
            await base.connect_preflight_with_recovery(client, connection="never-connect",
                readiness_timeout_seconds=1, abort_check=lambda: None,
                log_path=tmp_path / "log", evidence=evidence)
        assert evidence["attempts"] == []

    asyncio.run(scenario())


# 功能：
#   解锁尝试、永久故障、清理失败、超时和取消不得触发第二次连接。
# 输入：
#   fault：注入的中止条件。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", [
    "armed", "arm-during-attempt", "permanent", "cleanup", "timeout", "cancel",
])
def test_supervisor_cannot_retry_unsafe_or_unbounded_work(fault):
    # 功能：
    #   独立执行一组故障注入并核对尝试次数。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        calls, evidence = [], {}
        motion = [fault == "armed"]

        # 功能：
        #   注入预定失败，不操作任何飞控。
        # 输入：
        #   remaining、row：本轮准备参数。
        # 输出：
        #   None：本夹具只抛异常。
        async def attempt(remaining, row):
            calls.append("attempt")
            if fault == "timeout":
                await asyncio.sleep(10)
            if fault == "cancel":
                raise asyncio.CancelledError()
            if fault == "arm-during-attempt":
                motion[0] = True
            raise ConnectionError()

        # 功能：
        #   清理错误时拒绝产生新的连接。
        # 输入：
        #   无。
        # 输出：
        #   None：本夹具只抛异常。
        async def recover():
            calls.append("cleanup")
            raise RuntimeError("cleanup failure fixture")

        with pytest.raises((RuntimeError, TimeoutError, ConnectionError, asyncio.CancelledError)):
            await run_preflight_recovery(attempt=attempt, recover=recover,
                retryable=lambda error, row: fault != "permanent",
                motion_requested=lambda: motion[0], abort_check=lambda: None, evidence=evidence,
                timeout_seconds=.03 if fault == "timeout" else 1)
        assert calls.count("attempt") <= 1
        if fault in {"armed", "arm-during-attempt", "permanent", "cancel"}:
            assert "cleanup" not in calls
        assert evidence["status"] != "ready"

    asyncio.run(scenario())


# 功能：
#   持续健康需要完整时间窗口，断流后重新累计，单次好帧或重复广播不能取得资格。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_stable_window_restarts_without_extending_sensor_ttl():
    gate = NativePerceptionReadiness(stable_window_ms=1000, require_source_timestamps=True)
    for index in range(5):
        stamp = 1000 + 100*index
        packet = {**health(index+1, stamp), "perception_observed_at_unix_ms": stamp}
        assert not gate.observe(packet, now_unix_ms=stamp)
    assert not gate.observe({**packet, "stream_healthy": False}, now_unix_ms=1450)
    assert gate.independent_frames == 0
    for index in range(11):
        stamp = 1500 + 100*index
        packet = {**health(index+6, stamp), "perception_observed_at_unix_ms": stamp}
        assert gate.observe(packet, now_unix_ms=stamp) is (index == 10)
    assert not gate.observe(packet, now_unix_ms=2751)
    assert gate.independent_frames == 0


# 功能：
#   重新发布时间不能使过期或未来深度取得起飞就绪资格。
# 输入：
#   depth_time：待验证的深度来源时刻。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("depth_time", [None, True, 700, 1001, "1000"])
def test_strict_gate_rejects_invalid_depth_source(depth_time):
    gate = NativePerceptionReadiness(require_source_timestamps=True)
    assert not gate.observe({**health(), "perception_observed_at_unix_ms": depth_time},
                            now_unix_ms=1000)
    assert gate.last_issue == "NATIVE_PERCEPTION_DEPTH_SOURCE_INVALID_OR_EXPIRED"


# 功能：
#   保留有限诊断类别但不泄漏异常正文；缺失来源即使超时也保存故障计数。
# 输入：
#   tmp_path：本测试不存在的健康文件所在目录。
# 输出：
#   None：不返回业务数据。
def test_diagnostics_survive_failed_readiness_without_raw_error_leak(tmp_path):
    assert sensor_issue_code(ValueError("native IMU source has expired")) == "NATIVE_IMU_EXPIRED"
    assert sensor_issue_code(ValueError("private token fixture")) == "ValueError"
    assert sensor_issue_code(ValueError("NATIVE_ODOMETRY_RESET_WAITING_FOR_COHERENT_STATE")) == (
        "NATIVE_ODOMETRY_RESET_WAITING_FOR_COHERENT_STATE")
    issue = "ValueError:NATIVE_STATE_STREAM_UNAVAILABLE:SOURCE_SAMPLE_EXPIRED"
    assert sensor_issue_code(issue) == "SOURCE_SAMPLE_EXPIRED"
    assert len(sensor_issue_codes(["arbitrary"+str(i) for i in range(10000)])) <= 4
    evidence = {}
    with pytest.raises(RuntimeError, match="NOT_READY"):
        asyncio.run(wait_for_native_perception(tmp_path / "missing", timeout_seconds=.02,
                                               evidence=evidence))
    assert evidence["status"] == "timeout"
    assert evidence["last_issue"] in {
        "NATIVE_PERCEPTION_NOT_RECEIVED", "NATIVE_PERCEPTION_READ_TimeoutError",
        "NATIVE_PERCEPTION_READ_DEADLINE_EXCEEDED",
    }
    assert sum(evidence["issue_counts"].values()) > 0


# 功能：
#   必需遥测请求失败可以重试，不支持或矛盾回执不允许假报就绪。
# 输入：
#   tmp_path：本轮隔离日志目录。
#   state：第一轮采样率请求的测试状态。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("state", ["failed", "unsupported", "lying-success"])
def test_transport_checks_actual_required_rate_receipts(tmp_path, state):
    base = _base_module()

    class Client(base.FakeOffboardClient):
        calls = 0

        # 功能：
        #   返回有明确来源的速率请求夹具，第二轮模拟恢复。
        # 输入：
        #   self：本测试客户端。
        # 输出：
        #   receipt：逐来源请求回执，不代表物理传感器采样。
        async def configure_dynamics_telemetry_rates(self):
            self.calls += 1
            sources = {key: {"status": "requested"} for key in
                       ("position_velocity", "imu", "attitude", "odometry", "battery")}
            if self.calls == 1:
                sources["imu"]["status"] = "failed" if state == "lying-success" else state
            receipt = {"sources": sources,
                "required_rate_requests_succeeded": self.calls > 1 or state == "lying-success"}
            return receipt

    client, evidence = Client(), {}
    call = base.connect_preflight_with_recovery(client, connection="test-only",
        readiness_timeout_seconds=2, abort_check=lambda: None, log_path=tmp_path / "log",
        evidence=evidence)
    if state == "unsupported":
        with pytest.raises(RuntimeError, match="UNSUPPORTED"):
            asyncio.run(call)
        assert client.calls == 1
    else:
        assert asyncio.run(call).armable
        assert client.calls == 2


# 功能：证明视觉定位依赖的遥测先于就绪等待启动，且不跳过原有飞控就绪检查。
# 输入：tmp_path：隔离日志路径；就绪替身仅在获得采样率请求后返回。
# 输出：顺序断言以及真实执行过的就绪检查回执，不请求解锁。
def test_sensor_rates_precede_localization_readiness(tmp_path):
    base = _base_module()
    order = []

    class Client(base.FakeOffboardClient):
        # 功能：记录遥测配置阶段，提供完整且成功的独立请求回执。
        # 输入：无。
        # 输出：按来源列出的模拟速率请求。
        async def configure_dynamics_telemetry_rates(self):
            order.append('rates')
            return {'sources': {key: {'status': 'requested'} for key in
                    ('position_velocity', 'imu', 'attitude', 'odometry', 'battery')},
                    'required_rate_requests_succeeded': True}

        # 功能：检测原有循环依赖；没有遥测请求则拒绝模拟定位就绪。
        # 输入：timeout_seconds：继承的有界准备期限。
        # 输出：父类就绪状态。
        async def wait_until_ready(self, timeout_seconds):
            assert order == ['rates']
            order.append('readiness')
            return await super().wait_until_ready(timeout_seconds)

    result = asyncio.run(base.connect_preflight_with_recovery(Client(), connection='test-only',
        readiness_timeout_seconds=2, abort_check=lambda: None, log_path=tmp_path/'log',
        evidence={}))
    assert result.armable
    assert order == ['rates', 'readiness']


# 功能：本地模式准备只能显式选择，只返回原始健康，不伪造可解锁；默认路径仍拒绝。
# 输入：只有本地定位有效的客户端与临时日志。
# 输出：准备阶段可以返回 armable=False，默认完整准备仍失败，且没有控制命令。
def test_local_preparation_is_explicit_and_does_not_claim_armability(tmp_path):
    base = _base_module()
    native = base.TelemetryHealth(True, False, True, True, False)

    class Client(base.FakeOffboardClient):
        # 功能：提供模式准备前的真实式状态；输入：期限；输出：不可解锁的健康。
        async def wait_until_local_position_ready(self, timeout_seconds):
            return native

        # 功能：模拟原有完整准备；输入：期限；输出：同一不可解锁状态供检查器拒绝。
        async def wait_until_ready(self, timeout_seconds):
            return native

    common = dict(connection='test-only', readiness_timeout_seconds=2,
                  abort_check=lambda: None, log_path=tmp_path/'log')
    evidence = {}
    result = asyncio.run(base.connect_preflight_with_recovery(Client(), **common,
        evidence=evidence, local_mode_preparation_only=True))
    assert result is native and result.armable is False
    assert evidence['attempts'][0]['local_mode_preparation_only'] is True
    with pytest.raises(RuntimeError, match='FIRMWARE_NOT_READY'):
        asyncio.run(base.connect_preflight_with_recovery(Client(), **common, evidence={}))


# 功能：
#   真正的 MAVSDK 客户端订阅在停流后自行恢复，旧值撤销且最多一个订阅存活。
# 输入：
#   monkeypatch：缩短测试停流等待，不改变产品默认超时。
# 输出：
#   None：不返回业务数据。
def test_position_subscription_automatically_recovers_without_duplicate_streams(monkeypatch):
    base = _base_module()
    monkeypatch.setitem(base.DYNAMICS_STREAM_SAMPLE_TIMEOUT_SECONDS, "position_velocity", .02)

    # 功能：
    #   运行真实客户端的采样协程，核对中断及恢复后的唯一订阅。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        release, recovered = asyncio.Event(), asyncio.Event()

        class Telemetry:
            calls = active = maximum_active = 0

            # 功能：
            #   第一次订阅阻塞，下一次交付真实异步夹具样本，并跟踪资源回收顺序。
            # 输入：
            #   self：测试遥测接口。
            # 输出：
            #   sample：合成位置和速度，不用于飞行资格证据。
            async def position_velocity_ned(self):
                self.calls += 1
                self.active += 1
                self.maximum_active = max(self.active, self.maximum_active)
                try:
                    if self.calls == 1:
                        await release.wait()
                    sample = SimpleNamespace(
                        position=SimpleNamespace(north_m=1., east_m=2., down_m=-1.),
                        velocity=SimpleNamespace(north_m_s=0., east_m_s=0., down_m_s=0.))
                    recovered.set()
                    yield sample
                    await release.wait()
                finally:
                    self.active -= 1

        client = object.__new__(base.MavsdkOffboardClient)
        telemetry = Telemetry()
        client._system = SimpleNamespace(telemetry=telemetry)
        client._position_velocity_condition = asyncio.Condition()
        client._position_velocity_sample = None
        client._position_velocity_error = None
        client._dynamics_errors = {}
        client._dynamics_restart_counts = {}
        task = asyncio.create_task(client._collect_position_velocity_ned())
        client._position_velocity_task = task
        try:
            await asyncio.wait_for(recovered.wait(), 1.)
            sample = await client.sample_position_velocity_ned(.2)
            assert sample.north_m == 1.
            assert telemetry.calls == 2 and telemetry.maximum_active == 1
            assert client._position_velocity_error is None
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert telemetry.active == 0

    asyncio.run(scenario())
