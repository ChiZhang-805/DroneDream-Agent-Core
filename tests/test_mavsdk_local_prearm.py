"""Real adapter, fake transport: distinct local preparation and firmware armability."""

import asyncio
import time
from types import SimpleNamespace

import pytest
from test_px4_base_telemetry import _base_module

from dronedream_agent_core.flight_command_cleanup import FlightCommandAttempts


# 功能：运行真实客户端的模式准备路径，用无解锁接口的替身注入原生遥测与命令回执。
# 输入：故障类别和 monkeypatch；所有坐标是测试测量，不是真实飞行证据。
# 输出：原地保持、模式所有权及故障退出断言。
@pytest.mark.parametrize('failure', [None, 'stale', 'mode', 'health', 'armed'])
def test_adapter_uses_only_measured_hold_and_actual_mode_health(monkeypatch, failure):
    base = _base_module()

    async def scenario():
        events = []
        in_offboard = False

        class Telemetry:
            # 功能：提供解锁状态；输入：测试故障；输出：明确布尔值。
            async def armed(self):
                yield failure == 'armed'

            # 功能：提供当前模式的健康；输入：模式状态；输出：不伪造全球定位。
            async def health(self):
                yield SimpleNamespace(is_home_position_ok=True, is_local_position_ok=True,
                                      is_global_position_ok=False,
                                      is_armable=in_offboard and failure != 'health')

            # 功能：提供实际模式观察；输入：测试状态；输出：模式枚举等价物。
            async def flight_mode(self):
                yield SimpleNamespace(name='OFFBOARD' if in_offboard else 'HOLD')

        class Core:
            # 功能：提供连接状态；输入：无；输出：已连接遥测。
            async def connection_state(self):
                yield SimpleNamespace(is_connected=True)

        class Offboard:
            # 功能：记录实际下发保持；输入：原生坐标；输出：空回执。
            async def set_position_ned(self, hold):
                events.append(('hold', hold))

            # 功能：注入模式回执丢失；输入：无；输出：成功或超时，但均保留尝试。
            async def start(self):
                nonlocal in_offboard
                events.append(('start', None))
                in_offboard = True
                if failure == 'mode':
                    raise TimeoutError('mode-ack-lost')

            # 功能：退出模式；输入：无；输出：明确退出记录。
            async def stop(self):
                nonlocal in_offboard
                events.append(('stop', None))
                in_offboard = False

        client = object.__new__(base.MavsdkOffboardClient)
        client._system = SimpleNamespace(telemetry=Telemetry(), core=Core(), offboard=Offboard())
        client._position_cls = lambda *values: values
        client._offboard_error_cls = RuntimeError
        client._flight_command_requested = False

        # 功能：模拟已校验失联参数；输入：当前客户端；输出：记录准备顺序。
        async def failsafe(instance):
            assert instance is client
            events.append(('failsafe', None))
            return {'verified': True}

        # 功能：提供带真实接收时间格式的位置；输入：过期故障；输出：测量样本。
        async def position(timeout_seconds):
            stamp = int(time.time()*1000) - (1000 if failure == 'stale' else 0)
            return base.PositionVelocityNed(1., 2., -3., 0., 0., 0., stamp)

        # 功能：提供测量航向；输入：期限；输出：度数。
        async def heading(timeout_seconds):
            return 42.

        monkeypatch.setattr(base, 'configure_offboard_loss_failsafe', failsafe)
        client.sample_position_velocity_ned = position
        client.sample_heading_deg = heading
        attempts, evidence = FlightCommandAttempts(), {}
        call = client.prepare_local_offboard_mode(command_attempts=attempts, evidence=evidence)
        if failure:
            with pytest.raises((RuntimeError, TimeoutError)):
                await call
        else:
            health = await call
            assert health.armable and not health.global_position_ok
            assert evidence['handshake']['still_disarmed']
        assert not attempts.arm_requested
        assert all(hold == (1., 2., -3., 42.) for event, hold in events if event == 'hold')
        assert attempts.offboard_requested == (failure not in ('stale', 'armed'))
        assert sum(event == 'stop' for event, _ in events) == (failure in ('mode', 'health'))
        if failure != 'armed':
            assert events[0][0] == 'failsafe'

    asyncio.run(scenario())


# 功能：验证进入本地模式的第二次检查也使用准备预算，不退回隐蔽的两秒首帧限制。
# 输入：monkeypatch 和迟到/未知/超时的原生状态夹具；输出：未收到 False 前不发送模式请求。
@pytest.mark.parametrize('total_budget', [1., 10., 60.])
def test_local_mode_disarmed_wait_uses_bounded_preparation_budget(total_budget):
    base = _base_module()

    # 功能：在接触任何保持、模式或解锁接口前截获实际检查预算。
    # 输入：本次总预算；输出：真实方法传递预算和故障回执断言。
    async def scenario():
        client = object.__new__(base.MavsdkOffboardClient)
        client._system = SimpleNamespace()
        evidence, attempts, budgets = {}, FlightCommandAttempts(), []

        # 功能：模拟尚无明确未解锁状态；输入：检查预算；输出：故障，绝不生成通过回执。
        async def unavailable(*, timeout_seconds):
            budgets.append(timeout_seconds)
            raise TimeoutError('PREFLIGHT_DISARMED_TELEMETRY_TIMEOUT')

        client.verify_disarmed_before_preflight = unavailable
        with pytest.raises(TimeoutError, match='PREFLIGHT_DISARMED_TELEMETRY_TIMEOUT'):
            await client.prepare_local_offboard_mode(command_attempts=attempts,
                evidence=evidence, timeout_seconds=total_budget)
        assert budgets == [min(30., total_budget)]
        assert evidence['disarmed_timeout_seconds'] == min(30., total_budget)
        assert 'disarmed_check' not in evidence
        assert not attempts.offboard_requested and not attempts.arm_requested

    asyncio.run(scenario())
