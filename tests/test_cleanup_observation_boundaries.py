"""Fault injection for shutdown; no connection to a simulator or aircraft."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dronedream_agent_core.flight_command_cleanup import (
    FlightCommandAttempts,
    cleanup_flight_commands,
)


# 功能：
#   构造独立的命令客户端及本次清理参数，不连接飞控。
# 输入：
#   无。
# 输出：
#   arguments：客户端、尝试记录、落地观察与清理回执参数。
def cleanup_arguments():
    aircraft = SimpleNamespace(
        stop_offboard=AsyncMock(), land=AsyncMock(),
        wait_until_landed=AsyncMock(return_value={"state": "ON_GROUND", "confirmed": True}),
    )
    arguments = dict(client=aircraft,
                     attempts=FlightCommandAttempts(arm_requested=True, offboard_requested=True),
                     offboard_stopped=False, landed=False, landing_timeout_seconds=0.1,
                     on_landing=Mock(), evidence={})
    return arguments


# 功能：
#   验证阶段记录回调被取消也不能阻断后续降落尝试。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cancelled_phase_callback_cannot_prevent_landing():
    arguments = cleanup_arguments()
    arguments["on_landing"].side_effect = asyncio.CancelledError()
    asyncio.run(cleanup_flight_commands(**arguments))
    arguments["client"].land.assert_awaited_once()
    assert "CancelledError" in arguments["evidence"]["landing_phase_write"]
    assert arguments["evidence"]["landing_observation"]["confirmed"] is True


# 功能：
#   验证降落响应丢失时仍独立等待原生落地状态，保留命令失败而不伪造成功。
# 输入：
#   failure：降落发送等待期间出现的异常。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failure", [TimeoutError("lost reply"), asyncio.CancelledError()])
def test_lost_land_ack_does_not_skip_native_confirmation(failure):
    arguments = cleanup_arguments()
    arguments["client"].land.side_effect = failure
    asyncio.run(cleanup_flight_commands(**arguments))
    arguments["client"].wait_until_landed.assert_awaited_once_with(0.1)
    assert "failed:" in arguments["evidence"]["land_command"]
    assert arguments["evidence"]["land"] == "confirmed_on_ground_during_failure_cleanup"


# 功能：
#   验证本次未确认落地时不能沿用调用前残留的确认记录。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_new_observation_drops_previous_landing_confirmation():
    arguments = cleanup_arguments()
    arguments["evidence"]["landing_observation"] = {"state": "ON_GROUND", "confirmed": True}
    arguments["client"].wait_until_landed.side_effect = TimeoutError("no fresh telemetry")
    asyncio.run(cleanup_flight_commands(**arguments))
    assert "landing_observation" not in arguments["evidence"]
    assert arguments["evidence"]["land"].startswith("failed:")


# 功能：
#   验证停止命令卡住后在自有限时内退出等待，并继续降落清理。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_stuck_stop_request_cannot_block_landing_forever():
    arguments = cleanup_arguments()

    # 功能：
    #   模拟可被取消但一直没有回执的停止请求。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def stuck():
        await asyncio.Event().wait()

    arguments["client"].stop_offboard.side_effect = stuck
    asyncio.run(cleanup_flight_commands(**arguments, command_timeout_seconds=0.01))
    arguments["client"].land.assert_awaited_once()
    assert "TimeoutError" in arguments["evidence"]["stop_offboard"]


# 功能：
#   验证非布尔的真值不能充当已停控、已落地证据并跳过安全清理。
# 输入：
#   value：被污染的状态标志。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", ["false", 1, [True]])
def test_only_literal_true_may_skip_cleanup(value):
    arguments = cleanup_arguments()
    arguments.update(offboard_stopped=value, landed=value)
    asyncio.run(cleanup_flight_commands(**arguments))
    arguments["client"].stop_offboard.assert_awaited_once()
    arguments["client"].land.assert_awaited_once()


# 功能：
#   验证降落命令卡住也会结束等待，并由独立原生观察判断是否已经落地。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_stuck_land_request_still_reads_native_state():
    arguments = cleanup_arguments()

    # 功能：
    #   模拟降落请求一直等待响应，但原生状态通道仍能独立返回。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def stuck():
        await asyncio.Event().wait()

    arguments["client"].land.side_effect = stuck
    asyncio.run(cleanup_flight_commands(**arguments, command_timeout_seconds=0.01))
    assert "TimeoutError" in arguments["evidence"]["land_command"]
    assert arguments["evidence"]["landing_observation"]["confirmed"] is True


# 功能：
#   验证不实现内部限时的观察适配器也受清理入口的等待限制，不假造落地结果。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_observer_has_independent_outer_deadline():
    arguments = cleanup_arguments()
    arguments["landing_timeout_seconds"] = 0.01

    # 功能：
    #   模拟忽略超时参数但仍响应取消的原生观察器。
    # 输入：
    #   timeout：故意未在夹具内部执行的秒数限制。
    # 输出：
    #   None：不返回业务数据。
    async def stuck(timeout):
        await asyncio.Event().wait()

    arguments["client"].wait_until_landed.side_effect = stuck
    asyncio.run(cleanup_flight_commands(**arguments))
    assert "TimeoutError" in arguments["evidence"]["land"]
    assert "landing_observation" not in arguments["evidence"]


# 功能：
#   验证非法等待配置在调用任何飞行命令前明确拒绝。
# 输入：
#   timeout：非有限、非正或错误类型的等待参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("timeout", [0, -1, True, "1", float("nan"), float("inf")])
def test_invalid_command_timeout_is_not_coerced(timeout):
    arguments = cleanup_arguments()
    with pytest.raises(ValueError, match="CLEANUP_TIMEOUT_INVALID"):
        asyncio.run(cleanup_flight_commands(**arguments, command_timeout_seconds=timeout))
    arguments["client"].stop_offboard.assert_not_awaited()


# 功能：
#   验证独立运行时执行器在解锁回执丢失后，仍检查本次降落观测而不信任任意返回值。
# 输入：
#   tmp_path：隔离的日志与计时回执目录。
#   observation：不能证明已经落地的伪观察值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("observation", [None, {}, {"state": "IN_AIR", "confirmed": True},
                                        {"state": "ON_GROUND", "confirmed": 1}])
def test_standalone_cleanup_rejects_unconfirmed_landing(tmp_path, observation):
    from test_gazebo_adapter import _load_px4_executor_module

    base = _load_px4_executor_module()
    aircraft = base.FakeOffboardClient()
    aircraft.position_velocity_samples = [base.PositionVelocityNed(0, 0, 0, 0, 0, 0)]
    aircraft.arm = AsyncMock(side_effect=TimeoutError("lost arm acknowledgement"))
    aircraft.wait_until_landed = AsyncMock(return_value=observation)
    with pytest.raises(TimeoutError, match="lost arm acknowledgement"):
        asyncio.run(base.run_executor(
            aircraft, [base.Setpoint(0, 0, -0.5, 0)], connection="udp://:14540",
            takeoff_timeout_seconds=1., track_timeout_seconds=2., landing_timeout_seconds=1.,
            rate_hz=100., land_after=True, log_path=tmp_path / "executor.log",
            timing_path=tmp_path / "timing.json",
        ))
    receipt = json.loads((tmp_path / "timing.json").read_text())
    assert "LANDING_NOT_CONFIRMED" in receipt["cleanup"]["land"]
    assert "landing_observation" not in receipt["cleanup"]


# 功能：
#   在独立执行器中逐项注入退出故障，验证仍降落、观察、关闭并保留原始执行异常。
# 输入：
#   tmp_path：隔离的诊断目录。
#   monkeypatch：缩短清理限时或替换单一诊断入口。
#   fault：停止、降落回执、阶段发布或日志故障。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["stop_cancelled", "land_ack_lost", "phase_cancelled",
                                  "cleanup_log_failed", "stop_stuck"])
def test_standalone_failure_cleanup_continues_after_independent_fault(tmp_path, monkeypatch, fault):
    from test_gazebo_adapter import _load_px4_executor_module

    base = _load_px4_executor_module()
    aircraft = base.FakeOffboardClient()
    aircraft.position_velocity_samples = [base.PositionVelocityNed(0, 0, 0, 0, 0, 0)]
    aircraft.start_offboard = AsyncMock(side_effect=TimeoutError("original start failure"))
    aircraft.stop_offboard, aircraft.land, aircraft.close = AsyncMock(), AsyncMock(), AsyncMock()
    aircraft.wait_until_landed = AsyncMock(return_value={"state": "ON_GROUND", "confirmed": True})
    if fault == "stop_cancelled":
        aircraft.stop_offboard.side_effect = asyncio.CancelledError()
    elif fault == "land_ack_lost":
        aircraft.land.side_effect = TimeoutError("land acknowledgement lost")
    elif fault == "phase_cancelled":
        write_phase = base._write_runtime_phase

        # 功能：
        #   仅取消降落阶段的诊断发布，其他阶段保持原写入实现。
        # 输入：
        #   path：阶段文件路径。
        #   phase：阶段名。
        # 输出：
        #   None：不返回业务数据。
        def cancelled_phase(path, phase):
            if phase == "LANDING":
                raise asyncio.CancelledError()
            write_phase(path, phase)

        monkeypatch.setattr(base, "_write_runtime_phase", cancelled_phase)
    elif fault == "cleanup_log_failed":
        write_log = base._log

        # 功能：
        #   仅让清理日志写入失败，实际命令及原始执行日志保持正常。
        # 输入：
        #   path：日志文件路径。
        #   message：日志文字。
        # 输出：
        #   None：不返回业务数据。
        def failed_cleanup_log(path, message):
            if "failure cleanup" in message:
                raise OSError("cleanup log locked")
            write_log(path, message)

        monkeypatch.setattr(base, "_log", failed_cleanup_log)
    else:
        # 功能：
        #   模拟停止响应一直缺失，允许清理层通过取消结束等待。
        # 输入：
        #   无。
        # 输出：
        #   None：不返回业务数据。
        async def stuck():
            await asyncio.Event().wait()

        aircraft.stop_offboard.side_effect = stuck
        monkeypatch.setattr(base, "CLEANUP_COMMAND_TIMEOUT_SECONDS", 0.01)

    # 功能：
    #   给整个故障夹具额外设置测试级截止，避免错误的清理实现令测试无限等待。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def execute():
        await asyncio.wait_for(base.run_executor(
            aircraft, [base.Setpoint(0, 0, -0.5, 0)], connection="udp://:14540",
            takeoff_timeout_seconds=1., track_timeout_seconds=2., landing_timeout_seconds=0.1,
            rate_hz=100., land_after=True, log_path=tmp_path / "executor.log",
            timing_path=tmp_path / "timing.json", runtime_phase_path=tmp_path / "phase.json",
        ), timeout=1.)

    with pytest.raises(TimeoutError, match="original start failure"):
        asyncio.run(execute())
    aircraft.land.assert_awaited_once()
    aircraft.wait_until_landed.assert_awaited_once()
    aircraft.close.assert_awaited_once()
    receipt = json.loads((tmp_path / "timing.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["cleanup"]["land"] == "confirmed_on_ground_during_failure_cleanup"
    if fault == "cleanup_log_failed":
        assert receipt["cleanup"]["logging_errors"]
