"""Command fault injection; no real vehicle, connection or flight qualification."""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dronedream_agent_core.flight_command_cleanup import (
    FlightCommandAttempts,
    cleanup_flight_commands,
)


# 功能：
#   直接等待测试命令，不额外引入超时或吞掉异常。
# 输入：
#   command：本次异步命令。
# 输出：
#   result：命令返回值。
async def wait(command):
    result = await command
    return result


# 功能：
#   创建可独立注入故障的客户端替身，默认原生观察确认落地。
# 输入：
#   无。
# 输出：
#   aircraft：不连接硬件的异步客户端。
def client():
    aircraft = SimpleNamespace(
        arm=AsyncMock(), start_offboard=AsyncMock(), stop_offboard=AsyncMock(),
        land=AsyncMock(), wait_until_landed=AsyncMock(return_value={
            "state": "ON_GROUND", "confirmed": True}))
    return aircraft


# 功能：
#   验证解锁或进控回执丢失、执行被取消时，仍依据发送尝试完成清理。
# 输入：
#   failed_command：失败发生的命令名称。
#   failure：注入的超时或取消异常。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failed_command", ["arm", "start_offboard"])
@pytest.mark.parametrize("failure", [TimeoutError("lost reply"), asyncio.CancelledError()])
def test_lost_ack_or_cancellation_still_requires_cleanup(failed_command, failure):
    aircraft, state, evidence = client(), FlightCommandAttempts(), {}
    getattr(aircraft, failed_command).side_effect = failure

    # 功能：
    #   模拟原始执行抛错后进入 finally，清理不能覆盖原异常。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def exercise():
        try:
            await state.arm(aircraft, wait)
            await state.start_offboard(aircraft, wait)
        finally:
            await cleanup_flight_commands(aircraft, attempts=state, offboard_stopped=False,
                landed=False, landing_timeout_seconds=4., on_landing=lambda: None,
                evidence=evidence)

    with pytest.raises(type(failure)):
        asyncio.run(exercise())
    assert state.arm_requested
    assert state.arm_acknowledged is (failed_command != "arm")
    assert state.offboard_requested is (failed_command == "start_offboard")
    assert not state.offboard_acknowledged
    assert aircraft.stop_offboard.await_count == int(failed_command == "start_offboard")
    aircraft.land.assert_awaited_once()
    aircraft.wait_until_landed.assert_awaited_once_with(4.)
    assert evidence["land"] == "confirmed_on_ground_during_failure_cleanup"


# 功能：
#   验证从未解锁或已停控且确认落地时，不重复发出飞行命令。
# 输入：
#   already_landed：是否模拟已完成停控与落地。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("already_landed", [False, True])
def test_no_arm_or_already_landed_does_not_issue_unneeded_flight_commands(already_landed):
    aircraft, evidence = client(), {}
    state = FlightCommandAttempts(arm_requested=already_landed,
                                  offboard_requested=already_landed)
    asyncio.run(cleanup_flight_commands(aircraft, attempts=state,
        offboard_stopped=already_landed, landed=already_landed,
        landing_timeout_seconds=4., on_landing=Mock(), evidence=evidence))
    aircraft.land.assert_not_awaited()
    aircraft.stop_offboard.assert_not_awaited()
    assert not evidence


# 功能：
#   验证停控及阶段写入失败不跳过降落，无效原生观测不能生成确认回执。
# 输入：
#   bad_observation：缺失或未确认落地的观察值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad_observation", [{}, {"state": "IN_AIR", "confirmed": True},
    {"state": "ON_GROUND", "confirmed": False}, {"state": "ON_GROUND"}, None])
def test_stop_or_phase_write_failure_cannot_skip_landing_and_fake_confirmation_rejected(
    bad_observation,
):
    aircraft, evidence = client(), {}
    aircraft.stop_offboard.side_effect = TimeoutError("stop acknowledgement lost")
    aircraft.wait_until_landed.return_value = bad_observation
    asyncio.run(cleanup_flight_commands(aircraft,
        attempts=FlightCommandAttempts(arm_requested=True, offboard_requested=True),
        offboard_stopped=False, landed=False, landing_timeout_seconds=4.,
        on_landing=Mock(side_effect=OSError("phase file locked")), evidence=evidence))
    aircraft.land.assert_awaited_once()
    assert evidence["stop_offboard"].startswith("failed:")
    assert evidence["landing_phase_write"].startswith("failed:")
    assert "CLEANUP_NATIVE_LANDING_NOT_CONFIRMED" in evidence["land"]
    assert "landing_observation" not in evidence


# 功能：
#   静态核对检查点执行器统一登记命令尝试，并先做飞行清理再释放诊断资源。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_checkpoint_executor_uses_one_attempt_owner_and_cleans_before_diagnostics():
    source = (Path(__file__).parents[1] / "scripts/px4_checkpoint_executor.py").read_text(
        encoding="utf-8")
    run = next(n for n in ast.parse(source).body if isinstance(n, ast.AsyncFunctionDef)
               and n.name == "_run")
    text = ast.unparse(run)
    assert "await command_attempts.arm(" in text
    assert "await command_attempts.start_offboard(" in text
    assert "await client.arm(" not in text and "await client.start_offboard(" not in text
    cleanup_offset = text.index("await cleanup_flight_commands(")
    assert cleanup_offset < text.index("await finalize_executor_resources(")


# 功能：
#   验证调用命令时尝试标志已可见，返回回执与内部记录不共享可变状态。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_attempt_is_visible_inside_dispatch_and_evidence_is_detached():
    state, aircraft = FlightCommandAttempts(), client()
    # 功能：
    #   在命令执行内部检查发送前登记与发送后确认的先后次序。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def dispatched():
        assert state.arm_requested and not state.arm_acknowledged
    aircraft.arm.side_effect = dispatched
    asyncio.run(state.arm(aircraft, wait))
    evidence = state.evidence()
    assert evidence["arm_acknowledged"] is True
    evidence["arm_requested"] = False
    assert state.arm_requested


# 功能：
#   通过独立运行时执行器的假客户端核对丢失回执后的降落、状态与持久化回执。
# 输入：
#   tmp_path：隔离的日志和计时文件目录。
#   failed_command：故障命令名称。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failed_command", ["arm", "start_offboard"])
def test_standalone_runtime_executor_also_cleans_up_lost_ack(tmp_path, failed_command):
    from test_gazebo_adapter import _load_px4_executor_module

    base = _load_px4_executor_module()
    aircraft = base.FakeOffboardClient()
    aircraft.position_velocity_samples = [base.PositionVelocityNed(0, 0, 0, 0, 0, 0)]
    setattr(aircraft, failed_command, AsyncMock(side_effect=TimeoutError("lost acknowledgement")))
    aircraft.land = AsyncMock()
    aircraft.stop_offboard = AsyncMock()
    aircraft.wait_until_landed = AsyncMock(return_value={"state": "ON_GROUND", "confirmed": True})
    with pytest.raises(TimeoutError, match="lost acknowledgement"):
        asyncio.run(base.run_executor(aircraft, [base.Setpoint(0, 0, -.5, 0)],
            connection="udp://:14540", takeoff_timeout_seconds=1., track_timeout_seconds=2.,
            landing_timeout_seconds=1., rate_hz=100., land_after=True,
            log_path=tmp_path / "executor.log", timing_path=tmp_path / "timing.json"))
    aircraft.land.assert_awaited_once()
    assert aircraft.stop_offboard.await_count == int(failed_command == "start_offboard")
    receipt = json.loads((tmp_path / "timing.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["cleanup"]["land"] == "confirmed_on_ground_during_failure_cleanup"
    assert receipt["command_attempts"]["arm_requested"] is True
    assert receipt["command_attempts"]["arm_acknowledged"] is (failed_command != "arm")
