"""Measured transport reserve never renews sensor authority; synthetic tests only."""

import asyncio
from types import SimpleNamespace

import pytest
from test_control_execution_evidence import evidence
from test_runtime_commands import _load_executor

from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand


# 功能：把本次成功发送实测延迟用于下次预留；原命令时效不能增加。
# 输入：合成飞控传输及单调/UTC 时钟；输出：较慢发送后的预留提高且不得自动降低。
@pytest.mark.parametrize("combined", [False, True])
def test_measured_reserve_increases_without_renewing_input(combined, tmp_path):
    executor = _load_executor()
    _, inputs = evidence()
    command = RuntimeLocalSafetyCommand.model_validate(inputs["command_records"][0]["command"])
    utc, monotonic, cost, events = [1.1], [10.0], [0.040], []
    executor.time = SimpleNamespace(time=lambda: utc[0])
    executor.asyncio = SimpleNamespace(
        get_running_loop=lambda: SimpleNamespace(time=lambda: monotonic[0])
    )
    executor._record_local_safety_executor_event = lambda _a, **row: events.append(row)

    async def send(*_args):
        utc[0] += cost[0]
        monotonic[0] += cost[0]

    args = SimpleNamespace()
    kwargs = dict(
        args=args,
        base=SimpleNamespace(VelocitySetpoint=lambda **v: SimpleNamespace(**v)),
        client=SimpleNamespace(set_velocity_ned=send, set_position_velocity_ned=send),
        setpoint=SimpleNamespace(yaw_deg=0.0),
        velocity_ned_mps=(0.2, 0.0, 0.0),
        command=command,
        coordinate_contract=None,
        phase_path=tmp_path / "phase.json",
        position_control=combined,
    )
    assert asyncio.run(executor._dispatch_motion_or_brake(**kwargs)) == 1140
    reserve = args._measured_transport_reserve_ms
    assert 45 <= reserve <= 46
    assert command.valid_until_unix_ms == 1200
    assert events[0]["details"]["deadline_extended"] is False
    assert events[0]["details"]["elapsed_monotonic_ms"] == pytest.approx(40.0)
    cost[0] = 0.001
    asyncio.run(executor._dispatch_motion_or_brake(**kwargs))
    assert args._measured_transport_reserve_ms == reserve
    assert len(events) == 1


# 功能：自适应余量只会更早拒发，不允许缩短基础预算或伪造类型。
# 输入：两种飞控发送路径及不同余量；输出：未发送/显式错误，保持原输入期限。
@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("reserve", [19, True, 30.0, 252, 45])
def test_invalid_or_insufficient_reserve_never_sends(combined, reserve):
    executor = _load_executor()
    executor.time = SimpleNamespace(time=lambda: 1.17)
    calls = []

    async def send(*args):
        calls.append(args)

    kwargs = dict(
        base=SimpleNamespace(VelocitySetpoint=lambda **v: SimpleNamespace(**v)),
        client=SimpleNamespace(set_velocity_ned=send, set_position_velocity_ned=send),
        velocity_ned_mps=(0.2, 0.0, 0.0),
        deadline_unix_ms=1200,
        transport_reserve_ms=reserve,
    )
    if combined:
        kwargs["setpoint"] = SimpleNamespace(yaw_deg=0.0)
        method = executor._send_position_with_velocity
    else:
        kwargs["yaw_deg"] = 0.0
        method = executor._send_model_velocity
    with pytest.raises((ValueError, executor.ControlInputLeaseUnavailable)):
        asyncio.run(method(**kwargs))
    assert not calls
