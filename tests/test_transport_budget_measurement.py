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
    assert reserve == 52
    assert command.valid_until_unix_ms == 1200
    assert events[0]["details"]["deadline_extended"] is False
    assert events[0]["details"]["elapsed_monotonic_ms"] == pytest.approx(40.0)
    assert events[0]["details"]["elapsed_wall_ms"] == 40
    cost[0] = 0.001
    asyncio.run(executor._dispatch_motion_or_brake(**kwargs))
    assert args._measured_transport_reserve_ms == reserve
    assert len(events) == 1


# 功能：实测传输余量包含毫秒取整和调度余量，短租约在物理发送前转为保护性制动。
# 输入：一次 43.4 毫秒发送以及下一条仅剩 54 毫秒的命令；输出：学习到 56 毫秒余量且不再发送第二条运动。
def test_fractional_transport_spike_is_reserved_before_next_physical_send(tmp_path):
    executor = _load_executor()
    _, inputs = evidence()
    command = RuntimeLocalSafetyCommand.model_validate(inputs["command_records"][0]["command"])
    utc, monotonic, calls = [1.100], [10.0], []
    executor.time = SimpleNamespace(time=lambda: utc[0])
    executor.asyncio = SimpleNamespace(
        get_running_loop=lambda: SimpleNamespace(time=lambda: monotonic[0])
    )
    executor._record_local_safety_executor_event = lambda *_a, **_kw: None

    async def send(*_args):
        calls.append("motion")
        utc[0] += 0.0434
        monotonic[0] += 0.0434

    args = SimpleNamespace()
    kwargs = dict(
        args=args,
        base=SimpleNamespace(
            Setpoint=lambda **v: SimpleNamespace(**v),
            VelocitySetpoint=lambda **v: SimpleNamespace(**v),
        ),
        client=SimpleNamespace(set_velocity_ned=send),
        setpoint=SimpleNamespace(yaw_deg=0.0),
        velocity_ned_mps=(0.2, 0.0, 0.0),
        command=command,
        coordinate_contract=None,
        phase_path=tmp_path / "phase.json",
    )
    assert asyncio.run(executor._dispatch_motion_or_brake(**kwargs)) == 1143
    assert args._measured_transport_reserve_ms == 56

    # The next command has 54 ms remaining.  The real executor enters its
    # existing bounded hold path instead of starting another late transport.
    short = command.model_copy(update={"valid_until_unix_ms": 1197})
    executor._refresh_px4_identity_telemetry = lambda **_: _async_value(SimpleNamespace(
        north_m=0.0, east_m=0.0, down_m=-1.0,
        north_m_s=0.0, east_m_s=0.0, down_m_s=0.0,
    ))
    executor._hold_heading_or_brake = lambda **_: _async_value(None)
    assert asyncio.run(executor._dispatch_motion_or_brake(**{**kwargs, "command": short})) is None
    assert calls == ["motion"]


async def _async_value(value):
    return value


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
