"""Heading outages revoke motion without inventing a route-derived heading."""

import asyncio
from types import SimpleNamespace

import pytest
from test_px4_base_telemetry import _base_module
from test_runtime_commands import _load_executor

from dronedream_agent_core.contracts import Px4CoordinateContract


# 功能：
#   构造完整位置速度测试样本，不作为物理飞行证据。
# 输入：
#   north：本轮位置，用来识别恢复后是否复用了旧位置。
# 输出：
#   observed：六轴有限遥测测试对象。
def observation(north=1.0):
    return SimpleNamespace(north_m=north, east_m=2.0, down_m=-3.0,
                           north_m_s=0.1, east_m_s=0.0, down_m_s=0.0)


# 功能：
#   构造可切换过期状态的姿态样本，不改变生产有效期。
# 输入：
#   stale：是否超过原有 250 毫秒限制。
# 输出：
#   telemetry：固定航向的测试姿态数据。
def attitude(stale=False):
    return {"sources": {"attitude": {"yaw_deg": 114.0,
                                    "sample_age_seconds": 0.3 if stale else 0.01}}}


# 功能：
#   验证底层刹停调用机体系零速度、零角速度，绝不发送绝对朝北航向。
# 输入：
#   无。
# 输出：
#   None：传输类型和四个值均通过断言。
def test_brake_adapter_uses_zero_body_velocity_and_yaw_rate():
    base = _base_module()
    sent = []

    async def send(value):
        sent.append(value)

    client = object.__new__(base.MavsdkOffboardClient)
    client._system = SimpleNamespace(offboard=SimpleNamespace(set_velocity_body=send))
    client._body_velocity_cls = lambda *values: ("body-rate", values)
    asyncio.run(client.set_braking_hold())
    assert sent == [("body-rate", (0.0, 0.0, 0.0, 0.0))]
    fixture = base.FakeOffboardClient()
    asyncio.run(fixture.set_braking_hold())
    assert fixture.braking_hold_count == 1
    assert fixture.setpoints == fixture.velocity_setpoints == []


# 功能：
#   在真实控制循环注入航向断档，确认重采样后才恢复保护悬停，不生成示范动作。
# 输入：
#   tmp_path：隔离证据目录；teacher：模型控制或仿真示范模式。
# 输出：
#   None：两种权限下仅在新鲜遥测稳定后发送最新位置和实测航向。
@pytest.mark.parametrize("teacher", [False, True])
def test_loop_recovers_with_new_position_and_without_training_labels(tmp_path, teacher):
    executor = _load_executor()
    args = SimpleNamespace(require_model_control_authority=not teacher,
        simulation_teacher_control=teacher, local_safety_required=True,
        local_safety_target=None, local_safety_command=tmp_path / "absent.json",
        local_safety_repair_timeout_seconds=2.0, setpoint_rate_hz=100.0)
    samples, brakes, sent, events = [], [], [], []

    async def refresh(**_):
        samples.append(observation(float(len(samples) + 1)))
        return samples[-1]

    async def brake():
        brakes.append(True)

    class EndTest(Exception):
        pass

    async def send(position, velocity):
        sent.append((position, velocity))
        raise EndTest()

    executor._refresh_px4_identity_telemetry = refresh
    executor._read_local_safety_command = lambda _: None
    executor._record_local_safety_executor_event = lambda _, **event: events.append(event)
    executor._record_model_control_application = lambda *a, **k: pytest.fail("fake training label")
    client = SimpleNamespace(set_braking_hold=brake, set_position_velocity_ned=send,
        latest_dynamics_telemetry=lambda _: attitude(stale=not brakes))
    with pytest.raises(EndTest):
        asyncio.run(executor._apply_local_safety(args=args,
            base=SimpleNamespace(Setpoint=lambda **v: SimpleNamespace(**v),
                                 VelocitySetpoint=lambda **v: SimpleNamespace(**v)),
            client=client, planned_setpoint=SimpleNamespace(
                north_m=90.0, east_m=80.0, down_m=-70.0, yaw_deg=45.0),
            coordinate_contract=Px4CoordinateContract(model_root_world_enu_m=[0., 0., 0.],
                collision_center_offset_model_m=[0., 0., .2]), phase_path=tmp_path / "phase.json"))
    assert len(brakes) > 2
    assert len(sent) == 1
    position, velocity = sent[0]
    assert position.north_m == samples[-1].north_m > samples[0].north_m
    assert position.yaw_deg == 114.0
    assert (velocity.north_m_s, velocity.east_m_s, velocity.down_m_s) == (0., 0., 0.)
    assert any(event["details"]["recovered"] for event in events)
    assert all(not event["details"]["training_label_authorized"] for event in events)
    assert args._heading_gap_started_at is None


# 功能：
#   验证长期断档、无位置、缺失传输及传输超时均不能成为无限恢复或继续运动。
# 输入：
#   failure：要注入的失败类型。
# 输出：
#   None：明确抛出降落异常，不发送普通控制。
@pytest.mark.parametrize("failure", ["timeout", "position", "unsupported", "transport", "slow"])
def test_gap_failure_remains_fail_closed(failure):
    executor = _load_executor()
    args = SimpleNamespace(require_model_control_authority=True)
    calls = []

    async def brake():
        calls.append(True)
        if failure == "transport":
            raise RuntimeError("transport down")
        if failure == "slow":
            await asyncio.sleep(1.0)

    executor._record_local_safety_executor_event = lambda *a, **k: None
    client = SimpleNamespace(latest_dynamics_telemetry=lambda _: attitude(stale=True))
    if failure != "unsupported":
        client.set_braking_hold = brake

    async def scenario():
        if failure == "timeout":
            args._heading_gap_started_at = asyncio.get_running_loop().time() - 1.01
        await executor._hold_heading_or_brake(
            args, client, None if failure == "position" else observation(), 45.0)

    with pytest.raises(executor.UserDirectedLanding, match="NATIVE_HEADING_"):
        asyncio.run(scenario())
    assert len(calls) == (1 if failure in {"transport", "slow"} else 0)


# 功能：
#   验证恢复计时不因偶发新帧或外部取消而伪造恢复完成。
# 输入：
#   无。
# 输出：
#   None：断档起点保留，取消向上传播。
def test_intermittent_heading_does_not_reset_deadline_and_cancellation_propagates():
    executor = _load_executor()
    args = SimpleNamespace(require_model_control_authority=True)
    stale = True
    cancel = False

    async def brake():
        if cancel:
            raise asyncio.CancelledError()

    client = SimpleNamespace(latest_dynamics_telemetry=lambda _: attitude(stale),
                             set_braking_hold=brake)
    executor._record_local_safety_executor_event = lambda *a, **k: None

    async def scenario():
        nonlocal stale, cancel
        assert await executor._hold_heading_or_brake(args, client, observation(), 45.0) is None
        started = args._heading_gap_started_at
        stale = False
        assert await executor._hold_heading_or_brake(args, client, observation(), 45.0) is None
        stale = True
        assert await executor._hold_heading_or_brake(args, client, observation(), 45.0) is None
        assert args._heading_gap_started_at == started
        assert args._heading_recovery_fresh_since is None
        cancel = True
        with pytest.raises(asyncio.CancelledError):
            await executor._hold_heading_or_brake(args, client, observation(), 45.0)

    asyncio.run(scenario())


# 功能：
#   传输等待耗尽姿态有效期时不把旧帧计为稳定恢复。
# 输入：
#   无。
# 输出：
#   None：恢复计时被清除，保持刹停且偏航积分无效。
def test_heading_expiring_during_brake_cannot_complete_recovery():
    executor = _load_executor()
    args = SimpleNamespace(require_model_control_authority=True)
    stale = False

    async def brake():
        nonlocal stale
        stale = True

    client = SimpleNamespace(latest_dynamics_telemetry=lambda _: attitude(stale),
                             set_braking_hold=brake)
    executor._record_local_safety_executor_event = lambda *a, **k: None

    async def scenario():
        now = asyncio.get_running_loop().time()
        args._heading_gap_started_at = now - 0.5
        args._heading_recovery_fresh_since = now - 0.3
        assert await executor._hold_heading_or_brake(args, client, observation(), 45.0) is None
        assert args._heading_gap_started_at is not None
        assert args._heading_recovery_fresh_since is None
        assert args._model_body_control_yaw_deg is None

    asyncio.run(scenario())
