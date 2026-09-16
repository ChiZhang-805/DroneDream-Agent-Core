from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_runtime_follow import _args, _modules, _replacement, _write_assets


# 功能：
#   强制模型控制的动态跟随必须进入共用控制入口，不能通过直接位置发送绕过模型。
# 输入：
#   tmp_path：独立跟随证据目录。
#   monkeypatch：捕获控制入口，不启动模型或实际飞控。
# 输出：
#   None：不返回业务数据。
def test_follow_uses_model_control_and_clears_track_progress(tmp_path, monkeypatch) -> None:
    base, executor = _modules()
    semantic, vehicle = _write_assets(tmp_path)
    args = _args(tmp_path, semantic, vehicle)
    args.require_model_control_authority = True
    args.local_safety_required = True
    args.local_safety_command = tmp_path / "command.json"
    args.local_safety_target = tmp_path / "target.json"
    args._runtime_track_progress = object()
    client = base.FakeOffboardClient()
    client.gazebo_pose_samples = [{"x": 2, "y": 0, "z": 0, "topic": "/model/person/pose"}]
    calls = []

    # 功能：
    #   捕获跟随目标和有效期，模拟共用控制入口完成一次受控采样而不发送实际指令。
    # 输入：
    #   kwargs：跟随分支提交的目标与控制上下文。
    # 输出：
    #   applied：入口返回的设定值对象。
    async def controlled_tick(**kwargs):
        assert args._runtime_track_progress is None
        assert kwargs["navigation_goal_id"].startswith("follow-")
        assert kwargs["navigation_goal_deadline_monotonic"] > asyncio.get_running_loop().time()
        calls.append(kwargs)
        await asyncio.sleep(0.01)
        applied = kwargs["planned_setpoint"]
        return applied

    monkeypatch.setattr(executor, "_apply_local_safety", controlled_tick)
    asyncio.run(executor._follow_runtime_target(
        args=args, base=base, client=client, params=SimpleNamespace(), runtime_session=None,
        replacement=_replacement(executor, base),
        initial_setpoint=base.Setpoint(0, 0, -0.8, 0),
        phase_path=tmp_path / "phase.json", timing={"runtime_interruptions": []},
    ))
    assert calls
    assert all(sample.east_m == 0 for sample in client.setpoints)


# 功能：
#   已要求模型控制却缺少本地指令通道时提前拒绝，不能悄悄降级为位置跟随。
# 输入：
#   tmp_path：独立测试资产目录。
# 输出：
#   None：不返回业务数据。
def test_follow_cannot_downgrade_required_model_control(tmp_path: Path) -> None:
    base, executor = _modules()
    semantic, vehicle = _write_assets(tmp_path)
    args = _args(tmp_path, semantic, vehicle)
    args.require_model_control_authority = True
    client = base.FakeOffboardClient()
    with pytest.raises(executor.UserDirectedLanding, match="model control"):
        asyncio.run(executor._follow_runtime_target(
            args=args, base=base, client=client, params=SimpleNamespace(), runtime_session=None,
            replacement=_replacement(executor, base),
            initial_setpoint=base.Setpoint(0, 0, -0.8, 0),
            phase_path=tmp_path / "phase.json", timing={"runtime_interruptions": []},
        ))
    assert not client.setpoints


# 功能：
#   目标采样长期不返回时结束跟随并取消自有请求，不能把持续保持误报为成功跟随。
# 输入：
#   tmp_path：独立证据目录。
#   monkeypatch：使目标采样永久等待，仍运行真实跟随协程及期限处理。
# 输出：
#   None：不返回业务数据。
def test_follow_rejects_stalled_target_and_cancels_owned_request(tmp_path, monkeypatch) -> None:
    base, executor = _modules()
    semantic, vehicle = _write_assets(tmp_path)
    args = _args(tmp_path, semantic, vehicle)
    client = base.FakeOffboardClient()
    cancelled = []

    # 功能：
    #   模拟永远不返回的目标请求，记录跟随退出时是否回收该自有协程。
    # 输入：
    #   parameters：目标身份参数。
    # 输出：
    #   None：不返回业务数据。
    async def stalled(parameters):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(parameters)

    monkeypatch.setattr(client, "sample_gazebo_pose", stalled)
    with pytest.raises(executor.UserDirectedLanding, match="GOAL_EXPIRED"):
        asyncio.run(executor._follow_runtime_target(
            args=args, base=base, client=client, params=SimpleNamespace(), runtime_session=None,
            replacement=_replacement(executor, base), initial_setpoint=base.Setpoint(0, 0, -0.8, 0),
            phase_path=tmp_path / "phase.json", timing={"runtime_interruptions": []},
        ))
    assert len(cancelled) == 1
    assert all(sample.east_m == 0 for sample in client.setpoints)


# 功能：
#   动态目标即使在等待控制器的过程中到期，也必须在运动发送入口之前被拒绝。
# 输入：
#   tmp_path：未实际使用的控制输出路径。
# 输出：
#   None：不返回业务数据。
def test_expired_goal_is_rejected_before_motion_dispatch(tmp_path: Path) -> None:
    base, executor = _modules()
    client = base.FakeOffboardClient()
    with pytest.raises(executor.UserDirectedLanding, match="GOAL_EXPIRED"):
        asyncio.run(executor._dispatch_motion_or_brake(
            args=SimpleNamespace(), base=base, client=client,
            setpoint=base.Setpoint(0, 0, -0.8, 0), velocity_ned_mps=(1, 0, 0),
            command=SimpleNamespace(),
            coordinate_contract=_replacement(executor, base).coordinate_contract,
            phase_path=tmp_path / "phase.json", navigation_goal_deadline_monotonic=0,
        ))
    assert not client.setpoints


# 功能：
#   静态目标不要求过期时间；动态目标截止值不可为布尔、非有限或已到期时间。
# 输入：
#   deadline：待检查的截止时刻。
#   rejected：是否应该拒绝继续导航。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("deadline,rejected", [
    (None, False), (2, False), (1, True), (0, True), (True, True), (float("nan"), True),
])
def test_dynamic_goal_deadline_validation(deadline, rejected) -> None:
    _, executor = _modules()
    if rejected:
        with pytest.raises(executor.UserDirectedLanding, match="GOAL_EXPIRED"):
            executor._require_live_navigation_goal(deadline, 1)
    else:
        executor._require_live_navigation_goal(deadline, 1)


# 功能：
#   过期动态目标在共用控制入口即被拒绝，不能先发布目标或经无通道兼容分支发送运动。
# 输入：
#   tmp_path：不会实际写入的阶段路径。
# 输出：
#   None：不返回业务数据。
def test_expired_goal_is_rejected_before_local_control_publication(tmp_path: Path) -> None:
    base, executor = _modules()
    client = base.FakeOffboardClient()
    with pytest.raises(executor.UserDirectedLanding, match="GOAL_EXPIRED"):
        asyncio.run(executor._apply_local_safety(
            args=SimpleNamespace(), base=base, client=client,
            planned_setpoint=base.Setpoint(0, 0, -0.8, 0),
            coordinate_contract=_replacement(executor, base).coordinate_contract,
            phase_path=tmp_path / "phase.json", navigation_goal_deadline_monotonic=0,
        ))
    assert not client.setpoints
