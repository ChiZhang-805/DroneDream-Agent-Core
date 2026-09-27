"""Safety handovers must not rotate a model-controlled vehicle toward its route."""
import asyncio
from types import SimpleNamespace

import pytest
from test_runtime_commands import _load_executor

from dronedream_agent_core.contracts import Px4CoordinateContract


# 功能：
#   验证模型和示范在保护悬停后均从实测航向恢复，不追赶旧积分或路线航向。
# 输入：
#   teacher：是否使用明确的仿真示范权限。
# 输出：
#   None：恢复零角速度时保留实测航向。
@pytest.mark.parametrize('teacher', [False, True])
def test_hold_discards_old_integrated_target_and_zero_rate_resumes_at_measured_yaw(teacher):
    executor = _load_executor()
    args = SimpleNamespace(require_model_control_authority=not teacher,
                           simulation_teacher_control=teacher, setpoint_rate_hz=20.,
                           _model_body_control_yaw_deg=155.)
    client = SimpleNamespace(latest_dynamics_telemetry=lambda _: {
        "sources": {"attitude": {"yaw_deg": 114., "sample_age_seconds": .03}}})
    heading = executor._local_hold_yaw(args=args, client=client, fallback_heading_deg=90.)
    assert heading == 114.
    command = SimpleNamespace(decision=SimpleNamespace(control_source="local-model-body-control",
        action="continue", selected_yaw_rate_dps=0.), requested_control_intent=SimpleNamespace(
            yaw_control_mode="model-yaw-rate"))
    executor.sha256_json = lambda _: "test-command"
    result = executor._setpoint_with_model_body_yaw(
        args=args, base=SimpleNamespace(Setpoint=lambda **v: SimpleNamespace(**v)),
        setpoint=SimpleNamespace(north_m=0., east_m=0., down_m=-1., yaw_deg=90.),
        command=command,
    )
    assert result.yaw_deg == 114.


# 功能：
#   对启动、丢失、不可读和过期四种真实执行分支验证悬停不发生未授权转头。
# 输入：
#   tmp_path：隔离文件目录；kind：指令缺口类型；teacher：模型或示范权限。
# 输出：
#   None：实际发送设定值必须绑定实测位置与航向，速度为零。
@pytest.mark.parametrize('teacher', [False, True])
@pytest.mark.parametrize("kind", ["startup", "missing", "unreadable", "stale"])
def test_every_required_command_gap_brakes_at_native_heading(tmp_path, kind, teacher):
    executor = _load_executor()
    args = SimpleNamespace(require_model_control_authority=not teacher,
        simulation_teacher_control=teacher, local_safety_required=True,
        local_safety_target=None, local_safety_command=tmp_path / "command.json",
        local_safety_repair_timeout_seconds=1., setpoint_rate_hz=20.,
        _local_safety_command_established=kind == "missing")
    if kind in {"unreadable", "stale"}:
        args.local_safety_command.write_text("{}")
    executor._read_local_safety_command = lambda _: None
    executor._read_stale_local_safety_command = lambda _: (
        SimpleNamespace(valid_until_unix_ms=int(executor.time.time() * 1000))
        if kind == "stale" else None)
    executor._local_safety_pair_diagnostic = lambda _: {}
    executor._record_local_safety_executor_event = lambda *a, **kw: None

    async def measured(**_):
        return SimpleNamespace(north_m=1., east_m=2., down_m=-3., north_m_s=.2)

    executor._refresh_px4_identity_telemetry = measured
    sent = []

    class StopAfterSend(Exception):
        pass

    async def send(position, velocity):
        sent.append((position, velocity))
        raise StopAfterSend()

    client = SimpleNamespace(set_position_velocity_ned=send,
        latest_dynamics_telemetry=lambda _: {
            "sources": {"attitude": {"yaw_deg": 114., "sample_age_seconds": .02}}})
    with pytest.raises(StopAfterSend):
        asyncio.run(executor._apply_local_safety(args=args,
            base=SimpleNamespace(Setpoint=lambda **v: SimpleNamespace(**v),
                VelocitySetpoint=lambda **v: SimpleNamespace(**v)), client=client,
            planned_setpoint=SimpleNamespace(north_m=9., east_m=8., down_m=-7., yaw_deg=90.),
            coordinate_contract=Px4CoordinateContract(model_root_world_enu_m=[0., 0., 0.],
                collision_center_offset_model_m=[0., 0., .2]), phase_path=tmp_path / "phase.json"))
    assert len(sent) == 1
    position, velocity = sent[0]
    assert (position.north_m, position.east_m, position.down_m, position.yaw_deg) == (
        1., 2., -3., 114.)
    assert (velocity.north_m_s, velocity.east_m_s, velocity.down_m_s) == (0., 0., 0.)


# 功能：
#   验证实测航向缺失或过期时，两种控制模式均拒绝采用路线航向伪装新观测。
# 输入：
#   telemetry：无效姿态样本；teacher：模型或示范权限。
# 输出：
#   None：明确拒绝且不修改原积分状态。
@pytest.mark.parametrize('teacher', [False, True])
@pytest.mark.parametrize("telemetry", [None, {}, {"sources": {"attitude": {
    "yaw_deg": 114., "sample_age_seconds": .251}}}])
def test_missing_heading_never_substitutes_route_heading(telemetry, teacher):
    executor = _load_executor()
    args = SimpleNamespace(require_model_control_authority=not teacher,
                           simulation_teacher_control=teacher, _model_body_control_yaw_deg=155.)
    with pytest.raises(executor.UserDirectedLanding, match="FRESH_NATIVE_HEADING"):
        executor._local_hold_yaw(args=args, fallback_heading_deg=90.,
            client=SimpleNamespace(latest_dynamics_telemetry=lambda _: telemetry))
    assert args._model_body_control_yaw_deg == 155.
