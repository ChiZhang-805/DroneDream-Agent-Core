"""Test the real executor branch with a recording transport, not physics."""

import asyncio
from types import SimpleNamespace

import pytest
from clock_fixtures import isolate_time
from test_executed_control_training import teacher_evidence
from test_runtime_commands import _load_executor

from dronedream_agent_core.contracts import Px4CoordinateContract, Vector3


# 功能：
#   验证仿真教师发送速度与偏航控制而非点位控制，且保留独立的权限来源。
# 输入：
#   tmp_path：隔离控制文件的目录。
#   monkeypatch：只替换教师执行器墙钟的测试工具。
#   first_action：直接巡航，或先执行安全恢复再恢复巡航。
# 输出：
#   None：无返回值。
@pytest.mark.parametrize('first_action', ['continue', 'replan'])
@pytest.mark.parametrize('independent_route', [False, True])
def test_teacher_sends_velocity_yaw_with_distinct_authority(tmp_path, monkeypatch, first_action, independent_route):
    executor = _load_executor()
    _, command, _, _ = teacher_evidence()
    # Keep the fixture's epoch and the executor's transport clock consistent;
    # do not bypass the production lease check to accommodate historical data.
    isolate_time(monkeypatch, executor, time=lambda: 1.05)
    commands = [command]
    if first_action == 'replan':
        decision = command.decision.model_copy(update={'action': 'replan',
            'control_source': 'deterministic-brake', 'selected_yaw_rate_dps': 0.})
        commands.insert(0, command.model_copy(update={'decision': decision}))
    executor._read_local_safety_command = lambda args: commands.pop(0)

    async def ignore(**kwargs):
        return SimpleNamespace(north_m=0., east_m=0., down_m=-1.)

    executor._refresh_px4_identity_telemetry = ignore
    records = []

    class Client:
        def __init__(self):
            self.velocities = []

        async def set_velocity_ned(self, velocity):
            self.velocities.append(velocity)

        async def set_position_velocity_ned(self, *args):
            raise AssertionError("Teacher accidentally used position control")

        # 功能：
        #   为保护接管提供与路线不同的合成实测航向，暴露误用路线航向的回退。
        # 输入：
        #   max_age：执行器要求的最大样本年龄。
        # 输出：
        #   telemetry：满足时效约束的测试姿态。
        def latest_dynamics_telemetry(self, max_age):
            assert max_age <= .25
            telemetry = {'sources': {'attitude': {'yaw_deg': 114., 'sample_age_seconds': .01}}}
            return telemetry

    args = SimpleNamespace(
        simulation_teacher_control=not independent_route,
        independent_route_control=independent_route,
        local_safety_target=None,
        local_safety_command=tmp_path / "command.json",
        local_safety_repair_timeout_seconds=15.0,
        setpoint_rate_hz=20.0,
        run_dir=tmp_path,
        _control_application_writer=SimpleNamespace(submit=lambda path, row: records.append(row)),
    )
    client = Client()
    observed = asyncio.run(
        executor._apply_local_safety(
            args=args,
            base=SimpleNamespace(
                Setpoint=lambda **v: SimpleNamespace(**v),
                VelocitySetpoint=lambda **v: SimpleNamespace(**v),
            ),
            client=client,
            planned_setpoint=SimpleNamespace(north_m=99.0, east_m=98.0, down_m=-9.0, yaw_deg=20.0),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            planned_velocity_ned_mps=(1.0, 1.0, 1.0),
        )
    )
    velocity = client.velocities[0]
    assert (velocity.north_m_s, velocity.east_m_s, velocity.down_m_s) == pytest.approx(
        (-0.2, 0.4, -0.375)
    )
    # 首次接管也必须从实测114度起算，不能把路线20度当成当前航向突然转头。
    assert observed.yaw_deg == pytest.approx(114.5)
    events = [record for record in records if "status" in record]
    assert events[0]["status"] == "accepted"
    assert events[0]["command_observation_sha256"] == command.observation_sha256
    assert events[0]["command_generated_at_unix_ms"] == command.generated_at_unix_ms
    assert len(events[0]["command_sha256"]) == 64
    records = [record for record in records if "transport" in record]
    assert len(records) == (2 if first_action == 'replan' else 1)
    assert records[0]["model_authorized"] is False
    assert records[0]["intent"] is None
    assert records[0]["transport"] == "velocity-ned"
    assert records[-1]["yaw_rate_application"]["clockwise_rate_dps"] == 10.0
    if first_action == 'replan':
        assert records[0]['safety_action'] == 'replan'
        assert records[0]['control_source'] == 'deterministic-brake'
        assert records[0]['yaw_rate_application']['clockwise_rate_dps'] == 0.
        assert records[0].get('position_ned_m') is None
    assert getattr(args, "_model_authorized_control_applied_count", 0) == 0


def test_teacher_rejects_gazebo_truth_offset(tmp_path):
    executor = _load_executor()
    _, command, _, _ = teacher_evidence()
    command = command.model_copy(
        update={"estimator_to_world_position_offset_m": Vector3(x=1, y=0, z=0)}
    )
    executor._read_local_safety_command = lambda args: command

    async def ignore(**kwargs):
        pass

    executor._refresh_px4_identity_telemetry = ignore
    args = SimpleNamespace(
        simulation_teacher_control=True,
        local_safety_target=None,
        local_safety_command=tmp_path / "command.json",
        local_safety_repair_timeout_seconds=15.0,
        setpoint_rate_hz=20.0,
    )
    with pytest.raises(executor.UserDirectedLanding, match="truth offsets"):
        asyncio.run(
            executor._apply_local_safety(
                args=args,
                base=SimpleNamespace(),
                client=SimpleNamespace(),
                planned_setpoint=SimpleNamespace(north_m=1.0, east_m=1.0, down_m=-1.0, yaw_deg=0.0),
                coordinate_contract=Px4CoordinateContract(
                    model_root_world_enu_m=[0.0, 0.0, 0.0],
                    collision_center_offset_model_m=[0.0, 0.0, 0.2],
                ),
                phase_path=tmp_path / "phase.json",
            )
        )
