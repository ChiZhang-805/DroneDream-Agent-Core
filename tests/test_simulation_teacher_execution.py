"""Test the real executor branch with a recording transport, not physics."""

import asyncio
from types import SimpleNamespace

import pytest
from test_executed_control_training import teacher_evidence
from test_runtime_commands import _load_executor

from dronedream_agent_core.contracts import Px4CoordinateContract, Vector3


def test_teacher_sends_pure_velocity_yaw_with_distinct_authority(tmp_path, monkeypatch):
    executor = _load_executor()
    _, command, _, _ = teacher_evidence()
    # Keep the fixture's epoch and the executor's transport clock consistent;
    # do not bypass the production lease check to accommodate historical data.
    monkeypatch.setattr(executor.time, "time", lambda: 1.05)
    executor._read_local_safety_command = lambda args: command

    async def ignore(**kwargs):
        pass

    executor._refresh_px4_identity_telemetry = ignore
    records = []

    class Client:
        def __init__(self):
            self.velocities = []

        async def set_velocity_ned(self, velocity):
            self.velocities.append(velocity)

        async def set_position_velocity_ned(self, *args):
            raise AssertionError("Teacher accidentally used position control")

    args = SimpleNamespace(
        simulation_teacher_control=True,
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
    assert observed.yaw_deg == pytest.approx(20.5)
    events = [record for record in records if "status" in record]
    assert events[0]["status"] == "accepted"
    records = [record for record in records if "transport" in record]
    assert len(records) == 1
    assert records[0]["model_authorized"] is False
    assert records[0]["intent"] is None
    assert records[0]["transport"] == "velocity-ned"
    assert records[0]["yaw_rate_application"]["clockwise_rate_dps"] == 10.0
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
