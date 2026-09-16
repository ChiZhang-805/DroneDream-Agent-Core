import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from test_control_execution_evidence import evidence
from test_runtime_commands import _load_executor

from dronedream_agent_core.contracts import Px4CoordinateContract, RuntimeLocalSafetyCommand
from dronedream_agent_core.control_execution_evidence import verify_control_applications
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.runtime_evidence import BoundedRuntimeEvidenceWriter


@pytest.mark.parametrize("held_heading", [None, 78., -179.])
@pytest.mark.parametrize("model_heading", [-179., 22., 179.])
def test_actual_model_dispatch_preserves_approved_yaw_through_spawn_adapter(
    held_heading, model_heading,
):
    executor = _load_executor()
    executor.time = SimpleNamespace(time=lambda: 1.1)
    sent = []

    async def send(velocity):
        sent.append(velocity)

    wrapped = executor.SpawnRelativeOffboardClient(
        SimpleNamespace(set_velocity_ned=send),
        SimpleNamespace(north_m=100., east_m=200., down_m=-80.),
        heading_hold_deg=held_heading,
    )
    asyncio.run(executor._send_model_velocity(
        base=SimpleNamespace(VelocitySetpoint=lambda **v: SimpleNamespace(**v)),
        client=wrapped, velocity_ned_mps=(.2, -.1, -.05), yaw_deg=model_heading,
        deadline_unix_ms=1200,
    ))
    assert len(sent) == 1
    assert vars(sent[0]) == dict(north_m_s=.2, east_m_s=-.1, down_m_s=-.05,
                                yaw_deg=model_heading)


def test_explicit_model_adapter_failure_never_falls_back_to_route_heading_path():
    executor = _load_executor()
    calls = []

    async def explicit(_):
        calls.append("model")
        raise RuntimeError("adapter failure")

    async def legacy(_):
        calls.append("route")

    with pytest.raises(RuntimeError, match="adapter failure"):
        asyncio.run(executor._send_model_velocity(
            base=SimpleNamespace(VelocitySetpoint=lambda **v: SimpleNamespace(**v)),
            client=SimpleNamespace(set_model_velocity_ned=explicit, set_velocity_ned=legacy),
            velocity_ned_mps=(.1, 0., 0.), yaw_deg=2.,
        ))
    assert calls == ["model"]


@pytest.mark.parametrize("combined", [False, True])
def test_local_safety_hold_after_model_turn_preserves_measured_yaw(combined):
    executor = _load_executor()
    sent = []

    async def position(value):
        sent.append((value, None))

    async def position_velocity(value, velocity):
        sent.append((value, velocity))

    wrapped = executor.SpawnRelativeOffboardClient(
        SimpleNamespace(set_position_ned=position, set_position_velocity_ned=position_velocity),
        SimpleNamespace(north_m=10., east_m=20., down_m=-3.), heading_hold_deg=78.,
    )
    base = SimpleNamespace()
    if combined:
        base.VelocitySetpoint = lambda **v: SimpleNamespace(**v)
    asyncio.run(executor._send_position_with_velocity(
        base=base, client=wrapped,
        setpoint=SimpleNamespace(north_m=1., east_m=-2., down_m=-.5, yaw_deg=115.),
        velocity_ned_mps=(0., 0., 0.),
    ))
    assert len(sent) == 1
    assert vars(sent[0][0]) == dict(north_m=11., east_m=18., down_m=-3.5, yaw_deg=115.)
    if combined:
        assert vars(sent[0][1]) == dict(north_m_s=0., east_m_s=0., down_m_s=0., yaw_deg=115.)


def test_executor_trace_has_no_disk_write_in_the_command_to_transport_interval(tmp_path):
    executor = _load_executor()
    writing, release = threading.Event(), threading.Event()

    def serializer(record):
        writing.set()
        assert release.wait(2)
        return json.dumps(record)

    writer = BoundedRuntimeEvidenceWriter(
        tmp_path / "summary.json", summary_publisher=lambda *_: None,
        serializer=serializer,
    )
    args = SimpleNamespace(run_dir=tmp_path, _control_application_writer=writer)
    try:
        executor._record_local_safety_executor_event(
            args, status="accepted", details={"command_sequence": 1}
        )
        assert writing.wait(1)
        # The consumer can enqueue the next command while disk work is blocked.
        executor._record_local_safety_executor_event(
            args, status="accepted", details={"command_sequence": 2}
        )
        assert writer.summary()["submitted_count"] == 2
        assert writer.summary()["completed_count"] == 0
    finally:
        release.set()
        summary = writer.close()
    assert summary["complete"]
    rows = [json.loads(row) for row in (
        tmp_path / "runtime-state/local-safety-executor-history.jsonl"
    ).read_text().splitlines()]
    assert [row["command_sequence"] for row in rows] == [1, 2]


def test_executor_cannot_silently_restore_synchronous_trace_writes(tmp_path):
    executor = _load_executor()
    with pytest.raises(RuntimeError, match="EVIDENCE_WRITER_NOT_STARTED"):
        executor._record_local_safety_executor_event(
            SimpleNamespace(run_dir=tmp_path), status="accepted", details={}
        )


@pytest.mark.parametrize("action", ["continue", "slow", "replan"])
def test_dispatch_read_reserves_transport_after_scheduling_wait(tmp_path, action):
    executor = _load_executor()
    _, inputs = evidence()
    command = RuntimeLocalSafetyCommand.model_validate(inputs["command_records"][0]["command"])
    command = command.model_copy(update={"decision": command.decision.model_copy(
        update={"action": action})})
    path = tmp_path / "command.json"
    path.write_text(command.model_dump_json())
    args = SimpleNamespace(local_safety_command=path, local_safety_observation=None)
    executor.time = SimpleNamespace(time=lambda: 1.13)
    assert executor._read_local_safety_command(args) is not None
    executor.time = SimpleNamespace(time=lambda: 1.18)
    assert executor._read_local_safety_command(args) is not None
    executor.time = SimpleNamespace(time=lambda: 1.181)
    assert executor._read_local_safety_command(args) is None


def test_budget_for_one_wait_and_actual_transport_is_not_counted_twice(tmp_path):
    executor = _load_executor()
    _, inputs = evidence()
    command = RuntimeLocalSafetyCommand.model_validate(inputs["command_records"][0]["command"])
    now = [1.13]  # Publisher has 70 ms: next 50 ms tick plus 20 ms transport.
    executor.time = SimpleNamespace(time=lambda: now[0])
    path = tmp_path / "command.json"
    path.write_text(command.model_dump_json())
    args = SimpleNamespace(local_safety_command=path, local_safety_observation=None)
    now[0] = 1.18  # The executor has now reached that tick, not the next one.
    accepted_command = executor._read_local_safety_command(args)
    assert accepted_command is not None

    async def send(_value):
        now[0] = 1.195

    accepted_at = asyncio.run(executor._send_model_velocity(
        base=SimpleNamespace(VelocitySetpoint=lambda **v: SimpleNamespace(**v)),
        client=SimpleNamespace(set_velocity_ned=send), velocity_ned_mps=(.2, .1, -.05),
        yaw_deg=20., deadline_unix_ms=accepted_command.valid_until_unix_ms))
    assert accepted_at == 1195 < command.valid_until_unix_ms


def test_expired_before_dispatch_sends_nothing_and_late_acceptance_is_preserved(tmp_path):
    executor = _load_executor()
    now = [1.1]
    executor.time = SimpleNamespace(time=lambda: now[0])
    calls, rows = [], []

    async def send(value):
        calls.append(value)
        now[0] = 1.201

    kwargs = {"base": SimpleNamespace(VelocitySetpoint=lambda **v: SimpleNamespace(**v)),
              "client": SimpleNamespace(set_velocity_ned=send),
              "velocity_ned_mps": (.2, .1, -.05), "yaw_deg": 20.}
    with pytest.raises(executor.UserDirectedLanding, match="INSUFFICIENT_INPUT_LEASE"):
        asyncio.run(executor._send_model_velocity(**kwargs, deadline_unix_ms=1119))
    assert not calls
    accepted = asyncio.run(executor._send_model_velocity(**kwargs, deadline_unix_ms=1200))
    assert accepted == 1201 and len(calls) == 1
    _, inputs = evidence()
    command = RuntimeLocalSafetyCommand.model_validate(inputs["command_records"][0]["command"])
    args = SimpleNamespace(run_dir=tmp_path, _control_application_writer=SimpleNamespace(
        submit=lambda path, row: rows.append(row)))
    with pytest.raises(executor.UserDirectedLanding, match="EXCEEDED_INPUT_DEADLINE"):
        executor._record_model_control_application(
            args, command, velocity_ned_mps=(.2, .1, -.05), yaw_deg=20.,
            accepted_at_unix_ms=accepted,
        )
    assert rows[0]["accepted_at_unix_ms"] == 1201


def test_teacher_is_not_exempt_from_actual_transport_deadline():
    record, inputs = evidence()
    data = inputs["command_records"][0]["command"]
    data.update(navigation_control_authority="route-fallback", model_navigation_authorized=False,
                model_call_id=None, model_path_sha256=None, model_navigation_snapshot_sha256=None,
                requested_control_intent=None)
    data["decision"]["control_source"] = "route-target"
    command = RuntimeLocalSafetyCommand.model_validate(data)
    record.update(command_sha256=sha256_json(command), model_authorized=False, intent=None,
                  control_source="route-target", accepted_at_unix_ms=1201)
    result = verify_control_applications([record], **inputs)
    assert "CONTROL_APPLICATION_TRANSPORT_DEADLINE_VIOLATION" in result["issue_codes"]


@pytest.mark.parametrize("position_control", [False, True])
def test_unsent_expired_input_brakes_then_requires_new_dispatch(position_control, tmp_path):
    executor = _load_executor()
    _, inputs = evidence()
    command = RuntimeLocalSafetyCommand.model_validate(inputs["command_records"][0]["command"])
    now, elapsed, calls, rows = [1.181], [10.], [], []
    executor.time = SimpleNamespace(time=lambda: now[0])
    executor.asyncio = SimpleNamespace(
        get_running_loop=lambda: SimpleNamespace(time=lambda: elapsed[0]))
    position = [0.]

    async def telemetry(**_):
        return SimpleNamespace(north_m=position[0], east_m=0., down_m=-2.,
            north_m_s=.2, east_m_s=0., down_m_s=0.)

    async def velocity(value):
        calls.append(("velocity", value))

    async def combined(point, value):
        calls.append(("combined", point, value))

    executor._refresh_px4_identity_telemetry = telemetry
    executor._record_local_safety_executor_event = lambda _a, **row: rows.append(row)
    executor._publish_local_control_phase = lambda *_a, **_kw: None
    args = SimpleNamespace(setpoint_rate_hz=20., _model_body_control_yaw_deg=160.,
                           local_safety_runtime_stale_grace_seconds=8.)
    kwargs = dict(args=args, base=SimpleNamespace(
        Setpoint=lambda **v: SimpleNamespace(**v),
        VelocitySetpoint=lambda **v: SimpleNamespace(**v)),
        client=SimpleNamespace(set_velocity_ned=velocity, set_position_velocity_ned=combined,
            latest_dynamics_telemetry=lambda _: {"sources":{"attitude":{
                "yaw_deg":115., "sample_age_seconds":.01}}}),
        setpoint=SimpleNamespace(north_m=1., east_m=1., down_m=-2., yaw_deg=160.),
        velocity_ned_mps=(.4, .1, 0.), command=command,
        coordinate_contract=Px4CoordinateContract(model_root_world_enu_m=[0.,0.,0.],
            collision_center_offset_model_m=[0.,0.,.2]), phase_path=tmp_path/'phase.json',
        position_control=position_control)
    assert asyncio.run(executor._dispatch_motion_or_brake(**kwargs)) is None
    assert len(calls) == 1 and calls[0][0] == 'combined'
    assert calls[0][1].yaw_deg == 115.  # Unsent yaw integration is not replayed later.
    assert calls[0][2].north_m_s == calls[0][2].east_m_s == calls[0][2].down_m_s == 0.
    assert not rows[0]['details']['expired_motion_sent']
    assert not rows[0]['details']['schedule_advancement_authorized']
    assert args._dispatch_input_gap_started_at == 10.
    position[0], elapsed[0] = .1, 10.1
    assert asyncio.run(executor._dispatch_motion_or_brake(**kwargs)) is None
    assert calls[-1][1].north_m == .1  # Never return to the old braking point.
    assert args._dispatch_input_gap_started_at == 10.
    elapsed[0] = 18.01
    with pytest.raises(executor.UserDirectedLanding, match="LEASE_GAP_EXCEEDED"):
        asyncio.run(executor._dispatch_motion_or_brake(**kwargs))
    assert len(calls) == 2
    # A genuinely dispatchable later command ends the gap; arrival alone did not.
    now[0] = 1.3
    fresh = command.model_copy(update={"valid_until_unix_ms":1400})
    assert asyncio.run(executor._dispatch_motion_or_brake(**{**kwargs, 'command':fresh})) == 1300
    assert args._dispatch_input_gap_started_at is None
    assert len(calls) == 3


def test_dispatch_adapter_errors_are_not_reclassified_as_unsent_expiry(tmp_path):
    executor = _load_executor()
    _, inputs = evidence()
    command = RuntimeLocalSafetyCommand.model_validate(inputs['command_records'][0]['command'])
    executor.time = SimpleNamespace(time=lambda:1.1)

    async def failed_send(_):
        raise RuntimeError('transport outcome uncertain')

    with pytest.raises(RuntimeError, match='transport outcome uncertain'):
        asyncio.run(executor._dispatch_motion_or_brake(args=SimpleNamespace(),
            base=SimpleNamespace(VelocitySetpoint=lambda **v:SimpleNamespace(**v)),
            client=SimpleNamespace(set_velocity_ned=failed_send),
            setpoint=SimpleNamespace(yaw_deg=12.), velocity_ned_mps=(.2,0.,0.), command=command,
            coordinate_contract=None, phase_path=tmp_path/'phase.json'))


@pytest.mark.parametrize("action", ["continue", "slow"])
@pytest.mark.parametrize("teacher", [False, True])
def test_phase_disk_latency_cannot_spend_motion_transport_lease(tmp_path, action, teacher):
    """Both motion owners restore the phase after dispatch, not before it."""
    executor = _load_executor()
    _, inputs = evidence()
    data = inputs["command_records"][0]["command"]
    data["decision"]["action"] = action
    if teacher:
        data.update(navigation_control_authority="route-fallback",
                    model_navigation_authorized=False,
                    model_call_id=None, model_path_sha256=None,
                    model_navigation_snapshot_sha256=None, requested_control_intent=None)
        data["decision"]["control_source"] = "route-target"
    command = RuntimeLocalSafetyCommand.model_validate(data)
    clock, events = [1.17], []  # Original input expires at 1.20; never renewed.
    executor.time = SimpleNamespace(time=lambda: clock[0])
    executor._read_local_safety_command = lambda _args: command
    executor._local_safety_command_matches_control_context = lambda **_kwargs: True

    async def telemetry(**_kwargs):
        return SimpleNamespace(north_m=0., east_m=0., down_m=-1.)

    async def send(_velocity):
        events.append("transport")
        clock[0] = 1.175

    def slow_phase(*_args, **_kwargs):
        events.append("phase")
        clock[0] += .050

    executor._refresh_px4_identity_telemetry = telemetry
    executor._clear_local_control_phase = slow_phase
    executor._publish_local_control_phase = slow_phase
    # Keep real deadline checking/dispatch/branch selection; inspect receipt order.
    executor._record_model_control_application = lambda *a, **kw: events.append(
        ("receipt", kw["accepted_at_unix_ms"]))
    args = SimpleNamespace(local_safety_target=None, local_safety_command=tmp_path / "cmd",
        local_safety_repair_timeout_seconds=15., setpoint_rate_hz=20.,
        simulation_teacher_control=teacher)
    asyncio.run(executor._apply_local_safety(args=args,
        base=SimpleNamespace(Setpoint=lambda **v: SimpleNamespace(**v),
            VelocitySetpoint=lambda **v: SimpleNamespace(**v)),
        client=SimpleNamespace(set_velocity_ned=send, latest_dynamics_telemetry=lambda _: {
            "sources": {"attitude": {"yaw_deg": 0., "sample_age_seconds": .01}}}),
        planned_setpoint=SimpleNamespace(north_m=0., east_m=0., down_m=-1., yaw_deg=0.),
        coordinate_contract=Px4CoordinateContract(model_root_world_enu_m=[0., 0., 0.],
            collision_center_offset_model_m=[0., 0., .2]), phase_path=tmp_path / "phase.json"))
    assert events == ["transport", ("receipt", 1175), "phase"]
    assert command.valid_until_unix_ms == 1200


@pytest.mark.parametrize("writer_enabled", [False, True])
@pytest.mark.parametrize("action", ["continue", "slow", "replan"])
def test_late_acceptance_is_enforced_without_optional_evidence_writer(
    tmp_path, writer_enabled, action,
):
    """A diagnostics switch must never disable the post-dispatch safety check."""
    executor = _load_executor()
    _, inputs = evidence()
    data = inputs["command_records"][0]["command"]
    data["decision"]["action"] = action
    command = RuntimeLocalSafetyCommand.model_validate(data)
    rows = []
    args = SimpleNamespace(run_dir=tmp_path, _dispatch_input_gap_started_at=10.,
                           _dispatch_input_hold_setpoint="previous hold")
    if writer_enabled:
        args._control_application_writer = SimpleNamespace(submit=lambda _, row: rows.append(row))
    with pytest.raises(executor.UserDirectedLanding, match="EXCEEDED_INPUT_DEADLINE"):
        executor._record_model_control_application(args, command,
            velocity_ned_mps=(.2, .1, -.05), yaw_deg=20., accepted_at_unix_ms=1201)
    assert args._dispatch_input_gap_started_at == 10.
    assert args._dispatch_input_hold_setpoint == "previous hold"
    assert getattr(args, "_model_authorized_control_applied_count", 0) == 0
    assert len(rows) == int(writer_enabled)
    if rows:
        assert rows[0]["accepted_at_unix_ms"] == 1201


def test_valid_hold_ends_previous_input_gap_but_expired_hold_does_not():
    """Recovery depends on usable dispatched input, not on forward movement."""
    executor = _load_executor()
    _, inputs = evidence()
    data = inputs["command_records"][0]["command"]
    data["decision"].update(action="hold", control_source="deterministic-brake",
                            selected_velocity_mps={"x": 0., "y": 0., "z": 0.},
                            selected_yaw_rate_dps=0.)
    command = RuntimeLocalSafetyCommand.model_validate(data)
    args = SimpleNamespace(_dispatch_input_gap_started_at=10.,
                           _dispatch_input_hold_setpoint="previous hold")
    executor._record_model_control_application(args, command, accepted_at_unix_ms=1201)
    assert args._dispatch_input_gap_started_at == 10.
    executor._record_model_control_application(args, command, accepted_at_unix_ms=1200)
    assert args._dispatch_input_gap_started_at is None
    assert args._dispatch_input_hold_setpoint is None


@pytest.mark.parametrize("timestamp", [True, -1, 1.2, float("nan")])
def test_invalid_acceptance_clock_cannot_reset_recovery(timestamp):
    executor = _load_executor()
    _, inputs = evidence()
    command = RuntimeLocalSafetyCommand.model_validate(inputs["command_records"][0]["command"])
    args = SimpleNamespace(_dispatch_input_gap_started_at=10.)
    with pytest.raises(ValueError, match="ACCEPTANCE_TIME_INVALID"):
        executor._record_model_control_application(args, command, accepted_at_unix_ms=timestamp)
    assert args._dispatch_input_gap_started_at == 10.


def test_safety_position_replan_preserves_late_receipt_and_never_falls_back(tmp_path):
    executor = _load_executor()
    now, calls, rows = [1.1], [], []
    executor.time = SimpleNamespace(time=lambda: now[0])

    async def send(position, velocity):
        calls.append((position, velocity))
        now[0] = 1.201

    kwargs = dict(base=SimpleNamespace(VelocitySetpoint=lambda **v: SimpleNamespace(**v)),
        client=SimpleNamespace(set_position_velocity_ned=send),
        setpoint=SimpleNamespace(north_m=1., east_m=2., down_m=-3., yaw_deg=20.),
        velocity_ned_mps=(.1, .2, .0))
    with pytest.raises(executor.UserDirectedLanding, match="INSUFFICIENT_INPUT_LEASE"):
        asyncio.run(executor._send_position_with_velocity(**kwargs, deadline_unix_ms=1119))
    assert not calls
    accepted = asyncio.run(executor._send_position_with_velocity(**kwargs, deadline_unix_ms=1200))
    assert accepted == 1201 and len(calls) == 1
    with pytest.raises(executor.UserDirectedLanding, match="transport is unavailable"):
        asyncio.run(executor._send_position_with_velocity(
            **{**kwargs, "client": SimpleNamespace()}, deadline_unix_ms=1400))
    _, inputs = evidence()
    payload = inputs["command_records"][0]["command"]
    payload["decision"].update(action="replan", control_source="deterministic-brake")
    command = RuntimeLocalSafetyCommand.model_validate(payload)
    args = SimpleNamespace(run_dir=tmp_path, _control_application_writer=SimpleNamespace(
        submit=lambda path, row: rows.append(row)))
    with pytest.raises(executor.UserDirectedLanding, match="EXCEEDED_INPUT_DEADLINE"):
        executor._record_model_control_application(args, command,
            velocity_ned_mps=(.1, .2, 0.), yaw_deg=20., transport="position-velocity-ned",
            position_ned_m=(1., 2., -3.), accepted_at_unix_ms=accepted)
    assert rows[0]["accepted_at_unix_ms"] == accepted
    inputs["command_records"][0]["command"] = command.model_dump(mode="json")
    result = verify_control_applications(rows, **inputs)
    assert "CONTROL_APPLICATION_TRANSPORT_DEADLINE_VIOLATION" in result["issue_codes"]


@pytest.mark.parametrize("action", ["continue", "slow"])
@pytest.mark.parametrize("approved_rate", [0., 1.])
def test_velocity_safety_override_holds_measured_heading_and_records_real_yaw(
    tmp_path, action, approved_rate,
):
    executor = _load_executor()
    _, inputs = evidence()
    payload = inputs["command_records"][0]["command"]
    payload["decision"].update(action=action, control_source="deterministic-safety-override",
                               selected_yaw_rate_dps=approved_rate)
    command = RuntimeLocalSafetyCommand.model_validate(payload)
    executor.time = SimpleNamespace(time=lambda: 1.17)
    executor._read_local_safety_command = lambda _: command
    executor._local_safety_command_matches_control_context = lambda **_: True
    executor._clear_local_control_phase = lambda *_: None
    executor._publish_local_control_phase = lambda *_, **__: None

    async def telemetry(**_):
        return SimpleNamespace(north_m=0., east_m=0., down_m=-1.)

    sent, rows = [], []

    async def send(velocity):
        sent.append(velocity)

    executor._refresh_px4_identity_telemetry = telemetry
    args = SimpleNamespace(run_dir=tmp_path, local_safety_target=None,
        local_safety_command=tmp_path / "cmd", local_safety_repair_timeout_seconds=15.,
        setpoint_rate_hz=20., _model_body_control_yaw_deg=155.,
        _control_application_writer=SimpleNamespace(submit=lambda path, row: (
            rows.append(row) if path.name == "control-applications.jsonl" else None)))
    kwargs = dict(args=args,
        base=SimpleNamespace(Setpoint=lambda **v: SimpleNamespace(**v),
                             VelocitySetpoint=lambda **v: SimpleNamespace(**v)),
        client=SimpleNamespace(set_velocity_ned=send, latest_dynamics_telemetry=lambda _: {
            "sources": {"attitude": {"yaw_deg": 114., "sample_age_seconds": .01}}}),
        planned_setpoint=SimpleNamespace(north_m=0., east_m=0., down_m=-1., yaw_deg=90.),
        coordinate_contract=Px4CoordinateContract(model_root_world_enu_m=[0., 0., 0.],
            collision_center_offset_model_m=[0., 0., .2]), phase_path=tmp_path / "phase.json")
    if approved_rate:
        with pytest.raises(executor.UserDirectedLanding, match="CANNOT_AUTHOR_A_TURN"):
            asyncio.run(executor._apply_local_safety(**kwargs))
        assert not sent and not rows
        return
    asyncio.run(executor._apply_local_safety(**kwargs))
    assert len(sent) == len(rows) == 1
    assert sent[0].yaw_deg == rows[0]["yaw_heading_deg"] == 114.
    assert rows[0]["transport"] == "velocity-ned"
    assert rows[0]["yaw_rate_application"] == {
        "previous_heading_deg": 114., "clockwise_rate_dps": 0., "integration_seconds": .05}
    assert rows[0]["command_sha256"] == sha256_json(command)
    assert args._model_control_application_counts == {"safety-direction-override": 1}
