from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _base_module():
    path = Path(__file__).resolve().parents[1] / "runtime" / "px4_offboard_track_executor.py"
    spec = importlib.util.spec_from_file_location("test_px4_base_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_position_velocity_consumers_share_one_fresh_mavsdk_stream() -> None:
    base = _base_module()

    async def scenario() -> None:
        release = asyncio.Event()

        class Telemetry:
            calls = 0

            async def position_velocity_ned(self):
                self.calls += 1
                yield SimpleNamespace(
                    position=SimpleNamespace(north_m=1.0, east_m=2.0, down_m=-3.0),
                    velocity=SimpleNamespace(north_m_s=0.1, east_m_s=0.2, down_m_s=-0.3),
                )
                await release.wait()

        telemetry = Telemetry()
        client = object.__new__(base.MavsdkOffboardClient)
        client._system = SimpleNamespace(telemetry=telemetry)
        client._position_velocity_condition = asyncio.Condition()
        client._position_velocity_sample = None
        client._position_velocity_error = None
        client._position_velocity_task = asyncio.create_task(
            client._collect_position_velocity_ned()
        )
        try:
            first = await client.sample_position_velocity_ned(0.2)
            second = await client.sample_position_velocity_ned(0.2)
            assert first == second
            assert telemetry.calls == 1
        finally:
            client._position_velocity_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await client._position_velocity_task

    asyncio.run(scenario())


def test_position_velocity_cache_rejects_a_stale_sample() -> None:
    base = _base_module()

    async def scenario() -> None:
        release = asyncio.Event()

        class Telemetry:
            async def position_velocity_ned(self):
                yield SimpleNamespace(
                    position=SimpleNamespace(north_m=0.0, east_m=0.0, down_m=0.0),
                    velocity=SimpleNamespace(north_m_s=0.0, east_m_s=0.0, down_m_s=0.0),
                )
                await release.wait()

        client = object.__new__(base.MavsdkOffboardClient)
        client._system = SimpleNamespace(telemetry=Telemetry())
        client._position_velocity_condition = asyncio.Condition()
        client._position_velocity_sample = None
        client._position_velocity_error = None
        client._position_velocity_task = asyncio.create_task(
            client._collect_position_velocity_ned()
        )
        try:
            await client.sample_position_velocity_ned(0.2)
            await asyncio.sleep(0.02)
            with pytest.raises(TimeoutError, match="telemetry timeout"):
                await client.sample_position_velocity_ned(0.01)
        finally:
            client._position_velocity_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await client._position_velocity_task

    asyncio.run(scenario())


def test_position_velocity_stream_restart_replaces_stalled_subscription() -> None:
    base = _base_module()

    async def scenario() -> None:
        first_release = asyncio.Event()

        class Telemetry:
            calls = 0

            async def position_velocity_ned(self):
                self.calls += 1
                if self.calls == 1:
                    await first_release.wait()
                yield SimpleNamespace(
                    position=SimpleNamespace(north_m=4.0, east_m=5.0, down_m=-2.0),
                    velocity=SimpleNamespace(north_m_s=0.0, east_m_s=0.0, down_m_s=0.0),
                )
                await first_release.wait()

        telemetry = Telemetry()
        client = object.__new__(base.MavsdkOffboardClient)
        client._system = SimpleNamespace(telemetry=telemetry)
        client._position_velocity_condition = asyncio.Condition()
        client._position_velocity_sample = None
        client._position_velocity_error = None
        client._position_velocity_task = asyncio.create_task(
            client._collect_position_velocity_ned()
        )
        await asyncio.sleep(0)
        old_task = client._position_velocity_task
        await client.restart_position_velocity_ned_stream()
        observed = await client.sample_position_velocity_ned(0.2)
        assert old_task is not None and old_task.cancelled()
        assert telemetry.calls == 2
        assert observed.north_m == 4.0
        client._position_velocity_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await client._position_velocity_task

    asyncio.run(scenario())


def test_payload_dynamics_requires_fresh_fast_control_sources() -> None:
    base = _base_module()
    client = object.__new__(base.MavsdkOffboardClient)
    now = base.time.monotonic()
    client._dynamics_samples = {
        "imu": ({"acceleration_down_m_s2": -9.80665}, now),
        "attitude": ({"roll_deg": 1.0, "pitch_deg": -2.0}, now),
        "battery": ({"remaining_percent": 0.8, "voltage_v": 15.9}, now),
        "actuator_output": (
            {
                "actuator": [0.5, 0.5, 0.5, 0.5],
                "normalization_ready": True,
            },
            now,
        ),
    }
    client._dynamics_errors = {}
    client._dynamics_restart_counts = {}

    telemetry = client.latest_dynamics_telemetry(1.0)

    assert 0 <= int(base.time.time() * 1_000) - telemetry["collected_at_unix_ms"] < 100
    assert telemetry["ready_for_payload_inference"] is True
    assert telemetry["issue_codes"] == []
    assert telemetry["blocking_issue_codes"] == []
    assert telemetry["sources"]["imu"]["sample_age_seconds"] < 0.1
    client._dynamics_samples["battery"] = (
        {"remaining_percent": 0.8, "voltage_v": 15.9},
        now - 2.0,
    )
    advisory_stale = client.latest_dynamics_telemetry(1.0)
    assert advisory_stale["ready_for_payload_inference"] is True
    assert "battery:SAMPLE_STALE" in advisory_stale["issue_codes"]
    assert advisory_stale["blocking_issue_codes"] == []
    client._dynamics_samples["imu"] = (
        {"acceleration_down_m_s2": -9.80665},
        now - 2.0,
    )
    stale = client.latest_dynamics_telemetry(1.0)
    assert stale["ready_for_payload_inference"] is False
    assert "imu:SAMPLE_STALE" in stale["issue_codes"]
    assert "imu:SAMPLE_STALE" in stale["blocking_issue_codes"]


def test_dynamics_collectors_publish_real_sensor_shapes_without_resubscribing() -> None:
    base = _base_module()

    async def scenario() -> None:
        release = asyncio.Event()

        class Telemetry:
            async def imu(self):
                yield SimpleNamespace(
                    acceleration_frd=SimpleNamespace(
                        forward_m_s2=0.1,
                        right_m_s2=-0.2,
                        down_m_s2=-9.7,
                    ),
                    angular_velocity_frd=SimpleNamespace(
                        forward_rad_s=0.01,
                        right_rad_s=-0.02,
                        down_rad_s=0.03,
                    ),
                    timestamp_us=100,
                )
                await release.wait()

            async def attitude_euler(self):
                yield SimpleNamespace(
                    roll_deg=1.0,
                    pitch_deg=-2.0,
                    yaw_deg=80.0,
                    timestamp_us=101,
                )
                await release.wait()

            async def battery(self):
                yield SimpleNamespace(
                    remaining_percent=0.9,
                    voltage_v=16.2,
                    current_battery_a=4.5,
                )
                await release.wait()

            async def actuator_output_status(self):
                yield SimpleNamespace(active=15, actuator=[0.4, 0.5, 0.6, 0.5])
                await release.wait()

        client = object.__new__(base.MavsdkOffboardClient)
        client._system = SimpleNamespace(telemetry=Telemetry())
        client._dynamics_samples = {}
        client._dynamics_errors = {}
        client._dynamics_restart_counts = {}
        tasks = [
            asyncio.create_task(client._collect_imu()),
            asyncio.create_task(client._collect_attitude_euler()),
            asyncio.create_task(client._collect_dynamics_battery()),
            asyncio.create_task(client._collect_actuator_output_status()),
        ]
        for _ in range(100):
            if {"imu", "attitude", "battery", "actuator_output"} <= set(client._dynamics_samples):
                break
            await asyncio.sleep(0.001)
        telemetry = client.latest_dynamics_telemetry(1.0)
        assert telemetry["ready_for_payload_inference"] is True
        assert telemetry["sources"]["imu"]["acceleration_down_m_s2"] == -9.7
        assert telemetry["sources"]["attitude"]["yaw_deg"] == 80.0
        assert telemetry["sources"]["battery"]["current_battery_a"] == 4.5
        assert telemetry["sources"]["actuator_output"]["active_mask"] == 15
        assert telemetry["sources"]["actuator_output"]["normalization_ready"] is True
        assert (
            telemetry["sources"]["actuator_output"]["normalization_kind"] == "native-unit-interval"
        )
        for task in tasks:
            task.cancel()
        for task in tasks:
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_raw_sitl_actuator_outputs_use_explicit_vehicle_scale(monkeypatch) -> None:
    base = _base_module()
    monkeypatch.setenv(base.ACTUATOR_OUTPUT_ABSOLUTE_MAXIMUM_ENV, "1000")

    async def scenario() -> None:
        release = asyncio.Event()

        class Telemetry:
            async def actuator_output_status(self):
                yield SimpleNamespace(
                    active=4,
                    actuator=[800.0, 750.0, 0.0, 0.0],
                )
                await release.wait()

        client = object.__new__(base.MavsdkOffboardClient)
        client._system = SimpleNamespace(telemetry=Telemetry())
        client._dynamics_samples = {}
        client._dynamics_errors = {}
        client._dynamics_restart_counts = {}
        client._actuator_output_absolute_maximum = 1000.0
        task = asyncio.create_task(client._collect_actuator_output_status())
        try:
            for _ in range(100):
                if "actuator_output" in client._dynamics_samples:
                    break
                await asyncio.sleep(0.001)
            sample = client._dynamics_samples["actuator_output"][0]
            assert sample["raw_actuator"] == [800.0, 750.0, 0.0, 0.0]
            assert sample["actuator"] == [0.8, 0.75, 0.0, 0.0]
            assert sample["normalization_ready"] is True
            assert sample["normalization_absolute_maximum"] == 1000.0
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_unscaled_raw_actuator_output_blocks_payload_inference() -> None:
    base = _base_module()
    client = object.__new__(base.MavsdkOffboardClient)
    now = base.time.monotonic()
    client._dynamics_samples = {
        "imu": ({"acceleration_down_m_s2": -9.80665}, now),
        "attitude": ({"roll_deg": 0.0, "pitch_deg": 0.0}, now),
        "battery": ({"remaining_percent": 0.8, "voltage_v": 15.9}, now),
        "actuator_output": (
            {
                "actuator": [799.0, 799.0, 799.0, 799.0],
                "normalization_ready": False,
            },
            now,
        ),
    }
    client._dynamics_errors = {}
    client._dynamics_restart_counts = {}

    telemetry = client.latest_dynamics_telemetry(1.0)

    assert telemetry["ready_for_payload_inference"] is False
    assert "actuator_output:NORMALIZATION_UNAVAILABLE" in telemetry["issue_codes"]


def test_repeated_native_sensor_packet_does_not_renew_receipt_time(monkeypatch) -> None:
    base = _base_module()
    client = object.__new__(base.MavsdkOffboardClient)
    client._dynamics_samples = {}
    client._dynamics_errors = {}
    clock = [100.0]
    monkeypatch.setattr(base.time, "monotonic", lambda: clock[0])
    packet = {"timestamp_us": 100, "roll_deg": 0.0}
    client._store_dynamics_sample("attitude", packet)
    clock[0] = 101.0
    client._store_dynamics_sample("attitude", dict(packet))
    assert client._dynamics_samples["attitude"][1] == 100.0
    with pytest.raises(RuntimeError, match="conflicting"):
        client._store_dynamics_sample("attitude", packet | {"roll_deg": 1.0})


@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_odometry_collector_preserves_covariance_and_explicit_unknown(unknown, partial) -> None:
    from test_localization_evidence import covariance

    base = _base_module()

    async def scenario():
        release = asyncio.Event()
        packed = covariance()
        expected = list(packed)
        if partial:
            for index in (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 16, 17, 19):
                packed[index] = float("nan")
                expected[index] = None
        if unknown:
            packed[0] = float("nan")

        class Telemetry:
            async def odometry(self):
                yield SimpleNamespace(
                    time_usec=1000, frame_id=SimpleNamespace(name="ESTIM_NED"),
                    pose_covariance=SimpleNamespace(covariance_matrix=packed),
                )
                await release.wait()

        client = object.__new__(base.MavsdkOffboardClient)
        client._system = SimpleNamespace(telemetry=Telemetry())
        client._dynamics_samples = {}
        client._dynamics_errors = {}
        client._dynamics_restart_counts = {}
        task = asyncio.create_task(client._collect_odometry())
        try:
            for _ in range(100):
                if "odometry" in client._dynamics_samples:
                    break
                await asyncio.sleep(0.001)
            packet = client._dynamics_samples["odometry"][0]
            assert packet["frame_id"] == "ESTIM_NED"
            assert packet["pose_covariance_upper_m2"] == (None if unknown else expected)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_dynamics_telemetry_rates_are_explicitly_requested() -> None:
    base = _base_module()

    async def scenario() -> None:
        class Telemetry:
            requests: list[tuple[str, float]] = []

            async def set_rate_position_velocity_ned(self, rate_hz: float) -> None:
                self.requests.append(("position_velocity", rate_hz))

            async def set_rate_odometry(self, rate_hz: float) -> None:
                self.requests.append(("odometry", rate_hz))

            async def set_rate_imu(self, rate_hz: float) -> None:
                self.requests.append(("imu", rate_hz))

            async def set_rate_attitude_euler(self, rate_hz: float) -> None:
                self.requests.append(("attitude", rate_hz))

            async def set_rate_battery(self, rate_hz: float) -> None:
                self.requests.append(("battery", rate_hz))

            async def set_rate_actuator_output_status(self, rate_hz: float) -> None:
                self.requests.append(("actuator_output", rate_hz))

        telemetry = Telemetry()
        client = object.__new__(base.MavsdkOffboardClient)
        client._system = SimpleNamespace(telemetry=telemetry)

        evidence = await client.configure_dynamics_telemetry_rates()

        assert telemetry.requests == [
            ("position_velocity", 50.0),
            ("imu", 50.0),
            ("attitude", 50.0),
            ("battery", 2.0),
            ("actuator_output", 10.0),
            ("odometry", 50.0),
        ]
        assert evidence["required_rate_requests_succeeded"] is True
        assert evidence["sources"]["battery"]["status"] == "requested"

    asyncio.run(scenario())


def test_dynamics_collector_replaces_an_ended_subscription_without_competing_streams() -> None:
    base = _base_module()

    async def scenario() -> None:
        replacement_started = asyncio.Event()
        release = asyncio.Event()

        class Telemetry:
            calls = 0

            async def imu(self):
                self.calls += 1
                if self.calls == 1:
                    return
                yield SimpleNamespace(
                    acceleration_frd=SimpleNamespace(
                        forward_m_s2=0.2,
                        right_m_s2=-0.1,
                        down_m_s2=-9.8,
                    ),
                    angular_velocity_frd=SimpleNamespace(
                        forward_rad_s=0.01,
                        right_rad_s=0.02,
                        down_rad_s=-0.01,
                    ),
                    timestamp_us=202,
                )
                replacement_started.set()
                await release.wait()

        telemetry = Telemetry()
        client = object.__new__(base.MavsdkOffboardClient)
        client._system = SimpleNamespace(telemetry=telemetry)
        client._dynamics_samples = {}
        client._dynamics_errors = {}
        client._dynamics_restart_counts = {}
        task = asyncio.create_task(client._collect_imu())
        try:
            await asyncio.wait_for(replacement_started.wait(), timeout=0.5)
            evidence = client.latest_dynamics_telemetry(1.0)
            assert telemetry.calls == 2
            assert evidence["sources"]["imu"]["timestamp_us"] == 202
            assert evidence["restart_counts"] == {"imu": 1}
            assert "imu" not in client._dynamics_errors
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_dynamics_collector_replaces_a_connected_but_stalled_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_module()
    monkeypatch.setattr(
        base,
        "DYNAMICS_STREAM_SAMPLE_TIMEOUT_SECONDS",
        {"battery": 0.01},
    )

    async def scenario() -> None:
        first_stream_release = asyncio.Event()
        replacement_started = asyncio.Event()
        replacement_release = asyncio.Event()

        class Telemetry:
            calls = 0

            async def battery(self):
                self.calls += 1
                yield SimpleNamespace(
                    remaining_percent=0.9,
                    voltage_v=16.2,
                    current_battery_a=4.5,
                )
                if self.calls == 1:
                    await first_stream_release.wait()
                replacement_started.set()
                await replacement_release.wait()

        telemetry = Telemetry()
        client = object.__new__(base.MavsdkOffboardClient)
        client._system = SimpleNamespace(telemetry=telemetry)
        client._dynamics_samples = {}
        client._dynamics_errors = {}
        client._dynamics_restart_counts = {}
        task = asyncio.create_task(client._collect_dynamics_battery())
        try:
            await asyncio.wait_for(replacement_started.wait(), timeout=0.5)
            evidence = client.latest_dynamics_telemetry(1.0)
            assert telemetry.calls == 2
            assert evidence["restart_counts"] == {"battery": 1}
            assert "battery" not in client._dynamics_errors
            assert evidence["sources"]["battery"]["voltage_v"] == 16.2
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_takeoff_stability_finishes_window_started_before_nominal_deadline() -> None:
    base = _base_module()

    async def scenario() -> None:
        class StableClient(base.FakeOffboardClient):
            async def sample_position_velocity_ned(self, timeout_seconds: float):
                del timeout_seconds
                await asyncio.sleep(0.01)
                return base.PositionVelocityNed(
                    north_m=0.0,
                    east_m=0.0,
                    down_m=-1.0,
                    north_m_s=0.0,
                    east_m_s=0.0,
                    down_m_s=0.0,
                )

        evidence: dict[str, object] = {}
        await base._wait_for_takeoff_stability(
            StableClient(),
            base.Setpoint(0.0, 0.0, -1.0, 0.0),
            timeout_seconds=0.03,
            sample_rate_hz=200.0,
            stable_window_seconds=0.05,
            horizontal_tolerance_m=0.1,
            vertical_tolerance_m=0.1,
            horizontal_speed_tolerance_m_s=0.1,
            vertical_speed_tolerance_m_s=0.1,
            evidence=evidence,
        )

        assert evidence["status"] == "achieved"
        assert float(evidence["achieved_stable_window_s"]) >= 0.05
        grace = evidence["stability_completion_grace"]
        assert isinstance(grace, dict)
        assert grace["granted"] is True
        assert grace["used"] is True
        assert grace["completed"] is True

    asyncio.run(scenario())


def test_takeoff_stability_fails_when_envelope_is_lost_during_grace() -> None:
    base = _base_module()

    async def scenario() -> None:
        class DestabilizingClient(base.FakeOffboardClient):
            sample_count = 0

            async def sample_position_velocity_ned(self, timeout_seconds: float):
                del timeout_seconds
                self.sample_count += 1
                await asyncio.sleep(0.025)
                return base.PositionVelocityNed(
                    north_m=0.0 if self.sample_count < 3 else 1.0,
                    east_m=0.0,
                    down_m=-1.0,
                    north_m_s=0.0,
                    east_m_s=0.0,
                    down_m_s=0.0,
                )

        evidence: dict[str, object] = {}
        with pytest.raises(TimeoutError, match="left the stability envelope"):
            await base._wait_for_takeoff_stability(
                DestabilizingClient(),
                base.Setpoint(0.0, 0.0, -1.0, 0.0),
                timeout_seconds=0.06,
                sample_rate_hz=200.0,
                stable_window_seconds=0.1,
                horizontal_tolerance_m=0.1,
                vertical_tolerance_m=0.1,
                horizontal_speed_tolerance_m_s=0.1,
                vertical_speed_tolerance_m_s=0.1,
                evidence=evidence,
            )

        assert evidence["status"] == "failed"
        assert evidence["failure_reason"] == "takeoff_stability_lost_during_completion_grace"
        grace = evidence["stability_completion_grace"]
        assert isinstance(grace, dict)
        assert grace["granted"] is True
        assert grace["used"] is True
        assert grace["completed"] is False

    asyncio.run(scenario())


def test_mavsdk_client_uses_a_dedicated_embedded_server_port(monkeypatch) -> None:
    base = _base_module()

    async def scenario() -> None:
        release = asyncio.Event()

        class Telemetry:
            async def position_velocity_ned(self):
                await release.wait()
                if False:
                    yield None

        class System:
            created_ports: list[int] = []

            def __init__(self, *, port: int) -> None:
                self.created_ports.append(port)
                self.telemetry = Telemetry()
                self.stopped = False

            async def connect(self, *, system_address: str) -> None:
                assert system_address == "udpin://0.0.0.0:14540"

            def _stop_mavsdk_server(self) -> None:
                self.stopped = True

        monkeypatch.setattr(base, "_allocate_loopback_tcp_port", lambda: 43_219)
        client = object.__new__(base.MavsdkOffboardClient)
        client._system_cls = System
        client._system = None
        client._mavsdk_server_port = None
        client._position_velocity_condition = None
        client._position_velocity_sample = None
        client._position_velocity_error = None
        client._position_velocity_task = None

        await client.connect("udpin://0.0.0.0:14540")
        system = client._system
        assert System.created_ports == [43_219]
        assert client._mavsdk_server_port == 43_219
        await client.close()
        assert system.stopped is True

    asyncio.run(scenario())


def test_preflight_recovers_two_unavailable_transports_before_arm(tmp_path: Path) -> None:
    base = _base_module()

    async def scenario() -> None:
        class UnavailableCode:
            name = "UNAVAILABLE"

        class AioRpcError(Exception):
            def code(self):
                return UnavailableCode()

        class Client:
            def __init__(self) -> None:
                self.connect_count = 0
                self.ready_count = 0
                self.close_count = 0
                self._mavsdk_server_port = None

            async def connect(self, _connection: str) -> None:
                self.connect_count += 1
                self._mavsdk_server_port = 44_000 + self.connect_count

            async def wait_until_ready(self, _timeout_seconds: float):
                self.ready_count += 1
                if self.ready_count <= 2:
                    raise AioRpcError("Stream removed (Connection reset by peer)")
                return base.TelemetryHealth(
                    connected=True,
                    global_position_ok=True,
                    home_position_ok=True,
                    local_position_ok=True,
                    armable=True,
                )

            async def close(self) -> None:
                self.close_count += 1

        client = Client()
        evidence: dict[str, object] = {}
        result = await base.connect_preflight_with_recovery(
            client,
            connection="udpin://0.0.0.0:14540",
            readiness_timeout_seconds=2.0,
            abort_check=lambda: None,
            log_path=tmp_path / "preflight.log",
            evidence=evidence,
        )

        assert result.armable is True
        assert client.connect_count == 3
        assert client.close_count == 2
        assert evidence["status"] == "ready"
        assert evidence["successful_attempt"] == 3
        attempts = evidence["attempts"]
        assert isinstance(attempts, list)
        assert attempts[0]["recoverable_transport_failure"] is True
        assert attempts[0]["cleanup"] == "closed_and_reaped"
        assert attempts[1]["recoverable_transport_failure"] is True
        assert attempts[1]["cleanup"] == "closed_and_reaped"
        assert attempts[2]["status"] == "ready"

    asyncio.run(scenario())


def test_preflight_does_not_retry_an_estimator_readiness_failure(tmp_path: Path) -> None:
    base = _base_module()

    async def scenario() -> None:
        class Client:
            def __init__(self) -> None:
                self.connect_count = 0
                self.close_count = 0

            async def connect(self, _connection: str) -> None:
                self.connect_count += 1

            async def wait_until_ready(self, _timeout_seconds: float):
                raise RuntimeError("PX4 health stream ended before armable")

            async def close(self) -> None:
                self.close_count += 1

        client = Client()
        evidence: dict[str, object] = {}
        with pytest.raises(RuntimeError, match="before armable"):
            await base.connect_preflight_with_recovery(
                client,
                connection="udpin://0.0.0.0:14540",
                readiness_timeout_seconds=2.0,
                abort_check=lambda: None,
                log_path=tmp_path / "preflight.log",
                evidence=evidence,
            )
        assert client.connect_count == 1
        assert client.close_count == 0
        attempts = evidence["attempts"]
        assert isinstance(attempts, list)
        assert attempts[0]["recoverable_transport_failure"] is False

    asyncio.run(scenario())


def test_offboard_loss_failsafe_is_firmware_owned_hold_with_readback() -> None:
    base = _base_module()
    client = base.FakeOffboardClient()

    evidence = asyncio.run(base.configure_offboard_loss_failsafe(client))

    assert evidence["verified"] is True
    assert evidence["action"] == "HOLD"
    assert evidence["timeout"] == {
        "before": 1.0,
        "requested": 1.0,
        "applied": 1.0,
    }
    assert evidence["mode"] == {"before": 0, "requested": 5, "applied": 5}
    assert client.float_params["COM_OF_LOSS_T"] == 1.0
    assert client.int_params["COM_OBL_RC_ACT"] == 5


def test_payload_command_republishes_until_plugin_state_is_observed(monkeypatch) -> None:
    base = _base_module()

    class Empty:
        pass

    class StringMsg:
        def __init__(self, data: str = "") -> None:
            self.data = data

    class Publisher:
        def __init__(self, node) -> None:
            self.node = node
            self.publish_count = 0

        def has_connections(self) -> bool:
            return True

        def publish(self, _message) -> None:
            self.publish_count += 1
            if self.publish_count == 3:
                self.node.callback(StringMsg("detached"))

    class Node:
        last_publisher = None

        def subscribe(self, _message_type, _topic, callback) -> bool:
            self.callback = callback
            return True

        def advertise(self, _topic, _message_type):
            publisher = Publisher(self)
            Node.last_publisher = publisher
            self.callback(StringMsg("detached"))
            return publisher

        def unsubscribe(self, _topic) -> None:
            pass

    monkeypatch.setattr(
        base,
        "_gazebo_transport_bindings",
        lambda: (Node, object, object, object, Empty, StringMsg),
    )
    client = object.__new__(base.MavsdkOffboardClient)
    client._payload_detached = False

    result = asyncio.run(
        client.execute_payload_command(
            {
                "protocol": "gazebo-transport",
                "operation": "detach",
                "topic": "/payload/detach",
                "output_topic": "/payload/state",
            }
        )
    )

    assert result["confirmed"] is True
    assert result["detached"] is True
    assert result["command_publish_attempts"] == 3
    assert result["state_transition_timeout_seconds"] == 15.0
    assert Node.last_publisher is not None
    assert Node.last_publisher.publish_count == 3


def test_payload_attach_does_not_realign_when_rigid_mount_is_observed(monkeypatch) -> None:
    base = _base_module()

    class Empty:
        pass

    class StringMsg:
        def __init__(self, data: str = "") -> None:
            self.data = data

    class Publisher:
        def __init__(self, node) -> None:
            self.node = node
            self.publish_count = 0

        def has_connections(self) -> bool:
            return True

        def publish(self, _message) -> None:
            self.publish_count += 1
            if self.publish_count == 3:
                self.node.callback(StringMsg("attached"))

    class Node:
        last_publisher = None

        def subscribe(self, _message_type, _topic, callback) -> bool:
            self.callback = callback
            return True

        def advertise(self, _topic, _message_type):
            publisher = Publisher(self)
            Node.last_publisher = publisher
            self.callback(StringMsg("detached"))
            return publisher

        def unsubscribe(self, _topic) -> None:
            pass

    monkeypatch.setattr(
        base,
        "_gazebo_transport_bindings",
        lambda: (Node, object, object, object, Empty, StringMsg),
    )
    monkeypatch.setenv("PX4_GAZEBO_WORLD_NAME", "school")
    client = object.__new__(base.MavsdkOffboardClient)
    client._payload_detached = True
    alignment_count = 0

    async def align(_parameters, **_kwargs):
        nonlocal alignment_count
        alignment_count += 1
        return {"binding_sha256": "a" * 64}

    vehicle_pose = base.GazeboModelPose(x=1.0, y=2.0, z=3.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)
    payload_pose = base.GazeboModelPose(x=1.0, y=2.0, z=3.12, qx=0.0, qy=0.0, qz=0.0, qw=1.0)

    async def sample(**_kwargs):
        return {"my_drone": vehicle_pose, "takeout_payload": payload_pose}

    client._align_payload_to_mount = align
    client._sample_named_gazebo_poses = sample
    result = asyncio.run(
        client.execute_payload_command(
            {
                "protocol": "gazebo-transport",
                "operation": "attach",
                "topic": "/payload/attach",
                "output_topic": "/payload/state",
                "vehicle_model_name": "my_drone",
                "payload_model_name": "takeout_payload",
                "payload_mount_offset_model_m": [0.0, 0.0, 0.12],
                "payload_mount_max_alignment_error_m": 0.02,
                "payload_mount_binding_sha256": "a" * 64,
            }
        )
    )

    assert result["confirmed"] is True
    assert result["detached"] is False
    assert result["command_publish_attempts"] == 3
    assert result["payload_mount_alignment"]["alignment_attempts"] == 1
    assert alignment_count == 1


# 功能：
#   证明位置连续贴合不能替代关节连接事件，丢失事件时必须拒绝成功且不重复搬动载荷。
# 输入：
#   monkeypatch：隔离原生传输并提供静止、未确认连接的位姿。
# 输出：
#   None：断言事件超时、只对齐一次且确实尝试发送过命令。
def test_payload_attach_rejects_persistent_rigid_mount_when_state_event_is_lost(
    monkeypatch,
) -> None:
    base = _base_module()

    class Empty:
        pass

    class StringMsg:
        data = ""

    class Publisher:
        def __init__(self) -> None:
            self.publish_count = 0

        def has_connections(self) -> bool:
            return True

        def publish(self, _message) -> None:
            self.publish_count += 1

    class Node:
        last_publisher = None

        def subscribe(self, _message_type, _topic, _callback) -> bool:
            self.callback = _callback
            return True

        def advertise(self, _topic, _message_type):
            publisher = Publisher()
            Node.last_publisher = publisher
            self.callback(SimpleNamespace(data="detached"))
            return publisher

        def unsubscribe(self, _topic) -> None:
            pass

    monkeypatch.setattr(
        base,
        "_gazebo_transport_bindings",
        lambda: (Node, object, object, object, Empty, StringMsg),
    )
    monkeypatch.setattr(base, "GAZEBO_PAYLOAD_STATE_DISCOVERY_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(base, "GAZEBO_PAYLOAD_COMMAND_RETRY_SECONDS", 0.001)
    monkeypatch.setattr(base, "GAZEBO_PAYLOAD_STATE_TRANSITION_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setenv("PX4_GAZEBO_WORLD_NAME", "school")
    client = object.__new__(base.MavsdkOffboardClient)
    client._payload_detached = True
    alignment_count = 0

    async def align(_parameters, **_kwargs):
        nonlocal alignment_count
        alignment_count += 1
        return {"binding_sha256": "a" * 64}

    vehicle_pose = base.GazeboModelPose(x=1.0, y=2.0, z=3.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)
    payload_pose = base.GazeboModelPose(x=1.0, y=2.0, z=3.12, qx=0.0, qy=0.0, qz=0.0, qw=1.0)

    async def sample(**_kwargs):
        return {"my_drone": vehicle_pose, "takeout_payload": payload_pose}

    client._align_payload_to_mount = align
    client._sample_named_gazebo_poses = sample

    with pytest.raises(RuntimeError, match="native joint event required"):
        asyncio.run(
            client.execute_payload_command(
                {
                    "protocol": "gazebo-transport",
                    "operation": "attach",
                    "topic": "/payload/attach",
                    "output_topic": "/payload/state",
                    "vehicle_model_name": "my_drone",
                    "payload_model_name": "takeout_payload",
                    "payload_mount_offset_model_m": [0.0, 0.0, 0.12],
                    "payload_mount_max_alignment_error_m": 0.02,
                    "payload_mount_binding_sha256": "a" * 64,
                }
            )
        )
    assert alignment_count == 1
    assert Node.last_publisher is not None
    assert Node.last_publisher.publish_count > 0
    assert client._payload_observers["/payload/state"]["detached"] is None
