import asyncio
import importlib.util
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from clock_fixtures import isolate_time

import dronedream_agent_core.gazebo_adapter as gazebo_adapter
from dronedream_agent_core.contracts import Px4GazeboRunEvidence, Px4Track
from dronedream_agent_core.gazebo_adapter import (
    SIMULATION_TAKEOFF_STABILITY_TIMEOUT_SECONDS,
    SimulationRuntimeError,
    _advance_closed_loop_progress_anchors,
    _bounded_executor_subphase_extension_allowed,
    _detach_payload_before_flight,
    _distance_to_polyline,
    _effective_local_speed_limit_mps,
    _executor_track_timeout_seconds,
    _executor_wall_timeout_seconds,
    _fresh_semantic_progress_window,
    _fresh_tracking_progress,
    _fresh_tracking_progress_state,
    _gazebo_image_model_payload,
    _gazebo_image_png,
    _gazebo_semantic_label_png,
    _heading_executor_arguments,
    _identity_correction_limit_m,
    _identity_estimator_reset_counter,
    _is_tolerated_landing_contact,
    _landing_confirmed,
    _live_camera_sdf,
    _load_controller_limits,
    _maximum_model_participation_gap_seconds,
    _model_navigation_controller_lookahead_m,
    _model_navigation_cycle_is_applied,
    _partition_runtime_cpus,
    _payload_spawn_spec,
    _phase_reference_polyline,
    _progress_extension_budget_seconds,
    _publish_native_terminal_lifecycle,
    _px4_sitl_actuator_output_absolute_maximum,
    _recent_progress_extension_allowed,
    _reference_route_deviation_enforced,
    _require_local_navigation_sensor_runtime,
    _require_local_policy_runtime,
    _resolve_controlled_vehicle_pose,
    _ros_overlay_environment,
    _runtime_evidence_writer_complete,
    _runtime_snapshot_writer_complete,
    _terminal_completion_grace_allowed,
    _tracking_corridor_policy,
    _tracking_identity_required,
    _update_goal_progress,
    _with_runtime_cpu_affinity,
)


def test_runtime_evidence_writer_gate_recounts_persisted_records() -> None:
    summary = {
        "schema_version": "dronedream.runtime-evidence-writer.v1",
        "complete": True,
        "closed": True,
        "issue_code": None,
        "rejected_count": 0,
        "pending_count": 0,
        "thread_finished": True,
        "thread_alive": False,
        "submitted_count": 42,
        "completed_count": 42,
        "artifacts": [{"path": "known.jsonl", "submitted_count": 42, "completed_count": 42}],
    }
    counts = {"known.jsonl": 42}
    assert _runtime_evidence_writer_complete(summary, record_count=42, artifact_counts=counts)
    assert not _runtime_evidence_writer_complete(summary, record_count=41, artifact_counts=counts)
    assert not _runtime_evidence_writer_complete(
        {**summary, "rejected_count": 1},
        record_count=42,
        artifact_counts=counts,
    )
    assert not _runtime_evidence_writer_complete(None, record_count=0, artifact_counts={})

    snapshot_summary = {
        "schema_version": "dronedream.runtime-snapshot-writer.v1",
        "complete": True,
        "issue_code": None,
        "rejected_count": 0,
        "pending_count": 0,
        "thread_finished": True,
        "thread_alive": False,
        "submitted_count": 42,
        "completed_count": 30,
        "superseded_count": 12,
    }
    assert _runtime_snapshot_writer_complete(snapshot_summary)
    assert not _runtime_snapshot_writer_complete(
        {**snapshot_summary, "superseded_count": 11}
    )


def test_px4_sitl_actuator_output_scale_is_read_from_model_contract(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "Tools" / "simulation" / "gz" / "models" / "x500"
    model_root.mkdir(parents=True)
    (model_root / "model.sdf").write_text(
        "<sdf><model><plugin><maxRotVelocity>1000</maxRotVelocity></plugin>"
        "<plugin><maxRotVelocity>1000.0</maxRotVelocity></plugin></model></sdf>",
        encoding="utf-8",
    )

    assert _px4_sitl_actuator_output_absolute_maximum(tmp_path, "x500") == 1000.0


def test_px4_sitl_actuator_output_scale_rejects_mixed_propulsion(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "Tools" / "simulation" / "gz" / "models" / "vtol"
    model_root.mkdir(parents=True)
    (model_root / "model.sdf").write_text(
        "<sdf><model><plugin><maxRotVelocity>1000</maxRotVelocity></plugin>"
        "<plugin><maxRotVelocity>1500</maxRotVelocity></plugin></model></sdf>",
        encoding="utf-8",
    )

    assert _px4_sitl_actuator_output_absolute_maximum(tmp_path, "vtol") is None


def test_heading_executor_arguments_are_explicit_and_bounded() -> None:
    assert _heading_executor_arguments("measured-hold", 20.0) == [
        "--heading-policy",
        "measured-hold",
        "--maximum-yaw-rate-deg-s",
        "20",
    ]
    assert _heading_executor_arguments("route-tangent-relative", 12.5) == [
        "--heading-policy",
        "route-tangent-relative",
        "--maximum-yaw-rate-deg-s",
        "12.5",
    ]
    with pytest.raises(ValueError, match="unsupported PX4 heading policy"):
        _heading_executor_arguments("unbounded", 20.0)
    for invalid_rate in (0.0, 90.1, math.inf, math.nan):
        with pytest.raises(ValueError, match="maximum yaw rate"):
            _heading_executor_arguments("route-tangent-relative", invalid_rate)


def test_simulation_takeoff_budget_preserves_strict_stability_gate() -> None:
    # Recent qualified runs needed 73-84 seconds to satisfy the unchanged
    # position, speed, and five-second continuous-hover envelope.  Keep a
    # deterministic margin for the same PX4 controller under host load.
    assert SIMULATION_TAKEOFF_STABILITY_TIMEOUT_SECONDS == 180.0


def test_local_policy_runtime_preflight_uses_active_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def successful_probe(command, *, env, timeout):
        captured.update(command=command, env=env, timeout=timeout)
        return SimpleNamespace(returncode=0, stdout="1.29.0\n", stderr="")

    monkeypatch.setattr(gazebo_adapter, "_run", successful_probe)
    _require_local_policy_runtime()

    assert captured["command"] == [
        sys.executable,
        "-c",
        "import onnxruntime; print(onnxruntime.__version__)",
    ]
    assert captured["timeout"] == 15


def test_local_policy_runtime_preflight_fails_before_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gazebo_adapter,
        "_run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'onnxruntime'\n",
        ),
    )

    with pytest.raises(SimulationRuntimeError, match="LOCAL_POLICY_RUNTIME_UNAVAILABLE"):
        _require_local_policy_runtime()


def test_requested_local_navigation_requires_depth_worker_before_flight() -> None:
    with pytest.raises(
        SimulationRuntimeError,
        match="LOCAL_NAVIGATION_SENSOR_RUNTIME_UNAVAILABLE",
    ):
        _require_local_navigation_sensor_runtime(
            local_navigation_provider="local-policy",
            depth_safety_supported=False,
        )

    _require_local_navigation_sensor_runtime(
        local_navigation_provider=None,
        depth_safety_supported=False,
    )
    _require_local_navigation_sensor_runtime(
        local_navigation_provider="local-policy",
        depth_safety_supported=True,
    )


def test_runtime_cpu_partition_reserves_two_cpus_for_local_models() -> None:
    general, local_model = _partition_runtime_cpus(
        tuple(range(12)),
        local_policy_enabled=True,
    )

    assert general == tuple(range(10))
    assert local_model == (10, 11)
    assert set(general).isdisjoint(local_model)


def test_runtime_cpu_partition_reserves_two_complete_smt_cores_on_large_hosts() -> None:
    general, local_model = _partition_runtime_cpus(
        tuple(range(32)),
        local_policy_enabled=True,
    )

    assert general == tuple(range(28))
    assert local_model == (28, 29, 30, 31)
    assert set(general).isdisjoint(local_model)


def test_runtime_cpu_partition_preserves_small_or_non_model_hosts() -> None:
    assert _partition_runtime_cpus(
        (1, 3, 5),
        local_policy_enabled=True,
    ) == ((1, 3, 5), ())
    assert _partition_runtime_cpus(
        (1, 3, 5, 7),
        local_policy_enabled=False,
    ) == ((1, 3, 5, 7), ())


def test_runtime_cpu_affinity_wraps_command_with_bound_cpu_set() -> None:
    plan = {
        "enabled": True,
        "taskset_path": "/usr/bin/taskset",
        "general_cpu_ids": [0, 2, 4],
        "local_model_cpu_ids": [6, 8],
    }

    assert _with_runtime_cpu_affinity(
        ["python", "worker.py"],
        plan=plan,
        local_model=True,
    ) == ["/usr/bin/taskset", "--cpu-list", "6,8", "python", "worker.py"]
    assert _with_runtime_cpu_affinity(
        ["gz", "sim"],
        plan=plan,
        local_model=False,
    ) == ["/usr/bin/taskset", "--cpu-list", "0,2,4", "gz", "sim"]


def test_reference_route_deviation_yields_to_required_model_path_authority() -> None:
    assert _reference_route_deviation_enforced(
        phase="TRACK",
        goal_observed=False,
        model_control_authority_required=False,
    )
    assert not _reference_route_deviation_enforced(
        phase="TRACK",
        goal_observed=False,
        model_control_authority_required=True,
    )


@pytest.mark.parametrize("phase", [None, "PREFLIGHT", "TAKEOFF"])
def test_ground_and_climb_do_not_count_as_closed_mission_departure(phase):
    launch = (-10., -2., 2.)
    assert _update_goal_progress(center=(-10., -2., .217), start=launch, goal=launch,
        departure_observed=False, phase=phase) == (False, False)
    assert _update_goal_progress(center=launch, start=launch, goal=launch,
        departure_observed=True, phase=phase) == (True, False)


def test_phase_monitor_bounds_ground_and_climb_without_disabling_deviation():
    ground, launch = (-10., -2., .217), (-10., -2., 2.)
    route = [launch, (-9., -2., 2.), launch]
    for phase in (None, "PREFLIGHT"):
        reference = _phase_reference_polyline(phase=phase, spawn_center=ground, route=route)
        assert _distance_to_polyline(ground, reference) == 0.
        assert _distance_to_polyline(launch, reference) > 1.
    reference = _phase_reference_polyline(phase="TAKEOFF", spawn_center=ground, route=route)
    assert _distance_to_polyline((-10., -2., 1.), reference) == pytest.approx(0., abs=1e-12)
    assert _distance_to_polyline((-8.9, -2., 1.), reference) > 1.
    for phase in ("TRACK", "WAYPOINT_SETTLE", "invalid"):
        reference = _phase_reference_polyline(phase=phase, spawn_center=ground, route=route)
        assert reference == route
        assert _distance_to_polyline(ground, reference) > 1.
    with pytest.raises(ValueError):
        _phase_reference_polyline(phase="TAKEOFF", spawn_center=ground, route=[])


def test_model_navigation_applied_motion_recognizes_direct_pilot_control() -> None:
    call_id = "model-direct"
    assert _model_navigation_cycle_is_applied(
        {
            "model_action": "pilot-control",
            "model_call_id": call_id,
            "selected_candidate_id": None,
            "selected_metric_path_sha256": "a" * 64,
            "controller_target_m": {"x": 4.0, "y": 2.0, "z": 1.0},
        },
        call_by_id={
            call_id: {
                "local_expert_trace": {
                    "navigation_action": "pilot-control",
                    "pilot_control": {
                        "forward": 0.4,
                        "right": -0.1,
                        "up": 0.05,
                        "yaw": 0.0,
                    },
                }
            }
        },
    )


def test_model_navigation_applied_motion_rejects_unbound_pilot_control() -> None:
    assert not _model_navigation_cycle_is_applied(
        {
            "model_action": "pilot-control",
            "model_call_id": "missing",
            "selected_candidate_id": None,
            "selected_metric_path_sha256": "a" * 64,
            "controller_target_m": {"x": 4.0, "y": 2.0, "z": 1.0},
        },
        call_by_id={},
    )
    assert not _reference_route_deviation_enforced(
        phase="LANDING",
        goal_observed=False,
        model_control_authority_required=False,
    )


def test_model_navigation_lookahead_supports_cruise_without_becoming_unbounded() -> None:
    assert _model_navigation_controller_lookahead_m(maximum_speed_mps=0.2) == 0.25
    assert _model_navigation_controller_lookahead_m(maximum_speed_mps=1.2) == pytest.approx(0.9)
    assert _model_navigation_controller_lookahead_m(maximum_speed_mps=4.0) == 1.0

    with pytest.raises(SimulationRuntimeError, match="lookahead inputs"):
        _model_navigation_controller_lookahead_m(maximum_speed_mps=0.0)


def test_local_safety_speed_is_bounded_by_qualified_track() -> None:
    track = Px4Track.model_validate(
        {
            "points": [
                {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 1.0,
                    "phase": "launch",
                    "speed_limit_mps": 0.34,
                },
                {
                    "x": 1.0,
                    "y": 0.0,
                    "z": 1.0,
                    "phase": "transit",
                    "speed_limit_mps": 0.28,
                },
            ],
            "source_world_points": [
                {"east_m": 0.0, "north_m": 0.0, "up_m": 1.0},
                {"east_m": 1.0, "north_m": 0.0, "up_m": 1.0},
            ],
            "coordinate_contract": {
                "model_root_world_enu_m": [0.0, 0.0, 0.0],
                "collision_center_offset_model_m": [0.0, 0.0, 0.228],
            },
            "waypoint_hold_seconds": 0.4,
        }
    )

    assert _effective_local_speed_limit_mps(
        controller_speed_limit_mps=1.2, track=track
    ) == pytest.approx(0.34)


def test_controller_runtime_ignores_retired_gain_shaped_fields(tmp_path: Path) -> None:
    controller_path = tmp_path / "controller_params.json"
    controller_path.write_text(
        json.dumps(
            {
                "vel_limit": 1.2,
                "accel_limit": 0.8,
                "kp_xy": -999.0,
                "ki_xy": "not-a-controller-value",
                "kd_xy": None,
                "disturbance_rejection": 999.0,
            }
        ),
        encoding="utf-8",
    )

    assert _load_controller_limits(controller_path) == pytest.approx((1.2, 0.8))
    executor = _load_px4_executor_module()
    params = executor.load_controller_params(controller_path)
    assert params == executor.ControllerParams(vel_limit=1.2, accel_limit=0.8)
    assert not hasattr(params, "kp_xy")
    assert not hasattr(params, "disturbance_rejection")


def test_controller_runtime_rejects_missing_active_limits(tmp_path: Path) -> None:
    controller_path = tmp_path / "controller_params.json"
    controller_path.write_text(json.dumps({"vel_limit": 1.2}), encoding="utf-8")

    with pytest.raises(SimulationRuntimeError, match="accel_limit"):
        _load_controller_limits(controller_path)
    executor = _load_px4_executor_module()
    with pytest.raises(ValueError, match="missing active limits"):
        executor.load_controller_params(controller_path)


def test_ros_overlay_environment_preserves_native_library_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = tmp_path / "install" / "setup.bash"
    setup.parent.mkdir(parents=True)
    setup.write_text("# test overlay\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout=(
                b"BASE_TOKEN=kept\0"
                b"AMENT_PREFIX_PATH=/qualified/install\0"
                b"LD_LIBRARY_PATH=/qualified/install/lib\0"
            ),
            stderr=b"",
        )

    monkeypatch.setattr(gazebo_adapter.subprocess, "run", fake_run)

    result = _ros_overlay_environment(
        ros_workspace=tmp_path,
        env={"BASE_TOKEN": "kept"},
    )

    assert result["AMENT_PREFIX_PATH"] == "/qualified/install"
    assert result["LD_LIBRARY_PATH"] == "/qualified/install/lib"
    assert captured["env"] == {"BASE_TOKEN": "kept"}
    assert captured["command"][-1] == str(setup)


def test_wall_timeout_extension_requires_fresh_closed_loop_progress(tmp_path: Path) -> None:
    progress = tmp_path / "closed-loop-tracking.json"
    progress.write_text(
        json.dumps(
            {
                "state": "recovering",
                "schedule_index": 312,
                "updated_at_unix_ms": 10_000,
            }
        ),
        encoding="utf-8",
    )

    assert _fresh_tracking_progress(progress, now_unix_ms=12_000) == 312
    assert _fresh_tracking_progress(progress, now_unix_ms=16_000) is None

    progress.write_text(
        json.dumps(
            {
                "state": "model-progress-hold",
                "schedule_index": 312,
                "navigation_goal_id": "source-waypoint-0004",
                "model_goal_distance_m": 7.25,
                "observed_world_collision_center_m": {"x": 1.0, "y": 2.0, "z": 3.0},
                "model_navigation_authorized": True,
                "updated_at_unix_ms": 10_000,
            }
        ),
        encoding="utf-8",
    )
    semantic_progress = _fresh_tracking_progress_state(progress, now_unix_ms=12_000)
    assert semantic_progress is not None
    assert semantic_progress[:2] == (312, "source-waypoint-0004")
    assert semantic_progress[2] == pytest.approx(7.25)
    assert semantic_progress[3] == pytest.approx((1.0, 2.0, 3.0))
    assert semantic_progress[4] is True

    progress.write_text("{}", encoding="utf-8")
    assert _fresh_tracking_progress(progress, now_unix_ms=12_000) is None


def test_semantic_progress_window_survives_latest_authority_hold_aliasing(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "model-semantic-progress-window.json"
    progress.write_text(
        json.dumps(
            {
                "navigation_goal_id": "source-waypoint-0025",
                "best_model_goal_distance_m": 10.92,
                "progress_revision": 3333,
                "authorized_schedule_revision": 7259,
                "updated_at_unix_ms": 10_000,
            }
        ),
        encoding="utf-8",
    )

    assert _fresh_semantic_progress_window(progress, now_unix_ms=12_000) == (
        "source-waypoint-0025",
        10.92,
        3333,
        7259,
    )
    assert _fresh_semantic_progress_window(progress, now_unix_ms=16_000) is None


def test_wall_timeout_progress_accepts_model_authorized_motion_at_fixed_schedule_index() -> None:
    initial = _advance_closed_loop_progress_anchors(
        (6482, "pickup", 8.5, (40.0, 0.0, 2.0), True),
        last_schedule_index=6482,
        last_goal_id="pickup",
        semantic_anchor_m=8.5,
        physical_anchor_m=(40.0, 0.0, 2.0),
    )
    assert initial[0] is False

    advanced = _advance_closed_loop_progress_anchors(
        (6482, "pickup", 8.42, (40.08, 0.0, 2.0), True),
        last_schedule_index=initial[2],
        last_goal_id=initial[3],
        semantic_anchor_m=initial[4],
        physical_anchor_m=initial[5],
    )
    assert advanced[0] is True
    assert advanced[1] == "semantic_goal_distance_decreased"
    assert advanced[2] == 6482
    assert advanced[4] == pytest.approx(8.42)


def test_wall_timeout_progress_rejects_unproductive_fixed_index_telemetry() -> None:
    observed = _advance_closed_loop_progress_anchors(
        (6482, "pickup", 8.5, (40.02, 0.0, 2.0), False),
        last_schedule_index=6482,
        last_goal_id="pickup",
        semantic_anchor_m=8.5,
        physical_anchor_m=(40.0, 0.0, 2.0),
    )
    assert observed[0] is False
    assert observed[1] is None


def test_wall_timeout_keeps_recent_progress_across_transient_model_hold() -> None:
    assert _recent_progress_extension_allowed(
        deadline_progress_state=None,
        progress_evidence_revision=12,
        last_extended_progress_revision=11,
        last_progress_observed_at=100.0,
        now=109.0,
    )
    assert _recent_progress_extension_allowed(
        deadline_progress_state=None,
        progress_evidence_revision=13,
        last_extended_progress_revision=12,
        last_progress_observed_at=100.0,
        now=159.0,
    )


def test_wall_timeout_rejects_stale_or_already_consumed_progress() -> None:
    assert not _recent_progress_extension_allowed(
        deadline_progress_state=None,
        progress_evidence_revision=12,
        last_extended_progress_revision=11,
        last_progress_observed_at=100.0,
        now=161.0,
    )
    assert not _recent_progress_extension_allowed(
        deadline_progress_state=(1, "goal", 3.0, None, True),
        progress_evidence_revision=12,
        last_extended_progress_revision=12,
        last_progress_observed_at=100.0,
        now=101.0,
    )


def test_wall_timeout_recognizes_only_finite_executor_subphases() -> None:
    assert _bounded_executor_subphase_extension_allowed("WAYPOINT_SETTLE")
    assert _bounded_executor_subphase_extension_allowed("CHECKPOINT")
    assert _bounded_executor_subphase_extension_allowed("ACTION")
    assert not _bounded_executor_subphase_extension_allowed("TRACK")
    assert not _bounded_executor_subphase_extension_allowed("MODEL_AUTHORITY_HOLD")
    assert not _bounded_executor_subphase_extension_allowed(None)


def test_tracking_identity_gate_closes_at_terminal_flight_phase() -> None:
    assert _tracking_identity_required(None)
    assert not _tracking_identity_required("PREFLIGHT")
    assert _tracking_identity_required("TAKEOFF")
    assert _tracking_identity_required("TRACK")
    assert not _tracking_identity_required("LANDING")
    assert not _tracking_identity_required("LANDED")
    assert not _tracking_identity_required("COMPLETE")
    assert not _tracking_identity_required("FAILED")


def test_terminal_completion_grace_is_finite_and_phase_scoped() -> None:
    assert not _terminal_completion_grace_allowed("TRACK", already_used=False)
    assert _terminal_completion_grace_allowed("LANDING", already_used=False)
    assert _terminal_completion_grace_allowed("COMPLETE", already_used=False)
    assert not _terminal_completion_grace_allowed("LANDING", already_used=True)


def test_progress_extension_budget_scales_continuously_for_low_rtf_simulations() -> None:
    assert (
        _progress_extension_budget_seconds(
            estimated_flight_seconds=12.0,
            track_timeout_seconds=180.0,
        )
        == 900.0
    )
    assert _progress_extension_budget_seconds(
        estimated_flight_seconds=59.5,
        track_timeout_seconds=180.0,
    ) == pytest.approx(4_462.5)
    assert _progress_extension_budget_seconds(
        estimated_flight_seconds=60.5,
        track_timeout_seconds=180.0,
    ) == pytest.approx(4_537.5)
    assert (
        _progress_extension_budget_seconds(
            estimated_flight_seconds=288.0,
            track_timeout_seconds=579.0,
        )
        == 21_600.0
    )
    assert (
        _executor_track_timeout_seconds(
            estimated_flight_seconds=12.0,
            track_timeout_seconds=180.0,
        )
        == 1_080.0
    )
    assert (
        _executor_track_timeout_seconds(
            estimated_flight_seconds=288.0,
            track_timeout_seconds=579.0,
        )
        == 21_600.0
    )


def test_parent_wall_timeout_cannot_expire_before_child_executor_budget() -> None:
    assert _executor_wall_timeout_seconds(executor_track_timeout_seconds=1_080.0) == 1_350.0
    assert _executor_wall_timeout_seconds(executor_track_timeout_seconds=21_600.0) == 21_870.0


def test_model_participation_gap_includes_active_flight_boundaries() -> None:
    calls = [
        {"created_at": "1970-01-01T00:00:12Z"},
        {"created_at": "1970-01-01T00:00:24+00:00"},
    ]
    assert (
        _maximum_model_participation_gap_seconds(
            active_command_timestamps_ms=[10_000, 40_000],
            model_call_records=calls,
        )
        == 16.0
    )


def _load_px4_executor_module():
    script = Path(__file__).parents[1] / "runtime" / "px4_offboard_track_executor.py"
    spec = importlib.util.spec_from_file_location("dronedream_test_px4_executor", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_live_camera_frame_is_encoded_as_browser_png() -> None:
    class Frame:
        width = 2
        height = 1
        step = 6
        pixel_format_type = 3
        data = bytes((255, 0, 0, 0, 255, 0))

    payload = _gazebo_image_png(Frame())

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert payload.endswith(b"IEND\xaeB`\x82")


def test_live_camera_frame_can_be_encoded_at_exact_model_input_size() -> None:
    from PIL import Image

    class Frame:
        width = 4
        height = 2
        step = 12
        pixel_format_type = 3
        data = bytes((255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255) * 2)

    payload = _gazebo_image_png(Frame(), output_size=(2, 1))
    image = Image.open(io.BytesIO(payload))

    assert image.mode == "RGB"
    assert image.size == (2, 1)


def test_live_camera_model_payload_reuses_exact_rgb8_pixels() -> None:
    from PIL import Image

    class Frame:
        width = 2
        height = 1
        step = 6
        pixel_format_type = 8
        data = bytes((0, 0, 255, 0, 255, 0))

    payload, rgb8 = _gazebo_image_model_payload(Frame(), output_size=(2, 1))
    image = Image.open(io.BytesIO(payload))

    assert image.tobytes() == rgb8
    assert rgb8 == bytes((255, 0, 0, 0, 255, 0))


def test_semantic_camera_frame_preserves_discrete_class_identifiers() -> None:
    from PIL import Image

    class Frame:
        width = 3
        height = 1
        step = 9
        pixel_format_type = 3
        data = bytes((0, 0, 0, 0, 0, 3, 7, 7, 7))

    payload = _gazebo_semantic_label_png(Frame())
    image = Image.open(io.BytesIO(payload))

    assert image.mode == "L"
    assert list(image.getdata()) == [0, 3, 7]


def test_semantic_camera_rejects_classes_outside_the_reviewed_contract() -> None:
    class Frame:
        width = 1
        height = 1
        step = 3
        pixel_format_type = 3
        data = bytes((0, 0, 9))

    with pytest.raises(ValueError, match="unsupported class"):
        _gazebo_semantic_label_png(Frame())


def test_live_camera_is_positioned_from_the_real_route() -> None:
    sdf, camera = _live_camera_sdf([(0.0, 0.0, 2.0), (12.0, 8.0, 5.0)])

    assert "<topic>/dronedream/live/camera</topic>" in sdf
    assert "<width>1280</width><height>720</height>" in sdf
    assert camera[0] < 0.0
    assert camera[1] < 0.0
    assert camera[2] > 5.0
    assert "<pose>0 0 0 0 0 0</pose>" in sdf
    pose = re.search(r'<pose relative_to="__model__">0 0 0 0 ([^ ]+) ([^<]+)</pose>', sdf)
    assert pose is not None
    assert float(pose.group(1)) > 0.0


# 功能：控制优先只降低旁观画面负载，保留相同路线视角和真实图像主题。
def test_control_priority_live_camera_keeps_view_without_sensor_changes():
    points = [(0., 0., 2.), (12., 8., 5.)]
    original, old_pose = _live_camera_sdf(points)
    compact, pose = _live_camera_sdf(points, control_priority=True)
    assert pose == old_pose
    assert compact == original.replace('<update_rate>12</update_rate>', '<update_rate>6</update_rate>').replace(
        '<width>1280</width><height>720</height>', '<width>640</width><height>360</height>')
    with pytest.raises(ValueError, match='LIVE_CAMERA_PROFILE_INVALID'):
        _live_camera_sdf(points, control_priority=1)


def test_tracking_corridor_is_tighter_than_measured_route_clearance() -> None:
    policy = _tracking_corridor_policy(0.2921883192031175)

    assert policy["required_local_clearance_m"] >= 0.05
    assert policy["tracking_lag_limit_m"] < 0.12
    assert policy["tracking_rejoin_tolerance_m"] < policy["tracking_lag_limit_m"]
    spent = (
        policy["required_local_clearance_m"]
        + policy["reserved_uncertainty_m"]
        + policy["tracking_lag_limit_m"]
    )
    assert spent < policy["minimum_route_clearance_m"]


def test_full_route_rejoin_tolerance_uses_funded_corridor_without_stalling() -> None:
    policy = _tracking_corridor_policy(0.4185298992524089)

    assert policy["tracking_lag_limit_m"] == pytest.approx(0.1908239194019271)
    assert policy["tracking_rejoin_tolerance_m"] == pytest.approx(0.1717415274617344)
    assert policy["tracking_lag_limit_m"] - policy["tracking_rejoin_tolerance_m"] >= 0.015
    assert (
        policy["required_local_clearance_m"]
        + policy["reserved_uncertainty_m"]
        + policy["tracking_rejoin_tolerance_m"]
        < policy["minimum_route_clearance_m"]
    )


def test_narrow_route_rejoin_hysteresis_scales_with_funded_corridor() -> None:
    policy = _tracking_corridor_policy(0.16)

    assert policy["tracking_lag_limit_m"] == pytest.approx(0.0528)
    assert policy["tracking_rejoin_tolerance_m"] == pytest.approx(0.04752)
    assert policy["tracking_rejoin_tolerance_m"] == pytest.approx(
        policy["tracking_lag_limit_m"] * 0.90
    )
    assert (
        policy["required_local_clearance_m"]
        + policy["reserved_uncertainty_m"]
        + policy["tracking_lag_limit_m"]
        < policy["minimum_route_clearance_m"]
    )


def test_tracking_corridor_rejects_route_without_control_margin() -> None:
    with pytest.raises(SimulationRuntimeError, match="cannot fund"):
        _tracking_corridor_policy(0.10)


def test_identity_correction_budget_is_not_consumed_by_narrow_tracking_corridor() -> None:
    assert _identity_correction_limit_m(0.16) == pytest.approx(0.25)
    assert _identity_correction_limit_m(0.20) == pytest.approx(0.25)
    assert _identity_correction_limit_m(0.50) == pytest.approx(0.25)
    with pytest.raises(SimulationRuntimeError, match="positive route clearance"):
        _identity_correction_limit_m(0.0)


def test_identity_estimator_reset_counter_reads_native_odometry_epoch() -> None:
    payload = {
        "dynamics": {
            "sources": {
                "odometry": {"reset_counter": 14},
            }
        }
    }

    assert _identity_estimator_reset_counter(payload) == 14
    assert _identity_estimator_reset_counter({}) is None


@pytest.mark.parametrize("value", [True, -1, 256, "14"])
def test_identity_estimator_reset_counter_rejects_invalid_epochs(value: object) -> None:
    with pytest.raises(ValueError, match="reset counter"):
        _identity_estimator_reset_counter(
            {"dynamics": {"sources": {"odometry": {"reset_counter": value}}}}
        )


def test_controlled_vehicle_pose_prefers_moving_canonical_link() -> None:
    def pose(name: str, x: float, y: float, z: float) -> SimpleNamespace:
        return SimpleNamespace(
            name=name,
            position=SimpleNamespace(x=x, y=y, z=z),
            orientation=SimpleNamespace(w=1., x=0., y=0., z=0.),
        )

    resolved = _resolve_controlled_vehicle_pose(
        [
            pose("my_drone", -42.25, 15.3, 7.487),
            pose("base_link", 4.25, -3.3, 0.741),
        ],
        vehicle_name="my_drone",
        collision_center_offset_model_m=(0.0, 0.0, 0.228),
        frames={"vehicle_model_name": "my_drone", "canonical_link_name": "base_link",
                "canonical_at_rest": {"position_m": [0, 0, .24],
                                      "orientation_wxyz": [1, 0, 0, 0]},
                "collision_center_model_m": [0, 0, .228]},
    )

    assert resolved is not None
    model_root, selected_name, reference, candidates = resolved
    assert selected_name == "base_link"
    assert reference == "calibrated-collision-center-offset-adjusted"
    assert model_root == pytest.approx((-38.0, 12.0, 7.988))
    assert candidates == ["base_link", "my_drone"]


def test_payload_spawn_is_bound_to_the_attach_checkpoint(tmp_path: Path) -> None:
    vehicle_sdf = tmp_path / "model.sdf"
    vehicle_sdf.write_text(
        """<sdf version="1.9"><model name="my_drone">
        <plugin filename="gz-sim-detachable-joint-system" name="DetachableJoint">
        <child_model>takeout_payload</child_model>
        <attach_topic>/payload/attach</attach_topic>
        <detach_topic>/payload/detach</detach_topic>
        <output_topic>/payload/state</output_topic>
        </plugin></model></sdf>""",
        encoding="utf-8",
    )
    payload_sdf = tmp_path / "takeout-payload.sdf"
    payload_sdf.write_text(
        '<sdf version="1.9"><model name="takeout_payload"><link name="payload"/></model></sdf>',
        encoding="utf-8",
    )
    action_path = tmp_path / "runtime-actions.json"
    action_path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.runtime-action-execution.v1",
                "contract_id": "mission-1",
                "task_graph_sha256": "a" * 64,
                "domain_action_catalog_sha256": "b" * 64,
                "adapter_catalog_sha256": "c" * 64,
                "steps": [
                    {
                        "schema_version": "dronedream.runtime-action-step.v1",
                        "step_id": "action-001",
                        "task_id": "pickup",
                        "action": "pickup",
                        "target_node": "target",
                        "trigger": "checkpoint",
                        "checkpoint_id": "checkpoint-001",
                        "runtime_executor": "native.payload.pickup",
                        "adapter_id": "runtime.payload.gazebo-detachable-joint",
                        "driver": "gazebo-payload",
                        "parameters": {
                            "operation": "attach",
                            "payload_mount_offset_model_m": [0.0, 0.0, 0.12],
                        },
                        "required_success_evidence": ["payload attached"],
                        "max_attempts": 2,
                        "fallback": "return",
                        "timeout_seconds": 15,
                        "authority": "actuate",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "runtime-checkpoints.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.runtime-checkpoints.v1",
                "contract_id": "mission-1",
                "checkpoints": [
                    {
                        "checkpoint_id": "checkpoint-001",
                        "segment_id": "segment-001",
                        "task_id": "pickup",
                        "track_point_index": 1,
                        "target_node": "target",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    track = Px4Track.model_validate(
        {
            "coordinate_contract": {
                "model_root_world_enu_m": [0.0, 0.0, 0.0],
                "collision_center_offset_model_m": [0.0, 0.0, 0.228],
            },
            "points": [
                {"x": 0, "y": 0, "z": 1, "phase": "launch", "speed_limit_mps": 1},
                {"x": 1, "y": 2, "z": 1, "phase": "pickup", "speed_limit_mps": 1},
            ],
            "source_world_points": [
                {"east_m": 0, "north_m": 0, "up_m": 1},
                {"east_m": 48.5, "north_m": 1.25, "up_m": 1.35},
            ],
            "waypoint_hold_seconds": 0.4,
        }
    )

    spec = _payload_spawn_spec(
        vehicle_sdf=vehicle_sdf,
        runtime_action_contract_path=action_path,
        checkpoint_contract_path=checkpoint_path,
        track=track,
    )

    assert spec is not None
    assert spec["entity_name"] == "takeout_payload"
    assert spec["sdf_path"] == payload_sdf
    assert spec["pose"] == (48.5, 1.25, 1.242)
    assert spec["checkpoint_id"] == "checkpoint-001"
    assert spec["attach_topic"] == "/payload/attach"
    assert spec["detach_topic"] == "/payload/detach"
    assert spec["output_topic"] == "/payload/state"

    vehicle_sdf.write_text(
        """<sdf version="1.9"><model name="my_drone">
        <plugin filename="gz-sim-detachable-joint-system" name="DetachableJoint">
        <child_model>takeout_payload</child_model></plugin></model></sdf>""",
        encoding="utf-8",
    )
    with pytest.raises(SimulationRuntimeError, match="payload topics are missing"):
        _payload_spawn_spec(
            vehicle_sdf=vehicle_sdf,
            runtime_action_contract_path=action_path,
            checkpoint_contract_path=checkpoint_path,
            track=track,
        )


def test_payload_spawn_rejects_missing_current_action_mount_binding(tmp_path: Path) -> None:
    vehicle_sdf = tmp_path / "model.sdf"
    payload_sdf = tmp_path / "payload.sdf"
    vehicle_sdf.write_text(
        """<sdf version="1.9"><model name="my_drone">
        <plugin filename="gz-sim-detachable-joint-system" name="DetachableJoint">
        <child_model>takeout_payload</child_model><child_link>payload_link</child_link>
        <attach_topic>/payload/attach</attach_topic><detach_topic>/payload/detach</detach_topic>
        <output_topic>/payload/state</output_topic></plugin></model></sdf>""",
        encoding="utf-8",
    )
    payload_sdf.write_text(
        '<sdf version="1.9"><model name="takeout_payload"/></sdf>',
        encoding="utf-8",
    )
    action_path = tmp_path / "runtime-actions.json"
    action_path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.runtime-action-execution.v1",
                "contract_id": "mission-1",
                "task_graph_sha256": "a" * 64,
                "domain_action_catalog_sha256": "b" * 64,
                "adapter_catalog_sha256": "c" * 64,
                "steps": [
                    {
                        "schema_version": "dronedream.runtime-action-step.v1",
                        "step_id": "action-001",
                        "task_id": "pickup",
                        "action": "pickup",
                        "target_node": "target",
                        "trigger": "checkpoint",
                        "checkpoint_id": "checkpoint-001",
                        "runtime_executor": "native.payload.pickup",
                        "adapter_id": "runtime.payload.gazebo-detachable-joint",
                        "driver": "gazebo-payload",
                        "parameters": {"operation": "attach"},
                        "required_success_evidence": ["payload attached"],
                        "max_attempts": 1,
                        "fallback": "return",
                        "timeout_seconds": 15,
                        "authority": "actuate",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "runtime-checkpoints.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.runtime-checkpoints.v1",
                "contract_id": "mission-1",
                "checkpoints": [
                    {
                        "checkpoint_id": "checkpoint-001",
                        "segment_id": "segment-001",
                        "task_id": "pickup",
                        "track_point_index": 1,
                        "target_node": "target",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    track = Px4Track.model_validate(
        {
            "coordinate_contract": {
                "model_root_world_enu_m": [0.0, 0.0, 0.0],
                "collision_center_offset_model_m": [0.0, 0.0, 0.228],
            },
            "points": [
                {"x": 0, "y": 0, "z": 1, "phase": "launch", "speed_limit_mps": 1},
                {"x": 1, "y": 2, "z": 1, "phase": "pickup", "speed_limit_mps": 1},
            ],
            "source_world_points": [
                {"east_m": 0, "north_m": 0, "up_m": 1.0},
                {"east_m": 1, "north_m": 2, "up_m": 1.35},
            ],
            "waypoint_hold_seconds": 0.4,
        }
    )

    with pytest.raises(SimulationRuntimeError, match="action mount offset is invalid"):
        _payload_spawn_spec(
            vehicle_sdf=vehicle_sdf,
            runtime_action_contract_path=action_path,
            checkpoint_contract_path=checkpoint_path,
            track=track,
        )


@pytest.mark.parametrize(
    ("state", "accepted"),
    [
        ("data: true\n", True),
        ('data: "detached"\n', True),
        ("data: false\n", False),
        ('data: "attached"\n', False),
    ],
)
def test_payload_preflight_detach_requires_positive_state_readback(
    monkeypatch, state: str, accepted: bool
) -> None:
    class Listener:
        pid = 12345
        returncode: int | None = None

        def communicate(self, timeout=None):
            _ = timeout
            self.returncode = 0
            return state, ""

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

        # 功能：
        #   为卸载状态测试提供有界进程等待接口，不启动真实进程。
        # 输入：
        #   self：测试监听器；timeout：等待上限。
        # 输出：
        #   returncode：当前退出码。
        def wait(self, timeout=None):
            assert timeout is not None
            returncode = self.returncode
            return returncode

    commands: list[list[str]] = []

    def fake_run(argv, *, env, timeout):
        _ = env, timeout
        commands.append(argv)
        if argv[1:3] == ["topic", "-l"]:
            return subprocess.CompletedProcess(
                argv, 0, "/payload/attach\n/payload/detach\n/payload/state\n", ""
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(gazebo_adapter, "_run", fake_run)
    monkeypatch.setattr(gazebo_adapter.subprocess, "Popen", lambda *args, **kwargs: Listener())
    monkeypatch.setattr(gazebo_adapter.os, "kill", lambda pid, signal: None)
    isolate_time(monkeypatch, gazebo_adapter, sleep=lambda _: None)

    if accepted:
        evidence = _detach_payload_before_flight(
            "gz",
            detach_topic="/payload/detach",
            output_topic="/payload/state",
            env={},
        )
        assert evidence["confirmed"] is True
        assert evidence["detached"] is True
    else:
        with pytest.raises(SimulationRuntimeError, match="did not confirm a detached payload"):
            _detach_payload_before_flight(
                "gz",
                detach_topic="/payload/detach",
                output_topic="/payload/state",
                env={},
            )
    assert any(command[3] == "/payload/detach" for command in commands if len(command) > 3)


@pytest.mark.parametrize(
    ("state", "detached"),
    [
        ("data: true", True),
        ("data: false", False),
        ('data: "detached"', True),
        ('data: "attached"', False),
    ],
)
def test_runtime_payload_state_parser_supports_gazebo_harmonic_and_legacy(
    state: str, detached: bool
) -> None:
    executor = _load_px4_executor_module()

    assert executor._parse_gazebo_payload_detached_state(state) is detached


# 功能：
#   验证普通缓存值无法冒充原生载荷状态，缺少传输观察时必须失败。
# 输入：
#   monkeypatch：提供无原生状态消息的隔离订阅。
# 输出：
#   None：断言旧布尔缓存不会让状态读取直接成功。
def test_runtime_payload_state_rejects_unbound_boolean_cache(monkeypatch) -> None:
    executor = _load_px4_executor_module()
    client = object.__new__(executor.MavsdkOffboardClient)
    client._payload_detached = True
    class Node:
        def subscribe(self, *_args):
            return True

        def unsubscribe(self, *_args):
            return True

    monkeypatch.setattr(executor, "_gazebo_transport_bindings",
                        lambda: (Node, object, object, object, object, object))
    with pytest.raises(RuntimeError, match="unknown or invalid"):
        asyncio.run(client.sample_payload_state("/payload/state", 0.01))


def test_runtime_payload_mount_pose_parser_and_rotation_are_model_bound() -> None:
    executor = _load_px4_executor_module()
    raw = """
pose {
  name: "my_drone"
  position { x: 48.5 y: 1.25 z: 1.1 }
  orientation { x: 0 y: 0 z: 0.7071067811865476 w: 0.7071067811865476 }
}
pose {
  name: "takeout_payload"
  position { x: 48.5 y: 1.25 z: 0.08 }
  orientation { x: 0 y: 0 z: 0 w: 1 }
}
"""

    pose = executor._parse_gazebo_model_pose(raw, "my_drone")
    rotated = executor._rotate_gazebo_model_vector(pose, (1.0, 0.0, 0.12))

    assert pose.z == pytest.approx(1.1)
    assert rotated == pytest.approx((0.0, 1.0, 0.12), abs=1e-9)
    with pytest.raises(RuntimeError, match="did not contain model"):
        executor._parse_gazebo_model_pose(raw, "wrong_model")


def test_runtime_payload_mount_pose_parser_accepts_omitted_protobuf_zero_fields() -> None:
    executor = _load_px4_executor_module()
    raw = """
pose {
  name: "my_drone"
  position { z: 1.1 }
  orientation { w: 1 }
}
"""

    pose = executor._parse_gazebo_model_pose(raw, "my_drone")

    assert (pose.x, pose.y, pose.z) == pytest.approx((0.0, 0.0, 1.1))
    assert (pose.qx, pose.qy, pose.qz, pose.qw) == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_runtime_payload_mount_alignment_uses_vehicle_frame_and_records_move(
    monkeypatch,
) -> None:
    executor = _load_px4_executor_module()
    client = object.__new__(executor.MavsdkOffboardClient)
    vehicle_pose = executor.GazeboModelPose(
        x=10.0,
        y=20.0,
        z=3.0,
        qx=0.0,
        qy=0.0,
        qz=2**-0.5,
        qw=2**-0.5,
    )
    payload_pose = executor.GazeboModelPose(
        x=2.0,
        y=3.0,
        z=0.08,
        qx=0.0,
        qy=0.0,
        qz=0.0,
        qw=1.0,
    )
    moved: list[object] = []

    async def sample(**_kwargs):
        observed_payload = moved[-1] if moved else payload_pose
        return {"my_drone": vehicle_pose, "takeout_payload": observed_payload}

    async def set_pose(**kwargs):
        moved.append(kwargs["pose"])
        return {"accepted": True}

    client._sample_named_gazebo_poses = sample
    client._set_named_gazebo_pose = set_pose
    monkeypatch.setenv("PX4_GAZEBO_WORLD_NAME", "school")

    evidence = asyncio.run(
        client._align_payload_to_mount(
            {
                "vehicle_model_name": "my_drone",
                "payload_model_name": "takeout_payload",
                "payload_mount_offset_model_m": [0.1, 0.0, 0.12],
                "payload_mount_max_alignment_error_m": 0.02,
                "payload_mount_binding_sha256": "a" * 64,
            }
        )
    )

    assert len(moved) == 1
    assert (moved[0].x, moved[0].y, moved[0].z) == pytest.approx((10.0, 20.1, 3.12))
    assert evidence["payload_pose_before_world_enu"] == {"x": 2.0, "y": 3.0, "z": 0.08}
    assert evidence["set_pose_attempts"] == 1
    assert evidence["pre_attachment_pose_readback"]["accepted"] is True


def test_plain_qualification_does_not_require_optional_scenario_backend(monkeypatch) -> None:
    executor = _load_px4_executor_module()
    monkeypatch.delenv("PX4_TRIAL_SCENARIO_EFFECT_REQUEST_PATH", raising=False)
    monkeypatch.setattr(
        executor,
        "_load_scenario_effect_engine",
        lambda: (_ for _ in ()).throw(AssertionError("optional backend must stay unloaded")),
    )

    assert executor._load_runtime_effect_request() == (None, None, None)


# 功能：
#   验证发出挂接指令前不能耗尽最终容差，持续一厘米误差也须停止而不是勉强挂接。
# 输入：
#   monkeypatch：隔离真实设备与世界环境。
# 输出：
#   None：六次有界预对齐均不满足五毫米要求时拒绝。
def test_payload_alignment_reserves_final_mount_tolerance(monkeypatch):
    executor = _load_px4_executor_module()
    client = object.__new__(executor.MavsdkOffboardClient)
    vehicle = executor.GazeboModelPose(x=0, y=0, z=2, qx=0, qy=0, qz=0, qw=1)
    payload = executor.GazeboModelPose(x=0, y=0, z=2.11, qx=0, qy=0, qz=0, qw=1)
    moves = []
    async def sample(**kwargs):
        return {'drone': vehicle, 'payload': payload}
    async def set_pose(**kwargs):
        moves.append(kwargs)
        return {'accepted': True}
    monkeypatch.setenv('PX4_GAZEBO_WORLD_NAME', 'school')
    client._sample_named_gazebo_poses = sample
    client._set_named_gazebo_pose = set_pose
    with pytest.raises(RuntimeError, match='limit=0.005000m'):
        asyncio.run(client._align_payload_to_mount({'vehicle_model_name': 'drone',
            'payload_model_name': 'payload', 'payload_mount_offset_model_m': [0, 0, .12],
            'payload_mount_max_alignment_error_m': .02, 'payload_mount_binding_sha256': 'a' * 64}))
    assert len(moves) == 6


def test_setpoint_schedule_rebases_every_axis_to_measured_px4_origin() -> None:
    executor = _load_px4_executor_module()
    relative = [
        executor.Setpoint(north_m=0.0, east_m=0.0, down_m=-0.5, yaw_deg=0.0),
        executor.Setpoint(north_m=2.0, east_m=-3.0, down_m=-0.8, yaw_deg=90.0),
    ]
    origin = executor.PositionVelocityNed(
        north_m=-0.029,
        east_m=0.036,
        down_m=0.012,
        north_m_s=0.0,
        east_m_s=0.0,
        down_m_s=0.0,
    )

    rebased = executor.rebase_setpoint_schedule(relative, origin)

    assert rebased == [
        executor.Setpoint(north_m=-0.029, east_m=0.036, down_m=-0.488, yaw_deg=0.0),
        executor.Setpoint(north_m=1.971, east_m=-2.964, down_m=-0.788, yaw_deg=90.0),
    ]
    assert relative[0].north_m == 0.0
    assert relative[1].east_m == -3.0


def test_closed_route_requires_departure_before_return_goal_is_observed() -> None:
    start = (4.0, -3.0, 2.0)

    departed, observed = _update_goal_progress(
        center=start,
        start=start,
        goal=start,
        departure_observed=False,
    )
    assert departed is False
    assert observed is False

    departed, observed = _update_goal_progress(
        center=(5.0, -3.0, 2.0),
        start=start,
        goal=start,
        departure_observed=departed,
    )
    assert departed is True
    assert observed is False

    departed, observed = _update_goal_progress(
        center=(4.2, -3.0, 2.0),
        start=start,
        goal=start,
        departure_observed=departed,
    )
    assert departed is True
    assert observed is True


def test_failed_gazebo_evidence_preserves_typed_live_safety_event() -> None:
    evidence = Px4GazeboRunEvidence.model_validate(
        {
            "schema_version": "dronedream.generic-px4-gazebo-run.v1",
            "status": "failed",
            "world": "school_map_world",
            "vehicle": "my_drone",
            "gates": {
                "executor_completed": False,
                "offboard_timing_complete": False,
                "runtime_pose_samples_present": True,
                "ros_observations_present": True,
                "goal_observed": False,
                "landing_confirmed": True,
                "native_terminal_lifecycle_published": False,
                "no_live_abort": False,
                "px4_ulog_present": True,
                "static_route_clearance_bound": True,
                "local_policy_simulation_admission_recorded": True,
                "multimodal_dataset_summary_present": True,
                "multimodal_dataset_records_present": True,
                "multimodal_dataset_summary_consistent": True,
            },
            "measurements": {
                "pose_sample_count": 100,
                "ros_observation_rows": 50,
                "minimum_goal_distance_m": 3.0,
                "goal_departure_observed": True,
                "goal_observed_runtime": False,
                "live_safety_event": {
                    "schema_version": "dronedream.live-safety-event.v1",
                    "reason": "LIVE_STATIC_COLLISION",
                    "elapsed_s": 83.25,
                    "vehicle_collision_center_world_enu_m": {
                        "east_m": -42.73,
                        "north_m": 11.75,
                        "up_m": 7.63,
                    },
                    "clearance_m": -0.0017,
                    "collision_primitive": "teaching-floor-3-slab-west",
                    "route_deviation_m": 0.71,
                    "runtime_phase": "WAYPOINT_SETTLE",
                },
                "landing_state": "ON_GROUND",
                "abort_reason": "LIVE_STATIC_COLLISION",
                "executor_return_code": 1,
                "tolerated_landing_contact_samples": 0,
                "minimum_tolerated_landing_clearance_m": None,
                "model_navigation": {
                    "enabled": True,
                    "provider": "local-policy",
                    "cycle_count": 0,
                    "provider_call_count": 0,
                    "authorized_control_applied_count": 0,
                    "model_timeout_seconds": 10.0,
                    "cycle_period_seconds": 1.0,
                    "control_authority": "deterministic-local-safety",
                    "visual_enabled": True,
                    "visual_frame_count": 0,
                    "local_policy_selection": {
                        "map_sha256": "3" * 64,
                        "package_id": "dronedream.local-policy.test",
                        "package_sha256": "4" * 64,
                        "qualification_receipt_id": "simulation-admission-test",
                        "rollback_package_sha256": None,
                        "scope": "general",
                        "selection_reason": "general-simulation-admitted",
                        "sensor_contract_sha256": "5" * 64,
                        "simulation_only": True,
                        "vehicle_sha256": "6" * 64,
                        "development_payload_collection": False,
                        "flight_qualification_granted": None,
                    },
                },
                "multimodal_dataset": {
                    "enabled": False,
                    "flight_id": None,
                    "maximum_mib": None,
                    "record_count": 0,
                    "record_period_seconds": None,
                    "semantic_record_count": 0,
                    "semantic_topic": None,
                    "summary": None,
                },
                "external_abort_request": {
                    "reason": "NATIVE_RUNTIME_SAFETY_EVENT",
                    "world_paused": False,
                    "action": "safe_hold_then_land",
                    "contract_id": "mission-test",
                    "issue_codes": ["WATCHDOG_HEALTH_OR_DEADLINE_FAILURE"],
                    "observation_sequence": 55,
                    "observation_age_ms": 1275,
                    "deadline_ms": 1000,
                    "severity": 3,
                    "source": "dronedream_agent_ros.safety_event_guard",
                },
            },
            "artifacts": {
                "world_sha256": "a" * 64,
                "semantic_sha256": "b" * 64,
                "vehicle_sha256": "c" * 64,
                "route_sha256": "d" * 64,
                "track_sha256": "e" * 64,
                "clearance_sha256": "f" * 64,
                "controller_params_sha256": "0" * 64,
                "executor_sha256": "1" * 64,
                "local_policy_selection_sha256": "7" * 64,
                "multimodal_dataset_root": None,
                "multimodal_dataset_records_sha256": None,
                "semantic_label_map_sha256": None,
                "px4_ulogs": [{"path": "px4.ulg", "size_bytes": 1, "sha256": "2" * 64}],
                "ros_workspace": "/opt/dronedream/ros_ws",
            },
        }
    )

    assert evidence.measurements.live_safety_event is not None
    assert evidence.measurements.live_safety_event.runtime_phase == "WAYPOINT_SETTLE"
    assert evidence.measurements.external_abort_request is not None
    assert evidence.measurements.external_abort_request.observation_sequence == 55
    assert evidence.measurements.external_abort_request.observation_age_ms == 1275
    assert evidence.measurements.external_abort_request.deadline_ms == 1000
    assert evidence.gates.local_policy_simulation_admission_recorded is True
    assert evidence.measurements.model_navigation is not None
    assert evidence.measurements.model_navigation.local_policy_selection is not None
    assert (
        evidence.measurements.model_navigation.local_policy_selection.development_payload_collection
        is False
    )
    assert evidence.measurements.multimodal_dataset is not None
    assert evidence.gates.multimodal_dataset_summary_present is True
    assert evidence.gates.multimodal_dataset_records_present is True
    assert evidence.gates.multimodal_dataset_summary_consistent is True


def test_stop_at_waypoints_exposes_exact_pre_hold_arrival_indices() -> None:
    executor = _load_px4_executor_module()
    points = [
        executor.TrackPoint(x=0.0, y=0.0, z=1.0),
        executor.TrackPoint(x=1.0, y=0.0, z=1.0, speed_limit_mps=0.3),
        executor.TrackPoint(x=1.0, y=1.0, z=1.0, speed_limit_mps=0.3),
    ]
    params = executor.ControllerParams(
        vel_limit=1.0,
        accel_limit=1.0,
    )
    rate_hz = 10.0
    hold_samples = int(rate_hz * 0.4)

    plan = executor.build_setpoint_schedule_plan(
        points,
        params,
        rate_hz,
        stop_at_waypoints=True,
        waypoint_hold_seconds=0.4,
    )

    assert len(plan.waypoint_arrival_indices) == 2
    for index, waypoint in zip(plan.waypoint_arrival_indices, points[1:], strict=True):
        arrival = plan.schedule[index]
        expected = executor.enu_point_to_ned_setpoint(waypoint, yaw_deg=arrival.yaw_deg)
        assert arrival == expected
        assert plan.schedule[index + 1 : index + 1 + hold_samples] == [expected] * hold_samples


def test_model_navigation_goal_remains_stable_between_route_waypoint_arrivals() -> None:
    script = Path(__file__).parents[1] / "scripts" / "px4_checkpoint_executor.py"
    spec = importlib.util.spec_from_file_location(
        "dronedream_test_checkpoint_executor_navigation_goal", script
    )
    assert spec is not None and spec.loader is not None
    executor = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = executor
    spec.loader.exec_module(executor)
    track = Px4Track.model_validate(
        {
            "coordinate_contract": {
                "model_root_world_enu_m": [0.0, 0.0, 0.0],
                "collision_center_offset_model_m": [0.0, 0.0, 0.228],
            },
            "points": [
                {"x": 0, "y": 0, "z": 1, "phase": "launch", "speed_limit_mps": 1},
                {"x": 1, "y": 2, "z": 1, "phase": "transit", "speed_limit_mps": 1},
                {"x": 3, "y": 4, "z": 1, "phase": "pickup", "speed_limit_mps": 1},
            ],
            "source_world_points": [
                {"east_m": 0, "north_m": 0, "up_m": 1},
                {"east_m": 10, "north_m": 20, "up_m": 1.5},
                {"east_m": 30, "north_m": 40, "up_m": 1.5},
            ],
            "waypoint_hold_seconds": 0.4,
        }
    )

    first_goal, first_id = executor._navigation_goal_for_schedule(
        track=track,
        waypoint_arrival_indices=(25, 60),
        schedule_index=1,
    )
    same_goal, same_id = executor._navigation_goal_for_schedule(
        track=track,
        waypoint_arrival_indices=(25, 60),
        schedule_index=24,
    )
    next_goal, next_id = executor._navigation_goal_for_schedule(
        track=track,
        waypoint_arrival_indices=(25, 60),
        schedule_index=26,
    )

    assert first_goal == same_goal
    assert first_id == same_id == "source-waypoint-0001"
    assert first_goal.model_dump() == {"x": 10.0, "y": 20.0, "z": 1.5}
    assert next_id == "source-waypoint-0002"
    assert next_goal.model_dump() == {"x": 30.0, "y": 40.0, "z": 1.5}


def test_executor_streams_origin_rebased_schedule(monkeypatch, tmp_path: Path) -> None:
    executor = _load_px4_executor_module()
    client = executor.FakeOffboardClient()
    origin = executor.PositionVelocityNed(
        north_m=-0.029,
        east_m=0.036,
        down_m=0.012,
        north_m_s=0.0,
        east_m_s=0.0,
        down_m_s=0.0,
    )
    client.position_velocity_samples = [origin]
    client.heading_deg = 79.2
    relative = [
        executor.Setpoint(north_m=0.0, east_m=0.0, down_m=-0.5, yaw_deg=0.0),
        executor.Setpoint(north_m=1.0, east_m=-2.0, down_m=-0.5, yaw_deg=45.0),
    ]

    async def accept_takeoff(*_args, evidence, **_kwargs) -> None:
        evidence["status"] = "achieved"

    monkeypatch.setattr(executor, "_wait_for_takeoff_stability", accept_takeoff)
    timing_path = tmp_path / "timing.json"
    asyncio.run(
        executor.run_executor(
            client,
            relative,
            connection="udp://:14540",
            takeoff_timeout_seconds=1.0,
            track_timeout_seconds=2.0,
            rate_hz=100.0,
            land_after=False,
            log_path=tmp_path / "executor.log",
            timing_path=timing_path,
        )
    )

    assert client.setpoints == [
        executor.Setpoint(north_m=-0.029, east_m=0.036, down_m=0.012, yaw_deg=79.2),
        executor.Setpoint(north_m=-0.029, east_m=0.036, down_m=-0.488, yaw_deg=79.2),
        executor.Setpoint(north_m=0.971, east_m=-1.964, down_m=-0.488, yaw_deg=79.2),
    ]
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    rebase = timing["takeoff_gate"]["schedule_origin_rebase"]
    assert rebase["origin_ned"] == {
        "north_m": -0.029,
        "east_m": 0.036,
        "down_m": 0.012,
    }
    assert rebase["rebased_first_setpoint_ned"]["down_m"] == -0.488
    assert timing["takeoff_gate"]["heading_control"] == {
        "policy": "measured_prearm_heading_hold",
        "measured_heading_deg": 79.2,
        "measured_gazebo_body_heading_ned_deg": None,
        "px4_minus_gazebo_heading_offset_deg": None,
        "planned_yaw_min_deg": 0.0,
        "planned_yaw_max_deg": 45.0,
        "planned_first_yaw_deg": 0.0,
        "commanded_yaw_deg": 79.2,
        "maximum_yaw_rate_deg_s": None,
        "source_setpoint_count": 2,
        "commanded_setpoint_count": 2,
        "deliberate_yaw_requires_separate_qualified_action": True,
    }


def test_route_tangent_heading_is_relative_rate_limited_and_position_stationary() -> None:
    executor = _load_px4_executor_module()
    plan = executor.SetpointSchedulePlan(
        schedule=[
            executor.Setpoint(0.0, 0.0, -1.0, 0.0),
            executor.Setpoint(1.0, 0.0, -1.0, 90.0),
            executor.Setpoint(1.0, 1.0, -1.0, -90.0),
        ],
        track_start_index=0,
        track_end_index=2,
        waypoint_arrival_indices=(1, 2),
    )

    aligned = executor.align_setpoint_schedule_to_route_tangent(
        plan,
        measured_px4_heading_deg=79.0,
        measured_body_heading_ned_deg=0.0,
        rate_hz=10.0,
        maximum_yaw_rate_deg_s=20.0,
    )

    assert aligned.schedule[0].yaw_deg == pytest.approx(79.0)
    assert aligned.schedule[aligned.waypoint_arrival_indices[0]].north_m == 1.0
    assert aligned.schedule[aligned.waypoint_arrival_indices[1]].east_m == 1.0
    first_motion_index = next(
        index for index, point in enumerate(aligned.schedule) if point.north_m == 1.0
    )
    assert all(
        point.north_m == 0.0 and point.east_m == 0.0
        for point in aligned.schedule[1:first_motion_index]
    )
    for previous, current in zip(aligned.schedule, aligned.schedule[1:], strict=False):
        assert abs(
            executor._shortest_yaw_delta_deg(previous.yaw_deg, current.yaw_deg)
        ) <= 2.0 + 1e-9


def test_gazebo_body_heading_converts_enu_quaternion_to_ned() -> None:
    executor = _load_px4_executor_module()
    identity = executor.GazeboModelPose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    north_facing = executor.GazeboModelPose(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        math.sin(math.pi / 4.0),
        math.cos(math.pi / 4.0),
    )

    assert executor.gazebo_body_heading_ned_deg(identity) == pytest.approx(90.0)
    assert executor.gazebo_body_heading_ned_deg(north_facing) == pytest.approx(0.0)


def test_base_executor_publishes_landing_phase_before_land_command(
    monkeypatch, tmp_path: Path
) -> None:
    executor = _load_px4_executor_module()
    phase_path = tmp_path / "runtime-phase.json"

    class PhaseAwareClient(executor.FakeOffboardClient):
        phase_at_land: str | None = None

        async def land(self) -> None:
            self.phase_at_land = json.loads(phase_path.read_text(encoding="utf-8"))["phase"]
            await super().land()

    client = PhaseAwareClient()
    client.position_velocity_samples = [
        executor.PositionVelocityNed(
            north_m=0.0,
            east_m=0.0,
            down_m=0.0,
            north_m_s=0.0,
            east_m_s=0.0,
            down_m_s=0.0,
        )
    ]

    async def accept_takeoff(*_args, evidence, **_kwargs) -> None:
        evidence["status"] = "achieved"

    monkeypatch.setattr(executor, "_wait_for_takeoff_stability", accept_takeoff)
    asyncio.run(
        executor.run_executor(
            client,
            [executor.Setpoint(north_m=0.0, east_m=0.0, down_m=-0.5, yaw_deg=0.0)],
            connection="udp://:14540",
            takeoff_timeout_seconds=1.0,
            track_timeout_seconds=2.0,
            landing_timeout_seconds=1.0,
            rate_hz=100.0,
            land_after=True,
            log_path=tmp_path / "executor.log",
            runtime_phase_path=phase_path,
        )
    )

    assert client.phase_at_land == "LANDING"
    assert json.loads(phase_path.read_text(encoding="utf-8")) == {"phase": "COMPLETE"}


def test_landing_ground_contact_has_narrow_phase_surface_and_depth_boundary() -> None:
    assert _is_tolerated_landing_contact(
        phase="LANDING",
        primitive_name="school-map-ground",
        clearance_m=-0.0139,
    )
    assert not _is_tolerated_landing_contact(
        phase="TRACK",
        primitive_name="school-map-ground",
        clearance_m=-0.0139,
    )
    assert not _is_tolerated_landing_contact(
        phase="LANDING",
        primitive_name="campus-main-gate-header",
        clearance_m=-0.0139,
    )
    assert not _is_tolerated_landing_contact(
        phase="LANDING",
        primitive_name="school-map-ground",
        clearance_m=-0.0201,
    )
    assert _is_tolerated_landing_contact(
        phase="PREFLIGHT",
        primitive_name="floor-0-0",
        clearance_m=-0.0022,
    )
    assert _is_tolerated_landing_contact(
        phase="TAKEOFF",
        primitive_name="terrain-0",
        clearance_m=-0.0022,
    )
    assert not _is_tolerated_landing_contact(
        phase="PREFLIGHT",
        primitive_name="wall-0-0",
        clearance_m=-0.0022,
    )


def test_native_terminal_lifecycle_requires_px4_on_ground_evidence() -> None:
    accepted = {
        "cleanup": {
            "land": "confirmed_on_ground: telemetry",
            "landing_observation": {"state": "ON_GROUND"},
        }
    }
    assert _landing_confirmed(accepted)
    assert not _landing_confirmed({"cleanup": {"land": "requested"}})
    assert not _landing_confirmed(
        {
            "cleanup": {
                "land": "confirmed_on_ground: telemetry",
                "landing_observation": {"state": "IN_AIR"},
            }
        }
    )


def test_native_terminal_lifecycle_publication_is_contract_bound(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, *, env, timeout):
        captured["command"] = command
        captured["env"] = env
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(command, 0, "published", "")

    monkeypatch.setattr(gazebo_adapter, "_run", fake_run)
    receipt = _publish_native_terminal_lifecycle(
        contract_id="mission-contract-1",
        executor_return_code=0,
        env={"ROS_DOMAIN_ID": "74"},
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[3:6] == ["--once", "--wait-matching-subscriptions", "0"]
    payload = json.loads(command[-1])
    assert payload == {
        "contract_id": "mission-contract-1",
        "terminal_state": "ON_GROUND",
        "executor_return_code": 0,
        "landing_confirmed": True,
        "safe_to_stop_watchdog": True,
    }
    assert receipt["publisher_exit_code"] == 0


def test_native_terminal_lifecycle_preserves_failed_executor_outcome(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, *, env, timeout):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, "published", "")

    monkeypatch.setattr(gazebo_adapter, "_run", fake_run)

    receipt = _publish_native_terminal_lifecycle(
        contract_id="mission-contract-failed-but-landed",
        executor_return_code=2,
        env={"ROS_DOMAIN_ID": "74"},
    )

    command = captured["command"]
    assert isinstance(command, list)
    payload = json.loads(command[-1])
    assert payload["terminal_state"] == "ON_GROUND"
    assert payload["executor_return_code"] == 2
    assert payload["landing_confirmed"] is True
    assert payload["safe_to_stop_watchdog"] is True
    assert receipt["executor_return_code"] == 2
