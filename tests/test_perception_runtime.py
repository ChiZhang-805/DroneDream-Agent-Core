import hashlib
import math
import threading
import time
from types import SimpleNamespace

import pytest

import dronedream_agent_core.perception_runtime as perception_runtime
from dronedream_agent_core.contracts import (
    DynamicObstacleObservation,
    NormalizedPilotControl,
    OnboardPerceptionFrame,
    RangeRayObservation,
    TextNavigationDecision,
    Vector3,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.model_harness.model_port import ModelInvocationError
from dronedream_agent_core.perception_runtime import (
    EventDrivenIndoorNavigationCoordinator,
    RuntimePerceptionFusion,
    TextOnlyIndoorNavigationCoordinator,
    _bind_navigation_control_output_contract,
    _bounded_path_controller_target,
    _bounded_strategic_navigation_context,
    _revalidate_selected_navigation_path,
    request_text_navigation_decision,
    validate_text_navigation_decision,
)
from dronedream_agent_core.runtime_sensor_contracts import (
    RuntimeSensorContract,
    RuntimeSensorRegistry,
)


def test_revalidated_controller_lookahead_can_cross_multiple_voxels() -> None:
    target = _bounded_path_controller_target(
        path=(
            Vector3(x=0.0, y=0.0, z=0.0),
            Vector3(x=0.5, y=0.0, z=0.0),
            Vector3(x=0.5, y=1.0, z=0.0),
        ),
        current_position_m=Vector3(x=0.0, y=0.0, z=0.0),
        fallback_target_m=Vector3(x=0.5, y=1.0, z=0.0),
        resolution_m=0.25,
        maximum_step_m=0.75,
    )

    # It follows the revalidated polyline rather than cutting the corner, but
    # is no longer artificially limited to one 25 cm voxel.
    assert target == Vector3(x=0.5, y=0.25, z=0.0)
    distance = math.dist((0.0, 0.0, 0.0), tuple(target.model_dump().values()))
    assert 0.25 < distance <= 0.750001


def test_event_driven_navigation_rejects_unbounded_controller_lookahead() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )

    with pytest.raises(ValueError, match="revalidated lookahead bound"):
        EventDrivenIndoorNavigationCoordinator(
            fusion=fusion,
            port=SimpleNamespace(),
            required_clearance_m=0.0,
            maximum_controller_step_m=2.01,
        )


@pytest.mark.parametrize("field", [
    "maximum_stream_age_seconds", "maximum_localization_covariance_m2",
    "maximum_sensor_offset_m", "maximum_dynamic_speed_mps",
    "maximum_dynamic_acceleration_mps2", "maximum_dynamic_innovation_m",
])
@pytest.mark.parametrize("value", [math.nan, math.inf, True])
def test_fusion_rejects_nonphysical_limit_configuration(field, value):
    with pytest.raises(ValueError):
        RuntimePerceptionFusion(world=_world(), accepted_sensor_ids={"front-lidar"},
                                **{field: value})


@pytest.mark.parametrize("field", [
    "required_clearance_m", "candidate_speed_mps", "vehicle_radius_m", "vehicle_height_m",
    "maximum_decision_age_seconds", "control_lease_seconds", "maximum_revalidation_seconds",
    "maximum_snapshot_planning_seconds",
])
@pytest.mark.parametrize("value", [math.nan, math.inf, True])
def test_coordinator_rejects_nonphysical_configuration_before_starting_worker(field, value):
    fusion = RuntimePerceptionFusion(world=_world(), accepted_sensor_ids={"front-lidar"})
    coordinator = None
    try:
        with pytest.raises(ValueError):
            coordinator = EventDrivenIndoorNavigationCoordinator(
                fusion=fusion, port=SimpleNamespace(), **{"required_clearance_m": 0, field: value},
            )
    finally:
        if coordinator is not None:
            coordinator.close()


def test_event_driven_navigation_rejects_nonpositive_revalidation_deadline() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )

    with pytest.raises(ValueError, match="revalidation deadline"):
        EventDrivenIndoorNavigationCoordinator(
            fusion=fusion,
            port=SimpleNamespace(),
            required_clearance_m=0.0,
            maximum_revalidation_seconds=0.0,
        )


def test_event_driven_navigation_rejects_nonpositive_snapshot_planning_deadline() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )

    with pytest.raises(ValueError, match="snapshot planning deadline"):
        EventDrivenIndoorNavigationCoordinator(
            fusion=fusion,
            port=SimpleNamespace(),
            required_clearance_m=0.0,
            maximum_snapshot_planning_seconds=0.0,
        )


def test_event_driven_navigation_pins_only_its_model_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[tuple[int, set[int]]] = []
    monkeypatch.setattr(
        perception_runtime.os,
        "sched_setaffinity",
        lambda process_id, cpu_ids: applied.append((process_id, set(cpu_ids))),
        raising=False,
    )
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=SimpleNamespace(),
        required_clearance_m=0.0,
        model_worker_cpu_ids=(7, 9),
    )

    try:
        worker_name = coordinator._executor.submit(  # noqa: SLF001 - concurrency contract
            lambda: threading.current_thread().name
        ).result(timeout=2.0)
    finally:
        coordinator._executor.shutdown(wait=True)  # noqa: SLF001

    assert worker_name.startswith("dronedream-local-navigation-model-")
    assert applied == [(0, {7, 9})]


def test_event_driven_navigation_rejects_duplicate_model_worker_cpu_ids() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )

    with pytest.raises(ValueError, match="must be unique"):
        EventDrivenIndoorNavigationCoordinator(
            fusion=fusion,
            port=SimpleNamespace(),
            required_clearance_m=0.0,
            model_worker_cpu_ids=(7, 7),
        )


def test_invocation_age_limit_includes_bounded_snapshot_planning() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=SimpleNamespace(invocation_timeout_seconds=0.25),
        required_clearance_m=0.0,
        maximum_decision_age_seconds=2.25,
        maximum_snapshot_planning_seconds=2.0,
    )
    try:
        assert coordinator._next_invocation_age_limit_seconds() == 3.25
    finally:
        coordinator.close()


def test_continuous_deadline_uses_original_sources_and_quarantines_late_job(monkeypatch):
    from concurrent.futures import Future

    from control_fixtures import complete_feature_snapshot

    fusion = RuntimePerceptionFusion(
        world=_world(), accepted_sensor_ids={"front-lidar"}, minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_000)
    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion, port=SimpleNamespace(invocation_timeout_seconds=60),
        required_clearance_m=0, maximum_decision_age_seconds=12, control_lease_seconds=15,
        control_output_mode="normalized-body-velocity",
    )
    future = Future()
    assert future.set_running_or_notify_cancel()
    monkeypatch.setattr(coordinator._executor, "submit", lambda *args, **kwargs: future)
    try:
        assert coordinator._next_invocation_age_limit_seconds() == .25
        assert coordinator.control_lease_seconds == .25
        result = coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=.25, z=.25), now_unix_ms=1_100,
            trigger="periodic",
            realtime_feature_snapshot=complete_feature_snapshot().model_dump(mode="json"),
        )
        assert result is None
        assert coordinator._pending.maximum_decision_age_seconds == .15
        receipt = coordinator.poll(now_unix_ms=1_251)
        assert receipt.hold_reason == "MODEL_NAVIGATION_INVOCATION_TIMEOUT"
        assert coordinator.request_pending
        assert coordinator._executor_generation == 0
        for _ in range(20):
            assert coordinator.schedule(
                goal_position_m=Vector3(x=3.75, y=.25, z=.25), now_unix_ms=1_255,
                trigger="periodic",
            ) is None
        assert coordinator._sequence == 1
        future.set_result(None)
        assert not coordinator.request_pending
        assert coordinator.poll(now_unix_ms=1_260) is None
    finally:
        coordinator.close()


def test_continuous_missing_encoders_cannot_request_action():
    fusion = RuntimePerceptionFusion(world=_world(), accepted_sensor_ids={"front-lidar"})
    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion, port=SimpleNamespace(), required_clearance_m=0,
        control_output_mode="normalized-body-velocity",
    )
    try:
        receipt = coordinator.schedule(goal_position_m=Vector3(x=1, y=0, z=0),
                                       now_unix_ms=1000, trigger="initial")
        assert receipt.hold_reason == "CONTINUOUS_CONTROL_SOURCE_NOT_READY"
        assert not coordinator.request_pending
    finally:
        coordinator.close()


@pytest.mark.parametrize("mode", ["normalized-body-velocity", "legacy-candidate-selection"])
def test_coordinator_rejects_contradictory_context_before_invocation(mode):
    from control_fixtures import complete_feature_snapshot

    fusion = RuntimePerceptionFusion(world=_world(), accepted_sensor_ids={"front-lidar"})
    fusion.ingest(_frame(), now_unix_ms=1020)
    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion, port=SimpleNamespace(), required_clearance_m=0., control_output_mode=mode,
    )
    other = ("legacy-candidate-selection" if mode == "normalized-body-velocity"
             else "normalized-body-velocity")
    try:
        receipt = coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=.25, z=.25), now_unix_ms=1020, trigger="initial",
            strategic_context={"task": {"local_navigation_output_mode": other}},
            realtime_feature_snapshot=complete_feature_snapshot().model_dump(mode="json"),
        )
        assert receipt.hold_reason == "MODEL_NAVIGATION_CONTEXT_MODE_MISMATCH"
        assert not coordinator.request_pending
    finally:
        coordinator.close()


def test_continuous_mode_rejects_candidate_result_without_revalidation(monkeypatch):
    from concurrent.futures import Future

    from control_fixtures import complete_feature_snapshot

    fusion = RuntimePerceptionFusion(world=_world(), accepted_sensor_ids={"front-lidar"})
    fusion.ingest(_frame(), now_unix_ms=1020)
    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion, port=SimpleNamespace(), required_clearance_m=0.,
        control_output_mode="normalized-body-velocity",
    )
    future = Future()
    monkeypatch.setattr(coordinator._executor, "submit", lambda *args, **kwargs: future)
    try:
        coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=.25, z=.25), now_unix_ms=1020, trigger="initial",
            realtime_feature_snapshot=complete_feature_snapshot().model_dump(mode="json"),
        )
        record = SimpleNamespace(call_id="mismatched-candidate")
        future.set_result(perception_runtime._NavigationWorkerResult(
            snapshot={"snapshot_sha256": "1" * 64},
            model_result=SimpleNamespace(record=record, artifact=TextNavigationDecision(
                snapshot_sha256="1" * 64, action="select-candidate",
                selected_candidate_id="old-path", rationale_summary="Unexpected legacy output.",
            )), selected_candidate={"candidate_id": "old-path"},
        ))
        receipt = coordinator.poll(now_unix_ms=1030)
        assert receipt.hold_reason == "MODEL_NAVIGATION_OUTPUT_MODE_MISMATCH"
        assert coordinator._pending_revalidation is None
        stored_record = coordinator.pop_model_call_record()
        assert stored_record == record
        assert stored_record is not record
    finally:
        coordinator.close()


def test_revalidation_passes_hard_deadline_to_fallback_planner(monkeypatch) -> None:
    world = _world()
    frame = _frame()
    observed_deadlines: list[float | None] = []

    monkeypatch.setattr(
        MetricVoxelMap,
        "path_clearance_valid",
        lambda self, *args, **kwargs: False,
    )

    def expired_plan(self, **kwargs):
        observed_deadlines.append(kwargs.get("deadline_monotonic_seconds"))
        raise ValueError("metric path planning deadline exceeded")

    monkeypatch.setattr(MetricVoxelMap, "plan_path", expired_plan)
    deadline = time.monotonic() + 0.5
    result = _revalidate_selected_navigation_path(
        world=world,
        frame=frame,
        selected={
            "endpoint_m": {"x": 1.25, "y": 0.25, "z": 0.25},
            "path_points_m": [
                {"x": 0.25, "y": 0.25, "z": 0.25},
                {"x": 1.25, "y": 0.25, "z": 0.25},
            ],
        },
        required_clearance_m=0.0,
        vehicle_radius_m=0.1,
        vehicle_height_m=0.1,
        maximum_controller_step_m=0.12,
        planning_deadline_monotonic_seconds=deadline,
    )

    assert observed_deadlines == [deadline]
    assert result.failure_reason == "AUTHORIZED_ENDPOINT_REVALIDATION_DEADLINE"


def test_strategic_navigation_context_is_bounded_and_hash_bound() -> None:
    bounded = _bounded_strategic_navigation_context(
        {
            "task": {"phase": "TRACK", "navigation_goal_id": "pickup"},
            "map": {"qualified_route_point_count": 43},
            "sensor_contract": {"unknown_space_policy": "blocked"},
            "vehicle": {"maximum_takeoff_mass_kg": 2.2},
            "payload": {
                "state": "loaded-stable",
                "observed_payload_mass_kg": 0.1,
                "dynamics": {"ready": True, "roll_deg": 1.5},
            },
        }
    )
    assert bounded["task"] == {
        "navigation_goal_id": "pickup",
        "phase": "TRACK",
    }
    assert bounded["payload"] == {
        "dynamics": {"ready": True, "roll_deg": 1.5},
        "observed_payload_mass_kg": 0.1,
        "state": "loaded-stable",
    }
    assert sha256_json(bounded) == sha256_json(
        _bounded_strategic_navigation_context(bounded)
    )
    with pytest.raises(ValueError, match="unsupported sections"):
        _bounded_strategic_navigation_context({"instructions": {"move": "forward"}})


def test_event_driven_navigation_includes_strategic_context_in_snapshot() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)
    snapshots: list[dict[str, object]] = []

    class Port:
        def call(self, **kwargs):
            snapshot = kwargs["input_artifact"]["text_navigation_snapshot"]
            snapshots.append(snapshot)
            candidate = snapshot["authorized_candidate_paths"][0]
            return SimpleNamespace(
                artifact=TextNavigationDecision(
                    snapshot_sha256=snapshot["snapshot_sha256"],
                    action="select-candidate",
                    selected_candidate_id=candidate["candidate_id"],
                    rationale_summary="Use the metric path with current task context.",
                ),
                record=SimpleNamespace(call_id="call-strategic-context"),
            )

    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=Port(),
        required_clearance_m=0.0,
    )
    try:
        coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_020,
            trigger="initial",
            strategic_context={
                "task": {"phase": "TRACK", "navigation_goal_id": "pickup"},
                "payload": {"state": "loaded-stable"},
            },
        )
        receipt = _wait_for_receipt(coordinator, now_unix_ms=1_030)
        assert receipt.model_call_id == "call-strategic-context"
        assert snapshots[0]["strategic_context"] == {
            "payload": {"state": "loaded-stable"},
            "task": {"navigation_goal_id": "pickup", "phase": "TRACK",
                     "local_navigation_output_mode": "legacy-candidate-selection"},
        }
        snapshot_without_hash = dict(snapshots[0])
        expected_hash = snapshot_without_hash.pop("snapshot_sha256")
        assert expected_hash == sha256_json(snapshot_without_hash)
    finally:
        coordinator.close()


def test_continuous_policy_snapshot_declares_axis_not_coordinate_authority() -> None:
    snapshot = {
        "model_authority": {
            "may_select": "candidate_id or safe hold",
        }
    }

    _bind_navigation_control_output_contract(
        snapshot,
        {
            "task": {
                "local_navigation_output_mode": "normalized-body-velocity",
            }
        },
    )

    authority = snapshot["model_authority"]
    assert authority["may_output"]["axes"] == ["forward", "right", "up", "yaw"]
    assert "metric coordinates" in authority["may_not_author"]
    assert "may_select" not in authority


def _wait_for_receipt(
    coordinator: EventDrivenIndoorNavigationCoordinator,
    *,
    now_unix_ms: int,
):
    for _ in range(100):
        receipt = coordinator.poll(now_unix_ms=now_unix_ms)
        if receipt is not None:
            return receipt
        time.sleep(0.005)
    raise AssertionError("model navigation future did not complete")


def _world() -> MetricVoxelMap:
    return MetricVoxelMap(
        resolution_m=0.5,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=4.0, y=1.0, z=1.0),
    )


def _frame(*, sequence: int = 1, observed_at_unix_ms: int = 1_000) -> OnboardPerceptionFrame:
    ray = RangeRayObservation(
        origin_m=Vector3(x=0.25, y=0.25, z=0.25),
        endpoint_m=Vector3(x=3.75, y=0.25, z=0.25),
        hit=False,
        confidence=0.95,
        observed_at_monotonic_seconds=observed_at_unix_ms / 1000.,
    )
    return OnboardPerceptionFrame(
        sensor_id="front-lidar",
        sequence=sequence,
        observed_at_unix_ms=observed_at_unix_ms,
        localization_position_m=Vector3(x=0.25, y=0.25, z=0.25),
        localization_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        localization_covariance_m2=0.01,
        range_rays=[ray.model_copy(deep=True) for _ in range(8)],
    )


def _tracked_frame(
    *, sequence: int, observed_at_unix_ms: int, obstacle_x: float, obstacle_speed_x: float
) -> OnboardPerceptionFrame:
    return _frame(sequence=sequence, observed_at_unix_ms=observed_at_unix_ms).model_copy(
        update={
            "dynamic_obstacles": [
                DynamicObstacleObservation(
                    obstacle_id="person-1",
                    position_m=Vector3(x=obstacle_x, y=0.75, z=0.5),
                    velocity_mps=Vector3(x=obstacle_speed_x, y=0.0, z=0.0),
                    radius_m=0.3,
                    height_m=1.8,
                    confidence=0.9,
                    age_seconds=0.0,
                )
            ]
        }
    )


def test_metric_sensor_frame_enables_text_only_navigation_snapshot() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )

    health = fusion.ingest(_frame(), now_unix_ms=1_020)
    observation = fusion.local_safety_observation(
        target_position_m=Vector3(x=3.75, y=0.25, z=0.25),
        now_unix_ms=1_020,
    )
    snapshot = fusion.text_navigation_snapshot(
        goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
        required_clearance_m=0.0,
        now_unix_ms=1_020,
    )

    assert health.stream_healthy is True
    assert observation.source == "onboard"
    assert observation.stream_healthy is True
    assert snapshot["visual_input_required"] is False
    assert snapshot["authorized_candidate_paths"]
    assert snapshot["perception_health"]["sensor_id"] == "front-lidar"
    candidate_id = snapshot["authorized_candidate_paths"][0]["candidate_id"]
    selected = validate_text_navigation_decision(
        snapshot,
        TextNavigationDecision(
            snapshot_sha256=snapshot["snapshot_sha256"],
            action="select-candidate",
            selected_candidate_id=candidate_id,
            rationale_summary="The goal path is metric validated and the stream is healthy.",
        ),
    )
    assert selected is not None
    assert selected["candidate_id"] == candidate_id


def test_sensor_registry_evidence_crosses_fusion_and_harness_snapshot() -> None:
    base_monotonic = time.monotonic() - .02
    frame = _frame().model_copy(
        update={
            "range_rays": [
                ray.model_copy(
                    update={"observed_at_monotonic_seconds": base_monotonic}
                )
                for ray in _frame().range_rays
            ]
        }
    )
    contract = RuntimeSensorContract(
        sensor_id="front-lidar",
        vehicle_id="dronedream.x500-depth",
        modality="lidar",
        coordinate_frame="world-enu",
        unit="meter-ray",
        calibration_sha256="a" * 64,
        intrinsics_sha256="b" * 64,
        extrinsics_sha256="c" * 64,
        maximum_sample_age_seconds=0.2,
        maximum_transport_latency_seconds=0.05,
        minimum_quality=0.5,
        required_for_motion=True,
    )
    registry = RuntimeSensorRegistry([contract])
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
        sensor_registry=registry,
    )

    health = fusion.ingest(
        frame,
        now_unix_ms=1_020,
        now_monotonic_seconds=base_monotonic + 0.02,
    )
    multimodal = fusion.multimodal_sensor_snapshot(
        now_monotonic_seconds=base_monotonic + 0.03
    )

    assert health.stream_healthy is True
    assert multimodal is not None
    assert multimodal.ready_for_motion is True
    assert multimodal.active_sensor_ids == ["front-lidar"]
    assert multimodal.statuses[0].payload_sha256 == sha256_json(frame)

    captured: list[dict[str, object]] = []

    class Port:
        def call(self, **kwargs):
            snapshot = kwargs["input_artifact"]["text_navigation_snapshot"]
            captured.append(snapshot)
            candidate = snapshot["authorized_candidate_paths"][0]
            return SimpleNamespace(
                artifact=TextNavigationDecision(
                    snapshot_sha256=snapshot["snapshot_sha256"],
                    action="select-candidate",
                    selected_candidate_id=candidate["candidate_id"],
                    rationale_summary="Use the hash-bound multimodal sensor snapshot.",
                ),
                record=SimpleNamespace(call_id="call-sensor-envelope"),
            )

    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=Port(),
        required_clearance_m=0.0,
    )
    try:
        coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_030,
            trigger="initial",
        )
        receipt = _wait_for_receipt(coordinator, now_unix_ms=1_040)
        assert receipt.model_call_id == "call-sensor-envelope"
        captured_multimodal = captured[0]["multimodal_sensor_snapshot"]
        assert captured_multimodal["contract_set_sha256"] == (
            multimodal.contract_set_sha256
        )
        assert captured_multimodal["ready_for_motion"] is True
        assert captured_multimodal["statuses"][0]["payload_sha256"] == (
            sha256_json(frame)
        )
        without_hash = dict(captured[0])
        expected_hash = without_hash.pop("snapshot_sha256")
        assert expected_hash == sha256_json(without_hash)
    finally:
        coordinator.close()


def test_revalidation_retains_safe_path_continuity_for_same_endpoint() -> None:
    world = _world()
    fusion = RuntimePerceptionFusion(
        world=world,
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)
    frame = _frame(sequence=2, observed_at_unix_ms=1_030).model_copy(
        update={"localization_position_m": Vector3(x=0.75, y=0.25, z=0.25)}
    )
    endpoint = Vector3(x=3.75, y=0.25, z=0.25)
    preferred = (
        Vector3(x=0.25, y=0.25, z=0.25),
        Vector3(x=2.0, y=0.25, z=0.25),
        endpoint,
    )
    selected = {
        "endpoint_m": endpoint.model_dump(mode="json"),
        "path_points_m": [
            Vector3(x=0.25, y=0.75, z=0.25).model_dump(mode="json"),
            Vector3(x=2.0, y=0.75, z=0.25).model_dump(mode="json"),
            endpoint.model_dump(mode="json"),
        ],
    }

    result = _revalidate_selected_navigation_path(
        world=world,
        frame=frame,
        selected=selected,
        required_clearance_m=0.0,
        vehicle_radius_m=0.1,
        vehicle_height_m=0.1,
        maximum_controller_step_m=0.12,
        preferred_path=preferred,
    )

    assert result.failure_reason is None
    assert result.path_continuity_retained is True
    assert result.path[-1] == endpoint
    assert result.controller_target_m is not None
    assert result.controller_target_m.y == pytest.approx(0.25)


def test_revalidation_replaces_materially_longer_old_detour_near_endpoint() -> None:
    world = _world()
    world.seed_known_static_region(
        [
            (world.center_for((x, y, z)), False)
            for x in range(8)
            for y in range(2)
            for z in range(2)
        ],
        source_sha256="a" * 64,
    )
    current = Vector3(x=3.5, y=0.25, z=0.25)
    endpoint = Vector3(x=3.75, y=0.25, z=0.25)
    frame = _frame(sequence=2, observed_at_unix_ms=1_030).model_copy(
        update={"localization_position_m": current}
    )
    preferred = (
        current,
        Vector3(x=3.5, y=0.75, z=0.25),
        Vector3(x=3.75, y=0.75, z=0.25),
        endpoint,
    )
    selected = {
        "endpoint_m": endpoint.model_dump(mode="json"),
        "path_points_m": [
            current.model_dump(mode="json"),
            endpoint.model_dump(mode="json"),
        ],
    }

    result = _revalidate_selected_navigation_path(
        world=world,
        frame=frame,
        selected=selected,
        required_clearance_m=0.0,
        vehicle_radius_m=0.1,
        vehicle_height_m=0.1,
        maximum_controller_step_m=0.9,
        preferred_path=preferred,
    )

    assert result.failure_reason is None
    assert result.path_continuity_retained is False
    assert result.path == (current, endpoint)
    assert result.controller_target_m == endpoint


def test_revalidation_never_reuses_path_to_adjacent_voxel_endpoint() -> None:
    world = _world()
    world.seed_known_static_region(
        [
            (world.center_for((x, y, z)), False)
            for x in range(8)
            for y in range(2)
            for z in range(2)
        ],
        source_sha256="a" * 64,
    )
    current = Vector3(x=3.25, y=0.25, z=0.25)
    selected_endpoint = Vector3(x=3.75, y=0.25, z=0.25)
    old_endpoint = Vector3(x=3.50, y=0.25, z=0.25)
    frame = _frame(sequence=2, observed_at_unix_ms=1_030).model_copy(
        update={"localization_position_m": current}
    )
    selected = {
        "endpoint_m": selected_endpoint.model_dump(mode="json"),
        "path_points_m": [
            current.model_dump(mode="json"),
            selected_endpoint.model_dump(mode="json"),
        ],
    }

    result = _revalidate_selected_navigation_path(
        world=world,
        frame=frame,
        selected=selected,
        required_clearance_m=0.0,
        vehicle_radius_m=0.1,
        vehicle_height_m=0.1,
        maximum_controller_step_m=0.9,
        preferred_path=(current, old_endpoint),
    )

    assert result.failure_reason is None
    assert result.path_continuity_retained is False
    assert result.path[-1] == selected_endpoint
    assert result.controller_target_m == selected_endpoint


def test_revalidation_replans_stale_geometry_to_model_authorized_endpoint() -> None:
    world = MetricVoxelMap(
        resolution_m=1.0,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=5.99, y=2.99, z=0.99),
    )
    world.seed_known_static_region(
        [
            (world.center_for((x, y, 0)), False)
            for x in range(6)
            for y in range(3)
        ],
        source_sha256="a" * 64,
    )
    current = Vector3(x=0.5, y=0.5, z=0.5)
    endpoint = Vector3(x=5.5, y=0.5, z=0.5)
    selected = {
        "endpoint_m": endpoint.model_dump(mode="json"),
        # This was safe in the model snapshot, but the latest depth map now
        # blocks its straight middle segment.
        "path_points_m": [
            current.model_dump(mode="json"),
            endpoint.model_dump(mode="json"),
        ],
    }
    world.mark_box(
        minimum_m=Vector3(x=2.01, y=0.01, z=0.01),
        maximum_m=Vector3(x=2.99, y=0.99, z=0.99),
        occupied=True,
    )
    frame = _frame(sequence=2, observed_at_unix_ms=1_030).model_copy(
        update={"localization_position_m": current}
    )

    result = _revalidate_selected_navigation_path(
        world=world,
        frame=frame,
        selected=selected,
        required_clearance_m=0.0,
        vehicle_radius_m=0.1,
        vehicle_height_m=0.1,
        maximum_controller_step_m=0.9,
    )

    assert result.failure_reason is None
    assert result.path[-1] == endpoint
    assert any(point.y > 0.5 for point in result.path)
    assert result.path_continuity_retained is False


def test_perception_fusion_rejects_nonmonotonic_and_stale_frames() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)

    with pytest.raises(ValueError, match="SEQUENCE_NOT_MONOTONIC"):
        fusion.ingest(_frame(), now_unix_ms=1_030)
    assert fusion.health(now_unix_ms=1_500).stream_healthy is False
    with pytest.raises(ValueError, match="STREAM_NOT_HEALTHY"):
        fusion.text_navigation_snapshot(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            required_clearance_m=0.0,
            now_unix_ms=1_500,
        )


def test_perception_fusion_rejects_uncalibrated_sensor() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"rear-lidar"},
        minimum_rays_per_frame=8,
    )

    with pytest.raises(ValueError, match="SENSOR_NOT_CALIBRATED"):
        fusion.ingest(_frame(), now_unix_ms=1_020)


def test_perception_fusion_rejects_teleporting_dynamic_track() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
        maximum_dynamic_innovation_m=0.5,
    )
    fusion.ingest(
        _tracked_frame(
            sequence=1,
            observed_at_unix_ms=1_000,
            obstacle_x=1.0,
            obstacle_speed_x=0.2,
        ),
        now_unix_ms=1_020,
    )

    with pytest.raises(ValueError, match="POSITION_INNOVATION_EXCEEDED"):
        fusion.ingest(
            _tracked_frame(
                sequence=2,
                observed_at_unix_ms=1_100,
                obstacle_x=3.0,
                obstacle_speed_x=0.2,
            ),
            now_unix_ms=1_120,
        )


def test_low_confidence_dynamic_track_forces_unhealthy_fusion() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    frame = _tracked_frame(
        sequence=1,
        observed_at_unix_ms=1_000,
        obstacle_x=1.0,
        obstacle_speed_x=0.2,
    )
    frame.dynamic_obstacles[0].confidence = 0.2

    health = fusion.ingest(frame, now_unix_ms=1_020)

    assert health.stream_healthy is False
    assert "DYNAMIC_TRACK_CONFIDENCE_LOW:person-1" in health.issue_codes


def test_retained_track_does_not_refresh_original_detection():
    fusion = RuntimePerceptionFusion(world=_world(), accepted_sensor_ids={"front-lidar"})
    first = _tracked_frame(sequence=1, observed_at_unix_ms=1_000,
                           obstacle_x=1.0, obstacle_speed_x=0.2)
    fusion.ingest(first, now_unix_ms=1_020)
    retained = first.model_copy(deep=True, update={"sequence": 2, "observed_at_unix_ms": 1_100})
    retained.range_rays = _frame(observed_at_unix_ms=1100).range_rays
    retained.dynamic_obstacles[0].age_seconds = 0.1
    assert fusion.ingest(retained, now_unix_ms=1_120).stream_healthy
    assert fusion.latest_frame.dynamic_obstacles[0].age_seconds == pytest.approx(0.1)
    retained.sequence = 3
    retained.observed_at_unix_ms = 1_200
    retained.dynamic_obstacles[0].age_seconds = 0.2
    retained.dynamic_obstacles[0].position_m.x = 2
    with pytest.raises(ValueError, match="DYNAMIC_TRACK_REPLAY_CONFLICT"):
        fusion.ingest(retained, now_unix_ms=1_220)


def test_omitted_dynamic_track_is_not_silently_cleared_by_new_frames():
    fusion = RuntimePerceptionFusion(world=_world(), accepted_sensor_ids={"front-lidar"})
    first = _tracked_frame(sequence=1, observed_at_unix_ms=1_000,
                           obstacle_x=1.0, obstacle_speed_x=0.2)
    fusion.ingest(first, now_unix_ms=1_020)
    empty = first.model_copy(deep=True, update={
        "sequence": 2, "observed_at_unix_ms": 1_100, "dynamic_obstacles": [],
        "range_rays": _frame(observed_at_unix_ms=1100).range_rays,
    })
    assert fusion.ingest(empty, now_unix_ms=1_120).stream_healthy
    assert len(fusion.latest_frame.dynamic_obstacles) == 1
    empty.sequence = 3
    empty.observed_at_unix_ms = 2_100
    empty.range_rays = _frame(observed_at_unix_ms=2100).range_rays
    health = fusion.ingest(empty, now_unix_ms=2_120)
    assert not health.stream_healthy
    assert "DYNAMIC_TRACK_OBSERVATION_STALE:person-1" in health.issue_codes
    assert fusion.latest_frame.dynamic_obstacles[0].age_seconds == pytest.approx(1.1)
    empty.sequence = 4
    empty.observed_at_unix_ms = 40_000
    empty.range_rays = _frame(observed_at_unix_ms=40000).range_rays
    assert not fusion.ingest(empty, now_unix_ms=40_020).stream_healthy
    assert fusion.latest_frame.sequence == 4
    assert fusion.latest_frame.dynamic_obstacles[0].age_seconds == 39
    assert not fusion.health(now_unix_ms=40_020).stream_healthy


def test_duplicate_dynamic_id_is_rejected_before_fusion():
    fusion = RuntimePerceptionFusion(world=_world(), accepted_sensor_ids={"front-lidar"})
    frame = _tracked_frame(sequence=1, observed_at_unix_ms=1_000,
                           obstacle_x=1.0, obstacle_speed_x=0.2)
    frame.dynamic_obstacles.append(frame.dynamic_obstacles[0].model_copy(deep=True))
    with pytest.raises(ValueError, match="DUPLICATE_ID"):
        fusion.ingest(frame, now_unix_ms=1_020)
    assert fusion.latest_frame is None


def test_text_navigation_decision_cannot_select_unlisted_candidate() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)
    snapshot = fusion.text_navigation_snapshot(
        goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
        required_clearance_m=0.0,
        now_unix_ms=1_020,
    )

    with pytest.raises(ValueError, match="CANDIDATE_NOT_AUTHORIZED"):
        validate_text_navigation_decision(
            snapshot,
            TextNavigationDecision(
                snapshot_sha256=snapshot["snapshot_sha256"],
                action="select-candidate",
                selected_candidate_id="candidate-invented",
                rationale_summary="Invented candidate must fail validation.",
            ),
        )


def test_pilot_control_decision_does_not_require_or_invent_a_candidate() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)
    snapshot = fusion.text_navigation_snapshot(
        goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
        required_clearance_m=0.0,
        now_unix_ms=1_020,
    )

    selected = validate_text_navigation_decision(
        snapshot,
        TextNavigationDecision(
            snapshot_sha256=snapshot["snapshot_sha256"],
            action="pilot-control",
            rationale_summary="Issue a bounded body-frame control request.",
            pilot_control=NormalizedPilotControl(
                forward_axis=0.6,
                right_axis=-0.1,
                up_axis=0.2,
                yaw_axis=0.3,
            ),
        ),
    )

    assert selected is None


@pytest.mark.parametrize("mode", ["normalized-body-velocity", "legacy-candidate-selection"])
def test_event_driven_pilot_control_establishes_short_direct_control_lease(mode) -> None:
    from control_fixtures import complete_feature_snapshot

    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)
    pilot = NormalizedPilotControl(
        forward_axis=0.6,
        right_axis=-0.1,
        up_axis=0.2,
        yaw_axis=0.3,
    )

    class Port:
        def call(self, **kwargs):
            snapshot = kwargs["input_artifact"]["text_navigation_snapshot"]
            return SimpleNamespace(
                artifact=TextNavigationDecision(
                    snapshot_sha256=snapshot["snapshot_sha256"],
                    action="pilot-control",
                    rationale_summary="Issue bounded body-frame pilot axes.",
                    controller_step_scale=0.7,
                    pilot_control=pilot,
                ),
                record=SimpleNamespace(
                    call_id="call-direct-pilot-control",
                    local_expert_trace=SimpleNamespace(
                        selected_navigation_role="local-navigation-policy"
                    ),
                ),
            )

    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=Port(),
        required_clearance_m=0.0,
        control_output_mode=mode,
    )
    try:
        goal = Vector3(x=3.75, y=0.25, z=0.25)
        coordinator.schedule(
            goal_position_m=goal,
            navigation_goal_id="goal-direct-control",
            now_unix_ms=1_020,
            trigger="initial",
            realtime_feature_snapshot=complete_feature_snapshot().model_dump(mode="json"),
        )
        receipt = _wait_for_receipt(coordinator, now_unix_ms=1_030)
        directive = coordinator.controller_directive(
            current_position_m=_frame().localization_position_m,
            fallback_target_m=goal,
            now_unix_ms=1_040,
            navigation_goal_id="goal-direct-control",
        )

        if mode == "legacy-candidate-selection":
            assert receipt.hold_reason == "MODEL_NAVIGATION_OUTPUT_MODE_MISMATCH"
            assert not directive.model_navigation_authorized
            return

        assert receipt.model_action == "pilot-control"
        assert receipt.hold_reason is None
        assert receipt.selected_candidate_id is None
        assert directive.model_navigation_authorized is True
        assert directive.selected_candidate_id is None
        assert directive.model_call_id == "call-direct-pilot-control"
        assert directive.pilot_control == pilot
        assert directive.path_sha256 is not None
        assert directive.navigation_snapshot_sha256 == receipt.snapshot_sha256
        assert directive.navigation_snapshot_sha256 != directive.path_sha256
        assert 1_040 < directive.valid_until_unix_ms <= 1_250
        expired = coordinator.controller_directive(
            current_position_m=_frame().localization_position_m,
            fallback_target_m=goal,
            now_unix_ms=directive.valid_until_unix_ms,
            navigation_goal_id="goal-direct-control",
        )
        assert not expired.model_navigation_authorized
        for clock in (math.nan, True, -1):
            with pytest.raises(ValueError, match="CLOCK_INVALID"):
                coordinator.controller_directive(
                    current_position_m=_frame().localization_position_m,
                    fallback_target_m=goal, now_unix_ms=clock,
                )
    finally:
        coordinator.close()


def test_text_only_model_is_limited_to_hash_bound_candidate_ids() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)
    snapshot = fusion.text_navigation_snapshot(
        goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
        required_clearance_m=0.0,
        now_unix_ms=1_020,
    )
    candidate_id = snapshot["authorized_candidate_paths"][0]["candidate_id"]

    class Port:
        def call(self, **kwargs):
            assert kwargs["role"] == "local_navigation_advisor"
            assert kwargs["input_artifact"]["text_navigation_snapshot"] == snapshot
            return SimpleNamespace(
                artifact=TextNavigationDecision(
                    snapshot_sha256=snapshot["snapshot_sha256"],
                    action="select-candidate",
                    selected_candidate_id=candidate_id,
                    rationale_summary="Select the validated observed-free goal path.",
                )
            )

    result, selected = request_text_navigation_decision(port=Port(), snapshot=snapshot)

    assert result.artifact.selected_candidate_id == candidate_id
    assert selected is not None
    assert selected["candidate_id"] == candidate_id


def test_text_only_coordinator_revalidates_full_metric_path_before_control_target() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)

    class Port:
        def call(self, **kwargs):
            snapshot = kwargs["input_artifact"]["text_navigation_snapshot"]
            candidate = snapshot["authorized_candidate_paths"][0]
            return SimpleNamespace(
                artifact=TextNavigationDecision(
                    snapshot_sha256=snapshot["snapshot_sha256"],
                    action="select-candidate",
                    selected_candidate_id=candidate["candidate_id"],
                    rationale_summary="Use the metric-validated observed-free goal corridor.",
                )
            )

    receipt = TextOnlyIndoorNavigationCoordinator(
        fusion=fusion,
        port=Port(),
        required_clearance_m=0.0,
    ).evaluate(
        goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
        now_unix_ms=1_020,
        trigger="initial",
    )

    assert receipt.model_action == "select-candidate"
    assert receipt.deterministic_metric_path_revalidated is True
    assert receipt.dynamic_path_revalidated is True
    assert receipt.controller_target_m is not None
    assert receipt.control_authority == "deterministic-local-safety"


def test_text_only_coordinator_holds_without_model_when_perception_is_stale() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)

    class Port:
        def call(self, **_kwargs):
            raise AssertionError("model must not be called on stale perception")

    receipt = TextOnlyIndoorNavigationCoordinator(
        fusion=fusion,
        port=Port(),
        required_clearance_m=0.0,
    ).evaluate(
        goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
        now_unix_ms=1_500,
        trigger="periodic",
    )

    assert receipt.model_action == "not-invoked"
    assert receipt.controller_target_m is None
    assert receipt.hold_reason == "PERCEPTION_STREAM_NOT_HEALTHY"


def test_event_driven_navigation_records_stale_snapshot_without_model_call() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)

    class Port:
        def call(self, **_kwargs):
            raise AssertionError("model must not be called on stale perception")

    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=Port(),
        required_clearance_m=0.0,
    )
    try:
        receipt = coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_500,
            trigger="periodic",
            strategic_context={"task": {"phase": "TRANSIT"}},
        )
        snapshot = coordinator.pop_submitted_snapshot()
    finally:
        coordinator.close()

    assert receipt is not None
    assert receipt.model_action == "not-invoked"
    assert receipt.hold_reason == "PERCEPTION_STREAM_NOT_HEALTHY"
    assert snapshot is not None
    assert snapshot["perception_health"]["stream_healthy"] is False
    assert snapshot["perception_health"]["issue_codes"] == ["PERCEPTION_STREAM_STALE"]
    assert snapshot["strategic_context"]["task"]["phase"] == "TRANSIT"
    assert receipt.snapshot_sha256 == snapshot["snapshot_sha256"]
    assert snapshot["snapshot_sha256"] == sha256_json(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )


def test_event_driven_navigation_does_not_block_sensor_thread(monkeypatch) -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)
    release = threading.Event()
    original_snapshot = MetricVoxelMap.text_navigation_snapshot

    def delayed_snapshot(self, **kwargs):
        assert release.wait(timeout=2.0)
        return original_snapshot(self, **kwargs)

    monkeypatch.setattr(MetricVoxelMap, "text_navigation_snapshot", delayed_snapshot)

    class Port:
        def call(self, **kwargs):
            assert release.wait(timeout=2.0)
            snapshot = kwargs["input_artifact"]["text_navigation_snapshot"]
            candidate = snapshot["authorized_candidate_paths"][0]
            return SimpleNamespace(
                artifact=TextNavigationDecision(
                    snapshot_sha256=snapshot["snapshot_sha256"],
                    action="select-candidate",
                    selected_candidate_id=candidate["candidate_id"],
                    rationale_summary="Use the authorized metric corridor.",
                ),
                record=SimpleNamespace(call_id="call-local-navigation"),
            )

    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=Port(),
        required_clearance_m=0.0,
    )
    try:
        started = time.monotonic()
        immediate = coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_020,
            trigger="initial",
        )
        elapsed = time.monotonic() - started
        assert immediate is None
        assert elapsed < 0.1
        assert coordinator.request_pending is True
        assert coordinator.poll(now_unix_ms=1_025) is None

        release.set()
        receipt = _wait_for_receipt(coordinator, now_unix_ms=1_030)
        assert receipt.model_call_id == "call-local-navigation"
        assert receipt.controller_target_m is not None
        assert math.dist(
            (0.25, 0.25, 0.25),
            tuple(receipt.controller_target_m.model_dump().values()),
        ) <= 0.120001
    finally:
        release.set()
        coordinator.close()


def test_event_driven_navigation_reuses_callers_atomic_health_sample(monkeypatch) -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)

    class Port:
        def call(self, **kwargs):
            snapshot = kwargs["input_artifact"]["text_navigation_snapshot"]
            candidate = snapshot["authorized_candidate_paths"][0]
            return SimpleNamespace(
                artifact=TextNavigationDecision(
                    snapshot_sha256=snapshot["snapshot_sha256"],
                    action="select-candidate",
                    selected_candidate_id=candidate["candidate_id"],
                    rationale_summary="Use the freshly admitted metric corridor.",
                ),
                record=SimpleNamespace(call_id="call-atomic-health"),
            )

    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=Port(),
        required_clearance_m=0.0,
    )
    try:
        coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_020,
            trigger="initial",
        )
        for _ in range(100):
            pending = coordinator._pending
            if pending is not None and pending.future.done():
                break
            time.sleep(0.005)
        else:
            raise AssertionError("model navigation future did not complete")

        admitted_health = fusion.health(now_unix_ms=1_025)

        def unexpected_second_health_read(*_args, **_kwargs):
            raise AssertionError("poll must reuse the caller's atomic health sample")

        monkeypatch.setattr(RuntimePerceptionFusion, "health", unexpected_second_health_read)
        current_frame = _frame().model_copy(
            update={"localization_position_m": Vector3(x=0.75, y=0.25, z=0.25)},
            deep=True,
        )
        for _ in range(100):
            receipt = coordinator.poll(
                now_unix_ms=1_025,
                current_goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
                current_perception_health=admitted_health,
                current_frame=current_frame,
            )
            if receipt is not None:
                break
            time.sleep(0.005)
        else:
            raise AssertionError("model navigation revalidation did not complete")

        assert receipt.model_call_id == "call-atomic-health"
        assert receipt.perception_health == admitted_health
        assert receipt.hold_reason is None
        assert receipt.deterministic_metric_path_revalidated is True
        assert receipt.controller_target_m is not None
        assert math.dist(
            tuple(current_frame.localization_position_m.model_dump().values()),
            tuple(receipt.controller_target_m.model_dump().values()),
        ) <= coordinator.maximum_controller_step_m + 1e-6
    finally:
        coordinator.close()


def test_event_driven_navigation_recovers_after_outer_model_deadline() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)
    release = threading.Event()

    class Port:
        reset_count = 0

        def call(self, **kwargs):
            assert release.wait(timeout=2.0)
            snapshot = kwargs["input_artifact"]["text_navigation_snapshot"]
            candidate = snapshot["authorized_candidate_paths"][0]
            return SimpleNamespace(
                artifact=TextNavigationDecision(
                    snapshot_sha256=snapshot["snapshot_sha256"],
                    action="select-candidate",
                    selected_candidate_id=candidate["candidate_id"],
                    rationale_summary="Use the authorized metric corridor.",
                ),
                record=SimpleNamespace(call_id="call-after-outer-timeout"),
            )

        def reset_transport(self) -> None:
            self.reset_count += 1
            release.set()

    port = Port()
    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=port,
        required_clearance_m=0.0,
        maximum_decision_age_seconds=1.0,
    )
    try:
        coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_020,
            trigger="initial",
        )
        receipt = coordinator.poll(now_unix_ms=2_021)
        assert receipt is not None
        assert receipt.hold_reason == "MODEL_NAVIGATION_INVOCATION_TIMEOUT"
        assert port.reset_count == 1
        # 复位连接不能强杀旧 Python 工作；等其退出后才复用同一模型工作槽。
        deadline = time.monotonic() + 2.0
        while coordinator.request_pending and time.monotonic() < deadline:
            time.sleep(0.005)
        assert coordinator.request_pending is False
        assert coordinator._executor_generation == 0

        fusion.ingest(
            _frame(sequence=2, observed_at_unix_ms=2_030),
            now_unix_ms=2_030,
        )
        coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=2_030,
            trigger="periodic",
        )
        recovered = _wait_for_receipt(coordinator, now_unix_ms=2_040)
        assert recovered.model_call_id == "call-after-outer-timeout"
        assert recovered.controller_target_m is not None
    finally:
        release.set()
        coordinator.close()


def test_event_driven_navigation_uses_selected_provider_deadline() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)
    release = threading.Event()

    class SlowFallbackPort:
        invocation_timeout_seconds = 5.0

        def call(self, **_kwargs):
            assert release.wait(timeout=2.0)
            raise RuntimeError("released after coordinator timeout")

        def reset_transport(self) -> None:
            release.set()

    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=SlowFallbackPort(),
        required_clearance_m=0.0,
        maximum_decision_age_seconds=1.0,
    )
    try:
        coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_020,
            trigger="initial",
        )

        assert coordinator.poll(now_unix_ms=3_020) is None
        assert coordinator.poll(now_unix_ms=9_020) is None
        receipt = coordinator.poll(now_unix_ms=9_021)

        assert receipt is not None
        assert receipt.hold_reason == "MODEL_NAVIGATION_INVOCATION_TIMEOUT"
    finally:
        release.set()
        coordinator.close()


def test_event_driven_navigation_replans_from_latest_position_after_model_latency() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)
    release = threading.Event()
    selected_metric_hash: list[str] = []

    class Port:
        def call(self, **kwargs):
            snapshot = kwargs["input_artifact"]["text_navigation_snapshot"]
            candidate = snapshot["authorized_candidate_paths"][0]
            selected_metric_hash.append(candidate["metric_path_sha256"])
            assert release.wait(timeout=2.0)
            return SimpleNamespace(
                artifact=TextNavigationDecision(
                    snapshot_sha256=snapshot["snapshot_sha256"],
                    action="select-candidate",
                    selected_candidate_id=candidate["candidate_id"],
                    rationale_summary="Continue toward the observed goal corridor.",
                    controller_step_scale=0.5,
                    pilot_control=NormalizedPilotControl(
                        forward_axis=0.6,
                        right_axis=-0.1,
                        up_axis=0.2,
                        yaw_axis=0.3,
                    ),
                ),
                record=SimpleNamespace(
                    call_id="call-moved-position",
                    local_expert_trace=SimpleNamespace(
                        selected_navigation_role="precision-maneuver-policy"
                    ),
                ),
            )

    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=Port(),
        required_clearance_m=0.0,
    )
    try:
        coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            navigation_goal_id="goal-a",
            now_unix_ms=1_020,
            trigger="initial",
        )
        moved = _frame(sequence=2, observed_at_unix_ms=1_040).model_copy(
            update={"localization_position_m": Vector3(x=0.75, y=0.25, z=0.25)}
        )
        fusion.ingest(moved, now_unix_ms=1_050)
        release.set()
        receipt = _wait_for_receipt(coordinator, now_unix_ms=1_060)

        assert receipt.deterministic_metric_path_revalidated is True
        assert receipt.dynamic_path_revalidated is True
        assert receipt.controller_step_scale == 0.5
        assert receipt.selected_metric_path_sha256 != selected_metric_hash[0]
        target = coordinator.controller_target(
            current_position_m=moved.localization_position_m,
            fallback_target_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_060,
            navigation_goal_id="goal-a",
        )
        assert target.x > moved.localization_position_m.x
        advanced_position = Vector3(x=1.0, y=0.25, z=0.25)
        advanced_target = coordinator.controller_target(
            current_position_m=advanced_position,
            fallback_target_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_070,
            navigation_goal_id="goal-a",
        )
        assert advanced_target.x > advanced_position.x
        assert math.dist(
            tuple(advanced_position.model_dump().values()),
            tuple(advanced_target.model_dump().values()),
        ) <= 0.060001
        directive = coordinator.controller_directive(
            current_position_m=advanced_position,
            fallback_target_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_070,
            navigation_goal_id="goal-a",
        )
        assert directive.model_navigation_authorized is True
        assert directive.model_call_id == "call-moved-position"
        assert directive.selected_candidate_id is not None
        assert directive.path_sha256 == receipt.selected_metric_path_sha256
        assert directive.navigation_goal_id == "goal-a"
        assert directive.controller_step_scale == 0.5
        assert directive.source_expert == "precision-maneuver-policy"
        assert directive.pilot_control is not None
        assert directive.pilot_control.forward_axis == pytest.approx(0.6)
        precision_directive = coordinator.controller_directive(
            current_position_m=advanced_position,
            fallback_target_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_070,
            maximum_step_m=0.06,
            navigation_goal_id="goal-a",
        )
        assert math.dist(
            tuple(advanced_position.model_dump().values()),
            tuple(precision_directive.target_m.model_dump().values()),
        ) <= 0.030001
        with pytest.raises(ValueError, match="no larger"):
            coordinator.controller_directive(
                current_position_m=advanced_position,
                fallback_target_m=Vector3(x=3.75, y=0.25, z=0.25),
                now_unix_ms=1_070,
                maximum_step_m=0.13,
            )

        expired = coordinator.controller_directive(
            current_position_m=advanced_position,
            fallback_target_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=20_000,
            navigation_goal_id="goal-a",
        )
        assert expired.model_navigation_authorized is False
        assert expired.reason == "model-lease-expired"
        assert expired.target_m == Vector3(x=3.75, y=0.25, z=0.25)
        assert expired.pilot_control is None
        wrong_epoch = coordinator.controller_directive(
            current_position_m=advanced_position,
            fallback_target_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_080,
            navigation_goal_id="goal-b",
        )
        assert wrong_epoch.model_navigation_authorized is False
        assert wrong_epoch.reason == "model-lease-goal-changed"
        assert wrong_epoch.target_m == advanced_position
    finally:
        release.set()
        coordinator.close()


def test_event_driven_navigation_rejects_unlisted_model_candidate() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)

    class Port:
        def call(self, **kwargs):
            snapshot = kwargs["input_artifact"]["text_navigation_snapshot"]
            return SimpleNamespace(
                artifact=TextNavigationDecision(
                    snapshot_sha256=snapshot["snapshot_sha256"],
                    action="select-candidate",
                    selected_candidate_id="invented-path",
                    rationale_summary="Attempt to bypass the candidate authority boundary.",
                )
            )

    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=Port(),
        required_clearance_m=0.0,
    )
    try:
        coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_020,
            trigger="initial",
        )
        receipt = _wait_for_receipt(coordinator, now_unix_ms=1_030)
        assert receipt.model_action == "not-invoked"
        assert receipt.controller_target_m is None
        assert receipt.hold_reason == "MODEL_NAVIGATION_INVOCATION_FAILED"
        assert receipt.failure_stage == "model-invocation"
        assert receipt.failure_exception_type == "builtins.ValueError"
        assert receipt.failure_root_exception_type == "builtins.ValueError"
        assert receipt.failure_reason_code == "MODEL_INVOCATION_FAILED"
        assert receipt.failure_fingerprint is not None
        assert len(receipt.failure_fingerprint) == 64
    finally:
        coordinator.close()


def test_event_driven_navigation_preserves_bounded_failure_timings() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)

    class Port:
        def call(self, **kwargs):
            raise ModelInvocationError(
                "bounded local failure",
                attempts_used=1,
                reason_code="LOCAL_POLICY_QUALIFIED_LATENCY_EXCEEDED",
                diagnostic_metrics={
                    "total-latency-ms": 251.25,
                    "qualified-limit-ms": 250.0,
                },
            )

    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=Port(),
        required_clearance_m=0.0,
    )
    try:
        coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_020,
            trigger="initial",
        )
        receipt = _wait_for_receipt(coordinator, now_unix_ms=1_030)
        assert receipt.failure_reason_code == (
            "LOCAL_POLICY_QUALIFIED_LATENCY_EXCEEDED"
        )
        assert receipt.failure_diagnostic_metrics == {
            "total-latency-ms": 251.25,
            "qualified-limit-ms": 250.0,
        }
    finally:
        coordinator.close()


def test_event_driven_navigation_hash_binds_supplementary_visual_frame(tmp_path) -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)
    frame_path = tmp_path / "forward.png"
    frame_path.write_bytes(b"bounded-forward-rgb-frame")
    expected_sha256 = hashlib.sha256(frame_path.read_bytes()).hexdigest()

    class Port:
        def call(self, **kwargs):
            snapshot = kwargs["input_artifact"]["text_navigation_snapshot"]
            assert snapshot["visual_evidence"] == [
                {
                    "kind": "forward-rgb-camera",
                    "sha256": expected_sha256,
                    "content_type": "image/png",
                }
            ]
            assert kwargs["multimodal"][0]["path"] == str(frame_path)
            return SimpleNamespace(
                artifact=TextNavigationDecision(
                    snapshot_sha256=snapshot["snapshot_sha256"],
                    action="hold",
                    rationale_summary="Hold while the visual corridor is ambiguous.",
                ),
                record=SimpleNamespace(call_id="call-visual-navigation"),
            )

    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=Port(),
        required_clearance_m=0.0,
    )
    try:
        coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_020,
            trigger="initial",
            multimodal=[
                {
                    "kind": "image-file",
                    "path": str(frame_path),
                    "content_type": "image/png",
                }
            ],
        )
        receipt = _wait_for_receipt(coordinator, now_unix_ms=1_030)
        submitted = coordinator.pop_submitted_snapshot()
        assert submitted is not None
        assert submitted["control_reference_observed_at_unix_ms"] == 1_020
        assert submitted["visual_evidence"][0]["sha256"] == expected_sha256
        assert submitted["snapshot_sha256"] == sha256_json(
            {key: value for key, value in submitted.items() if key != "snapshot_sha256"}
        )
        assert coordinator.pop_submitted_snapshot() is None
        assert receipt.model_action == "hold"
        assert receipt.model_call_id == "call-visual-navigation"
    finally:
        coordinator.close()


def test_event_driven_navigation_accepts_hash_bound_embedded_frame_before_disk_write(
    tmp_path, monkeypatch,
) -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)
    frame_path = tmp_path / "pending-forward.png"
    frame_bytes = b"bounded-forward-rgb-frame"
    expected_sha256 = hashlib.sha256(frame_bytes).hexdigest()

    class Port:
        def call(self, **kwargs):
            snapshot = kwargs["input_artifact"]["text_navigation_snapshot"]
            assert snapshot["visual_evidence"][0]["sha256"] == expected_sha256
            assert kwargs["multimodal"][0]["content_bytes"] == frame_bytes
            return SimpleNamespace(
                artifact=TextNavigationDecision(
                    snapshot_sha256=snapshot["snapshot_sha256"],
                    action="hold",
                    rationale_summary="Hold while the visual corridor is ambiguous.",
                ),
                record=SimpleNamespace(call_id="call-embedded-visual-navigation"),
            )

    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=Port(),
        required_clearance_m=0.0,
    )
    def reject_filesystem_resolution(*args, **kwargs):
        raise AssertionError("Embedded visual input must not resolve a mounted path")

    monkeypatch.setattr(perception_runtime.Path, "resolve", reject_filesystem_resolution)
    try:
        coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            now_unix_ms=1_020,
            trigger="initial",
            multimodal=[
                {
                    "kind": "image-file",
                    "path": str(frame_path),
                    "content_type": "image/png",
                    "content_bytes": frame_bytes,
                    "content_sha256": expected_sha256,
                }
            ],
        )
        receipt = _wait_for_receipt(coordinator, now_unix_ms=1_030)
        assert receipt.model_action == "hold"
        assert receipt.model_call_id == "call-embedded-visual-navigation"
        assert not frame_path.exists()
    finally:
        coordinator.close()


def test_event_driven_navigation_discards_response_when_route_goal_epoch_changes() -> None:
    fusion = RuntimePerceptionFusion(
        world=_world(),
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    fusion.ingest(_frame(), now_unix_ms=1_020)

    class Port:
        def call(self, **kwargs):
            snapshot = kwargs["input_artifact"]["text_navigation_snapshot"]
            candidate = snapshot["authorized_candidate_paths"][0]
            return SimpleNamespace(
                artifact=TextNavigationDecision(
                    snapshot_sha256=snapshot["snapshot_sha256"],
                    action="select-candidate",
                    selected_candidate_id=candidate["candidate_id"],
                    rationale_summary="Select the old goal candidate.",
                ),
                record=SimpleNamespace(call_id="call-old-goal"),
            )

    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=Port(),
        required_clearance_m=0.0,
    )
    try:
        coordinator.schedule(
            goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
            navigation_goal_id="goal-outbound",
            now_unix_ms=1_020,
            trigger="initial",
        )
        for _ in range(100):
            receipt = coordinator.poll(
                now_unix_ms=1_030,
                current_goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
                current_navigation_goal_id="goal-return",
            )
            if receipt is not None:
                break
            time.sleep(0.005)
        else:
            raise AssertionError("model navigation future did not complete")
        assert receipt.controller_target_m is None
        assert receipt.hold_reason == "MODEL_NAVIGATION_GOAL_CHANGED_DURING_CALL"
    finally:
        coordinator.close()
