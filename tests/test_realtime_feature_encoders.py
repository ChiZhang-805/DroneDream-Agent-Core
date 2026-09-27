import math

import pytest
from control_fixtures import complete_feature_snapshot

from dronedream_agent_core.contracts import (
    CalibratedRangeSensorMount,
    DynamicObstacleObservation,
    NormalizedPilotControl,
    QuaternionWxyz,
    RawMetricRangeScan,
    RawRangeSample,
    RuntimeLocalSafetyCommand,
    RuntimeLocalSafetyObservation,
    Vector3,
    VehicleAsset,
)
from dronedream_agent_core.control_feature_contract import (
    DYNAMIC_TARGET_FEATURE_COUNT,
    GEOMETRY_FEATURE_COUNT,
)
from dronedream_agent_core.local_policy_port import compile_local_policy_features
from dronedream_agent_core.realtime_feature_encoders import (
    POLICY_CONTROL_REFERENCE_FEATURE_COUNT,
    FlightStateEncoder,
    FlightStateSample,
    RealtimeFeatureSnapshot,
    body_control_intent_for_pilot_control,
    body_control_intent_for_target,
    body_to_world_enu,
    encode_dynamic_targets,
    encode_metric_geometry,
    fuse_realtime_features,
    world_enu_to_body,
)
from dronedream_agent_core.runtime_local_safety import evaluate_runtime_local_safety

IDENTITY = QuaternionWxyz(w=1.0, x=0.0, y=0.0, z=0.0)


def _mount(
    *, orientation_body_from_sensor: QuaternionWxyz = IDENTITY
) -> CalibratedRangeSensorMount:
    return CalibratedRangeSensorMount(
        sensor_id="test-lidar",
        translation_body_m=Vector3(x=0.0, y=0.0, z=0.0),
        orientation_body_from_sensor=orientation_body_from_sensor,
        minimum_range_m=0.1,
        maximum_range_m=5.0,
    )


def _scan(*, observed_at_unix_ms: int = 1_000) -> RawMetricRangeScan:
    samples = []
    for sector in range(8):
        angle = sector * math.pi / 4.0
        samples.extend(
            RawRangeSample(
                direction_sensor=Vector3(
                    x=math.cos(angle),
                    y=math.sin(angle),
                    z=0.0,
                ),
                range_m=2.0 + sample_index,
                hit=sample_index == 0,
                confidence=0.9,
            )
            for sample_index in range(4)
        )
    return RawMetricRangeScan(
        sensor_id="test-lidar",
        sequence=1,
        observed_at_unix_ms=observed_at_unix_ms,
        observed_at_monotonic_seconds=1.0,
        body_position_world_enu_m=Vector3(x=0.0, y=0.0, z=1.0),
        body_orientation_world_from_body=IDENTITY,
        body_velocity_world_enu_mps=Vector3(x=0.0, y=0.0, z=0.0),
        localization_covariance_m2=0.01,
        samples=samples,
    )


def test_metric_geometry_encodes_measured_sector_content_and_masks() -> None:
    encoding = encode_metric_geometry(_scan(), sensor_mount=_mount(), encoded_at_unix_ms=1_001)

    assert encoding.encoder_role == "metric-geometry-encoder"
    assert len(encoding.features) == GEOMETRY_FEATURE_COUNT
    assert encoding.valid_mask[:40] == [1.0] * 40
    assert encoding.valid_mask[40:] == [0.0] * (GEOMETRY_FEATURE_COUNT - 40)
    assert encoding.coverage == 1.0
    assert encoding.features[0] == pytest.approx(2.0 / 5.0)
    assert encoding.features[2] == pytest.approx(0.25)
    assert encoding.source_sha256 != "0" * 64


def test_dynamic_target_encoder_reports_closing_speed_and_ttc() -> None:
    encoding = encode_dynamic_targets(
        [
            DynamicObstacleObservation(
                obstacle_id="closing-target",
                position_m=Vector3(x=10.0, y=0.0, z=1.0),
                velocity_mps=Vector3(x=-2.0, y=0.0, z=0.0),
                radius_m=0.4,
                height_m=1.7,
                confidence=0.95,
                age_seconds=0.02,
            )
        ],
        body_position_world_enu_m=Vector3(x=0.0, y=0.0, z=1.0),
        body_orientation_world_from_body=IDENTITY,
        body_velocity_world_enu_mps=Vector3(x=0.0, y=0.0, z=0.0),
        observed_at_unix_ms=2_000,
        encoded_at_unix_ms=2_001,
    )

    assert len(encoding.features) == DYNAMIC_TARGET_FEATURE_COUNT
    assert encoding.features[:5] == pytest.approx([
        9.958 / 30.0, 0.2, 4.779 / 30.0, 0.95, 0.042,
    ])
    assert encoding.valid_mask[:11] == [1.0] * 11
    assert encoding.valid_mask[11:] == [0.0] * (DYNAMIC_TARGET_FEATURE_COUNT - 11)


def test_geometry_applies_sensor_mount_rotation_before_sectorization() -> None:
    yaw_ninety = QuaternionWxyz(
        w=math.sqrt(0.5),
        x=0.0,
        y=0.0,
        z=math.sqrt(0.5),
    )
    scan = _scan().model_copy(update={"samples": _scan().samples[:4]})

    encoding = encode_metric_geometry(
        scan,
        sensor_mount=_mount(orientation_body_from_sensor=yaw_ninety),
        encoded_at_unix_ms=1_001,
    )

    assert encoding.valid_mask[:5] == [0.0] * 5
    assert encoding.valid_mask[30:35] == [1.0] * 5
    assert encoding.features[30] == pytest.approx(2.0 / 5.0)


def test_forward_camera_coverage_is_measured_against_its_calibrated_fov() -> None:
    full_scan = _scan()
    forward_samples = [
        sample for index, sample in enumerate(full_scan.samples) if index // 4 in {0, 1, 7}
    ] * 3
    scan = full_scan.model_copy(update={"samples": forward_samples})

    encoding = encode_metric_geometry(
        scan,
        sensor_mount=_mount(),
        expected_horizontal_fov_rad=1.274,
        encoded_at_unix_ms=1_001,
    )

    assert encoding.coverage == 1.0
    assert "GEOMETRY_ANGULAR_COVERAGE_LOW" not in encoding.issue_codes


def test_dynamic_target_uses_vehicle_relative_velocity() -> None:
    encoding = encode_dynamic_targets(
        [
            DynamicObstacleObservation(
                obstacle_id="stationary-target",
                position_m=Vector3(x=10.0, y=0.0, z=1.0),
                velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
                radius_m=0.4,
                height_m=1.7,
                confidence=1.0,
                age_seconds=0.0,
            )
        ],
        body_position_world_enu_m=Vector3(x=0.0, y=0.0, z=1.0),
        body_orientation_world_from_body=IDENTITY,
        body_velocity_world_enu_mps=Vector3(x=2.0, y=0.0, z=0.0),
        observed_at_unix_ms=2_000,
        encoded_at_unix_ms=2_001,
    )

    assert encoding.features[1] == pytest.approx(0.2)
    assert encoding.features[2] == pytest.approx(4.799 / 30.0)


def test_flight_state_encoder_preserves_missing_mask_and_temporal_variation() -> None:
    encoder = FlightStateEncoder(history_length=4)
    first = encoder.encode(
        FlightStateSample(
            source_id="px4-ekf",
            observed_at_unix_ms=3_000,
            orientation_world_from_body=IDENTITY,
            velocity_world_enu_mps=Vector3(x=1.0, y=0.0, z=0.0),
            localization_covariance_m2=0.01,
        )
    )
    second = encoder.encode(
        FlightStateSample(
            source_id="px4-ekf",
            observed_at_unix_ms=3_050,
            orientation_world_from_body=IDENTITY,
            velocity_world_enu_mps=Vector3(x=2.0, y=0.0, z=0.0),
            acceleration_body_mps2=Vector3(x=1.0, y=0.0, z=0.0),
            angular_velocity_body_rad_s=Vector3(x=0.0, y=0.0, z=0.2),
            localization_covariance_m2=0.01,
        )
    )

    assert first.valid_mask[8:14] == [0.0] * 6
    assert second.valid_mask[8:14] == [1.0] * 6
    assert second.coverage == 0.5
    assert second.features[-5] > 0.0


def test_feature_fusion_fails_closed_when_required_encoding_is_stale() -> None:
    geometry = encode_metric_geometry(_scan(), sensor_mount=_mount(), encoded_at_unix_ms=1_001)
    dynamic = encode_dynamic_targets(
        [],
        body_position_world_enu_m=Vector3(x=0.0, y=0.0, z=1.0),
        body_orientation_world_from_body=IDENTITY,
        body_velocity_world_enu_mps=Vector3(x=0.0, y=0.0, z=0.0),
        observed_at_unix_ms=1_000,
        encoded_at_unix_ms=1_001,
    )
    state = FlightStateEncoder(history_length=4).encode(
        FlightStateSample(
            source_id="px4-ekf",
            observed_at_unix_ms=1_000,
            orientation_world_from_body=IDENTITY,
            velocity_world_enu_mps=Vector3(x=0.0, y=0.0, z=0.0),
            localization_covariance_m2=0.01,
        ),
        encoded_at_unix_ms=1_001,
    )

    snapshot = fuse_realtime_features(
        (geometry, dynamic, state),
        captured_at_unix_ms=1_600,
    )

    assert snapshot.ready_for_control is False
    assert "REQUIRED_ENCODER_STALE:metric-geometry-encoder" in snapshot.issue_codes
    assert len(snapshot.fused_features) == len(snapshot.fused_valid_mask)


def test_feature_fusion_blocks_until_temporal_state_history_is_ready() -> None:
    observed_at = 4_000
    geometry = encode_metric_geometry(
        _scan(observed_at_unix_ms=observed_at),
        sensor_mount=_mount(),
        encoded_at_unix_ms=observed_at,
    )
    dynamic = encode_dynamic_targets(
        [],
        body_position_world_enu_m=Vector3(x=0.0, y=0.0, z=1.0),
        body_orientation_world_from_body=IDENTITY,
        body_velocity_world_enu_mps=Vector3(x=0.0, y=0.0, z=0.0),
        observed_at_unix_ms=observed_at,
        encoded_at_unix_ms=observed_at,
    )
    encoder = FlightStateEncoder(history_length=4)
    states = []
    for offset in (0, 50, 100, 150):
        states.append(
            encoder.encode(
                FlightStateSample(
                    source_id="px4-ekf",
                    observed_at_unix_ms=observed_at + offset,
                    orientation_world_from_body=IDENTITY,
                    velocity_world_enu_mps=Vector3(x=0.0, y=0.0, z=0.0),
                    acceleration_body_mps2=Vector3(x=0.0, y=0.0, z=0.0),
                    angular_velocity_body_rad_s=Vector3(x=0.0, y=0.0, z=0.0),
                    localization_covariance_m2=0.01,
                ),
                encoded_at_unix_ms=observed_at + offset,
            )
        )

    warming = fuse_realtime_features(
        (geometry, dynamic, states[0]),
        captured_at_unix_ms=observed_at,
    )
    ready = fuse_realtime_features(
        (
            geometry.model_copy(
                update={
                    "observed_at_unix_ms": observed_at + 150,
                    "encoded_at_unix_ms": observed_at + 150,
                }
            ),
            dynamic.model_copy(
                update={
                    "observed_at_unix_ms": observed_at + 150,
                    "encoded_at_unix_ms": observed_at + 150,
                }
            ),
            states[-1],
        ),
        captured_at_unix_ms=observed_at + 150,
    )

    assert warming.ready_for_control is False
    assert any("FLIGHT_STATE_HISTORY_WARMING" in issue for issue in warming.issue_codes)
    assert ready.ready_for_control is True
    assert [item.encoder_role for item in ready.encodings] == [
        "metric-geometry-encoder",
        "dynamic-target-encoder",
        "flight-state-encoder",
    ]
    batch = compile_local_policy_features(
        {"realtime_feature_snapshot": ready.model_dump(mode="json")}
    )
    assert batch.realtime_features_ready is True
    assert batch.realtime_snapshot_sha256 == ready.snapshot_sha256
    assert batch.realtime_features[: len(ready.fused_features)] == pytest.approx(
        ready.fused_features
    )
    assert batch.realtime_valid_mask[: len(ready.fused_valid_mask)] == pytest.approx(
        ready.fused_valid_mask
    )
    assert len(batch.realtime_features) == (
        len(ready.fused_features) + POLICY_CONTROL_REFERENCE_FEATURE_COUNT
    )
    assert (
        batch.realtime_valid_mask[-POLICY_CONTROL_REFERENCE_FEATURE_COUNT:]
        == (1.0,) * POLICY_CONTROL_REFERENCE_FEATURE_COUNT
    )


def test_body_world_transforms_are_inverse_and_model_intent_drives_velocity() -> None:
    yaw_ninety = QuaternionWxyz(
        w=math.sqrt(0.5),
        x=0.0,
        y=0.0,
        z=math.sqrt(0.5),
    )
    body = Vector3(x=1.0, y=0.2, z=-0.1)
    round_trip = world_enu_to_body(yaw_ninety, body_to_world_enu(yaw_ninety, body))
    assert (round_trip.x, round_trip.y, round_trip.z) == pytest.approx((body.x, body.y, body.z))
    world_right = body_to_world_enu(
        IDENTITY,
        Vector3(x=0.0, y=1.0, z=0.0),
    )
    assert (world_right.x, world_right.y, world_right.z) == pytest.approx((0.0, -1.0, 0.0))
    assert world_enu_to_body(IDENTITY, world_right).y == pytest.approx(1.0)

    call_id = "model-" + "a" * 24
    path_sha256 = "b" * 64
    intent = body_control_intent_for_target(
        source_expert="local-navigation-policy",
        model_call_id=call_id,
        navigation_snapshot_sha256=path_sha256,
        task_reference_sha256="a" * 64,
        current_position_world_enu_m=Vector3(x=0.0, y=0.0, z=1.0),
        target_position_world_enu_m=Vector3(x=0.0, y=0.2, z=1.0),
        body_orientation_world_from_body=yaw_ninety,
        generated_at_unix_ms=5_000,
        maximum_speed_mps=1.0,
        maximum_acceleration_mps2=2.0,
        maximum_jerk_mps3=10.0,
    )
    vehicle = VehicleAsset(
        asset_id="test-x500",
        name="Test X500",
        dry_mass_kg=2.0,
        max_takeoff_mass_kg=3.0,
        body_radius_m=0.38,
        body_height_m=0.43,
        max_speed_mps=2.0,
        max_acceleration_mps2=4.0,
        qualified_range_m=100.0,
        reserve_battery_percent=30.0,
        max_pickup_payload_kg=0.2,
        sensors=["depth-camera", "stereo-vio"],
    )
    observation = RuntimeLocalSafetyObservation(
        sequence=1,
        observed_at_unix_ms=5_000,
        source="onboard",
        stream_healthy=True,
        stream_age_seconds=0.01,
        localization_covariance_m2=0.01,
        current_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        body_orientation_world_from_body=yaw_ninety,
        target_position_m=Vector3(x=0.0, y=0.2, z=1.0),
    )
    command = evaluate_runtime_local_safety(
        observation=observation,
        vehicle=vehicle,
        static_primitives=[],
        required_clearance_m=0.35,
        generated_at_unix_ms=5_010,
        navigation_goal_id="goal-1",
        navigation_control_authority="model-required",
        model_navigation_authorized=True,
        model_call_id=call_id,
        model_selected_candidate_id=None,
        model_path_sha256="a" * 64,
        model_navigation_snapshot_sha256=path_sha256,
        model_authority_reason="model-path-lease-active",
        requested_control_intent=intent,
    )

    assert command.requested_control_intent == intent
    assert command.decision.control_source == "local-model-body-control"
    assert command.decision.selected_velocity_mps.y > 0.0
    assert command.decision.maximum_acceleration_mps2 == 2.0
    assert command.decision.maximum_jerk_mps3 == 10.0
    assert command.model_path_sha256 != command.model_navigation_snapshot_sha256
    assert command.valid_until_unix_ms <= intent.valid_until_unix_ms
    serialized = command.model_dump(mode="json")
    with pytest.raises(ValueError, match="input snapshot"):
        RuntimeLocalSafetyCommand.model_validate({
            **serialized, "model_navigation_snapshot_sha256": command.model_path_sha256,
        })
    with pytest.raises(ValueError, match="observation budget"):
        RuntimeLocalSafetyCommand.model_validate({
            **serialized, "valid_until_unix_ms": intent.valid_until_unix_ms + 1,
        })
    # The intent lease is an independent upper bound, including legacy
    # commands with no optional observation-budget receipt.
    with pytest.raises(ValueError, match="outlive"):
        RuntimeLocalSafetyCommand.model_validate({
            **serialized, "observation_budget": None,
            "valid_until_unix_ms": intent.valid_until_unix_ms + 1,
        })
    recovery_command = evaluate_runtime_local_safety(
        observation=observation, vehicle=vehicle, static_primitives=[],
        required_clearance_m=0.35, generated_at_unix_ms=5_010,
        maximum_speed_mps=0.03, tracking_recovery_active=True,
        navigation_goal_id="goal-1", navigation_control_authority="model-required",
        model_navigation_authorized=True, model_call_id=call_id,
        model_path_sha256=intent.task_reference_sha256,
        model_navigation_snapshot_sha256=path_sha256, requested_control_intent=intent,
    )
    selected_velocity = recovery_command.decision.selected_velocity_mps
    assert math.sqrt(sum(v**2 for v in selected_velocity.model_dump().values())) <= 0.03 + 1e-9


def test_body_control_rejects_an_overlong_command_lease() -> None:
    with pytest.raises(ValueError, match="control horizon"):
        body_control_intent_for_target(
            source_expert="local-navigation-policy",
            model_call_id="model-" + "c" * 24,
            navigation_snapshot_sha256="d" * 64,
            task_reference_sha256="a" * 64,
            current_position_world_enu_m=Vector3(x=0.0, y=0.0, z=0.0),
            target_position_world_enu_m=Vector3(x=1.0, y=0.0, z=0.0),
            body_orientation_world_from_body=IDENTITY,
            generated_at_unix_ms=10_000,
            maximum_speed_mps=1.0,
            maximum_acceleration_mps2=1.0,
            maximum_jerk_mps3=5.0,
            validity_milliseconds=3_000,
        )


@pytest.mark.parametrize("control_scale", [1.0, 0.4])
def test_continuous_pilot_axes_map_to_physical_body_control(control_scale: float) -> None:
    intent = body_control_intent_for_pilot_control(
        source_expert="precision-maneuver-policy",
        model_call_id="model-" + "e" * 24,
        navigation_snapshot_sha256="f" * 64,
        task_reference_sha256="a" * 64,
        pilot_control=NormalizedPilotControl(
            forward_axis=0.5,
            right_axis=-0.25,
            up_axis=0.75,
            yaw_axis=-0.4,
        ),
        generated_at_unix_ms=20_000,
        maximum_horizontal_speed_mps=2.0,
        maximum_vertical_speed_mps=0.8,
        maximum_yaw_rate_dps=30.0,
        maximum_acceleration_mps2=1.5,
        maximum_jerk_mps3=6.0,
        harness_control_scale=control_scale,
    )

    assert intent.control_origin == "continuous-model-output"
    assert intent.forward_velocity_mps == pytest.approx(1.0 * control_scale)
    assert intent.right_velocity_mps == pytest.approx(-0.5 * control_scale)
    assert intent.up_velocity_mps == pytest.approx(0.6 * control_scale)
    assert intent.yaw_rate_dps == pytest.approx(-12.0 * control_scale)
    assert intent.harness_control_scale == control_scale
    assert intent.maximum_acceleration_mps2 == 1.5
    assert intent.valid_until_unix_ms == 20_500


def test_continuous_pilot_axes_reject_invalid_physical_limits() -> None:
    with pytest.raises(ValueError, match="physical limits"):
        body_control_intent_for_pilot_control(
            source_expert="local-navigation-policy",
            model_call_id="model-" + "1" * 24,
            navigation_snapshot_sha256="2" * 64,
            task_reference_sha256="a" * 64,
            pilot_control=NormalizedPilotControl(
                forward_axis=0.0,
                right_axis=0.0,
                up_axis=0.0,
                yaw_axis=0.0,
            ),
            generated_at_unix_ms=30_000,
            maximum_horizontal_speed_mps=0.0,
            maximum_vertical_speed_mps=0.5,
            maximum_yaw_rate_dps=20.0,
            maximum_acceleration_mps2=1.0,
            maximum_jerk_mps3=5.0,
        )


@pytest.mark.parametrize("mutation", ["features", "mask", "ready", "hash", "roles"])
def test_feature_snapshot_rejects_tampered_fusion(mutation: str) -> None:
    geometry = encode_metric_geometry(_scan(), sensor_mount=_mount(), encoded_at_unix_ms=1_001)
    snapshot = fuse_realtime_features(
        [geometry], captured_at_unix_ms=1_001, required_roles=["metric-geometry-encoder"]
    )
    payload = snapshot.model_dump(mode="json")
    if mutation == "features":
        payload["fused_features"][0] = 0.99
    elif mutation == "mask":
        payload["fused_valid_mask"][0] = 0.0
    elif mutation == "ready":
        payload["ready_for_control"] = False
    elif mutation == "hash":
        payload["snapshot_sha256"] = "0" * 64
    else:
        payload["required_roles"].append("flight-state-encoder")
    with pytest.raises(ValueError):
        RealtimeFeatureSnapshot.model_validate(payload)


def test_stale_dynamic_track_is_masked_and_blocks_control() -> None:
    encoding = encode_dynamic_targets(
        [DynamicObstacleObservation(
            obstacle_id="stale-target", position_m=Vector3(x=1.0, y=0.0, z=1.0),
            velocity_mps=Vector3(x=-1.0, y=0.0, z=0.0), radius_m=0.4,
            height_m=1.0, confidence=1.0, age_seconds=0.49,
        )],
        body_position_world_enu_m=Vector3(x=0.0, y=0.0, z=1.0),
        body_orientation_world_from_body=IDENTITY,
        body_velocity_world_enu_mps=Vector3(x=0.0, y=0.0, z=0.0),
        observed_at_unix_ms=1000, encoded_at_unix_ms=1020,
    )
    assert encoding.valid_mask == [0.0] * DYNAMIC_TARGET_FEATURE_COUNT
    assert "DYNAMIC_TARGET_STALE" in encoding.issue_codes
    snapshot = fuse_realtime_features(
        [encoding], captured_at_unix_ms=1020, required_roles=["dynamic-target-encoder"]
    )
    assert not snapshot.ready_for_control


def test_fusion_timestamp_does_not_extend_individual_sensor_deadline() -> None:
    snapshot = complete_feature_snapshot(1000)
    assert snapshot.fresh_at(1250)
    assert not snapshot.fresh_at(1251)
    assert not snapshot.fresh_at(999)


def test_control_deadline_is_unavailable_at_exact_expiry() -> None:
    snapshot = complete_feature_snapshot(1000)
    assert snapshot.control_deadline_unix_ms(now_unix_ms=1249) == 1250
    # Retention is inclusive; permission for a new action is strictly exclusive.
    assert snapshot.control_deadline_unix_ms(now_unix_ms=1250) == 0


@pytest.mark.parametrize("now", [999, 1000, 1249, 1250, 1251])
def test_single_parse_control_admission_matches_independent_checks(now: int) -> None:
    from dronedream_agent_core.realtime_feature_encoders import parse_realtime_control_input
    snapshot = complete_feature_snapshot(1000)
    detached, fresh, deadline = parse_realtime_control_input(snapshot, now_unix_ms=now)
    assert fresh == snapshot.fresh_at(now)
    assert deadline == snapshot.control_deadline_unix_ms(now_unix_ms=now)
    snapshot.encodings[0].features[0] = 999.0
    assert detached.encodings[0].features[0] != 999.0
    with pytest.raises(ValueError):
        parse_realtime_control_input(snapshot, now_unix_ms=now)


def test_snapshot_constructor_revalidates_borrowed_encoding_instances() -> None:
    snapshot = complete_feature_snapshot(1000)
    payload = snapshot.model_dump(mode="python")
    payload["encodings"] = snapshot.encodings
    snapshot.encodings[0].valid_mask[0] = .5
    with pytest.raises(ValueError):
        RealtimeFeatureSnapshot.model_validate(payload)


def test_large_finite_direction_is_normalized_without_squaring_overflow() -> None:
    scan = _scan()
    for sample in scan.samples:
        sample.direction_sensor = Vector3(
            x=sample.direction_sensor.x * 1e200,
            y=sample.direction_sensor.y * 1e200,
            z=0.,
        )
    actual = encode_metric_geometry(scan, sensor_mount=_mount(), encoded_at_unix_ms=1001)
    ordinary = encode_metric_geometry(_scan(), sensor_mount=_mount(), encoded_at_unix_ms=1001)
    assert actual.features == pytest.approx(ordinary.features)
    assert actual.valid_mask == ordinary.valid_mask


@pytest.mark.parametrize("source,timestamp,issue", [
    ("other-ekf", 1200, "FLIGHT_STATE_SOURCE_CHANGED"),
    ("px4-ekf", 2000, "FLIGHT_STATE_SAMPLE_GAP"),
])
def test_state_history_cannot_mix_sources_or_discontinuous_samples(
    source, timestamp, issue,
) -> None:
    encoder = FlightStateEncoder(history_length=4)
    for observed_at in (1000, 1050, 1100, 1150):
        encoder.encode(FlightStateSample(
            source_id="px4-ekf", observed_at_unix_ms=observed_at,
            orientation_world_from_body=IDENTITY,
            velocity_world_enu_mps=Vector3(x=3.0, y=0.0, z=0.0),
            localization_covariance_m2=0.01,
        ), encoded_at_unix_ms=observed_at)
    result = encoder.encode(FlightStateSample(
        source_id=source, observed_at_unix_ms=timestamp,
        orientation_world_from_body=IDENTITY,
        velocity_world_enu_mps=Vector3(x=0.0, y=0.0, z=0.0),
        localization_covariance_m2=0.01,
    ), encoded_at_unix_ms=timestamp)
    assert issue in result.issue_codes
    assert "FLIGHT_STATE_HISTORY_WARMING" in result.issue_codes
    assert result.coverage == 0.25
    assert result.features[-6:] == [0.0] * 6
