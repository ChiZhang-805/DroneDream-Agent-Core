"""Adversarial tests at the sensor-to-state judgment boundary, not flight accuracy."""

import math

import pytest
from test_depth_obstacle_tracker import _frame
from test_realtime_feature_encoders import _mount, _scan

from dronedream_agent_core.contracts import DynamicObstacleObservation, QuaternionWxyz, Vector3
from dronedream_agent_core.depth_obstacle_tracker import DepthMotionTracker
from dronedream_agent_core.realtime_feature_encoders import (
    FlightStateEncoder,
    FlightStateSample,
    RealtimeFeatureSnapshot,
    encode_dynamic_targets,
    encode_metric_geometry,
    fuse_realtime_features,
)


def _state(timestamp, **updates):
    return FlightStateSample(
        source_id="native-state", observed_at_unix_ms=timestamp,
        orientation_world_from_body=QuaternionWxyz(w=1, x=0, y=0, z=0),
        velocity_world_enu_mps=Vector3(x=0, y=0, z=0),
        acceleration_body_mps2=Vector3(x=0, y=0, z=0),
        angular_velocity_body_rad_s=Vector3(x=0, y=0, z=0),
        localization_covariance_m2=.01,
    ).model_copy(update=updates)


def _cloud(sequence, *, broad=False):
    frame = _frame(sequence, 2.)
    if broad:
        points = [(x*.25, y*.25, z*.25)
                  for x in range(13) for y in range(13) for z in (0, 1)]
    else:
        # Distinct points, strongly nonuniform density along a connected object.
        points = [(0., 0., z*.01) for z in range(20)]
        points += [(0., y*.25, y*.25) for y in range(1, 12)]
    frame.range_rays = [frame.range_rays[0].model_copy(update={
        "endpoint_m": Vector3(x=2.+x+sequence*.1, y=y, z=1.+z),
    }) for x, y, z in points]
    return frame


@pytest.mark.parametrize("broad", [False, True])
def test_track_cylinder_contains_every_observed_endpoint(broad):
    tracker = DepthMotionTracker()
    tracker.update(_cloud(1, broad=broad), observed_at_monotonic_seconds=.1)
    frame = _cloud(2, broad=broad)
    track, = tracker.update(frame, observed_at_monotonic_seconds=.2)
    for ray in frame.range_rays:
        point = ray.endpoint_m
        assert math.hypot(point.x-track.position_m.x, point.y-track.position_m.y) <= track.radius_m
        assert abs(point.z-track.position_m.z) <= track.height_m/2


@pytest.mark.parametrize("case", ["zero-confidence", "duplicate-points"])
def test_invalid_depth_evidence_cannot_accumulate_motion_confidence(case):
    tracker = DepthMotionTracker()
    for sequence in range(1, 8):
        frame = _frame(sequence, 1.+sequence*.1)
        if case == "zero-confidence":
            for ray in frame.range_rays:
                ray.confidence = 0
        else:
            frame.range_rays = [frame.range_rays[0].model_copy(deep=True) for _ in range(3)]
        assert tracker.update(frame, observed_at_monotonic_seconds=sequence*.1) == []


def test_tracker_rejects_mixed_scan_clock_without_consuming_next_frame():
    tracker = DepthMotionTracker()
    tracker.update(_frame(1, 1.), observed_at_monotonic_seconds=.1)
    bad = _frame(2, 1.2)
    bad.range_rays[-1].observed_at_monotonic_seconds = .1
    with pytest.raises(ValueError, match="CLOCK"):
        tracker.update(bad, observed_at_monotonic_seconds=.2)
    assert len(tracker.update(_frame(2, 1.2), observed_at_monotonic_seconds=.2)) == 1


@pytest.mark.parametrize("case", ["future", "negative-clock", "invalid-covariance"])
def test_rejected_state_does_not_poison_temporal_history(case):
    encoder = FlightStateEncoder(history_length=4)
    encoder.encode(_state(1000), encoded_at_unix_ms=1000)
    sample, now = _state(1100), 1100
    if case == "future":
        now = 1050
    elif case == "negative-clock":
        now = -1
    else:
        sample = sample.model_copy(update={"localization_covariance_m2": -.01})
    with pytest.raises(ValueError):
        encoder.encode(sample, encoded_at_unix_ms=now)
    result = encoder.encode(_state(1050), encoded_at_unix_ms=1050)
    assert result.coverage == .5


@pytest.mark.parametrize("updates", [
    {"localization_covariance_m2": None}, {"localization_covariance_m2": .26},
    {"acceleration_body_mps2": None}, {"angular_velocity_body_rad_s": None},
])
def test_unknown_or_uncertain_flight_state_never_becomes_control_ready(updates):
    encoder = FlightStateEncoder(history_length=4)
    for stamp in (1000, 1050, 1100, 1150):
        encoding = encoder.encode(_state(stamp, **updates), encoded_at_unix_ms=stamp)
    snapshot = fuse_realtime_features([encoding], captured_at_unix_ms=1150,
                                      required_roles=["flight-state-encoder"])
    assert not snapshot.ready_for_control


def test_geometry_rejects_future_clock_and_binds_localization_uncertainty():
    with pytest.raises(ValueError, match="future"):
        encode_metric_geometry(_scan(), sensor_mount=_mount(), encoded_at_unix_ms=999)
    first = encode_metric_geometry(_scan(), sensor_mount=_mount(), encoded_at_unix_ms=1000)
    changed = _scan().model_copy(update={"localization_covariance_m2": .2})
    second = encode_metric_geometry(changed, sensor_mount=_mount(), encoded_at_unix_ms=1000)
    assert first.source_sha256 != second.source_sha256


@pytest.mark.parametrize("case", ["negative-covariance", "invalid-confidence", "invalid-mount"])
def test_geometry_revalidates_even_when_called_without_the_sensor_bridge(case):
    scan, mount = _scan(), _mount()
    if case == "negative-covariance":
        scan = scan.model_copy(update={"localization_covariance_m2": -.1})
    elif case == "invalid-confidence":
        scan.samples[0] = scan.samples[0].model_copy(update={"confidence": 2.})
    else:
        mount = mount.model_copy(update={"minimum_range_m": -1.})
    with pytest.raises(ValueError):
        encode_metric_geometry(scan, sensor_mount=mount, encoded_at_unix_ms=1000)


def test_dynamic_encoder_rejects_negative_source_age_not_clamping_it_to_fresh():
    obstacle = DynamicObstacleObservation(
        obstacle_id="unknown-track", position_m=Vector3(x=2., y=0., z=1.),
        velocity_mps=Vector3(x=0., y=0., z=0.), radius_m=.3, height_m=.6,
        confidence=.8, age_seconds=0.,
    ).model_copy(update={"age_seconds": -.1})
    with pytest.raises(ValueError):
        encode_dynamic_targets(
            [obstacle], body_position_world_enu_m=Vector3(x=0., y=0., z=1.),
            body_orientation_world_from_body=QuaternionWxyz(w=1., x=0., y=0., z=0.),
            body_velocity_world_enu_mps=Vector3(x=0., y=0., z=0.),
            observed_at_unix_ms=1000, encoded_at_unix_ms=1000,
        )


def test_fusion_owns_its_encoding_and_revalidates_mutated_values():
    geometry = encode_metric_geometry(_scan(), sensor_mount=_mount(), encoded_at_unix_ms=1000)
    snapshot = fuse_realtime_features([geometry], captured_at_unix_ms=1000,
                                      required_roles=["metric-geometry-encoder"])
    geometry.features[0] = .987
    RealtimeFeatureSnapshot.model_validate(snapshot.model_dump())
    geometry.valid_mask[0] = .5
    with pytest.raises(ValueError, match="binary"):
        fuse_realtime_features([geometry], captured_at_unix_ms=1000,
                               required_roles=["metric-geometry-encoder"])


def test_discarded_tracking_proposal_cannot_pollute_later_motion():
    tracker = DepthMotionTracker()
    proposal = tracker.prepare(_frame(1, 1.), observed_at_monotonic_seconds=.1)
    with pytest.raises(ValueError, match="SUPERSEDED"):
        DepthMotionTracker().commit(proposal)
    # Simulate fusion rejecting this frame. No commit, no first detection.
    assert tracker.update(_frame(2, 1.2), observed_at_monotonic_seconds=.2) == []
    with pytest.raises(ValueError, match="SUPERSEDED"):
        tracker.commit(proposal)


@pytest.mark.parametrize("changed", [{"sensor_id": "different-depth"}, {"sequence": 1}])
def test_tracker_source_and_sequence_cannot_change_or_replay(changed):
    tracker = DepthMotionTracker()
    tracker.update(_frame(1, 1.), observed_at_monotonic_seconds=.1)
    with pytest.raises(ValueError):
        tracker.update(_frame(2, 1.2).model_copy(update=changed),
                       observed_at_monotonic_seconds=.2)
    assert len(tracker.update(_frame(2, 1.2), observed_at_monotonic_seconds=.2)) == 1


def test_tracker_confidence_never_exceeds_input_coverage_or_ray_quality():
    tracker = DepthMotionTracker()
    for sequence in range(1, 10):
        frame = _frame(sequence, 1.+sequence*.1)
        frame.source_coverage = .2
        for ray in frame.range_rays:
            ray.confidence = .4
        tracks = tracker.update(frame, observed_at_monotonic_seconds=sequence*.1)
        assert all(track.confidence <= .2 for track in tracks)
    assert tracks


def test_static_map_is_owned_and_large_geometry_does_not_expand_unbounded_bins():
    wall = {"center_x": 1.1, "center_y": 0., "center_z": 1.,
            "size_x": 1., "size_y": 100000., "size_z": 100000., "yaw_rad": 0.}
    tracker = DepthMotionTracker(known_static_primitives=[wall])
    wall["center_x"] = 100.
    for seq in (1, 2):
        assert tracker.update(_frame(seq, 1.+seq*.1),
                              observed_at_monotonic_seconds=seq*.1) == []


def test_encoding_failure_after_history_preparation_is_atomic():
    encoder = FlightStateEncoder(history_length=4)
    encoder.encode(_state(1000), encoded_at_unix_ms=1000)
    bad = _state(1100, velocity_world_enu_mps=Vector3(x=1e308, y=1e308, z=1e308))
    with pytest.raises((ValueError, OverflowError)):
        encoder.encode(bad, encoded_at_unix_ms=1100)
    assert encoder.encode(_state(1050), encoded_at_unix_ms=1050).coverage == .5


@pytest.mark.parametrize("clock", [math.nan, math.inf, -1, True])
def test_invalid_fusion_clock_cannot_commit_map_or_registry(clock):
    from test_perception_runtime import _frame as fusion_frame
    from test_perception_runtime import _world

    from dronedream_agent_core.perception_runtime import RuntimePerceptionFusion

    target = _world()
    fusion = RuntimePerceptionFusion(world=target, accepted_sensor_ids={"front-lidar"})
    before = target.snapshot()
    with pytest.raises(ValueError, match="CLOCK"):
        fusion.ingest(fusion_frame(), now_unix_ms=clock, now_monotonic_seconds=1.)
    assert target.snapshot() == before
    assert fusion.frozen_frame is None


def test_bridge_mount_and_input_scan_remain_owned():
    from dronedream_agent_core.sensor_bridge import MetricRangeSensorBridge

    source = _scan()
    bridge = MetricRangeSensorBridge(_mount())
    bridge.mount.translation_body_m.x = 100
    frame = bridge.assemble(source)
    original = frame.model_dump()
    source.body_position_world_enu_m.x = 50
    assert frame.model_dump() == original
    assert bridge.mount.translation_body_m.x == 0
    bad = _scan().model_copy(update={"source_coverage": 2.})
    with pytest.raises(ValueError):
        bridge.assemble(bad)


def test_measured_covariance_bound_covers_random_correlated_state_errors():
    import numpy as np

    from dronedream_agent_core.localization_evidence import position_variance_bound_m2

    rng = np.random.default_rng(734)
    for _ in range(200):
        matrix = rng.normal(size=(3, 3)) * .05
        covariance = matrix @ matrix.T
        packed = [0.]*21
        for index, (row, col) in zip((0, 1, 2, 6, 7, 11),
                                    ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)),
                                    strict=True):
            packed[index] = float(covariance[row, col])
        actual = float(np.linalg.eigvalsh(covariance)[-1])
        assert position_variance_bound_m2(packed) + 1e-12 >= actual
        for index in (1, 2, 7):
            packed[index] = None
        assert position_variance_bound_m2(packed) + 1e-12 >= actual
