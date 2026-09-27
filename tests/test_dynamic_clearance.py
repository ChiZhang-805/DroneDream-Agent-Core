from itertools import product

import pytest

from dronedream_agent_core.contracts import (
    DynamicObstacleObservation,
    OnboardPerceptionFrame,
    RangeRayObservation,
    Vector3,
)
from dronedream_agent_core.dynamic_clearance import (
    fresh_free_voxels,
    reachable_track_box_observed_free,
)
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.perception_runtime import RuntimePerceptionFusion


def track():
    return DynamicObstacleObservation(
        obstacle_id="moving-object",
        position_m=Vector3(x=1.4, y=0, z=0),
        velocity_mps=Vector3(x=0.1, y=0, z=0),
        radius_m=0.1,
        height_m=0.2,
        confidence=0.9,
        age_seconds=0,
    )


def frame(sequence, timestamp, *, detected=False, ray_time=None):
    rays = [
        RangeRayObservation(
            origin_m=Vector3(x=0, y=y, z=z),
            endpoint_m=Vector3(x=4, y=y, z=z),
            hit=False,
            confidence=0.95,
            observed_at_monotonic_seconds=timestamp / 1000 if ray_time is None else ray_time,
        )
        for y, z in product((-0.375, -0.125, 0.125, 0.375), repeat=2)
    ]
    return OnboardPerceptionFrame(
        sensor_id="depth",
        sequence=sequence,
        observed_at_unix_ms=timestamp,
        localization_position_m=Vector3(x=0, y=0, z=0),
        localization_velocity_mps=Vector3(x=0, y=0, z=0),
        localization_covariance_m2=0.000001,
        range_rays=rays,
        dynamic_obstacles=[track()] if detected else [],
    )


def fusion():
    return RuntimePerceptionFusion(
        world=MetricVoxelMap(
            resolution_m=0.25,
            minimum_bound_m=Vector3(x=-1, y=-1, z=-1),
            maximum_bound_m=Vector3(x=5, y=1, z=1),
        ),
        accepted_sensor_ids={"depth"},
        maximum_dynamic_acceleration_mps2=1.0,
    )


def test_lost_track_requires_two_independent_full_volume_scans():
    f = fusion()
    f.ingest(frame(1, 1000, detected=True), now_unix_ms=1000, now_monotonic_seconds=1)
    f.ingest(frame(2, 1100), now_unix_ms=1100, now_monotonic_seconds=1.1)
    assert len(f.latest_frame.dynamic_obstacles) == 1
    previous = f.latest_frame
    observation_count = f.world.observation_count
    # A re-read depth frame is now rejected before either map evidence or
    # track-clearance state can be renewed, even with a new wrapper sequence.
    with pytest.raises(ValueError, match="METRIC_SCAN_SOURCE_NOT_NEW"):
        f.ingest(frame(3, 1150, ray_time=1.1), now_unix_ms=1150, now_monotonic_seconds=1.15)
    assert f.latest_frame == previous
    assert f.world.observation_count == observation_count
    assert len(f.latest_frame.dynamic_obstacles) == 1
    health = f.ingest(frame(4, 1200), now_unix_ms=1200, now_monotonic_seconds=1.2)
    assert health.stream_healthy
    assert not f.latest_frame.dynamic_obstacles
    receipt = f.last_dynamic_clearances[0]
    assert receipt["obstacle_id"] == "moving-object"
    assert receipt["prior_scan_sha256"] != receipt["scan_sha256"]


def test_a_center_ray_cannot_clear_unseen_sides_or_vertical_volume():
    scan = frame(2, 1100)
    free = fresh_free_voxels(scan.range_rays[:1], resolution_m=0.25, now_monotonic_seconds=1.1)
    assert not reachable_track_box_observed_free(
        track(),
        age_seconds=0.1,
        free_voxels=free,
        resolution_m=0.25,
        localization_variance_m2=0.000001,
        maximum_acceleration_mps2=1,
    )


def test_long_lost_track_preserves_real_age_and_denies_control_without_crashing_fusion():
    f = fusion()
    f.ingest(frame(1, 1000, detected=True), now_unix_ms=1000, now_monotonic_seconds=1)
    health = f.ingest(frame(2, 61_000), now_unix_ms=61_000, now_monotonic_seconds=61)
    assert not health.stream_healthy
    assert "DYNAMIC_TRACK_OBSERVATION_STALE:moving-object" in health.issue_codes
    assert len(f.latest_frame.dynamic_obstacles) == 1
    assert f.latest_frame.dynamic_obstacles[0].age_seconds == 60
    assert not f.last_dynamic_clearances
    observation = f.local_safety_observation(target_position_m=Vector3(x=4, y=0, z=0),
                                            now_unix_ms=61_000)
    assert not observation.stream_healthy
    assert observation.dynamic_obstacles[0].age_seconds == 60


# 功能：过期轨迹不触发无效全图清除计算，但保留原观测、年龄和阻塞状态。
# 输入：一条超过短时清除包络期限的失联轨迹；输出：不计算自由网格且不会误放行。
def test_long_lost_track_skips_unusable_clearance_work_without_becoming_clear(monkeypatch):
    f = fusion()
    f.ingest(frame(1, 1000, detected=True), now_unix_ms=1000, now_monotonic_seconds=1)
    def forbidden(*args, **kwargs):
        raise AssertionError('an expired envelope cannot consume a free-space scan')
    monkeypatch.setattr('dronedream_agent_core.perception_runtime.fresh_free_voxels', forbidden)
    health = f.ingest(frame(2, 2001), now_unix_ms=2001, now_monotonic_seconds=2.001)
    assert not health.stream_healthy
    assert f.latest_frame.dynamic_obstacles[0].age_seconds == 1.001
    assert not f.last_dynamic_clearances


@pytest.mark.parametrize("kind", ["old", "future", "low-confidence", "over-budget"])
def test_unusable_rays_never_provide_clearance(kind):
    scan = frame(2, 1100)
    for ray in scan.range_rays:
        if kind == "old":
            ray.observed_at_monotonic_seconds = 0.1
        elif kind == "future":
            ray.observed_at_monotonic_seconds = 2.0
        elif kind == "low-confidence":
            ray.confidence = 0.1
    free = fresh_free_voxels(
        scan.range_rays,
        resolution_m=0.25,
        now_monotonic_seconds=1.1,
        maximum_samples=1 if kind == "over-budget" else 100000,
    )
    assert not free


def test_hit_blocks_clearance_even_if_other_rays_cross_that_voxel():
    scan = frame(2, 1100)
    hit = scan.range_rays[0].model_copy(deep=True)
    hit.endpoint_m = Vector3(x=1.4, y=-0.375, z=-0.375)
    hit.hit = True
    free = fresh_free_voxels([*scan.range_rays, hit], resolution_m=0.25, now_monotonic_seconds=1.1)
    assert (5, -2, -2) not in free


def test_acceleration_uncertainty_and_old_tracks_cannot_be_cleared_by_small_volume():
    scan = frame(2, 1100)
    free = fresh_free_voxels(scan.range_rays, resolution_m=0.25, now_monotonic_seconds=1.1)
    for age, variance, acceleration in ((0.9, 0.25, 12.0), (1.1, 0.000001, 1.0), (0.1, 0.0, 1.0)):
        assert not reachable_track_box_observed_free(
            track(),
            age_seconds=age,
            free_voxels=free,
            resolution_m=0.25,
            localization_variance_m2=variance,
            maximum_acceleration_mps2=acceleration,
        )


def test_overflowing_motion_prediction_keeps_lost_track_unresolved():
    obstacle = track().model_copy(update={
        "position_m": Vector3(x=1.7e308, y=0, z=0),
        "velocity_mps": Vector3(x=1.7e308, y=0, z=0),
    })
    assert not reachable_track_box_observed_free(
        obstacle, age_seconds=1., free_voxels=set(), resolution_m=.05,
        localization_variance_m2=.1, maximum_acceleration_mps2=1.,
    )


def test_overflowing_ray_does_not_manufacture_partial_free_space():
    ray = RangeRayObservation(
        origin_m=Vector3(x=-1.7e308, y=0, z=0),
        endpoint_m=Vector3(x=1.7e308, y=0, z=0),
        hit=False, confidence=.9, observed_at_monotonic_seconds=1.,
    )
    assert not fresh_free_voxels([ray], resolution_m=.25, now_monotonic_seconds=1.)
