"""Ownership and timestamp contracts for fast perception snapshots."""

from test_perception_runtime import _tracked_frame, _world

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.perception_runtime import RuntimePerceptionFusion


def test_snapshot_is_detached_at_every_mutable_level_and_keeps_source_clock():
    fusion = RuntimePerceptionFusion(world=_world(), accepted_sensor_ids={"front-lidar"})
    assert not fusion.has_frame
    assert fusion.latest_frame is None
    source = _tracked_frame(sequence=1, observed_at_unix_ms=1000,
                            obstacle_x=2., obstacle_speed_x=0.)
    fusion.ingest(source, now_unix_ms=1000, now_monotonic_seconds=1.)
    assert fusion.has_frame
    left, right = fusion.latest_frame, fusion.latest_frame
    digest = sha256_json(right)
    assert sha256_json(left) == digest
    assert left.observed_at_unix_ms == 1000
    assert left.range_rays[0].observed_at_monotonic_seconds == 1.
    left.localization_position_m.x += .1
    left.localization_velocity_mps.x += .1
    left.range_rays[0].origin_m.x += .1
    left.range_rays[0].endpoint_m.x += .1
    left.range_rays[0].confidence = .1
    left.range_rays.append(left.range_rays[0])
    left.dynamic_obstacles[0].velocity_mps.x += .1
    left.dynamic_obstacles[0].position_m.x += .1
    left.dynamic_obstacles.clear()
    # Neither another consumer, the admitted source, nor the fusion is changed.
    assert sha256_json(fusion.latest_frame) == digest == sha256_json(right)
    assert source.range_rays[0].confidence == .95
    source.range_rays.clear()
    source.dynamic_obstacles[0].position_m.x += 1.
    assert sha256_json(fusion.latest_frame) == digest
    # Health queries keep aging original evidence; snapshots do not renew it.
    assert not fusion.health(now_unix_ms=3000, now_monotonic_seconds=3.).stream_healthy
