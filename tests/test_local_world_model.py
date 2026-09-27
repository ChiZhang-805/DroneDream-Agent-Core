import math
import random
import time

import pytest

from dronedream_agent_core.contracts import (
    DynamicObstacleObservation,
    RangeRayObservation,
    Vector3,
)
from dronedream_agent_core.local_world_model import MetricVoxelMap


def test_hot_path_preserves_original_ray_sampling_and_order():
    # Compare every visited voxel/update, including negative coordinates,
    # zero-length rays, boundary endpoints, hits and repeated intersections.
    rng = random.Random(51)
    world = MetricVoxelMap(resolution_m=.25,
                           minimum_bound_m=Vector3(x=-4, y=-4, z=-4),
                           maximum_bound_m=Vector3(x=4, y=4, z=4))
    points = [(-4., -4., -4.), (4., 4., 4.), (0., 0., 0.)]
    points += [tuple(rng.uniform(-4, 4) for _ in range(3)) for _ in range(100)]
    for index, origin in enumerate(points):
        endpoint = points[-index - 1] if index % 3 else origin
        distance = math.dist(origin, endpoint)
        count = max(1, math.ceil(distance / (world.resolution_m * .45)))
        keys = []
        for n in range(count + 1):
            ratio = n / count
            key = tuple(int(math.floor((origin[a] + (endpoint[a] - origin[a]) * ratio
                                       - world._minimum[a]) * (1. / world.resolution_m)))
                        for a in range(3))
            if not keys or keys[-1] != key:
                keys.append(key)
        hit = bool(index % 2)
        events = []
        world._update_log_odds = lambda key, sink=events, **kwargs: sink.append((key, kwargs))
        world.integrate_ray(RangeRayObservation(
            origin_m=Vector3(**dict(zip(('x', 'y', 'z'), origin, strict=True))),
            endpoint_m=Vector3(**dict(zip(('x', 'y', 'z'), endpoint, strict=True))),
            confidence=.92, hit=hit, observed_at_monotonic_seconds=1.,
        ))
        assert [event[0] for event in events] == keys
        assert all(event[1]['measurement'] < 0 for event in events[:-1])
        assert (events[-1][1]['measurement'] > 0) == hit


def _map() -> MetricVoxelMap:
    world = MetricVoxelMap(
        resolution_m=1.0,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=9.99, y=5.99, z=0.99),
    )
    world.mark_box(
        minimum_m=Vector3(x=0.01, y=0.01, z=0.01),
        maximum_m=Vector3(x=9.9, y=5.9, z=0.9),
        occupied=False,
        confidence=0.70,
    )
    return world


def test_clearance_aware_path_uses_observed_doorway() -> None:
    world = _map()
    world.mark_box(
        minimum_m=Vector3(x=4.01, y=0.01, z=0.01),
        maximum_m=Vector3(x=4.9, y=1.9, z=0.9),
        occupied=True,
    )
    world.mark_box(
        minimum_m=Vector3(x=4.01, y=3.01, z=0.01),
        maximum_m=Vector3(x=4.9, y=5.9, z=0.9),
        occupied=True,
    )
    path = world.plan_path(
        start_m=Vector3(x=1.5, y=2.5, z=0.5),
        goal_m=Vector3(x=8.5, y=2.5, z=0.5),
        required_clearance_m=0.1,
    )
    assert path[0].x == 1.5
    assert path[-1].x == 8.5
    assert any(4.0 <= point.x < 5.0 and 2.0 <= point.y < 3.0 for point in path)


def test_unknown_space_is_not_treated_as_free() -> None:
    world = MetricVoxelMap(
        resolution_m=0.5,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=4.0, y=1.0, z=1.0),
    )
    world.integrate_ray(
        RangeRayObservation(
            origin_m=Vector3(x=0.1, y=0.25, z=0.25),
            endpoint_m=Vector3(x=1.9, y=0.25, z=0.25),
            hit=False,
            confidence=0.95,
            observed_at_monotonic_seconds=1.0,
        )
    )
    with pytest.raises(ValueError, match="goal is not in observed"):
        world.plan_path(
            start_m=Vector3(x=0.25, y=0.25, z=0.25),
            goal_m=Vector3(x=3.75, y=0.25, z=0.25),
            required_clearance_m=0.0,
        )


def test_metric_path_planning_honors_hard_monotonic_deadline() -> None:
    with pytest.raises(ValueError, match="planning deadline"):
        _map().plan_path(
            start_m=Vector3(x=1.5, y=2.5, z=0.5),
            goal_m=Vector3(x=8.5, y=2.5, z=0.5),
            required_clearance_m=0.1,
            deadline_monotonic_seconds=time.monotonic() - 1.0,
        )


def test_unknown_voxels_inside_clearance_envelope_block_the_path() -> None:
    world = MetricVoxelMap(
        resolution_m=1.0,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=5.99, y=2.99, z=0.99),
    )
    world.mark_box(
        minimum_m=Vector3(x=0.01, y=1.01, z=0.01),
        maximum_m=Vector3(x=5.9, y=1.9, z=0.9),
        occupied=False,
        confidence=0.95,
    )

    with pytest.raises(ValueError, match="start is not in observed"):
        world.plan_path(
            start_m=Vector3(x=0.5, y=1.5, z=0.5),
            goal_m=Vector3(x=5.5, y=1.5, z=0.5),
            required_clearance_m=0.3,
        )


def test_text_summary_exposes_metric_evidence_and_authority_boundary() -> None:
    summary = _map().text_map_summary()
    assert summary["source_of_truth"] == "metric-range-observations-not-rendered-image"
    assert summary["unknown_space_policy"] == "blocked-until-observed"
    assert int(summary["observed_free_voxel_count"]) > 0
    assert "actuator commands" in str(summary["model_authority"])


def test_runtime_summary_is_compact_and_tracks_threshold_indexes() -> None:
    world = _map()
    summary = world.runtime_evidence_summary()

    assert summary["source_of_truth"] == "metric-range-observations-not-rendered-image"
    assert int(summary["observed_free_voxel_count"]) > 0
    assert int(summary["occupied_voxel_count"]) == 0
    assert "frontier_centers_m" not in summary

    world.mark_box(
        minimum_m=Vector3(x=4.01, y=1.01, z=0.01),
        maximum_m=Vector3(x=4.9, y=1.9, z=0.9),
        occupied=True,
    )
    assert int(world.runtime_evidence_summary()["occupied_voxel_count"]) == 1


def test_runtime_summary_incremental_total_does_not_double_count_static_overlap() -> None:
    world = MetricVoxelMap(
        resolution_m=1.0,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=4.99, y=1.99, z=1.99),
    )
    overlapping = Vector3(x=1.5, y=0.5, z=0.5)
    live_only = Vector3(x=2.5, y=0.5, z=0.5)
    world.mark_box(minimum_m=overlapping, maximum_m=overlapping, occupied=False)
    assert world.runtime_evidence_summary()["evidence_voxel_count"] == 1

    world.seed_known_static_region([(overlapping, False)], source_sha256="c" * 64)
    assert world.runtime_evidence_summary()["evidence_voxel_count"] == 1
    world.mark_box(minimum_m=live_only, maximum_m=live_only, occupied=True)
    assert world.runtime_evidence_summary()["evidence_voxel_count"] == 2


def test_bounded_navigation_clone_freezes_only_local_live_evidence() -> None:
    world = MetricVoxelMap(
        resolution_m=1.0,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=39.99, y=3.99, z=1.99),
    )
    near = Vector3(x=2.5, y=1.5, z=0.5)
    far = Vector3(x=32.5, y=1.5, z=0.5)
    static_far = Vector3(x=36.5, y=1.5, z=0.5)
    world.mark_box(minimum_m=near, maximum_m=near, occupied=True)
    world.mark_box(minimum_m=far, maximum_m=far, occupied=True)
    world.seed_known_static_region(
        [(static_far, True)],
        source_sha256="a" * 64,
    )

    clone = world.navigation_clone(center_m=near, radius_m=4.0)

    assert clone.occupancy_probability(clone.key_for(near)) is not None
    assert clone.occupancy_probability(clone.key_for(far)) is None
    assert clone.is_occupied(clone.key_for(static_far)) is True

    world.mark_box(minimum_m=near, maximum_m=near, occupied=False)
    assert clone.is_occupied(clone.key_for(near)) is True


def test_bounded_navigation_clone_requires_center_and_radius_together() -> None:
    with pytest.raises(ValueError, match="center and radius"):
        _map().navigation_clone(center_m=Vector3(x=1.5, y=1.5, z=0.5))


# 功能：
#   用逐体素原始计算核对整桶快捷路径，覆盖负世界坐标、边界桶和球面附近半径。
# 输入：
#   radius：查询半径，包含小于桶边长和跨多个桶的情况。
# 输出：
#   None：断言键集合完全相同且克隆索引不共享可变集合。
@pytest.mark.parametrize("radius", [.001, .25, 1., 4., 12., 100.])
def test_chunk_selection_matches_scalar_distance_and_owns_indexes(radius):
    import math
    import random

    rng = random.Random(25)
    world = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=-50, y=-50, z=-25),
        maximum_bound_m=Vector3(x=50, y=50, z=25))
    center = Vector3(x=-3.5, y=-2.25, z=4.25)
    center_key = world.key_for(center)
    for _ in range(4000):
        key = tuple(value + rng.randint(-56, 56) for value in center_key)
        world._update_log_odds(key, measurement=.8, observed_at=1.)
    expected = {key for key in world._evidence if math.dist(
        (center.x, center.y, center.z), world._center_point_for_key(key)) <= radius + math.sqrt(3.) * .25 / 2}
    clone = world.navigation_clone(center_m=center, radius_m=radius)
    assert set(clone._evidence) == expected
    assert set().union(set(), *clone._evidence_keys_by_chunk.values()) == expected
    for chunk, members in clone._evidence_keys_by_chunk.items():
        assert members is not world._evidence_keys_by_chunk[chunk]
        assert members == world._evidence_keys_by_chunk[chunk] & expected
    assert clone._non_static_evidence_count == len(expected)


def test_navigation_snapshots_share_values_but_detach_writes_on_either_side():
    world = MetricVoxelMap(resolution_m=1., minimum_bound_m=Vector3(x=0, y=0, z=0),
                           maximum_bound_m=Vector3(x=4, y=4, z=4))
    key, other = (1, 1, 0), (2, 1, 0)
    world._update_log_odds(key, measurement=1., observed_at=10.)
    world._update_log_odds(other, measurement=-1., observed_at=10.)
    first = world.navigation_clone()
    second = first.navigation_clone()
    original = world._evidence[key]
    assert first._evidence[key] is second._evidence[key] is original
    assert not hasattr(original, "__dict__")
    world._update_log_odds(key, measurement=1., observed_at=11.)
    detached = world._evidence[key]
    assert detached is not original
    world._update_log_odds(key, measurement=1., observed_at=9.)
    assert world._evidence[key] is detached  # no repeated allocation per ray
    assert (detached.log_odds, detached.observations, detached.latest_monotonic_seconds) == (
        3., 3, 11.)
    first._update_log_odds(key, measurement=-3., observed_at=12.)
    assert first._evidence[key].log_odds == -2.
    assert original.log_odds == second._evidence[key].log_odds == 1.
    assert original.observations == 1
    assert world._evidence[other] is first._evidence[other] is second._evidence[other]
    assert key in world._occupied_keys and key not in first._occupied_keys
    assert key in second._occupied_keys


def test_prune_live_evidence_expires_dynamic_state_but_preserves_static_prior() -> None:
    world = MetricVoxelMap(
        resolution_m=0.5,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=30.0, y=4.0, z=2.0),
    )
    near_old = Vector3(x=2.25, y=1.25, z=0.75)
    near_fresh = Vector3(x=3.25, y=1.25, z=0.75)
    far_fresh = Vector3(x=24.25, y=1.25, z=0.75)
    static_far = Vector3(x=27.25, y=1.25, z=0.75)
    world.mark_box(
        minimum_m=near_old,
        maximum_m=near_old,
        occupied=True,
        observed_at_monotonic_seconds=1.0,
    )
    world.mark_box(
        minimum_m=near_fresh,
        maximum_m=near_fresh,
        occupied=False,
        observed_at_monotonic_seconds=9.5,
    )
    world.mark_box(
        minimum_m=far_fresh,
        maximum_m=far_fresh,
        occupied=True,
        observed_at_monotonic_seconds=9.5,
    )
    world.seed_known_static_region([(static_far, True)], source_sha256="b" * 64)

    removed = world.prune_live_evidence(
        center_m=Vector3(x=3.0, y=1.0, z=1.0),
        radius_m=5.0,
        now_monotonic_seconds=10.0,
        maximum_age_seconds=2.0,
    )

    assert removed == 2
    assert world.occupancy_probability(world.key_for(near_old)) is None
    assert world.is_observed_free(world.key_for(near_fresh)) is True
    assert world.occupancy_probability(world.key_for(far_fresh)) is None
    assert world.is_occupied(world.key_for(static_far)) is True
    assert world.pruned_live_evidence_count == 2
    assert world.runtime_evidence_summary()["pruned_live_evidence_count"] == 2
    near_old_key = world.key_for(near_old)
    assert all(near_old_key not in keys for keys in world._evidence_keys_by_chunk.values())


@pytest.mark.parametrize(
    ("radius_m", "maximum_age_seconds", "now_monotonic_seconds", "message"),
    [
        (0.0, 1.0, 1.0, "retention radius"),
        (1.0, 0.0, 1.0, "maximum age"),
        (1.0, 1.0, float("nan"), "pruning time"),
    ],
)
def test_prune_live_evidence_rejects_invalid_bounds(
    radius_m: float,
    maximum_age_seconds: float,
    now_monotonic_seconds: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _map().prune_live_evidence(
            center_m=Vector3(x=1.5, y=1.5, z=0.5),
            radius_m=radius_m,
            now_monotonic_seconds=now_monotonic_seconds,
            maximum_age_seconds=maximum_age_seconds,
        )


def test_recent_occupied_voxels_become_bounded_local_collision_boxes() -> None:
    world = MetricVoxelMap(
        resolution_m=0.25,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=4.0, y=2.0, z=2.0),
    )
    world.integrate_ray(
        RangeRayObservation(
            origin_m=Vector3(x=0.1, y=1.0, z=1.0),
            endpoint_m=Vector3(x=1.9, y=1.0, z=1.0),
            hit=True,
            confidence=0.95,
            observed_at_monotonic_seconds=10.0,
        )
    )

    primitives = world.local_occupied_box_primitives(
        center_m=Vector3(x=1.0, y=1.0, z=1.0),
        radius_m=2.0,
        now_monotonic_seconds=10.2,
    )

    assert len(primitives) == 1
    assert primitives[0]["shape"] == "box"
    assert primitives[0]["size_x"] == 0.25
    assert str(primitives[0]["name"]).startswith("perception-voxel-")


def test_stale_depth_endpoints_do_not_become_permanent_ghost_walls() -> None:
    world = MetricVoxelMap(
        resolution_m=0.25,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=4.0, y=2.0, z=2.0),
    )
    world.integrate_ray(
        RangeRayObservation(
            origin_m=Vector3(x=0.1, y=1.0, z=1.0),
            endpoint_m=Vector3(x=1.9, y=1.0, z=1.0),
            hit=True,
            confidence=0.95,
            observed_at_monotonic_seconds=10.0,
        )
    )

    assert world.local_occupied_box_primitives(
        center_m=Vector3(x=1.0, y=1.0, z=1.0),
        radius_m=2.0,
        now_monotonic_seconds=11.0,
        maximum_age_seconds=0.75,
    ) == []


def test_text_navigation_snapshot_only_offers_metric_validated_candidates() -> None:
    snapshot = _map().text_navigation_snapshot(
        current_position_m=Vector3(x=1.5, y=2.5, z=0.5),
        current_velocity_mps=Vector3(x=1.0, y=0.0, z=0.0),
        goal_position_m=Vector3(x=8.5, y=2.5, z=0.5),
        dynamic_obstacles=[
            DynamicObstacleObservation(
                obstacle_id="person-1",
                position_m=Vector3(x=3.5, y=4.5, z=0.5),
                velocity_mps=Vector3(x=-0.5, y=0.0, z=0.0),
                radius_m=0.3,
                height_m=1.8,
                confidence=0.9,
                age_seconds=0.05,
            )
        ],
        required_clearance_m=0.1,
    )

    assert snapshot["visual_input_required"] is False
    assert snapshot["unknown_space_policy"] == "blocked-until-observed"
    assert len(str(snapshot["snapshot_sha256"])) == 64
    assert len(snapshot["egocentric_sectors"]) == 8
    candidates = snapshot["authorized_candidate_paths"]
    assert candidates
    # 连续体素碰撞检查可能把直接弦线换成几何寻路结果；不锁定内部候选生成策略。
    assert candidates[0]["kind"] in {"goal-direct", "goal", "goal-dynamic-detour"}
    assert candidates[0]["deterministic_metric_path_validated"] is True
    assert candidates[0]["dynamic_path_validated"] is True
    assert candidates[0]["candidate_id"].startswith("candidate-")
    obstacle = snapshot["dynamic_obstacles"][0]
    assert obstacle["obstacle_id"] == "person-1"
    assert obstacle["time_to_closest_approach_seconds"] > 0.0
    assert snapshot["model_authority"]["may_select"] == "candidate_id or safe hold"


def test_continuous_control_snapshot_skips_legacy_candidate_search() -> None:
    snapshot = _map().text_navigation_snapshot(
        current_position_m=Vector3(x=1.5, y=2.5, z=0.5),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        goal_position_m=Vector3(x=8.5, y=2.5, z=0.5),
        required_clearance_m=0.1,
        include_candidate_paths=False,
    )

    assert snapshot["authorized_candidate_paths"] == []
    assert snapshot["candidate_generation_issues"] == []


def test_text_navigation_snapshot_does_not_offer_goal_through_unknown_space() -> None:
    world = MetricVoxelMap(
        resolution_m=0.5,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=4.0, y=1.0, z=1.0),
    )
    world.integrate_ray(
        RangeRayObservation(
            origin_m=Vector3(x=0.1, y=0.25, z=0.25),
            endpoint_m=Vector3(x=1.9, y=0.25, z=0.25),
            hit=False,
            confidence=0.95,
            observed_at_monotonic_seconds=1.0,
        )
    )

    snapshot = world.text_navigation_snapshot(
        current_position_m=Vector3(x=0.25, y=0.25, z=0.25),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        goal_position_m=Vector3(x=3.75, y=0.25, z=0.25),
        required_clearance_m=0.0,
        maximum_frontier_candidates=0,
    )

    assert snapshot["authorized_candidate_paths"] == []


def test_qualified_static_map_adds_hash_bound_route_lookahead_candidates() -> None:
    world = MetricVoxelMap(
        resolution_m=1.0,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=9.99, y=4.99, z=0.99),
    )
    classifications = [
        (world.center_for((x, y, 0)), False)
        for x in range(10)
        for y in range(5)
    ]
    semantic_sha256 = "a" * 64
    route_sha256 = "b" * 64
    world.seed_known_static_region(
        classifications,
        source_sha256=semantic_sha256,
    )
    world.bind_qualified_route(
        [Vector3(x=1.5, y=2.5, z=0.5), Vector3(x=8.5, y=2.5, z=0.5)],
        route_sha256=route_sha256,
        minimum_clearance_m=0.4,
    )

    snapshot = world.text_navigation_snapshot(
        current_position_m=Vector3(x=1.5, y=2.5, z=0.5),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        goal_position_m=Vector3(x=8.5, y=2.5, z=0.5),
        required_clearance_m=0.1,
        maximum_frontier_candidates=0,
    )

    assert snapshot["known_static_map"] == {
        "source_sha256": semantic_sha256,
        "qualified_route_sha256": route_sha256,
        "qualified_route_minimum_clearance_m": 0.4,
        "free_voxel_count": 50,
        "occupied_voxel_count": 0,
    }
    assert any(
        candidate["kind"] == "qualified-route-lookahead"
        for candidate in snapshot["authorized_candidate_paths"]
    )
    assert snapshot["vehicle_envelope_m"]["required_local_clearance"] == 0.1


def test_final_approach_offers_only_exact_metric_goal_when_direct_path_is_safe() -> None:
    world = MetricVoxelMap(
        resolution_m=1.0,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=9.99, y=4.99, z=0.99),
    )
    world.seed_known_static_region(
        [
            (world.center_for((x, y, 0)), False)
            for x in range(10)
            for y in range(5)
        ],
        source_sha256="a" * 64,
    )
    goal = Vector3(x=8.2, y=2.4, z=0.45)
    world.bind_qualified_route(
        [Vector3(x=1.5, y=2.5, z=0.5), goal],
        route_sha256="b" * 64,
        minimum_clearance_m=0.4,
    )

    snapshot = world.text_navigation_snapshot(
        current_position_m=Vector3(x=6.7, y=2.5, z=0.5),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        goal_position_m=goal,
        required_clearance_m=0.1,
        local_radius_m=8.0,
        maximum_frontier_candidates=0,
    )

    candidates = snapshot["authorized_candidate_paths"]
    assert len(candidates) == 1
    assert candidates[0]["kind"] == "goal-direct"
    assert candidates[0]["endpoint_m"] == goal.model_dump(mode="json")


def test_far_qualified_goal_uses_bounded_local_lookahead_instead_of_full_astar() -> None:
    world = MetricVoxelMap(
        resolution_m=1.0,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=49.99, y=4.99, z=0.99),
    )
    world.seed_known_static_region(
        [
            (world.center_for((x, y, 0)), False)
            for x in range(50)
            for y in range(5)
        ],
        source_sha256="a" * 64,
    )
    start = Vector3(x=1.5, y=2.5, z=0.5)
    goal = Vector3(x=48.5, y=2.5, z=0.5)
    world.bind_qualified_route(
        [start, goal],
        route_sha256="b" * 64,
        minimum_clearance_m=0.4,
    )

    snapshot = world.text_navigation_snapshot(
        current_position_m=start,
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        goal_position_m=goal,
        required_clearance_m=0.1,
        local_radius_m=8.0,
        maximum_frontier_candidates=0,
    )

    candidates = snapshot["authorized_candidate_paths"]
    assert candidates
    assert all(candidate["kind"] != "goal" for candidate in candidates)
    local = next(
        candidate
        for candidate in candidates
        if candidate["kind"] == "qualified-route-local-lookahead"
    )
    assert float(local["path_length_m"]) <= 4.5
    assert "GOAL_OUTSIDE_LOCAL_PLANNING_HORIZON" in snapshot[
        "candidate_generation_issues"
    ]


def test_near_goal_prioritizes_qualified_route_candidate_before_goal_astar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = MetricVoxelMap(
        resolution_m=1.0,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=11.99, y=3.99, z=0.99),
    )
    world.seed_known_static_region(
        [
            (world.center_for((x, y, 0)), False)
            for x in range(12)
            for y in range(4)
        ],
        source_sha256="a" * 64,
    )
    start = Vector3(x=1.5, y=1.5, z=0.5)
    goal = Vector3(x=10.5, y=1.5, z=0.5)
    world.bind_qualified_route(
        [start, goal],
        route_sha256="b" * 64,
        minimum_clearance_m=0.4,
    )

    def unexpected_goal_search(**_kwargs: object) -> list[Vector3]:
        raise AssertionError("a safe route lookahead must preempt full goal A*")

    monkeypatch.setattr(world, "plan_path", unexpected_goal_search)
    snapshot = world.text_navigation_snapshot(
        current_position_m=Vector3(x=5.5, y=1.5, z=0.5),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        goal_position_m=goal,
        required_clearance_m=0.1,
        local_radius_m=8.0,
        maximum_frontier_candidates=0,
    )

    assert snapshot["authorized_candidate_paths"]
    assert snapshot["authorized_candidate_paths"][0]["kind"] == (
        "qualified-route-local-lookahead"
    )


def test_far_qualified_goal_uses_local_astar_to_rejoin_from_safe_detour() -> None:
    world = MetricVoxelMap(
        resolution_m=1.0,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=19.99, y=5.99, z=0.99),
    )
    classifications = [
        (world.center_for((x, y, 0)), (x, y) == (10, 2))
        for x in range(20)
        for y in range(6)
    ]
    world.seed_known_static_region(
        classifications,
        source_sha256="a" * 64,
    )
    start = Vector3(x=1.5, y=2.5, z=0.5)
    goal = Vector3(x=18.5, y=2.5, z=0.5)
    world.bind_qualified_route(
        [start, goal],
        route_sha256="b" * 64,
        minimum_clearance_m=0.4,
    )

    snapshot = world.text_navigation_snapshot(
        current_position_m=Vector3(x=8.5, y=1.5, z=0.5),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        goal_position_m=goal,
        required_clearance_m=0.1,
        local_radius_m=8.0,
        maximum_frontier_candidates=0,
    )

    local = next(
        candidate
        for candidate in snapshot["authorized_candidate_paths"]
        if candidate["kind"] == "qualified-route-local-lookahead"
    )
    points = [Vector3.model_validate(point) for point in local["path_points_m"]]
    assert len(points) > 2
    assert all(not (10.0 <= point.x < 11.0 and 2.0 <= point.y < 3.0) for point in points)
    assert local["deterministic_metric_path_validated"] is True
    assert "QUALIFIED_ROUTE_LOCAL_REJOIN_UNAVAILABLE" not in snapshot[
        "candidate_generation_issues"
    ]


def test_repeated_round_trip_goal_selects_nearest_active_segment() -> None:
    world = MetricVoxelMap(
        resolution_m=1.0,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=49.99, y=9.99, z=0.99),
    )
    world.seed_known_static_region(
        [
            (world.center_for((x, y, 0)), False)
            for x in range(50)
            for y in range(10)
        ],
        source_sha256="a" * 64,
    )
    start = Vector3(x=1.5, y=2.5, z=0.5)
    repeated_goal = Vector3(x=40.5, y=2.5, z=0.5)
    return_turn = Vector3(x=42.5, y=8.5, z=0.5)
    world.bind_qualified_route(
        [start, repeated_goal, return_turn, repeated_goal, start],
        route_sha256="b" * 64,
        minimum_clearance_m=0.4,
    )

    snapshot = world.text_navigation_snapshot(
        current_position_m=Vector3(x=20.5, y=2.5, z=0.5),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        goal_position_m=repeated_goal,
        required_clearance_m=0.1,
        local_radius_m=8.0,
        maximum_frontier_candidates=0,
    )

    local = next(
        candidate
        for candidate in snapshot["authorized_candidate_paths"]
        if candidate["kind"] == "qualified-route-local-lookahead"
    )
    assert local["endpoint_m"]["x"] > 20.5
    assert float(local["path_length_m"]) <= 4.5


def test_text_navigation_generates_metric_detour_around_moving_person() -> None:
    snapshot = _map().text_navigation_snapshot(
        current_position_m=Vector3(x=1.5, y=2.5, z=0.5),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        goal_position_m=Vector3(x=8.5, y=2.5, z=0.5),
        dynamic_obstacles=[
            DynamicObstacleObservation(
                obstacle_id="person-crossing",
                position_m=Vector3(x=4.5, y=2.5, z=0.5),
                velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
                radius_m=0.35,
                height_m=1.8,
                confidence=0.95,
                age_seconds=0.02,
            )
        ],
        required_clearance_m=0.1,
        maximum_frontier_candidates=0,
    )

    candidates = snapshot["authorized_candidate_paths"]
    assert candidates
    assert candidates[0]["kind"] == "goal-dynamic-detour"
    assert candidates[0]["dynamic_path_validated"] is True
    assert "DYNAMIC_PATH_BLOCKED:person-crossing" in snapshot["candidate_generation_issues"]
