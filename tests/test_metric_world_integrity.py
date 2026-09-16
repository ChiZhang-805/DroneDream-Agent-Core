import pytest

from dronedream_agent_core.contracts import RangeRayObservation, Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap, _segment_hits_box


# 功能：
#   创建明确用于单元测试的米制栅格，不冒充真实传感器或飞行证据。
# 输入：
#   无。
# 输出：
#   world：尚未接收任何观测的局部地图。
def _world():
    world = MetricVoxelMap(
        resolution_m=1.0,
        minimum_bound_m=Vector3(x=0, y=0, z=0),
        maximum_bound_m=Vector3(x=5.99, y=5.99, z=2.99),
    )
    return world


# 功能：
#   验证不同来源摘要的地图不能把旧空地与新障碍拼成同一当前地图。
# 输入：
#   无。
# 输出：
#   无：拒绝替换且原快照不变的断言结果。
def test_static_source_cannot_be_relabelled_over_old_cells():
    world = _world()
    point = Vector3(x=1.5, y=1.5, z=1.5)
    world.seed_known_static_region([(point, False)], source_sha256="a" * 64)
    before = world.snapshot()
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        world.seed_known_static_region(
            [(Vector3(x=4.5, y=4.5, z=1.5), True)], source_sha256="b" * 64
        )
    assert world.snapshot() == before
    assert world.runtime_evidence_summary()["known_static_source_sha256"] == "a" * 64


# 功能：
#   验证实测障碍过期只变成未知，不能恢复被否定的历史空地权限。
# 输入：
#   无。
# 输出：
#   无：原地图和局部副本均不放行旧空地的断言结果。
def test_expired_hit_never_revives_invalidated_static_free_space():
    world = _world()
    point = Vector3(x=1.5, y=1.5, z=1.5)
    key = world.key_for(point)
    world.seed_known_static_region([(point, False)], source_sha256="a" * 64)
    old_clone = world.navigation_clone()
    world.integrate_scan(
        [
            RangeRayObservation(
                origin_m=point,
                endpoint_m=point,
                hit=True,
                confidence=0.95,
                observed_at_monotonic_seconds=1.0,
            )
        ]
    )
    distant_clone = world.navigation_clone(center_m=Vector3(x=5.5, y=5.5, z=1.5), radius_m=0.1)
    world.prune_live_evidence(
        center_m=point, radius_m=2.0, now_monotonic_seconds=5.0, maximum_age_seconds=1.0
    )
    assert old_clone.is_observed_free(key)
    assert world.occupancy_probability(key) is None
    for target in (world, distant_clone):
        assert not target.is_observed_free(key)
        assert target.is_occupied(key)
        assert key not in target.snapshot().free_voxels


# 功能：
#   验证路线与地图边界不借用调用者可变向量。
# 输入：
#   无。
# 输出：
#   无：调用后修改原向量不会改写已绑定几何的断言结果。
def test_map_bounds_and_route_are_owned_snapshots():
    minimum, maximum = Vector3(x=0, y=0, z=0), Vector3(x=5, y=5, z=5)
    world = MetricVoxelMap(resolution_m=1.0, minimum_bound_m=minimum, maximum_bound_m=maximum)
    points = [Vector3(x=0.5, y=0.5, z=0.5), Vector3(x=4.5, y=0.5, z=0.5)]
    world.bind_qualified_route(points, route_sha256="c" * 64)
    minimum.x = -100
    maximum.x = 100
    points[1].x = 500
    assert world.snapshot().minimum_bound_m.x == 0
    assert world.snapshot().maximum_bound_m.x == 5
    assert world._qualified_route_points[1].x == 4.5


# 功能：
#   验证构造后损坏的布尔观测不能通过 Python 真值转换写入地图。
# 输入：
#   无。
# 输出：
#   无：非法扫描被整体拒绝且无局部更新的断言结果。
def test_mutated_ray_flag_is_rejected_before_map_changes():
    world = _world()
    point = Vector3(x=1.5, y=1.5, z=1.5)
    ray = RangeRayObservation(
        origin_m=point,
        endpoint_m=point,
        hit=False,
        confidence=0.95,
        observed_at_monotonic_seconds=1.0,
    )
    ray = ray.model_copy(update={"hit": "false"})
    with pytest.raises(ValueError):
        world.integrate_scan([ray])
    assert world.observation_count == 0


# 功能：
#   验证两端可通行不代表中间斜穿障碍拐角可通行。
# 输入：
#   无。
# 输出：
#   无：零净空要求仍不穿过实体体积的断言结果。
def test_short_corner_crossing_is_not_missed_between_samples():
    world = _world()
    world.mark_box(
        minimum_m=Vector3(x=0.01, y=0.01, z=0.01),
        maximum_m=Vector3(x=5.9, y=5.9, z=2.9),
        occupied=False,
    )
    obstacle = Vector3(x=2.5, y=1.5, z=1.5)
    world.mark_box(minimum_m=obstacle, maximum_m=obstacle, occupied=True)
    path = [Vector3(x=1.98, y=1.9, z=1.5), Vector3(x=2.1, y=2.02, z=1.5)]
    assert not world.path_clearance_valid(path, required_clearance_m=0.0)


# 功能：
#   验证超大局部查询只查已有证据，不按空空间体积建立循环。
# 输入：
#   无。
# 输出：
#   无：超大半径返回有限结果的断言结果。
def test_large_finite_radius_is_bounded_by_existing_evidence():
    world = _world()
    point = Vector3(x=1.5, y=1.5, z=1.5)
    world.mark_box(minimum_m=point, maximum_m=point, occupied=False)
    assert world._local_navigation_keys(center_m=point, radius_m=1e8) == {world.key_for(point)}


# 功能：
#   验证非法净空与查询参数不能进入寻路循环。
# 输入：
#   invalid：NaN、无穷大或伪装为数值的布尔值。
# 输出：
#   无：参数在边界被拒绝的断言结果。
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True])
def test_invalid_clearance_rejected(invalid):
    with pytest.raises(ValueError):
        _world().path_clearance_valid([], required_clearance_m=invalid)


# 功能：
#   验证连续相交计算区分穿角和绕角，支持零长度、平行及反向线段。
# 输入：
#   first、second：测试端点；expected：独立指定的几何事实。
# 输出：
#   无：正向及反向相交结果均符合预期的断言结果。
@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ((1.98, 1.9, 1.5), (2.1, 2.02, 1.5), True),
        ((1.9, 1.98, 1.5), (2.02, 2.1, 1.5), False),
        ((2.0, 0.5, 1.5), (2.0, 2.5, 1.5), True),
        ((1.99, 0.5, 1.5), (1.99, 2.5, 1.5), False),
        ((2.5, 1.5, 1.5), (2.5, 1.5, 1.5), True),
        ((1.5, 1.5, 1.5), (1.5, 1.5, 1.5), False),
    ],
)
def test_segment_box_geometry_including_nonintersecting_corner(first, second, expected):
    assert _segment_hits_box(first, second, (2.5, 1.5, 1.5), 0.5) is expected
    assert _segment_hits_box(second, first, (2.5, 1.5, 1.5), 0.5) is expected


# 功能：
#   验证地图副本继续写入时不污染原图或更早副本的静态撤销状态。
# 输入：
#   无。
# 输出：
#   无：写时复制和静态权限隔离的断言结果。
def test_cloned_static_invalidation_detaches_in_both_directions():
    world = _world()
    a, b = Vector3(x=1.5, y=1.5, z=1.5), Vector3(x=2.5, y=1.5, z=1.5)
    world.seed_known_static_region([(a, False), (b, False)], source_sha256="a" * 64)
    clone = world.navigation_clone()
    world.mark_box(minimum_m=a, maximum_m=a, occupied=True)
    clone.mark_box(minimum_m=b, maximum_m=b, occupied=True)
    assert world.is_observed_free(world.key_for(b))
    assert clone.is_observed_free(clone.key_for(a))
    assert world._invalidated_static_free_keys == {world.key_for(a)}
    assert clone._invalidated_static_free_keys == {world.key_for(b)}


# 功能：
#   验证连续控制快照明确告诉本地模型可提交有界机体系控制意图，不混用候选模式说明。
# 输入：
#   无。
# 输出：
#   无：模式及候选集合一致的断言结果。
def test_continuous_mode_authority_matches_control_role():
    world = _world()
    point = Vector3(x=1.5, y=1.5, z=1.5)
    snapshot = world.text_navigation_snapshot(
        current_position_m=point,
        current_velocity_mps=Vector3(x=0, y=0, z=0),
        goal_position_m=point,
        required_clearance_m=0.1,
        include_candidate_paths=False,
    )
    assert snapshot["authorized_candidate_paths"] == []
    assert "body-frame control intent" in snapshot["model_authority"]["may_select"]


# 功能：
#   验证大整数、NaN 和布尔配置均在计算前被拒绝，不发生整数到浮点溢出。
# 输入：
#   invalid：待拒绝的参数。
# 输出：
#   无：类型和预算边界生效的断言结果。
@pytest.mark.parametrize("invalid", [10**1000, float("nan"), True])
def test_grid_query_numeric_boundaries(invalid):
    with pytest.raises(ValueError):
        _world().navigation_clone(center_m=Vector3(x=1, y=1, z=1), radius_m=invalid)


# 功能：
#   验证一批静态分类末尾损坏不会留下前面半批分类。
# 输入：
#   无。
# 输出：
#   无：导入失败后地图保持空白的断言结果。
def test_static_classification_failure_is_atomic():
    world = _world()
    a, b = Vector3(x=1.5, y=1.5, z=1.5), Vector3(x=2.5, y=1.5, z=1.5)
    with pytest.raises(ValueError):
        world.seed_known_static_region([(a, False), (b, "false")], source_sha256="a" * 64)
    assert world._known_static_source_sha256 is None
    assert world.snapshot().free_voxels == []
