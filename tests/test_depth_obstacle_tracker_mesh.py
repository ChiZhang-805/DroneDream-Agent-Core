"""Static mesh identity must never remove measured occupancy or real motion."""

import math

import numpy as np
import pytest

from dronedream_agent_core.contracts import OnboardPerceptionFrame, RangeRayObservation, Vector3
from dronedream_agent_core.depth_obstacle_tracker import DepthMotionTracker
from dronedream_agent_core.local_map_alignment import MapAlignmentLimits, MapSurfaceIndex


# 功能：生成闭合且有真实朝向的合成树冠壳；输入中心与尺寸；输出真实三角面索引原语。
def _mesh(*, center=(3., 0., 1.), half=(1., 3., 4.), yaw=0.35):
    vertices = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return dict(center_x=center[0], center_y=center[1], center_z=center[2], yaw_rad=yaw,
                triangles=[[[vertices[i][a] * half[a] for a in range(3)] for i in face]
                           for face in faces])


# 功能：构造明确来自原生坐标规则的合成量测；输入端点/起点；输出不可偷换来源的完整帧。
def _frame(sequence, endpoints, origin=(0., 0., 1.)):
    return OnboardPerceptionFrame(
        sensor_id="oakd-lite-depth", sequence=sequence, observed_at_unix_ms=sequence * 100,
        localization_position_m=Vector3(x=origin[0], y=origin[1], z=origin[2]),
        localization_velocity_mps=Vector3(x=0., y=0., z=0.), localization_covariance_m2=.0001,
        range_rays=[RangeRayObservation(
            origin_m=Vector3(x=origin[0], y=origin[1], z=origin[2]),
            endpoint_m=Vector3(x=p[0], y=p[1], z=p[2]), hit=True, confidence=.95,
            observed_at_monotonic_seconds=sequence * .1) for p in endpoints])


# 功能：静态斜面可见斑块上下变化时不产生假运动；所有原始占据点保持不变。
def test_static_mesh_visibility_shift_is_filtered_without_changing_raw_rays():
    tracker = DepthMotionTracker(known_static_surfaces=MapSurfaceIndex([_mesh()]))
    unbound = DepthMotionTracker()
    observed_unbound_motion = False
    for sequence in range(1, 6):
        points = [(3. - math.cos(.35) - math.sin(.35) * offset,
                   -math.sin(.35) + math.cos(.35) * offset,
                   1. + sequence * .15 + offset) for offset in (-.12, 0., .12)]
        frame = _frame(sequence, points)
        raw = frame.model_dump(mode="json")
        observed_unbound_motion |= bool(unbound.update(
            frame, observed_at_monotonic_seconds=sequence * .1))
        assert tracker.update(frame, observed_at_monotonic_seconds=sequence * .1) == []
        assert frame.model_dump(mode="json") == raw
    assert observed_unbound_motion  # This exercises the former missing-mesh behavior.


# 功能：树前的独立运动簇、壳体孔隙以及实际网格高度之外均不能被静态身份吞掉。
@pytest.mark.parametrize("meshes,z", [
    ([_mesh()], 1.),
    ([_mesh(center=(3., -2., 1.), half=(1., .5, 4.), yaw=0.),
      _mesh(center=(3., 2., 1.), half=(1., .5, 4.), yaw=0.)], 1.),
    ([_mesh(half=(1., 3., .5), yaw=0.)], 3.),
])
def test_separate_moving_target_remains_observable_near_static_mesh(meshes, z):
    tracker = DepthMotionTracker(known_static_surfaces=MapSurfaceIndex(meshes))
    for sequence, x in enumerate((1., 1.2), 1):
        frame = _frame(sequence, [(x, d, z + d) for d in (-.12, 0., .12)], origin=(0., 0., z))
        result = tracker.update(frame, observed_at_monotonic_seconds=sequence * .1)
    assert len(result) == 1
    assert result[0].velocity_mps.x > .1


# 功能：近切向射线虽到无限平面的法向距离很小，实际深度不同仍不得标为静态。
def test_surface_normal_proximity_alone_does_not_hide_separate_target():
    index = MapSurfaceIndex([_mesh(center=(3., 0., 1.), half=(1., 20., 4.), yaw=0.)])
    tracker = DepthMotionTracker(known_static_surfaces=index)
    point, origin = (1.95, 10., 1.), (0., -10., 1.)
    match = index.match([point], MapAlignmentLimits(), sensor_origins_world_m=origin)
    assert match.valid[0] and match.distances[0] < .15
    assert abs(np.linalg.norm(np.array(point) - origin) - match.surface_ranges_m[0]) > .15
    assert tracker._known_surface_points({point: origin}, uncertainty_m=.03) == set()


# 功能：同一定位平移误差在斜视平面上产生更大深度差，不能被等长前缀误截。
def test_oblique_pose_uncertainty_uses_complete_first_hit_and_projected_error():
    index = MapSurfaceIndex([_mesh(center=(3., 0., 1.), half=(1., 20., 4.), yaw=0.)])
    tracker = DepthMotionTracker(known_static_surfaces=index)
    # 真值起点(0,-10,1)与端点(2,10,1)同时被原生定位向西平移0.1m。
    point, origin = (1.9, 10., 1.), (-.1, -10., 1.)
    limits = MapAlignmentLimits(maximum_association_distance_m=.22)
    clipped = index.match([point], limits, sensor_origins_world_m=origin,
                          observed_range_prefix=True)
    full = index.match([point], limits, sensor_origins_world_m=origin)
    assert not clipped.valid[0] and full.valid[0]
    assert full.distances[0] == pytest.approx(.1)
    assert abs(np.linalg.norm(np.array(point) - origin) - full.surface_ranges_m[0]) > .22
    assert tracker._known_surface_points({point: origin}, uncertainty_m=.1) == {point}
    # 同样的定位误差不能解释真正位于前方的另一目标。
    foreground = (1.5, 10., 1.)
    assert tracker._known_surface_points({foreground: origin}, uncertainty_m=.1) == set()


# 功能：误差正确传播后不把静态可见区域变化变为动态目标，原始占据量测仍完整保留。
def test_oblique_static_patch_shift_preserves_original_occupancy():
    tracker = DepthMotionTracker(known_static_surfaces=MapSurfaceIndex([
        _mesh(center=(3., 0., 1.), half=(1., 20., 4.), yaw=0.)]))
    for sequence in range(1, 5):
        points = [(1.9, 10. + d, 1. + sequence * .3 + d) for d in (-.12, 0., .12)]
        frame = _frame(sequence, points, origin=(-.1, -10., 1.))
        frame.localization_covariance_m2 = (.1 / 3.) ** 2
        raw = frame.model_dump(mode="json")
        assert tracker.update(frame, observed_at_monotonic_seconds=sequence * .1) == []
        assert frame.model_dump(mode="json") == raw


# 功能：超过一批的端点全部接受证明，不把第 513 个之后的真实静态点或独立目标截掉。
def test_all_batches_are_matched_and_unexplained_tail_is_retained(monkeypatch):
    index = MapSurfaceIndex([_mesh(yaw=0.)])
    tracker = DepthMotionTracker(known_static_surfaces=index)
    points = {(2., -.8 + i * .001, 1.3): (0., 0., 1.) for i in range(1025)}
    independent = (1., .2, 1.3)
    points[independent] = (0., 0., 1.)
    batch_sizes = []
    original = index.match

    def match(values, limits, **kwargs):
        batch_sizes.append(len(values))
        return original(values, limits, **kwargs)

    monkeypatch.setattr(index, "match", match)
    known = tracker._known_surface_points(points, uncertainty_m=.03)
    assert batch_sizes == [512, 512, 2]
    assert known == set(points) - {independent}


# 功能：过大不确定度或无效表面索引不能成为静态目标忽略许可。
def test_uncertainty_outside_surface_envelope_does_not_exclude_points():
    tracker = DepthMotionTracker(known_static_surfaces=MapSurfaceIndex([_mesh(yaw=0.)]))
    assert tracker._known_surface_points({(2., 0., 1.): (0., 0., 1.)}, uncertainty_m=3.) == set()
    with pytest.raises(ValueError, match="DEPTH_TRACK_STATIC_SURFACES_INVALID"):
        DepthMotionTracker(known_static_surfaces=object())
