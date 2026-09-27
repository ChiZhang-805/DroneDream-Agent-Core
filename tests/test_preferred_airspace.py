"""Independent geometric examples, not production flight qualification."""

import math

import pytest

from dronedream_agent_core.preferred_airspace import AirspacePreferences, PreferredAirspace


# 功能：
#   构造有真实尺寸的单元测试箱体，避免测试使用生产地图的硬编码预期答案。
# 输入：
#   x、y、z、sx、sy、sz、semantic：中心、尺寸和类型。
# 输出：
#   primitive：测试碰撞图元。
def box(x=0., y=0., z=-.1, sx=20., sy=20., sz=.2, semantic="terrain"):
    primitive = dict(center_x=x, center_y=y, center_z=z, size_x=sx, size_y=sy,
                     size_z=sz, semantic=semantic)
    return primitive


# 功能：
#   为测试创建绑定空间场，所有飞行权限仍未授予。
# 输入：
#   extra：地形之外的障碍。
# 输出：
#   field：测试用全图软空间。
def field(*extra):
    return PreferredAirspace({"coordinate_frame": "ENU", "collision_primitives": [box(), *extra]},
                             {"map": "test-only"}, .2, .4)


def test_indoor_usable_interval_matches_five_band_example():
    space = field(box(z=3.1, sz=.2, semantic="ceiling"))
    band = space.band_at((0., 0., 1.5))
    assert band.minimum_center_z_m == pytest.approx(.5)
    assert band.maximum_center_z_m == pytest.approx(2.5)
    assert band.preferred_lower_z_m == pytest.approx(1.3)
    assert band.preferred_upper_z_m == pytest.approx(2.1)


def test_multi_floor_and_broad_outdoor_regions_not_route_tube():
    space = field(box(z=3.1, sx=8., sy=8., sz=.2, semantic="floor"),
                  box(z=6.3, sx=8., sy=8., sz=.2, semantic="ceiling"))
    assert len(space.column((0, 0))) >= 2
    assert space.band_at((0., 0., 1.5)).ceiling_z_m == pytest.approx(3.)
    assert space.band_at((0., 0., 4.5)).floor_z_m == pytest.approx(3.2)
    assert space.band_at((6., 6., 6.)).preferred_lower_z_m == 5.
    assert space.band_at((-6., -6., 6.)).preferred_upper_z_m == 8.


def test_tree_clearance_hole_does_not_raise_agl_from_crown():
    space = field(box(z=5., sx=2., sy=2., sz=10., semantic="tree"))
    assert space.band_at((0., 0., 6.)) is None
    assert space.band_at((0., 0., 16.)) is None
    assert space.band_at((4., 0., 6.)) is not None


def test_unknown_and_whole_body_map_boundary_are_not_free():
    space = field()
    assert space.band_at((10., 0., 6.)) is None
    assert space.band_at((9.5, 0., 6.)) is None
    empty = PreferredAirspace({"collision_primitives": [box(semantic="floor")]}, {}, .2, .4)
    assert not empty.snapshot()["volumes"]


def test_identity_changes_with_geometry_vehicle_payload_and_preferences():
    original = field()
    for replacement in (
        PreferredAirspace({"coordinate_frame": "ENU", "collision_primitives": [box()]},
                          {"payload_kg": .1}, .2, .4),
        PreferredAirspace({"coordinate_frame": "ENU", "collision_primitives": [box()]},
                          {"map": "test-only"}, .3, .4),
        PreferredAirspace({"coordinate_frame": "ENU", "collision_primitives": [box()]},
                          {"map": "test-only"}, .2, .4, AirspacePreferences(margin_m=.4)),
    ):
        assert replacement.sha256 != original.sha256


def test_snapshot_roundtrip_consistency_and_stable_hash():
    space = field(box(z=3.1, sx=6., sy=6., sz=.2, semantic="ceiling"))
    first = space.snapshot()
    assert first == space.snapshot()
    for cx, cy, cz, sx, sy, sz, floor, ceiling in first["volumes"]:
        assert sx > 0 and sy > 0 and sz > 0
        band = space.band_at((cx, cy, cz))
        assert band is not None
        assert band.preferred_lower_z_m <= cz <= band.preferred_upper_z_m
        assert floor == band.floor_z_m and ceiling == band.ceiling_z_m


def test_runtime_collision_envelope_also_removes_preferred_space():
    space = PreferredAirspace({"coordinate_frame": "ENU", "collision_primitives": [box()],
                              "runtime_collision_primitives": [box(z=5, sx=3, sy=3, sz=10)]},
                             {}, .2, .4)
    assert space.band_at((0, 0, 6)) is None


@pytest.mark.parametrize("value", [math.nan, math.inf, True, -1])
def test_invalid_preferences_rejected(value):
    with pytest.raises(ValueError):
        AirspacePreferences(margin_m=value)


@pytest.mark.parametrize("point", [(math.nan, 0, 0), (True, 0, 0), (1, 2)])
def test_invalid_query_rejected(point):
    with pytest.raises(ValueError):
        field().context(point)


def test_stair_gradient_changes_preference_not_obstacles():
    space = field(*[box(x=-3 + i * .5, z=i * .15 / 2, sx=.5, sy=3,
                       sz=max(.01, i * .15), semantic="stair") for i in range(12)],
                  box(z=4.1, sz=.2, semantic="ceiling"))
    assert space.context((0., 0., 2.))['floor_gradient_xy'][0] > 0
    assert space.band_at((0., 0., .1)) is None
