import math
import random

import pytest
from test_collision_batch import mixed_shapes

from dronedream_agent_core.collision import vehicle_clearance
from dronedream_agent_core.depth_obstacle_tracker import _StaticPrimitiveIndex


# 功能：
#   对比缓存索引与完整标量几何，覆盖混合形状、负坐标和跨格定位余量。
# 输入：
#   seed：生成合成地图和查询点的固定随机种子。
# 输出：
#   None：比较结果不一致时测试失败。
@pytest.mark.parametrize("seed", range(8))
def test_cached_index_matches_complete_scalar_map(seed):
    rng = random.Random(seed)
    shapes = mixed_shapes(seed)
    index = _StaticPrimitiveIndex(shapes, exclusion_m=.1)
    for _ in range(120):
        point = tuple(rng.uniform(-5, 5) for _ in range(3))
        uncertainty = rng.choice((0., .001, .1, .5, 2., 100.))
        expected = any(vehicle_clearance(point, shape, radius_m=.001, half_height_m=.001)
                       <= .1 + uncertainty for shape in shapes)
        assert index.contains_known_static(point, uncertainty_m=uncertainty) is expected
        assert index.contains_known_static(point, uncertainty_m=uncertainty) is expected


# 功能：
#   验证同一格内的两个点只共享候选编号，不能共享第一个点的包含结论。
# 输入：
#   无。
# 输出：
#   None：距离计算被错误缓存时测试失败。
def test_same_bin_never_reuses_point_containment_result():
    sphere = dict(center_x=.2, center_y=.2, center_z=.2, radius_m=.1)
    index = _StaticPrimitiveIndex([sphere], exclusion_m=0.)
    assert index.contains_known_static((.2, .2, .2))
    key = ((0, 0, 0), (0, 0, 0))
    cached = index._candidate_cache[key]
    assert not index.contains_known_static((1.8, 1.8, 1.8))
    assert index._candidate_cache[key] is cached


# 功能：
#   验证查询缓存有条目上限，淘汰后仍可重新检出原位置的障碍。
# 输入：
#   monkeypatch：缩小缓存容量以测试最近使用策略。
# 输出：
#   None：缓存越界或淘汰改变几何结果时测试失败。
def test_candidate_cache_is_bounded_and_preserves_evicted_geometry(monkeypatch):
    monkeypatch.setattr(_StaticPrimitiveIndex, "_MAX_CACHED_QUERIES", 2)
    index = _StaticPrimitiveIndex(
        [dict(center_x=.2, center_y=.2, center_z=.2, radius_m=.1)], exclusion_m=0.)
    assert index.contains_known_static((.2, .2, .2))
    index.contains_known_static((5., 5., 5.))
    assert index.contains_known_static((.2, .2, .2))  # 刷新最近使用顺序。
    index.contains_known_static((9., 9., 9.))
    assert len(index._candidate_cache) == 2
    assert ((2, 2, 2), (2, 2, 2)) not in index._candidate_cache
    for x in range(20, 100):
        assert not index.contains_known_static((float(x), 0., 0.))
        assert len(index._candidate_cache) <= 2
    assert index.contains_known_static((.2, .2, .2))


# 功能：
#   验证候选过多或格范围过大时不写缓存，但仍检查未索引的大基元。
# 输入：
#   monkeypatch：缩小单项缓存及格索引容量的测试工具。
# 输出：
#   None：超预算时遗漏障碍或缓存无界增长则测试失败。
def test_oversized_queries_remain_complete_without_cache(monkeypatch):
    monkeypatch.setattr(_StaticPrimitiveIndex, "_MAX_CACHED_CANDIDATES", 1)
    monkeypatch.setattr(_StaticPrimitiveIndex, "_MAX_BIN_ENTRIES", 0)
    shapes = [dict(center_x=float(x), center_y=0., center_z=0., radius_m=.1)
              for x in (0, 1, 2)]
    index = _StaticPrimitiveIndex(shapes, exclusion_m=0.)
    assert len(index._large_primitives) == 3
    # 定位余量使查询跨越相邻两格，正确空间筛选后仍有三个候选，继续检验缓存容量。
    assert index.contains_known_static((2., 0., 0.), uncertainty_m=.1)
    assert index.contains_known_static((80., 0., 0.), uncertainty_m=100.)
    assert not index.contains_known_static((200., 0., 0.), uncertainty_m=100.)
    assert not index._candidate_cache


# 功能：
#   验证外部输入与对外只读视图不能改变索引固定的地图和膨胀距离。
# 输入：
#   无。
# 输出：
#   None：已验证几何能被后续修改时测试失败。
def test_index_freezes_geometry_and_exclusion_for_all_queries():
    sphere = dict(center_x=0., center_y=0., center_z=0., radius_m=.1)
    index = _StaticPrimitiveIndex([sphere], exclusion_m=.1)
    sphere["radius_m"] = 100.
    assert not index.contains_known_static((1., 0., 0.))
    with pytest.raises(TypeError):
        index.primitives[0]["radius_m"] = 100.
    with pytest.raises(AttributeError):
        index.primitives = ()
    with pytest.raises(AttributeError):
        index.exclusion_m = 100.


# 功能：
#   验证初始化后未重复验证几何，但每个新查询仍拒绝非法点与溢出净空。
# 输入：
#   monkeypatch：隔离校验调用及模拟净空溢出的测试工具。
# 输出：
#   None：热路径重复校验或非法数值绕过检查时测试失败。
def test_frozen_geometry_avoids_revalidation_but_keeps_numeric_guards(monkeypatch):
    from dronedream_agent_core import depth_obstacle_tracker as module

    index = _StaticPrimitiveIndex(
        [dict(center_x=0., center_y=0., center_z=0., radius_m=.1)], exclusion_m=.1)

    # 功能：
    #   让已冻结后的再次几何验证显式失败，以检测热点退回重复验证的情况。
    # 输入：
    #   primitive：不应再次传入验证器的几何。
    # 输出：
    #   无：调用本辅助函数会使测试失败。
    def unexpected_validation(primitive):
        pytest.fail("frozen geometry was revalidated")

    monkeypatch.setattr(module, "_validated_primitive", unexpected_validation)
    assert index.contains_known_static((0., 0., 0.))
    with pytest.raises(ValueError, match="QUERY_INVALID"):
        index.contains_known_static((math.nan, 0., 0.))
    monkeypatch.setattr(module, "_clearance", lambda *args, **kwargs: math.inf)
    with pytest.raises(ValueError, match="clearance overflow"):
        index.contains_known_static((0., 0., 0.))


# 功能：强制所有形状走大基元旁表，核对完整有符号净空、跨格余量与负坐标。
# 输入：monkeypatch：关闭格写入；seed：固定混合形状种子。输出：所有查询与标量一致。
@pytest.mark.parametrize("seed", range(4))
def test_large_primitive_broad_phase_preserves_complete_scalar_query(monkeypatch, seed):
    monkeypatch.setattr(_StaticPrimitiveIndex, "_MAX_BIN_ENTRIES", 0)
    rng = random.Random(seed)
    shapes = mixed_shapes(seed)
    index = _StaticPrimitiveIndex(shapes, exclusion_m=.1)
    for _ in range(200):
        point = tuple(rng.uniform(-6, 6) for _ in range(3))
        uncertainty = rng.choice((0., .001, .1, .5, 2., 100.))
        expected = any(vehicle_clearance(point, shape, radius_m=.001, half_height_m=.001)
                       <= .1 + uncertainty for shape in shapes)
        assert index.contains_known_static(point, uncertainty_m=uncertainty) is expected


# 功能：格边界附近的圆形探针对角半径仍纳入宽阶段；不得漏掉内点或接触边界。
# 输入：monkeypatch、force_large：切换普通格与大基元旁表。输出：两路均等于完整标量。
@pytest.mark.parametrize("sign", (-1, 1))
@pytest.mark.parametrize("force_large", (False, True))
def test_round_primitive_preserves_diagonal_probe_at_bin_boundary(monkeypatch, sign, force_large):
    if force_large:
        monkeypatch.setattr(_StaticPrimitiveIndex, "_MAX_BIN_ENTRIES", 0)
    shape = dict(center_x=sign * 1.8988, center_y=0., center_z=0., radius_m=.1)
    index = _StaticPrimitiveIndex([shape], exclusion_m=0.)
    point = (sign * 2.0001, 0., 0.)
    assert vehicle_clearance(point, shape, radius_m=.001, half_height_m=.001) < 0.
    assert index.contains_known_static(point)
    assert index.contains_known_static((shape["center_x"], 0., 0.))


# 功能：排除远方旁表基元但保留本地与跨格可达者，缓存只存几何候选编号。
# 输入：monkeypatch：强制大基元索引。输出：候选数量受空间范围约束且结论完整。
def test_large_primitive_candidates_prune_distant_geometry(monkeypatch):
    monkeypatch.setattr(_StaticPrimitiveIndex, "_MAX_BIN_ENTRIES", 0)
    shapes = [dict(center_x=float(x), center_y=0., center_z=0., radius_m=.1)
              for x in (0, 100, 200)]
    index = _StaticPrimitiveIndex(shapes, exclusion_m=.1)
    assert index._candidates((0, 0, 0), (0, 0, 0)) == (0,)
    assert index.contains_known_static((100., 0., 0.))
    assert index.contains_known_static((99.7, 0., 0.), uncertainty_m=.2)


# 功能：大箱体中心距边界很远仍保留内部和表面邻域，旋转与定位余量不缩小宽阶段。
# 输入：yaw、sign、uncertainty：合成姿态与边界两侧；输出：全部结果等于有符号标量净空。
@pytest.mark.parametrize("yaw", (0., math.pi / 5))
@pytest.mark.parametrize("sign", (-1, 1))
@pytest.mark.parametrize("uncertainty", (0., .5))
def test_large_box_signed_interior_and_surface_margin(yaw, sign, uncertainty):
    shape = dict(center_x=-100., center_y=60., center_z=4., size_x=200.,
                 size_y=20., size_z=8., yaw_rad=yaw)
    index = _StaticPrimitiveIndex([shape], exclusion_m=.1)
    assert index._large_primitives == [0]
    assert index.contains_known_static((-100., 60., 4.))
    for gap in (-5., -.001, .05, .1009, .1011, .6011, 4.):
        local_x = sign * (100. + gap)
        point = (-100. + math.cos(yaw) * local_x, 60. + math.sin(yaw) * local_x, 4.)
        expected = vehicle_clearance(point, shape, radius_m=.001, half_height_m=.001)
        assert index.contains_known_static(point, uncertainty_m=uncertainty) is (
            expected <= .1 + uncertainty)
