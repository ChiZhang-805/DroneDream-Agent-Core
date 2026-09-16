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
    assert index.contains_known_static((2., 0., 0.))
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
