import itertools
import math
import random

import pytest

from dronedream_agent_core.contracts import RangeRayObservation, Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.occupancy_collision import (
    bounded_occupied_cell_boxes,
    box_fully_inside,
    fully_contained_box_mask,
    merged_occupied_cell_boxes,
)
from scripts.runtime_depth_safety_worker import _remove_known_static_perception_duplicates


# 功能：
#   构造可覆盖字段的单位盒夹具，用于包含关系和非法几何对照。
# 输入：
#   changes：需要覆盖的几何字段。
# 输出：
#   result：盒几何字典。
def box(**changes):
    result = {"shape": "box", "center_x": 0., "center_y": 0., "center_z": 0.,
            "size_x": 1., "size_y": 1., "size_z": 1., "yaw_rad": 0., **changes}
    return result


# 功能：
#   将小规模半开区间盒穷举成整数格集合，作为测试中的独立并集基准。
# 输入：
#   bounds：三轴下界及对应的排他上界。
# 输出：
#   result：该盒实际覆盖的整数格集合。
def cells(bounds):
    result = set(itertools.product(*(range(bounds[a], bounds[a + 3]) for a in range(3))))
    return result


# 功能：
#   计算测试包围盒覆盖的整数格并集，用于检查孔洞、重复覆盖和漏格。
# 输入：
#   boxes：小规模测试包围盒列表。
# 输出：
#   result：全部盒覆盖的格集合。
def union(boxes):
    result = set().union(*(cells(bounds) for bounds in boxes))
    return result


# 功能：
#   核对去重仅删除完整包含的盒，并保留穿出静态障碍边界的感知盒。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_full_containment_removes_only_a_redundant_collision_volume():
    enclosing = box(size_x=3., size_y=3., size_z=3.)
    contained, protruding = box(), box(center_x=1.1)
    result, count = _remove_known_static_perception_duplicates([contained, protruding], [enclosing])
    assert result == [protruding]
    assert count == 1


# 功能：
#   检查部分重叠、接触边界、未知形状和非法数值都不能证明完整包含。
# 输入：
#   changes：覆盖单位盒的边界或非法字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("changes", [
    {"center_x": .51}, {"center_y": .51}, {"center_z": .51},
    {"size_x": 2.01}, {"size_y": 2.01}, {"size_z": 2.01},
    {"center_x": .5},  # Boundary is retained, not removed using a positive tolerance.
    {"shape": "sphere"}, {"roll_rad": 1e-12}, {"pitch_rad": .3},
    {"yaw_rad": float("nan")}, {"center_x": float("inf")},
    {"size_y": -1.}, {"size_z": 0.}, {"size_x": None},
])
def test_overlap_touching_and_unsupported_shapes_do_not_prove_containment(changes):
    assert not box_fully_inside(box(**changes), box(size_x=2., size_y=2., size_z=2.))


# 功能：
#   检查包含判断使用外盒局部轴，而非仅比较世界轴尺寸或中心距离。
# 输入：
#   yaw：内外盒共同的偏航角，单位弧度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("yaw", [0., .17, math.pi / 4, math.pi / 2, -2.3])
def test_containment_is_in_enclosing_box_axes(yaw):
    outer = box(size_x=3., size_y=.5, size_z=1., yaw_rad=yaw)
    inner = box(size_x=2., size_y=.2, size_z=.5, yaw_rad=yaw)
    assert box_fully_inside(inner, outer)
    assert not box_fully_inside({**inner, "center_x": -.4 * math.sin(yaw),
                                 "center_y": .4 * math.cos(yaw)}, outer)


# 功能：
#   检查同中心旋转后的内盒角点不能越界，并拒绝缺少尺寸的外盒。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_rotated_inner_corners_must_fit_even_with_same_center():
    assert not box_fully_inside(box(yaw_rad=math.pi / 4), box(size_x=1.2, size_y=1.2))
    assert not box_fully_inside(box(), {"shape": "box"})


# 功能：
#   用跨越两维分块边界的随机盒批次核对向量筛选与逐对标量证明一致。
# 输入：
#   seed：可复现的随机种子。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("seed", range(6))
def test_batched_containment_preserves_scalar_proof_across_both_block_boundaries(seed):
    rng = random.Random(seed)

    # 功能：
    #   在固定随机源下生成不同中心、尺寸和朝向的盒。
    # 输入：
    #   count：盒数量。
    #   scale：各轴尺寸的采样上限。
    # 输出：
    #   result：本次生成的盒列表。
    def boxes(count, scale):
        result = [box(**{f"center_{axis}": rng.uniform(-4, 4) for axis in "xyz"},
                    **{f"size_{axis}": rng.uniform(.01, scale) for axis in "xyz"},
                    yaw_rad=rng.uniform(-math.pi, math.pi)) for _ in range(count)]
        return result

    inner, outer = boxes(111, 2), boxes(143, 8)
    expected = [any(box_fully_inside(candidate, primitive) for primitive in outer)
                for candidate in inner]
    assert fully_contained_box_mask(inner, outer) == expected


# 功能：
#   检查批量筛选保留畸形、倾斜、接触及不可证明的极端数值几何，并覆盖空输入。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_batch_never_removes_malformed_rotated_or_touching_evidence():
    candidates = [None, {}, box(shape="sphere"), box(center_x=.5), box(roll_rad=.001),
                  box(size_x=float("inf")), box(yaw_rad=1e308), box()]
    enclosing = [None, {}, box(pitch_rad=.001), box(yaw_rad=-1e308),
                 box(size_x=2., size_y=2., size_z=2.)]
    expected = [any(box_fully_inside(candidate, primitive) for primitive in enclosing)
                for candidate in candidates]
    assert fully_contained_box_mask(candidates, enclosing) == expected
    assert expected[:6] == [False] * 6
    assert expected[-1]
    assert fully_contained_box_mask([], enclosing) == []
    assert fully_contained_box_mask(candidates, []) == [False] * len(candidates)


# 功能：
#   检查粗筛只按盒解析，不为每个无关配对重复解析；任何删除仍调用标量证明。
# 输入：
#   monkeypatch：注入计数包装，不改变几何判定结果。
# 输出：
#   None：不返回业务数据。
def test_broad_phase_does_not_reparse_every_irrelevant_pair(monkeypatch):
    import dronedream_agent_core.occupancy_collision as module

    rows, proofs = [], []
    original_row, original_proof = module._box_row, module.box_fully_inside

    # 功能：
    #   记录每次盒解析并透传真实解析结果。
    # 输入：
    #   value：待解析几何。
    # 输出：
    #   result：原解析器的数值行或 None。
    def row(value):
        rows.append(value)
        result = original_row(value)
        return result

    # 功能：
    #   记录真正进入删除授权的几何配对，并保留原有标量证明。
    # 输入：
    #   candidate：候选盒。
    #   enclosing：外盒。
    # 输出：
    #   result：原标量包含判定。
    def proof(candidate, enclosing):
        proofs.append((candidate, enclosing))
        result = original_proof(candidate, enclosing)
        return result

    monkeypatch.setattr(module, "_box_row", row)
    monkeypatch.setattr(module, "box_fully_inside", proof)
    candidates, enclosing = [box()] * 96, [box(center_x=20.)] * 256
    assert fully_contained_box_mask(candidates, enclosing) == [False] * 96
    assert len(rows) == 96 + 256  # Not 96 * 256 * 2 scalar row validations.
    assert proofs == []
    # A prospective removal still has to pass the guarded scalar authority.
    assert fully_contained_box_mask([box()], [box(size_x=2., size_y=2., size_z=2.)]) == [True]
    assert len(proofs) == 1


# 功能：
#   检查实心体精确合并、挖孔后并集不变，并保持仅对角接触的格相互独立。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_solid_volume_merges_exactly_and_holes_never_fill():
    full = list(itertools.product(range(-3, 4), range(-2, 3), range(1, 5)))
    assert merged_occupied_cell_boxes(full) == [(-3, -2, 1, 4, 3, 5)]
    full.remove((0, 0, 2))
    merged = merged_occupied_cell_boxes(full)
    assert union(merged) == set(full)
    assert sum(len(cells(item)) for item in merged) == len(full)  # No overlapping output.
    assert merged_occupied_cell_boxes([(0, 0, 0), (1, 1, 1)]) == [
        (0, 0, 0, 1, 1, 1), (1, 1, 1, 2, 2, 2)]


# 功能：
#   随机检查输入乱序和重复不影响精确并集，预算粗化后仍覆盖每一个真实占用格。
# 输入：
#   seed：可复现的稀疏占用格种子。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("seed", range(12))
def test_random_sparse_cells_keep_exact_union_order_and_bounded_conservative_cover(seed):
    rng = random.Random(seed)
    source = list({tuple(rng.randrange(-4, 5) for _ in range(3)) for _ in range(120)})
    expected = merged_occupied_cell_boxes(source)
    rng.shuffle(source)
    assert merged_occupied_cell_boxes(source + source[:10]) == expected
    assert union(expected) == set(source)
    for limit in (1, 2, 5, 96):
        bounded = bounded_occupied_cell_boxes(source, center_in_cells=(.3, -.5, .9), limit=limit)
        assert len(bounded) <= limit
        assert set(source) <= union([bounds for bounds, _ in bounded])
        assert sum(coarse for _, coarse in bounded) <= 1
        assert bool(any(coarse for _, coarse in bounded)) == (len(expected) > limit)
        for bounds, coarse in bounded:
            if not coarse:
                assert cells(bounds) <= set(source)


# 功能：
#   检查输出预算必须是正整数，布尔值和非有限值不能混作数量。
# 输入：
#   limit：非法输出预算。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("limit", [0, -1, True, 1.5, float("nan")])
def test_invalid_output_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="limit"):
        bounded_occupied_cell_boxes([], center_in_cells=(0., 0., 0.), limit=limit)


# 功能：
#   检查空占用输入不生成盒，同时仍拒绝非有限查询中心。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_empty_occupancy_has_no_invented_boxes():
    assert bounded_occupied_cell_boxes([], center_in_cells=(0., 0., 0.), limit=1) == []
    with pytest.raises(ValueError, match="center"):
        bounded_occupied_cell_boxes([], center_in_cells=(float("nan"), 0., 0.), limit=1)


# 功能：
#   向测试地图写入零长度命中射线，避免清空较早占用格，隔离测试合并几何本身。
# 输入：
#   world：待更新的测试体素地图。
#   point：命中位置。
#   stamp：合成观测单调时刻。
# 输出：
#   None：不返回业务数据。
def add_hit(world, point, *, stamp=10.):
    # Zero-length hit is legitimate evidence of the occupied source cell. It
    # avoids clearing earlier cells in this synthetic geometry contract test.
    world.integrate_ray(RangeRayObservation(origin_m=point, endpoint_m=point, hit=True,
                                           confidence=.99, observed_at_monotonic_seconds=stamp))


# 功能：
#   通过实际体素地图出口检查预算溢出时远处占用格的全部角点仍被保守覆盖。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_world_collision_budget_does_not_discard_farther_occupied_cells():
    world = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=-5., y=-2., z=-1.),
                           maximum_bound_m=Vector3(x=5., y=2., z=2.))
    points = [Vector3(x=x, y=.1, z=.5) for x in (-3.8, -1.8, .2, 2.2, 4.2)]
    for point in points:
        add_hit(world, point)
    result = world.local_occupied_box_primitives(center_m=Vector3(x=0., y=0., z=0.),
        radius_m=6., now_monotonic_seconds=10.1, limit=2)
    assert len(result) == 2
    assert any(item["name"].startswith("perception-coarsened-overflow-") for item in result)
    for point in points:
        cell_center = world.center_for(world.key_for(point))
        for signs in itertools.product((-1, 1), repeat=3):
            corner = [getattr(cell_center, axis) + sign * .125
                      for axis, sign in zip("xyz", signs, strict=True)]
            assert any(all(abs(corner[i] - item[f"center_{axis}"]) <= item[f"size_{axis}"] / 2
                           for i, axis in enumerate("xyz")) for item in result)


# 功能：
#   检查查询球接触体素近面时仍保留该格，不因格中心在半径外而漏检。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cell_face_intersection_is_not_missed_by_center_radius_filter():
    world = MetricVoxelMap(resolution_m=1., minimum_bound_m=Vector3(x=0., y=0., z=0.),
                           maximum_bound_m=Vector3(x=5., y=5., z=5.))
    add_hit(world, Vector3(x=2.5, y=2.5, z=2.5))
    result = world.local_occupied_box_primitives(center_m=Vector3(x=1., y=2.5, z=2.5),
        radius_m=1., now_monotonic_seconds=10.1)
    assert len(result) == 1  # Cell center is 1.5 m away; its near face is 1 m away.


# 功能：
#   检查体素查询的半径、时钟、年龄和数量预算遵守数值契约。
# 输入：
#   field：待替换的查询参数名。
#   value：对应的非法参数值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [("radius_m", float("nan")),
    ("maximum_age_seconds", float("inf")), ("now_monotonic_seconds", -1.),
    ("now_monotonic_seconds", float("nan")), ("limit", True)])
def test_world_query_rejects_invalid_numeric_contract(field, value):
    world = MetricVoxelMap(resolution_m=1., minimum_bound_m=Vector3(x=0., y=0., z=0.),
                           maximum_bound_m=Vector3(x=5., y=5., z=5.))
    kwargs = dict(center_m=Vector3(x=1., y=1., z=1.), radius_m=2.,
                  now_monotonic_seconds=10., maximum_age_seconds=.5, limit=96)
    kwargs[field] = value
    with pytest.raises(ValueError):
        world.local_occupied_box_primitives(**kwargs)
