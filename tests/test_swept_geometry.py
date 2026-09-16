import math
import random

import pytest

from dronedream_agent_core.training.outcome_verifier import swept_clearance_bound
from dronedream_agent_core.training.swept_geometry import SweptMapGeometry


# 功能：
#   对多种形状随机地图比较批量及标量保守距离，并验证调用方改动不能改写编译地图。
# 输入：
#   seed：可复现的随机地图种子。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("seed", range(8))
def test_batched_geometry_keeps_all_shapes_and_scalar_clearance(seed):
    rng = random.Random(seed)
    primitives = []
    for i in range(31):
        p = dict(zip(("center_x", "center_y", "center_z"),
                     [rng.uniform(-3, 3) for _ in range(3)], strict=True))
        if i % 4 == 0:
            p.update(size_x=rng.uniform(.001, 5), size_y=rng.uniform(.001, 5),
                     size_z=rng.uniform(.001, 5), yaw_rad=rng.uniform(-math.pi, math.pi))
        else:
            p.update(radius_m=rng.uniform(.01, 1))
            if i % 4 == 1:
                p.update(height_m=2., roll_rad=.3, pitch_rad=-.1)
            elif i % 4 == 2:
                p.update(length_m=2., yaw_rad=1.)
        primitives.append(p)
    positions = [tuple(rng.uniform(-2, 2) for _ in range(3)) for _ in range(4)]
    expected = min(swept_clearance_bound(a, b, primitives, radius_m=.38, half_height_m=.215)
                   for a, b in zip(positions, positions[1:], strict=False))
    geometry = SweptMapGeometry(primitives, radius_m=.38, half_height_m=.215)
    assert geometry.clearance(positions) == pytest.approx(expected, abs=1e-12)
    primitives.clear()  # Mutating a caller's list cannot rewrite the frozen map.
    assert geometry.clearance(positions) == pytest.approx(expected, abs=1e-12)


# 功能：
#   验证端点之间的薄墙仍被识别为碰撞，过大位移或 NaN 轨迹必须拒绝。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_thin_wall_between_endpoints_remains_collision():
    wall = dict(center_x=0., center_y=0., center_z=0., size_x=.001, size_y=2., size_z=2.)
    geometry = SweptMapGeometry([wall], radius_m=.1, half_height_m=.1)
    assert geometry.clearance([(-1, 0, 0), (1, 0, 0)]) < 0
    with pytest.raises(ValueError, match="JUMP"):
        geometry.clearance([(0, 0, 0), (11, 0, 0)])
    with pytest.raises(ValueError, match="JUMP"):
        geometry.clearance([(0, 0, 0), (float("nan"), 0, 0)])


# 功能：
#   验证超过单个地图块及采样块的数据仍全量参与查询，不漏掉末块薄墙。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_map_and_sample_batches_do_not_drop_far_or_late_geometry():
    boxes = [dict(center_x=i + 10., center_y=0., center_z=0.,
                  size_x=.001, size_y=2., size_z=2.) for i in range(1024)]
    boxes.append(dict(center_x=-1.499, center_y=0., center_z=0.,
                      size_x=.001, size_y=2., size_z=2.))
    positions = [(2., 0., 0.), (-2., 0., 0.)]
    expected = swept_clearance_bound(*positions, boxes, radius_m=.1, half_height_m=.1)
    assert SweptMapGeometry(boxes, radius_m=.1, half_height_m=.1).clearance(positions) == (
        pytest.approx(expected, abs=1e-12))
