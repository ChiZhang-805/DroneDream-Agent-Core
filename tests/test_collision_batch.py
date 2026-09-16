import math
import random

import numpy as np
import pytest
from test_dynamic_safety import _request

from dronedream_agent_core import dynamic_safety
from dronedream_agent_core.collision import vehicle_clearance
from dronedream_agent_core.collision_batch import static_point_clearances
from dronedream_agent_core.contracts import DynamicObstacleObservation, Vector3


# 功能：
#   按固定随机种子生成箱体、直立或倾斜圆柱、胶囊和球体，包含退化短轴用于数值比较。
# 输入：
#   seed：可复现的随机种子。
# 输出：
#   shapes：包含二十四个不同几何基元的测试列表。
def mixed_shapes(seed):
    rng = random.Random(seed)
    shapes = []
    for i in range(24):
        shape = {f"center_{axis}": rng.uniform(-3, 3) for axis in "xyz"}
        shape["name"] = str(i)
        kind = i % 6
        if kind == 0:
            shape.update({f"size_{axis}": rng.uniform(0.01, 3) for axis in "xyz"})
            shape["yaw_rad"] = rng.uniform(-math.pi, math.pi)
        else:
            shape["radius_m"] = rng.uniform(0.01, 1)
            if kind in {1, 2}:
                shape["height_m"] = rng.uniform(0.01, 3)
            if kind in {2, 3, 4}:
                shape.update(roll_rad=0.1, pitch_rad=-0.3, yaw_rad=0.7)
            if kind in {3, 4}:
                shape["length_m"] = 1e-12 if kind == 4 else 3.0
        shapes.append(shape)
    return shapes


# 功能：
#   比较全部形状的标量与批量距离及最近障碍身份，数值一致不等同于真实地图已验证。
# 输入：
#   seed：生成当前几何与采样点的可复现种子。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("seed", range(10))
def test_all_shapes_match_scalar_clearances_and_obstacle_identity(seed):
    rng, shapes = random.Random(seed), mixed_shapes(seed)
    points = [tuple(rng.uniform(-4, 4) for _ in range(3)) for _ in range(81)]
    actual, indices = static_point_clearances(points, shapes, radius_m=0.38, half_height_m=0.215)
    expected = [
        [vehicle_clearance(p, s, radius_m=0.38, half_height_m=0.215) for s in shapes]
        for p in points
    ]
    np.testing.assert_allclose(actual, np.min(expected, axis=1), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(indices, np.argmin(expected, axis=1))


# 功能：
#   跨数组分块仍保留最后一个查询点，同分时保留首个障碍，空地图保持明确哨兵值。
# 输入：
#   无：使用一千零二十五个点及两个相同障碍。
# 输出：
#   None：不返回业务数据。
def test_batches_keep_last_point_and_first_primitive_on_equal_clearance():
    shapes = [dict(center_x=0.0, center_y=0.0, center_z=0.0, radius_m=1.0, name="first")]
    shapes.append({**shapes[0], "name": "second"})
    points = [(10.0, 0.0, 0.0)] * 1024 + [(0.0, 0.0, 0.0)]
    clearance, nearest = static_point_clearances(points, shapes, radius_m=0.1, half_height_m=0.1)
    assert len(clearance) == 1025 and clearance[-1] < 0 and set(nearest) == {0}
    clearance, nearest = static_point_clearances(points, [], radius_m=0.1, half_height_m=0.1)
    assert set(clearance) == {999.0} and set(nearest) == {-1}
    with pytest.raises(ValueError, match="INPUT_INVALID"):
        static_point_clearances([(float("nan"), 0.0, 0.0)], [], radius_m=0.1, half_height_m=0.1)


# 功能：
#   将候选预测和最终安全决策与逐点标量实现对比，确保批量优化不改变控制或威胁来源。
# 输入：
#   seed：混合静态形状的可复现随机种子。
#   monkeypatch：将批量预测临时替换为标量循环的工具。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("seed", range(10))
def test_candidate_predictions_and_selected_safety_decision_match_scalar(seed, monkeypatch):
    request = _request(
        obstacles=[
            DynamicObstacleObservation(
                obstacle_id="moving",
                position_m=Vector3(x=2, y=-1, z=1),
                velocity_mps=Vector3(x=-0.1, y=0.5, z=0.1),
                radius_m=0.2,
                height_m=1.5,
                confidence=0.8,
                age_seconds=0.15,
            )
        ]
    )
    shapes = mixed_shapes(seed)
    candidates = dynamic_safety._candidate_velocities(request)
    actual = dynamic_safety._predict_candidates(request, candidates, shapes)
    expected = [dynamic_safety._predict(request, velocity, shapes) for velocity in candidates]
    for a, e in zip(actual, expected, strict=True):
        assert a[0] == e[0] and a[2:] == e[2:]
        assert a[1] == pytest.approx(e[1], abs=1e-12)
    result = dynamic_safety.predictive_safety_decision(request, shapes)
    monkeypatch.setattr(
        dynamic_safety,
        "_predict_candidates",
        lambda r, vs, ps: [dynamic_safety._predict(r, v, ps) for v in vs],
    )
    scalar = dynamic_safety.predictive_safety_decision(request, shapes)
    assert result == scalar  # Final full-map scalar verification retains identical evidence.
