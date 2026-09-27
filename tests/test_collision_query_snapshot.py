import copy
import random

import pytest
from test_collision_batch import mixed_shapes
from test_dynamic_safety import _request

from dronedream_agent_core import collision, dynamic_safety
from dronedream_agent_core.contracts import DynamicObstacleObservation, Vector3


# 功能：
#   用未优化的逐点公开查询计算独立参考答案，保持原来的哨兵值与首个同分障碍。
# 输入：
#   points、primitives：测试位置与完整几何；radius_m、half_height_m：机体尺寸。
# 输出：
#   samples：每个位置的最小净空与障碍索引。
def scalar_reference(points, primitives, *, radius_m, half_height_m):
    samples = []
    for point in points:
        minimum, nearest = 999.0, -1
        for index, primitive in enumerate(primitives):
            clearance = collision.vehicle_clearance(point, primitive, radius_m=radius_m, half_height_m=half_height_m)
            if clearance < minimum:
                minimum, nearest = clearance, index
        samples.append((minimum, nearest))
    return samples


# 功能：
#   对混合形状、倾斜箱体和随机位置逐值验证批次校验与原实现完全相同，且不修改调用者输入。
# 输入：
#   seed：可复现的测试种子。
# 输出：
#   None：断言精确净空、身份和输入不变。
@pytest.mark.parametrize("seed", range(20))
def test_snapshot_query_is_bitwise_equal_to_scalar(seed):
    rng, primitives = random.Random(seed), mixed_shapes(seed)
    primitives[0].update(roll_rad=0.37, pitch_rad=-0.24)
    points = [tuple(rng.uniform(-5, 5) for _ in range(3)) for _ in range(81)]
    before = copy.deepcopy((points, primitives))
    actual = collision.minimum_vehicle_clearances(points, primitives, radius_m=0.38, half_height_m=0.215)
    assert actual == scalar_reference(points, primitives, radius_m=0.38, half_height_m=0.215)
    assert (points, primitives) == before


# 功能：
#   确认同分身份、远障碍哨兵、空查询与最后一个位置均与原标量契约一致。
# 输入：
#   无：使用同位置球体与跨千点的查询。
# 输出：
#   None：断言边界结果。
def test_query_ties_empty_and_last_point():
    shape = dict(center_x=0.0, center_y=0.0, center_z=0.0, radius_m=1.0)
    points = [(2000.0, 0.0, 0.0)] * 1024 + [(0.0, 0.0, 0.0)]
    actual = collision.minimum_vehicle_clearances(points, [shape, shape], radius_m=0.1, half_height_m=0.1)
    assert actual[:-1] == [(999.0, -1)] * 1024
    assert actual[-1][1] == 0 and actual[-1][0] < 0
    assert collision.minimum_vehicle_clearances(points, [], radius_m=0.1, half_height_m=0.1) == [(999.0, -1)] * len(points)
    assert collision.minimum_vehicle_clearances([], [shape], radius_m=0.1, half_height_m=0.1) == []


# 功能：
#   即使不存在可配对的位置或障碍，仍拒绝无效数值、形状、机体尺寸及过大输入。
# 输入：
#   change：本例注入的非法参数。
# 输出：
#   None：断言不能发布有误导性的有效净空。
@pytest.mark.parametrize("change", [
    {"points": [(float("nan"), 0, 0)]},
    {"points": [(float("inf"), 0, 0)]},
    {"points": [(True, 0, 0)]},
    {"points": [("1", 0, 0)]},
    {"points": [(10 ** 1000, 0, 0)]},
    {"points": [(0, 0)]},
    {"points": iter([(0, 0, 0)])},
    {"points": [(0, 0, 0)] * 250_001},
    {"primitives": [{}]},
    {"primitives": [dict(center_x=0, center_y=0, center_z=0, size_x=1)]},
    {"primitives": [{}] * 100_001},
    {"primitives": ()},
    {"radius_m": 0}, {"half_height_m": True}, {"half_height_m": float("nan")},
])
def test_query_rejects_invalid_input(change):
    arguments = dict(points=[], primitives=[], radius_m=0.1, half_height_m=0.1)
    arguments.update(change)
    with pytest.raises(ValueError):
        collision.minimum_vehicle_clearances(**arguments)


# 功能：
#   防止有限输入在运算中溢出后被哨兵值掩盖，禁止把无限距离当作安全。
# 输入：
#   无：构造相向的极大坐标。
# 输出：
#   None：断言查询拒绝溢出。
def test_query_rejects_arithmetic_overflow():
    shape = dict(center_x=-1e308, center_y=0.0, center_z=0.0, radius_m=1.0)
    with pytest.raises(ValueError, match="overflow"):
        collision.minimum_vehicle_clearances([(1e308, 0.0, 0.0)], [shape], radius_m=0.1, half_height_m=0.1)


# 功能：
#   验证物理字段和位置只冻结一次，后续外部修改不会混进本次查询。
# 输入：
#   monkeypatch：在几何计算开始时注入外部更新。
# 输出：
#   None：断言计算结果不变，校验次数不随预测点数增长。
def test_query_owns_physical_inputs_and_validates_once(monkeypatch):
    points, primitives = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], mixed_shapes(2)
    expected = scalar_reference(points, primitives, radius_m=0.1, half_height_m=0.1)
    original_clearance, original_validate = collision._clearance, collision._validated_primitive
    validations = []

    # 功能：
    #   记录真实校验次数，不替换校验规则。
    # 输入：
    #   primitive：当前障碍。
    # 输出：
    #   validated：原校验器生成的独立几何。
    def count_validate(primitive):
        validations.append(1)
        validated = original_validate(primitive)
        return validated

    # 功能：
    #   在冻结完成后的计算阶段改变原输入，模拟调用者更新。
    # 输入：
    #   point、primitive、kwargs：已固定的查询参数。
    # 输出：
    #   clearance：独立输入上的原始净空结果。
    def mutate_original(point, primitive, **kwargs):
        points[1][0] = 10000.0
        primitives[0]["center_x"] = 10000.0
        clearance = original_clearance(point, primitive, **kwargs)
        return clearance

    monkeypatch.setattr(collision, "_validated_primitive", count_validate)
    monkeypatch.setattr(collision, "_clearance", mutate_original)
    actual = collision.minimum_vehicle_clearances(points, primitives, radius_m=0.1, half_height_m=0.1)
    assert actual == expected
    assert len(validations) == len(primitives)


# 功能：
#   验证优化没有改变动态区间判断、最终安全动作、威胁身份和完整几何复核。
# 输入：
#   seed：混合几何种子；monkeypatch：切换独立标量参考。
# 输出：
#   None：断言预测与最终决策严格相等。
@pytest.mark.parametrize("seed", range(10))
def test_full_prediction_and_decision_match_original(seed, monkeypatch):
    request = _request(obstacles=[DynamicObstacleObservation(
        obstacle_id="crossing", position_m=Vector3(x=2, y=-1, z=1),
        velocity_mps=Vector3(x=-0.1, y=0.5, z=0.1), radius_m=0.2,
        height_m=1.5, confidence=0.8, age_seconds=0.15,
    )])
    primitives = mixed_shapes(seed)
    velocity = (0.3, -0.2, 0.1)
    actual = dynamic_safety._predict(request, velocity, primitives)
    decision = dynamic_safety.predictive_safety_decision(request, primitives)
    monkeypatch.setattr(dynamic_safety, "minimum_vehicle_clearances", scalar_reference)
    assert dynamic_safety._predict(request, velocity, primitives) == actual
    assert dynamic_safety.predictive_safety_decision(request, primitives) == decision


# 功能：
#   静止或重复位置只在本次独立几何上复用净空，换位置或换快照仍计算全部障碍。
# 输入：
#   monkeypatch：记录真正的静态计算次数。
# 输出：
#   None：断言计算次数减少且修改地图后没有旧净空泄漏。
def test_identical_positions_share_only_current_static_query(monkeypatch):
    primitives = mixed_shapes(3)
    points = [(0.0, 0.0, 1.0)] * 8 + [(0.1, 0.0, 1.0)] * 3 + [(0.0, 0.0, 1.0)]
    expected = scalar_reference(points, primitives, radius_m=0.2, half_height_m=0.15)
    original = collision._clearance
    calls = []

    # 功能：
    #   统计完整静态几何的实际调用，不改变原标量公式。
    # 输入：
    #   point、primitive、kwargs：原始查询参数。
    # 输出：
    #   clearance：原始净空。
    def record_clearance(point, primitive, **kwargs):
        calls.append(1)
        clearance = original(point, primitive, **kwargs)
        return clearance

    monkeypatch.setattr(collision, "_clearance", record_clearance)
    assert collision.minimum_vehicle_clearances(points, primitives, radius_m=0.2, half_height_m=0.15) == expected
    assert len(calls) == 3 * len(primitives)
    primitives.append(dict(center_x=0.0, center_y=0.0, center_z=1.0, radius_m=2.0))
    calls.clear()
    result = collision.minimum_vehicle_clearances(points, primitives, radius_m=0.2, half_height_m=0.15)
    assert len(calls) == 3 * len(primitives)
    assert result != expected


# 功能：
#   机体静止不代表动态环境静止；保留每个预测区间，识别随后进入机体包络的物体。
# 输入：
#   monkeypatch：切换静态查询的独立标量参考。
# 输出：
#   None：断言未来动态碰撞仍被识别且与原预测严格相等。
def test_stationary_vehicle_still_predicts_incoming_dynamic_obstacle(monkeypatch):
    request = _request(obstacles=[DynamicObstacleObservation(
        obstacle_id="incoming", position_m=Vector3(x=2, y=0, z=1),
        velocity_mps=Vector3(x=-2, y=0, z=0), radius_m=0.2,
        height_m=1.0, confidence=1.0, age_seconds=0.0,
    )])
    primitives = [dict(center_x=5.0, center_y=0.0, center_z=1.0, radius_m=0.2)]
    actual = dynamic_safety._predict(request, (0.0, 0.0, 0.0), primitives)
    assert len(actual[0]) == 10
    assert actual[1] < 0 and actual[2] > 0 and "incoming" in actual[3]
    monkeypatch.setattr(dynamic_safety, "minimum_vehicle_clearances", scalar_reference)
    assert dynamic_safety._predict(request, (0.0, 0.0, 0.0), primitives) == actual
