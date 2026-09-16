import math
import random

from dronedream_agent_core.contracts import CalibratedRangeSensorMount, QuaternionWxyz, Vector3
from dronedream_agent_core.realtime_feature_encoders import (
    _calibrated_spatial_support,
    _expected_spatial_cells,
    _rotate,
    _spatial_index,
)


# 功能：
#   创建带固定平移及指定姿态的测试安装契约，保持明确的物理量程。
# 输入：
#   quaternion：传感器到机体的安装四元数。
# 输出：
#   sensor：当前测试安装契约。
def mount(quaternion):
    sensor = CalibratedRangeSensorMount(
        sensor_id="support-camera", translation_body_m=Vector3(x=.1, y=0, z=.05),
        orientation_body_from_sensor=quaternion, minimum_range_m=.2, maximum_range_m=19.1)
    return sensor


# 功能：
#   直接枚举视场方向作为缓存实现的对照，并将 FLU 左轴转换到空间格使用的右轴。
# 输入：
#   sensor：安装外参及量程。
#   horizontal：水平视场角，单位弧度。
#   vertical：竖直视场角，零表示平面激光扫描。
# 输出：
#   cells：视场方向覆盖的空间索引集合。
def reference_support(sensor, horizontal, vertical):
    cells = set()
    for azimuth_index in range(33):
        azimuth = horizontal * (azimuth_index / 32 - .5)
        for elevation_index in range(17 if vertical else 1):
            elevation = vertical * (elevation_index / 16 - .5)
            direction = _rotate(sensor.orientation_body_from_sensor, (
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth), math.sin(elevation)))
            cells.add(_spatial_index(direction[0], -direction[1], direction[2]))
    return cells


# 功能：
#   在可复现的随机安装姿态下，对照验证倾斜相机及平面雷达的缓存视场索引。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cached_calibration_is_exact_for_tilted_cameras_and_planar_lidar():
    rng = random.Random(805)
    for _ in range(40):
        values = [rng.uniform(-1, 1) for _ in range(4)]
        norm = math.sqrt(sum(v*v for v in values))
        q = QuaternionWxyz(**dict(zip("wxyz", (v/norm for v in values), strict=True)))
        sensor = mount(q)
        for horizontal, vertical in ((2*math.pi, 0), (1.274, .987), (.05, math.pi)):
            assert _expected_spatial_cells(sensor, horizontal, vertical) == reference_support(
                sensor, horizontal, vertical)


# 功能：
#   相同校准值应复用不可变集合，改换安装姿态必须重新计算，不能按传感器名沿用旧结果。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cache_uses_current_calibration_values_not_mutable_sensor_identity():
    sensor = mount(QuaternionWxyz(w=1, x=0, y=0, z=0))
    _calibrated_spatial_support.cache_clear()
    before = _expected_spatial_cells(sensor, .5, .3)
    assert isinstance(before, frozenset)
    assert _expected_spatial_cells(sensor, .5, .3) is before
    assert _calibrated_spatial_support.cache_info().hits == 1
    tilted = sensor.model_copy(update={"orientation_body_from_sensor":
        QuaternionWxyz(w=math.sqrt(.5), x=0, y=math.sqrt(.5), z=0)})
    after = _expected_spatial_cells(tilted, .5, .3)
    assert before != after
    assert after == reference_support(tilted, .5, .3)
    assert _expected_spatial_cells(sensor, .5, .3) == before


# 功能：
#   验证视场变化不会复用不匹配结果，且缓存容量保持六十四项上限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cache_is_bounded_and_fov_changes_are_not_reused():
    sensor = mount(QuaternionWxyz(w=1, x=0, y=0, z=0))
    _calibrated_spatial_support.cache_clear()
    for i in range(80):
        horizontal = .05 + i*.05
        result = _expected_spatial_cells(sensor, horizontal, .5)
        assert result == reference_support(sensor, horizontal, .5)
    assert _calibrated_spatial_support.cache_info().currsize == 64
