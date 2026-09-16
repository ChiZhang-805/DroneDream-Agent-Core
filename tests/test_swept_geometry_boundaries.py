"""No unsafe coercion or silently ignored non-finite geometry in batched scoring."""

import pytest
from test_training_runtime_evidence import WALL

from dronedream_agent_core.training.swept_geometry import SweptMapGeometry


# 功能：
#   验证批量几何与标量评分一样拒绝布尔尺寸，不把开关值转为一米包络。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_batched_geometry_rejects_boolean_envelope():
    with pytest.raises(ValueError):
        SweptMapGeometry([WALL], radius_m=True, half_height_m=.1)


# 功能：
#   验证来自原语字段的布尔坐标或数值字符串不能被浮点数组静默转换。
# 输入：
#   value：错误类型的几何中心。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, "0"])
def test_primitive_coordinates_are_real_numbers(value):
    with pytest.raises(ValueError):
        SweptMapGeometry([{**WALL, "center_x": value}], radius_m=.1, half_height_m=.1)


# 功能：
#   验证批量查询拒绝布尔位置，即使它能参与 Python 距离计算。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_positions_are_not_boolean_metrics():
    geometry = SweptMapGeometry([WALL], radius_m=.1, half_height_m=.1)
    with pytest.raises(ValueError):
        geometry.clearance([(False, 0, 1), (1, 0, 1)])


# 功能：
#   验证某类原语溢出形成 NaN 时不能被另一类正常障碍的有限距离掩盖。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_non_finite_capsule_computation_is_not_ignored():
    primitives = [WALL, {"center_x": 1e200, "center_y": 0., "center_z": 0.,
                         "radius_m": .1, "length_m": 1e200}]
    geometry = SweptMapGeometry(primitives, radius_m=.1, half_height_m=.1)
    with pytest.raises(ValueError):
        geometry.clearance([(0., 0., 1.), (1., 0., 1.)])
