"""Physical joystick admission boundaries without a simulator or flight controller."""

import pytest

from dronedream_agent_core.contracts import NormalizedPilotControl
from dronedream_agent_core.pilot_control_mapping import (
    PilotControlLimits,
    action_risk_features,
    physical_pilot_request,
)


# 功能：
#   构造一个各方向均有幅度的合法控制，避免错误被全零输入掩盖。
# 输入：
#   无。
# 输出：
#   control：前、右、上与偏航都有显式幅度的归一化控制。
def control_axes():
    control = NormalizedPilotControl(forward_axis=.5, right_axis=-.25, up_axis=.2, yaw_axis=-.1)
    return control


# 功能：
#   物理限额的数值溢出和非法类型都必须受控拒绝。
# 输入：
#   field：水平速度、垂直速度或偏航限额。
#   value：破坏限额的超界或非法值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["horizontal_speed_mps", "vertical_speed_mps", "yaw_rate_dps"])
@pytest.mark.parametrize("value", [2**4096, True, None, float("nan"), 0.])
def test_limit_validation_never_overflows(field, value):
    values = dict(horizontal_speed_mps=2., vertical_speed_mps=1., yaw_rate_dps=30.)
    values[field] = value
    with pytest.raises(ValueError):
        PilotControlLimits(**values)


# 功能：
#   Harness 比例在运算前按类型与范围检查，不通过隐式转换接受字符串或布尔值。
# 输入：
#   scale：非法缩放比例。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("scale", [None, "1", 2**4096, True, float("nan"), 0.])
def test_scale_rejection_is_a_controlled_error(scale):
    with pytest.raises(ValueError):
        physical_pilot_request(control_axes(), PilotControlLimits(2, 1, 30), harness_scale=scale)


# 功能：
#   绕过模型验证后的错误四轴值必须同时被物理映射和风险编码拒绝。
# 输入：
#   field：被破坏的控制轴。
#   value：非法轴值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["forward_axis", "right_axis", "up_axis", "yaw_axis"])
@pytest.mark.parametrize("value", [float("nan"), True, "0", 1.01])
def test_mutated_axes_never_reach_physical_or_risk_output(field, value):
    control = control_axes().model_copy(update={field: value})
    limits = PilotControlLimits(2, 1, 30)
    for convert in (physical_pilot_request, action_risk_features):
        with pytest.raises(ValueError):
            convert(control, limits, harness_scale=.5)


# 功能：
#   即使冻结限额被低层强行改写，执行边界仍重新检查，不产生非法物理指令。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_limits_are_revalidated_at_the_mapping_boundary():
    limits = PilotControlLimits(2, 1, 30)
    object.__setattr__(limits, "yaw_rate_dps", float("nan"))
    with pytest.raises(ValueError):
        physical_pilot_request(control_axes(), limits, harness_scale=.5)


# 功能：
#   错误的控制或限额容器不能触发属性错误后逃出控制校验边界。
# 输入：
#   field：被替换的参数名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["control", "limits"])
def test_mapping_rejects_untyped_objects(field):
    inputs = dict(control=control_axes(), limits=PilotControlLimits(2, 1, 30), harness_scale=.5)
    inputs[field] = {}
    with pytest.raises(ValueError):
        physical_pilot_request(**inputs)


# 功能：
#   不擅自归一化斜向摇杆；合成速度超过总包络时拒绝，合法方向和缩放保持不变。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_diagonal_motion_uses_the_shared_vector_envelope():
    control = NormalizedPilotControl(forward_axis=1, right_axis=1, up_axis=0, yaw_axis=0)
    limits = PilotControlLimits(20, 1, 30)
    with pytest.raises(ValueError, match="velocity envelope"):
        physical_pilot_request(control, limits, harness_scale=1.)
    assert physical_pilot_request(control, limits, harness_scale=.5) == (10., 10., 0., 0.)
