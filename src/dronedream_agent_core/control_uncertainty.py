"""One localization allowance for planning and runtime control.

The variance is an estimator-provided direction-independent upper bound, not
truth error. Three sigma is the existing engineering risk policy, not a formal
bounded-error guarantee. Relative-map cancellation needs separate joint evidence.
"""

import math

PLANNING_LOCALIZATION_ALLOWANCE_M = .03
LOCALIZATION_SIGMA_MULTIPLIER = 3.


# 功能：
#   严格检查外部数值预算，拒绝布尔、字符串、非有限数及无法转换的巨大整数。
# 输入：
#   value：待验证的数值。
# 输出：
#   valid：该值是否为有限正数。
def finite_positive_number(value: object) -> bool:
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    return valid


# 功能：
#   将定位方差上界换算为共享的三倍标准差余量；这是工程风险策略，不是真值误差保证。
#   零只作为数值边界，上游仍须证明定位证据有效，不能据此宣称传感器完美。
# 输入：
#   variance_bound_m2：非负且有限的方向无关方差上界，单位平方米。
# 输出：
#   margin_m：用于规划和控制的定位不确定性余量，单位米。
def localization_uncertainty_margin_m(variance_bound_m2: float) -> float:
    try:
        valid = (type(variance_bound_m2) in (int, float)
                 and math.isfinite(variance_bound_m2) and variance_bound_m2 >= 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("LOCALIZATION_VARIANCE_BOUND_INVALID")
    margin_m = LOCALIZATION_SIGMA_MULTIPLIER * math.sqrt(variance_bound_m2)
    return margin_m
