"""Metric bounds for local prediction, independent of a model or simulator.

These bound the declared constant-velocity prediction, not unobserved flight
dynamics. Actual tracking, sensor uncertainty and actuator limits retain their
own independent gates.
"""

import math

SAFETY_SCORING_CLEARANCE_BONUS_M = 2.0
RUNTIME_PREDICTION_HORIZON_SECONDS = 3.0


# 功能：
#   检查物理标量的类型和有限性，拒绝布尔值及超出浮点表示范围的整数。
# 输入：
#   value：待用于运动包络计算的数值。
# 输出：
#   valid：是否为可表示的有限实数。
def _finite_scalar(value: object) -> bool:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    return valid


# 功能：
#   在净空关于距离满足 1-Lipschitz 条件时，用端点距离锥的交集计算区间保守下界。
#   下界不是准确碰撞时间；动态障碍的行程必须使用相对运动，不能仅用机体位移。
# 输入：
#   left：区间起点净空，单位米，可为负数。
#   right：区间终点净空，单位米，可为负数。
#   travel_m：区间相对行程，单位米。
# 输出：
#   lower_bound：整个区间的保守净空下界，单位米。
def interval_clearance_lower_bound(left: float, right: float, travel_m: float) -> float:
    if not all(_finite_scalar(v) for v in (left, right, travel_m)) or travel_m < 0:
        raise ValueError("PREDICTION_CLEARANCE_INTERVAL_INVALID")
    # 先分别除以二，避免正的有限端点相加先溢出；最终结果仍单独检查。
    lower_bound = min(left, right, left / 2 + right / 2 - travel_m / 2)
    if not math.isfinite(lower_bound):
        raise ValueError("PREDICTION_CLEARANCE_INTERVAL_INVALID")
    return lower_bound


# 功能：
#   包围预测行程、机体尺寸和净空评分范围，使几何粗筛不漏掉当前超速及残留加速影响。
#   粗筛必须检查基元包络而不是仅检查中心；查询容量不足不能截断后当作完整几何。
# 输入：
#   current_velocity_mps：当前实测三轴速度，单位米每秒。
#   maximum_speed_mps：任务允许的最大速度，单位米每秒。
#   horizon_seconds：本次预测时域，单位秒。
#   vehicle_radius_m：机体水平半径，单位米。
#   vehicle_height_m：机体高度，单位米。
#   required_clearance_m：必须保留的净空，单位米。
#   acceleration_speed_margin_mps：下一控制步可能保留的额外速度，单位米每秒。
# 输出：
#   radius：以当前机体位置为中心的保守查询半径，单位米。
def prediction_query_radius_m(
    *,
    current_velocity_mps,
    maximum_speed_mps: float,
    horizon_seconds: float,
    vehicle_radius_m: float,
    vehicle_height_m: float,
    required_clearance_m: float,
    acceleration_speed_margin_mps: float = 0.0,
) -> float:
    if (
        not isinstance(current_velocity_mps, (tuple, list))
        or len(current_velocity_mps) != 3
        or not all(_finite_scalar(v) for v in current_velocity_mps)
        or not all(
            _finite_scalar(v) and v > 0
            for v in (maximum_speed_mps, horizon_seconds, vehicle_radius_m, vehicle_height_m)
        )
        or not _finite_scalar(required_clearance_m)
        or required_clearance_m < 0
        or not _finite_scalar(acceleration_speed_margin_mps)
        or acceleration_speed_margin_mps < 0
    ):
        raise ValueError("PREDICTION_QUERY_ENVELOPE_INVALID")
    speed = max(
        math.hypot(*current_velocity_mps) + acceleration_speed_margin_mps, maximum_speed_mps
    )
    radius = (
        speed * horizon_seconds
        + math.hypot(vehicle_radius_m, vehicle_height_m / 2)
        + required_clearance_m
        + SAFETY_SCORING_CLEARANCE_BONUS_M
    )
    if not math.isfinite(radius):
        raise ValueError("PREDICTION_QUERY_ENVELOPE_INVALID")
    return radius
