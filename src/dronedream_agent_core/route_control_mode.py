"""Explicit simulation route control, independent of learned actor availability."""

import math

PROFILE = "independent-route-v1"


# 功能：校验独立路线控制的互斥权限；不授权真实飞行或跳过感知/碰撞检查。
# 输入：显式模式与运行依赖；输出：非法组合在创建进程前拒绝。
def validate_route_control_mode(*, enabled, model_required=False, provider=None,
                                hybrid=False, teacher=False, training=False,
                                fusion=True, heading="route-tangent-relative"):
    if type(enabled) is not bool:
        raise ValueError("ROUTE_CONTROL_FLAG_INVALID")
    if not enabled:
        return
    if model_required or provider is not None or hybrid or teacher or training:
        raise ValueError("ROUTE_CONTROL_AUTHORITY_CONFLICT")
    if not fusion or heading != "route-tangent-relative":
        raise ValueError("ROUTE_CONTROL_REQUIRES_FUSED_POSE_AND_ROUTE_HEADING")


# 功能：只接纳具有独立传感器来源的当前状态；路线绝不作为定位测量。
# 输入：观测来源、原始健康结论；输出：可送入局部安全求解器，不等于允许运动。
def route_observation_eligible(source, healthy):
    return healthy is True and source == "onboard"


# 功能：
#   仅在仿真中，当三维静态路线、定位和当前真值观测仍新鲜时，允许短时视觉掉帧
#   降级为低速静态路线跟踪。该函数不授权真机，也不绕过后续全路径覆盖和净空复验。
# 输入：
#   source：当前位姿观测来源。
#   identity_ok：PX4 与 Gazebo 受控实体身份是否仍匹配。
#   stream_age_seconds：仿真真值观测年龄。
#   qualified_route_clearance_m：签名路线的最小三维净空。
#   required_clearance_m：当前机体、负载和误差需要的净空。
#   localization_covariance_m2：当前原生定位协方差上界。
#   dynamic_obstacle_count：仿真真值观测到的动态障碍数量。
# 输出：
#   eligible：可以进入后续低速几何复验时为 True。
def qualified_simulation_route_fallback_eligible(
    *,
    source,
    identity_ok,
    stream_age_seconds,
    qualified_route_clearance_m,
    required_clearance_m,
    localization_covariance_m2,
    dynamic_obstacle_count,
    maximum_stream_age_seconds=0.5,
):
    numeric = (
        stream_age_seconds,
        qualified_route_clearance_m,
        required_clearance_m,
        localization_covariance_m2,
        maximum_stream_age_seconds,
    )
    if (
        source != "simulation-ground-truth"
        or identity_ok is not True
        or type(dynamic_obstacle_count) is not int
        or dynamic_obstacle_count != 0
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in numeric)
    ):
        return False
    return bool(
        0.0 <= stream_age_seconds <= maximum_stream_age_seconds
        and maximum_stream_age_seconds > 0.0
        and required_clearance_m > 0.0
        and qualified_route_clearance_m >= required_clearance_m
        and localization_covariance_m2 >= 0.0
    )
