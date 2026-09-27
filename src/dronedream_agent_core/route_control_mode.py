"""Explicit simulation route control, independent of learned actor availability."""

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
