"""Explicit deterministic demonstration aid, never a learned flight policy."""

import math

from .contracts import DynamicObstacleObservation, QuaternionWxyz, Vector3
from .control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from .realtime_feature_encoders import RealtimeFeatureSnapshot, world_enu_to_body


# 功能：
#   复用学生控制的原始来源期限，教师采集不延长寿命，也不重复进行整份快照的深验证。
# 输入：
#   features：当前融合编码；缺失或特征契约不符时不允许动作。
#   now_ms：本次检查的非负整数 Unix 毫秒时刻。
# 输出：
#   deadline：原控制期限，零表示不可用。
def teacher_input_deadline(features: RealtimeFeatureSnapshot | None, *, now_ms: int) -> int:
    if (
        features is None
        or type(now_ms) is not int or now_ms < 0
        or features.policy_feature_contract_sha256() != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
    ):
        return 0
    # control_deadline_unix_ms 自身先执行 fresh_at，仍会核对篡改、缺失与过期。
    return features.control_deadline_unix_ms(now_unix_ms=now_ms)


# 功能：
#   1. 为仿真示范选择观察方向；存在动态目标时优先朝向最近目标，而不随路线转头丢失它。
#   2. 陈旧目标只触发零偏航参考，不刷新观测、不删除目标，也不授权平移或绕过安全检查。
# 输入：
#   position：当前世界 ENU 米制位置。
#   goal：原路线目标；没有动态目标时沿用其观察方向。
#   obstacles：当前感知保留的动态目标，最多 256 个。
# 输出：
#   attention_target：仅用于计算偏航的参考位置，不替换导航目的地。
def teacher_observation_target(*, position: Vector3, goal: Vector3, obstacles: list[DynamicObstacleObservation]) -> Vector3:
    position = Vector3.model_validate(position.model_dump(), strict=True)
    goal = Vector3.model_validate(goal.model_dump(), strict=True)
    if type(obstacles) is not list or len(obstacles) > 256:
        raise ValueError('SIMULATION_TEACHER_OBSTACLES_INVALID')
    checked = [DynamicObstacleObservation.model_validate(item.model_dump(), strict=True)
               for item in obstacles]
    if len({item.obstacle_id for item in checked}) != len(checked):
        raise ValueError('SIMULATION_TEACHER_DUPLICATE_OBSTACLE')
    if any(item.age_seconds > .25 or item.confidence <= 0 for item in checked):
        # 这里的零偏航不等于允许悬停；原特征时效、碰撞和执行租约仍独立拒绝动作。
        attention_target = position
    elif checked:
        # 固定 ID 打破等距平局，避免输入列表顺序改变时左右反复转头。
        nearest = min(checked, key=lambda item: (
            math.dist((position.x, position.y, position.z),
                      (item.position_m.x, item.position_m.y, item.position_m.z))
            - max(item.radius_m, item.height_m / 2), item.obstacle_id))
        attention_target = nearest.position_m.model_copy(deep=True)
    else:
        attention_target = goal
    return attention_target


# 功能：
#   根据目标相对方位计算有界的教师偏航参考；它不是神经网络输出，也不直接驱动飞机。
# 输入：
#   orientation：当前机体姿态四元数。
#   position、goal：世界 ENU 米制位置及目标。
#   maximum_rate_dps：允许的偏航角速度上限，单位度／秒。
# 输出：
#   yaw_rate_dps：机体顺时针为正的偏航角速度参考。
def teacher_heading_rate(*, orientation: QuaternionWxyz, position: Vector3, goal: Vector3, maximum_rate_dps: float) -> float:
    if type(maximum_rate_dps) not in (int, float) or not 0 < maximum_rate_dps <= 45:
        raise ValueError("SIMULATION_TEACHER_YAW_LIMIT_INVALID")
    direction = world_enu_to_body(
        orientation,
        Vector3(
            x=goal.x - position.x,
            y=goal.y - position.y,
            z=0.0,
        ),
    )
    if math.hypot(direction.x, direction.y) < 0.1:
        # Near-coincident horizontal positions have no useful bearing.
        return 0.0
    # Body right is positive clockwise. A one-second proportional yaw
    # reference is only a deterministic teacher, not a fabricated neural output.
    error_deg = math.degrees(math.atan2(direction.y, direction.x))
    yaw_rate_dps = max(-maximum_rate_dps, min(maximum_rate_dps, error_deg))
    return yaw_rate_dps
