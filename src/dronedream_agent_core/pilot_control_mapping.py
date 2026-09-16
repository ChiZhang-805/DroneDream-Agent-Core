"""Shared joystick-to-physical mapping for inference and actuator requests."""

import math
from dataclasses import dataclass

from .contracts import NormalizedPilotControl

ACTION_RISK_FEATURE_COUNT = 4


@dataclass(frozen=True)
class PilotControlLimits:
    """Per-axis physical scales; the downstream envelope still limits vector norms."""
    horizontal_speed_mps: float
    vertical_speed_mps: float
    yaw_rate_dps: float

    # 功能：
    #   在构造时检查三轴物理尺度，先比较范围再调用浮点函数，避免超大整数转换溢出。
    # 输入：
    #   self：带水平／垂直米每秒限额和偏航度每秒限额的冻结实例。
    # 输出：
    #   None：不返回业务数据。
    def __post_init__(self) -> None:
        for value, maximum in ((self.horizontal_speed_mps, 20.0),
                               (self.vertical_speed_mps, 20.0), (self.yaw_rate_dps, 180.0)):
            if (isinstance(value, bool) or not isinstance(value, int | float)
                    or not 0 < value <= maximum or not math.isfinite(value)):
                raise ValueError("pilot control limits must be finite and within physical bounds")


# 功能：
#   1. 将连续四轴幅度映射为机体前、右、上速度和顺时针偏航角速度，应用 Harness 缩放。
#   2. 重新检查输入及总速度包络；加速度、加加速度、避障和命令期限仍由下游独立约束。
# 输入：
#   control：范围为负一到一的四轴控制契约实例。
#   limits：当前飞机绑定的各轴物理尺度。
#   harness_scale：零点一到一的缩放比例。
# 输出：
#   result：依次为前／右／上米每秒速度、顺时针度每秒偏航角速度的元组。
def physical_pilot_request(control: NormalizedPilotControl, limits: PilotControlLimits, *,
                           harness_scale: float) -> tuple[float, float, float, float]:
    if (not isinstance(control, NormalizedPilotControl)
            or not isinstance(limits, PilotControlLimits)):
        raise ValueError("pilot control and limits must use their declared contracts")
    control = NormalizedPilotControl.model_validate(control.model_dump(mode="python"), strict=True)
    limits = PilotControlLimits(limits.horizontal_speed_mps, limits.vertical_speed_mps,
                                limits.yaw_rate_dps)
    if (isinstance(harness_scale, bool) or not isinstance(harness_scale, (int, float))
            or not 0.1 <= harness_scale <= 1.0 or not math.isfinite(harness_scale)):
        raise ValueError("Harness control scale must be in [0.1, 1]")
    # 不在这里偷偷归一化斜向输入，确保风险专家与执行器看到同一个真实物理请求。
    result = (
        control.forward_axis * limits.horizontal_speed_mps * harness_scale,
        control.right_axis * limits.horizontal_speed_mps * harness_scale,
        control.up_axis * limits.vertical_speed_mps * harness_scale,
        control.yaw_axis * limits.yaw_rate_dps * harness_scale,
    )
    if math.hypot(*result[:3]) > 20 + 1e-9:
        raise ValueError("pilot control request exceeds physical velocity envelope")
    return result


# 功能：
#   按训练与部署共同的固定尺度编码将要请求的动作，使风险专家判断同一份实际控制。
# 输入：
#   control：归一化四轴控制。
#   limits：当前飞机的物理尺度。
#   harness_scale：已经由 Harness 确定的控制缩放比例。
# 输出：
#   features：前／右／上速度除以二十、顺时针偏航角速度除以一百八十的四元组。
def action_risk_features(control: NormalizedPilotControl, limits: PilotControlLimits, *,
                         harness_scale: float) -> tuple[float, float, float, float]:
    forward, right, up, yaw = physical_pilot_request(control, limits, harness_scale=harness_scale)
    features = forward / 20, right / 20, up / 20, yaw / 180
    return features
