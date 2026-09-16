"""Controller-neutral rules for short-lived local-model control authority.

No provider, filesystem, simulator or clock access belongs here. Adapters pass
their measured times and headings explicitly, so replay uses the same rules as
live execution. A numerical zero is a command, never an implicit handover.
"""

from __future__ import annotations

import sys

from .control_timing import LOCAL_CONTROL_MAXIMUM_AGE_SECONDS


# 功能：
#   使用新鲜原生观测的航向停止转动，不用粗略路径朝向替代缺失或过期的姿态。
# 输入：
#   yaw_deg：原生观测航向，单位为度。
#   sample_age_seconds：适配器测得的单调时钟源年龄秒数。
# 输出：
#   heading：规范到 [-180, 180) 度的保持航向。
def measured_hold_heading(*, yaw_deg: float, sample_age_seconds: float) -> float:
    values = (yaw_deg, sample_age_seconds)
    if any(type(v) not in (int, float) or not -sys.float_info.max <= v <= sys.float_info.max
           for v in values):
        raise ValueError("HOLD_HEADING_REQUIRES_FINITE_NATIVE_ATTITUDE")
    if not 0 <= sample_age_seconds <= LOCAL_CONTROL_MAXIMUM_AGE_SECONDS:
        raise ValueError("HOLD_HEADING_NATIVE_ATTITUDE_IS_NOT_FRESH")
    # 先取模再平移，避免巨大累计角度吞掉 180 度的平移量。
    heading = (yaw_deg % 360.0 + 180.0) % 360.0 - 180.0
    return heading


# 功能：
#   按 NED 顺时针为正的约定积分模型偏航速度，限制转速和单次积分时长，不补偿长停顿。
# 输入：
#   previous_heading_deg：上次已授权航向角度。
#   requested_rate_dps：模型请求的有符号偏航速度，单位为度每秒。
#   maximum_rate_dps：允许的偏航速度绝对值上限。
#   step_seconds：本次积分时间，最多 0.25 秒。
# 输出：
#   heading：积分后规范到 [-180, 180) 度的航向。
def integrate_model_yaw(
    *, previous_heading_deg: float, requested_rate_dps: float,
    maximum_rate_dps: float, step_seconds: float,
) -> float:
    values = (previous_heading_deg, requested_rate_dps, maximum_rate_dps, step_seconds)
    if any(type(value) not in (int, float)
           or not -sys.float_info.max <= value <= sys.float_info.max for value in values):
        raise ValueError("yaw integration requires finite values")
    if not 0.0 < maximum_rate_dps <= 180.0:
        raise ValueError("yaw rate limit must be in (0, 180] degrees per second")
    if not 0.0 < step_seconds <= 0.25:
        raise ValueError("yaw integration step must be in (0, 0.25] seconds")
    rate = max(-maximum_rate_dps, min(maximum_rate_dps, requested_rate_dps))
    # 已有航向先化为一个周期，再叠加小控制量，避免小转角被大数舍入丢失。
    heading = (previous_heading_deg % 360.0 + rate * step_seconds + 180.0) % 360.0 - 180.0
    return heading


# 功能：
#   将命令有效期裁剪到原始决策期限，更新几何只能否决旧动作，不能刷新其授权。
# 输入：
#   now_unix_ms：当前 UNIX 毫秒时刻。
#   authority_deadline_unix_ms：原始决策不可延长的截止时刻。
#   requested_validity_ms：请求的命令有效毫秒数。
#   minimum_publication_window_ms：发送所需的最小有效窗口。
# 输出：
#   remaining：可用有效期毫秒数；不足最小窗口时为 None。
def remaining_control_validity_ms(
    *, now_unix_ms: int, authority_deadline_unix_ms: int, requested_validity_ms: int,
    minimum_publication_window_ms: int = 50,
) -> int | None:
    if any(type(value) is not int or not 0 <= value < 2**63
           for value in (now_unix_ms, authority_deadline_unix_ms)):
        raise ValueError("control timestamps must be nonnegative signed-64 integers")
    if (type(requested_validity_ms) is not int or type(minimum_publication_window_ms) is not int
            or not 20 <= minimum_publication_window_ms <= requested_validity_ms <= 2_000):
        raise ValueError("control publication windows are invalid")
    remaining = min(requested_validity_ms, authority_deadline_unix_ms - now_unix_ms)
    if remaining < minimum_publication_window_ms:
        remaining = None
    return remaining


# 功能：
#   对已成功发送的控制作来源归因，区分模型包络、安全替代和路径控制，不冒充任务成功。
# 输入：
#   model_authorized：该命令是否确实具备模型授权。
#   action：实际应用的动作名称。
#   control_source：适配器记录的实际控制来源。
#   heading_assisted：模型命令是否使用路径航向辅助。
# 输出：
#   category：本次已发送动作的来源分类。
def control_application_category(
    *, model_authorized: bool, action: str, control_source: str, heading_assisted: bool,
) -> str:
    if (type(model_authorized) is not bool or type(heading_assisted) is not bool
            or any(type(value) is not str or not value.strip() or len(value) > 128
                   for value in (action, control_source))):
        raise ValueError("CONTROL_APPLICATION_EVIDENCE_INVALID")
    # 模型包络包含确定性限速、限加速度，不能宣称原始神经网络向量原样到达执行器。
    if not model_authorized:
        category = "without-model-authority"
    elif action in {"hold", "replan"} or control_source == "deterministic-brake":
        category = "safety-brake-or-hold"
    elif control_source == "deterministic-safety-override":
        category = "safety-direction-override"
    elif control_source == "local-model-body-control":
        category = "model-enveloped-heading-assisted" if heading_assisted else "model-enveloped"
    elif control_source == "route-target":
        category = "route-derived"
    else:
        raise ValueError("unrecognized applied control source")
    return category
