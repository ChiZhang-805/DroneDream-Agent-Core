"""Canonical host-time budgets for local joystick control, not cloud planning.

These are enforceable design limits, not claimed measurements. Runtime and
qualification import the same values; a slow machine must brake rather than
silently lengthen an action's lifetime to a cloud request timeout.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass

CONTINUOUS_CONTROL_MODE = "normalized-body-velocity"
LOCAL_CONTROL_PERIOD_SECONDS = 0.05
LOCAL_CONTROL_MAXIMUM_AGE_SECONDS = 0.25
LOCAL_CONTROL_JOINT_P99_TARGET_SECONDS = 0.10
# Producer admission reserves the future control period plus transport. Once
# the executor has reached its dispatch tick, only transport remains; do not
# count that same scheduling wait twice. Pre-send and actual-acceptance checks
# still enforce the original input deadline, without renewing any lease.
LOCAL_TRANSPORT_BUDGET_MS = 20
LOCAL_DISPATCH_RESERVE_MS = round(LOCAL_CONTROL_PERIOD_SECONDS * 1000) + LOCAL_TRANSPORT_BUDGET_MS


# 功能：
#   为本地策略及仿真训练统一选择连续控制证据门槛，不把云端建议视为运动授权。
# 输入：
#   provider：提供者标识；未知或非法标识不匹配本地控制提供者。
#   authority_required：是否明确要求模型拥有运动控制权。
# 输出：
#   required：是否必须验证连续控制证据。
def continuous_control_evidence_required(provider: str | None, authority_required: bool) -> bool:
    required = (authority_required is True and type(provider) is str
                and provider in {"local-policy", "simulation-training"})
    return required


# 功能：
#   验证成功控制之间的最大间隔；该必要条件不代表达到 20 Hz 或完成飞行任务。
#   单条指令期限及证据归属仍须单独验收，本函数不会续期。
# 输入：
#   verification：控制证据检查结果。
#   maximum_gap_seconds：相邻成功控制的最大间隔，单位秒。
# 输出：
#   bounded：证据已通过且间隔位于允许范围内。
def continuous_control_cadence_bounded(
    verification: dict, maximum_gap_seconds: float | None,
) -> bool:
    # 先做有界比较，避免巨大整数在转换为浮点数时溢出；NaN 也不会通过区间检查。
    bounded = (isinstance(verification, dict) and verification.get("accepted") is True
               and type(maximum_gap_seconds) in (int, float)
               and 0 <= maximum_gap_seconds <= LOCAL_CONTROL_MAXIMUM_AGE_SECONDS)
    return bounded


@dataclass(frozen=True)
class ControlTiming:
    """Effective worker configuration; limits are not measured performance."""
    sensor_tick_hz: float
    decision_period_seconds: float
    maximum_decision_age_seconds: float
    control_lease_seconds: float

    # 功能：
    #   导出生效的工作线程时序配置；配置上限不冒充实测性能。
    # 输入：
    #   self：已解析的时序配置。
    # 输出：
    #   values：独立的配置字典。
    def as_dict(self) -> dict[str, float]:
        values = asdict(self)
        return values


# 功能：
#   按工作线程实际生效配置验收频率及期限，拒绝云端接管运动或非法回执。
# 输入：
#   evidence：包含控制模式、云端接管标志及生效配置的回执。
# 输出：
#   bounded：实际配置是否满足连续控制设计门槛。
def continuous_timing_is_bounded(evidence: object) -> bool:
    if not isinstance(evidence, dict) or (
        evidence.get("control_output_mode") != CONTINUOUS_CONTROL_MODE
        or evidence.get("cloud_fallback_controls_motion") is not False
    ):
        return False
    effective = evidence.get("effective")
    if not isinstance(effective, dict):
        return False
    try:
        timing = ControlTiming(**effective)
        values = (timing.sensor_tick_hz, timing.decision_period_seconds,
                  timing.maximum_decision_age_seconds, timing.control_lease_seconds)
        if any(type(v) not in (int, float) or not 0 < v <= sys.float_info.max for v in values):
            return False
        bounded = (
            timing.decision_period_seconds <= LOCAL_CONTROL_PERIOD_SECONDS
            and timing.sensor_tick_hz >= 1 / timing.decision_period_seconds
            and timing.maximum_decision_age_seconds <= LOCAL_CONTROL_MAXIMUM_AGE_SECONDS
            and timing.control_lease_seconds <= LOCAL_CONTROL_MAXIMUM_AGE_SECONDS
        )
        return bounded
    except (TypeError, ValueError):
        return False


# 功能：
#   1. 为连续控制限制本地周期及授权期限，不随云端超时延长。
#   2. 仅在明确选择兼容模式时保留粗粒度建议节奏，该模式不能替代连续控制验收。
#   3. 拒绝非法输入及溢出的派生配置，避免发布无限预算。
# 输入：
#   output_mode：明确选择的控制输出模式。
#   sensor_tick_hz：请求的传感器处理频率，单位 Hz。
#   decision_period_seconds：请求的决策周期，单位秒。
#   provider_timeout_seconds：提供者超时，单位秒。
# 输出：
#   timing：解析后的有效时序配置。
def resolve_control_timing(
    *, output_mode: str, sensor_tick_hz: float, decision_period_seconds: float,
    provider_timeout_seconds: float,
) -> ControlTiming:
    values = (sensor_tick_hz, decision_period_seconds, provider_timeout_seconds)
    if any(type(v) not in (int, float) or not 0 < v <= sys.float_info.max for v in values):
        raise ValueError("control timing values must be finite and positive")
    if output_mode == CONTINUOUS_CONTROL_MODE:
        period = min(decision_period_seconds, LOCAL_CONTROL_PERIOD_SECONDS)
        timing = ControlTiming(
            sensor_tick_hz=max(sensor_tick_hz, 1.0 / period),
            decision_period_seconds=period,
            maximum_decision_age_seconds=LOCAL_CONTROL_MAXIMUM_AGE_SECONDS,
            control_lease_seconds=LOCAL_CONTROL_MAXIMUM_AGE_SECONDS,
        )
    elif output_mode == "legacy-candidate-selection":
        # 保留仍有明确调用的非执行型建议模式；其长预算不能通过连续控制验收。
        timing = ControlTiming(
            sensor_tick_hz=sensor_tick_hz,
            decision_period_seconds=decision_period_seconds,
            maximum_decision_age_seconds=max(provider_timeout_seconds + 3.0,
                                             decision_period_seconds * 3.0),
            control_lease_seconds=min(15.0, max(6.0, decision_period_seconds * 6.0,
                                               provider_timeout_seconds + 2.0)),
        )
    else:
        raise ValueError("unsupported control timing mode")
    # 有限输入的倒数／倍乘仍可能溢出，必须检查计算结果而不只是输入。
    if not all(math.isfinite(v) for v in timing.as_dict().values()):
        raise ValueError("derived control timing must be finite")
    return timing
