"""Bounded, non-renewable observation deadlines; never collision permission.

Two inputs are deliberately different. A checked trajectory's residual margin
can bound additional temporal error. A raw swept-space distance additionally
needs a qualified braking lower bound; a vehicle's MAX acceleration is not one.
No pixels, point clouds, inference, I/O, iteration or wall-clock reads here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .control_timing import LOCAL_CONTROL_MAXIMUM_AGE_SECONDS


@dataclass(frozen=True)
class ObservationValidity:
    """A time-budget decision; control-eligible still requires all other safety gates."""

    disposition: Literal["control-eligible", "context-only", "reject"]
    source_observed_at_unix_ms: int
    control_deadline_unix_ms: int
    remaining_ms: int
    reason: str

    # 功能：
    #   发令前扣除剩余工作预算；临界相等即过期，背景信息或已拒绝的图不能重新获得控制资格。
    # 输入：
    #   now_unix_ms：当前 Unix 毫秒时间。
    #   reserve_ms：发令前仍需消耗的毫秒预算。
    # 输出：
    #   usable：仍在本次原始有效期内；不代替碰撞、覆盖或动力学安全判断。
    def usable_at(self, now_unix_ms: int, *, reserve_ms: int = 0) -> bool:
        # An accepted receipt is not a renewable lease. Context-only/rejected
        # receipts cannot be resurrected by asking again with a smaller reserve.
        usable = (
            type(now_unix_ms) is int
            and type(reserve_ms) is int
            and type(self.source_observed_at_unix_ms) is int
            and type(self.control_deadline_unix_ms) is int
            and self.source_observed_at_unix_ms >= 0
            and reserve_ms >= 0
            and now_unix_ms >= self.source_observed_at_unix_ms
            and self.disposition == "control-eligible"
            and now_unix_ms + reserve_ms < self.control_deadline_unix_ms
        )
        return usable


@dataclass
class SourceDeadlineLatch:
    """One latest-value stream, constant memory, no same-source rejuvenation.

    Owned by the scheduling thread. A sharper turn may shorten a source's
    deadline; slowing the turn later cannot restore the already-lost view.
    Out-of-order samples need new ingress evidence, not a restarted deadline.
    """

    latest_source_unix_ms: int = -1
    deadline_unix_ms: int = 0

    # 功能：
    #   同一来源只允许收紧截止时间；后来减速也不能恢复之前已失去的图像参考价值。
    # 输入：
    #   source_unix_ms：当前帧原始来源时间。
    #   deadline_unix_ms：当前条件下计算的截止时间。
    # 输出：
    #   deadline：锁存后不再延长的截止时间，非法或倒序来源返回零。
    def restrict(self, *, source_unix_ms: int, deadline_unix_ms: int) -> int:
        if any(type(t) is not int or t < 0 for t in (source_unix_ms, deadline_unix_ms)):
            return 0
        if source_unix_ms < self.latest_source_unix_ms:
            return 0
        if source_unix_ms > self.latest_source_unix_ms:
            self.latest_source_unix_ms = source_unix_ms
            self.deadline_unix_ms = deadline_unix_ms
        else:
            self.deadline_unix_ms = min(self.deadline_unix_ms, deadline_unix_ms)
        deadline = self.deadline_unix_ms
        return deadline


# 功能：
#   接纳可表示的非负有限物理量，拒绝布尔值和超出浮点范围的大整数。
# 输入：
#   value：待检查数值。
# 输出：
#   valid：数值满足范围及类型要求。
def _nonnegative(value: object) -> bool:
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    return valid


# 功能：
#   1. 常数时间求解平移、转动与不确定性下的保守帧龄预算，不读取图像或调用模型。
#   2. 已检验轨迹余量使用 (v+u)t+a*t²/2；原始扫掠距离还扣除本机及障碍在制动期间的行程。
#   3. 制动减速度必须是当前质量、姿态和环境下已验证的下界，不能用最大加速度代替。
#   4. 处理、排队和执行器响应共同消耗原始来源时间预算，向更早毫秒取整；这只是附加否决门。
# 输入：
#   source_observed_at_unix_ms：原始采集时间，不是接收或处理完成时间。
#   now_unix_ms：本次评估时间。
#   inherited_deadline_unix_ms：上游不可延长的截止时间。
#   clearance_margin_m：已验证轨迹余量或原始扫掠空闲距离。
#   ego_speed_bound_mps、obstacle_speed_bound_mps：全时段本机和障碍速度上界。
#   acceleration_bound_mps2：全时段可能加速度上界。
#   uncertainty_margin_m：定位、几何及模型误差占用的距离余量。
#   downstream_reserve_ms：后续处理和发令保留时间。
#   angular_speed_bound_rad_s、angular_error_budget_rad：角速度上界与可承受视角变化。
#   margin_kind：已验证轨迹余量或原始扫掠距离，二者不能混用。
#   minimum_braking_deceleration_mps2：原始扫掠距离模式必须具备的已验证制动下界。
#   actuator_response_ms：执行器响应及制动力建立所需时间。
# 输出：
#   validity：控制时效、仅背景参考或拒绝的判定，以及来源、截止时间和原因。
def observation_validity(
    *,
    source_observed_at_unix_ms: int,
    now_unix_ms: int,
    inherited_deadline_unix_ms: int,
    clearance_margin_m: float,
    ego_speed_bound_mps: float,
    obstacle_speed_bound_mps: float,
    acceleration_bound_mps2: float,
    uncertainty_margin_m: float,
    downstream_reserve_ms: int,
    angular_speed_bound_rad_s: float = 0.0,
    angular_error_budget_rad: float = math.radians(10.0),
    margin_kind: Literal[
        "checked-trajectory-slack", "swept-free-distance"
    ] = "checked-trajectory-slack",
    minimum_braking_deceleration_mps2: float | None = None,
    actuator_response_ms: int = 0,
) -> ObservationValidity:
    source = source_observed_at_unix_ms

    # 功能：
    #   保留原始采集身份并生成不会续期的判定，即使只能作为背景信息也不改写时间。
    # 输入：
    #   disposition：当前用途分类。
    #   deadline：已收紧的截止时间。
    #   reason：判定原因。
    # 输出：
    #   validity：不可变的时效记录。
    def result(disposition, deadline, reason):
        validity = ObservationValidity(
            disposition, source, deadline, max(0, deadline - now_unix_ms), reason
        )
        return validity

    if (
        any(
            type(t) is not int or t < 0
            for t in (
                source,
                now_unix_ms,
                inherited_deadline_unix_ms,
                downstream_reserve_ms,
                actuator_response_ms,
            )
        )
        or source > now_unix_ms
    ):
        return ObservationValidity("reject", 0, 0, 0, "invalid-source-clock")
    numbers = (
        clearance_margin_m,
        ego_speed_bound_mps,
        obstacle_speed_bound_mps,
        acceleration_bound_mps2,
        uncertainty_margin_m,
        angular_speed_bound_rad_s,
        angular_error_budget_rad,
    )
    if not all(_nonnegative(n) for n in numbers) or angular_error_budget_rad <= 0:
        return result("reject", 0, "invalid-motion-or-uncertainty-bound")
    if margin_kind not in ("checked-trajectory-slack", "swept-free-distance"):
        return result("reject", 0, "invalid-margin-kind")
    hard_ms = math.floor(LOCAL_CONTROL_MAXIMUM_AGE_SECONDS * 1000)
    deadline = min(source + hard_ms, inherited_deadline_unix_ms)
    if deadline <= now_unix_ms:
        return result("context-only", deadline, "source-deadline-exhausted")

    # Normalize validated JSON integers before products. Python's arbitrary-size
    # integer square can otherwise throw during the later float division; IEEE
    # overflow instead reaches the explicit finite-coefficient rejection below.
    v, u, a = map(float, (ego_speed_bound_mps, obstacle_speed_bound_mps, acceleration_bound_mps2))
    margin = float(clearance_margin_m) - float(uncertainty_margin_m)
    linear, quadratic = v + u, a / 2
    if margin_kind == "swept-free-distance":
        b = minimum_braking_deceleration_mps2
        if not _nonnegative(b) or b == 0:
            return result("context-only", 0, "qualified-braking-bound-unavailable")
        b = float(b)
        margin -= v * v / (2 * b) + u * v / b
        linear += a * (v + u) / b
        quadratic += a * a / (2 * b)
    elif minimum_braking_deceleration_mps2 is not None:
        return result("reject", 0, "ambiguous-margin-and-braking-bound")
    if not all(math.isfinite(n) for n in (margin, linear, quadratic)):
        return result("reject", 0, "motion-bound-overflow")
    if margin <= 0:
        return result("context-only", 0, "no-temporal-clearance-margin")

    # Stable positive root avoids cancellation for tiny acceleration. First
    # compare at the hard horizon so huge finite distances cannot overflow a
    # discriminant unnecessarily. hypot avoids squaring the linear coefficient.
    horizon = hard_ms / 1000
    consumed = linear * horizon + quadratic * horizon * horizon
    if not math.isfinite(consumed):
        return result("reject", 0, "motion-bound-overflow")
    if consumed > margin:
        if quadratic:
            root = math.hypot(linear, 2 * math.sqrt(quadratic) * math.sqrt(margin))
            horizon = margin / (linear / 2 + root / 2)
        else:
            horizon = margin / linear
    if angular_speed_bound_rad_s:
        horizon = min(horizon, angular_error_budget_rad / angular_speed_bound_rad_s)
    deadline = min(deadline, source + math.floor(horizon * 1000))
    if now_unix_ms + downstream_reserve_ms + actuator_response_ms >= deadline:
        return result("context-only", deadline, "processing-or-response-budget-exhausted")
    return result("control-eligible", deadline, "within-adaptive-age-budget")


# 功能：
#   发布只消耗已评估的有效期，慢计算结束后不能重新启动计时；截止边界相等即失效。
# 输入：
#   evaluated_deadline_unix_ms：既有评估的截止时间。
#   published_at_unix_ms：实际发布时刻。
#   reserve_ms：后续所需预算。
# 输出：
#   deadline：仍足够时返回原截止时间，否则为 None。
def publication_deadline(
    *, evaluated_deadline_unix_ms: int, published_at_unix_ms: int, reserve_ms: int
) -> int | None:
    if any(
        type(t) is not int or t < 0
        for t in (evaluated_deadline_unix_ms, published_at_unix_ms, reserve_ms)
    ):
        raise ValueError("invalid publication clock or reserve")
    deadline = (
        evaluated_deadline_unix_ms
        if published_at_unix_ms + reserve_ms < evaluated_deadline_unix_ms
        else None
    )
    return deadline


# 功能：
#   由验证过的仿真场景采集时间形成图像硬截止；真机必须新增验证过的曝光时钟适配，不能冒用接收时间。
# 输入：
#   image：携原始场景和主机接收时间的图像元数据。
#   now_unix_ms：当前 Unix 毫秒时间。
# 输出：
#   deadline：有效原始截止时间，来源不明、非法或已过期时为零。
def image_control_deadline(image: Mapping[str, object], *, now_unix_ms: int) -> int:
    if not isinstance(image, Mapping) or type(now_unix_ms) is not int or now_unix_ms < 0:
        return 0
    if image.get("timestamp_basis") != "simulation-scene-capture":
        return 0
    source_ns, received_ns = image.get("scene_source_unix_ns"), image.get("host_received_unix_ns")
    if (
        type(source_ns) is not int
        or type(received_ns) is not int
        or not 0 <= source_ns <= received_ns <= now_unix_ms * 1_000_000 + 999_999
    ):
        return 0
    deadline = source_ns // 1_000_000 + math.floor(LOCAL_CONTROL_MAXIMUM_AGE_SECONDS * 1000)
    return deadline if deadline > now_unix_ms else 0
