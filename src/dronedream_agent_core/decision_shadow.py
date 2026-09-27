"""Offline stage decisions; deliberately has no execution or transport dependency."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

ACTIONS = ("follow_route", "slow_down", "wait", "replan", "request_observation")
DESCRIPTIONS = {
    "follow_route": "Continue on the independently verified clear route.",
    "slow_down": "Continue at reduced speed when the route is clear but caution is needed.",
    "wait": "Wait for a temporary crossing obstacle; preserve the mission.",
    "replan": "Request a new local path around a persistent blockage; do not execute it yet.",
    "request_observation": "Request fresh or missing critical observations; preserve the mission.",
}


@dataclass(frozen=True)
class StageState:
    """Only current causal facts enter the model; labels/outcomes remain separate."""

    pose_fresh: bool | None
    geometry_fresh: bool | None
    route_verified: bool | None
    dynamic_crossing: bool | None
    persistent_blockage: bool | None
    caution_required: bool | None
    optional_media_available: bool | None = None
    speed_mps: float | None = None
    observation_age_ms: float | None = None

    # 功能：拒绝把字符串、NaN、缺失信息等误解释为有效传感器读数。
    # 输入：只读阶段状态；None 明确表示未知，不转换成安全值。
    # 输出：合法实例；数据类型或范围不合法时抛出 ValueError。
    def __post_init__(self):
        for key, value in asdict(self).items():
            if key in {"speed_mps", "observation_age_ms"}:
                if value is not None and (type(value) not in (int, float)
                        or not math.isfinite(value) or value < 0):
                    raise ValueError(f"STAGE_NUMERIC_INVALID:{key}")
            elif value is not None and type(value) is not bool:
                raise ValueError(f"STAGE_BOOLEAN_INVALID:{key}")


# 功能：提供可审计的阶段规则基线；规则不授予飞行权限，也不伪装成模型概率。
# 输入：当前感知事实；可选媒体丢失不会改变阶段决策。
# 输出：建议行为；等待/补充观测均不等同于放弃任务。
def rule_decision(state: StageState) -> str:
    if state.pose_fresh is not True or state.geometry_fresh is not True:
        return "request_observation"
    if state.persistent_blockage is True:
        return "replan"
    if state.dynamic_crossing is True:
        return "wait"
    if (state.route_verified is not True or state.dynamic_crossing is None
            or state.persistent_blockage is None):
        return "request_observation"
    if state.caution_required is not False:
        return "slow_down"
    return "follow_route"


# 功能：编码固定宽度的数值特征并保留缺失掩码，供小分类器和可重复对比使用。
# 输入：不含标签与执行结果的状态。
# 输出：特征列表；数值字段采用单调有界缩放，不从验证集拟合归一化参数。
def numeric_features(state: StageState) -> list[float]:
    features = []
    for value in asdict(state).values():
        features.extend((0.0 if value is None else float(value) / (1 + abs(float(value))),
                         float(value is not None)))
    return features


# 功能：生成固定选项的文字接口，避免把路线、日志或任务原文当作系统指令。
# 输入：仅包含类型约束后的短状态。
# 输出：Laya 的状态字符串与问题契约；不请求内部思维链。
def laya_request(state: StageState):
    instructions = ("Suggest one navigation stage, not motor commands. Unknown means unobserved. "
            "Missing optional media alone must not stop progress. Require fresh pose and "
            "geometry. Persistent blockage requires replanning; crossing obstacles require "
            "waiting. Unknown critical route/hazard facts require observation. "
            "A clear route with caution requires reduced speed. Otherwise follow the route. "
            "All outputs are offline suggestions.")
    # 将规则放在问题指令，将当前事实放在 state；不把所有候选场景的描述混入观测。
    # 不写出规则答案，也不将 None 翻译成 false。
    meanings = {
        "pose_fresh": "The drone position estimate is fresh",
        "geometry_fresh": "Obstacle distances and local geometry are fresh",
        "route_verified": "The current route segment has been verified",
        "dynamic_crossing": "A moving obstacle is crossing the flight path",
        "persistent_blockage": "A persistent obstacle blocks the flight path",
        "caution_required": "The current conditions require reduced speed",
        "optional_media_available": "Optional live-view video is available",
    }
    facts = []
    for field, value in asdict(state).items():
        if field in meanings:
            facts.append(meanings[field] + ": " + ("unknown" if value is None else "yes" if value else "no") + ".")
        elif value is not None:
            facts.append(f"{field}: {value}.")
    return "\n".join(facts), {"stage": {"type": "choice", "instructions": instructions,
                            "criteria": dict(DESCRIPTIONS)}}


# 功能：校验模型原始概率；不通过屏蔽和重新归一化制造高置信度。
# 输入：五个固定动作的完整概率分布，允许四位小数造成的小舍入误差。
# 输出：新分布副本；缺项、额外项、非法数字或总和异常时抛错。
def validate_probabilities(probabilities: dict) -> dict[str, float]:
    if type(probabilities) is not dict or set(probabilities) != set(ACTIONS):
        raise ValueError("STAGE_PROBABILITY_KEYS_INVALID")
    values = list(probabilities.values())
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           for v in values) or abs(sum(values) - 1) > .0005:
        raise ValueError("STAGE_PROBABILITY_VALUES_INVALID")
    return {k: float(probabilities[k]) for k in ACTIONS}


# 功能：离线记录选择、原始差距及可执行性；不输出任何飞控许可或执行消息。
# 输入：候选概率、独立提供的允许动作；None 表示没有可执行性证据。
# 输出：影子记录，建议可能为空；概率高也不能创造独立可执行性证据。
def shadow_suggestion(probabilities: dict, allowed_actions: tuple[str, ...] | None = None):
    probabilities = validate_probabilities(probabilities)
    if allowed_actions is not None and (type(allowed_actions) is not tuple
            or len(set(allowed_actions)) != len(allowed_actions)
            or any(a not in ACTIONS for a in allowed_actions)):
        raise ValueError("STAGE_ALLOWED_ACTIONS_INVALID")
    ranked = sorted(ACTIONS, key=lambda a: (-probabilities[a], ACTIONS.index(a)))
    top = ranked[0]
    reason = "uncalibrated-shadow-only"
    if allowed_actions is None:
        reason = "independent-admissibility-unavailable"
    elif top not in allowed_actions:
        reason = "top-choice-inadmissible"
    elif probabilities[top] < .8 or probabilities[top] - probabilities[ranked[1]] < .2:
        reason = "ambiguous-distribution"
    return {"raw_choice": top, "raw_probability": probabilities[top],
            "margin": probabilities[top] - probabilities[ranked[1]],
            "reason": reason, "execution_authority": False,
            "probabilities": probabilities}


# 功能：按完整任务组固定划分数据，避免相邻帧跨训练/验证泄漏。
# 输入：稳定任务组身份；相同路线重复任务应由调用方使用同一个组身份。
# 输出：train/calibration/test，约 60/20/20；这不是标签质量认证。
def group_split(group_id: str) -> str:
    if type(group_id) is not str or not group_id.strip():
        raise ValueError("STAGE_GROUP_REQUIRED")
    bucket = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 6 else "calibration" if bucket < 8 else "test"
