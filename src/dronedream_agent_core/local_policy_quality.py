"""Shared evidence and acceptance rules for training and ONNX admission.

Metrics with no class support are not evidence of success. The numerical
recalls remain serializable as zero in that case; their explicit denominators
distinguish unavailable evidence from a measured failure. Training and runtime
admission must not implement different definitions of a qualified pilot.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field, model_serializer, model_validator

from .contracts import StrictModel
from .training.risk_admission_evidence import (
    ActionRiskAdmissionEvidence,
    action_risk_evidence_issues,
    action_risk_summary,
)

PILOT_AXES = ("forward", "right", "up", "yaw")
AXIS_ACTIVITY_THRESHOLD = 0.05


class PilotAxisEvidence(StrictModel):
    sample_count: int = Field(ge=1)
    positive_sample_count: int = Field(ge=0)
    negative_sample_count: int = Field(ge=0)
    neutral_sample_count: int = Field(ge=0)
    mean_absolute_error: float = Field(ge=0.0, le=2.0)
    active_mean_absolute_error: float | None = Field(default=None, ge=0.0, le=2.0)
    maximum_absolute_error: float = Field(ge=0.0, le=2.0)

    # 功能：
    #   核对各方向样本数之和与总量一致，活动误差必须有非中性样本支撑。
    # 输入：
    #   self：解析后的单轴控制证据。
    # 输出：
    #   self：支持数量与误差存在性一致的证据。
    @model_validator(mode="after")
    def validate_support(self) -> PilotAxisEvidence:
        active = self.positive_sample_count + self.negative_sample_count
        if active + self.neutral_sample_count != self.sample_count:
            raise ValueError("pilot axis support counts do not add up")
        if (active == 0) != (self.active_mean_absolute_error is None):
            raise ValueError("active axis error must have supporting samples")
        return self


class LocalRiskCriticMetrics(StrictModel):
    """Risk-head quality with explicit positive and negative support."""

    sample_count: int = Field(ge=1)
    risky_sample_count: int = Field(default=0, ge=0)
    safe_sample_count: int = Field(default=0, ge=0)
    risk_hold_recall: float = Field(ge=0.0, le=1.0)
    safe_motion_recall: float = Field(ge=0.0, le=1.0)
    risk_mean_absolute_error: float = Field(ge=0.0, le=1.0)
    mean_binary_cross_entropy: float = Field(ge=0.0)


class LocalPolicyTrainingMetrics(StrictModel):
    action_risk_evidence: ActionRiskAdmissionEvidence | None = None
    sample_count: int = Field(ge=1)
    motion_sample_count: int = Field(default=0, ge=0)
    non_motion_sample_count: int = Field(default=0, ge=0)
    authorized_motion_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    non_motion_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    risky_sample_count: int = Field(default=0, ge=0)
    safe_sample_count: int = Field(default=0, ge=0)
    candidate_sample_count: int = Field(default=0, ge=0)
    action_accuracy: float = Field(ge=0.0, le=1.0)
    motion_authorization_accuracy: float = Field(ge=0.0, le=1.0)
    candidate_selection_accuracy: float = Field(ge=0.0, le=1.0)
    risk_hold_recall: float = Field(ge=0.0, le=1.0)
    safe_motion_recall: float = Field(ge=0.0, le=1.0)
    risk_mean_absolute_error: float = Field(ge=0.0, le=1.0)
    mean_cross_entropy: float = Field(ge=0.0)
    pilot_control_mean_absolute_error: float | None = Field(default=None, ge=0.0, le=2.0)
    pilot_axis_evidence: dict[str, PilotAxisEvidence] = Field(default_factory=dict, max_length=4)

    # 功能：
    #   未使用独立动作风险验收的旧回执保持原字段身份，新回执显式保留不同分母的风险证据。
    # 输入：
    #   self、handler：当前指标与标准序列化器。
    # 输出：
    #   payload：不会给旧来源补造风险证据的指标对象。
    @model_serializer(mode='wrap')
    def serialize_action_risk_evidence(self, handler):
        payload = handler(self)
        if self.action_risk_evidence is None:
            payload.pop('action_risk_evidence', None)
        return payload


# 功能：
#   1. 分别统计前后、左右、升降、偏航四轴的方向覆盖及控制误差。
#   2. 活动样本单独计算误差，不能用大量零动作稀释真实操纵误差；输入应已排除非运动占位样本。
# 输入：
#   predictions：每个运动样本对应的四轴归一化预测值。
#   targets：同顺序、同形状的目标控制值。
# 输出：
#   evidence：按轴名称索引的支持数量及误差统计；没有目标样本时为空字典。
def summarize_pilot_axes(
    predictions: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
) -> dict[str, PilotAxisEvidence]:
    import math

    if len(predictions) != len(targets):
        raise ValueError("pilot predictions and targets must have the same sample count")
    evidence = {}
    if not len(targets):
        return evidence
    for rows in (predictions, targets):
        for row in rows:
            if len(row) != 4 or any(not math.isfinite(v) or abs(v) > 1.0 for v in row):
                raise ValueError("pilot evidence requires four finite normalized axes")
    for index, axis in enumerate(PILOT_AXES):
        errors = [
            abs(float(p[index]) - float(t[index]))
            for p, t in zip(predictions, targets, strict=True)
        ]
        positive = sum(t[index] > AXIS_ACTIVITY_THRESHOLD for t in targets)
        negative = sum(t[index] < -AXIS_ACTIVITY_THRESHOLD for t in targets)
        active_errors = [
            error
            for error, target in zip(errors, targets, strict=True)
            if abs(target[index]) > AXIS_ACTIVITY_THRESHOLD
        ]
        evidence[axis] = PilotAxisEvidence(
            sample_count=len(targets),
            positive_sample_count=int(positive),
            negative_sample_count=int(negative),
            neutral_sample_count=len(targets) - int(positive) - int(negative),
            mean_absolute_error=sum(errors) / len(errors),
            active_mean_absolute_error=(
                sum(active_errors) / len(active_errors) if active_errors else None
            ),
            maximum_absolute_error=max(errors),
        )
    return evidence


# 功能：
#   1. 重新校验模型质量证据，统一训练与运行包准入的判定，收集所有不满足项。
#   2. 连续控制只免去候选排序检查，仍要求危险、非危险、运动、非运动以及四轴双向样本支持。
# 输入：
#   metrics：训练或独立准入验证产生的统计指标。
#   continuous_control：是否验收四轴连续控制模型，必须为布尔值。
#   maximum_pilot_control_mae：允许的控制平均绝对误差上限。
#   minimum_risk_class_samples：每个风险类别最少样本数。
#   minimum_axis_direction_samples：每轴每个活动方向最少样本数。
# 输出：
#   issues：全部质量问题代码；空列表只表示这些统计门限通过，不等于闭环飞行验收。
def navigation_quality_issues(
    metrics: LocalPolicyTrainingMetrics,
    *,
    continuous_control: bool,
    maximum_pilot_control_mae: float = 0.2,
    minimum_risk_class_samples: int = 20,
    minimum_axis_direction_samples: int = 5,
) -> list[str]:
    if type(continuous_control) is not bool:
        raise ValueError("continuous control mode must be boolean")
    if (type(maximum_pilot_control_mae) not in (int, float)
            or not 0.0 < maximum_pilot_control_mae <= 2.0):
        raise ValueError("pilot error threshold must be in (0, 2]")
    if (type(minimum_risk_class_samples) is not int
            or type(minimum_axis_direction_samples) is not int
            or minimum_risk_class_samples < 1 or minimum_axis_direction_samples < 1):
        raise ValueError("evidence thresholds must require positive sample counts")
    # 类型标注不能保证对象仍合法；model_copy 或嵌套字典原地修改可能绕过构造器校验。
    metrics = LocalPolicyTrainingMetrics.model_validate(
        metrics.model_dump(mode="python"), strict=True,
    )
    issues: list[str] = []
    if metrics.action_risk_evidence is None:
        if metrics.risky_sample_count + metrics.safe_sample_count != metrics.sample_count:
            issues.append("RISK_CLASS_SUPPORT_INCOMPLETE")
    else:
        # 行为样本与风险探针各有自己的分母；仍执行同等或更严格的逐区域风险门槛。
        expected = action_risk_summary(metrics.action_risk_evidence)
        if any(getattr(metrics, name) != value for name, value in expected.items()):
            issues.append('ACTION_RISK_SUMMARY_MISMATCH')
        issues.extend(action_risk_evidence_issues(metrics.action_risk_evidence))
    if metrics.risky_sample_count < minimum_risk_class_samples:
        issues.append("INSUFFICIENT_RISKY_SAMPLES")
    if metrics.safe_sample_count < minimum_risk_class_samples:
        issues.append("INSUFFICIENT_SAFE_SAMPLES")
    if metrics.motion_authorization_accuracy < 0.95:
        issues.append("MOTION_AUTHORIZATION_ACCURACY_TOO_LOW")
    if (
        metrics.motion_sample_count + metrics.non_motion_sample_count != metrics.sample_count
        or min(metrics.motion_sample_count, metrics.non_motion_sample_count) == 0
    ):
        issues.append("MOTION_CLASS_SUPPORT_INCOMPLETE")
    if metrics.authorized_motion_recall < 0.95:
        issues.append("AUTHORIZED_MOTION_RECALL_TOO_LOW")
    if metrics.non_motion_recall < 0.95:
        issues.append("NON_MOTION_RECALL_TOO_LOW")
    if not continuous_control:
        if metrics.candidate_sample_count == 0:
            issues.append("CANDIDATE_EVIDENCE_MISSING")
        if metrics.candidate_selection_accuracy < 0.9:
            issues.append("CANDIDATE_SELECTION_ACCURACY_TOO_LOW")
    if metrics.risk_hold_recall < 0.95:
        issues.append("RISK_HOLD_RECALL_TOO_LOW")
    if metrics.safe_motion_recall < 0.95:
        issues.append("SAFE_MOTION_RECALL_TOO_LOW")
    if metrics.risk_mean_absolute_error > 0.2:
        issues.append("RISK_MEAN_ABSOLUTE_ERROR_TOO_HIGH")
    if continuous_control:
        if metrics.pilot_control_mean_absolute_error is None:
            issues.append("PILOT_CONTROL_ERROR_EVIDENCE_MISSING")
        elif metrics.pilot_control_mean_absolute_error > maximum_pilot_control_mae:
            issues.append("PILOT_CONTROL_ERROR_TOO_HIGH")
        if set(metrics.pilot_axis_evidence) != set(PILOT_AXES):
            issues.append("PILOT_AXIS_EVIDENCE_INCOMPLETE")
        for axis in PILOT_AXES:
            evidence = metrics.pilot_axis_evidence.get(axis)
            if evidence is None:
                continue
            suffix = axis.upper()
            if evidence.sample_count != metrics.motion_sample_count:
                issues.append(f"PILOT_AXIS_SAMPLE_SUPPORT_MISMATCH_{suffix}")
            if min(evidence.positive_sample_count, evidence.negative_sample_count) < (
                minimum_axis_direction_samples
            ):
                issues.append(f"PILOT_AXIS_DIRECTION_COVERAGE_INSUFFICIENT_{suffix}")
            if evidence.mean_absolute_error > maximum_pilot_control_mae or (
                evidence.active_mean_absolute_error is not None
                and evidence.active_mean_absolute_error > maximum_pilot_control_mae
            ):
                issues.append(f"PILOT_AXIS_ERROR_TOO_HIGH_{suffix}")
    return issues
