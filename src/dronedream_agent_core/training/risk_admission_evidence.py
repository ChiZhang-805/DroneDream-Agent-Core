"""Separate denominators for action-risk probes and executed actor demonstrations."""

import math
from typing import Annotated

from pydantic import Field, model_validator

from ..contracts import StrictModel

Sha256 = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]


class ActionRiskRegionEvidence(StrictModel):
    dataset_receipt_sha256: list[Sha256] = Field(min_length=1, max_length=16)
    test_groups: list[Sha256] = Field(min_length=1, max_length=10000)
    sample_count: int = Field(ge=1, le=250000)
    risky_sample_count: int = Field(ge=0)
    safe_sample_count: int = Field(ge=0)
    independent_observation_count: int = Field(ge=1)
    safe_observation_count: int = Field(ge=0)
    unsafe_observation_count: int = Field(ge=0)
    action_contrast_observation_count: int = Field(ge=0)
    risk_hold_recall: float = Field(ge=0., le=1.)
    safe_motion_recall: float = Field(ge=0., le=1.)
    risk_mean_absolute_error: float = Field(ge=0., le=1.)
    action_discrimination_fraction: float = Field(ge=0., le=1.)
    p99_input_and_inference_ms: float = Field(ge=0., le=60000.)

    # 功能：
    #   核对风险探针和独立观测各自的分母，拒绝重复来源及超出真实支持量的计数。
    # 输入：
    #   self：单一区域的动作风险证据。
    # 输出：
    #   self：计数与来源一致的证据，不代表质量通过。
    @model_validator(mode='after')
    def validate_counts(self):
        if (self.risky_sample_count + self.safe_sample_count != self.sample_count
                or self.independent_observation_count > self.sample_count
                or self.safe_observation_count > min(self.safe_sample_count,
                                                     self.independent_observation_count)
                or self.unsafe_observation_count > min(self.risky_sample_count,
                                                       self.independent_observation_count)
                or self.action_contrast_observation_count > self.independent_observation_count
                or len(set(self.test_groups)) != len(self.test_groups)
                or len(set(self.dataset_receipt_sha256)) != len(self.dataset_receipt_sha256)):
            raise ValueError('ACTION_RISK_ADMISSION_SUPPORT_INVALID')
        return self


class ActionRiskAdmissionEvidence(StrictModel):
    model_sha256: Sha256
    teacher_config_sha256: Sha256
    regions: list[ActionRiskRegionEvidence] = Field(min_length=1, max_length=16)

    # 功能：
    #   阻止跨区域重复引用同一来源；保留按区域验收，不用总平均抵消薄弱区域。
    # 输入：
    #   self：绑定同一风险模型及教师的全部独立区域证据。
    # 输出：
    #   self：来源互斥的风险证据。
    @model_validator(mode='after')
    def validate_regions(self):
        groups, receipts = set(), set()
        for region in self.regions:
            if groups.intersection(region.test_groups) or receipts.intersection(
                    region.dataset_receipt_sha256):
                raise ValueError('ACTION_RISK_ADMISSION_DUPLICATE_REGION')
            groups.update(region.test_groups)
            receipts.update(region.dataset_receipt_sha256)
        return self


# 功能：
#   将实际图评估结果转换为与行为样本分开的类型化证据，未推理的覆盖失败不能伪造数值。
# 输入：
#   reports：来源读取器及真实 ONNX 推理产生的区域结果。
# 输出：
#   evidence：绑定单一模型、教师和所有区域的独立风险证据。
def action_risk_evidence_from_reports(reports):
    if type(reports) is not list or not 1 <= len(reports) <= 16:
        raise ValueError('ACTION_RISK_ADMISSION_REPORTS_INVALID')
    regions, model, teacher = [], None, None
    for report in reports:
        if (type(report) is not dict or report.get('inference_performed') is not True
                or report.get('qualified_for_flight') is not False):
            raise ValueError('ACTION_RISK_ADMISSION_MEASUREMENTS_REQUIRED')
        if model is not None and (model != report['model_sha256']
                                 or teacher != report['teacher_config_sha256']):
            raise ValueError('ACTION_RISK_ADMISSION_MODEL_OR_TEACHER_CHANGED')
        model, teacher = report['model_sha256'], report['teacher_config_sha256']
        metrics, coverage = report['metrics'], report['coverage']
        discrimination = report['action_discrimination']
        contrast_count = coverage['action_contrast_observation_count']
        correct_count = discrimination['correct_count']
        if (metrics['sample_count'] != coverage['sample_count']
                or type(contrast_count) is not int or contrast_count < 1
                or discrimination['observation_count'] != contrast_count
                or type(correct_count) is not int or not 0 <= correct_count <= contrast_count
                or discrimination['correct_fraction'] != correct_count / contrast_count):
            raise ValueError('ACTION_RISK_ADMISSION_SUPPORT_INVALID')
        fields = {key: metrics[key] for key in ('sample_count', 'risky_sample_count',
            'safe_sample_count', 'risk_hold_recall', 'safe_motion_recall',
            'risk_mean_absolute_error')}
        fields.update({key: coverage[key] for key in ('independent_observation_count',
            'safe_observation_count', 'unsafe_observation_count',
            'action_contrast_observation_count')})
        regions.append(ActionRiskRegionEvidence(**fields,
            dataset_receipt_sha256=report['dataset_receipt_sha256'],
            test_groups=report['test_groups'],
            action_discrimination_fraction=report['action_discrimination']['correct_fraction'],
            p99_input_and_inference_ms=report['p99_input_and_inference_ms']))
    evidence = ActionRiskAdmissionEvidence(model_sha256=model, teacher_config_sha256=teacher,
                                          regions=regions)
    return evidence


# 功能：
#   按每个独立区域执行原风险门槛，缺少类别、动作对照或低召回均不能被其他区域平均掩盖。
# 输入：
#   evidence：风险专用证据；maximum_latency_ms：当前模型包的原始延迟预算。
# 输出：
#   issues：不满足风险接纳标准的问题代码。
def action_risk_evidence_issues(evidence, *, maximum_latency_ms=60000.):
    if (type(maximum_latency_ms) not in (int, float) or not math.isfinite(maximum_latency_ms)
            or not 0 < maximum_latency_ms <= 60000):
        raise ValueError('ACTION_RISK_ADMISSION_LATENCY_BUDGET_INVALID')
    evidence = ActionRiskAdmissionEvidence.model_validate(evidence.model_dump(), strict=True)
    issues = []
    for index, region in enumerate(evidence.regions):
        if min(region.safe_observation_count, region.unsafe_observation_count,
               region.action_contrast_observation_count) < 20:
            issues.append(f'ACTION_RISK_REGION_{index}_OBSERVATION_SUPPORT_INSUFFICIENT')
        if min(region.risk_hold_recall, region.safe_motion_recall) < .95:
            issues.append(f'ACTION_RISK_REGION_{index}_RECALL_TOO_LOW')
        if region.risk_mean_absolute_error > .2:
            issues.append(f'ACTION_RISK_REGION_{index}_ERROR_TOO_HIGH')
        if region.action_discrimination_fraction < .95:
            issues.append(f'ACTION_RISK_REGION_{index}_DISCRIMINATION_TOO_LOW')
        if region.p99_input_and_inference_ms > maximum_latency_ms:
            issues.append(f'ACTION_RISK_REGION_{index}_LATENCY_TOO_HIGH')
    return issues


# 功能：
#   汇总独立风险指标并保留其自身样本分母，不能把风险探针计入真实行为样本总量。
# 输入：
#   evidence：类型化区域证据。
# 输出：
#   fields：可用于报告的风险类别数、召回和加权误差。
def action_risk_summary(evidence):
    evidence = ActionRiskAdmissionEvidence.model_validate(evidence.model_dump(), strict=True)
    risky = sum(region.risky_sample_count for region in evidence.regions)
    safe = sum(region.safe_sample_count for region in evidence.regions)
    fields = dict(risky_sample_count=risky, safe_sample_count=safe,
        risk_hold_recall=sum(r.risk_hold_recall*r.risky_sample_count
                            for r in evidence.regions)/risky if risky else 0.,
        safe_motion_recall=sum(r.safe_motion_recall*r.safe_sample_count
                              for r in evidence.regions)/safe if safe else 0.,
        risk_mean_absolute_error=sum(r.risk_mean_absolute_error*r.sample_count
                                     for r in evidence.regions)/(risky+safe))
    return fields
