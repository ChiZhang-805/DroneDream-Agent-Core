"""Evaluate frozen action-conditioned risk graphs against matching hypothetical actions.

Executed successful demonstrations are not labels for a different proposed
action. This evaluator consumes only provenance-checked risk-only datasets;
it does not confer actor, flight or release qualification.
"""

import hashlib
import math
import re
import time
from pathlib import Path

from ..plugin_files import read_plugin_file
from .action_risk_artifacts import load_action_risk_dataset
from .action_risk_training import action_discrimination, risk_class_coverage


# 功能：
#   核对固定模型、教师及已使用分组身份，拒绝无界或歧义的评估参数。
# 输入：
#   model_sha256、teacher_sha256：预先固定的来源摘要。
#   excluded_groups：训练、开发以及已用于调整的测试分组；maximum_latency_ms：推理预算。
# 输出：
#   groups：格式验证后的禁止重用分组集合。
def _validate_identity(model_sha256, teacher_sha256, excluded_groups, maximum_latency_ms):
    if (type(excluded_groups) not in (set, frozenset) or not excluded_groups
            or len(excluded_groups) > 10000):
        raise ValueError('RISK_EVALUATION_EXCLUDED_GROUPS_INVALID')
    for value in (model_sha256, teacher_sha256, *excluded_groups):
        if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
            raise ValueError('RISK_EVALUATION_IDENTITY_INVALID')
    if (type(maximum_latency_ms) not in (int, float) or not math.isfinite(maximum_latency_ms)
            or not 0 < maximum_latency_ms <= 60000):
        raise ValueError('RISK_EVALUATION_LATENCY_BUDGET_INVALID')
    groups = set(excluded_groups)
    return groups


# 功能：
#   先核验独立来源和类别覆盖，再以每条记录实际绑定的动作输入运行原始风险图。
#   未覆盖危险类别时不计算虚假的危险召回，也不以行为示范的零占位标签替代风险答案。
# 输入：
#   model_path：固定风险图；dataset_roots：同一区域的完整独立风险数据集目录。
#   model_sha256、teacher_sha256：预期模型与离线教师身份；excluded_groups：不可用于终测的分组。
#   maximum_latency_ms：单条输入构造与模型推理的尾延迟预算。
# 输出：
#   report：真实来源、类别数量、指标和拒绝原因；不包含飞行资格。
def evaluate_action_risk_artifact(model_path: Path, dataset_roots, *, model_sha256: str,
                                  teacher_sha256: str, excluded_groups, maximum_latency_ms=20.):
    import numpy as np
    import onnxruntime as ort

    used = _validate_identity(model_sha256, teacher_sha256, excluded_groups, maximum_latency_ms)
    if type(dataset_roots) not in (list, tuple) or not 1 <= len(dataset_roots) <= 16:
        raise ValueError('RISK_EVALUATION_DATASET_COUNT_INVALID')
    content = read_plugin_file(model_path, limit=16 * 1024**2)
    if hashlib.sha256(content).hexdigest() != model_sha256:
        raise ValueError('RISK_EVALUATION_MODEL_CHANGED')
    datasets, test_groups = [], set()
    for root in dataset_roots:
        data = load_action_risk_dataset(root)
        if (not data.groups or used & data.groups
                or data.teacher_config_sha256 != teacher_sha256):
            raise ValueError('RISK_EVALUATION_SOURCE_OVERLAP_OR_TEACHER_MISMATCH')
        used.update(data.groups)
        test_groups.update(data.groups)
        datasets.append(data)
    coverage = risk_class_coverage(datasets)
    issues = [name.upper() + '_BELOW_TWENTY'
              for name in ('safe_observation_count', 'unsafe_observation_count',
                           'action_contrast_observation_count') if coverage[name] < 20]
    report = dict(model_sha256=model_sha256, teacher_config_sha256=teacher_sha256,
                  dataset_receipt_sha256=[data.receipt_sha256 for data in datasets],
                  test_groups=sorted(test_groups), coverage=coverage, metrics=None,
                  action_discrimination=None, p99_input_and_inference_ms=None,
                  inference_performed=False, independent_test_passed=False,
                  qualified_for_flight=False, issue_codes=issues)
    # 来源和覆盖失败先于模型调用，既节省时间，也避免选择全安全数据得到好看的模型成绩。
    if issues:
        return report
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(content, options, providers=['CPUExecutionProvider'])
    fields = [('state_features', 'state_features'), ('realtime_features', 'realtime_features'),
              ('realtime_valid_mask', 'realtime_valid_mask'),
              ('proposed_control', 'risk_proposed_control')]
    if ({item.name for item in session.get_inputs()} != {key for key, _ in fields}
            or [item.name for item in session.get_outputs()] != ['risk_score']):
        raise ValueError('RISK_EVALUATION_GRAPH_INTERFACE_INVALID')
    predictions, targets, latencies = [], [], []
    for data in datasets:
        for sample in data.samples:
            started = time.perf_counter()
            feeds = {key: np.asarray([getattr(sample, field)], dtype=np.float32)
                     for key, field in fields}
            result = session.run(['risk_score'], feeds)[0]
            if (result.shape != (1, 1) or result.dtype != np.float32
                    or not np.isfinite(result).all() or not 0 <= result.item() <= 1):
                raise ValueError('RISK_EVALUATION_OUTPUT_INVALID')
            elapsed = (time.perf_counter() - started) * 1000
            if len(predictions) >= 20:
                latencies.append(elapsed)
            predictions.append(float(result.item()))
            targets.append(sample.risk_target)
    prediction, target = np.asarray(predictions), np.asarray(targets)
    risky, predicted_risky = target >= .5, prediction >= .5
    metrics = dict(sample_count=len(target), risky_sample_count=int(risky.sum()),
                   safe_sample_count=int((~risky).sum()),
                   risk_hold_recall=float(predicted_risky[risky].mean()),
                   safe_motion_recall=float((~predicted_risky[~risky]).mean()),
                   risk_mean_absolute_error=float(np.abs(prediction - target).mean()),
                   false_safe_count=int((risky & ~predicted_risky).sum()),
                   false_hold_count=int((~risky & predicted_risky).sum()))
    discrimination = action_discrimination(datasets, prediction)
    for name in ('risk_hold_recall', 'safe_motion_recall'):
        if metrics[name] < .95:
            issues.append(name.upper() + '_BELOW_THRESHOLD')
    if metrics['risk_mean_absolute_error'] > .2:
        issues.append('RISK_ERROR_ABOVE_THRESHOLD')
    if discrimination['correct_fraction'] is None or discrimination['correct_fraction'] < .95:
        issues.append('ACTION_DISCRIMINATION_BELOW_THRESHOLD')
    p99 = float(np.percentile(latencies, 99))
    if not math.isfinite(p99) or p99 > maximum_latency_ms:
        issues.append('RISK_INFERENCE_LATENCY_ABOVE_THRESHOLD')
    report.update(metrics=metrics, action_discrimination=discrimination,
                  p99_input_and_inference_ms=p99, inference_performed=True,
                  independent_test_passed=not issues)
    return report
