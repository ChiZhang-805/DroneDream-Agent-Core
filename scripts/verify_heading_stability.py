"""CPU return verification and causal feedback stress, never flight qualification."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from dronedream_agent_core.heading_context import HEADING_CONTEXT_SHA256, validate_geometry
from dronedream_agent_core.training.heading_stability import load_stable_heading_policy


# 功能：
#   同时保留平均、尾部和最大误差，避免把少数失败隐藏于均值。
# 输入：
#   prediction、target：同一窗口的归一化偏航输出和历史执行标签。
# 输出：
#   metrics：三个误差统计量和窗口数量。
def errors(prediction, target):
    difference = np.abs(prediction-target)
    metrics = dict(count=len(difference), mae=float(difference.mean()), p95=float(np.quantile(difference, .95)), maximum=float(difference.max()))
    return metrics


# 功能：
#   严格沿过去配对替换历史动作进行压力测试；不改写后续观测，不把回放冒充物理闭环。
# 输入：
#   session：本地候选；features：原输入；previous、valid：同任务严格过去的配对索引及有效位。
# 输出：
#   prediction：逐窗口递归输出；replaced：使用自身历史输出的窗口数。
def feedback_probe(session, features, previous, valid):
    prediction = np.zeros(len(features), dtype=np.float32)
    replaced = 0
    for index, source in enumerate(features):
        current = source.copy()
        if valid[index]:
            parent = int(previous[index])
            if not 0 <= parent < index:
                raise ValueError('STABILITY_FEEDBACK_PAIR_NOT_PAST')
            if current[12] == 1.:
                current[10] = prediction[parent]
                replaced += 1
        prediction[index] = session.run(None, {'heading_context': current[None]})[0][0, 0]
    return prediction, replaced


# 功能：
#   核验所有回传权重、CPU 导出一致性、单窗口延迟及历史缺失和自身输出反馈压力。
# 输入：
#   args：包含冻结计划、数据和六组回传模型的目录，以及独占报告路径。
# 输出：
#   report：本机核验结果；不更换产品模型，也不授予飞行资格。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    plan_hash = hashlib.sha256((directory/'plan.json').read_bytes()).hexdigest()
    data_hash = hashlib.sha256((directory/'stability-inputs.npz').read_bytes()).hexdigest()
    manifest = json.loads((directory/'input-manifest.json').read_text())
    if manifest['data_sha256'] != data_hash or manifest['feature_contract_sha256'] != HEADING_CONTEXT_SHA256:
        raise ValueError('STABILITY_RETURN_DATA_MISMATCH')
    arrays = np.load(directory/'stability-inputs.npz', allow_pickle=False)
    features, target, mask = arrays['validation_context'], arrays['validation_target'][:, 3], arrays['validation_compatible']
    for row in np.concatenate((features, arrays['training_context'])):
        validate_geometry(row[:8].tolist())
        validate_geometry(row[13:21].tolist())
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    torch.set_num_threads(1)
    report = dict(data_sha256=data_hash, plan_sha256=plan_hash, final_test_read=False,
        qualified_for_flight=False, recursive_probe_is_physical_closed_loop=False, models={})
    for name in ('circular', 'context', 'context-noise', 'context-regularized', 'repeat-805', 'repeat-807'):
        folder = directory/name
        remote = json.loads((folder/'metrics.json').read_text())
        weights, graph = (folder/'heading.pt').read_bytes(), (folder/'heading.onnx').read_bytes()
        if (hashlib.sha256(weights).hexdigest() != remote['checkpoint_sha256']
                or hashlib.sha256(graph).hexdigest() != remote['onnx_sha256']):
            raise ValueError('STABILITY_RETURN_WEIGHT_HASH_MISMATCH')
        model = load_stable_heading_policy(weights, plan_sha256=plan_hash, data_sha256=data_hash)
        session = ort.InferenceSession(graph, sess_options=options, providers=['CPUExecutionProvider'])
        metadata = session.get_modelmeta().custom_metadata_map
        if (metadata.get('feature_contract_sha256') != HEADING_CONTEXT_SHA256 or metadata.get('plan_sha256') != plan_hash
                or metadata.get('data_sha256') != data_hash or metadata.get('qualified_for_flight') != 'false'
                or metadata.get('use_context') != str(model.use_context).lower()):
            raise ValueError('STABILITY_RETURN_ONNX_CONTRACT_MISMATCH')
        predicted = session.run(None, {'heading_context': features})[0][:, 0]
        with torch.inference_mode():
            reference = model(torch.from_numpy(features)).numpy()[:, 0]
        parity = float(np.max(np.abs(reference-predicted)))
        if parity > 1e-5 or not np.isfinite(predicted).all() or np.any(np.abs(predicted) > 1.):
            raise ValueError('STABILITY_RETURN_PARITY_FAILED')
        measured = errors(predicted[mask], target[mask])
        if abs(measured['mae']-remote['splits']['validation']['compatible']['mae']) > 1e-5:
            raise ValueError('STABILITY_RETURN_METRIC_MISMATCH')
        for _ in range(50):
            session.run(None, {'heading_context': features[:1]})
        latencies = []
        for index in range(2000):
            item = features[index % len(features):index % len(features)+1]
            started = time.perf_counter_ns()
            session.run(None, {'heading_context': item})
            latencies.append((time.perf_counter_ns()-started)/1e6)
        feedback, replaced = feedback_probe(session, features, arrays['validation_previous_index'], arrays['validation_pair_mask'])
        if not np.isfinite(feedback).all() or np.any(np.abs(feedback) > 1.):
            raise ValueError('STABILITY_FEEDBACK_NONFINITE_OR_UNBOUNDED')
        row = dict(seed=remote['seed'], parameters=remote['parameters'], checkpoint_sha256=remote['checkpoint_sha256'],
            onnx_sha256=remote['onnx_sha256'], checkpoint_bytes=len(weights), onnx_bytes=len(graph), parity_max_abs=parity,
            compatible=measured, all_windows=errors(predicted, target),
            latency_ms=dict(p50=float(np.quantile(latencies, .5)), p95=float(np.quantile(latencies, .95)),
                p99=float(np.quantile(latencies, .99)), maximum=max(latencies)),
            recursive_feedback=errors(feedback[mask], target[mask]), replaced_feedback_windows=replaced,
            command_missing=remote['command_missing'], noise_probe=remote['probe'])
        report['models'][name] = row
        print(json.dumps(dict(name=name, compatible=measured, recursive_feedback=row['recursive_feedback'], latency_ms=row['latency_ms'])), flush=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
