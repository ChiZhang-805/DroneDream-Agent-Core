"""Frozen two-variant CPU comparison; preserves all validation windows and original labels."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from dronedream_agent_core.heading_context import HEADING_CONTEXT_SHA256
from dronedream_agent_core.training.heading_availability import AvailabilityHeadingPolicy
from dronedream_agent_core.training.heading_stability import load_stable_heading_policy
from train_heading_stability import probe
from verify_heading_stability import errors, feedback_probe


# 功能：
#   按事先冻结规则对缺测分流方案核验数值、误差和扰动，不再训练或改写原始数据。
# 输入：
#   args：冻结训练来源与已有比较计划目录。
# 输出：
#   report：两种组合的全量指标及选中方案，未通过门槛时选中项为空。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan_bytes = (args.output/'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    training_plan = hashlib.sha256((args.source/'plan.json').read_bytes()).hexdigest()
    if hashlib.sha256((args.source/'stability-inputs.npz').read_bytes()).hexdigest() != plan['data_sha256']:
        raise ValueError('HEADING_AVAILABILITY_DATA_CHANGED')
    reference = json.loads((args.source/'local-verification.json').read_text())
    branches = {}
    for name in ('context', 'circular'):
        content = (args.source/name/'heading.pt').read_bytes()
        if hashlib.sha256(content).hexdigest() != reference['models'][name]['checkpoint_sha256']:
            raise ValueError('HEADING_AVAILABILITY_WEIGHT_CHANGED')
        branches[name] = load_stable_heading_policy(content, plan_sha256=training_plan, data_sha256=plan['data_sha256'])
    if reference['models']['context']['checkpoint_sha256'] != plan['context_checkpoint_sha256']:
        raise ValueError('HEADING_AVAILABILITY_CONTEXT_CHANGED')
    arrays = np.load(args.source/'stability-inputs.npz', allow_pickle=False)
    x, y, mask = arrays['validation_context'], arrays['validation_target'][:, 3], arrays['validation_compatible']
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    torch.set_num_threads(1)
    report = dict(plan_sha256=hashlib.sha256(plan_bytes).hexdigest(), candidates={}, selected=None, qualified_for_flight=False)
    for name in plan['variants']:
        model = AvailabilityHeadingPolicy(branches['context'], branches['circular'], age_weighted=name == 'age-weighted')
        buffer = io.BytesIO()
        torch.onnx.export(model, (torch.zeros(1, 23),), buffer, input_names=['heading_context'], output_names=['yaw_axis'],
            dynamic_axes={'heading_context': {0: 'batch'}, 'yaw_axis': {0: 'batch'}}, opset_version=17, dynamo=False)
        graph = onnx.load_model_from_string(buffer.getvalue())
        metadata = dict(architecture='heading-availability-v1', variant=name, feature_contract_sha256=HEADING_CONTEXT_SHA256,
            plan_sha256=report['plan_sha256'], data_sha256=plan['data_sha256'], qualified_for_flight='false')
        for key, value in metadata.items():
            entry = graph.metadata_props.add()
            entry.key, entry.value = key, value
        onnx.checker.check_model(graph)
        content = graph.SerializeToString()
        session = ort.InferenceSession(content, sess_options=options, providers=['CPUExecutionProvider'])
        predicted = session.run(None, {'heading_context': x})[0][:, 0]
        with torch.inference_mode():
            direct = model(torch.from_numpy(x)).numpy()[:, 0]
        if (not np.isfinite(predicted).all() or np.any(np.abs(predicted) > 1.) or np.max(np.abs(predicted-direct)) > 1e-5):
            raise ValueError('HEADING_AVAILABILITY_EXPORT_INVALID')
        row = dict(compatible=errors(predicted[mask], y[mask]), all_windows=errors(predicted, y),
            legacy_dynamic=errors(predicted[~mask], y[~mask]), noise=probe(session, x[mask], y[mask]),
            sha256=hashlib.sha256(content).hexdigest(), parameters=sum(p.numel() for p in model.parameters()))
        recursive, replaced = feedback_probe(session, x, arrays['validation_previous_index'], arrays['validation_pair_mask'])
        row['recursive_feedback'], row['feedback_windows'] = errors(recursive[mask], y[mask]), replaced
        missing = x.copy()
        missing[:, 10:13] = 0.
        row['command_missing'] = errors(session.run(None, {'heading_context': missing})[0][mask, 0], y[mask])
        limits = plan['selection']
        row['eligible'] = (row['compatible']['mae'] <= limits['mae_maximum'] and row['compatible']['p95'] <= limits['p95_maximum']
            and row['compatible']['maximum'] <= limits['worst_window_maximum']
            and row['noise']['current_only']['change_p95'] <= limits['jitter_p95_maximum']
            and row['noise']['current_only']['change_max'] <= limits['jitter_maximum'])
        with (args.output/(name+'.onnx')).open('xb') as stream:
            stream.write(content)
        report['candidates'][name] = row
    eligible = [name for name, row in report['candidates'].items() if row['eligible']]
    if eligible:
        report['selected'] = min(eligible, key=lambda name: (report['candidates'][name]['compatible']['maximum'], report['candidates'][name]['compatible']['mae']))
    with (args.output/'evaluation.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
