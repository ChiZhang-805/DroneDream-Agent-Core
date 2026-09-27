"""Bounded CUDA experiments for causal, periodic, noise-aware yaw prediction."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from dronedream_agent_core.heading_context import HEADING_CONTEXT_SHA256
from dronedream_agent_core.training.heading_refinement import heading_training_weights
from dronedream_agent_core.training.heading_stability import (
    ARCHITECTURE, StableHeadingPolicy, perturb_context, reflect_context, temporal_delta_loss,
)


# 功能：
#   同时报告平均误差、尾部和最差任务组，不只选择有利的整体均值。
# 输入：
#   prediction、target：逐窗口偏航预测与原执行标签；groups：固定任务身份。
# 输出：
#   report：逐组和整体误差。
def measure(prediction, target, groups):
    error = np.abs(prediction-target)
    report = dict(count=len(error), mae=float(error.mean()), p95=float(np.quantile(error, .95)), maximum=float(error.max()), groups={})
    for group in np.unique(groups):
        selected = groups == group
        report['groups'][str(group)] = dict(count=int(selected.sum()), mae=float(error[selected].mean()), p95=float(np.quantile(error[selected], .95)))
    report['worst_group_mae'] = max(row['mae'] for row in report['groups'].values())
    return report


# 功能：
#   在相同角度范围测量小扰动反应，并另外报告同一潜在状态下的噪声预测误差。
# 输入：
#   session：候选 ONNX；features、target：兼容验证输入及原执行标签。
# 输出：
#   report：当前帧与相关历史扰动指标；不把假设噪声说成实测传感器标定。
def probe(session, features, target):
    original = session.run(None, {'heading_context': features})[0][:, 0]
    safe = np.all(np.abs(features[:, [0, 3]]) < .95, axis=1)
    selected = features[safe]
    report = dict(source_windows=int(safe.sum()), noise_bound_degrees=.25, measured_noise_calibration=False)
    for correlated in (False, True):
        changes, errors = [], []
        for sign in (-1, 1):
            noisy = selected.copy()
            for angle, valid, factor in ((0, 2, 1.), (3, 5, 1.), (13, 15, .5), (16, 18, .5)):
                if angle >= 13 and not correlated:
                    continue
                noisy[:, angle] = ((noisy[:, angle]+sign*.25/180*factor+1)%2-1)*noisy[:, valid]
            prediction = session.run(None, {'heading_context': noisy})[0][:, 0]
            changes.extend(np.abs(prediction-original[safe]).tolist())
            errors.extend(np.abs(prediction-target[safe]).tolist())
        report['correlated' if correlated else 'current_only'] = dict(change_p95=float(np.quantile(changes, .95)),
            change_max=float(max(changes)), noisy_mae=float(np.mean(errors)), noisy_error_p95=float(np.quantile(errors, .95)))
    return report


# 功能：
#   固定数据与计划训练一个候选，镜像和历史缺失增强不修改原记录；验证集不参与梯度。
# 输入：
#   args：冻结目录、方案、种子和独占输出路径。
# 输出：
#   report：原标签验证、噪声敏感度、时序变化、权重及导出身份。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan_bytes, manifest_bytes = (args.directory/'plan.json').read_bytes(), (args.directory/'input-manifest.json').read_bytes()
    plan, manifest = json.loads(plan_bytes), json.loads(manifest_bytes)
    data_bytes = (args.directory/'stability-inputs.npz').read_bytes()
    if (hashlib.sha256(data_bytes).hexdigest() != manifest['data_sha256']
            or manifest['feature_contract_sha256'] != HEADING_CONTEXT_SHA256
            or args.variant not in plan['variants'] or args.seed not in [plan['seed'], *plan['repeat_seeds']]):
        raise ValueError('STABILITY_FROZEN_INPUT_INVALID')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('STABILITY_EXACTLY_ONE_CUDA_REQUIRED')
    arrays = np.load(io.BytesIO(data_bytes), allow_pickle=False)
    keep = arrays['training_compatible']
    x, y = torch.from_numpy(arrays['training_context'][keep]), torch.from_numpy(arrays['training_target'][keep, 3:4])
    previous_index = arrays['training_previous_index'][keep]
    px, py = torch.from_numpy(arrays['training_context'][previous_index]), torch.from_numpy(arrays['training_target'][previous_index, 3:4])
    valid = torch.from_numpy(arrays['training_pair_mask'][keep])
    weights = heading_training_weights(x[:, :8], y, arrays['training_groups'][keep].tolist(), balance_turns=False)
    mx, my = reflect_context(x, y)
    mpx, mpy = reflect_context(px, py)
    x, y, px, py = torch.cat((x, mx)), torch.cat((y, my)), torch.cat((px, mpx)), torch.cat((py, mpy))
    weights, valid = weights.repeat(2), valid.repeat(2)
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    model = StableHeadingPolicy(use_context=args.variant != 'circular').cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=1e-5)
    x, y, px, py, weights, valid = (value.cuda() for value in (x, y, px, py, weights, valid))
    started = time.monotonic()
    for epoch in range(plan['epochs']):
        for slots in torch.randperm(len(x), generator=generator).split(plan['batch_size']):
            if time.monotonic()-started > plan['maximum_job_seconds']:
                raise TimeoutError('STABILITY_JOB_TIMEOUT')
            slots = slots.cuda()
            features = x[slots].clone()
            if model.use_context:
                dropped = torch.rand(len(slots), generator=generator).cuda() < plan['history_dropout']
                features[dropped, 10:] = 0.
            predicted = model(features)
            loss = (((predicted-y[slots]).square()[:, 0])*weights[slots]).sum()/weights[slots].sum()
            if args.variant in ('context-noise', 'context-regularized'):
                delta = ((torch.rand(len(slots), 1, generator=generator)*2-1)*plan['noise_degrees']/180).cuda()
                noisy = model(perturb_context(features, delta))
                loss = loss + .5*((noisy-y[slots]).square()[:, 0]*weights[slots]).sum()/weights[slots].sum()
                if args.variant == 'context-regularized':
                    loss = loss + plan['spatial_regularization']*(predicted-noisy).square().mean()
                    loss = loss + plan['temporal_regularization']*temporal_delta_loss(predicted, model(px[slots]), y[slots], py[slots], valid[slots])
            if not torch.isfinite(loss):
                raise ValueError('STABILITY_NONFINITE_LOSS')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        if (epoch+1) % 60 == 0:
            print(json.dumps(dict(epoch=epoch+1, loss=loss.detach().item())), flush=True)
    elapsed = time.monotonic()-started
    model.cpu().eval()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('STABILITY_NONFINITE_WEIGHTS')
    payload = dict(architecture=ARCHITECTURE, feature_contract_sha256=HEADING_CONTEXT_SHA256,
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(), data_sha256=manifest['data_sha256'],
        use_context=model.use_context, yaw_limit_dps=20., state_dict=state, qualified_for_flight=False)
    with (args.output/'heading.pt').open('xb') as stream:
        torch.save(payload, stream)
    buffer = io.BytesIO()
    torch.onnx.export(model, (torch.zeros(1, 23),), buffer, input_names=['heading_context'], output_names=['yaw_axis'],
        dynamic_axes={'heading_context': {0: 'batch'}, 'yaw_axis': {0: 'batch'}}, opset_version=17, dynamo=False)
    graph = onnx.load_model_from_string(buffer.getvalue())
    for key, value in payload.items():
        if key == 'state_dict':
            continue
        item = graph.metadata_props.add()
        item.key, item.value = key, str(value).lower() if isinstance(value, bool) else str(value)
    onnx.checker.check_model(graph)
    graph_bytes = graph.SerializeToString()
    with (args.output/'heading.onnx').open('xb') as stream:
        stream.write(graph_bytes)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(graph_bytes, sess_options=options, providers=['CPUExecutionProvider'])
    report = dict(variant=args.variant, seed=args.seed, training_seconds=elapsed, training_windows=int(keep.sum()),
        augmented_rows=len(x), parameters=sum(p.numel() for p in model.parameters()), plan_sha256=payload['plan_sha256'],
        data_sha256=manifest['data_sha256'], onnx_sha256=hashlib.sha256(graph_bytes).hexdigest(),
        checkpoint_sha256=hashlib.sha256((args.output/'heading.pt').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), final_test_read=False, qualified_for_flight=False, splits={})
    for split in ('training', 'validation'):
        features, target, groups = arrays[split+'_context'], arrays[split+'_target'][:, 3], arrays[split+'_groups']
        predicted = session.run(None, {'heading_context': features})[0][:, 0]
        with torch.inference_mode():
            reference = model(torch.from_numpy(features)).numpy()[:, 0]
        parity = float(np.abs(reference-predicted).max())
        if not np.isfinite(predicted).all() or (np.abs(predicted)>1).any() or parity>1e-5:
            raise ValueError('STABILITY_EXPORT_PARITY_FAILED')
        mask = arrays[split+'_compatible']
        pair, index = arrays[split+'_pair_mask'], arrays[split+'_previous_index']
        delta_error = np.abs((predicted-predicted[index])-(target-target[index]))[pair]
        report['splits'][split] = dict(parity_max_abs=parity, all=measure(predicted, target, groups),
            compatible=measure(predicted[mask], target[mask], groups[mask]), legacy_dynamic=measure(predicted[~mask], target[~mask], groups[~mask]),
            temporal_delta_mae=float(delta_error.mean()) if len(delta_error) else None, temporal_pairs=int(pair.sum()))
    mask = arrays['validation_compatible']
    report['probe'] = probe(session, arrays['validation_context'][mask], arrays['validation_target'][mask, 3])
    # 遮蔽过去动作揭示是否只会复制上一杆；是离线压力测试，不是假装闭环成功。
    masked = arrays['validation_context'][mask].copy()
    masked[:, 10:13] = 0.
    report['command_missing'] = measure(session.run(None, {'heading_context': masked})[0][:, 0], arrays['validation_target'][mask, 3], arrays['validation_groups'][mask])
    with (args.output/'metrics.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
