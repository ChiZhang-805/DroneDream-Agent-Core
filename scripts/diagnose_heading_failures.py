"""Trace fixed validation failures to original observations and transport receipts."""

import argparse
from bisect import bisect_left
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


# 功能：
#   只读取冻结摘要匹配的原记录，避免用后续修改后的文件解释旧结果。
# 输入：
#   path、digest：原文件路径和预期摘要。
# 输出：
#   rows：逐行原始 JSON 记录。
def read_records(path, digest):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError('HEADING_DIAGNOSIS_SOURCE_CHANGED')
    rows = [json.loads(line) for line in content.splitlines()]
    return rows


# 功能：
#   对照最差窗口及输入缺失分组，追溯过去回执缺失的具体时序和传输原因。
# 输入：
#   args：冻结数据目录、原始侧车证据目录和新报告路径。
# 输出：
#   report：原始失败、分组指标及回执时间证据，不删除样本或修改标签。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--sidecar-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root/'input-manifest.json').read_text())
    if hashlib.sha256((root/'stability-inputs.npz').read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('HEADING_DIAGNOSIS_DATA_CHANGED')
    arrays = np.load(root/'stability-inputs.npz', allow_pickle=False)
    reference = json.loads((root/'local-verification.json').read_text())
    predictions = {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    for name in ('context', 'circular'):
        graph = (root/name/'heading.onnx').read_bytes()
        if hashlib.sha256(graph).hexdigest() != reference['models'][name]['onnx_sha256']:
            raise ValueError('HEADING_DIAGNOSIS_MODEL_CHANGED')
        session = ort.InferenceSession(graph, sess_options=options, providers=['CPUExecutionProvider'])
        predictions[name] = session.run(None, {'heading_context': arrays['validation_context']})[0][:, 0]
    side_manifest = json.loads((args.sidecar_directory/'heading-ablation-manifest.json').read_text())
    sidecar = {row['source_snapshot_sha256']: row for row in read_records(args.sidecar_directory/'heading-sidecar.jsonl', side_manifest['heading_sidecar_sha256'])}
    error = np.abs(predictions['context']-arrays['validation_target'][:, 3])
    compatible = arrays['validation_compatible']
    indices = np.flatnonzero(compatible)
    selected = indices[np.argsort(error[indices])[-20:][::-1]]
    report = dict(final_test_read=False, labels_modified=False, physical_simulation=False,
        data_sha256=manifest['data_sha256'], missing_groups={}, worst_windows=[])
    for split in ('training', 'validation'):
        features, mask = arrays[split+'_context'], arrays[split+'_compatible']
        groups = {}
        for ack in (False, True):
            for history in (False, True):
                keep = mask & ((features[:, 12] == 1) == ack) & ((features[:, 22] == 1) == history)
                row = dict(count=int(keep.sum()))
                if split == 'validation' and keep.any():
                    for name, predicted in predictions.items():
                        errors = np.abs(predicted[keep]-arrays[split+'_target'][keep, 3])
                        row[name] = dict(mae=float(errors.mean()), maximum=float(errors.max()), p95=float(np.quantile(errors, .95)))
                groups[f'ack-{ack}-history-{history}'] = row
        report['missing_groups'][split] = groups
    loaded = {}
    for index in selected:
        identity = str(arrays['validation_identities'][index])
        side = sidecar[identity]
        source = side['source_root']
        if source not in loaded:
            hashes = manifest['sources'][source]
            observations = read_records(Path(source)/'learning-observations.jsonl', hashes['observations'])
            applications = read_records(Path(source)/'runtime-state/control-applications.jsonl', hashes['applications'])
            loaded[source] = ({row['snapshot']['snapshot_sha256']: row for row in observations}, applications)
        observations, applications = loaded[source]
        snapshot = observations[identity]['snapshot']
        source_ms = snapshot['control_reference_observed_at_unix_ms']
        position = bisect_left([row['accepted_at_unix_ms'] for row in applications], source_ms)-1
        previous = applications[position] if position >= 0 else None
        receipt = None
        if previous:
            receipt = {key: previous[key] for key in ('sequence', 'accepted_at_unix_ms', 'command_generated_at_unix_ms',
                'command_valid_until_unix_ms', 'transport', 'yaw_rate_application', 'safety_action')}
            receipt['age_at_observation_ms'] = source_ms-previous['accepted_at_unix_ms']
        report['worst_windows'].append(dict(index=int(index), identity=identity, source_root=source,
            source_observed_at_unix_ms=source_ms, current_position=snapshot['current_position_m'], goal=snapshot['goal_position_m'],
            target=float(arrays['validation_target'][index, 3]), context_prediction=float(predictions['context'][index]),
            circular_prediction=float(predictions['circular'][index]), context=arrays['validation_context'][index].tolist(),
            previous_receipt=receipt))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(groups=report['missing_groups'], worst=report['worst_windows'][:2])), flush=True)


if __name__ == '__main__':
    main()
