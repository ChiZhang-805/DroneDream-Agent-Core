"""Preserve returned candidates and exact training/review code without replacing a product model."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


# 功能：
#   将已核验候选、报告与当前审查源码复制到新归档，逐文件核对摘要，不覆盖既有模型。
# 输入：
#   args：核验证据目录、当前源码根目录和不存在的归档目标。
# 输出：
#   manifest：含文件字节数与摘要的归档回执。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verification = json.loads((args.evidence/'local-verification.json').read_text())
    files = []
    for name, record in verification['models'].items():
        if name not in ('circular', 'context', 'context-noise', 'context-regularized', 'repeat-805', 'repeat-807'):
            raise ValueError('STABILITY_ARCHIVE_MODEL_NAME_INVALID')
        for filename, key in (('heading.pt', 'checkpoint_sha256'), ('heading.onnx', 'onnx_sha256')):
            source = args.evidence/name/filename
            if hashlib.sha256(source.read_bytes()).hexdigest() != record[key]:
                raise ValueError('STABILITY_ARCHIVE_WEIGHT_CHANGED')
            files.append((source, Path(name)/filename))
        files.append((args.evidence/name/'metrics.json', Path(name)/'metrics.json'))
    for name in ('plan.json', 'input-manifest.json', 'selection.json', 'completion.json',
                 'local-verification.json', 'regression.xml', 'regression-final.xml'):
        files.append((args.evidence/name, Path(name)))
    code = ['heading_context.py', 'yaw_command_envelope.py', 'runtime_local_safety.py', 'training/heading_stability.py']
    paths = ['src/dronedream_agent_core/'+name for name in code]
    paths += ['scripts/'+name for name in ('prepare_heading_stability.py', 'train_heading_stability.py',
        'run_heading_stability_matrix.py', 'verify_heading_stability.py', 'archive_heading_stability.py')]
    paths += ['tests/test_heading_stability.py', 'tests/test_yaw_command_envelope.py']
    for name in paths:
        files.append((args.repository/name, Path('review-source')/name))
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for source, relative in files:
        target = args.output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError('STABILITY_ARCHIVE_COPY_MISMATCH')
        records.append(dict(path=relative.as_posix(), bytes=target.stat().st_size, sha256=digest))
    manifest = dict(schema='dronedream.heading-stability-archive.v1', selected='context',
        qualified_for_flight=False, product_models_replaced=False, source_data_location=str(args.evidence/'stability-inputs.npz'),
        data_sha256=verification['data_sha256'], files=records, total_files=len(records), total_bytes=sum(item['bytes'] for item in records),
        source_note='review-source includes later loader and harness changes; the frozen cloud wheel is recorded separately')
    with (args.output/'archive-manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(dict(files=len(records), bytes=manifest['total_bytes'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
