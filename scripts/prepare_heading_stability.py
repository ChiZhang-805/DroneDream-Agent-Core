"""Build source-bound, train/validation-only context without future action leakage."""

import argparse
from bisect import bisect_left
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.contracts import DynamicObstacleObservation, QuaternionWxyz, RuntimeLocalSafetyCommand, Vector3
from dronedream_agent_core.heading_context import HEADING_CONTEXT_SHA256, encode_heading_context
from dronedream_agent_core.heading_observation import encode_heading_observation


# 功能：
#   读取与已有独立核对报告摘要一致的原始文件，不信任路径名或修改时间。
# 输入：
#   path：源文件；digest：冻结的 SHA256。
# 输出：
#   content：经过身份核对的原始字节。
def read_bound(path, digest):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError('STABILITY_SOURCE_CHANGED:'+str(path))
    return content


# 功能：
#   计算历史连续性身份，目标位置或最近目标发生变化即重置几何历史。
# 输入：
#   snapshot：原观测快照；tracks：由已绑定命令记录保留的原始感知目标。
# 输出：
#   identity：任务目标与被关注目标身份摘要。
def context_identity(snapshot, tracks):
    position = snapshot['current_position_m']
    nearest = min(tracks, key=lambda item: (math.dist(tuple(position[a] for a in 'xyz'),
        tuple(item['position_m'][a] for a in 'xyz'))-max(item['radius_m'], item['height_m']/2), item['obstacle_id'])) if tracks else None
    identity = sha256_json(dict(goal=snapshot['goal_position_m'], track=nearest['obstacle_id'] if nearest else None))
    return identity


# 功能：
#   按已冻结的历史记录关联报告重建几何，不改写缺少精确几何的旧快照。
# 输入：
#   snapshot：原快照；tracks：其已绑定命令记录里的感知目标。
# 输出：
#   geometry：与冻结来源契约相同的八维历史研究特征。
def reconstruct_geometry(snapshot, tracks):
    flight = next(e for e in snapshot['realtime_feature_snapshot']['encodings'] if e['encoder_role']=='flight-state-encoder')
    geometry = encode_heading_observation(position=Vector3.model_validate(snapshot['current_position_m']),
        goal=Vector3.model_validate(snapshot['goal_position_m']),
        orientation=QuaternionWxyz(**dict(zip(('w', 'x', 'y', 'z'), flight['features'][:4], strict=True))),
        obstacles=[DynamicObstacleObservation.model_validate(row) for row in tracks], local_radius_m=snapshot['local_radius_m'])
    return geometry


# 功能：
#   从冻结原始观测和执行回执提取真实过去信息，保持基线窗口不变。
# 输入：
#   args：冻结证据目录、原组装数据目录及新的独占输出目录。
# 输出：
#   report：源身份、输入契约、缺测和连续窗口覆盖，不新增实采数量。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.previous/'heading-ablation-manifest.json').read_text())
    read_bound(args.previous/'heading-ablation-inputs.npz', manifest['data_sha256'])
    read_bound(args.previous/'heading-sidecar.jsonl', manifest['heading_sidecar_sha256'])
    assembly = json.loads(read_bound(args.data/'assembly-receipt.json', manifest['assembly_sha256']))
    audited = json.loads((args.previous/'heading-alignment-verification.json').read_text())
    arrays = dict(np.load(args.previous/'heading-ablation-inputs.npz', allow_pickle=False))
    cohort = json.loads((args.previous/'heading-cohort-manifest.json').read_text())
    read_bound(args.previous/'heading-cohort-masks.npz', cohort['masks_sha256'])
    arrays.update(dict(np.load(args.previous/'heading-cohort-masks.npz', allow_pickle=False)))
    wanted = {str(identity) for split in ('training', 'validation') for identity in arrays[split+'_identities']}
    roots = defaultdict(dict)
    for line in (args.previous/'heading-sidecar.jsonl').read_bytes().splitlines():
        row = json.loads(line)
        if row['source_snapshot_sha256'] in wanted:
            roots[row['source_root']][row['source_snapshot_sha256']] = row
    extracted, source_inventory = {}, {}
    for root, rows in roots.items():
        path, expected = Path(root), audited['files'][root]
        observations_bytes = read_bound(path/'learning-observations.jsonl', expected['observations'])
        applications_bytes = read_bound(path/'runtime-state/control-applications.jsonl', expected['applications'])
        commands_bytes = read_bound(path/'depth-local-safety-history.jsonl', expected['commands'])
        commands = {}
        for line in commands_bytes.splitlines():
            row = json.loads(line)
            if row.get('command') is not None:
                commands[sha256_json(RuntimeLocalSafetyCommand.model_validate(row['command']))] = row
        applications = [json.loads(line) for line in applications_bytes.splitlines()]
        times = [row['accepted_at_unix_ms'] for row in applications]
        if any(type(t) is not int for t in times) or times != sorted(times):
            raise ValueError('STABILITY_APPLICATION_CLOCK_INVALID')
        observations = [json.loads(line) for line in observations_bytes.splitlines()]
        observations.sort(key=lambda row: row['snapshot']['control_reference_observed_at_unix_ms'])
        previous, previous_key = None, None
        for recorded in observations:
            snapshot = recorded['snapshot']
            identity = snapshot['snapshot_sha256']
            source_ms = snapshot['control_reference_observed_at_unix_ms']
            command = commands.get(recorded.get('evaluated_command_sha256'))
            if command is None:
                if identity in rows:
                    raise ValueError('STABILITY_CURRENT_BOUND_COMMAND_MISSING')
                previous, previous_key = None, None
                continue
            tracks = command['observation']['dynamic_obstacles']
            key = context_identity(snapshot, tracks)
            if identity in rows:
                if (sha256_json({k: v for k, v in snapshot.items() if k != 'snapshot_sha256'}) != identity
                        or recorded['evaluated_command_sha256'] != rows[identity]['command_sha256']):
                    raise ValueError('STABILITY_SNAPSHOT_BINDING_INVALID')
                geometry = reconstruct_geometry(snapshot, tracks)
                if not np.allclose(geometry, rows[identity]['features'], rtol=0, atol=1e-7):
                    raise ValueError('STABILITY_GEOMETRY_CHANGED')
                flight = next(e for e in snapshot['realtime_feature_snapshot']['encodings'] if e['encoder_role']=='flight-state-encoder')
                gyro = 5*flight['features'][13] if flight['valid_mask'][13] == 1 and abs(flight['features'][13]) < 4 else None
                history, history_time = None, None
                if previous is not None and previous_key == key and 0 < source_ms-previous[0]['control_reference_observed_at_unix_ms'] <= 250:
                    history_time = previous[0]['control_reference_observed_at_unix_ms']
                    try:
                        history = reconstruct_geometry(*previous)
                    except ValueError:
                        history_time = None
                # bisect_left 排除同毫秒及未来回执，当前监督动作绝不充当自己的输入。
                index = bisect_left(times, source_ms)-1
                yaw, accepted = None, None
                if index >= 0:
                    application = applications[index]
                    rate = application.get('yaw_rate_application')
                    if (source_ms-times[index] <= 250 and application['transport']=='velocity-ned'
                            and rate is not None and abs(rate['clockwise_rate_dps']) <= 20.
                            and application['command_generated_at_unix_ms'] <= times[index] <= application['command_valid_until_unix_ms']):
                        yaw, accepted = rate['clockwise_rate_dps'], times[index]
                features = encode_heading_context(geometry, source_ms, gyro_flu_z_rad_s=gyro,
                    previous_geometry=history, previous_source_ms=history_time, previous_yaw_dps=yaw, accepted_ms=accepted)
                if identity in extracted:
                    raise ValueError('STABILITY_DUPLICATE_SNAPSHOT')
                extracted[identity] = (features, root, key, source_ms)
            if previous is None or source_ms > previous[0]['control_reference_observed_at_unix_ms']:
                previous, previous_key = (snapshot, tracks), key
        source_inventory[root] = dict(observations=expected['observations'], applications=expected['applications'], commands=expected['commands'])
        print(json.dumps(dict(source=root, matched=len(extracted))), flush=True)
    report = dict(schema='dronedream.heading-stability-input.v1', feature_contract_sha256=HEADING_CONTEXT_SHA256,
        source_data_sha256=manifest['data_sha256'], assembly_sha256=manifest['assembly_sha256'], sources=source_inventory,
        final_test_read=False, new_physical_observations=0, labels_modified=False, splits={})
    for split in ('training', 'validation'):
        raw = [json.loads(line) for line in read_bound(args.data/f'{split}-replay.jsonl', assembly['file_sha256'][f'{split}-replay.jsonl']).splitlines()]
        temporal = {row['source_snapshot_sha256']: row['temporal_evidence'] for row in raw}
        identities = arrays[split+'_identities'].tolist()
        features = np.asarray([extracted[identity][0] for identity in identities], dtype=np.float32)
        previous_index, pair_mask = np.zeros(len(identities), dtype=np.int64), np.zeros(len(identities), dtype=bool)
        latest = {}
        order = sorted(range(len(identities)), key=lambda i: (temporal[identities[i]]['stream_id'], temporal[identities[i]]['observed_at_unix_ms']))
        for index in order:
            identity = identities[index]
            evidence = temporal[identity]
            stream = evidence['stream_id']
            before = latest.get(stream)
            if (before is not None and not evidence['reset_history'] and arrays[split+'_compatible'][index]
                    and arrays[split+'_compatible'][before] and extracted[identity][1:3] == extracted[identities[before]][1:3]
                    and 0 < extracted[identity][3]-extracted[identities[before]][3] <= 250):
                previous_index[index], pair_mask[index] = before, True
            latest[stream] = index
        arrays[split+'_context'], arrays[split+'_previous_index'], arrays[split+'_pair_mask'] = features, previous_index, pair_mask
        report['splits'][split] = dict(windows=len(features), gyro_valid=int(features[:, 9].sum()),
            prior_command_valid=int(features[:, 12].sum()), history_valid=int(features[:, 22].sum()), temporal_pairs=int(pair_mask.sum()))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'stability-inputs.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    report['data_sha256'] = hashlib.sha256((args.output/'stability-inputs.npz').read_bytes()).hexdigest()
    with (args.output/'input-manifest.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report['splits']), flush=True)


if __name__ == '__main__':
    main()
