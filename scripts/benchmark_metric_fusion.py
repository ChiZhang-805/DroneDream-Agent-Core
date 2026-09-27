"""Synthetic CPU fusion benchmark with exact evidence parity, never a flight qualification."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from unittest.mock import patch

from dronedream_agent_core.contracts import OnboardPerceptionFrame, RangeRayObservation, Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.metric_scan_native import backend
from dronedream_agent_core.perception_runtime import RuntimePerceptionFusion


# 功能：
#   固定多种射线距离的合成环境，分别测量完整融合与快照复制并逐帧比较证据。
# 输入：
#   distance：前方平面距离；native：是否明确启用编译内核。
# 输出：
#   result：计时样本、每帧完整数值证据摘要及证据数量。
def measure(distance, native):
    world = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=-40, y=-40, z=-40),
                           maximum_bound_m=Vector3(x=40, y=40, z=40))
    position = Vector3(x=0., y=0., z=0.)
    fusion = RuntimePerceptionFusion(world=world, accepted_sensor_ids={'front-lidar'})
    times, clone_times, digests = [], [], []
    with patch('dronedream_agent_core.local_world_model.NATIVE_SCAN_AVAILABLE', native):
        for sequence in range(1, 31):
            rays = [RangeRayObservation(origin_m=position,
                endpoint_m=Vector3(x=distance, y=(i % 20 - 10)*.5, z=(i // 20 - 7)*.5),
                hit=i % 7 != 0, confidence=.92 if sequence % 2 else .7,
                observed_at_monotonic_seconds=float(sequence)) for i in range(300)]
            frame = OnboardPerceptionFrame(sensor_id='front-lidar', sequence=sequence,
                observed_at_unix_ms=sequence*1000, localization_position_m=position,
                localization_velocity_mps=position, localization_covariance_m2=.01, range_rays=rays)
            started = time.perf_counter()
            fusion.ingest(frame, now_unix_ms=sequence*1000, now_monotonic_seconds=float(sequence))
            times.append((time.perf_counter()-started)*1000.)
            started = time.perf_counter()
            clone = world.navigation_clone(center_m=position, radius_m=10.)
            clone_times.append((time.perf_counter()-started)*1000.)
            # 时钟、浮点原位、计数和索引全部比较，不只比较体素数量。
            state = [(key, value.log_odds.hex(), value.observations, value.latest_monotonic_seconds.hex())
                     for key, value in world._evidence.items()]
            state.extend([sorted(world._occupied_keys), sorted(world._observed_free_keys),
                          world.observation_count, len(clone._evidence)])
            digests.append(hashlib.sha256(json.dumps(state).encode()).hexdigest())
    result = {'native': native, 'fusion_samples_ms': times[5:], 'clone_samples_ms': clone_times[5:],
        'fusion_median_ms': statistics.median(times[5:]),
        'fusion_p95_ms': sorted(times[5:])[math.ceil(.95*25)-1], 'evidence_sha256': digests,
        'evidence_voxels': len(world._evidence)}
    return result


# 功能：
#   执行固定合成对照并写入独占报告，任何逐帧差异立即失败。
# 输入：
#   output：新的 JSON 输出文件路径。
# 输出：
#   report：三种距离的计时、证据对照和明确的不授予飞行资格标志。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if backend is None:
        raise RuntimeError('BENCHMARK_REQUIRES_BUILT_NATIVE_KERNEL')
    report = {'training_rows_promoted': 0, 'qualified_for_flight': False,
              'native_binary_sha256': hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest(),
              'cases': []}
    for distance in [3., 10., 25.]:
        reference, native = measure(distance, False), measure(distance, True)
        if reference['evidence_sha256'] != native['evidence_sha256']:
            raise ValueError('METRIC_FUSION_EXACT_PARITY_FAILED')
        report['cases'].append({'distance_m': distance, 'reference': reference, 'native': native})
        print(json.dumps({'distance': distance, 'reference_ms': reference['fusion_median_ms'],
                          'native_ms': native['fusion_median_ms'], 'exact_parity': True}), flush=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
