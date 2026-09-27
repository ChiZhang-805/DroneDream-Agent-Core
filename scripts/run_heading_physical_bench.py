"""Bounded learned-yaw bench through the existing guarded PX4/Gazebo training port.

Translation requests are explicitly zero; this tests physical yaw response and
history-loss handling, not navigation, delivery completion, or full model flight qualification.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from dronedream_agent_core.live_heading_context import LiveHeadingContext
from dronedream_agent_core.metric_scan_native import backend as metric_backend
from dronedream_agent_core.training.flight_environment import ControlPreparationExpired, PilotAction
from dronedream_agent_core.training.px4_environment import Px4GazeboTrainingEnvironment, Px4TrainingConfig


# 功能：
#   在现有仿真训练端口运行短时学习偏航测试，记录真实观测与接受证据，最后请求安全落地。
# 输入：
#   args：已固定路线和模型的实验目录，以及新的运行目录名。
# 输出：
#   report：每次实际输入和候选、关闭状态与回合位置，不把发送请求视为成功飞行。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--expert-role', default='local-navigation-policy', choices=['local-navigation-policy', 'precision-maneuver-policy'])
    args = parser.parse_args()
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.run_name):
        raise ValueError('HEADING_BENCH_RUN_NAME_INVALID')
    root = Path('/mnt/q/DroneDream-Workspace')
    repository = Path(__file__).resolve().parents[1]
    directory = args.directory.resolve()
    evaluation = json.loads((directory/'evaluation.json').read_text())
    selected = evaluation['selected']
    if selected not in ('availability', 'age-weighted') or not evaluation['candidates'][selected]['eligible']:
        raise ValueError('HEADING_BENCH_REQUIRES_ELIGIBLE_RESEARCH_CANDIDATE')
    graph = (directory/(selected+'.onnx')).read_bytes()
    digest = hashlib.sha256(graph).hexdigest()
    if digest != evaluation['candidates'][selected]['sha256']:
        raise ValueError('HEADING_BENCH_MODEL_CHANGED')
    assets = root/'Build/Native-Learning-Inputs-20260907'
    paths = dict(route=directory/'yaw-bench-route.json',
        semantic=root/'TestRuns/local-realtime-control-20260910/native-motion-calibration-inputs/semantic.json',
        world_sdf=assets/'map/normalized/map/gazebo/world.sdf',
        vehicle_sdf=assets/'vehicle/normalized/vehicle/gazebo/model.sdf',
        vehicle=assets/'vehicle/normalized/vehicle/vehicle.json',
        controller_params=assets/'vehicle/normalized/vehicle/controller_params.json')
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    output = directory/args.run_name
    output.mkdir(exist_ok=False)
    config = Px4TrainingConfig(runner=repository/'scripts/run_school_map_depth_qualification.py', output_root=output,
        mission_id='heading-availability-physical-bench', expert_role=args.expert_role, **paths, asset_sha256=hashes,
        minimum_enu_m=(-13., -5., 0.), maximum_enu_m=(-7., 3., 4.), speed_limit_mps=.4, required_clearance_m=1.,
        episode_steps=160, startup_timeout_seconds=300, quiesce_timeout_seconds=180,
        initial_collection_mode='stream-imitation',
        batch_static_world_visuals=True,
        native_sensor_runtime=root/'Build/Release-Acceptance-20260916/native-sensors')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(graph, sess_options=options, providers=['CPUExecutionProvider'])
    # 在起飞前触发推理运行库初始化；不让首帧付出线程/算子初始化成本。
    # 零输入只用于加载预热，不作为训练记录或控制请求。
    session.run(None, {'heading_context': np.zeros((1, 23), dtype=np.float32)})
    state = LiveHeadingContext()
    env = Px4GazeboTrainingEnvironment(config)
    env.bind_policy_identity(digest)
    report = dict(model_sha256=digest, simulation_only=True, qualified_for_flight=False, translation_request='zero velocity',
        purpose='physical-yaw-and-acknowledgement-loss-bench', mask_seconds=[4., 7.], rows=[], error=None, safely_closed=False)
    report['metric_scan_kernel'] = ({'contract': metric_backend.CONTRACT,
        'sha256': hashlib.sha256(Path(metric_backend.__file__).read_bytes()).hexdigest()}
        if metric_backend is not None else {'implementation': 'numpy-python'})
    try:
        observation = env.reset(seed=806)
        started = time.monotonic()
        while len(report['rows']) < 160 and time.monotonic()-started < 12.:
            elapsed = time.monotonic()-started
            context_started = time.perf_counter()
            snapshot, receipt = env.current_control_context()
            context_ms = (time.perf_counter()-context_started)*1000.
            masked = 4. <= elapsed < 7.
            encoding_started = time.perf_counter()
            features = state.encode(snapshot, observation.sample.temporal_evidence.stream_id, receipt,
                now_unix_ms=int(time.time()*1000), mask_command=masked)
            encoding_ms = (time.perf_counter()-encoding_started)*1000.
            inference_started = time.perf_counter()
            yaw = float(session.run(None, {'heading_context': np.asarray([features], dtype=np.float32)})[0][0, 0])
            inference_ms = (time.perf_counter()-inference_started)*1000.
            row = dict(elapsed=elapsed, source_snapshot_sha256=snapshot['snapshot_sha256'],
                source_ms=snapshot['control_reference_observed_at_unix_ms'], current_position=snapshot['current_position_m'],
                heading_features=list(features), proposed_yaw_axis=yaw, injected_command_mask=masked, submitted=False,
                context_ms=context_ms, encoding_ms=encoding_ms, inference_ms=inference_ms)
            report['rows'].append(row)
            submission_started = time.perf_counter()
            try:
                env.submit_stream_action(PilotAction(mode='pilot-control', axes=[0., 0., 0., yaw]))
                row['submitted'] = True
            except ControlPreparationExpired:
                row['expired_before_submission'] = True
            finally:
                row['submission_ms'] = (time.perf_counter()-submission_started)*1000.
            if time.monotonic()-started >= 12.:
                break
            try:
                observation = env.next_stream_observation(deadline=min(started+12., time.monotonic()+1.9))
            except TimeoutError:
                if time.monotonic() >= started+12.:
                    report['duration_elapsed'] = True
                    break
                raise
    except Exception as error:
        report['error'] = type(error).__name__+': '+str(error)
        print(report['error'], flush=True)
    finally:
        try:
            env.close()
            report['safely_closed'] = True
            # 只有原生落地确认之后才连接提案、命令和执行回执；不将已发送冒充已执行。
            if any(row['submitted'] for row in report['rows']):
                visits = env.finalize_stream_captures()
                report['actually_accepted_decisions'] = len(visits)
                report['accepted_safety_interventions'] = sum(visit.safety_intervened for visit in visits)
                report['accepted_motion_decisions'] = sum(
                    visit.applied_action.mode == 'pilot-control' and any(abs(axis) > 1e-6 for axis in visit.applied_action.axes)
                    for visit in visits)
            else:
                report['actually_accepted_decisions'] = 0
                report['accepted_motion_decisions'] = 0
            if not report['accepted_motion_decisions'] and report['error'] is None:
                report['error'] = 'HEADING_BENCH_NO_ACCEPTED_MOTION'
        except Exception as error:
            report['close_error'] = type(error).__name__+': '+str(error)
            print(report['close_error'], flush=True)
        report['episode_path'] = str(env.episode_path) if env.episode_path else None
        with (output/'bench-report.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(dict(rows=len(report['rows']), error=report['error'], safely_closed=report['safely_closed'], episode=report['episode_path'])), flush=True)
    if report['error'] or not report['safely_closed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
