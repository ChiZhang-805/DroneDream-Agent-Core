"""Read-only comparison of learned-heading handoff experiments, including failures."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


# 功能：
#   汇总完整数值序列，空序列保持未知，不能把没有动作报告为零延迟。
# 输入：
#   values：同一口径的有限毫秒数值列表。
# 输出：
#   metrics：计数、均值、最近秩 P95 和最大值。
def distribution(values):
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError('HEADING_DIAGNOSIS_NONFINITE_METRIC')
    if not values:
        return {'count': 0, 'mean': None, 'p95': None, 'maximum': None}
    ordered = sorted(values)
    metrics = {'count': len(values), 'mean': mean(values),
               'p95': ordered[max(0, (95 * len(values) + 99) // 100 - 1)],
               'maximum': ordered[-1]}
    return metrics


# 功能：
#   读取一个独占台架目录的原始证据并计算摘要，对齐调用返回、调度及命令的真实时钟。
#   安全介入后的悬停与模型运动分开计数，任何离线统计均不授予飞行资格。
# 输入：
#   run：已结束台架目录。
# 输出：
#   result：该次成功或失败的汇总、原文件摘要及各段延迟。
def summarize(run):
    report_path = run / 'bench-report.json'
    report_bytes = report_path.read_bytes()
    bench = json.loads(report_bytes)
    episodes = [item for item in run.glob('episode-*') if item.is_dir()]
    if len(episodes) != 1:
        raise ValueError('HEADING_DIAGNOSIS_EPISODE_AMBIGUOUS')
    simulation = episodes[0] / 'flight/simulation'
    sources = {'bench-report.json': hashlib.sha256(report_bytes).hexdigest()}

    # 功能：
    #   加载实际存在的单个证据文件，缺失时明确返回空，不伪造落地或调用记录。
    # 输入：
    #   name：固定证据文件名；lines：是否为 JSONL。
    # 输出：
    #   data：解析记录或缺失时的空值。
    def read(name, *, lines=False):
        path = simulation / name
        if not path.is_file():
            return [] if lines else None
        raw = path.read_bytes()
        sources[name] = hashlib.sha256(raw).hexdigest()
        data = [json.loads(line) for line in raw.splitlines()] if lines else json.loads(raw)
        return data

    calls = {row['call_id']: row for row in read('model-navigation-model-calls.jsonl', lines=True)}
    records = read('depth-local-safety-history.jsonl', lines=True)
    preparation = read('model-navigation-timing.jsonl', lines=True)
    lifecycle = read('native-terminal-lifecycle.json')
    kernel = read('metric-scan-kernel.json')
    if kernel is not None and kernel != bench.get('metric_scan_kernel'):
        raise ValueError('HEADING_DIAGNOSIS_KERNEL_IDENTITY_MISMATCH')
    phase_names = sorted({name for row in records for name in (row.get('pipeline_phase_ms') or {})})
    phases = {name: distribution([row['pipeline_phase_ms'][name] for row in records
              if name in (row.get('pipeline_phase_ms') or {})]) for name in phase_names}
    preparation_names = sorted({name for row in preparation for name in row.get('input_preparation_ms', {})})
    preparation_metrics = {name: distribution([row['input_preparation_ms'][name] for row in preparation
        if name in row.get('input_preparation_ms', {})]) for name in preparation_names}
    pairs = []
    for row in records:
        command = row.get('command') or {}
        call = calls.get(command.get('model_call_id'))
        if call is None:
            continue
        stamp = datetime.fromisoformat(call['created_at'])
        if stamp.tzinfo is None:
            raise ValueError('HEADING_DIAGNOSIS_CALL_TIMEZONE_REQUIRED')
        returned = int(stamp.timestamp() * 1000)
        timing = row.get('handoff_scheduling') or {}
        pairs.append({'call_id': call['call_id'], 'provider_latency_ms': call['latency_ms'],
            'return_to_command_ms': command['generated_at_unix_ms'] - returned,
            'return_relative_to_tick_start_ms': returned - timing['sampled_at_unix_ms']
                if timing else None,
            'priority_handoff': row['prioritized_ready_control'],
            'control_source': command['decision']['control_source']})
    # 一个有效租期可能产生多条执行命令；首次交接与后续保持命令分开，避免把保持时间当推理延迟。
    first_handoffs = {}
    for pair in pairs:
        identity = pair['call_id']
        latency = pair['return_to_command_ms']
        first_handoffs[identity] = min(first_handoffs.get(identity, latency), latency)
    result = {'run': run.name, 'model_sha256': bench['model_sha256'],
        'inputs': len(bench['rows']), 'error': bench['error'],
        'accepted_motion_decisions': bench.get('accepted_motion_decisions', 0),
        'accepted_safety_interventions': bench.get('accepted_safety_interventions', 0),
        'metric_scan_kernel': kernel,
        'pipeline_phase_ms': phases,
        'input_preparation_ms': preparation_metrics,
        'provider_call_ms': distribution([call['latency_ms'] for call in calls.values()]),
        'native_landing_confirmed': bool(lifecycle and lifecycle.get('terminal_state') == 'ON_GROUND'
            and lifecycle.get('landing_confirmed') is True and lifecycle.get('safe_to_stop_watchdog') is True),
        'input_inference_ms': distribution([row['inference_ms'] for row in bench['rows']]),
        'return_to_command_ms': distribution([row['return_to_command_ms'] for row in pairs]),
        'first_return_to_command_ms': distribution(list(first_handoffs.values())),
        'subsequent_commands_for_same_call': len(pairs) - len(first_handoffs),
        'priority_handoffs': sum(row['prioritized_ready_control'] for row in records),
        'matched_commands': pairs, 'source_sha256': sources, 'flight_qualification_granted': False}
    return result


# 功能：
#   对明确列出的所有实验统一汇总，不自动挑选成功运行或覆盖既有统计文件。
# 输入：
#   runs：命令行传入的已结束目录；output：独占的新 JSON 输出路径。
# 输出：
#   report：包含失败、证据摘要和统计口径的对比文件。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'purpose': 'diagnostic-only; safety holds are not learned motion',
        'training_rows_promoted': 0, 'flight_qualification_granted': False,
        'runs': [summarize(run) for run in args.runs]}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps([{key: value for key, value in row.items()
                      if key not in ('matched_commands', 'source_sha256')} for row in report['runs']],
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
