"""Diagnostic report provenance and safe-hold distinction, with synthetic files only."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('heading_handoff_diagnosis',
    Path(__file__).resolve().parents[1] / 'scripts/diagnose_heading_handoff.py')
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)


# 功能：
#   区分无数据、有限延迟与无效指标，不让空实验获得零延迟成绩。
# 输入：
#   无：固定空样本和有限样本。
# 输出：
#   None：统计与拒绝规则通过。
def test_distribution_preserves_empty_and_rejects_invalid():
    assert DIAG.distribution([])['mean'] is None
    assert DIAG.distribution([1., 3., 2.]) == {'count': 3, 'mean': 2., 'p95': 3., 'maximum': 3.}
    for invalid in [float('nan'), float('inf'), True, '1']:
        with pytest.raises(ValueError, match='NONFINITE'):
            DIAG.distribution([invalid])


# 功能：
#   验证调用时间保留毫秒、失败和安全悬停不被改写为模型运动，落地证据缺失仍为未确认。
# 输入：
#   tmp_path：独立临时测试目录；landed：是否提供原生安全落地证据。
# 输出：
#   None：结果、时间差和原证据摘要逐项符合预期。
@pytest.mark.parametrize('landed', [False, True])
def test_summary_retains_failure_holds_and_fractional_timestamps(tmp_path, landed):
    simulation = tmp_path / 'episode-fixture/flight/simulation'
    simulation.mkdir(parents=True)
    (tmp_path / 'bench-report.json').write_text(json.dumps({'model_sha256': 'a' * 64,
        'rows': [{'inference_ms': .2}], 'error': 'HEADING_BENCH_NO_ACCEPTED_MOTION',
        'accepted_motion_decisions': 0, 'accepted_safety_interventions': 1}), encoding='utf-8')
    (simulation / 'model-navigation-model-calls.jsonl').write_text(json.dumps({
        'call_id': 'c1', 'created_at': '1970-01-01T00:00:01.123+00:00', 'latency_ms': 3.}), encoding='utf-8')
    (simulation / 'depth-local-safety-history.jsonl').write_text(json.dumps({
        'command': {'model_call_id': 'c1', 'generated_at_unix_ms': 1140,
                    'decision': {'control_source': 'safe-hold'}},
        'handoff_scheduling': {'sampled_at_unix_ms': 1125}, 'prioritized_ready_control': True,
        'pipeline_phase_ms': {'tracking_and_metric_fusion': 12.5}}), encoding='utf-8')
    if landed:
        (simulation / 'native-terminal-lifecycle.json').write_text(json.dumps({
            'terminal_state': 'ON_GROUND', 'landing_confirmed': True, 'safe_to_stop_watchdog': True}), encoding='utf-8')
    summary = DIAG.summarize(tmp_path)
    assert summary['return_to_command_ms']['mean'] == 17
    assert summary['matched_commands'][0]['return_relative_to_tick_start_ms'] == -2
    assert summary['accepted_motion_decisions'] == 0
    assert summary['accepted_safety_interventions'] == 1
    assert summary['error'] == 'HEADING_BENCH_NO_ACCEPTED_MOTION'
    assert summary['native_landing_confirmed'] is landed
    assert summary['flight_qualification_granted'] is False
    assert len(summary['source_sha256']['bench-report.json']) == 64
    assert summary['pipeline_phase_ms']['tracking_and_metric_fusion']['mean'] == 12.5
    assert summary['metric_scan_kernel'] is None
    assert summary['input_preparation_ms'] == {}
    assert summary['provider_call_ms']['mean'] == 3.
    (simulation / 'model-navigation-timing.jsonl').write_text(json.dumps({
        'input_preparation_ms': {'world_freeze_ms': 0., 'direct_observation_ms': 4.5}}), encoding='utf-8')
    enriched = DIAG.summarize(tmp_path)
    assert enriched['input_preparation_ms']['direct_observation_ms']['mean'] == 4.5
    assert len(enriched['source_sha256']['model-navigation-timing.jsonl']) == 64
    path = simulation / 'depth-local-safety-history.jsonl'
    first = json.loads(path.read_text(encoding='utf-8'))
    later = json.loads(json.dumps(first))
    later['command']['generated_at_unix_ms'] = 1240
    path.write_text(json.dumps(later) + '\n' + json.dumps(first), encoding='utf-8')
    repeated = DIAG.summarize(tmp_path)
    assert repeated['first_return_to_command_ms']['mean'] == 17
    assert repeated['return_to_command_ms']['mean'] == 67
    assert repeated['subsequent_commands_for_same_call'] == 1
    (simulation / 'metric-scan-kernel.json').write_text(json.dumps({'contract': 'unexpected'}), encoding='utf-8')
    with pytest.raises(ValueError, match='KERNEL_IDENTITY_MISMATCH'):
        DIAG.summarize(tmp_path)
