"""Synthetic offline risk semantics, not a qualification or physical flight."""

import json

import pytest
from test_native_corrections import fixture
from test_native_action_risk_artifacts import dataset, rehash

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.action_risk_artifacts import load_action_risk_dataset
from dronedream_agent_core.training.counterfactual_teacher import CounterfactualConfig
from dronedream_agent_core.training.flight_environment import PilotAction
from dronedream_agent_core.training.risk_clearance_contract import (
    OBSERVED_CLEARANCE_LABELS, clearance_risk_score, observed_required_clearance,
)
from dronedream_agent_core.training.risk_reannotation import reannotate_clearance_labels


# 功能：核对不同净空请求下风险阈值对应实际净空要求，而非要求的一半。
# 输入：required：覆盖狭窄室内、普通通道与开阔场景的测试净空。
# 输出：None；仅执行数值契约断言。
@pytest.mark.parametrize('required', [.05, .0970286546, .15, .25, 1., 4.])
def test_score_boundary_and_monotonicity(required):
    assert clearance_risk_score(required, required) == .5
    assert clearance_risk_score(required * .999, required) > .5
    assert clearance_risk_score(required * 1.001, required) < .5
    assert clearance_risk_score(0., required) == 1.
    assert clearance_risk_score(-.1, required) == 1.
    assert clearance_risk_score(required * 2, required) == 0.
    assert clearance_risk_score(required / 2, required) == .75


# 功能：阻止缺测、饱和、布尔和非有限净空被静默当作默认请求。
# 输入：value：非法特征值。
# 输出：None；断言拒绝有歧义的状态。
@pytest.mark.parametrize('value', [0, -1, 1, 2, True, '0.1', float('nan'), float('inf')])
def test_invalid_or_saturated_clearance_is_not_a_default(value):
    features = [0.] * 46
    features[13] = value
    with pytest.raises(ValueError, match='CONDITION_UNAVAILABLE'):
        observed_required_clearance(features)


# 功能：证明旧配置摘要保持不变，新版本显式改变身份，不能混淆两种监督定义。
# 输入：无。
# 输出：None；断言序列化及身份一致性。
def test_legacy_configuration_identity_is_preserved():
    config = CounterfactualConfig(acceleration_mps2=2., braking_acceleration_mps2=1.)
    legacy = config.model_dump()
    assert 'risk_label_semantics' not in legacy
    assert sha256_json(CounterfactualConfig.model_validate(legacy)) == sha256_json(legacy)
    config.risk_label_semantics = OBSERVED_CLEARANCE_LABELS
    assert config.model_dump()['risk_label_semantics'] == OBSERVED_CLEARANCE_LABELS
    assert sha256_json(config) != sha256_json(legacy)


# 功能：验证新教师仅更改标签定义，保留原始观测、独立几何预测与执行结果。
# 输入：tmp_path：合成回合目录。
# 输出：None；验证请求净空来源和回执绑定。
def test_grounded_teacher_uses_observed_margin_without_rewriting_sources(tmp_path):
    oracle, observation, _, _ = fixture(tmp_path)
    action = PilotAction(mode='pilot-control', axes=[0., 0., 0., 0.])
    original = sha256_json(observation)
    old = oracle.risk(observation, action)
    old_receipt = oracle.receipts[-1]['receipt']
    oracle.teacher.config.risk_label_semantics = OBSERVED_CLEARANCE_LABELS
    new = oracle.risk(observation, action)
    receipt = oracle.receipts[-1]['receipt']
    assert sha256_json(observation) == original
    assert receipt['positions_m'] == old_receipt['positions_m']
    assert receipt['clearance_lower_bound_m'] == old_receipt['clearance_lower_bound_m']
    required = observed_required_clearance(observation.sample.state_features)
    assert receipt['risk_label_contract']['required_clearance_m'] == required
    assert new.risk == clearance_risk_score(receipt['clearance_lower_bound_m'], required)
    assert new.verifier_receipt_sha256 != old.verifier_receipt_sha256


# 功能：即使篡改者同步更新全套摘要，也不能将别的净空条件或评分公式混入训练标签。
# 输入：tmp_path、monkeypatch：隔离数据与命令行；target：注入错误的字段。
# 输出：None；断言数据准入重新计算语义而非仅比较摘要。
@pytest.mark.parametrize('target', ['clearance', 'score'])
def test_admission_recomputes_label_even_when_all_hashes_are_updated(tmp_path, monkeypatch, target):
    root = dataset(tmp_path, monkeypatch)
    load_action_risk_dataset(root)
    rows = [json.loads(line) for line in (root / 'counterfactual-receipts.jsonl').read_text().splitlines()]
    changed = next(row for row in rows if 'receipt' in row)
    old_digest = changed['receipt_sha256']
    if target == 'clearance':
        changed['receipt']['risk_label_contract']['required_clearance_m'] += .1
    else:
        changed['receipt']['risk_target'] = .123456
    changed['receipt_sha256'] = sha256_json(changed['receipt'])
    rehash(root, 'counterfactual-receipts', lambda items: items.__setitem__(slice(None), rows))
    records = [json.loads(line) for line in (root / 'action-records.jsonl').read_text().splitlines()]
    index = next(i for i, row in enumerate(records)
                 if row['assessment']['verifier_receipt_sha256'] == old_digest)
    record = records[index]
    record['assessment']['verifier_receipt_sha256'] = changed['receipt_sha256']
    record['assessment']['risk'] = changed['receipt']['risk_target']
    labels = [json.loads(line) for line in (root / 'action-risk.jsonl').read_text().splitlines()]
    labels[index]['risk_target'] = changed['receipt']['risk_target']
    record['label_sha256'] = sha256_json(labels[index])
    rehash(root, 'action-risk', lambda items: items.__setitem__(slice(None), labels))
    rehash(root, 'action-records', lambda items: items.__setitem__(slice(None), records))
    with pytest.raises(ValueError, match='CLEARANCE_LABEL_MISMATCH'):
        load_action_risk_dataset(root)


# 功能：验证旧数据重新标注保留传感器、物理轨迹和分组，不把派生标签计为新飞行样本。
# 输入：tmp_path、monkeypatch：合成数据目录及命令行工具。
# 输出：None；断言来源不变、迁移可复核且不能覆盖或反复迁移。
def test_reannotation_preserves_source_and_refuses_overwrite(tmp_path, monkeypatch):
    source = dataset(tmp_path, monkeypatch, legacy=True)
    original = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    target = tmp_path / 'reannotated'
    receipt = reannotate_clearance_labels(source, target)
    assert receipt['reannotation']['new_physical_observation_count'] == 0
    assert receipt['reannotation']['changed_sensor_features'] is False
    assert original == {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    a, b = load_action_risk_dataset(source), load_action_risk_dataset(target)
    assert a.observations == b.observations and a.groups == b.groups
    assert a.teacher_config_sha256 != b.teacher_config_sha256
    old = [json.loads(line)['receipt'] for line in original['counterfactual-receipts.jsonl'].splitlines()
           if 'receipt' in json.loads(line)]
    new = [json.loads(line)['receipt'] for line in (target / 'counterfactual-receipts.jsonl').read_text().splitlines()
           if 'receipt' in json.loads(line)]
    for first, second in zip(old, new, strict=True):
        for field in ('positions_m', 'times_seconds', 'clearance_lower_bound_m', 'physical_request', 'state'):
            assert first[field] == second[field]
    with pytest.raises(ValueError, match='DESTINATION_MUST_BE_NEW'):
        reannotate_clearance_labels(source, target)
    with pytest.raises(ValueError, match='LEGACY_SOURCE_REQUIRED'):
        reannotate_clearance_labels(target, tmp_path / 'second')
