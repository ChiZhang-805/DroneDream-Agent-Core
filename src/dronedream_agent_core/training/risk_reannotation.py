"""Recompute risk-label semantics from preserved nominal geometry receipts.

No sensor values, trajectories, dynamics, route groups or executed outcomes
are changed. Reannotations are not new physical observations or demonstrations.
"""

from pathlib import Path

from ..hashing import sha256_json
from ..plugin_files import check_plain_plugin_path
from .action_risk_artifacts import FILES, load_action_risk_dataset
from .counterfactual_teacher import CounterfactualConfig
from .dagger_artifacts import write_rows
from .evidence_files import decode_evidence_rows, read_evidence_dataset
from .risk_clearance_contract import (
    LEGACY_CLEARANCE_LABELS, OBSERVED_CLEARANCE_LABELS, observation_clearance_label,
)


# 功能：将已经核验的旧风险评分按原请求净空重新标注，保留源文件及完整物理预测证据。
# 输入：source：旧风险数据目录；destination：必须不存在的目标目录。
# 输出：receipt：新标签、来源身份和零新增物理观测声明；不训练、不部署、不授予飞行权限。
def reannotate_clearance_labels(source: Path, destination: Path):
    source, destination = source.absolute(), destination.absolute()
    check_plain_plugin_path(source)
    check_plain_plugin_path(destination)
    if source == destination or source in destination.parents or destination.exists():
        raise ValueError("ACTION_RISK_REANNOTATION_DESTINATION_MUST_BE_NEW")
    dataset = load_action_risk_dataset(source)
    config = CounterfactualConfig.model_validate(dataset.receipt['teacher_config'])
    if config.risk_label_semantics != LEGACY_CLEARANCE_LABELS:
        raise ValueError("ACTION_RISK_REANNOTATION_LEGACY_SOURCE_REQUIRED")
    # 再次读取仍逐字节核对已固定的摘要，不能在检查和写出之间接纳被换掉的原文件。
    contents = read_evidence_dataset(source, FILES, dataset.receipt['file_sha256'],
                                     error_prefix='ACTION_RISK_DATASET_CONTENT_CHANGED')
    rows = {name: decode_evidence_rows(content) for name, content in contents.items()}
    predictions = {row['receipt_sha256']: row for row in rows['counterfactual-receipts']
                   if 'receipt' in row}
    observed = {sha256_json(row): row for row in dataset.observations}
    config.risk_label_semantics = OBSERVED_CLEARANCE_LABELS
    changed = 0
    for label, record in zip(rows['action-risk'], rows['action-records'], strict=True):
        prediction = predictions[record['assessment']['verifier_receipt_sha256']]
        receipt = prediction['receipt']
        bound = receipt['clearance_lower_bound_m']
        # 旧公式也重新计算，防止仅凭自洽摘要把有误的旧评分当成可信迁移来源。
        legacy = max(0., min(1., 1. - bound / config.required_clearance_m))
        if receipt['risk_target'] != legacy:
            raise ValueError("ACTION_RISK_REANNOTATION_LEGACY_SCORE_MISMATCH")
        observation = observed[record['observation_sha256']]
        score, contract = observation_clearance_label(observation.sample.state_features, bound)
        changed += score != label['risk_target']
        receipt['risk_target'] = label['risk_target'] = score
        receipt['config'] = config.model_dump()
        receipt['risk_label_contract'] = contract
        prediction['receipt_sha256'] = sha256_json(receipt)
        record['assessment']['risk'] = score
        record['assessment']['verifier_receipt_sha256'] = prediction['receipt_sha256']
        record['label_sha256'] = sha256_json(label)
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {name: write_rows(destination / FILES[name], value) for name, value in rows.items()}
    from collections import Counter
    receipt = {**dataset.receipt, 'teacher_config': config.model_dump(),
               'file_sha256': hashes,
               'class_counts': dict(Counter('unsafe' if row['risk_target'] >= .5 else 'safe'
                                           for row in rows['action-risk'])),
               'reannotation': dict(kind='observed-clearance-label-migration-v2',
                   source_dataset_receipt_sha256=dataset.receipt_sha256,
                   source_file_sha256=dataset.receipt['file_sha256'],
                   previous_teacher_config=dataset.receipt['teacher_config'],
                   changed_label_count=changed, new_physical_observation_count=0,
                   changed_sensor_features=False, changed_physical_prediction=False,
                   changed_route_groups=False, behavior_cloning_dataset=False)}
    write_rows(destination / 'dataset-receipt.jsonl', [receipt])
    admitted = load_action_risk_dataset(destination)
    if admitted.observations != dataset.observations or admitted.groups != dataset.groups:
        raise ValueError("ACTION_RISK_REANNOTATION_SOURCE_DRIFT")
    return receipt
