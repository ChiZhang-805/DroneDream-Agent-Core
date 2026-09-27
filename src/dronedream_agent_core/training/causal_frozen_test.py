"""Evaluate a separately frozen candidate on test sources; never choose or optimize weights."""

import hashlib
import io
from pathlib import Path

from dronedream_plugin_sdk.protocol import decode_json

from ..local_expert_harness import NAVIGATION_EXPERT_ROLES
from ..local_policy_training import LocalPolicyObservation, LocalPolicyTrainingSample
from ..plugin_files import read_plugin_file
from .causal_export_validation import evaluate_causal_export
from .causal_policy import causal_examples, load_causal_checkpoint
from .causal_replay import (
    MAX_REPLAY_RECORD_BYTES,
    MAX_REPLAY_RECORDS,
    REPLAY_FILES,
    decode_replay,
    protected_validation_groups,
    read_bound_replay,
    require_unseen_validation,
    validate_split_sources,
)
from .causal_training_inputs import training_cpu_threads
from .control_shards import verify_assembled_control_data
from .mission_groups import MissionGroupManifest, merge_mission_group_manifests
from .visual_lineage import (
    require_matching_visual_input,
    verify_causal_checkpoint_receipt,
    verify_visual_split,
)


# 功能：
#   读取并绑定预先冻结的普通文件字节，摘要不匹配时禁止进入最终测试。
# 输入：
#   path、expected、limit：明确文件、独立保存的 SHA-256 和字节上限。
# 输出：
#   content：实际校验过的同一份字节，不重新按路径替换内容。
def frozen_bytes(path, expected, limit):
    if type(expected) is not str or len(expected) != 64 or set(expected) - set('0123456789abcdef'):
        raise ValueError('CAUSAL_FROZEN_TEST_DIGEST_INVALID')
    content = read_plugin_file(path, limit=limit)
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('CAUSAL_FROZEN_TEST_INPUT_CHANGED:' + path.name)
    return content


# 功能：
#   从有界原始 JSONL 逐行严格解析观测或标签，保留 JSON 模式的 dataclass 转换而不放宽字段类型。
# 输入：
#   content、contract：已绑定字节和当前观测/标签契约。
# 输出：
#   rows：完整且没有重复键、截断行或超预算记录的模型对象列表。
def frozen_rows(content, contract):
    rows = []
    with io.BytesIO(content) as stream:
        while line := stream.readline(MAX_REPLAY_RECORD_BYTES + 1):
            if len(rows) >= MAX_REPLAY_RECORDS or len(line) > MAX_REPLAY_RECORD_BYTES or not line.endswith(b'\n'):
                raise ValueError('CAUSAL_FROZEN_TEST_RECORD_BUDGET_OR_INCOMPLETE')
            if not isinstance(decode_json(line, limit=MAX_REPLAY_RECORD_BYTES), dict):
                raise ValueError('CAUSAL_FROZEN_TEST_RECORD_INVALID')
            rows.append(contract.model_validate_json(line, strict=True))
    return rows


# 功能：
#   1. 按外部冻结的训练回执和数据摘要，核对实际权重、视觉身份及全部训练/调参/祖先来源隔离。
#   2. 在最终测试集运行固定 CPU 权重和 ONNX，不更新参数、不选择候选、不授予飞行权限。
# 输入：
#   candidate：已训练单专家目录。
#   dataset：三集合完整组装目录。
#   training_receipt_sha256、assembly_sha256：读取测试预测前独立冻结的两个摘要。
# 输出：
#   report：明确标记 test 的实际部署输出、误差、来源与内容身份。
def evaluate_frozen_causal_candidate(candidate: Path, dataset: Path, training_receipt_sha256: str, assembly_sha256: str):
    receipt_raw = frozen_bytes(candidate / 'training-receipt.json', training_receipt_sha256, 4 * 1024**2)
    receipt = decode_json(receipt_raw, limit=4 * 1024**2)
    if type(receipt) is not dict or receipt.get('expert_role') not in NAVIGATION_EXPERT_ROLES:
        raise ValueError('CAUSAL_FROZEN_TEST_ROLE_INVALID')
    role = receipt['expert_role']
    checkpoint = frozen_bytes(candidate / (role + '.pt'), receipt.get('checkpoint_sha256'), 256 * 1024**2)
    visual = verify_causal_checkpoint_receipt(checkpoint, receipt_raw, expert_role=role)
    graph = frozen_bytes(candidate / (role + '.onnx'), receipt.get('artifact_sha256'), 256 * 1024**2)
    # 整包搬运检查后，下面仍按同一冻结回执重新绑定实际消费字节，防止两次读取之间替换。
    verify_assembled_control_data(dataset, assembly_sha256)
    assembly_raw = frozen_bytes(dataset / 'assembly-receipt.json', assembly_sha256, 4 * 1024**2)
    assembly = decode_json(assembly_raw, limit=4 * 1024**2)
    hashes = assembly['file_sha256']
    expected_inputs = {'train': 'training-replay.jsonl', 'validation': 'validation-replay.jsonl',
        'training_observations': 'training-observations.jsonl',
        'validation_observations': 'validation-observations.jsonl', 'stream_groups': 'stream-groups.json'}
    if any(receipt.get('input_sha256', {}).get(name) != hashes[filename]
           for name, filename in expected_inputs.items()):
        raise ValueError('CAUSAL_FROZEN_TEST_WRONG_TRAINING_ASSEMBLY')
    replay = read_bound_replay({name: candidate / filename for name, filename in REPLAY_FILES.items()}, receipt)
    decoded, prior_manifest, _ = decode_replay(replay)
    manifest_raw = frozen_bytes(dataset / 'stream-groups.json', hashes['stream-groups.json'], 64 * 1024**2)
    manifest = MissionGroupManifest.model_validate(decode_json(manifest_raw, limit=64 * 1024**2))
    manifest = merge_mission_group_manifests(prior_manifest, manifest)
    remaining = 512 * 1024**2 - sum(len(content) for content in replay.values()) - len(manifest_raw)
    if remaining <= 0:
        raise ValueError('CAUSAL_FROZEN_TEST_BYTE_BUDGET')
    labels_raw = frozen_bytes(dataset / 'test-replay.jsonl', hashes['test-replay.jsonl'], min(256 * 1024**2, remaining))
    remaining -= len(labels_raw)
    if remaining <= 0:
        raise ValueError('CAUSAL_FROZEN_TEST_BYTE_BUDGET')
    history_raw = frozen_bytes(dataset / 'test-observations.jsonl', hashes['test-observations.jsonl'], min(256 * 1024**2, remaining))
    labels = frozen_rows(labels_raw, LocalPolicyTrainingSample)
    history = frozen_rows(history_raw, LocalPolicyObservation)
    groups = validate_split_sources(
        [row for rows in decoded.values() for row in rows], [*labels, *history], manifest)[1]
    # 当前回放之外仍可能存在迁移祖先；历史训练组和历史调参组都不能变成最终测试。
    require_unseen_validation(receipt, groups)
    if protected_validation_groups(receipt) & groups:
        raise ValueError('CAUSAL_FROZEN_TEST_PREVIOUSLY_USED_FOR_TUNING')
    with training_cpu_threads(1):
        model = load_causal_checkpoint(checkpoint)
        if receipt.get('config') != model.config.model_dump():
            raise ValueError('CAUSAL_FROZEN_TEST_CONFIG_MISMATCH')
        if model.config.visual_feature_count:
            encoding = frozen_bytes(dataset / 'test-visual-receipt.json', hashes['test-visual-receipt.json'], 4 * 1024**2)
            contract = verify_visual_split(labels_raw, encoding, feature_count=model.config.visual_feature_count)
            require_matching_visual_input(contract, visual)
        examples = causal_examples(labels, stream_groups=manifest.groups,
            history_length=model.config.history_length, navigation_role=role, history_observations=history)
        report = evaluate_causal_export(model, graph, examples)
    # 共用数值计算，不将 test 结果命名成 validation，从输出层防止后续误用。
    torch_test = report.pop('torch_validation')
    report['torch_test'] = {key.replace('validation_', 'test_', 1): value for key, value in torch_test.items()}
    report['test_groups'] = report.pop('validation_groups')
    report.update(schema_version='dronedream.causal-frozen-test.v1', evaluation_split='test',
        expert_role=role, training_receipt_sha256=training_receipt_sha256,
        assembly_sha256=assembly_sha256, checkpoint_sha256=receipt['checkpoint_sha256'],
        onnx_sha256=receipt['artifact_sha256'], test_set_read=True, weights_modified=False,
        candidate_selection_performed=False, protected_test_source_groups=sorted(groups))
    return report
