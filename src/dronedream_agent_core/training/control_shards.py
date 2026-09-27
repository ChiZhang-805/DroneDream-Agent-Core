"""Lossless, content-bound assembly of executed control shards and frozen holdouts."""

import hashlib
import itertools
import json
from pathlib import Path

from dronedream_plugin_sdk.protocol import decode_json

from ..local_expert_harness import NAVIGATION_EXPERT_ROLES
from ..local_policy_training import LocalPolicyObservation, parse_training_samples
from ..plugin_files import check_plain_plugin_path, read_plugin_file
from .causal_replay import MAX_REPLAY_FILE_BYTES, MAX_REPLAY_TOTAL_BYTES
from .demonstrations import DemonstrationCorpus, validate_demonstration_splits
from .evidence_files import decode_evidence_rows
from .evidence_publication import publish_evidence_bytes, write_evidence_object
from .mission_groups import MissionGroupManifest, merge_mission_group_manifests
from .visual_lineage import verify_visual_split

SPLITS = ('training', 'validation', 'test')


# 功能：
#   按另存的组装回执摘要复核可搬运数据目录，不读取回执中的来源绝对路径或授予训练质量资格。
# 输入：
#   directory：本地或上传后的完整组装目录。
#   expected_sha256：传输前独立保存的 assembly-receipt.json 摘要。
# 输出：
#   report：实际文件数、字节数和完整性结果。
def verify_assembled_control_data(directory: Path, expected_sha256: str):
    directory = directory.absolute()
    check_plain_plugin_path(directory)
    receipt_path = directory / 'assembly-receipt.json'
    receipt_bytes = _bound(receipt_path, expected_sha256, 4 * 1024 * 1024)
    receipt = decode_json(receipt_bytes, limit=4 * 1024 * 1024)
    names = {'stream-groups.json'} | {
        f'{split}-{suffix}' for split in SPLITS
        for suffix in ('replay.jsonl', 'observations.jsonl', 'visual-receipt.json')}
    if (type(receipt) is not dict
            or receipt.get('schema_version') != 'dronedream.control-shard-assembly.v1'
            or type(receipt.get('file_sha256')) is not dict
            or set(receipt['file_sha256']) != names
            or receipt.get('weights_trained') is not False
            or receipt.get('qualified_for_flight') is not False):
        raise ValueError('CONTROL_TRANSFER_RECEIPT_INVALID')
    # 只允许这十份已绑定输入和回执；多出来的旧目录、脚本或数据不能静默混入。
    expected_names = names | {'assembly-receipt.json'}
    seen = set()
    for path in directory.iterdir():
        if path.name not in expected_names:
            raise ValueError('CONTROL_TRANSFER_UNLISTED_PATH:' + path.name)
        check_plain_plugin_path(path)
        if not path.is_file():
            raise ValueError('CONTROL_TRANSFER_FILE_REQUIRED:' + path.name)
        seen.add(path.name)
    if seen != expected_names:
        raise ValueError('CONTROL_TRANSFER_FILE_MISSING')
    total_bytes = 0
    for name in sorted(names):
        content = _bound(directory / name, receipt['file_sha256'][name],
                         min(MAX_REPLAY_FILE_BYTES, MAX_REPLAY_TOTAL_BYTES - total_bytes))
        total_bytes += len(content)
    # 不把一次完整性检查当成文件永久锁定；训练入口仍须重新读取并校验自身输入。
    _bound(receipt_path, expected_sha256, 4 * 1024 * 1024)
    report = dict(schema_version='dronedream.control-transfer-verification.v1',
                  verified=True, assembly_sha256=expected_sha256, data_files=len(names),
                  data_bytes=total_bytes, weights_trained=False, qualified_for_flight=False)
    return report


# 功能：
#   从普通文件读取固定字节并复核明确摘要，拒绝链接、读取漂移与内容替换。
# 输入：
#   path：输入路径；digest：清单中原始 SHA-256；limit：最大读取字节数。
# 输出：
#   content：通过身份核验的同一份字节，下游不重新打开路径。
def _bound(path, digest, limit):
    if type(digest) is not str or len(digest) != 64 or set(digest) - set('0123456789abcdef'):
        raise ValueError('CONTROL_ASSEMBLY_DIGEST_INVALID')
    content = read_plugin_file(path, limit=limit)
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError('CONTROL_ASSEMBLY_SOURCE_CHANGED:' + path.name)
    return content


# 功能：
#   原字节串接同一划分的视觉分片，保留每份原编码回执而非冒充新的推理回执。
# 输入：
#   contents：已验证的非空分片字节列表；receipts：对应原编码回执对象。
#   feature_count：共同视觉维数。
# 输出：
#   content、receipt：可逐段重验的组合字节与组合回执。
def compose_visual_shards(contents, receipts, feature_count):
    if (type(contents) is not list or type(receipts) is not list
            or not 1 <= len(contents) <= 64 or len(contents) != len(receipts)
            or any(type(part) is not bytes for part in contents)
            or any(type(item) is not dict or type(item.get('sample_count')) is not int
                   or item['sample_count'] < 1 for item in receipts)
            or sum(map(len, contents)) > MAX_REPLAY_FILE_BYTES):
        raise ValueError('CONTROL_ASSEMBLY_VISUAL_PARTS_INVALID')
    content = b''.join(contents)
    receipt = dict(schema_version='dronedream.visual-shard-composition.v1',
        output_sha256=hashlib.sha256(content).hexdigest(),
        sample_count=sum(item.get('sample_count', 0) for item in receipts if type(item) is dict),
        qualification_granted=False,
        segments=[dict(byte_length=len(part), encoding_receipt=item)
                  for part, item in zip(contents, receipts, strict=True)])
    verify_visual_split(content, json.dumps(receipt, allow_nan=False).encode(), feature_count=feature_count)
    return content, receipt


# 功能：
#   1. 按冻结清单逐字节核验三类分片，原样合并样本与历史，不重标记、不插帧。
#   2. 重新计算真实因果窗口，逐对检查路线、流、图像与来源隔离，不信任旧计数代替检查。
#   3. 全部核验后才独占发布；最终回执最后写入，失败现场不得被当作已完成训练包。
# 输入：
#   inventory_path：显式控制分片清单；output：尚不存在的新目录。
#   feature_count：本次视觉输入维数，必须与每份实际编码回执一致。
# 输出：
#   report：全部文件摘要、真实窗口及缺口；不授予模型训练结果或飞行资格。
def assemble_control_shards(inventory_path: Path, output: Path, *, feature_count: int):
    # 传输完整性检查不需要训练框架；仅真实重建学习窗口时加载 Torch 相关实现。
    from .causal_policy import causal_examples
    from .control_coverage import control_window_coverage

    output = output.absolute()
    check_plain_plugin_path(output)
    if output.exists():
        raise FileExistsError(output)
    inventory_bytes = read_plugin_file(inventory_path, limit=4 * 1024 * 1024)
    inventory = decode_json(inventory_bytes, limit=4 * 1024 * 1024)
    if (type(inventory) is not dict
            or inventory.get('schema_version') != 'dronedream.control-preparation-inventory.v1'
            or type(inventory.get('shards')) is not list or not 3 <= len(inventory['shards']) <= 64):
        raise ValueError('CONTROL_ASSEMBLY_INVENTORY_INVALID')
    collected = {name: dict(data=[], history=[], receipts=[], manifests=[]) for name in SPLITS}
    total_bytes, raw_bytes, visual_identity = 0, 0, None
    for shard in inventory['shards']:
        if type(shard) is not dict or shard.get('split') not in SPLITS:
            raise ValueError('CONTROL_ASSEMBLY_SPLIT_INVALID')
        split, directory = shard['split'], Path(shard['directory'])
        expected = shard['file_sha256']
        if type(expected) is not dict or set(expected) != {'raw', 'encoded', 'history', 'groups'}:
            raise ValueError('CONTROL_ASSEMBLY_CONTENT_BINDING_INVALID')
        receipt_bytes = _bound(directory / f'{split}-visual-receipt.json',
                              shard['visual_receipt_sha256'], 4 * 1024 * 1024)
        encoded = decode_json(receipt_bytes, limit=4 * 1024 * 1024)
        if type(encoded) is not dict or encoded.get('source_policy_data_sha256') != expected['raw']:
            raise ValueError('CONTROL_ASSEMBLY_ORIGINAL_LABEL_BINDING_INVALID')
        parts = {}
        for name, filename in [('raw', f'{split}.jsonl'), ('encoded', f'{split}-visual.jsonl'),
                               ('history', f'{split}-observations.jsonl'), ('groups', 'stream-groups.json')]:
            # 原标签仅用于比对且不会导出，独立限制其累计字节；不能把原标签与
            # 编码标签重复计入输出预算，导致仍可被训练入口读取的数据提前被拒绝。
            consumed = raw_bytes if name == 'raw' else total_bytes
            parts[name] = _bound(directory / filename, expected[name],
                                 min(MAX_REPLAY_FILE_BYTES, MAX_REPLAY_TOTAL_BYTES - consumed))
            if name == 'raw':
                raw_bytes += len(parts[name])
            else:
                total_bytes += len(parts[name])
        contract = verify_visual_split(parts['encoded'], receipt_bytes, feature_count=feature_count)
        originals = parse_training_samples(parts['raw'])
        encoded_samples = parse_training_samples(parts['encoded'])
        # 视觉推理只允许替换特征向量，不能借重新编码改写教师动作、来源或时序标签。
        if len(originals) != len(encoded_samples) or any(
            original.model_dump(exclude={'visual_features'}) != encoded_row.model_dump(exclude={'visual_features'})
            for original, encoded_row in zip(originals, encoded_samples, strict=True)
        ):
            raise ValueError('CONTROL_ASSEMBLY_ENCODING_CHANGED_SUPERVISION')
        del originals, encoded_samples
        del parts['raw']
        if visual_identity is not None and contract != visual_identity:
            raise ValueError('CONTROL_ASSEMBLY_VISUAL_IDENTITY_MISMATCH')
        if contract.perception_encoder_sha256 != inventory.get('visual_encoder_sha256'):
            raise ValueError('CONTROL_ASSEMBLY_INDEX_ENCODER_MISMATCH')
        visual_identity = contract
        # 原数据必须完整换行；串接不修补截断、不排序改变因果顺序。
        for name in ('encoded', 'history'):
            if not parts[name] or not parts[name].endswith(b'\n'):
                raise ValueError('CONTROL_ASSEMBLY_INCOMPLETE_SHARD')
        group = MissionGroupManifest.model_validate_json(parts['groups'])
        selected = collected[split]
        selected['data'].append(parts['encoded'])
        selected['history'].append(parts['history'])
        selected['receipts'].append(encoded)
        selected['manifests'].append(group)
    if any(not collected[name]['data'] for name in SPLITS):
        raise ValueError('CONTROL_ASSEMBLY_REQUIRES_THREE_SPLITS')
    outputs, corpora, groups, counts = {}, {}, [], {}
    for split, item in collected.items():
        data, receipt = compose_visual_shards(item['data'], item['receipts'], feature_count)
        history_content = b''.join(item['history'])
        if len(history_content) > MAX_REPLAY_FILE_BYTES:
            raise ValueError('CONTROL_ASSEMBLY_HISTORY_TOO_LARGE')
        samples = parse_training_samples(data)
        # JSON 严格模式允许按合同重建嵌套数据类；Python strict 模式会误要求
        # pilot_control_limits 已是数据类实例，而磁盘上的合法值只能是 JSON 对象。
        history = [LocalPolicyObservation.model_validate_json(json.dumps(row, allow_nan=False), strict=True)
                   for row in decode_evidence_rows(history_content)]
        if any(row.temporal_evidence is None for row in [*samples, *history]):
            raise ValueError('CONTROL_ASSEMBLY_TEMPORAL_SOURCE_REQUIRED')
        if any(len(row.visual_features) != feature_count for row in samples):
            raise ValueError('CONTROL_ASSEMBLY_VISUAL_WIDTH_MISMATCH')
        manifest = merge_mission_group_manifests(*item['manifests'])
        streams = {row.temporal_evidence.stream_id for row in [*samples, *history]}
        if streams - manifest.groups.keys():
            raise ValueError('CONTROL_ASSEMBLY_MISSING_STREAM_GROUP')
        selected_groups = {stream: manifest.groups[stream] for stream in streams}
        manifest = MissionGroupManifest(groups=selected_groups, evidence=manifest.evidence)
        groups.append(manifest)
        # 标签与自己的历史允许同源；每个集合内部重复行不得提升样本或窗口数量。
        for rows in (samples, history):
            identities = [row.temporal_evidence.sample_sha256 for row in rows]
            if len(set(identities)) != len(identities):
                raise ValueError('CONTROL_ASSEMBLY_DUPLICATE_SOURCE')
        corpora[split] = DemonstrationCorpus(samples, selected_groups, [], {}, history)
        roles, coverage = {}, {}
        for role in NAVIGATION_EXPERT_ROLES:
            try:
                examples = causal_examples(samples, stream_groups=selected_groups,
                    navigation_role=role, history_observations=history)
                coverage[role] = control_window_coverage(examples)
                roles[role] = len(examples)
            except ValueError as error:
                if str(error) != 'CAUSAL_TRAINING_HAS_NO_COMPLETE_WINDOWS':
                    raise
                roles[role] = 0
                coverage[role] = control_window_coverage([])
        counts[split] = dict(samples=len(samples), observations=len(history), windows=roles,
                             window_coverage=coverage)
        outputs[f'{split}-replay.jsonl'] = data
        outputs[f'{split}-observations.jsonl'] = history_content
        outputs[f'{split}-visual-receipt.json'] = json.dumps(receipt, allow_nan=False).encode()
    for left, right in itertools.combinations(SPLITS, 2):
        validate_demonstration_splits(corpora[left], corpora[right])
    merged = merge_mission_group_manifests(*groups)
    outputs['stream-groups.json'] = merged.model_dump_json().encode()
    if sum(map(len, outputs.values())) > MAX_REPLAY_TOTAL_BYTES:
        raise ValueError('CONTROL_ASSEMBLY_TOTAL_TOO_LARGE')
    missing = [f'{split}:{role}' for split in SPLITS for role, count in counts[split]['windows'].items() if count == 0]
    report = dict(schema_version='dronedream.control-shard-assembly.v1',
        inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
        visual_input_contract=visual_identity.model_dump(mode='json'), counts=counts,
        file_sha256={name: hashlib.sha256(data).hexdigest() for name, data in outputs.items()},
        missing_window_roles=missing, minimum_window_coverage_present=not missing,
        training_ready=False, weights_trained=False, qualified_for_flight=False)
    # 有窗口只是最低接口条件；样本多样性与独立表现仍需训练方案单独验收。
    output.mkdir(parents=True, exist_ok=False)
    for name, data in outputs.items():
        publish_evidence_bytes(output / name, data, limit=MAX_REPLAY_FILE_BYTES)
    write_evidence_object(output / 'assembly-receipt.json', report)
    return report
