"""Control transfer integrity is separate from learning quality or flight acceptance."""

import hashlib
import json
import shutil

import pytest

from dronedream_agent_core.training.control_shards import SPLITS, verify_assembled_control_data


# 功能：
#   构造仅用于完整性边界的十文件目录，不将占位内容当作有效学习样本。
# 输入：
#   tmp_path：隔离测试根目录。
# 输出：
#   root、digest：输入目录及独立保存的回执摘要。
def transfer_fixture(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    names = {'stream-groups.json'} | {f'{split}-{suffix}' for split in SPLITS
        for suffix in ('replay.jsonl', 'observations.jsonl', 'visual-receipt.json')}
    hashes = {}
    for name in sorted(names):
        content = name.encode() + b'\n'
        (root / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    receipt = dict(schema_version='dronedream.control-shard-assembly.v1', file_sha256=hashes,
                   weights_trained=False, qualified_for_flight=False)
    raw = json.dumps(receipt).encode()
    (root / 'assembly-receipt.json').write_bytes(raw)
    return root, hashlib.sha256(raw).hexdigest()


# 功能：
#   数据搬到另一路径仍可核验，无需原始 Windows 路径或原始分片在线。
# 输入：
#   tmp_path：隔离复制目录。
# 输出：
#   None：两份目录结果相同且均不授予飞行资格。
def test_control_transfer_is_relocatable(tmp_path):
    root, digest = transfer_fixture(tmp_path)
    destination = tmp_path / 'relocated'
    shutil.copytree(root, destination)
    first = verify_assembled_control_data(root, digest)
    second = verify_assembled_control_data(destination, digest)
    assert first == second
    assert first['data_files'] == 10
    assert first['verified'] and not first['qualified_for_flight']


# 功能：
#   拒绝缺失、篡改、清单外旧文件、伪造摘要及把输入替换为目录。
# 输入：
#   tmp_path、change：隔离数据和异常种类。
# 输出：
#   None：损坏目录不能通过传输检查。
@pytest.mark.parametrize('change', ['missing', 'tamper', 'extra', 'directory', 'digest', 'manifest'])
def test_control_transfer_rejects_changed_inventory(tmp_path, change):
    root, digest = transfer_fixture(tmp_path)
    target = root / 'training-replay.jsonl'
    if change == 'missing':
        target.unlink()
    elif change == 'tamper':
        target.write_bytes(b'changed')
    elif change == 'extra':
        (root / 'old.py').write_text('# stale')
    elif change == 'directory':
        target.unlink()
        target.mkdir()
    elif change == 'digest':
        digest = 'z' * 64
    else:
        (root / 'assembly-receipt.json').write_bytes(b'{}')
    with pytest.raises(ValueError):
        verify_assembled_control_data(root, digest)


# 功能：
#   即使攻击者重算回执摘要，也不能引入外部路径、额外输入或虚报已训练权限。
# 输入：
#   tmp_path、change：隔离目录和被改动的回执字段。
# 输出：
#   None：固定结构和未授权状态必须保留。
@pytest.mark.parametrize('change', ['path', 'extra', 'trained', 'qualified', 'schema'])
def test_control_transfer_rejects_invalid_bound_manifest(tmp_path, change):
    root, _ = transfer_fixture(tmp_path)
    path = root / 'assembly-receipt.json'
    value = json.loads(path.read_bytes())
    if change in ('path', 'extra'):
        key = '../training-replay.jsonl' if change == 'path' else 'other.jsonl'
        value['file_sha256'][key] = 'a' * 64
    elif change == 'trained':
        value['weights_trained'] = True
    elif change == 'qualified':
        value['qualified_for_flight'] = True
    else:
        value['schema_version'] = 'old'
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match='CONTROL_TRANSFER_RECEIPT_INVALID'):
        verify_assembled_control_data(root, hashlib.sha256(raw).hexdigest())
