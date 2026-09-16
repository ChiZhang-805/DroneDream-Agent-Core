"""Recorded-teacher corpus tests bind admission to exactly the consumed bytes."""

import hashlib
from dataclasses import replace

import pytest
from test_executed_demonstration_dataset import rebind, source_fixture, update_file

from dronedream_agent_core.training import demonstrations as module


# 功能：
#   验证摘要正确但内容存在歧义、未完成末行或空白行的运行记录仍然被拒绝。
# 输入：
#   tmp_path：独立运行目录的父目录。
#   suffix：对已有 JSONL 末尾施加的畸形形式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("suffix", ["duplicate-key", "missing-newline", "blank-row"])
def test_content_hash_does_not_excuse_malformed_records(tmp_path, suffix):
    root = tmp_path / "run"
    source_fixture(root)
    path = root / "learning-observations.jsonl"
    raw = path.read_bytes()
    if suffix == "duplicate-key":
        raw = raw.replace(b'"model_invoked": false', b'"model_invoked": true,"model_invoked":false')
    elif suffix == "missing-newline":
        raw = raw.rstrip(b"\n")
    else:
        raw += b"\n"
    path.write_bytes(raw)
    rebind(root)
    with pytest.raises(ValueError):
        module.collect_demonstrations([root], require_visual=False)


# 功能：
#   验证执行序号缺失及布尔计数不能伪装成一条完整的已执行训练样本。
# 输入：
#   tmp_path：独立运行目录的父目录。
#   corruption：选择执行序号或汇总计数缺陷。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("corruption", ["sequence-gap", "boolean-count"])
def test_dataset_requires_continuous_execution_and_integer_counts(tmp_path, corruption):
    root = tmp_path / "run"
    source_fixture(root)
    if corruption == "sequence-gap":
        update_file(
            root, "runtime-state/control-applications.jsonl", lambda row: row.update(sequence=3)
        )
    else:
        update_file(
            root,
            "learning-observation-summary.json",
            lambda row: row.update(submitted=True, completed=True),
        )
    with pytest.raises(ValueError):
        module.collect_demonstrations([root], require_visual=False)


# 功能：
#   验证收集期间源文件被更改后，回执仍绑定实际读取内容，不能给旧输入附上新摘要。
# 输入：
#   tmp_path：独立运行目录。
#   monkeypatch：在特征编译时触发源文件改写。
# 输出：
#   None：不返回业务数据。
def test_receipt_hashes_match_consumed_metadata(tmp_path, monkeypatch):
    root = tmp_path / "run"
    source_fixture(root)
    path = root / "mission_evidence.json"
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    compile_features = module.compile_local_policy_features

    # 功能：
    #   模拟计算期间其他进程改写文件，仍调用真实特征编译器。
    # 输入：
    #   args：原特征编译位置参数。
    #   kwargs：原特征编译关键字参数。
    # 输出：
    #   batch：当前快照的真实特征批次。
    def compile_and_change(*args, **kwargs):
        path.write_bytes(b'{"replaced":true}')
        batch = compile_features(*args, **kwargs)
        return batch

    monkeypatch.setattr(module, "compile_local_policy_features", compile_and_change)
    corpus = module.collect_demonstrations([root], require_visual=False)
    assert corpus.receipts[0]["mission_evidence_sha256"] == expected


# 功能：
#   验证空数据集不能仅因没有交集就被视为有效训练／验证划分。
# 输入：
#   tmp_path：合成运行目录的父目录。
# 输出：
#   None：不返回业务数据。
def test_empty_corpus_cannot_pass_split_admission(tmp_path):
    root = tmp_path / "run"
    source_fixture(root)
    corpus = module.collect_demonstrations([root], require_visual=False)
    empty = replace(corpus, samples=[], observations=[], stream_groups={})
    with pytest.raises(ValueError, match="CORPUS"):
        module.validate_demonstration_splits(empty, empty)
