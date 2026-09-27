"""Dataset publication tests; source admission is tested separately with real collectors."""

import hashlib
import json
from dataclasses import replace

import pytest
from test_executed_demonstration_dataset import source_fixture

from dronedream_agent_core.training.demonstrations import collect_demonstrations
from dronedream_agent_core.training.mission_groups import mission_group_evidence
from scripts import build_executed_policy_dataset as cli


# 功能：
#   先用真实读取器接纳一份合成语料，再隔离划分验证，以单独测试目录发布事务。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：仅替换命令行层的来源收集与划分验证。
# 输出：
#   corpus：通过实际来源读取的合成样本语料。
@pytest.fixture
def admitted_corpus(tmp_path, monkeypatch):
    root = tmp_path / "run"
    source_fixture(root)
    corpus = collect_demonstrations([root], require_visual=False)
    monkeypatch.setattr(cli, "collect_demonstrations", lambda *args, **kwargs: corpus)
    monkeypatch.setattr(cli, "validate_demonstration_splits", lambda *args: None)
    return corpus


# 功能：
#   验证最终目录一次发布六个完整文件，回执摘要对应磁盘真实字节且不冒称已编码或可飞。
# 输入：
#   tmp_path：独立输出目录的父目录。
#   admitted_corpus：隔离来源验证的合成语料。
# 输出：
#   None：不返回业务数据。
def test_dataset_publication_includes_matching_receipt(tmp_path, admitted_corpus):
    output = tmp_path / "dataset"
    counts = cli.build_dataset([], [], output, require_visual=False)
    assert counts["training"] == admitted_corpus.counts
    assert len(list(output.iterdir())) == 6
    receipt = json.loads((output / "dataset-receipt.json").read_bytes())
    assert not receipt["qualified_for_flight"] and not receipt["visual_features_encoded"]
    for split in ("training", "validation"):
        raw = (output / f"{split}.jsonl").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == receipt["outputs"][split]["sha256"]
        assert raw.endswith(b"\n")


# 功能：
#   验证写入第二个划分失败时，最终目录不存在，不暴露没有回执的半成品数据集。
# 输入：
#   tmp_path：独立输出目录的父目录。
#   monkeypatch：注入写入错误。
#   admitted_corpus：可供正常写入前半部分的语料。
# 输出：
#   None：不返回业务数据。
def test_write_failure_does_not_publish_partial_dataset(tmp_path, monkeypatch, admitted_corpus):
    original = cli.write_rows

    # 功能：
    #   前半部分使用真实写入器，在验证样本文件创建前模拟磁盘故障。
    # 输入：
    #   path：当前写入路径。
    #   rows：待持久化记录。
    # 输出：
    #   digest：成功文件的实际写入摘要。
    def fail_validation(path, rows):
        if path.name == "validation.jsonl":
            raise OSError("simulated disk error")
        digest = original(path, rows)
        return digest

    monkeypatch.setattr(cli, "write_rows", fail_validation)
    output = tmp_path / "dataset"
    with pytest.raises(OSError, match="disk error"):
        cli.build_dataset([], [], output, require_visual=False)
    assert not output.exists()


# 功能：
#   验证构建期间另一个写入者创建最终目录时，无覆盖发布保留对方数据。
# 输入：
#   tmp_path：独立输出目录的父目录。
#   monkeypatch：在暂存写完后模拟最终目录竞争。
#   admitted_corpus：正常可序列化的合成语料。
# 输出：
#   None：不返回业务数据。
def test_concurrent_destination_is_not_replaced(tmp_path, monkeypatch, admitted_corpus):
    output = tmp_path / "dataset"
    original = cli._write_dataset

    # 功能：
    #   完成真实暂存后写入另一使用者的目录，使最终发布遇到明确冲突。
    # 输入：
    #   args：暂存写入位置参数。
    #   kwargs：暂存写入关键字参数。
    # 输出：
    #   receipt：已经完成的本次暂存回执。
    def occupy_destination(*args, **kwargs):
        receipt = original(*args, **kwargs)
        output.mkdir()
        (output / "preserve.txt").write_text("another writer", encoding="utf-8")
        return receipt

    monkeypatch.setattr(cli, "_write_dataset", occupy_destination)
    with pytest.raises(OSError):
        cli.build_dataset([], [], output, require_visual=False)
    assert (output / "preserve.txt").read_text(encoding="utf-8") == "another writer"
    assert len(list(output.iterdir())) == 1


# 功能：
#   验证第三套留出单独发布，并且三对来源均经过隔离检查；只测试发布，不冒充物理数据。
# 输入：
#   tmp_path：独立暂存空间。
#   monkeypatch：为每次收集提供不同语料对象并记录隔离调用。
#   admitted_corpus：真实读取器接纳的合成夹具。
# 输出：
#   None：不返回业务数据。
def test_test_split_is_separate_and_every_pair_is_checked(tmp_path, monkeypatch, admitted_corpus):
    train, validation, test = [replace(admitted_corpus) for _ in range(3)]
    sequence = iter((train, validation, test))
    checked = []
    monkeypatch.setattr(cli, "collect_demonstrations", lambda *args, **kwargs: next(sequence))
    monkeypatch.setattr(cli, "validate_demonstration_splits", lambda a, b: checked.append((id(a), id(b))))
    output = tmp_path / "three-splits"
    counts = cli.build_dataset([], [], output, require_visual=False, test_runs=[tmp_path / "test-run"])
    assert checked == [(id(train), id(validation)), (id(train), id(test)), (id(validation), id(test))]
    assert set(counts) == {"training", "validation", "test"}
    assert len(list(output.iterdir())) == 8
    receipt = json.loads((output / "dataset-receipt.json").read_bytes())
    for split in counts:
        assert receipt["outputs"][split]["sha256"] == hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest()
        assert receipt["outputs"][split]["observation_history_sha256"] == hashlib.sha256((output / f"{split}-observations.jsonl").read_bytes()).hexdigest()


# 功能：
#   测试集与验证集冲突时必须在创建输出目录前失败，不能只检查训练集。
# 输入：
#   tmp_path：测试输出父目录。
#   monkeypatch：在第三对来源的检查注入泄漏错误。
#   admitted_corpus：合成语料。
# 输出：
#   None：不返回业务数据。
def test_test_validation_leakage_prevents_publication(tmp_path, monkeypatch, admitted_corpus):
    calls = []

    # 功能：
    #   前两对允许通过，第三对模拟验证与测试来源交叉。
    # 输入：
    #   first：左侧语料。
    #   second：右侧语料。
    # 输出：
    #   None：不返回业务数据。
    def reject_third_pair(first, second):
        calls.append((first, second))
        if len(calls) == 3:
            raise ValueError("DEMONSTRATION_STREAM_LEAKAGE")

    monkeypatch.setattr(cli, "validate_demonstration_splits", reject_third_pair)
    output = tmp_path / "not-published"
    with pytest.raises(ValueError, match="STREAM_LEAKAGE"):
        cli.build_dataset([], [], output, require_visual=False, test_runs=[tmp_path / "test-run"])
    assert len(calls) == 3
    assert not output.exists()


# 功能：
#   拒绝显式空测试集及错误容器，避免用户误以为独立测试已接纳。
# 输入：
#   tmp_path：输出父目录。
#   admitted_corpus：合成语料。
#   invalid：错误测试来源参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("invalid", [[], (), "test-run", False])
def test_explicit_invalid_test_split_is_rejected(tmp_path, admitted_corpus, invalid):
    output = tmp_path / "not-published"
    with pytest.raises(ValueError, match="TEST_ROOTS_INVALID"):
        cli.build_dataset([], [], output, require_visual=False, test_runs=invalid)
    assert not output.exists()


# 功能：
#   第三套样本的写入失败时也不能发布已写完的训练和验证部分，暂存由事务回收。
# 输入：
#   tmp_path：输出父目录。
#   monkeypatch：注入测试历史写入错误。
#   admitted_corpus：合成语料。
# 输出：
#   None：不返回业务数据。
def test_test_history_failure_cleans_staging(tmp_path, monkeypatch, admitted_corpus):
    original = cli.write_rows

    # 功能：
    #   仅在测试历史落盘时模拟失败，其余文件走真实写入器。
    # 输入：
    #   path：写入文件路径。
    #   rows：待写记录。
    # 输出：
    #   digest：成功写入内容摘要。
    def fail_test_history(path, rows):
        if path.name == "test-observations.jsonl":
            raise OSError("test disk failure")
        digest = original(path, rows)
        return digest

    monkeypatch.setattr(cli, "write_rows", fail_test_history)
    output = tmp_path / "not-published"
    with pytest.raises(OSError, match="test disk failure"):
        cli.build_dataset([], [], output, require_visual=False, test_runs=[tmp_path / "test-run"])
    assert not output.exists()
    assert not list(tmp_path.glob(".not-published.staging-*"))


# 功能：
#   合并分组时拒绝同一流被重新归到另一条路线，即使上游隔离检查失效也不静默覆盖。
# 输入：
#   tmp_path：输出父目录。
#   monkeypatch：隔离发布层的上游验证。
#   admitted_corpus：具有真实可重算路线身份的合成语料。
# 输出：
#   None：不返回业务数据。
def test_manifest_merge_rejects_stream_reassignment(tmp_path, monkeypatch, admitted_corpus):
    route = json.dumps({"positions_m": [{"x": 5., "y": 0., "z": 1.}, {"x": 7., "y": 0., "z": 1.}]}).encode()
    evidence = mission_group_evidence(route, "e" * 64)
    conflict = replace(admitted_corpus,
        stream_groups={stream: evidence.group_sha256 for stream in admitted_corpus.stream_groups},
        receipts=[{"mission_split": evidence.model_dump(mode="json")}])
    sequence = iter((admitted_corpus, conflict))
    monkeypatch.setattr(cli, "collect_demonstrations", lambda *args, **kwargs: next(sequence))
    output = tmp_path / "not-published"
    with pytest.raises(ValueError, match="MISSION_GROUP_STREAM_CONFLICT"):
        cli.build_dataset([], [], output, require_visual=False)
    assert not output.exists()
    assert not list(tmp_path.glob(".not-published.staging-*"))
