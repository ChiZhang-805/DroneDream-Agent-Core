"""Dataset publication tests; source admission is tested separately with real collectors."""

import hashlib
import json

import pytest
from test_executed_demonstration_dataset import source_fixture

from dronedream_agent_core.training.demonstrations import collect_demonstrations
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
