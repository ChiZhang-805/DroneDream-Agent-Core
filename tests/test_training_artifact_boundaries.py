"""Strict and bounded offline dataset I/O; no cloud calls or product models."""

import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_dagger_artifacts import dataset

from dronedream_agent_core.training.action_risk_artifacts import _rows
from dronedream_agent_core.training.dagger_artifacts import (
    load_dagger_training_artifacts,
    write_rows,
)
from dronedream_agent_core.training.evidence_files import decode_evidence_rows


# 功能：
#   验证物理计算回执中的元组仍以数组持久化，中文按 UTF-8 写入且摘要对应真实字节。
# 输入：
#   tmp_path：测试独占输出目录。
# 输出：
#   None：不返回业务数据。
def test_writer_preserves_tuple_and_unicode_evidence(tmp_path):
    path = tmp_path / "physics.jsonl"
    digest = write_rows(path, [{"速度": (1.0, 2.0, 3.0), "nested": [("左右", 0.5)]}])
    content = path.read_bytes()
    assert digest == hashlib.sha256(content).hexdigest()
    assert decode_evidence_rows(content) == [{"速度": [1.0, 2.0, 3.0], "nested": [["左右", 0.5]]}]


# 功能：
#   验证记录迭代器修改工作目录后，写入器仍核对最初打开的同一绝对路径。
# 输入：
#   tmp_path：包含两个工作目录的测试根。
#   monkeypatch：测试结束时恢复工作目录。
# 输出：
#   None：不返回业务数据。
def test_writer_freezes_path_before_consuming_rows(tmp_path, monkeypatch):
    from pathlib import Path

    second = tmp_path / "second"
    second.mkdir()
    monkeypatch.chdir(tmp_path)

    # 功能：
    #   在开始消费第一条记录时切换工作目录，模拟调用方的迭代副作用。
    # 输入：
    #   无。
    # 输出：
    #   row：一条合法记录。
    def rows():
        monkeypatch.chdir(second)
        row = {"value": 1}
        yield row

    digest = write_rows(Path("records.jsonl"), rows())
    assert hashlib.sha256((tmp_path / "records.jsonl").read_bytes()).hexdigest() == digest
    assert not (second / "records.jsonl").exists()


# 功能：
#   验证循环容器和自定义对象不会触发无界复制或任意序列化。
# 输入：
#   tmp_path：独占输出目录。
#   fault：循环引用或非标准类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["cycle", "object"])
def test_writer_rejects_non_json_graphs(tmp_path, fault):
    row = {"part": object()}
    if fault == "cycle":
        row["part"] = row
    with pytest.raises(ValueError, match="COMPLEXITY|VALUE_TYPE"):
        write_rows(tmp_path / "invalid.jsonl", [row])


# 功能：
#   验证写入器拒绝非对象、非字符串 JSON 键及过大记录，不返回成功摘要。
# 输入：
#   tmp_path：独占证据目录。
#   row：违反写入契约的记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("row", ["scalar", {1: "numeric key"}, {"large": "a" * (4 * 1024 * 1024)}])
def test_writer_rejects_invalid_or_oversized_record(tmp_path, row):
    with pytest.raises(ValueError):
        write_rows(tmp_path / "records.jsonl", [row])


# 功能：
#   验证实际写入字节数不足时不得返回完整文件摘要。
# 输入：
#   tmp_path：独占输出目录。
#   monkeypatch：受控替换文件打开方法。
# 输出：
#   None：不返回业务数据。
def test_writer_detects_short_write(tmp_path, monkeypatch):
    path = tmp_path / "short.jsonl"
    original_open = type(path).open

    # 功能：
    #   保留实际文件描述符但模拟零字节写入，退出时正常关闭该测试文件。
    # 输入：
    #   selected：准备打开的测试路径。
    #   args、kwargs：真实打开参数。
    # 输出：
    #   handle：除写入返回值外沿用实际文件操作的测试句柄。
    @contextmanager
    def short_open(selected, *args, **kwargs):
        with original_open(selected, *args, **kwargs) as source:
            handle = SimpleNamespace(write=lambda content: 0, flush=source.flush,
                                     fileno=source.fileno)
            yield handle

    monkeypatch.setattr(type(path), "open", short_open)
    with pytest.raises((ValueError, OSError), match="WRITE|write"):
        write_rows(path, [{"row": 1}])


# 功能：
#   在合成数据文件故意变化后重绑摘要，只用于检验 JSONL 和标签语义门槛。
# 输入：
#   root：合成数据集目录。
#   name：数据清单中的文件类别。
#   content：替代的原始文件字节。
# 输出：
#   None：不返回业务数据。
def replace_bound_file(root, name, content):
    path = root / (name + ".jsonl")
    path.write_bytes(content)
    receipt_path = root / "collection-receipt.jsonl"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["file_sha256"][name] = hashlib.sha256(content).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


# 功能：
#   验证重算摘要后的空白行、缺失末尾换行或重复键仍不是合法训练记录。
# 输入：
#   tmp_path：合成数据集目录。
#   fault：破坏 JSONL 完整性的方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["blank", "unterminated", "duplicate"])
def test_loader_requires_complete_unambiguous_rows(tmp_path, fault):
    dataset(tmp_path)
    path = tmp_path / "behavior-corrections.jsonl"
    content = path.read_bytes()
    if fault == "blank":
        content += b"\n"
    elif fault == "unterminated":
        content = content.rstrip(b"\n")
    else:
        first, *rest = content.splitlines()
        key, value = next(iter(json.loads(first).items()))
        duplicate = json.dumps(key).encode() + b":" + json.dumps(value).encode() + b","
        content = b"{" + duplicate + first[1:] + b"\n" + b"\n".join(rest) + b"\n"
    replace_bound_file(tmp_path, "behavior-corrections", content)
    with pytest.raises(ValueError):
        load_dagger_training_artifacts(tmp_path)


# 功能：
#   验证风险专用解析器同样拒绝 NaN、重复键、非对象和不完整末行。
# 输入：
#   content：违反严格行契约的原始 JSONL 字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("content", [b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b'[]\n', b'{}'])
def test_risk_rows_are_strict_complete_objects(content):
    with pytest.raises(ValueError):
        _rows(content)
