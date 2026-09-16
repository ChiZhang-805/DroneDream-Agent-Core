"""Shared publication integrity without simulator, cloud, or product model changes."""

import json
from unittest.mock import Mock

import pytest

from dronedream_agent_core.training import evidence_publication as publication
from dronedream_agent_core.training import px4_environment


# 功能：
#   检查运行环境与命令行共用唯一对象发布实现，并允许合法元组明确编码为数组。
# 输入：
#   tmp_path：独立证据目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_uses_shared_publication_and_tuple_encoding(tmp_path):
    assert px4_environment._write_new is publication.write_evidence_object
    path = tmp_path / "receipt.json"
    publication.write_evidence_object(path, {"axes": (0.2, -0.1, 0.0, 0.0)})
    assert json.loads(path.read_bytes()) == {"axes": [0.2, -0.1, 0.0, 0.0]}
    assert list(tmp_path.iterdir()) == [path]


# 功能：
#   错误字节类型、布尔预算和超额内容在创建暂存文件之前拒绝。
# 输入：
#   tmp_path：独立目录。
#   content：错误类型或超过预算的内容。
#   limit：对应的测试预算。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "content,limit", [(bytearray(b"a"), 8), (b"a", True), (b"a", 0), (b"ab", 1)]
)
def test_invalid_publication_leaves_no_files(tmp_path, content, limit):
    with pytest.raises(ValueError, match="BYTES_INVALID"):
        publication.publish_evidence_bytes(tmp_path / "evidence.bin", content, limit=limit)
    assert not list(tmp_path.iterdir())


# 功能：
#   已有文件不能被覆盖，失败后原字节及目录清单保持不变。
# 输入：
#   tmp_path：含原有证据的独立目录。
# 输出：
#   None：不返回业务数据。
def test_existing_evidence_is_never_replaced(tmp_path):
    path = tmp_path / "evidence.bin"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        publication.publish_evidence_bytes(path, b"new", limit=16)
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


# 功能：
#   最终硬链接发布失败时删除本次独占暂存文件，不留下被误认为成功的目标。
# 输入：
#   tmp_path：独立输出目录。
#   monkeypatch：注入文件系统发布失败的工具。
# 输出：
#   None：不返回业务数据。
def test_publish_failure_removes_owned_staging_only(tmp_path, monkeypatch):
    monkeypatch.setattr(publication.os, "link", Mock(side_effect=OSError("link-failed")))
    with pytest.raises(OSError, match="link-failed"):
        publication.publish_evidence_bytes(tmp_path / "evidence.bin", b"valid", limit=16)
    assert not list(tmp_path.iterdir())


# 功能：
#   非对象或含非有限值的对象不能产生任何最终或暂存回执。
# 输入：
#   tmp_path：独立目录。
#   value：不能严格编码为训练对象的输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [[], {"loss": float("inf")}])
def test_invalid_object_cannot_be_published(tmp_path, value):
    with pytest.raises(ValueError):
        publication.write_evidence_object(tmp_path / "receipt.json", value)
    assert not list(tmp_path.iterdir())
