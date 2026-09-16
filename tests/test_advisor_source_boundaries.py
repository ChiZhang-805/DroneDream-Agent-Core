"""Frozen advisor bytes must also respect file identity and memory budgets."""

from pathlib import Path

import pytest
from test_mission_groups import split_fixture

from dronedream_agent_core import plugin_files
from dronedream_agent_core.training import advisor_sources as sources


# 功能：
#   验证冻结来源不能保存可变字节数组，避免同一记录在训练期间变化。
# 输入：
#   tmp_path：来源路径所在的测试目录。
# 输出：
#   None：不返回业务数据。
def test_direct_source_construction_requires_immutable_bytes(tmp_path):
    with pytest.raises(ValueError):
        sources.RecordedSource(tmp_path / "source.json", bytearray(b"{}"))


# 功能：
#   验证来源名称和原始字节一起固定，后续工作目录变化不能改变记录指向的位置。
# 输入：
#   tmp_path：测试工作根。
#   monkeypatch：测试结束时恢复工作目录。
# 输出：
#   None：不返回业务数据。
def test_frozen_source_path_is_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("source.json").write_bytes(b"{}")
    recorded = sources.RecordedSource.read(Path("source.json"))
    assert recorded.path == tmp_path / "source.json"
    assert str(recorded) == str(tmp_path / "source.json")


# 功能：
#   验证来源读取使用共同的普通文件校验，不能绕开链接及并发替换检查。
# 输入：
#   tmp_path：测试来源目录。
#   monkeypatch：注入路径拒绝。
# 输出：
#   None：不返回业务数据。
def test_source_read_honors_plain_file_boundary(tmp_path, monkeypatch):
    path = tmp_path / "source.json"
    path.write_bytes(b"{}")

    # 功能：
    #   模拟共享路径校验拒绝不可安全读取的来源。
    # 输入：
    #   path：待检查的源路径。
    # 输出：
    #   None：不返回业务数据。
    def reject(path):
        raise ValueError("TEST_ADVISOR_PATH_REJECTED")

    monkeypatch.setattr(plugin_files, "check_plain_plugin_path", reject)
    with pytest.raises(ValueError, match="TEST_ADVISOR_PATH_REJECTED"):
        sources.RecordedSource.read(path)


# 功能：
#   用缩小的测试预算验证读取前执行文件大小限制，不实际创建巨大测试文件。
# 输入：
#   tmp_path：独占文件目录。
#   monkeypatch：仅缩小来源字节预算。
# 输出：
#   None：不返回业务数据。
def test_source_budget_is_enforced_before_read(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "MAX_RECORDED_SOURCE_BYTES", 8, raising=False)
    path = tmp_path / "source.json"
    path.write_bytes(b"012345678")
    with pytest.raises(ValueError):
        sources.RecordedSource.read(path)


# 功能：
#   验证独立来源列表有数量上限，避免重复同条路线制造无界验证工作。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_spatial_sources_have_a_batch_budget():
    evidence = split_fixture()
    source = {"mission_split": evidence.model_dump(), "semantic_sha256": evidence.semantic_sha256}
    with pytest.raises(ValueError):
        sources.advisor_spatial_groups([source] * 10_001)
