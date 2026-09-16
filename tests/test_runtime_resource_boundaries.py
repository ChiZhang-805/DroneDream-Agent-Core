"""Validate packaged Runtime files using temporary fixtures, without accessing WSL."""

import json

import pytest
from test_runtime_manager import _manager, _runtime_resources

from dronedream_agent_app import runtime_resource_manifest as resources_io


# 功能：
#   验证安装器可能加载的额外 wheel、ROS 源码或目录不能脱离发布索引进入运行。
# 输入：
#   tmp_path：隔离测试目录。
#   relative：需要插入的未索引文件路径。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "relative", ["wheels/old.whl", "ros_ws/src/old/setup.py", "local-policy/catalog.json"]
)
def test_unindexed_runtime_content_is_not_ready(tmp_path, relative):
    resources = _runtime_resources(tmp_path / "resources")
    extra = resources / "runtime" / relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"unindexed")
    assert not _manager(tmp_path / "store", resources)._resources_ready()


# 功能：
#   验证清单的版本、路径和实际字节数均参与校验，不只比较文件摘要。
# 输入：
#   tmp_path：隔离测试目录。
#   mutation：要构造的清单缺陷。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "mutation", ["schema", "alias", "size", "boolean_size", "missing_size", "duplicate_key"]
)
def test_runtime_manifest_rejects_inconsistent_metadata(tmp_path, mutation):
    resources = _runtime_resources(tmp_path / "resources")
    manifest_path = resources / "runtime" / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    entry = manifest["files"][0]
    if mutation == "schema":
        manifest["schema_version"] = "unsupported"
    elif mutation == "alias":
        entry["path"] = "./" + entry["path"]
    elif mutation == "size":
        entry["bytes"] += 1
    elif mutation == "boolean_size":
        entry["bytes"] = True
    elif mutation == "missing_size":
        entry.pop("bytes")
    payload = json.dumps(manifest)
    if mutation == "duplicate_key":
        payload = '{"files":[],' + payload[1:]
    manifest_path.write_text(payload, encoding="utf-8")
    assert not _manager(tmp_path / "store", resources)._resources_ready()


# 功能：
#   验证原始合法夹具仍可通过，避免以无条件拒绝实现边界保护。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_current_runtime_manifest_remains_usable(tmp_path):
    resources = _runtime_resources(tmp_path / "resources")
    assert _manager(tmp_path / "store", resources)._resources_ready()


# 功能：
#   验证目录项、清单字节和总资源字节限制分别生效，避免无限枚举或加载。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：临时收紧资源限制的测试工具。
#   budget：需要验证的资源限制名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "budget",
    ["MAX_RESOURCE_FILES", "MAX_RESOURCE_ENTRIES", "MAX_MANIFEST_BYTES", "MAX_RESOURCE_BYTES"],
)
def test_runtime_resource_budgets_are_enforced(tmp_path, monkeypatch, budget):
    resources = _runtime_resources(tmp_path / "resources")
    monkeypatch.setattr(resources_io, budget, 1)
    with pytest.raises(ValueError):
        resources_io.verify_runtime_resource_manifest(resources / "runtime")


# 功能：
#   验证文件校验期间更换发布清单不能把旧索引的结果绑定到新清单。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：在首次文件摘要后模拟清单变化的测试工具。
# 输出：
#   None：不返回业务数据。
def test_runtime_manifest_change_during_verification_is_rejected(tmp_path, monkeypatch):
    resources = _runtime_resources(tmp_path / "resources")
    manifest_path = resources / "runtime" / "runtime-manifest.json"
    original_hash = resources_io.hash_plugin_file
    changed = []

    # 功能：
    #   保持实际文件摘要计算，仅在第一次调用后改变清单字节。
    # 输入：
    #   path：当前被校验的资源文件。
    #   limit：当前文件的声明字节上限。
    # 输出：
    #   digest：真实文件摘要。
    def replace_manifest_after_hash(path, *, limit):
        digest = original_hash(path, limit=limit)
        if not changed:
            manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
            changed.append(True)
        return digest

    monkeypatch.setattr(resources_io, "hash_plugin_file", replace_manifest_after_hash)
    with pytest.raises(ValueError, match="MANIFEST_CHANGED"):
        resources_io.verify_runtime_resource_manifest(resources / "runtime")
