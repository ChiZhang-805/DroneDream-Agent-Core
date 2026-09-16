"""Artifact identity fixtures only; the native probe renders actual Gazebo images."""

import json
from io import BytesIO
from pathlib import Path

import pytest

from dronedream_agent_core import render_replica_runtime as runtime


# 功能：
#   建立字节可控的假原生构建及依赖回执，不执行编译器或加载真实动态库。
# 输入：
#   tmp_path：隔离源码、构建及依赖目录。
#   monkeypatch：临时替换源码根目录。
# 输出：
#   artifacts：构建目录、回执字典及源码目录。
def fixture(tmp_path, monkeypatch):
    source = tmp_path / "sources"
    source.mkdir()
    (source / "source.cpp").write_bytes(b"fixture-source")
    monkeypatch.setattr(runtime, "SOURCE_ROOT", source)
    root = tmp_path / "build"
    root.mkdir()
    for name in runtime.BINARIES:
        (root / name).write_bytes(b"fixture-not-native-code")
    libraries = {}
    for name in ("libOgreNextMain.so.2.3.1", "RenderSystem_GL3Plus.so.2.3.1",
                 "libgz-rendering8-ogre2.so.8.2.3", "libgz-sim8-sensors-system.so.8.14.0"):
        path = tmp_path / name
        path.write_bytes(b"fixture-not-a-library")
        libraries[str(path)] = runtime.file_hash(path)
    header = tmp_path / "Pose.hh"
    header.write_bytes(b"fixture-header")
    receipt = {"sources": runtime.source_files(),
        "binaries": {n: runtime.file_hash(root / n) for n in runtime.BINARIES},
        "libraries": libraries, "component_headers": {str(header): runtime.file_hash(header)},
        "native_tests_passed": True, "qualification_granted": False,
        "sensors_plugin": str(tmp_path / "libgz-sim8-sensors-system.so.8.14.0")}
    (root / runtime.RECEIPT).write_text(json.dumps(receipt))
    artifacts = root, receipt, source
    return artifacts


# 功能：
#   验证完整内容绑定可以通过，但保持不授予飞行资格的明确标志。
# 输入：
#   tmp_path：隔离制品目录。
#   monkeypatch：替换源码根的测试工具。
# 输出：
#   None：不返回业务数据。
def test_complete_source_bound_fixture_never_grants_flight_qualification(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path, monkeypatch)
    assert runtime.validate_replica_runtime(root)["qualification_granted"] is False


# 功能：
#   验证回执读入被替换为超预算来源时拒绝，不仅依赖首次 stat 声明的大小。
# 输入：
#   tmp_path：隔离制品目录。
#   monkeypatch：替换单次文件打开的测试工具。
# 输出：
#   None：不返回业务数据。
def test_receipt_read_is_bounded_even_if_file_grows_after_stat(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path, monkeypatch)
    real_open = Path.open

    class GrowingReceipt(BytesIO):
        # 功能：
        #   对调用方实际读取设置字节预算断言，不提供文件描述符供身份验证绕过。
        # 输入：
        #   size：请求读取的字节数。
        # 输出：
        #   content：最多请求长度的夹具字节。
        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024 + 1
            content = super().read(size)
            return content

    # 功能：
    #   仅替换回执打开结果，其他路径仍访问本测试创建的普通文件。
    # 输入：
    #   path：正在打开的路径。
    #   args：原始打开位置参数。
    #   kwargs：原始打开关键字参数。
    # 输出：
    #   stream：超限回执替身或真实文件流。
    def open_snapshot(path, *args, **kwargs):
        if path == root / runtime.RECEIPT:
            stream = GrowingReceipt(b" " * (1024 * 1024 + 2))
        else:
            stream = real_open(path, *args, **kwargs)
        return stream

    monkeypatch.setattr(Path, "open", open_snapshot)
    with pytest.raises(ValueError, match="RECEIPT_UNAVAILABLE"):
        runtime.validate_replica_runtime(root)


# 功能：
#   验证超深 JSON 回执在依赖文件读取前被明确拒绝。
# 输入：
#   tmp_path：隔离制品目录。
#   monkeypatch：替换源码根的测试工具。
# 输出：
#   None：不返回业务数据。
def test_deeply_nested_receipt_is_rejected_before_dependency_lookup(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path, monkeypatch)
    (root / runtime.RECEIPT).write_text("[" * 2000 + "]" * 2000)
    with pytest.raises(ValueError, match="RECEIPT_UNAVAILABLE"):
        runtime.validate_replica_runtime(root)


# 功能：
#   分别替换二进制、源码、头文件与动态库，核对旧构建回执不再有效。
# 输入：
#   tmp_path：隔离制品目录。
#   monkeypatch：替换源码根的测试工具。
#   mutation：本次更换的制品类别。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["binary", "source", "header", "library"])
def test_replaced_dependency_or_source_cannot_reuse_frozen_build(tmp_path, monkeypatch, mutation):
    root, _, source = fixture(tmp_path, monkeypatch)
    path = {"binary": root / runtime.EXECUTABLE, "source": source / "source.cpp",
            "header": tmp_path / "Pose.hh", "library": tmp_path / "libOgreNextMain.so.2.3.1"}
    path[mutation].write_bytes(b"changed")
    with pytest.raises(ValueError):
        runtime.validate_replica_runtime(root)


# 功能：
#   验证错误资格标志、缺少渲染依赖或混合 ABI 的回执不能通过。
# 输入：
#   tmp_path：隔离制品目录。
#   monkeypatch：替换源码根的测试工具。
#   mutation：要破坏的资格或依赖声明。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["qualification", "tests", "engine", "plugin", "mixed"])
def test_incomplete_or_mixed_build_identity_is_rejected(tmp_path, monkeypatch, mutation):
    root, receipt, _ = fixture(tmp_path, monkeypatch)
    if mutation == "qualification":
        receipt["qualification_granted"] = True
    elif mutation == "tests":
        receipt["native_tests_passed"] = False
    elif mutation == "engine":
        receipt["libraries"].pop(str(tmp_path / "libgz-rendering8-ogre2.so.8.2.3"))
    elif mutation == "plugin":
        receipt["sensors_plugin"] = str(tmp_path / "different.so")
    else:
        path = tmp_path / "libOgreMain.so.1.9.0"
        path.write_bytes(b"old-generation-fixture")
        receipt["libraries"][str(path)] = runtime.file_hash(path)
    (root / runtime.RECEIPT).write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        runtime.validate_replica_runtime(root)
