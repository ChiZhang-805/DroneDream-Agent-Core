"""Filesystem and streaming tests for asset storage, using only isolated local files."""

import hashlib
import os
from io import BytesIO

import pytest
from test_asset_import_boundaries import _installed_asset

import dronedream_agent_core.asset_package_storage as storage
from dronedream_agent_core.asset_packages import AssetFile, inspect_ddpkg


# 功能：
#   验证目录发布保留已经存在的空目录和非空目录，不仅保护普通文件。
# 输入：
#   tmp_path：测试独立目录。
#   populated：是否在已有目录中放入文件。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("populated", [False, True])
def test_directory_publication_never_overwrites_existing_directory(tmp_path, populated):
    source = tmp_path / "staging"
    source.mkdir()
    (source / "payload").write_bytes(b"new content")
    destination = tmp_path / "published"
    destination.mkdir()
    previous_identity = destination.stat()
    if populated:
        (destination / "existing").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        storage.publish_asset_directory(source, destination)
    assert os.path.samestat(previous_identity, destination.stat())
    assert (source / "payload").read_bytes() == b"new content"
    if populated:
        assert (destination / "existing").read_bytes() == b"keep"
    else:
        assert list(destination.iterdir()) == []


# 功能：
#   验证目标不存在时，整个暂存目录被一次发布且字节保持不变。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_directory_publication_moves_complete_directory(tmp_path):
    source = tmp_path / "staging"
    source.mkdir()
    (source / "payload").write_bytes(b"current content")
    destination = tmp_path / "published"
    storage.publish_asset_directory(source, destination)
    assert not source.exists()
    assert (destination / "payload").read_bytes() == b"current content"


# 功能：
#   验证导出可重新检查为相同内容版本，且不会在原目录内残留临时文件。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_export_preserves_verified_manifest_and_payloads(tmp_path):
    _, _, inspected, root = _installed_asset(tmp_path)
    result = storage.export_stored_asset(
        root,
        tmp_path / "export.ddpkg",
        expected_asset_id=inspected.manifest.asset_id,
        expected_content_sha256=inspected.manifest.content_sha256,
    )
    copied = inspect_ddpkg(result)
    assert copied.model_dump(mode="json") == inspected.model_dump(mode="json")
    assert list(tmp_path.glob(".export.ddpkg-*.tmp")) == []
    storage.verify_stored_asset(root, inspected.manifest)


# 功能：
#   验证导出目标不能落在源版本内部，防止改变正在被导出的不可变目录。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_export_cannot_write_inside_source_version(tmp_path):
    _, _, inspected, root = _installed_asset(tmp_path)
    with pytest.raises(ValueError, match="INSIDE_SOURCE"):
        storage.export_stored_asset(
            root,
            root / "export.ddpkg",
            expected_asset_id=inspected.manifest.asset_id,
            expected_content_sha256=inspected.manifest.content_sha256,
        )
    storage.verify_stored_asset(root, inspected.manifest)


# 功能：
#   验证原始文件超过限制时不向目标写入任何字节。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_source_copy_enforces_budget_before_writing(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"123456789")
    target = BytesIO()
    with pytest.raises(ValueError, match="SIZE_OR_TYPE"):
        storage.copy_asset_source(source, target, 8)
    assert target.getvalue() == b""


# 功能：
#   验证源文件复制期间发生变化时不能产生稳定摘要，即使字节数未变。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_source_copy_rejects_change_during_output(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"123")

    class ChangingOutput(BytesIO):
        # 功能：
        #   写入测试缓冲区时改变源文件时间戳，重现复制期间发生的源变更。
        # 输入：
        #   self：隔离内存缓冲区。
        #   payload：需要写入的测试字节。
        # 输出：
        #   written：实际写入的字节数。
        def write(self, payload):
            written = super().write(payload)
            metadata = source.stat()
            os.utime(source, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 2_000_000_000))
            return written

    with pytest.raises(ValueError, match="FILE_CHANGED"):
        storage.copy_asset_source(source, ChangingOutput(), 3)


# 功能：
#   验证成员复制检测超长、截断和同长度不同内容，不能只检查字节数。
# 输入：
#   actual：本次模拟的实际读取字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("actual", [b"abcd", b"ab", b"xyz"])
def test_indexed_member_copy_rejects_changed_content(actual):
    entry = AssetFile(
        path="payload.bin",
        role="metadata",
        media_type="application/octet-stream",
        sha256=hashlib.sha256(b"abc").hexdigest(),
        size_bytes=3,
    )
    with pytest.raises(ValueError):
        storage.copy_indexed_asset_stream(BytesIO(actual), BytesIO(), entry)


# 功能：
#   验证有界复制发布写入完整新文件，同时释放自己的临时路径。
# 输入：
#   tmp_path：测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_file_publication_preserves_checked_bytes(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"checked payload")
    destination = tmp_path / "export.bin"
    result = storage.publish_asset_file(
        source, destination, expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        limit=100,
    )
    assert result == destination
    assert destination.read_bytes() == source.read_bytes()
    assert not list(tmp_path.glob(".export.bin-*.tmp"))


# 功能：
#   验证复制预算不足或实际内容与预期摘要不同均无法发布，并清理本次暂存文件。
# 输入：
#   tmp_path：测试独占目录。
#   failure：需要模拟的容量或摘要错误。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failure", ["limit", "hash"])
def test_file_publication_rejects_unverified_copy(tmp_path, failure):
    source = tmp_path / "source"
    source.write_bytes(b"checked payload")
    destination = tmp_path / "export.bin"
    expected = hashlib.sha256(source.read_bytes()).hexdigest() if failure == "limit" else "0" * 64
    with pytest.raises(ValueError):
        storage.publish_asset_file(
            source, destination, expected_sha256=expected, limit=2 if failure == "limit" else 100
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".export.bin-*.tmp"))


# 功能：
#   验证复制目标已存在时不覆盖也不清理它，即使新内容校验正常。
# 输入：
#   tmp_path：测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_file_publication_never_overwrites_existing_file(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"new")
    destination = tmp_path / "export.bin"
    destination.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        storage.publish_asset_file(
            source, destination, expected_sha256=hashlib.sha256(b"new").hexdigest(), limit=10
        )
    assert destination.read_bytes() == b"keep"


# 功能：
#   验证发布前暂存路径被替换时拒绝发布，清理也不能删除替换者的文件。
# 输入：
#   tmp_path：测试独占目录。
#   monkeypatch：在原描述符关闭后替换暂存路径的测试替换器。
# 输出：
#   None：不返回业务数据。
def test_file_publication_preserves_replaced_temporary(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    destination = tmp_path / "export.bin"
    original_check = storage.check_plain_plugin_path
    replacements = []

    # 功能：
    #   首次检查已写完的暂存路径时移走原文件并放入另一文件，模拟所有权变化。
    # 输入：
    #   path：需要检查的本地路径。
    # 输出：
    #   None：不返回业务数据。
    def replace_after_close(path):
        if path.name.startswith(".export.bin-") and not replacements:
            path.rename(tmp_path / "retained-original")
            path.write_bytes(b"replacement owner")
            replacements.append(path)
        original_check(path)

    monkeypatch.setattr(storage, "check_plain_plugin_path", replace_after_close)
    with pytest.raises(ValueError, match="STAGING_CHANGED"):
        storage.publish_asset_file(
            source, destination, expected_sha256=hashlib.sha256(b"original").hexdigest(), limit=10
        )
    assert not destination.exists()
    assert replacements[0].read_bytes() == b"replacement owner"
    assert (tmp_path / "retained-original").read_bytes() == b"original"
