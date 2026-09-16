"""信任库读写、锁身份及资源回收边界；只操作独立测试目录。"""

import os
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

import dronedream_agent_core.plugin_trust as trust_module
from dronedream_agent_core.plugin_trust import PluginTrustStore, TrustStoreError


# 功能：
#   验证锁文件或快照文件的流包装失败时，关闭本次描述符且保持原信任数据。
# 输入：
#   tmp_path：信任库所在的独立目录。
#   monkeypatch：注入流包装失败。
#   operation：需要检查的锁打开或快照写入操作。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("operation", ["lock", "snapshot"])
def test_trust_fdopen_failure_closes_owned_descriptor(tmp_path, monkeypatch, operation):
    store = PluginTrustStore(tmp_path / "trust.json")
    original = store.path.read_bytes()
    captured = {}

    # 功能：
    #   记录原描述符身份后模拟流包装失败，供测试确认资源所有权。
    # 输入：
    #   descriptor：已打开但尚未转交给流的描述符。
    #   args：原流包装的位置参数。
    #   kwargs：原流包装的关键字参数。
    # 输出：
    #   None：不返回业务数据。
    def reject_stream(descriptor, *args, **kwargs):
        captured.update(descriptor=descriptor, identity=os.fstat(descriptor))
        raise OSError("fixture stream construction failed")

    try:
        # 写入测试先正常持有锁，再注入错误，以免错误落到不相关的锁打开阶段。
        with store._locked() if operation == "snapshot" else nullcontext():
            with monkeypatch.context() as patch:
                patch.setattr(trust_module.os, "fdopen", reject_stream)
                with pytest.raises(OSError, match="fixture stream construction failed"):
                    if operation == "snapshot":
                        store._write(store._empty())
                    else:
                        with store._locked():
                            pytest.fail("a failed lock stream must not enter the transaction")
            with pytest.raises(OSError):
                os.fstat(captured["descriptor"])
        assert store.path.read_bytes() == original
        assert not list(tmp_path.glob(".plugin-trust-*.json"))
    finally:
        # 复现旧实现时，也只关闭测试自身泄漏且仍保持原身份的描述符。
        try:
            current = os.fstat(captured["descriptor"])
        except OSError:
            pass
        else:
            if os.path.samestat(current, captured["identity"]):
                os.close(captured["descriptor"])


# 功能：
#   验证暂存路径被其他文件占用时，既不发布替换内容，也不误删另一写入者的文件。
# 输入：
#   tmp_path：信任库和暂存文件所在目录。
#   monkeypatch：在暂存流关闭后替换其路径。
# 输出：
#   None：不返回业务数据。
def test_trust_replaced_staging_is_preserved_and_not_published(tmp_path, monkeypatch):
    store = PluginTrustStore(tmp_path / "trust.json")
    original = store.path.read_bytes()
    original_mkstemp = trust_module.tempfile.mkstemp
    original_fdopen = os.fdopen
    captured = {}

    # 功能：
    #   记录当前信任快照临时文件，保留真实创建行为。
    # 输入：
    #   kwargs：临时文件的创建参数。
    # 输出：
    #   temporary：描述符和暂存路径的元组。
    def record_temporary(**kwargs):
        temporary = original_mkstemp(**kwargs)
        captured.update(descriptor=temporary[0], path=Path(temporary[1]))
        return temporary

    # 功能：
    #   关闭原暂存流后制造实际路径替换，不在 Windows 上移动仍打开的文件。
    # 输入：
    #   descriptor：快照文件描述符。
    #   args：原流包装的位置参数。
    #   kwargs：原流包装的关键字参数。
    # 输出：
    #   stream：原始快照流。
    @contextmanager
    def replace_closed_staging(descriptor, *args, **kwargs):
        with original_fdopen(descriptor, *args, **kwargs) as stream:
            yield stream
        path = captured["path"]
        path.rename(tmp_path / "retained-original.json")
        path.write_bytes(b"replacement owned by another writer")

    with store._locked(), monkeypatch.context() as patch:
        patch.setattr(trust_module.tempfile, "mkstemp", record_temporary)
        patch.setattr(trust_module.os, "fdopen", replace_closed_staging)
        with pytest.raises(TrustStoreError, match="STAGING_CHANGED"):
            store._write(store._empty())
    assert store.path.read_bytes() == original
    assert captured["path"].read_bytes() == b"replacement owned by another writer"


# 功能：
#   验证不遵守锁的外部写入在快照准备期间改动信任库时，不被旧事务悄悄覆盖。
# 输入：
#   tmp_path：测试信任库目录。
#   monkeypatch：在暂存快照刷盘时修改目标信任库。
# 输出：
#   None：不返回业务数据。
def test_trust_external_update_during_write_is_not_overwritten(tmp_path, monkeypatch):
    store = PluginTrustStore(tmp_path / "trust.json")
    original_fsync = os.fsync
    foreign = store.path.read_bytes() + b"\n"

    # 功能：
    #   保留真实刷盘动作，并模拟另一个未取得锁的写入者修改目标内容。
    # 输入：
    #   descriptor：当前暂存文件描述符。
    # 输出：
    #   None：不返回业务数据。
    def update_destination(descriptor):
        original_fsync(descriptor)
        store.path.write_bytes(foreign)

    with store._locked(), monkeypatch.context() as patch:
        patch.setattr(trust_module.os, "fsync", update_destination)
        with pytest.raises(TrustStoreError, match="STORE_CHANGED"):
            store._write(store._empty())
    assert store.path.read_bytes() == foreign
    assert not list(tmp_path.glob(".plugin-trust-*.json"))


# 功能：
#   验证读取结束后源文件被替换时，不将已经过时的权限快照交给签名判断。
# 输入：
#   tmp_path：信任库目录。
#   monkeypatch：只在信任库读取流关闭后替换该文件。
# 输出：
#   None：不返回业务数据。
def test_trust_read_rejects_replaced_source(tmp_path, monkeypatch):
    store = PluginTrustStore(tmp_path / "trust.json")
    original_open = Path.open

    # 功能：
    #   在真实读取完成后更换源文件身份，模拟快照读取窗口内的更新。
    # 输入：
    #   path：被打开的路径。
    #   args：原打开操作的位置参数。
    #   kwargs：原打开操作的关键字参数。
    # 输出：
    #   stream：原文件流。
    @contextmanager
    def replace_after_read(path, *args, **kwargs):
        with original_open(path, *args, **kwargs) as stream:
            yield stream
        if path == store.path:
            path.rename(tmp_path / "retained-trust.json")
            with original_open(path, "wb") as replacement:
                replacement.write(b"{}")

    with store._locked(), monkeypatch.context() as patch:
        patch.setattr(Path, "open", replace_after_read)
        with pytest.raises(TrustStoreError, match="STORE_INVALID"):
            store._load()


# 功能：
#   验证获得文件锁后仍复核锁路径身份，避免两个写入者锁住不同文件却修改同一信任库。
# 输入：
#   tmp_path：锁和信任库所在目录。
#   monkeypatch：模拟锁路径在打开后指向另一普通文件的元数据。
# 输出：
#   None：不返回业务数据。
def test_trust_lock_identity_is_checked_after_acquisition(tmp_path, monkeypatch):
    store = PluginTrustStore(tmp_path / "trust.json")
    lock_path = store.path.with_name("trust.json.lock")
    foreign = tmp_path / "another-lock"
    foreign.write_bytes(b"x")
    foreign_identity = foreign.stat()
    original_stat = Path.stat

    # 功能：
    #   仅替换锁路径 stat 的结果，保留 lstat 和真实文件锁，避免依赖 Windows 删除共享权限。
    # 输入：
    #   path：待查询路径。
    #   args：原 stat 位置参数。
    #   kwargs：原 stat 关键字参数。
    # 输出：
    #   metadata：模拟路径身份或真实文件元数据。
    def changed_lock_stat(path, *args, **kwargs):
        metadata = (
            foreign_identity
            if path == lock_path and kwargs.get("follow_symlinks", True)
            else original_stat(path, *args, **kwargs)
        )
        return metadata

    monkeypatch.setattr(Path, "stat", changed_lock_stat)
    with pytest.raises(TrustStoreError, match="LOCK_CHANGED"), store._locked():
        pytest.fail("the transaction must not run under the wrong lock identity")
