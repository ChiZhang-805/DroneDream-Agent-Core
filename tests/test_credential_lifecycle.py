from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

from dronedream_agent_app.connector_credentials import ConnectorCredentialService
from dronedream_agent_app.custom_models import (
    CustomModelService,
    WindowsCredentialVault,
    detect_provider,
    validate_custom_base_url,
)
from dronedream_agent_app.storage import AppStore


class MemoryVault:
    # 功能：
    #   建立仅供当前测试使用的内存秘密表。
    # 输入：
    #   self：测试凭证库。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self):
        self.values = {}

    # 功能：
    #   保存测试秘密，不访问操作系统用户凭证。
    # 输入：
    #   self：测试凭证库。
    #   name：不透明凭证标识。
    #   secret：测试值。
    # 输出：
    #   None：不返回业务数据。
    def put(self, name, secret):
        self.values[name] = secret

    # 功能：
    #   读取当前测试值，缺失时抛出 KeyError。
    # 输入：
    #   self：测试凭证库。
    #   name：凭证标识。
    # 输出：
    #   secret：对应的测试值。
    def get(self, name):
        secret = self.values[name]
        return secret

    # 功能：
    #   幂等移除当前测试秘密，支持失败补偿检查。
    # 输入：
    #   self：测试凭证库。
    #   name：凭证标识。
    # 输出：
    #   None：不返回业务数据。
    def delete(self, name):
        self.values.pop(name, None)


# 功能：
#   给每个模型测试提供独立 SQLite 存储和内存凭证库，并在退出时关闭数据库。
# 输入：
#   tmp_path：该测试独占临时目录。
# 输出：
#   service：只引用隔离存储的模型服务。
@pytest.fixture
def service(tmp_path):
    store = AppStore(tmp_path)
    service = CustomModelService(store, MemoryVault())
    try:
        yield service
    finally:
        store.close()


# 功能：
#   创建含首尾空格测试秘密的合法自定义模型，检验秘密不会被自动修改。
# 输入：
#   service：隔离的模型服务。
# 输出：
#   profile：新建的非秘密连接资料。
def _profile(service):
    profile = service.create(
        display_name="Planner",
        base_url="https://models.example/v1",
        model_id="planner",
        api_key="  test-key-preserve  ",
        api_style="chat-completions",
    )
    return profile


# 功能：
#   URL 的空用户信息、错误端口、空查询及内部换行不能被当作合法模型地址保存。
# 输入：
#   url：非法自定义连接地址。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "url",
    [
        "https://@models.example/v1",
        "https://models.example:bad/v1",
        "https://models.example:0/v1",
        "https://models.example/v1?",
        "https://models.example/v1#",
        "https://models.exa\nmple/v1",
        123,
    ],
)
def test_custom_url_has_no_ambiguous_authority(url):
    with pytest.raises(ValueError, match="CUSTOM_MODEL_BASE_URL_INVALID"):
        validate_custom_base_url(url)


# 功能：
#   本地模型品牌只按完整端口匹配，不能让 12345 端口误判为 1234 的服务。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_local_provider_hint_matches_exact_port():
    assert detect_provider("http://127.0.0.1:1234/v1")["provider"] == "lmstudio"
    assert detect_provider("http://127.0.0.1:12345/v1")["provider"] == "ollama"
    assert detect_provider("http://[::1]:1234/v1")["provider"] == "lmstudio"


# 功能：
#   模型秘密部分写入后失败，必须补偿移除秘密且不能留下可选模型元数据。
# 输入：
#   service：隔离模型服务。
#   monkeypatch：凭证写入故障注入工具。
# 输出：
#   None：不返回业务数据。
def test_failed_model_secret_write_has_no_orphan(service, monkeypatch):
    # 功能：
    #   模拟凭证库已经写入后才返回错误，避免只测试无副作用的失败。
    # 输入：
    #   name：此次凭证标识。
    #   secret：待写入测试值。
    # 输出：
    #   None：不返回业务数据。
    def partial_write(name, secret):
        service.vault.values[name] = secret
        raise OSError("credential write failed")

    monkeypatch.setattr(service.vault, "put", partial_write)
    with pytest.raises(OSError, match="credential write failed"):
        _profile(service)
    assert service.vault.values == {}
    assert service.store.list_custom_models() == []


# 功能：
#   元数据提交失败时回收已写秘密；回收再失败也必须保留原始失败及固定补偿说明。
# 输入：
#   service：隔离模型服务。
#   monkeypatch：存储故障注入工具。
#   cleanup_fails：是否同时模拟秘密回收失败。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_model_metadata_failure_preserves_original_error(service, monkeypatch, cleanup_fails):
    failure = OSError("metadata write failed")
    monkeypatch.setattr(service.store, "save_custom_model", Mock(side_effect=failure))
    if cleanup_fails:
        monkeypatch.setattr(service.vault, "delete", Mock(side_effect=OSError("cleanup failed")))
    with pytest.raises(OSError, match="metadata write failed") as caught:
        _profile(service)
    assert caught.value is failure
    if cleanup_fails:
        assert any("CREDENTIAL_CLEANUP_FAILED" in note for note in caught.value.__notes__)
    else:
        assert service.vault.values == {}


# 功能：
#   模型连接或秘密在授权后改变时，旧授权不能继续执行旧配置。
# 输入：
#   service：隔离模型服务。
#   change：模拟的资料或凭证变更类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("change", ["base_url", "model_id", "api_style", "secret"])
def test_model_grant_rejects_changed_connection(service, change):
    profile = _profile(service)
    profile_id = profile["profile_id"]
    selection_id = profile["selection_id"]
    thread = service.store.create_thread("Task", selection_id)
    issued = service.issue_grant(profile_id, thread["thread_id"])
    if change == "secret":
        service.vault.put(profile_id, "replacement-test-key")
    else:
        columns = {
            "base_url": "https://other.example/v1",
            "model_id": "another",
            "api_style": "responses",
        }
        with service.store.transaction() as connection:
            connection.execute(
                f"UPDATE custom_models SET {change}=? WHERE profile_id=?",
                (columns[change], profile_id),
            )
    with pytest.raises(ValueError, match="CUSTOM_MODEL_GRANT_STALE"):
        service.consume_grant(issued["grant"], thread["thread_id"], selection_id)


# 功能：
#   对不存在的任务不能签发模型调用能力，避免悬空授权被后续复用。
# 输入：
#   service：隔离模型服务。
# 输出：
#   None：不返回业务数据。
def test_model_grant_requires_existing_thread(service):
    profile = _profile(service)
    with pytest.raises(KeyError):
        service.issue_grant(profile["profile_id"], "absent-thread")
    assert service._grants == {}


# 功能：
#   合法授权仍保留精确秘密，只消费一次；连接 repr 不暴露实际 API Key。
# 输入：
#   service：隔离模型服务。
# 输出：
#   None：不返回业务数据。
def test_valid_grant_preserves_secret_without_public_exposure(service):
    profile = _profile(service)
    thread = service.store.create_thread("Task", profile["selection_id"])
    grant = service.issue_grant(profile["profile_id"], thread["thread_id"])
    connection = service.consume_grant(grant["grant"], thread["thread_id"], profile["selection_id"])
    assert connection.api_key == "  test-key-preserve  "
    assert connection.api_key not in repr(connection)
    assert connection.api_key not in str(service.catalog())
    with pytest.raises(ValueError, match="GRANT_INVALID"):
        service.consume_grant(grant["grant"], thread["thread_id"], profile["selection_id"])


# 功能：
#   连接器范围必须是有界的合法插件标识集合，异常类型不能先进入 set 或正则崩溃。
# 输入：
#   scope：非法插件作用域输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("scope", [[{}], [1], "connector.test", None, ["connector.test"] * 33])
def test_connector_scope_rejects_invalid_iterables(scope):
    store = Mock(spec=AppStore)
    store.list_plugins.return_value = [{"plugin_id": "connector.test"}]
    vault = MemoryVault()
    service = ConnectorCredentialService(store, vault)
    with pytest.raises(ValueError, match="CONNECTOR_CREDENTIAL_SCOPE_INVALID"):
        service.create(display_name="Test", secret="test-secret", allowed_plugin_ids=scope)
    assert vault.values == {}


# 功能：
#   连接器保存失败及回收失败不能互相覆盖根因，补偿说明不能包含秘密。
# 输入：
#   monkeypatch：故障注入工具。
# 输出：
#   None：不返回业务数据。
def test_connector_cleanup_failure_preserves_root_error(monkeypatch):
    store = Mock(spec=AppStore)
    store.list_plugins.return_value = [{"plugin_id": "connector.test"}]
    failure = OSError("metadata unavailable")
    store.save_connector_credential.side_effect = failure
    vault = MemoryVault()
    monkeypatch.setattr(vault, "delete", Mock(side_effect=OSError("cleanup failed")))
    with pytest.raises(OSError, match="metadata unavailable") as caught:
        ConnectorCredentialService(store, vault).create(
            display_name="Test", secret="test-secret", allowed_plugin_ids=["connector.test"]
        )
    assert caught.value is failure
    assert "test-secret" not in str(caught.value.__notes__)


# 功能：
#   两个同标识凭证并发保存不能共用固定暂存文件；最终值必须是完整的一次写入。
# 输入：
#   tmp_path：独占的测试凭证目录。
#   monkeypatch：原子替换前同步两个写入的工具。
#   _attempt：独立重复用例编号，各次使用不同测试目录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI contract")
@pytest.mark.parametrize("_attempt", range(12))
def test_vault_concurrent_replacement_uses_distinct_staging(tmp_path, monkeypatch, _attempt):
    vault = WindowsCredentialVault(tmp_path)
    barrier = threading.Barrier(2, timeout=5)
    staged = []
    original_publish = vault._publish

    # 功能：
    #   让两个写入都完成暂存后才申请发布锁，覆盖并发竞争而不把同步屏障放到锁内。
    # 输入：
    #   temporary：当前写入的暂存文件。
    #   owned：暂存创建时的身份。
    #   target：最终凭证文件。
    # 输出：
    #   None：不返回业务数据。
    def synchronized_publish(temporary, owned, target):
        staged.append(temporary)
        barrier.wait()
        original_publish(temporary, owned, target)

    monkeypatch.setattr(vault, "_publish", synchronized_publish)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(vault.put, "cmp-12345678", secret)
            for secret in ("first-test-secret", "second-test-secret")
        ]
        for future in futures:
            future.result(timeout=10)
    assert vault.get("cmp-12345678") in {"first-test-secret", "second-test-secret"}
    assert len(set(staged)) == 2
    assert list(tmp_path.glob("*.tmp")) == []


# 功能：
#   发布失败后保留原加密凭证，并回收本次独占暂存；不误删其他文件。
# 输入：
#   tmp_path：测试凭证目录。
#   monkeypatch：原子发布失败注入工具。
# 输出：
#   None：不返回业务数据。
@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI contract")
def test_vault_failed_publication_preserves_previous_secret(tmp_path, monkeypatch):
    vault = WindowsCredentialVault(tmp_path)
    vault.put("cmp-12345678", "previous-test-secret")
    monkeypatch.setattr(Path, "replace", Mock(side_effect=OSError("publication failed")))
    with pytest.raises(OSError, match="publication failed"):
        vault.put("cmp-12345678", "next-test-secret")
    assert vault.get("cmp-12345678") == "previous-test-secret"
    assert list(tmp_path.glob("*.tmp")) == []


# 功能：
#   超大损坏凭证必须在解密前拒绝，不能无界读取后交给原生库。
# 输入：
#   tmp_path：测试凭证目录。
#   monkeypatch：跟踪解密入口的工具。
# 输出：
#   None：不返回业务数据。
@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI contract")
def test_vault_bounds_ciphertext_before_decrypting(tmp_path, monkeypatch):
    vault = WindowsCredentialVault(tmp_path)
    (tmp_path / "cmp-12345678.dpapi").write_bytes(b"x" * (128 * 1024 + 1))
    decrypt = Mock(return_value=b"unexpected")
    monkeypatch.setattr(vault, "_unprotect", decrypt)
    with pytest.raises(ValueError):
        vault.get("cmp-12345678")
    decrypt.assert_not_called()


# 功能：
#   元数据撤销后秘密删除若临时失败，重复删除必须能回收剩余秘密，而非永久卡在缺失元数据。
# 输入：
#   service：隔离模型服务及其存储和内存凭证库。
#   monkeypatch：注入一次删除故障和已安装测试插件的工具。
#   kind：要验证的模型或连接器服务。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["model", "connector"])
def test_revoked_credential_deletion_can_be_retried(service, monkeypatch, kind):
    if kind == "model":
        reference = _profile(service)["profile_id"]
        owner = service
    else:
        monkeypatch.setattr(
            service.store, "list_plugins", Mock(return_value=[{"plugin_id": "connector.test"}])
        )
        owner = ConnectorCredentialService(service.store, service.vault)
        reference = owner.create(
            display_name="Test", secret="test-credential", allowed_plugin_ids=["connector.test"]
        )["reference"]
    original_delete = service.vault.delete
    monkeypatch.setattr(service.vault, "delete", Mock(side_effect=OSError("temporarily locked")))
    with pytest.raises(OSError, match="temporarily locked"):
        owner.delete(reference)
    assert reference in service.vault.values
    if kind == "model":
        assert service.catalog() == []
    else:
        assert owner.list() == []
    monkeypatch.setattr(service.vault, "delete", original_delete)
    owner.delete(reference)
    assert reference not in service.vault.values


# 功能：
#   暂存被其他文件替换后不能继续发布或回收它，原目标凭证也保持原值。
# 输入：
#   tmp_path：独占测试凭证目录。
#   monkeypatch：模拟发布前文件替换的工具。
# 输出：
#   None：不返回业务数据。
@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI contract")
def test_replaced_staging_is_not_published_or_deleted(tmp_path, monkeypatch):
    vault = WindowsCredentialVault(tmp_path)
    vault.put("cmp-12345678", "original-secret")
    original_publish = vault._publish
    substituted = []

    # 功能：
    #   在独占测试目录保留原暂存并换入另一文件，验证发布和清理都按身份判断归属。
    # 输入：
    #   temporary：本次准备发布的暂存路径。
    #   owned：原暂存创建时的文件身份。
    #   target：原密文目标。
    # 输出：
    #   None：不返回业务数据。
    def substitute(temporary, owned, target):
        temporary.rename(temporary.with_suffix(".preserved"))
        temporary.write_bytes(b"foreign replacement")
        substituted.append(temporary)
        original_publish(temporary, owned, target)

    monkeypatch.setattr(vault, "_publish", substitute)
    with pytest.raises(ValueError, match="CREDENTIAL_STAGING_CHANGED") as caught:
        vault.put("cmp-12345678", "next-secret")
    assert substituted[0].read_bytes() == b"foreign replacement"
    assert "CREDENTIAL_STAGING_OWNERSHIP_CHANGED" in caught.value.__notes__
    assert vault.get("cmp-12345678") == "original-secret"


# 功能：
#   保存后的资料回读若失败，数据库插入也必须回滚，不能只删除秘密却留下已启用的空壳资料。
# 输入：
#   service：隔离模型服务及存储。
#   monkeypatch：在数据库插入后的回读位置注入故障。
#   kind：模型资料或连接器资料。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["model", "connector"])
def test_metadata_readback_failure_rolls_back_insert(service, monkeypatch, kind):
    failure = OSError("readback failed")
    if kind == "model":
        monkeypatch.setattr(service.store, "get_custom_model", Mock(side_effect=failure))
        with pytest.raises(OSError, match="readback failed"):
            _profile(service)
        assert service.store.list_custom_models() == []
    else:
        monkeypatch.setattr(service.store, "get_connector_credential", Mock(side_effect=failure))
        monkeypatch.setattr(
            service.store, "list_plugins", Mock(return_value=[{"plugin_id": "connector.test"}])
        )
        owner = ConnectorCredentialService(service.store, service.vault)
        with pytest.raises(OSError, match="readback failed"):
            owner.create(
                display_name="Test", secret="test-credential", allowed_plugin_ids=["connector.test"]
            )
        assert owner.list() == []
    assert service.vault.values == {}
