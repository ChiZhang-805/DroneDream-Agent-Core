from __future__ import annotations

import pytest

from dronedream_agent_app.connector_credentials import ConnectorCredentialService
from dronedream_agent_app.plugin_manager import PluginManager
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.capability_broker import CapabilityBrokerError


class MemoryVault:
    """In-memory test double; never open the operating system credential vault."""

    # 功能：
    #   建立测试专用秘密表，不访问操作系统凭证。
    # 输入：
    #   self：测试凭证库。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    # 功能：
    #   将测试秘密存入当前夹具。
    # 输入：
    #   self：测试凭证库。
    #   name：凭证标识。
    #   secret：测试秘密。
    # 输出：
    #   None：不返回业务数据。
    def put(self, name: str, secret: str) -> None:
        self.values[name] = secret

    # 功能：
    #   读取当前测试秘密，缺失时抛出 KeyError。
    # 输入：
    #   self：测试凭证库。
    #   name：凭证标识。
    # 输出：
    #   secret：对应测试值。
    def get(self, name: str) -> str:
        secret = self.values[name]
        return secret

    # 功能：
    #   幂等移除测试秘密，用于核对服务撤销结果。
    # 输入：
    #   self：测试凭证库。
    #   name：要撤销的标识。
    # 输出：
    #   None：不返回业务数据。
    def delete(self, name: str) -> None:
        self.values.pop(name, None)


# 功能：
#   创建隔离数据库，测试结束后关闭连接。
# 输入：
#   tmp_path：当前测试的独占目录。
# 输出：
#   store：本次使用的本机存储。
@pytest.fixture
def store(tmp_path):
    store = AppStore(tmp_path)
    try:
        yield store
    finally:
        store.close()


# 功能：
#   为需要已安装插件的测试加载产品目录，并先于存储关闭插件管理器。
# 输入：
#   store：测试存储。
# 输出：
#   manager：当前测试的插件管理器。
@pytest.fixture
def manager(store):
    manager = PluginManager(store)
    try:
        yield manager
    finally:
        manager.close()


# 功能：
#   只有指定插件可解析秘密，列表不泄密，删除后旧引用不能再次使用。
# 输入：
#   store：隔离凭证元数据存储。
#   manager：已加载产品插件目录的管理器夹具。
# 输出：
#   None：不返回业务数据。
def test_connector_credential_is_opaque_and_plugin_scoped(store, manager):
    vault = MemoryVault()
    service = ConnectorCredentialService(store, vault)

    value = service.create(
        display_name="PagerDuty production",
        secret="never-return-this-secret",
        allowed_plugin_ids=["connector.alerts.pagerduty"],
    )

    reference = str(value["reference"])
    assert reference.startswith("cred-")
    assert "never-return-this-secret" not in str(value)
    assert "never-return-this-secret" not in str(service.list())
    assert service.resolve(reference, plugin_id="connector.alerts.pagerduty") == (
        "never-return-this-secret"
    )
    with pytest.raises(CapabilityBrokerError, match="BROKER_CREDENTIAL_SCOPE_DENIED"):
        service.resolve(reference, plugin_id="connector.erp.notion")

    service.delete(reference)
    assert reference not in vault.values
    with pytest.raises(CapabilityBrokerError, match="BROKER_CREDENTIAL_UNAVAILABLE"):
        service.resolve(reference, plugin_id="connector.alerts.pagerduty")


# 功能：
#   未安装插件不能仅靠提交名称取得新凭证的允许范围。
# 输入：
#   store：不含产品插件目录的隔离存储。
# 输出：
#   None：不返回业务数据。
def test_connector_credential_rejects_uninstalled_scope(store):
    service = ConnectorCredentialService(store, MemoryVault())
    with pytest.raises(ValueError, match="CONNECTOR_CREDENTIAL_PLUGIN_NOT_INSTALLED"):
        service.create(
            display_name="unknown",
            secret="secret",
            allowed_plugin_ids=["unknown.connector"],
        )
