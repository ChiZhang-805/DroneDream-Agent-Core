"""Mutating catalog boundaries use only a temporary built-in plugin catalog."""

import pytest

from dronedream_agent_app.plugin_manager import PluginManager
from dronedream_agent_app.storage import AppStore


# 功能：
#   创建仅供本文件测试使用的内置插件目录，先回收会话池再关闭存储，不访问真实账户。
# 输入：
#   tmp_path_factory：生成本模块独立临时目录的 pytest 工厂。
# 输出：
#   store：已初始化内置插件表的独立数据库。
@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    store = AppStore(tmp_path_factory.mktemp("catalog"))
    manager = PluginManager(store)
    try:
        yield store
    finally:
        try:
            manager.close()
        finally:
            store.close()


# 功能：
#   验证单条与批量更新都拒绝非布尔启用值，不能把文本 false 存成启用状态。
# 输入：
#   catalog：独立测试存储。
#   bad：非法启用值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", ["false", 0, 1, [], {}])
def test_plugin_enablement_is_never_coerced(catalog, bad):
    plugin_id = "model.openai"
    before = catalog.get_plugin(plugin_id)
    with pytest.raises(ValueError, match="BOOLEAN"):
        catalog.set_plugin_lifecycle(plugin_id, enabled=bad)
    with pytest.raises(ValueError, match="BOOLEAN"):
        catalog.set_plugin_lifecycles({plugin_id: {"enabled": bad}})
    assert catalog.get_plugin(plugin_id) == before


# 功能：
#   验证批量变更末尾出现非法布尔值时，前面成员的生命周期也不被部分保存。
# 输入：
#   catalog：独立测试存储。
# 输出：
#   None：不返回业务数据。
def test_plugin_batch_rolls_back_prior_changes_if_a_later_flag_is_invalid(catalog):
    first, second = "model.openai", "navigation.shortest-route"
    before = [catalog.get_plugin(first), catalog.get_plugin(second)]
    with pytest.raises(ValueError, match="BOOLEAN"):
        catalog.set_plugin_lifecycles({first: {"enabled": False}, second: {"enabled": "false"}})
    assert [catalog.get_plugin(first), catalog.get_plugin(second)] == before


# 功能：
#   验证选择来源、选择者和凭据摘要都受契约约束，非法来源不会替换已有选择记录。
# 输入：
#   catalog：独立测试存储。
#   patch：含错误来源或错误类型摘要的选择更新。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "patch",
    [
        {"selection_source": "unknown", "selected_by": "account_configurable"},
        {"selection_source": "explicit", "selected_by": "unknown"},
        {
            "selection_source": "explicit",
            "selected_by": "account_configurable",
            "selection_receipt_sha256": 12,
        },
    ],
)
def test_selection_provenance_rejects_unknown_identities(catalog, patch):
    before = catalog.get_plugin("model.openai")
    with pytest.raises(ValueError, match="PLUGIN_SELECTION"):
        catalog.set_plugin_selection_provenance("model.openai", **patch)
    assert catalog.get_plugin("model.openai") == before


# 功能：
#   验证生命周期和治理两种回执都拒绝文本形式的接受标记，防止拒绝被当作真值持久化。
# 输入：
#   catalog：独立测试存储。
#   method：本例使用的回执写入方法名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("method", ["record_plugin_event", "record_plugin_governance_decision"])
def test_denied_receipt_cannot_be_saved_as_truthy_text(catalog, method):
    receipt = {
        "receipt_id": "event",
        "decision_id": "decision",
        "plugin_id": "model.openai",
        "operation": "enable",
        "accepted": "false",
        "created_at": "2026-09-11T00:00:00Z",
    }
    with pytest.raises(ValueError, match="BOOLEAN"):
        getattr(catalog, method)(receipt)
    assert catalog.list_plugin_events("model.openai") == []
    assert catalog.list_plugin_governance_decisions("model.openai") == []


# 功能：
#   验证嵌套非有限数值导致配置写入失败时，原保存配置仍完整保留。
# 输入：
#   catalog：独立测试存储。
# 输出：
#   None：不返回业务数据。
def test_plugin_configuration_rejects_nonfinite_json_without_replacing_saved_config(catalog):
    before = catalog.get_plugin_configuration("model.openai")
    with pytest.raises(ValueError):
        catalog.save_plugin_configuration("model.openai", {"nested": [float("nan")]})
    assert catalog.get_plugin_configuration("model.openai") == before
