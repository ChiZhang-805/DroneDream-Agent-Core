"""治理决策的可变输入、数量边界与精确摘要测试，不执行插件。"""

from contextlib import ExitStack

import pytest
from pydantic import ValidationError

from dronedream_agent_app.plugin_manager import PluginManager, PluginManagerError
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_contracts import PluginGovernancePolicy, PluginManifest
from dronedream_agent_core.plugin_governance import evaluate_plugin_governance


# 功能：
#   提供无执行权限的声明式清单，隔离治理计算与包安装、外部进程和网络。
# 输入：
#   无。
# 输出：
#   manifest：通过契约校验的惰性面板清单。
@pytest.fixture
def manifest():
    manifest = PluginManifest.model_validate(
        {
            "plugin_id": "example.governance",
            "name": "Governance fixture",
            "version": "1.0.0",
            "publisher": "Fixture",
            "description": "Governance evaluation without external code.",
            "runtime": {"kind": "ui-declarative"},
            "permissions": ["ui.panel"],
            "capabilities": [
                {
                    "capability_id": "example.governance.panel",
                    "name": "Panel",
                    "kind": "ui-panel",
                    "authority": "read",
                    "description": "Inert panel.",
                    "metadata": {"entrypoint": "ui/panel.json"},
                }
            ],
            "file_sha256": {"ui/panel.json": "a" * 64},
        }
    )
    return manifest


# 功能：
#   验证外部插件计数不能用负值、布尔值或非整数绕过导入数量上限。
# 输入：
#   manifest：惰性测试清单。
#   count：故意非法的外部插件计数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("count", [-1, True, False, 1.5, float("nan"), "0", None])
def test_governance_count_is_a_nonnegative_integer(manifest, count):
    with pytest.raises(ValueError, match="PLUGIN_GOVERNANCE_COUNT_INVALID"):
        evaluate_plugin_governance(
            policy=PluginGovernancePolicy(maximum_external_plugins=0),
            manifest=manifest,
            operation="import",
            trust_status="verified",
            installed_external_plugins=count,
        )


# 功能：
#   验证已实例化策略也需重新校验，字符串 false 不能按真值批准本地包。
# 输入：
#   manifest：惰性测试清单。
# 输出：
#   None：不返回业务数据。
def test_governance_revalidates_mutated_policy(manifest):
    policy = PluginGovernancePolicy()
    policy.allow_local_approval = "false"
    with pytest.raises(ValidationError):
        evaluate_plugin_governance(
            policy=policy,
            manifest=manifest,
            operation="trust-local-package",
            trust_status="unverified",
            installed_external_plugins=0,
        )


# 功能：
#   验证清单创建后的嵌套修改也被拒绝，不能给包含未声明权限的能力签发接受决策。
# 输入：
#   manifest：随后会被故意修改的清单。
# 输出：
#   None：不返回业务数据。
def test_governance_revalidates_mutated_capability(manifest):
    manifest.capabilities[0].required_permissions = ["mission.read"]
    with pytest.raises(ValidationError):
        evaluate_plugin_governance(
            policy=PluginGovernancePolicy(),
            manifest=manifest,
            operation="enable",
            trust_status="verified",
            installed_external_plugins=0,
        )


# 功能：
#   验证导入不代表执行许可，签名门槛在启用和提升时生效，撤销始终优先。
# 输入：
#   manifest：惰性测试清单。
# 输出：
#   None：不返回业务数据。
def test_import_execution_and_revocation_have_distinct_gates(manifest):
    policy = PluginGovernancePolicy(require_verified_signatures=True)
    imported = evaluate_plugin_governance(
        policy=policy,
        manifest=manifest,
        operation="import",
        trust_status="unverified",
        installed_external_plugins=0,
    )
    assert imported.accepted
    assert imported.policy_sha256 == sha256_json(policy.model_dump(mode="json"))
    assert imported.manifest_sha256 == sha256_json(manifest.model_dump(mode="json"))
    for operation in ("enable", "promote"):
        decision = evaluate_plugin_governance(
            policy=policy,
            manifest=manifest,
            operation=operation,
            trust_status="local-approved",
            installed_external_plugins=0,
        )
        assert not decision.accepted
        assert "GOVERNANCE_VERIFIED_SIGNATURE_REQUIRED" in decision.issue_codes
    revoked = evaluate_plugin_governance(
        policy=PluginGovernancePolicy(),
        manifest=manifest,
        operation="import",
        trust_status="revoked",
        installed_external_plugins=0,
    )
    assert not revoked.accepted
    assert "GOVERNANCE_PACKAGE_REVOKED" in revoked.issue_codes


# 功能：
#   验证非法修改后的策略即使没有外部插件也不能保存，原策略和设置保持不变。
# 输入：
#   tmp_path：只含本例数据库的临时目录。
# 输出：
#   None：不返回业务数据。
def test_invalid_policy_cannot_poison_an_empty_external_catalog(tmp_path):
    with ExitStack() as cleanup:
        store = AppStore(tmp_path)
        cleanup.callback(store.close)
        manager = PluginManager(store)
        cleanup.callback(manager.close)
        before = store.get_settings()
        policy = PluginGovernancePolicy()
        policy.maximum_external_plugins = -1
        with pytest.raises(ValidationError):
            manager.set_governance_policy(policy)
        assert store.get_settings() == before


# 功能：
#   验证数量上限同样约束已启用的外部包，允许恰好达到上限，并允许原包更新不重复计数。
# 输入：
#   tmp_path：独立数据库目录。
#   manifest：不会启动进程的测试清单。
# 输出：
#   None：不返回业务数据。
def test_enabled_catalog_obeys_total_limit_without_double_counting_updates(tmp_path, manifest):
    with ExitStack() as cleanup:
        store = AppStore(tmp_path)
        cleanup.callback(store.close)
        manager = PluginManager(store)
        cleanup.callback(manager.close)
        store.upsert_plugin(
            manifest=manifest,
            package_sha256="b" * 64,
            bundle_root=tmp_path,
            builtin=False,
            enabled=True,
            status="healthy",
            health="healthy",
        )
        before = store.get_settings()
        with pytest.raises(PluginManagerError, match="GOVERNANCE_EXTERNAL_PLUGIN_LIMIT"):
            manager.set_governance_policy(PluginGovernancePolicy(maximum_external_plugins=0))
        assert store.get_settings() == before
        manager.set_governance_policy(PluginGovernancePolicy(maximum_external_plugins=1))
        decision = manager._govern(manifest=manifest, operation="import", trust_status="verified")
        assert decision["accepted"] is True
