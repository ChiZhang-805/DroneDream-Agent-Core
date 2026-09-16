from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import audit_plugin_catalog as catalog_module
from scripts.audit_plugin_catalog import audit, discover_host_hook_bindings


# 功能：
#   验证目录检查器识别直接调用、策略映射和工具中间件，不补造源码中不存在的入口。
# 输入：
#   tmp_path：存放最小宿主源码夹具的临时目录。
# 输出：
#   None：不返回业务数据。
def test_host_hook_scanner_recognizes_direct_and_dynamic_dispatch(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "host.py").write_text(
        """
policy_hooks = {
    "retry": ("harness.retry-policy", "resolve_retry"),
}

def dispatch(registry, slot_id, hook_name, value):
    registry.invoke_single("models.role-policy", "select_port")
    registry.invoke_single(slot_id, hook_name)
    registry._middleware("before_tool_call", value)
""".strip()
        + "\n",
        encoding="utf-8",
    )

    bindings = discover_host_hook_bindings(source_root)

    assert ("models.role-policy", "select_port") in bindings
    assert ("harness.retry-policy", "resolve_retry") in bindings
    assert ("tools.middleware", "before_tool_call") in bindings
    assert ("assets.map-importer", "import") not in bindings
    assert ("assets.vehicle-importer", "import") not in bindings


# 功能：
#   验证目录中存在扩展但宿主源码未连接钩子时被拒绝，不把声明误当成已接入能力。
# 输入：
#   tmp_path：隔离的空源码及插件存储目录。
# 输出：
#   None：不返回业务数据。
def test_catalog_audit_rejects_extensions_without_host_wiring(tmp_path: Path) -> None:
    empty_source_root = tmp_path / "empty-src"
    empty_source_root.mkdir()

    result = audit(tmp_path / "store", source_root=empty_source_root)

    assert result["status"] == "rejected"
    wiring = result["extension_host_wiring"]
    assert wiring["status"] == "rejected"
    assert wiring["missing_bindings"]
    assert any(
        str(issue).startswith("PLUGIN_EXTENSION_HOST_BINDING_MISSING:")
        for issue in result["issues"]
    )


# 功能：
#   验证当前启用扩展都有可识别的宿主调用入口，覆盖能力与绑定数量的基本规模。
# 输入：
#   tmp_path：本例独立的插件目录存储位置。
# 输出：
#   None：不返回业务数据。
def test_current_catalog_has_a_host_binding_for_every_enabled_extension(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"

    result = audit(tmp_path / "store", source_root=source_root)

    assert result["status"] == "accepted"
    assert result["extension_host_wiring"] == {
        "source_root": str(source_root.resolve()),
        "status": "accepted",
        "missing_bindings": [],
    }
    summary = result["summary"]
    assert summary["extension_capability_count"] >= 100
    assert summary["extension_hook_binding_count"] >= 60


# 功能：
#   验证空源码不被默认认定为已有资产导入钩子，防止检查器自己补造接入证据。
# 输入：
#   tmp_path：本例的空源码目录。
# 输出：
#   None：不返回业务数据。
def test_empty_source_does_not_manufacture_import_bindings(tmp_path):
    assert discover_host_hook_bindings(tmp_path) == set()


# 功能：
#   验证管理器构造或关闭异常时，检查器仍回收本次创建的数据库。
# 输入：
#   tmp_path：提供给检查器的隔离工作位置。
#   monkeypatch：退出测试后恢复模块工厂的夹具。
#   failure_point：管理器构造或关闭的故障注入位置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failure_point", ["construct", "close"])
def test_audit_closes_store_across_manager_failures(tmp_path, monkeypatch, failure_point):
    events = []
    monkeypatch.setattr(
        catalog_module,
        "AppStore",
        lambda _: SimpleNamespace(close=lambda: events.append("store.closed")),
    )

    # 功能：
    #   模拟管理器关闭失败，留下调用顺序供测试确认数据库是否继续回收。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def close_manager():
        events.append("manager.close")
        raise RuntimeError("manager failure")

    # 功能：
    #   构建无外部资源的管理器替身，或在构造阶段报告故障。
    # 输入：
    #   store：检查器创建的存储替身。
    #   kwargs：构造参数，本故障夹具不使用。
    # 输出：
    #   manager：提供空目录和故障关闭方法的管理器替身。
    def make_manager(store, **kwargs):
        if failure_point == "construct":
            raise RuntimeError("manager failure")
        manager = SimpleNamespace(
            list_plugins=lambda: [],
            snapshot=lambda: None,
            build_extension_registry=lambda **_: SimpleNamespace(catalog=lambda: []),
            close=close_manager,
        )
        return manager

    monkeypatch.setattr(catalog_module, "PluginManager", make_manager)
    with pytest.raises(RuntimeError, match="manager failure"):
        audit(tmp_path / "store", source_root=tmp_path)
    expected = ["manager.close", "store.closed"] if failure_point == "close" else ["store.closed"]
    assert events == expected
