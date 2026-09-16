"""Packaging/trust fixtures operate only inside pytest-owned directories."""

from __future__ import annotations

import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from dronedream_agent_core.plugin_contracts import PluginManifest
from dronedream_agent_core.plugin_panels import materialize_panel, validate_panel_document
from dronedream_agent_core.plugin_sdk import (
    _load_signing_key,
    _package_files,
    build_plugin_bundle,
    generate_publisher_key,
    scaffold_plugin,
    validate_plugin_source,
)
from dronedream_agent_core.plugin_trust import PluginTrustStore, TrustStoreError


# 功能：
#   在测试目录内创建声明式 UI 插件源码，不安装插件或运行外部程序。
# 输入：
#   tmp_path：本测试拥有的临时根目录。
# 输出：
#   source：生成的插件源码目录。
def source_fixture(tmp_path):
    source = scaffold_plugin(
        tmp_path / "source",
        plugin_id="fixture.panel",
        name="Fixture",
        publisher="Fixture",
        kind="ui",
    )
    return source


# 功能：
#   构造含配置表单和刷新动作的最小面板文档，供语义校验及引用隔离测试使用。
# 输入：
#   无。
# 输出：
#   panel：独立创建的声明式面板字典。
def panel_fixture():
    panel = {
        "schema_version": "dronedream.ui-panel.v1",
        "title": "Fixture",
        "sections": [
            {
                "section_id": "main",
                "title": "Main",
                "widgets": [
                    {
                        "widget_id": "config",
                        "kind": "configuration-form",
                        "label": "Config",
                        "source": "configuration",
                    }
                ],
                "actions": [{"action_id": "panel.refresh", "label": "Refresh"}],
            }
        ],
    }
    return panel


# 功能：
#   验证打包排除指定签名私钥和常见凭据文件，同时保留子目录中普通同名清单数据。
# 输入：
#   tmp_path：源码、一次性测试密钥和输出包所在的独立目录。
# 输出：
#   None：不返回业务数据。
def test_packaging_never_copies_selected_key_or_known_credential_files(tmp_path):
    source = source_fixture(tmp_path)
    key = source / "private-custom-name.json"
    generate_publisher_key(key, key_id="fixture.publisher", publisher="Fixture")
    for name in (".env", ".env.local", "TEST.PEM", "secret.pfx"):
        (source / name).write_text("inert-secret-fixture", encoding="utf-8")
    nested = source / "data"
    nested.mkdir()
    (nested / "plugin.json").write_text('{"nested":"data"}', encoding="utf-8")
    output = tmp_path / "plugin.zip"
    build_plugin_bundle(source, output, signing_key=key)
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "data/plugin.json" in names
    assert not names.intersection({key.name, ".env", ".env.local", "TEST.PEM", "secret.pfx"})


# 功能：
#   验证包输出不能覆盖源文件或已存在的产物，失败后原始字节保持不变。
# 输入：
#   tmp_path：隔离的源码及产物目录。
# 输出：
#   None：不返回业务数据。
def test_packaging_does_not_overwrite_source_or_existing_outputs(tmp_path):
    source = source_fixture(tmp_path)
    manifest = (source / "plugin.json").read_bytes()
    with pytest.raises(ValueError, match="OUTSIDE_SOURCE"):
        build_plugin_bundle(source, source / "plugin.json")
    assert (source / "plugin.json").read_bytes() == manifest
    output = tmp_path / "existing.zip"
    output.write_bytes(b"existing-user-artifact")
    with pytest.raises(ValueError, match="OUTPUT_EXISTS"):
        build_plugin_bundle(source, output)
    assert output.read_bytes() == b"existing-user-artifact"


# 功能：
#   验证检查之后才出现的竞争输出仍不会被覆盖，且本次构建暂存文件能正确回收。
# 输入：
#   tmp_path：隔离的源码及产物目录。
#   monkeypatch：在独占发布时插入竞争写入。
# 输出：
#   None：不返回业务数据。
def test_exclusive_publish_preserves_a_concurrently_created_output(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    output = tmp_path / "plugin.zip"
    original_link = os.link

    # 功能：
    #   在正式硬链接发布前写入另一份目标，构造检查与发布之间的竞争。
    # 输入：
    #   staging：正式代码创建的暂存文件。
    #   destination：准备发布的目标路径。
    # 输出：
    #   result：底层硬链接操作的返回值。
    def create_before_link(staging, destination):
        destination.write_bytes(b"other-writer")
        result = original_link(staging, destination)
        return result

    monkeypatch.setattr("dronedream_agent_core.plugin_sdk.os.link", create_before_link)
    with pytest.raises(FileExistsError):
        build_plugin_bundle(source, output)
    assert output.read_bytes() == b"other-writer"
    assert not list(tmp_path.glob(".plugin-build-*"))


# 功能：
#   验证源码遍历分别限制文件数与所有条目数，不先无限收集后再检查预算。
# 输入：
#   tmp_path：插件夹具目录。
#   monkeypatch：缩小遍历预算以便用小夹具触发边界。
# 输出：
#   None：不返回业务数据。
def test_package_file_and_entry_budgets_apply_before_unbounded_collection(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    monkeypatch.setattr("dronedream_agent_core.plugin_sdk._MAX_SOURCE_FILES", 1)
    with pytest.raises(ValueError, match="FILE_LIMIT"):
        _package_files(source)
    monkeypatch.setattr("dronedream_agent_core.plugin_sdk._MAX_SOURCE_FILES", 100)
    monkeypatch.setattr("dronedream_agent_core.plugin_sdk._MAX_SOURCE_ENTRIES", 1)
    with pytest.raises(ValueError, match="ENTRY_LIMIT"):
        _package_files(source)


# 功能：
#   验证打包拒绝指向源码目录外的文件链接；宿主不允许创建链接时明确跳过此用例。
# 输入：
#   tmp_path：包含源目录与外部模拟文件的测试目录。
# 输出：
#   None：不返回业务数据。
def test_package_rejects_symlink_to_files_outside_the_source(tmp_path):
    source = source_fixture(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("do not package", encoding="utf-8")
    try:
        (source / "link.txt").symlink_to(outside)
    except OSError:
        pytest.skip("This host does not permit test symlink creation")
    with pytest.raises(ValueError, match="LINK_NOT_ALLOWED"):
        _package_files(source)


# 功能：
#   验证一次性测试密钥不会覆盖现存密钥、公钥响应不携带私钥，且加载时检查密钥配对。
# 输入：
#   tmp_path：仅存放本测试密钥的目录。
# 输出：
#   None：不返回业务数据。
def test_key_generation_is_exclusive_and_key_pairs_are_validated(tmp_path):
    key = tmp_path / "key.json"
    public = generate_publisher_key(key, key_id="fixture.publisher", publisher="Fixture")
    assert "private_key_base64" not in public
    before = key.read_bytes()
    with pytest.raises(ValueError, match="KEY_EXISTS"):
        generate_publisher_key(key, key_id="fixture.publisher", publisher="Fixture")
    assert key.read_bytes() == before
    value = json.loads(before)
    value["public_key_base64"] = "not-the-public-key"
    key.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="PAIR_MISMATCH"):
        _load_signing_key(key)


# 功能：
#   验证不支持的插件种类、越界标识或空名称在创建目录前即被拒绝。
# 输入：
#   tmp_path：用于检查是否遗留半成品目录的测试根目录。
# 输出：
#   None：不返回业务数据。
def test_invalid_scaffold_arguments_do_not_create_partial_source(tmp_path):
    root = tmp_path / "invalid"
    for changes in ({"kind": "other"}, {"plugin_id": "../invalid"}, {"name": ""}):
        arguments = {
            "plugin_id": "fixture.panel",
            "name": "Fixture",
            "publisher": "Fixture",
            "kind": "ui",
        }
        arguments.update(changes)
        with pytest.raises(ValueError):
            scaffold_plugin(root, **arguments)
        assert not root.exists()


# 功能：
#   验证源码检查阶段就校验面板语义，布尔值不能冒充数值限制混入包内。
# 输入：
#   tmp_path：可修改的测试插件源码目录。
# 输出：
#   None：不返回业务数据。
def test_source_validation_checks_panel_semantics_before_packaging(tmp_path):
    source = source_fixture(tmp_path)
    panel = source / "ui/panel.json"
    value = json.loads(panel.read_text(encoding="utf-8"))
    value["sections"][0]["widgets"][0]["limit"] = True
    panel.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValidationError):
        validate_plugin_source(source)


# 功能：
#   验证修改已渲染面板中的配置或 Schema，不会回写宿主持有的原始对象。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_panel_output_detaches_configuration_and_schema():
    document = validate_panel_document(panel_fixture())
    configuration = {"nested": {"value": 1}}
    schema = {"type": "object", "properties": {"nested": {"type": "object"}}}
    output = materialize_panel(
        document, sources={"configuration": configuration}, configuration_schema=schema
    )
    widget = output["sections"][0]["widgets"][0]
    widget["resolved"]["nested"]["value"] = 2
    widget["schema"]["properties"].clear()
    assert configuration["nested"]["value"] == 1
    assert "nested" in schema["properties"]


# 功能：
#   验证面板拒绝重复分区、重复动作及非法数值，不允许有歧义的交互标识。
# 输入：
#   change：本次注入的面板错误种类。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("change", ["section", "action", "boolean-limit", "nonfinite"])
def test_panel_rejects_ambiguous_identity_and_numeric_coercion(change):
    value = panel_fixture()
    if change == "section":
        value["sections"].append({"section_id": "main", "title": "Duplicate"})
    elif change == "action":
        value["sections"][0]["actions"] *= 2
    elif change == "boolean-limit":
        value["sections"][0]["widgets"][0]["limit"] = True
    else:
        value["sections"][0]["widgets"][0]["value"] = float("nan")
    with pytest.raises(ValidationError):
        validate_panel_document(value)


# 功能：
#   验证面板生成时重新校验被修改的模型，并限制运行配置的序列化大小。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_panel_revalidates_mutable_models_and_bounds_nested_runtime_values():
    document = validate_panel_document(panel_fixture())
    with pytest.raises(ValueError, match="SIZE_LIMIT"):
        materialize_panel(
            document, sources={"configuration": {"large": "x" * 300_000}}, configuration_schema={}
        )
    document.sections[0].widgets[0].limit = 101
    with pytest.raises(ValidationError):
        materialize_panel(document, sources={}, configuration_schema={})


# 功能：
#   验证多个信任库实例并发撤销时通过锁协调读改写，所有撤销摘要最终都被保存。
# 输入：
#   tmp_path：模拟信任库及其锁文件所在的测试目录。
# 输出：
#   None：不返回业务数据。
def test_trust_store_read_modify_write_does_not_lose_parallel_revocations(tmp_path):
    path = tmp_path / "trust.json"
    stores = [PluginTrustStore(path) for _ in range(4)]
    digests = [f"{index:064x}" for index in range(24)]
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(
            workers.map(
                lambda item: stores[item[0] % 4].revoke_package(item[1]), enumerate(digests)
            )
        )
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert set(persisted["revoked_packages"]) == set(digests)
    assert path.with_name("trust.json.lock").is_file()


# 功能：
#   验证同一发行者标识绑定后不能悄悄换成另一把公钥，不影响真实产品签名密钥。
# 输入：
#   tmp_path：两组一次性测试密钥及隔离信任库所在目录。
# 输出：
#   None：不返回业务数据。
def test_publisher_id_cannot_silently_rotate_to_another_key(tmp_path):
    first = generate_publisher_key(
        tmp_path / "first.json", key_id="fixture.publisher", publisher="Fixture"
    )
    second = generate_publisher_key(
        tmp_path / "second.json", key_id="fixture.publisher", publisher="Fixture"
    )
    trust = PluginTrustStore(tmp_path / "trust.json")
    trust.add_publisher(
        key_id=first["key_id"],
        publisher=first["publisher"],
        public_key_base64=first["public_key_base64"],
    )
    with pytest.raises(TrustStoreError, match="ALREADY_BOUND"):
        trust.add_publisher(
            key_id=second["key_id"],
            publisher=second["publisher"],
            public_key_base64=second["public_key_base64"],
        )


# 功能：
#   验证本地批准绑定具体清单内容，且已撤销的包不能通过再次批准重新获得信任。
# 输入：
#   tmp_path：源码夹具及模拟信任库所在目录。
# 输出：
#   None：不返回业务数据。
def test_local_approval_binds_manifest_and_revocation_cannot_be_undone_by_approval(tmp_path):
    source = source_fixture(tmp_path)
    manifest = PluginManifest.model_validate_json(
        (source / "plugin.json").read_text(encoding="utf-8")
    )
    trust = PluginTrustStore(tmp_path / "trust.json")
    digest = "a" * 64
    trust.approve_local(manifest, digest)
    assert trust.verify(manifest, digest).status == "local-approved"
    manifest.description = "Different contents, same plugin coordinate"
    assert trust.verify(manifest, digest).issue_codes == ["PLUGIN_APPROVED_MANIFEST_MISMATCH"]
    trust.revoke_package(digest)
    with pytest.raises(TrustStoreError, match="REVOKED"):
        trust.approve_local(manifest, digest)
    assert trust.verify(manifest, digest).status == "revoked"


# 功能：
#   验证信任库容器类型损坏时拒绝打开，不将损坏的撤销记录静默当作空列表。
# 输入：
#   tmp_path：用于损坏测试的独立信任库目录。
# 输出：
#   None：不返回业务数据。
def test_corrupt_trust_containers_fail_closed_on_open(tmp_path):
    path = tmp_path / "trust.json"
    PluginTrustStore(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["revoked_packages"] = {}
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(TrustStoreError, match="STORE_INVALID"):
        PluginTrustStore(path)
