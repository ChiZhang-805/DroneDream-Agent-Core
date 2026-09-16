"""Local SQLite corruption and write-boundary tests; no installed account state."""

import hashlib
import json
from datetime import UTC, datetime

import pytest

from dronedream_agent_app.plugin_manager import PluginManager
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.asset_packages import AssetImportJob
from dronedream_agent_core.asset_pair_qualification import AssetPairQualificationJob
from dronedream_agent_core.model_harness.boundary import ModelHarnessExecutionAuthority


# 功能：
#   为每条反例创建独立数据库，失败也关闭连接，不操作用户账户。
# 输入：
#   tmp_path：pytest 分配的独立目录。
# 输出：
#   store：本测试独占的应用存储。
@pytest.fixture
def store(tmp_path):
    store = AppStore(tmp_path)
    try:
        yield store
    finally:
        store.close()


# 功能：
#   建立具有真实隔离源文件的最小导入任务，供索引身份检查使用。
# 输入：
#   store：独立测试存储。
# 输出：
#   job：已保存的导入任务。
@pytest.fixture
def import_job(store):
    source = store.asset_quarantine_root / "source.ddpkg"
    source.write_bytes(b"fixture")
    now = datetime.now(UTC)
    job = AssetImportJob(
        job_id="asset-job-" + "a" * 24,
        owner_id="test-owner",
        source_name="source.ddpkg",
        source_format="ddpkg",
        package_sha256="a" * 64,
        progress_percent=0,
        revision=0,
        created_at=now,
        updated_at=now,
    )
    store.save_asset_import_job(
        job, source_format="ddpkg", source_path=source, package_sha256=job.package_sha256
    )
    return job


# 功能：
#   建立待验收资产对及本地目录，只测试存储绑定，不伪造飞行合格结果。
# 输入：
#   store：独立测试存储。
# 输出：
#   job：已保存的 created 状态资格任务。
@pytest.fixture
def pair_job(store):
    now = datetime.now(UTC)
    job = AssetPairQualificationJob(
        job_id="asset-qualification-job-" + "a" * 24,
        owner_id="test-owner",
        map_asset_id="fixture-map",
        map_content_sha256="a" * 64,
        vehicle_asset_id="fixture-vehicle",
        vehicle_content_sha256="b" * 64,
        created_at=now,
        updated_at=now,
    )
    workspace = store.asset_qualification_root / job.job_id
    map_root = store.asset_versions_root / "map"
    vehicle_root = store.asset_versions_root / "vehicle"
    for path in (workspace, map_root, vehicle_root):
        path.mkdir()
    store.save_asset_pair_qualification_job(
        job, workspace_root=workspace, map_bundle_root=map_root, vehicle_bundle_root=vehicle_root
    )
    return job


# 功能：
#   验证数据库中的非 0/1 标记不能被宽松转换成已启用或已置顶。
# 输入：
#   store：独立测试存储。
#   bad：模拟损坏的 SQLite 标记。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", ["false", 2, -1])
def test_corrupt_sqlite_flags_are_not_enablement(store, bad):
    thread = store.create_thread("fixture", "fixture")
    with store.transaction() as connection:
        connection.execute("UPDATE threads SET pinned=?", (bad,))
    with pytest.raises(ValueError, match="BOOLEAN"):
        store.get_thread(thread["thread_id"])


# 功能：
#   验证已落库的记忆开关也严格校验，不能只在写入 API 入口检查。
# 输入：
#   store：独立测试存储。
#   bad：非法持久化开关值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", ["false", 0, None])
def test_stored_memory_consent_requires_boolean(store, bad):
    with store.transaction() as connection:
        connection.execute("INSERT INTO settings VALUES('memory_enabled',?)", (json.dumps(bad),))
    with pytest.raises(ValueError, match="BOOLEAN"):
        store.get_settings()


# 功能：
#   验证配置及消息重复 JSON 键会被拒绝，防止读取时静默保留最后一个值。
# 输入：
#   store：独立测试存储。
#   source：需要注入重复键的持久化入口。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("source", ["settings", "message"])
def test_ambiguous_json_is_rejected_on_read(store, source):
    thread = store.create_thread("fixture", "fixture")
    if source == "settings":
        with store.transaction() as connection:
            connection.execute("INSERT INTO settings VALUES('extra',?)", ('{"x":1,"x":2}',))
        read = store.get_settings
    else:
        store.append_message(thread["thread_id"], role="user", kind="text", content="fixture")
        with store.transaction() as connection:
            connection.execute("UPDATE messages SET metadata_json=?", ('{"x":1,"x":2}',))

        # 功能：
        #   通过任务读取入口触发元数据解码，而不是直接测试 JSON 库。
        # 输入：
        #   无；使用外层 store 与 thread。
        # 输出：
        #   result：包含消息的任务记录。
        def read():
            result = store.get_thread(thread["thread_id"])
            return result

    with pytest.raises(ValueError, match="DUPLICATE"):
        read()


# 功能：
#   验证写入非字符串 JSON 键时拒绝整个更新，不把不同类型的键合并成同名字段。
# 输入：
#   store：独立测试存储。
# 输出：
#   None：不返回业务数据。
def test_settings_write_rejects_key_coercion_atomically(store):
    with pytest.raises(ValueError, match="KEY"):
        store.patch_settings({"theme": "dark", "extra": {1: "first", "1": "second"}})
    assert store.get_settings()["theme"] == "system"


# 功能：
#   验证合法形状但属于另一任务的 JSON 不能从当前导入任务 ID 下返回。
# 输入：
#   store：独立测试存储。
#   import_job：合法导入任务。
# 输出：
#   None：不返回业务数据。
def test_import_json_must_match_requested_job_id(store, import_job):
    other = import_job.model_copy(update={"job_id": "asset-job-" + "b" * 24})
    with store.transaction() as connection:
        connection.execute("UPDATE asset_import_jobs SET job_json=?", (other.model_dump_json(),))
    with pytest.raises(ValueError, match="MISMATCH"):
        store.load_asset_import_job(import_job.job_id)


# 功能：
#   验证资格任务的 JSON 与独立查询索引一致，防止串任务或串地图。
# 输入：
#   store：独立测试存储。
#   pair_job：合法资格任务。
#   field：被替换的身份字段。
#   value：另一合法形状的字段值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "field,value",
    [
        ("job_id", "asset-qualification-job-" + "b" * 24),
        ("map_asset_id", "other-map"),
        ("vehicle_content_sha256", "c" * 64),
    ],
)
def test_pair_json_must_match_indexed_identity(store, pair_job, field, value):
    other = pair_job.model_copy(update={field: value})
    with store.transaction() as connection:
        connection.execute(
            "UPDATE asset_pair_qualification_jobs SET job_json=?", (other.model_dump_json(),)
        )
    with pytest.raises(ValueError, match="MISMATCH"):
        store.load_asset_pair_qualification_job(pair_job.job_id)


# 功能：
#   验证后续保存不能悄悄忽略调用者替换的工作路径，否则返回成功会掩盖源目录不一致。
# 输入：
#   store：独立测试存储。
#   pair_job：已保存的资格任务。
# 输出：
#   None：不返回业务数据。
def test_pair_update_rejects_workspace_drift(store, pair_job):
    before = store.asset_pair_qualification_paths(pair_job.job_id)
    other = store.asset_qualification_root / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="IDENTITY_CHANGED"):
        store.save_asset_pair_qualification_job(pair_job, workspace_root=other)
    assert store.asset_pair_qualification_paths(pair_job.job_id) == before


# 功能：
#   验证版本清单身份与请求键不一致时拒绝激活，不能覆盖另一插件的目录项。
# 输入：
#   store：独立测试存储。
# 输出：
#   None：不返回业务数据。
def test_plugin_activation_rejects_foreign_manifest(store):
    manager = PluginManager(store)
    try:
        own = store.get_plugin("model.openai")
        foreign = store.get_plugin("navigation.shortest-route")
        with store.transaction() as connection:
            connection.execute(
                "UPDATE plugin_versions SET manifest_json=? WHERE plugin_id=? AND version=?",
                (json.dumps(foreign["manifest"]), own["plugin_id"], own["version"]),
            )
        with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
            store.activate_plugin_version(own["plugin_id"], own["version"])
        assert store.get_plugin(foreign["plugin_id"]) == foreign
    finally:
        manager.close()


# 功能：
#   验证没有调用历史时成功数及错误数为零，不能把 SQL 空聚合误当成未知数。
# 输入：
#   store：独立测试存储。
# 输出：
#   None：不返回业务数据。
def test_empty_usage_has_zero_counts(store):
    summary = store.summarize_plugin_usage("fixture.plugin")
    assert summary["calls"] == summary["successes"] == summary["errors"] == 0


# 功能：
#   验证延迟和流量只接受有限非负数，非法事件不能影响后续汇总。
# 输入：
#   store：独立测试存储。
#   field：被损坏的指标名。
#   value：非法指标值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "field,value",
    [
        ("duration_ms", float("inf")),
        ("duration_ms", True),
        ("duration_ms", -1),
        ("input_bytes", -1),
        ("output_bytes", 1.5),
        ("input_bytes", True),
    ],
)
def test_usage_rejects_invalid_measurements(store, field, value):
    event = dict(
        invocation_id="fixture-call",
        plugin_id="fixture.plugin",
        plugin_version="1.0.0",
        capability_id="fixture",
        slot_id="fixture",
        invocation_kind="tool",
        outcome="success",
        duration_ms=2.5,
        input_bytes=10,
        output_bytes=20,
        created_at=datetime.now(UTC).isoformat(),
    )
    event[field] = value
    with pytest.raises(ValueError, match="METRIC"):
        store.record_plugin_usage(event)
    assert store.list_plugin_usage() == []


# 功能：
#   为隔离数据库构造具备一致内容摘要的一次性绑定，不运行规划、仿真或真实授权。
# 输入：
#   store：独立测试存储。
# 输出：
#   authority：已写入测试数据库的一次性绑定。
@pytest.fixture
def authority(store):
    thread = store.create_thread("fixture", "fixture")
    payload = dict(
        schema_version="dronedream.model-harness-execution-authority.v1",
        authority_id="execution-authority-" + "a" * 32,
        thread_id=thread["thread_id"],
        plan_revision_id="plan-" + "b" * 32,
        contract_id="fixture-contract",
        prepared_mission_sha256="c" * 64,
        owner_binding_sha256="d" * 64,
        tenant_binding_sha256="e" * 64,
        control_plane_selection_sha256="f" * 64,
        plugin_snapshot_id="plugin-snapshot-" + "a" * 24,
        plugin_catalog_sha256="b" * 64,
        issued_at=datetime.now(UTC).isoformat(),
        single_use=True,
        reusable=False,
    )
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    authority = ModelHarnessExecutionAuthority(**payload, authority_sha256=digest)
    store.issue_execution_authority(authority)
    return authority


# 功能：
#   验证执行绑定索引漂移同时阻止读取与消费，失败不改变已签发状态。
# 输入：
#   store：独立测试存储。
#   authority：已经保存的测试绑定。
#   field：需要损坏的索引字段。
#   value：与正文不一致的索引值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "field,value",
    [
        ("plan_revision_id", "plan-" + "c" * 32),
        ("authority_sha256", "1" * 64),
    ],
)
def test_authority_index_drift_blocks_read_and_consume(store, authority, field, value):
    with store.transaction() as connection:
        connection.execute(f"UPDATE model_harness_execution_authorities SET {field}=?", (value,))
    with pytest.raises(ValueError, match="BINDING_MISMATCH"):
        store.execution_authority(authority.authority_id)
    with pytest.raises(ValueError, match="BINDING_MISMATCH"):
        store.consume_execution_authority(authority, execution_id="fixture-execution")
    status = store._connection.execute(
        "SELECT status FROM model_harness_execution_authorities"
    ).fetchone()[0]
    assert status == "issued"


# 功能：
#   验证被修改但未重新绑定摘要的模型在保存前拒绝，不作废已有的有效绑定。
# 输入：
#   store：独立测试存储。
#   authority：原有效绑定。
# 输出：
#   None：不返回业务数据。
def test_mutated_authority_cannot_supersede_valid_grant(store, authority):
    invalid = authority.model_copy(update={"plan_revision_id": "plan-" + "e" * 32})
    with pytest.raises(ValueError):
        store.issue_execution_authority(invalid)
    assert store.execution_authority(authority.authority_id) == (authority, "issued")


# 功能：
#   验证消费操作必须有非空执行身份，空值不会提前消耗一次性绑定。
# 输入：
#   store：独立测试存储。
#   authority：原有效绑定。
#   value：非法执行标识。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", ["", " ", None, True])
def test_empty_execution_identity_does_not_consume(store, authority, value):
    with pytest.raises(ValueError, match="EXECUTION_ID_INVALID"):
        store.consume_execution_authority(authority, execution_id=value)
    assert store.execution_authority(authority.authority_id)[1] == "issued"


# 功能：
#   验证退役操作不能删除资格根目录内另一任务的目录，使用真实临时文件检查保留结果。
# 输入：
#   store：独立测试存储。
#   pair_job：原资格任务。
# 输出：
#   None：不返回业务数据。
def test_retirement_cannot_remove_another_job_workspace(store, pair_job):
    qualified = pair_job.model_copy(
        update=dict(
            state="qualified",
            progress_percent=100,
            qualification_id="asset-qualification-" + "a" * 24,
            result_map_content_sha256="c" * 64,
            result_vehicle_content_sha256="d" * 64,
        )
    )
    store.save_asset_pair_qualification_job(qualified)
    foreign = store.asset_qualification_root / "another-job"
    foreign.mkdir()
    preserved = foreign / "user-evidence.txt"
    preserved.write_text("preserve", encoding="utf-8")
    with store.transaction() as connection:
        connection.execute(
            "UPDATE asset_pair_qualification_jobs SET workspace_root=?", (str(foreign),)
        )
    with pytest.raises(ValueError, match="PATH_INVALID"):
        store.retire_bundled_asset_pair_qualification(
            qualification_id=qualified.qualification_id,
            map_content_sha256=qualified.result_map_content_sha256,
            vehicle_content_sha256=qualified.result_vehicle_content_sha256,
        )
    assert preserved.read_text(encoding="utf-8") == "preserve"
    assert store.load_asset_pair_qualification_job(pair_job.job_id) == qualified


# 功能：
#   验证正规任务目录仍能退役，且重复调用只返回未找到，不影响两个源资产包目录。
# 输入：
#   store：独立测试存储。
#   pair_job：原资格任务。
# 输出：
#   None：不返回业务数据。
def test_retirement_removes_only_exact_canonical_workspace(store, pair_job):
    qualified = pair_job.model_copy(
        update=dict(
            state="qualified",
            progress_percent=100,
            qualification_id="asset-qualification-" + "a" * 24,
            result_map_content_sha256="c" * 64,
            result_vehicle_content_sha256="d" * 64,
        )
    )
    store.save_asset_pair_qualification_job(qualified)
    workspace, map_root, vehicle_root = store.asset_pair_qualification_paths(pair_job.job_id)
    arguments = dict(
        qualification_id=qualified.qualification_id,
        map_content_sha256=qualified.result_map_content_sha256,
        vehicle_content_sha256=qualified.result_vehicle_content_sha256,
    )
    assert store.retire_bundled_asset_pair_qualification(**arguments) is True
    assert not workspace.exists()
    assert map_root.is_dir() and vehicle_root.is_dir()
    assert store.retire_bundled_asset_pair_qualification(**arguments) is False


# 功能：
#   验证插件配置拒绝会发生键合并的 JSON，并保留先前配置，不留下已提交但无法读取的记录。
# 输入：
#   store：独立测试存储。
# 输出：
#   None：不返回业务数据。
def test_plugin_configuration_rejects_ambiguous_write_without_losing_original(store):
    manager = PluginManager(store)
    try:
        before = store.save_plugin_configuration("model.openai", {"fixture": "preserved"})
        with pytest.raises(ValueError, match="KEY"):
            store.save_plugin_configuration("model.openai", {"extra": {1: "a", "1": "b"}})
        assert store.get_plugin_configuration("model.openai") == before
    finally:
        manager.close()


# 功能：
#   验证非法顶层设置键不能被 SQLite 转为字符串后覆盖其他设置。
# 输入：
#   store：独立测试存储。
# 输出：
#   None：不返回业务数据。
def test_setting_names_require_strings_before_writing(store):
    with pytest.raises(ValueError, match="KEY"):
        store.patch_settings({"theme": "dark", 1: "a", "1": "b"})
    assert store.get_settings()["theme"] == "system"


# 功能：
#   验证模型连接开关与本机记忆偏好真正跨关闭重开保留，而不是只存在于进程缓存。
# 输入：
#   store：独立测试存储。
# 输出：
#   None：不返回业务数据。
def test_model_and_memory_flags_survive_reopen(store):
    store.patch_settings({"memory_enabled": False, "remember_task_preferences": False})
    profile = dict(
        profile_id="fixture",
        display_name="fixture",
        provider="custom",
        icon="custom",
        base_url="https://example.invalid/v1",
        api_style="openai-chat",
        model_id="fixture",
        enabled=False,
    )
    store.save_custom_model(profile)
    store.close()
    reopened = AppStore(store.root)
    try:
        assert reopened.get_settings()["memory_enabled"] is False
        assert reopened.get_settings()["remember_task_preferences"] is False
        assert reopened.get_custom_model("fixture")["enabled"] is False
    finally:
        reopened.close()
