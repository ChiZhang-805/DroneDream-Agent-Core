"""Asset identity and development-input fault injection, without live flight."""

import hashlib
import json
from uuid import UUID

import pytest
from test_acceptance_input_governance import _write_current_mission_manifest
from test_qualification_boundaries import _prepared_pair

import dronedream_agent_app.asset_runtime_resolver as resolver
from dronedream_agent_app.asset_import_service import AssetImportService
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.asset_pair_qualification import bind_qualification_receipt


# 功能：
#   将同一次模拟验收的两个包发布到隔离存储，返回地图记录用于边界测试。
# 输入：
#   tmp_path：本次测试目录。
# 输出：
#   record：已安装的测试地图版本记录。
def _qualified_map(tmp_path):
    source, _, _, receipt = _prepared_pair(tmp_path)
    map_source = bind_qualification_receipt(
        source_archive=source, destination_archive=tmp_path / "map-qualified.ddpkg", receipt=receipt
    )
    vehicle_source = bind_qualification_receipt(
        source_archive=tmp_path / "vehicle.ddpkg",
        destination_archive=tmp_path / "vehicle-qualified.ddpkg",
        receipt=receipt,
    )
    service = AssetImportService(AppStore(tmp_path / "store"))
    record, _ = service.promote_locally_verified_pair(
        map_source=map_source,
        map_asset_id=receipt.map_asset_id,
        vehicle_source=vehicle_source,
        vehicle_asset_id=receipt.vehicle_asset_id,
        environment_versions=receipt.environment_versions,
        operation_id="resolver-test",
    )
    return record


# 功能：
#   验证数据库中的规范化信息不能独立于清单和实际文件被替换。
# 输入：
#   tmp_path：本次测试目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_rejects_database_ir_different_from_frozen_file(tmp_path):
    record = _qualified_map(tmp_path)
    record["asset_ir"]["name"] = "Database-only replacement"
    with pytest.raises(resolver.AssetRuntimeResolutionError, match="IR"):
        resolver.resolve_versioned_map(record)


# 功能：
#   验证环境匹配不能忽略记录本身的内容摘要、成熟度或资产身份。
# 输入：
#   tmp_path：本次测试目录。
#   field：需要制造不一致的记录字段。
#   replacement：对应的错误值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    ("field", "replacement"),
    [("content_sha256", "0" * 64), ("maturity", "raw"), ("asset_id", "another-map")],
)
def test_environment_match_rejects_record_identity_drift(tmp_path, field, replacement):
    record = _qualified_map(tmp_path)
    record[field] = replacement
    assert not resolver.qualification_matches_environment(record, {"gazebo": "fixture"})


# 功能：
#   更新一个测试成员及其声明摘要，让后续失败聚焦内容语义而非普通摘要不符。
# 输入：
#   manifest：测试开发输入清单。
#   role：需要替换的文件角色。
#   content：替换后的完整文件内容。
# 输出：
#   None：不返回业务数据。
def _replace_declared_file(manifest, role, content):
    payload = json.loads(manifest.read_bytes())
    entry = payload["files"][role]
    path = manifest.parent / entry["path"]
    path.write_bytes(content)
    entry["sha256"] = hashlib.sha256(content).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")


# 功能：
#   验证重复键不能让后写入的控制器参数悄悄覆盖前一份定义。
# 输入：
#   tmp_path：隔离开发输入目录。
# 输出：
#   None：不返回业务数据。
def test_development_controller_rejects_duplicate_json_keys(tmp_path):
    manifest = _write_current_mission_manifest(tmp_path)
    _replace_declared_file(
        manifest,
        "controller_params",
        b'{"vel_limit":100,"vel_limit":1.0,"accel_limit":0.8}',
    )
    with pytest.raises(resolver.AssetRuntimeResolutionError, match="CONTROLLER_PARAMS_INVALID"):
        resolver.resolve_development_mission_input(manifest)


# 功能：
#   验证负载偏移只能使用有限数值，布尔值和字符串不能被隐式当成物理量。
# 输入：
#   tmp_path：隔离开发输入目录。
#   invalid：注入的非数值或非有限偏移。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("invalid", [True, "0.12", float("nan")])
def test_development_summary_rejects_non_numeric_offset(tmp_path, invalid):
    manifest = _write_current_mission_manifest(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_bytes())
    summary["mission_payload"]["center_above_model_root_m"] = invalid
    _replace_declared_file(manifest, "vehicle_summary", json.dumps(summary).encode())
    with pytest.raises(resolver.AssetRuntimeResolutionError, match="SUMMARY"):
        resolver.resolve_development_mission_input(manifest)


# 功能：
#   验证多模型 SDF 不会依靠取第一个模型来通过开发输入检查。
# 输入：
#   tmp_path：隔离开发输入目录。
# 输出：
#   None：不返回业务数据。
def test_development_vehicle_rejects_second_top_level_model(tmp_path):
    manifest = _write_current_mission_manifest(tmp_path)
    sdf = (
        (tmp_path / "model.sdf").read_bytes().replace(b"</sdf>", b"<model name='unrelated'/></sdf>")
    )
    _replace_declared_file(manifest, "vehicle_sdf", sdf)
    summary = json.loads((tmp_path / "summary.json").read_bytes())
    summary["payload_files_sha256"]["model.sdf"] = hashlib.sha256(sdf).hexdigest()
    _replace_declared_file(manifest, "vehicle_summary", json.dumps(summary).encode())
    with pytest.raises(resolver.AssetRuntimeResolutionError, match="SDF"):
        resolver.resolve_development_mission_input(manifest)


# 功能：
#   验证清单路径不能使用先进入子目录再回退的非规范写法。
# 输入：
#   tmp_path：隔离开发输入目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_file_rejects_noncanonical_internal_path(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "model.sdf").write_text("<sdf/>")
    with pytest.raises(resolver.AssetRuntimeResolutionError, match="FILE_INVALID"):
        resolver._checked_file(tmp_path, "nested/../model.sdf")


# 功能：
#   准备发布器的有效开发输入，测试只改变指定边界条件。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   inputs：源根目录、角色文件、固定摘要和完整清单组成的元组。
def _publication_inputs(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    manifest = _write_current_mission_manifest(root)
    payload = json.loads(manifest.read_bytes())
    files = {role: root / entry["path"] for role, entry in payload["files"].items()}
    hashes = {role: entry["sha256"] for role, entry in payload["files"].items()}
    inputs = root, files, hashes, payload
    return inputs


# 功能：
#   验证复制前被改写的源文件不能用新计算的摘要悄悄替换已固定的开发输入。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：用于控制文件变化时机的测试替换器。
# 输出：
#   None：不返回业务数据。
def test_development_publication_rejects_source_change_before_copy(tmp_path, monkeypatch):
    root, files, hashes, payload = _publication_inputs(tmp_path)
    original_copy = resolver.copy_asset_source

    # 功能：
    #   在复制控制器前修改数值，复现校验与复制之间的内容切换。
    # 输入：
    #   source：正在复制的源路径。
    #   target：调用方独占的目标输出流。
    #   limit：最大复制字节数。
    # 输出：
    #   digest：实际复制的新字节摘要。
    def changing_copy(source, target, limit):
        if source == files["controller_params"]:
            source.write_bytes(b'{"vel_limit":0.5,"accel_limit":0.8}')
        digest = original_copy(source, target, limit)
        return digest

    monkeypatch.setattr(resolver, "copy_asset_source", changing_copy)
    with pytest.raises(resolver.AssetRuntimeResolutionError, match="SOURCE_CHANGED"):
        resolver._publish_development_input(
            files, hashes, payload, tmp_path / "output", protected_roots=(root,)
        )
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.*.tmp"))
    assert root.exists()


# 功能：
#   验证发布前另一流程创建的目标不能被替换，失败清理也不能删除对方内容。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：用于控制发布时机的测试替换器。
# 输出：
#   None：不返回业务数据。
def test_development_publication_preserves_concurrent_destination(tmp_path, monkeypatch):
    root, files, hashes, payload = _publication_inputs(tmp_path)
    original_publish = resolver.publish_asset_directory

    # 功能：
    #   紧挨发布前插入另一个所有者的目标目录。
    # 输入：
    #   source：当前流程的独占暂存目录。
    #   destination：即将发生竞争的发布目标。
    # 输出：
    #   None：不返回业务数据。
    def concurrent_publish(source, destination):
        destination.mkdir()
        (destination / "owner.txt").write_text("another writer")
        original_publish(source, destination)

    monkeypatch.setattr(resolver, "publish_asset_directory", concurrent_publish)
    with pytest.raises(FileExistsError):
        resolver._publish_development_input(
            files, hashes, payload, tmp_path / "output", protected_roots=(root,)
        )
    assert (tmp_path / "output" / "owner.txt").read_text() == "another writer"
    assert not list(tmp_path.glob(".output.*.tmp"))


# 功能：
#   验证随机暂存名恰好碰撞时，不清理或复用已有目录。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：用于固定随机标识的测试替换器。
# 输出：
#   None：不返回业务数据。
def test_development_publication_preserves_colliding_staging(tmp_path, monkeypatch):
    root, files, hashes, payload = _publication_inputs(tmp_path)
    identifier = UUID("11111111-1111-1111-1111-111111111111")
    monkeypatch.setattr(resolver, "uuid4", lambda: identifier)
    staging = tmp_path / f".output.{identifier.hex}.tmp"
    staging.mkdir()
    (staging / "owner.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        resolver._publish_development_input(
            files, hashes, payload, tmp_path / "output", protected_roots=(root,)
        )
    assert (staging / "owner.txt").read_text() == "preserve"


# 功能：
#   验证平铺文件名碰撞在写入前被拒绝，不把后复制的角色覆盖前一个角色。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_development_publication_rejects_filename_collision(tmp_path):
    root, files, hashes, payload = _publication_inputs(tmp_path)
    files["controller_params"] = files["vehicle_metadata"]
    with pytest.raises(resolver.AssetRuntimeResolutionError, match="FILENAME_COLLISION"):
        resolver._publish_development_input(
            files, hashes, payload, tmp_path / "output", protected_roots=(root,)
        )
    assert not list(tmp_path.glob(".output.*.tmp"))
