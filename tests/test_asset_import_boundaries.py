"""Local synthetic import failures; these tests do not certify any real flight."""

import json
from pathlib import Path

import pytest
from test_asset_packages import _write_package
from test_asset_pair_qualification import _install_fixture_version
from test_qualification_boundaries import _prepared_pair

import dronedream_agent_app.asset_import_service as imports
from dronedream_agent_app.storage import AppStore, AssetImportError
from dronedream_agent_core.asset_packages import inspect_ddpkg
from dronedream_agent_core.asset_pair_qualification import (
    bind_qualification_receipt,
    build_pair_qualification_receipt,
)


# 功能：
#   构造一个已安装但未取得飞行资格的隔离资产，供导出与目录检查使用。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   fixture：导入服务、源包、检查结果与已安装目录组成的元组。
def _installed_asset(tmp_path):
    archive = tmp_path / "source.ddpkg"
    _write_package(archive)
    store = AppStore(tmp_path / "store")
    identity = _install_fixture_version(store, archive)
    version = store.get_asset_version(*identity)
    fixture = (
        imports.AssetImportService(store),
        archive,
        inspect_ddpkg(archive),
        Path(version["bundle_root"]),
    )
    return fixture


# 功能：
#   验证历史迁移声明无效时，在默认资产安装动作发生前就拒绝索引。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：记录是否提前进入安装流程的测试工具。
# 输出：
#   None：不返回业务数据。
def test_invalid_retirement_index_is_rejected_before_installation(tmp_path, monkeypatch):
    service = imports.AssetImportService(AppStore(tmp_path / "store"))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "dronedream.bundled-assets.v2",
                "qualified_pair": {"packages": []},
                "superseded_qualified_pairs": [False],
            }
        )
    )
    installed = []
    monkeypatch.setattr(
        service, "_seed_bundled_qualified_pair", lambda **kwargs: installed.append(True) or []
    )
    with pytest.raises(AssetImportError, match="SUPERSEDED_PAIR_INVALID"):
        service.seed_bundled_sources(bundle)
    assert installed == []
    assert service.store.list_asset_versions() == []


# 功能：
#   验证两个各自有效、但携带不同回执的包不能被当成同一次配对验收入库。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_pair_requires_identical_receipt_bytes_before_any_publication(tmp_path):
    source, work_root, plan, receipt = _prepared_pair(tmp_path)
    modified_evidence = {**receipt.runtime_evidence, "test_note": "different receipt"}
    second_receipt = build_pair_qualification_receipt(
        plan=plan,
        work_root=work_root,
        runtime_evidence=modified_evidence,
        environment_versions=receipt.environment_versions,
        qualified_at=receipt.qualified_at,
    )
    map_source = bind_qualification_receipt(
        source_archive=source, destination_archive=tmp_path / "map-qualified.ddpkg", receipt=receipt
    )
    vehicle_source = bind_qualification_receipt(
        source_archive=tmp_path / "vehicle.ddpkg",
        destination_archive=tmp_path / "vehicle-qualified.ddpkg",
        receipt=second_receipt,
    )
    service = imports.AssetImportService(AppStore(tmp_path / "store"))
    with pytest.raises(AssetImportError, match="PAIR_RECEIPT_MISMATCH"):
        service.promote_locally_verified_pair(
            map_source=map_source,
            map_asset_id=receipt.map_asset_id,
            vehicle_source=vehicle_source,
            vehicle_asset_id=receipt.vehicle_asset_id,
            environment_versions=receipt.environment_versions,
            operation_id="mixed-pair",
        )
    assert service.store.list_asset_versions() == []
    assert list(service.store.asset_versions_root.iterdir()) == []


# 功能：
#   验证导出不能覆盖已有目标，也不能清理另一流程的旧式临时文件。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_export_preserves_existing_destination_and_temporary(tmp_path):
    service, _, inspected, _ = _installed_asset(tmp_path)
    destination = tmp_path / "result.ddpkg"
    temporary = tmp_path / "result.ddpkg.tmp"
    destination.write_bytes(b"existing output")
    temporary.write_bytes(b"another writer")
    with pytest.raises((ValueError, FileExistsError)):
        service.export_version(
            asset_id=inspected.manifest.asset_id,
            content_sha256=inspected.manifest.content_sha256,
            destination=destination,
        )
    assert destination.read_bytes() == b"existing output"
    assert temporary.read_bytes() == b"another writer"


# 功能：
#   验证复用已安装目录前，清单本身及额外未索引文件也受到检查。
# 输入：
#   tmp_path：测试独立目录。
#   corruption：本次制造的清单或额外文件缺陷。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("corruption", ["manifest", "extra-file"])
def test_reusing_content_directory_rechecks_complete_index(tmp_path, corruption):
    service, archive, inspected, root = _installed_asset(tmp_path)
    if corruption == "manifest":
        (root / "manifest.json").write_text("{}", encoding="utf-8")
    else:
        (root / "normalized" / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        service._promote(archive, inspected, "repeat")


# 功能：
#   验证非法默认资产索引统一转换为产品错误，不直接泄露解析器类型异常。
# 输入：
#   tmp_path：测试独立目录。
#   raw：列表或重复键构成的非法索引。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        '{"schema_version":"obsolete","schema_version":"dronedream.bundled-assets.v2"}',
    ],
)
def test_bundled_index_rejects_ambiguous_root(tmp_path, raw):
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "index.json").write_text(raw, encoding="utf-8")
    service = imports.AssetImportService(AppStore(tmp_path / "store"))
    with pytest.raises(AssetImportError, match="^BUNDLED_ASSET_INDEX_INVALID$"):
        service.seed_bundled_sources(resources)


# 功能：
#   验证外部本地验收器必须返回字面 True，不能用非空字符串或数字放行。
# 输入：
#   tmp_path：测试独立目录。
#   verdict：容易被真假值判断误认为通过的返回值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("verdict", ["false", 1, {"accepted": False}])
def test_import_verifier_requires_literal_true(tmp_path, verdict):
    archive = tmp_path / "source.ddpkg"
    environment = {"gazebo": "fixture"}
    _write_package(archive, qualification_environment=environment)
    service = imports.AssetImportService(
        AppStore(tmp_path / "store"),
        environment_versions_provider=lambda: environment,
        local_qualification_verifier=lambda _asset, _environment: verdict,
    )
    created = service.create(source=archive, source_name=archive.name, source_format="ddpkg")
    waiting = service.process(created["job_id"])
    assert waiting["state"] == "needs_input"
    assert waiting["required_inputs"] == ["local_qualification_run"]
    assert service.store.list_asset_versions() == []


# 功能：
#   验证复制源文件前执行大小限制，超额源文件不会产生导入任务。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：可恢复的限制替换工具。
# 输出：
#   None：不返回业务数据。
def test_quarantine_checks_source_size_before_copy(tmp_path, monkeypatch):
    source = tmp_path / "source.blend"
    source.write_bytes(b"123456789")
    monkeypatch.setattr(imports, "MAX_PACKAGE_ARCHIVE_BYTES", 8, raising=False)
    service = imports.AssetImportService(AppStore(tmp_path / "store"))
    with pytest.raises(ValueError):
        service.create(source=source, source_name=source.name, source_format="auto")
    assert service.store.list_asset_import_jobs() == []


# 功能：
#   验证第二个资产安装失败时，不会删除第一处目录中并发写入者的内容。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：可恢复的安装入口替换工具。
# 输出：
#   None：不返回业务数据。
def test_pair_failure_does_not_delete_concurrent_directory(tmp_path, monkeypatch):
    source, _, _, receipt = _prepared_pair(tmp_path)
    map_archive = bind_qualification_receipt(
        source_archive=source,
        destination_archive=tmp_path / "map-qualified.ddpkg",
        receipt=receipt,
    )
    vehicle_archive = bind_qualification_receipt(
        source_archive=tmp_path / "vehicle.ddpkg",
        destination_archive=tmp_path / "vehicle-qualified.ddpkg",
        receipt=receipt,
    )
    service = imports.AssetImportService(AppStore(tmp_path / "store"))
    winner = service._version_destination(inspect_ddpkg(map_archive))
    marker = winner / "owned-by-other-writer.txt"

    # 功能：
    #   模拟另一流程先创建地图目录，而当前流程在安装飞机时失败。
    # 输入：
    #   archive：本次待安装的资产包。
    #   inspected：该资产包的检查结果。
    #   operation：本次操作标识。
    # 输出：
    #   winner：由另一流程创建的测试目录。
    def competing_promotion(archive, inspected, operation):
        if inspected.manifest.asset_kind == "vehicle":
            raise OSError("isolated second-asset failure")
        winner.mkdir(parents=True)
        marker.write_text("must survive", encoding="utf-8")
        return winner

    monkeypatch.setattr(service, "_promote", competing_promotion)
    with pytest.raises(OSError):
        service.promote_locally_verified_pair(
            map_source=map_archive,
            map_asset_id=receipt.map_asset_id,
            vehicle_source=vehicle_archive,
            vehicle_asset_id=receipt.vehicle_asset_id,
            environment_versions=receipt.environment_versions,
            operation_id="test-pair",
        )
    assert marker.read_text(encoding="utf-8") == "must survive"
    assert service.store.list_asset_versions() == []


# 功能：
#   验证重新导出时拒绝带重复字段的清单，不根据最后一个字段解释资产身份。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_export_rejects_duplicate_manifest_identity(tmp_path):
    service, _, inspected, root = _installed_asset(tmp_path)
    manifest = (root / "manifest.json").read_text(encoding="utf-8")
    duplicate = '{"asset_id":"different",' + manifest.lstrip()[1:]
    assert json.loads(duplicate)["asset_id"] == inspected.manifest.asset_id
    (root / "manifest.json").write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError):
        service.export_version(
            asset_id=inspected.manifest.asset_id,
            content_sha256=inspected.manifest.content_sha256,
            destination=tmp_path / "export.ddpkg",
        )
