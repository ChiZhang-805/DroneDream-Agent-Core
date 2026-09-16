"""Synthetic package and import-state tests; local verifier stubs are not flight evidence."""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from dronedream_agent_app.asset_import_service import AssetImportService
from dronedream_agent_app.storage import AppStore, AssetImportError
from dronedream_agent_core.asset_packages import (
    AssetFile,
    AssetImportJob,
    AssetIR,
    AssetPackageError,
    AssetSource,
    DDPkgManifest,
    InspectedDDPkg,
    QualificationBinding,
    RuntimeExtension,
    SimulationTarget,
    inspect_ddpkg,
    package_content_sha256,
    qualification_is_current,
    transition_import_job,
)


# 功能：
#   根据实际写入测试包的字节生成摘要，使夹具声明与文件内容一致。
# 输入：
#   payload：测试文件字节。
# 输出：
#   digest：该字节串的 SHA-256。
def _sha(payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    return digest


# 功能：
#   构造内外索引一致的最小测试包，可注入指定准入缺陷；不产生真实飞行证据。
# 输入：
#   path：测试包输出路径。
#   world：世界 SDF 字节。
#   extra_member：可选的未索引成员，用于测试拒绝行为。
#   runtime_extensions：可选的运行扩展声明。
#   qualification_environment：可选的模拟资格环境。
#   provenance_source_sha256：原始输入的声明摘要。
#   provenance_source_format：原始输入格式。
#   provenance_adapter_id：规范化适配器标识。
# 输出：
#   None：不返回业务数据。
def _write_package(
    path: Path,
    *,
    world: bytes = b'<sdf version="1.10"><world name="test"/></sdf>',
    extra_member: tuple[str, bytes] | None = None,
    runtime_extensions: list[RuntimeExtension] | None = None,
    qualification_environment: dict[str, str] | None = None,
    provenance_source_sha256: str | None = None,
    provenance_source_format: str = "blend",
    provenance_adapter_id: str = "blender.phobos",
) -> None:
    source = b"source-placeholder"
    semantic = b'{"coordinate_frame":"ENU"}'
    payloads = {
        "source/model.blend": source,
        "normalized/world.sdf": world,
        "normalized/semantic.json": semantic,
    }
    # 真实验收会新增索引内证据，因而产生新内容摘要；不能仅改清单中的资格标签。
    if qualification_environment is not None:
        payloads["evidence/test-qualification.json"] = json.dumps(
            {"fixture_only": True, "environment_versions": qualification_environment},
            sort_keys=True,
        ).encode()
    ir_files = [
        AssetFile(
            path=name,
            role="source"
            if name.startswith("source/")
            else "sdf"
            if name.endswith(".sdf")
            else "evidence"
            if name.startswith("evidence/")
            else "semantic",
            media_type="application/octet-stream",
            sha256=_sha(payload),
            size_bytes=len(payload),
        )
        for name, payload in payloads.items()
    ]
    asset_ir = AssetIR(
        asset_id="campus-map",
        name="Campus Map",
        version="1.0.0",
        kind="map",
        source=AssetSource(
            adapter_id=provenance_adapter_id,
            adapter_version="1.0.0",
            application="Blender",
            application_version="4.5",
            source_format=provenance_source_format,
            source_sha256=provenance_source_sha256 or _sha(source),
        ),
        coordinate_frame="map_enu",
        files=ir_files,
        simulation_targets=[
            SimulationTarget(
                target_id="gazebo-harmonic",
                simulator="gazebo-harmonic",
                simulator_version="8.9",
                ros_distribution="jazzy",
                entrypoint="normalized/world.sdf",
            )
        ],
        runtime_extensions=runtime_extensions or [],
        semantic_layers=["collision", "roads", "doors"],
        capabilities=["ros2", "gazebo"],
        license_expression="LicenseRef-Proprietary",
    )
    ir_payload = (json.dumps(asset_ir.model_dump(mode="json"), sort_keys=True) + "\n").encode()
    payloads["normalized/asset-ir.json"] = ir_payload
    package_files = [
        AssetFile(
            path=name,
            role="asset_ir"
            if name.endswith("asset-ir.json")
            else next(item.role for item in ir_files if item.path == name),
            media_type="application/json" if name.endswith(".json") else "application/octet-stream",
            sha256=_sha(payload),
            size_bytes=len(payload),
        )
        for name, payload in payloads.items()
    ]
    content_sha256 = package_content_sha256(package_files)
    qualification = (
        QualificationBinding(
            maturity="qualified",
            content_sha256=content_sha256,
            environment_versions=qualification_environment,
            evidence_paths=["evidence/test-qualification.json"],
            qualified_at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        if qualification_environment is not None
        else None
    )
    manifest = DDPkgManifest(
        package_id="campus-map.1.0.0",
        asset_id="campus-map",
        asset_kind="map",
        content_sha256=content_sha256,
        files=package_files,
        qualification=qualification,
        created_at=datetime(2026, 8, 20, tzinfo=UTC),
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "manifest.json",
            json.dumps(manifest.model_dump(mode="json"), sort_keys=True),
        )
        for name, payload in payloads.items():
            bundle.writestr(name, payload)
        if extra_member is not None:
            bundle.writestr(*extra_member)


# 功能：
#   验证检查器能读取声明身份和入口，不需要加载仿真库或执行包内插件。
# 输入：
#   tmp_path：测试包独立目录。
# 输出：
#   None：不返回业务数据。
def test_inspects_content_addressed_package_without_executing_plugins(tmp_path: Path) -> None:
    archive = tmp_path / "campus.ddpkg"
    _write_package(archive)

    inspected = inspect_ddpkg(archive)

    assert inspected.asset_ir.asset_id == "campus-map"
    assert inspected.manifest.asset_ir_path == "normalized/asset-ir.json"
    assert inspected.detected_runtime_extensions == []


# 功能：
#   验证未索引成员和父目录穿越在提取前被拒绝。
# 输入：
#   tmp_path：包含两个不同缺陷测试包的目录。
# 输出：
#   None：不返回业务数据。
def test_rejects_undeclared_members_and_path_traversal(tmp_path: Path) -> None:
    undeclared = tmp_path / "undeclared.ddpkg"
    _write_package(undeclared, extra_member=("unexpected.txt", b"not indexed"))
    with pytest.raises(AssetPackageError, match="DDPKG_FILE_INDEX_MISMATCH"):
        inspect_ddpkg(undeclared)

    traversal = tmp_path / "traversal.ddpkg"
    _write_package(traversal)
    with zipfile.ZipFile(traversal, "a") as bundle:
        bundle.writestr("../escape.txt", b"escape")
    with pytest.raises(AssetPackageError, match="DDPKG_MEMBER_PATH_INVALID"):
        inspect_ddpkg(traversal)


# 功能：
#   验证 ZIP 成员不能向导入目录引入符号链接或可执行脚本。
# 输入：
#   tmp_path：包含链接和脚本测试包的目录。
# 输出：
#   None：不返回业务数据。
def test_rejects_symlinks_and_executables_before_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.ddpkg"
    _write_package(archive)
    with zipfile.ZipFile(archive, "a") as bundle:
        link = zipfile.ZipInfo("normalized/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(link, "../../outside")
    with pytest.raises(AssetPackageError, match="DDPKG_SYMLINK_FORBIDDEN"):
        inspect_ddpkg(archive)

    executable = tmp_path / "executable.ddpkg"
    _write_package(executable, extra_member=("runtime/start.sh", b"#!/bin/sh"))
    with pytest.raises(AssetPackageError, match="DDPKG_EXECUTABLE_MEMBER_FORBIDDEN"):
        inspect_ddpkg(executable)


# 功能：
#   验证 SDF 扩展必须显式声明，声明存在也不能自动启用执行权限。
# 输入：
#   tmp_path：运行扩展声明的测试包目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_extensions_are_declared_and_disabled_by_default(tmp_path: Path) -> None:
    world = (
        b'<sdf version="1.10"><world name="test"><plugin name="wind" '
        b'filename="libWind.so"/></world></sdf>'
    )
    undeclared = tmp_path / "undeclared-plugin.ddpkg"
    _write_package(undeclared, world=world)
    with pytest.raises(AssetPackageError, match="DDPKG_RUNTIME_EXTENSION_UNDECLARED"):
        inspect_ddpkg(undeclared)

    declared = tmp_path / "declared-plugin.ddpkg"
    _write_package(
        declared,
        world=world,
        runtime_extensions=[
            RuntimeExtension(
                name="wind",
                filename="libWind.so",
                source_path="normalized/world.sdf",
            )
        ],
    )
    inspected = inspect_ddpkg(declared)
    assert inspected.detected_runtime_extensions[0].admission == "blocked"
    assert inspected.detected_runtime_extensions[0].enabled_by_default is False

    with pytest.raises(ValidationError):
        RuntimeExtension(
            name="unsafe",
            source_path="normalized/world.sdf",
            enabled_by_default=True,  # type: ignore[arg-type]
        )


# 功能：
#   验证导入任务可从等待继续推进，但进入合格终态后不可退回验证阶段。
# 输入：
#   无：使用本函数创建的独立状态机对象。
# 输出：
#   None：不返回业务数据。
def test_import_job_is_resumable_but_terminal_states_are_immutable() -> None:
    created = datetime(2026, 8, 20, tzinfo=UTC)
    job = AssetImportJob(
        job_id="asset-job-0123456789abcdef01234567",
        owner_id="owner-1",
        source_name="campus.blend",
        source_format="blend",
        progress_percent=0,
        revision=0,
        created_at=created,
        updated_at=created,
    )
    job = transition_import_job(job, "quarantining", progress_percent=5)
    job = AssetImportJob.model_validate(job.model_copy(update={"package_sha256": "a" * 64}))
    job = transition_import_job(job, "parsing", progress_percent=15)
    job = transition_import_job(
        job,
        "needs_input",
        progress_percent=25,
        required_inputs=["coordinate_frame"],
    )
    job = transition_import_job(
        job,
        "normalizing",
        progress_percent=35,
        asset_id="campus-map",
        asset_kind="map",
    )
    job = transition_import_job(job, "building", progress_percent=65)
    job = transition_import_job(job, "validating", progress_percent=90)
    job = transition_import_job(job, "qualified", progress_percent=100)

    assert job.revision == 7
    assert job.state == "qualified"
    with pytest.raises(AssetPackageError, match="ASSET_IMPORT_TRANSITION_INVALID"):
        transition_import_job(job, "validating", progress_percent=100)


# 功能：
#   验证资格绑定必须同时匹配内容和运行环境，任一变化都会使匹配失效。
# 输入：
#   无：使用本函数构造的模拟资格绑定。
# 输出：
#   None：不返回业务数据。
def test_qualification_is_invalidated_by_content_or_environment_drift() -> None:
    content_hash = "a" * 64
    environment = {"gazebo": "8.9", "px4": "1.16", "ros": "jazzy"}
    binding = QualificationBinding(
        maturity="qualified",
        content_sha256=content_hash,
        environment_versions=environment,
        evidence_paths=["evidence/px4-sitl.json"],
        qualified_at=datetime(2026, 8, 20, tzinfo=UTC),
    )

    assert qualification_is_current(
        binding,
        content_sha256=content_hash,
        environment_versions=environment,
    )
    assert not qualification_is_current(
        binding,
        content_sha256="b" * 64,
        environment_versions=environment,
    )
    assert not qualification_is_current(
        binding,
        content_sha256=content_hash,
        environment_versions={**environment, "gazebo": "9.0"},
    )


# 功能：
#   使用验证器替身检查隔离、状态推进、目录发布及暂存清理，不测试实际飞行。
# 输入：
#   tmp_path：源包和本地数据库的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_import_service_quarantines_resumes_and_atomically_promotes(tmp_path: Path) -> None:
    environment = {"gazebo": "8.9", "px4": "1.16", "ros": "jazzy"}
    archive = tmp_path / "campus.ddpkg"
    _write_package(archive, qualification_environment=environment)
    store = AppStore(tmp_path / "store")
    service = AssetImportService(
        store,
        environment_versions_provider=lambda: environment,
        local_qualification_verifier=lambda _asset, _environment: True,
    )

    created = service.create(
        source=archive,
        source_name="campus.ddpkg",
        source_format="ddpkg",
        expected_kind="map",
    )

    assert created["state"] == "parsing"
    assert created["progress_percent"] == 15
    assert store.get_asset_import_source(str(created["job_id"])).is_file()
    qualified = service.process(str(created["job_id"]))
    assert qualified["state"] == "qualified"
    assert qualified["progress_percent"] == 100
    versions = store.list_asset_versions("campus-map")
    assert len(versions) == 1
    bundle_root = Path(str(versions[0]["bundle_root"]))
    assert bundle_root.is_dir()
    assert (bundle_root / "normalized/world.sdf").is_file()
    assert not any(
        path.name.startswith(".staging-") for path in store.asset_versions_root.iterdir()
    )


# 功能：
#   验证普通检查只能准入视觉用途内容，缺少飞行资格时仍等待补充材料。
# 输入：
#   tmp_path：未验收测试包和数据库目录。
# 输出：
#   None：不返回业务数据。
def test_import_service_keeps_unqualified_content_in_needs_input(tmp_path: Path) -> None:
    archive = tmp_path / "draft.ddpkg"
    _write_package(archive)
    store = AppStore(tmp_path / "store")
    service = AssetImportService(store)
    created = service.create(
        source=archive,
        source_name="draft.ddpkg",
        source_format="ddpkg",
    )

    waiting = service.process(str(created["job_id"]))

    assert waiting["state"] == "needs_input"
    assert waiting["required_inputs"] == ["qualification_evidence"]
    cancelled = service.cancel(str(created["job_id"]))
    assert cancelled["state"] == "cancelled"
    versions = store.list_asset_versions()
    assert len(versions) == 1
    assert versions[0]["maturity"] == "visual_only"


# 功能：
#   验证伴随工具结果必须绑定原始输入，规范化成功后仍需另行取得飞行资格。
# 输入：
#   tmp_path：原始源文件、伴随结果和数据库目录。
# 输出：
#   None：不返回业务数据。
def test_companion_result_is_hash_bound_admitted_and_resumable(tmp_path: Path) -> None:
    source_payload = b"native blender project"
    source = tmp_path / "campus.blend"
    source.write_bytes(source_payload)
    result = tmp_path / "campus.ddpkg"
    _write_package(
        result,
        provenance_source_sha256=_sha(source_payload),
        provenance_source_format="blender-blend",
    )
    store = AppStore(tmp_path / "store")
    service = AssetImportService(store)
    created = service.create(
        source=source,
        source_name=source.name,
        source_format="auto",
        expected_kind="map",
    )
    waiting = service.process(str(created["job_id"]))
    assert waiting["required_inputs"] == ["plugin_adapter:blender.phobos"]

    admitted = service.submit_companion_result(
        str(created["job_id"]),
        result=result,
        source_package_sha256=_sha(source_payload),
        adapter_id="blender.phobos",
    )

    assert admitted["state"] == "needs_input"
    assert admitted["required_inputs"] == ["qualification_evidence"]
    assert admitted["normalized_content_sha256"]
    assert service._sha256(store.get_asset_import_source(str(created["job_id"]))) == _sha(
        source_payload
    )
    versions = store.list_asset_versions("campus-map")
    assert len(versions) == 1
    assert versions[0]["maturity"] == "visual_only"


# 功能：
#   验证错误来源、错误适配器和外部自授资格均不能作为当前任务的伴随结果。
# 输入：
#   tmp_path：构造不同来源错误的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_companion_result_rejects_wrong_binding_and_external_qualification(
    tmp_path: Path,
) -> None:
    source_payload = b"native blender project"
    source = tmp_path / "campus.blend"
    source.write_bytes(source_payload)
    result = tmp_path / "campus.ddpkg"
    _write_package(
        result,
        provenance_source_sha256=_sha(source_payload),
        provenance_source_format="blender-blend",
    )
    store = AppStore(tmp_path / "store")
    service = AssetImportService(store)
    created = service.create(
        source=source,
        source_name=source.name,
        source_format="auto",
        expected_kind="map",
    )
    service.process(str(created["job_id"]))

    with pytest.raises(AssetImportError, match="ASSET_COMPANION_SOURCE_HASH_MISMATCH"):
        service.submit_companion_result(
            str(created["job_id"]),
            result=result,
            source_package_sha256="f" * 64,
            adapter_id="blender.phobos",
        )
    with pytest.raises(AssetImportError, match="ASSET_COMPANION_ADAPTER_MISMATCH"):
        service.submit_companion_result(
            str(created["job_id"]),
            result=result,
            source_package_sha256=_sha(source_payload),
            adapter_id="ros2.xacro",
        )

    externally_qualified = tmp_path / "externally-qualified.ddpkg"
    _write_package(
        externally_qualified,
        provenance_source_sha256=_sha(source_payload),
        provenance_source_format="blender-blend",
        qualification_environment={"gazebo": "8.9"},
    )
    with pytest.raises(AssetImportError, match="ASSET_COMPANION_QUALIFICATION_FORBIDDEN"):
        service.submit_companion_result(
            str(created["job_id"]),
            result=externally_qualified,
            source_package_sha256=_sha(source_payload),
            adapter_id="blender.phobos",
        )
    assert not (store.asset_quarantine_root / str(created["job_id"]) / "normalized.ddpkg").exists()


# 功能：
#   验证本地结果按原始与验收后内容摘要关闭对应等待任务，不混淆两种版本身份。
# 输入：
#   tmp_path：模拟配对结果和等待导入任务的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_local_qualification_closes_matching_companion_import_job(tmp_path: Path) -> None:
    source_payload = b"native blender project"
    environment = {"gazebo": "8.9", "px4": "1.16", "ros": "jazzy"}
    source = tmp_path / "campus.blend"
    source.write_bytes(source_payload)
    normalized = tmp_path / "campus.ddpkg"
    _write_package(
        normalized,
        provenance_source_sha256=_sha(source_payload),
        provenance_source_format="blender-blend",
    )
    store = AppStore(tmp_path / "store")
    service = AssetImportService(store)
    created = service.create(
        source=source,
        source_name=source.name,
        source_format="auto",
        expected_kind="map",
    )
    service.process(str(created["job_id"]))
    waiting = service.submit_companion_result(
        str(created["job_id"]),
        result=normalized,
        source_package_sha256=_sha(source_payload),
        adapter_id="blender.phobos",
    )
    qualified = tmp_path / "campus-qualified.ddpkg"
    _write_package(
        qualified,
        provenance_source_sha256=_sha(source_payload),
        provenance_source_format="blender-blend",
        qualification_environment=environment,
    )
    result = service.promote_locally_verified_package(
        qualified,
        expected_asset_id="campus-map",
        environment_versions=environment,
        operation_id="local-qualification",
    )

    completed = service.finalize_qualified_admissions(
        asset_id="campus-map",
        source_content_sha256=str(waiting["normalized_content_sha256"]),
        result_content_sha256=str(result["content_sha256"]),
    )

    assert len(completed) == 1
    assert completed[0]["state"] == "qualified"
    assert completed[0]["qualified_content_sha256"] == result["content_sha256"]


# 功能：
#   验证外部环境声明即使完全匹配，也不能替代本地资格核查器。
# 输入：
#   tmp_path：带外部资格标签的测试包与数据库目录。
# 输出：
#   None：不返回业务数据。
def test_import_service_never_trusts_external_qualification_without_local_run(
    tmp_path: Path,
) -> None:
    environment = {"gazebo": "8.9", "px4": "1.16", "ros": "jazzy"}
    archive = tmp_path / "external-qualified.ddpkg"
    _write_package(archive, qualification_environment=environment)
    store = AppStore(tmp_path / "store")
    service = AssetImportService(
        store,
        environment_versions_provider=lambda: environment,
    )
    created = service.create(
        source=archive,
        source_name="external-qualified.ddpkg",
        source_format="ddpkg",
    )

    waiting = service.process(str(created["job_id"]))

    assert waiting["state"] == "needs_input"
    assert waiting["required_inputs"] == ["local_qualification_run"]
    assert store.list_asset_versions() == []


# 功能：
#   验证持久化验证阶段恢复后只产生一个内容版本，不重复安装或降低进度。
# 输入：
#   tmp_path：需要模拟中断恢复的源包与数据库目录。
# 输出：
#   None：不返回业务数据。
def test_import_service_resumes_from_persisted_validating_state(tmp_path: Path) -> None:
    environment = {"gazebo": "8.9", "px4": "1.16", "ros": "jazzy"}
    archive = tmp_path / "resume.ddpkg"
    _write_package(archive, qualification_environment=environment)
    store = AppStore(tmp_path / "store")
    service = AssetImportService(
        store,
        environment_versions_provider=lambda: environment,
        local_qualification_verifier=lambda _asset, _environment: True,
    )
    created = service.create(
        source=archive,
        source_name="resume.ddpkg",
        source_format="ddpkg",
        expected_kind="map",
    )
    job = store.load_asset_import_job(str(created["job_id"]))
    job = transition_import_job(
        job,
        "normalizing",
        progress_percent=40,
        asset_id="campus-map",
        asset_kind="map",
    )
    job = transition_import_job(job, "building", progress_percent=65)
    job = transition_import_job(job, "validating", progress_percent=90)
    store.save_asset_import_job(job)

    qualified = service.process(job.job_id)

    assert qualified["state"] == "qualified"
    assert qualified["progress_percent"] == 100
    assert len(store.list_asset_versions("campus-map")) == 1


# 功能：
#   验证本地核查器异常只保存稳定错误码，不暴露原始异常中的内部信息。
# 输入：
#   tmp_path：模拟核查器故障的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_import_service_sanitizes_local_verifier_failures(tmp_path: Path) -> None:
    environment = {"gazebo": "8.9", "px4": "1.16", "ros": "jazzy"}
    archive = tmp_path / "verifier-error.ddpkg"
    _write_package(archive, qualification_environment=environment)
    store = AppStore(tmp_path / "store")

    # 功能：
    #   抛出含虚构内部信息的异常，检查错误清洗，不读取真实敏感数据。
    # 输入：
    #   _asset：当前已检查包；本替身不使用。
    #   _environment：当前环境；本替身不使用。
    # 输出：
    #   None：不返回业务数据。
    def fail_verification(_asset: InspectedDDPkg, _environment: dict[str, str]) -> bool:
        raise RuntimeError("secret internal path and diagnostic")

    service = AssetImportService(
        store,
        environment_versions_provider=lambda: environment,
        local_qualification_verifier=fail_verification,
    )
    created = service.create(
        source=archive,
        source_name="verifier-error.ddpkg",
        source_format="ddpkg",
    )

    with pytest.raises(AssetImportError, match="^LOCAL_QUALIFICATION_VERIFIER_FAILED$"):
        service.process(str(created["job_id"]))

    failed = store.get_asset_import_job(str(created["job_id"]))
    assert failed["state"] == "failed"
    assert failed["issue_codes"] == ["LOCAL_QUALIFICATION_VERIFIER_FAILED"]


# 功能：
#   验证复用已有版本前重新核对文件内容，不把摘要格式的目录名当成安全证明。
# 输入：
#   tmp_path：含首次导入与后续篡改文件的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_import_service_rejects_tampered_existing_content_version(tmp_path: Path) -> None:
    environment = {"gazebo": "8.9", "px4": "1.16", "ros": "jazzy"}
    archive = tmp_path / "tamper.ddpkg"
    _write_package(archive, qualification_environment=environment)
    store = AppStore(tmp_path / "store")
    service = AssetImportService(
        store,
        environment_versions_provider=lambda: environment,
        local_qualification_verifier=lambda _asset, _environment: True,
    )
    first = service.create(
        source=archive,
        source_name="tamper.ddpkg",
        source_format="ddpkg",
    )
    service.process(str(first["job_id"]))
    version = store.list_asset_versions("campus-map")[0]
    (Path(str(version["bundle_root"])) / "normalized/world.sdf").write_text(
        "tampered",
        encoding="utf-8",
    )
    second = service.create(
        source=archive,
        source_name="tamper-again.ddpkg",
        source_format="ddpkg",
    )

    with pytest.raises(AssetImportError, match="^ASSET_VERSION_HASH_MISMATCH$"):
        service.process(str(second["job_id"]))

    failed = store.get_asset_import_job(str(second["job_id"]))
    assert failed["state"] == "failed"
    assert failed["issue_codes"] == ["ASSET_VERSION_HASH_MISMATCH"]
