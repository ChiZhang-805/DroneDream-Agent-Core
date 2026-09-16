from __future__ import annotations

import hashlib
import json
import threading
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dronedream_agent_app.asset_import_service import AssetImportService
from dronedream_agent_app.asset_qualification_service import AssetQualificationService
from dronedream_agent_app.asset_runtime_resolver import (
    qualification_matches_environment,
    resolve_versioned_map,
    resolve_versioned_vehicle,
)
from dronedream_agent_app.storage import AppStore, AssetImportError
from dronedream_agent_core import asset_qualification_cli
from dronedream_agent_core.asset_packages import (
    AssetFile,
    AssetIR,
    AssetSource,
    DDPkgManifest,
    PhysicsSummary,
    SensorSummary,
    SimulationTarget,
    VehicleSummary,
    inspect_ddpkg,
    package_content_sha256,
    qualification_is_current,
)
from dronedream_agent_core.asset_pair_qualification import (
    AssetPairQualificationError,
    bind_qualification_receipt,
    build_pair_qualification_receipt,
    prepare_asset_pair_qualification,
)
from dronedream_agent_core.asset_source_adapters import detect_asset_source


# 功能：
#   为隔离测试材料计算 SHA-256，用于验证绑定的内容是否发生变化。
# 输入：
#   payload：测试材料字节。
# 输出：
#   digest：十六进制摘要。
def _sha(payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    return digest


# 功能：
#   将可信测试夹具按固定排序写成 JSON 字节，不作为产品的不可信输入解析器。
# 输入：
#   value：当前测试构造的 JSON 数据。
# 输出：
#   payload：带末尾换行的 UTF-8 字节。
def _json(value: object) -> bytes:
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    return payload


# 功能：
#   1. 构造索引与摘要一致的隔离 DDPKG，供静态准入和服务测试使用。
#   2. 按参数声明测试物理字段；这些声明不代表真实仿真或真实飞行结果。
# 输入：
#   path：测试包输出路径。
#   kind：地图或飞机类别。
#   files：相对路径到角色及字节内容的映射。
#   entrypoint：测试 Gazebo 入口。
#   capabilities：测试所需的能力声明。
#   physics_complete：是否声明完整物理信息。
# 输出：
#   None：不返回业务数据。
def _write_package(
    path: Path,
    *,
    kind: str,
    files: dict[str, tuple[str, bytes]],
    entrypoint: str,
    capabilities: list[str],
    physics_complete: bool = True,
) -> None:
    asset_id = f"qualification-{kind}"
    source_payload = b"source"
    ir_files = [
        AssetFile(
            path=name,
            role=role,  # type: ignore[arg-type]
            media_type="application/json" if name.endswith(".json") else "application/xml",
            sha256=_sha(payload),
            size_bytes=len(payload),
        )
        for name, (role, payload) in files.items()
    ]
    ir = AssetIR(
        asset_id=asset_id,
        name=asset_id,
        version="1.0.0",
        kind=kind,  # type: ignore[arg-type]
        source=AssetSource(
            adapter_id="tests.fixture",
            adapter_version="1",
            application="pytest",
            source_format="fixture",
            source_sha256=_sha(source_payload),
        ),
        coordinate_frame="map_enu" if kind == "map" else "base_link",
        files=ir_files,
        simulation_targets=[
            SimulationTarget(
                target_id="gazebo-harmonic",
                simulator="gazebo-harmonic",
                simulator_version="8.14.0",
                ros_distribution="jazzy",
                autopilot="px4" if kind == "vehicle" else "none",
                entrypoint=entrypoint,
            )
        ],
        physics=(
            PhysicsSummary(
                link_count=1,
                mass_entry_count=1,
                inertia_entry_count=1,
                collision_complete=physics_complete,
                mass_complete=physics_complete,
                inertia_complete=physics_complete,
            )
            if kind == "vehicle"
            else PhysicsSummary(collision_complete=True)
        ),
        vehicle=(
            VehicleSummary(
                vehicle_class="multirotor",
                actuator_count=4,
                rotor_count=4,
                autopilot_profile="x500",
                payload_limit_kg=0.2,
            )
            if kind == "vehicle"
            else None
        ),
        sensors=(
            [
                SensorSummary(kind="imu", source_path="normalized/vehicle.json"),
                SensorSummary(kind="odometry", source_path="normalized/vehicle.json"),
                SensorSummary(kind="depth-camera", source_path="normalized/vehicle.json"),
            ]
            if kind == "vehicle"
            else []
        ),
        capabilities=capabilities,
        license_expression="LicenseRef-Test",
    )
    ir_payload = _json(ir.model_dump(mode="json"))
    ir_file = AssetFile(
        path="normalized/asset-ir.json",
        role="asset_ir",
        media_type="application/json",
        sha256=_sha(ir_payload),
        size_bytes=len(ir_payload),
    )
    package_files = [*ir_files, ir_file]
    manifest = DDPkgManifest(
        package_id=f"{asset_id}.1.0.0",
        asset_id=asset_id,
        asset_kind=kind,  # type: ignore[arg-type]
        content_sha256=package_content_sha256(package_files),
        files=package_files,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", _json(manifest.model_dump(mode="json")))
        for name, (_, payload) in files.items():
            bundle.writestr(name, payload)
        bundle.writestr("normalized/asset-ir.json", ir_payload)


# 功能：
#   构造最小地图和飞机包，通过指定开关分别制造缺失能力或越界参数。
# 输入：
#   tmp_path：隔离输出目录。
#   px4_capability：是否声明 PX4 机型绑定。
#   valid_vehicle_contract：是否提供飞机航程字段。
#   current_normalized_contract：是否使用当前规范化声明。
#   complete_vehicle_physics：是否声明完整质量、惯性及碰撞字段。
#   controller_over_limit：是否故意设置超过飞机能力的速度上限。
#   complete_sensor_suite：是否声明状态与视觉感知传感器。
# 输出：
#   archives：地图包和飞机包路径组成的元组。
def _fixtures(
    tmp_path: Path,
    *,
    px4_capability: bool = True,
    valid_vehicle_contract: bool = True,
    current_normalized_contract: bool = True,
    complete_vehicle_physics: bool = True,
    controller_over_limit: bool = False,
    complete_sensor_suite: bool = True,
) -> tuple[Path, Path]:
    graph = {
        "schema_version": "dronedream.map-graph.v1",
        "asset_id": "qualification-map",
        "name": "Qualification Map",
        "coordinate_frame": "map_enu",
        "nodes": [
            {
                "node_id": "launch",
                "label": "Launch",
                "position_m": {"x": 0.0, "y": 0.0, "z": 2.0},
                "semantic": "launch",
            },
            {
                "node_id": "turn",
                "label": "Turn",
                "position_m": {"x": 2.0, "y": 0.0, "z": 2.0},
                "semantic": "outdoor",
            },
        ],
        "edges": [
            {
                "edge_id": "qualification-leg",
                "from_node": "launch",
                "to_node": "turn",
                "distance_m": 2.0,
                "minimum_clearance_m": 4.0,
                "speed_limit_mps": 0.8,
                "bidirectional": True,
                "qualification": "geometry-derived",
            }
        ],
        "named_entities": {"launch": "launch"},
    }
    semantic = {
        "schema_version": "dronedream.map-semantic.v1",
        "coordinate_frame": "ENU",
        "scene_id": "qualification-map",
        "entities": [
            {
                "entity_id": "launch",
                "aliases": ["launch pad"],
                "position_m": {"x": 0.0, "y": 0.0, "z": 2.0},
                "semantic": "launch",
            }
        ],
        "runtime_bindings": {
            "schema_version": "dronedream.map-runtime-bindings.v1",
            "simulator": "gazebo-harmonic",
            "coordinate_frame": "ENU",
            "vehicle_spawn": {"x": 0.0, "y": 0.0, "z": 0.0},
            "mission_launch_waypoint": {"x": 0.0, "y": 0.0, "z": 2.0},
        },
        "collision_primitives": [
            {
                "name": "distant-wall",
                "center_x": 100.0,
                "center_y": 100.0,
                "center_z": 2.0,
                "size_x": 1.0,
                "size_y": 1.0,
                "size_z": 4.0,
            }
        ],
    }
    vehicle = {
        "schema_version": "dronedream.vehicle.v1",
        "asset_id": "qualification-vehicle",
        "name": "Qualification Vehicle",
        "coordinate_frame": "base_link_frd",
        "dry_mass_kg": 1.5,
        "max_takeoff_mass_kg": 2.0,
        "body_radius_m": 0.25,
        "body_height_m": 0.2,
        "collision_center_offset_model_m": {"x": 0.0, "y": 0.0, "z": 0.1},
        "max_speed_mps": 1.5,
        "max_acceleration_mps2": 1.0,
        "reserve_battery_percent": 30.0,
        "max_pickup_payload_kg": 0.2,
        "sensors": (["imu", "odometry", "depth-camera"] if complete_sensor_suite else ["camera"]),
    }
    if valid_vehicle_contract:
        vehicle["qualified_range_m"] = 100.0
    controller = {
        "vel_limit": 10.0 if controller_over_limit else 1.0,
        "accel_limit": 0.8,
    }
    map_archive = tmp_path / "map.ddpkg"
    vehicle_archive = tmp_path / "vehicle.ddpkg"
    _write_package(
        map_archive,
        kind="map",
        files={
            "normalized/world.sdf": (
                "sdf",
                b'<sdf version="1.10"><world name="qualification_world"/></sdf>',
            ),
            "normalized/semantic.json": ("semantic", _json(semantic)),
            "normalized/navigation-graph.json": ("metadata", _json(graph)),
        },
        entrypoint="normalized/world.sdf",
        capabilities=[
            "gazebo",
            *([] if current_normalized_contract else ["legacy-migrated"]),
        ],
    )
    _write_package(
        vehicle_archive,
        kind="vehicle",
        files={
            "normalized/model.sdf": (
                "sdf",
                b'<sdf version="1.10"><model name="qualification_vehicle"/></sdf>',
            ),
            "normalized/vehicle.json": ("metadata", _json(vehicle)),
            "normalized/controller.json": ("controller", _json(controller)),
        },
        entrypoint="normalized/model.sdf",
        capabilities=["gazebo", *(["px4.sitl-model=x500"] if px4_capability else [])],
        physics_complete=complete_vehicle_physics,
    )
    archives = map_archive, vehicle_archive
    return archives


# 功能：
#   1. 生成仅供单元测试使用的模拟运行结果，绑定准备目录中的实际文件摘要。
#   2. 门禁、采样数量及 ULog 内容均为测试构造，不能用作真实飞行验收证据。
# 输入：
#   plan：测试生成的资格计划。
#   work_root：测试准备材料目录。
# 输出：
#   evidence：模拟运行结果字典。
def _runtime_evidence(plan: object, work_root: Path) -> dict[str, object]:
    inputs = plan.inputs  # type: ignore[attr-defined]
    gates = {gate: True for gate in plan.required_runtime_gates}  # type: ignore[attr-defined]
    evidence = {
        "schema_version": "dronedream.generic-px4-gazebo-run.v1",
        "status": "verified",
        "world": inputs.world_name,
        "vehicle": inputs.vehicle_name,
        "gates": gates,
        "measurements": {"pose_sample_count": 20},
        "artifacts": {
            "world_sha256": _sha((work_root / inputs.world_sdf).read_bytes()),
            "semantic_sha256": _sha((work_root / inputs.semantic).read_bytes()),
            "vehicle_sha256": _sha((work_root / inputs.vehicle_sdf).read_bytes()),
            "route_sha256": plan.route_sha256,  # type: ignore[attr-defined]
            "track_sha256": plan.track_sha256,  # type: ignore[attr-defined]
            "clearance_sha256": plan.clearance_sha256,  # type: ignore[attr-defined]
            "controller_params_sha256": _sha((work_root / inputs.controller_params).read_bytes()),
            "executor_sha256": "a" * 64,
            "px4_ulogs": [{"path": "flight.ulg", "size_bytes": 1, "sha256": "b" * 64}],
            "ros_workspace": "/opt/dronedream/ros_ws",
        },
    }
    return evidence


# 功能：
#   验证准备阶段保存往返路线及明确的世界、飞机和 PX4 机型，不启动模拟器。
# 输入：
#   tmp_path：隔离资产目录。
# 输出：
#   None：不返回业务数据。
def test_prepares_hash_bound_round_trip_for_real_runtime(tmp_path: Path) -> None:
    map_archive, vehicle_archive = _fixtures(tmp_path)
    work_root = tmp_path / "run"

    plan = prepare_asset_pair_qualification(
        map_archive=map_archive,
        vehicle_archive=vehicle_archive,
        work_root=work_root,
    )

    assert plan.route.node_ids == ["launch", "turn", "launch"]
    assert plan.route.route_length_m == 4.0
    assert plan.inputs.world_name == "qualification_world"
    assert plan.inputs.vehicle_name == "qualification_vehicle"
    assert plan.inputs.px4_sitl_model == "x500"
    assert (work_root / "qualification-plan.json").is_file()


# 功能：
#   验证缺少机型绑定时停止准备，不能悄悄使用默认飞机。
# 输入：
#   tmp_path：隔离资产目录。
# 输出：
#   None：不返回业务数据。
def test_fails_closed_when_px4_binding_is_not_declared(tmp_path: Path) -> None:
    map_archive, vehicle_archive = _fixtures(tmp_path, px4_capability=False)

    with pytest.raises(
        AssetPairQualificationError,
        match="ASSET_QUALIFICATION_PX4_SITL_MODEL_REQUIRED",
    ):
        prepare_asset_pair_qualification(
            map_archive=map_archive,
            vehicle_archive=vehicle_archive,
            work_root=tmp_path / "run",
        )


# 功能：
#   验证缺少航程的飞机契约返回稳定错误代码，便于调用方正确提示。
# 输入：
#   tmp_path：隔离资产目录。
# 输出：
#   None：不返回业务数据。
def test_invalid_vehicle_contract_has_stable_issue_code(tmp_path: Path) -> None:
    map_archive, vehicle_archive = _fixtures(tmp_path, valid_vehicle_contract=False)

    with pytest.raises(
        AssetPairQualificationError,
        match="^ASSET_QUALIFICATION_VEHICLE_CONTRACT_INVALID$",
    ):
        prepare_asset_pair_qualification(
            map_archive=map_archive,
            vehicle_archive=vehicle_archive,
            work_root=tmp_path / "run",
        )


# 功能：
#   验证只迁移旧格式、未完成当前规范化的包不能进入运行验收。
# 输入：
#   tmp_path：隔离资产目录。
# 输出：
#   None：不返回业务数据。
def test_qualification_rejects_migration_only_package(tmp_path: Path) -> None:
    map_archive, vehicle_archive = _fixtures(
        tmp_path,
        current_normalized_contract=False,
    )

    with pytest.raises(
        AssetPairQualificationError,
        match="^ASSET_QUALIFICATION_CURRENT_NORMALIZED_CONTRACT_REQUIRED$",
    ):
        prepare_asset_pair_qualification(
            map_archive=map_archive,
            vehicle_archive=vehicle_archive,
            work_root=tmp_path / "run",
        )


# 功能：
#   分别验证物理字段缺失、状态传感器缺失及控制参数越界都会阻止准入。
# 输入：
#   tmp_path：隔离资产目录。
#   fixture_overrides：需要制造的单个静态缺陷。
#   issue_code：预期报告的错误代码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    ("fixture_overrides", "issue_code"),
    [
        (
            {"complete_vehicle_physics": False},
            "ASSET_QUALIFICATION_VEHICLE_PHYSICS_INCOMPLETE",
        ),
        (
            {"complete_sensor_suite": False},
            "ASSET_QUALIFICATION_VEHICLE_STATE_SENSORS_INCOMPLETE",
        ),
        (
            {"controller_over_limit": True},
            "ASSET_QUALIFICATION_CONTROLLER_EXCEEDS_VEHICLE_ENVELOPE",
        ),
    ],
)
def test_qualification_rejects_incomplete_vehicle_static_contract(
    tmp_path: Path,
    fixture_overrides: dict[str, bool],
    issue_code: str,
) -> None:
    map_archive, vehicle_archive = _fixtures(tmp_path, **fixture_overrides)

    with pytest.raises(AssetPairQualificationError, match=f"^{issue_code}$"):
        prepare_asset_pair_qualification(
            map_archive=map_archive,
            vehicle_archive=vehicle_archive,
            work_root=tmp_path / "run",
        )


# 功能：
#   验证命令行入口将当前资产的标识和机型传给执行器，不使用历史默认值。
# 输入：
#   tmp_path：隔离资产目录。
#   monkeypatch：可恢复的执行器替换工具。
# 输出：
#   None：不返回业务数据。
def test_runtime_cli_binds_paths_hashes_and_declared_px4_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    map_archive, vehicle_archive = _fixtures(tmp_path)
    work_root = tmp_path / "prepared"
    plan = prepare_asset_pair_qualification(
        map_archive=map_archive,
        vehicle_archive=vehicle_archive,
        work_root=work_root,
    )
    captured: dict[str, object] = {}

    # 功能：
    #   捕获执行器调用参数，并返回最小模拟结果，不启动 PX4 或 Gazebo。
    # 输入：
    #   kwargs：命令行入口交给执行器的关键字参数。
    # 输出：
    #   evidence：模拟执行结果。
    def fake_runner(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        evidence = {"status": "verified", "gates": {"executor_completed": True}}
        return evidence

    monkeypatch.setattr(asset_qualification_cli, "run_px4_gazebo_track", fake_runner)
    evidence = asset_qualification_cli.run_prepared_asset_qualification(
        work_root=work_root,
        run_dir=tmp_path / "runtime-evidence",
        px4_root=tmp_path / "PX4-Autopilot",
        executor=tmp_path / "executor.py",
        ros_workspace=tmp_path / "ros_ws",
    )

    assert evidence["status"] == "verified"
    assert captured["contract_id"] == plan.qualification_id
    assert captured["px4_sitl_model"] == "x500"
    assert captured["world_name"] == "qualification_world"
    assert captured["vehicle_name"] == "qualification_vehicle"


# 功能：
#   验证轨迹摘要不一致时立即拒绝启动，避免执行被替换的准备材料。
# 输入：
#   tmp_path：隔离资产目录。
#   monkeypatch：可恢复的执行器替换工具。
# 输出：
#   None：不返回业务数据。
def test_runtime_cli_rejects_tampered_track_before_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    map_archive, vehicle_archive = _fixtures(tmp_path)
    work_root = tmp_path / "prepared"
    prepare_asset_pair_qualification(
        map_archive=map_archive,
        vehicle_archive=vehicle_archive,
        work_root=work_root,
    )
    (work_root / "track.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        asset_qualification_cli,
        "run_px4_gazebo_track",
        lambda **_kwargs: pytest.fail("simulator must not start after hash mismatch"),
    )

    with pytest.raises(
        asset_qualification_cli.AssetQualificationCliError,
        match="^ASSET_QUALIFICATION_PLAN_ARTIFACT_MISMATCH:track.json$",
    ):
        asset_qualification_cli.run_prepared_asset_qualification(
            work_root=work_root,
            run_dir=tmp_path / "runtime-evidence",
            px4_root=tmp_path / "PX4-Autopilot",
            executor=tmp_path / "executor.py",
            ros_workspace=tmp_path / "ros_ws",
        )


# 功能：
#   验证模拟回执生成新的内容版本，并使导入、环境匹配和运行路径解析相互一致。
# 输入：
#   tmp_path：隔离资产及数据库目录。
# 输出：
#   None：不返回业务数据。
def test_binds_verified_pair_receipt_into_new_content_versions(tmp_path: Path) -> None:
    map_archive, vehicle_archive = _fixtures(tmp_path)
    work_root = tmp_path / "run"
    plan = prepare_asset_pair_qualification(
        map_archive=map_archive,
        vehicle_archive=vehicle_archive,
        work_root=work_root,
    )
    environment = {"ros": "jazzy", "gazebo": "8.14.0", "px4_commit": "abc123"}
    receipt = build_pair_qualification_receipt(
        plan=plan,
        work_root=work_root,
        runtime_evidence=_runtime_evidence(plan, work_root),
        environment_versions=environment,
    )

    qualified_map = bind_qualification_receipt(
        source_archive=map_archive,
        destination_archive=tmp_path / "map-qualified.ddpkg",
        receipt=receipt,
    )
    qualified_vehicle = bind_qualification_receipt(
        source_archive=vehicle_archive,
        destination_archive=tmp_path / "vehicle-qualified.ddpkg",
        receipt=receipt,
    )

    for archive in (qualified_map, qualified_vehicle):
        inspected = inspect_ddpkg(archive)
        assert qualification_is_current(
            inspected.manifest.qualification,
            content_sha256=inspected.manifest.content_sha256,
            environment_versions=environment,
        )
        assert inspected.manifest.qualification is not None
        assert inspected.manifest.qualification.maturity == "qualified"
        assert inspected.manifest.qualification.evidence_paths == [
            f"evidence/qualification/{plan.qualification_id}.json"
        ]

    store = AppStore(tmp_path / "store")
    imported_map, imported_vehicle = AssetImportService(store).promote_locally_verified_pair(
        map_source=qualified_map,
        map_asset_id=plan.inputs.map_asset_id,
        vehicle_source=qualified_vehicle,
        vehicle_asset_id=plan.inputs.vehicle_asset_id,
        environment_versions=environment,
        operation_id="resolver-test",
    )
    resolved_map = resolve_versioned_map(imported_map)
    resolved_vehicle = resolve_versioned_vehicle(imported_vehicle)
    assert resolved_map.content_sha256 == imported_map["content_sha256"]
    assert resolved_vehicle.content_sha256 == imported_vehicle["content_sha256"]
    assert resolved_map.world_sdf.is_file()
    assert resolved_map.semantic.is_file()
    assert resolved_map.graph.is_file()
    assert resolved_vehicle.vehicle_sdf.is_file()
    assert resolved_vehicle.vehicle_metadata.is_file()
    assert resolved_vehicle.controller_params.is_file()
    assert qualification_matches_environment(imported_map, environment)
    assert not qualification_matches_environment(imported_map, {"gazebo": "different"})


# 功能：
#   验证落地门禁未通过时，即使其他门禁通过，也不会生成验收回执。
# 输入：
#   tmp_path：隔离资产目录。
# 输出：
#   None：不返回业务数据。
def test_rejects_incomplete_runtime_evidence(tmp_path: Path) -> None:
    map_archive, vehicle_archive = _fixtures(tmp_path)
    work_root = tmp_path / "run"
    plan = prepare_asset_pair_qualification(
        map_archive=map_archive,
        vehicle_archive=vehicle_archive,
        work_root=work_root,
    )
    evidence = _runtime_evidence(plan, work_root)
    evidence["gates"]["landing_confirmed"] = False  # type: ignore[index]

    with pytest.raises(AssetPairQualificationError, match="landing_confirmed"):
        build_pair_qualification_receipt(
            plan=plan,
            work_root=work_root,
            runtime_evidence=evidence,
            environment_versions={"gazebo": "8.14.0"},
        )


# 功能：
#   验证回执不能使用另一套控制器参数对应的运行结果。
# 输入：
#   tmp_path：隔离资产目录。
# 输出：
#   None：不返回业务数据。
def test_rejects_runtime_evidence_for_different_controller_params(tmp_path: Path) -> None:
    map_archive, vehicle_archive = _fixtures(tmp_path)
    work_root = tmp_path / "run"
    plan = prepare_asset_pair_qualification(
        map_archive=map_archive,
        vehicle_archive=vehicle_archive,
        work_root=work_root,
    )
    evidence = _runtime_evidence(plan, work_root)
    evidence["artifacts"]["controller_params_sha256"] = "0" * 64  # type: ignore[index]

    with pytest.raises(
        AssetPairQualificationError,
        match="controller_params_sha256",
    ):
        build_pair_qualification_receipt(
            plan=plan,
            work_root=work_root,
            runtime_evidence=evidence,
            environment_versions={"gazebo": "8.14.0"},
        )


# 功能：
#   验证安装资源中的默认地图和飞机使用当前 DDPKG 格式。
# 输入：
#   无输入参数。
# 输出：
#   None：不返回业务数据。
def test_bundled_assets_expose_only_current_ddpkg_sources() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "desktop"
        / "src-tauri"
        / "resources"
        / "default-assets"
    )

    assert detect_asset_source(root / "school-map.ddpkg").source_format == "ddpkg"
    assert detect_asset_source(root / "my-drone.ddpkg").source_format == "ddpkg"


# 功能：
#   将本文件创建且已校验的测试包安装到隔离数据库，不作为产品导入路径。
# 输入：
#   store：当前测试的独立存储。
#   archive：可信测试夹具包。
# 输出：
#   identity：资产标识与内容摘要组成的元组。
def _install_fixture_version(store: AppStore, archive: Path) -> tuple[str, str]:
    inspected = inspect_ddpkg(archive)
    destination = (
        store.asset_versions_root
        / inspected.manifest.asset_kind
        / inspected.manifest.asset_id
        / inspected.manifest.content_sha256
    )
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source:
        source.extractall(destination)
    store.record_asset_version(inspected, bundle_root=destination)
    identity = inspected.manifest.asset_id, inspected.manifest.content_sha256
    return identity


# 功能：
#   在十秒内等待测试任务到达目标状态，超时则使测试失败，避免无限挂起。
# 输入：
#   store：测试存储。
#   job_id：等待的任务标识。
#   states：任一可接受的目标状态。
# 输出：
#   value：到达目标状态时读取的任务记录。
def _wait_for_job(store: AppStore, job_id: str, *states: str) -> dict[str, object]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        value = store.get_asset_pair_qualification_job(job_id)
        if value["state"] in states:
            return value
        time.sleep(0.02)
    pytest.fail(f"asset qualification job did not reach {states}")


# 功能：
#   验证后台服务消费模拟运行结果后，同时记录地图和飞机的新合格版本。
# 输入：
#   tmp_path：隔离资产及数据库目录。
# 输出：
#   None：不返回业务数据。
def test_durable_service_promotes_both_qualified_versions_together(tmp_path: Path) -> None:
    map_archive, vehicle_archive = _fixtures(tmp_path)
    store = AppStore(tmp_path / "store")
    map_id, map_hash = _install_fixture_version(store, map_archive)
    vehicle_id, vehicle_hash = _install_fixture_version(store, vehicle_archive)
    environment = {"ros": "jazzy", "gazebo": "8.14.0", "px4_commit": "abc123"}

    # 功能：
    #   读取服务本次写入的计划，返回绑定当前材料的模拟结果。
    # 输入：
    #   kwargs：服务传入的运行参数。
    # 输出：
    #   evidence：模拟运行结果。
    def runtime_runner(**kwargs: object) -> dict[str, object]:
        work_root = Path(str(kwargs["work_root"]))
        plan = asset_qualification_cli.AssetPairQualificationPlan.model_validate_json(
            (work_root / "qualification-plan.json").read_bytes()
        )
        evidence = _runtime_evidence(plan, work_root)
        return evidence

    service = AssetQualificationService(
        store,
        AssetImportService(store),
        runtime_runner=runtime_runner,
        runtime_canceller=lambda _job_id: False,
        environment_versions_provider=lambda: environment,
    )
    created = service.create(
        map_asset_id=map_id,
        map_content_sha256=map_hash,
        vehicle_asset_id=vehicle_id,
        vehicle_content_sha256=vehicle_hash,
    )
    service.start(str(created["job_id"]))
    qualified = _wait_for_job(store, str(created["job_id"]), "qualified", "failed")

    assert qualified["state"] == "qualified", qualified
    assert qualified["result_map_content_sha256"] != map_hash
    assert qualified["result_vehicle_content_sha256"] != vehicle_hash
    resulting = {str(row["content_sha256"]): row for row in store.list_asset_versions()}
    assert resulting[str(qualified["result_map_content_sha256"])]["maturity"] == "qualified"
    assert resulting[str(qualified["result_vehicle_content_sha256"])]["maturity"] == "qualified"


# 功能：
#   验证暂停中的旧运行结果不被发布，恢复后使用独立的新尝试目录。
# 输入：
#   tmp_path：隔离资产及数据库目录。
# 输出：
#   None：不返回业务数据。
def test_running_qualification_can_pause_and_resume_from_a_new_attempt(tmp_path: Path) -> None:
    map_archive, vehicle_archive = _fixtures(tmp_path)
    store = AppStore(tmp_path / "store")
    map_id, map_hash = _install_fixture_version(store, map_archive)
    vehicle_id, vehicle_hash = _install_fixture_version(store, vehicle_archive)
    environment = {"ros": "jazzy", "gazebo": "8.14.0", "px4_commit": "abc123"}
    entered_runtime = threading.Event()
    release_runtime = threading.Event()
    cancelled: list[str] = []

    # 功能：
    #   用事件暂停模拟执行器，稳定重现运行与用户暂停请求交错的情况。
    # 输入：
    #   kwargs：服务传入的当前运行参数。
    # 输出：
    #   evidence：解除阻塞后生成的模拟结果。
    def blocking_runner(**kwargs: object) -> dict[str, object]:
        entered_runtime.set()
        assert release_runtime.wait(timeout=5)
        work_root = Path(str(kwargs["work_root"]))
        plan = asset_qualification_cli.AssetPairQualificationPlan.model_validate_json(
            (work_root / "qualification-plan.json").read_bytes()
        )
        evidence = _runtime_evidence(plan, work_root)
        return evidence

    service = AssetQualificationService(
        store,
        AssetImportService(store),
        runtime_runner=blocking_runner,
        runtime_canceller=lambda job_id: not cancelled.append(job_id),
        environment_versions_provider=lambda: environment,
    )
    created = service.create(
        map_asset_id=map_id,
        map_content_sha256=map_hash,
        vehicle_asset_id=vehicle_id,
        vehicle_content_sha256=vehicle_hash,
    )
    job_id = str(created["job_id"])
    service.start(job_id)
    assert entered_runtime.wait(timeout=5)
    paused = service.pause(job_id)
    release_runtime.set()
    assert paused["state"] == "paused"
    assert cancelled == [job_id]
    _wait_for_job(store, job_id, "paused")
    deadline = time.monotonic() + 5
    while service._workers.get(job_id) is not None and time.monotonic() < deadline:
        time.sleep(0.02)

    service.runtime_runner = lambda **kwargs: _runtime_evidence(
        asset_qualification_cli.AssetPairQualificationPlan.model_validate_json(
            (Path(str(kwargs["work_root"])) / "qualification-plan.json").read_bytes()
        ),
        Path(str(kwargs["work_root"])),
    )
    service.start(job_id)
    qualified = _wait_for_job(store, job_id, "qualified", "failed")
    assert qualified["state"] == "qualified", qualified
    workspace, _map_root, _vehicle_root = store.asset_pair_qualification_paths(job_id)
    assert len(list(workspace.glob("attempt-*"))) == 2


# 功能：
#   验证默认资源重复初始化仍只保存同一配对，并能解析已绑定的内容与回执。
# 输入：
#   tmp_path：模拟全新安装的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_release_default_pair_is_immediately_resolvable_after_fresh_seed(tmp_path: Path) -> None:
    resources = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "desktop"
        / "src-tauri"
        / "resources"
        / "default-assets"
    )
    bundled_index = json.loads((resources / "index.json").read_text(encoding="utf-8"))
    qualified_pair = bundled_index["qualified_pair"]
    packages = {str(package["kind"]): package for package in qualified_pair["packages"]}
    map_hash = str(packages["map"]["content_sha256"])
    vehicle_hash = str(packages["vehicle"]["content_sha256"])
    store = AppStore(tmp_path / "fresh-install")
    service = AssetImportService(store)

    first = service.seed_bundled_sources(resources)
    second = service.seed_bundled_sources(resources)
    pair = store.qualified_asset_pair(
        map_asset_id="dronedream.school-map.v1",
        map_content_sha256=map_hash,
        vehicle_asset_id="dronedream.my-drone.v1",
        vehicle_content_sha256=vehicle_hash,
    )

    assert len(first) == 2
    assert len(second) == 2
    assert {
        (str(version["asset_id"]), str(version["content_sha256"]), str(version["maturity"]))
        for version in store.list_asset_versions()
    } == {
        (
            "dronedream.school-map.v1",
            map_hash,
            "qualified",
        ),
        (
            "dronedream.my-drone.v1",
            vehicle_hash,
            "qualified",
        ),
    }
    assert pair is not None
    receipt, receipt_sha256 = store.verified_asset_pair_receipt(pair)
    assert receipt.qualification_id == qualified_pair["qualification_id"]
    assert receipt_sha256 == qualified_pair["receipt_sha256"]
    binding = store.verified_asset_pair_execution_binding(pair)
    assert binding.map_qualified_content_sha256 == (map_hash)
    assert binding.vehicle_qualified_content_sha256 == (vehicle_hash)
    assert len(store.list_asset_pair_qualification_jobs()) == 1


# 功能：
#   验证当前默认资源入口拒绝退役的旧索引结构，不重新启用旧导入流程。
# 输入：
#   tmp_path：隔离资源及数据库目录。
# 输出：
#   None：不返回业务数据。
def test_default_asset_seed_rejects_the_retired_zip_index(tmp_path: Path) -> None:
    retired_resources = tmp_path / "retired-resources"
    retired_resources.mkdir()
    (retired_resources / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "dronedream.bundled-assets.v1",
                "bundles": [],
            }
        ),
        encoding="utf-8",
    )

    service = AssetImportService(AppStore(tmp_path / "store"))
    with pytest.raises(AssetImportError, match="BUNDLED_ASSET_INDEX_INVALID"):
        service.seed_bundled_sources(retired_resources)
