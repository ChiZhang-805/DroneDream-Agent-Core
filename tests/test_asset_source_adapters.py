"""Exercise declarative adapters and service integration without launching CAD applications."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from dronedream_agent_app.asset_import_service import AssetImportService
from dronedream_agent_app.storage import AppStore, AssetImportError
from dronedream_agent_core.asset_packages import inspect_ddpkg
from dronedream_agent_core.asset_source_adapters import (
    AssetSourceAdapterError,
    AssetSourceDetection,
    detect_asset_source,
    normalize_asset_source,
    source_adapter_catalog,
)


def test_catalog_separates_builtin_parsers_from_external_companions() -> None:
    """Catalog capabilities distinguish parsing from conversion that needs another process."""
    catalog = {entry["adapter_id"]: entry for entry in source_adapter_catalog()}

    assert catalog["gazebo.sdf"]["availability"] == "builtin"
    assert catalog["ros2.urdf"]["execution_boundary"] == "declarative-parser"
    assert catalog["blender.phobos"]["availability"] == "companion_required"
    assert catalog["solidworks.urdf"]["availability"] == "plugin_required"
    assert catalog["gazebo.fuel"]["required_application"] == "Gazebo Fuel Tools"
    assert catalog["nvidia.usd"]["availability"] == "plugin_required"
    assert catalog["autodesk.maya"]["execution_boundary"] == "isolated-plugin"
    assert catalog["unity.simulation"]["required_application"] == "Unity Editor"
    assert catalog["mesh.interchange"]["asset_kinds"] == ["map", "world", "vehicle"]


def test_normalizes_raw_gazebo_world_without_enabling_plugins(tmp_path: Path) -> None:
    """Normalization records embedded extensions as blocked and drops qualification claims."""
    source = tmp_path / "school.world"
    source.write_text(
        '<sdf version="1.10"><world name="School Map"><plugin name="wind" '
        'filename="libWindEffects.so"/></world></sdf>',
        encoding="utf-8",
    )
    detection = detect_asset_source(source)

    output = normalize_asset_source(
        source,
        detection,
        tmp_path / "school.ddpkg",
        expected_kind="map",
    )
    inspected = inspect_ddpkg(output)

    assert detection.source_format == "gazebo-world"
    assert inspected.asset_ir.kind == "map"
    assert inspected.asset_ir.simulation_targets[0].entrypoint.endswith("school.sdf")
    assert inspected.detected_runtime_extensions[0].name == "wind"
    assert inspected.detected_runtime_extensions[0].enabled_by_default is False
    assert inspected.detected_runtime_extensions[0].admission == "blocked"
    assert inspected.manifest.qualification is None


def test_normalizes_ros2_urdf_but_keeps_xacro_fail_closed(tmp_path: Path) -> None:
    """Plain URDF is declarative; macro expansion requires the isolated companion path."""
    urdf = tmp_path / "x500.urdf"
    urdf.write_text('<robot name="X500"><link name="base_link"/></robot>', encoding="utf-8")
    detection = detect_asset_source(urdf)
    inspected = inspect_ddpkg(normalize_asset_source(urdf, detection, tmp_path / "x500.ddpkg"))

    assert detection.adapter_id == "ros2.urdf"
    assert inspected.asset_ir.asset_id == "x500"
    assert inspected.asset_ir.kind == "vehicle"
    assert inspected.asset_ir.capabilities == ["ros2"]

    xacro = tmp_path / "x500.xacro"
    xacro.write_text(
        '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="X500"/>',
        encoding="utf-8",
    )
    xacro_detection = detect_asset_source(xacro)
    assert xacro_detection.can_normalize_locally is False
    assert xacro_detection.required_inputs == ["isolated_xacro_expansion"]
    with pytest.raises(AssetSourceAdapterError, match="ASSET_SOURCE_ADAPTER_REQUIRED"):
        normalize_asset_source(xacro, xacro_detection, tmp_path / "unsafe.ddpkg")


def test_normalizes_gazebo_bundle_and_rejects_executable_members(tmp_path: Path) -> None:
    """A valid model bundle may carry meshes, not executable files for later extraction."""
    bundle = tmp_path / "x500.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("x500/model.config", "<model><name>X500</name></model>")
        archive.writestr(
            "x500/model.sdf",
            '<sdf version="1.10"><model name="X500"><link name="base_link"/></model></sdf>',
        )
        archive.writestr("x500/meshes/body.stl", b"solid body\nendsolid body\n")
    detection = detect_asset_source(bundle)
    inspected = inspect_ddpkg(normalize_asset_source(bundle, detection, tmp_path / "x500.ddpkg"))

    assert detection.source_format == "gazebo-model-bundle"
    assert inspected.asset_ir.kind == "vehicle"
    assert any(entry.role == "visual" for entry in inspected.asset_ir.files)

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("model.config", "<model/>")
        archive.writestr("model.sdf", '<sdf version="1.10"><model name="bad"/></sdf>')
        archive.writestr("plugins/attack.sh", "#!/bin/sh")
    unsafe_detection = detect_asset_source(unsafe)
    with pytest.raises(
        AssetSourceAdapterError,
        match="ASSET_SOURCE_EXECUTABLE_MEMBER_FORBIDDEN",
    ):
        normalize_asset_source(unsafe, unsafe_detection, tmp_path / "unsafe.ddpkg")


def test_release_normalization_can_retain_provenance_without_embedding_source_zip(
    tmp_path: Path,
) -> None:
    """Reviewed bundled resources keep an immutable source digest, not a nested archive."""

    bundle = tmp_path / "map.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("world.sdf", '<sdf version="1.10"><world name="map"/></sdf>')
    detection = detect_asset_source(bundle)
    output = normalize_asset_source(
        bundle,
        detection,
        tmp_path / "map.ddpkg",
        expected_kind="map",
        embed_source_snapshot=False,
    )
    inspected = inspect_ddpkg(output)

    paths = {entry.path for entry in inspected.manifest.files}
    assert "source/source-reference.json" in paths
    assert not any(path.casefold().endswith(".zip") for path in paths)
    with zipfile.ZipFile(output) as archive:
        reference = json.loads(archive.read("source/source-reference.json"))
    assert reference["embedded"] is False
    assert reference["sha256"] == inspected.asset_ir.source.source_sha256


def test_migrates_legacy_zip_into_ddpkg_and_invalidates_old_qualification(tmp_path: Path) -> None:
    """Compatibility is an explicit conversion boundary, never reuse of old flight approval."""
    bundle = tmp_path / "school-map.zip"
    legacy_manifest = {
        "schema_version": "dronedream.asset-bundle.v1",
        "kind": "map",
        "asset_id": "dronedream.school-map.v1",
        "name": "School Map",
        "qualification_status": "qualified",
        "files": {
            "graph": "graph.json",
            "semantic": "semantic.json",
            "world_sdf": "world.sdf",
        },
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("manifest.json", json.dumps(legacy_manifest))
        archive.writestr("graph.json", "{}")
        archive.writestr("semantic.json", "{}")
        archive.writestr(
            "world.sdf",
            '<sdf version="1.10"><world name="School Map"/></sdf>',
        )

    detection = detect_asset_source(bundle)
    inspected = inspect_ddpkg(
        normalize_asset_source(bundle, detection, tmp_path / "school-map.ddpkg")
    )

    assert detection.source_format == "dronedream-legacy-bundle"
    assert detection.adapter_id == "dronedream.legacy-bundle"
    assert inspected.asset_ir.asset_id == "dronedream.school-map.v1"
    assert inspected.asset_ir.name == "School Map"
    assert inspected.asset_ir.capabilities == ["gazebo", "legacy-migrated"]
    assert inspected.asset_ir.semantics.navigation_graph_path == (
        "normalized/legacy/graph.json"
    )
    assert any(entry.path == "normalized/legacy/graph.json" for entry in inspected.asset_ir.files)
    assert inspected.manifest.qualification is None


@pytest.mark.parametrize(
    ("filename", "adapter_id", "required_input"),
    [
        ("airframe.blend", "blender.phobos", "blender_phobos_companion"),
        ("airframe.sldasm", "solidworks.urdf", "solidworks_connector"),
        ("airframe.f3d", "autodesk.fusion", "fusion_connector"),
        ("airframe.ma", "autodesk.maya", "maya_connector"),
        ("campus.max", "autodesk.3dsmax", "3dsmax_connector"),
        ("campus.rvt", "autodesk.revit", "revit_connector"),
        ("campus.skp", "trimble.sketchup", "sketchup_connector"),
        ("campus.usd", "nvidia.usd", "openusd_semantics_and_physics"),
        ("campus.unitypackage", "unity.simulation", "unity_connector"),
        ("campus.uproject", "unreal.robotics", "unreal_connector"),
        ("campus.glb", "mesh.interchange", "mesh_semantics_and_physics"),
        ("airframe.scad", "openscad.robotics", "openscad_connector"),
        ("campus.tif", "gis.geospatial", "gis_connector"),
    ],
)
def test_detects_native_projects_without_claiming_local_conversion(
    tmp_path: Path,
    filename: str,
    adapter_id: str,
    required_input: str,
) -> None:
    """Recognizing a native format must not imply that its converter is installed."""
    source = tmp_path / filename
    source.write_bytes(b"native-project")

    detection = detect_asset_source(source)

    assert detection.adapter_id == adapter_id
    assert detection.can_normalize_locally is False
    assert detection.required_inputs == [required_input]


def test_import_service_persists_detection_and_waits_for_real_qualification(
    tmp_path: Path,
) -> None:
    """Persist the detected adapter and normalized visual asset while awaiting evidence."""
    source = tmp_path / "campus.sdf"
    source.write_text(
        '<sdf version="1.10"><world name="Campus"><model name="building"/></world></sdf>',
        encoding="utf-8",
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

    assert waiting["state"] == "needs_input"
    assert waiting["detected_source_format"] == "gazebo-world"
    assert waiting["source_adapter_id"] == "gazebo.sdf"
    assert waiting["asset_id"] == "campus"
    assert waiting["required_inputs"] == ["qualification_evidence"]
    assert (store.asset_quarantine_root / str(created["job_id"]) / "normalized.ddpkg").is_file()
    versions = store.list_asset_versions()
    assert len(versions) == 1
    assert versions[0]["maturity"] == "visual_only"


def test_import_service_reports_missing_external_adapter_as_required_input(
    tmp_path: Path,
) -> None:
    """Missing converters produce an actionable input request and no invented asset."""
    source = tmp_path / "airframe.blend"
    source.write_bytes(b"BLENDER")
    store = AppStore(tmp_path / "store")
    service = AssetImportService(store)
    created = service.create(
        source=source,
        source_name=source.name,
        source_format="auto",
        expected_kind="vehicle",
    )

    waiting = service.process(str(created["job_id"]))

    assert waiting["state"] == "needs_input"
    assert waiting["source_adapter_id"] == "blender.phobos"
    assert waiting["required_inputs"] == ["plugin_adapter:blender.phobos"]
    assert store.list_asset_versions() == []


def test_import_service_uses_enabled_plugin_normalizer_before_requesting_companion(
    tmp_path: Path,
) -> None:
    """An available adapter callback can normalize but cannot certify its own result."""
    source = tmp_path / "airframe.blend"
    source.write_bytes(b"BLENDER")
    store = AppStore(tmp_path / "store")
    calls: list[str] = []

    def plugin_normalizer(**values: object) -> Path:
        """Stand in for a converter with a tiny SDF; no Blender process is exercised."""
        detection = values["detection"]
        destination = values["destination"]
        assert isinstance(detection, AssetSourceDetection)
        assert detection.adapter_id == "blender.phobos"
        assert isinstance(destination, Path)
        calls.append(detection.adapter_id)
        staged_sdf = tmp_path / "converted.sdf"
        staged_sdf.write_text(
            '<sdf version="1.10"><model name="ConvertedDrone">'
            '<link name="base_link"/></model></sdf>',
            encoding="utf-8",
        )
        staged_detection = detect_asset_source(staged_sdf)
        return normalize_asset_source(
            staged_sdf,
            staged_detection,
            destination,
            expected_kind="vehicle",
        )

    service = AssetImportService(store, plugin_source_normalizer=plugin_normalizer)
    created = service.create(
        source=source,
        source_name=source.name,
        source_format="auto",
        expected_kind="vehicle",
    )

    waiting = service.process(str(created["job_id"]))

    assert calls == ["blender.phobos"]
    assert waiting["state"] == "needs_input"
    assert waiting["required_inputs"] == ["qualification_evidence"]
    assert waiting["asset_id"] == "converteddrone"
    assert store.list_asset_versions()[0]["maturity"] == "visual_only"


def test_declared_format_mismatch_fails_without_leaking_parser_details(tmp_path: Path) -> None:
    """An explicit format conflicting with detected content becomes a stable job failure."""
    source = tmp_path / "vehicle.urdf"
    source.write_text('<robot name="vehicle"/>', encoding="utf-8")
    store = AppStore(tmp_path / "store")
    service = AssetImportService(store)
    created = service.create(
        source=source,
        source_name=source.name,
        source_format="gazebo-sdf",
    )

    with pytest.raises(AssetImportError, match="^ASSET_SOURCE_FORMAT_MISMATCH$"):
        service.process(str(created["job_id"]))

    failed = store.get_asset_import_job(str(created["job_id"]))
    assert failed["issue_codes"] == ["ASSET_SOURCE_FORMAT_MISMATCH"]
