"""Resolve immutable asset versions into the files consumed by planning and flight."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dronedream_agent_core.asset_package_storage import (
    copy_asset_source,
    publish_asset_directory,
    read_stored_manifest,
)
from dronedream_agent_core.asset_packages import (
    MAX_ASSET_IR_BYTES,
    MAX_MEMBER_UNCOMPRESSED_BYTES,
    AssetIR,
    DDPkgManifest,
    normalized_member_path,
    qualification_is_current,
)
from dronedream_agent_core.contracts import VehicleAsset
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    read_plugin_file,
)
from dronedream_agent_core.xml_values import parse_xml
from dronedream_plugin_sdk.protocol import decode_json

_MAX_SDF_BYTES = 64 * 1024 * 1024


class AssetRuntimeResolutionError(ValueError):
    """A selected asset cannot safely enter planning or execution."""


@dataclass(frozen=True)
class ResolvedMapAsset:
    asset_id: str
    content_sha256: str | None
    root: Path
    world_sdf: Path
    semantic: Path
    graph: Path
    versioned: bool


@dataclass(frozen=True)
class ResolvedVehicleAsset:
    asset_id: str
    content_sha256: str | None
    root: Path
    vehicle_sdf: Path
    vehicle_metadata: Path
    vehicle_summary: Path | None
    controller_params: Path
    payload_sdf: Path | None
    versioned: bool


@dataclass(frozen=True)
class ResolvedDevelopmentMissionInput:
    """One explicitly selected, hash-pinned development mission input.

    Development inputs are deliberately separate from qualified release assets.
    A caller must select the manifest, every vehicle file is re-hashed before use,
    and the exact qualified map is bound by content and executable-file hashes.
    This prevents a current vehicle from being planned against a historical map.
    """

    manifest_path: Path
    manifest_sha256: str
    asset_id: str
    map_asset_id: str
    map_content_sha256: str
    map_world_sdf_sha256: str
    map_semantic_sha256: str
    map_graph_sha256: str
    root: Path
    vehicle_sdf: Path
    vehicle_metadata: Path
    vehicle_summary: Path
    controller_params: Path
    payload_sdf: Path
    payload_mass_kg: float


# 功能：
#   对普通资产文件进行有界哈希，并检测读取期间的文件替换或大小变化。
# 输入：
#   path：待核对的资产文件。
# 输出：
#   digest：实际读取字节的 SHA-256。
def _sha256(path: Path) -> str:
    try:
        digest = hash_plugin_file(path, limit=MAX_MEMBER_UNCOMPRESSED_BYTES)
    except (OSError, ValueError) as error:
        raise AssetRuntimeResolutionError("ASSET_RUNTIME_FILE_INVALID") from error
    return digest


# 功能：
#   检查声明路径的规范性、目录归属和文件摘要，拒绝链接及路径回退。
# 输入：
#   root：所属资产根目录。
#   relative：清单中的规范相对路径。
#   expected_sha256：可选的声明摘要。
# 输出：
#   candidate：通过当前检查的绝对文件路径。
def _checked_file(root: Path, relative: str, expected_sha256: str | None = None) -> Path:
    try:
        if normalized_member_path(relative) != relative:
            raise ValueError("noncanonical path")
        check_plain_plugin_path(root)
        candidate = root / relative
        check_plain_plugin_path(candidate)
        resolved_root = root.resolve()
        candidate = candidate.resolve()
        if resolved_root not in candidate.parents or not candidate.is_file():
            raise ValueError("file outside root or missing")
    except (OSError, TypeError, ValueError) as error:
        raise AssetRuntimeResolutionError("ASSET_RUNTIME_FILE_INVALID") from error
    if expected_sha256 is not None and _sha256(candidate) != expected_sha256:
        raise AssetRuntimeResolutionError("ASSET_RUNTIME_FILE_HASH_MISMATCH")
    return candidate


# 功能：
#   验证外部摘要使用完整的小写 SHA-256 表示，不修正或截断非法声明。
# 输入：
#   value：清单中的摘要值。
#   error_code：声明不合法时使用的领域错误码。
# 输出：
#   value：通过验证的摘要字符串。
def _declared_sha256(value: object, error_code: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AssetRuntimeResolutionError(error_code)
    return value


# 功能：
#   读取具有明确大小与摘要约束的同一份字节，供解析器消费。
# 输入：
#   path：待解析文件。
#   limit：允许读取的最大字节数。
#   expected_sha256：可选的预期内容摘要。
# 输出：
#   content：通过普通文件和摘要检查的字节串。
def _read_bound_bytes(path: Path, limit: int, expected_sha256: str | None = None) -> bytes:
    content = read_plugin_file(path, limit=limit)
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise AssetRuntimeResolutionError("ASSET_RUNTIME_FILE_HASH_MISMATCH")
    return content


# 功能：
#   将文件解析成受大小、结构、重复键和有限数值约束的 JSON 字典。
# 输入：
#   path：待解析的 JSON 文件。
#   expected_sha256：可选的预期内容摘要。
# 输出：
#   payload：已验证的 JSON 字典。
def _read_json(path: Path, expected_sha256: str | None = None) -> dict[str, object]:
    raw = _read_bound_bytes(path, MAX_ASSET_IR_BYTES, expected_sha256)
    payload = decode_json(raw, limit=MAX_ASSET_IR_BYTES, node_limit=2_000_000)
    if not isinstance(payload, dict):
        raise AssetRuntimeResolutionError("ASSET_RUNTIME_JSON_OBJECT_REQUIRED")
    return payload


# 功能：
#   验证物理量确实为有限数值，不将布尔值、字符串或溢出整数转换成测量值。
# 输入：
#   value：控制器或负载摘要中的待检查值。
# 输出：
#   number：通过有限性检查的浮点数。
def _finite_number(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("physical quantity must be numeric")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("physical quantity overflow") from error
    if not math.isfinite(number):
        raise ValueError("physical quantity must be finite")
    return number


# 功能：
#   提取 SDF 的唯一顶层模型，避免将嵌套零件或第二个模型当成当前飞机。
# 输入：
#   root：已通过有界 XML 解析的根元素。
# 输出：
#   model：唯一顶层模型元素。
def _one_sdf_model(root: ET.Element) -> ET.Element:
    models = [root] if root.tag == "model" else list(root.findall("model"))
    if root.tag not in {"sdf", "model"} or len(models) != 1:
        raise ValueError("one unambiguous top-level SDF model is required")
    model = models[0]
    return model


# 功能：
#   1. 解析当前开发输入，将飞机、控制器、负载与指定地图绑定到确切字节。
#   2. 验证开发用挂载约定和质量约束；此入口只支持仿真，不授予真机资格。
# 输入：
#   manifest_path：用户或验收流程明确选定的开发输入清单。
# 输出：
#   selection：具有内容绑定的开发输入文件与负载信息。
def resolve_development_mission_input(manifest_path: Path) -> ResolvedDevelopmentMissionInput:
    try:
        # 先检查原始路径再解析绝对路径，避免 resolve 抹去符号链接信息。
        manifest_bytes = _read_bound_bytes(manifest_path, MAX_ASSET_IR_BYTES)
        resolved_manifest = manifest_path.resolve()
        payload = decode_json(manifest_bytes, limit=MAX_ASSET_IR_BYTES, node_limit=2_000_000)
    except FileNotFoundError as error:
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_MANIFEST_MISSING") from error
    except (OSError, ValueError) as error:
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_MANIFEST_INVALID") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "dronedream.current-development-mission-input.v1"
        or payload.get("status") != "current-development"
        or not isinstance(payload.get("asset_id"), str)
        or not payload["asset_id"]
    ):
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_MANIFEST_NOT_CURRENT")
    map_binding = payload.get("map_binding")
    if (
        not isinstance(map_binding, dict)
        or not isinstance(map_binding.get("asset_id"), str)
        or not map_binding["asset_id"]
    ):
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_MAP_BINDING_INVALID")
    map_asset_id = str(map_binding["asset_id"])
    map_content_sha256 = _declared_sha256(
        map_binding.get("content_sha256"), "DEVELOPMENT_INPUT_MAP_BINDING_INVALID"
    )
    map_world_sdf_sha256 = _declared_sha256(
        map_binding.get("world_sdf_sha256"), "DEVELOPMENT_INPUT_MAP_BINDING_INVALID"
    )
    map_semantic_sha256 = _declared_sha256(
        map_binding.get("semantic_sha256"), "DEVELOPMENT_INPUT_MAP_BINDING_INVALID"
    )
    map_graph_sha256 = _declared_sha256(
        map_binding.get("graph_sha256"), "DEVELOPMENT_INPUT_MAP_BINDING_INVALID"
    )
    files = payload.get("files")
    if not isinstance(files, dict):
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_MANIFEST_INVALID")
    root = resolved_manifest.parent.resolve()

    # 功能：
    #   解析当前清单中唯一角色的文件声明，并核对其实际内容摘要。
    # 输入：
    #   name：需要解析的文件角色。
    # 输出：
    #   path：归属于当前开发输入的已核对文件。
    def declared_file(name: str) -> Path:
        declaration = files.get(name)
        if not isinstance(declaration, dict):
            raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_FILE_UNDECLARED")
        relative = declaration.get("path")
        expected_sha256 = declaration.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha256, str):
            raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_FILE_DECLARATION_INVALID")
        path = _checked_file(
            root,
            relative,
            _declared_sha256(expected_sha256, "DEVELOPMENT_INPUT_FILE_HASH_INVALID"),
        )
        return path

    vehicle_sdf = declared_file("vehicle_sdf")
    vehicle_metadata = declared_file("vehicle_metadata")
    vehicle_summary = declared_file("vehicle_summary")
    controller_params = declared_file("controller_params")
    payload_sdf = declared_file("payload_sdf")
    try:
        controller = _read_json(controller_params, files["controller_params"]["sha256"])
        for key in ("vel_limit", "accel_limit"):
            if _finite_number(controller.get(key)) <= 0.0:
                raise ValueError("control limit must be positive")
    except (OSError, ValueError) as error:
        raise AssetRuntimeResolutionError("DEVELOPMENT_CONTROLLER_PARAMS_INVALID") from error
    try:
        vehicle = VehicleAsset.model_validate(
            _read_json(vehicle_metadata, files["vehicle_metadata"]["sha256"])
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise AssetRuntimeResolutionError("DEVELOPMENT_VEHICLE_METADATA_INVALID") from error
    asset_id = str(payload["asset_id"])
    if vehicle.asset_id != asset_id:
        raise AssetRuntimeResolutionError("DEVELOPMENT_VEHICLE_IDENTITY_MISMATCH")
    try:
        payload_bytes = _read_bound_bytes(
            payload_sdf, _MAX_SDF_BYTES, files["payload_sdf"]["sha256"]
        )
        payload_tree = _one_sdf_model(
            parse_xml(payload_bytes, maximum_bytes=_MAX_SDF_BYTES, maximum_elements=1_000_000)
        )
        payload_link_masses = [
            float(node.text)
            for node in payload_tree.findall(".//link/inertial/mass")
            if node.text is not None
        ]
    except (ET.ParseError, OSError, TypeError, ValueError) as error:
        raise AssetRuntimeResolutionError("DEVELOPMENT_PAYLOAD_SDF_INVALID") from error
    if not payload_link_masses or any(
        not math.isfinite(value) or value <= 0.0 for value in payload_link_masses
    ):
        raise AssetRuntimeResolutionError("DEVELOPMENT_PAYLOAD_MASS_INVALID")
    payload_mass_kg = sum(payload_link_masses)
    if not math.isfinite(payload_mass_kg):
        raise AssetRuntimeResolutionError("DEVELOPMENT_PAYLOAD_MASS_INVALID")
    permitted_payload_kg = min(
        vehicle.max_pickup_payload_kg,
        max(0.0, vehicle.max_takeoff_mass_kg - vehicle.dry_mass_kg),
    )
    if payload_mass_kg > permitted_payload_kg + 1e-9:
        raise AssetRuntimeResolutionError("DEVELOPMENT_PAYLOAD_EXCEEDS_VEHICLE_CAPACITY")
    try:
        summary = _read_json(vehicle_summary, files["vehicle_summary"]["sha256"])
    except (OSError, ValueError) as error:
        raise AssetRuntimeResolutionError("DEVELOPMENT_VEHICLE_SUMMARY_INVALID") from error
    payload_summary = summary.get("mission_payload") if isinstance(summary, dict) else None
    payload_hashes = summary.get("payload_files_sha256") if isinstance(summary, dict) else None
    if not isinstance(payload_summary, dict) or not isinstance(payload_hashes, dict):
        raise AssetRuntimeResolutionError("DEVELOPMENT_VEHICLE_SUMMARY_INVALID")
    try:
        declared_payload_mass_kg = _finite_number(payload_summary["mass_kg"])
        loaded_mass_kg = _finite_number(payload_summary["loaded_mass_kg"])
        attachment_offset_m = _finite_number(payload_summary["center_above_model_root_m"])
        attachment_error_m = _finite_number(payload_summary["maximum_attachment_error_m"])
        declared_dry_mass_kg = _finite_number(summary["dry_mass_kg"])
        declared_maximum_payload_kg = _finite_number(summary["maximum_qualified_payload_kg"])
    except (KeyError, TypeError, ValueError) as error:
        raise AssetRuntimeResolutionError("DEVELOPMENT_VEHICLE_SUMMARY_INVALID") from error
    try:
        vehicle_bytes = _read_bound_bytes(
            vehicle_sdf, _MAX_SDF_BYTES, files["vehicle_sdf"]["sha256"]
        )
        vehicle_model = _one_sdf_model(
            parse_xml(vehicle_bytes, maximum_bytes=_MAX_SDF_BYTES, maximum_elements=1_000_000)
        )
        detachable_plugins = [
            element
            for element in vehicle_model.findall("plugin")
            if "detachable"
            in (f"{element.attrib.get('name', '')} {element.attrib.get('filename', '')}").casefold()
        ]
        if len(detachable_plugins) != 1:
            raise ValueError("one detachable plugin required")
        plugin_values: dict[str, str] = {}
        for element in detachable_plugins[0]:
            if element.tag in plugin_values:
                raise ValueError("duplicate detachable plugin property")
            plugin_values[element.tag] = (element.text or "").strip()
    except (ET.ParseError, OSError, ValueError) as error:
        raise AssetRuntimeResolutionError("DEVELOPMENT_VEHICLE_SDF_INVALID") from error
    if (
        vehicle_model.attrib.get("name") != summary.get("model_name")
        or summary.get("model_name") != "my_drone"
        or summary.get("source_model") != "model://x500_depth"
        or payload_summary.get("model_name") != "takeout_payload"
        or payload_summary.get("link_name") != "payload_link"
        or plugin_values.get("child_model") != payload_summary.get("model_name")
        or plugin_values.get("child_link") != payload_summary.get("link_name")
        or plugin_values.get("attach_topic") != payload_summary.get("attach_topic")
        or plugin_values.get("detach_topic") != payload_summary.get("detach_topic")
        or plugin_values.get("output_topic") != payload_summary.get("state_topic")
        or payload_summary.get("attach_topic") != "/model/my_drone/takeout_payload/attach"
        or payload_summary.get("detach_topic") != "/model/my_drone/takeout_payload/detach"
        or payload_summary.get("state_topic") != "/model/my_drone/takeout_payload/state"
        or not math.isclose(declared_payload_mass_kg, payload_mass_kg, abs_tol=1e-9)
        or not math.isclose(loaded_mass_kg, vehicle.dry_mass_kg + payload_mass_kg, abs_tol=1e-9)
        or not math.isclose(declared_dry_mass_kg, vehicle.dry_mass_kg, abs_tol=1e-9)
        or not math.isclose(
            declared_maximum_payload_kg,
            vehicle.max_pickup_payload_kg,
            abs_tol=1e-9,
        )
        or not 0.0 < attachment_offset_m <= 1.0
        or not 0.0 < attachment_error_m <= 0.1
        or payload_hashes.get("model.sdf") != hashlib.sha256(vehicle_bytes).hexdigest()
        or payload_hashes.get("takeout-payload.sdf") != hashlib.sha256(payload_bytes).hexdigest()
    ):
        raise AssetRuntimeResolutionError("DEVELOPMENT_VEHICLE_SUMMARY_MISMATCH")
    capabilities = payload.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or any(not isinstance(value, str) or not value for value in capabilities)
        or "forward-depth" not in capabilities
        or "payload-attachment" not in capabilities
        or "model://x500_depth" not in [node.text for node in vehicle_model.findall("include/uri")]
        or "DetachableJoint"
        not in (
            f"{detachable_plugins[0].get('name', '')} {detachable_plugins[0].get('filename', '')}"
        )
    ):
        raise AssetRuntimeResolutionError("DEVELOPMENT_VEHICLE_CAPABILITY_MISMATCH")
    # 返回路径前再次检查已解析材料，防止跨文件读取期间混入另一批内容。
    for role in (
        "vehicle_sdf",
        "vehicle_metadata",
        "vehicle_summary",
        "controller_params",
        "payload_sdf",
    ):
        declared_file(role)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if _sha256(resolved_manifest) != manifest_sha256:
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_MANIFEST_CHANGED")
    selection = ResolvedDevelopmentMissionInput(
        manifest_path=resolved_manifest,
        manifest_sha256=manifest_sha256,
        asset_id=asset_id,
        map_asset_id=map_asset_id,
        map_content_sha256=map_content_sha256,
        map_world_sdf_sha256=map_world_sdf_sha256,
        map_semantic_sha256=map_semantic_sha256,
        map_graph_sha256=map_graph_sha256,
        root=root,
        vehicle_sdf=vehicle_sdf,
        vehicle_metadata=vehicle_metadata,
        vehicle_summary=vehicle_summary,
        controller_params=controller_params,
        payload_sdf=payload_sdf,
        payload_mass_kg=payload_mass_kg,
    )
    return selection


# 功能：
#   拒绝将当前开发飞机输入用于另一张地图或被替换的地图执行文件。
# 输入：
#   development_input：已经冻结地图身份与摘要的开发输入。
#   selected_map：本次规划或执行明确选择的地图。
# 输出：
#   None：不返回业务数据。
def assert_development_map_binding(
    development_input: ResolvedDevelopmentMissionInput,
    selected_map: ResolvedMapAsset,
) -> None:
    if (
        selected_map.asset_id != development_input.map_asset_id
        or selected_map.content_sha256 != development_input.map_content_sha256
        or _sha256(selected_map.world_sdf) != development_input.map_world_sdf_sha256
        or _sha256(selected_map.semantic) != development_input.map_semantic_sha256
        or _sha256(selected_map.graph) != development_input.map_graph_sha256
    ):
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_MAP_BINDING_MISMATCH")


# 功能：
#   冻结原开发飞机文件并绑定当前版本化地图，完整验证后发布；仍只供仿真使用。
# 输入：
#   source_manifest：原开发输入清单。
#   selected_map：已经过资产准入的本次地图选择。
#   output_root：不可覆盖的新输入目录。
# 输出：
#   manifest_path：新发布的开发输入清单。
def rebind_development_mission_input_to_map(
    source_manifest: Path,
    selected_map: ResolvedMapAsset,
    output_root: Path,
) -> Path:
    source = resolve_development_mission_input(source_manifest)
    if not selected_map.versioned or not selected_map.content_sha256:
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_MAP_NOT_VERSIONED")
    source_payload = _read_json(source.manifest_path, source.manifest_sha256)
    files = {
        "vehicle_sdf": source.vehicle_sdf,
        "vehicle_metadata": source.vehicle_metadata,
        "vehicle_summary": source.vehicle_summary,
        "controller_params": source.controller_params,
        "payload_sdf": source.payload_sdf,
    }
    hashes = {role: source_payload["files"][role]["sha256"] for role in files}
    manifest = {
        **source_payload,
        "map_binding": {
            "asset_id": selected_map.asset_id,
            "content_sha256": selected_map.content_sha256,
            "world_sdf_sha256": _sha256(selected_map.world_sdf),
            "semantic_sha256": _sha256(selected_map.semantic),
            "graph_sha256": _sha256(selected_map.graph),
        },
        "limitations": ["simulation-only until promoted through the qualified asset-pair pipeline"],
    }
    manifest_path = _publish_development_input(
        files,
        hashes,
        manifest,
        output_root,
        protected_roots=(source.root, selected_map.root),
        verify_binding=lambda rebound: assert_development_map_binding(rebound, selected_map),
    )
    return manifest_path


# 功能：
#   替换开发输入的飞机契约，要求执行模型、控制器和负载字节一致；不授予真机资格。
# 输入：
#   source_manifest：原开发输入清单。
#   selected_vehicle：已经过资产准入的飞机选择。
#   output_root：不可覆盖的新开发输入目录。
# 输出：
#   manifest_path：保留原地图绑定的新开发输入清单。
def rebind_development_mission_input_to_qualified_vehicle(
    source_manifest: Path,
    selected_vehicle: ResolvedVehicleAsset,
    output_root: Path,
) -> Path:
    source = resolve_development_mission_input(source_manifest)
    if not selected_vehicle.versioned or not selected_vehicle.content_sha256:
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_VEHICLE_NOT_VERSIONED")
    if selected_vehicle.vehicle_summary is None or selected_vehicle.payload_sdf is None:
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_VEHICLE_PAYLOAD_CONTRACT_MISSING")
    source_payload = _read_json(source.manifest_path, source.manifest_sha256)
    files = {
        "vehicle_sdf": selected_vehicle.vehicle_sdf,
        "vehicle_metadata": selected_vehicle.vehicle_metadata,
        "vehicle_summary": selected_vehicle.vehicle_summary,
        "controller_params": selected_vehicle.controller_params,
        "payload_sdf": selected_vehicle.payload_sdf,
    }
    hashes = {role: _sha256(path) for role, path in files.items()}
    if any(
        hashes[role] != source_payload["files"][role]["sha256"]
        for role in ("vehicle_sdf", "controller_params", "payload_sdf")
    ):
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_VEHICLE_RUNTIME_MISMATCH")
    try:
        promoted = VehicleAsset.model_validate(
            _read_json(selected_vehicle.vehicle_metadata, hashes["vehicle_metadata"])
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_VEHICLE_METADATA_INVALID") from error
    if (
        promoted.asset_id != selected_vehicle.asset_id
        or "oakd-lite-depth" not in promoted.sensors
        or promoted.max_pickup_payload_kg <= 0.0
    ):
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_VEHICLE_IDENTITY_MISMATCH")

    manifest = {
        **source_payload,
        "asset_id": selected_vehicle.asset_id,
        "limitations": ["simulation-only bootstrap using an already qualified vehicle contract"],
    }
    manifest_path = _publish_development_input(
        files,
        hashes,
        manifest,
        output_root,
        protected_roots=(source.root, selected_vehicle.root),
    )
    return manifest_path


# 功能：
#   1. 按固定摘要复制开发输入，拒绝同名碰撞和复制期间的源文件变化。
#   2. 完整验证后不可覆盖地发布目录，只清理本次独占创建的暂存目录。
# 输入：
#   files：文件角色与已核对源路径的映射。
#   hashes：各角色必须保持的源文件摘要。
#   manifest：需要发布的开发清单内容。
#   output_root：新输入目录。
#   protected_roots：禁止写入的现有源资产根目录。
#   verify_binding：可选的发布前地图绑定复核函数。
# 输出：
#   manifest_path：最终发布目录中的清单路径。
def _publish_development_input(
    files: dict[str, Path],
    hashes: dict[str, str],
    manifest: dict[str, object],
    output_root: Path,
    *,
    protected_roots: tuple[Path, ...],
    verify_binding: Callable[[ResolvedDevelopmentMissionInput], None] | None = None,
) -> Path:
    check_plain_plugin_path(output_root)
    destination = output_root.resolve()
    if any(destination.is_relative_to(root.resolve()) for root in protected_roots):
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_OUTPUT_INSIDE_SOURCE")
    if destination.exists():
        raise FileExistsError("development input output already exists")
    names = [path.name.casefold() for path in files.values()]
    if len(names) != len(set(names)) or "current-input.json" in names:
        raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_FILENAME_COLLISION")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.mkdir()
    owned = temporary.stat()
    try:
        declarations: dict[str, dict[str, str]] = {}
        for role, source_path in files.items():
            relative = normalized_member_path(source_path.name)
            target = temporary / relative
            with target.open("xb") as stream:
                digest = copy_asset_source(source_path, stream, MAX_MEMBER_UNCOMPRESSED_BYTES)
            if digest != hashes[role]:
                raise AssetRuntimeResolutionError("DEVELOPMENT_INPUT_SOURCE_CHANGED")
            declarations[role] = {"path": relative, "sha256": digest}
        manifest = {**manifest, "files": declarations}
        manifest_path = temporary / "current-input.json"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        rebound = resolve_development_mission_input(manifest_path)
        if verify_binding is not None:
            verify_binding(rebound)
        publish_asset_directory(temporary, destination)
    finally:
        if temporary.exists():
            check_plain_plugin_path(temporary)
            if temporary.resolve().parent == destination.parent and os.path.samestat(
                owned, temporary.stat()
            ):
                shutil.rmtree(temporary)
    manifest_path = destination / "current-input.json"
    return manifest_path


# 功能：
#   将数据库身份、资格、规范化信息与磁盘清单和实际 IR 字节交叉核对。
# 输入：
#   record：选定的资产版本数据库记录。
#   expected_kind：调用方需要的资产种类。
# 输出：
#   version：资产根目录、清单和规范化信息组成的元组。
def _validated_version(
    record: dict[str, object],
    expected_kind: Literal["map", "world", "vehicle"],
) -> tuple[Path, DDPkgManifest, AssetIR]:
    try:
        manifest = DDPkgManifest.model_validate(record["manifest"])
        asset_ir = AssetIR.model_validate(record["asset_ir"])
        root = Path(str(record["bundle_root"]))
        check_plain_plugin_path(root)
        root = root.resolve()
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise AssetRuntimeResolutionError("ASSET_VERSION_RECORD_INVALID") from error
    if (
        manifest.asset_id != asset_ir.asset_id
        or manifest.asset_kind != asset_ir.kind
        or record.get("asset_id") != manifest.asset_id
        or record.get("kind") != manifest.asset_kind
    ):
        raise AssetRuntimeResolutionError("ASSET_VERSION_IDENTITY_MISMATCH")
    if manifest.asset_kind != expected_kind and not (
        expected_kind == "map" and manifest.asset_kind == "world"
    ):
        raise AssetRuntimeResolutionError("ASSET_VERSION_KIND_MISMATCH")
    if str(record.get("content_sha256")) != manifest.content_sha256:
        raise AssetRuntimeResolutionError("ASSET_VERSION_CONTENT_HASH_MISMATCH")
    if str(record.get("maturity")) != "qualified":
        raise AssetRuntimeResolutionError("ASSET_VERSION_NOT_QUALIFIED")
    binding = manifest.qualification
    if (
        binding is None
        or binding.maturity != "qualified"
        or binding.content_sha256 != manifest.content_sha256
    ):
        raise AssetRuntimeResolutionError("ASSET_VERSION_QUALIFICATION_INVALID")
    try:
        stored_manifest, _ = read_stored_manifest(root)
        if stored_manifest.model_dump(mode="json") != manifest.model_dump(mode="json"):
            raise ValueError("stored manifest differs from database")
    except (OSError, ValueError) as error:
        raise AssetRuntimeResolutionError("ASSET_VERSION_STORED_MANIFEST_INVALID") from error
    try:
        entry = next(item for item in manifest.files if item.path == manifest.asset_ir_path)
        payload = _read_bound_bytes(
            root / entry.path, min(entry.size_bytes, MAX_ASSET_IR_BYTES), entry.sha256
        )
        if len(payload) != entry.size_bytes:
            raise ValueError("asset IR size mismatch")
        stored_ir = AssetIR.model_validate(
            decode_json(payload, limit=MAX_ASSET_IR_BYTES, node_limit=2_000_000)
        )
        if stored_ir.model_dump(mode="json") != asset_ir.model_dump(mode="json"):
            raise ValueError("stored asset IR differs from database")
        if {item.path: (item.sha256, item.size_bytes, item.role) for item in asset_ir.files} != {
            item.path: (item.sha256, item.size_bytes, item.role)
            for item in manifest.files
            if item.path != manifest.asset_ir_path
        }:
            raise ValueError("asset IR index differs from manifest")
    except (OSError, ValueError, StopIteration) as error:
        raise AssetRuntimeResolutionError("ASSET_VERSION_STORED_IR_INVALID") from error
    version = root, manifest, asset_ir
    return version


# 功能：
#   只解析规范化索引明确声明的文件，不允许入口配置引用未索引内容。
# 输入：
#   root：当前资产版本根目录。
#   asset_ir：规范化文件索引。
#   relative：待解析的入口相对路径。
# 输出：
#   path：已核对内容摘要的入口文件。
def _declared_file(root: Path, asset_ir: AssetIR, relative: str) -> Path:
    declared = {entry.path: entry for entry in asset_ir.files}
    entry = declared.get(relative)
    if entry is None:
        raise AssetRuntimeResolutionError("ASSET_RUNTIME_FILE_UNDECLARED")
    path = _checked_file(root, relative, entry.sha256)
    return path


# 功能：
#   根据职责选择唯一文件，缺失或重复时拒绝用排列顺序决定执行内容。
# 输入：
#   root：当前资产版本根目录。
#   asset_ir：规范化文件索引。
#   role：语义、控制器等所需文件职责。
# 输出：
#   path：该职责唯一且摘要匹配的文件。
def _one_role(root: Path, asset_ir: AssetIR, role: str) -> Path:
    matches = [entry for entry in asset_ir.files if entry.role == role]
    if len(matches) != 1:
        raise AssetRuntimeResolutionError(f"ASSET_RUNTIME_{role.upper()}_AMBIGUOUS")
    path = _checked_file(root, matches[0].path, matches[0].sha256)
    return path


# 功能：
#   选择必须存在的唯一 JSON 契约，缺失或歧义时停止解析。
# 输入：
#   root：当前资产版本根目录。
#   asset_ir：规范化文件索引。
#   schema_version：所需契约的模式标识。
# 输出：
#   path：经过有界解析和摘要检查的契约文件。
def _json_contract(root: Path, asset_ir: AssetIR, schema_version: str) -> Path:
    path = _optional_json_contract(root, asset_ir, schema_version)
    if path is None:
        raise AssetRuntimeResolutionError("ASSET_RUNTIME_JSON_CONTRACT_AMBIGUOUS")
    return path


# 功能：
#   查找可选 JSON 契约；允许未提供，不允许重复声明或忽略损坏的 JSON 材料。
# 输入：
#   root：当前资产版本根目录。
#   asset_ir：规范化文件索引。
#   schema_version：需要查找的契约模式标识。
# 输出：
#   selection：唯一契约路径；未提供时为 None。
def _optional_json_contract(root: Path, asset_ir: AssetIR, schema_version: str) -> Path | None:
    matches: list[Path] = []
    for entry in asset_ir.files:
        if entry.role not in {"metadata", "source"} or not entry.path.endswith(".json"):
            continue
        path = _checked_file(root, entry.path, entry.sha256)
        try:
            payload = _read_json(path, entry.sha256)
        except (OSError, ValueError) as error:
            raise AssetRuntimeResolutionError("ASSET_RUNTIME_JSON_CONTRACT_INVALID") from error
        if isinstance(payload, dict) and payload.get("schema_version") == schema_version:
            matches.append(path)
    if len(matches) > 1:
        raise AssetRuntimeResolutionError("ASSET_RUNTIME_JSON_CONTRACT_AMBIGUOUS")
    selection = matches[0] if matches else None
    return selection


# 功能：
#   查找可选命名文件，并拒绝不同目录下的同名候选造成不确定选择。
# 输入：
#   root：当前资产版本根目录。
#   asset_ir：规范化文件索引。
#   filename：需要查找的文件名。
# 输出：
#   selection：匹配且摘要有效的路径；未提供时为 None。
def _optional_named_file(root: Path, asset_ir: AssetIR, filename: str) -> Path | None:
    matches = [entry for entry in asset_ir.files if Path(entry.path).name == filename]
    if len(matches) > 1:
        raise AssetRuntimeResolutionError("ASSET_RUNTIME_NAMED_FILE_AMBIGUOUS")
    selection = _checked_file(root, matches[0].path, matches[0].sha256) if matches else None
    return selection


# 功能：
#   优先解析唯一 Harmonic 入口；仅在资产明确声明 Classic 时保留该兼容入口。
# 输入：
#   root：当前资产版本根目录。
#   asset_ir：包含仿真目标的规范化信息。
# 输出：
#   entrypoint：内容摘要与索引匹配的仿真入口文件。
def _gazebo_entrypoint(root: Path, asset_ir: AssetIR) -> Path:
    targets = [
        target
        for target in asset_ir.simulation_targets
        if target.simulator in {"gazebo-harmonic", "gazebo-classic"}
    ]
    harmonic = [target for target in targets if target.simulator == "gazebo-harmonic"]
    candidates = harmonic or targets
    if len(candidates) != 1:
        raise AssetRuntimeResolutionError("ASSET_RUNTIME_GAZEBO_TARGET_AMBIGUOUS")
    entrypoint = _declared_file(root, asset_ir, candidates[0].entrypoint)
    return entrypoint


# 功能：
#   将已验收的地图版本解析为世界、语义和规划图文件，不使用目录名推测版本。
# 输入：
#   record：当前明确选择的地图或世界版本记录。
# 输出：
#   selection：绑定资产身份、内容摘要和执行文件的地图选择。
def resolve_versioned_map(record: dict[str, object]) -> ResolvedMapAsset:
    root, manifest, asset_ir = _validated_version(record, "map")
    selection = ResolvedMapAsset(
        asset_id=manifest.asset_id,
        content_sha256=manifest.content_sha256,
        root=root,
        world_sdf=_gazebo_entrypoint(root, asset_ir),
        semantic=_one_role(root, asset_ir, "semantic"),
        graph=_json_contract(root, asset_ir, "dronedream.map-graph.v1"),
        versioned=True,
    )
    return selection


# 功能：
#   将已验收飞机版本解析为模型、物理参数、控制器及可选负载文件。
# 输入：
#   record：当前明确选择的飞机版本记录。
# 输出：
#   selection：绑定资产身份、内容摘要和执行文件的飞机选择。
def resolve_versioned_vehicle(record: dict[str, object]) -> ResolvedVehicleAsset:
    root, manifest, asset_ir = _validated_version(record, "vehicle")
    selection = ResolvedVehicleAsset(
        asset_id=manifest.asset_id,
        content_sha256=manifest.content_sha256,
        root=root,
        vehicle_sdf=_gazebo_entrypoint(root, asset_ir),
        vehicle_metadata=_json_contract(root, asset_ir, "dronedream.vehicle.v1"),
        vehicle_summary=_optional_json_contract(
            root,
            asset_ir,
            "dronedream.my-drone-gazebo-artifact.v1",
        ),
        controller_params=_one_role(root, asset_ir, "controller"),
        payload_sdf=_optional_named_file(root, asset_ir, "takeout-payload.sdf"),
        versioned=True,
    )
    return selection


# 功能：
#   核对记录身份、成熟度和验收环境；此元数据判断不能替代使用前的文件检查。
# 输入：
#   record：待判断的资产版本记录。
#   environment_versions：本次使用的运行环境身份。
# 输出：
#   matches：记录与验收绑定、目标环境是否一致。
def qualification_matches_environment(
    record: dict[str, object], environment_versions: dict[str, str]
) -> bool:
    matches = False
    try:
        manifest = DDPkgManifest.model_validate(record["manifest"])
    except (KeyError, TypeError, ValueError):
        return matches
    matches = (
        record.get("asset_id") == manifest.asset_id
        and record.get("kind") == manifest.asset_kind
        and record.get("content_sha256") == manifest.content_sha256
        and record.get("maturity") == "qualified"
        and qualification_is_current(
            manifest.qualification,
            content_sha256=manifest.content_sha256,
            environment_versions=environment_versions,
        )
        and manifest.qualification is not None
        and manifest.qualification.maturity == "qualified"
    )
    return matches


# 功能：
#   1. 为显式候选仿真实验核对旧资产证据和同一 Gazebo/PX4/ROS 基础环境。
#   2. 仅允许应用 Runtime 资源修订改变，不将旧证据升级为新 Runtime 的正式资格。
# 输入：
#   record：内容绑定的资产记录；environment_versions：当前实际基础环境与资源摘要。
# 输出：
#   matches：是否仅存在允许试验的资源修订差异。
def trial_asset_environment_matches(record, environment_versions):
    try:
        manifest = DDPkgManifest.model_validate(record["manifest"])
        if manifest.qualification is None:
            return False
        previous = manifest.qualification.environment_versions
        required = {"runtime_manifest_sha256", "ros_distribution", "gazebo_sim", "px4_commit"}
        if set(previous) != required or set(environment_versions) != required:
            return False
        if not all(previous[key] == environment_versions[key]
                   for key in required - {"runtime_manifest_sha256"}):
            return False
        matches = qualification_matches_environment(record, previous)
    except (KeyError, TypeError, ValueError):
        matches = False
    return matches
