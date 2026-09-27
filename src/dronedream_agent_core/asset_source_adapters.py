"""Detect and normalize external simulation assets without executing their code.

This module is the narrow import boundary between modeling / robotics tools and
DroneDream's content-addressed ``.ddpkg`` format.  Built-in adapters only parse
declarative files. Native project formats which require Blender, CAD, cloud or
GIS runtimes are detected, but remain fail-closed until an isolated companion
adapter is installed by the host application.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import zipfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import uuid4
from xml.etree import ElementTree

from pydantic import Field, model_validator

from dronedream_plugin_sdk.protocol import decode_json

from .asset_packages import (
    _EXECUTABLE_SUFFIXES,
    MAX_MANIFEST_BYTES,
    MAX_MEMBER_UNCOMPRESSED_BYTES,
    MAX_PACKAGE_MEMBERS,
    MAX_PACKAGE_UNCOMPRESSED_BYTES,
    AssetFile,
    AssetIR,
    AssetKind,
    AssetPackageError,
    AssetSource,
    DDPkgManifest,
    GeometrySummary,
    InterfaceSummary,
    PhysicsSummary,
    ReadinessSummary,
    RuntimeExtension,
    SemanticSummary,
    SensorSummary,
    SimulationTarget,
    VehicleSummary,
    inspect_ddpkg,
    inspect_zip_members,
    normalized_member_path,
    package_content_sha256,
)
from .contracts import StrictModel
from .plugin_files import check_plain_plugin_path, hash_plugin_file, read_plugin_file
from .rmf_building_conversion import RmfBuildingConversionError, convert_rmf_building_map
from .xml_values import parse_xml
from .zip_index import validate_zip_index

SourceKind = Literal["map", "world", "vehicle"]
AdapterAvailability = Literal["builtin", "companion_required", "plugin_required"]

_MESH_SUFFIXES = {".dae", ".glb", ".gltf", ".obj", ".ply", ".stl"}
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tga", ".tif", ".tiff", ".webp"}
_MAX_XML_SOURCE_BYTES = 64 * 1024 * 1024
_MAX_EMBEDDED_SOURCE_BYTES = 256 * 1024 * 1024


class AssetSourceAdapterError(ValueError):
    """The source cannot be safely detected or normalized."""


class AssetSourceAdapterDescriptor(StrictModel):
    """Describe supported conversion boundaries without claiming an external tool is installed."""

    adapter_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", min_length=3, max_length=160)
    name: str = Field(min_length=1, max_length=160)
    version: str = Field(min_length=1, max_length=80)
    availability: AdapterAvailability
    source_formats: list[str] = Field(min_length=1, max_length=64)
    file_extensions: list[str] = Field(default_factory=list, max_length=64)
    asset_kinds: list[SourceKind] = Field(min_length=1, max_length=3)
    output_format: Literal["ddpkg"] = "ddpkg"
    execution_boundary: Literal[
        "declarative-parser",
        "isolated-local-companion",
        "isolated-plugin",
    ]
    required_application: str | None = Field(default=None, max_length=160)
    documentation_url: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_file_extensions(self) -> AssetSourceAdapterDescriptor:
        """Keep extension discovery deterministic and case-insensitive across hosts."""
        normalized = [value.casefold() for value in self.file_extensions]
        if any(not re.fullmatch(r"\.[a-z0-9][a-z0-9._-]{0,15}", value) for value in normalized):
            raise ValueError("source adapter file extensions must be normalized suffixes")
        if len(normalized) != len(set(normalized)) or normalized != self.file_extensions:
            raise ValueError("source adapter file extensions must be unique lowercase suffixes")
        return self


class AssetSourceDetection(StrictModel):
    """Format evidence, not permission to execute source code or skip asset qualification."""

    source_format: str = Field(min_length=1, max_length=80)
    adapter_id: str = Field(min_length=3, max_length=160)
    asset_kind: SourceKind | None = None
    confidence: Literal["exact", "strong", "extension"]
    can_normalize_locally: bool = Field(strict=True)
    required_inputs: list[str] = Field(default_factory=list, max_length=32)


_ADAPTERS = (
    AssetSourceAdapterDescriptor(
        adapter_id="dronedream.ddpkg",
        name="DroneDream Package",
        version="1.0.0",
        availability="builtin",
        source_formats=["ddpkg"],
        file_extensions=[".ddpkg"],
        asset_kinds=["map", "world", "vehicle"],
        execution_boundary="declarative-parser",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="dronedream.legacy-bundle",
        name="DroneDream Legacy Asset Bundle",
        version="1.0.0",
        availability="builtin",
        source_formats=["dronedream-legacy-bundle"],
        file_extensions=[".zip"],
        asset_kinds=["map", "vehicle"],
        execution_boundary="declarative-parser",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="gazebo.sdf",
        name="Gazebo SDF",
        version="1.0.0",
        availability="builtin",
        source_formats=["gazebo-sdf", "gazebo-world", "gazebo-model-bundle", "gazebo-world-bundle"],
        file_extensions=[".sdf", ".world", ".zip"],
        asset_kinds=["map", "world", "vehicle"],
        execution_boundary="declarative-parser",
        documentation_url="https://gazebosim.org/docs/latest/sdf_worlds/",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="ros2.urdf",
        name="ROS 2 URDF",
        version="1.0.0",
        availability="builtin",
        source_formats=["ros2-urdf", "ros2-urdf-package"],
        file_extensions=[".urdf", ".zip"],
        asset_kinds=["vehicle"],
        execution_boundary="declarative-parser",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="ros2.xacro",
        name="ROS 2 Xacro",
        version="1.0.0",
        availability="companion_required",
        source_formats=["ros2-xacro", "ros2-xacro-package"],
        file_extensions=[".xacro", ".zip"],
        asset_kinds=["vehicle"],
        execution_boundary="isolated-local-companion",
        required_application="DroneDreamRuntime / ROS 2 Xacro",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="open-rmf.building-map",
        name="Open-RMF Building Map",
        version="1.0.0",
        availability="builtin",
        source_formats=["rmf-building-map", "rmf-building-map-package"],
        file_extensions=[".yaml", ".zip"],
        asset_kinds=["map", "world"],
        execution_boundary="declarative-parser",
        required_application=None,
        documentation_url="https://github.com/open-rmf/rmf_traffic_editor",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="blender.phobos",
        name="Blender + Phobos",
        version="1.0.0",
        availability="companion_required",
        source_formats=["blender-blend", "phobos-smurf", "phobos-smurf-package"],
        file_extensions=[".blend", ".smurf", ".zip"],
        asset_kinds=["map", "world", "vehicle"],
        execution_boundary="isolated-local-companion",
        required_application="Blender + Phobos",
        documentation_url="https://github.com/dfki-ric/phobos",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="solidworks.urdf",
        name="SolidWorks",
        version="1.0.0",
        availability="plugin_required",
        source_formats=["solidworks-native", "cad-step"],
        file_extensions=[".sldasm", ".sldprt", ".step", ".stp"],
        asset_kinds=["vehicle"],
        execution_boundary="isolated-plugin",
        required_application="SolidWorks",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="autodesk.fusion",
        name="Autodesk Fusion",
        version="1.0.0",
        availability="plugin_required",
        source_formats=["fusion-native", "cad-step"],
        file_extensions=[".f3d", ".f3z", ".step", ".stp"],
        asset_kinds=["map", "vehicle"],
        execution_boundary="isolated-plugin",
        required_application="Autodesk Fusion",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="onshape.translation",
        name="Onshape",
        version="1.0.0",
        availability="plugin_required",
        source_formats=["onshape-document", "cad-step"],
        asset_kinds=["map", "vehicle"],
        execution_boundary="isolated-plugin",
        required_application="Onshape API",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="freecad.robotics",
        name="FreeCAD",
        version="1.0.0",
        availability="plugin_required",
        source_formats=["freecad-native", "cad-step"],
        file_extensions=[".fcstd", ".step", ".stp"],
        asset_kinds=["map", "vehicle"],
        execution_boundary="isolated-plugin",
        required_application="FreeCAD",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="autodesk.maya",
        name="Autodesk Maya",
        version="1.0.0",
        availability="plugin_required",
        source_formats=["maya-native"],
        file_extensions=[".ma", ".mb"],
        asset_kinds=["map", "world", "vehicle"],
        execution_boundary="isolated-plugin",
        required_application="Autodesk Maya",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="autodesk.3dsmax",
        name="Autodesk 3ds Max",
        version="1.0.0",
        availability="plugin_required",
        source_formats=["3dsmax-native"],
        file_extensions=[".max"],
        asset_kinds=["map", "world", "vehicle"],
        execution_boundary="isolated-plugin",
        required_application="Autodesk 3ds Max",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="autodesk.revit",
        name="Autodesk Revit",
        version="1.0.0",
        availability="plugin_required",
        source_formats=["revit-native", "ifc-bim"],
        file_extensions=[".rvt", ".ifc"],
        asset_kinds=["map", "world"],
        execution_boundary="isolated-plugin",
        required_application="Autodesk Revit or an IFC toolchain",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="trimble.sketchup",
        name="SketchUp",
        version="1.0.0",
        availability="plugin_required",
        source_formats=["sketchup-native"],
        file_extensions=[".skp"],
        asset_kinds=["map", "world"],
        execution_boundary="isolated-plugin",
        required_application="SketchUp",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="nvidia.usd",
        name="OpenUSD / NVIDIA Omniverse",
        version="1.0.0",
        availability="plugin_required",
        source_formats=["openusd", "isaac-sim-usd"],
        file_extensions=[".usd", ".usda", ".usdc"],
        asset_kinds=["map", "world", "vehicle"],
        execution_boundary="isolated-plugin",
        required_application="OpenUSD or NVIDIA Omniverse / Isaac Sim",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="unity.simulation",
        name="Unity",
        version="1.0.0",
        availability="plugin_required",
        source_formats=["unity-package", "unity-project"],
        file_extensions=[".unitypackage"],
        asset_kinds=["map", "world", "vehicle"],
        execution_boundary="isolated-plugin",
        required_application="Unity Editor",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="unreal.robotics",
        name="Unreal Engine",
        version="1.0.0",
        availability="plugin_required",
        source_formats=["unreal-project"],
        file_extensions=[".uproject"],
        asset_kinds=["map", "world", "vehicle"],
        execution_boundary="isolated-plugin",
        required_application="Unreal Engine",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="mesh.interchange",
        name="3D Mesh Interchange",
        version="1.0.0",
        availability="plugin_required",
        source_formats=["fbx", "gltf", "obj", "collada", "ply", "stl"],
        file_extensions=[".fbx", ".gltf", ".glb", ".obj", ".dae", ".ply", ".stl"],
        asset_kinds=["map", "world", "vehicle"],
        execution_boundary="isolated-plugin",
        required_application=("A geometry, semantics, material, and physics conversion plugin"),
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="openscad.robotics",
        name="OpenSCAD",
        version="1.0.0",
        availability="plugin_required",
        source_formats=["openscad-native"],
        file_extensions=[".scad"],
        asset_kinds=["map", "vehicle"],
        execution_boundary="isolated-plugin",
        required_application="OpenSCAD",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="gazebo.fuel",
        name="Gazebo Fuel",
        version="1.0.0",
        availability="companion_required",
        source_formats=["gazebo-fuel-uri"],
        asset_kinds=["map", "world", "vehicle"],
        execution_boundary="isolated-local-companion",
        required_application="Gazebo Fuel Tools",
        documentation_url="https://gazebosim.org/libs/fuel_tools/",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="px4.gazebo-models",
        name="PX4 Gazebo Models",
        version="1.0.0",
        availability="companion_required",
        source_formats=["px4-gazebo-model", "px4-gazebo-world"],
        asset_kinds=["world", "vehicle"],
        execution_boundary="isolated-local-companion",
        required_application="PX4 SITL + Gazebo",
        documentation_url="https://docs.px4.io/main/en/sim_gazebo_gz/",
    ),
    AssetSourceAdapterDescriptor(
        adapter_id="gis.geospatial",
        name="GIS / DEM",
        version="1.0.0",
        availability="plugin_required",
        source_formats=["geotiff", "citygml", "osm", "las-laz"],
        file_extensions=[".tif", ".tiff", ".osm", ".gml", ".las", ".laz"],
        asset_kinds=["map", "world"],
        execution_boundary="isolated-plugin",
        required_application="GDAL-compatible GIS toolchain",
    ),
)


def source_adapter_catalog() -> list[dict[str, object]]:
    """Return detached presentation values, not mutable access to the adapter registry."""
    return [entry.model_dump(mode="json") for entry in _ADAPTERS]


def _sha256(path: Path) -> str:
    """Hash bounded stable source bytes without loading a multi-gigabyte asset into RAM."""
    return hash_plugin_file(path, limit=MAX_PACKAGE_UNCOMPRESSED_BYTES)


def _local_name(tag: str) -> str:
    """Compare declarative tag names independently of an XML namespace prefix."""
    return tag.rsplit("}", 1)[-1]


def _xml_detection(source: Path) -> AssetSourceDetection | None:
    """Recognize XML without interpreting macros, entities or embedded executable plugins."""
    with source.open("rb") as handle:
        prefix = handle.read(256).lstrip()
    if not prefix.startswith(
        (b"<", b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff", b"\x00<", b"\x00\x00\xfe\xff")
    ):
        return None
    if source.stat().st_size > _MAX_XML_SOURCE_BYTES:
        if source.suffix.casefold() not in {".sdf", ".world", ".urdf", ".xacro"}:
            # Large native XML geometry belongs to its external converter, not
            # the bounded local SDF/URDF parser.
            return None
        raise AssetSourceAdapterError("ASSET_SOURCE_XML_SIZE_EXCEEDED")
    try:
        root = _parse_source_xml(read_plugin_file(source, limit=_MAX_XML_SOURCE_BYTES))
    except (ElementTree.ParseError, OSError):
        return None
    local = _local_name(root.tag)
    if local == "robot":
        # Renaming a macro-bearing robot to .urdf does not make it declarative.
        xacro = source.suffix.casefold() == ".xacro" or any(
            "xacro" in element.tag.casefold()
            or any("${" in value or "$(" in value for value in element.attrib.values())
            for element in root.iter()
        )
        return AssetSourceDetection(
            source_format="ros2-xacro" if xacro else "ros2-urdf",
            adapter_id="ros2.xacro" if xacro else "ros2.urdf",
            asset_kind="vehicle",
            confidence="exact",
            can_normalize_locally=not xacro,
            required_inputs=["isolated_xacro_expansion"] if xacro else [],
        )
    if local != "sdf":
        return None
    children = {_local_name(child.tag) for child in root}
    if "world" in children:
        return AssetSourceDetection(
            source_format="gazebo-world",
            adapter_id="gazebo.sdf",
            asset_kind="world",
            confidence="exact",
            can_normalize_locally=True,
        )
    if "model" in children:
        return AssetSourceDetection(
            source_format="gazebo-sdf",
            adapter_id="gazebo.sdf",
            asset_kind="vehicle",
            confidence="exact",
            can_normalize_locally=True,
        )
    return None


def _validated_zip_names(source: Path) -> set[str]:
    """Bound the ZIP index before allocation and share the package's portable member rules."""
    try:
        check_plain_plugin_path(source)
        with source.open("rb") as handle:
            validate_zip_index(handle, maximum_members=MAX_PACKAGE_MEMBERS)
            with zipfile.ZipFile(handle) as bundle:
                # Detection may identify an inert companion project containing
                # code; built-in normalization independently forbids those files.
                return set(inspect_zip_members(bundle, allow_executables=True))
    except zipfile.BadZipFile as error:
        raise AssetSourceAdapterError("ASSET_SOURCE_INVALID_ZIP") from error
    except AssetPackageError as error:
        raise AssetSourceAdapterError(str(error).replace("DDPKG_", "ASSET_SOURCE_", 1)) from error
    except ValueError as error:
        raise AssetSourceAdapterError("ASSET_SOURCE_INDEX_INVALID") from error


def _zip_detection(source: Path) -> AssetSourceDetection:
    """Recognize declarations and conventional entrypoints without extracting an archive."""
    names = _validated_zip_names(source)
    lowered = {name.casefold(): name for name in names}
    if "manifest.json" in lowered:
        try:
            with zipfile.ZipFile(source) as bundle:
                manifest = _read_manifest(bundle, lowered["manifest.json"])
            if (
                isinstance(manifest, dict)
                and manifest.get("schema_version") == "dronedream.ddpkg.v1"
            ):
                kind = manifest.get("asset_kind")
                return AssetSourceDetection(
                    source_format="ddpkg",
                    adapter_id="dronedream.ddpkg",
                    asset_kind=kind
                    if isinstance(kind, str) and kind in {"map", "world", "vehicle"}
                    else None,
                    confidence="exact",
                    can_normalize_locally=True,
                )
            if (
                isinstance(manifest, dict)
                and manifest.get("schema_version") == "dronedream.asset-bundle.v1"
            ):
                kind = manifest.get("kind")
                return AssetSourceDetection(
                    source_format="dronedream-legacy-bundle",
                    adapter_id="dronedream.legacy-bundle",
                    asset_kind=kind
                    if isinstance(kind, str) and kind in {"map", "vehicle"}
                    else None,
                    confidence="exact",
                    can_normalize_locally=True,
                )
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
    suffixes = {PurePosixPath(name).suffix.casefold() for name in names}
    if any(name.casefold().endswith(".building.yaml") for name in names):
        return AssetSourceDetection(
            source_format="rmf-building-map-package",
            adapter_id="open-rmf.building-map",
            asset_kind="map",
            confidence="exact",
            can_normalize_locally=True,
            required_inputs=[],
        )
    if any(name.endswith("model.config") for name in lowered) and any(
        name.endswith("model.sdf") for name in lowered
    ):
        return AssetSourceDetection(
            source_format="gazebo-model-bundle",
            adapter_id="gazebo.sdf",
            asset_kind="vehicle",
            confidence="strong",
            can_normalize_locally=True,
        )
    if any(name.endswith(".world") or name.endswith("world.sdf") for name in lowered):
        return AssetSourceDetection(
            source_format="gazebo-world-bundle",
            adapter_id="gazebo.sdf",
            asset_kind="world",
            confidence="strong",
            can_normalize_locally=True,
        )
    if ".smurf" in suffixes:
        return AssetSourceDetection(
            source_format="phobos-smurf-package",
            adapter_id="blender.phobos",
            asset_kind="vehicle",
            confidence="strong",
            can_normalize_locally=False,
            required_inputs=["blender_phobos_companion"],
        )
    if ".xacro" in suffixes:
        return AssetSourceDetection(
            source_format="ros2-xacro-package",
            adapter_id="ros2.xacro",
            asset_kind="vehicle",
            confidence="strong",
            can_normalize_locally=False,
            required_inputs=["isolated_xacro_expansion"],
        )
    if ".urdf" in suffixes:
        return AssetSourceDetection(
            source_format="ros2-urdf-package",
            adapter_id="ros2.urdf",
            asset_kind="vehicle",
            confidence="strong",
            can_normalize_locally=True,
        )
    raise AssetSourceAdapterError("ASSET_SOURCE_FORMAT_UNRECOGNIZED")


def detect_asset_source(source: Path, declared_format: str = "auto") -> AssetSourceDetection:
    """Prefer actual format evidence; a native extension only selects a still-disabled converter."""
    check_plain_plugin_path(source)
    if not isinstance(declared_format, str):
        raise AssetSourceAdapterError("ASSET_SOURCE_FORMAT_MISMATCH")
    if not source.is_file() or source.stat().st_size <= 0:
        raise AssetSourceAdapterError("ASSET_SOURCE_FILE_INVALID")
    if source.stat().st_size > MAX_PACKAGE_UNCOMPRESSED_BYTES:
        raise AssetSourceAdapterError("ASSET_SOURCE_TOTAL_SIZE_EXCEEDED")
    if declared_format == "ddpkg" and not zipfile.is_zipfile(source):
        raise AssetSourceAdapterError("DDPKG_NOT_ZIP")
    detected = None
    is_archive = zipfile.is_zipfile(source)
    if is_archive:
        try:
            detected = _zip_detection(source)
        except AssetSourceAdapterError as error:
            if str(error) != "ASSET_SOURCE_FORMAT_UNRECOGNIZED":
                raise
            # CAD native projects often are ZIP containers. Unknown contents
            # may still select their external converter by extension below.
    if detected is None:
        detected = None if is_archive else _xml_detection(source)
        if detected is None:
            suffix = source.suffix.casefold()
            if source.name.casefold().endswith(".building.yaml"):
                detected = AssetSourceDetection(
                    source_format="rmf-building-map",
                    adapter_id="open-rmf.building-map",
                    asset_kind="map",
                    confidence="exact",
                    can_normalize_locally=True,
                    required_inputs=[],
                )
            if detected is not None:
                suffix = ""
            native: dict[str, tuple[str, str, SourceKind | None, str]] = {
                ".blend": ("blender-blend", "blender.phobos", None, "blender_phobos_companion"),
                ".smurf": ("phobos-smurf", "blender.phobos", "vehicle", "blender_phobos_companion"),
                ".sldasm": (
                    "solidworks-native",
                    "solidworks.urdf",
                    "vehicle",
                    "solidworks_connector",
                ),
                ".sldprt": (
                    "solidworks-native",
                    "solidworks.urdf",
                    "vehicle",
                    "solidworks_connector",
                ),
                ".f3d": ("fusion-native", "autodesk.fusion", None, "fusion_connector"),
                ".f3z": ("fusion-native", "autodesk.fusion", None, "fusion_connector"),
                ".fcstd": ("freecad-native", "freecad.robotics", None, "freecad_connector"),
                ".ma": ("maya-native", "autodesk.maya", None, "maya_connector"),
                ".mb": ("maya-native", "autodesk.maya", None, "maya_connector"),
                ".max": ("3dsmax-native", "autodesk.3dsmax", None, "3dsmax_connector"),
                ".rvt": ("revit-native", "autodesk.revit", "map", "revit_connector"),
                ".ifc": ("ifc-bim", "autodesk.revit", "map", "bim_semantics_and_physics"),
                ".skp": ("sketchup-native", "trimble.sketchup", "map", "sketchup_connector"),
                ".usd": ("openusd", "nvidia.usd", None, "openusd_semantics_and_physics"),
                ".usda": ("openusd", "nvidia.usd", None, "openusd_semantics_and_physics"),
                ".usdc": ("openusd", "nvidia.usd", None, "openusd_semantics_and_physics"),
                ".unitypackage": (
                    "unity-package",
                    "unity.simulation",
                    None,
                    "unity_connector",
                ),
                ".uproject": (
                    "unreal-project",
                    "unreal.robotics",
                    None,
                    "unreal_connector",
                ),
                ".fbx": ("fbx", "mesh.interchange", None, "mesh_semantics_and_physics"),
                ".gltf": ("gltf", "mesh.interchange", None, "mesh_semantics_and_physics"),
                ".glb": ("gltf", "mesh.interchange", None, "mesh_semantics_and_physics"),
                ".obj": ("obj", "mesh.interchange", None, "mesh_semantics_and_physics"),
                ".dae": ("collada", "mesh.interchange", None, "mesh_semantics_and_physics"),
                ".ply": ("ply", "mesh.interchange", None, "mesh_semantics_and_physics"),
                ".stl": ("stl", "mesh.interchange", None, "mesh_semantics_and_physics"),
                ".scad": (
                    "openscad-native",
                    "openscad.robotics",
                    None,
                    "openscad_connector",
                ),
                ".step": ("cad-step", "freecad.robotics", None, "cad_semantics_and_physics"),
                ".stp": ("cad-step", "freecad.robotics", None, "cad_semantics_and_physics"),
                ".tif": ("geotiff", "gis.geospatial", "map", "gis_connector"),
                ".tiff": ("geotiff", "gis.geospatial", "map", "gis_connector"),
                ".osm": ("osm", "gis.geospatial", "map", "gis_connector"),
                ".gml": ("citygml", "gis.geospatial", "map", "gis_connector"),
                ".las": ("las-laz", "gis.geospatial", "map", "gis_connector"),
                ".laz": ("las-laz", "gis.geospatial", "map", "gis_connector"),
            }
            if detected is None:
                value = native.get(suffix)
                if value is None:
                    raise AssetSourceAdapterError("ASSET_SOURCE_FORMAT_UNRECOGNIZED")
                source_format, adapter_id, kind, required = value
                detected = AssetSourceDetection(
                    source_format=source_format,
                    adapter_id=adapter_id,
                    asset_kind=kind,
                    confidence="extension",
                    can_normalize_locally=False,
                    required_inputs=[required],
                )
    if declared_format not in {"", "auto", detected.source_format}:
        raise AssetSourceAdapterError("ASSET_SOURCE_FORMAT_MISMATCH")
    return detected


def _slug(value: str, fallback: str) -> str:
    """Create a portable display-derived identity, not a fallback map or coordinate binding."""
    result = re.sub(r"[^a-z0-9._-]+", "-", value.casefold()).strip("-._")
    if len(result) < 3:
        prefix = re.sub(r"[^a-z0-9._-]+", "-", fallback.casefold()).strip("-._") or "asset"
        # Reserve the digest suffix even when a caller supplies a long fallback name.
        result = f"{prefix[:147]}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}"
    return result[:160]


def _role(
    path: str,
) -> Literal[
    "source",
    "visual",
    "collision",
    "sdf",
    "urdf",
    "xacro",
    "controller",
    "semantic",
    "thumbnail",
    "evidence",
    "license",
    "metadata",
]:
    """Assign a presentation/inspection role; no role may bypass executable XML checks."""
    lowered = path.casefold()
    suffix = PurePosixPath(path).suffix.casefold()
    if suffix in {".sdf", ".world"}:
        return "sdf"
    if suffix == ".urdf":
        return "urdf"
    if suffix == ".xacro":
        return "xacro"
    if suffix in _MESH_SUFFIXES:
        return "collision" if "collision" in lowered else "visual"
    if suffix in _IMAGE_SUFFIXES:
        return "thumbnail" if "thumbnail" in lowered or "preview" in lowered else "visual"
    if suffix in {".json", ".yaml", ".yml"} and any(
        token in lowered for token in ("control", "motor", "actuator", "parameter")
    ):
        return "controller"
    if "semantic" in lowered:
        return "semantic"
    if suffix == ".json" and ("qualification" in lowered or "evidence" in lowered):
        return "evidence"
    if "license" in lowered or "licence" in lowered:
        return "license"
    return "metadata"


def _runtime_extensions(xml_payload: bytes, source_path: str) -> list[RuntimeExtension]:
    """Record every discovered plugin as blocked, including raw-source copies."""
    root = _parse_source_xml(xml_payload)
    return [
        RuntimeExtension(
            name=element.attrib.get("name") or "unnamed-plugin",
            filename=element.attrib.get("filename"),
            source_path=source_path,
        )
        for element in root.iter()
        if _local_name(element.tag) == "plugin"
    ]


def _root_identity(xml_payload: bytes, fallback: str) -> tuple[str, str]:
    """Use the declared world/model/robot name; filename fallback supplies no physics."""
    root = _parse_source_xml(xml_payload)
    candidates = [root, *list(root)]
    name = next(
        (item.attrib.get("name") for item in candidates if item.attrib.get("name")), fallback
    )
    return _slug(name, fallback), name


def _xml_child_text(element: ElementTree.Element, name: str) -> str | None:
    """Read a direct child, not an unrelated nested plugin's similarly named setting."""
    for child in element:
        if _local_name(child.tag) == name and child.text and child.text.strip():
            return child.text.strip()
    return None


def _structured_summaries(
    xml_payloads: list[tuple[str, bytes]],
    files: list[AssetFile],
    *,
    kind: AssetKind,
    targets: list[SimulationTarget],
    capabilities: list[str],
) -> tuple[
    GeometrySummary,
    PhysicsSummary,
    VehicleSummary | None,
    list[SensorSummary],
    SemanticSummary,
    InterfaceSummary,
    ReadinessSummary,
]:
    """Summarize declared structure; each link must carry its own physics fields.

    These counts are not dynamics validation or sensor freshness evidence. No
    physics completeness is inferred from extra fields on some other link.
    """
    links = 0
    joints = 0
    visuals = 0
    collisions = 0
    masses = 0
    inertias = 0
    explicit_actuators = 0
    explicit_rotors = 0
    frames: set[str] = set()
    materials: set[str] = set()
    sensors: list[SensorSummary] = []
    per_link_coverage: list[tuple[bool, bool, bool]] = []
    for source_path, payload in xml_payloads:
        root = _parse_source_xml(payload)
        for element in root.iter():
            local = _local_name(element.tag)
            if local == "link":
                links += 1
                inertials = [child for child in element if _local_name(child.tag) == "inertial"]
                inertial_fields = {
                    _local_name(child.tag) for inertial in inertials for child in inertial
                }
                per_link_coverage.append(
                    (
                        any(_local_name(child.tag) == "collision" for child in element),
                        "mass" in inertial_fields,
                        "inertia" in inertial_fields,
                    )
                )
                if element.attrib.get("name"):
                    frames.add(element.attrib["name"])
            elif local == "joint":
                joints += 1
            elif local == "visual":
                visuals += 1
            elif local == "collision":
                collisions += 1
            elif local == "mass" and (element.attrib.get("value") or (element.text or "").strip()):
                masses += 1
            elif local == "inertia":
                inertias += 1
            elif local == "actuator":
                explicit_actuators += 1
            elif local == "rotor":
                explicit_rotors += 1
            elif local == "material":
                material_name = element.attrib.get("name")
                if material_name:
                    materials.add(material_name)
            elif local == "sensor":
                update_rate_text = _xml_child_text(element, "update_rate")
                try:
                    update_rate = float(update_rate_text) if update_rate_text is not None else None
                    if update_rate is not None and (
                        not math.isfinite(update_rate) or not 0 < update_rate <= 1_000_000
                    ):
                        raise ValueError("invalid rate")
                except ValueError as error:
                    raise AssetSourceAdapterError("ASSET_SOURCE_SENSOR_RATE_INVALID") from error
                sensors.append(
                    SensorSummary(
                        kind=element.attrib.get("type") or element.attrib.get("name") or "unknown",
                        frame_id=element.attrib.get("name"),
                        source_path=source_path,
                        update_rate_hz=update_rate,
                    )
                )
    visual_paths = sorted({entry.path for entry in files if entry.role == "visual"})
    collision_paths = sorted({entry.path for entry in files if entry.role == "collision"})
    geometry = GeometrySummary(
        visual_paths=visual_paths,
        collision_paths=collision_paths,
        coordinate_frames=sorted(frames),
        visual_count=visuals or len(visual_paths),
        collision_count=collisions or len(collision_paths),
    )
    physics = PhysicsSummary(
        link_count=links,
        joint_count=joints,
        mass_entry_count=masses,
        inertia_entry_count=inertias,
        collision_complete=bool(per_link_coverage) and all(row[0] for row in per_link_coverage),
        mass_complete=bool(per_link_coverage) and all(row[1] for row in per_link_coverage),
        inertia_complete=bool(per_link_coverage) and all(row[2] for row in per_link_coverage),
        contact_materials=sorted(materials),
    )
    px4_profiles = sorted(
        capability.split("=", 1)[1]
        for capability in capabilities
        if capability.startswith("px4.sitl-model=")
    )
    vehicle = (
        VehicleSummary(
            actuator_count=explicit_actuators or None,
            rotor_count=explicit_rotors or None,
            autopilot_profile=px4_profiles[0] if len(px4_profiles) == 1 else None,
        )
        if kind == "vehicle"
        else None
    )
    interfaces = InterfaceSummary(
        gazebo_entrypoints=sorted(
            target.entrypoint for target in targets if target.simulator.startswith("gazebo")
        ),
        ros2_entrypoints=sorted(entry.path for entry in files if entry.role in {"urdf", "xacro"}),
        px4_profiles=px4_profiles,
    )
    semantic_paths = sorted(entry.path for entry in files if entry.role == "semantic")
    semantics = SemanticSummary(
        navigation_graph_path=next(
            (path for path in semantic_paths if "graph" in path.casefold()), None
        ),
        traversability_path=next(
            (path for path in semantic_paths if "travers" in path.casefold()), None
        ),
    )
    missing: list[str] = []
    if not geometry.collision_count:
        missing.append("geometry.collision")
    if not physics.mass_complete:
        missing.append("physics.mass")
    if not physics.inertia_complete:
        missing.append("physics.inertia")
    if kind == "vehicle" and not px4_profiles:
        missing.append("interfaces.px4_profile")
    if kind in {"map", "world"} and not semantic_paths:
        missing.append("semantics.navigation")
    readiness = ReadinessSummary(
        maturity_ceiling=(
            "physics_ready"
            if physics.collision_complete and physics.mass_complete and physics.inertia_complete
            else "visual_only"
        ),
        missing_fields=missing,
    )
    return geometry, physics, vehicle, sensors, semantics, interfaces, readiness


def _stream_zip_member(
    source_bundle: zipfile.ZipFile,
    source_name: str,
    output_bundle: zipfile.ZipFile,
    output_name: str,
) -> AssetFile:
    """Hash exactly the bounded bytes copied to a new member, not a separate earlier read."""
    digest = hashlib.sha256()
    size = 0
    maximum = min(source_bundle.getinfo(source_name).file_size, MAX_MEMBER_UNCOMPRESSED_BYTES)
    with source_bundle.open(source_name) as source, output_bundle.open(output_name, "w") as output:
        while chunk := source.read(min(1024 * 1024, maximum - size + 1)):
            size += len(chunk)
            if size > maximum:
                raise AssetSourceAdapterError("ASSET_SOURCE_MEMBER_SIZE_EXCEEDED")
            digest.update(chunk)
            output.write(chunk)
    return AssetFile(
        path=output_name,
        role=_role(output_name),
        media_type="application/octet-stream",
        sha256=digest.hexdigest(),
        size_bytes=size,
    )


def _store_source_snapshot(
    source: Path, output_bundle: zipfile.ZipFile, *, source_sha256: str
) -> AssetFile:
    """Embed a verified source copy or an explicit reference; never mislabel the latter as bytes."""
    if source.stat().st_size > _MAX_EMBEDDED_SOURCE_BYTES:
        reference = (
            json.dumps(
                {
                    "schema_version": "dronedream.asset-source-reference.v1",
                    "display_name": source.name,
                    "size_bytes": source.stat().st_size,
                    "sha256": source_sha256,
                    "embedded": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        source_name = "source/source-reference.json"
        output_bundle.writestr(source_name, reference)
        return AssetFile(
            path=source_name,
            role="source_reference",
            media_type="application/json",
            sha256=hashlib.sha256(reference).hexdigest(),
            size_bytes=len(reference),
        )
    suffix = source.suffix.casefold()
    source_name = normalized_member_path(
        f"source/{_slug(source.stem, 'source')}{suffix if len(suffix) <= 16 else '.bin'}"
    )
    # Stream/hashes cover the same copied bytes. A later source hash is checked
    # again before publication to bind all normalization passes to this input.
    digest = hashlib.sha256()
    size = 0
    info = zipfile.ZipInfo(source_name)
    info.compress_type = zipfile.ZIP_STORED
    with source.open("rb") as raw, output_bundle.open(info, "w") as target:
        while chunk := raw.read(min(1024 * 1024, _MAX_EMBEDDED_SOURCE_BYTES - size + 1)):
            size += len(chunk)
            if size > _MAX_EMBEDDED_SOURCE_BYTES:
                raise AssetSourceAdapterError("ASSET_SOURCE_CHANGED")
            digest.update(chunk)
            target.write(chunk)
    if digest.hexdigest() != source_sha256:
        raise AssetSourceAdapterError("ASSET_SOURCE_CHANGED")
    return AssetFile(
        path=source_name,
        role="source",
        media_type="application/octet-stream",
        sha256=source_sha256,
        size_bytes=size,
    )


def _store_source_reference(
    source: Path, output_bundle: zipfile.ZipFile, *, source_sha256: str
) -> AssetFile:
    """Retain immutable provenance without embedding the complete upstream archive."""

    reference = (
        json.dumps(
            {
                "schema_version": "dronedream.asset-source-reference.v1",
                "display_name": source.name,
                "size_bytes": source.stat().st_size,
                "sha256": source_sha256,
                "embedded": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    source_name = "source/source-reference.json"
    output_bundle.writestr(source_name, reference)
    return AssetFile(
        path=source_name,
        role="source_reference",
        media_type="application/json",
        sha256=hashlib.sha256(reference).hexdigest(),
        size_bytes=len(reference),
    )


def _normalization_inputs(
    source: Path,
    detection: AssetSourceDetection,
    output_bundle: zipfile.ZipFile,
) -> tuple[list[AssetFile], str, bytes, list[tuple[str, bytes]]]:
    """Copy declarative files; ambiguous entrypoints require an explicit selection."""
    if detection.adapter_id == "open-rmf.building-map":
        files: list[AssetFile] = []
        if zipfile.is_zipfile(source):
            names = _validated_zip_names(source)
            yaml_names = sorted(
                name for name in names if name.casefold().endswith(".building.yaml")
            )
            if len(yaml_names) != 1:
                raise AssetSourceAdapterError(
                    "ASSET_SOURCE_ENTRYPOINT_MISSING"
                    if not yaml_names
                    else "ASSET_SOURCE_ENTRYPOINT_AMBIGUOUS"
                )
            if any(PurePosixPath(name).suffix.casefold() in _EXECUTABLE_SUFFIXES for name in names):
                raise AssetSourceAdapterError("ASSET_SOURCE_EXECUTABLE_MEMBER_FORBIDDEN")
            with zipfile.ZipFile(source) as input_bundle:
                yaml_name = yaml_names[0]
                info = input_bundle.getinfo(yaml_name)
                if info.file_size > _MAX_XML_SOURCE_BYTES:
                    raise AssetSourceAdapterError("ASSET_SOURCE_XML_SIZE_EXCEEDED")
                yaml_payload = input_bundle.read(yaml_name)
                for name in sorted(names):
                    output_name = normalized_member_path(f"normalized/rmf/source/{name}")
                    files.append(_stream_zip_member(input_bundle, name, output_bundle, output_name))
        else:
            yaml_name = source.name
            yaml_payload = read_plugin_file(source, limit=_MAX_XML_SOURCE_BYTES)
            output_name = normalized_member_path(f"normalized/rmf/source/{source.name}")
            output_bundle.writestr(output_name, yaml_payload)
            files.append(
                AssetFile(
                    path=output_name,
                    role="metadata",
                    media_type="application/yaml",
                    sha256=hashlib.sha256(yaml_payload).hexdigest(),
                    size_bytes=len(yaml_payload),
                )
            )
        try:
            conversion = convert_rmf_building_map(yaml_payload, source_name=yaml_name)
        except RmfBuildingConversionError as error:
            raise AssetSourceAdapterError(str(error)) from error
        generated = (
            ("normalized/rmf/generated/world.sdf", "sdf", "application/xml", conversion.sdf),
            (
                "normalized/rmf/generated/map-semantic.json",
                "semantic",
                "application/json",
                conversion.semantic,
            ),
            (
                "normalized/rmf/generated/navigation-semantic-graph.json",
                "metadata",
                "application/json",
                conversion.topology,
            ),
            (
                "normalized/rmf/generated/conversion-evidence.json",
                "evidence",
                "application/json",
                conversion.report,
            ),
        )
        for path, role, media_type, content in generated:
            output_bundle.writestr(path, content)
            files.append(
                AssetFile(
                    path=path,
                    role=role,
                    media_type=media_type,
                    sha256=hashlib.sha256(content).hexdigest(),
                    size_bytes=len(content),
                )
            )
        entrypoint = "normalized/rmf/generated/world.sdf"
        return files, entrypoint, conversion.sdf, [(entrypoint, conversion.sdf)]
    if zipfile.is_zipfile(source):
        names = _validated_zip_names(source)
        if any(PurePosixPath(name).suffix.casefold() in _EXECUTABLE_SUFFIXES for name in names):
            raise AssetSourceAdapterError("ASSET_SOURCE_EXECUTABLE_MEMBER_FORBIDDEN")
        if detection.source_format == "dronedream-legacy-bundle":
            with zipfile.ZipFile(source) as input_bundle:
                try:
                    manifest = _read_manifest(input_bundle, "manifest.json")
                except (KeyError, TypeError, json.JSONDecodeError) as error:
                    raise AssetSourceAdapterError("ASSET_SOURCE_LEGACY_MANIFEST_INVALID") from error
                if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
                    raise AssetSourceAdapterError("ASSET_SOURCE_LEGACY_MANIFEST_INVALID")
                entrypoint_key = "world_sdf" if manifest.get("kind") == "map" else "vehicle_sdf"
                entrypoint_value = manifest["files"].get(entrypoint_key)
                if not isinstance(entrypoint_value, str):
                    raise AssetSourceAdapterError("ASSET_SOURCE_ENTRYPOINT_MISSING")
                try:
                    entrypoint_source = normalized_member_path(entrypoint_value)
                except AssetPackageError as error:
                    raise AssetSourceAdapterError("ASSET_SOURCE_ENTRYPOINT_INVALID") from error
                if entrypoint_source not in names:
                    raise AssetSourceAdapterError("ASSET_SOURCE_ENTRYPOINT_MISSING")
                files: list[AssetFile] = []
                entrypoint_payload = b""
                xml_payloads: list[tuple[str, bytes]] = []
                for name in sorted(names):
                    output_name = normalized_member_path(f"normalized/legacy/{name}")
                    files.append(_stream_zip_member(input_bundle, name, output_bundle, output_name))
                    if PurePosixPath(name).suffix.casefold() in {
                        ".sdf",
                        ".world",
                        ".urdf",
                        ".xacro",
                    }:
                        info = input_bundle.getinfo(name)
                        if info.file_size > _MAX_XML_SOURCE_BYTES:
                            raise AssetSourceAdapterError("ASSET_SOURCE_XML_SIZE_EXCEEDED")
                        xml_payloads.append((output_name, input_bundle.read(name)))
                    if name == entrypoint_source:
                        entrypoint_payload = input_bundle.read(name)
                return (
                    files,
                    f"normalized/legacy/{entrypoint_source}",
                    entrypoint_payload,
                    xml_payloads,
                )
        entrypoint_candidates = [
            name
            for name in sorted(names)
            if (
                detection.source_format == "ros2-urdf-package"
                and PurePosixPath(name).suffix.casefold() == ".urdf"
            )
            or (
                detection.source_format == "gazebo-model-bundle"
                and name.casefold().endswith("model.sdf")
            )
            or (
                detection.source_format == "gazebo-world-bundle"
                and (name.casefold().endswith(".world") or name.casefold().endswith("world.sdf"))
            )
        ]
        if not entrypoint_candidates:
            raise AssetSourceAdapterError("ASSET_SOURCE_ENTRYPOINT_MISSING")
        if len(entrypoint_candidates) != 1:
            raise AssetSourceAdapterError("ASSET_SOURCE_ENTRYPOINT_AMBIGUOUS")
        entrypoint_source = entrypoint_candidates[0]
        prefix = "normalized/ros2" if detection.adapter_id == "ros2.urdf" else "normalized/gazebo"
        files: list[AssetFile] = []
        entrypoint_payload = b""
        xml_payloads: list[tuple[str, bytes]] = []
        with zipfile.ZipFile(source) as input_bundle:
            for name in sorted(names):
                output_name = normalized_member_path(f"{prefix}/{name}")
                files.append(_stream_zip_member(input_bundle, name, output_bundle, output_name))
                if PurePosixPath(name).suffix.casefold() in {".sdf", ".world", ".urdf", ".xacro"}:
                    info = input_bundle.getinfo(name)
                    if info.file_size > _MAX_XML_SOURCE_BYTES:
                        raise AssetSourceAdapterError("ASSET_SOURCE_XML_SIZE_EXCEEDED")
                    xml_payloads.append((output_name, input_bundle.read(name)))
                if name == entrypoint_source:
                    entrypoint_payload = input_bundle.read(name)
        return files, f"{prefix}/{entrypoint_source}", entrypoint_payload, xml_payloads
    if source.stat().st_size > _MAX_XML_SOURCE_BYTES:
        raise AssetSourceAdapterError("ASSET_SOURCE_XML_SIZE_EXCEEDED")
    payload = read_plugin_file(source, limit=_MAX_XML_SOURCE_BYTES)
    suffix = ".urdf" if detection.adapter_id == "ros2.urdf" else ".sdf"
    output_name = f"normalized/{_slug(source.stem, 'asset')}{suffix}"
    output_bundle.writestr(output_name, payload)
    return (
        [
            AssetFile(
                path=output_name,
                role="urdf" if suffix == ".urdf" else "sdf",
                media_type="application/xml",
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
        ],
        output_name,
        payload,
        [(output_name, payload)],
    )


def normalize_asset_source(
    source: Path,
    detection: AssetSourceDetection,
    destination: Path,
    *,
    expected_kind: AssetKind | None = None,
    embed_source_snapshot: bool = True,
) -> Path:
    """Normalize to an inspected, unqualified package without replacing existing output.

    Input lives in host-owned quarantine. Re-detection and before/after hashes
    reject stale detection objects and ordinary source changes; they are not an
    OS sandbox against a privileged concurrent writer restoring the original.
    """

    detection = AssetSourceDetection.model_validate(detection.model_dump(mode="python"))
    if expected_kind is not None and (
        not isinstance(expected_kind, str) or expected_kind not in {"map", "world", "vehicle"}
    ):
        raise AssetSourceAdapterError("ASSET_SOURCE_KIND_MISMATCH")
    current_detection = detect_asset_source(source)
    if current_detection != detection:
        raise AssetSourceAdapterError("ASSET_SOURCE_DETECTION_STALE")
    if detection.source_format == "ddpkg":
        inspected = inspect_ddpkg(source)
        if (
            expected_kind is not None
            and inspected.asset_ir.kind != expected_kind
            and not ({inspected.asset_ir.kind, expected_kind} <= {"map", "world"})
        ):
            raise AssetSourceAdapterError("ASSET_SOURCE_KIND_MISMATCH")
        return source
    if not detection.can_normalize_locally:
        raise AssetSourceAdapterError("ASSET_SOURCE_ADAPTER_REQUIRED")
    if detection.adapter_id not in {
        "dronedream.legacy-bundle",
        "gazebo.sdf",
        "open-rmf.building-map",
        "ros2.urdf",
    }:
        raise AssetSourceAdapterError("ASSET_SOURCE_ADAPTER_UNAVAILABLE")
    kind = detection.asset_kind or expected_kind
    if kind is None:
        raise AssetSourceAdapterError("ASSET_SOURCE_KIND_REQUIRED")
    if expected_kind is not None and kind != expected_kind:
        if not ({kind, expected_kind} <= {"map", "world"}):
            raise AssetSourceAdapterError("ASSET_SOURCE_KIND_MISMATCH")
        kind = expected_kind
    check_plain_plugin_path(destination)
    if destination.exists():
        raise AssetSourceAdapterError("ASSET_SOURCE_DESTINATION_EXISTS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sha256 = _sha256(source)
    temporary = destination.with_name(f".{destination.name}-{uuid4().hex}.tmp")
    owns_temporary = False
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as output:
            owns_temporary = True
            files, entrypoint, entrypoint_payload, xml_payloads = _normalization_inputs(
                source, detection, output
            )
            source_file = (
                _store_source_snapshot(source, output, source_sha256=source_sha256)
                if embed_source_snapshot
                else _store_source_reference(source, output, source_sha256=source_sha256)
            )
            files = [source_file, *files]
            # Preserved raw XML is also inspected at package admission. Keep its
            # disabled plugin declarations without double-counting its physics.
            source_xml = (
                [(source_file.path, entrypoint_payload)]
                if not zipfile.is_zipfile(source)
                and PurePosixPath(source_file.path).suffix.casefold()
                in {".sdf", ".world", ".urdf", ".xacro"}
                else []
            )
            if detection.adapter_id == "dronedream.legacy-bundle":
                with zipfile.ZipFile(source) as legacy_bundle:
                    legacy_manifest = _read_manifest(legacy_bundle, "manifest.json")
                asset_id = _slug(str(legacy_manifest.get("asset_id", "")), source.stem)
                name = str(legacy_manifest.get("name") or asset_id)
            else:
                asset_id, name = _root_identity(entrypoint_payload, source.stem)
            extensions = [
                extension
                for xml_path, xml_payload in [*xml_payloads, *source_xml]
                for extension in _runtime_extensions(xml_payload, xml_path)
            ]
            targets = []
            px4_sitl_model: str | None = None
            if detection.adapter_id == "dronedream.legacy-bundle" and kind == "vehicle":
                identity = legacy_manifest.get("artifact_identity")
                if isinstance(identity, dict) and identity.get("control_interface") == "mavsdk":
                    source_model = identity.get("source_model")
                    if isinstance(source_model, str) and source_model.startswith("model://"):
                        candidate = source_model.removeprefix("model://")
                        if candidate.replace("_", "").replace("-", "").isalnum():
                            px4_sitl_model = candidate
            if detection.adapter_id in {
                "dronedream.legacy-bundle",
                "gazebo.sdf",
                "open-rmf.building-map",
            }:
                targets.append(
                    SimulationTarget(
                        target_id="gazebo-harmonic",
                        simulator="gazebo-harmonic",
                        simulator_version="runtime-detected",
                        autopilot="px4" if px4_sitl_model else "none",
                        entrypoint=entrypoint,
                    )
                )
            capabilities = (
                ["gazebo", "legacy-migrated"]
                if detection.adapter_id == "dronedream.legacy-bundle"
                else ["gazebo"]
                if detection.adapter_id in {"gazebo.sdf", "open-rmf.building-map"}
                else ["ros2"]
            )
            if detection.adapter_id == "open-rmf.building-map":
                capabilities.extend(["rmf-semantic-topology", "uav-airspace-source"])
            if px4_sitl_model is not None:
                capabilities.append(f"px4.sitl-model={px4_sitl_model}")
            (
                geometry,
                physics,
                vehicle,
                sensors,
                semantics,
                interfaces,
                readiness,
            ) = _structured_summaries(
                xml_payloads,
                files,
                kind=kind,
                targets=targets,
                capabilities=capabilities,
            )
            if detection.adapter_id == "open-rmf.building-map":
                # Gazebo static world geometry does not need dynamic-link mass
                # or inertia.  Runtime and aircraft-pair qualification remain
                # mandatory and are represented independently.
                graph_path = "normalized/rmf/generated/navigation-semantic-graph.json"
                semantics = semantics.model_copy(update={"navigation_graph_path": graph_path})
                readiness = readiness.model_copy(
                    update={
                        "maturity_ceiling": "physics_ready",
                        "missing_fields": [
                            field
                            for field in readiness.missing_fields
                            if field
                            not in {
                                "physics.mass",
                                "physics.inertia",
                                "semantics.navigation",
                            }
                        ],
                    }
                )
            if detection.adapter_id == "dronedream.legacy-bundle" and kind in {
                "map",
                "world",
            }:
                legacy_files = legacy_manifest.get("files")
                if not isinstance(legacy_files, dict):
                    raise AssetSourceAdapterError("ASSET_SOURCE_LEGACY_MANIFEST_INVALID")
                graph_value = legacy_files.get("graph")
                if isinstance(graph_value, str):
                    try:
                        graph_source = normalized_member_path(graph_value)
                    except AssetPackageError as error:
                        raise AssetSourceAdapterError(
                            "ASSET_SOURCE_LEGACY_GRAPH_INVALID"
                        ) from error
                    graph_path = f"normalized/legacy/{graph_source}"
                    if not any(entry.path == graph_path for entry in files):
                        raise AssetSourceAdapterError("ASSET_SOURCE_LEGACY_GRAPH_MISSING")
                    semantics = semantics.model_copy(update={"navigation_graph_path": graph_path})
                    readiness = readiness.model_copy(
                        update={
                            "missing_fields": [
                                field
                                for field in readiness.missing_fields
                                if field != "semantics.navigation"
                            ]
                        }
                    )
            asset_ir = AssetIR(
                asset_id=asset_id,
                name=name,
                version="0.1.0",
                kind=kind,
                source=AssetSource(
                    adapter_id=detection.adapter_id,
                    adapter_version="1.0.0",
                    application=(
                        "DroneDream Legacy"
                        if detection.adapter_id == "dronedream.legacy-bundle"
                        else "Open-RMF / DroneDream Built-in Converter"
                        if detection.adapter_id == "open-rmf.building-map"
                        else "Gazebo"
                        if detection.adapter_id == "gazebo.sdf"
                        else "ROS 2"
                    ),
                    source_format=detection.source_format,
                    source_sha256=source_sha256,
                ),
                coordinate_frame="map_enu" if kind in {"map", "world"} else "base_link",
                files=files,
                simulation_targets=targets,
                runtime_extensions=extensions,
                geometry=geometry,
                physics=physics,
                vehicle=vehicle,
                sensors=sensors,
                semantics=semantics,
                interfaces=interfaces,
                readiness=readiness,
                capabilities=capabilities,
                license_expression="NOASSERTION",
            )
            ir_payload = (
                json.dumps(asset_ir.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
                + "\n"
            ).encode()
            ir_file = AssetFile(
                path="normalized/asset-ir.json",
                role="asset_ir",
                media_type="application/json",
                sha256=hashlib.sha256(ir_payload).hexdigest(),
                size_bytes=len(ir_payload),
            )
            output.writestr(ir_file.path, ir_payload)
            manifest_files = [*files, ir_file]
            manifest = DDPkgManifest(
                package_id=f"{asset_id}.0.1.0",
                asset_id=asset_id,
                asset_kind=kind,
                content_sha256=package_content_sha256(manifest_files),
                files=manifest_files,
                created_at=datetime.now(UTC),
            )
            output.writestr(
                "manifest.json",
                json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
            )
        if _sha256(source) != source_sha256:
            raise AssetSourceAdapterError("ASSET_SOURCE_CHANGED")
        inspect_ddpkg(temporary)
        check_plain_plugin_path(destination)
        # A same-filesystem hard link publishes atomically without overwriting
        # an output created by another importer after the existence check.
        os.link(temporary, destination)
        # Successful publication must not turn into a failed import solely
        # because an antivirus temporarily holds the disposable link open.
        with suppress(OSError):
            temporary.unlink()
        owns_temporary = False
        return destination
    except BaseException:
        if owns_temporary:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise


def _parse_source_xml(payload: bytes) -> ElementTree.Element:
    """Share byte/element/DTD limits with package admission instead of an unrestricted parser."""
    try:
        return parse_xml(payload, maximum_bytes=_MAX_XML_SOURCE_BYTES, maximum_elements=1_000_000)
    except (ElementTree.ParseError, ValueError) as error:
        raise AssetSourceAdapterError("ASSET_SOURCE_XML_INVALID") from error


def _read_manifest(bundle: zipfile.ZipFile, path: str) -> dict:
    """Bound legacy/current metadata reads before interpreting schema identity or provenance."""
    try:
        with bundle.open(path) as source:
            raw = source.read(MAX_MANIFEST_BYTES + 1)
        value = decode_json(raw, limit=MAX_MANIFEST_BYTES, node_limit=2_000_000)
        if not isinstance(value, dict):
            raise ValueError("object required")
        return value
    except ValueError as error:
        raise AssetSourceAdapterError("ASSET_SOURCE_MANIFEST_INVALID") from error
