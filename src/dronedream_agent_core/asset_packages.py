"""Secure, content-addressed contracts for external simulation assets.

The package boundary is intentionally smaller than a modeling application. Source
adapters normalize Blender/CAD/GIS/ROS/Gazebo output into Asset IR, then this module
admits a deterministic ``.ddpkg`` without executing embedded simulator plugins.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal
from xml.etree import ElementTree

from pydantic import Field, model_validator

from dronedream_plugin_sdk.protocol import decode_json

from .contracts import StrictModel
from .plugin_files import check_plain_plugin_path, portable_plugin_path
from .xml_values import parse_xml
from .zip_index import validate_zip_index

AssetKind = Literal["map", "world", "vehicle"]
AssetMaturity = Literal[
    "visual_only",
    "physics_ready",
    "simulation_ready",
    "flight_ready",
    "qualified",
]
AssetImportState = Literal[
    "created",
    "quarantining",
    "parsing",
    "needs_input",
    "normalizing",
    "building",
    "validating",
    "qualified",
    "failed",
    "cancelled",
]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SafeIdentifier = Annotated[
    str,
    Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", min_length=3, max_length=160),
]

MAX_PACKAGE_MEMBERS = 20_000
MAX_PACKAGE_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
MAX_MEMBER_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
# The bundled School Map contains a deliberately flat PPM collision/preview texture
# whose legitimate DEFLATE ratio is about 272:1.  Keep the bomb defence finite while
# admitting that known data shape; total/member byte limits remain independently
# enforced before extraction.
MAX_COMPRESSION_RATIO = 300.0
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_ASSET_IR_BYTES = 16 * 1024 * 1024
MAX_XML_INSPECTION_BYTES = 64 * 1024 * 1024
MAX_PACKAGE_ARCHIVE_BYTES = MAX_PACKAGE_UNCOMPRESSED_BYTES + 64 * 1024 * 1024
_EXECUTABLE_SUFFIXES = {
    ".bat",
    ".cmd",
    ".com",
    ".dll",
    ".exe",
    ".msi",
    ".ps1",
    ".py",
    ".sh",
    ".so",
}


class AssetPackageError(ValueError):
    """A package cannot cross the quarantine boundary."""


class AssetSource(StrictModel):
    """Declared source provenance; source hashes still need local admission verification."""

    adapter_id: SafeIdentifier
    adapter_version: str = Field(min_length=1, max_length=80)
    application: str = Field(min_length=1, max_length=120)
    application_version: str | None = Field(default=None, max_length=80)
    source_format: str = Field(min_length=1, max_length=80)
    source_sha256: Sha256
    source_uri_redacted: str | None = Field(default=None, max_length=240)


class AssetFile(StrictModel):
    """One package member's identity, byte size and authority-bearing role."""

    path: str = Field(min_length=1, max_length=500)
    role: Literal[
        "asset_ir",
        "source",
        "source_reference",
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
    ]
    media_type: str = Field(min_length=1, max_length=160)
    sha256: Sha256
    size_bytes: int = Field(ge=0, le=MAX_MEMBER_UNCOMPRESSED_BYTES, strict=True)

    # 功能：
    #   拒绝跨系统含义不一致或需要重新规范化的成员路径。
    # 输入：
    #   self：包含路径、角色、大小和摘要的文件索引项。
    # 输出：
    #   self：通过路径检查的原索引项。
    @model_validator(mode="after")
    def validate_path(self) -> AssetFile:
        normalized = normalized_member_path(self.path)
        if normalized != self.path:
            raise ValueError("asset file paths must already be normalized")
        return self


class RuntimeExtension(StrictModel):
    """A discovered executable reference remains disabled; declaration is not authorization."""

    name: str = Field(min_length=1, max_length=200)
    filename: str | None = Field(default=None, max_length=500)
    source_path: str = Field(min_length=1, max_length=500)
    enabled_by_default: Literal[False] = False
    admission: Literal["blocked", "requires_explicit_trust"] = "blocked"

    # 功能：
    #   将插件声明的来源限定为包内相对路径；声明本身不授予执行权限。
    # 输入：
    #   self：待校验的运行插件声明。
    # 输出：
    #   self：来源路径合法的原插件声明。
    @model_validator(mode="after")
    def validate_path(self) -> RuntimeExtension:
        if normalized_member_path(self.source_path) != self.source_path:
            raise ValueError("runtime extension source_path must be normalized")
        return self


class SimulationTarget(StrictModel):
    """Declared simulator interface; compatibility is checked again during qualification."""

    target_id: SafeIdentifier
    simulator: Literal["gazebo-classic", "gazebo-harmonic", "isaac-sim", "webots", "other"]
    simulator_version: str = Field(min_length=1, max_length=80)
    ros_distribution: str | None = Field(default=None, max_length=80)
    autopilot: Literal["px4", "ardupilot", "none", "other"] = "none"
    entrypoint: str = Field(min_length=1, max_length=500)

    # 功能：
    #   检查仿真入口的路径拼写；文件是否存在由资产索引校验。
    # 输入：
    #   self：声明仿真器、飞控及入口文件的目标配置。
    # 输出：
    #   self：入口路径合法的原目标配置。
    @model_validator(mode="after")
    def validate_entrypoint(self) -> SimulationTarget:
        if normalized_member_path(self.entrypoint) != self.entrypoint:
            raise ValueError("simulation entrypoint must be normalized")
        return self


class GeometrySummary(StrictModel):
    """Geometry inventory and units, not a collision-clearance or navigability proof."""

    visual_paths: list[str] = Field(default_factory=list, max_length=10_000)
    collision_paths: list[str] = Field(default_factory=list, max_length=10_000)
    coordinate_frames: list[str] = Field(default_factory=list, max_length=10_000)
    up_axis: Literal["x", "y", "z", "unknown"] = "unknown"
    paths_normalized: bool = Field(default=True, strict=True)
    visual_count: int = Field(default=0, ge=0, le=1_000_000, strict=True)
    collision_count: int = Field(default=0, ge=0, le=1_000_000, strict=True)

    # 功能：
    #   检查视觉和碰撞几何引用，避免渲染器解析绝对路径或歧义路径。
    # 输入：
    #   self：包含几何文件列表和坐标约定的摘要。
    # 输出：
    #   self：几何路径全部合法的原摘要。
    @model_validator(mode="after")
    def validate_geometry_paths(self) -> GeometrySummary:
        for path in [*self.visual_paths, *self.collision_paths]:
            if normalized_member_path(path) != path:
                raise ValueError("geometry paths must be normalized package paths")
        return self


class PhysicsSummary(StrictModel):
    """Declared physics coverage; numeric/model feasibility needs independent qualification."""

    link_count: int = Field(default=0, ge=0, le=1_000_000, strict=True)
    joint_count: int = Field(default=0, ge=0, le=1_000_000, strict=True)
    mass_entry_count: int = Field(default=0, ge=0, le=1_000_000, strict=True)
    inertia_entry_count: int = Field(default=0, ge=0, le=1_000_000, strict=True)
    collision_complete: bool = Field(default=False, strict=True)
    mass_complete: bool = Field(default=False, strict=True)
    inertia_complete: bool = Field(default=False, strict=True)
    contact_materials: list[str] = Field(default_factory=list, max_length=1_000)


class VehicleSummary(StrictModel):
    """Optional dynamics descriptors, with unknown distinct from a guessed zero value."""

    vehicle_class: Literal[
        "multirotor",
        "fixed_wing",
        "vtol",
        "ground",
        "other",
        "unknown",
    ] = "unknown"
    actuator_count: int | None = Field(default=None, ge=0, le=1_000, strict=True)
    rotor_count: int | None = Field(default=None, ge=0, le=1_000, strict=True)
    autopilot_profile: str | None = Field(default=None, max_length=160)
    battery_model: str | None = Field(default=None, max_length=160)
    payload_limit_kg: float | None = Field(default=None, ge=0.0, le=10_000.0, strict=True)


class SensorSummary(StrictModel):
    """Declared sensor source/rate, not a measurement or proof of live sensor delivery."""

    kind: str = Field(min_length=1, max_length=120)
    frame_id: str | None = Field(default=None, max_length=160)
    source_path: str = Field(min_length=1, max_length=500)
    update_rate_hz: float | None = Field(default=None, gt=0.0, le=1_000_000.0, strict=True)

    # 功能：
    #   检查传感器声明的来源路径；不以声明代替实际传感器数据。
    # 输入：
    #   self：包含传感器种类、坐标系、来源和频率的摘要。
    # 输出：
    #   self：来源路径合法的原传感器摘要。
    @model_validator(mode="after")
    def validate_source_path(self) -> SensorSummary:
        if normalized_member_path(self.source_path) != self.source_path:
            raise ValueError("sensor source_path must be normalized")
        return self


class SemanticSummary(StrictModel):
    """Semantic/navigation file references whose existence is checked by AssetIR."""

    named_place_count: int = Field(default=0, ge=0, le=1_000_000, strict=True)
    region_count: int = Field(default=0, ge=0, le=1_000_000, strict=True)
    navigation_graph_path: str | None = Field(default=None, max_length=500)
    traversability_path: str | None = Field(default=None, max_length=500)

    # 功能：
    #   允许缺省语义文件，但已提供的导航图与可通行性路径必须是包内规范路径。
    # 输入：
    #   self：地点、区域计数及可选文件引用组成的语义摘要。
    # 输出：
    #   self：文件引用合法的原语义摘要。
    @model_validator(mode="after")
    def validate_semantic_paths(self) -> SemanticSummary:
        for path in (self.navigation_graph_path, self.traversability_path):
            if path is not None and normalized_member_path(path) != path:
                raise ValueError("semantic paths must be normalized")
        return self


class InterfaceSummary(StrictModel):
    """Entrypoint declarations do not grant permission to execute their contents."""

    gazebo_entrypoints: list[str] = Field(default_factory=list, max_length=64)
    ros2_entrypoints: list[str] = Field(default_factory=list, max_length=64)
    px4_profiles: list[str] = Field(default_factory=list, max_length=64)

    # 功能：
    #   检查 Gazebo 与 ROS 入口路径；PX4 配置标识不是文件路径，不在此转换。
    # 输入：
    #   self：运行接口入口与配置标识的摘要。
    # 输出：
    #   self：文件入口路径合法的原接口摘要。
    @model_validator(mode="after")
    def validate_interface_paths(self) -> InterfaceSummary:
        for path in [*self.gazebo_entrypoints, *self.ros2_entrypoints]:
            if normalized_member_path(path) != path:
                raise ValueError("runtime interface paths must be normalized")
        return self


class ReadinessSummary(StrictModel):
    """Source readiness claims remain bounded by locally verified qualification evidence."""

    maturity_ceiling: AssetMaturity = "visual_only"
    qualification_required: bool = Field(default=True, strict=True)
    runtime_validation_performed: bool = Field(default=False, strict=True)
    missing_fields: list[str] = Field(default_factory=list, max_length=1_000)
    evidence_paths: list[str] = Field(default_factory=list, max_length=1_000)

    # 功能：
    #   1. 拒绝未声明运行验证却声称达到运行成熟度的矛盾状态。
    #   2. 检查证据路径；声明通过不等于实际证据已经验收。
    # 输入：
    #   self：来源适配器报告的成熟度与证据摘要。
    # 输出：
    #   self：声明内部一致、证据路径合法的原摘要。
    @model_validator(mode="after")
    def validate_readiness(self) -> ReadinessSummary:
        if (
            self.maturity_ceiling in {"simulation_ready", "flight_ready", "qualified"}
            and not self.runtime_validation_performed
        ):
            raise ValueError("runtime maturity requires runtime validation")
        for path in self.evidence_paths:
            if normalized_member_path(path) != path:
                raise ValueError("readiness evidence paths must be normalized")
        return self


class AssetIR(StrictModel):
    """Normalized metadata independent of source-editor formats and simulator execution."""

    schema_version: Literal["dronedream.asset-ir.v1"] = "dronedream.asset-ir.v1"
    asset_id: SafeIdentifier
    name: str = Field(min_length=1, max_length=160)
    version: str = Field(min_length=1, max_length=80)
    kind: AssetKind
    source: AssetSource
    coordinate_frame: str = Field(min_length=1, max_length=80)
    length_unit: Literal["metre"] = "metre"
    mass_unit: Literal["kilogram"] = "kilogram"
    angle_unit: Literal["radian"] = "radian"
    files: list[AssetFile] = Field(min_length=1, max_length=20_000)
    simulation_targets: list[SimulationTarget] = Field(default_factory=list, max_length=32)
    runtime_extensions: list[RuntimeExtension] = Field(default_factory=list, max_length=256)
    geometry: GeometrySummary = Field(default_factory=GeometrySummary)
    physics: PhysicsSummary = Field(default_factory=PhysicsSummary)
    vehicle: VehicleSummary | None = None
    sensors: list[SensorSummary] = Field(default_factory=list, max_length=1_000)
    semantics: SemanticSummary = Field(default_factory=SemanticSummary)
    interfaces: InterfaceSummary = Field(default_factory=InterfaceSummary)
    readiness: ReadinessSummary = Field(default_factory=ReadinessSummary)
    semantic_layers: list[str] = Field(default_factory=list, max_length=256)
    capabilities: list[str] = Field(default_factory=list, max_length=256)
    dependencies: list[str] = Field(default_factory=list, max_length=256)
    license_expression: str = Field(min_length=1, max_length=240)

    # 功能：
    #   保证文件名不因大小写冲突，并要求所有入口、几何、传感器和证据引用存在于索引。
    # 输入：
    #   self：规范化资产的完整中间表示。
    # 输出：
    #   self：内部文件引用一致的原资产表示。
    @model_validator(mode="after")
    def validate_file_index(self) -> AssetIR:
        paths = [entry.path.casefold() for entry in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("asset file paths must be unique case-insensitively")
        entrypoints = {target.entrypoint for target in self.simulation_targets}
        declared = {entry.path for entry in self.files}
        if not entrypoints.issubset(declared):
            raise ValueError("simulation target references an undeclared file")
        extension_paths = {extension.source_path for extension in self.runtime_extensions}
        if not extension_paths.issubset(declared):
            raise ValueError("runtime extension references an undeclared file")
        interface_paths = {
            *self.interfaces.gazebo_entrypoints,
            *self.interfaces.ros2_entrypoints,
        }
        if not interface_paths.issubset(declared):
            raise ValueError("runtime interface references an undeclared file")
        sensor_paths = {sensor.source_path for sensor in self.sensors}
        if not sensor_paths.issubset(declared):
            raise ValueError("sensor references an undeclared file")
        data_paths = {
            *self.geometry.visual_paths,
            *self.geometry.collision_paths,
            *self.readiness.evidence_paths,
            *[
                path
                for path in (
                    self.semantics.navigation_graph_path,
                    self.semantics.traversability_path,
                )
                if path is not None
            ],
        }
        if not data_paths.issubset(declared):
            raise ValueError("asset metadata references an undeclared file")
        return self


class QualificationBinding(StrictModel):
    """Content/environment binding; caller must verify its locally produced evidence."""

    schema_version: Literal["dronedream.asset-qualification-binding.v1"] = (
        "dronedream.asset-qualification-binding.v1"
    )
    maturity: AssetMaturity
    content_sha256: Sha256
    environment_versions: dict[str, str] = Field(min_length=1, max_length=64)
    evidence_paths: list[str] = Field(default_factory=list, max_length=1_000)
    qualified_at: datetime

    # 功能：
    #   要求验收时间含时区、证据路径规范，并为飞行资格声明保留证据引用。
    # 输入：
    #   self：绑定资产摘要与运行环境的资格声明。
    # 输出：
    #   self：时间和证据引用合法的原资格声明。
    @model_validator(mode="after")
    def validate_evidence_paths(self) -> QualificationBinding:
        if self.qualified_at.utcoffset() is None:
            raise ValueError("qualification time must include a timezone")
        if any(normalized_member_path(path) != path for path in self.evidence_paths):
            raise ValueError("qualification evidence paths must be normalized")
        if self.maturity in {"flight_ready", "qualified"} and not self.evidence_paths:
            raise ValueError("flight-ready assets require evidence paths")
        return self


class DDPkgManifest(StrictModel):
    """Content-addressed file index; its digest is integrity, not a publisher signature."""

    schema_version: Literal["dronedream.ddpkg.v1"] = "dronedream.ddpkg.v1"
    package_id: SafeIdentifier
    asset_id: SafeIdentifier
    asset_kind: AssetKind
    asset_ir_path: str = "normalized/asset-ir.json"
    content_sha256: Sha256
    files: list[AssetFile] = Field(min_length=1, max_length=20_000)
    qualification: QualificationBinding | None = None
    created_at: datetime

    # 功能：
    #   1. 校验唯一文件索引、确定性摘要、资产中间表示入口和创建时间。
    #   2. 将资格声明绑定到当前内容和证据，拒绝复用旧包的资格标签。
    # 输入：
    #   self：待校验的 DDPKG 清单。
    # 输出：
    #   self：索引与声明内部一致的原清单。
    @model_validator(mode="after")
    def validate_content_index(self) -> DDPkgManifest:
        if normalized_member_path(self.asset_ir_path) != self.asset_ir_path:
            raise ValueError("asset_ir_path must be normalized")
        paths = [entry.path.casefold() for entry in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("package paths must be unique case-insensitively")
        if self.asset_ir_path not in {entry.path for entry in self.files}:
            raise ValueError("package file index must contain Asset IR")
        if self.content_sha256 != package_content_sha256(self.files):
            raise ValueError("package content hash does not match its file index")
        if self.created_at.utcoffset() is None:
            raise ValueError("package creation time must include a timezone")
        if any(entry.path == "manifest.json" for entry in self.files):
            raise ValueError("manifest cannot recursively index itself")
        if (
            next(entry for entry in self.files if entry.path == self.asset_ir_path).role
            != "asset_ir"
        ):
            raise ValueError("asset_ir_path must identify the Asset IR role")
        if self.qualification is not None and (
            self.qualification.content_sha256 != self.content_sha256
            or not set(self.qualification.evidence_paths).issubset(
                entry.path for entry in self.files
            )
        ):
            raise ValueError("qualification does not bind this package's content and evidence")
        return self


class AssetImportJob(StrictModel):
    """Persisted import state; transitions validate a new revision without mutating this one."""

    schema_version: Literal["dronedream.asset-import-job.v1"] = "dronedream.asset-import-job.v1"
    job_id: Annotated[str, Field(pattern=r"^asset-job-[0-9a-f]{24}$")]
    owner_id: str = Field(min_length=1, max_length=160)
    source_name: str = Field(min_length=1, max_length=260)
    source_format: str = Field(min_length=1, max_length=80)
    detected_source_format: str | None = Field(default=None, max_length=80)
    source_adapter_id: str | None = Field(default=None, max_length=160)
    package_sha256: Sha256 | None = None
    normalized_content_sha256: Sha256 | None = None
    qualified_content_sha256: Sha256 | None = None
    state: AssetImportState = "created"
    progress_percent: int = Field(ge=0, le=100, strict=True)
    revision: int = Field(ge=0, strict=True)
    asset_id: str | None = Field(default=None, max_length=160)
    asset_kind: AssetKind | None = None
    issue_codes: list[str] = Field(default_factory=list, max_length=1_000)
    required_inputs: list[str] = Field(default_factory=list, max_length=256)
    created_at: datetime
    updated_at: datetime

    # 功能：
    #   检查导入状态所需的源摘要、资产身份及带时区的时间顺序。
    # 输入：
    #   self：待持久化或重新加载的导入任务记录。
    # 输出：
    #   self：状态与基础元数据一致的原任务记录。
    @model_validator(mode="after")
    def validate_state_metadata(self) -> AssetImportJob:
        if self.created_at.utcoffset() is None or self.updated_at.utcoffset() is None:
            raise ValueError("import times must include a timezone")
        if self.updated_at < self.created_at:
            raise ValueError("import update time precedes creation")
        if self.state not in {"created", "quarantining"} and self.package_sha256 is None:
            raise ValueError("quarantined jobs require package_sha256")
        if self.state == "qualified" and (self.asset_id is None or self.asset_kind is None):
            raise ValueError("qualified jobs require an asset identity")
        return self


class InspectedDDPkg(StrictModel):
    """Verified archive metadata, with detected executable references still blocked by default."""

    manifest: DDPkgManifest
    asset_ir: AssetIR
    detected_runtime_extensions: list[RuntimeExtension] = Field(default_factory=list)


_IMPORT_TRANSITIONS: dict[AssetImportState, frozenset[AssetImportState]] = {
    "created": frozenset({"quarantining", "cancelled"}),
    "quarantining": frozenset({"parsing", "failed", "cancelled"}),
    "parsing": frozenset({"needs_input", "normalizing", "failed", "cancelled"}),
    "needs_input": frozenset({"parsing", "normalizing", "failed", "cancelled"}),
    "normalizing": frozenset({"needs_input", "building", "failed", "cancelled"}),
    "building": frozenset({"validating", "failed", "cancelled"}),
    "validating": frozenset({"qualified", "failed", "cancelled"}),
    "qualified": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


# 功能：
#   校验跨系统一致的包内路径，拒绝目录穿越、设备名与 Windows 附加数据流。
# 输入：
#   value：待检查的成员路径字符串。
# 输出：
#   normalized：与输入拼写完全一致的规范相对路径。
def normalized_member_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 500
        or "\x00" in value
        or "\\" in value
    ):
        raise AssetPackageError("DDPKG_MEMBER_PATH_INVALID")
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or ":" in path.parts[0]
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise AssetPackageError("DDPKG_MEMBER_PATH_INVALID")
    normalized = path.as_posix()
    if normalized != value or value.endswith("/"):
        raise AssetPackageError("DDPKG_MEMBER_PATH_NOT_NORMALIZED")
    try:
        portable_plugin_path(value)
    except ValueError as error:
        raise AssetPackageError("DDPKG_MEMBER_PATH_INVALID") from error
    return normalized


# 功能：
#   重验文件索引并按稳定顺序计算摘要，使身份不受 ZIP 排列与压缩方式影响。
# 输入：
#   files：待绑定的全部内容文件索引，不包含清单自身。
# 输出：
#   content_sha256：规范化索引的 SHA-256 十六进制摘要。
def package_content_sha256(files: list[AssetFile]) -> str:
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_PACKAGE_MEMBERS:
        raise AssetPackageError("DDPKG_FILE_INDEX_INVALID")
    # 已构造的模型内部仍可能被修改，因此不能只相信对象的 Python 类型。
    files = [AssetFile.model_validate(entry.model_dump(mode="python")) for entry in files]
    if len({entry.path.casefold() for entry in files}) != len(files):
        raise AssetPackageError("DDPKG_DUPLICATE_MEMBER")
    payload = [
        entry.model_dump(mode="json")
        for entry in sorted(files, key=lambda candidate: candidate.path.casefold())
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    content_sha256 = hashlib.sha256(encoded.encode()).hexdigest()
    return content_sha256


# 功能：
#   比较资格声明与当前内容及环境；只判断时效绑定，不验证签名或运行证据。
# 输入：
#   binding：可缺省的资格声明。
#   content_sha256：当前选中资产的内容摘要。
#   environment_versions：当前运行组件的版本快照。
# 输出：
#   current：声明是否合法且与当前内容、非空环境完全一致。
def qualification_is_current(
    binding: QualificationBinding | None,
    *,
    content_sha256: str,
    environment_versions: dict[str, str],
) -> bool:
    current = False
    if binding is None:
        return current
    try:
        binding = QualificationBinding.model_validate(binding.model_dump(mode="python"))
    except ValueError:
        return current
    current = bool(
        environment_versions
        and binding.content_sha256 == content_sha256
        and binding.environment_versions == environment_versions
    )
    return current


# 功能：
#   1. 按允许的状态边迁移导入任务，拒绝进度或时间倒退及缺失的失败／补充信息。
#   2. 创建独立的新记录并增加修订计数，不修改调用方持有的旧记录。
# 输入：
#   job：迁移前的任务快照。
#   next_state：目标状态。
#   progress_percent：目标进度百分比。
#   issue_codes：失败原因代码列表。
#   required_inputs：等待用户补充的输入项。
#   asset_id：本次确定的资产标识，缺省时沿用当前记录。
#   asset_kind：本次确定的资产种类，缺省时沿用当前记录。
#   normalized_content_sha256：规范化包摘要，缺省时沿用当前记录。
#   qualified_content_sha256：验收后包摘要，缺省时沿用当前记录。
#   now：带时区的迁移时间，缺省时读取当前 UTC 时间。
# 输出：
#   transitioned：通过状态和元数据校验的新任务记录。
def transition_import_job(
    job: AssetImportJob,
    next_state: AssetImportState,
    *,
    progress_percent: int,
    issue_codes: list[str] | None = None,
    required_inputs: list[str] | None = None,
    asset_id: str | None = None,
    asset_kind: AssetKind | None = None,
    normalized_content_sha256: str | None = None,
    qualified_content_sha256: str | None = None,
    now: datetime | None = None,
) -> AssetImportJob:
    job = AssetImportJob.model_validate(job.model_dump(mode="python"))
    if not isinstance(next_state, str) or next_state not in _IMPORT_TRANSITIONS:
        raise AssetPackageError("ASSET_IMPORT_TRANSITION_INVALID")
    if type(progress_percent) is not int or not 0 <= progress_percent <= 100:
        raise AssetPackageError("ASSET_IMPORT_PROGRESS_INVALID")
    if next_state not in _IMPORT_TRANSITIONS[job.state]:
        raise AssetPackageError(f"ASSET_IMPORT_TRANSITION_INVALID:{job.state}:{next_state}")
    if progress_percent < job.progress_percent:
        raise AssetPackageError("ASSET_IMPORT_PROGRESS_REGRESSION")
    if next_state == "qualified" and progress_percent != 100:
        raise AssetPackageError("ASSET_IMPORT_QUALIFIED_REQUIRES_COMPLETE_PROGRESS")
    if next_state == "needs_input" and not required_inputs:
        raise AssetPackageError("ASSET_IMPORT_REQUIRED_INPUTS_MISSING")
    if next_state == "failed" and not issue_codes:
        raise AssetPackageError("ASSET_IMPORT_FAILURE_ISSUES_MISSING")
    for values in (issue_codes, required_inputs):
        if values is not None and (
            not isinstance(values, list)
            or any(not isinstance(item, str) or not item.strip() for item in values)
        ):
            raise AssetPackageError("ASSET_IMPORT_MESSAGE_LIST_INVALID")
    current_time = now if now is not None else datetime.now(UTC)
    if not isinstance(current_time, datetime) or current_time.utcoffset() is None:
        raise AssetPackageError("ASSET_IMPORT_TIME_INVALID")
    if current_time < job.updated_at:
        raise AssetPackageError("ASSET_IMPORT_TIME_REGRESSION")
    value = job.model_dump(mode="python")
    value.update(
        {
            "state": next_state,
            "progress_percent": progress_percent,
            "revision": job.revision + 1,
            "issue_codes": list(dict.fromkeys(issue_codes or [])),
            "required_inputs": list(dict.fromkeys(required_inputs or [])),
            "asset_id": asset_id if asset_id is not None else job.asset_id,
            "asset_kind": asset_kind if asset_kind is not None else job.asset_kind,
            "normalized_content_sha256": (
                normalized_content_sha256
                if normalized_content_sha256 is not None
                else job.normalized_content_sha256
            ),
            "qualified_content_sha256": (
                qualified_content_sha256
                if qualified_content_sha256 is not None
                else job.qualified_content_sha256
            ),
            "updated_at": current_time,
        }
    )
    transitioned = AssetImportJob.model_validate(value)
    return transitioned


# 功能：
#   1. 在提取前检查全部 ZIP 成员的路径、类型、大小、压缩比与文件目录冲突。
#   2. 默认禁止可执行文件；返回索引不代表授予执行权限。
# 输入：
#   bundle：已经打开、尚未解压到磁盘的归档。
#   allow_executables：是否允许索引中出现已知可执行文件扩展名。
# 输出：
#   indexed：以规范路径为键的普通文件成员索引。
def inspect_zip_members(
    bundle: zipfile.ZipFile, *, allow_executables: bool = False
) -> dict[str, zipfile.ZipInfo]:
    members = bundle.infolist()
    if not members or len(members) > MAX_PACKAGE_MEMBERS:
        raise AssetPackageError("DDPKG_MEMBER_LIMIT_EXCEEDED")
    total_size = 0
    indexed: dict[str, zipfile.ZipInfo] = {}
    casefolded: set[str] = set()
    directories: set[str] = set()
    for info in members:
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise AssetPackageError("DDPKG_SYMLINK_FORBIDDEN")
        if stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise AssetPackageError("DDPKG_SPECIAL_MEMBER_FORBIDDEN")
        if info.orig_filename != info.filename:
            raise AssetPackageError("DDPKG_MEMBER_PATH_INVALID")
        is_directory = info.is_dir()
        if (stat.S_IFMT(mode) == stat.S_IFDIR and not is_directory) or (
            stat.S_IFMT(mode) == stat.S_IFREG and is_directory
        ):
            raise AssetPackageError("DDPKG_MEMBER_TYPE_CONFLICT")
        name = normalized_member_path(info.filename[:-1] if is_directory else info.filename)
        folded = name.casefold()
        if folded in casefolded:
            raise AssetPackageError("DDPKG_DUPLICATE_MEMBER")
        casefolded.add(folded)
        if info.flag_bits & 0x1:
            raise AssetPackageError("DDPKG_ENCRYPTED_MEMBER_FORBIDDEN")
        if is_directory:
            if info.file_size != 0:
                raise AssetPackageError("DDPKG_DIRECTORY_DATA_FORBIDDEN")
            directories.add(folded)
            continue
        if info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise AssetPackageError("DDPKG_MEMBER_SIZE_EXCEEDED")
        ratio = info.file_size / max(info.compress_size, 1)
        if info.file_size > 1024 * 1024 and ratio > MAX_COMPRESSION_RATIO:
            raise AssetPackageError("DDPKG_COMPRESSION_RATIO_EXCEEDED")
        if not allow_executables and PurePosixPath(name).suffix.casefold() in _EXECUTABLE_SUFFIXES:
            raise AssetPackageError("DDPKG_EXECUTABLE_MEMBER_FORBIDDEN")
        total_size += info.file_size
        indexed[name] = info
    if total_size > MAX_PACKAGE_UNCOMPRESSED_BYTES:
        raise AssetPackageError("DDPKG_TOTAL_SIZE_EXCEEDED")
    # 大小写不敏感地检查父路径，防止跨平台提取时文件占据另一个成员的目录。
    files = {name.casefold() for name in indexed}
    for name in files | directories:
        if any(
            parent.as_posix() in files for parent in PurePosixPath(name).parents if parent.parts
        ):
            raise AssetPackageError("DDPKG_FILE_DIRECTORY_COLLISION")
    return indexed


# 功能：
#   在声明大小和单成员上限内流式统计实际解压字节，读取同时接受 ZIP 的 CRC 校验。
# 输入：
#   bundle：已经完成成员索引检查的归档。
#   path：需要计算摘要的成员路径。
# 输出：
#   measured：实际字节数与 SHA-256 十六进制摘要组成的元组。
def _member_digest(bundle: zipfile.ZipFile, path: str) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    maximum = min(bundle.getinfo(path).file_size, MAX_MEMBER_UNCOMPRESSED_BYTES)
    with bundle.open(path) as source:
        # 多读一个字节只为发现超额，不能无界展开后再检查 ZIP 声明。
        while chunk := source.read(min(1024 * 1024, maximum - size + 1)):
            size += len(chunk)
            if size > maximum:
                raise AssetPackageError("DDPKG_MEMBER_SIZE_EXCEEDED")
            digest.update(chunk)
    measured = size, digest.hexdigest()
    return measured


# 功能：
#   扫描已知 XML 格式及显式仿真入口中的插件声明，防止伪装文件角色绕过检查。
# 输入：
#   bundle：已验证成员大小和摘要的归档。
#   files：清单声明的内容文件索引。
#   explicit_xml_paths：接口和仿真目标明确声明为 XML 的入口路径。
# 输出：
#   extensions：从 XML 实际发现、默认禁止执行的插件声明列表。
def _detected_extensions(
    bundle: zipfile.ZipFile,
    files: list[AssetFile],
    *,
    explicit_xml_paths: set[str],
) -> list[RuntimeExtension]:
    extensions: list[RuntimeExtension] = []
    for entry in files:
        if (
            entry.role not in {"sdf", "urdf", "xacro"}
            and PurePosixPath(entry.path).suffix.casefold()
            not in {".sdf", ".world", ".urdf", ".xacro"}
            and entry.path not in explicit_xml_paths
        ):
            continue
        if entry.size_bytes > MAX_XML_INSPECTION_BYTES:
            raise AssetPackageError(f"DDPKG_XML_INSPECTION_SIZE_EXCEEDED:{entry.path}")
        try:
            with bundle.open(entry.path) as source:
                raw = source.read(MAX_XML_INSPECTION_BYTES + 1)
            root = parse_xml(
                raw, maximum_bytes=MAX_XML_INSPECTION_BYTES, maximum_elements=1_000_000
            )
        except (ElementTree.ParseError, KeyError, ValueError) as error:
            raise AssetPackageError(f"DDPKG_XML_INVALID:{entry.path}") from error
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] != "plugin":
                continue
            if len(extensions) >= 256:
                raise AssetPackageError("DDPKG_RUNTIME_EXTENSION_LIMIT")
            extensions.append(
                RuntimeExtension(
                    name=element.attrib.get("name") or "unnamed-plugin",
                    filename=element.attrib.get("filename"),
                    source_path=entry.path,
                )
            )
    return extensions


# 功能：
#   校验资产包并返回元数据，不提取文件或执行包内插件。
# 输入：
#   archive：待校验的本地 DDPKG 路径。
# 输出：
#   inspected：通过实际内容、索引和摘要检查的包元数据。
def inspect_ddpkg(archive: Path) -> InspectedDDPkg:
    with open_verified_ddpkg(archive) as verified_bundle:
        _, inspected = verified_bundle
    return inspected


# 功能：
#   1. 校验资产包，并在调用方读取期间保持同一个归档描述符打开。
#   2. 退出时再次检查文件身份；不替代抵御恶意并发修改的系统沙箱。
# 输入：
#   archive：资产管理器隔离目录中的 DDPKG 路径。
# 输出：
#   verified_bundle：在 with 作用域内有效的 ZIP 读取器与已校验元数据。
@contextmanager
def open_verified_ddpkg(archive: Path) -> Iterator[tuple[zipfile.ZipFile, InspectedDDPkg]]:
    check_plain_plugin_path(archive)
    before = archive.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_PACKAGE_ARCHIVE_BYTES:
        raise AssetPackageError("DDPKG_ARCHIVE_SIZE_OR_TYPE_INVALID")
    try:
        with archive.open("rb") as source:
            opened = os.fstat(source.fileno())
            if not os.path.samestat(before, opened) or opened.st_size != before.st_size:
                raise AssetPackageError("DDPKG_ARCHIVE_CHANGED")
            try:
                validate_zip_index(source, maximum_members=MAX_PACKAGE_MEMBERS)
            except ValueError as error:
                raise AssetPackageError("DDPKG_INDEX_INVALID") from error
            with zipfile.ZipFile(source) as bundle:
                inspected = _inspect_bundle(bundle)
                # 校验和后续复制共用描述符，不能校验旧路径后再打开新文件。
                verified_bundle = bundle, inspected
                yield verified_bundle
            after = os.fstat(source.fileno())
            check_plain_plugin_path(archive)
            if (
                not os.path.samestat(after, archive.stat())
                or after.st_size != opened.st_size
                or after.st_mtime_ns != opened.st_mtime_ns
            ):
                raise AssetPackageError("DDPKG_ARCHIVE_CHANGED")
    except (zipfile.BadZipFile, NotImplementedError) as error:
        raise AssetPackageError("DDPKG_INVALID_ZIP") from error


# 功能：
#   交叉验证 ZIP 实际成员、清单、逐文件字节、资产中间表示和插件声明。
# 输入：
#   bundle：经归档中央索引预检、保持打开的 ZIP 读取器。
# 输出：
#   inspected：完整性校验后的元数据与实际检测出的插件列表。
def _inspect_bundle(bundle: zipfile.ZipFile) -> InspectedDDPkg:
    members = inspect_zip_members(bundle)
    if "manifest.json" not in members:
        raise AssetPackageError("DDPKG_MANIFEST_MISSING")
    if members["manifest.json"].file_size > MAX_MANIFEST_BYTES:
        raise AssetPackageError("DDPKG_MANIFEST_SIZE_EXCEEDED")
    manifest = DDPkgManifest.model_validate(
        _member_json(bundle, "manifest.json", maximum_bytes=MAX_MANIFEST_BYTES)
    )
    declared = {entry.path for entry in manifest.files}
    actual = set(members) - {"manifest.json"}
    if declared != actual:
        raise AssetPackageError("DDPKG_FILE_INDEX_MISMATCH")
    for entry in manifest.files:
        size, digest = _member_digest(bundle, entry.path)
        if size != entry.size_bytes:
            raise AssetPackageError(f"DDPKG_FILE_SIZE_MISMATCH:{entry.path}")
        if digest != entry.sha256:
            raise AssetPackageError(f"DDPKG_FILE_HASH_MISMATCH:{entry.path}")
    if members[manifest.asset_ir_path].file_size > MAX_ASSET_IR_BYTES:
        raise AssetPackageError("DDPKG_ASSET_IR_SIZE_EXCEEDED")
    asset_ir = AssetIR.model_validate(
        _member_json(bundle, manifest.asset_ir_path, maximum_bytes=MAX_ASSET_IR_BYTES)
    )
    if asset_ir.asset_id != manifest.asset_id or asset_ir.kind != manifest.asset_kind:
        raise AssetPackageError("DDPKG_ASSET_IDENTITY_MISMATCH")
    if {entry.path: (entry.sha256, entry.size_bytes, entry.role) for entry in asset_ir.files} != {
        entry.path: (entry.sha256, entry.size_bytes, entry.role)
        for entry in manifest.files
        if entry.path != manifest.asset_ir_path
    }:
        raise AssetPackageError("DDPKG_ASSET_IR_FILE_INDEX_MISMATCH")
    detected = _detected_extensions(
        bundle,
        manifest.files,
        explicit_xml_paths={
            *asset_ir.interfaces.gazebo_entrypoints,
            *asset_ir.interfaces.ros2_entrypoints,
            *[
                target.entrypoint
                for target in asset_ir.simulation_targets
                if target.simulator.startswith("gazebo")
            ],
        },
    )
    declared_extensions = {
        (item.source_path, item.name, item.filename) for item in asset_ir.runtime_extensions
    }
    if any(
        (item.source_path, item.name, item.filename) not in declared_extensions for item in detected
    ):
        raise AssetPackageError("DDPKG_RUNTIME_EXTENSION_UNDECLARED")
    inspected = InspectedDDPkg(
        manifest=manifest,
        asset_ir=asset_ir,
        detected_runtime_extensions=detected,
    )
    return inspected


# 功能：
#   有界读取并解析成员 JSON，拒绝重复键、非有限数值和过深的元数据。
# 输入：
#   bundle：保持打开的 ZIP 读取器。
#   path：JSON 成员路径。
#   maximum_bytes：允许读取的最大实际字节数。
# 输出：
#   decoded：解析后的 JSON 值，具体字段契约由上层校验。
def _member_json(bundle: zipfile.ZipFile, path: str, *, maximum_bytes: int) -> object:
    try:
        with bundle.open(path) as source:
            raw = source.read(maximum_bytes + 1)
        decoded = decode_json(raw, limit=maximum_bytes, node_limit=2_000_000)
        return decoded
    except ValueError as error:
        raise AssetPackageError("DDPKG_JSON_INVALID") from error
