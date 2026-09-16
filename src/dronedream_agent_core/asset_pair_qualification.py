"""Prepare and bind evidence for real map/vehicle qualification runs.

This module deliberately stops at the simulator execution boundary.  It turns two
already-quarantined DDPKG archives into an explicit, hash-bound PX4/Gazebo exercise
and can bind the resulting evidence back into new package versions.  The caller is
responsible for invoking :func:`dronedream_agent_core.gazebo_adapter.run_px4_gazebo_track`
inside the pinned DroneDream runtime; no imported source code is executed here.
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
from pathlib import Path
from typing import Annotated, Any, BinaryIO, Literal, TypeVar
from uuid import uuid4
from xml.etree import ElementTree

from pydantic import Field, field_validator, model_validator

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .asset_package_storage import (
    copy_indexed_asset_stream,
    extract_verified_asset,
    read_stored_manifest,
    verify_stored_asset,
)
from .asset_packages import (
    AssetFile,
    AssetIR,
    DDPkgManifest,
    InspectedDDPkg,
    QualificationBinding,
    ReadinessSummary,
    inspect_ddpkg,
    normalized_member_path,
    open_verified_ddpkg,
    package_content_sha256,
)
from .collision import validate_route_clearance
from .contracts import GraphRoute, MapAsset, RouteQuery, StrictModel, VehicleAsset
from .hashing import sha256_json
from .navigation import shortest_route
from .plugin_contracts import PluginHookReceipt, PluginSnapshot
from .plugin_files import check_plain_plugin_path, hash_plugin_file, read_plugin_file
from .px4_track import route_to_px4_track
from .xml_values import parse_xml

_MAX_CONTRACT_BYTES = 16 * 1024 * 1024
_MAX_XML_BYTES = 64 * 1024 * 1024
_MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ContractT = TypeVar("ContractT", bound=StrictModel)
AssetPairQualificationState = Literal[
    "created",
    "preparing",
    "running",
    "validating",
    "paused",
    "qualified",
    "failed",
    "cancelled",
]


class AssetPairQualificationError(ValueError):
    """A map/vehicle pair cannot yet cross the real-simulation boundary."""


class AssetQualificationInputs(StrictModel):
    schema_version: Literal["dronedream.asset-qualification-inputs.v1"] = (
        "dronedream.asset-qualification-inputs.v1"
    )
    map_asset_id: str
    map_content_sha256: Sha256
    vehicle_asset_id: str
    vehicle_content_sha256: Sha256
    world_sdf: str
    semantic: str
    graph: str
    vehicle_sdf: str
    vehicle_metadata: str
    controller_params: str
    world_name: str
    vehicle_name: str
    px4_sitl_model: str

    # 功能：
    #   限制计划中的文件路径只能表示工作区内的可移植相对文件。
    # 输入：
    #   cls：当前输入契约类型。
    #   value：尚未规范化的文件路径。
    # 输出：
    #   normalized：经过路径校验的相对路径。
    @field_validator(
        "world_sdf",
        "semantic",
        "graph",
        "vehicle_sdf",
        "vehicle_metadata",
        "controller_params",
        mode="before",
    )
    @classmethod
    def validate_input_path(cls, value: str) -> str:
        normalized = normalized_member_path(value)
        return normalized


class AssetPairQualificationPlan(StrictModel):
    schema_version: Literal["dronedream.asset-pair-qualification-plan.v1"] = (
        "dronedream.asset-pair-qualification-plan.v1"
    )
    qualification_id: str = Field(pattern=r"^asset-qualification-[0-9a-f]{24}$")
    created_at: datetime
    inputs: AssetQualificationInputs
    route: GraphRoute
    route_sha256: Sha256
    clearance_sha256: Sha256
    track_sha256: Sha256
    required_runtime_gates: list[str] = Field(min_length=1, max_length=64)

    # 功能：
    #   校验计划时间、路线摘要和最低验收条件，禁止计划自行缩减必需门禁。
    # 输入：
    #   self：待验证的资格计划。
    # 输出：
    #   self：通过一致性校验的计划对象。
    @model_validator(mode="after")
    def validate_plan(self) -> AssetPairQualificationPlan:
        if self.created_at.utcoffset() is None:
            raise ValueError("qualification plan requires an aware creation time")
        if len(set(self.required_runtime_gates)) != len(self.required_runtime_gates):
            raise ValueError("duplicate qualification gates")
        if not set(REQUIRED_RUNTIME_GATES).issubset(self.required_runtime_gates):
            raise ValueError("qualification plan omits required runtime gates")
        if self.route_sha256 != _sha256_bytes(_json_bytes(self.route.model_dump(mode="json"))):
            raise ValueError("qualification plan route hash mismatch")
        return self


class AssetQualificationPluginCheck(StrictModel):
    """One additive plugin verdict bound into the immutable pair receipt.

    Plugin checks may reject a pair or add evidence, but they never replace the
    non-negotiable runtime gates validated by this module.
    """

    check_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,159}$")
    plugin_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,119}$")
    capability_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,119}$")
    accepted: bool = Field(strict=True)
    issue_codes: list[str] = Field(default_factory=list, max_length=128)
    details: dict[str, Any] = Field(default_factory=dict)

    # 功能：
    #   保证插件判定与问题列表一致，并拒绝无法安全记录的详情数据。
    # 输入：
    #   self：插件提交的附加验收判定。
    # 输出：
    #   self：满足一致性约束的插件判定。
    @model_validator(mode="after")
    def validate_verdict(self) -> AssetQualificationPluginCheck:
        _json_bytes(self.details)
        if self.accepted and self.issue_codes:
            raise ValueError("accepted plugin checks cannot report issues")
        if not self.accepted and not self.issue_codes:
            raise ValueError("rejected plugin checks require issue codes")
        return self


class AssetPairQualificationReceipt(StrictModel):
    schema_version: Literal["dronedream.asset-pair-qualification-receipt.v1"] = (
        "dronedream.asset-pair-qualification-receipt.v1"
    )
    qualification_id: str = Field(pattern=r"^asset-qualification-[0-9a-f]{24}$")
    status: Literal["verified"]
    map_asset_id: str
    map_content_sha256: Sha256
    vehicle_asset_id: str
    vehicle_content_sha256: Sha256
    environment_versions: dict[str, str] = Field(min_length=1, max_length=64)
    runtime_evidence_sha256: Sha256
    runtime_evidence: dict[str, Any]
    plugin_snapshot: PluginSnapshot | None = None
    plugin_snapshot_sha256: Sha256 | None = None
    plugin_checks: list[AssetQualificationPluginCheck] = Field(default_factory=list, max_length=128)
    plugin_hook_receipts: list[PluginHookReceipt] = Field(default_factory=list, max_length=128)
    qualified_at: datetime

    # 功能：
    #   1. 校验运行证据摘要、固定门禁以及附加插件检查结果。
    #   2. 将插件回执逐一绑定到同一插件快照，拒绝重复或缺失的身份。
    # 输入：
    #   self：待校验的地图与飞机配对验收回执。
    # 输出：
    #   self：通过结构和绑定校验的回执对象。
    @model_validator(mode="after")
    def validate_runtime_evidence(self) -> AssetPairQualificationReceipt:
        if self.qualified_at.utcoffset() is None:
            raise ValueError("qualification receipt requires an aware time")
        if self.runtime_evidence_sha256 != _sha256_bytes(_json_bytes(self.runtime_evidence)):
            raise ValueError("runtime evidence hash mismatch")
        if self.runtime_evidence.get("status") != "verified":
            raise ValueError("runtime evidence must be verified")
        gates = self.runtime_evidence.get("gates")
        if (
            not isinstance(gates, dict)
            or not set(REQUIRED_RUNTIME_GATES).issubset(gates)
            or not all(value is True for value in gates.values())
        ):
            raise ValueError("all runtime evidence gates must be true")
        if any(not check.accepted for check in self.plugin_checks):
            raise ValueError("all additive plugin qualification checks must be accepted")
        has_plugin_evidence = bool(self.plugin_checks or self.plugin_hook_receipts)
        if has_plugin_evidence and (
            self.plugin_snapshot is None or self.plugin_snapshot_sha256 is None
        ):
            raise ValueError("plugin evidence requires the exact plugin snapshot and hash")
        if self.plugin_snapshot is not None:
            if self.plugin_snapshot_sha256 != sha256_json(self.plugin_snapshot):
                raise ValueError("plugin snapshot hash mismatch")
        elif self.plugin_snapshot_sha256 is not None:
            raise ValueError("plugin snapshot hash cannot exist without the snapshot")
        if any(receipt.outcome != "accepted" for receipt in self.plugin_hook_receipts):
            raise ValueError("qualification plugin hook receipts must be accepted")
        check_identity_list = [
            (check.plugin_id, check.capability_id) for check in self.plugin_checks
        ]
        receipt_identity_list = [
            (receipt.plugin_id, receipt.capability_id) for receipt in self.plugin_hook_receipts
        ]
        if len(check_identity_list) != len(set(check_identity_list)):
            raise ValueError("duplicate qualification plugin check identity")
        if len(receipt_identity_list) != len(set(receipt_identity_list)):
            raise ValueError("duplicate qualification plugin receipt identity")
        check_identities = set(check_identity_list)
        receipt_identities = set(receipt_identity_list)
        if check_identities != receipt_identities:
            raise ValueError("qualification plugin checks and hook receipts must match")
        if self.plugin_snapshot is not None:
            if any(entry.bundle_root is not None for entry in self.plugin_snapshot.plugins):
                raise ValueError("portable qualification snapshots cannot contain bundle paths")
            snapshot_by_plugin = {entry.plugin_id: entry for entry in self.plugin_snapshot.plugins}
            if len(snapshot_by_plugin) != len(self.plugin_snapshot.plugins):
                raise ValueError("plugin snapshot contains duplicate plugin identities")
            for receipt in self.plugin_hook_receipts:
                entry = snapshot_by_plugin.get(receipt.plugin_id)
                if entry is None:
                    raise ValueError("qualification receipt plugin is absent from snapshot")
                if (
                    receipt.plugin_version != entry.version
                    or receipt.plugin_package_sha256 != entry.package_sha256
                    or receipt.capability_id not in entry.capability_ids
                ):
                    raise ValueError("qualification receipt does not match plugin snapshot")
                if entry.manifest is not None and (
                    receipt.slot_id != entry.manifest.placement.slot_id
                    or receipt.capability_id
                    not in {capability.capability_id for capability in entry.manifest.capabilities}
                ):
                    raise ValueError("qualification receipt does not match plugin manifest")
        return self


class AssetPairQualificationJob(StrictModel):
    schema_version: Literal["dronedream.asset-pair-qualification-job.v1"] = (
        "dronedream.asset-pair-qualification-job.v1"
    )
    job_id: str = Field(pattern=r"^asset-qualification-job-[0-9a-f]{24}$")
    owner_id: str = Field(min_length=1, max_length=160)
    map_asset_id: str = Field(min_length=1, max_length=160)
    map_content_sha256: Sha256
    vehicle_asset_id: str = Field(min_length=1, max_length=160)
    vehicle_content_sha256: Sha256
    state: AssetPairQualificationState = "created"
    progress_percent: int = Field(default=0, ge=0, le=100, strict=True)
    revision: int = Field(default=0, ge=0, strict=True)
    qualification_id: str | None = Field(
        default=None, pattern=r"^asset-qualification-[0-9a-f]{24}$"
    )
    result_map_content_sha256: Sha256 | None = None
    result_vehicle_content_sha256: Sha256 | None = None
    issue_codes: list[str] = Field(default_factory=list, max_length=256)
    cancel_requested: bool = Field(default=False, strict=True)
    created_at: datetime
    updated_at: datetime

    # 功能：
    #   校验任务时间顺序，以及合格、失败两种终态必须携带的结果。
    # 输入：
    #   self：读取或更新后的资格任务。
    # 输出：
    #   self：满足终态及时间约束的任务对象。
    @model_validator(mode="after")
    def validate_terminal_result(self) -> AssetPairQualificationJob:
        if self.created_at.utcoffset() is None or self.updated_at.utcoffset() is None:
            raise ValueError("qualification job timestamps must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError("qualification job timestamps are reversed")
        if self.state == "qualified" and (
            self.progress_percent != 100
            or self.result_map_content_sha256 is None
            or self.result_vehicle_content_sha256 is None
        ):
            raise ValueError("qualified pair jobs require both resulting content hashes")
        if self.state == "failed" and not self.issue_codes:
            raise ValueError("failed pair jobs require issue codes")
        return self


_PAIR_JOB_TRANSITIONS: dict[AssetPairQualificationState, frozenset[AssetPairQualificationState]] = {
    "created": frozenset({"preparing", "cancelled"}),
    "preparing": frozenset({"running", "paused", "failed", "cancelled"}),
    "running": frozenset({"validating", "paused", "failed", "cancelled"}),
    "validating": frozenset({"qualified", "paused", "failed", "cancelled"}),
    "paused": frozenset({"preparing", "cancelled"}),
    "qualified": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


# 功能：
#   1. 按状态转换表生成下一版资格任务，不修改传入对象。
#   2. 拒绝进度倒退、时间倒退、非法输入类型及缺失的终态结果。
# 输入：
#   job：当前任务。
#   next_state：目标状态。
#   progress_percent：新进度，范围 0—100。
#   qualification_id：可选的新验收标识；不提供时沿用当前值。
#   result_map_content_sha256：可选的验收后地图包摘要。
#   result_vehicle_content_sha256：可选的验收后飞机包摘要。
#   issue_codes：本次问题代码列表。
#   cancel_requested：可选取消标记。
#   now：带时区的更新时间，未提供时使用当前 UTC 时间。
# 输出：
#   updated_job：经过重新校验、修订计数递增后的独立任务对象。
def transition_pair_qualification_job(
    job: AssetPairQualificationJob,
    next_state: AssetPairQualificationState,
    *,
    progress_percent: int,
    qualification_id: str | None = None,
    result_map_content_sha256: str | None = None,
    result_vehicle_content_sha256: str | None = None,
    issue_codes: list[str] | None = None,
    cancel_requested: bool | None = None,
    now: datetime | None = None,
) -> AssetPairQualificationJob:
    job = AssetPairQualificationJob.model_validate(job.model_dump(mode="python"))
    if not isinstance(next_state, str) or next_state not in _PAIR_JOB_TRANSITIONS[job.state]:
        raise AssetPairQualificationError(
            f"ASSET_QUALIFICATION_JOB_TRANSITION_INVALID:{job.state}:{next_state}"
        )
    if type(progress_percent) is not int or not 0 <= progress_percent <= 100:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_JOB_PROGRESS_INVALID")
    if issue_codes is not None and (
        not isinstance(issue_codes, list) or any(not isinstance(code, str) for code in issue_codes)
    ):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_JOB_ISSUES_INVALID")
    current_time = datetime.now(UTC) if now is None else now
    if (
        not isinstance(current_time, datetime)
        or current_time.utcoffset() is None
        or current_time < job.updated_at
    ):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_JOB_TIME_REGRESSION")
    if progress_percent < job.progress_percent:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_JOB_PROGRESS_REGRESSION")
    if next_state == "qualified" and progress_percent != 100:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_JOB_INCOMPLETE")
    if next_state == "failed" and not issue_codes:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_JOB_FAILURE_ISSUES_MISSING")
    value = job.model_dump(mode="python")
    value.update(
        {
            "state": next_state,
            "progress_percent": progress_percent,
            "revision": job.revision + 1,
            "qualification_id": (
                job.qualification_id if qualification_id is None else qualification_id
            ),
            "result_map_content_sha256": (
                job.result_map_content_sha256
                if result_map_content_sha256 is None
                else result_map_content_sha256
            ),
            "result_vehicle_content_sha256": (
                job.result_vehicle_content_sha256
                if result_vehicle_content_sha256 is None
                else result_vehicle_content_sha256
            ),
            "issue_codes": list(dict.fromkeys(issue_codes or [])),
            "cancel_requested": (
                cancel_requested if cancel_requested is not None else job.cancel_requested
            ),
            "updated_at": current_time,
        }
    )
    updated_job = AssetPairQualificationJob.model_validate(value)
    return updated_job


REQUIRED_RUNTIME_GATES = (
    "executor_completed",
    "offboard_timing_complete",
    "runtime_pose_samples_present",
    "ros_observations_present",
    "goal_observed",
    "landing_confirmed",
    "native_terminal_lifecycle_published",
    "no_live_abort",
    "px4_ulog_present",
    "static_route_clearance_bound",
)


# 功能：
#   计算指定字节的 SHA-256，用于绑定实际写入的资格材料。
# 输入：
#   payload：需要计算摘要的完整字节内容。
# 输出：
#   digest：小写十六进制 SHA-256 字符串。
def _sha256_bytes(payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    return digest


# 功能：
#   对本地资格材料进行有界流式摘要计算，并检查读取期间的文件变化。
# 输入：
#   path：需要计算摘要的普通文件路径。
# 输出：
#   digest：实际读取字节的 SHA-256 字符串。
def _sha256_file(path: Path) -> str:
    digest = hash_plugin_file(path, limit=_MAX_MEMBER_BYTES)
    return digest


# 功能：
#   1. 限制资格 JSON 的深度、节点数和字节数，拒绝非有限数值及循环引用。
#   2. 保持既有稳定序列化格式，避免仅修改排版就改变证据摘要。
# 输入：
#   value：由 JSON 基本类型构成的资格材料。
# 输出：
#   payload：稳定排序、UTF-8 编码且末尾带换行的 JSON 字节。
def _json_bytes(value: Any) -> bytes:
    encode_json(value, limit=_MAX_CONTRACT_BYTES, node_limit=2_000_000)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, separators=(",", ": "))
        + "\n"
    ).encode()
    if len(payload) > _MAX_CONTRACT_BYTES:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_JSON_TOO_LARGE")
    return payload


# 功能：
#   按预先验证的大小和摘要复制单个包成员，限制实际解压字节数。
# 输入：
#   bundle：保持打开的已校验 ZIP。
#   entry：该成员的预期元数据。
#   target：调用方独占创建的输出流。
# 输出：
#   copied：实际复制的字节数。
def _copy_verified_member(bundle: zipfile.ZipFile, entry: AssetFile, target: BinaryIO) -> int:
    try:
        with bundle.open(entry.path) as source:
            copied = copy_indexed_asset_stream(source, target, entry)
    except ValueError as error:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_MEMBER_CHANGED") from error
    return copied


# 功能：
#   1. 从同一已校验归档描述符提取文件，拒绝复用不同内容的旧检查结果。
#   2. 独占创建提取目录；失败后保留本次现场，不删除已有用户目录。
# 输入：
#   archive：已隔离的资产包。
#   destination：本次新建的提取目录。
#   inspected：调用方先前检查得到的包元数据。
# 输出：
#   None：不返回业务数据。
def _extract_verified(archive: Path, destination: Path, inspected: InspectedDDPkg) -> None:
    check_plain_plugin_path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        extract_verified_asset(archive, destination, inspected)
    except ValueError as error:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_EXTRACTION_INVALID") from error


# 功能：
#   从已校验的内部文件索引中筛选指定角色，不依据文件名猜测角色。
# 输入：
#   inspected：包检查结果。
#   role：需要筛选的文件角色。
# 输出：
#   paths：匹配角色的相对路径列表。
def _role_paths(inspected: InspectedDDPkg, role: str) -> list[str]:
    paths = [entry.path for entry in inspected.asset_ir.files if entry.role == role]
    return paths


# 功能：
#   有界读取一个 JSON 文件，拒绝重复键、非有限数值和过深嵌套。
# 输入：
#   path：本地 JSON 路径。
#   code：读取或解析失败时对外报告的稳定错误代码。
# 输出：
#   value：解析后的 JSON 基本类型数据。
def _load_json(path: Path, code: str) -> Any:
    try:
        payload = read_plugin_file(path, limit=_MAX_CONTRACT_BYTES)
        value = decode_json(payload, limit=_MAX_CONTRACT_BYTES, node_limit=2_000_000)
    except (OSError, ValueError, RecursionError) as error:
        raise AssetPairQualificationError(code) from error
    return value


# 功能：
#   查找唯一的指定 JSON 契约，多个同类契约必须消除歧义后才能继续。
# 输入：
#   root：包提取目录。
#   inspected：包检查结果。
#   schema_version：所需契约标识。
# 输出：
#   contract_path：唯一匹配契约的相对路径。
def _json_contract_path(root: Path, inspected: InspectedDDPkg, schema_version: str) -> str:
    candidates = [
        entry.path
        for entry in inspected.asset_ir.files
        if entry.media_type == "application/json" or entry.path.casefold().endswith(".json")
    ]
    matches: list[str] = []
    for candidate in candidates:
        try:
            value = _load_json(
                root / normalized_member_path(candidate), "ASSET_QUALIFICATION_CONTRACT_INVALID"
            )
        except AssetPairQualificationError:
            continue
        if isinstance(value, dict) and value.get("schema_version") == schema_version:
            matches.append(candidate)
    if not matches:
        raise AssetPairQualificationError(f"ASSET_QUALIFICATION_CONTRACT_MISSING:{schema_version}")
    if len(matches) != 1:
        raise AssetPairQualificationError(
            f"ASSET_QUALIFICATION_CONTRACT_AMBIGUOUS:{schema_version}"
        )
    contract_path = matches[0]
    return contract_path


# 功能：
#   读取并验证指定数据契约，对外只报告稳定错误代码。
# 输入：
#   path：本地契约路径。
#   model：目标契约类型。
#   code：失败时使用的错误代码。
# 输出：
#   contract：通过内容和类型校验的契约对象。
def _validated_contract(path: Path, model: type[ContractT], code: str) -> ContractT:
    try:
        contract = model.model_validate(_load_json(path, code))
    except (OSError, ValueError) as error:
        raise AssetPairQualificationError(code) from error
    return contract


# 功能：
#   判断候选值是否为有限正数，拒绝布尔值和无法表示的超大整数。
# 输入：
#   value：待检查的数值。
# 输出：
#   valid：数值是否满足有限且大于零的条件。
def _is_positive_finite(value: Any) -> bool:
    valid = type(value) in (int, float)
    if valid:
        try:
            valid = math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
    return valid


# 功能：
#   读取控制器参数并要求速度、加速度上限为有限正数，不执行控制指令。
# 输入：
#   path：控制器 JSON 路径；速度采用 m/s，加速度采用 m/s²。
# 输出：
#   value：经过校验的控制参数字典。
def _controller_contract(path: Path) -> dict[str, Any]:
    value = _load_json(path, "ASSET_QUALIFICATION_CONTROLLER_CONTRACT_INVALID")
    if not isinstance(value, dict) or not {"vel_limit", "accel_limit"}.issubset(value):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_CONTROLLER_CONTRACT_INVALID")
    for key in ("vel_limit", "accel_limit"):
        candidate = value[key]
        if not _is_positive_finite(candidate):
            raise AssetPairQualificationError("ASSET_QUALIFICATION_CONTROLLER_CONTRACT_INVALID")
    return value


# 功能：
#   阻止仅完成旧格式迁移的资产直接进入可执行验收，要求提供当前规范化结果。
# 输入：
#   inspected：包检查结果。
# 输出：
#   None：不返回业务数据。
def _require_current_normalized_contract(inspected: InspectedDDPkg) -> None:
    if "legacy-migrated" in inspected.asset_ir.capabilities or any(
        "/legacy/" in f"/{entry.path.casefold().strip('/')}/" for entry in inspected.asset_ir.files
    ):
        raise AssetPairQualificationError(
            "ASSET_QUALIFICATION_CURRENT_NORMALIZED_CONTRACT_REQUIRED"
        )


# 功能：
#   验证地图使用当前语义结构和运行绑定，拒绝混入已退役的绑定字段。
# 输入：
#   path：地图语义 JSON 的本地路径。
# 输出：
#   value：满足当前基础结构要求的语义字典。
def _current_map_semantic_contract(path: Path) -> dict[str, Any]:
    value = _load_json(path, "ASSET_QUALIFICATION_MAP_SEMANTIC_INVALID")
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "dronedream.map-semantic.v1"
        or "simulation_bindings" in value
        or not isinstance(value.get("runtime_bindings"), dict)
        or not isinstance(value.get("entities"), list)
        or not value["entities"]
    ):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_CURRENT_MAP_SEMANTIC_REQUIRED")
    return value


# 功能：
#   1. 校验飞机的物理字段、状态与感知传感器及负载能力是否齐备。
#   2. 拒绝控制器上限超过飞机能力；字段齐备不代替真实仿真验收。
# 输入：
#   inspected：飞机包检查结果。
#   vehicle：飞机能力契约。
#   controller：速度及加速度上限参数。
# 输出：
#   None：不返回业务数据。
def _validate_vehicle_static_contract(
    inspected: InspectedDDPkg,
    vehicle: VehicleAsset,
    controller: dict[str, Any],
) -> None:
    physics = inspected.asset_ir.physics
    if not (
        physics.link_count >= 1
        and physics.mass_entry_count >= 1
        and physics.inertia_entry_count >= 1
        and physics.collision_complete
        and physics.mass_complete
        and physics.inertia_complete
    ):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_VEHICLE_PHYSICS_INCOMPLETE")
    summary = inspected.asset_ir.vehicle
    if summary is None or (summary.rotor_count or 0) < 4:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_VEHICLE_DYNAMICS_INCOMPLETE")
    sensors = {value.casefold() for value in vehicle.sensors}
    if "imu" not in sensors or "odometry" not in sensors:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_VEHICLE_STATE_SENSORS_INCOMPLETE")
    perception_tokens = ("camera", "depth", "lidar", "stereo", "rgb")
    if not any(token in sensor for sensor in sensors for token in perception_tokens):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_VEHICLE_PERCEPTION_INCOMPLETE")
    if vehicle.max_takeoff_mass_kg + 1e-9 < (vehicle.dry_mass_kg + vehicle.max_pickup_payload_kg):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_VEHICLE_PAYLOAD_ENVELOPE_INVALID")
    if (
        float(controller["vel_limit"]) > vehicle.max_speed_mps
        or float(controller["accel_limit"]) > vehicle.max_acceleration_mps2
    ):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_CONTROLLER_EXCEEDS_VEHICLE_ENVELOPE")


# 功能：
#   要求指定文件角色恰好对应一个文件，缺失或重复时拒绝继续。
# 输入：
#   inspected：包检查结果。
#   role：需要的角色。
#   code：失败错误代码。
# 输出：
#   role_path：唯一匹配文件的相对路径。
def _one_role(inspected: InspectedDDPkg, role: str, code: str) -> str:
    candidates = _role_paths(inspected, role)
    if len(candidates) != 1:
        raise AssetPairQualificationError(code)
    role_path = candidates[0]
    return role_path


# 功能：
#   找到唯一声明的 Gazebo Harmonic 入口，不按文件名猜测世界或飞机模型。
# 输入：
#   inspected：包检查结果。
# 输出：
#   target_path：目标仿真入口的相对路径。
def _gazebo_target(inspected: InspectedDDPkg) -> str:
    candidates = [
        target.entrypoint
        for target in inspected.asset_ir.simulation_targets
        if target.simulator == "gazebo-harmonic"
    ]
    if len(candidates) != 1:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_GAZEBO_TARGET_REQUIRED")
    target_path = candidates[0]
    return target_path


# 功能：
#   有界解析 SDF 并读取唯一的顶层世界或模型名，不将嵌套部件当作另一架飞机。
# 输入：
#   path：SDF 文件路径。
#   entity：需要的顶层类型，仅允许 world 或 model。
# 输出：
#   entity_name：顶层世界或模型的名称。
def _xml_entity_name(path: Path, entity: str) -> str:
    try:
        root = parse_xml(
            read_plugin_file(path, limit=_MAX_XML_BYTES),
            maximum_bytes=_MAX_XML_BYTES,
            maximum_elements=1_000_000,
        )
    except (OSError, ValueError, ElementTree.ParseError) as error:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_SDF_INVALID") from error
    if entity not in {"world", "model"} or root.tag.rsplit("}", 1)[-1] != "sdf":
        raise AssetPairQualificationError("ASSET_QUALIFICATION_SDF_INVALID")
    values = [item.attrib.get("name") for item in root if item.tag.rsplit("}", 1)[-1] == entity]
    names = [value for value in values if isinstance(value, str) and value]
    if len(values) != 1 or len(names) != 1:
        raise AssetPairQualificationError(f"ASSET_QUALIFICATION_{entity.upper()}_NAME_AMBIGUOUS")
    entity_name = names[0]
    if (
        len(entity_name) > 256
        or entity_name != entity_name.strip()
        or any(ord(char) < 32 for char in entity_name)
    ):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_SDF_INVALID")
    return entity_name


# 功能：
#   读取唯一声明的 PX4 仿真机型，不根据飞机名称或旧资产推断默认机型。
# 输入：
#   inspected：飞机包检查结果。
# 输出：
#   model_name：符合 ASCII 标识规则的 PX4 SITL 机型名。
def _px4_sitl_model(inspected: InspectedDDPkg) -> str:
    prefix = "px4.sitl-model="
    declared = [
        value[len(prefix) :]
        for value in inspected.asset_ir.capabilities
        if value.startswith(prefix)
    ]
    if len(declared) != 1 or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,159}", declared[0]) is None:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_PX4_SITL_MODEL_REQUIRED")
    model_name = declared[0]
    return model_name


# 功能：
#   从当前地图的起飞语义及通用起飞别名中确定唯一节点，不偏向学校专用名称。
# 输入：
#   graph：当前地图导航图。
# 输出：
#   launch_node：唯一的起飞节点标识。
def _launch_node(graph: MapAsset) -> str:
    launch_nodes = {node.node_id for node in graph.nodes if node.semantic == "launch"}
    launch_nodes.update(
        graph.named_entities[key] for key in ("launch-pad", "launch") if key in graph.named_entities
    )
    if len(launch_nodes) != 1:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_LAUNCH_NODE_REQUIRED")
    launch_node = next(iter(launch_nodes))
    return launch_node


# 功能：
#   1. 搜索距离范围内可往返的候选节点，并检查静态路线净空。
#   2. 选择符合条件的最远候选，用于独立验收练习，不替代实时模型控制。
# 输入：
#   graph：当前地图图结构。
#   semantic_path：对应地图语义文件。
#   vehicle：飞机几何与能力契约。
#   minimum_translation_m：单程路线长度下限，单位米。
#   maximum_translation_m：单程路线长度上限，单位米。
# 输出：
#   round_trip：往返路线和该路线的静态净空检查结果组成的元组。
def _round_trip_route(
    graph: MapAsset,
    semantic_path: Path,
    vehicle: VehicleAsset,
    *,
    minimum_translation_m: float,
    maximum_translation_m: float,
) -> tuple[GraphRoute, Any]:
    launch = _launch_node(graph)
    candidates: list[tuple[float, GraphRoute, Any]] = []
    for node in graph.nodes:
        if node.node_id == launch:
            continue
        try:
            outbound = shortest_route(graph, RouteQuery(start_node=launch, goal_node=node.node_id))
        except ValueError:
            continue
        if not (minimum_translation_m <= outbound.route_length_m <= maximum_translation_m):
            continue
        try:
            inbound = shortest_route(graph, RouteQuery(start_node=node.node_id, goal_node=launch))
        except ValueError:
            # 有去路不代表有回路；单向死路不能让其余合格候选一起失败。
            continue
        route = GraphRoute(
            start_node=launch,
            goal_node=launch,
            node_ids=[*outbound.node_ids, *inbound.node_ids[1:]],
            edge_ids=[*outbound.edge_ids, *inbound.edge_ids],
            positions_m=[*outbound.positions_m, *inbound.positions_m[1:]],
            route_length_m=outbound.route_length_m + inbound.route_length_m,
            all_edges_flight_verified=outbound.all_edges_flight_verified
            and inbound.all_edges_flight_verified,
        )
        clearance = validate_route_clearance(
            route,
            semantic_path,
            vehicle_diameter_m=vehicle.body_radius_m * 2,
            vehicle_height_m=vehicle.body_height_m,
        )
        if clearance.accepted:
            candidates.append((outbound.route_length_m, route, clearance))
    if not candidates:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_CLEAR_ROUND_TRIP_REQUIRED")
    _, route, clearance = max(candidates, key=lambda item: item[0])
    round_trip = route, clearance
    return round_trip


# 功能：
#   独占写入本次准备目录中的新文件，并在返回前刷新数据到磁盘。
# 输入：
#   path：目标普通文件路径。
#   payload：要完整写入的字节。
# 输出：
#   None：不返回业务数据。
def _write_prepared_artifact(path: Path, payload: bytes) -> None:
    check_plain_plugin_path(path)
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


# 功能：
#   1. 校验并提取当前地图、飞机及控制器，生成静态往返练习与输入摘要。
#   2. 在独占新目录中保存计划；不启动仿真，也不自行授予飞行资格。
# 输入：
#   map_archive：已隔离的地图包路径。
#   vehicle_archive：已隔离的飞机包路径。
#   work_root：本次新建的计划目录，不允许复用已有目录。
#   minimum_translation_m：单程练习长度下限，单位米。
#   maximum_translation_m：单程练习长度上限，单位米。
# 输出：
#   plan：绑定输入资产、路线、净空、轨迹及必需运行门禁的资格计划。
def prepare_asset_pair_qualification(
    *,
    map_archive: Path,
    vehicle_archive: Path,
    work_root: Path,
    minimum_translation_m: float = 1.0,
    maximum_translation_m: float = 30.0,
) -> AssetPairQualificationPlan:
    if not _is_positive_finite(minimum_translation_m):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_MINIMUM_TRANSLATION_INVALID")
    if (
        not _is_positive_finite(maximum_translation_m)
        or maximum_translation_m < minimum_translation_m
    ):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_MAXIMUM_TRANSLATION_INVALID")
    check_plain_plugin_path(work_root)
    if work_root.exists():
        raise AssetPairQualificationError("ASSET_QUALIFICATION_WORKSPACE_NOT_EMPTY")
    work_root.mkdir(parents=True, exist_ok=False)

    map_inspected = inspect_ddpkg(map_archive)
    vehicle_inspected = inspect_ddpkg(vehicle_archive)
    if map_inspected.manifest.asset_kind not in {"map", "world"}:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_MAP_KIND_REQUIRED")
    if vehicle_inspected.manifest.asset_kind != "vehicle":
        raise AssetPairQualificationError("ASSET_QUALIFICATION_VEHICLE_KIND_REQUIRED")
    _require_current_normalized_contract(map_inspected)
    _require_current_normalized_contract(vehicle_inspected)

    map_root = work_root / "map"
    vehicle_root = work_root / "vehicle"
    _extract_verified(map_archive, map_root, map_inspected)
    _extract_verified(vehicle_archive, vehicle_root, vehicle_inspected)

    world_path = _gazebo_target(map_inspected)
    vehicle_sdf_path = _gazebo_target(vehicle_inspected)
    semantic_path = _one_role(
        map_inspected,
        "semantic",
        "ASSET_QUALIFICATION_MAP_SEMANTIC_REQUIRED",
    )
    graph_path = _json_contract_path(map_root, map_inspected, "dronedream.map-graph.v1")
    vehicle_metadata_path = _json_contract_path(
        vehicle_root, vehicle_inspected, "dronedream.vehicle.v1"
    )
    controller_path = _one_role(
        vehicle_inspected,
        "controller",
        "ASSET_QUALIFICATION_CONTROLLER_REQUIRED",
    )

    graph = _validated_contract(
        map_root / graph_path,
        MapAsset,
        "ASSET_QUALIFICATION_MAP_GRAPH_INVALID",
    )
    vehicle = _validated_contract(
        vehicle_root / vehicle_metadata_path,
        VehicleAsset,
        "ASSET_QUALIFICATION_VEHICLE_CONTRACT_INVALID",
    )
    if (
        graph.asset_id != map_inspected.manifest.asset_id
        or vehicle.asset_id != vehicle_inspected.manifest.asset_id
    ):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_CONTRACT_IDENTITY_MISMATCH")
    _current_map_semantic_contract(map_root / semantic_path)
    controller = _controller_contract(vehicle_root / controller_path)
    _validate_vehicle_static_contract(vehicle_inspected, vehicle, controller)

    route, clearance = _round_trip_route(
        graph,
        map_root / semantic_path,
        vehicle,
        minimum_translation_m=minimum_translation_m,
        maximum_translation_m=maximum_translation_m,
    )
    track = route_to_px4_track(
        route,
        graph,
        map_root / semantic_path,
        vehicle=vehicle,
        waypoint_hold_seconds=1.0,
    )

    route_payload = _json_bytes(route.model_dump(mode="json"))
    clearance_payload = _json_bytes(clearance.model_dump(mode="json"))
    track_payload = _json_bytes(track.model_dump(mode="json"))
    _write_prepared_artifact(work_root / "route.json", route_payload)
    _write_prepared_artifact(work_root / "clearance.json", clearance_payload)
    _write_prepared_artifact(work_root / "track.json", track_payload)

    seed = (
        map_inspected.manifest.content_sha256
        + vehicle_inspected.manifest.content_sha256
        + _sha256_bytes(route_payload)
    )
    qualification_id = f"asset-qualification-{hashlib.sha256(seed.encode()).hexdigest()[:24]}"
    plan = AssetPairQualificationPlan(
        qualification_id=qualification_id,
        created_at=datetime.now(UTC),
        inputs=AssetQualificationInputs(
            map_asset_id=map_inspected.manifest.asset_id,
            map_content_sha256=map_inspected.manifest.content_sha256,
            vehicle_asset_id=vehicle_inspected.manifest.asset_id,
            vehicle_content_sha256=vehicle_inspected.manifest.content_sha256,
            world_sdf=f"map/{world_path}",
            semantic=f"map/{semantic_path}",
            graph=f"map/{graph_path}",
            vehicle_sdf=f"vehicle/{vehicle_sdf_path}",
            vehicle_metadata=f"vehicle/{vehicle_metadata_path}",
            controller_params=f"vehicle/{controller_path}",
            world_name=_xml_entity_name(map_root / world_path, "world"),
            vehicle_name=_xml_entity_name(vehicle_root / vehicle_sdf_path, "model"),
            px4_sitl_model=_px4_sitl_model(vehicle_inspected),
        ),
        route=route,
        route_sha256=_sha256_bytes(route_payload),
        clearance_sha256=_sha256_bytes(clearance_payload),
        track_sha256=_sha256_bytes(track_payload),
        required_runtime_gates=list(REQUIRED_RUNTIME_GATES),
    )
    _write_prepared_artifact(
        work_root / "qualification-plan.json", _json_bytes(plan.model_dump(mode="json"))
    )
    return plan


# 功能：
#   复核准备目录中地图与飞机的全部文件，将实际运行材料绑定到计划的原包摘要。
# 输入：
#   plan：本次资产配对计划。
#   work_root：计划创建的隔离准备目录。
# 输出：
#   None：不返回业务数据。
def verify_prepared_asset_inputs(plan: AssetPairQualificationPlan, work_root: Path) -> None:
    plan = AssetPairQualificationPlan.model_validate(plan.model_dump(mode="python"))
    try:
        check_plain_plugin_path(work_root)
        root = work_root.resolve()
        bindings = (
            (
                "map",
                plan.inputs.map_asset_id,
                plan.inputs.map_content_sha256,
                {"map", "world"},
                (plan.inputs.world_sdf, plan.inputs.semantic, plan.inputs.graph),
            ),
            (
                "vehicle",
                plan.inputs.vehicle_asset_id,
                plan.inputs.vehicle_content_sha256,
                {"vehicle"},
                (
                    plan.inputs.vehicle_sdf,
                    plan.inputs.vehicle_metadata,
                    plan.inputs.controller_params,
                ),
            ),
        )
        for directory, asset_id, content_sha256, kinds, inputs in bindings:
            asset_root = root / directory
            manifest, _ = read_stored_manifest(asset_root)
            if (
                manifest.asset_id != asset_id
                or manifest.content_sha256 != content_sha256
                or manifest.asset_kind not in kinds
            ):
                raise ValueError("prepared asset identity mismatch")
            declared = {f"{directory}/{entry.path}" for entry in manifest.files}
            if not set(inputs).issubset(declared):
                raise ValueError("prepared input is not in the bound asset")
            # 路线摘要正确不代表模型、控制器、语义或其依赖仍是准备时的那一份。
            verify_stored_asset(asset_root, manifest)
    except (OSError, ValueError) as error:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_PREPARED_ASSET_INVALID") from error


# 功能：
#   1. 重新验证计划及运行证据，检查固定门禁和实际准备文件的摘要。
#   2. 生成同时绑定地图、飞机、运行环境及插件检查的回执，不执行新的仿真。
# 输入：
#   plan：本次资格计划。
#   work_root：准备材料目录。
#   runtime_evidence：仿真运行结果。
#   environment_versions：实际运行环境标识。
#   plugin_snapshot：可选插件快照。
#   plugin_snapshot_sha256：可选插件快照的摘要。
#   plugin_checks：可选附加检查结果。
#   plugin_hook_receipts：附加检查对应的插件调用回执。
#   qualified_at：带时区的验收时间，不提供时使用当前 UTC 时间。
# 输出：
#   receipt：经过验证且与调用方可变数据隔离的配对验收回执。
def build_pair_qualification_receipt(
    *,
    plan: AssetPairQualificationPlan,
    work_root: Path,
    runtime_evidence: dict[str, Any],
    environment_versions: dict[str, str],
    plugin_snapshot: dict[str, Any] | PluginSnapshot | None = None,
    plugin_snapshot_sha256: str | None = None,
    plugin_checks: list[dict[str, Any]] | None = None,
    plugin_hook_receipts: list[dict[str, Any]] | None = None,
    qualified_at: datetime | None = None,
) -> AssetPairQualificationReceipt:
    for optional_checks in (plugin_checks, plugin_hook_receipts):
        if optional_checks is not None and not isinstance(optional_checks, list):
            raise AssetPairQualificationError("ASSET_QUALIFICATION_PLUGIN_CHECKS_INVALID")
    if qualified_at is not None and (
        not isinstance(qualified_at, datetime) or qualified_at.utcoffset() is None
    ):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_TIME_INVALID")
    # Pydantic 对象本身可被修改，不能把“已经是模型实例”当作仍然有效的证明。
    plan = AssetPairQualificationPlan.model_validate(plan.model_dump(mode="python"))
    runtime_evidence = decode_json(
        _json_bytes(runtime_evidence), limit=_MAX_CONTRACT_BYTES, node_limit=2_000_000
    )
    if not isinstance(runtime_evidence, dict):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_RUNTIME_EVIDENCE_INVALID")
    missing = [
        gate
        for gate in plan.required_runtime_gates
        if not isinstance(runtime_evidence.get("gates"), dict)
        or runtime_evidence["gates"].get(gate) is not True
    ]
    if runtime_evidence.get("status") != "verified" or missing:
        suffix = ",".join(missing)
        raise AssetPairQualificationError(f"ASSET_QUALIFICATION_RUNTIME_GATES_FAILED:{suffix}")
    artifacts = runtime_evidence.get("artifacts")
    if not isinstance(artifacts, dict):
        raise AssetPairQualificationError("ASSET_QUALIFICATION_RUNTIME_ARTIFACTS_MISSING")
    verify_prepared_asset_inputs(plan, work_root)
    expected = {
        "world_sha256": _sha256_file(work_root / plan.inputs.world_sdf),
        "semantic_sha256": _sha256_file(work_root / plan.inputs.semantic),
        "vehicle_sha256": _sha256_file(work_root / plan.inputs.vehicle_sdf),
        "controller_params_sha256": _sha256_file(work_root / plan.inputs.controller_params),
        "route_sha256": plan.route_sha256,
        "track_sha256": plan.track_sha256,
        "clearance_sha256": plan.clearance_sha256,
    }
    mismatched = [key for key, value in expected.items() if artifacts.get(key) != value]
    if mismatched:
        raise AssetPairQualificationError(
            f"ASSET_QUALIFICATION_RUNTIME_ARTIFACT_MISMATCH:{','.join(mismatched)}"
        )
    for name, expected_hash in (
        ("route.json", plan.route_sha256),
        ("track.json", plan.track_sha256),
        ("clearance.json", plan.clearance_sha256),
    ):
        if _sha256_file(work_root / name) != expected_hash:
            raise AssetPairQualificationError(f"ASSET_QUALIFICATION_PLAN_ARTIFACT_CHANGED:{name}")
    if not environment_versions:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_ENVIRONMENT_REQUIRED")
    payload = _json_bytes(runtime_evidence)
    receipt = AssetPairQualificationReceipt(
        qualification_id=plan.qualification_id,
        status="verified",
        map_asset_id=plan.inputs.map_asset_id,
        map_content_sha256=plan.inputs.map_content_sha256,
        vehicle_asset_id=plan.inputs.vehicle_asset_id,
        vehicle_content_sha256=plan.inputs.vehicle_content_sha256,
        environment_versions=environment_versions,
        runtime_evidence_sha256=_sha256_bytes(payload),
        runtime_evidence=runtime_evidence,
        plugin_snapshot=(
            PluginSnapshot.model_validate(
                plugin_snapshot.model_dump(mode="python")
                if isinstance(plugin_snapshot, PluginSnapshot)
                else plugin_snapshot
            )
            if plugin_snapshot is not None
            else None
        ),
        plugin_snapshot_sha256=plugin_snapshot_sha256,
        plugin_checks=[
            AssetQualificationPluginCheck.model_validate(value) for value in (plugin_checks or [])
        ],
        plugin_hook_receipts=[
            PluginHookReceipt.model_validate(value) for value in (plugin_hook_receipts or [])
        ],
        qualified_at=datetime.now(UTC) if qualified_at is None else qualified_at,
    )
    receipt = AssetPairQualificationReceipt.model_validate_json(
        _json_bytes(receipt.model_dump(mode="json"))
    )
    return receipt


# 功能：
#   1. 重新校验回执，将其绑定到对应种类和内容摘要的源包。
#   2. 独占构建临时包，验证后原子发布；不覆盖已有目标或清理其他流程的文件。
# 输入：
#   source_archive：未绑定本回执的源资产包。
#   destination_archive：新资产包目标路径，必须尚不存在。
#   receipt：与源资产内容匹配的配对验收回执。
# 输出：
#   destination_archive：完成校验并发布的新 DDPKG 路径。
def bind_qualification_receipt(
    *,
    source_archive: Path,
    destination_archive: Path,
    receipt: AssetPairQualificationReceipt,
) -> Path:
    receipt = AssetPairQualificationReceipt.model_validate(receipt.model_dump(mode="python"))
    check_plain_plugin_path(destination_archive)
    if destination_archive.exists():
        raise AssetPairQualificationError("ASSET_QUALIFICATION_DESTINATION_EXISTS")
    inspected = inspect_ddpkg(source_archive)
    if inspected.manifest.asset_kind in {"map", "world"}:
        expected_id, expected_hash = receipt.map_asset_id, receipt.map_content_sha256
    else:
        expected_id, expected_hash = receipt.vehicle_asset_id, receipt.vehicle_content_sha256
    if inspected.manifest.asset_id != expected_id:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_RECEIPT_IDENTITY_MISMATCH")
    if inspected.manifest.content_sha256 != expected_hash:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_RECEIPT_CONTENT_MISMATCH")

    evidence_path = f"evidence/qualification/{receipt.qualification_id}.json"
    if evidence_path in {entry.path for entry in inspected.manifest.files}:
        raise AssetPairQualificationError("ASSET_QUALIFICATION_EVIDENCE_ALREADY_PRESENT")
    receipt_payload = _json_bytes(receipt.model_dump(mode="json"))
    evidence_file = AssetFile(
        path=evidence_path,
        role="evidence",
        media_type="application/json",
        sha256=_sha256_bytes(receipt_payload),
        size_bytes=len(receipt_payload),
    )
    asset_ir = inspected.asset_ir.model_copy(
        update={
            "files": [*inspected.asset_ir.files, evidence_file],
            "readiness": ReadinessSummary(
                maturity_ceiling="qualified",
                qualification_required=False,
                runtime_validation_performed=True,
                missing_fields=[],
                evidence_paths=[evidence_path],
            ),
        }
    )
    asset_ir = AssetIR.model_validate(asset_ir.model_dump(mode="python"))
    ir_payload = _json_bytes(asset_ir.model_dump(mode="json"))
    ir_file = AssetFile(
        path=inspected.manifest.asset_ir_path,
        role="asset_ir",
        media_type="application/json",
        sha256=_sha256_bytes(ir_payload),
        size_bytes=len(ir_payload),
    )
    files = [
        entry
        for entry in inspected.manifest.files
        if entry.path != inspected.manifest.asset_ir_path
    ]
    files.extend([evidence_file, ir_file])
    content_sha256 = package_content_sha256(files)
    manifest = DDPkgManifest(
        package_id=inspected.manifest.package_id,
        asset_id=inspected.manifest.asset_id,
        asset_kind=inspected.manifest.asset_kind,
        asset_ir_path=inspected.manifest.asset_ir_path,
        content_sha256=content_sha256,
        files=files,
        qualification=QualificationBinding(
            maturity="qualified",
            content_sha256=content_sha256,
            environment_versions=receipt.environment_versions,
            evidence_paths=[evidence_path],
            qualified_at=receipt.qualified_at,
        ),
        created_at=datetime.now(UTC),
    )
    manifest_payload = _json_bytes(manifest.model_dump(mode="json"))
    destination_archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_archive.with_name(f".{destination_archive.name}-{uuid4().hex}.tmp")
    owns_temporary = False
    try:
        with (
            open_verified_ddpkg(source_archive) as (source, current),
            zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as output,
        ):
            owns_temporary = True
            if current.model_dump(mode="json") != inspected.model_dump(mode="json"):
                raise AssetPairQualificationError("ASSET_QUALIFICATION_SOURCE_CHANGED")
            for entry in inspected.manifest.files:
                if entry.path == inspected.manifest.asset_ir_path:
                    continue
                with output.open(entry.path, "w", force_zip64=True) as member_output:
                    _copy_verified_member(source, entry, member_output)
            output.writestr(evidence_path, receipt_payload)
            output.writestr(inspected.manifest.asset_ir_path, ir_payload)
            output.writestr("manifest.json", manifest_payload)
        inspect_ddpkg(temporary)
        check_plain_plugin_path(destination_archive)
        # 硬链接发布要求目标不存在；竞争者先发布时，本次只能失败，不能覆盖它。
        os.link(temporary, destination_archive)
        with suppress(OSError):
            temporary.unlink()
        owns_temporary = False
        return destination_archive
    except BaseException:
        if owns_temporary:
            with suppress(OSError):
                temporary.unlink()
        raise
