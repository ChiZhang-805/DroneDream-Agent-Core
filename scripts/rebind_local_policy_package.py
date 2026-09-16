#!/usr/bin/env python3
"""Bind an immutable local-expert ensemble to the current qualified vehicle.

This operation never trains, edits, or silently substitutes an ONNX artifact.
It creates a new, unqualified package identity whose only allowed change is the
vehicle binding, and records the complete byte-preservation lineage.  A fresh
offline simulation admission and a current-asset closed-loop run are still
required before the package can be staged for the product Runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from dronedream_agent_core.asset_package_storage import publish_asset_directory
from dronedream_agent_core.asset_packages import inspect_ddpkg
from dronedream_agent_core.contracts import VehicleAsset
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_policy_packages import (
    LocalPolicySimulationAdmissionReceipt,
    load_local_policy_package,
    local_policy_receipt_supports_control_contract,
)
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    read_plugin_file,
)
from dronedream_agent_core.training.ensemble_lineage import preserve_complete_expert_lineage
from dronedream_agent_core.training.evidence_publication import write_evidence_object
from dronedream_plugin_sdk.protocol import decode_json


# 功能：
#   对固定的本次载具包执行有界流式散列，检查普通文件身份而不无界读取。
# 输入：
#   path：本次私有载具包副本。
# 输出：
#   digest：最多 1 GiB 文件的实际 SHA-256。
def _sha256(path: Path) -> str:
    digest = hash_plugin_file(path, limit=1024 * 1024 * 1024)
    return digest


# 功能：
#   解析固定副本中的原生载具及资格绑定，严格读取唯一非 legacy 元数据，不沿用旧载具目录。
# 输入：
#   package_path：主入口在独占目录内冻结的载具 DDPKG。
# 输出：
#   vehicle：通过严格字段校验的载具模型。
#   evidence：载具包身份与原有资格证据引用，不授予新模型包资格。
def _qualified_vehicle(package_path: Path) -> tuple[VehicleAsset, dict[str, object]]:
    check_plain_plugin_path(package_path)
    inspected = inspect_ddpkg(package_path)
    manifest = inspected.manifest
    qualification = manifest.qualification
    if manifest.asset_kind != "vehicle":
        raise ValueError("LOCAL_POLICY_BINDING_VEHICLE_PACKAGE_REQUIRED")
    if qualification is None or qualification.maturity != "qualified":
        raise ValueError("LOCAL_POLICY_BINDING_QUALIFIED_VEHICLE_REQUIRED")
    if qualification.content_sha256 != manifest.content_sha256:
        raise ValueError("LOCAL_POLICY_BINDING_VEHICLE_QUALIFICATION_MISMATCH")
    if inspected.asset_ir.source.source_format != "dronedream-native":
        raise ValueError("LOCAL_POLICY_BINDING_CURRENT_NATIVE_VEHICLE_REQUIRED")
    candidates = [
        entry.path
        for entry in manifest.files
        if PurePosixPath(entry.path).name == "vehicle.json"
        and "/legacy/" not in f"/{entry.path.casefold()}"
    ]
    if len(candidates) != 1:
        raise ValueError("LOCAL_POLICY_BINDING_VEHICLE_METADATA_INVALID")
    with zipfile.ZipFile(package_path) as bundle, bundle.open(candidates[0]) as member:
        content = member.read(2 * 1024 * 1024 + 1)
    vehicle = VehicleAsset.model_validate(
        decode_json(content, limit=2 * 1024 * 1024, node_limit=100_000)
    )
    evidence = {
        "asset_id": manifest.asset_id,
        "content_sha256": manifest.content_sha256,
        "package_sha256": _sha256(package_path),
        "qualification_evidence_paths": qualification.evidence_paths,
        "qualification_environment_versions": qualification.environment_versions,
    }
    return vehicle, evidence


# 功能：
#   1. 核验旧包准入、固定载具字节并检查传感器与载荷契约，只改变新包标识及载具绑定。
#   2. 保留模型实际字节，无覆盖发布候选和来源回执；两者不是跨文件事务，仍须重新准入和闭环验证。
# 输入：
#   无：命令行提供来源包、原仿真回执、合格载具、新包及重绑定回执路径。
# 输出：
#   status：无资格继承的候选及完整来源回执发布成功时为零。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--source-simulation-admission", type=Path, required=True)
    parser.add_argument("--qualified-vehicle-package", type=Path, required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--rebinding-receipt", type=Path, required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--display-name", required=True)
    args = parser.parse_args()

    for name in ("source_package", "output_package", "rebinding_receipt"):
        check_plain_plugin_path(getattr(args, name))
        setattr(args, name, getattr(args, name).resolve())
    if (
        args.output_package.is_relative_to(args.source_package)
        or args.rebinding_receipt.is_relative_to(args.source_package)
        or args.rebinding_receipt.is_relative_to(args.output_package)
    ):
        raise ValueError("rebinding outputs must stay outside source and candidate packages")
    for output in (args.output_package, args.rebinding_receipt):
        if output.exists():
            raise FileExistsError(output)

    source = load_local_policy_package(args.source_package)
    admission_content = read_plugin_file(args.source_simulation_admission, limit=4 * 1024 * 1024)
    admission_sha256 = hashlib.sha256(admission_content).hexdigest()
    inherited = LocalPolicySimulationAdmissionReceipt.model_validate(
        decode_json(admission_content, limit=4 * 1024 * 1024, node_limit=1_000_000)
    )
    combined_latency = max(
        inherited.combined_p99_inference_latency_ms or 0.0,
        inherited.p99_inference_latency_ms
        + (inherited.visual_preprocess_p99_latency_ms or 0.0)
        + (inherited.visual_encoder_p99_inference_latency_ms or 0.0),
    )
    if (
        not inherited.admitted_to_simulation
        or inherited.issue_codes
        or inherited.admission_scope != "standard-simulation"
        or inherited.policy_package_sha256 != source.package_sha256
        or inherited.vehicle_sha256 != source.manifest.vehicle_sha256
        or inherited.sensor_contract_sha256 != source.manifest.sensor_contract_sha256
        or inherited.map_sha256 != source.manifest.map_sha256
        or not local_policy_receipt_supports_control_contract(source, inherited)
        or combined_latency > source.manifest.maximum_inference_latency_ms
    ):
        raise ValueError("LOCAL_POLICY_BINDING_SOURCE_ADMISSION_INVALID")
    if source.manifest.scope != "general" or source.manifest.map_sha256 is not None:
        raise ValueError("LOCAL_POLICY_BINDING_GENERAL_PACKAGE_REQUIRED")

    args.output_package.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=args.output_package.parent, prefix=".rebinding-vehicle-"
    ) as directory:
        vehicle_copy = Path(directory) / "vehicle.ddpkg"
        vehicle_digest = hash_plugin_file(
            args.qualified_vehicle_package, limit=1024 * 1024 * 1024, destination=vehicle_copy
        )
        vehicle, vehicle_evidence = _qualified_vehicle(vehicle_copy)
        if vehicle_evidence["package_sha256"] != vehicle_digest:
            raise ValueError("LOCAL_POLICY_BINDING_VEHICLE_IDENTITY_CHANGED")
    vehicle_sha256 = sha256_json(vehicle)
    if vehicle_sha256 == source.manifest.vehicle_sha256:
        raise ValueError("LOCAL_POLICY_BINDING_VEHICLE_UNCHANGED")
    required_sensors = {"imu", "magnetometer", "barometer", "gps", "odometry"}
    if "perception-encoder" in source.artifact_paths:
        required_sensors.add("oakd-lite-depth")
    if not required_sensors.issubset(set(vehicle.sensors)):
        raise ValueError("LOCAL_POLICY_BINDING_REQUIRED_SENSOR_MISSING")
    if vehicle.max_pickup_payload_kg <= 0 or vehicle.max_takeoff_mass_kg <= vehicle.dry_mass_kg:
        raise ValueError("LOCAL_POLICY_BINDING_PAYLOAD_CONTRACT_INVALID")

    args.output_package.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=args.output_package.parent,
        prefix=f".{args.output_package.name}.rebinding-",
    ) as temporary_directory:
        staged = Path(temporary_directory) / "package"
        staged.mkdir()
        remaining = 2 * 1024 * 1024 * 1024
        for artifact in source.manifest.artifacts:
            destination = staged.joinpath(*PurePosixPath(artifact.relative_path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hash_plugin_file(
                source.artifact_paths[artifact.role],
                limit=remaining,
                destination=destination,
            )
            remaining -= destination.stat().st_size
            if digest != artifact.sha256:
                raise ValueError("LOCAL_POLICY_BINDING_SOURCE_ARTIFACT_CHANGED")
        manifest = source.manifest.model_copy(
            update={
                "package_id": args.package_id,
                "display_name": args.display_name,
                "vehicle_sha256": vehicle_sha256,
            }
        )
        write_evidence_object(staged / "manifest.json", manifest.model_dump(mode="json"))
        rebound = load_local_policy_package(staged)
        source_hashes = {artifact.role: artifact.sha256 for artifact in source.manifest.artifacts}
        rebound_hashes = {artifact.role: artifact.sha256 for artifact in rebound.manifest.artifacts}
        if source_hashes != rebound_hashes:
            raise RuntimeError("LOCAL_POLICY_BINDING_ARTIFACT_CHANGED")
        # 训练事实仍属于相同权重，必须完整携带；旧载具的准入结论则不能随之继承。
        if source.manifest.navigation_architecture == "causal-gru-control":
            preserve_complete_expert_lineage(source, staged, rebound, {}, rebind_vehicle=True)
        publish_asset_directory(staged, args.output_package)

    rebound = load_local_policy_package(args.output_package)
    receipt = {
        "schema_version": "dronedream.local-policy-rebinding-receipt.v1",
        "source_package_id": source.manifest.package_id,
        "source_package_sha256": source.package_sha256,
        "source_vehicle_sha256": source.manifest.vehicle_sha256,
        "source_simulation_admission_sha256": admission_sha256,
        "package_id": rebound.manifest.package_id,
        "package_sha256": rebound.package_sha256,
        "vehicle_sha256": rebound.manifest.vehicle_sha256,
        "sensor_contract_sha256": rebound.manifest.sensor_contract_sha256,
        "map_sha256": rebound.manifest.map_sha256,
        "qualified_vehicle": vehicle_evidence,
        "preserved_artifact_sha256": {
            role: digest for role, digest in sorted(source_hashes.items())
        },
        "runtime_input_contract_unchanged": True,
        "deployment_scope": "unqualified",
        "fresh_simulation_admission_required": True,
        "current_asset_closed_loop_required": True,
        "qualification_granted": False,
    }
    args.rebinding_receipt.parent.mkdir(parents=True, exist_ok=True)
    write_evidence_object(args.rebinding_receipt, receipt)
    print(
        json.dumps(
            {
                "package": str(args.output_package),
                "package_sha256": rebound.package_sha256,
                "vehicle_sha256": rebound.manifest.vehicle_sha256,
                "qualification_granted": False,
            },
            sort_keys=True,
        )
    )
    status = 0
    return status


if __name__ == "__main__":
    raise SystemExit(main())
