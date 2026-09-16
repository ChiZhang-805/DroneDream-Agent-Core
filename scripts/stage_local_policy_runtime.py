#!/usr/bin/env python3
"""Stage one admitted, content-bound local-expert ensemble for Runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path, PurePosixPath

from dronedream_agent_core.asset_package_storage import publish_asset_directory
from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.local_policy_packages import (
    LocalPolicyQualificationReceipt,
    LocalPolicySimulationAdmissionReceipt,
    load_local_policy_package,
    local_policy_receipt_supports_control_contract,
)
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    read_plugin_file,
)
from dronedream_agent_core.training.evidence_publication import (
    publish_evidence_bytes,
    write_evidence_object,
)
from dronedream_plugin_sdk.protocol import decode_json

REQUIRED_EXPERT_ROLES = {
    "local-navigation-policy",
    "precision-maneuver-policy",
    "recovery-policy",
    "risk-critic",
    "perception-encoder",
    "perception-health-critic",
    "settle-stability-critic",
    "payload-dynamics-adapter",
    "state-anomaly-detector",
    "cross-modal-consistency-critic",
}
MAXIMUM_PACKAGE_BYTES = 1024 * 1024 * 1024


# 功能：
#   核对逐专家的分发许可记录与实际权重绑定，保留完整许可文本，不自动赋予第三方 MIT 许可。
# 输入：
#   path：维护者提供的模型分发许可清单。
#   package：当前模型包。
# 输出：
#   content：验证后可随 Runtime 分发的原始清单字节。
def _verified_distribution_licenses(path: Path, package) -> bytes:
    content = read_plugin_file(path, limit=4 * 1024**2)
    value = decode_json(content, limit=4 * 1024**2)
    if (not isinstance(value, dict)
            or value.get("schema_version") != "dronedream.model-distribution-licenses.v1"
            or value.get("package_sha256") != package.package_sha256
            or not isinstance(value.get("artifacts"), list)):
        raise ValueError("LOCAL_POLICY_DISTRIBUTION_LICENSES_INVALID")
    expected = {artifact.role: artifact.sha256 for artifact in package.manifest.artifacts}
    seen = set()
    for entry in value["artifacts"]:
        if (not isinstance(entry, dict) or not isinstance(entry.get("role"), str)
                or entry["role"] in seen or entry["role"] not in expected
                or entry.get("sha256") != expected[entry["role"]]
                or entry.get("redistribution_approved") is not True
                or any(not isinstance(entry.get(field), str) or not entry[field].strip()
                       for field in ("license_id", "source", "license_text"))):
            raise ValueError("LOCAL_POLICY_DISTRIBUTION_LICENSE_ENTRY_INVALID")
        seen.add(entry["role"])
    if seen != expected.keys():
        raise ValueError("LOCAL_POLICY_DISTRIBUTION_LICENSES_INCOMPLETE")
    # 这是对维护者许可记录的内容核验，并非凭一个布尔字段判定法律授权成立。
    return content


# 功能：
#   核对十个专家训练来源和跨专家留出独立性，冻结随安装包保留的证据字节。
# 输入：
#   package：已验证文件摘要的模型包。
# 输出：
#   evidence：相对路径到已核对训练证据字节的映射。
def _verified_training_evidence(package) -> dict[str, bytes]:
    from dronedream_agent_core.training.artifact_assembly import (
        expert_spatial_groups,
        read_bound_content,
        validate_expert_training_receipt,
    )
    from dronedream_agent_core.training.ensemble_lineage import load_base_lineage

    lineage, digest = load_base_lineage(package)
    root = package.manifest_path.parent
    evidence = {"assembly-receipt.json": read_bound_content(
        root / "assembly-receipt.json", digest, maximum_bytes=4 * 1024**2,
    )}
    training, validation = set(), set()
    for artifact in package.manifest.artifacts:
        entry = lineage["expert_evidence"][artifact.role]
        relative = f"training-evidence/{artifact.role}.json"
        if (not isinstance(entry, dict) or entry.get("training_receipt_path") != relative
                or entry.get("artifact_sha256") != artifact.sha256):
            raise ValueError("LOCAL_POLICY_RUNTIME_EXPERT_LINEAGE_MISMATCH")
        content = read_bound_content(root / relative, entry.get("training_receipt_sha256"),
                                     maximum_bytes=4 * 1024**2)
        receipt = decode_json(content, limit=4 * 1024**2)
        validate_expert_training_receipt(artifact.role, artifact.sha256, receipt, package.manifest)
        if artifact.role != "perception-encoder":
            trained, held_out = expert_spatial_groups(artifact.role, receipt)
            training.update(trained)
            validation.update(held_out)
        evidence[relative] = content
    if training & validation:
        raise ValueError("LOCAL_POLICY_RUNTIME_CROSS_EXPERT_HOLDOUT_LEAKAGE")
    return evidence


# 功能：
#   有界复核暂存回执字节身份，不把路径及未受限读取当作内容证明。
# 输入：
#   path：本次独占目录中复制好的回执路径。
# 输出：
#   digest：最多 4 MiB 普通文件的实际摘要。
def _sha256(path: Path) -> str:
    digest = hash_plugin_file(path, limit=4 * 1024 * 1024)
    return digest


# 功能：
#   1. 核验完整专家包及资格、仿真范围和延迟，复制时固定已验证的回执字节。
#   2. 无覆盖发布 Runtime 目录；只准备资源，不安装、不训练，也不扩大原回执资格。
#   3. 只读预检不写输出；生产构建额外实载 ONNX 并验证训练来源。
# 输入：
#   无：命令行提供模型包、互斥的资格或仿真回执及新输出目录。
# 输出：
#   status：整套资源及目录索引发布成功时为零。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    receipt = parser.add_mutually_exclusive_group(required=True)
    receipt.add_argument("--qualification-receipt", type=Path)
    receipt.add_argument("--simulation-admission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--verify-onnx", action="store_true")
    parser.add_argument("--distribution-licenses", type=Path)
    args = parser.parse_args()
    if not args.check_only and args.output is None:
        parser.error("--output is required unless --check-only is selected")
    if args.check_only and args.output is not None:
        parser.error("--check-only does not accept --output")

    for name in ("package", "output"):
        if getattr(args, name) is None:
            continue
        check_plain_plugin_path(getattr(args, name))
        setattr(args, name, getattr(args, name).resolve())
    if args.output is not None and args.output.is_relative_to(args.package):
        raise ValueError("runtime output must stay outside the source package")
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    package = load_local_policy_package(args.package)
    roles = set(package.artifact_paths)
    if roles != REQUIRED_EXPERT_ROLES:
        missing = sorted(REQUIRED_EXPERT_ROLES - roles)
        extra = sorted(roles - REQUIRED_EXPERT_ROLES)
        raise ValueError(f"LOCAL_POLICY_RUNTIME_ENSEMBLE_INCOMPLETE:{missing}:{extra}")
    # 历史包仍可供离线读取，但安装入口只发布当前四轴控制包，不能以角色数量冒充控制能力。
    if (
        package.manifest.pilot_control_mode != "normalized-body-velocity"
        or package.manifest.control_feature_contract_sha256
        != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
    ):
        raise ValueError("LOCAL_POLICY_RUNTIME_CURRENT_CONTROL_REQUIRED")
    package_bytes = sum(path.stat().st_size for path in package.artifact_paths.values())
    if package_bytes <= 0 or package_bytes > MAXIMUM_PACKAGE_BYTES:
        raise ValueError("LOCAL_POLICY_RUNTIME_PACKAGE_SIZE_INVALID")

    receipt_path = args.simulation_admission or args.qualification_receipt
    receipt_content = read_plugin_file(receipt_path, limit=4 * 1024 * 1024)
    receipt_payload = decode_json(receipt_content, limit=4 * 1024 * 1024, node_limit=1_000_000)
    receipt_sha256 = hashlib.sha256(receipt_content).hexdigest()
    if args.simulation_admission is not None:
        admission = LocalPolicySimulationAdmissionReceipt.model_validate(receipt_payload)
        combined_latency = max(
            admission.combined_p99_inference_latency_ms or 0.0,
            admission.p99_inference_latency_ms
            + (admission.visual_preprocess_p99_latency_ms or 0.0)
            + (admission.visual_encoder_p99_inference_latency_ms or 0.0),
        )
        if (
            not admission.admitted_to_simulation
            or admission.issue_codes
            or admission.admission_scope != "standard-simulation"
            or admission.policy_package_sha256 != package.package_sha256
            or admission.vehicle_sha256 != package.manifest.vehicle_sha256
            or admission.sensor_contract_sha256 != package.manifest.sensor_contract_sha256
            or admission.map_sha256 != package.manifest.map_sha256
            or not local_policy_receipt_supports_control_contract(package, admission)
            or combined_latency > package.manifest.maximum_inference_latency_ms
        ):
            raise ValueError("LOCAL_POLICY_RUNTIME_SIMULATION_ADMISSION_INVALID")
        deployment_scope = "simulation-only"
        receipt_kind = "simulation-admission"
    else:
        assert args.qualification_receipt is not None
        qualification = LocalPolicyQualificationReceipt.model_validate(receipt_payload)
        if (
            not qualification.qualified
            or qualification.policy_package_sha256 != package.package_sha256
            or qualification.vehicle_sha256 != package.manifest.vehicle_sha256
            or qualification.sensor_contract_sha256 != package.manifest.sensor_contract_sha256
            or qualification.map_sha256 != package.manifest.map_sha256
            or not local_policy_receipt_supports_control_contract(package, qualification)
            or qualification.p99_inference_latency_ms
            > package.manifest.maximum_inference_latency_ms
        ):
            raise ValueError("LOCAL_POLICY_RUNTIME_QUALIFICATION_INVALID")
        deployment_scope = "production-qualified"
        receipt_kind = "qualification"

    # 安装预检必须实际加载当前推理接口，不能只凭清单中的名称和摘要认定图可执行。
    training_evidence = {}
    if args.verify_onnx:
        from dronedream_agent_core.local_policy_port import OnnxLocalPolicyBackend

        training_evidence = _verified_training_evidence(package)
        backend = OnnxLocalPolicyBackend(package, execution_providers=["CPUExecutionProvider"])
        backend.close()
        if args.distribution_licenses is None:
            raise ValueError("LOCAL_POLICY_DISTRIBUTION_LICENSES_REQUIRED")
    license_content = None
    if args.distribution_licenses is not None:
        license_content = _verified_distribution_licenses(args.distribution_licenses, package)
    if args.check_only:
        print(json.dumps({
            "package_sha256": package.package_sha256,
            "receipt_sha256": receipt_sha256,
            "deployment_scope": deployment_scope,
            "expert_count": len(roles),
            "model_bytes": package_bytes,
            "onnx_verified": args.verify_onnx,
        }, sort_keys=True))
        return 0

    assert args.output is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=args.output.parent,
        prefix=f".{args.output.name}.staging-",
    ) as temporary_directory:
        staged = Path(temporary_directory) / "local-policy"
        package_output = staged / "packages" / "general-navigation"
        receipt_output = staged / "receipts" / "active.json"
        package_output.mkdir(parents=True)
        receipt_output.parent.mkdir(parents=True)
        copied_bytes = 0
        for artifact in package.manifest.artifacts:
            source = package.artifact_paths[artifact.role]
            destination = package_output.joinpath(*PurePosixPath(artifact.relative_path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (
                hash_plugin_file(source, limit=MAXIMUM_PACKAGE_BYTES, destination=destination)
                != artifact.sha256
            ):
                raise RuntimeError("LOCAL_POLICY_RUNTIME_COPY_HASH_MISMATCH")
            copied_bytes += destination.stat().st_size
            if copied_bytes > MAXIMUM_PACKAGE_BYTES:
                raise ValueError("LOCAL_POLICY_RUNTIME_PACKAGE_SIZE_INVALID")
        manifest_content = read_plugin_file(package.manifest_path, limit=2 * 1024 * 1024)
        publish_evidence_bytes(
            package_output / "manifest.json", manifest_content, limit=2 * 1024 * 1024
        )
        publish_evidence_bytes(receipt_output, receipt_content, limit=4 * 1024 * 1024)
        if license_content is not None:
            publish_evidence_bytes(staged / "licenses.json", license_content, limit=4 * 1024**2)
        for relative, content in training_evidence.items():
            destination = package_output.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            publish_evidence_bytes(destination, content, limit=4 * 1024**2)
        if _sha256(receipt_output) != receipt_sha256:
            raise RuntimeError("LOCAL_POLICY_RUNTIME_RECEIPT_IDENTITY_CHANGED")
        copied = load_local_policy_package(package_output)
        if copied.package_sha256 != package.package_sha256:
            raise RuntimeError("LOCAL_POLICY_RUNTIME_PACKAGE_IDENTITY_CHANGED")
        catalog = {
            "schema_version": "dronedream.local-policy-runtime-catalog.v2",
            "deployment_scope": deployment_scope,
            "entries": [
                {
                    "package": "packages/general-navigation",
                    "package_sha256": package.package_sha256,
                    "receipt": {
                        "kind": receipt_kind,
                        "path": "receipts/active.json",
                        "sha256": receipt_sha256,
                    },
                }
            ],
        }
        write_evidence_object(staged / "catalog.json", catalog)
        publish_asset_directory(staged, args.output)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "package_sha256": package.package_sha256,
                "receipt_sha256": receipt_sha256,
                "deployment_scope": deployment_scope,
                "model_bytes": copied_bytes,
                "expert_count": len(roles),
            },
            sort_keys=True,
        )
    )
    status = 0
    return status


if __name__ == "__main__":
    raise SystemExit(main())
