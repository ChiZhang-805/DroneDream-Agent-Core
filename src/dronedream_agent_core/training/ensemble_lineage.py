"""Preserve complete expert evidence across atomic, explicit replacements."""

import hashlib

from dronedream_plugin_sdk.protocol import decode_json

from ..plugin_files import read_plugin_file
from .evidence_publication import publish_evidence_bytes, write_evidence_object


# 功能：
#   有界读取基座的完整来源回执，严格核对包摘要和十个专家角色，不按附近文件猜测来源。
# 输入：
#   base：已加载的基座模型包。
# 输出：
#   result：通过校验的来源对象及其原始字节摘要组成的元组。
def load_base_lineage(base):
    from .policy_packaging import REQUIRED_RECURRENT_ENSEMBLE_ROLES

    path = base.manifest_path.parent / "assembly-receipt.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("CAUSAL_COMPOSITION_REQUIRES_COMPLETE_TRAINING_LINEAGE")
    content = read_plugin_file(path, limit=4 * 1024 * 1024)
    lineage = decode_json(content, limit=4 * 1024 * 1024)
    if (
        not isinstance(lineage, dict)
        or lineage.get("package_sha256") != base.package_sha256
        or not isinstance(lineage.get("expert_evidence"), dict)
        or set(lineage.get("expert_evidence", {})) != REQUIRED_RECURRENT_ENSEMBLE_ROLES
    ):
        raise ValueError("CAUSAL_COMPOSITION_BASE_LINEAGE_MISMATCH")
    result = lineage, hashlib.sha256(content).hexdigest()
    return result


# 功能：
#   1. 将新专家的训练回执与未替换专家的内容绑定证据组合，保持十角色来源完整。
#   2. 重新检查各角色身份和跨专家留出独立性，写入仍需重新准入的组合回执。
# 输入：
#   base：原始基座包。
#   staged：调用方拥有的候选暂存目录。
#   package：已加载的候选模型包。
#   replacements：替换角色与已捕获训练回执字节的映射。
#   rebind_vehicle：明确只更换载具时允许零个权重替换，但重验其余契约完全不变。
# 输出：
#   None：不返回业务数据。
def preserve_complete_expert_lineage(
    base, staged, package, replacements: dict[str, bytes], *, rebind_vehicle: bool = False
):
    from .artifact_assembly import (
        expert_spatial_groups,
        read_bound_content,
        validate_expert_training_receipt,
    )
    from .policy_packaging import REQUIRED_RECURRENT_ENSEMBLE_ROLES

    lineage, lineage_sha = load_base_lineage(base)
    entries = lineage["expert_evidence"]
    if type(rebind_vehicle) is not bool:
        raise ValueError("CAUSAL_LINEAGE_REBINDING_FLAG_INVALID")
    if rebind_vehicle:
        unchanged = set(type(base.manifest).model_fields) - {
            "package_id",
            "display_name",
            "vehicle_sha256",
        }
        if (
            replacements
            or base.manifest.vehicle_sha256 == package.manifest.vehicle_sha256
            or any(
                getattr(base.manifest, name) != getattr(package.manifest, name)
                for name in unchanged
            )
        ):
            raise ValueError("CAUSAL_LINEAGE_REBINDING_CHANGED_MORE_THAN_VEHICLE")
    if (
        not isinstance(replacements, dict)
        or (not replacements and not rebind_vehicle)
        or not set(replacements) <= REQUIRED_RECURRENT_ENSEMBLE_ROLES
    ):
        raise ValueError("CAUSAL_COMPOSITION_REPLACEMENT_ROLES_INVALID")
    replacements = dict(replacements)
    if any(not isinstance(content, bytes) for content in replacements.values()):
        raise ValueError("CAUSAL_COMPOSITION_REPLACEMENT_RECEIPT_NOT_BYTES")
    if set(package.artifact_paths) != REQUIRED_RECURRENT_ENSEMBLE_ROLES:
        raise ValueError("CAUSAL_COMPOSITION_REQUIRES_EXACTLY_TEN_ROLES")
    training, validation, updated = set(), set(), {}
    (staged / "training-evidence").mkdir(exist_ok=True)
    for artifact in package.manifest.artifacts:
        role = artifact.role
        filename = f"training-evidence/{role}.json"
        if role in replacements:
            content = replacements[role]
        else:
            entry = entries[role]
            if (
                not isinstance(entry, dict)
                or entry.get("training_receipt_path") != filename
                or entry.get("artifact_sha256") != artifact.sha256
            ):
                raise ValueError("CAUSAL_COMPOSITION_PRESERVED_EXPERT_LINEAGE_MISMATCH")
            content = read_bound_content(
                base.manifest_path.parent / filename,
                entry.get("training_receipt_sha256"),
                maximum_bytes=4 * 1024 * 1024,
            )
        receipt = decode_json(content, limit=4 * 1024 * 1024)
        validate_expert_training_receipt(role, artifact.sha256, receipt, package.manifest)
        # Holdout separation must hold across experts, not just inside each
        # individual receipt, or composition could leak evaluation locations.
        if role != "perception-encoder":
            trained, held_out = expert_spatial_groups(role, receipt)
            training.update(trained)
            validation.update(held_out)
            if training & validation:
                raise ValueError("CAUSAL_COMPOSITION_CROSS_EXPERT_HOLDOUT_LEAKAGE")
        publish_evidence_bytes(staged / filename, content, limit=4 * 1024 * 1024)
        updated[role] = {
            "artifact_sha256": artifact.sha256,
            "training_receipt_sha256": hashlib.sha256(content).hexdigest(),
            "training_receipt_path": filename,
        }
    result = {
        "purpose": (
            "complete-current-ensemble-vehicle-rebinding"
            if rebind_vehicle
            else "complete-current-ensemble-composition"
        ),
        "package_sha256": package.package_sha256,
        "source_package_sha256": base.package_sha256,
        "source_assembly_receipt_sha256": lineage_sha,
        "control_feature_contract_sha256": package.manifest.control_feature_contract_sha256,
        "expert_evidence": updated,
        "control_expert_training_groups": sorted(training),
        "control_expert_validation_groups": sorted(validation),
        "inherited_admission": False,
        "fresh_offline_admission_required": True,
        "fresh_simulation_admission_required": True,
        "qualification_granted": False,
        "qualified_for_flight": False,
    }
    write_evidence_object(staged / "assembly-receipt.json", result)
