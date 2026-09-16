"""Atomically assemble all recurrent manoeuvre roles without changing advisors."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from dronedream_plugin_sdk.protocol import copy_json, encode_json

from ..asset_package_storage import publish_asset_directory
from ..causal_control import CONTROL_HISTORY_CONTRACT_SHA256
from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..local_expert_harness import NAVIGATION_EXPERT_ROLES
from ..local_policy_packages import LocalPolicyPackageManifest, load_local_policy_package
from ..plugin_files import check_plain_plugin_path

REQUIRED_RECURRENT_ENSEMBLE_ROLES = {
    *NAVIGATION_EXPERT_ROLES,
    "risk-critic",
    "perception-encoder",
    "perception-health-critic",
    "settle-stability-critic",
    "payload-dynamics-adapter",
    "state-anomaly-detector",
    "cross-modal-consistency-critic",
}


# 功能：
#   检查基座包含当前特征契约、连续控制接口和必需专家，可额外核对视觉特征维数。
# 输入：
#   base_root：基座模型包目录。
#   visual_feature_count：可选的预期视觉特征数，只接受正整数。
# 输出：
#   base：通过兼容性检查的基座包。
def validate_causal_base(base_root: Path, *, visual_feature_count: int | None = None):
    if visual_feature_count is not None and (
        type(visual_feature_count) is not int or visual_feature_count < 1
    ):
        raise ValueError("CAUSAL_PACKAGE_VISUAL_WIDTH_INVALID")
    base = load_local_policy_package(base_root)
    if base.manifest.control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256:
        raise ValueError("CAUSAL_PACKAGE_BASE_FEATURE_CONTRACT_MISMATCH")
    if not set(base.artifact_paths) >= REQUIRED_RECURRENT_ENSEMBLE_ROLES:
        raise ValueError("CAUSAL_PACKAGE_BASE_EXPERTS_INCOMPLETE")
    if base.manifest.pilot_control_mode != "normalized-body-velocity":
        raise ValueError("CAUSAL_PACKAGE_BASE_REQUIRES_CONTINUOUS_CONTROL")
    if (
        visual_feature_count is not None
        and (base.manifest.visual_feature_count or 0) != visual_feature_count
    ):
        raise ValueError("CAUSAL_PACKAGE_VISUAL_WIDTH_MISMATCH")
    return base


# 功能：
#   1. 同时替换三个因果操纵专家，保留其余专家的权重、契约与可核查训练证据。
#   2. 冻结输入回执，验证完整暂存包后无覆盖发布；新包仍须重新通过准入与仿真。
# 输入：
#   base_root：包含完整来源证据的基座模型包。
#   navigation_models：三个操纵角色与新 ONNX 文件路径的映射。
#   output_root：必须尚不存在的候选包目录。
#   package_id：新候选包标识。
#   history_length：三个新专家一致使用的因果历史长度。
#   navigation_receipts：三个新专家分别绑定权重、训练划分和视觉输入的回执。
# 输出：
#   package：发布后重新加载并校验的候选模型包。
def assemble_causal_package(
    *,
    base_root: Path,
    navigation_models: dict[str, Path],
    output_root: Path,
    package_id: str,
    history_length: int,
    navigation_receipts: dict[str, dict] | None = None,
):
    import onnx

    from .artifact_assembly import read_bound_content, validate_embedded_graph

    base = validate_causal_base(base_root)
    if not isinstance(navigation_models, dict) or set(navigation_models) != set(
        NAVIGATION_EXPERT_ROLES
    ):
        raise ValueError("CAUSAL_PACKAGE_REQUIRES_ALL_THREE_MANOEUVRE_ROLES")
    navigation_models = dict(navigation_models)
    if not isinstance(navigation_receipts, dict) or set(navigation_receipts) != set(
        navigation_models
    ):
        raise ValueError("CAUSAL_PACKAGE_REQUIRES_ALL_TRAINING_RECEIPTS")
    # 权重读取、回执验证与落盘必须引用同一快照，不能在中途重新读取调用方字典。
    if any(not isinstance(value, dict) for value in navigation_receipts.values()):
        raise ValueError("ENSEMBLE_TRAINING_RECEIPT_NOT_OBJECT")
    navigation_receipts = {
        role: copy_json(value, limit=4 * 1024 * 1024) for role, value in navigation_receipts.items()
    }
    check_plain_plugin_path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".causal-policy-staging-", dir=output_root.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        payload = base.manifest.model_dump(mode="json")
        payload.update(
            package_id=package_id,
            navigation_architecture="causal-gru-control",
            navigation_history_length=history_length,
            navigation_history_contract_sha256=CONTROL_HISTORY_CONTRACT_SHA256,
        )
        entries = []
        for original in base.manifest.artifacts:
            entry = original.model_dump(mode="json")
            source = navigation_models.get(original.role, base.artifact_paths[original.role])
            expected_sha = (
                navigation_receipts[original.role].get("artifact_sha256")
                if original.role in navigation_models
                else original.sha256
            )
            source_content = read_bound_content(
                source, expected_sha, maximum_bytes=1024 * 1024 * 1024
            )
            if original.role in navigation_models:
                graph = onnx.load_model_from_string(source_content)
                metadata = {item.key: item.value for item in graph.metadata_props}
                if (
                    metadata.get("architecture") != "causal-gru-control"
                    or metadata.get("history_contract_sha256") != CONTROL_HISTORY_CONTRACT_SHA256
                    or metadata.get("history_length") != str(history_length)
                    or metadata.get("feature_contract_sha256")
                    != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
                ):
                    raise ValueError("CAUSAL_PACKAGE_MODEL_IDENTITY_MISMATCH")
                entry.update(
                    input_names=[i.name for i in graph.graph.input],
                    output_names=[i.name for i in graph.graph.output],
                )
            validate_embedded_graph(
                source_content, input_names=entry["input_names"], output_names=entry["output_names"]
            )
            destination = staging / original.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as target:
                if target.write(source_content) != len(source_content):
                    raise OSError("CAUSAL_PACKAGE_ARTIFACT_SHORT_WRITE")
            entry["sha256"] = hashlib.sha256(source_content).hexdigest()
            if original.role not in navigation_models and entry["sha256"] != original.sha256:
                raise ValueError("CAUSAL_PACKAGE_ADVISOR_CHANGED_DURING_COPY")
            entries.append(entry)
        payload["artifacts"] = entries
        manifest = LocalPolicyPackageManifest.model_validate(payload)
        from .artifact_assembly import validate_expert_training_receipt

        replacement_evidence = {}
        for entry in manifest.artifacts:
            if entry.role in navigation_models:
                receipt = navigation_receipts[entry.role]
                validate_expert_training_receipt(entry.role, entry.sha256, receipt, manifest)
                content = encode_json(receipt, limit=4 * 1024 * 1024).encode("utf-8")
                # 统一由完整来源组合器落盘一次，不能先写新证据再以覆盖方式重新发布。
                replacement_evidence[entry.role] = content
        (staging / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        loaded = load_local_policy_package(staging)
        # Execute tensor/metadata validation through the SAME production loader.
        from ..local_policy_port import OnnxLocalPolicyBackend

        backend = OnnxLocalPolicyBackend(loaded, execution_providers=["CPUExecutionProvider"])
        backend.close()
        del backend
        from .ensemble_lineage import preserve_complete_expert_lineage

        preserve_complete_expert_lineage(base, staging, loaded, replacement_evidence)
        (staging / "training-lineage.json").write_text(
            json.dumps(
                {
                    "schema_version": "dronedream.local-policy-training-receipt.v1",
                    "source_package_sha256": base.package_sha256,
                    "base_package_sha256": base.package_sha256,
                    "package_sha256": loaded.package_sha256,
                    "source_vehicle_sha256": base.manifest.vehicle_sha256,
                    "vehicle_sha256": manifest.vehicle_sha256,
                    "sensor_contract_sha256": manifest.sensor_contract_sha256,
                    "map_sha256": manifest.map_sha256,
                    "control_feature_contract_sha256": manifest.control_feature_contract_sha256,
                    "preserved_artifact_sha256": {
                        artifact.role: artifact.sha256
                        for artifact in manifest.artifacts
                        if artifact.role not in navigation_models
                    },
                    "replacement_roles": list(NAVIGATION_EXPERT_ROLES),
                    "replacement_training_receipts_sha256": {
                        role: hashlib.sha256(replacement_evidence[role]).hexdigest()
                        for role in NAVIGATION_EXPERT_ROLES
                    },
                    "replacement_artifact_sha256": {
                        artifact.role: artifact.sha256
                        for artifact in manifest.artifacts
                        if artifact.role in navigation_models
                    },
                    # Only manoeuvre architecture/weights change. The copied manifest
                    # binds identical advisor inputs, scales, vision and sensor layout.
                    # This permits evidence verification, NEVER automatic admission.
                    "preserved_advisor_input_contract_unchanged": True,
                    "fresh_simulation_admission_required": True,
                    "qualification_granted": False,
                    "qualified_for_flight": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Readers see the complete validated directory or nothing. Existing
        # packages, receipts, and weights are never overwritten or activated.
        publish_asset_directory(staging, output_root)
    package = load_local_policy_package(output_root)
    return package
