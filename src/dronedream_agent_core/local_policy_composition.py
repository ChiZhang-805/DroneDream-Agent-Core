"""Content-bound composition of independently trained local policy experts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .asset_package_storage import publish_asset_directory
from .local_policy_packages import (
    LoadedLocalPolicyPackage,
    LocalPolicyArtifact,
    LocalPolicyPackageManifest,
    PolicyArtifactRole,
    load_local_policy_package,
)
from .local_policy_port import OnnxLocalPolicyBackend
from .plugin_files import check_plain_plugin_path, hash_plugin_file
from .training.evidence_files import read_evidence_object
from .training.evidence_publication import write_evidence_object

COMPOSABLE_ADVISOR_ROLES: frozenset[PolicyArtifactRole] = frozenset(
    {
        "perception-health-critic",
        "settle-stability-critic",
        "payload-dynamics-adapter",
        "state-anomaly-detector",
        "cross-modal-consistency-critic",
    }
)


# 功能：
#   对普通模型文件进行有界流式散列，检查读取过程中身份和内容是否变化。
# 输入：
#   path：明确指定的模型文件。
# 输出：
#   digest：至多 1 GiB 文件实际字节的 SHA-256。
def _sha256(path: Path) -> str:
    digest = hash_plugin_file(path, limit=1024 * 1024 * 1024)
    return digest


@dataclass(frozen=True)
class LocalAdvisorArtifactEvidence:
    role: PolicyArtifactRole
    artifact_path: Path
    artifact_sha256: str
    training_receipt_path: Path
    training_receipt_sha256: str
    training_data_sha256: str
    validation_data_sha256: str
    validation_used_for_configuration_selection: bool


# 功能：
#   核验顾问模型与训练回执的身份、接受状态和数据摘要，拒绝重复键及宽松布尔转换。
# 输入：
#   role：允许组合的顾问角色，不能据此替换导航或风险控制模型。
#   artifact_path：顾问 ONNX 文件路径。
#   training_receipt_path：对应训练回执路径。
# 输出：
#   evidence：已绑定来源路径、摘要及验证集用途的不可变选择记录。
def load_local_advisor_artifact_evidence(
    *,
    role: PolicyArtifactRole,
    artifact_path: Path,
    training_receipt_path: Path,
) -> LocalAdvisorArtifactEvidence:
    if role not in COMPOSABLE_ADVISOR_ROLES:
        raise ValueError(f"local policy role is not a composable advisor: {role}")
    check_plain_plugin_path(artifact_path)
    check_plain_plugin_path(training_receipt_path)
    artifact_path = artifact_path.resolve(strict=True)
    training_receipt_path = training_receipt_path.resolve(strict=True)
    receipt, receipt_sha256 = read_evidence_object(training_receipt_path, limit=4 * 1024 * 1024)
    if receipt.get("schema_version") != "dronedream.local-advisor-training-receipt.v1":
        raise ValueError("local advisor training receipt schema is invalid")
    if receipt.get("training_accepted") is not True:
        raise ValueError("local advisor training receipt was not accepted")
    if receipt.get("qualification_granted") is not False:
        raise ValueError("training receipt cannot grant local policy qualification")
    if receipt.get("fresh_admission_validation_required") is not True:
        raise ValueError("local advisor receipt does not require fresh admission evidence")
    issues = receipt.get("issue_codes", [])
    selection_used = receipt.get("validation_used_for_configuration_selection", False)
    if type(issues) is not list or issues or type(selection_used) is not bool:
        raise ValueError("local advisor receipt has invalid issues or selection flag")
    records = receipt.get("advisors")
    record = records.get(role) if isinstance(records, dict) else None
    if not isinstance(record, dict) or record.get("accepted") is not True:
        raise ValueError(f"local advisor training record was not accepted: {role}")
    artifact = artifact_path
    if not artifact.is_file() or artifact.is_symlink() or artifact.suffix.lower() != ".onnx":
        raise ValueError("local advisor artifact is not a regular ONNX file")
    expected_name = record.get("artifact")
    if expected_name != artifact.name:
        raise ValueError("local advisor artifact name differs from its training receipt")
    artifact_sha256 = _sha256(artifact)
    if record.get("artifact_sha256") != artifact_sha256:
        raise ValueError("local advisor artifact hash differs from its training receipt")
    training_data_sha256 = receipt.get("training_data_sha256")
    validation_data_sha256 = receipt.get("validation_data_sha256")
    if (
        not all(
            isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
            for value in (training_data_sha256, validation_data_sha256)
        )
        or training_data_sha256 == validation_data_sha256
    ):
        raise ValueError("local advisor receipt data hashes are invalid")
    evidence = LocalAdvisorArtifactEvidence(
        role=role,
        artifact_path=artifact,
        artifact_sha256=artifact_sha256,
        training_receipt_path=training_receipt_path,
        training_receipt_sha256=receipt_sha256,
        training_data_sha256=str(training_data_sha256),
        validation_data_sha256=str(validation_data_sha256),
        validation_used_for_configuration_selection=selection_used,
    )
    return evidence


# 功能：
#   1. 按明确选择替换顾问图，保留其他模型原始字节及因果控制契约，不转换成旧前馈结构。
#   2. 验证实际后端和因果集合来源，再无覆盖发布候选与回执，不授予运行或飞行权限。
# 输入：
#   base_package_path：已有基座目录。
#   output_package_path：不在基座内部且尚不存在的候选目录。
#   composition_receipt_path：位于两个模型包之外的独立新回执路径。
#   package_id：新包标识。
#   display_name：新包显示名称。
#   additions：至多五种、角色互不重复的内容绑定顾问证据。
# 输出：
#   package：发布后重新加载的候选包。
def compose_local_policy_advisors(
    *,
    base_package_path: Path,
    output_package_path: Path,
    composition_receipt_path: Path,
    package_id: str,
    display_name: str,
    additions: tuple[LocalAdvisorArtifactEvidence, ...],
) -> LoadedLocalPolicyPackage:
    additions = tuple(additions)
    if not 1 <= len(additions) <= len(COMPOSABLE_ADVISOR_ROLES):
        raise ValueError("local policy composition requires at least one advisor")
    if len({item.role for item in additions}) != len(additions):
        raise ValueError("local policy composition contains duplicate advisor roles")
    for path in (base_package_path, output_package_path, composition_receipt_path):
        check_plain_plugin_path(path)
    base_package_path = base_package_path.resolve(strict=True)
    output_package_path = output_package_path.resolve()
    composition_receipt_path = composition_receipt_path.resolve()
    if (
        output_package_path.is_relative_to(base_package_path)
        or composition_receipt_path.is_relative_to(base_package_path)
        or composition_receipt_path.is_relative_to(output_package_path)
    ):
        raise ValueError("composition outputs must remain outside source and candidate packages")
    if output_package_path.exists():
        raise FileExistsError(output_package_path)
    if composition_receipt_path.exists():
        raise FileExistsError(composition_receipt_path)
    base = load_local_policy_package(base_package_path)
    if "risk-critic" not in base.artifact_paths:
        raise ValueError("local policy composition requires an independent risk critic")
    from .training.artifact_assembly import read_bound_content, validate_embedded_graph

    replacement_receipts = {}
    for item in additions:
        checked = load_local_advisor_artifact_evidence(
            role=item.role,
            artifact_path=item.artifact_path,
            training_receipt_path=item.training_receipt_path,
        )
        if checked != item:
            raise ValueError("local advisor evidence changed after selection")
        replacement_receipts[item.role] = read_bound_content(
            item.training_receipt_path, item.training_receipt_sha256, maximum_bytes=4 * 1024 * 1024
        )
    output_package_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=output_package_path.parent,
        prefix=f".{output_package_path.name}.composition-",
    ) as temporary_root:
        staged = Path(temporary_root) / "package"
        # Composition is byte-preserving, not model conversion. In particular,
        # a causal graph must never pass through the old feedforward serializer
        # or lose its feature/history contracts while replacing one advisor.
        import onnx

        staged.mkdir()
        entries = {item.role: item for item in base.manifest.artifacts}
        sources = dict(base.artifact_paths)
        for item in additions:
            data = read_bound_content(
                item.artifact_path, item.artifact_sha256, maximum_bytes=1024 * 1024 * 1024
            )
            graph = onnx.load_model_from_string(data)
            entries[item.role] = LocalPolicyArtifact(
                role=item.role,
                relative_path=f"{item.role}.onnx",
                sha256=item.artifact_sha256,
                input_names=[v.name for v in graph.graph.input],
                output_names=[v.name for v in graph.graph.output],
            )
            sources[item.role] = item.artifact_path
        paths = [entry.relative_path.casefold() for entry in entries.values()]
        if len(paths) != len(set(paths)):
            raise ValueError("composed artifacts have colliding output paths")
        for role, entry in entries.items():
            content = read_bound_content(
                sources[role], entry.sha256, maximum_bytes=1024 * 1024 * 1024
            )
            validate_embedded_graph(
                content, input_names=entry.input_names, output_names=entry.output_names
            )
            destination = staged / entry.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        payload = base.manifest.model_dump(mode="json")
        payload.update(
            package_id=package_id,
            display_name=display_name,
            artifacts=[entry.model_dump(mode="json") for entry in entries.values()],
        )
        manifest = LocalPolicyPackageManifest.model_validate(payload)
        (staged / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        staged_package = load_local_policy_package(staged)
        backend = OnnxLocalPolicyBackend(
            staged_package,
            execution_providers=["CPUExecutionProvider"],
        )
        backend.close()
        del backend
        base_hashes = {artifact.role: artifact.sha256 for artifact in base.manifest.artifacts}
        staged_hashes = {
            artifact.role: artifact.sha256 for artifact in staged_package.manifest.artifacts
        }
        changed_roles = {item.role for item in additions}
        for role, digest in base_hashes.items():
            if role not in changed_roles and staged_hashes.get(role) != digest:
                raise RuntimeError(f"local policy composition changed base artifact: {role}")
        for evidence in additions:
            if staged_hashes.get(evidence.role) != evidence.artifact_sha256:
                raise RuntimeError(
                    f"composed advisor hash differs from training artifact: {evidence.role}"
                )
        if base.manifest.navigation_architecture == "causal-gru-control":
            from .training.ensemble_lineage import preserve_complete_expert_lineage

            preserve_complete_expert_lineage(base, staged, staged_package, replacement_receipts)
        publish_asset_directory(staged, output_package_path)

    package = load_local_policy_package(output_package_path)
    package_hashes = {artifact.role: artifact.sha256 for artifact in package.manifest.artifacts}
    changed_roles = {item.role for item in additions}
    preserved_hashes = {
        role: digest for role, digest in sorted(base_hashes.items()) if role not in changed_roles
    }
    if any(package_hashes.get(role) != digest for role, digest in preserved_hashes.items()):
        raise RuntimeError("composed package does not preserve its declared base artifacts")
    receipt = {
        "schema_version": "dronedream.local-policy-composition-receipt.v1",
        "source_package_id": base.manifest.package_id,
        "source_package_sha256": base.package_sha256,
        "package_id": package.manifest.package_id,
        "package_sha256": package.package_sha256,
        "vehicle_sha256": package.manifest.vehicle_sha256,
        "sensor_contract_sha256": package.manifest.sensor_contract_sha256,
        "preserved_artifact_sha256": preserved_hashes,
        "advisor_evidence": [
            {
                "role": item.role,
                "artifact_sha256": item.artifact_sha256,
                "training_receipt_sha256": item.training_receipt_sha256,
                "training_data_sha256": item.training_data_sha256,
                "validation_data_sha256": item.validation_data_sha256,
                "validation_used_for_configuration_selection": (
                    item.validation_used_for_configuration_selection
                ),
            }
            for item in sorted(additions, key=lambda value: value.role)
        ],
        "fresh_admission_validation_required": True,
        "qualification_granted": False,
    }
    composition_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    # 包和外部回执是两个独立发布对象；回执冲突保留候选用于诊断，不能覆盖旧回执。
    write_evidence_object(composition_receipt_path, receipt)
    return package


__all__ = [
    "COMPOSABLE_ADVISOR_ROLES",
    "LocalAdvisorArtifactEvidence",
    "compose_local_policy_advisors",
    "load_local_advisor_artifact_evidence",
]
