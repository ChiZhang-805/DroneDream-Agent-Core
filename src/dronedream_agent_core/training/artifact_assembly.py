"""Build a complete current ensemble from explicit, independently trained artifacts.

No base package, filename search, weight conversion, inherited admission or
activation is implied. The recipe is content-bound; each source is read once
for hashing, validation and copying. A candidate is still subject to fresh
offline admission and physical simulation through the existing product gates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import Field, model_validator

from dronedream_plugin_sdk.protocol import copy_json, decode_json

from ..asset_package_storage import publish_asset_directory
from ..contracts import StrictModel
from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..hashing import sha256_json
from ..local_expert_harness import NAVIGATION_EXPERT_ROLES
from ..local_policy_packages import (
    LocalPolicyPackageManifest,
    PolicyArtifactRole,
    Sha256,
    load_local_policy_package,
)
from ..plugin_files import check_plain_plugin_path, read_plugin_file
from .causal_replay import REPLAY_FILES
from .mission_groups import SPATIAL_SPLIT_CONTRACT
from .policy_packaging import REQUIRED_RECURRENT_ENSEMBLE_ROLES
from .visual_lineage import VisualInputContract, require_matching_visual_input


class ExpertArtifactSource(StrictModel):
    """Explicit weight/receipt pair; paths are supplied, never discovered by filename guesses."""
    role: PolicyArtifactRole
    artifact_path: Path
    training_receipt_path: Path
    training_receipt_sha256: Sha256


class CompleteEnsembleRecipe(StrictModel):
    """Complete current-role manifest plus independent provenance for every model artifact."""
    manifest: LocalPolicyPackageManifest
    sources: list[ExpertArtifactSource] = Field(min_length=10, max_length=10)

    # 功能：
    #   1. 验证当前因果控制契约与十个互不重复的专家角色。
    #   2. 限制输出名称，避免模型覆盖清单或训练证据文件。
    # 输入：
    #   self：包含目标清单与十份来源记录的配方。
    # 输出：
    #   self：通过完整性和命名校验的配方。
    @model_validator(mode="after")
    def validate_current_complete_ensemble(self):
        manifest = self.manifest
        if (
            manifest.navigation_architecture != "causal-gru-control"
            or manifest.pilot_control_mode != "normalized-body-velocity"
            or manifest.control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
        ):
            raise ValueError("ENSEMBLE_REQUIRES_CURRENT_CAUSAL_CONTROL_CONTRACT")
        required = REQUIRED_RECURRENT_ENSEMBLE_ROLES
        if {s.role for s in self.sources} != required or {
            a.role for a in manifest.artifacts
        } != required:
            raise ValueError("ENSEMBLE_REQUIRES_EXACTLY_TEN_DISTINCT_ROLES")
        # Canonical output names avoid reserved manifest/evidence files, path
        # aliasing, Windows case collisions and links into an existing package.
        if any(a.relative_path != f"{a.role}.onnx" for a in manifest.artifacts):
            raise ValueError("ENSEMBLE_ARTIFACT_OUTPUT_NAMES_MUST_BE_CANONICAL")
        return self


# 功能：
#   读取有界普通文件并复核身份、大小和摘要，后续校验与复制使用同一份字节。
# 输入：
#   path：明确指定的模型或回执文件。
#   expected：预期的 SHA-256 摘要。
#   maximum_bytes：最多允许读取的字节数，范围为 1 至 1 GiB。
# 输出：
#   data：身份与摘要检查通过的文件字节。
def read_bound_content(path: Path, expected: str, *, maximum_bytes: int) -> bytes:
    # 负读取长度代表无限读取；先限制预算，再交给共享的身份绑定读取器。
    if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= 1024**3:
        raise ValueError("ENSEMBLE_SOURCE_BYTE_LIMIT_INVALID")
    if not _hash(expected):
        raise ValueError("ENSEMBLE_SOURCE_DIGEST_INVALID")
    check_plain_plugin_path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("ENSEMBLE_SOURCE_MUST_BE_REGULAR_FILE")
    data = read_plugin_file(path, limit=maximum_bytes)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("ENSEMBLE_SOURCE_CONTENT_CHANGED:" + path.name)
    return data


# 功能：
#   1. 检查 ONNX 图及嵌套张量，禁止未绑定在模型字节内的外部权重。
#   2. 核对具名输入输出与元数据唯一性；图通过校验不代表模型具备飞行能力。
# 输入：
#   data：已经按摘要绑定的 ONNX 字节。
#   input_names：清单声明的输入名称。
#   output_names：清单声明的输出名称。
# 输出：
#   None：不返回业务数据。
def validate_embedded_graph(data: bytes, *, input_names: list[str], output_names: list[str]):
    import onnx

    graph = onnx.load_model_from_string(data)

    # 功能：
    #   递归检查全部 protobuf 子消息，防止常量、稀疏张量或子图隐藏外部文件引用。
    # 输入：
    #   message：当前检查的 ONNX protobuf 消息。
    # 输出：
    #   None：不返回业务数据。
    def embedded(message):
        if message.DESCRIPTOR.full_name == "onnx.TensorProto" and (
            message.data_location == onnx.TensorProto.EXTERNAL or message.external_data
        ):
            raise ValueError("ENSEMBLE_EXTERNAL_WEIGHTS_NOT_CONTENT_BOUND")
        for field, value in message.ListFields():
            if field.message_type is not None:
                if field.is_repeated:
                    for item in value:
                        embedded(item)
                else:
                    embedded(value)

    embedded(graph)
    onnx.checker.check_model(graph)
    # Runtime feeds and requested outputs are named, not positional. Existing
    # producers may sort their manifest names without changing ONNX semantics.
    for tensors, names in ((graph.graph.input, input_names), (graph.graph.output, output_names)):
        actual = [item.name for item in tensors]
        if (
            len(names) != len(set(names))
            or len(actual) != len(set(actual))
            or set(actual) != set(names)
        ):
            raise ValueError("ENSEMBLE_GRAPH_TENSOR_NAMES_DIFFER_FROM_MANIFEST")
    keys = [item.key for item in graph.metadata_props]
    if len(keys) != len(set(keys)):
        raise ValueError("ENSEMBLE_GRAPH_METADATA_DUPLICATED")


# 功能：
#   判断一个值是否为标准小写 SHA-256 十六进制字符串。
# 输入：
#   value：待检查的摘要值。
# 输出：
#   valid：字符串长度和字符集是否符合摘要格式。
def _hash(value) -> bool:
    valid = isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
    return valid


# 功能：
#   验证训练与验证数据具有不同的有效摘要，不把内容独立性等同于空间独立性。
# 输入：
#   receipt：当前专家的训练回执。
# 输出：
#   None：不返回业务数据。
def _split_hashes(receipt):
    train, validation = receipt.get("training_data_sha256"), receipt.get("validation_data_sha256")
    if not _hash(train) or not _hash(validation) or train == validation:
        raise ValueError("ENSEMBLE_TRAINING_SPLITS_NOT_INDEPENDENT")


# 功能：
#   读取必须为对象的嵌套回执字段，拒绝空值、数组等错误结构。
# 输入：
#   receipt：包含该字段的回执对象。
#   field：需要读取的字段名。
# 输出：
#   value：通过对象类型检查的字段值。
def _object_field(receipt: dict, field: str) -> dict:
    value = receipt.get(field)
    if not isinstance(value, dict):
        raise ValueError("ENSEMBLE_TRAINING_FIELD_NOT_OBJECT:" + field)
    return value


# 功能：
#   验证当前空间划分契约，提取不重叠的训练与留出分组供跨专家检查。
# 输入：
#   role：当前控制或顾问专家角色。
#   receipt：该专家的训练回执。
# 输出：
#   groups：按训练、验证顺序排列的两个分组摘要集合。
def expert_spatial_groups(role, receipt):
    field = "dataset_split_method" if "advisors" in receipt else "split_contract"
    if receipt.get(field) != SPATIAL_SPLIT_CONTRACT:
        raise ValueError("ENSEMBLE_REQUIRES_CURRENT_SPATIAL_SPLIT_CONTRACT")
    metrics = _object_field(receipt, "metrics") if role in NAVIGATION_EXPERT_ROLES else receipt
    groups = []
    for split in ("training", "validation"):
        values = metrics.get(split + "_groups")
        if not isinstance(values, list) or not values or not all(_hash(v) for v in values):
            raise ValueError("ENSEMBLE_SPATIAL_GROUP_IDENTITIES_MISSING")
        groups.append(set(values))
    if groups[0] & groups[1]:
        raise ValueError("ENSEMBLE_SPATIAL_GROUPS_OVERLAP")
    return groups


# 功能：
#   1. 按专家类型检查训练来源、视觉契约、划分独立性与目标模型身份。
#   2. 拒绝非法 JSON、失败记录和自行宣称的飞行资格；不授予模型准入权限。
# 输入：
#   role：需要验证的专家角色。
#   digest：目标模型权重的预期摘要。
#   receipt：训练生产者提供的回执对象。
#   manifest：组装后的目标模型包清单。
# 输出：
#   None：不返回业务数据。
def validate_expert_training_receipt(role: str, digest: str, receipt: dict, manifest):
    if not isinstance(receipt, dict):
        raise ValueError("ENSEMBLE_TRAINING_RECEIPT_NOT_OBJECT")
    receipt = copy_json(receipt, limit=4 * 1024 * 1024)
    if role not in REQUIRED_RECURRENT_ENSEMBLE_ROLES or not _hash(digest):
        raise ValueError("ENSEMBLE_TRAINING_TARGET_INVALID")
    for key in ("qualification_granted", "qualified_for_flight", "flight_qualification_granted"):
        if key in receipt and receipt[key] is not False:
            raise ValueError("ENSEMBLE_TRAINING_CANNOT_GRANT_FLIGHT_QUALIFICATION")
    if "issue_codes" in receipt and not isinstance(receipt["issue_codes"], list):
        raise ValueError("ENSEMBLE_TRAINING_ISSUES_NOT_LIST")
    if receipt.get("issue_codes"):
        raise ValueError("ENSEMBLE_TRAINING_RECEIPT_HAS_FAILURES")
    if role != "perception-encoder":
        expert_spatial_groups(role, receipt)
    if role in NAVIGATION_EXPERT_ROLES:
        perception = next(a for a in manifest.artifacts if a.role == "perception-encoder")
        require_matching_visual_input(
            receipt.get("visual_input_contract"),
            VisualInputContract.from_manifest(manifest, perception.sha256).model_dump(),
        )
        config, metrics = _object_field(receipt, "config"), _object_field(receipt, "metrics")
        if (
            receipt.get("expert_role") != role
            or receipt.get("artifact_sha256") != digest
            or receipt.get("architecture") != "causal-gru-control"
            or manifest.navigation_architecture != "causal-gru-control"
            or manifest.pilot_control_mode != "normalized-body-velocity"
            or receipt.get("feature_contract_sha256") != manifest.control_feature_contract_sha256
            or receipt.get("qualified_for_flight") is not False
            or type(config.get("history_length")) is not int
            or type(config.get("visual_feature_count")) is not int
            or config.get("history_length") != manifest.navigation_history_length
            or config.get("visual_feature_count") != manifest.visual_feature_count
        ):
            raise ValueError("ENSEMBLE_NAVIGATION_TRAINING_IDENTITY_MISMATCH")
        for split in ("training", "validation"):
            count = metrics.get(f"{split}_window_count")
            groups = metrics.get(f"{split}_groups")
            if type(count) is not int or count < 1 or not isinstance(groups, list) or not groups:
                raise ValueError("ENSEMBLE_NAVIGATION_TRAINING_EVIDENCE_MISSING")
            if any(not isinstance(group, str) or not group.strip() for group in groups):
                raise ValueError("ENSEMBLE_NAVIGATION_GROUP_INVALID")
        if set(metrics["training_groups"]) & set(metrics["validation_groups"]):
            raise ValueError("ENSEMBLE_NAVIGATION_GROUPS_OVERLAP")
        inputs = receipt.get("input_sha256")
        if not isinstance(inputs, dict) or not inputs or not all(_hash(v) for v in inputs.values()):
            raise ValueError("ENSEMBLE_NAVIGATION_INPUT_IDENTITIES_MISSING")
        replay = receipt.get("replay_artifact_sha256")
        if (not isinstance(replay, dict) or set(replay) != set(REPLAY_FILES)
                or not all(_hash(v) for v in replay.values())):
            raise ValueError("ENSEMBLE_NAVIGATION_BOUND_REPLAY_MISSING")
    elif role == "risk-critic":
        if (
            receipt.get("purpose") != "native-action-risk-offline-training"
            or receipt.get("model_sha256") != digest
            or receipt.get("control_feature_contract_sha256")
            != manifest.control_feature_contract_sha256
            or receipt.get("optimized") is not True
            or receipt.get("offline_validation_passed") is not True
            or receipt.get("qualified_for_flight") is not False
            or receipt.get("independent_physical_validation_required") is not True
            or receipt.get("teacher_config_sha256") != sha256_json(receipt.get("teacher_config"))
        ):
            raise ValueError("ENSEMBLE_ACTION_RISK_TRAINING_IDENTITY_MISMATCH")
        splits = _object_field(receipt, "dataset_receipts_sha256")
        for name in ("training", "validation"):
            values = splits.get(name)
            if not isinstance(values, list) or not values or not all(_hash(v) for v in values):
                raise ValueError("ENSEMBLE_ACTION_RISK_DATASET_IDENTITIES_MISSING")
        if set(splits["training"]) & set(splits["validation"]):
            raise ValueError("ENSEMBLE_ACTION_RISK_SPLITS_OVERLAP")
    elif role == "perception-encoder":
        _split_hashes(receipt)
        config = _object_field(receipt, "config")
        if (
            receipt.get("artifact_sha256") != digest
            or receipt.get("training_accepted") is not True
            or receipt.get("flight_qualification_granted") is not False
            or receipt.get("split_method") != "held-out-complete-flight"
            or receipt.get("backbone_initialization") != "mobilenet-v3-large-imagenet1k-v2"
            or type(receipt.get("visual_feature_count")) is not int
            or type(config.get("width")) is not int
            or type(config.get("height")) is not int
            or receipt.get("visual_feature_count") != manifest.visual_feature_count
            or config.get("width") != manifest.visual_width
            or config.get("height") != manifest.visual_height
            or manifest.visual_normalization != "imagenet"
        ):
            raise ValueError("ENSEMBLE_VISUAL_TRAINING_IDENTITY_MISMATCH")
    else:
        _split_hashes(receipt)
        advisor = _object_field(_object_field(receipt, "advisors"), role)
        if (
            receipt.get("schema_version") != "dronedream.local-advisor-training-receipt.v1"
            or receipt.get("training_accepted") is not True
            or receipt.get("qualification_granted") is not False
            or receipt.get("fresh_admission_validation_required") is not True
            or receipt.get("feature_contract_sha256") != manifest.control_feature_contract_sha256
            or advisor.get("accepted") is not True
            or advisor.get("artifact_sha256") != digest
        ):
            raise ValueError("ENSEMBLE_ADVISOR_TRAINING_IDENTITY_MISMATCH")


# 功能：
#   1. 从十份独立来源组装候选包，绑定模型与训练证据并检查跨专家留出泄漏。
#   2. 用实际生产后端计算十专家接口探针后无覆盖发布，不修改当前选择或继承飞行资格。
# 输入：
#   recipe：已明确模型角色、摘要、输入输出和训练回执的完整配方。
#   source_root：相对来源路径的基准目录；显式绝对路径保持原含义。
#   output_root：必须尚不存在的候选包目录。
# 输出：
#   package：发布后重新加载并校验的候选模型包。
def assemble_complete_ensemble(
    *, recipe: CompleteEnsembleRecipe, source_root: Path, output_root: Path
):
    # 重验并冻结配方，不能信任 model_copy(update=...) 或之后被调用方改写的对象。
    recipe = CompleteEnsembleRecipe.model_validate(recipe.model_dump(mode="json"))
    check_plain_plugin_path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {a.role: a for a in recipe.manifest.artifacts}
    with TemporaryDirectory(prefix=".complete-ensemble-", dir=output_root.parent) as temporary:
        staging = Path(temporary) / "package"
        (staging / "training-evidence").mkdir(parents=True)
        evidence = {}
        training_groups, validation_groups = set(), set()
        for source in sorted(recipe.sources, key=lambda value: value.role):
            artifact = artifacts[source.role]
            data = read_bound_content(
                source_root / source.artifact_path,
                artifact.sha256,
                maximum_bytes=1024 * 1024 * 1024,
            )
            receipt_bytes = read_bound_content(
                source_root / source.training_receipt_path,
                source.training_receipt_sha256,
                maximum_bytes=4 * 1024 * 1024,
            )
            receipt = decode_json(receipt_bytes, limit=4 * 1024 * 1024)
            validate_expert_training_receipt(source.role, artifact.sha256, receipt, recipe.manifest)
            if source.role != "perception-encoder":
                # A locally independent split may still train on another expert's holdout.
                trained, held_out = expert_spatial_groups(source.role, receipt)
                training_groups.update(trained)
                validation_groups.update(held_out)
                if training_groups & validation_groups:
                    raise ValueError("ENSEMBLE_CROSS_EXPERT_TRAINING_VALIDATION_LEAKAGE")
            validate_embedded_graph(
                data, input_names=artifact.input_names, output_names=artifact.output_names
            )
            (staging / artifact.relative_path).write_bytes(data)
            receipt_name = f"training-evidence/{source.role}.json"
            (staging / receipt_name).write_bytes(receipt_bytes)
            evidence[source.role] = {
                "artifact_sha256": artifact.sha256,
                "training_receipt_sha256": source.training_receipt_sha256,
                "training_receipt_path": receipt_name,
            }
        (staging / "manifest.json").write_text(
            recipe.manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        package = load_local_policy_package(staging)
        from ..local_policy_port import OnnxLocalPolicyBackend

        backend = OnnxLocalPolicyBackend(package, execution_providers=["CPUExecutionProvider"])
        try:
            io_report = backend.verify_runtime_io()
        finally:
            backend.close()
        del backend
        lineage = {
            "purpose": "complete-current-ensemble-assembly",
            "recipe_sha256": sha256_json(recipe.model_dump(mode="json")),
            "package_sha256": package.package_sha256,
            "control_feature_contract_sha256": recipe.manifest.control_feature_contract_sha256,
            "expert_evidence": evidence,
            "runtime_io_probe": io_report,
            "control_expert_training_groups": sorted(training_groups),
            "control_expert_validation_groups": sorted(validation_groups),
            "inherited_admission": False,
            "fresh_offline_admission_required": True,
            "fresh_simulation_admission_required": True,
            "qualification_granted": False,
            "qualified_for_flight": False,
        }
        (staging / "assembly-receipt.json").write_text(
            json.dumps(lineage, indent=2), encoding="utf-8"
        )
        # No existing package, checkpoint, qualification or active selection changes.
        # 共用 Windows / Linux 无覆盖发布，Linux 不能使用会覆盖已有空目录的 rename。
        publish_asset_directory(staging, output_root)
    package = load_local_policy_package(output_root)
    return package
