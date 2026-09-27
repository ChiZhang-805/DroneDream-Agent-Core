"""Self-contained source evidence for a composed precision graph, not flight admission."""

import base64
import binascii
import hashlib

from dronedream_plugin_sdk.protocol import copy_json

from ..heading_context import HEADING_CONTEXT_SHA256
from ..local_policy_packages import LocalPolicyPackageManifest
from .mission_groups import SPATIAL_SPLIT_CONTRACT

COMPOSITION_PURPOSE = "frozen-precision-heading-composition"
NAVIGATION_COMPOSITION_PURPOSE = "frozen-navigation-heading-composition"


# 功能：把已声明组合用途映射到唯一控制角色，拒绝来源证据的角色互换。
# 输入：receipt：组合记录。输出：精细或巡航角色。
def composition_role(receipt):
    purposes = {COMPOSITION_PURPOSE: 'precision-maneuver-policy',
                NAVIGATION_COMPOSITION_PURPOSE: 'local-navigation-policy'}
    if type(receipt) is not dict or receipt.get('purpose') not in purposes:
        raise ValueError('PRECISION_LINEAGE_PURPOSE_INVALID')
    return purposes[receipt['purpose']]


# 功能：
#   从组合证据中提取两分支的全部训练及开发分组，拒绝跨分支留出泄漏。
# 输入：
#   receipt：包含原平移训练回执和偏航来源的组合记录。
# 输出：
#   groups：训练组与开发验证组的两个集合。
def composition_spatial_groups(receipt):
    from .artifact_assembly import expert_spatial_groups

    role = composition_role(receipt)
    translation = receipt.get("translation_training_receipt")
    heading = receipt.get("heading_training_lineage")
    if (
        type(translation) is not dict
        or translation.get("purpose") in (COMPOSITION_PURPOSE, NAVIGATION_COMPOSITION_PURPOSE)
        or type(heading) is not dict
        or heading.get("split_contract") != SPATIAL_SPLIT_CONTRACT
    ):
        raise ValueError("PRECISION_LINEAGE_SOURCE_INVALID")
    groups = expert_spatial_groups(role, translation)
    for index, name in enumerate(("training_groups", "validation_groups")):
        values = heading.get(name)
        if (
            type(values) is not list
            or not values
            or len(values) > 10000
            or any(not _digest(value) for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError("PRECISION_LINEAGE_GROUPS_INVALID")
        groups[index].update(values)
    if groups[0] & groups[1]:
        raise ValueError("PRECISION_LINEAGE_CROSS_BRANCH_HOLDOUT_LEAKAGE")
    return groups


# 功能：
#   检查精确的小写 SHA-256 身份，拒绝空值或非字符串。
# 输入：
#   value：待验证的内容摘要。
# 输出：
#   valid：摘要格式是否合法。
def _digest(value):
    valid = type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef")
    return valid


# 功能：
#   有界解码嵌入式源图；来源与组合图一起保存，不依靠工作目录中同名文件。
# 输入：
#   value：源图的标准 Base64 字符串。
# 输出：
#   content：字节副本，内容与结构后续交由同一图组合器核验。
def _source_bytes(value):
    if type(value) is not str or not 0 < len(value) <= 3 * 1024**2:
        raise ValueError("PRECISION_LINEAGE_SOURCE_BYTES_INVALID")
    try:
        content = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("PRECISION_LINEAGE_SOURCE_BYTES_INVALID") from error
    if not content or base64.b64encode(content).decode("ascii") != value:
        raise ValueError("PRECISION_LINEAGE_SOURCE_BYTES_INVALID")
    return content


# 功能：
#   1. 根据两份绑定源图重新生成组合图，核对实际产物，不将原训练成绩转移给新模型。
#   2. 分别校验原平移训练回执、偏航训练来源和跨分支数据隔离；不授予飞行资格。
# 输入：
#   digest：目标组合模型摘要。
#   receipt：自包含、最多四 MiB 的组合来源记录。
#   manifest：显式声明偏航输入的完整包清单。
# 输出：
#   None：来源、输入及组合不一致时抛出异常。
def validate_precision_composition_receipt(digest, receipt, manifest, *, role='precision-maneuver-policy'):
    import onnx

    from .artifact_assembly import validate_expert_training_receipt
    from .precision_heading_composition import compose_precision_heading

    receipt = copy_json(receipt, limit=4 * 1024**2)
    if composition_role(receipt) != role:
        raise ValueError('HEADING_LINEAGE_ROLE_MISMATCH')
    composition_spatial_groups(receipt)
    if (
        set(receipt)
        != {
            "purpose",
            "composition",
            "translation_graph_base64",
            "heading_graph_base64",
            "translation_training_receipt",
            "heading_training_lineage",
            "qualified_for_flight",
        }
        or receipt["qualified_for_flight"] is not False
        or manifest.heading_context_for_role(role) != HEADING_CONTEXT_SHA256
    ):
        raise ValueError("PRECISION_LINEAGE_CONTRACT_INVALID")
    translation = _source_bytes(receipt["translation_graph_base64"])
    heading = _source_bytes(receipt["heading_graph_base64"])
    translation_sha = hashlib.sha256(translation).hexdigest()
    heading_sha = hashlib.sha256(heading).hexdigest()
    content, expected = compose_precision_heading(
        translation, heading, translation_sha256=translation_sha, heading_sha256=heading_sha, role=role
    )
    if receipt["composition"] != expected or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("PRECISION_LINEAGE_RECOMPOSITION_MISMATCH")

    lineage = receipt["heading_training_lineage"]
    metadata = {
        item.key: item.value for item in onnx.load_model_from_string(heading).metadata_props
    }
    if (
        lineage.get("purpose") != "trained-heading-availability-lineage"
        or lineage.get("artifact_sha256") != heading_sha
        or lineage.get("feature_contract_sha256") != HEADING_CONTEXT_SHA256
        or lineage.get("qualified_for_flight") is not False
        or lineage.get("new_training") is not False
        or not _digest(lineage.get("data_sha256"))
        or not _digest(lineage.get("training_plan_sha256"))
        or not _digest(lineage.get("availability_plan_sha256"))
        or metadata.get("data_sha256") != lineage["data_sha256"]
        or metadata.get("plan_sha256") != lineage["availability_plan_sha256"]
        or metadata.get("variant") != "availability"
    ):
        raise ValueError("PRECISION_LINEAGE_HEADING_TRAINING_INVALID")
    checkpoints = lineage.get("trained_checkpoint_sha256")
    if (
        type(checkpoints) is not dict
        or set(checkpoints) != {"context", "circular"}
        or any(not _digest(value) for value in checkpoints.values())
        or len(set(checkpoints.values())) != 2
    ):
        raise ValueError("PRECISION_LINEAGE_TRAINED_CHECKPOINTS_INVALID")

    # 原回执只能验证原平移权重，不能把组合后的摘要塞进旧训练回执冒充重训。
    original = manifest.model_dump(mode="json")
    field = ('navigation_heading_context_sha256' if role == 'local-navigation-policy'
             else 'precision_heading_context_sha256')
    original.pop(field, None)
    artifact = next(a for a in original["artifacts"] if a["role"] == role)
    artifact["sha256"] = translation_sha
    artifact["input_names"] = [
        name for name in artifact["input_names"] if name != "heading_context"
    ]
    original_manifest = LocalPolicyPackageManifest.model_validate(original)
    validate_expert_training_receipt(
        role,
        translation_sha,
        receipt["translation_training_receipt"],
        original_manifest,
    )


# 功能：
#   为显式提供的冻结图及各自训练来源建立自包含证据，不继承任何旧模型验收成绩。
# 输入：
#   translation、heading：冻结源图字节。
#   translation_receipt、heading_lineage：分别属于两分支的原始来源记录。
# 输出：
#   receipt：有界深拷贝组合记录；完整配方组装时仍必须重新验证。
def make_precision_composition_receipt(translation, heading, translation_receipt, heading_lineage,
                                       *, role='precision-maneuver-policy'):
    from .precision_heading_composition import compose_precision_heading

    _, composition = compose_precision_heading(
        translation,
        heading,
        translation_sha256=hashlib.sha256(translation).hexdigest(),
        heading_sha256=hashlib.sha256(heading).hexdigest(),
        role=role,
    )
    receipt = copy_json(
        dict(
            purpose=composition['purpose'],
            composition=composition,
            translation_graph_base64=base64.b64encode(translation).decode("ascii"),
            heading_graph_base64=base64.b64encode(heading).decode("ascii"),
            translation_training_receipt=translation_receipt,
            heading_training_lineage=heading_lineage,
            qualified_for_flight=False,
        ),
        limit=4 * 1024**2,
    )
    composition_spatial_groups(receipt)
    return receipt
