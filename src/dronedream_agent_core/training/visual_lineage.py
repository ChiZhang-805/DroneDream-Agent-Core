"""Embedding identity is its encoder AND preprocessing, not just tensor width."""

import hashlib
import io
from typing import Literal

from pydantic import Field

from dronedream_plugin_sdk.protocol import decode_json

from ..contracts import StrictModel
from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..local_expert_harness import NAVIGATION_EXPERT_ROLES
from ..local_policy_packages import Sha256


class VisualInputContract(StrictModel):
    """Identify embeddings by actual weights and preprocessing, not dimensions alone."""

    perception_encoder_sha256: Sha256
    visual_width: int = Field(strict=True, ge=32, le=1024)
    visual_height: int = Field(strict=True, ge=32, le=1024)
    visual_feature_count: int = Field(strict=True, ge=1, le=65536)
    visual_normalization: Literal["zero-to-one", "minus-one-to-one", "imagenet"]

    # 功能：
    #   将实际编码器摘要与清单中的尺寸、维数和归一化方式绑定，不能仅靠维数识别特征。
    # 输入：
    #   cls：视觉输入契约类。
    #   manifest：当前模型包清单。
    #   encoder_sha256：已经校验的编码器权重摘要。
    # 输出：
    #   contract：通过类型与范围验证的视觉输入契约。
    @classmethod
    def from_manifest(cls, manifest, encoder_sha256: str):
        contract = cls(
            perception_encoder_sha256=encoder_sha256,
            **{
                name: getattr(manifest, name)
                for name in cls.model_fields
                if name != "perception_encoder_sha256"
            },
        )
        return contract


# 功能：
#   1. 验证训练与验证数据的视觉编码回执，要求字节身份、样本数和预处理契约一致。
#   2. 拒绝空白记录和相同划分内容；此处的内容独立检查不能代替后续空间划分校验。
# 输入：
#   training_content：训练数据原始字节。
#   validation_content：验证数据原始字节。
#   receipt_contents：按训练、验证顺序提供的两份编码回执字节。
#   feature_count：视觉特征维数，零表示显式选择无视觉策略。
# 输出：
#   result：共同视觉契约的字典；无视觉策略为 None。
def verify_visual_training_inputs(
    *,
    training_content: bytes,
    validation_content: bytes,
    receipt_contents: list[bytes],
    feature_count: int,
):
    # Zero explicitly selects a nonvisual policy; bool/float coercion must not
    # silently change the architecture or impersonate a one-feature encoder.
    if type(feature_count) is not int or not 0 <= feature_count <= 65536:
        raise ValueError("VISUAL_FEATURE_COUNT_INVALID")
    if not isinstance(receipt_contents, (list, tuple)):
        raise ValueError("VISUAL_TRAINING_RECEIPTS_CONTAINER_INVALID")
    if any(not isinstance(value, bytes) or len(value) > 256 * 1024 * 1024
           for value in (training_content, validation_content)):
        raise ValueError("VISUAL_TRAINING_CONTENT_INVALID_OR_TOO_LARGE")
    if feature_count == 0:
        if receipt_contents:
            raise ValueError("VISUAL_RECEIPTS_PROVIDED_FOR_NON_VISUAL_POLICY")
        result = None
        return result
    if len(receipt_contents) != 2:
        raise ValueError("VISUAL_TRAINING_REQUIRES_BOTH_ENCODING_RECEIPTS")
    contracts = []
    for content, encoded in zip(
        (training_content, validation_content), receipt_contents, strict=True
    ):
        receipt = decode_json(encoded, limit=4 * 1024 * 1024)
        # 逐行计数避免 splitlines 为整份数据再分配大量对象；不静默忽略空白样本。
        sample_count = 0
        for row in io.BytesIO(content):
            if not row.strip():
                raise ValueError("VISUAL_ENCODING_RECEIPT_BLANK_SAMPLE")
            sample_count += 1
        if not isinstance(receipt, dict) or (
            receipt.get("schema_version") != "dronedream.local-policy-visual-encoding-receipt.v1"
            or receipt.get("qualification_granted") is not False
            or receipt.get("output_sha256") != hashlib.sha256(content).hexdigest()
            or type(receipt.get("visual_feature_count")) is not int
            or receipt.get("visual_feature_count") != feature_count
            or type(receipt.get("sample_count")) is not int
            or receipt.get("sample_count") != sample_count
            or sample_count == 0
        ):
            raise ValueError("VISUAL_ENCODING_RECEIPT_DOES_NOT_BIND_TRAINING_INPUT")
        contracts.append(
            VisualInputContract.model_validate(
                {key: receipt.get(key) for key in VisualInputContract.model_fields}
            )
        )
    if contracts[0] != contracts[1]:
        raise ValueError("VISUAL_TRAINING_SPLITS_USE_DIFFERENT_ENCODERS_OR_PREPROCESSING")
    if training_content == validation_content:
        raise ValueError("VISUAL_TRAINING_SPLITS_HAVE_IDENTICAL_CONTENT")
    result = contracts[0].model_dump(mode="json")
    return result


# 功能：
#   重验并比较视觉契约，拒绝错误的可变实例、视觉模式差异及编码器或预处理不符。
# 输入：
#   contract：待复用特征的视觉契约，允许字典、契约实例或 None。
#   expected：本次训练所需的视觉契约，格式同 contract。
# 输出：
#   None：不返回业务数据。
def require_matching_visual_input(contract, expected):
    if isinstance(contract, VisualInputContract):
        contract = contract.model_dump(mode="python")
    if isinstance(expected, VisualInputContract):
        expected = expected.model_dump(mode="python")
    actual = VisualInputContract.model_validate(contract) if contract is not None else None
    wanted = VisualInputContract.model_validate(expected) if expected is not None else None
    if actual != wanted:
        raise ValueError("VISUAL_INPUT_IDENTITY_MISMATCH")


# 功能：
#   在复用检查点前绑定权重、专家角色与当前特征契约，核对视觉模式和维数。
# 输入：
#   content：检查点原始字节。
#   receipt_content：其训练回执原始字节。
#   expert_role：本次需要复用检查点的操纵专家角色。
# 输出：
#   result：已核对的视觉契约字典；无视觉策略为 None。
def verify_causal_checkpoint_receipt(content: bytes, receipt_content: bytes, *, expert_role: str):
    if not isinstance(content, bytes) or not content or len(content) > 256 * 1024 * 1024:
        raise ValueError("CAUSAL_CHECKPOINT_CONTENT_INVALID_OR_TOO_LARGE")
    if expert_role not in NAVIGATION_EXPERT_ROLES:
        raise ValueError("CAUSAL_CHECKPOINT_EXPERT_ROLE_INVALID")
    receipt = decode_json(receipt_content, limit=4 * 1024 * 1024)
    if not isinstance(receipt, dict) or (
        receipt.get("checkpoint_sha256") != hashlib.sha256(content).hexdigest()
        or receipt.get("expert_role") != expert_role
        or receipt.get("architecture") != "causal-gru-control"
        or receipt.get("feature_contract_sha256") != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
    ):
        raise ValueError("CAUSAL_CHECKPOINT_RECEIPT_IDENTITY_MISMATCH")
    contract = receipt.get("visual_input_contract")
    visual = VisualInputContract.model_validate(contract) if contract is not None else None
    config = receipt.get("config")
    expected_count = visual.visual_feature_count if visual else 0
    if (not isinstance(config, dict)
            or type(config.get("visual_feature_count")) is not int
            or config.get("visual_feature_count") != expected_count):
        raise ValueError("CAUSAL_CHECKPOINT_VISUAL_RECEIPT_INCOMPLETE")
    result = visual.model_dump(mode="json") if visual else None
    return result
