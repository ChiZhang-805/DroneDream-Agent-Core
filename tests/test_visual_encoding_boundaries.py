"""Bounded visual identity and tensor tests; no online models or aircraft."""

import base64
import hashlib
import io
import json
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from test_visual_prefetch_boundary import frozen_encoder, media  # noqa: F401
from test_visual_training_lineage import encoding

from dronedream_agent_core import visual_control_input
from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.training import visual_observation
from dronedream_agent_core.training.visual_lineage import (
    VisualInputContract,
    require_matching_visual_input,
    verify_causal_checkpoint_receipt,
    verify_visual_training_inputs,
)
from dronedream_agent_core.visual_encoding_input import freeze_visual_input


# 功能：
#   验证视觉编码回执不能通过重复键、非有限值或无界文本隐藏错误信息。
# 输入：
#   invalid：需要注入的回执错误。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("invalid", ["duplicate", "nan", "large"])
def test_visual_receipt_is_strict_and_bounded(invalid):
    train, validation = b"training\n", b"validation\n"
    left, right = encoding(train), encoding(validation)
    prefix = {
        "duplicate": '"qualification_granted":true,',
        "nan": '"metric":NaN,',
        "large": '"unused":"' + "x" * (4 * 1024 * 1024) + '",',
    }[invalid]
    receipt = ("{" + prefix + json.dumps(left)[1:]).encode()
    with pytest.raises(ValueError):
        verify_visual_training_inputs(training_content=train, validation_content=validation,
            receipt_contents=[receipt, json.dumps(right).encode()], feature_count=1)


# 功能：
#   验证视觉训练拒绝空数据、相同划分内容及含空白样本行的数据。
# 输入：
#   train：训练数据字节。
#   validation：验证数据字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("train,validation", [
    (b"", b"validation\n"), (b"same\n", b"same\n"), (b"row\n\n", b"validation\n"),
])
def test_visual_split_requires_nonempty_distinct_records(train, validation):
    receipts = [encoding(value) for value in (train, validation)]
    for value, receipt in zip((train, validation), receipts, strict=True):
        receipt["sample_count"] = sum(bool(row.strip()) for row in value.splitlines())
    with pytest.raises(ValueError):
        verify_visual_training_inputs(training_content=train, validation_content=validation,
            receipt_contents=[json.dumps(value).encode() for value in receipts], feature_count=1)


# 功能：
#   验证可变契约实例需要重新校验，布尔维数不能与正确的一维特征契约比较为相等。
# 输入：
#   无：测试自行构造视觉契约并模拟验证后的错误修改。
# 输出：
#   None：不返回业务数据。
def test_matching_contract_revalidates_mutable_instances():
    values = {key: encoding(b"row\n")[key] for key in VisualInputContract.model_fields}
    actual = VisualInputContract.model_validate(values).model_copy(
        update={"visual_feature_count": True}
    )
    # 普通赋值原本就会被验证；未校验的 model_copy 才是要覆盖的入口。
    assert VisualInputContract.model_validate(actual) is actual
    with pytest.raises(ValueError):
        require_matching_visual_input(actual, values)


# 功能：
#   验证单独检查点回执也不能使用重复的权重摘要或非有限指标。
# 输入：
#   invalid：要注入的 JSON 歧义。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("invalid", ["duplicate", "nan"])
def test_checkpoint_receipt_rejects_ambiguous_json(invalid):
    content = b"offline checkpoint"
    receipt = {
        "checkpoint_sha256": hashlib.sha256(content).hexdigest(),
        "expert_role": "local-navigation-policy",
        "architecture": "causal-gru-control",
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "config": {"visual_feature_count": 0},
        "visual_input_contract": None,
    }
    prefix = '"checkpoint_sha256":"wrong",' if invalid == "duplicate" else '"metric":NaN,'
    encoded = ("{" + prefix + json.dumps(receipt)[1:]).encode()
    with pytest.raises(ValueError):
        verify_causal_checkpoint_receipt(content, encoded, expert_role="local-navigation-policy")


# 功能：
#   验证编码器不能展平多批次错误维数或自动转换字符串、布尔值。
# 输入：
#   frozen_encoder：使用合成 CPU ONNX 图的测试编码器。
#   output：需要模拟的非法后端输出。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("output", [
    np.ones((2, 6), dtype=np.float32),
    np.ones((1, 12), dtype=bool), np.full((1, 12), "1"),
    np.full((1, 12), np.nan, dtype=np.float32),
])
def test_features_reject_wrong_batch_shape_and_dtype(frozen_encoder, output):  # noqa: F811
    frozen_encoder.session = Mock(run=Mock(return_value=[output]))
    with pytest.raises(ValueError, match="OUTPUT_INVALID"):
        frozen_encoder._features(np.zeros((1, 3, 2, 2), dtype=np.float32))


# 功能：
#   验证显式单批次和无批次轴的特征向量均可使用，返回数组不共享后端可变缓冲。
# 输入：
#   frozen_encoder：合成视觉编码器。
#   shape：合法的特征向量布局。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("shape", [(12,), (1, 12)])
def test_features_accept_vector_and_single_batch(frozen_encoder, shape):  # noqa: F811
    output = np.ones(shape, dtype=np.float32)
    frozen_encoder.session = Mock(run=Mock(return_value=[output]))
    features = frozen_encoder._features(np.zeros((1, 3, 2, 2), dtype=np.float32))
    assert features.shape == (12,)
    output[...] = 0
    assert np.all(features == 1)


# 功能：
#   验证超出像素所需长度的 Base64 输入在解码分配内存前拒绝。
# 输入：
#   frozen_encoder：合成视觉编码器。
#   monkeypatch：替换 Base64 解码调用的夹具。
# 输出：
#   None：不返回业务数据。
def test_oversized_base64_is_rejected_before_decoding(frozen_encoder, monkeypatch):  # noqa: F811
    item = media()
    item["model_rgb_bytes"] = {"base64_bytes": base64.b64encode(bytes(100)).decode()}
    decoder = Mock(side_effect=AssertionError("oversized input reached decoder"))
    monkeypatch.setattr(visual_observation.base64, "b64decode", decoder)
    with pytest.raises(ValueError, match="RGB"):
        frozen_encoder._source([item])
    decoder.assert_not_called()


# 功能：
#   验证错误媒体容器被统一拒绝，不因 len、get 或下标操作抛出非契约异常。
# 输入：
#   invalid：非法媒体容器或记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("invalid", [None, 1, [None], ["not an object"]])
def test_visual_source_rejects_malformed_container(invalid):
    with pytest.raises((ValueError, RuntimeError), match="RGB"):
        freeze_visual_input(invalid, width=2, height=2,
                            normalization="zero-to-one", encoder_sha256="a" * 64)


# 功能：
#   验证压缩图片即使文件很小，解码后的像素超过预算仍会在转换像素前被拒绝。
# 输入：
#   monkeypatch：缩小测试像素预算的夹具。
# 输出：
#   None：不返回业务数据。
def test_compressed_image_cannot_bypass_decoded_pixel_budget(monkeypatch):
    with io.BytesIO() as buffer:
        with Image.new("RGB", (64, 64), "white") as image:
            image.save(buffer, format="PNG")
        content = buffer.getvalue()
    assert len(content) < 3072
    monkeypatch.setattr(visual_control_input, "MAXIMUM_IMAGE_BYTES", 3072, raising=False)
    with pytest.raises(RuntimeError, match="PIXEL_BUDGET"):
        visual_control_input.forward_rgb_tensor({"content_bytes": content}, width=2, height=2,
                                                normalization="zero-to-one")


# 功能：
#   验证实际共享图像预处理保留 RGB 通道顺序，并执行各编码器声明的归一化公式。
# 输入：
#   normalization：待检查的归一化方法。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("normalization", ["zero-to-one", "minus-one-to-one", "imagenet"])
def test_shared_preprocessing_preserves_channel_semantics(normalization):
    record = {"model_rgb_bytes": bytes([255, 128, 0]),
              "model_rgb_width": 1, "model_rgb_height": 1}
    tensor = visual_control_input.forward_rgb_tensor(record, width=1, height=1,
                                                     normalization=normalization)
    expected = np.array([1, 128 / 255, 0], dtype=np.float32)
    if normalization == "minus-one-to-one":
        expected = expected * 2 - 1
    elif normalization == "imagenet":
        expected = (expected - [.485, .456, .406]) / [.229, .224, .225]
    assert tensor.shape == (1, 3, 1, 1) and tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0, :, 0, 0], expected, rtol=1e-6, atol=1e-6)
