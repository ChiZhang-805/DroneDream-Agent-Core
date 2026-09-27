"""Owned RGB prefetch contracts; tiny synthetic CPU graph is not a product model."""

import base64
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from clock_fixtures import isolate_time
from PIL import Image
from test_local_policy_composition import _write_base_package
from test_local_policy_packages import _write_brightness_perception_encoder
from test_offline_flight_learning import UnitEnvironment
from test_px4_training_lifecycle import environment
from test_training_observation_boundary import current_request

from dronedream_agent_core.latest_inference_worker import LatestInferenceWorker
from dronedream_agent_core.local_policy_port import OnnxLocalPolicyBackend
from dronedream_agent_core.training import px4_environment as native
from dronedream_agent_core.training import visual_observation
from dronedream_agent_core.training.visual_observation import FrozenVisualEncoder
from dronedream_agent_core.visual_encoding_input import freeze_visual_input


# 功能：
#   创建确定的两行两列 RGB 图像记录并绑定内容摘要。
# 输入：
#   无：像素序列由测试固定生成。
# 输出：
#   item：包含原始像素、尺寸和摘要的相机记录。
def media():
    raw = bytes(range(12))
    item = {"kind": "image-file", "model_rgb_bytes": raw,
            "model_rgb_sha256": hashlib.sha256(raw).hexdigest(),
            "model_rgb_width": 2, "model_rgb_height": 2}
    return item


# 功能：
#   使用测试默认预处理和编码器摘要冻结一张图，允许显式覆盖所测契约字段。
# 输入：
#   item：测试相机记录。
#   updates：需要覆盖的尺寸、归一化或编码器配置。
# 输出：
#   source：通过共同入口冻结的图像输入。
def freeze(item, **updates):
    source = freeze_visual_input([item], **{"width": 2, "height": 2,
                                        "encoder_sha256": "a" * 64,
                                        "normalization": "zero-to-one", **updates})
    return source


# 功能：
#   验证冻结像素及缓存键不受外部记录修改影响，而预处理或编码器改变会产生不同缓存键。
# 输入：
#   无：测试自行构造和修改像素记录。
# 输出：
#   None：不返回业务数据。
def test_owned_source_key_and_pixels_cannot_be_mutated_by_caller():
    original = media()
    frozen = freeze(original)
    key = frozen.key
    original["model_rgb_bytes"] = b"corrupted"
    original["model_rgb_width"] = 9
    assert frozen.key == key
    assert frozen.content == bytes(range(12))
    assert frozen.tensor().shape == (1, 3, 2, 2)
    assert freeze(media(), normalization="imagenet").key != key
    assert freeze(media(), encoder_sha256="b" * 64).key != key


# 功能：
#   验证缓存身份中的编码器摘要必须是有效的小写 SHA-256。
# 输入：
#   digest：非法编码器摘要。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("digest", [None, "", "A" * 64, "g" * 64, "a" * 63, 123])
def test_encoder_identity_requires_a_valid_digest(digest):
    with pytest.raises(ValueError, match="VISUAL_ENCODER_DIGEST_INVALID"):
        freeze(media(), encoder_sha256=digest)


# 功能：
#   验证相同像素携带错误尺寸、长度或摘要时仍被拒绝，不能利用缓存跳过输入校验。
# 输入：
#   update：需要注入的错误媒体字段。
#   error：预期错误标识。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("update,error", [
    ({"model_rgb_width": 4, "model_rgb_height": 1}, "DIMENSIONS_MISMATCH"),
    ({"model_rgb_width": True}, "DIMENSIONS_MISMATCH"),
    ({"model_rgb_bytes": b"wrong"}, "SIZE_MISMATCH"),
    ({"model_rgb_sha256": "a" * 64}, "HASH_MISMATCH"),
])
def test_bad_same_pixels_metadata_cannot_hit_a_cached_feature(update, error):
    with pytest.raises(RuntimeError, match=error):
        freeze({**media(), **update})


# 功能：
#   验证文件内容在后台编码前已经冻结，后来替换原文件不会改变待编码画面。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_file_bytes_are_frozen_before_background_encoding(tmp_path):
    path = tmp_path / "pixels.png"
    with Image.new("RGB", (2, 2), "white") as image:
        image.save(path)
    frozen = freeze({"kind": "image-file", "path": str(path)})
    path.write_bytes(b"changed after admission")
    assert np.all(frozen.tensor() == 1.)


# 功能：
#   创建实际 CPU ONNX 展平图与预取工作器，结束时关闭线程；合成图不代表产品能力。
# 输入：
#   无：夹具自行生成两行两列输入的测试图。
# 输出：
#   encoder：供单个测试使用的冻结编码器实例。
@pytest.fixture
def frozen_encoder():
    # Real ONNX Runtime mechanics, synthetic identity graph and manifest only.
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Flatten", ["forward_rgb"], ["visual_features"], axis=1)],
        "unit-visual", [onnx.helper.make_tensor_value_info("forward_rgb", onnx.TensorProto.FLOAT,
                                                         [1, 3, 2, 2])],
        [onnx.helper.make_tensor_value_info("visual_features", onnx.TensorProto.FLOAT, [1, 12])])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 10
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    encoder = object.__new__(FrozenVisualEncoder)
    encoder.sha256 = hashlib.sha256(model.SerializeToString()).hexdigest()
    encoder.manifest = SimpleNamespace(visual_width=2, visual_height=2, visual_feature_count=12,
                                       visual_normalization="zero-to-one")
    encoder.session = ort.InferenceSession(model.SerializeToString(), options,
                                           providers=["CPUExecutionProvider"])
    encoder._worker = LatestInferenceWorker(encoder._encode_source, name="unit-frozen-encoder")
    yield encoder
    encoder.close()


# 功能：
#   验证同步与预取编码结果相同，原观测和历史证据不被修改，缓存不能掩盖错误布局。
# 输入：
#   frozen_encoder：合成 CPU 视觉编码器。
# 输出：
#   None：不返回业务数据。
def test_training_prefetch_matches_fresh_encoding_and_preserves_original_sample(frozen_encoder):
    item = media()
    item["model_rgb_bytes"] = {"base64_bytes": base64.b64encode(bytes(range(12))).decode()}
    original = UnitEnvironment().reset(seed=805).sample
    synchronous = frozen_encoder.attach(original, [item])
    assert frozen_encoder._worker._thread is None
    frozen_encoder.prime([item])
    asynchronous = frozen_encoder.attach(original, [item])
    assert synchronous == asynchronous
    assert asynchronous.temporal_evidence == original.temporal_evidence
    assert not original.visual_features
    assert asynchronous.source_visual_sha256 == media()["model_rgb_sha256"]
    with pytest.raises(RuntimeError, match="DIMENSIONS_MISMATCH"):
        frozen_encoder.attach(original, [{**item, "model_rgb_width": 4, "model_rgb_height": 1}])


# 功能：
#   验证实际部署后端在命中视觉缓存时仍检查当前图像尺寸，并可靠关闭预取工作器。
# 输入：
#   frozen_encoder：提供合成图和输入契约的编码器。
# 输出：
#   None：不返回业务数据。
def test_deployed_prefetch_cannot_bypass_shape_validation_on_cache_hit(frozen_encoder):
    backend = object.__new__(OnnxLocalPolicyBackend)
    backend._manifest = frozen_encoder.manifest
    backend._perception_session = frozen_encoder.session
    backend._perception_sha256 = frozen_encoder.sha256
    backend._visual_worker = LatestInferenceWorker(
        backend._encode_visual, name="unit-deploy-encoder")
    try:
        backend.prime_visual([media()])
        result, _ = backend._resolve_visual([media()])
        assert result.source_sha256 == media()["model_rgb_sha256"]
        with pytest.raises(RuntimeError, match="DIMENSIONS_MISMATCH"):
            backend._resolve_visual([{**media(), "model_rgb_width": 4, "model_rgb_height": 1}])
    finally:
        backend.close()


# 功能：
#   验证只有剩余时效满足阈值才提前预取视觉，预取本身不会回复或接纳操纵请求。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：替换时间与观测编译入口的夹具。
#   remaining：请求距离失效的毫秒数。
#   primed：本场景是否应触发预取。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("remaining,primed", [
    (native.LOCAL_DISPATCH_RESERVE_MS + native.TRAINING_REPLY_PREPARATION_RESERVE_MS, True),
    (native.LOCAL_DISPATCH_RESERVE_MS + native.TRAINING_REPLY_PREPARATION_RESERVE_MS - 1, False),
    (1, False)])
def test_live_visual_prefetch_precedes_numeric_validation_without_admitting_input(
    tmp_path, monkeypatch, remaining, primed
):
    env = environment(tmp_path)
    request = current_request()
    request["valid_until_unix_ms"] = 1000 + remaining
    env._exchange = SimpleNamespace(wait_request=Mock(return_value=request), reply=Mock())
    env.visual = SimpleNamespace(prime=Mock())
    env._retain_source_observation = Mock()

    # 功能：
    #   检查数值编译开始时视觉预取次数，并返回固定的已拥有样本替身。
    # 输入：
    #   value：送入编译入口的请求对象。
    # 输出：
    #   compiled：只包含受控样本的测试编译结果。
    def compile_source(value):
        assert value is request
        assert env.visual.prime.call_count == int(primed)
        compiled = SimpleNamespace(sample="owned-numeric-sample")
        return compiled

    monkeypatch.setattr(
        native, "PreparedTrainingInput", SimpleNamespace(from_request=compile_source))
    isolate_time(monkeypatch, native, time=lambda: 1.)
    assert env._receive_source_request(timeout_seconds=.1) is request
    env._exchange.reply.assert_not_called()


# 功能：
#   验证原生环境静止等待失败时仍关闭视觉线程，同时保留原始失败供调用方处理。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_environment_closes_visual_thread_even_when_native_quiescence_raises(tmp_path):
    env = environment(tmp_path)
    env.visual = SimpleNamespace(close=Mock())
    env.quiesce = Mock(side_effect=TimeoutError("native not grounded"))
    with pytest.raises(TimeoutError, match="native not grounded"):
        env.close()
    env.visual.close.assert_called_once()


# 功能：
#   验证清单校验后被替换的模型仍会在实际部署后端加载时被拒绝。
# 输入：
#   tmp_path：测试独立目录。
#   role：需要篡改权重文件的角色。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("role", ["local-navigation-policy", "risk-critic"])
def test_deployment_rejects_artifacts_replaced_after_manifest_validation(tmp_path, role):
    package = _write_base_package(tmp_path / "unit-package")
    package.artifact_paths[role].write_bytes(b"changed after package validation")
    with pytest.raises(RuntimeError, match="ARTIFACT_CHANGED_DURING_LOAD"):
        OnnxLocalPolicyBackend(package)


# 功能：
#   验证构造器加载并预热实际 CPU 图、绑定实际摘要，但在首次预取前不启动线程。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：提供明确测试包的夹具。
# 输出：
#   None：不返回业务数据。
def test_frozen_encoder_constructor_warms_actual_cpu_graph_without_starting_worker(
    tmp_path, monkeypatch
):
    path = tmp_path / "synthetic-encoder.onnx"
    _write_brightness_perception_encoder(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = SimpleNamespace(
        visual_width=32, visual_height=32, visual_feature_count=1,
        visual_normalization="zero-to-one",
        artifacts=[SimpleNamespace(role="perception-encoder", sha256=digest,
                                   input_names=["forward_rgb"], output_names=["visual_features"])])
    package = SimpleNamespace(manifest=manifest, artifact_paths={"perception-encoder": path})
    monkeypatch.setattr(visual_observation, "load_local_policy_package", lambda _root: package)
    encoder = FrozenVisualEncoder(tmp_path)
    try:
        assert encoder._worker._thread is None
        assert encoder.input_contract["perception_encoder_sha256"] == digest
    finally:
        assert encoder.close()["thread_stopped"]
