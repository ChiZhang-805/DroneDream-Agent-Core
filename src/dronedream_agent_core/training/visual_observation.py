"""Load only the frozen RGB encoder, not any historical navigation/control weights."""

import base64
import hashlib
from pathlib import Path

from ..latest_inference_worker import LatestInferenceWorker
from ..local_policy_packages import load_local_policy_package
from ..plugin_files import read_plugin_file
from ..visual_control_input import forward_rgb_tensor
from ..visual_encoding_input import freeze_visual_input
from .visual_lineage import VisualInputContract


class FrozenVisualEncoder:
    """Training-side RGB embeddings using the deployable encoder and preprocessing."""

    # 功能：
    #   校验冻结视觉编码器的权重与图接口，建立单线程 CPU 会话并用合成像素预热。
    # 输入：
    #   self：待初始化的编码器。
    #   package_root：提供编码器与预处理契约的模型包目录。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, package_root: Path):
        import onnxruntime as ort

        from .artifact_assembly import validate_embedded_graph

        package = load_local_policy_package(package_root)
        self.manifest = package.manifest
        path = package.artifact_paths.get("perception-encoder")
        if path is None or not self.manifest.visual_feature_count:
            raise ValueError("TRAINING_VISUAL_ENCODER_MISSING")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        content = read_plugin_file(path, limit=256 * 1024 * 1024)
        self.sha256 = hashlib.sha256(content).hexdigest()
        expected = next(
            a.sha256 for a in package.manifest.artifacts if a.role == "perception-encoder"
        )
        if self.sha256 != expected:
            raise ValueError("TRAINING_VISUAL_ENCODER_CHANGED_DURING_LOAD")
        validate_embedded_graph(
            content, input_names=["forward_rgb"], output_names=["visual_features"]
        )
        self.session = ort.InferenceSession(content, options, providers=["CPUExecutionProvider"])
        self.input_contract = VisualInputContract.from_manifest(
            self.manifest, self.sha256
        ).model_dump()
        # Load image libraries and materialize ONNX kernels BEFORE reset/arming.
        # This synthetic compute warmup is not an observation or a training label.
        width, height = self.manifest.visual_width, self.manifest.visual_height
        tensor = forward_rgb_tensor(
            {
                "model_rgb_bytes": bytes(width * height * 3),
                "model_rgb_width": width,
                "model_rgb_height": height,
            },
            width=width,
            height=height,
            normalization=self.manifest.visual_normalization,
        )
        self._features(tensor)
        self._worker = LatestInferenceWorker(self._encode_source, name="training-frozen-visual")

    # 功能：
    #   执行冻结视觉图，要求输出为指定维数的有限 float32 向量或单批次矩阵。
    # 输入：
    #   self：持有冻结 CPU 会话和视觉契约的编码器。
    #   tensor：经过共同预处理的 RGB 输入张量。
    # 输出：
    #   features：长度等于视觉特征维数的一维 float32 数组。
    def _features(self, tensor):
        import numpy as np

        outputs = self.session.run(["visual_features"], {"forward_rgb": tensor})
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise ValueError("TRAINING_VISUAL_ENCODER_OUTPUT_INVALID")
        output = outputs[0]
        if (
            not isinstance(output, np.ndarray)
            or output.dtype != np.dtype(np.float32)
            or output.shape not in {
                (self.manifest.visual_feature_count,), (1, self.manifest.visual_feature_count)
            }
            or not np.isfinite(output).all()
        ):
            raise ValueError("TRAINING_VISUAL_ENCODER_OUTPUT_INVALID")
        # 两种既有向量接口均有效；不能靠展平把多张图或任意布局伪装成一个向量。
        features = output.reshape(-1).copy()
        return features

    # 功能：
    #   对单张内嵌相机图像先限制 Base64 长度，再核对实际字节、尺寸和摘要并冻结输入。
    # 输入：
    #   self：持有目标像素尺寸与编码器摘要的对象。
    #   media：包含唯一前向相机记录的列表或元组。
    # 输出：
    #   source：不可变的像素与预处理身份，不附加观测时效或动作权限。
    def _source(self, media):
        if (not isinstance(media, (list, tuple)) or len(media) != 1
                or not isinstance(media[0], dict)):
            raise ValueError("TRAINING_FORWARD_CAMERA_REQUIRED")
        item = dict(media[0])
        encoded = item.get("model_rgb_bytes")
        if not isinstance(encoded, dict) or set(encoded) != {"base64_bytes"}:
            raise ValueError("TRAINING_REQUIRES_EMBEDDED_MODEL_RGB")
        text = encoded["base64_bytes"]
        expected_size = self.manifest.visual_width * self.manifest.visual_height * 3
        if not isinstance(text, str) or len(text) != 4 * ((expected_size + 2) // 3):
            raise ValueError("TRAINING_MODEL_RGB_BASE64_SIZE_INVALID")
        raw = base64.b64decode(text, validate=True)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item.get("model_rgb_sha256"):
            raise ValueError("TRAINING_FORWARD_RGB_HASH_MISMATCH")
        item["model_rgb_bytes"] = raw
        source = freeze_visual_input(
            [item],
            width=self.manifest.visual_width,
            height=self.manifest.visual_height,
            normalization=self.manifest.visual_normalization,
            encoder_sha256=self.sha256,
        )
        return source

    # 功能：
    #   将冻结像素编码为不可变特征，并附带原始内容摘要以便异步取回时核对。
    # 输入：
    #   self：冻结编码器。
    #   source：已绑定像素、预处理和编码器身份的输入。
    # 输出：
    #   result：原始像素摘要与特征元组组成的二元组。
    def _encode_source(self, source):
        result = source.source_sha256, tuple(float(v) for v in self._features(source.tensor()))
        return result

    # 功能：
    #   提交最新冻结图像以预取特征，不续期观测时效或授予移动权限。
    # 输入：
    #   self：持有最新任务工作线程的编码器。
    #   media：唯一的内嵌前向相机记录。
    # 输出：
    #   None：不返回业务数据。
    def prime(self, media):
        source = self._source(media)
        self._worker.submit(source.key, source)

    # 功能：
    #   将身份匹配的视觉特征附加到观测副本，缓存缺失时同步编码，不修改原观测。
    # 输入：
    #   self：冻结编码器。
    #   observation：需要附加视觉特征的类型化观测。
    #   media：对应的唯一相机记录。
    # 输出：
    #   result：重新通过观测契约校验的视觉观测。
    def attach(self, observation, media):
        source = self._source(media)
        future = self._worker.cached(source.key)
        # Offline readback is synchronous and starts no background worker.
        digest, features = (future.result(timeout=.25) if future is not None
                            else self._encode_source(source))
        if digest != source.source_sha256:
            raise ValueError("TRAINING_VISUAL_PREFETCH_IDENTITY_MISMATCH")
        payload = observation.model_dump(mode="json")
        payload.update(visual_features=list(features), source_visual_sha256=digest)
        result = type(observation).model_validate(payload)
        return result

    # 功能：
    #   关闭视觉预取并返回线程状态，不强行终止仍在执行的推理。
    # 输入：
    #   self：需要关闭的冻结编码器。
    # 输出：
    #   result：工作线程关闭状态和统计回执。
    def close(self):
        result = self._worker.close()
        return result
