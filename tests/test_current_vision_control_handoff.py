"""Actual five-head export must be reusable by the frozen control-training encoder."""

import hashlib
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from dronedream_agent_core import local_vision_training as training
from dronedream_agent_core.training import visual_observation
from dronedream_agent_core.training.vision_export import OUTPUT_NAMES
from dronedream_agent_core.training.visual_lineage import VisualInputContract
from dronedream_agent_core.training.px4_environment import ASSET_FIELDS, Px4TrainingConfig


# 功能：
#   将实际五输出视觉网络导出后交给控制训练的冻结编码器，确认额外监督头不会阻断特征复用。
# 输入：
#   tmp_path、monkeypatch：隔离模型路径及只替换资产索引的测试夹具。
# 输出：
#   None：真实 ORT 的 139 维输出与 PyTorch 一致，且未自动创建采集线程。
def test_current_five_head_export_enters_control_training(tmp_path, monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        config = training.LocalVisionTrainingConfig(width=64, height=64)
        model = training.build_local_vision_model(config, pretrained_backbone=False).eval()
        path = tmp_path / "actual-five-head.onnx"
        training.export_local_vision_onnx(model, path, config)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact = SimpleNamespace(role="perception-encoder", sha256=digest,
            input_names=["forward_rgb"], output_names=list(OUTPUT_NAMES))
        manifest = SimpleNamespace(visual_width=64, visual_height=64, visual_feature_count=139,
            visual_normalization="imagenet", artifacts=[artifact])
        package = SimpleNamespace(manifest=manifest, artifact_paths={"perception-encoder": path})
        monkeypatch.setattr(visual_observation, "load_local_policy_package", lambda _: package)
        encoder = visual_observation.FrozenVisualEncoder(tmp_path)
        try:
            probe = torch.randn(1, 3, 64, 64)
            with torch.inference_mode():
                expected = model(probe)[0].numpy()[0]
            np.testing.assert_allclose(
                encoder._features(probe.numpy()), expected, atol=1e-4, rtol=1e-4
            )
            assert encoder._worker._thread is None
        finally:
            encoder.close()
        contract = VisualInputContract(perception_encoder_sha256=digest, visual_width=64,
            visual_height=64, visual_feature_count=139, visual_normalization="imagenet")
        # 独立视觉入口不依赖完整包，更不能悄悄复用包内的旧控制专家。
        monkeypatch.setattr(visual_observation, "load_local_policy_package",
                            lambda _: pytest.fail("standalone encoder loaded a control package"))
        standalone = visual_observation.FrozenVisualEncoder.from_artifact(path, contract)
        try:
            np.testing.assert_allclose(standalone._features(probe.numpy()), expected, atol=1e-4, rtol=1e-4)
            assert standalone.input_contract == contract.model_dump()
        finally:
            standalone.close()
        for updates, message in [({'perception_encoder_sha256': 'a' * 64}, 'CHANGED_DURING_LOAD'),
                                 ({'visual_width': 32}, 'INPUT_INVALID'),
                                 ({'visual_feature_count': 140}, 'OUTPUT_INVALID')]:
            with pytest.raises(ValueError, match=message):
                visual_observation.FrozenVisualEncoder.from_artifact(path, contract.model_copy(update=updates))
    finally:
        torch.set_num_threads(previous)


# 功能：
#   验证独立新视觉权重与输入契约成对提供，不能与旧模型包同时配置。
# 输入：
#   无：仅使用合成路径验证配置，不启动环境或读取模型文件。
# 输出：
#   None：正确组合通过，缺失身份及混合来源被拒绝。
def test_standalone_visual_configuration_is_unambiguous():
    base = dict(runner='runner.py', output_root='runs', mission_id='synthetic',
        expert_role='local-navigation-policy', **{name: 'asset' for name in ASSET_FIELDS},
        asset_sha256={}, minimum_enu_m=(0, 0, 0), maximum_enu_m=(1, 1, 1))
    contract = VisualInputContract(perception_encoder_sha256='a' * 64,
        visual_width=224, visual_height=128, visual_feature_count=139,
        visual_normalization='imagenet')
    config = Px4TrainingConfig(**base, visual_encoder='vision.onnx', visual_input_contract=contract)
    assert config.visual_input_contract == contract and config.visual_package is None
    for fields in ({'visual_encoder': 'vision.onnx'}, {'visual_input_contract': contract},
                   {'visual_encoder': 'vision.onnx', 'visual_input_contract': contract,
                    'visual_package': 'old-package'}):
        with pytest.raises(ValueError):
            Px4TrainingConfig(**base, **fields)
