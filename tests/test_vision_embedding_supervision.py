"""Real gradient and update checks; synthetic fixtures never qualify flight models."""

import numpy as np
import onnxruntime as ort
import pytest
import torch
from test_local_vision_training import _dataset

from dronedream_agent_core import local_vision_training as training


# 功能：
#   为小型真实网络回归固定 CPU 线程数并在退出时恢复，不将耗时当产品性能数据。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


# 功能：
#   分别证明每个辅助任务的损失确实回传到嵌入投影，且冻结骨干时仍有非零梯度。
# 输入：
#   output_index：可通行性、场景或质量的输出位置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("output_index", [2, 3, 4])
def test_each_auxiliary_task_supervises_embedding(output_index):
    config = training.LocalVisionTrainingConfig(width=64, height=64)
    model = training.build_local_vision_model(config, pretrained_backbone=False).eval()
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    outputs = model(torch.randn(2, 3, 64, 64))
    logits = outputs[output_index]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    loss.backward()
    for parameter in model.embedding_head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


# 功能：
#   用实际优化器更新证明嵌入权重改变，冻结骨干的参数及运行统计均保持不变。
# 输入：
#   tmp_path：仅容纳合成训练样本的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_optimizer_updates_embedding_without_mutating_frozen_backbone(tmp_path):
    config = training.LocalVisionTrainingConfig(
        width=64, height=64, epoch_count=1, batch_size=2, freeze_backbone_epochs=1
    )
    samples = training.resolve_local_vision_samples(tmp_path, _dataset(tmp_path))
    model = training.build_local_vision_model(config, pretrained_backbone=False)
    before_backbone = {name: value.clone() for name, value in model.backbone.state_dict().items()}
    before_embedding = {
        name: value.clone() for name, value in model.embedding_head.state_dict().items()
    }
    trained, metrics = training.train_local_vision_model(model, samples, config)
    assert metrics.sample_count == 4
    assert all(torch.equal(value, trained.backbone.state_dict()[name])
               for name, value in before_backbone.items())
    assert all(not torch.equal(value, trained.embedding_head.state_dict()[name])
               for name, value in before_embedding.items())


# 功能：
#   人为切断嵌入的监督路径时，真实训练必须在优化器更新前报错，不能返回成功指标。
# 输入：
#   tmp_path：隔离样本目录。
# 输出：
#   None：不返回业务数据。
def test_detached_embedding_cannot_pass_training(tmp_path):
    config = training.LocalVisionTrainingConfig(width=64, height=64, epoch_count=1, batch_size=2)
    samples = training.resolve_local_vision_samples(tmp_path, _dataset(tmp_path))
    model = training.build_local_vision_model(config, pretrained_backbone=False)
    before = {name: value.clone() for name, value in model.named_parameters()}
    handle = model.embedding_head.register_forward_hook(
        lambda _module, _args, output: output.detach()
    )
    try:
        with pytest.raises(RuntimeError, match="EMBEDDING_SUPERVISION_DISCONNECTED"):
            training.train_local_vision_model(model, samples, config)
    finally:
        handle.remove()
    assert all(torch.equal(value, dict(model.named_parameters())[name])
               for name, value in before.items())


# 功能：
#   禁止只监督分割却声称训练了控制端向量的配置。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_segmentation_only_is_not_a_supervised_control_embedding():
    with pytest.raises(ValueError, match="embedding supervision"):
        training.LocalVisionTrainingConfig(
            semantic_loss_weight=1.0, traversability_loss_weight=0.0,
            scene_loss_weight=0.0, quality_loss_weight=0.0,
        )


# 功能：
#   CUDA 环境缺失时明确拒绝，而 CPU 请求仍按显式设备解析。
# 输入：
#   monkeypatch：仅模拟设备可用性，不声称本机已完成 GPU 训练。
# 输出：
#   None：不返回业务数据。
def test_cuda_request_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA_UNAVAILABLE"):
        training.local_vision_training_device(training.LocalVisionTrainingConfig(device="cuda"))
    assert str(training.local_vision_training_device(training.LocalVisionTrainingConfig())) == "cpu"


# 功能：
#   实际导出五输出图，核对各张量与 PyTorch 一致，并确认导出不改变原模型模式。
# 输入：
#   tmp_path：隔离 ONNX 输出目录，不修改任何产品模型。
# 输出：
#   None：不返回业务数据。
def test_supervised_embedding_onnx_matches_all_outputs(tmp_path):
    config = training.LocalVisionTrainingConfig(width=64, height=64)
    model = training.build_local_vision_model(config, pretrained_backbone=False).eval()
    inputs = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        expected = [tensor.numpy() for tensor in model(inputs)]
    model.train()
    model.backbone.eval()
    previous_modes = {name: module.training for name, module in model.named_modules()}
    path = tmp_path / "synthetic-only.onnx"
    training.export_local_vision_onnx(model, path, config)
    assert previous_modes == {name: module.training for name, module in model.named_modules()}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    outputs = session.run(None, {"forward_rgb": inputs.numpy()})
    assert len(outputs) == len(expected) == 5
    assert outputs[0].shape == (1, 139)
    for observed, reference in zip(outputs, expected, strict=True):
        np.testing.assert_allclose(observed, reference, atol=1e-5, rtol=1e-4)


# 功能：
#   对同一组隔离样本比较导出 ONNX 和 PyTorch 的实际评估指标，禁止以替身成绩代替。
# 输入：
#   tmp_path：合成图像与导出文件的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_frozen_onnx_evaluator_matches_torch_metrics(tmp_path):
    import importlib.util
    from pathlib import Path

    source = Path(__file__).parents[1] / "scripts/evaluate_frozen_vision.py"
    spec = importlib.util.spec_from_file_location("frozen_vision_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = training.LocalVisionTrainingConfig(width=64, height=64)
    samples = training.resolve_local_vision_samples(tmp_path, _dataset(tmp_path))
    model = training.build_local_vision_model(config, pretrained_backbone=False).eval()
    path = tmp_path / "eval.onnx"
    training.export_local_vision_onnx(model, path, config)
    expected = training.evaluate_local_vision_model(model, samples, config)
    actual = module.evaluate_bytes(path.read_bytes(), config, samples)
    assert actual.mean_loss == pytest.approx(expected.mean_loss, abs=1e-5)
    assert actual.semantic_class_target_pixels == expected.semantic_class_target_pixels
    assert actual.traversability_mean_absolute_error == pytest.approx(
        expected.traversability_mean_absolute_error, abs=1e-5)


# 功能：
#   用真实优化器确认失效曝光图只训练质量与共享特征，不用隐藏场景答案训练感知头。
# 输入：
#   tmp_path：合成回归样本目录，不代表正式数据或视觉精度。
# 输出：
#   None：感知头无更新、质量与嵌入有更新且感知覆盖不被计数。
def test_quality_only_training_excludes_hidden_perception_targets(tmp_path):
    config = training.LocalVisionTrainingConfig(width=64, height=64, epoch_count=1,
        batch_size=2, freeze_backbone_epochs=1, weight_decay=0.0)
    samples = [training.LocalVisionTrainingSample.model_validate({
        **sample.model_dump(), "perception_supervision_enabled": False,
        "quality_targets": [1.0, 0.0, 0.0, 0.0],
    }) for sample in _dataset(tmp_path)]
    resolved = training.resolve_local_vision_samples(tmp_path, samples)
    model = training.build_local_vision_model(config, pretrained_backbone=False)
    before = {name: value.clone() for name, value in model.named_parameters()}
    trained, metrics = training.train_local_vision_model(model, resolved, config)
    after = dict(trained.named_parameters())
    # 关闭 weight decay，隔离“不可观测标签是否产生梯度”这一条因果关系。
    for prefix in ("semantic_head.", "traversability_head.", "scene_head."):
        perception = [name for name in before if name.startswith(prefix)]
        assert perception
        assert all(torch.equal(before[name], after[name]) for name in perception)
    for prefix in ("quality_head.", "embedding_head."):
        assert any(not torch.equal(before[name], after[name])
                   for name in before if name.startswith(prefix))
    assert metrics.semantic_sample_count == 0
    assert metrics.semantic_class_target_pixels == [0] * 8
    assert metrics.semantic_mean_iou is None
    assert metrics.scene_binary_accuracy == 0.0
    assert metrics.traversability_mean_absolute_error == 1.0


# 功能：
#   失效图的不可观测答案被改变时，共享评估结果必须保持不变，防止评价口径泄漏。
# 输入：
#   tmp_path：同一批真实解码的合成样本。
# 输出：
#   None：只有有效质量标签参与失效图评价。
def test_quality_only_metrics_ignore_arbitrary_hidden_answers(tmp_path):
    config = training.LocalVisionTrainingConfig(width=64, height=64)
    model = training.build_local_vision_model(config, pretrained_backbone=False).eval()
    samples = [training.LocalVisionTrainingSample.model_validate({
        **sample.model_dump(), "perception_supervision_enabled": False,
        "quality_targets": [1.0, 0.0, 0.0, 0.0],
    }) for sample in _dataset(tmp_path)]
    altered = [training.LocalVisionTrainingSample.model_validate({
        **sample.model_dump(), "traversability_target": 1.0 - sample.traversability_target,
        "scene_targets": [1.0 - value for value in sample.scene_targets],
    }) for sample in samples]
    expected = training.evaluate_local_vision_model(model,
        training.resolve_local_vision_samples(tmp_path, samples), config)
    actual = training.evaluate_local_vision_model(model,
        training.resolve_local_vision_samples(tmp_path, altered), config)
    assert actual == expected
