"""Local training input and lifecycle checks, not product model qualification."""

import numpy as np
import pytest
from PIL import Image
from test_local_vision_dataset_builder import _label_map
from test_local_vision_training import _dataset

from dronedream_agent_core import local_vision_training as training


# 功能：
#   标签定义中的重复键及布尔类别编号不得冒充精确的模拟器类别契约。
# 输入：
#   tmp_path：隔离标签文件目录。
#   kind：被篡改的标签字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["duplicate", "class-bool", "background-bool"])
def test_label_map_rejects_ambiguous_ids(tmp_path, kind):
    path = tmp_path / "labels.json"
    _label_map(path)
    text = path.read_text()
    if kind == "duplicate":
        text = text.replace(
            '"background_class_id":0', '"background_class_id":9,"background_class_id":0'
        )
    elif kind == "class-bool":
        text = text.replace('"class_id":1', '"class_id":true')
    else:
        text = text.replace('"background_class_id":0', '"background_class_id":false')
    path.write_text(text)
    with pytest.raises(ValueError):
        training.load_local_vision_label_map(path)


# 功能：
#   监督目标越界或非有限值不能进入训练损失计算。
# 输入：
#   tmp_path：隔离样本目录。
#   target：非法目标值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("target", [-1.0, 2.0, float("nan"), True])
def test_sample_rejects_invalid_targets(tmp_path, target):
    raw = _dataset(tmp_path)[0].model_dump(mode="python")
    raw["scene_targets"][0] = target
    with pytest.raises(ValueError):
        training.LocalVisionTrainingSample.model_validate(raw)


# 功能：
#   已解析样本在训练前被改动时，应在张量解码前拒绝摘要漂移。
# 输入：
#   tmp_path：隔离样本目录。
# 输出：
#   None：不返回业务数据。
def test_training_tensor_rechecks_bound_image(tmp_path):
    sample, image, _ = training.resolve_local_vision_samples(tmp_path, _dataset(tmp_path))[0]
    with Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)) as replacement:
        replacement.save(image)
    config = training.LocalVisionTrainingConfig(width=64, height=64)
    with pytest.raises(ValueError, match="hash"):
        training._image_tensor(image, config, expected_sha256=sample.image_sha256)


# 功能：
#   调色板掩码按类别索引读取，不把显示颜色转成类别编号。
# 输入：
#   tmp_path：隔离掩码目录。
# 输出：
#   None：不返回业务数据。
def test_palette_mask_preserves_class_indices(tmp_path):
    path = tmp_path / "mask.png"
    with Image.new("P", (64, 64), 1) as image:
        image.putpalette([0, 0, 0, 255, 255, 255] + [0] * (768 - 6))
        image.save(path)
    mask = training._mask_tensor(path, training.LocalVisionTrainingConfig(width=64, height=64))
    assert mask.unique().tolist() == [1]


# 功能：
#   导出路径已存在时在加载模型／依赖之前拒绝，不覆盖原有模型文件。
# 输入：
#   tmp_path：隔离模型目录。
# 输出：
#   None：不返回业务数据。
def test_export_preserves_existing_model(tmp_path):
    path = tmp_path / "existing.onnx"
    path.write_bytes(b"existing model")
    with pytest.raises(FileExistsError):
        training.export_local_vision_onnx(None, path, training.LocalVisionTrainingConfig())
    assert path.read_bytes() == b"existing model"


# 功能：
#   预训练下载开关只接受精确布尔值，不将字符串 false 当作下载许可。
# 输入：
#   monkeypatch：阻止任何实际依赖加载或网络动作。
# 输出：
#   None：不返回业务数据。
def test_pretrained_flag_checked_before_dependencies(monkeypatch):
    monkeypatch.setattr(training, "_require_torch", lambda: pytest.fail("dependency reached"))
    with pytest.raises(ValueError, match="boolean"):
        training.build_local_vision_model(training.LocalVisionTrainingConfig(),
                                          pretrained_backbone="false")


# 功能：
#   无有效任务损失的配置不能运行一次看似成功但没有训练目标的循环。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_config_requires_an_active_loss():
    with pytest.raises(ValueError, match="loss"):
        training.LocalVisionTrainingConfig(semantic_loss_weight=0, traversability_loss_weight=0,
                                            scene_loss_weight=0, quality_loss_weight=0)


# 功能：
#   最近邻缩小不能掩盖原始掩码中的非法类别像素。
# 输入：
#   tmp_path：隔离图像目录。
# 输出：
#   None：不返回业务数据。
def test_invalid_label_cannot_disappear_during_resize(tmp_path):
    path = tmp_path / "labels.png"
    labels = np.ones((128, 128), dtype=np.uint8)
    labels[0, 0] = 255
    with Image.fromarray(labels) as image:
        image.save(path)
    with pytest.raises(ValueError, match="class"):
        training._mask_tensor(path, training.LocalVisionTrainingConfig(width=64, height=64))
