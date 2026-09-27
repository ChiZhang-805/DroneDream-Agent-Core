"""Real tensor transfer tests; source tensors are fixtures, not pretrained performance evidence."""

import hashlib

import pytest
import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from dronedream_agent_core.local_vision_training import LOCAL_VISION_SEMANTIC_CLASSES
from dronedream_agent_core.training import vision_initialization as initialization


# 功能：
#   构造含二十一类的实际网络权重并通过离线文件迁移，核对通用层及 person 分类行。
# 输入：
#   tmp_path：隔离的权重文件目录。
# 输出：
#   None：不返回业务数据。
def test_full_segmentation_transfer_preserves_generic_layers_and_person(tmp_path):
    source = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=21)
    with torch.no_grad():
        source.classifier.low_classifier.weight.fill_(0.123)
        source.classifier.high_classifier.bias.fill_(0.456)
    path = tmp_path / "fixture.pt"
    torch.save(source.state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    model, record = initialization.initialize_vision_base(
        LOCAL_VISION_SEMANTIC_CLASSES, pretrained=True, source="coco-voc-segmentation",
        weights_path=path, weights_sha256=digest)
    assert record["copied_semantic_classes"] == ["person"]
    assert record["file_sha256"] == digest
    for name, value in source.classifier.cbr.state_dict().items():
        assert torch.equal(value, model.classifier.cbr.state_dict()[name])
    assert torch.equal(source.classifier.low_classifier.weight[15],
                       model.classifier.low_classifier.weight[5])
    assert torch.equal(source.classifier.high_classifier.bias[15],
                       model.classifier.high_classifier.bias[5])
    assert not torch.equal(source.classifier.low_classifier.weight[0],
                           model.classifier.low_classifier.weight[0])
    assert model.classifier.low_classifier.out_channels == 8


# 功能：
#   不匹配摘要或非有限参数不能作为初始化输入。
# 输入：
#   tmp_path、bad：隔离文件目录及故障类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", ["hash", "nan"])
def test_weight_integrity_is_checked_before_transfer(tmp_path, bad):
    path = tmp_path / "fixture.pt"
    torch.save({"test": torch.tensor([float("nan") if bad == "nan" else 1.0])}, path)
    digest = "0" * 64 if bad == "hash" else hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        initialization.read_vision_weights(path, digest)


# 功能：
#   摘要不受字典顺序影响，但能发现单个参数内容变化。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_tensor_digest_binds_actual_content():
    state = {"a": torch.tensor(3), "b": torch.tensor([0.5])}
    digest = initialization.tensor_state_sha256(state)
    assert digest == initialization.tensor_state_sha256(dict(reversed(list(state.items()))))
    state["b"].add_(0.1)
    assert digest != initialization.tensor_state_sha256(state)
