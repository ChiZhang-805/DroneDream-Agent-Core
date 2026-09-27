"""Every exported head is a publication boundary, including non-feature outputs."""

import numpy as np
import onnxruntime as ort
import pytest
import torch

from dronedream_agent_core import local_vision_training as training


# 功能：
#   对真实导出流程分别破坏每个 ONNX 输出，确认拒绝发布并清理自有暂存文件。
# 输入：
#   tmp_path：合成模型的隔离导出目录。
#   monkeypatch：仅替换导出数值复核所用会话，模拟损坏输出。
#   head：五个输出之一的索引。
# 输出：
#   None：任意输出损坏都必须报错，不能遗留可被误用的发布文件。
@pytest.mark.parametrize("head", range(5))
def test_corrupted_head_cannot_publish(tmp_path, monkeypatch, head):
    config = training.LocalVisionTrainingConfig(width=64, height=64)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = training.build_local_vision_model(config, pretrained_backbone=False).eval()

        class CorruptedSession:
            # 功能：
            #   创建只用于负向测试的输出提供器。
            # 输入：
            #   args、kwargs：与 ORT 会话构造器相同的调用参数。
            # 输出：
            #   None：初始化不修改模型。
            def __init__(self, *args, **kwargs):
                pass

            # 功能：
            #   生成与真实网络一致的参考输出，仅将被测输出整体偏移。
            # 输入：
            #   names：调用方要求的输出名称。
            #   feeds：输入图像字典。
            # 输出：
            #   outputs：恰好一项损坏的五输出数组。
            def run(self, names, feeds):
                with torch.inference_mode():
                    outputs = [value.numpy().copy() for value in model(
                        torch.from_numpy(feeds["forward_rgb"]))]
                outputs[head] += np.float32(0.1)
                return outputs

        monkeypatch.setattr(ort, "InferenceSession", CorruptedSession)
        output = tmp_path / "must-not-publish.onnx"
        with pytest.raises(RuntimeError, match="LOCAL_VISION_EXPORT_PARITY_MISMATCH"):
            training.export_local_vision_onnx(model, output, config)
        assert not output.exists()
        assert not list(tmp_path.glob(".vision-export-*"))
    finally:
        torch.set_num_threads(previous_threads)
