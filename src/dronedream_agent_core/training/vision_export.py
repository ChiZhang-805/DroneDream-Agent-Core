"""Numerical publication boundary for the five-output local vision network."""

OUTPUT_NAMES = ("visual_features", "semantic_logits", "traversability_logits",
                "scene_logits", "quality_logits")


# 功能：
#   1. 对零值、纹理和渐变三种固定探针比较 CPU PyTorch 与待发布 ONNX 的全部五个输出。
#   2. 任一输出维度、有限性或数值一致性不符即拒绝发布；本检查不替代精度或飞行验收。
# 输入：
#   raw：已经校验图结构且禁止外部权重的 ONNX 字节。
#   model：导出专用 CPU、eval 模型，不得传入正在训练的原模型。
#   height、width：固定图像输入尺寸。
#   feature_count：部署端消费的视觉特征宽度。
# 输出：
#   matched：所有探针的五个输出均在固定容差内一致时为 True。
def verify_vision_export(raw, model, height, width, feature_count):
    import numpy as np
    import onnxruntime as ort
    import torch

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(raw, sess_options=options, providers=["CPUExecutionProvider"])
    shape = (1, 3, height, width)
    # 独立 RNG 不推进训练随机状态；非零探针避免只有常量输入相同的错误图蒙混通过。
    rng = np.random.default_rng(805)
    probes = (np.zeros(shape, dtype=np.float32),
              rng.uniform(-2.0, 2.0, size=shape).astype(np.float32),
              np.linspace(-2.0, 2.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape))
    shapes = ((1, feature_count), (1, 8, height, width), (1, 1), (1, 6), (1, 4))
    for probe in probes:
        with torch.inference_mode():
            expected = model(torch.from_numpy(probe))
        actual = session.run(list(OUTPUT_NAMES), {"forward_rgb": probe})
        if len(actual) != 5 or len(expected) != 5:
            raise RuntimeError("LOCAL_VISION_EXPORT_PARITY_OUTPUT_COUNT")
        for name, observed, reference, required in zip(
            OUTPUT_NAMES, actual, expected, shapes, strict=True
        ):
            reference = reference.detach().cpu().numpy()
            if (observed.shape != required or reference.shape != required
                    or not np.isfinite(observed).all() or not np.isfinite(reference).all()
                    or not np.allclose(observed, reference, atol=1e-4, rtol=1e-4)):
                raise RuntimeError("LOCAL_VISION_EXPORT_PARITY_MISMATCH:" + name)
    matched = True
    return matched
