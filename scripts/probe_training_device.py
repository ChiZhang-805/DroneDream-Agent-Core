"""Measure the actual fine-tuning step on an explicitly selected device."""

import argparse
import json
import time
from pathlib import Path

from dronedream_agent_core.local_vision_training import (
    LocalVisionTrainingConfig,
    build_local_vision_model,
    local_vision_training_device,
)
from dronedream_agent_core.runtime_control_io import publish_runtime_json


# 功能：
#   以实际网络前向、反向和优化器更新测量单卡负载，明确区别于真实数据精度验收。
# 输入：
#   config、weights、digest、steps：训练尺寸、预训练文件、摘要及有界实测批次数。
# 输出：
#   report：实测步耗时、吞吐及峰值显存；不授予模型或飞行资格。
def probe(config, weights, digest, steps):
    import torch

    if type(steps) is not int or not 2 <= steps <= 100:
        raise ValueError("TRAINING_PROBE_STEP_BUDGET_INVALID")
    device = local_vision_training_device(config)
    model = build_local_vision_model(config, pretrained_backbone=True,
        weights_path=weights, weights_sha256=digest).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    images = torch.rand(config.batch_size, 3, config.height, config.width, device=device)
    embedding_parameter = next(model.embedding_head.parameters())
    before = embedding_parameter.detach().clone()
    timings = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = sum(tensor.square().mean() for tensor in outputs[1:])
        if not torch.isfinite(loss):
            raise ValueError("TRAINING_PROBE_NONFINITE_LOSS")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - started) * 1000)
    if torch.equal(before, embedding_parameter):
        raise ValueError("TRAINING_PROBE_EMBEDDING_NOT_UPDATED")
    report = {"schema": "dronedream.training-device-probe.v1", "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "config": config.model_dump(mode="json"), "steps": steps, "step_ms": timings,
        "steady_mean_step_ms": sum(timings[1:]) / len(timings[1:]),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda" else None,
        "initialization": model.initialization_record, "synthetic_inputs": True,
        "data_pipeline_throughput_measured": False, "flight_qualification_granted": False}
    return report


# 功能：
#   显式指定 CPU/CUDA 设备进行有界测量，并独占保存实测回执。
# 输入：
#   命令行参数：设备、批次、步骤、权重路径和回执路径。
# 输出：
#   exit_code：测量完成返回零，设备或运算异常直接失败。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    config = LocalVisionTrainingConfig(device=args.device, batch_size=args.batch_size)
    result = probe(config, args.weights, args.sha256, args.steps)
    publish_runtime_json(args.receipt, result, replace_existing=False)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
