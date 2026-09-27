"""Explicit offline optimization devices; deployment and export remain CPU-compatible."""

import torch


# 功能：
#   校验显式训练设备并实际探测 CUDA 分配及内核执行，不将不可用的 GPU 静默替换为 CPU。
# 输入：
#   requested：cpu 或 cuda；CUDA 使用当前进程可见的默认设备。
# 输出：
#   device：已通过可用性检查的 PyTorch 训练设备。
def causal_training_device(requested: str) -> torch.device:
    if type(requested) is not str or requested not in {'cpu', 'cuda'}:
        raise ValueError('CAUSAL_TRAINING_DEVICE_INVALID')
    device = torch.device(requested)
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CAUSAL_TRAINING_CUDA_UNAVAILABLE')
        try:
            # 驱动枚举成功不代表可分配和执行；读取标量会等待这个小内核真正完成。
            probe = torch.ones(1, device=device).sum().item()
            if probe != 1.:
                raise RuntimeError('CAUSAL_TRAINING_CUDA_PROBE_INVALID')
        except RuntimeError as error:
            raise RuntimeError('CAUSAL_TRAINING_CUDA_INITIALIZATION_FAILED') from error
    return device


# 功能：
#   记录实际优化设备和框架版本，明确冻结评价与导出仍在 CPU 进行。
# 输入：
#   device：已完成设备检查且实际承载优化的设备。
# 输出：
#   evidence：可写入训练回执的设备信息，不代表实时飞行性能。
def causal_device_evidence(device: torch.device) -> dict:
    evidence = dict(training_device=device.type,
        training_device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        evaluation_device='cpu', export_device='cpu', torch_version=str(torch.__version__),
        torch_cuda_version=torch.version.cuda if device.type == 'cuda' else None)
    return evidence
