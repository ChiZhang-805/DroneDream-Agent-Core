"""Optional exact depth kernel; old scan-only wheels keep the NumPy reference."""

from .metric_scan_native import backend

_CONTRACT = 'all-pixels-nearest-radial-v1'
_declared = getattr(backend, 'DEPTH_CONTRACT', None)
if _declared is not None and (
    _declared != _CONTRACT or not callable(getattr(backend, 'project_depth', None))
):
    raise RuntimeError('DEPTH_NATIVE_CONTRACT_MISMATCH')
NATIVE_DEPTH_AVAILABLE = _declared == _CONTRACT


# 功能：用已安装且协议匹配的纯 CPU 内核完整归约原始像素，失败不静默切换实现。
# 输入：已冻结图像及经过验证的校准；无路径、设备、源钟或任务信息。
# 输出：原始像素/距离/命中元组和有效像素数；无编译、下载、飞行许可。
def project_native_depth(data, step, calibration):
    if not NATIVE_DEPTH_AVAILABLE:
        raise RuntimeError('DEPTH_NATIVE_NOT_INSTALLED')
    return backend.project_depth(data, calibration.width, calibration.height, step,
        calibration.sample_stride_pixels, calibration.intrinsics,
        calibration.minimum_depth_m, calibration.maximum_depth_m,
        calibration.no_return_mode == 'gazebo-far-clip')
