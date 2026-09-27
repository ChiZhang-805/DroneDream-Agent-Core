"""Optional exact CPU kernel, imported before flight; never compiles or downloads at runtime."""

try:
    import _dronedream_metric_scan as backend
except ModuleNotFoundError as error:
    if error.name != '_dronedream_metric_scan':
        raise
    backend = None

if backend is not None and (getattr(backend, 'CONTRACT', None) != 'exact-scan-max-hit-first-v1'
                            or not callable(getattr(backend, 'integrate', None))):
    raise RuntimeError('METRIC_NATIVE_CONTRACT_MISMATCH')

NATIVE_SCAN_AVAILABLE = backend is not None


# 功能：
#   将已冻结且完整验证的射线交给本地精确归约器，失败时不上交部分证据也不隐式重试。
# 输入：
#   rays：冻结原点、端点、步数、强度和命中位；minimum、resolution：地图几何。
# 输出：
#   updates：占用与自由证据二元组。
def prepare_native_scan(rays, minimum, resolution):
    if backend is None:
        raise RuntimeError('METRIC_NATIVE_NOT_INSTALLED')
    updates = backend.integrate(rays, minimum, resolution)
    return updates
