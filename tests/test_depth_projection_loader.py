"""Missing old-wheel support is distinct from a broken installed depth kernel."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from dronedream_agent_core import metric_scan_native


# 功能：在隔离命名空间加载实际适配器，不改写应用已导入模块；输入：假内核；输出：适配器。
def load(monkeypatch, backend):
    monkeypatch.setattr(metric_scan_native, 'backend', backend)
    path = Path(metric_scan_native.__file__).with_name('depth_projection_native.py')
    spec = importlib.util.spec_from_file_location('dronedream_agent_core._depth_loader_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 功能：未安装或旧扫描内核不冒称已启用深度加速；输入：兼容缺失；输出：明确参考路径。
@pytest.mark.parametrize('backend', [None, SimpleNamespace(CONTRACT='exact-scan-max-hit-first-v1')])
def test_absent_depth_kernel_keeps_reference(monkeypatch, backend):
    module = load(monkeypatch, backend)
    assert not module.NATIVE_DEPTH_AVAILABLE
    with pytest.raises(RuntimeError, match='NOT_INSTALLED'):
        module.project_native_depth(b'', 0, None)


# 功能：声称支持但协议/入口错误的内核必须报错；输入：不兼容模块；输出：拒绝加载。
@pytest.mark.parametrize('backend', [SimpleNamespace(DEPTH_CONTRACT='wrong'),
    SimpleNamespace(DEPTH_CONTRACT='all-pixels-nearest-radial-v1', project_depth=None)])
def test_incompatible_depth_kernel_does_not_fall_back(monkeypatch, backend):
    with pytest.raises(RuntimeError, match='CONTRACT_MISMATCH'):
        load(monkeypatch, backend)
