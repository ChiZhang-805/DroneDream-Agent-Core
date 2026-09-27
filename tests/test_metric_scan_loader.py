"""Import boundaries: an absent optional accelerator differs from an incompatible installed one."""

import builtins
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'src/dronedream_agent_core/metric_scan_native.py'
NAME = '_dronedream_metric_scan'


# 功能：
#   没有安装可选内核时明确保持参考实现，禁止把缺失报告为已经加速。
# 输入：
#   monkeypatch：测试导入隔离器。
# 输出：
#   None。
def test_absent_kernel_is_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, NAME, None)
    loaded = runpy.run_path(str(MODULE))
    assert loaded['NATIVE_SCAN_AVAILABLE'] is False
    with pytest.raises(RuntimeError, match='NOT_INSTALLED'):
        loaded['prepare_native_scan']([], (0., 0., 0.), .25)


# 功能：
#   已安装模块的协议或调用入口不兼容时直接拒绝，不能暗中改用其他实现。
# 输入：
#   monkeypatch：测试隔离器；kind：模块错误类型。
# 输出：
#   None。
@pytest.mark.parametrize('kind', ['missing-contract', 'wrong-contract', 'missing-entrypoint'])
def test_incompatible_kernel_is_not_hidden(monkeypatch, kind):
    fake = SimpleNamespace(CONTRACT='exact-scan-max-hit-first-v1', integrate=lambda *args: ())
    if kind == 'missing-contract': del fake.CONTRACT
    elif kind == 'wrong-contract': fake.CONTRACT = 'old-contract'
    else: del fake.integrate
    monkeypatch.setitem(sys.modules, NAME, fake)
    with pytest.raises(RuntimeError, match='CONTRACT_MISMATCH'):
        runpy.run_path(str(MODULE))


# 功能：
#   内核内部依赖丢失不能被误判成未安装，防止加载失败被静默掩盖。
# 输入：
#   monkeypatch：测试导入隔离器。
# 输出：
#   None。
def test_nested_import_failure_remains_fatal(monkeypatch):
    original = builtins.__import__

    # 功能：
    #   仅模拟当前扩展内部的依赖缺失，其余导入仍使用原解释器。
    # 输入：
    #   name：导入模块名；args、kwargs：原导入参数。
    # 输出：
    #   result：原解释器导入结果。
    def broken_import(name, *args, **kwargs):
        if name == NAME:
            raise ModuleNotFoundError('missing dependent module', name='dependent_module')
        result = original(name, *args, **kwargs)
        return result

    monkeypatch.setattr(builtins, '__import__', broken_import)
    with pytest.raises(ModuleNotFoundError) as caught:
        runpy.run_path(str(MODULE))
    assert caught.value.name == 'dependent_module'
