"""Process-scoped GC configuration must not leak into callers or disable GC."""
import gc
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from dronedream_agent_core import runtime_scheduling as scheduling


# 功能：
#   用隔离的配置替身检查进入条件、嵌套和异常恢复，不改变 pytest 自身的回收策略。
# 输入：
#   monkeypatch：测试替换器。
# 输出：
#   state：可观察的代阈值、启用状态和修改记录。
@pytest.fixture
def configuration(monkeypatch):
    state = {'thresholds': (700, 10, 10), 'enabled': True, 'writes': []}

    # 功能：
    #   记录测试中的回收阈值设置。
    # 输入：
    #   values：三个代阈值。
    # 输出：
    #   无：只更新测试状态。
    def set_threshold(*values):
        state['thresholds'] = values
        state['writes'].append(values)

    monkeypatch.setattr(gc, 'get_threshold', lambda: state['thresholds'])
    monkeypatch.setattr(gc, 'set_threshold', set_threshold)
    monkeypatch.setattr(gc, 'isenabled', lambda: state['enabled'])
    monkeypatch.setattr(scheduling.sys, 'version_info', (3, 12, 3))
    return state


# 功能：
#   检查嵌套不重复配置，高代阈值不变，异常退出恢复原阈值。
# 输入：
#   configuration：回收配置替身。
# 输出：
#   无：断言不满足则测试失败。
def test_nested_scope_restores_after_exception(configuration):
    configuration['thresholds'] = (700, 7, 13)
    with pytest.raises(RuntimeError):
        with scheduling.sensor_worker_gc_budget() as applied:
            assert applied == (4096, 7, 13)
            with scheduling.sensor_worker_gc_budget() as nested:
                assert nested == applied
            assert configuration['thresholds'] == applied
            raise RuntimeError('worker failed')
    assert configuration['thresholds'] == (700, 7, 13)
    assert len(configuration['writes']) == 2


# 功能：
#   验证自动回收已关闭或更宽阈值时保持调用者配置。
# 输入：
#   configuration：配置替身；enabled、threshold：需要保留的现有设置。
# 输出：
#   无：断言配置没有被改写。
@pytest.mark.parametrize('enabled,threshold', [(False, 700), (True, 0), (True, 4096), (True, 8000)])
def test_existing_configuration_is_preserved(configuration, enabled, threshold):
    configuration.update(enabled=enabled, thresholds=(threshold, 10, 10))
    with scheduling.sensor_worker_gc_budget() as applied:
        assert applied == (threshold, 10, 10)
    assert configuration['writes'] == []


# 功能：
#   未验证的解释器版本或非主线程不能启用进程级调度修改。
# 输入：
#   monkeypatch、configuration：测试替身；unsupported：需要测试的拒绝条件。
# 输出：
#   无：断言没有修改全局配置。
@pytest.mark.parametrize('unsupported', ['version', 'thread'])
def test_unsupported_context_is_unchanged(monkeypatch, configuration, unsupported):
    if unsupported == 'version':
        monkeypatch.setattr(scheduling.sys, 'version_info', (3, 14, 0))
    else:
        monkeypatch.setattr(threading, 'current_thread', lambda: object())
    with scheduling.sensor_worker_gc_budget():
        pass
    assert configuration['writes'] == []


# 功能：
#   上下文退出不得覆盖其他所有者中途设置的新阈值。
# 输入：
#   configuration：测试配置。
# 输出：
#   无：断言外部修改保留。
def test_external_change_survives_cleanup(configuration):
    with scheduling.sensor_worker_gc_budget():
        configuration['thresholds'] = (1000, 8, 9)
    assert configuration['thresholds'] == (1000, 8, 9)


# 功能：
#   在独立解释器通过真实循环引用分配证明自动回收仍工作，退出不泄露配置。
# 输入：
#   无。
# 输出：
#   无：子进程断言或超时使测试失败。
def test_actual_cycles_are_automatically_collected():
    program = '''
import gc, weakref
from dronedream_agent_core.runtime_scheduling import sensor_worker_gc_budget
class Cycle:
    pass
original = gc.get_threshold()
references = []
before = sum(item['collections'] for item in gc.get_stats())
with sensor_worker_gc_budget():
    for _ in range(30000):
        cycle = Cycle()
        cycle.self = cycle
        references.append(weakref.ref(cycle))
        del cycle
    assert gc.isenabled()
    assert sum(item['collections'] for item in gc.get_stats()) > before
    assert sum(ref() is not None for ref in references) < 4096
assert gc.get_threshold() == original
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    result = subprocess.run([sys.executable, '-c', program], env=env, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
