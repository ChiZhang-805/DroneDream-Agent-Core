"""Guard the test clock boundary without changing production timing or flight limits."""

import asyncio
import time
from types import SimpleNamespace

import pytest
from clock_fixtures import isolate_monotonic, isolate_time


# 功能：
#   验证模块时钟冻结时真实异步等待仍会完成，退出替换范围后恢复原绑定。
# 输入：
#   monkeypatch：控制测试替换作用域的工具。
# 输出：
#   None：无返回值。
def test_local_clock_does_not_freeze_asyncio_or_other_modules(monkeypatch):
    module = SimpleNamespace(time=time)
    original = time.monotonic
    with monkeypatch.context() as patch:
        isolate_monotonic(patch, module, lambda: 100.0)
        assert time.monotonic is original
        assert module.time.monotonic() == 100.0
        asyncio.run(asyncio.wait_for(asyncio.sleep(0.001), timeout=1.0))
        assert module.time.monotonic() == 100.0
    assert module.time is time


# 功能：
#   验证连续替换单调钟、墙钟及睡眠时保留其他接口，且不修改进程共享模块。
# 输入：
#   monkeypatch：隔离模块时间接口的工具。
# 输出：
#   None：无返回值。
def test_sequential_clock_overrides_preserve_unmodified_functions(monkeypatch):
    module = SimpleNamespace(time=time)
    original_sleep, original_wall = time.sleep, time.time
    sleeps = []
    isolate_monotonic(monkeypatch, module, lambda: 4.0)
    isolate_time(monkeypatch, module, time=lambda: 5.0, sleep=sleeps.append)
    module.time.sleep(0.2)
    assert sleeps == [0.2]
    assert module.time.monotonic() == 4.0 and module.time.time() == 5.0
    assert time.sleep is original_sleep and time.time is original_wall
    assert module.time.perf_counter is time.perf_counter


# 功能：
#   拒绝不存在或不可调用的时间替换，错误输入不能留下半应用的模块绑定。
# 输入：
#   monkeypatch：隔离模块绑定的工具。
#   replacement：非法时间接口名称或替换值。
# 输出：
#   None：无返回值。
@pytest.mark.parametrize("replacement", [{"missing_clock": lambda: 1}, {"time": None}])
def test_bad_clock_override_does_not_replace_module(monkeypatch, replacement):
    module = SimpleNamespace(time=time)
    with pytest.raises(ValueError, match="existing callable"):
        isolate_time(monkeypatch, module, **replacement)
    assert module.time is time
