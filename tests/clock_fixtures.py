"""Module-local clock substitution that leaves real schedulers and other tests alone."""

from types import SimpleNamespace


# 功能：
#   复制被测模块的时间接口后替换指定函数，避免影响其他模块、线程和调度器。
# 输入：
#   monkeypatch：测试结束后恢复模块绑定的测试工具。
#   module：持有 time 接口的被测模块。
#   clocks：需要替换的时间函数名及对应测试函数。
# 输出：
#   None：无返回值。
def isolate_time(monkeypatch, module, **clocks):
    replacement = SimpleNamespace(**vars(module.time))
    for name, clock in clocks.items():
        if not callable(getattr(replacement, name, None)) or not callable(clock):
            raise ValueError("Clock substitution requires an existing callable time function")
        setattr(replacement, name, clock)
    monkeypatch.setattr(module, "time", replacement)


# 功能：
#   1. 只替换被测模块的单调时钟，不修改进程共享的 time 模块。
#   2. 保留其余时间函数，使 asyncio、线程等待和真实超时继续正常推进。
# 输入：
#   monkeypatch：测试结束后恢复模块绑定的测试工具。
#   module：使用 module.time.monotonic 的被测模块。
#   clock：当前用例控制的单调时钟函数。
# 输出：
#   None：无返回值。
def isolate_monotonic(monkeypatch, module, clock):
    isolate_time(monkeypatch, module, monotonic=clock)
