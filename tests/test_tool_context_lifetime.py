"""Offline tool-context tests with explicitly released worker threads and no external effects."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from test_tool_control_boundaries import registry_for

from dronedream_agent_core.extensions import ExtensionRegistry
from dronedream_agent_core.tools import ToolExecutionError


# 功能：
#   验证有正在执行的工具调用时拒绝重配，完成后才允许切换并清除旧结果缓存。
# 输入：
#   setting：本例尝试切换的运行预算或扩展配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("setting", ["limits", "extensions"])
def test_configuration_cannot_cross_an_inflight_tool_call(setting):
    started = threading.Event()
    release = threading.Event()
    calls = []

    # 功能：
    #   暂停首次工具调用，返回递增观测编号以识别新上下文是否错误命中旧缓存。
    # 输入：
    #   value：本例未使用的工具输入。
    # 输出：
    #   observation：本次实际执行的观测编号。
    def handler(value):
        calls.append("called")
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        observation = {"sample": len(calls)}
        return observation

    registry = registry_for(handler)
    registry.configure_runtime_limits(maximum_calls=10, timeout_seconds=5)

    # 功能：
    #   经公开配置入口切换本例选定的上下文设置，不直接操作缓存内部字段。
    # 输入：
    #   无；使用本例闭包内的 registry 与 setting。
    # 输出：
    #   None：不返回业务数据。
    def configure():
        if setting == "limits":
            registry.configure_runtime_limits(maximum_calls=10, timeout_seconds=5)
        else:
            registry.configure_extensions(ExtensionRegistry())

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(registry.call, "test.boundary", {})
        try:
            assert started.wait(5)
            with pytest.raises(RuntimeError, match="TOOL_RUNTIME_CONFIGURATION_IN_USE"):
                configure()
        finally:
            release.set()
        assert future.result(timeout=5)[0] == {"sample": 1}

    configure()
    output, receipt = registry.call("test.boundary", {})
    assert output == {"sample": 2}
    assert "CACHE_HIT" not in receipt.issue_codes


# 功能：
#   验证处理器异常后释放逻辑调用占用，不使注册表永久停留在不可配置状态。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_call_releases_configuration_lifetime():
    # 功能：
    #   模拟本地处理器失败，不产生外部副作用。
    # 输入：
    #   value：未使用的测试输入。
    # 输出：
    #   None：不返回业务数据。
    def broken(value):
        raise RuntimeError("fixture failure")

    registry = registry_for(broken)
    with pytest.raises(ToolExecutionError):
        registry.call("test.boundary", {})
    registry.configure_runtime_limits(maximum_calls=1, timeout_seconds=1)
    registry.configure_extensions(ExtensionRegistry())


# 功能：
#   验证非字符串键不能靠规范摘要的历史键转换伪装为合法缓存输入。
# 输入：
#   source：非法键来自调用方或输入中间件。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("source", ["caller", "middleware"])
def test_invalid_keys_cannot_alias_a_valid_cache_entry(source):
    registry = registry_for(lambda _: {"accepted": True})
    registry.call("test.boundary", {"1": "same"})
    invalid = {1: "lost", "1": "same"}
    if source == "middleware":
        registry._middleware = lambda hook, value, **_: (
            invalid if hook == "before_tool_call" else value,
            None,
        )
    with pytest.raises((ValueError, ToolExecutionError)):
        registry.call("test.boundary", invalid if source == "caller" else {"1": "same"})


# 功能：
#   验证工具或输出中间件返回冲突键时拒绝结果，不能生成丢字段后的接受回执。
# 输入：
#   source：非法键来自处理器或输出中间件。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("source", ["handler", "middleware"])
def test_invalid_output_keys_do_not_receive_accepted_receipts(source):
    invalid = {1: "lost", "1": "same"}
    registry = registry_for(lambda _: invalid if source == "handler" else {"1": "same"})
    if source == "middleware":
        registry._middleware = lambda hook, value, **_: (
            invalid if hook == "after_tool_call" else value,
            None,
        )
    with pytest.raises(ToolExecutionError):
        registry.call("test.boundary", {})
    assert not registry._cache


# 功能：
#   验证批量调用从策略解析到全部结果收集都固定上下文，不能在派发之前换入新配置。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_batch_policy_resolution_holds_configuration_lifetime():
    registry = registry_for(lambda value: value)

    # 功能：
    #   在调度策略回调中尝试切换配置，验证批次在尚未启动处理器时已占用上下文。
    # 输入：
    #   plugin：未使用的当前工具定义。
    # 输出：
    #   policy：本例的串行单次调度策略。
    def resolve(plugin):
        with pytest.raises(RuntimeError, match="TOOL_RUNTIME_CONFIGURATION_IN_USE"):
            registry.configure_runtime_limits(maximum_calls=10, timeout_seconds=1)
        policy = {"maximum_attempts": 1, "parallelism": 1}
        return policy

    registry._execution_policy = resolve
    results = registry.call_batch([("test.boundary", {"value": 1})])
    assert results[0][0] == {"value": 1}
    registry.configure_runtime_limits(maximum_calls=10, timeout_seconds=1)


# 功能：
#   验证工作提交失败也会关闭每个已创建的线程池，而不是仅在获得 Future 后清理。
# 输入：
#   monkeypatch：pytest 属性替换工具。
# 输出：
#   None：不返回业务数据。
def test_submission_failure_closes_each_executor(monkeypatch):
    pools = []

    class RejectingExecutor:
        # 功能：
        #   记录没有启动线程的合成执行器，使测试只观察资源清理行为。
        # 输入：
        #   max_workers：宿主申请的工作线程数量。
        # 输出：
        #   None：不返回业务数据。
        def __init__(self, *, max_workers):
            assert max_workers == 1
            self.closed = False
            pools.append(self)

        # 功能：
        #   模拟线程任务尚未交付就失败，不调用处理器。
        # 输入：
        #   handler：未执行的工具处理器。
        #   value：未执行的工具输入。
        # 输出：
        #   None：不返回业务数据。
        def submit(self, handler, value):
            raise RuntimeError("fixture submission failure")

        # 功能：
        #   记录非阻塞且取消等待任务的清理请求。
        # 输入：
        #   wait：宿主是否等待线程结束。
        #   cancel_futures：是否取消尚未执行的任务。
        # 输出：
        #   None：不返回业务数据。
        def shutdown(self, *, wait, cancel_futures):
            assert wait is False
            assert cancel_futures is True
            self.closed = True

    monkeypatch.setattr("dronedream_agent_core.tools.ThreadPoolExecutor", RejectingExecutor)
    registry = registry_for(lambda value: value)
    with pytest.raises(ToolExecutionError):
        registry.call("test.boundary", {})
    assert len(pools) == 3
    assert all(pool.closed for pool in pools)


# 功能：
#   验证混合批次采用最严格的并行上限，不能用另一工具的宽松策略覆盖串行要求。
# 输入：
#   monkeypatch：pytest 属性替换工具。
# 输出：
#   None：不返回业务数据。
def test_batch_respects_the_most_restrictive_parallelism(monkeypatch):
    pool_sizes = []

    # 功能：
    #   记录实际申请的线程池大小后创建真实本地线程池，由宿主负责关闭。
    # 输入：
    #   max_workers：宿主申请的线程数量。
    # 输出：
    #   executor：本例实际使用的线程池。
    def create_executor(*, max_workers):
        pool_sizes.append(max_workers)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        return executor

    monkeypatch.setattr("dronedream_agent_core.tools.ThreadPoolExecutor", create_executor)
    registry = registry_for(lambda value: value)
    registry.register(replace(registry._plugins["test.boundary"], tool_id="test.parallel"))
    registry._execution_policy = lambda plugin: {
        "parallelism": 1 if plugin.tool_id == "test.boundary" else 4,
    }
    results = registry.call_batch([("test.boundary", {}), ("test.parallel", {})])
    assert len(results) == 2
    assert pool_sizes[0] == 1


# 功能：
#   验证错误类型或非正并行度在批次派发前拒绝，不通过整数强制转换悄悄启用执行。
# 输入：
#   parallelism：待拒绝的调度值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("parallelism", [0, -1, True, 1.5, "2", None])
def test_invalid_parallelism_cannot_dispatch_tools(parallelism):
    calls = []
    registry = registry_for(lambda value: calls.append(value) or value)
    registry._execution_policy = lambda _: {"parallelism": parallelism}
    with pytest.raises(RuntimeError, match="TOOL_EXECUTION_PARALLELISM_INVALID"):
        registry.call_batch([("test.boundary", {})])
    assert not calls
