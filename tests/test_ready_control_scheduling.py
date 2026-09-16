"""Scheduling tests only; readiness never proves a valid flight command."""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from dronedream_agent_core.perception_runtime import (
    EventDrivenIndoorNavigationCoordinator,
    _NavigationWorkerResult,
)
from dronedream_agent_core.runtime_scheduling import ReadyControlScheduler


# 功能：
#   Future 就绪仅提供调度提示，不消费结果、不授权动作、不提升旧候选模式的权限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_completed_future_is_only_a_hint_and_cannot_promote_legacy_results():
    coordinator = object.__new__(EventDrivenIndoorNavigationCoordinator)
    coordinator.control_output_mode = "normalized-body-velocity"
    future = Future()
    coordinator._pending = SimpleNamespace(future=future)
    assert not coordinator.continuous_result_ready
    assert coordinator.continuous_request_pending
    future.set_result(_NavigationWorkerResult(snapshot={}, model_result=object()))
    assert coordinator.continuous_result_ready
    assert not coordinator.continuous_request_pending
    assert coordinator._pending.future is future  # Hint does not consume or authorize it.
    coordinator.control_output_mode = "legacy-candidate-selection"
    assert not coordinator.continuous_result_ready
    assert not coordinator.continuous_request_pending
    coordinator._pending = None
    coordinator.control_output_mode = "normalized-body-velocity"
    assert not coordinator.continuous_result_ready
    assert not coordinator.continuous_request_pending


# 功能：
#   失败或缺失的结果不能延迟新深度接入，但仍保留给原有结果收取流程。
# 输入：
#   kind：结果失败方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["failure", "exception", "cancelled", "missing", "empty"])
def test_failed_result_does_not_delay_fresh_depth_but_remains_available_for_poll(kind):
    coordinator = object.__new__(EventDrivenIndoorNavigationCoordinator)
    coordinator.control_output_mode = "normalized-body-velocity"
    future = Future()
    coordinator._pending = SimpleNamespace(future=future)
    if kind == "failure":
        future.set_result(
            _NavigationWorkerResult(snapshot={}, failure_reason="INPUT_BUDGET_EXHAUSTED")
        )
    elif kind == "exception":
        future.set_exception(RuntimeError("invocation failed"))
    elif kind == "cancelled":
        future.cancel()
    elif kind == "missing":
        future.set_result(None)
    else:
        future.set_result(_NavigationWorkerResult(snapshot={}))
    assert not coordinator.continuous_result_ready
    assert not coordinator.continuous_request_pending
    assert coordinator._pending.future is future and future.done()


# 功能：
#   建立仅用于调度测试的合法提示条件，不代表可飞行动作。
# 输入：
#   无。
# 输出：
#   flags：独立序列、结果、健康、预算和故障提示。
def conditions():
    flags = dict(
        depth_sequence=1,
        result_ready=True,
        perception_healthy=True,
        features_retain_dispatch_budget=True,
        depth_processing_failed=False,
        fault_active=False,
    )
    return flags


# 功能：
#   同一深度序列最多消费一次优先机会，后续独立新深度才能再次交接。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_priority_cannot_starve_independent_depth_ingestion():
    scheduler = ReadyControlScheduler()
    assert scheduler.eligible(**conditions())
    scheduler.consume(1)
    assert not scheduler.eligible(**conditions())
    assert not scheduler.eligible(**{**conditions(), "depth_sequence": 0})
    assert scheduler.eligible(**{**conditions(), "depth_sequence": 2})
    with pytest.raises(ValueError, match="INDEPENDENT_DEPTH_SEQUENCE"):
        scheduler.consume(1)
    scheduler.consume(2)
    assert not scheduler.eligible(**{**conditions(), "depth_sequence": 2})


# 功能：
#   缺少健康、预算或新序列时，不提供控制结果交接机会。
# 输入：
#   change：不满足调度要求的字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "change",
    [
        {"result_ready": False},
        {"perception_healthy": False},
        {"features_retain_dispatch_budget": False},
        {"depth_processing_failed": True},
        {"fault_active": True},
        {"depth_sequence": 0},
    ],
)
def test_missing_source_or_budget_cannot_enable_priority(change):
    scheduler = ReadyControlScheduler()
    assert not scheduler.eligible(**{**conditions(), **change})
    assert scheduler.eligible(**conditions())


# 功能：
#   将合法条件改为模型仍在运行的有限等待状态。
# 输入：
#   updates：本用例明确覆盖的时钟或提示字段。
# 输出：
#   flags：等待测试使用的完整条件。
def waiting(**updates):
    flags = {
        **conditions(),
        "now_monotonic": 1.0,
        "continuous_pending": True,
        "result_ready": False,
        **updates,
    }
    return flags


# 功能：
#   每个新深度只允许一次有限等待，结果返回或消费后不能重新阻塞感知。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_running_inference_gets_one_finite_handoff_without_depth_starvation():
    scheduler = ReadyControlScheduler()
    assert scheduler.await_pending_result(**waiting())
    assert scheduler.await_pending_result(**waiting(now_monotonic=1.034))
    assert not scheduler.await_pending_result(**waiting(now_monotonic=1.036))
    assert not scheduler.await_pending_result(**waiting(now_monotonic=2.0))
    assert scheduler.await_pending_result(**waiting(depth_sequence=2, now_monotonic=2.0))
    # A reply terminates the waiting opportunity and gets the normal admission
    # checks; it does not become authorized merely because it is ready.
    assert not scheduler.await_pending_result(
        **waiting(depth_sequence=2, result_ready=True, now_monotonic=2.01)
    )
    assert scheduler.eligible(**{**conditions(), "depth_sequence": 2})
    scheduler.consume(2)
    assert not scheduler.await_pending_result(**waiting(depth_sequence=2, now_monotonic=2.02))


# 功能：
#   一旦失去等待条件立即终止该次机会，即使随后条件恢复也不重复等待。
# 输入：
#   change：中途失去的等待条件。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "change",
    [
        {"continuous_pending": False},
        {"result_ready": True},
        {"perception_healthy": False},
        {"features_retain_dispatch_budget": False},
        {"depth_processing_failed": True},
        {"fault_active": True},
    ],
)
def test_handoff_cancels_immediately_on_loss_of_permission_and_cannot_restart(change):
    scheduler = ReadyControlScheduler()
    assert scheduler.await_pending_result(**waiting())
    assert not scheduler.await_pending_result(**waiting(now_monotonic=1.001, **change))
    assert not scheduler.await_pending_result(**waiting(now_monotonic=1.002))


# 功能：
#   以手工时钟核对有界暂停统计，测试期间不关闭 GC，也不遗留回调。
# 输入：
#   monkeypatch：只替换诊断器时钟的局部工具。
# 输出：
#   None：不返回业务数据。
def test_gc_pause_measurement_is_bounded_and_never_disables_collection(monkeypatch):
    import gc

    from dronedream_agent_core import runtime_scheduling as module

    enabled, callbacks = gc.isenabled(), list(gc.callbacks)
    now = [1.0]
    monkeypatch.setattr(
        module, "time", SimpleNamespace(perf_counter=lambda: now[0], time=lambda: now[0])
    )
    monitor = module.InterpreterPauseMonitor()
    assert monitor._callback in gc.callbacks
    # Manual clock/events below must not mix with an unrelated real collection
    # caused by pytest allocations while automatic collection remains enabled.
    gc.callbacks.remove(monitor._callback)
    try:
        for _ in range(70):
            monitor._observe("start", {"generation": 2})
            now[0] += 0.05
            monitor._observe("stop", {"generation": 2, "collected": 20})
    finally:
        result = monitor.close()
    assert gc.isenabled() == enabled and list(gc.callbacks) == callbacks
    assert result["collection_count"] == 70
    assert len(result["longest_pauses"]) == 16
    assert result["generations"]["2"]["collected"] == 1400
    assert result["maximum_pause_ms"] == pytest.approx(50.0)
    assert len(result["latest_pauses_at_least_one_ms"]) == 64
