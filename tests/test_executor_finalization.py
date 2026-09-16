import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dronedream_agent_core.executor_finalization import finalize_executor_resources


# 功能：
#   构造可注入单点故障的资源组合，记录释放顺序及回执写入时的独立快照。
# 输入：
#   fault：需要失败的操作名称，None 表示正常返回。
#   failure：注入异常，缺省使用磁盘错误。
#   completed：初始飞行状态是否为已完成。
# 输出：
#   resources_result：调用参数、调用顺序及已写回执列表。
def resources(fault=None, *, failure=None, completed=False):
    calls, receipts = [], []
    # 功能：
    #   为指定操作创建记录调用、返回结果或抛出故障的回调。
    # 输入：
    #   name：该操作的回执名称。
    #   result：正常路径的返回值。
    # 输出：
    #   run：本操作的同步回调。
    def operation(name, result=None):
        # 功能：
        #   记录调用顺序，在目标操作注入异常，并在写回执时复制当前完整状态。
        # 输入：
        #   args：操作实参，写回执时首项为计时字典。
        # 输出：
        #   result：夹具预设的返回值。
        def run(*args):
            calls.append(name)
            if name == fault:
                raise failure if failure is not None else OSError(name + " failed")
            if name == "timing_receipt":
                receipts.append(copy.deepcopy(args[0]))
            return result
        return run

    # 功能：
    #   模拟异步关闭客户端并登记其调用顺序。
    # 输入：
    #   无。
    # 输出：
    #   result：夹具关闭结果。
    async def close_client():
        result = operation("client_close")()
        return result

    # 功能：
    #   模拟原生发布器异步排空，成功时返回明确的 drained 标记。
    # 输入：
    #   无。
    # 输出：
    #   result：发布器排空回执。
    async def close_native():
        result = operation("native_state_publication", {"drained": True})()
        return result

    timing = {"status": "complete" if completed else "failed",
              "cleanup": {"land": "confirmed_on_ground", "landing_observation": {
                  "state": "ON_GROUND", "confirmed": True}}}
    kwargs = dict(
        client=SimpleNamespace(flush_heading_tracking_evidence=operation("heading_tracking", {}),
                               close=close_client),
        native_publisher=SimpleNamespace(close=close_native),
        application_writer=SimpleNamespace(close=operation(
            "control_application_writer", {"complete": True})),
        control_pacer=SimpleNamespace(summary=operation("local_control_cadence", {})),
        safety_receiver=SimpleNamespace(summary=operation("local_safety_transport", {})),
        timing=timing, write_terminal_phase=operation("terminal_phase"),
        write_timing=operation("timing_receipt"),
        native_closed=operation("native_publication_flag"), primary_failure=False,
    )
    resources_result = kwargs, calls, receipts
    return resources_result


# 功能：
#   逐个注入资源故障，核对其余资源仍释放且最终尝试写入失败回执。
# 输入：
#   fault：当前故障操作名称。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["heading_tracking", "terminal_phase",
    "native_state_publication", "native_publication_flag", "control_application_writer",
    "local_control_cadence", "local_safety_transport", "client_close", "timing_receipt"])
def test_each_failure_still_closes_other_resources_and_attempts_final_receipt(fault):
    kwargs, calls, receipts = resources(fault)
    with pytest.raises(RuntimeError, match="EXECUTOR_RESOURCE_FINALIZATION_FAILED:" + fault):
        asyncio.run(finalize_executor_resources(**kwargs))
    assert "control_application_writer" in calls
    assert calls[-2:] == ["client_close", "timing_receipt"]
    assert kwargs["timing"]["resource_finalization"]["complete"] is False
    assert list(kwargs["timing"]["resource_finalization"]["errors"]) == [fault]
    assert kwargs["timing"]["status"] == "failed"
    assert bool(receipts) is (fault != "timing_receipt")
    if receipts:
        assert receipts[0]["resource_finalization"]["complete"] is False


# 功能：
#   验证清理发生异常或取消时，原始飞行异常对象及原因不被替换。
# 输入：
#   failure：清理阶段的故障。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failure", [OSError("disk locked"), asyncio.CancelledError()])
def test_primary_flight_failure_is_preserved_not_replaced_by_cleanup_failure(failure):
    kwargs, calls, receipts = resources("heading_tracking", failure=failure)
    original = RuntimeError("actual flight failure")
    kwargs["timing"]["failure"] = str(original)
    # 功能：
    #   在原始异常传播的 finally 中执行资源清理。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def run():
        try:
            raise original
        finally:
            await finalize_executor_resources(**{**kwargs, "primary_failure": True})
    with pytest.raises(RuntimeError) as result:
        asyncio.run(run())
    assert result.value is original
    assert calls[-2:] == ["client_close", "timing_receipt"]
    assert receipts[0]["failure"] == str(original)
    assert "heading_tracking" in receipts[0]["resource_finalization"]["errors"]


# 功能：
#   验证已完成的飞行遇到资源失败时，保留飞行事实但整体回执必须为失败。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_finalization_cannot_leave_a_success_receipt():
    kwargs, _, receipts = resources("control_application_writer", completed=True)
    with pytest.raises(RuntimeError):
        asyncio.run(finalize_executor_resources(**kwargs))
    assert receipts[0]["status"] == "failed"
    assert receipts[0]["flight_execution_status"] == "complete"
    assert receipts[0]["cleanup"]["landing_observation"]["confirmed"] is True


# 功能：
#   验证结束阶段只依赖合法原生落地确认，不依赖看似成功的说明文字。
# 输入：
#   observation：注入的落地观测。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("observation", [{}, None, {"state": "IN_AIR", "confirmed": True},
    {"state": "ON_GROUND", "confirmed": False}, {"state": "ON_GROUND", "confirmed": True}])
def test_terminal_phase_never_uses_landing_text_without_native_confirmation(observation):
    kwargs, _, _ = resources()
    kwargs["timing"]["cleanup"]["landing_observation"] = observation
    phase = Mock()
    kwargs["write_terminal_phase"] = phase
    asyncio.run(finalize_executor_resources(**kwargs))
    confirmed = observation == {"state": "ON_GROUND", "confirmed": True}
    phase.assert_called_once_with("LANDED" if confirmed else "FAILED")


# 功能：
#   验证未排空发布器不能触发已关闭标志，但其他资源仍被处理。
# 输入：
#   publication：未确认排空的发布器返回值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("publication", [{"drained": False}, {}, None])
def test_undrained_publisher_is_not_marked_closed(publication):
    kwargs, calls, receipts = resources()
    # 功能：
    #   返回预设的未排空结果，模拟没有抛错但未完成收尾的发布器。
    # 输入：
    #   无。
    # 输出：
    #   publication：外层夹具指定的返回值。
    async def close():
        return publication
    kwargs["native_publisher"].close = close
    with pytest.raises(RuntimeError, match="native_state_publication"):
        asyncio.run(finalize_executor_resources(**kwargs))
    assert "native_publication_flag" not in calls
    assert calls[-2:] == ["client_close", "timing_receipt"]
    assert receipts[0]["resource_finalization"]["complete"] is False


# 功能：
#   验证正常释放严格保持依赖顺序，并保存飞行成功与资源完成两个独立结果。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_success_closes_in_dependency_order_without_changing_flight_status():
    kwargs, calls, receipts = resources(completed=True)
    asyncio.run(finalize_executor_resources(**kwargs))
    assert calls == ["heading_tracking", "native_state_publication", "native_publication_flag",
                     "local_control_cadence", "control_application_writer",
                     "local_safety_transport", "client_close", "timing_receipt"]
    assert receipts[0]["status"] == "complete"
    assert receipts[0]["resource_finalization"] == {"complete": True, "errors": {}}


# 功能：
#   验证缺省资源不生成虚假的关闭报告，文件兼容路径仍有明确标识。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_optional_resources_need_no_fake_default_completion():
    kwargs, calls, receipts = resources()
    for key in ("native_publisher", "application_writer", "control_pacer", "safety_receiver"):
        kwargs[key] = None
    del kwargs["client"].flush_heading_tracking_evidence
    asyncio.run(finalize_executor_resources(**kwargs))
    assert calls == ["terminal_phase", "client_close", "timing_receipt"]
    assert "native_state_publication" not in receipts[0]
    assert "control_application_writer" not in receipts[0]
    assert receipts[0]["local_safety_transport"] == {"transport": "explicit-file-compatibility"}


# 功能：
#   验证记录器正常返回却没有持久化完成标志时，仍判定收尾失败。
# 输入：
#   writer_result：记录器缺失完成确认的返回值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("writer_result", [{"complete": False}, {}, None])
def test_writer_returning_without_durable_completion_fails_closed(writer_result):
    kwargs, calls, receipts = resources(completed=True)
    kwargs["application_writer"].close = lambda: writer_result
    with pytest.raises(RuntimeError, match="control_application_writer"):
        asyncio.run(finalize_executor_resources(**kwargs))
    assert calls[-2:] == ["client_close", "timing_receipt"]
    assert receipts[0]["status"] == "failed"
    assert receipts[0]["resource_finalization"]["errors"] == {
        "control_application_writer": "CONTROL_APPLICATION_WRITER_NOT_DRAINED"}


# 功能：
#   验证受损清理字典被保留为异常证据，同时不阻止资源关闭和最终写回。
# 输入：
#   bad_cleanup：错误类型的清理记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad_cleanup", [None, [], "not-an-object"])
def test_malformed_cleanup_evidence_cannot_prevent_resource_release(bad_cleanup):
    kwargs, calls, receipts = resources()
    kwargs["timing"]["cleanup"] = bad_cleanup
    with pytest.raises(RuntimeError, match="EXECUTOR_RESOURCE_FINALIZATION_FAILED"):
        asyncio.run(finalize_executor_resources(**kwargs))
    assert calls[-2:] == ["client_close", "timing_receipt"]
    assert receipts[0]["resource_finalization"]["complete"] is False
    assert receipts[0]["invalid_cleanup_evidence"] == bad_cleanup


# 功能：
#   验证资源方法缺失也被当作该资源的失败，不能绕过其他释放及最终回执写入。
# 输入：
#   resource：缺失方法的资源参数名。
#   method：应调用但缺失的方法名。
#   error_name：该资源在回执中的失败名称。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("resource,method,error_name", [
    ("native_publisher", "close", "native_state_publication"),
    ("control_pacer", "summary", "local_control_cadence"),
    ("client", "close", "client_close"),
])
def test_missing_resource_operation_cannot_abort_remaining_cleanup(resource, method, error_name):
    kwargs, calls, receipts = resources()
    delattr(kwargs[resource], method)
    with pytest.raises(RuntimeError, match="EXECUTOR_RESOURCE_FINALIZATION_FAILED:" + error_name):
        asyncio.run(finalize_executor_resources(**kwargs))
    assert calls[-1] == "timing_receipt"
    assert "control_application_writer" in calls
    if resource != "client":
        assert "client_close" in calls
    assert "AttributeError" in receipts[0]["resource_finalization"]["errors"][error_name]
