"""Finish independent executor resources even when a diagnostic operation fails.

This runs after flight-command cleanup. It cannot confirm landing, send motion,
or turn a failed flight into a completed one. Failures remain in the receipt;
an original flight exception must not be replaced by a later logging error.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any


# 功能：
#   1. 在飞行命令清理之后独立释放发布器、记录器和客户端，逐项保留失败。
#   2. 只有原生落地观测能写 LANDED 阶段；资源收尾不改变飞行本身的成败事实。
#   3. 最后尝试保存回执；已有飞行异常时不以清理异常覆盖它，否则汇总抛错。
# 输入：
#   client：提供航向证据与关闭方法的执行客户端。
#   native_publisher：可选原生状态发布器，关闭须返回 drained 确认。
#   application_writer：可选同步控制应用记录器，关闭须返回 complete 确认。
#   control_pacer：可选控制节拍统计源。
#   safety_receiver：可选安全传输统计源，缺省使用显式文件兼容标记。
#   timing：就地记录飞行状态、清理结果与失败的字典。
#   write_terminal_phase：同步写入结束阶段的回调。
#   write_timing：持久化完整计时回执的回调。
#   native_closed：原生发布已排空后登记关闭的回调。
#   primary_failure：调用者是否正在传播先前的飞行异常。
#   snapshots：可选的当前执行器后台证据快照写入器。
#   phase_broadcaster：可选的当前执行器实时阶段广播器。
# 输出：
#   None：不返回业务数据。
async def finalize_executor_resources(
    *, client: Any, native_publisher: Any, application_writer: Any,
    control_pacer: Any, safety_receiver: Any, timing: dict,
    write_terminal_phase: Callable[[str], None], write_timing: Callable[[dict], None],
    native_closed: Callable[[], None], primary_failure: bool,
    snapshots: Any = None, phase_broadcaster: Any = None,
) -> None:
    errors: dict[str, str] = {}
    cleanup = timing.get("cleanup", {})
    if not isinstance(cleanup, dict):
        # 坏诊断值不阻断关闭：保留原值，并另建正常字典记录后续资源处理。
        timing["invalid_cleanup_evidence"] = cleanup
        errors["cleanup_evidence"] = "EXECUTOR_CLEANUP_EVIDENCE_INVALID"
        cleanup = {}
    timing["cleanup"] = cleanup

    # 功能：
    #   执行一项同步或异步操作，把异常和取消记入独立失败表以继续处理其他资源。
    # 输入：
    #   name：该资源在失败表中的唯一名称。
    #   operation：延迟取用资源方法的零参数回调。
    # 输出：
    #   result：操作返回值，失败时为 None。
    async def attempt(name: str, operation: Callable):
        try:
            value = operation()
            result = await value if inspect.isawaitable(value) else value
        except (Exception, asyncio.CancelledError) as error:
            errors[name] = f"{type(error).__name__}: {error}"
            result = None
        return result

    # 功能：
    #   优先读取已排空的航向报告；没有新接口时保留明确的旧报告接口兼容。
    # 输入：
    #   无。
    # 输出：
    #   heading：报告或可等待对象，两种接口都没有时为 None。
    def heading_evidence():
        getter = getattr(client, "flush_heading_tracking_evidence", None)
        if not callable(getter):
            getter = getattr(client, "heading_tracking_evidence", None)
        heading = getter() if callable(getter) else None
        return heading

    heading = await attempt("heading_tracking", heading_evidence)
    if heading is not None:
        timing["heading_tracking"] = heading
        if isinstance(heading, dict) and heading.get("writer_issue") is not None:
            errors["heading_tracking"] = "HEADING_DIAGNOSTIC_PUBLICATION_INCOMPLETE"
    if timing.get("status") != "complete":
        observation = cleanup.get("landing_observation", {})
        # 文本前缀或阶段名称不能代替原生落地确认。
        confirmed = (isinstance(observation, dict) and observation.get("state") == "ON_GROUND"
                     and observation.get("confirmed") is True)
        await attempt("terminal_phase", lambda: write_terminal_phase(
            "LANDED" if confirmed else "FAILED"))

    if native_publisher is not None:
        # 属性求值也属于资源操作：对象部分初始化或属性抛错不应绕过 attempt。
        publication = await attempt("native_state_publication", lambda: native_publisher.close())
        if publication is not None:
            timing["native_state_publication"] = publication
        if isinstance(publication, dict) and publication.get("drained") is True:
            await attempt("native_publication_flag", native_closed)
        else:
            errors.setdefault("native_state_publication", "NATIVE_PUBLICATION_NOT_DRAINED")
    if control_pacer is not None:
        cadence = await attempt("local_control_cadence", lambda: control_pacer.summary())
        if cadence is not None:
            timing["local_control_cadence"] = cadence
    if application_writer is not None:
        writer = await attempt("control_application_writer",
                               lambda: asyncio.to_thread(application_writer.close))
        if writer is not None:
            timing["control_application_writer"] = writer
        if not isinstance(writer, dict) or writer.get("complete") is not True:
            errors.setdefault("control_application_writer",
                              "CONTROL_APPLICATION_WRITER_NOT_DRAINED")
    transport = await attempt("local_safety_transport", lambda: (
        safety_receiver.summary() if safety_receiver is not None
        else {"transport": "explicit-file-compatibility"}))
    if transport is not None:
        timing["local_safety_transport"] = transport
    # 最终阶段先发布；随后停止广播和落盘，完整性进入同一执行回执。
    for name, resource in (("phase_broadcaster", phase_broadcaster), ("snapshots", snapshots)):
        if resource is None:
            continue
        result = await attempt(name, lambda resource=resource, name=name: (
            resource.close() if name == "phase_broadcaster"
            else asyncio.to_thread(resource.close)))
        timing[name] = result
        if not isinstance(result, dict) or result.get("complete") is not True:
            errors.setdefault(name, "EXECUTOR_LIVE_STATE_RESOURCE_INCOMPLETE")
    await attempt("client_close", lambda: client.close())
    timing.setdefault("cleanup", {})["close"] = (
        "failed: " + errors["client_close"] if "client_close" in errors else "completed")

    # 功能：
    #   刷新收尾成败与错误副本，保留原飞行状态及失败，不能把失败改成成功。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def record_errors():
        timing["resource_finalization"] = {"complete": not errors, "errors": dict(errors)}
        if errors:
            timing.setdefault("flight_execution_status", timing.get("status"))
            timing["status"] = "failed"
            timing.setdefault("failure", "EXECUTOR_RESOURCE_FINALIZATION_FAILED")

    record_errors()
    # 即使前面的排空失败也尝试落盘；磁盘拒绝时只能保留内存错误，不能声称已持久化。
    await attempt("timing_receipt", lambda: write_timing(timing))
    record_errors()
    if errors and not primary_failure:
        raise RuntimeError("EXECUTOR_RESOURCE_FINALIZATION_FAILED:" + ",".join(errors))
