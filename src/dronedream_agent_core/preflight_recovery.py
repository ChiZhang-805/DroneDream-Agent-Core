"""Pre-arm startup supervision only; never owns actuators or in-flight recovery.

Transport adapters supply one attempt, their transient-error classifier and
owned-resource cleanup. A single monotonic deadline includes attempts/backoff;
cleanup has its own finite bound. Sensor readiness remains a separate check.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


class TelemetryRateSetupPending(RuntimeError):
    """A required rate request failed transiently; unsupported sources are not retryable."""


# 功能：
#   在任何重连之前核对实际命令尝试，取消与解锁回执丢失都不能被当作尚未解锁。
# 输入：
#   motion_requested：读取本次客户端是否已经请求解锁或外部控制的回调。
#   abort_check：用户停止及运行终止条件检查。
# 输出：
#   None：不返回业务数据。
def assert_preflight_recovery_allowed(
    motion_requested: Callable[[], bool], abort_check: Callable[[], None],
) -> None:
    abort_check()
    if motion_requested() is not False:
        raise RuntimeError("PREFLIGHT_RECOVERY_FORBIDDEN_AFTER_MOTION_REQUEST")


# 功能：
#   1. 在唯一总期限内执行起飞前准备，对明确的临时故障清理后退避重试。
#   2. 停止、取消、安全故障和清理失败立即终止；解锁请求后绝不重连或重新起飞。
#   3. 保存每次尝试及失败阶段，最终失败仍由资源拥有者收尾，不隐瞒残留资源。
# 输入：
#   attempt：执行一轮准备的异步回调，接收剩余秒数及本轮证据字典。
#   recover：释放上一轮所拥有资源的异步回调。
#   retryable：仅判断当前异常及失败阶段能否进行起飞前恢复。
#   motion_requested、abort_check：命令尝试和外部停止检查。
#   evidence：由调用方保存的准备证据字典。
#   timeout_seconds、maximum_attempts：总等待秒数及有限尝试次数。
#   cleanup_timeout_seconds：单次清理的秒数上限。
# 输出：
#   result：准备回调返回的真实结果，不代表可以跳过传感器或解锁检查。
async def run_preflight_recovery(
    *, attempt: Callable[[float, dict], Awaitable[T]], recover: Callable[[], Awaitable[None]],
    retryable: Callable[[BaseException, dict], bool], motion_requested: Callable[[], bool],
    abort_check: Callable[[], None], evidence: dict[str, Any], timeout_seconds: float,
    maximum_attempts: int = 3, cleanup_timeout_seconds: float = 5.,
) -> T:
    for value in (timeout_seconds, cleanup_timeout_seconds):
        if type(value) not in (float, int) or not 0 < value <= 3600:
            raise ValueError("PREFLIGHT_TIMEOUT_INVALID")
    if type(maximum_attempts) is not int or not 1 <= maximum_attempts <= 3:
        raise ValueError("PREFLIGHT_ATTEMPT_LIMIT_INVALID")
    started = time.monotonic()
    deadline = started + timeout_seconds
    evidence.clear()
    evidence.update(status="running", maximum_attempts=maximum_attempts, attempts=[])
    last_error = None
    try:
        for number in range(1, maximum_attempts + 1):
            assert_preflight_recovery_allowed(motion_requested, abort_check)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            row = {"attempt": number, "stage": "connect", "status": "running"}
            evidence["attempts"].append(row)
            try:
                result = await asyncio.wait_for(attempt(remaining, row), timeout=remaining)
                assert_preflight_recovery_allowed(motion_requested, abort_check)
                if time.monotonic() >= deadline:
                    raise TimeoutError("PREFLIGHT_TOTAL_DEADLINE_EXCEEDED")
                row["status"] = evidence["status"] = "ready"
                evidence["successful_attempt"] = number
                return result
            except BaseException as error:
                last_error = error
                row.update(status="failed", error_type=type(error).__name__)
                # 取消、进程退出不进入恢复分类；保留异常，不把取消包成超时。
                can_retry = isinstance(error, Exception) and retryable(error, row) is True
                row["recoverable_transport_failure"] = can_retry
                if not can_retry or number == maximum_attempts:
                    raise
                assert_preflight_recovery_allowed(motion_requested, abort_check)
                try:
                    await asyncio.wait_for(recover(), timeout=cleanup_timeout_seconds)
                    row["cleanup"] = "closed_and_reaped"
                except asyncio.CancelledError:
                    row["cleanup"] = "cancelled"
                    raise
                except Exception as cleanup_error:
                    row["cleanup"] = "failed:" + type(cleanup_error).__name__
                    raise RuntimeError("PREFLIGHT_RECOVERY_CLEANUP_FAILED") from cleanup_error
                delay = min(.25 * 2 ** (number - 1), 1., max(0., deadline-time.monotonic()))
                row["backoff_seconds"] = delay
                until = time.monotonic() + delay
                while time.monotonic() < until:
                    assert_preflight_recovery_allowed(motion_requested, abort_check)
                    await asyncio.sleep(min(.05, max(0., until-time.monotonic())))
        raise TimeoutError("PREFLIGHT_TOTAL_DEADLINE_EXCEEDED") from last_error
    except BaseException as error:
        evidence["status"] = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
        evidence["failure_type"] = type(error).__name__
        raise
    finally:
        evidence["elapsed_seconds"] = time.monotonic() - started
