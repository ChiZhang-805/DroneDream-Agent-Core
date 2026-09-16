"""Conservative cleanup of requests whose transport acknowledgements can be lost.

Attempted is intentionally distinct from acknowledged. These flags describe
commands, never observed flight state or permission to skip landing telemetry.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class FlightCommandAttempts:
    """Track dispatch separately from acknowledgements for failure cleanup."""

    arm_requested: bool = False
    arm_acknowledged: bool = False
    offboard_requested: bool = False
    offboard_acknowledged: bool = False

    # 功能：
    #   在发送解锁请求前登记尝试，回执到达后才登记确认；请求不等于已经起飞。
    # 输入：
    #   client：提供解锁命令的客户端。
    #   wait：执行调用者等待、超时及取消策略的异步入口。
    # 输出：
    #   None：不返回业务数据。
    async def arm(self, client: Any, wait: Callable[[Awaitable], Awaitable]) -> None:
        # 必须先记尝试：发送后取消或丢失回执不能被误判为从未解锁。
        self.arm_requested = True
        await wait(client.arm())
        self.arm_acknowledged = True

    # 功能：
    #   分别记录外部控制请求与确认，保留回执丢失后的退出清理依据。
    # 输入：
    #   client：提供进入外部控制命令的客户端。
    #   wait：执行调用者等待策略的异步入口。
    # 输出：
    #   None：不返回业务数据。
    async def start_offboard(self, client: Any, wait: Callable[[Awaitable], Awaitable]) -> None:
        self.offboard_requested = True
        await wait(client.start_offboard())
        self.offboard_acknowledged = True

    # 功能：
    #   复制命令尝试与回执记录；这些标志不能替代原生落地遥测。
    # 输入：
    #   无。
    # 输出：
    #   evidence：不共享可变状态的命令记录字典。
    def evidence(self) -> dict[str, bool]:
        evidence = asdict(self)
        return evidence


# 功能：
#   1. 对已尝试的飞行命令执行退出清理；停止失败或记录失败不阻断降落请求。
#   2. 分开记录命令回执与原生落地状态，丢失命令回执仍尝试确认实际状态。
#   3. 独立限制命令和遥测等待；客户端必须响应异步取消，不能在事件循环中阻塞。
# 输入：
#   client：提供停控、降落与原生落地观测的异步客户端。
#   attempts：本次执行实际登记的命令尝试。
#   offboard_stopped：只有字面 True 才表示已完成停控。
#   landed：只有字面 True 才表示已确认落地。
#   landing_timeout_seconds：原生落地观察的有限正等待秒数。
#   on_landing：记录进入降落阶段的同步回调。
#   evidence：就地更新的本次清理记录，确认失败时清除残留落地观测。
#   command_timeout_seconds：每个清理命令的有限正等待秒数。
# 输出：
#   None：不返回业务数据。
async def cleanup_flight_commands(
    client: Any, *, attempts: FlightCommandAttempts, offboard_stopped: bool,
    landed: bool, landing_timeout_seconds: float, on_landing: Callable[[], None],
    evidence: dict[str, Any], command_timeout_seconds: float = 5.0,
) -> None:
    for timeout in (command_timeout_seconds, landing_timeout_seconds):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("CLEANUP_TIMEOUT_INVALID")
    if attempts.offboard_requested and offboard_stopped is not True:
        try:
            await asyncio.wait_for(client.stop_offboard(), timeout=command_timeout_seconds)
            evidence["stop_offboard"] = "completed_during_failure_cleanup"
        except (Exception, asyncio.CancelledError) as error:
            evidence["stop_offboard"] = f"failed: {type(error).__name__}: {error}"
    if attempts.arm_requested and landed is not True:
        # 本次重试失败时，后续资源收尾不能把上一次留下的观测当作新确认。
        evidence.pop("landing_observation", None)
        try:
            on_landing()
        except (Exception, asyncio.CancelledError) as error:
            evidence["landing_phase_write"] = f"failed: {type(error).__name__}: {error}"
        try:
            await asyncio.wait_for(client.land(), timeout=command_timeout_seconds)
            evidence["land_command"] = "acknowledged_during_failure_cleanup"
        except (Exception, asyncio.CancelledError) as error:
            evidence["land_command"] = f"failed: {type(error).__name__}: {error}"
        # 回执丢失不说明命令没有生效；实际落地只能由这次原生观察确认。
        try:
            observation = await asyncio.wait_for(
                client.wait_until_landed(landing_timeout_seconds), timeout=landing_timeout_seconds
            )
            if (not isinstance(observation, dict) or observation.get("state") != "ON_GROUND"
                    or observation.get("confirmed") is not True):
                raise RuntimeError("CLEANUP_NATIVE_LANDING_NOT_CONFIRMED")
            evidence["land"] = "confirmed_on_ground_during_failure_cleanup"
            evidence["landing_observation"] = observation
        except (Exception, asyncio.CancelledError) as error:
            evidence["land"] = f"failed: {type(error).__name__}: {error}"
