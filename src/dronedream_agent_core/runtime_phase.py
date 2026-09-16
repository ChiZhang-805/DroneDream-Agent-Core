"""Bounded mission-stage context. Stage labels never confer flight authority."""

from __future__ import annotations

from pathlib import Path

from .runtime_control_io import read_runtime_object

ENDING_PHASES = frozenset({"LANDING", "LANDED", "COMPLETE", "FAILED"})
MAXIMUM_PHASE_BYTES = 64 * 1024


# 功能：
#   只提取有限阶段和检查点，暂停时保留所属执行阶段，终止标签优先于旧活跃标签。
# 输入：
#   payload：阶段消息对象，非法或缺失输入表示未知。
# 输出：
#   context：三个有限标签字段组成的上下文，不包含动作、推进或落地权限。
def phase_context(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        payload = {}

    # 功能：
    #   验证阶段标签，空白、控制字符及无效 Unicode 不能进入缓存或诊断。
    # 输入：
    #   value：候选标签。
    # 输出：
    #   label：合法原始标签，或 UNKNOWN。
    def phase(value: object) -> str:
        valid = (type(value) is str and 0 < len(value) <= 128 and bool(value.strip())
                 and all(ord(char) >= 32 and ord(char) != 127
                         and not 0xD800 <= ord(char) <= 0xDFFF for char in value))
        label = value if valid else "UNKNOWN"
        return label

    enclosing = payload.get("enclosing_executor_state")
    current = phase(payload.get("phase"))
    underlying = phase(enclosing.get("phase")) if isinstance(enclosing, dict) else current
    # 冲突状态只能让请求停止，不能让已经结束的回合重新开始。
    if current in ENDING_PHASES:
        underlying = current
    checkpoint = payload.get("checkpoint_id")
    context = {
        "phase": current,
        "executor_phase": underlying,
        "checkpoint_id": (
            checkpoint if type(checkpoint) is str and 0 < len(checkpoint) <= 256
            and bool(checkpoint.strip()) and all(ord(char) >= 32 and ord(char) != 127
                and not 0xD800 <= ord(char) <= 0xDFFF for char in checkpoint) else None
        ),
    }
    return context


# 功能：
#   有界读取普通阶段文件，重复键、非有限值、链接和读取中替换均降为未知，不缓存旧结果。
# 输入：
#   path：本次运行的阶段文件路径。
# 输出：
#   context：有限阶段上下文；可复用多久由消费端的原始时间窗口决定。
def runtime_phase_context(path: Path) -> dict[str, object]:
    try:
        context = phase_context(read_runtime_object(path, maximum_bytes=MAXIMUM_PHASE_BYTES))
    except (OSError, ValueError, UnicodeError, RecursionError):
        context = phase_context(None)
    return context
