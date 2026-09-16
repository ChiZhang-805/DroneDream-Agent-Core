"""One bounded history of independent observations, shared by training and inference."""

from __future__ import annotations

import math
from collections import deque
from numbers import Real

from pydantic import Field

from .contracts import StrictModel


class TemporalEvidence(StrictModel):
    stream_id: str = Field(min_length=1, max_length=256)
    sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_at_unix_ms: int = Field(ge=0, strict=True)
    reset_history: bool = False


# 功能：
#   冻结有限数值行，拒绝巨大容器、布尔值和非有限数值；精确维度及物理语义仍由上游契约检查。
# 输入：
#   values：状态或载荷特征的一维数值列表或元组。
# 输出：
#   row：最多八千一百九十二项的独立浮点元组。
def _feature_row(values) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)) or len(values) > 8192:
        raise ValueError("TEMPORAL_FEATURE_ROW_INVALID")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError("TEMPORAL_FEATURE_ROW_INVALID")
    try:
        row = tuple(float(value) for value in values)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError("TEMPORAL_FEATURE_ROW_INVALID") from exc
    if any(not math.isfinite(value) for value in row):
        raise ValueError("TEMPORAL_FEATURE_ROW_INVALID")
    return row


class ObservationHistory:
    """No implicit wall clock, padding-as-data, cross-mission state or replay renewal."""

    # 功能：
    #   建立有限长度的来源历史，只有独立真实观测才能增加有效行数。
    # 输入：
    #   self：新建的历史窗口。
    #   length：一到一百二十八之间的整数窗口长度。
    #   maximum_gap_ms：同一来源连续样本允许的最大毫秒间隔，不超过二百五十。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, length: int, maximum_gap_ms: int = 250) -> None:
        if (type(length) is not int or type(maximum_gap_ms) is not int
                or not 1 <= length <= 128 or not 1 <= maximum_gap_ms <= 250):
            raise ValueError("TEMPORAL_HISTORY_BOUNDS_INVALID")
        self.length, self.maximum_gap_ms = length, maximum_gap_ms
        self.rows: deque[tuple[tuple[float, ...], tuple[float, ...]]] = deque(maxlen=length)
        self.latest: TemporalEvidence | None = None

    # 功能：
    #   1. 验证并冻结来源与特征，同一观测重放不增加行数，也不续期。
    #   2. 来源切换、间隔过大或显式重启重建窗口；未授权时钟回退清空窗口并拒绝。
    # 输入：
    #   self：当前历史窗口。
    #   evidence：带摘要、来源时刻及真实布尔重置标志的身份。
    #   state：已经按模型契约编码的状态特征行。
    #   payload：已经按模型契约编码的附加特征行，可为空。
    # 输出：
    #   accepted：成功加入一份新来源样本时为 True，精确来源重放时为 False。
    def append(
        self, evidence: TemporalEvidence, state: tuple[float, ...], payload: tuple[float, ...]
    ) -> bool:
        if not isinstance(evidence, TemporalEvidence):
            raise ValueError("TEMPORAL_EVIDENCE_INVALID")
        evidence = TemporalEvidence.model_validate(evidence.model_dump(mode="python"), strict=True)
        if evidence.observed_at_unix_ms >= 2**63:
            raise ValueError("TEMPORAL_EVIDENCE_CLOCK_INVALID")
        state, payload = _feature_row(state), _feature_row(payload)
        previous = self.latest
        # 身份重放保持幂等；例如因果编码器再次调用时的派生 elapsed 字段不能创建新样本。
        if previous is not None and evidence == previous:
            return False
        if previous is not None and evidence.stream_id == previous.stream_id:
            if evidence.sample_sha256 == previous.sample_sha256:
                raise ValueError("TEMPORAL_SAMPLE_REDATED")
            delta = evidence.observed_at_unix_ms - previous.observed_at_unix_ms
            if delta == 0:
                raise ValueError("TEMPORAL_SAME_TIME_CONFLICT")
            if delta < 0 and not evidence.reset_history:
                self.rows.clear()
                self.latest = None
                raise ValueError("TEMPORAL_CLOCK_REGRESSED")
            if evidence.reset_history or delta > self.maximum_gap_ms:
                self.rows.clear()
        else:
            self.rows.clear()
        self.latest = evidence
        self.rows.append((state, payload))
        accepted = True
        return accepted

    # 功能：
    #   仅检查已累计样本是否填满窗口；当前新鲜度仍必须由控制端独立检查。
    # 输入：
    #   self：当前历史窗口。
    # 输出：
    #   ready：已包含配置数量的独立来源行时为 True。
    @property
    def ready(self) -> bool:
        ready = len(self.rows) == self.length
        return ready

    # 功能：
    #   在任务或来源生命周期边界同时移除特征行及最后身份，防止跨回合复用。
    # 输入：
    #   self：当前历史窗口。
    # 输出：
    #   None：不返回业务数据。
    def clear(self) -> None:
        self.rows.clear()
        self.latest = None
