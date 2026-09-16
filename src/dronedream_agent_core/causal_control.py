"""Shared causal observation packing for control training and ONNX inference."""

import math
from numbers import Real

from .control_feature_contract import SENSOR_FEATURE_COUNT
from .hashing import sha256_json
from .temporal_evidence import ObservationHistory, TemporalEvidence

CONTROL_HISTORY_LENGTH = 16
CONTROL_STATE_WIDTH = 46
CONTROL_REALTIME_WIDTH = SENSOR_FEATURE_COUNT + 10
CONTROL_OBSERVATION_WIDTH = CONTROL_STATE_WIDTH + 2 * CONTROL_REALTIME_WIDTH
CONTROL_HISTORY_WIDTH = CONTROL_OBSERVATION_WIDTH + 1
CONTROL_HISTORY_CONTRACT_SHA256 = sha256_json({
    "fields": ["state", "masked-realtime", "realtime-valid-mask", "elapsed-source-time/250ms"],
    "widths": [CONTROL_STATE_WIDTH, CONTROL_REALTIME_WIDTH, CONTROL_REALTIME_WIDTH, 1],
    "first_or_reset_delta": 0., "maximum_source_gap_ms": 250,
    "semantics": "causal source history; independent observations; current row included",
})


# 功能：
#   1. 按训练与推理共用的顺序冻结状态、遮蔽后的实时特征和有效位。
#   2. 拒绝非法维度及数值；独立有效位区分“未观测”和“确实测得零”。
# 输入：
#   state：固定宽度的状态实数列表或元组。
#   realtime：固定宽度的传感器与局部控制参考特征。
#   mask：逐特征的数值零／一有效位，不接受布尔值。
# 输出：
#   row：不包含 elapsed 列的独立浮点历史行。
def control_history_row(state, realtime, mask) -> tuple[float, ...]:
    if (any(not isinstance(values, (list, tuple)) for values in (state, realtime, mask))
            or len(state) != CONTROL_STATE_WIDTH or len(realtime) != CONTROL_REALTIME_WIDTH
            or len(mask) != CONTROL_REALTIME_WIDTH):
        raise ValueError("CAUSAL_CONTROL_FEATURE_WIDTH_INVALID")
    if any(isinstance(v, bool) or not isinstance(v, Real) or v not in (0., 1.) for v in mask):
        raise ValueError("CAUSAL_CONTROL_FEATURE_MASK_INVALID")
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in (*state, *realtime)):
        raise ValueError("CAUSAL_CONTROL_FEATURE_NONFINITE")
    try:
        state, realtime, mask = (tuple(float(v) for v in values)
                                 for values in (state, realtime, mask))
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError("CAUSAL_CONTROL_FEATURE_NONFINITE") from exc
    if any(not math.isfinite(v) for v in (*state, *realtime)):
        raise ValueError("CAUSAL_CONTROL_FEATURE_NONFINITE")
    row = (*state, *(v * valid for v, valid in zip(realtime, mask, strict=True)), *mask)
    return row


class CausalControlHistory:
    """Maintain source-time causal rows with explicit left-padding validity."""

    # 功能：
    #   创建四到三十二行的因果窗口，来源顺序和最大间隔交给共享历史层检查。
    # 输入：
    #   self：新建的控制历史对象。
    #   length：整数窗口长度。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, length: int = CONTROL_HISTORY_LENGTH) -> None:
        if type(length) is not int or not 4 <= length <= 32:
            raise ValueError("CAUSAL_CONTROL_HISTORY_LENGTH_INVALID")
        self.history = ObservationHistory(length)

    # 功能：
    #   先检查来源，再把真实来源间隔加入控制行；重复轮询不能增加历史样本数。
    # 输入：
    #   self：当前因果窗口。
    #   evidence：独立传感器观测的来源身份、毫秒时刻及重置标志。
    #   state：固定宽度状态特征。
    #   realtime：固定宽度实时特征。
    #   mask：实时特征的逐项有效位。
    # 输出：
    #   accepted：新观测被接收时为 True，精确重复来源时为 False。
    def append(self, evidence: TemporalEvidence, state, realtime, mask) -> bool:
        if not isinstance(evidence, TemporalEvidence):
            raise ValueError("TEMPORAL_EVIDENCE_INVALID")
        evidence = TemporalEvidence.model_validate(evidence.model_dump(mode="python"), strict=True)
        if evidence.observed_at_unix_ms >= 2**63:
            raise ValueError("TEMPORAL_EVIDENCE_CLOCK_INVALID")
        previous = self.history.latest
        elapsed_ms = (evidence.observed_at_unix_ms - previous.observed_at_unix_ms
                      if previous is not None and previous.stream_id == evidence.stream_id
                      and not evidence.reset_history else 0)
        elapsed = elapsed_ms / 250. if 0 < elapsed_ms <= 250 else 0.
        row = (*control_history_row(state, realtime, mask), elapsed)
        accepted = self.history.append(evidence, row, ())
        return accepted

    # 功能：
    #   导出独立的定长窗口；左侧填充行必须由零有效位与实际观测区分。
    # 输入：
    #   self：当前因果窗口。
    # 输出：
    #   values：左填充后的二维浮点特征列表。
    #   valid：每行是否为真实独立观测的零／一列表。
    def values(self) -> tuple[list[list[float]], list[float]]:
        missing = self.history.length - len(self.history.rows)
        values = ([[0.] * CONTROL_HISTORY_WIDTH for _ in range(missing)]
                  + [list(row) for row, _ in self.history.rows])
        valid = [0.] * missing + [1.] * len(self.history.rows)
        return values, valid

    # 功能：
    #   暴露独立样本是否填满窗口；不据此续期传感器或授权无人机移动。
    # 输入：
    #   self：当前因果窗口。
    # 输出：
    #   ready：窗口已包含足够独立观测时为 True。
    @property
    def ready(self) -> bool:
        ready = self.history.ready
        return ready

    # 功能：
    #   在模型部署或来源切换边界主动清除上一生命周期的历史。
    # 输入：
    #   self：当前因果窗口。
    # 输出：
    #   None：不返回业务数据。
    def clear(self) -> None:
        self.history.clear()
