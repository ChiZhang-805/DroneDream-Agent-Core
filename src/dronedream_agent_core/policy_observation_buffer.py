"""Bounded source-ordered observations, independent of permission to infer or fly."""

from collections import deque
from dataclasses import dataclass
from threading import Lock

from .temporal_evidence import TemporalEvidence


@dataclass(frozen=True)
class PolicyObservation:
    temporal_evidence: TemporalEvidence
    state_features: tuple
    payload_features: tuple
    realtime_features: tuple
    realtime_valid_mask: tuple


class PolicyObservationBuffer:
    # 功能：建立仅保存真实特征的有界移交队列；队列不产生动作，也不延长来源有效期。
    # 输入：容量（至少覆盖一个控制窗口）；输出：独立于模型推理线程的队列。
    def __init__(self, capacity=64):
        if type(capacity) is not int or not 32 <= capacity <= 256:
            raise ValueError('POLICY_OBSERVATION_CAPACITY_INVALID')
        self._rows = deque(maxlen=capacity)
        self._lock = Lock()
        self._latest = None

    # 功能：保存编译器已冻结的数值与原始身份，严格有序；同槽修订不能增加独立帧数。
    # 输入：已通过统一特征编译器的 batch；输出：是否入队，不修改推理后端的当前历史。
    def stage(self, batch):
        if batch.temporal_evidence is None or not batch.realtime_features_ready:
            return False
        evidence = TemporalEvidence.model_validate(
            batch.temporal_evidence.model_dump(mode='python'), strict=True)
        row = PolicyObservation(evidence, tuple(batch.state_features), tuple(batch.payload_features),
                                tuple(batch.realtime_features), tuple(batch.realtime_valid_mask))
        with self._lock:
            previous = self._latest
            if previous is not None and previous.stream_id == evidence.stream_id:
                if previous == evidence:
                    return False
                delta = evidence.observed_at_unix_ms - previous.observed_at_unix_ms
                if (delta < 0 and not evidence.reset_history
                        or delta == 0 and (evidence.history_slot_revision <= previous.history_slot_revision
                                         or evidence.reset_history != previous.reset_history)
                        or evidence.sample_sha256 == previous.sample_sha256):
                    raise ValueError('POLICY_OBSERVATION_SOURCE_ORDER_INVALID')
                if evidence.reset_history:
                    self._rows.clear()
                elif delta == 0 and self._rows:
                    self._rows.pop()
            else:
                self._rows.clear()
            self._rows.append(row)
            self._latest = evidence
        return True

    # 功能：只取不晚于本次输入的同来源历史，绝不让后来到达的帧或修订泄漏进旧推理。
    # 输入：本次推理的原始时刻、来源及修订身份；输出：有界历史，未来观测保留待后续调用。
    def take_through(self, evidence):
        if evidence is None:
            return ()
        selected = []
        with self._lock:
            if self._latest is None or self._latest.stream_id != evidence.stream_id:
                return ()
            while self._rows:
                row = self._rows[0]
                source = row.temporal_evidence
                if (source.observed_at_unix_ms > evidence.observed_at_unix_ms
                        or source.observed_at_unix_ms == evidence.observed_at_unix_ms
                        and source.history_slot_revision > evidence.history_slot_revision):
                    break
                selected.append(self._rows.popleft())
        return tuple(selected)
