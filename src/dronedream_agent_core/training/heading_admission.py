"""Bind extra heading inputs to original observations, never to action labels."""

from dataclasses import dataclass
from io import BytesIO
from types import MappingProxyType

from dronedream_plugin_sdk.protocol import decode_json

from ..local_policy_port import compile_local_policy_features
from ..plugin_files import read_plugin_file
from ..precision_heading_input import current_precision_heading_input


@dataclass(frozen=True)
class HeadingAdmissionInputs:
    records: object

    # 功能：
    #   将额外精确几何绑定到同一训练观测的原始摘要、物理尺度、状态及传感器输入。
    # 输入：
    #   self：已冻结的源观测索引；sample：当前需评分的行为样本。
    # 输出：
    #   features：只含当时观测的二十三维偏航输入，无历史动作或未来标签。
    def features_for(self, sample):
        entry = self.records.get(sample.source_snapshot_sha256)
        if entry is None:
            raise ValueError("HEADING_ADMISSION_SOURCE_MISSING")
        features, batch = entry
        if (
            batch.temporal_evidence is None
            or sample.temporal_evidence is None
            or batch.temporal_evidence != sample.temporal_evidence
            or batch.pilot_control_limits != sample.pilot_control_limits
            or batch.control_feature_contract_sha256 != sample.control_feature_contract_sha256
            or batch.state_features != tuple(sample.state_features)
            or batch.realtime_features != tuple(sample.realtime_features)
            or batch.realtime_valid_mask != tuple(sample.realtime_valid_mask)
        ):
            raise ValueError("HEADING_ADMISSION_SOURCE_FEATURES_DIFFER")
        return features


# 功能：
#   读取有界且摘要完整的真实快照侧文件，在其原记录时刻重跑生产编码器，拒绝改写、过期或重复源。
# 输入：
#   path：调用方已冻结的 JSONL 文件，每行包含 snapshot 与 recorded_at_unix_ms。
# 输出：
#   inputs：只读来源索引，不能用自报八个数或教师控制替代原始感知。
def read_heading_admission_inputs(path):
    content = read_plugin_file(path, limit=256 * 1024**2)
    records = {}
    for index, line in enumerate(BytesIO(content)):
        if (index >= 250000 or not line.strip() or len(line) > 4 * 1024**2
                or not line.endswith(b"\n")):
            raise ValueError("HEADING_ADMISSION_SOURCE_LIMIT_INVALID")
        row = decode_json(line, limit=4 * 1024**2)
        if type(row) is not dict or set(row) != {"snapshot", "recorded_at_unix_ms"}:
            raise ValueError("HEADING_ADMISSION_RECORD_INVALID")
        snapshot = row["snapshot"]
        features = current_precision_heading_input(snapshot, now_unix_ms=row["recorded_at_unix_ms"])
        identity = snapshot["snapshot_sha256"]
        if identity in records:
            raise ValueError("HEADING_ADMISSION_SOURCE_DUPLICATED")
        batch = compile_local_policy_features(snapshot, include_candidate_features=False)
        if not batch.realtime_features_ready:
            raise ValueError("HEADING_ADMISSION_SOURCE_NOT_READY")
        records[identity] = features, batch
    if not records:
        raise ValueError("HEADING_ADMISSION_SOURCE_EMPTY")
    inputs = HeadingAdmissionInputs(MappingProxyType(records))
    return inputs
