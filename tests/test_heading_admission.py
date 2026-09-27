"""Extra heading observations must reproduce the same measured control input."""

import json

import pytest
from test_heading_availability import live_snapshot

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_policy_port import compile_local_policy_features
from dronedream_agent_core.local_policy_training import LocalPolicyTrainingSample
from dronedream_agent_core.training.heading_admission import read_heading_admission_inputs


# 功能：
#   建立同一合成快照及行为样本，保留实际生产编码器的时间、状态和单位。
# 输入：
#   tmp_path：测试目录。
# 输出：
#   result：原始记录、样本及文件位置。
def source_case(tmp_path):
    snapshot = live_snapshot()
    snapshot["strategic_context"] = {
        "task": {
            "local_navigation_output_mode": "normalized-body-velocity",
            "normalized_pilot_control_limits": {
                "horizontal_speed_mps": 0.4,
                "vertical_speed_mps": 0.3,
                "yaw_rate_dps": 20.0,
            },
        }
    }
    snapshot["snapshot_sha256"] = sha256_json(
        {k: v for k, v in snapshot.items() if k != "snapshot_sha256"}
    )
    batch = compile_local_policy_features(snapshot, include_candidate_features=False)
    sample = LocalPolicyTrainingSample(
        temporal_evidence=batch.temporal_evidence,
        pilot_control_limits=batch.pilot_control_limits,
        source_snapshot_sha256=snapshot["snapshot_sha256"],
        control_feature_contract_sha256=batch.control_feature_contract_sha256,
        state_features=list(batch.state_features),
        realtime_features=list(batch.realtime_features),
        realtime_valid_mask=list(batch.realtime_valid_mask),
        candidate_features=[list(row) for row in batch.candidate_features],
        candidate_mask=list(batch.candidate_mask),
        navigation_expert_role="precision-maneuver-policy",
        target_action_index=11,
        target_pilot_control=[0.1, 0.1, 0.1, 0.1],
        risk_target=0.0,
    )
    record = dict(snapshot=snapshot, recorded_at_unix_ms=1001)
    path = tmp_path / "snapshots.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    result = record, sample, path
    return result


# 功能：
#   核对实际记录能生成与生产相同的无历史输入，且改变监督动作不改变模型可见的几何。
# 输入：
#   tmp_path：测试目录。
# 输出：
#   None：原始观测绑定与标签隔离符合断言。
def test_heading_admission_is_bound_and_does_not_read_labels(tmp_path):
    _, sample, path = source_case(tmp_path)
    inputs = read_heading_admission_inputs(path)
    features = inputs.features_for(sample)
    assert features[8:] == (0.0,) * 15
    sample.target_pilot_control = [-0.8, 0.4, -0.2, 0.9]
    assert inputs.features_for(sample) == features
    with pytest.raises(TypeError):
        inputs.records["new"] = (features, None)


# 功能：
#   拒绝改写快照、过期或重复来源，不把事后填写的数字当作原始图像和状态的替身。
# 输入：
#   damage：源记录损坏种类；tmp_path：测试目录。
# 输出：
#   None：所有损坏都在推理前拒绝。
@pytest.mark.parametrize(
    "damage", ["digest", "stale", "duplicate", "empty", "row", "future", "truncated"]
)
def test_heading_admission_rejects_source_damage(damage, tmp_path):
    record, _, path = source_case(tmp_path)
    if damage == "digest":
        record["snapshot"]["goal_position_m"]["x"] += 1.0
    elif damage == "stale":
        record["recorded_at_unix_ms"] = 1300
    elif damage == "future":
        record["recorded_at_unix_ms"] = 999
    elif damage == "row":
        record["features"] = [0.0] * 23
    content = json.dumps(record) + "\n"
    if damage == "truncated":
        content = content.rstrip("\n")
    path.write_text(
        "" if damage == "empty" else content * (2 if damage == "duplicate" else 1), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        read_heading_admission_inputs(path)


# 功能：
#   拒绝同名样本搭配另一份状态、时间、控制尺度或传感器特征，不能仅检查侧文件存在。
# 输入：
#   damage：样本损坏种类；tmp_path：测试目录。
# 输出：
#   None：损坏样本不得取得偏航输入。
@pytest.mark.parametrize(
    "damage", ["state", "sensor", "mask", "time", "identity", "source", "scale", "stream"]
)
def test_heading_admission_rejects_sample_substitution(damage, tmp_path):
    _, sample, path = source_case(tmp_path)
    inputs = read_heading_admission_inputs(path)
    if damage == "state":
        sample.state_features[0] += 0.1
    elif damage == "sensor":
        sample.realtime_features[0] += 0.1
    elif damage == "mask":
        sample.realtime_valid_mask[0] = 1.0 - sample.realtime_valid_mask[0]
    elif damage == "time":
        sample.temporal_evidence = sample.temporal_evidence.model_copy(
            update={"observed_at_unix_ms": 1002}
        )
    elif damage == "identity":
        sample.temporal_evidence = sample.temporal_evidence.model_copy(
            update={"sample_sha256": "f" * 64}
        )
    elif damage == "source":
        sample.source_snapshot_sha256 = "a" * 64
    elif damage == "stream":
        sample.temporal_evidence = sample.temporal_evidence.model_copy(
            update={"stream_id": "unrelated-flight-stream"}
        )
    else:
        sample.pilot_control_limits = None
    with pytest.raises(ValueError, match="HEADING_ADMISSION_SOURCE"):
        inputs.features_for(sample)
