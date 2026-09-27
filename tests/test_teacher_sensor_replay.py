"""Replay input alignment tests use synthetic records only."""

from copy import deepcopy

import pytest
from control_fixtures import complete_feature_snapshot
from test_executed_control_training import teacher_evidence

from dronedream_agent_core.training.teacher_sensor_replay import replay_teacher_sensor_diagnostics


# 功能：
#   创建同一来源的导航上下文与当前故障记录，便于检查是否错误沿用旧健康结论。
# 输入：
#   无。
# 输出：
#   snapshots、records：仅用于重放边界验证的合成输入。
def replay_fixture():
    snapshot = deepcopy(teacher_evidence()[0])
    snapshot["strategic_context"]["task"]["navigation_goal_id"] = "goal"
    snapshot["perception_health"] = {"stream_healthy": True}
    snapshot["multimodal_sensor_snapshot"] = {"ready_for_motion": True}
    observation = {
        "sequence": 1, "observed_at_unix_ms": 1000, "source": "onboard",
        "stream_healthy": False, "stream_age_seconds": 1.0, "localization_covariance_m2": .01,
        "current_position_m": {"x": 1., "y": 0., "z": 1.},
        "current_velocity_mps": {"x": .1, "y": 0., "z": 0.},
        "target_position_m": {"x": 9., "y": 7., "z": 2.},
    }
    record = {
        "recorded_at_unix_ms": 1000, "navigation_goal_id": "goal",
        "identity_accepted": True, "observation": observation,
        "realtime_feature_snapshot": complete_feature_snapshot().model_dump(mode="json"),
    }
    return [(1000, snapshot)], [record]


# 功能：
#   拒绝用旧导航快照的健康模态填补当前缺测；单帧健康模型不依赖未使用的历史。
# 输入：
#   无。
# 输出：
#   None：只生成真实健康故障标签，跨模态及异常保持无样本。
def test_no_old_health_or_multimodal_fallback():
    snapshots, records = replay_fixture()
    result = replay_teacher_sensor_diagnostics(snapshots, records,
        roles=("perception-health-critic", "cross-modal-consistency-critic"),
        source_identity="a" * 64)
    assert len(result) == 1
    sample = next(iter(result.values()))
    assert sample.role == "perception-health-critic" and sample.risk_target == 1.0
    assert sample.state_features[7] == 0.0 and sample.history_mask == [0.0] * 8


# 功能：
#   验证当前模态故障成为跨模态监督，不被导航快照里的旧就绪状态覆盖。
# 输入：
#   无。
# 输出：
#   None：两类故障标签及当前状态时钟的单行历史均符合原始记录。
def test_real_diagnostic_status_drives_targets():
    snapshots, records = replay_fixture()
    records[0]["multimodal_sensor_snapshot"] = {
        "captured_at_monotonic_seconds": 1., "contract_set_sha256": "c" * 64,
        "ready_for_motion": False, "active_sensor_ids": ["depth"],
        "statuses": [{"sensor_id": "depth", "modality": "depth-camera", "required_for_motion": True,
                      "latest_sequence": 1, "sample_age_seconds": 1., "health": "unavailable"}],
        "issue_codes": ["DEPTH_STALE"],
    }
    result = replay_teacher_sensor_diagnostics(snapshots, records,
        roles=("cross-modal-consistency-critic", "state-anomaly-detector"),
        source_identity="a" * 64)
    assert len(result) == 2
    assert all(s.risk_target == 1.0 for s in result.values())
    assert all(sum(s.history_mask) == 1 for s in result.values())


# 功能：
#   防止目标切换后复用旧任务上下文；记录时钟倒退也不能靠排序掩盖。
# 输入：
#   无。
# 输出：
#   None：错误目标无样本，时钟倒退明确拒绝。
def test_target_and_clock_boundaries():
    snapshots, records = replay_fixture()
    records[0]["navigation_goal_id"] = "other"
    assert not replay_teacher_sensor_diagnostics(snapshots, records,
        roles=("perception-health-critic",), source_identity="a" * 64)
    records.append({**records[0], "recorded_at_unix_ms": 999})
    with pytest.raises(ValueError, match="CLOCK_INVALID"):
        replay_teacher_sensor_diagnostics(snapshots, records,
            roles=("perception-health-critic",), source_identity="a" * 64)


# 功能：
#   特权真值不能冒充机载输入，导航时间线也不能通过重新排序隐藏回退。
# 输入：
#   fault：真值混入或导航时钟倒退。
# 输出：
#   None：两类来源错误均被显式拒绝。
@pytest.mark.parametrize("fault", ["truth", "clock"])
def test_diagnostic_actor_input_and_navigation_time(fault):
    snapshots, records = replay_fixture()
    if fault == "truth":
        records[0]["observation"]["source"] = "simulation-ground-truth"
    else:
        snapshots.append((999, deepcopy(snapshots[0][1])))
    with pytest.raises(ValueError, match="REQUIRES_ONBOARD|NAVIGATION_CLOCK"):
        replay_teacher_sensor_diagnostics(snapshots, records,
            roles=("perception-health-critic",), source_identity="a" * 64)
