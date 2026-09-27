"""Same-slot source updates across physical encoding, inference and training."""

import copy
import json
import threading
from dataclasses import replace

import pytest
from control_fixtures import complete_feature_snapshot
from test_causal_policy import samples
from test_current_advisor_sources import changed_observation
from test_executed_demonstration_dataset import rebind, source_fixture
from test_learning_observation_recorder import request_fixture
from test_native_state_stream import packet
from test_temporal_evidence import evidence

from dronedream_agent_core.causal_control import CausalControlHistory
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.learning_observation_recorder import LearningObservationRecorder
from dronedream_agent_core.local_policy_port import (
    OnnxLocalPolicyBackend,
    compile_local_policy_features,
)
from dronedream_agent_core.native_state_stream import NativeStateBuffer
from dronedream_agent_core.realtime_feature_encoders import fuse_realtime_features
from dronedream_agent_core.temporal_evidence import ObservationHistory, TemporalEvidence
from dronedream_agent_core.training.causal_policy import causal_examples
from dronedream_agent_core.training.demonstrations import collect_demonstrations


# 功能：
#   检查修订只更新末槽；重放幂等、修订倒退和未经标识的冲突仍被拒绝。
# 输入：
#   无。
# 输出：
#   None：独立样本数量和原采样时刻均保持不变。
def test_revision_replaces_one_slot_without_renewing_history():
    history = ObservationHistory(2)
    original = evidence(1)
    history.append(original, (1.,), ())
    revised = evidence(2, timestamp=1050).model_copy(update={"history_slot_revision": 1})
    assert history.append(revised, (2.,), ()) is False
    assert history.latest == revised and len(history.rows) == 1 and not history.ready
    assert history.append(revised, (99.,), ()) is False
    assert history.rows[-1][0] == (2.,)
    for bad in (original, evidence(3, timestamp=1050),
                revised.model_copy(update={"reset_history": True, "sample_sha256": "f" * 64})):
        with pytest.raises(ValueError, match="SAME_TIME_CONFLICT"):
            history.append(bad, (3.,), ())
    with pytest.raises(ValueError, match="REDATED"):
        history.append(revised.model_copy(update={"observed_at_unix_ms": 1060}), (3.,), ())
    assert history.latest == revised


# 功能：
#   拒绝非法修订类型；不增加默认字段而改变已有来源及传感器快照的序列化身份。
# 输入：
#   value：非法修订号。
# 输出：
#   None：非法输入被拒绝，原有零修订序列化字典保持兼容。
@pytest.mark.parametrize("value", [True, "1", -1, 1.5, 2**31])
def test_revision_validation_and_legacy_identity(value):
    original = evidence(0)
    assert "history_slot_revision" not in original.model_dump(mode="json")
    with pytest.raises(ValueError):
        TemporalEvidence.model_validate({**original.model_dump(), "history_slot_revision": value})
    snapshot = complete_feature_snapshot(1000)
    assert all("history_slot_revision" not in row for row in snapshot.model_dump()["encodings"])
    restored = type(snapshot).model_validate(snapshot.model_dump())
    assert restored.snapshot_sha256 == snapshot.snapshot_sha256


# 功能：
#   将真实编码入口产生的异步姿态修订送入生产模型后端，不替换成合成时间步。
# 输入：
#   无。
# 输出：
#   None：历史保持一行、姿态同步、来源时刻仍为原始时刻。
def test_native_partial_update_reaches_online_history():
    native = NativeStateBuffer()
    for stamp in range(1000, 1320, 20):
        native.ingest(packet(stamp), now_unix_ms=stamp)
    first = native.latest(now_unix_ms=1300).encoding
    update = packet(1320)
    update["dynamics"]["sources"]["imu"] = packet(1300)["dynamics"]["sources"]["imu"]
    update["dynamics"]["sources"]["attitude"]["yaw_deg"] = 110.
    native.ingest(update, now_unix_ms=1320)
    latest = native.latest(now_unix_ms=1320).encoding
    backend = object.__new__(OnnxLocalPolicyBackend)
    backend._history_lock = threading.Lock()
    backend._control_history = CausalControlHistory(4)
    backend._observation_history = ObservationHistory(8)
    request, _ = request_fixture()
    from dronedream_agent_core.navigation_snapshot import compile_navigation_snapshot
    snapshot = compile_navigation_snapshot(request)
    for encoding in (first, latest):
        encodings = complete_feature_snapshot(1320).encodings
        encodings[-1] = encoding
        snapshot["realtime_feature_snapshot"] = fuse_realtime_features(
            encodings, captured_at_unix_ms=1320).model_dump(mode="json")
        snapshot["control_reference_observed_at_unix_ms"] = 1320
        batch = compile_local_policy_features(snapshot)
        backend.prepare_temporal_context(batch)
    assert batch.temporal_evidence.history_slot_revision == 1
    assert backend._observation_history.latest.observed_at_unix_ms == 1300
    assert len(backend._observation_history.rows) == 1
    assert len(backend._control_history.history.rows) == 1
    assert backend._control_history.history.latest == batch.temporal_evidence


# 功能：
#   修订当前控制槽时保留真实时间间隔及来源列表，并与在线控制窗口逐值比较。
# 输入：
#   无。
# 输出：
#   None：训练窗口与在线窗口一致，先前输出不被后来修订反向覆盖。
def test_causal_training_revision_matches_runtime_and_preserves_elapsed():
    rows = samples()[:5]
    previous = rows[3]
    revised = previous.model_copy(update={
        "temporal_evidence": evidence(90, timestamp=previous.temporal_evidence.observed_at_unix_ms)
            .model_copy(update={"stream_id": "train", "history_slot_revision": 1}),
        "state_features": [.123] * 46,
    })
    observations = [*rows[:4], revised, rows[4]]
    examples = causal_examples(observations, stream_groups={"train": "route-a"}, history_length=4)
    online = CausalControlHistory(4)
    for row in observations:
        online.append(row.temporal_evidence, row.state_features, row.realtime_features,
                      row.realtime_valid_mask)
    assert len(examples) == 3
    assert examples[1].history[-1][-1] == examples[0].history[-1][-1] == .2
    assert examples[0].source_sha256[-1] == previous.temporal_evidence.sample_sha256
    assert examples[1].source_sha256[-1] == revised.temporal_evidence.sample_sha256
    assert examples[-1].history == online.values()[0]
    assert len(examples[-1].source_sha256) == 4


# 功能：
#   记录器保存显式修订但不保存重放，完整保留模型输入供相同历史算法重放。
# 输入：
#   无。
# 输出：
#   None：两份记录具有相同时刻与不同修订身份，没有任何伪造动作标签。
def test_recorder_retains_revision_for_offline_replay():
    request, _ = request_fixture()
    records = []
    recorder = LearningObservationRecorder(lambda row: records.append(row) is None,
                                          synchronous=True)
    try:
        assert recorder.submit(request, None)
        features = copy.deepcopy(request.realtime_feature_snapshot)
        from dronedream_agent_core.realtime_feature_encoders import RealtimeFeatureSnapshot
        encodings = RealtimeFeatureSnapshot.model_validate(features).encodings
        encodings[-1] = encodings[-1].model_copy(update={
            "source_sha256": "b" * 64, "history_slot_revision": 1})
        changed = replace(request, realtime_feature_snapshot=fuse_realtime_features(
            encodings, captured_at_unix_ms=1000).model_dump(mode="json"))
        assert recorder.submit(changed, None)
        assert recorder.submit(changed, None) is False
    finally:
        assert recorder.close()["complete"]
    assert len(records) == 2
    batch = compile_local_policy_features(records[-1]["snapshot"])
    assert batch.temporal_evidence.history_slot_revision == 1


# 功能：
#   完整证据读取允许同槽显式修订，仅保留观测而不为没有执行回执的记录制造动作。
# 输入：
#   tmp_path：合成证据目录。
# 输出：
#   None：收集到两个来源版本，原行为标签数量不增加。
def test_demonstration_collector_accepts_bound_revision(tmp_path):
    root = tmp_path / "run"
    first = source_fixture(root)
    revised = changed_observation(first, 0)
    revised["recorded_at_unix_ms"] = 1000
    revised["evaluated_command_sha256"] = None
    revised["control_evaluation_status"] = "no-command"
    snapshot = revised["snapshot"]
    encodings = type(complete_feature_snapshot(1000)).model_validate(
        snapshot["realtime_feature_snapshot"]).encodings
    encodings[-1] = encodings[-1].model_copy(update={"history_slot_revision": 1})
    snapshot["realtime_feature_snapshot"] = fuse_realtime_features(
        encodings, captured_at_unix_ms=1000).model_dump(mode="json")
    snapshot.pop("snapshot_sha256")
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    (root / "learning-observations.jsonl").write_text(
        json.dumps(first) + "\n" + json.dumps(revised) + "\n")
    (root / "learning-observation-summary.json").write_text(
        json.dumps({"complete": True, "submitted": 2, "completed": 2}))
    witness = {"recorded_at_unix_ms": 1000, "command": None, "identity_accepted": True,
               "navigation_goal_id": snapshot["strategic_context"]["task"]["navigation_goal_id"],
               "realtime_feature_snapshot": snapshot["realtime_feature_snapshot"]}
    path = root / "depth-local-safety-history.jsonl"
    path.write_text(path.read_text() + json.dumps(witness) + "\n")
    rebind(root)
    result = collect_demonstrations([root], require_visual=False, allow_nonvisual_history=True)
    assert len(result.observations) == 2
    assert len(result.samples) == 1
