"""Current recorder integration; synthetic fixtures are not flight evidence."""

import copy
import json
import threading

import pytest
from control_fixtures import complete_feature_snapshot
from test_executed_demonstration_dataset import rebind, source_fixture

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_policy_packages import LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH
from dronedream_agent_core.local_policy_port import (
    OnnxLocalPolicyBackend,
    compile_local_policy_features,
)
from dronedream_agent_core.realtime_feature_encoders import fuse_realtime_features
from dronedream_agent_core.temporal_evidence import ObservationHistory
from dronedream_agent_core.training import current_advisor_sources as loader
from dronedream_agent_core.training.advisor_sources import RecordedSource
from scripts.build_local_advisor_dataset import _collect_run_samples, _source_receipt


# 功能：
#   以默认严格来源选项读取测试目录，不开启历史开发证据豁免。
# 输入：
#   root：合成记录所在目录。
# 输出：
#   source：脚本实际使用的五项来源元组。
def source(root):
    result = _source_receipt(root, require_verified=True, allow_verified_fault_recovery=False,
                            allow_verified_payload_collection=False,
                            allow_verified_payload_recovery=False)
    return result


# 功能：
#   验证当前原始观测能贯通辅助专家读取、编译，并且没有伪造模型调用周期。
# 输入：
#   tmp_path：独立合成记录目录。
# 输出：
#   None：断言真实入口返回非空样本和原始来源绑定。
def test_current_source_compiles_without_legacy_cycles(tmp_path):
    root = tmp_path / "run"
    source_fixture(root)
    receipt, observations, cycles, depth, multimodal = source(root)
    assert cycles is depth is multimodal is None
    assert receipt["source_kind"] == "executed-teacher-learning-observations"
    assert receipt["flight_qualification_granted"] is False
    assert observations.content == (root / "learning-observations.jsonl").read_bytes()
    samples = _collect_run_samples(observations, cycles, depth, multimodal,
        roles=("perception-health-critic", "state-anomaly-detector"),
        source_identity=receipt["mission_evidence_sha256"])
    assert {sample.role for sample in samples.values()} == {
        "perception-health-critic", "state-anomaly-detector"}
    assert not (root / "model-navigation-cycles.jsonl").exists()


# 功能：
#   现有文件损坏时拒绝当前来源，不因旧文件也在目录里而静默回退。
# 输入：
#   tmp_path：独立合成记录目录；fault：损坏文件或教师契约的方式。
# 输出：
#   None：实际入口必须拒绝不一致来源。
@pytest.mark.parametrize("fault", ["bytes", "teacher", "summary"])
def test_current_source_refuses_damage_and_old_contract(tmp_path, fault):
    root = tmp_path / "run"
    source_fixture(root)
    (root / "model-navigation-cycles.jsonl").write_text("{}\n")
    if fault == "bytes":
        with (root / "learning-observations.jsonl").open("ab") as stream:
            stream.write(b"{}\n")
    elif fault == "teacher":
        path = root / "mission_evidence.json"
        value = json.loads(path.read_text())
        value["measurements"]["simulation_learning"]["teacher_contract_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
    else:
        (root / "learning-observation-summary.json").write_text(
            json.dumps({"complete": False, "submitted": 1, "completed": 1}))
        rebind(root)
    with pytest.raises(ValueError):
        source(root)


# 功能：
#   验证来源校验完成后被替换的观测不能成为辅助专家实际使用的字节。
# 输入：
#   tmp_path：独立合成记录目录；monkeypatch：确定性插入文件替换的测试设施。
# 输出：
#   None：读取器必须报告核验后内容变化。
def test_post_validation_replacement_rejected(tmp_path, monkeypatch):
    root = tmp_path / "run"
    source_fixture(root)
    original = loader.collect_demonstrations

    # 功能：
    #   在真实验证完成后模拟来源文件被替换。
    # 输入：
    #   args、kwargs：原始验证调用参数。
    # 输出：
    #   corpus：原始验证结果，不篡改其摘要。
    def replace_after_validation(*args, **kwargs):
        corpus = original(*args, **kwargs)
        (root / "learning-observations.jsonl").write_text("{}\n")
        return corpus

    monkeypatch.setattr(loader, "collect_demonstrations", replace_after_validation)
    with pytest.raises(ValueError, match="CHANGED_AFTER_VALIDATION"):
        source(root)


# 功能：
#   当前来源不可借用旧开发故障豁免跳过验证。
# 输入：
#   tmp_path：独立合成记录目录。
# 输出：
#   None：入口拒绝混用来源模式。
def test_current_source_rejects_legacy_overrides(tmp_path):
    root = tmp_path / "run"
    source_fixture(root)
    with pytest.raises(ValueError, match="LEGACY_RECOVERY_OVERRIDE"):
        _source_receipt(root, require_verified=True, allow_verified_fault_recovery=True,
                        allow_verified_payload_collection=False,
                        allow_verified_payload_recovery=False)


# 功能：
#   构造传感器时间、日志时间和来源身份独立可控的第二条合成观测。
# 输入：
#   record：第一条合成观测；gap：传感器时间增量；identity：第二份来源摘要。
#   reset：是否显式重置；session：可选的新会话身份。
# 输出：
#   later：只用于单元测试的第二条记录，不作为真实运行证据。
def changed_observation(record, gap, identity="b" * 64, reset=False, session=None):
    later = copy.deepcopy(record)
    later["recorded_at_unix_ms"] += 10_000
    snapshot = later["snapshot"]
    snapshot["current_position_m"]["x"] += .001
    encodings = complete_feature_snapshot(1000 + gap).encodings
    encodings[-1] = encodings[-1].model_copy(update={
        "source_sha256": identity,
        "issue_codes": ["FLIGHT_STATE_IMU_CLOCK_RESET"] if reset else [],
    })
    snapshot["realtime_feature_snapshot"] = fuse_realtime_features(
        encodings, captured_at_unix_ms=1000 + gap).model_dump(mode="json")
    snapshot["control_reference_observed_at_unix_ms"] = 1000 + gap
    if session is not None:
        snapshot["strategic_context"]["task"]["control_session_id"] = session
    snapshot.pop("snapshot_sha256")
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    return later


# 功能：
#   使用生产后端的实际历史入口核对训练窗口，覆盖日志延迟、断流、重复来源及会话重启。
# 输入：
#   tmp_path：合成目录；gap、identity、reset、session：来源变化；count：预期有效行数。
# 输出：
#   None：训练窗口与在线后端必须逐值相同。
@pytest.mark.parametrize("gap,identity,reset,session,count", [
    (100, "b" * 64, False, None, 2),
    (250, "b" * 64, False, None, 2),
    (251, "b" * 64, False, None, 1),
    (0, "a" * 64, False, None, 1),
    (100, "b" * 64, False, "new-session", 1),
    (100, "b" * 64, True, None, 1),
    (-1, "b" * 64, True, None, 1),
])
def test_current_history_matches_runtime(tmp_path, gap, identity, reset, session, count):
    root = tmp_path / "run"
    record = source_fixture(root)
    later = changed_observation(record, gap, identity, reset, session)
    raw = RecordedSource(root / "synthetic-history.jsonl",
        (json.dumps(record) + "\n" + json.dumps(later) + "\n").encode())
    samples = _collect_run_samples(raw, None, None, None,
        roles=("state-anomaly-detector",), source_identity="a" * 64)
    assert len(samples) == 2
    final = list(samples.values())[-1]
    backend = object.__new__(OnnxLocalPolicyBackend)
    backend._history_lock = threading.Lock()
    backend._control_history = None
    backend._observation_history = ObservationHistory(LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH)
    for row in (record, later):
        backend.prepare_temporal_context(compile_local_policy_features(row["snapshot"]))
    rows = [list(row[0]) for row in backend._observation_history.rows]
    assert len(rows) == sum(final.history_mask) == count
    assert final.state_history[-count:] == rows


# 功能：
#   拒绝同刻冲突、摘要续期和未授权时钟回退，不通过排序掩盖损坏时序。
# 输入：
#   tmp_path：合成目录；gap、identity：冲突来源；error：生产历史应报告的错误。
# 输出：
#   None：构建器保持与在线来源校验一致的拒绝行为。
@pytest.mark.parametrize("gap,identity,error", [
    (0, "b" * 64, "SAME_TIME_CONFLICT"),
    (100, "a" * 64, "SAMPLE_REDATED"),
    (-1, "b" * 64, "CLOCK_REGRESSED"),
])
def test_current_history_rejects_source_conflicts(tmp_path, gap, identity, error):
    root = tmp_path / "run"
    record = source_fixture(root)
    later = changed_observation(record, gap, identity)
    raw = RecordedSource(root / "synthetic-history.jsonl",
        (json.dumps(record) + "\n" + json.dumps(later) + "\n").encode())
    with pytest.raises(ValueError, match=error):
        _collect_run_samples(raw, None, None, None,
            roles=("state-anomaly-detector",), source_identity="a" * 64)
