"""Only synthetic evidence is used here; no flight qualification is granted."""

import hashlib
import json

import pytest
from test_executed_demonstration_dataset import source_fixture

from dronedream_agent_core.training.demonstrations import collect_demonstrations
from dronedream_agent_core.training.teacher_fault_sources import (
    read_teacher_fault_advisor_source,
    verify_teacher_fault_window,
)
from scripts.build_local_advisor_dataset import _collect_sources


# 功能：
#   构造有明确注入、停车及恢复的合成教师证据，所有文件摘要独立绑定。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   root：合成故障数据目录。
@pytest.fixture
def fault_source(tmp_path):
    root = tmp_path / "run"
    source_fixture(root)
    fault = {"schema_version": "dronedream.development-depth-fault.v1",
             "development_only": True, "fault_kind": "depth-frame-drop",
             "activated": True, "recovered": True, "activated_at_unix_ms": 2000,
             "recovered_at_unix_ms": 5000, "drop_duration_seconds": 3.0}
    rows = [
        {"recorded_at_unix_ms": 1000, "status": "accepted", "action": "continue"},
        {"recorded_at_unix_ms": 2100, "status": "command-unavailable"},
        {"recorded_at_unix_ms": 6000, "status": "accepted", "action": "continue"},
    ]
    paths = {"depth-local-safety-history.jsonl": ("depth_safety_history_sha256", _depth_evidence()),
             "development-depth-fault.json": ("development_fault_injection_sha256", [fault]),
             "runtime-state/local-safety-executor-history.jsonl": (
                 "local_safety_executor_history_sha256", rows)}
    evidence_path = root / "mission_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["status"] = "failed"
    evidence["gates"].update(development_fault_injection_absent=False,
                             landing_confirmed=True, executor_completed=True)
    for name, (key, values) in paths.items():
        path = root / name
        path.write_text("".join(json.dumps(v) + "\n" for v in values), encoding="utf-8")
        evidence["artifacts"][key] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return root


# 功能：
#   构造摘要绑定的故障前运动和故障后制动，明确保留观测原时间。
# 输入：
#   无。
# 输出：
#   rows：仅用于时序边界单元测试的两条合成决策。
def _depth_evidence():
    from test_runtime_local_safety import _vehicle

    from dronedream_agent_core.contracts import RuntimeLocalSafetyObservation, Vector3
    from dronedream_agent_core.runtime_local_safety import evaluate_runtime_local_safety

    observation = RuntimeLocalSafetyObservation(sequence=1, observed_at_unix_ms=1950,
        source="onboard", stream_healthy=True, stream_age_seconds=.02,
        localization_covariance_m2=.01, current_position_m=Vector3(x=0, y=0, z=2),
        current_velocity_mps=Vector3(x=0, y=0, z=0),
        target_position_m=Vector3(x=1, y=0, z=2))
    rows = []
    for stamp, healthy in ((1980, True), (2010, False)):
        current = observation.model_copy(update={"stream_healthy": healthy})
        command = evaluate_runtime_local_safety(observation=current, vehicle=_vehicle(),
            static_primitives=[], required_clearance_m=.35, generated_at_unix_ms=stamp)
        rows.append({"recorded_at_unix_ms": stamp - 1,
                     "observation": current.model_dump(mode="json"),
                     "command": command.model_dump(mode="json")})
    return rows


# 功能：
#   区分已证实的在途旧命令与故障后运动，拒绝缺身份、伪续期和制动后继续运行。
# 输入：
#   fault_source：合成故障来源；damage：待破坏的身份或时序边界。
# 输出：
#   None：仅完整证实的短暂在途命令可进入感知辅助来源。
@pytest.mark.parametrize("damage", [None, "missing_hash", "new_command", "expiry",
    "brake_published", "wrong_observation", "wrong_sequence", "no_brake", "source_binding"])
def test_fault_inflight_command_requires_exact_bounded_identity(fault_source, damage):
    from dronedream_agent_core.hashing import sha256_json

    fault = json.loads((fault_source / "development-depth-fault.json").read_text())
    depth = _depth_evidence()
    command = depth[0]["command"]
    row = {"recorded_at_unix_ms": 2005, "status": "accepted", "action": "continue",
           "command_sha256": sha256_json(command),
           "command_observation_sha256": command["observation_sha256"],
           "command_generated_at_unix_ms": command["generated_at_unix_ms"],
           "command_valid_until_unix_ms": command["valid_until_unix_ms"],
           "command_sequence": command["observation_sequence"]}
    if damage == "missing_hash":
        row.pop("command_sha256")
    elif damage == "new_command":
        row["command_generated_at_unix_ms"] = 2001
    elif damage == "expiry":
        row["command_valid_until_unix_ms"] += 1
    elif damage == "brake_published":
        row["recorded_at_unix_ms"] = 2010
    elif damage == "wrong_observation":
        row["command_observation_sha256"] = "0" * 64
    elif damage == "wrong_sequence":
        row["command_sequence"] += 1
    elif damage == "no_brake":
        depth.pop()
    elif damage == "source_binding":
        depth[0]["observation"]["sequence"] += 1
    rows = [{"recorded_at_unix_ms": 1000, "status": "accepted", "action": "continue"},
            row, {"recorded_at_unix_ms": 2300, "status": "command-unavailable"},
            {"recorded_at_unix_ms": 6000, "status": "accepted", "action": "continue"}]
    if damage:
        with pytest.raises(ValueError, match="TEACHER_ADVISOR_"):
            verify_teacher_fault_window(fault, rows, depth)
    else:
        counts = verify_teacher_fault_window(fault, rows, depth)
        assert counts == {"before": 1, "held_or_unavailable": 1, "after": 1,
                          "inflight_pre_fault_commands": 1}
        with pytest.raises(ValueError, match="MOTION_DURING_FAULT"):
            verify_teacher_fault_window(fault, rows)


# 功能：
#   确认感知故障观察可被读取，但同一数据仍不能充当成功动作示范或稳定性标签。
# 输入：
#   fault_source：合成故障记录目录。
# 输出：
#   None：用途隔离、原失败状态和行为入口拒绝通过断言。
def test_fault_observation_scope(fault_source):
    receipt, observations, depth = read_teacher_fault_advisor_source(fault_source)
    assert receipt["mission_status"] == "failed"
    assert receipt["behavior_action_labels_granted"] is False
    assert receipt["flight_qualification_granted"] is False
    assert observations.content and depth.content
    assert receipt["fault_window_counts"] == {"before": 1, "held_or_unavailable": 1, "after": 1}
    with pytest.raises(ValueError, match="PHYSICAL_RUN_NOT_VERIFIED"):
        collect_demonstrations([fault_source], require_visual=False)
    with pytest.raises(ValueError, match="SOURCE_ROLE_NOT_ALLOWED"):
        _collect_sources([fault_source], roles=("settle-stability-critic",),
            require_verified=True, allow_verified_fault_recovery=True,
            allow_verified_payload_collection=False, allow_verified_payload_recovery=False)


# 功能：
#   即使攻击者重绑定摘要，也拒绝真实语义错误；普通文件被修改也必须被摘要拦截。
# 输入：
#   fault_source：合成来源；damage：破坏的故障、安全或来源边界。
# 输出：
#   None：每种破坏均不能得到有效辅助来源。
@pytest.mark.parametrize("damage", ["hash", "unsafe", "unrecovered", "no_after",
                                    "clock", "other_gate", "old_teacher", "not_landed"])
def test_fault_source_rejects_damage(fault_source, damage):
    root = fault_source
    path = root / "mission_evidence.json"
    evidence = json.loads(path.read_text())
    if damage in ("hash", "unsafe", "no_after", "clock"):
        target = root / "runtime-state/local-safety-executor-history.jsonl"
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        if damage in ("hash", "unsafe"):
            rows[1].update(status="accepted", action="continue")
        elif damage == "no_after":
            rows.pop()
        else:
            rows.reverse()
        target.write_text("".join(json.dumps(v) + "\n" for v in rows))
        if damage != "hash":
            evidence["artifacts"]["local_safety_executor_history_sha256"] = hashlib.sha256(
                target.read_bytes()).hexdigest()
    elif damage == "unrecovered":
        target = root / "development-depth-fault.json"
        fault = json.loads(target.read_text())
        fault["recovered"] = False
        target.write_text(json.dumps(fault))
        evidence["artifacts"]["development_fault_injection_sha256"] = hashlib.sha256(
            target.read_bytes()).hexdigest()
    elif damage == "other_gate":
        evidence["gates"]["collision_free"] = False
    elif damage == "not_landed":
        evidence["gates"]["landing_confirmed"] = False
    else:
        evidence["measurements"]["simulation_learning"]["teacher_contract_sha256"] = "0" * 64
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="TEACHER_ADVISOR_"):
        read_teacher_fault_advisor_source(root)
