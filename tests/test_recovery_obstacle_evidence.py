import json

import pytest
from test_runtime_local_safety import _vehicle

from dronedream_agent_core.contracts import (
    DynamicObstacleObservation,
    RuntimeLocalSafetyObservation,
    Vector3,
)
from dronedream_agent_core.recovery_obstacle_evidence import (
    RecoveryObstacleWitness,
    recovery_clearance_metrics,
    recovery_obstacle_metrics,
)
from dronedream_agent_core.runtime_local_safety import evaluate_runtime_local_safety


# 功能：
#   创建名称不同但几何一致的独立真值/原生感知夹具，不冒充真实采样。
# 输入：
#   source：观测来源。
#   stamp：原始采集毫秒时间。
# 输出：
#   observation：类型化合成观测。
def observation(source, stamp=10000):
    obstacle = DynamicObstacleObservation(obstacle_id=("dronedream_dynamic_box" if source == "simulation-ground-truth" else "depth-track-1"),
        position_m=Vector3(x=1., y=0., z=1.5), velocity_mps=Vector3(x=0., y=0., z=0.),
        radius_m=.4, height_m=1., confidence=1., age_seconds=0.)
    result = RuntimeLocalSafetyObservation(sequence=1, observed_at_unix_ms=stamp, source=source,
        stream_healthy=True, stream_age_seconds=.01, localization_covariance_m2=.01,
        current_position_m=Vector3(x=0., y=0., z=1.5), current_velocity_mps=Vector3(x=.5, y=0., z=0.),
        target_position_m=Vector3(x=5., y=0., z=1.5), dynamic_obstacles=[obstacle])
    return result


# 功能：
#   为单元测试保存完整独立见证与摘要匹配的原生命令，可显式注入拒绝条件。
# 输入：
#   root：隔离目录。
#   fault：需要注入的异常名称。
# 输出：
#   None：无业务数据。
def evidence(root, fault="none"):
    witness = observation("simulation-ground-truth", 9500 if fault == "old" else 10000)
    if fault == "ambiguous":
        witness.dynamic_obstacles.append(witness.dynamic_obstacles[0].model_copy(update={"obstacle_id": "dronedream_dynamic_other"}))
    recorder = RecoveryObstacleWitness(root, lambda p, r: p.write_text(json.dumps(r)))
    assert recorder.record(witness)
    assert recorder.close()["complete"]
    native = observation("onboard")
    if fault == "wrong-height":
        native.dynamic_obstacles[0].position_m.z = 8.
    if fault == "unhealthy":
        native.stream_healthy = False
    if fault == "truth-input":
        native.source = "simulation-ground-truth"
    if fault == "stale-track":
        native.dynamic_obstacles[0].age_seconds = 1.
    if fault == "split-clusters":
        native.dynamic_obstacles.insert(0, native.dynamic_obstacles[0].model_copy(update={"obstacle_id": "not-the-threat"}))
    command = evaluate_runtime_local_safety(observation=native, vehicle=_vehicle(), static_primitives=[],
        required_clearance_m=.35, generated_at_unix_ms=10010, validity_milliseconds=150)
    if fault == "hash":
        command = command.model_copy(update={"observation_sha256": "a" * 64})
    if fault == "long-lease":
        command = command.model_copy(update={"valid_until_unix_ms": 10500})
    if fault == "split-clusters":
        command.decision.threat_obstacle_id = "depth-track-1"
        command.decision.avoidance_obstacle_id = "depth-track-1"
    if fault == "nearest-only":
        command.decision.avoidance_obstacle_id = None
    row = {"observation": native.model_dump(mode="json"), "command": command.model_dump(mode="json")}
    (root / "depth-local-safety-history.jsonl").write_text(json.dumps(row) + "\n")


# 功能：
#   核对当前原生轨迹可关联真值实体，且不需要旧真值控制历史。
# 输入：
#   tmp_path：隔离目录。
# 输出：
#   None：无返回数据。
def test_current_native_threat_matches_independent_witness(tmp_path):
    evidence(tmp_path)
    result = recovery_obstacle_metrics(tmp_path, ["dronedream_dynamic_box"])
    assert result["dronedream_dynamic_box"] == {"observation_count": 1, "threat_count": 1, "moving_observation_count": 0,
        "minimum_horizontal_distance_m": 1.}
    assert not (tmp_path / "local-safety-history.jsonl").exists()


# 功能：
#   过期、几何不符、关联歧义、非原生或不健康观测不能证明挑战被模型感知。
# 输入：
#   tmp_path：隔离目录。
#   fault：拒绝条件。
# 输出：
#   None：无返回数据。
@pytest.mark.parametrize("fault", ["old", "ambiguous", "wrong-height", "unhealthy", "truth-input", "stale-track"])
def test_invalid_associations_do_not_count_as_perception(tmp_path, fault):
    evidence(tmp_path, fault)
    result = recovery_obstacle_metrics(tmp_path, ["dronedream_dynamic_box"])
    assert result["dronedream_dynamic_box"]["observation_count"] == 0


# 功能：
#   感知到障碍不等于控制器把它判为威胁，摘要错配或租约过长均不计威胁。
# 输入：
#   tmp_path：隔离目录。
#   fault：命令关联故障。
# 输出：
#   None：无返回数据。
@pytest.mark.parametrize("fault", ["hash", "long-lease", "nearest-only"])
def test_unbound_or_expired_commands_do_not_count_as_threats(tmp_path, fault):
    evidence(tmp_path, fault)
    result = recovery_obstacle_metrics(tmp_path, ["dronedream_dynamic_box"])
    assert result["dronedream_dynamic_box"]["observation_count"] == 1
    assert result["dronedream_dynamic_box"]["threat_count"] == 0


# 功能：
#   记录器保留失败状态，接收原生控制数据不能冒充独立见证。
# 输入：
#   tmp_path：隔离目录。
# 输出：
#   None：无返回数据。
def test_recorder_latches_wrong_source_and_rejects_overwrite(tmp_path):
    recorder = RecoveryObstacleWitness(tmp_path, lambda p, r: p.write_text(json.dumps(r)))
    assert not recorder.record(observation("onboard"))
    assert not recorder.record(observation("simulation-ground-truth"))
    assert not recorder.close()["complete"]
    with pytest.raises(FileExistsError):
        RecoveryObstacleWitness(tmp_path, lambda p, r: None)


# 功能：
#   真值文件截断或回执数量不符必须中断验收，而非按有效前缀宣告成功。
# 输入：
#   tmp_path：隔离目录。
#   fault：证据完整性故障。
# 输出：
#   None：无返回数据。
@pytest.mark.parametrize("fault", ["truncated", "count"])
def test_incomplete_witness_cannot_pass(tmp_path, fault):
    evidence(tmp_path)
    if fault == "truncated":
        path = tmp_path / "recovery-obstacle-witness.jsonl"
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
    else:
        path = tmp_path / "recovery-obstacle-witness-summary.json"
        payload = json.loads(path.read_text())
        payload["witness_count"] += 1
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="RECOVERY_"):
        recovery_obstacle_metrics(tmp_path, ["dronedream_dynamic_box"])


# 功能：
#   同一实体的多个原生簇与重复历史只计一次，且不漏掉稍后出现的威胁簇。
# 输入：
#   tmp_path：隔离目录。
# 输出：
#   None：无业务数据。
def test_split_cluster_threat_is_not_swallowed_or_double_counted(tmp_path):
    evidence(tmp_path, "split-clusters")
    path = tmp_path / "depth-local-safety-history.jsonl"
    path.write_bytes(path.read_bytes() * 2)
    result = recovery_obstacle_metrics(tmp_path, ["dronedream_dynamic_box"])
    assert result["dronedream_dynamic_box"]["threat_count"] == 1
    assert result["dronedream_dynamic_box"]["observation_count"] == 1


# 功能：
#   布尔值不得冒充已完成数量，非零待处理或失败回执不得绕过关闭检查。
# 输入：
#   tmp_path：隔离目录。
#   field、value：回执故障。
# 输出：
#   None：无业务数据。
@pytest.mark.parametrize("field,value", [("witness_count", True), ("completed_count", True),
    ("submitted_count", 2), ("pending_count", 1), ("rejected_count", 1),
    ("thread_alive", True), ("witness_issue", "failed")])
def test_receipt_integrity_is_fail_closed(tmp_path, field, value):
    evidence(tmp_path)
    path = tmp_path / "recovery-obstacle-witness-summary.json"
    summary = json.loads(path.read_text())
    summary[field] = value
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="RECOVERY_"):
        recovery_obstacle_metrics(tmp_path, ["dronedream_dynamic_box"])


# 功能：
#   已关闭见证记录器不接纳新数据，不覆盖其最终回执；采样时钟回退锁定失败。
# 输入：
#   tmp_path：隔离目录。
# 输出：
#   None：无业务数据。
def test_recorder_close_and_clock_boundaries(tmp_path):
    recorder = RecoveryObstacleWitness(tmp_path, lambda p, r: p.write_text(json.dumps(r)))
    assert recorder.record(observation("simulation-ground-truth"))
    assert not recorder.record(observation("simulation-ground-truth", 9800))
    summary = recorder.close()
    assert not summary["complete"]
    assert not recorder.record(observation("simulation-ground-truth", 10400))
    assert recorder.close() == summary


# 功能：
#   缺失、损坏及意外中断不能留下基础飞行通过状态，完整挑战也不能抹去基础失败。
# 输入：
#   tmp_path、monkeypatch：隔离证据与异常注入夹具。
#   mode：挑战证据或基础飞行状态。
# 输出：
#   None：无业务数据。
@pytest.mark.parametrize("mode", ["valid", "missing", "truncated", "base-failed", "interrupted", "motion-missing"])
def test_challenge_finalization_never_preserves_stale_success(tmp_path, monkeypatch, mode):
    from scripts import run_school_map_depth_qualification as runner
    if mode != "missing":
        evidence(tmp_path)
    if mode == "truncated":
        (tmp_path / "recovery-obstacle-witness.jsonl").write_text("{")
    base = {"status": "verified", "gates": {"physical_run": mode != "base-failed"},
        "measurements": {"local_safety": {"required_clearance_m": .15}}, "artifacts": {}}
    challenge = {"entity_names": ["dronedream_dynamic_box"], "receipt_sha256": "a" * 64,
        "required_encounter_distance_m": 2.}
    if mode == "motion-missing":
        challenge["moving_entity_names"] = ["dronedream_dynamic_box"]
    if mode == "interrupted":
        # 在异常传播前必须已经记录未通过，防止外部调用误读基础证据。
        def interrupt(*args):
            raise RuntimeError("synthetic interruption")
        monkeypatch.setattr(runner, "recovery_obstacle_metrics", interrupt)
        with pytest.raises(RuntimeError):
            runner._finalize_recovery_challenge(tmp_path, base, challenge, _vehicle())
    else:
        result = runner._finalize_recovery_challenge(tmp_path, base, challenge, _vehicle())
        assert (result["status"] == "verified") == (mode == "valid")
    saved = json.loads((tmp_path / "mission_evidence.json").read_text())
    assert (saved["status"] == "verified") == (mode == "valid")
    assert base["status"] == "verified" and len(base["gates"]) == 1


# 功能：
#   核对包围尺寸、竖直分离及穿透符号，中心距离足够不能掩盖机体净空不足。
# 输入：
#   tmp_path：隔离证据目录。
#   separation、height、expected：水平中心距离、竖直中心距离与期望包围体净空。
# 输出：
#   None：无业务返回值。
@pytest.mark.parametrize("separation,height,expected", [(1., 0., .22), (.8, 0., .02),
    (.5, 0., -.28), (.5, 1., .285), (.98, 1.015, (0.2 ** 2 + 0.3 ** 2) ** .5)])
def test_sampled_clearance_uses_complete_body_envelopes(tmp_path, separation, height, expected):
    evidence(tmp_path)
    path = tmp_path / "recovery-obstacle-witness.jsonl"
    row = json.loads(path.read_text())
    row["dynamic_obstacles"][0]["position_m"] = {"x": separation, "y": 0., "z": 1.5 + height}
    path.write_text(json.dumps(row) + "\n")
    result = recovery_clearance_metrics(tmp_path, ["dronedream_dynamic_box"], _vehicle(), .15)
    assert result["entities"]["dronedream_dynamic_box"]["minimum_clearance_m"] == pytest.approx(expected)
    assert result["sampled_clearance_respected"] == (expected >= .15)
    assert result["continuous_clearance_verified"] is False


# 功能：
#   禁止缺少实体、不可用见证或非法实时余量产生安全通过结果，且陈旧近障不能被过滤。
# 输入：
#   tmp_path：隔离证据目录。
#   fault：独立见证的故障模式。
# 输出：
#   None：无业务返回值。
@pytest.mark.parametrize("fault", ["missing", "unhealthy", "stale", "uncertain", "stale-close"])
def test_sampled_clearance_cannot_hide_missing_or_close_witnesses(tmp_path, fault):
    evidence(tmp_path)
    path = tmp_path / "recovery-obstacle-witness.jsonl"
    row = json.loads(path.read_text())
    if fault == "missing":
        row["dynamic_obstacles"] = []
    elif fault == "unhealthy":
        row["stream_healthy"] = False
    elif fault == "uncertain":
        row["dynamic_obstacles"][0]["confidence"] = .5
    else:
        row["stream_age_seconds"] = .2
    rows = [row]
    if fault == "stale-close":
        row["dynamic_obstacles"][0]["position_m"]["x"] = .5
        rows.append(observation("simulation-ground-truth", 10200).model_dump(mode="json"))
        summary_path = tmp_path / "recovery-obstacle-witness-summary.json"
        summary = json.loads(summary_path.read_text())
        for key in ("witness_count", "submitted_count", "completed_count"):
            summary[key] = 2
        summary_path.write_text(json.dumps(summary))
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    result = recovery_clearance_metrics(tmp_path, ["dronedream_dynamic_box"], _vehicle(), .15)
    assert result["sampled_clearance_respected"] is False
    if fault == "stale-close":
        assert result["maximum_witness_gap_ms"] == 200
        assert result["entities"]["dronedream_dynamic_box"]["usable_sample_count"] == 1


# 功能：
#   拒绝用字符串、布尔值、非有限值或非正数替换当前实时安全余量。
# 输入：
#   tmp_path：隔离目录。
#   margin：非法余量。
# 输出：
#   None：无业务返回值。
@pytest.mark.parametrize("margin", [True, "0.15", None, 0., -1., float("nan"), float("inf")])
def test_invalid_dynamic_clearance_contract_is_rejected(tmp_path, margin):
    with pytest.raises(ValueError, match="RECOVERY_CLEARANCE_CONTRACT_INVALID"):
        recovery_clearance_metrics(tmp_path, ["dronedream_dynamic_box"], _vehicle(), margin)


# 功能：
#   感知关联和威胁事件全部通过时，独立净空不足仍阻止最终验收；缺失实时余量也拒绝。
# 输入：
#   tmp_path：隔离证据目录。
#   margin：有效但不满足的余量，或缺失余量。
# 输出：
#   None：无业务返回值。
@pytest.mark.parametrize("margin", [.3, None])
def test_clearance_failure_overrides_encounter_success(tmp_path, margin):
    from scripts import run_school_map_depth_qualification as runner
    evidence(tmp_path)
    base = {"status": "verified", "gates": {"physical_run": True},
        "measurements": {"local_safety": {"required_clearance_m": margin}}, "artifacts": {}}
    challenge = {"entity_names": ["dronedream_dynamic_box"], "receipt_sha256": "a" * 64,
        "required_encounter_distance_m": 2.}
    result = runner._finalize_recovery_challenge(tmp_path, base, challenge, _vehicle())
    assert result["status"] == "failed"
    assert result["gates"]["dynamic_obstacle_challenge_sampled_clearance_respected"] is False
    assert json.loads((tmp_path / "mission_evidence.json").read_text())["status"] == "failed"
