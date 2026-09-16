"""工作进程与真实文件合同的反例测试；不连接飞控、不调用付费模型、不伪造飞行。"""

import contextlib
import errno
import json
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from test_perception_control_integrity import _coordinator
from test_runtime_commands import _load_executor
from test_runtime_depth_context import _vehicle

import scripts.runtime_depth_safety_worker as worker
from dronedream_agent_core.contracts import Px4CoordinateContract, RuntimeLocalSafetyObservation
from dronedream_agent_core.perception_runtime import _NavigationWorkerResult


# 功能：
#   通过真实执行器发布目标文件，覆盖生产者与工作进程消费端之间的协议。
# 输入：
#   path：独立测试目标文件。
# 输出：
#   payload：实际落盘的目标数据。
def _publish_target(path):
    executor = _load_executor()
    executor._publish_local_safety_target(
        path=path, setpoint=SimpleNamespace(north_m=2., east_m=3., down_m=-1.),
        coordinate_contract=Px4CoordinateContract(
            model_root_world_enu_m=[0., 0., 0.],
            collision_center_offset_model_m=[0., 0., .3]),
        navigation_goal_id="customer-pickup", action_checkpoint_goal=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload


# 功能：
#   验证生产者的真实目标可以读取，固定目标不会因为持续时间长就被误当成过期传感器。
# 输入：
#   tmp_path：独立文件目录。
# 输出：
#   None：不返回业务数据。
def test_executor_target_roundtrip_and_persistent_goal(tmp_path):
    path = tmp_path / "target.json"
    payload = _publish_target(path)
    target = worker._read_control_target(path, now_unix_ms=payload["updated_at_unix_ms"] + 600_000)
    assert target == payload
    assert target["target_position_m"] == {"x": 3., "y": 2., "z": 1.3}


# 功能：
#   验证目标中的假布尔、坏向量和未来时间不能通过隐式转换获得控制权限。
# 输入：
#   field、value：要破坏的字段和值。
#   tmp_path：独立文件目录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(("field", "value"), [
    ("action_checkpoint_goal", "false"), ("tracking_recovery_active", 1),
    ("navigation_goal_position_m", {"x": "9", "y": 0., "z": 1.}),
    ("updated_at_unix_ms", True), ("updated_at_unix_ms", 10**15),
    ("navigation_goal_id", " "), ("control_profile", []),
])
def test_target_boundary_rejects_untyped_control(field, value, tmp_path):
    path = tmp_path / "target.json"
    payload = _publish_target(path)
    now = payload["updated_at_unix_ms"]
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        worker._read_control_target(path, now_unix_ms=now)


# 功能：
#   验证重复键被整份拒绝，不接受某个解析器保留的最后一个控制标志。
# 输入：
#   tmp_path：独立目标目录。
# 输出：
#   None：不返回业务数据。
def test_target_duplicate_flags_are_rejected(tmp_path):
    path = tmp_path / "target.json"
    payload = _publish_target(path)
    path.write_text('{"action_checkpoint_goal": false,' + json.dumps(payload)[1:], encoding="utf-8")
    with pytest.raises(ValueError):
        worker._read_control_target(path, now_unix_ms=payload["updated_at_unix_ms"])


# 功能：
#   验证机体元数据缺失时无默认质量兜底，声明不符或超过实际性能限制时拒绝启动。
# 输入：
#   tmp_path：独立机体文件目录。
# 输出：
#   None：不返回业务数据。
def test_worker_vehicle_requires_real_metadata_and_limits(tmp_path):
    vehicle = _vehicle()
    path = tmp_path / "vehicle.json"
    limits = dict(radius_m=vehicle.body_radius_m, height_m=vehicle.body_height_m,
                  speed_mps=vehicle.max_speed_mps, acceleration_mps2=vehicle.max_acceleration_mps2)
    with pytest.raises(FileNotFoundError):
        worker._load_worker_vehicle(path, **limits)
    path.write_text(vehicle.model_dump_json(), encoding="utf-8")
    assert worker._load_worker_vehicle(path, **limits) == vehicle
    for name in limits:
        with pytest.raises(ValueError):
            worker._load_worker_vehicle(path, **{**limits, name: limits[name] * 2})


# 功能：
#   验证未传机体元数据的命令行在创建原生资源之前就被明确拒绝。
# 输入：
#   monkeypatch：提供缺少必要元数据的命令行。
# 输出：
#   None：不返回业务数据。
def test_worker_cli_has_no_dummy_vehicle_fallback(monkeypatch):
    monkeypatch.setattr(worker.sys, "argv", ["worker"])
    with contextlib.ExitStack() as cleanup, pytest.raises(SystemExit) as error:
        worker._run_worker(cleanup)
    assert error.value.code == 2


# 功能：
#   验证未知输出模式不能静默退回旧坐标候选，布尔开关必须保持真实类型。
# 输入：
#   mode、omitted：非法输出契约或开关。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(("mode", "omitted"), [
    ("candidate-selection", False), ("mistyped-control", True),
    ("normalized-body-velocity", "false"), ([], False),
])
def test_unknown_output_contract_never_revives_legacy_search(mode, omitted):
    with pytest.raises(ValueError):
        worker._coordinate_candidates_enabled(omitted=omitted, control_output_mode=mode)


# 功能：
#   验证过期观测保留真实年龄，不能把一分钟前的图像截成三十秒。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_expired_observation_preserves_actual_age():
    observation = RuntimeLocalSafetyObservation(
        sequence=1, observed_at_unix_ms=1000, source="onboard", stream_healthy=True,
        stream_age_seconds=.01, localization_covariance_m2=.01,
        current_position_m={"x": 0., "y": 0., "z": 1.},
        current_velocity_mps={"x": 0., "y": 0., "z": 0.},
        target_position_m={"x": 1., "y": 0., "z": 1.})
    expired = worker._expired_model_control_observation(observation, published_at_unix_ms=62_000)
    assert expired.stream_age_seconds == 61.
    assert not expired.stream_healthy
    assert observation.stream_healthy
    with pytest.raises(ValueError, match="CLOCK"):
        worker._expired_model_control_observation(observation, published_at_unix_ms=999)


# 功能：
#   验证等待发布读锁时，暂存文件被替换不能发布或被当成本次文件删除。
# 输入：
#   tmp_path、monkeypatch：独立目录与可控的文件替换故障。
# 输出：
#   None：不返回业务数据。
def test_retry_does_not_publish_replaced_temporary(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.write_bytes(b"original")
    retained = tmp_path / "retained-owned"
    replaced = []

    # 功能：
    #   保留本次原暂存并放入不同文件，再模拟读锁，避免依赖文件系统重用 inode 的时机。
    # 输入：
    #   source、destination：本次发布路径。
    # 输出：
    #   None：不返回业务数据。
    def replace_during_lock(source, destination):
        assert destination == target
        source.rename(retained)
        source.write_bytes(b"not-owned")
        replaced.append(source)
        raise OSError(errno.EBUSY, "injected lock")

    monkeypatch.setattr(worker.os, "replace", replace_during_lock)
    with pytest.raises(ValueError, match="TEMPORARY_REPLACED"):
        worker._atomic_bytes(target, b"new-owned")
    assert target.read_bytes() == b"original"
    assert retained.read_bytes() == b"new-owned"
    assert len(replaced) == 1 and replaced[0].read_bytes() == b"not-owned"


# 功能：
#   验证超深 JSON 在创建输出目录之前被拒绝。
# 输入：
#   tmp_path：独立证据目录。
# 输出：
#   None：不返回业务数据。
def test_snapshot_budget_precedes_filesystem_creation(tmp_path):
    value = {}
    for _ in range(70):
        value = {"nested": value}
    path = tmp_path / "not-created" / "snapshot.json"
    with pytest.raises(ValueError):
        worker._atomic_json(path, value)
    assert not path.parent.exists()


# 功能：
#   验证换目标立即撤销旧恢复事件，单调时钟倒退不得继续延长事件。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_recovery_episode_respects_goal_and_clock():
    values = dict(navigation_goal_id="goal-a", dynamic_signature="a" * 64)
    state = worker._advance_dynamic_recovery_episode(
        state=None, dynamic_present=True, now_monotonic=10., **values)
    assert worker._advance_dynamic_recovery_episode(state=state, dynamic_present=False,
        now_monotonic=11., **{**values, "navigation_goal_id": "goal-b"}) is None
    with pytest.raises(ValueError, match="CLOCK"):
        worker._advance_dynamic_recovery_episode(
            state=state, dynamic_present=True, now_monotonic=9., **values)


# 功能：
#   验证没有新的可执行动作时，全部迟到调用仍交给真实 FIFO 并完成文件回读。
# 输入：
#   tmp_path：独立模型调用证据目录。
# 输出：
#   None：不返回业务数据。
def test_drain_model_calls_without_motion_receipt(tmp_path):
    records = deque([{"call_id": "rejected"}, {"call_id": "late"}])
    coordinator = SimpleNamespace(
        pop_model_call_record=lambda: records.popleft() if records else None)
    path = tmp_path / "calls.jsonl"
    writer = worker._BoundedRuntimeEvidenceWriter(tmp_path / "summary.json")
    try:
        assert worker._drain_model_calls(coordinator, writer, path) == 2
    finally:
        summary = writer.close(timeout_seconds=2.)
    assert summary["complete"]
    call_ids = [json.loads(line)["call_id"] for line in path.read_text().splitlines()]
    assert call_ids == ["rejected", "late"]


# 功能：
#   验证协调器关闭后仍能回收真实已完成调用，只记录一次且不恢复运动授权。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_completed_quarantined_call_can_be_saved_after_close():
    coordinator = _coordinator()
    future = Future()
    assert future.set_running_or_notify_cancel()
    try:
        coordinator._quarantine_timed_out_work(future, reset_transport=False)
        coordinator.close()
        assert coordinator.request_pending
        future.set_result(_NavigationWorkerResult(
            snapshot=None, failure_reason="expired",
            discarded_call_record=SimpleNamespace(call_id="completed-after-close")))
        assert coordinator.pop_model_call_record().call_id == "completed-after-close"
        assert coordinator.pop_model_call_record() is None
        assert not coordinator.request_pending and coordinator._active_pilot_control is None
    finally:
        coordinator.close()


# 功能：
#   验证可变状态快照也有队列上限，配置错误不能启动无界内存后台队列。
# 输入：
#   tmp_path：独立输出目录。
# 输出：
#   None：不返回业务数据。
def test_snapshot_path_capacity_has_upper_bound(tmp_path):
    with pytest.raises(ValueError):
        worker._LatestRuntimeSnapshotWriter(tmp_path / "summary.json", maximum_pending_paths=65)
    assert not list(tmp_path.iterdir())
