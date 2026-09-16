from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import scripts.runtime_depth_safety_worker as depth_worker
from dronedream_agent_core.contracts import (
    PerceptionFusionHealth,
    RuntimeLocalSafetyObservation,
    Vector3,
    VehicleAsset,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.perception_runtime import (
    _bounded_strategic_navigation_context,
)
from dronedream_agent_core.runtime_sensor_contracts import (
    RuntimeMultimodalSensorSnapshot,
    RuntimeSensorStatus,
)
from scripts.runtime_depth_safety_worker import (
    _advance_dynamic_recovery_episode,
    _BoundedRuntimeEvidenceWriter,
    _controller_step_for_profile,
    _coordinate_candidates_enabled,
    _development_depth_frame_suppressed,
    _effective_control_profile,
    _expired_model_control_observation,
    _fault_adjusted_perception_health,
    _fresh_perception_can_finalize_model_cycle,
    _LatestOnlyDatasetWriter,
    _LatestRuntimeSnapshotWriter,
    _model_cycle_trigger,
    _model_navigation_goal_contract,
    _NativeIdentityTracker,
    _navigation_target_for_control_authority,
    _nearest_synchronized_sensor_pair,
    _payload_context,
    _remove_known_static_perception_duplicates,
    _runtime_phase_context,
    _strategic_sensor_context,
)


# 功能：
#   验证本地推理及仿真训练的图像传递不等待同步 PNG 落盘。
# 输入：
#   provider：本地提供方类型。
#   tmp_path、monkeypatch：独立目录与同步写入拦截器。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("provider", ["local-policy", "simulation-training"])
def test_local_visual_transports_do_not_wait_for_png_disk_write(provider, tmp_path, monkeypatch):
    submitted = []
    # 功能：
    #   同步落盘被意外调用时使测试失败。
    # 输入：
    #   args、kwargs：同步写入参数。
    # 输出：
    #   None：不返回业务数据。
    def unexpected_write(*args, **kwargs):
        raise AssertionError("Local camera transfer must not wait for filesystem writes")
    monkeypatch.setattr(depth_worker, "_atomic_bytes", unexpected_write)
    path = tmp_path / "not-yet-written.png"
    depth_worker._persist_navigation_image(
        provider=provider,
        writer=SimpleNamespace(submit_bytes=lambda *args: submitted.append(args)),
        path=path, png=b"real-camera-bytes",
    )
    assert submitted == [(path, b"real-camera-bytes")]
    assert not path.exists()


# 功能：
#   验证云端视觉适配仍生成真实文件，供现有云端数据地址读取。
# 输入：
#   tmp_path：独立图像输出目录。
# 输出：
#   None：不返回业务数据。
def test_cloud_visual_transport_preserves_file_backed_contract(tmp_path):
    path = tmp_path / "cloud-frame.png"
    depth_worker._persist_navigation_image(provider="openai", writer=None,
                                            path=path, png=b"real-camera-bytes")
    assert path.read_bytes() == b"real-camera-bytes"


# 功能：
#   验证已就绪推理使用当前感知与位姿复验，不必等待下一张图；不健康时禁止完成。
# 输入：
#   healthy：当前感知健康状态。
#   monkeypatch：固定时钟。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("healthy", [False, True])
def test_ready_continuous_result_uses_current_perception_without_waiting(healthy, monkeypatch):
    from test_perception_runtime import _frame

    monkeypatch.setattr(depth_worker.time, "time", lambda: 1.1)
    health = SimpleNamespace(stream_healthy=healthy)
    fusion = SimpleNamespace(health=Mock(return_value=health))
    coordinator = SimpleNamespace(poll=Mock(return_value="ready-receipt"))
    position, velocity = Vector3(x=1, y=2, z=3), Vector3(x=.1, y=.2, z=.3)
    result = depth_worker._poll_ready_navigation_cycle(
        coordinator, fusion=fusion, stale_tick=True, goal=position, goal_id="current-goal",
        frame=_frame(), position=position, velocity=velocity,
    )
    if healthy:
        assert result == "ready-receipt"
        kwargs = coordinator.poll.call_args.kwargs
        assert kwargs["now_unix_ms"] == 1100
        assert kwargs["current_perception_health"] is health
        assert kwargs["current_navigation_goal_id"] == "current-goal"
        assert kwargs["current_frame"].localization_position_m == position
        assert kwargs["current_frame"].localization_velocity_mps == velocity
    else:
        assert result is None
        coordinator.poll.assert_not_called()


# 功能：
#   验证直接速度控制模型绑定语义目标，不跟随教师不断改变的短前视点。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_direct_model_uses_semantic_goal_not_teachers_advancing_route_reference() -> None:
    route_target = Vector3(x=1.0, y=2.0, z=3.0)
    navigation_goal = Vector3(x=8.0, y=9.0, z=3.0)

    assert _model_navigation_goal_contract(
        route_target=route_target,
        navigation_goal=navigation_goal,
        omitted_coordinate_candidates=False,
        control_output_mode="normalized-body-velocity",
    ) == (navigation_goal, None, "normalized-body-velocity")
    assert _model_navigation_goal_contract(
        route_target=route_target,
        navigation_goal=navigation_goal,
        omitted_coordinate_candidates=False,
        control_output_mode="legacy-candidate-selection",
    ) == (navigation_goal, navigation_goal, "legacy-candidate-selection")


# 功能：
#   验证教师采集输入与部署时共享语义目标和连续输出契约，不混入旧坐标候选。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_teacher_collection_uses_same_semantic_goal_without_coordinate_candidates() -> None:
    route_target = Vector3(x=1.0, y=2.0, z=3.0)
    navigation_goal = Vector3(x=8.0, y=9.0, z=3.0)

    assert _model_navigation_goal_contract(
        route_target=route_target,
        navigation_goal=navigation_goal,
        omitted_coordinate_candidates=True,
        control_output_mode="legacy-candidate-selection",
    ) == (navigation_goal, None, "normalized-body-velocity")


# 功能：
#   验证旁观模型不能改动教师路线，只有要求模型授权的模式才采用获准模型目标。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_shadow_model_never_changes_deterministic_teacher_route_target() -> None:
    route_target = Vector3(x=4.0, y=2.0, z=1.0)
    position = Vector3(x=0.0, y=0.0, z=1.0)
    stale_model_target = Vector3(x=0.1, y=-0.2, z=1.1)
    directive = SimpleNamespace(
        model_navigation_authorized=True,
        target_m=stale_model_target,
    )

    assert _navigation_target_for_control_authority(
        route_target=route_target,
        current_position=position,
        model_directive=directive,
        require_model_control_authority=False,
    ) == route_target
    assert _navigation_target_for_control_authority(
        route_target=route_target,
        current_position=position,
        model_directive=directive,
        require_model_control_authority=True,
    ) == stale_model_target


# 功能：
#   验证模型控制模式没有有效授权时保持当前位置，不恢复教师运动。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_required_model_authority_holds_position_without_a_live_lease() -> None:
    route_target = Vector3(x=4.0, y=2.0, z=1.0)
    position = Vector3(x=0.0, y=0.0, z=1.0)

    assert _navigation_target_for_control_authority(
        route_target=route_target,
        current_position=position,
        model_directive=SimpleNamespace(
            model_navigation_authorized=False,
            target_m=route_target,
        ),
        require_model_control_authority=True,
    ) == position


# 功能：
#   验证晚到控制观测仍保留原始状态，但更新真实帧龄并撤回健康标记。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_late_model_control_tick_becomes_a_freshly_validated_unhealthy_observation() -> None:
    observation = RuntimeLocalSafetyObservation(
        sequence=4,
        observed_at_unix_ms=1_000,
        source="onboard",
        stream_healthy=True,
        stream_age_seconds=0.02,
        localization_covariance_m2=0.01,
        current_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        current_velocity_mps=Vector3(x=0.3, y=0.0, z=0.0),
        target_position_m=Vector3(x=1.0, y=0.0, z=1.0),
    )

    expired = _expired_model_control_observation(
        observation,
        published_at_unix_ms=2_250,
    )

    assert expired.stream_healthy is False
    assert expired.stream_age_seconds == pytest.approx(1.25)
    assert expired.current_position_m == observation.current_position_m


# 功能：
#   验证显式断流注入立即置缓存感知不健康，而未注入时保持原始健康对象。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_development_depth_fault_invalidates_cached_health_immediately() -> None:
    healthy = PerceptionFusionHealth(
        sensor_id="depth",
        latest_sequence=17,
        stream_age_seconds=0.01,
        localization_covariance_m2=0.01,
        accepted_ray_count=300,
        map_observation_count=300,
        stream_healthy=True,
        issue_codes=[],
    )

    unchanged = _fault_adjusted_perception_health(
        healthy,
        development_depth_fault_active=False,
    )
    suppressed = _fault_adjusted_perception_health(
        healthy,
        development_depth_fault_active=True,
    )

    assert unchanged is healthy
    assert suppressed.stream_healthy is False
    assert suppressed.issue_codes == ["DEVELOPMENT_DEPTH_FRAME_SUPPRESSED"]
    assert suppressed.latest_sequence == healthy.latest_sequence


# 功能：
#   验证异步完成记录使用提交时的目标身份，不能改写为消费时已推进的新目标。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_completed_snapshot_keeps_its_submitted_navigation_goal_epoch() -> None:
    snapshot = {
        "strategic_context": {
            "task": {
                "navigation_goal_id": "source-waypoint-0056",
            }
        }
    }

    assert depth_worker._submitted_snapshot_goal_id(
        snapshot,
        fallback_goal_id="source-waypoint-0057",
    ) == "source-waypoint-0056"
    assert depth_worker._submitted_snapshot_goal_id(
        {},
        fallback_goal_id="source-waypoint-0057",
    ) == "source-waypoint-0057"


# 功能：
#   阻塞首条序列化时验证提交仍快速返回，解阻后跨文件证据保持完整顺序。
# 输入：
#   tmp_path：独立证据目录。
#   monkeypatch：阻塞后台序列化以制造队列积压。
# 输出：
#   None：不返回业务数据。
def test_runtime_evidence_writer_is_nonblocking_and_preserves_every_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    original_jsonl_line = depth_worker._jsonl_line

    # 功能：
    #   只阻塞第一条记录，留出稳定窗口验证主线程提交不被磁盘侧拖住。
    # 输入：
    #   payload：待编码记录。
    # 输出：
    #   line：原序列化器输出的 JSON 行。
    def blocking_jsonl_line(payload: object) -> str:
        if isinstance(payload, dict) and payload.get("sequence") == 1:
            first_started.set()
            assert release_first.wait(timeout=2.0)
        return original_jsonl_line(payload)

    monkeypatch.setattr(depth_worker, "_jsonl_line", blocking_jsonl_line)
    writer = _BoundedRuntimeEvidenceWriter(
        tmp_path / "runtime-evidence-writer-summary.json",
        maximum_pending_records=8,
        flush_interval_seconds=0.01,
    )
    first_path = tmp_path / "safety.jsonl"
    second_path = tmp_path / "model.jsonl"
    assert writer.submit(first_path, {"sequence": 1})
    assert first_started.wait(timeout=1.0)

    started = time.monotonic()
    assert writer.submit(first_path, {"sequence": 2})
    assert writer.submit(second_path, {"sequence": 3})
    assert time.monotonic() - started < 0.1

    release_first.set()
    summary = writer.close(timeout_seconds=2.0)
    assert summary["complete"] is True
    assert summary["submitted_count"] == summary["completed_count"] == 3
    assert summary["rejected_count"] == 0
    assert [json.loads(line)["sequence"] for line in first_path.read_text().splitlines()] == [
        1,
        2,
    ]
    assert [json.loads(line)["sequence"] for line in second_path.read_text().splitlines()] == [3]


# 功能：
#   验证超容量拒绝明确留痕，并继续排空已经接收的证据，不能将拒绝算作成功。
# 输入：
#   tmp_path、monkeypatch：独立目录和首记录阻塞器。
# 输出：
#   None：不返回业务数据。
def test_runtime_evidence_writer_capacity_failure_is_explicit_and_drains_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    original_jsonl_line = depth_worker._jsonl_line

    # 功能：
    #   阻塞正在处理的首记录，使一个待处理槽位可被可重复地填满。
    # 输入：
    #   payload：待编码记录。
    # 输出：
    #   line：解阻后生成的 JSON 行。
    def blocking_jsonl_line(payload: object) -> str:
        if isinstance(payload, dict) and payload.get("sequence") == 1:
            first_started.set()
            assert release_first.wait(timeout=2.0)
        return original_jsonl_line(payload)

    monkeypatch.setattr(depth_worker, "_jsonl_line", blocking_jsonl_line)
    writer = _BoundedRuntimeEvidenceWriter(
        tmp_path / "runtime-evidence-writer-summary.json",
        maximum_pending_records=1,
        flush_interval_seconds=0.01,
    )
    evidence_path = tmp_path / "evidence.jsonl"
    assert writer.submit(evidence_path, {"sequence": 1})
    assert first_started.wait(timeout=1.0)
    assert writer.submit(evidence_path, {"sequence": 2})
    assert not writer.submit(evidence_path, {"sequence": 3})
    assert writer.issue == "RUNTIME_EVIDENCE_QUEUE_CAPACITY_EXCEEDED"

    release_first.set()
    summary = writer.close(timeout_seconds=2.0)
    assert summary["complete"] is False
    assert summary["submitted_count"] == summary["completed_count"] == 2
    assert summary["rejected_count"] == 1
    assert summary["pending_count"] == 0
    assert [json.loads(line)["sequence"] for line in evidence_path.read_text().splitlines()] == [
        1,
        2,
    ]


# 功能：
#   验证同路径待写快照可被更新覆盖，已开始写出的快照不会被修改，计数守恒。
# 输入：
#   tmp_path、monkeypatch：独立目录和写入阻塞器。
# 输出：
#   None：不返回业务数据。
def test_runtime_snapshot_writer_coalesces_only_superseded_mutable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = tmp_path / "snapshot-summary.json"
    snapshot_path = tmp_path / "health.json"
    writer = _LatestRuntimeSnapshotWriter(summary_path)
    first_started = threading.Event()
    release_first = threading.Event()
    original_atomic_bytes = depth_worker._atomic_bytes

    # 功能：
    #   暂停首份状态快照发布，让测试连续提交两个更新版本。
    # 输入：
    #   path、payload、kwargs：原子写入路径、内容及选项。
    # 输出：
    #   None：不返回业务数据。
    def blocking_atomic_bytes(path: Path, payload: bytes, **kwargs: object) -> None:
        if path == snapshot_path and json.loads(payload).get("sequence") == 1:
            first_started.set()
            assert release_first.wait(timeout=2.0)
        original_atomic_bytes(path, payload, **kwargs)

    monkeypatch.setattr(depth_worker, "_atomic_bytes", blocking_atomic_bytes)
    assert writer.submit_json(snapshot_path, {"sequence": 1})
    assert first_started.wait(timeout=1.0)
    assert writer.submit_json(snapshot_path, {"sequence": 2})
    assert writer.submit_json(snapshot_path, {"sequence": 3})
    release_first.set()

    summary = writer.close(timeout_seconds=2.0)
    assert summary["complete"] is True
    assert summary["submitted_count"] == 3
    assert summary["completed_count"] == 2
    assert summary["superseded_count"] == 1
    assert json.loads(snapshot_path.read_text())["sequence"] == 3


# 功能：
#   验证独立路径的二进制图像证据实际落盘，关闭报告完成。
# 输入：
#   tmp_path：独立输出目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_snapshot_writer_persists_unique_binary_evidence(tmp_path: Path) -> None:
    writer = _LatestRuntimeSnapshotWriter(tmp_path / "snapshot-summary.json")
    frame_path = tmp_path / "frames" / "forward.png"
    assert writer.submit_bytes(frame_path, b"png-evidence")

    summary = writer.close(timeout_seconds=2.0)
    assert summary["complete"] is True
    assert frame_path.read_bytes() == b"png-evidence"


# 功能：
#   验证多线程连续替换同一快照不争用暂存文件，最终内容始终是一份完整 JSON。
# 输入：
#   tmp_path：独立并发写入目录。
# 输出：
#   None：不返回业务数据。
def test_atomic_json_uses_independent_thread_temporaries(tmp_path: Path) -> None:
    target = tmp_path / "shared-health.json"
    errors: list[Exception] = []

    # 功能：
    #   由一个线程连续发布多份快照，异常交还主测试统一断言。
    # 输入：
    #   writer_id：用于区分两个并发生产者的身份。
    # 输出：
    #   None：不返回业务数据。
    def write_many(writer_id: int) -> None:
        try:
            for sequence in range(50):
                depth_worker._atomic_json(
                    target,
                    {"writer_id": writer_id, "sequence": sequence},
                )
        except Exception as error:  # pragma: no cover - asserted through shared state
            errors.append(error)

    threads = [threading.Thread(target=write_many, args=(writer_id,)) for writer_id in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3.0)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    payload = json.loads(target.read_text())
    assert payload["writer_id"] in {0, 1}
    assert 0 <= payload["sequence"] < 50


# 功能：
#   验证模拟器真值不能修改原生位姿，飞行中更改地图绑定也会被拒绝。
# 输入：
#   tmp_path：独立原生身份文件目录。
# 输出：
#   None：不返回业务数据。
def test_identity_tracker_never_corrects_native_pose_or_rebinds_midflight(tmp_path: Path) -> None:
    from test_native_flight_state import _identity

    from dronedream_agent_core.hashing import sha256_json

    path = tmp_path / "identity.json"
    tracker = _NativeIdentityTracker()
    payload = _identity()
    path.write_text(json.dumps(payload), encoding="utf-8")
    first = tracker.evaluate(path, now_unix_ms=1020)
    assert first.position_world_enu_m == Vector3(x=0, y=0, z=0)
    payload["observed_world_collision_center_m"] = {"x": 800, "y": 900, "z": -5}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert tracker.evaluate(path, now_unix_ms=1030) == first
    payload["map_frame_binding"]["collision_center_origin_world_enu_m"]["x"] = 1.
    payload["map_frame_binding_sha256"] = sha256_json(payload["map_frame_binding"])
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="BINDING_CHANGED"):
        tracker.evaluate(path, now_unix_ms=1030)


# 功能：
#   验证缺失速度、未来时钟和非对象遥测不会被补造为可用状态。
# 输入：
#   tmp_path：独立遥测目录。
#   invalid：待注入的非法遥测类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("invalid", ["missing-velocity", "future-time", "non-object"])
def test_identity_tracker_does_not_fabricate_missing_or_future_state(tmp_path, invalid) -> None:
    from test_native_flight_state import _identity
    telemetry = _identity()
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(telemetry), encoding="utf-8")
    tracker = _NativeIdentityTracker()
    assert tracker.evaluate(path, now_unix_ms=1_000)
    if invalid == "missing-velocity":
        del telemetry["observed_velocity_ned_mps"]
    elif invalid == "future-time":
        telemetry["updated_at_unix_ms"] = 1_100
    path.write_text(json.dumps([] if invalid == "non-object" else telemetry), encoding="utf-8")
    with pytest.raises(ValueError):
        tracker.evaluate(path, now_unix_ms=1_050)


# 功能：
#   穷举是否有新帧和当前健康状态，验证健康度而非新帧到达事件控制结果复验入口。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_model_completion_uses_latest_admitted_frame_while_it_remains_healthy() -> None:
    assert _fresh_perception_can_finalize_model_cycle(
        stale_tick=False,
        stream_healthy=True,
    )
    assert _fresh_perception_can_finalize_model_cycle(
        stale_tick=True,
        stream_healthy=True,
    )
    assert not _fresh_perception_can_finalize_model_cycle(
        stale_tick=False,
        stream_healthy=False,
    )
    assert not _fresh_perception_can_finalize_model_cycle(
        stale_tick=True,
        stream_healthy=False,
    )


class _BlockingDatasetRecorder:
    # 功能：
    #   创建可控制首个写入何时完成的记录器替身。
    # 输入：
    #   self：新替身。
    #   root：尚不存在的独立数据集目录。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir()
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.calls: list[bytes] = []

    # 功能：
    #   阻塞首样本直到测试释放，并记录最终真正写出的图像顺序。
    # 输入：
    #   self：记录器替身。
    #   payload：样本字段，包含 RGB PNG 字节。
    # 输出：
    #   None：不返回业务数据。
    def record(self, **payload: object) -> None:
        if not self.calls:
            self.first_started.set()
            assert self.release_first.wait(timeout=2.0)
        self.calls.append(payload["rgb_png"])  # type: ignore[arg-type]

    # 功能：
    #   报告替身实际接收的记录数。
    # 输入：
    #   self：记录器替身。
    # 输出：
    #   summary：包含已写记录数的统计。
    def summary(self) -> dict[str, object]:
        return {"record_count": len(self.calls)}


# 功能：
#   验证训练队列保留正在写出的首帧和最新待写帧，主动舍弃中间积压而不阻塞安全周期。
# 输入：
#   tmp_path、monkeypatch：独立目录和图像编码替身。
# 输出：
#   None：不返回业务数据。
def test_dataset_writer_drops_old_pending_work_without_blocking_safety_tick(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        depth_worker,
        "_gazebo_image_png",
        lambda image: image.payload,
    )
    recorder = _BlockingDatasetRecorder(tmp_path / "dataset")
    writer = _LatestOnlyDatasetWriter(
        recorder,  # type: ignore[arg-type]
        semantic_label_map_sha256=None,
        semantic_label_class_ids=frozenset(),
    )

    # 功能：
    #   提交一帧并检查排队耗时，统一该测试中的样本构造方式。
    # 输入：
    #   payload、sample_time：图像字节与对应样本单调时间。
    # 输出：
    #   None：不返回业务数据。
    def submit(payload: bytes, sample_time: float) -> None:
        image = type("Image", (), {"payload": payload, "width": 1, "height": 1})()
        started = time.monotonic()
        assert writer.submit(
            rgb_image=image,
            frame=object(),  # type: ignore[arg-type]
            sensor_snapshot=object(),
            recorded_at_unix_ms=1,
            recorded_at_monotonic_seconds=sample_time,
            rgb_sample_monotonic_seconds=sample_time,
            semantic_image=None,
            semantic_sample_monotonic_seconds=None,
            state={},
        )
        assert time.monotonic() - started < 0.1

    submit(b"first", 1.0)
    assert recorder.first_started.wait(timeout=1.0)
    submit(b"superseded", 2.0)
    submit(b"latest", 3.0)
    recorder.release_first.set()
    writer.close(timeout_seconds=2.0)

    assert recorder.calls == [b"first", b"latest"]
    assert writer.dropped_pending_count == 1
    assert writer.issue is None


# 功能：
#   验证独立渲染的 RGB 与语义帧按时间最近邻配对，不使用过期或偏差过大的邻帧。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_training_capture_pairs_independent_sensor_histories_by_nearest_time() -> None:
    rgb_old = object()
    rgb_new = object()
    semantic_old = object()
    semantic_near = object()

    selected = _nearest_synchronized_sensor_pair(
        ((rgb_old, 9.70), (rgb_new, 9.92)),
        ((semantic_old, 9.51), (semantic_near, 9.86)),
        now_monotonic_seconds=10.0,
        newest_rgb_after_monotonic_seconds=9.75,
    )
    assert selected == ((rgb_new, 9.92), (semantic_near, 9.86))
    assert (
        _nearest_synchronized_sensor_pair(
            ((rgb_new, 9.92),),
            ((semantic_old, 9.51),),
            now_monotonic_seconds=10.0,
            newest_rgb_after_monotonic_seconds=9.75,
        )
        is None
    )


# 功能：
#   验证部分重叠门框的体素及新椅子都被保留，不能用粗重叠删除潜在障碍。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_partial_door_frame_overlap_does_not_erase_unmodelled_occupied_volume() -> None:
    perceived = [
        {
            "name": "perception-voxel-175-326-138",
            "center_x": -41.125,
            "center_y": 10.6675,
            "center_z": 8.625,
            "size_x": 0.25,
            "size_y": 0.25,
            "size_z": 0.25,
            "yaw_rad": 0.0,
        },
        {
            "name": "perception-voxel-novel-chair",
            "center_x": -40.4,
            "center_y": 10.1,
            "center_z": 8.6,
            "size_x": 0.25,
            "size_y": 0.25,
            "size_z": 0.25,
            "yaw_rad": 0.0,
        },
    ]
    known_static = [
        {
            "name": "office-frame-east",
            "center_x": -41.01,
            "center_y": 10.6,
            "center_z": 8.52,
            "size_x": 0.08,
            "size_y": 0.11,
            "size_z": 2.2,
            "yaw_rad": 0.0,
        }
    ]

    novel, duplicate_count = _remove_known_static_perception_duplicates(
        perceived,
        known_static,
    )

    assert duplicate_count == 0
    assert novel == perceived


# 功能：
#   构造具有明确载荷上限、完整机体尺寸和运动限制的测试机体。
# 输入：
#   无。
# 输出：
#   vehicle：用于负载上下文测试的机体元数据。
def _vehicle() -> VehicleAsset:
    return VehicleAsset(
        asset_id="test-aircraft",
        name="Test aircraft",
        dry_mass_kg=2.0,
        max_takeoff_mass_kg=2.2,
        body_radius_m=0.38,
        body_height_m=0.43,
        max_speed_mps=1.2,
        max_acceleration_mps2=0.8,
        qualified_range_m=400.0,
        reserve_battery_percent=30.0,
        max_pickup_payload_kg=0.1,
        sensors=["gps", "odometry"],
    )


# 功能：
#   写入测试输入；有意允许生成含 NaN 的坏 JSON，以验证生产读取器拒绝它。
# 输入：
#   path、payload：临时路径和测试数据。
# 输出：
#   None：不返回业务数据。
def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# 功能：
#   验证取件、绑定与带载稳定回执依次组成当前负载状态，质量按机体上限核对。
# 输入：
#   tmp_path：独立动作回执目录。
# 输出：
#   None：不返回业务数据。
def test_payload_context_tracks_verified_attachment_and_loaded_stability(
    tmp_path: Path,
) -> None:
    receipts = tmp_path / "runtime-actions" / "receipts"
    _write(
        receipts / "action-001.receipt.json",
        {
            "status": "accepted",
            "task_id": "pickup",
            "output": {"detached": False},
        },
    )
    _write(
        receipts / "action-002.receipt.json",
        {
            "status": "accepted",
            "task_id": "confirm-custody",
            "output": {
                "detached": False,
                "payload_mass_kg": 0.1,
                "payload_physics_binding_confirmed": True,
            },
        },
    )
    _write(
        receipts / "action-003.receipt.json",
        {
            "status": "accepted",
            "task_id": "verify-loaded-stability",
            "output": {"detached": False, "loaded_hover_stable": True},
        },
    )

    assert _payload_context(receipts, _vehicle()) == {
        "state": "loaded-stable",
        "accepted_transition_steps": [
            "pickup",
            "confirm-custody",
            "verify-loaded-stability",
        ],
        "observed_payload_mass_kg": 0.1,
        "maximum_payload_kg": 0.1,
        "within_declared_payload_limit": True,
    }


# 功能：
#   验证错误负载质量不能通过限制；损坏 JSON 必须在接受装卸状态之前直接拒绝。
# 输入：
#   tmp_path：独立回执目录。
#   invalid_mass：布尔、负数或非有限质量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("invalid_mass", [True, -0.1, float("nan"), float("inf")])
def test_payload_context_fails_closed_for_invalid_accepted_mass(
    tmp_path: Path,
    invalid_mass: object,
) -> None:
    receipts = tmp_path / "runtime-actions" / "receipts"
    _write(
        receipts / "action-001.receipt.json",
        {
            "status": "accepted",
            "task_id": "confirm-custody",
            "output": {
                "detached": False,
                "payload_mass_kg": invalid_mass,
                "payload_physics_binding_confirmed": True,
            },
        },
    )

    if isinstance(invalid_mass, float) and not math.isfinite(invalid_mass):
        # 非有限数使整份 JSON 证据失效，必须在接受其装卸状态之前拒绝。
        with pytest.raises(ValueError, match="RUNTIME_PAYLOAD_RECEIPT_UNREADABLE"):
            _payload_context(receipts, _vehicle())
    else:
        context = _payload_context(receipts, _vehicle())
        assert context["state"] == "custody-confirmed"
        assert context["observed_payload_mass_kg"] is None
        assert context["within_declared_payload_limit"] is False


# 功能：
#   验证原生动力学字段和摘要正确投影，已有当前内存包时不能重读另一个时间点的文件。
# 输入：
#   tmp_path、monkeypatch：独立遥测目录与文件重读拦截器。
#   in_memory：是否提供已接入的内存遥测包。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("in_memory", [False, True])
def test_payload_context_flattens_verified_px4_dynamics(tmp_path: Path, monkeypatch,
                                                        in_memory) -> None:
    telemetry = tmp_path / "px4-identity-telemetry.json"
    _write(
        telemetry,
        {
            "dynamics": {
                "schema_version": "dronedream.px4-dynamics-telemetry.v1",
                "maximum_sample_age_seconds": 3.0,
                "ready_for_payload_inference": True,
                "issue_codes": [],
                "restart_counts": {"imu": 1},
                "sources": {
                    "imu": {
                        "sample_age_seconds": 0.01,
                        "acceleration_forward_m_s2": 0.1,
                        "acceleration_right_m_s2": -0.2,
                        "acceleration_down_m_s2": -9.7,
                        "angular_velocity_forward_rad_s": 0.01,
                        "angular_velocity_right_rad_s": -0.02,
                        "angular_velocity_down_rad_s": 0.03,
                    },
                    "attitude": {
                        "sample_age_seconds": 0.02,
                        "roll_deg": 1.5,
                        "pitch_deg": -2.5,
                    },
                    "battery": {
                        "sample_age_seconds": 0.03,
                        "current_battery_a": 5.0,
                        "voltage_v": 15.8,
                    },
                    "actuator_output": {
                        "sample_age_seconds": 0.04,
                        "actuator": [0.4, 0.5, 0.6, 0.5],
                        "normalization_ready": True,
                        "normalization_kind": "configured-absolute-maximum",
                        "normalization_absolute_maximum": 1000.0,
                    },
                },
            }
        },
    )

    captured_payload = json.loads(telemetry.read_text(encoding="utf-8"))
    with monkeypatch.context() as patch:
        if in_memory:
            # 功能：
            #   阻止已有内存包的控制上下文重新读取可能已更新的原生状态文件。
            # 输入：
            #   args、kwargs：文件读取参数。
            # 输出：
            #   None：不返回业务数据。
            def forbid_reopening(*args, **kwargs):
                raise AssertionError("Native context must reuse the admitted tick packet")
            patch.setattr(Path, "read_text", forbid_reopening)
        context = _payload_context(
            tmp_path / "missing-receipts", _vehicle(),
            identity_telemetry_path=telemetry,
            identity_telemetry_payload=captured_payload if in_memory else None,
        )

    assert context["dynamics"] == {
        "available": True,
        "ready": True,
        "telemetry_schema_version": "dronedream.px4-dynamics-telemetry.v1",
        "telemetry_payload_sha256": sha256_json(
            json.loads(telemetry.read_text(encoding="utf-8"))["dynamics"]
        ),
        "maximum_sample_age_seconds": 3.0,
        "source_sample_age_seconds": {
            "imu": 0.01,
            "attitude": 0.02,
            "battery": 0.03,
            "actuator_output": 0.04,
        },
        "issue_codes": [],
        "blocking_issue_codes": [],
        "restart_counts": {"imu": 1},
        "acceleration_forward_m_s2": 0.1,
        "acceleration_right_m_s2": -0.2,
        "acceleration_down_m_s2": -9.7,
        "angular_velocity_forward_rad_s": 0.01,
        "angular_velocity_right_rad_s": -0.02,
        "angular_velocity_down_rad_s": 0.03,
        "roll_deg": 1.5,
        "pitch_deg": -2.5,
        "current_battery_a": 5.0,
        "voltage_v": 15.8,
        "actuator_normalization_ready": True,
        "actuator_normalization_kind": "configured-absolute-maximum",
        "actuator_normalization_absolute_maximum": 1000.0,
        "actuator_mean": 0.5,
        "actuator_max_abs": 0.6,
    }


# 功能：
#   验证上游声称就绪仍不能覆盖 IMU 过期及必要执行器缺失。
# 输入：
#   tmp_path：独立动力学遥测目录。
# 输出：
#   None：不返回业务数据。
def test_payload_context_rejects_stale_dynamics_despite_upstream_ready_flag(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "px4-identity-telemetry.json"
    _write(
        telemetry,
        {
            "dynamics": {
                "schema_version": "dronedream.px4-dynamics-telemetry.v1",
                "maximum_sample_age_seconds": 3.0,
                "ready_for_payload_inference": True,
                "issue_codes": [],
                "restart_counts": {},
                "sources": {
                    "imu": {"sample_age_seconds": 3.1},
                    "attitude": {"sample_age_seconds": 0.01},
                    "battery": {"sample_age_seconds": 0.01},
                },
            }
        },
    )

    context = _payload_context(
        tmp_path / "missing-receipts",
        _vehicle(),
        identity_telemetry_path=telemetry,
    )

    assert context["dynamics"]["available"] is True
    assert context["dynamics"]["ready"] is False


# 功能：
#   验证缺失阶段文件只返回未知背景，不伪造执行阶段或检查点。
# 输入：
#   tmp_path：没有阶段文件的独立目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_phase_context_fails_closed_when_artifact_is_missing(
    tmp_path: Path,
) -> None:
    assert _runtime_phase_context(tmp_path / "runtime-phase.json") == {
        "phase": "UNKNOWN",
        "executor_phase": "UNKNOWN",
        "checkpoint_id": None,
    }


# 功能：
#   验证战略传感器摘要保留关键身份与状态，同时满足模型背景的浅层嵌套限制。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_strategic_sensor_context_stays_within_prompt_depth_contract() -> None:
    snapshot = RuntimeMultimodalSensorSnapshot(
        captured_at_monotonic_seconds=10.0,
        contract_set_sha256="a" * 64,
        ready_for_motion=True,
        statuses=[
            RuntimeSensorStatus(
                sensor_id="oakd-lite-forward-rgb",
                modality="rgb-camera",
                required_for_motion=False,
                latest_sequence=11,
                sample_age_seconds=0.02,
                transport_latency_seconds=0.01,
                quality=0.98,
                coverage=1.0,
                health="healthy",
                payload_sha256="b" * 64,
            ),
            RuntimeSensorStatus(
                sensor_id="oakd-lite-depth",
                modality="depth-camera",
                required_for_motion=True,
                latest_sequence=12,
                sample_age_seconds=0.01,
                transport_latency_seconds=0.01,
                quality=0.99,
                coverage=1.0,
                health="healthy",
                payload_sha256="c" * 64,
            ),
        ],
        active_sensor_ids=["oakd-lite-depth", "oakd-lite-forward-rgb"],
        issue_codes=[],
    )
    sensor_context = {
        "metric_authority": "calibrated depth plus localization",
        **_strategic_sensor_context(snapshot),
    }

    bounded = _bounded_strategic_navigation_context({"sensor_contract": sensor_context})

    assert bounded["sensor_contract"]["ready_for_motion"] is True
    assert bounded["sensor_contract"]["sensor_statuses"] == [
        {
            "health": "healthy",
            "modality": "rgb-camera",
            "required_for_motion": False,
            "sensor_id": "oakd-lite-forward-rgb",
        },
        {
            "health": "healthy",
            "modality": "depth-camera",
            "required_for_motion": True,
            "sensor_id": "oakd-lite-depth",
        },
    ]


# 功能：
#   验证近目标、动作检查点、精细锁存及制动距离均只收紧局部控制工况与前视距离。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_precision_profile_uses_micro_targets_without_expanding_cruise_bound() -> None:
    assert (
        _controller_step_for_profile(
            profile="cruise",
            maximum_step_m=0.9,
            world_resolution_m=0.25,
        )
        == 0.9
    )
    assert (
        _controller_step_for_profile(
            profile="precision",
            maximum_step_m=0.9,
            world_resolution_m=0.25,
        )
        == 0.1875
    )
    assert (
        _effective_control_profile(
            requested_profile="cruise",
            action_checkpoint_goal=True,
            observed_goal_distance_m=3.9,
        )
        == "precision"
    )
    assert (
        _effective_control_profile(
            requested_profile="cruise",
            action_checkpoint_goal=False,
            observed_goal_distance_m=0.7,
        )
        == "precision"
    )
    assert (
        _effective_control_profile(
            requested_profile="cruise",
            action_checkpoint_goal=False,
            observed_goal_distance_m=1.0,
        )
        == "cruise"
    )
    assert (
        _effective_control_profile(
            requested_profile="cruise",
            action_checkpoint_goal=False,
            observed_goal_distance_m=1.2,
            precision_latched=True,
        )
        == "precision"
    )
    assert (
        _effective_control_profile(
            requested_profile="cruise",
            action_checkpoint_goal=False,
            observed_goal_distance_m=1.1,
            observed_speed_mps=0.8,
            maximum_acceleration_mps2=0.8,
        )
        == "precision"
    )


# 功能：
#   验证显式丢帧注入的开始和结束边界，未配置时不影响正常传感器流。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_development_depth_drop_window_is_explicit_and_bounded() -> None:
    disabled = _development_depth_frame_suppressed(
        first_accepted_frame_monotonic=10.0,
        now_monotonic=12.0,
        drop_after_seconds=None,
        drop_duration_seconds=None,
    )
    before = _development_depth_frame_suppressed(
        first_accepted_frame_monotonic=10.0,
        now_monotonic=11.9,
        drop_after_seconds=2.0,
        drop_duration_seconds=2.5,
    )
    active = _development_depth_frame_suppressed(
        first_accepted_frame_monotonic=10.0,
        now_monotonic=12.0,
        drop_after_seconds=2.0,
        drop_duration_seconds=2.5,
    )
    recovered = _development_depth_frame_suppressed(
        first_accepted_frame_monotonic=10.0,
        now_monotonic=14.5,
        drop_after_seconds=2.0,
        drop_duration_seconds=2.5,
    )

    assert disabled is False
    assert before is False
    assert active is True
    assert recovered is False


# 功能：
#   验证模型周期按初始、目标变化、动态障碍和停滞事件选择明确触发原因。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_model_cycle_trigger_escalates_semantic_progress_stall() -> None:
    assert (
        _model_cycle_trigger(
            model_cycle_started=True,
            now_monotonic=30.0,
            next_model_cycle_monotonic=29.0,
            target_changed=False,
            dynamic_obstacle_changed=False,
            progress_recovery_requested=True,
        )
        == "progress-stalled"
    )
    assert (
        _model_cycle_trigger(
            model_cycle_started=True,
            now_monotonic=30.0,
            next_model_cycle_monotonic=31.0,
            target_changed=False,
            dynamic_obstacle_changed=False,
            progress_recovery_requested=True,
        )
        is None
    )


# 功能：
#   验证持续动态障碍共享恢复身份，静默超时或目标切换后才新建事件。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_dynamic_obstacle_recovery_episode_groups_one_continuous_encounter() -> None:
    first = _advance_dynamic_recovery_episode(
        state=None,
        navigation_goal_id="goal-001",
        dynamic_signature="a" * 64,
        dynamic_present=True,
        now_monotonic=10.0,
    )
    repeated = _advance_dynamic_recovery_episode(
        state=first,
        navigation_goal_id="goal-001",
        dynamic_signature="b" * 64,
        dynamic_present=True,
        now_monotonic=13.0,
    )
    intermittent = _advance_dynamic_recovery_episode(
        state=repeated,
        navigation_goal_id="goal-001",
        dynamic_signature="c" * 64,
        dynamic_present=True,
        now_monotonic=39.0,
    )
    quiet = _advance_dynamic_recovery_episode(
        state=intermittent,
        navigation_goal_id="goal-001",
        dynamic_signature="d" * 64,
        dynamic_present=False,
        now_monotonic=70.0,
    )
    changed_scene = _advance_dynamic_recovery_episode(
        state=quiet,
        navigation_goal_id="goal-001",
        dynamic_signature="d" * 64,
        dynamic_present=True,
        now_monotonic=71.0,
    )

    assert first is repeated
    assert repeated is intermittent
    assert quiet is None
    assert changed_scene is not None
    assert first["episode_id"] != changed_scene["episode_id"]
    assert len(str(first["episode_id"])) == 33


# 功能：
#   验证狭窄空间速度上限服从路线已有更严限制，不为精细模式反向提速。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_precision_profile_uses_confined_space_speed_envelope() -> None:
    assert depth_worker._planner_speed_for_profile(
        profile="cruise", route_speed_limit_mps=0.34
    ) == pytest.approx(0.34)
    assert depth_worker._planner_speed_for_profile(
        profile="precision", route_speed_limit_mps=0.34
    ) == pytest.approx(0.2)
    assert depth_worker._planner_speed_for_profile(
        profile="precision", route_speed_limit_mps=0.15
    ) == pytest.approx(0.15)


# 功能：
#   验证直接控制及教师采集都可禁用旧坐标候选搜索，兼容模式保持显式可选。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_direct_control_and_teacher_collection_can_disable_coordinate_search() -> None:
    assert not _coordinate_candidates_enabled(
        omitted=False,
        control_output_mode="normalized-body-velocity",
    )
    assert not _coordinate_candidates_enabled(
        omitted=True,
        control_output_mode="legacy-candidate-selection",
    )
    assert _coordinate_candidates_enabled(
        omitted=False,
        control_output_mode="legacy-candidate-selection",
    )
# 功能：
#   验证传输保留时间边界和拒绝调用归因，拒绝回执只能解释悬停，不能取得运动权限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_dispatch_budget_is_required_and_rejected_call_gets_only_hold_attribution():
    from types import SimpleNamespace

    from test_runtime_commands import _load_depth_worker

    worker = _load_depth_worker()
    assert worker._model_control_dispatch_unavailable(
        published_at_unix_ms=1000, authority_deadline_unix_ms=1069)
    assert not worker._model_control_dispatch_unavailable(
        published_at_unix_ms=1000, authority_deadline_unix_ms=1070)
    rejected = SimpleNamespace(hold_reason="MODEL_NAVIGATION_DECISION_STALE",
                               model_call_id="new-rejected-call")
    held = SimpleNamespace(model_navigation_authorized=False, model_call_id="old-call")
    active = SimpleNamespace(model_navigation_authorized=True, model_call_id="active-call")
    assert worker._safety_hold_call_id(held, rejected) == "new-rejected-call"
    assert worker._safety_hold_call_id(None, rejected) == "new-rejected-call"
    assert worker._safety_hold_call_id(active, rejected) == "active-call"
    assert worker._safety_hold_call_id(held, None) == "old-call"
