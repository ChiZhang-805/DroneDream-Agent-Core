from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dronedream_agent_core.contracts import (
    OnboardPerceptionFrame,
    RangeRayObservation,
    Vector3,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.runtime_multimodal_dataset import (
    RuntimeMultimodalDatasetRecord,
    RuntimeMultimodalDatasetRecorder,
)
from dronedream_agent_core.runtime_sensor_contracts import (
    RuntimeMultimodalSensorSnapshot,
    RuntimeSensorStatus,
)


# 功能：
#   构造固定几何和时钟的合成深度帧，用于记录结构验证而非真实传感器验收。
# 输入：
#   sequence：本例帧序号。
# 输出：
#   frame：带一条无障碍射线的测试帧。
def _frame(sequence: int = 1) -> OnboardPerceptionFrame:
    ray = RangeRayObservation(
        origin_m=Vector3(x=0.0, y=0.0, z=1.0),
        endpoint_m=Vector3(x=1.0, y=0.0, z=1.0),
        hit=False,
        confidence=1.0,
        observed_at_monotonic_seconds=10.0,
    )
    frame = OnboardPerceptionFrame(
        sensor_id="oakd-lite-depth",
        sequence=sequence,
        observed_at_unix_ms=1_000,
        localization_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        localization_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        localization_covariance_m2=0.01,
        range_rays=[ray],
    )
    return frame


# 功能：
#   构造与深度帧对应的健康传感器状态，不代表真实硬件已就绪。
# 输入：
#   无。
# 输出：
#   snapshot：固定时间和摘要的测试状态快照。
def _snapshot() -> RuntimeMultimodalSensorSnapshot:
    snapshot = RuntimeMultimodalSensorSnapshot(
        captured_at_monotonic_seconds=10.01,
        contract_set_sha256="a" * 64,
        ready_for_motion=True,
        statuses=[
            RuntimeSensorStatus(
                sensor_id="oakd-lite-depth",
                modality="depth-camera",
                required_for_motion=True,
                latest_sequence=1,
                sample_age_seconds=0.01,
                transport_latency_seconds=0.01,
                quality=1.0,
                coverage=1.0,
                health="healthy",
                payload_sha256="b" * 64,
            )
        ],
        active_sensor_ids=["oakd-lite-depth"],
    )
    return snapshot


# 功能：
#   验证合法样本限频、图像去重、记录哈希链和增量文件摘要的一致性。
# 输入：
#   tmp_path：独立测试数据集根。
# 输出：
#   None：不返回业务数据。
def test_records_hash_chained_deduplicated_bounded_samples(tmp_path: Path) -> None:
    root = tmp_path / "multimodal"
    recorder = RuntimeMultimodalDatasetRecorder(
        root,
        flight_id="school-map-flight",
        map_sha256="c" * 64,
        maximum_bytes=1024 * 1024,
        minimum_period_seconds=0.1,
    )

    first = recorder.record(
        rgb_png=b"valid-png-placeholder",
        frame=_frame(),
        sensor_snapshot=_snapshot(),
        recorded_at_unix_ms=1_000,
        recorded_at_monotonic_seconds=10.01,
        state={"phase": "TRACK"},
    )
    skipped = recorder.record(
        rgb_png=b"valid-png-placeholder",
        frame=_frame(sequence=2),
        sensor_snapshot=_snapshot(),
        recorded_at_unix_ms=1_050,
        recorded_at_monotonic_seconds=10.05,
        state={"phase": "TRACK"},
    )
    second = recorder.record(
        rgb_png=b"valid-png-placeholder",
        frame=_frame(sequence=2),
        sensor_snapshot=_snapshot(),
        recorded_at_unix_ms=1_200,
        recorded_at_monotonic_seconds=10.21,
        state={"phase": "TRACK"},
    )

    assert first is not None
    assert skipped is None
    assert second is not None
    assert second.previous_record_sha256 == first.record_sha256
    assert len(list((root / "rgb").glob("*.png"))) == 1
    records = [
        RuntimeMultimodalDatasetRecord.model_validate(json.loads(line))
        for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    for record in records:
        payload = record.model_dump(mode="json")
        observed_hash = payload.pop("record_sha256")
        assert observed_hash == sha256_json(payload)
    summary = recorder.summary()
    assert summary["record_count"] == 2
    assert summary["latest_record_sha256"] == second.record_sha256
    assert summary["records_sha256"] == hashlib.sha256(
        (root / "records.jsonl").read_bytes()
    ).hexdigest()


# 功能：
#   宿主时间前进但相机源时间不变时不重复记录；源时间倒退须拒绝。
# 输入：
#   tmp_path：隔离采集目录。
# 输出：
#   None：不返回业务数据。
def test_camera_repoll_is_not_a_new_training_observation(tmp_path):
    recorder = RuntimeMultimodalDatasetRecorder(tmp_path / "capture",
        flight_id="same-source-frame", map_sha256="c" * 64)
    arguments = {"rgb_png": b"synthetic", "frame": _frame(), "sensor_snapshot": _snapshot(),
                 "recorded_at_unix_ms": 1000, "state": {},
                 "rgb_sample_monotonic_seconds": 10.0}
    assert recorder.record(**arguments, recorded_at_monotonic_seconds=10.1) is not None
    assert recorder.record(**arguments, recorded_at_monotonic_seconds=10.3) is None
    arguments["rgb_sample_monotonic_seconds"] = 9.9
    with pytest.raises(ValueError, match="RGB_CLOCK_REVERSED"):
        recorder.record(**arguments, recorded_at_monotonic_seconds=10.5)
    assert recorder.summary()["record_count"] == 1


# 功能：
#   汇总统计不得随记录文件增长而重复读取文件内容。
# 输入：
#   tmp_path：隔离数据集根。
#   monkeypatch：封禁无界内容读取的替换工具。
# 输出：
#   None：不返回业务数据。
def test_summary_does_not_reread_the_growing_record_file(tmp_path, monkeypatch) -> None:
    recorder = RuntimeMultimodalDatasetRecorder(
        tmp_path / "multimodal",
        flight_id="constant-time-summary",
        map_sha256="c" * 64,
        maximum_bytes=1024 * 1024,
    )
    recorder.record(
        rgb_png=b"valid-png-placeholder",
        frame=_frame(),
        sensor_snapshot=_snapshot(),
        recorded_at_unix_ms=1_000,
        recorded_at_monotonic_seconds=10.01,
        state={},
    )

    monkeypatch.setattr(Path, "read_bytes", lambda _path: (_ for _ in ()).throw(AssertionError))

    assert recorder.summary()["record_count"] == 1


# 功能：
#   拒绝空、越界、含路径分隔符及非规范大小写的飞行标识。
# 输入：
#   tmp_path：隔离数据集根。
#   flight_id：非法飞行标识。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("flight_id", ["Uppercase", "bad/id", "", "a" * 121])
def test_recorder_rejects_invalid_flight_identity(tmp_path: Path, flight_id: str) -> None:
    with pytest.raises(ValueError, match="identity"):
        RuntimeMultimodalDatasetRecorder(
            tmp_path / flight_id.replace("/", "-"),
            flight_id=flight_id,
            map_sha256="c" * 64,
            maximum_bytes=1024 * 1024,
        )


# 功能：
#   图像和记录总增长超过预算时应在写入前拒绝。
# 输入：
#   tmp_path：隔离数据集根。
# 输出：
#   None：不返回业务数据。
def test_recorder_fails_before_exceeding_quota(tmp_path: Path) -> None:
    recorder = RuntimeMultimodalDatasetRecorder(
        tmp_path / "multimodal",
        flight_id="quota-flight",
        map_sha256="c" * 64,
        maximum_bytes=1024 * 1024,
    )

    with pytest.raises(RuntimeError, match="QUOTA_EXCEEDED"):
        recorder.record(
            rgb_png=b"x" * (1024 * 1024),
            frame=_frame(),
            sensor_snapshot=_snapshot(),
            recorded_at_unix_ms=1_000,
            recorded_at_monotonic_seconds=10.01,
            state={},
        )


# 功能：
#   验证时间对齐的 RGB 和语义掩码绑定相同记录，摘要与配对间隔均保留。
# 输入：
#   tmp_path：隔离数据集根。
# 输出：
#   None：不返回业务数据。
def test_recorder_binds_synchronized_semantic_supervision(tmp_path: Path) -> None:
    recorder = RuntimeMultimodalDatasetRecorder(
        tmp_path / "multimodal",
        flight_id="semantic-flight",
        map_sha256="c" * 64,
        maximum_bytes=1024 * 1024,
    )

    record = recorder.record(
        rgb_png=b"rgb-png",
        frame=_frame(),
        sensor_snapshot=_snapshot(),
        recorded_at_unix_ms=1_000,
        recorded_at_monotonic_seconds=10.05,
        rgb_sample_monotonic_seconds=10.0,
        semantic_mask_png=b"semantic-png",
        semantic_label_map_sha256="d" * 64,
        semantic_sample_monotonic_seconds=10.04,
        state={},
    )

    assert record is not None
    assert record.semantic_mask_relative_path is not None
    assert (recorder.root / record.semantic_mask_relative_path).read_bytes() == b"semantic-png"
    assert record.rgb_semantic_time_offset_seconds == pytest.approx(0.04)
    assert recorder.summary()["map_sha256"] == "c" * 64


# 功能：
#   两种图像时间差超过同步契约时拒绝监督配对，不生成假同步训练记录。
# 输入：
#   tmp_path：隔离数据集根。
# 输出：
#   None：不返回业务数据。
def test_recorder_rejects_unsynchronized_semantic_supervision(tmp_path: Path) -> None:
    recorder = RuntimeMultimodalDatasetRecorder(
        tmp_path / "multimodal",
        flight_id="semantic-sync-failure",
        map_sha256="c" * 64,
        maximum_bytes=1024 * 1024,
    )

    with pytest.raises(ValueError, match="synchronization"):
        recorder.record(
            rgb_png=b"rgb-png",
            frame=_frame(),
            sensor_snapshot=_snapshot(),
            recorded_at_unix_ms=1_000,
            recorded_at_monotonic_seconds=10.5,
            rgb_sample_monotonic_seconds=10.0,
            semantic_mask_png=b"semantic-png",
            semantic_label_map_sha256="d" * 64,
            semantic_sample_monotonic_seconds=10.11,
            state={},
        )
