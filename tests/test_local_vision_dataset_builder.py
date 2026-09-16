from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dronedream_agent_core.contracts import (
    OnboardPerceptionFrame,
    RangeRayObservation,
    Vector3,
)
from dronedream_agent_core.local_vision_training import LocalVisionTrainingSample
from dronedream_agent_core.runtime_multimodal_dataset import (
    RuntimeMultimodalDatasetRecorder,
)
from dronedream_agent_core.runtime_sensor_contracts import (
    RuntimeMultimodalSensorSnapshot,
    RuntimeSensorStatus,
)
from scripts.build_local_vision_dataset import (
    _flight_records,
    _forward_navigation_frame_eligible,
)


# 功能：
#   将合成 RGB 或标签矩阵编码为真实 PNG 字节，供数据集构建器实际解码。
# 输入：
#   pixels：测试像素矩阵。
# 输出：
#   payload：完整 PNG 字节。
def _png(pixels: np.ndarray) -> bytes:
    from io import BytesIO

    with BytesIO() as output, Image.fromarray(pixels) as image:
        image.save(output, format="PNG")
        payload = output.getvalue()
    return payload


# 功能：
#   构造测试用单射线深度帧，不代表实际感知精度。
# 输入：
#   无。
# 输出：
#   frame：固定姿态位置和时间的合成帧。
def _frame() -> OnboardPerceptionFrame:
    frame = OnboardPerceptionFrame(
        sensor_id="oakd-lite-depth",
        sequence=1,
        observed_at_unix_ms=1_000,
        localization_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        localization_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        localization_covariance_m2=0.01,
        range_rays=[
            RangeRayObservation(
                origin_m=Vector3(x=0.0, y=0.0, z=1.0),
                endpoint_m=Vector3(x=1.0, y=0.0, z=1.0),
                hit=False,
                confidence=1.0,
                observed_at_monotonic_seconds=10.0,
            )
        ],
    )
    return frame


# 功能：
#   构造与测试深度帧匹配的健康快照，用于训练来源记录结构验证。
# 输入：
#   无。
# 输出：
#   snapshot：合成传感器状态。
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
#   写出固定语义类别定义并返回实际文件摘要，不授予模型训练资格。
# 输入：
#   path：隔离临时目录中的标签定义路径。
# 输出：
#   digest：实际标签文件的 SHA-256。
def _label_map(path: Path) -> str:
    import hashlib

    path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.vision-label-map.v1",
                "background_class_id": 0,
                "classes": [
                    {"class_id": class_id, "name": name}
                    for class_id, name in enumerate(
                        (
                            "background",
                            "traversable",
                            "obstacle",
                            "doorway",
                            "stairs",
                            "person",
                            "glass",
                            "pickup-marker",
                        )
                    )
                ],
                "qualification_granted": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


# 功能：
#   生成含真实 PNG、语义配对及显式资格状态的合成数据集，测试构建器准入与划分。
# 输入：
#   root、flight_id：测试输出根及合成飞行标识。
#   label_map_sha256：标签定义摘要。
#   value：每张图像的填充值，用于精确控制跨集合重复。
#   status：夹具的资格标记，不是实际飞行成功证据。
#   camera_aligned：视野方向证据；None 表示缺失该契约。
# 输出：
#   None：不返回业务数据。
def _flight(root: Path, flight_id: str, label_map_sha256: str, value: int | tuple[int, ...],
            *, status: str = "verified", camera_aligned: bool | None = True) -> None:
    recorder = RuntimeMultimodalDatasetRecorder(
        root,
        flight_id=flight_id,
        map_sha256="c" * 64,
        maximum_bytes=1024 * 1024,
    )
    values = (value,) if isinstance(value, int) else value
    for index, sample_value in enumerate(values):
        rgb = np.full((18, 32, 3), sample_value, dtype=np.uint8)
        mask = np.full((18, 32), 1, dtype=np.uint8)
        mask[:6, 12:20] = 3 if sample_value < 100 else 4
        recorder.record(
            rgb_png=_png(rgb),
            frame=_frame(),
            sensor_snapshot=_snapshot(),
            recorded_at_unix_ms=1_000 + index,
            recorded_at_monotonic_seconds=10.05 + index,
            rgb_sample_monotonic_seconds=10.0 + index,
            semantic_mask_png=_png(mask),
            semantic_label_map_sha256=label_map_sha256,
            semantic_sample_monotonic_seconds=10.04 + index,
            state=(
                {}
                if camera_aligned is None
                else {
                    "forward_camera_motion_alignment": {
                        "schema_version": "dronedream.forward-camera-motion-alignment.v1",
                        "sensor_id": "oakd-lite-forward-rgb",
                        "camera_forward_world_enu": {"x": 1.0, "y": 0.0, "z": 0.0},
                        "horizontal_speed_mps": 0.2,
                        "minimum_alignment_speed_mps": 0.1,
                        "alignment_required": True,
                        "alignment_cosine": 1.0 if camera_aligned else -1.0,
                        "minimum_alignment_cosine": 0.7071067811865476,
                        "aligned_for_forward_navigation": camera_aligned,
                        "issue_codes": (
                            []
                            if camera_aligned
                            else ["FORWARD_CAMERA_NOT_MOTION_ALIGNED"]
                        ),
                    }
                }
            ),
        )
    summary = recorder.summary()
    (root / "summary.json").write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    (root.parent / "qualification-summary.json").write_text(
        json.dumps(
            {
                "status": status,
                "measurements": {
                    "multimodal_dataset": {
                        "summary": summary,
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


# 功能：
#   失败飞行默认不得进入训练，即使显式允许也不能成为运行验收证据。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_failed_flight_requires_explicit_training_admission(tmp_path: Path) -> None:
    label_map = tmp_path / "vision-label-map.json"
    label_map_sha256 = _label_map(label_map)
    root = tmp_path / "failed-run" / "multimodal-dataset"
    _flight(root, "failed-flight", label_map_sha256, 40, status="failed")

    with pytest.raises(ValueError, match="did not verify"):
        _flight_records(root, allow_failed_source=False)
    _, _, mission = _flight_records(root, allow_failed_source=True)

    assert mission["failed_source_explicitly_allowed"] is True
    assert mission["operational_validation_eligible"] is False


# 功能：
#   通过真实构建子进程验证按整次飞行划分训练／验证集及监督权重、摘要回执。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_builds_hash_verified_complete_flight_vision_splits(tmp_path: Path) -> None:
    label_map = tmp_path / "vision-label-map.json"
    label_map_sha256 = _label_map(label_map)
    first = tmp_path / "run-one" / "multimodal-dataset"
    second = tmp_path / "run-two" / "multimodal-dataset"
    _flight(first, "flight-one", label_map_sha256, 40)
    _flight(second, "flight-two", label_map_sha256, 180)
    output = tmp_path / "compiled"
    script = Path(__file__).parents[1] / "scripts" / "build_local_vision_dataset.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--flight-dataset-root",
            str(first),
            "--flight-dataset-root",
            str(second),
            "--vision-label-map",
            str(label_map),
            "--output-root",
            str(output),
            "--validation-flight-id",
            "flight-two",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    training = [
        LocalVisionTrainingSample.model_validate_json(line)
        for line in (output / "training" / "samples.jsonl").read_text().splitlines()
    ]
    validation = [
        LocalVisionTrainingSample.model_validate_json(line)
        for line in (output / "validation" / "samples.jsonl").read_text().splitlines()
    ]
    assert {sample.flight_id for sample in training} == {"flight-one"}
    assert {sample.flight_id for sample in validation} == {"flight-two"}
    assert training[0].scene_target_weights == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert training[0].quality_target_weights == [1.0, 1.0, 0.0, 0.0]
    assert training[0].source_record_sha256 is not None
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["qualification_granted"] is False
    assert receipt["split_method"] == "held-out-complete-flight"


# 功能：
#   前向导航监督拒绝视野未对齐或缺少方向证据的帧。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_forward_navigation_training_rejects_unaligned_or_unproven_frames(tmp_path: Path) -> None:
    label_map = tmp_path / "vision-label-map.json"
    label_map_sha256 = _label_map(label_map)
    unaligned = tmp_path / "unaligned" / "multimodal-dataset"
    missing = tmp_path / "missing" / "multimodal-dataset"
    _flight(
        unaligned,
        "unaligned-flight",
        label_map_sha256,
        40,
        camera_aligned=False,
    )
    _flight(
        missing,
        "missing-contract-flight",
        label_map_sha256,
        80,
        camera_aligned=None,
    )

    unaligned_records, _, _ = _flight_records(
        unaligned,
        allow_failed_source=False,
    )
    missing_records, _, _ = _flight_records(missing, allow_failed_source=False)

    assert _forward_navigation_frame_eligible(unaligned_records[0]) is False
    with pytest.raises(ValueError, match="no forward-camera motion-alignment contract"):
        _forward_navigation_frame_eligible(missing_records[0])


# 功能：
#   跨飞行重复图像只保留训练副本，避免验证集泄漏且正确记录剔除计数。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_cross_split_duplicate_rgb_is_retained_only_in_training(tmp_path: Path) -> None:
    label_map = tmp_path / "vision-label-map.json"
    label_map_sha256 = _label_map(label_map)
    first = tmp_path / "run-one" / "multimodal-dataset"
    second = tmp_path / "run-two" / "multimodal-dataset"
    _flight(first, "flight-one", label_map_sha256, (40, 60))
    _flight(second, "flight-two", label_map_sha256, (40, 180))
    output = tmp_path / "compiled"
    script = Path(__file__).parents[1] / "scripts" / "build_local_vision_dataset.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--flight-dataset-root",
            str(first),
            "--flight-dataset-root",
            str(second),
            "--vision-label-map",
            str(label_map),
            "--output-root",
            str(output),
            "--validation-flight-id",
            "flight-two",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    training = (output / "training" / "samples.jsonl").read_text().splitlines()
    validation = (output / "validation" / "samples.jsonl").read_text().splitlines()
    receipt = json.loads((output / "receipt.json").read_text())
    second_receipt = next(
        item for item in receipt["source_flights"] if item["flight_id"] == "flight-two"
    )
    assert len(training) == 2
    assert len(validation) == 1
    assert receipt["cross_split_duplicate_rgb_count"] == 1
    assert receipt["cross_split_duplicate_policy"] == (
        "retain-training-exclude-validation"
    )
    assert second_receipt["rejected_cross_split_duplicate_count"] == 1
