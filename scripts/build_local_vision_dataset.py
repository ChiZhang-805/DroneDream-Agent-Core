#!/usr/bin/env python3
"""Compile hash-bound complete-flight RGB supervision for local vision training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dronedream_agent_core.asset_package_storage import publish_asset_file
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_vision_training import (
    LOCAL_VISION_SEMANTIC_CLASSES,
    LocalVisionTrainingSample,
    load_local_vision_label_map,
)
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    portable_plugin_path,
    read_plugin_file,
)
from dronedream_agent_core.runtime_control_io import publish_runtime_json
from dronedream_agent_core.runtime_multimodal_dataset import (
    RuntimeMultimodalDatasetRecord,
)
from dronedream_plugin_sdk.protocol import MAX_MESSAGE_BYTES, decode_json, encode_json

MAX_FLIGHT_RECORD_BYTES = 256 * 1024**2
MAX_FLIGHT_RECORDS = 50_000
MAX_TOTAL_RECORDS = 100_000
MAX_ARTIFACT_BYTES = 16 * 1024**2


# 功能：
#   通过共享普通文件读取器流式计算摘要，限制文件大小并检查读取期间替换。
# 输入：
#   path：明确的本地制品路径。
#   limit：允许读取的最大字节数。
# 输出：
#   digest：本次实际读取内容的摘要。
def _sha256(path: Path, *, limit: int = MAX_FLIGHT_RECORD_BYTES) -> str:
    digest = hash_plugin_file(path, limit=limit)
    return digest


# 功能：
#   从同一批有界字节解析 JSON 并计算摘要，拒绝重复键、非法数值和非对象回执。
# 输入：
#   path：需要绑定的摘要或任务回执文件。
# 输出：
#   payload、digest：独立对象及其实际来源字节摘要。
def _read_object_snapshot(path: Path) -> tuple[dict, str]:
    raw = read_plugin_file(path, limit=MAX_MESSAGE_BYTES)
    payload = decode_json(raw)
    if type(payload) is not dict:
        raise ValueError("vision dataset receipt is not an object")
    digest = hashlib.sha256(raw).hexdigest()
    return payload, digest


# 功能：
#   在保留原路径的情况下检查包内普通制品与摘要，不先解析掉链接痕迹。
# 输入：
#   root：当前飞行数据集目录。
#   relative_path：规范包内路径。
#   expected_sha256：记录所绑定的内容摘要。
# 输出：
#   path：完成点时摘要检查的制品路径，后续复制／解码仍须独立绑定实际字节。
def _resolved_artifact(root: Path, relative_path: str, expected_sha256: str) -> Path:
    root = root.absolute()
    path = root / portable_plugin_path(relative_path)
    check_plain_plugin_path(path)
    if root not in path.parents or not path.is_file():
        raise ValueError("multimodal artifact escapes its flight dataset root")
    if _sha256(path, limit=MAX_ARTIFACT_BYTES) != expected_sha256:
        raise ValueError("multimodal artifact content hash mismatch")
    return path


# 功能：
#   拒绝显式持久化失败或未完整关闭的摘要；旧摘要省略这些新字段不等于伪造完整标记。
# 输入：
#   summary：类型已经确认为对象的记录摘要。
# 输出：
#   None：不返回业务数据。
def _require_complete_recording(summary: dict) -> None:
    if (summary.get("issue_code") is not None or summary.get("writer_issue") is not None
            or any(key in summary and summary[key] is not True
                   for key in ("writer_complete", "bytes_accounting_complete"))):
        raise ValueError("multimodal recording is incomplete or failed")


# 功能：
#   将记录链绑定到同一次任务回执，失败来源只有显式允许时可用于恢复训练而非验证集。
# 输入：
#   root：当前飞行数据集目录。
#   dataset_summary：已验证的记录链摘要。
#   allow_failed_source：是否明确允许失败任务进入训练。
# 输出：
#   mission：任务状态、实际回执摘要和训练／验证准入范围。
def _source_mission_receipt(root: Path, *, dataset_summary: dict[str, Any],
                            allow_failed_source: bool) -> dict[str, Any]:
    if type(allow_failed_source) is not bool:
        raise ValueError("failed source admission requires an explicit boolean")

    path = root.parent / "qualification-summary.json"
    if not path.is_file():
        raise ValueError("multimodal dataset has no source mission receipt")
    payload, receipt_sha256 = _read_object_snapshot(path)
    status = payload.get("status")
    if type(status) is not str or not 0 < len(status) <= 64:
        raise ValueError("source mission status is invalid")
    measurements = payload.get("measurements")
    multimodal = measurements.get("multimodal_dataset") if isinstance(measurements, dict) else None
    bound_summary = multimodal.get("summary") if isinstance(multimodal, dict) else None
    binding_keys = (
        "flight_id",
        "map_sha256",
        "record_count",
        "latest_record_sha256",
        "records_sha256",
    )
    if not isinstance(bound_summary, dict) or any(
        type(bound_summary.get(key)) is not type(dataset_summary.get(key))
        or bound_summary.get(key) != dataset_summary.get(key) for key in binding_keys
    ):
        raise ValueError("source mission receipt does not bind the sensor record chain")
    _require_complete_recording(bound_summary)
    if status != "verified" and not allow_failed_source:
        raise ValueError(
            "source mission did not verify; pass --allow-failed-source only for "
            "explicit failure/recovery training"
        )
    mission = {
        "status": status,
        "receipt_sha256": receipt_sha256,
        "failed_source_explicitly_allowed": status != "verified",
        "operational_validation_eligible": status == "verified",
    }
    return mission


# 功能：
#   有界读取完整飞行记录，以同一流同时解析与散列，验证链顺序、记录状态及任务绑定。
# 输入：
#   root：已结束采集的飞行数据集目录。
#   allow_failed_source：允许失败任务用于训练的显式标记。
# 输出：
#   records、summary、mission：已验证的记录序列、数据集摘要和任务来源回执。
def _flight_records(root: Path, *, allow_failed_source: bool,
                   ) -> tuple[list[RuntimeMultimodalDatasetRecord], dict[str, Any], dict[str, Any]]:
    if type(allow_failed_source) is not bool:
        raise ValueError("failed source admission requires an explicit boolean")
    root = root.absolute()
    records_path = root / "records.jsonl"
    summary_path = root / "summary.json"
    check_plain_plugin_path(records_path)
    before = records_path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FLIGHT_RECORD_BYTES:
        raise ValueError("multimodal flight records are unavailable or oversized")
    records: list[RuntimeMultimodalDatasetRecord] = []
    previous = "0" * 64
    records_digest = hashlib.sha256()
    total = 0
    sample_ids = set()
    previous_time = -1.0
    with records_path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if not os.path.samestat(before, opened) or opened.st_size != before.st_size:
            raise ValueError("multimodal records changed before opening")
        line_number = 0
        # readline 本身携带上限，而不是先分配整条超大行才检查 len。
        while line := handle.readline(MAX_MESSAGE_BYTES + 1):
            line_number += 1
            total += len(line)
            if total > min(before.st_size, MAX_FLIGHT_RECORD_BYTES):
                raise ValueError("multimodal records changed or exceed the byte budget")
            if len(line) > MAX_MESSAGE_BYTES:
                raise ValueError("multimodal record exceeds the bounded line size")
            if not line.endswith(b"\n"):
                raise ValueError("multimodal final record is incomplete")
            records_digest.update(line)
            if not line.strip():
                continue
            if len(records) >= MAX_FLIGHT_RECORDS:
                raise ValueError("multimodal flight exceeds the record count budget")
            record = RuntimeMultimodalDatasetRecord.model_validate(decode_json(line))
            payload = record.model_dump(mode="json")
            observed_hash = payload.pop("record_sha256")
            if observed_hash != sha256_json(payload):
                raise ValueError(f"multimodal record hash mismatch at line {line_number}")
            if record.previous_record_sha256 != previous:
                raise ValueError(f"multimodal record chain mismatch at line {line_number}")
            if record.semantic_mask_relative_path is None:
                raise ValueError("vision training requires semantic supervision on every record")
            if (record.sample_id in sample_ids
                    or record.recorded_at_monotonic_seconds <= previous_time):
                raise ValueError("multimodal records repeat samples or reverse source order")
            records.append(record)
            sample_ids.add(record.sample_id)
            previous_time = record.recorded_at_monotonic_seconds
            previous = record.record_sha256
        after = os.fstat(handle.fileno())
    check_plain_plugin_path(records_path)
    if (total != opened.st_size or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or not os.path.samestat(after, records_path.stat())):
        raise ValueError("multimodal records changed while reading")
    if not records:
        raise ValueError("multimodal flight dataset is empty")
    summary, summary_sha256 = _read_object_snapshot(summary_path)
    _require_complete_recording(summary)
    expected = {
        "flight_id": records[0].flight_id,
        "map_sha256": records[0].map_sha256,
        "record_count": len(records),
        "latest_record_sha256": previous,
        "records_sha256": records_digest.hexdigest(),
        "qualification_granted": False,
    }
    if any(type(summary.get(key)) is not type(value) or summary.get(key) != value
           for key, value in expected.items()):
        raise ValueError("multimodal flight summary does not bind the record chain")
    if any(record.flight_id != records[0].flight_id for record in records):
        raise ValueError("multimodal records mix flight identities")
    if any(record.map_sha256 != records[0].map_sha256 for record in records):
        raise ValueError("multimodal records mix map identities")
    mission = _source_mission_receipt(
        root,
        dataset_summary=summary,
        allow_failed_source=allow_failed_source,
    )
    mission["dataset_summary_sha256"] = summary_sha256
    return records, summary, mission


# 功能：
#   按预期摘要复制到独占暂存后发布；同内容已存在时去重，冲突保留且拒绝覆盖。
# 输入：
#   source、destination：已验证源路径与编译输出路径。
#   expected_sha256：该图像或标签文件的预期摘要。
# 输出：
#   None：不返回业务数据。
def _copy_content(source: Path, destination: Path, expected_sha256: str) -> None:
    check_plain_plugin_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination, limit=MAX_ARTIFACT_BYTES) != expected_sha256:
            raise ValueError("compiled vision dataset content collision")
        return
    publish_asset_file(source, destination, expected_sha256=expected_sha256,
                       limit=MAX_ARTIFACT_BYTES)


# 功能：
#   检查显式相机朝向契约，缺失或非布尔判断不得混入前向导航监督。
# 输入：
#   record：已验证来源链的单帧记录。
# 输出：
#   eligible：无需水平对齐或明确已经对齐时为 True。
def _forward_navigation_frame_eligible(record: RuntimeMultimodalDatasetRecord) -> bool:
    raw = record.state.get("forward_camera_motion_alignment")
    if not isinstance(raw, dict) or raw.get("schema_version") != (
        "dronedream.forward-camera-motion-alignment.v1"
    ):
        raise ValueError("vision record has no forward-camera motion-alignment contract")
    required = raw.get("alignment_required")
    aligned = raw.get("aligned_for_forward_navigation")
    if not isinstance(required, bool) or not isinstance(aligned, bool):
        raise ValueError("vision record camera alignment verdict is invalid")
    eligible = not required or aligned
    return eligible


# 功能：
#   从摘要绑定的 PNG 字节计算像素监督，限制尺寸并保留未知任务标签的零权重。
# 输入：
#   image_path、mask_path：RGB 图像和单通道类别掩码路径。
#   image_sha256、mask_sha256：生产构建必须提供的来源摘要；省略仅作独立计算。
# 输出：
#   targets：可通行比例、对象存在／曝光目标、有效监督权重及类别像素比例。
def _targets(image_path: Path, mask_path: Path, *, image_sha256: str | None = None,
             mask_sha256: str | None = None) -> dict[str, object]:
    mask_raw = read_plugin_file(mask_path, limit=4 * 1024**2)
    rgb_raw = read_plugin_file(image_path, limit=MAX_ARTIFACT_BYTES)
    if ((mask_sha256 is not None and hashlib.sha256(mask_raw).hexdigest() != mask_sha256)
            or (image_sha256 is not None and hashlib.sha256(rgb_raw).hexdigest() != image_sha256)):
        raise ValueError("decoded vision artifact hash mismatch")
    with BytesIO(mask_raw) as stream, Image.open(stream, formats=["PNG"]) as image:
        if (not 1 <= image.width <= 4096 or not 1 <= image.height <= 2160
                or getattr(image, "n_frames", 1) != 1):
            raise ValueError("semantic mask dimensions or frame count are invalid")
        if image.mode not in {"L", "P"}:
            raise ValueError("semantic mask mode must preserve integer class IDs")
        # 调色板 P 模式保留类别索引，不能转成灰度后把调色板亮度误认为类别。
        mask = np.array(image, dtype=np.uint8)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("semantic mask is empty")
    class_ids = set(int(value) for value in np.unique(mask))
    if class_ids - set(range(len(LOCAL_VISION_SEMANTIC_CLASSES))):
        raise ValueError("semantic mask contains an unsupported class")
    height, width = mask.shape
    left = int(width * 0.2)
    right = max(left + 1, int(width * 0.8))
    bottom_center = mask[int(height * 0.55):, left:right]
    traversability = float(np.isin(bottom_center, (1, 7)).mean())
    ratios = {
        class_id: float((mask == class_id).mean())
        for class_id in range(len(LOCAL_VISION_SEMANTIC_CLASSES))
    }
    with BytesIO(rgb_raw) as stream, Image.open(stream, formats=["PNG"]) as image:
        if (not 1 <= image.width <= 4096 or not 1 <= image.height <= 2160
                or getattr(image, "n_frames", 1) != 1):
            raise ValueError("RGB dimensions or frame count are invalid")
        if image.width * height != image.height * width:
            raise ValueError("RGB and semantic mask aspect ratios do not match")
        if image.mode != "RGB":
            raise ValueError("RGB mode must have three visible color channels")
        rgb = np.array(image, dtype=np.float32)
    luminance = (
        rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
    )
    underexposed = float((luminance < 16.0).mean() >= 0.6)
    overexposed = float((luminance > 239.0).mean() >= 0.6)
    scene_targets = [
        0.0,
        0.0,
        float(ratios[3] >= 0.005),
        float(ratios[4] >= 0.005),
        float(ratios[5] >= 0.002),
        float(ratios[6] >= 0.005),
    ]
    targets = {
        "traversability_target": traversability,
        "scene_targets": scene_targets,
        # Indoor/outdoor cannot be inferred from this eight-class forward
        # mask alone; the four object-presence labels are exact simulator truth.
        "scene_target_weights": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "quality_targets": [underexposed, overexposed, 0.0, 0.0],
        # Blur and physical lens occlusion require separate corruption truth.
        "quality_target_weights": [1.0, 1.0, 0.0, 0.0],
        "class_pixel_ratios": ratios,
    }
    return targets


# 功能：
#   按整次飞行确定可重现的留出集合，拒绝重复标识或造成空训练集的指定划分。
# 输入：
#   flight_ids：不重复的全部飞行标识。
#   requested：用户显式指定的验证飞行标识，空列表表示自动划分。
#   fraction：自动留出比例，允许范围 0.05 至 0.5。
# 输出：
#   result：用于验证而不进入训练的飞行标识集合。
def _validation_flights(flight_ids: list[str], requested: list[str], fraction: float) -> set[str]:
    if (type(fraction) not in (int, float) or not 0.05 <= fraction <= 0.5
            or type(flight_ids) is not list or type(requested) is not list
            or any(type(value) is not str or not value for value in (*flight_ids, *requested))
            or len(set(flight_ids)) != len(flight_ids) or len(set(requested)) != len(requested)):
        raise ValueError("complete-flight split inputs are invalid")
    if len(flight_ids) < 2:
        raise ValueError("complete-flight validation requires at least two flights")
    if requested:
        result = set(requested)
        if not result.issubset(flight_ids) or len(result) == len(flight_ids):
            raise ValueError("requested validation flights do not define a proper split")
        return result
    count = min(len(flight_ids) - 1, max(1, round(len(flight_ids) * fraction)))
    ranked = sorted(
        flight_ids,
        key=lambda flight_id: hashlib.sha256(
            f"complete-flight-split:{flight_id}".encode()
        ).hexdigest(),
    )
    result = set(ranked[:count])
    return result


# 功能：
#   1. 验证完整来源链及类别定义，按整次飞行划分并移除跨集合重复图像。
#   2. 从同摘要图像生成监督，独占输出样本；最终来源复核成功后才发布构建回执。
# 输入：
#   无：命令行提供来源根、标签文件、输出根和验证划分选项。
# 输出：
#   exit_code：构建成功返回 0；异常不发布成功回执、不删除失败现场。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight-dataset-root", type=Path, action="append", required=True)
    parser.add_argument("--vision-label-map", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-flight-id", action="append", default=[])
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--allow-failed-source",
        action="store_true",
        help="admit failed flights only as explicit failure/recovery training evidence",
    )
    args = parser.parse_args()
    if not 0.05 <= args.validation_fraction <= 0.5:
        parser.error("--validation-fraction must be between 0.05 and 0.5")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    check_plain_plugin_path(args.output_root.absolute())
    label_map_sha256, allowed_class_ids = load_local_vision_label_map(
        args.vision_label_map
    )
    if allowed_class_ids != frozenset(range(len(LOCAL_VISION_SEMANTIC_CLASSES))):
        raise ValueError("vision label map does not cover every model class")

    flights: dict[
        str,
        tuple[Path, list[RuntimeMultimodalDatasetRecord], dict[str, Any], dict[str, Any]],
    ] = {}
    if len(args.flight_dataset_root) > 64:
        raise ValueError("vision compilation exceeds the flight count budget")
    total_records = 0
    total_record_bytes = 0
    for supplied_root in args.flight_dataset_root:
        root = supplied_root.absolute()
        check_plain_plugin_path(root)
        records, summary, mission = _flight_records(
            root,
            allow_failed_source=args.allow_failed_source,
        )
        total_records += len(records)
        total_record_bytes += (root / "records.jsonl").stat().st_size
        if total_records > MAX_TOTAL_RECORDS or total_record_bytes > 2 * MAX_FLIGHT_RECORD_BYTES:
            raise ValueError("vision compilation exceeds the combined record budget")
        flight_id = records[0].flight_id
        if flight_id in flights:
            raise ValueError("duplicate multimodal flight identity")
        if any(record.semantic_label_map_sha256 != label_map_sha256 for record in records):
            raise ValueError("flight semantic supervision uses a different label map")
        flights[flight_id] = (root, records, summary, mission)
    validation_flights = _validation_flights(
        sorted(flights),
        args.validation_flight_id,
        args.validation_fraction,
    )
    args.output_root.mkdir(parents=True)
    _copy_content(
        args.vision_label_map.absolute(),
        args.output_root / "vision-label-map.json",
        label_map_sha256,
    )

    samples_by_split: dict[str, list[LocalVisionTrainingSample]] = {
        "training": [],
        "validation": [],
    }
    split_image_masks: dict[str, dict[str, set[str]]] = {
        "training": {},
        "validation": {},
    }
    for flight_id in sorted(flights):
        _, records, _, mission = flights[flight_id]
        split = "validation" if flight_id in validation_flights else "training"
        if split == "validation" and not mission["operational_validation_eligible"]:
            raise ValueError("failed source flight cannot be used as validation evidence")
        for record in records:
            if not _forward_navigation_frame_eligible(record):
                continue
            if record.semantic_mask_sha256 is None:
                raise ValueError("semantic record binding unexpectedly disappeared")
            split_image_masks[split].setdefault(record.rgb_sha256, set()).add(
                record.semantic_mask_sha256
            )
    cross_split_duplicate_rgb = set(split_image_masks["training"]) & set(
        split_image_masks["validation"]
    )
    for image_sha256 in cross_split_duplicate_rgb:
        semantic_masks = (
            split_image_masks["training"][image_sha256]
            | split_image_masks["validation"][image_sha256]
        )
        if len(semantic_masks) != 1:
            raise ValueError("identical RGB content is bound to conflicting semantic masks")

    image_to_split: dict[str, str] = {}
    image_to_mask: dict[str, str] = {}
    flight_receipts = []
    class_pixel_totals = {str(class_id): 0.0 for class_id in allowed_class_ids}
    for flight_id in sorted(flights):
        root, records, summary, mission = flights[flight_id]
        split = "validation" if flight_id in validation_flights else "training"
        if split == "validation" and not mission["operational_validation_eligible"]:
            raise ValueError("failed source flight cannot be used as validation evidence")
        unique_pairs: set[tuple[str, str]] = set()
        rejected_motion_unaligned_count = 0
        rejected_cross_split_duplicate_count = 0
        compiled_unique_sample_count = 0
        for record in records:
            if not _forward_navigation_frame_eligible(record):
                rejected_motion_unaligned_count += 1
                continue
            if record.semantic_mask_relative_path is None or record.semantic_mask_sha256 is None:
                raise ValueError("semantic record binding unexpectedly disappeared")
            pair = (record.rgb_sha256, record.semantic_mask_sha256)
            if pair in unique_pairs:
                continue
            unique_pairs.add(pair)
            if (
                split == "validation"
                and record.rgb_sha256 in cross_split_duplicate_rgb
            ):
                rejected_cross_split_duplicate_count += 1
                continue
            previous_split = image_to_split.get(record.rgb_sha256)
            if previous_split is not None and previous_split != split:
                raise ValueError("RGB content overlaps complete-flight dataset splits")
            previous_mask = image_to_mask.get(record.rgb_sha256)
            if previous_mask is not None and previous_mask != record.semantic_mask_sha256:
                raise ValueError("identical RGB content is bound to conflicting semantic masks")
            image_to_split[record.rgb_sha256] = split
            image_to_mask[record.rgb_sha256] = record.semantic_mask_sha256
            image_path = _resolved_artifact(
                root, record.rgb_relative_path, record.rgb_sha256
            )
            mask_path = _resolved_artifact(
                root,
                record.semantic_mask_relative_path,
                record.semantic_mask_sha256,
            )
            targets = _targets(image_path, mask_path, image_sha256=record.rgb_sha256,
                               mask_sha256=record.semantic_mask_sha256)
            for class_id, ratio in targets.pop("class_pixel_ratios").items():
                class_pixel_totals[str(class_id)] += float(ratio)
            image_relative_path = f"rgb/{record.rgb_sha256}.png"
            mask_relative_path = f"semantic/{record.semantic_mask_sha256}.png"
            split_root = args.output_root / split
            _copy_content(
                image_path,
                split_root / image_relative_path,
                record.rgb_sha256,
            )
            _copy_content(
                mask_path,
                split_root / mask_relative_path,
                record.semantic_mask_sha256,
            )
            samples_by_split[split].append(
                LocalVisionTrainingSample(
                    flight_id=flight_id,
                    map_sha256=record.map_sha256,
                    image_relative_path=image_relative_path,
                    image_sha256=record.rgb_sha256,
                    semantic_mask_relative_path=mask_relative_path,
                    semantic_mask_sha256=record.semantic_mask_sha256,
                    source_record_sha256=record.record_sha256,
                    source_sensor_snapshot_sha256=sha256_json(record.sensor_snapshot),
                    rgb_semantic_time_offset_seconds=(
                        record.rgb_semantic_time_offset_seconds
                    ),
                    **targets,
                )
            )
            compiled_unique_sample_count += 1
        flight_receipts.append(
            {
                "flight_id": flight_id,
                "split": split,
                "source_root": str(root),
                "source_summary": summary,
                "source_mission": mission,
                "source_record_count": len(records),
                "source_unique_pair_count": len(unique_pairs),
                "compiled_unique_sample_count": compiled_unique_sample_count,
                "rejected_motion_unaligned_count": rejected_motion_unaligned_count,
                "rejected_cross_split_duplicate_count": (
                    rejected_cross_split_duplicate_count
                ),
            }
        )
    if any(len(samples) == 0 for samples in samples_by_split.values()):
        raise ValueError("compiled vision split is empty")
    for split, samples in samples_by_split.items():
        path = args.output_root / split / "samples.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        check_plain_plugin_path(path)
        # 流式独占写出，失败文件不冒充完整数据集；成功回执只在最后发布。
        with path.open("xb") as stream:
            for sample in samples:
                encoded = (encode_json(sample.model_dump(mode="json")) + "\n").encode("utf-8")
                if stream.write(encoded) != len(encoded):
                    raise OSError("vision sample short write")
            stream.flush()
            os.fsync(stream.fileno())
    # 复核来源没有在解码、去重和复制期间被更改，避免把不同采样批次混进同一构建回执。
    for root, _, summary, mission in flights.values():
        for path, expected, limit in (
            (root / "records.jsonl", summary["records_sha256"], MAX_FLIGHT_RECORD_BYTES),
            (root / "summary.json", mission["dataset_summary_sha256"], MAX_MESSAGE_BYTES),
            (root.parent / "qualification-summary.json", mission["receipt_sha256"],
             MAX_MESSAGE_BYTES),
        ):
            if _sha256(path, limit=limit) != expected:
                raise ValueError("vision source changed during compilation")
    if _sha256(args.vision_label_map, limit=64 * 1024) != label_map_sha256:
        raise ValueError("vision label map changed during compilation")
    receipt = {
        "schema_version": "dronedream.local-vision-dataset.v1",
        "vision_label_map_sha256": label_map_sha256,
        "split_method": "held-out-complete-flight",
        "training_flight_ids": sorted(set(flights) - validation_flights),
        "validation_flight_ids": sorted(validation_flights),
        "training_sample_count": len(samples_by_split["training"]),
        "validation_sample_count": len(samples_by_split["validation"]),
        "cross_split_duplicate_rgb_count": len(cross_split_duplicate_rgb),
        "cross_split_duplicate_policy": (
            "retain-training-exclude-validation"
        ),
        "class_pixel_ratio_sums": class_pixel_totals,
        "source_flights": flight_receipts,
        "supervision_policy": {
            "indoor_outdoor": "unknown-zero-weight",
            "blur_occlusion": "unknown-zero-weight",
            "object_presence": "semantic-mask-derived",
            "traversability": "bottom-center-semantic-mask-derived",
            "exposure": "rgb-luminance-derived",
        },
        "qualification_granted": False,
    }
    publish_runtime_json(args.output_root / "receipt.json", receipt, replace_existing=False,
                         maximum_bytes=16 * 1024**2)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    exit_code = 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
