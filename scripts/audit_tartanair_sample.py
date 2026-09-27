"""Validate paired forward RGB-D data without promoting it to control training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


# 功能：逐块摘要以核对下载回执，避免信任被修改后的外部数据。
# 输入：单个本地文件；输出：SHA256。
def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


# 功能：按官方 PNG 字节格式还原米制深度，不将 RGBA 颜色解释为距离。
# 输入：OpenCV 读取的 BGRA uint8 数组；输出：二维 float32 米制数组。
def decode_depth(raw):
    import numpy as np
    if raw is None or raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != 4:
        raise ValueError("DEPTH_ENCODING_INVALID")
    return np.ascontiguousarray(raw).view("<f4").squeeze(-1)


# 功能：兼容历史 Windows 清单并拒绝跨目录引用，保证 Linux 上传后仍能定位文件。
# 输入：清单相对路径；输出：平台无关的相对 Path。
def portable_path(value):
    from pathlib import PurePosixPath
    if not isinstance(value, str) or not value or ":" in value or "\x00" in value:
        raise ValueError("DATASET_MANIFEST_PATH_INVALID")
    normalized = PurePosixPath(value.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError("DATASET_MANIFEST_PATH_INVALID")
    return Path(*normalized.parts)


# 功能：重新校验实际数据而非只信任旧质量报告，拒绝链接越界、重复清单和篡改内容。
# 输入：解压根、下载回执；输出：已校验的规范相对路径集合。
def verify_files(root, receipt):
    names = set()
    for record in receipt["files"]:
        relative = portable_path(record["path"])
        path = (root / relative).resolve()
        name = relative.as_posix()
        if (name in names or not path.is_relative_to(root.resolve())
                or not path.is_file() or path.stat().st_size != record["bytes"]
                or digest(path) != record["sha256"]):
            raise ValueError("DATASET_FILE_INTEGRITY_FAILED")
        names.add(name)
    return names


# 功能：核对所有文件摘要、模态配对、相机位姿、时间轴和有效深度。
# 输入：已下载缓存根及新报告路径；输出：独立轨迹划分及质量报告，不生成飞控标签。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    root = args.dataset / "unpacked"
    receipt_path = args.dataset / "dataset-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    semantic_root = args.dataset / "semantic-names"
    semantic_classes = None
    if (semantic_root / "semantic-receipt.json").is_file():
        semantic_receipt = json.loads((semantic_root / "semantic-receipt.json").read_text())
        semantic_file = semantic_root / "seg_label_map.json"
        if digest(semantic_file) != semantic_receipt["mapping_sha256"]:
            raise ValueError("SEMANTIC_MAPPING_HASH_MISMATCH")
        semantic_classes = json.loads(semantic_file.read_text())["name_map"]
        if (not isinstance(semantic_classes, dict)
                or any(not isinstance(name, str) or type(label) is not int or not 0 <= label <= 255
                       for name, label in semantic_classes.items())
                or len(set(semantic_classes.values())) != len(semantic_classes)):
            raise ValueError("SEMANTIC_MAPPING_INVALID")
    verify_files(root, receipt)
    prefix = Path("ArchVizTinyHouseDay/Data_easy")
    trajectories = sorted((root / "image_lcam_front" / prefix).glob("P*"))
    report = {"receipt_sha256": digest(receipt_path), "license": receipt["license"],
        "attribution": receipt["attribution"], "source": "https://tartanair.org/modalities.html",
        "intrinsics": {"width": 640, "height": 640, "fx": 320, "fy": 320, "cx": 320, "cy": 320},
        "pose_frame": "NED", "formal_training_additions": 0,
        "control_labels_available": False, "semantic_class_mapping_available": semantic_classes is not None,
        "semantic_classes": semantic_classes,
        "trajectories": [], "pairs": []}
    for index, trajectory in enumerate(trajectories):
        images = sorted((trajectory / "image_lcam_front").glob("*.png"))
        pose = np.loadtxt(trajectory / "pose_lcam_front.txt", ndmin=2)
        if pose.shape != (len(images), 7) or not np.isfinite(pose).all() or not np.allclose(np.linalg.norm(pose[:, 3:], axis=1), 1, atol=.002):
            raise ValueError("POSE_IMAGE_ALIGNMENT_INVALID:" + trajectory.name)
        imu = root / "imu" / prefix / trajectory.name / "imu"
        cam_time = np.load(imu / "cam_time.npy", allow_pickle=False)
        imu_time = np.load(imu / "imu_time.npy", allow_pickle=False)
        if (cam_time.shape != (len(images),) or imu_time.ndim != 1
                or not np.isfinite(cam_time).all() or not np.isfinite(imu_time).all()
                or not np.allclose(np.diff(cam_time), .1) or not np.allclose(np.diff(imu_time), .01)):
            raise ValueError("IMU_CAMERA_TIMELINE_INVALID:" + trajectory.name)
        for modality in ("acc", "gyro"):
            values = np.load(imu / (modality + ".npy"), allow_pickle=False)
            if values.shape != (len(imu_time), 3) or not np.isfinite(values).all():
                raise ValueError("IMU_VALUES_INVALID")
        # 整段轨迹拆分；本小样本同一房屋，不宣称跨建筑独立泛化。
        split = "test" if index == len(trajectories)-1 else "validation" if index == len(trajectories)-2 else "train"
        depth_fractions, low_texture, duplicates, labels = [], 0, 0, Counter()
        previous_digest = None
        for frame_index, image_path in enumerate(images):
            stem = image_path.name.removesuffix(".png")
            depth_path = root / "depth_lcam_front" / prefix / trajectory.name / "depth_lcam_front" / (stem + "_depth.png")
            seg_path = root / "seg_lcam_front" / prefix / trajectory.name / "seg_lcam_front" / (stem + "_seg.png")
            rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            depth = decode_depth(cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED))
            seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or rgb.shape != (640, 640, 3) or depth.shape != (640, 640) or seg is None or seg.shape != (640, 640):
                raise ValueError("PAIRED_IMAGE_SHAPE_INVALID")
            fraction = float((np.isfinite(depth) & (depth > .05) & (depth < 40)).mean())
            depth_fractions.append(fraction)
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            low_texture += float(cv2.Laplacian(gray, cv2.CV_32F).var()) < 15
            current_digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            duplicates += current_digest == previous_digest
            previous_digest = current_digest
            unique, counts = np.unique(seg, return_counts=True)
            if semantic_classes is not None and not set(map(int, unique)).issubset(set(semantic_classes.values()) | {0}):
                raise ValueError("SEMANTIC_LABEL_NOT_IN_OFFICIAL_MAPPING")
            labels.update({int(k): int(v) for k, v in zip(unique, counts)})
            report["pairs"].append({"trajectory": trajectory.name, "split": split,
                "frame": frame_index, "rgb": image_path.relative_to(root).as_posix(),
                "depth": depth_path.relative_to(root).as_posix(), "seg": seg_path.relative_to(root).as_posix(),
                "timestamp_seconds": float(cam_time[frame_index]), "valid_depth_fraction": fraction,
                "imu_bracketed": bool(imu_time[0] <= cam_time[frame_index] <= imu_time[-1]),
                "control_training_eligible": False})
        report["trajectories"].append({"id": trajectory.name, "split": split, "rgb_count": len(images),
            "imu_samples": len(imu_time), "valid_depth_fraction_min": min(depth_fractions),
            "low_texture_frames": int(low_texture), "adjacent_exact_duplicates": int(duplicates),
            "semantic_pixel_counts": dict(labels)})
    report["rgb_count"] = len(report["pairs"])
    report["warning"] = "Same-building trajectory holdout only. IMU is synthetically derived. Missing end brackets are not extrapolated."
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"rgb_count": report["rgb_count"], "trajectories": report["trajectories"]}))


if __name__ == "__main__":
    main()
