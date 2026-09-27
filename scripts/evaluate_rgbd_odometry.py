"""Offline RGB-D motion baseline; ground truth is read only after predictions."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
from audit_tartanair_sample import decode_depth, digest, portable_path, verify_files


# 功能：由前后两帧 RGB 和前帧深度估计相对运动；无地图/真值位姿输入。
# 输入：两个彩色图像、米制深度、固定相机内参；输出：PnP 运动及内点数或明确拒绝。
def estimate_motion(previous, current, depth, camera):
    import cv2
    import numpy as np
    orb = cv2.ORB_create(nfeatures=1800)
    first, desc1 = orb.detectAndCompute(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), None)
    second, desc2 = orb.detectAndCompute(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), None)
    if desc1 is None or desc2 is None or len(desc2) < 2:
        return {"accepted": False, "reason": "insufficient-features"}
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc1, desc2, k=2)
    points3d, points2d = [], []
    for pair in matches:
        if len(pair) != 2 or pair[0].distance >= .75 * pair[1].distance:
            continue
        match = pair[0]
        x, y = first[match.queryIdx].pt
        z = float(depth[min(round(y), depth.shape[0]-1), min(round(x), depth.shape[1]-1)])
        if not math.isfinite(z) or not .15 < z < 20:
            continue
        points3d.append(((x-camera[0, 2])*z/camera[0, 0], (y-camera[1, 2])*z/camera[1, 1], z))
        points2d.append(second[match.trainIdx].pt)
    if len(points3d) < 12:
        return {"accepted": False, "reason": "insufficient-depth-matches", "matches": len(points3d)}
    ok, rotation, translation, inliers = cv2.solvePnPRansac(
        np.array(points3d, dtype=np.float64), np.array(points2d, dtype=np.float64), camera, None,
        iterationsCount=100, reprojectionError=2., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 12 or len(inliers)/len(points3d) < .4:
        return {"accepted": False, "reason": "pnp-consensus-insufficient", "matches": len(points3d)}
    return {"accepted": True, "translation_optical_m": translation.ravel().tolist(),
        "rotation_optical_axis_angle": rotation.ravel().tolist(),
        "translation_norm_m": float(np.linalg.norm(translation)),
        "rotation_degrees": float(np.linalg.norm(rotation)*180/math.pi),
        "matches": len(points3d), "inliers": len(inliers)}


# 功能：对保留测试轨迹作无训练运动估计，之后独立读取真值计算尺度/转角误差。
# 输入：经过质量检查的数据集和报告；输出：离线基线，绝不写 Runtime 或飞控。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    cv2.setRNGSeed(24)
    quality = json.loads(args.quality.read_text())
    if quality["receipt_sha256"] != digest(args.dataset / "dataset-receipt.json"):
        raise ValueError("QUALITY_RECEIPT_MISMATCH")
    root = args.dataset / "unpacked"
    verified = verify_files(root, json.loads((args.dataset / "dataset-receipt.json").read_text()))
    camera = np.array([[320., 0., 320.], [0., 320., 320.], [0., 0., 1.]])
    pairs = [p for p in quality["pairs"] if p["split"] == "test"]
    for pair in pairs:
        if any(portable_path(pair[key]).as_posix() not in verified for key in ("rgb", "depth")):
            raise ValueError("QUALITY_REFERENCES_UNVERIFIED_FILE")
    results = []
    for previous, current in zip(pairs, pairs[1:]):
        if previous["trajectory"] != current["trajectory"]:
            continue
        images = [cv2.imread(str(root / portable_path(p["rgb"]))) for p in (previous, current)]
        depth = decode_depth(cv2.imread(str(root / portable_path(previous["depth"])), cv2.IMREAD_UNCHANGED))
        if any(image is None for image in images):
            raise ValueError("RGB_READ_FAILED")
        started = time.perf_counter()
        prediction = estimate_motion(*images, depth, camera)
        results.append({"trajectory": previous["trajectory"], "previous_frame": previous["frame"],
            "current_frame": current["frame"], "latency_ms": (time.perf_counter()-started)*1000, **prediction})
    # 真值只用于预测完成后的评价；不进入特征检测、匹配或 PnP。
    poses = {}
    for row in results:
        name = row["trajectory"]
        if name not in poses:
            relative = portable_path("image_lcam_front/ArchVizTinyHouseDay/Data_easy/" + name + "/pose_lcam_front.txt")
            if relative.as_posix() not in verified:
                raise ValueError("QUALITY_REFERENCES_UNVERIFIED_POSE")
            poses[name] = np.loadtxt(root / relative)
        a, b = poses[name][row["previous_frame"]], poses[name][row["current_frame"]]
        truth_distance = float(np.linalg.norm(b[:3]-a[:3]))
        dot = float(np.dot(a[3:]/np.linalg.norm(a[3:]), b[3:]/np.linalg.norm(b[3:])))
        truth_rotation = math.degrees(2*math.acos(min(1., abs(dot))))
        row.update(truth_distance_m=truth_distance, truth_rotation_degrees=truth_rotation)
        if row["accepted"]:
            row["distance_magnitude_error_m"] = abs(row["translation_norm_m"]-truth_distance)
            row["rotation_magnitude_error_degrees"] = abs(row["rotation_degrees"]-truth_rotation)
    accepted = [r for r in results if r["accepted"]]
    report = {"algorithm": "ORB RGB-D PnP RANSAC", "quality_sha256": digest(args.quality),
        "opencv_version": cv2.__version__, "pairs": len(results), "accepted": len(accepted),
        "p50_ms": statistics.median(r["latency_ms"] for r in results) if results else None,
        "median_distance_magnitude_error_m": statistics.median(r["distance_magnitude_error_m"] for r in accepted) if accepted else None,
        "median_rotation_magnitude_error_degrees": statistics.median(r["rotation_magnitude_error_degrees"] for r in accepted) if accepted else None,
        "flight_authority": False, "formal_training_additions": 0,
        "limitations": "Uses dataset depth, not predicted depth. No IMU fusion, loop closure, accumulated drift or vector frame alignment qualification. Same-building holdout only.",
        "results": results}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))


if __name__ == "__main__":
    main()
