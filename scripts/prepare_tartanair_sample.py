#!/usr/bin/env python3
"""Pinned forward-camera indoor data, retained outside formal training inventory."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

REPOSITORY = "theairlabcmu/tartanair2"
REVISION = "0d2d145e973832742a2aaa04b7d2ebffc8d82817"
PREFIX = "ArchVizTinyHouseDay/Data_easy/"
FILES = tuple(PREFIX + name + ".zip" for name in
              ("image_lcam_front", "depth_lcam_front", "seg_lcam_front", "imu"))


# 功能：验证压缩包路径及展开预算，防止跨目录写入、重复覆盖和解压炸弹。
# 输入：ZIP 成员信息及目标目录。
# 输出：安全目标路径；不允许链接、设备路径或任意可执行代码。
def member_target(member, root):
    path = PurePosixPath(member.filename)
    mode = member.external_attr >> 16
    if (path.is_absolute() or ".." in path.parts or "\\" in member.filename
            or "\\" in member.orig_filename or "\x00" in member.orig_filename
            or ":" in member.filename or stat.S_ISLNK(mode)
            or (not member.is_dir() and path.suffix.lower() not in {".png", ".txt", ".csv", ".json", ".npy", ".yaml"})):
        raise ValueError("DATASET_UNSAFE_ARCHIVE_MEMBER")
    result = root.joinpath(*path.parts).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("DATASET_ARCHIVE_PATH_ESCAPE")
    if member.file_size > 256 * 1024**2:
        raise ValueError("DATASET_MEMBER_TOO_LARGE")
    return result


# 功能：流式计算来源及解压文件摘要。
# 输入：本地文件。
# 输出：SHA256 字符串。
def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


# 功能：下载特定室内前视 RGB/深度/语义/IMU 数据，核验固定版本和 LFS 内容。
# 输入：新缓存目录；数据仅用于实验，尚未赋予控制示范标签。
# 输出：不可覆盖的下载与解压清单；完整保留下载档案。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().dataset_info(REPOSITORY, revision=REVISION, files_metadata=True, token=False)
    selected = {item.rfilename: item for item in info.siblings if item.rfilename in FILES}
    if info.sha != REVISION or set(selected) != set(FILES):
        raise ValueError("DATASET_REVISION_INVENTORY_MISMATCH")
    if any(item.size is None for item in selected.values()) or sum(item.size for item in selected.values()) > 1024**3:
        raise ValueError("DATASET_DOWNLOAD_BUDGET_EXCEEDED")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "dataset-receipt.json"
    unpacked = args.output / "unpacked"
    if receipt_path.exists() or unpacked.exists():
        raise FileExistsError("retained dataset output already exists")
    snapshot_download(REPOSITORY, repo_type="dataset", revision=REVISION,
        allow_patterns=list(FILES), local_dir=args.output / "archives", token=False, max_workers=2)
    receipt = {"repository": REPOSITORY, "revision": REVISION,
        "license": "CC-BY-4.0", "license_source": "https://tartanair.org/",
        "attribution": "TartanAir V2, The AirLab, Carnegie Mellon University",
        "formal_training_eligible": False, "control_demonstration_labels": False,
        "purpose": "indoor forward-view geometry/visual odometry evaluation",
        "archives": {}, "files": []}
    seen, members, total = set(), [], 0
    for name, source in selected.items():
        archive = args.output / "archives" / name
        sha = digest(archive)
        if source.size != archive.stat().st_size or not source.lfs or source.lfs.sha256 != sha:
            raise ValueError("DATASET_CONTENT_MISMATCH:" + name)
        receipt["archives"][name] = {"sha256": sha, "bytes": source.size}
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                # 各模态包可能重复附带同名位姿文件；分包保存，绝不互相覆盖。
                target = member_target(member, unpacked / Path(name).stem)
                if member.is_dir():
                    continue
                relative = target.relative_to(unpacked.resolve()).as_posix()
                if relative.casefold() in seen:
                    raise ValueError("DATASET_ARCHIVE_DUPLICATE_TARGET")
                seen.add(relative.casefold())
                total += member.file_size
                if total > 4 * 1024**3 or len(seen) > 100_000:
                    raise ValueError("DATASET_UNPACK_BUDGET_EXCEEDED")
                members.append((archive, member.filename, target))
    unpacked.mkdir()
    # 每个成员使用独占文件创建，不执行外部 unzip，也不信任压缩包中的路径。
    for archive in selected:
        with zipfile.ZipFile(args.output / "archives" / archive) as bundle:
            for origin, name, target in members:
                if origin.name != Path(archive).name:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, target.open("xb") as destination:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
                receipt["files"].append({"path": target.relative_to(unpacked.resolve()).as_posix(),
                    "bytes": target.stat().st_size, "sha256": digest(target)})
    receipt["rgb_images"] = sum("image_lcam_front" in r["path"] and r["path"].endswith(".png")
                                for r in receipt["files"])
    receipt["file_count"] = len(receipt["files"])
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({"rgb_images": receipt["rgb_images"], "file_count": receipt["file_count"],
                      "formal_training_additions": 0}), flush=True)


if __name__ == "__main__":
    main()
