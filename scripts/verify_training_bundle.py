"""Verify a transferred training bundle before installing or executing its contents."""

import argparse
import hashlib
import json
import os
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath


# 功能：
#   检查普通文件及每一级父目录，拒绝符号链接、重解析点与过大输入。
# 输入：
#   path、limit：明确路径和最大文件字节数。
# 输出：
#   digest、size：实际读完的文件摘要和长度。
def file_digest(path, limit):
    for candidate in (path, *path.parents):
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("TRAINING_BUNDLE_LINK_NOT_ALLOWED")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("TRAINING_BUNDLE_FILE_BUDGET")
    hasher, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (not os.path.samestat(before, opened) or opened.st_size != before.st_size
                or opened.st_mtime_ns != before.st_mtime_ns):
            raise ValueError("TRAINING_BUNDLE_FILE_CHANGED")
        while chunk := stream.read(1024**2):
            size += len(chunk)
            if size > min(limit, before.st_size):
                raise ValueError("TRAINING_BUNDLE_FILE_CHANGED")
            hasher.update(chunk)
        closed = os.fstat(stream.fileno())
    after = path.stat()
    if (not os.path.samestat(before, after) or before.st_mtime_ns != after.st_mtime_ns
            or not os.path.samestat(opened, closed) or closed.st_size != size
            or closed.st_mtime_ns != opened.st_mtime_ns
            or size != before.st_size or size != after.st_size):
        raise ValueError("TRAINING_BUNDLE_FILE_CHANGED")
    digest = hasher.hexdigest()
    return digest, size


# 功能：
#   有界枚举上传目录全部普通文件，拒绝链接、特殊文件和清单外文件或空目录。
# 输入：
#   root、allowed：上传根目录及清单明确列出的相对文件名。
# 输出：
#   None：目录只包含清单及其必要父目录时通过。
def verify_inventory(root, allowed):
    allowed = {*allowed, "bundle-manifest.json"}
    directories = {parent.as_posix() for name in allowed
                   for parent in PurePosixPath(name).parents if parent.as_posix() != "."}
    pending, seen = [root], set()
    count = 0
    while pending:
        directory = pending.pop()
        info = directory.lstat()
        if (not stat.S_ISDIR(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError("TRAINING_BUNDLE_LINK_NOT_ALLOWED")
        for path in directory.iterdir():
            count += 1
            if count > 5000:
                raise ValueError("TRAINING_BUNDLE_INVENTORY_BUDGET")
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("TRAINING_BUNDLE_LINK_NOT_ALLOWED")
            if stat.S_ISDIR(info.st_mode) and relative in directories:
                pending.append(path)
            elif stat.S_ISREG(info.st_mode) and relative in allowed:
                seen.add(relative)
            else:
                raise ValueError("TRAINING_BUNDLE_UNLISTED_PATH:" + relative)
    if seen != allowed:
        raise ValueError("TRAINING_BUNDLE_INVENTORY_MISMATCH")


# 功能：
#   按本地已保存的清单摘要核对上传文件的完整内容，缺失、越界或篡改立即失败。
# 输入：
#   root、expected：上传根目录及上传前另外保存的清单 SHA-256。
# 输出：
#   report：验证文件数和总字节，不启动安装或任何计费资源。
def verify(root, expected):
    if type(expected) is not str or re.fullmatch(r"[a-f0-9]{64}", expected) is None:
        raise ValueError("TRAINING_BUNDLE_MANIFEST_DIGEST_REQUIRED")
    manifest = root / "bundle-manifest.json"
    digest, _ = file_digest(manifest, 2 * 1024**2)
    if digest != expected:
        raise ValueError("TRAINING_BUNDLE_MANIFEST_CHANGED")
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("TRAINING_BUNDLE_MANIFEST_CHANGED")
    payload = json.loads(raw)
    if (type(payload) is not dict
            or payload.get("schema") != "dronedream.training-transfer-bundle.v1"
            or type(payload.get("files")) is not list or not 1 <= len(payload["files"]) <= 1000):
        raise ValueError("TRAINING_BUNDLE_MANIFEST_INVALID")
    seen, allowed, total = set(), set(), 0
    for entry in payload["files"]:
        if type(entry) is not dict or set(entry) != {"path", "sha256", "size_bytes"}:
            raise ValueError("TRAINING_BUNDLE_ENTRY_INVALID")
        relative = entry.get("path")
        if (type(relative) is not str or "\\" in relative or ":" in relative
                or not relative or relative.startswith("/")
                or any(part in ("", ".", "..") for part in relative.split("/"))
                or relative.casefold() == "bundle-manifest.json"
                or any(part.endswith((".", " ")) for part in relative.split("/"))
                or relative.casefold() in seen or PurePosixPath(relative).as_posix() != relative):
            raise ValueError("TRAINING_BUNDLE_PATH_INVALID")
        seen.add(relative.casefold())
        allowed.add(relative)
        digest, size = file_digest(root / relative, 256 * 1024**2)
        if (digest != entry.get("sha256") or type(entry.get("size_bytes")) is not int
                or size != entry["size_bytes"]):
            raise ValueError("TRAINING_BUNDLE_FILE_MISMATCH:" + relative)
        total += size
        if total > 2 * 1024**3:
            raise ValueError("TRAINING_BUNDLE_TOTAL_BUDGET")
    verify_inventory(root, allowed)
    if file_digest(manifest, 2 * 1024**2)[0] != expected:
        raise ValueError("TRAINING_BUNDLE_MANIFEST_CHANGED")
    report = {"verified": True, "files": len(seen), "bytes": total,
              "manifest_sha256": expected, "gpu_started": False}
    return report


# 功能：
#   核对当前解释器实际安装的 Core 模块与本次已验证 wheel，禁止旧环境配新脚本运行。
# 输入：
#   root：已完成清单与完整目录校验的上传根目录。
# 输出：
#   count：与 wheel 字节完全一致的已安装 Python 模块数量。
def verify_installed_core(root):
    from importlib.metadata import distribution

    wheels = list(root.glob("dronedream_flight_agent_core-*.whl"))
    if len(wheels) != 1:
        raise ValueError("TRAINING_ENVIRONMENT_WHEEL_COUNT_INVALID")
    installed = distribution("dronedream-flight-agent-core")
    expected = set()
    with zipfile.ZipFile(wheels[0]) as archive:
        for entry in archive.infolist():
            if not entry.filename.endswith(".py"):
                continue
            name = PurePosixPath(entry.filename)
            if (name.is_absolute() or ".." in name.parts or "\\" in entry.filename
                    or not name.parts[0].startswith("dronedream_")
                    or entry.file_size > 2 * 1024**2 or len(expected) >= 10_000
                    or entry.filename in expected):
                raise ValueError("TRAINING_ENVIRONMENT_WHEEL_MODULE_INVALID")
            digest = hashlib.sha256(archive.read(entry)).hexdigest()
            if file_digest(Path(installed.locate_file(entry.filename)), 2 * 1024**2)[0] != digest:
                raise ValueError("TRAINING_ENVIRONMENT_CORE_MISMATCH:" + entry.filename)
            expected.add(entry.filename)
    if not expected:
        raise ValueError("TRAINING_ENVIRONMENT_CORE_MISSING")
    # 一个同名旧模块即使不在 wheel 中，也会被 Python 导入；实际安装树同样必须匹配。
    site_root = Path(installed.locate_file(""))
    actual = {path.relative_to(site_root).as_posix()
              for package in {PurePosixPath(name).parts[0] for name in expected}
              for path in (site_root / package).rglob("*.py")}
    if actual != expected:
        raise ValueError("TRAINING_ENVIRONMENT_EXTRA_OR_MISSING_CORE_MODULE")
    count = len(expected)
    return count


# 功能：
#   在没有第三方 Python 依赖时验证上传物，为后续环境安装提供独立入口。
# 输入：
#   命令行参数：bundle 目录和另外保存的清单摘要。
# 输出：
#   exit_code：所有索引文件验证通过为零。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--check-installed", action="store_true")
    args = parser.parse_args()
    root = args.bundle.absolute()
    report = verify(root, args.sha256)
    if args.check_installed:
        report["installed_modules_verified"] = verify_installed_core(root)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
