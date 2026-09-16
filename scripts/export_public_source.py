"""Create a reviewable source-only snapshot without exporting private Git history."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
from pathlib import Path, PurePosixPath

SOURCE_ROOTS = frozenset({
    ".github", "app", "docs", "examples", "native", "official_plugins", "ros_ws",
    "runtime", "schemas", "scripts", "shared", "src", "tests",
})
ROOT_FILES = frozenset({
    ".env.example", ".gitattributes", ".gitignore", ".gitmodules", "LICENSE",
    "README.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
})
PRIVATE_PARTS = frozenset({
    ".git", ".venv", "node_modules", "__pycache__", "artifacts", "runs", "evidence",
    "logs", "log", "data", "datasets", "credentials", "secrets", "backups", "cache",
    "target", "dist", "build", "install", "gen",
})
SOURCE_SUFFIXES = frozenset({
    ".py", ".rs", ".cpp", ".hpp", ".h", ".c", ".ts", ".tsx", ".js", ".jsx",
    ".mjs", ".cjs", ".css", ".html", ".json", ".toml", ".lock", ".yaml", ".yml",
    ".xml", ".sdf", ".config", ".msg", ".srv", ".action", ".sh", ".ps1",
    ".md", ".txt", ".in", ".nsh", ".svg", ".png", ".ico", ".icns", ".bmp", ".cfg",
})
PRIVATE_SUFFIXES = frozenset({
    ".ddpkg", ".zip", ".onnx", ".pt", ".pth", ".safetensors", ".sqlite", ".sqlite3",
    ".db", ".csv", ".parquet", ".ulg", ".bag", ".mcap", ".vhdx", ".pfx", ".p12",
    ".key", ".pem", ".exe", ".dll", ".so", ".dylib", ".pyc",
})
RUNTIME_GITLINK = "shared/dronedream-runtime-source"
ROS_RESOURCE_MARKER = "ros_ws/src/dronedream_agent_ros/resource/dronedream_agent_ros"
APPROVED_ASSETS = {
    "app/desktop/src-tauri/resources/default-assets/school-map.ddpkg":
        "441364e913d588984851120322080131f87b93565b8880205105817aa25c64a8",
    "app/desktop/src-tauri/resources/default-assets/my-drone.ddpkg":
        "695cb08542bfec997a58b14d8beaa11bcc5c6927216c2dd2d4244731dc6434d0",
}


# 功能：
#   判断文件是否属于可审阅的源码范围，拒绝历史数据、密钥及未单独授权的二进制资产。
# 输入：
#   name：仓库内采用正斜杠的相对路径。
# 输出：
#   reason：排除原因；空字符串表示可进入后续内容检查。
def exclusion_reason(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or "\\" in name or ":" in name or ".." in path.parts:
        raise ValueError("Unsafe source path")
    if not path.parts or path.as_posix() != name:
        raise ValueError("Non-canonical source path")
    parts = tuple(part.casefold() for part in path.parts)
    if name == "docs/CODE_ANNOTATION_HANDOFF.md":
        return "private-development-handoff"
    if any(part in PRIVATE_PARTS for part in parts):
        return "private-or-generated-directory"
    if path.name.startswith(".env") and name != ".env.example":
        return "environment-file"
    if path.suffix.casefold() in PRIVATE_SUFFIXES:
        return "data-key-or-unreviewed-binary"
    if len(path.parts) == 1:
        return "" if name in ROOT_FILES else "unreviewed-root-file"
    if path.parts[0] not in SOURCE_ROOTS:
        return "unreviewed-source-root"
    if (path.suffix.casefold() not in SOURCE_SUFFIXES
            and path.name != "CMakeLists.txt" and name != ROS_RESOURCE_MARKER):
        return "unreviewed-file-type"
    return ""


# 功能：
#   仅读取 Git 已跟踪文件和未忽略的新源码，不遍历整个开发目录或导出 Git 数据库。
# 输入：
#   repository：待导出的源码仓库目录。
# 输出：
#   names：去重并排序后的候选相对路径。
def source_names(repository: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=repository, capture_output=True, check=True,
    )
    names = sorted({name.decode("utf-8") for name in result.stdout.split(b"\0") if name})
    return names


# 功能：
#   确认源码路径的每一级均为普通目录或文件，阻止符号链接和 Windows 重解析点越界。
# 输入：
#   repository：已解析的仓库根目录。
#   name：已经过路径格式检查的相对路径。
# 输出：
#   source：经过类型和目录边界校验的普通源码文件。
def checked_source(repository: Path, name: str) -> Path:
    source = repository
    for part in PurePosixPath(name).parts:
        source = source / part
        info = source.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Linked source is not exportable: {name}")
    if not source.is_file() or not source.resolve().is_relative_to(repository):
        raise ValueError(f"Source is not a contained ordinary file: {name}")
    return source


# 功能：
#   在全新目录生成源码副本和逐文件校验清单；不删除、覆盖或提交原有工作。
#   子模块只记录原始固定提交；复制后仍须独立检查隐私与许可，再允许公开发布。
# 输入：
#   repository：包含最新已保存修改的源码仓库。
#   destination：尚不存在且位于仓库外的新快照目录。
# 输出：
#   manifest：导出文件哈希、排除数量及子模块绑定，不包含原机绝对路径。
def export_source(repository: Path, destination: Path) -> dict:
    repository = repository.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination must not exist; existing snapshots are preserved")
    parent = destination.parent.resolve(strict=True)
    if parent.is_relative_to(repository) or repository.is_relative_to(parent / destination.name):
        raise ValueError("Source and destination must not contain each other")
    destination = parent / destination.name
    selected: list[tuple[str, bytes, int]] = []
    excluded: dict[str, int] = {}
    gitlinks = []
    for name in source_names(repository):
        if name == RUNTIME_GITLINK:
            result = subprocess.run(
                ["git", "ls-files", "--stage", "--", name], cwd=repository,
                capture_output=True, text=True, check=True,
            ).stdout.split()
            if len(result) != 4 or result[0] != "160000" or result[2] != "0":
                raise ValueError("Runtime gitlink is not a resolved pinned submodule")
            gitlinks.append({"path": name, "commit": result[1]})
            continue
        # Approval is for two reviewed, immutable archives, not arbitrary imports.
        reason = "" if name in APPROVED_ASSETS else exclusion_reason(name)
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        source = checked_source(repository, name)
        if source.stat().st_size > 16 * 1024 * 1024:
            raise ValueError(f"Source requires separate size review: {name}")
        payload = source.read_bytes()
        if (name in APPROVED_ASSETS
                and hashlib.sha256(payload).hexdigest() != APPROVED_ASSETS[name]):
            raise ValueError(f"Approved asset bytes changed; review required: {name}")
        selected.append((name, payload, stat.S_IMODE(source.stat().st_mode)))
    if not selected or not any(name == "LICENSE" for name, _, _ in selected):
        raise ValueError("No licensed source to export")

    # All candidates are checked before creating output. Exclusive writes prevent
    # accidental replacement; a failed export remains inspectable, never auto-deleted.
    destination.mkdir(exist_ok=False)
    files = []
    for name, payload, mode in selected:
        output = destination / name
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(payload)
        output.chmod(mode)
        files.append({"path": name, "bytes": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {
        "schema_version": "dronedream.public-source-snapshot.v1",
        "scope": "source and two licensed default assets; not an installer or model distribution",
        "privacy_review_required": True,
        "files": files, "excluded_counts": excluded, "gitlinks": gitlinks,
    }
    with (destination / "PUBLIC_SOURCE_SNAPSHOT.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


# 功能：
#   接收显式仓库和目标目录，生成待审阅快照并只报告非敏感文件数量。
# 输入：
#   无；命令行提供 repository 和 destination 路径。
# 输出：
#   无；快照写入指定新目录，失败时命令返回非零状态。
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_source(args.repository, args.destination)
    print(json.dumps({"exported_files": len(manifest["files"]),
                      "excluded_counts": manifest["excluded_counts"],
                      "privacy_review_required": True}))


if __name__ == "__main__":
    main()
