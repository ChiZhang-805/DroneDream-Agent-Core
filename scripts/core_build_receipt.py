"""Bind built Core components to one clean source revision before product staging."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

from dronedream_agent_core.plugin_files import check_plain_plugin_path, hash_plugin_file
from dronedream_agent_core.source_identity import source_repository_identity

SCHEMA = "dronedream.core-components-build.v1"
RESOURCE_ROOT = "app/desktop/src-tauri/resources"
BINARY_ROOT = "app/desktop/src-tauri/binaries"
RECEIPT = "artifacts/desktop/core-components-build.json"
BINARIES = (
    "dronedream-autonomy-core-x86_64-pc-windows-msvc.exe",
    "dronedream-plugin-isolator-x86_64-pc-windows-msvc.exe",
)


# 功能：
#   读取显式仓库的 Git 身份或状态，命令失败时停止，不猜测来源。
# 输入：
#   root：当前 Core 源码根目录。
#   arguments：Git 子命令及参数。
# 输出：
#   value：Git 命令的去尾部空白结果。
def git_value(root: Path, *arguments: str) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True, encoding="utf-8", timeout=30,
    ).strip()
    return value


# 功能：
#   验证清单中的普通相对路径，拒绝目录穿越、Windows 别名和链接。
# 输入：
#   root：允许访问的文件根目录。
#   relative：清单提供的正斜杠相对路径。
# 输出：
#   path：通过边界检查的实际文件路径。
def bound_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("CORE_BUILD_PATH_INVALID")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} or re.search(r'[<>:"|?*\x00-\x1f]', part)
           or part.endswith((".", " ")) or re.fullmatch(
               r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part,
           ) for part in parts):
        raise ValueError("CORE_BUILD_PATH_INVALID")
    path = root.joinpath(*PurePosixPath(relative).parts)
    check_plain_plugin_path(path)
    if not path.is_file():
        raise ValueError("CORE_BUILD_FILE_MISSING:" + relative)
    return path


# 功能：
#   有界读取构建清单，按同一批字节验证可选摘要，并拒绝非对象根节点和重复字段。
# 输入：
#   path：待读取的 JSON 文件。
#   expected_sha256：调用方已冻结的文件摘要；None 表示只核对 JSON。
# 输出：
#   value：唯一字段的 JSON 对象。
def read_object(path: Path, expected_sha256: str | None = None) -> dict:
    from dronedream_agent_core.plugin_files import read_plugin_file
    from dronedream_plugin_sdk.protocol import decode_json

    content = read_plugin_file(path, limit=8 * 1024**2)
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("CORE_BUILD_RECEIPT_BYTES_CHANGED")
    value = decode_json(content, limit=8 * 1024**2)
    if not isinstance(value, dict):
        raise ValueError("CORE_BUILD_OBJECT_REQUIRED")
    return value


# 功能：
#   按实际 Runtime、插件和资产清单收集安装文件，逐个验证原清单摘要和大小。
# 输入：
#   root：显式 Core 源码与构建产物所在目录。
# 输出：
#   files：以仓库相对路径为键、实际摘要和字节数为值的文件集合。
def component_files(root: Path) -> dict:
    files = {}

    # 功能：
    #   对单个安装文件进行有界摘要检查，并禁止重复路径覆盖先前的清单条目。
    # 输入：
    #   relative：文件相对 Core 根目录的路径。
    #   expected：可选的上游清单摘要。
    #   size：可选的上游清单字节数。
    # 输出：
    #   path：已纳入文件集合的实际路径。
    def add(relative: str, expected: str | None = None, size: int | None = None) -> Path:
        if relative.casefold() in {name.casefold() for name in files}:
            raise ValueError("CORE_BUILD_DUPLICATE_FILE")
        path = bound_file(root, relative)
        digest = hash_plugin_file(path, limit=2 * 1024**3)
        actual_size = path.stat().st_size
        if expected is not None and digest != expected:
            raise ValueError("CORE_BUILD_FILE_HASH_MISMATCH:" + relative)
        if size is not None and (type(size) is not int or actual_size != size):
            raise ValueError("CORE_BUILD_FILE_SIZE_MISMATCH:" + relative)
        files[relative] = {"sha256": digest, "bytes": actual_size}
        return path

    for name in BINARIES:
        path = add(f"{BINARY_ROOT}/{name}")
        with path.open("rb") as stream:
            if stream.read(2) != b"MZ":
                raise ValueError("CORE_BUILD_WINDOWS_BINARY_REQUIRED")
    runtime = read_object(add(f"{RESOURCE_ROOT}/runtime/runtime-manifest.json"))
    for entry in runtime["files"]:
        add(f"{RESOURCE_ROOT}/runtime/{entry['path']}", entry["sha256"], entry["bytes"])
    required_runtime = {f"{RESOURCE_ROOT}/runtime/{name}" for name in (
        "provenance.json", "native-sensors/native-sensor-runtime.json",
        "licenses/LICENSE", "licenses/default-assets-licenses.json",
        "payload-placement/payload-placement-runtime.json",
        "payload-placement/libdronedream-payload-placement.so",
        "camera-clock/camera-clock-runtime.json", "camera-clock/libdronedream-camera-clock.so",
    )}
    if not required_runtime <= files.keys():
        raise ValueError("CORE_BUILD_REQUIRED_RUNTIME_FILES_MISSING")
    local_policy_prefix = f"{RESOURCE_ROOT}/runtime/local-policy/"
    local_policy_files = {name for name in files if name.startswith(local_policy_prefix)}
    required_local_policy = {
        f"{local_policy_prefix}catalog.json",
        f"{local_policy_prefix}licenses.json",
    }
    if local_policy_files and not required_local_policy <= local_policy_files:
        raise ValueError("CORE_BUILD_LOCAL_POLICY_INCOMPLETE")
    provenance = read_object(bound_file(root, f"{RESOURCE_ROOT}/runtime/provenance.json"))
    if (provenance.get("source_commit") != git_value(root, "rev-parse", "HEAD")
            or provenance.get("source_tree") != git_value(root, "rev-parse", "HEAD^{tree}")
            or provenance.get("source_repository") != source_repository_identity(root)
            or provenance.get("working_tree_dirty") is not False):
        raise ValueError("CORE_BUILD_RUNTIME_SOURCE_MISMATCH")
    plugins = read_object(add(f"{RESOURCE_ROOT}/official-plugins/index.json"))
    if not plugins.get("plugins"):
        raise ValueError("CORE_BUILD_PLUGINS_MISSING")
    for entry in plugins["plugins"]:
        add(f"{RESOURCE_ROOT}/official-plugins/{entry['file']}", entry["sha256"])
    assets = read_object(add(f"{RESOURCE_ROOT}/default-assets/index.json"))
    if assets.get("schema_version") == "dronedream.bundled-assets.v2":
        pairs = [assets.get("qualified_pair")]
    elif assets.get("schema_version") == "dronedream.bundled-assets.v3":
        pairs = assets.get("qualified_pairs")
        if (
            not isinstance(pairs, list)
            or assets.get("qualified_pair") not in pairs
            or assets.get("default_qualification_id")
            != assets.get("qualified_pair", {}).get("qualification_id")
        ):
            raise ValueError("CORE_BUILD_DEFAULT_ASSETS_INCOMPLETE")
    else:
        raise ValueError("CORE_BUILD_DEFAULT_ASSETS_INCOMPLETE")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("CORE_BUILD_DEFAULT_ASSETS_INCOMPLETE")
    seen_files: set[str] = set()
    for pair in pairs:
        entries = pair.get("packages") if isinstance(pair, dict) else None
        if (
            not isinstance(entries, list)
            or len(entries) != 2
            or {entry.get("kind") for entry in entries if isinstance(entry, dict)}
            != {"map", "vehicle"}
        ):
            raise ValueError("CORE_BUILD_DEFAULT_ASSETS_INCOMPLETE")
        for entry in entries:
            filename = entry["file"]
            if filename in seen_files:
                raise ValueError("CORE_BUILD_DEFAULT_ASSET_FILE_DUPLICATE")
            seen_files.add(filename)
            add(f"{RESOURCE_ROOT}/default-assets/{filename}", entry["sha256"])
    return files


# 功能：
#   对固定干净源码及安装组件建立或复验回执，拒绝回执绑定后变化的二进制和资源。
# 输入：
#   root：当前 Core 根目录。
#   expected_commit：构建启动时固定的源码提交。
#   verify：为 True 时只核对已有回执，为 False 时生成新回执。
#   expected_receipt_sha256：暂存调用方冻结的回执摘要，只允许与 verify 一起使用。
# 输出：
#   receipt：绑定真实文件字节与源码身份的组件构建回执。
def build_receipt(root: Path, expected_commit: str, *, verify: bool,
                  expected_receipt_sha256: str | None = None) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("CORE_BUILD_COMMIT_INVALID")
    if expected_receipt_sha256 is not None and (
        not verify or not re.fullmatch(r"[0-9a-f]{64}", expected_receipt_sha256)
    ):
        raise ValueError("CORE_BUILD_RECEIPT_HASH_ARGUMENT_INVALID")
    check_plain_plugin_path(root)
    identity = source_repository_identity(root)
    if identity is None or git_value(root, "rev-parse", "HEAD") != expected_commit:
        raise ValueError("CORE_BUILD_SOURCE_MISMATCH")
    if git_value(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("CORE_BUILD_SOURCE_DIRTY")
    receipt = {
        "schema_version": SCHEMA, "source_repository": identity,
        "source_commit": expected_commit,
        "source_tree": git_value(root, "rev-parse", "HEAD^{tree}"),
        "files": component_files(root),
    }
    if git_value(root, "rev-parse", "HEAD") != expected_commit or git_value(
        root, "status", "--porcelain=v1", "--untracked-files=all",
    ):
        raise ValueError("CORE_BUILD_SOURCE_CHANGED")
    target = root / RECEIPT
    check_plain_plugin_path(target)
    if verify:
        if read_object(target, expected_receipt_sha256) != receipt:
            raise ValueError("CORE_BUILD_RECEIPT_MISMATCH")
    else:
        # 只写构建专属回执，不赋予模型准入或飞行资格；后续暂存逐字节复验。
        target.parent.mkdir(parents=True, exist_ok=True)
        from dronedream_agent_core.runtime_control_io import publish_runtime_json

        publish_runtime_json(target, receipt, maximum_bytes=8 * 1024**2)
    return receipt


# 功能：
#   提供组件构建回执的生成和只读复验入口，不启动安装或飞行。
# 输入：
#   无：参数来自命令行。
# 输出：
#   exit_code：验证或生成成功时为零。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--expected-receipt-sha256")
    args = parser.parse_args()
    receipt = build_receipt(args.repository, args.expected_commit, verify=args.verify,
                            expected_receipt_sha256=args.expected_receipt_sha256)
    print(json.dumps({"source_commit": receipt["source_commit"],
                      "files": len(receipt["files"]), "verified": args.verify}))
    exit_code = 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
