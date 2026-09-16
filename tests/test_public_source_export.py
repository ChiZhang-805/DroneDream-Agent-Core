from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.export_public_source import APPROVED_ASSETS, exclusion_reason, export_source


# 功能：
#   验证源码导出默认排除真实数据、环境文件和未审查模型/资产二进制。
# 输入：
#   name：需要阻止公开的相对路径样例。
# 输出：
#   无；排除规则不生效时断言失败。
@pytest.mark.parametrize("name", [
    ".env", ".env.production", "evidence/user-closed-loop.json",
    "app/.env.example", "src/secrets/key.json", "artifacts/run.json",
    "app/desktop/src-tauri/resources/default-assets/school-map.ddpkg",
    "runtime/model.onnx", "src/data/accounts.json", "app/account.sqlite",
    "signing.pfx", "runtime/runtime.vhdx", "app/node_modules/a/index.js",
    "docs/CODE_ANNOTATION_HANDOFF.md",
    ".github/workflows/windows-installer.yml",
])
def test_private_material_is_excluded(name: str) -> None:
    assert exclusion_reason(name)


# 功能：
#   验证源码、合成测试和空环境模板可以进入内容检查，不误删核心实现。
# 输入：
#   name：允许审阅的源码相对路径。
# 输出：
#   无；合法源码被错误排除时断言失败。
@pytest.mark.parametrize("name", [
    "LICENSE", ".env.example", "src/dronedream_agent_core/contracts.py",
    "tests/test_memory.py", "ros_ws/src/bridge/CMakeLists.txt",
    "app/frontend/package-lock.json", ".github/workflows/ci.yml",
    "ros_ws/src/dronedream_agent_ros/resource/dronedream_agent_ros",
    "ros_ws/src/dronedream_agent_ros/setup.cfg",
    "app/desktop/src-tauri/installer/sidebar.bmp",
])
def test_source_is_retained(name: str) -> None:
    assert exclusion_reason(name) == ""


# 功能：
#   验证路径逃逸和非规范路径不能被当成普通排除项忽略。
# 输入：
#   name：恶意或非规范相对路径。
# 输出：
#   无；路径未被拒绝时断言失败。
@pytest.mark.parametrize("name", ["../key", "/etc/passwd", "C:/key", "a\\b", "a//b"])
def test_unsafe_path_is_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        exclusion_reason(name)


# 功能：
#   验证未提交的新源码被保留、私有历史不复制、哈希准确且旧快照不会被覆盖。
# 输入：
#   tmp_path：pytest 隔离的临时目录。
# 输出：
#   无；快照边界或内容发生错误时断言失败。
def test_export_preserves_source_and_excludes_history(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    (source / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (source / ".env").write_text("PRIVATE_SENTINEL=not-a-real-key", encoding="utf-8")
    (source / "src").mkdir()
    (source / "src" / "new.py").write_text("answer = 42\n", encoding="utf-8")
    (source / "evidence").mkdir()
    (source / "evidence" / "run.json").write_text('{"user_message":"private"}')
    destination = tmp_path / "candidate"
    manifest = export_source(source, destination)
    assert {item["path"] for item in manifest["files"]} == {"LICENSE", "src/new.py"}
    assert not (destination / ".git").exists()
    assert not (destination / ".env").exists()
    assert not (destination / "evidence").exists()
    for item in manifest["files"]:
        digest = hashlib.sha256((destination / item["path"]).read_bytes()).hexdigest()
        assert digest == item["sha256"]
    assert json.loads((destination / "PUBLIC_SOURCE_SNAPSHOT.json").read_text()) == manifest
    assert (source / ".env").is_file()
    with pytest.raises(ValueError, match="must not exist"):
        export_source(source, destination)


# 功能：
#   验证目标目录不能位于原仓库内部，避免把历史导出位置再次包含进快照。
# 输入：
#   tmp_path：pytest 隔离的临时目录。
# 输出：
#   无；嵌套导出未被拒绝时断言失败。
def test_nested_output_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not contain"):
        export_source(tmp_path, tmp_path / "export")


# 功能：
#   验证授权清单与导出器及实际资产字节一致，避免批准范围随文件内容悄悄变化。
# 输入：
#   无；读取当前仓库的两份默认资产及独立授权清单。
# 输出：
#   无；授权范围、许可证或哈希不一致时断言失败。
def test_approved_asset_license_matches_exact_archives() -> None:
    root = Path(__file__).resolve().parents[1]
    grant = json.loads((root / "runtime/default-assets-licenses.json").read_text("utf-8"))
    assert grant["license_expression"] == "MIT"
    assert {asset["path"]: asset["sha256"] for asset in grant["assets"]} == APPROVED_ASSETS
    for name, expected in APPROVED_ASSETS.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected


# 功能：
#   验证即使文件名与批准资产相同，只要字节改变就禁止导出，并保留原工作区。
# 输入：
#   tmp_path：隔离的临时目录。
# 输出：
#   无；不匹配的资产被导出时断言失败。
def test_changed_approved_archive_is_rejected_before_export(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    (source / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    name = next(iter(APPROVED_ASSETS))
    asset = source / name
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"unapproved replacement")
    destination = tmp_path / "candidate"
    with pytest.raises(ValueError, match="Approved asset bytes changed"):
        export_source(source, destination)
    assert not destination.exists()
    assert asset.read_bytes() == b"unapproved replacement"
