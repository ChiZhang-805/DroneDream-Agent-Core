"""Desktop canonical paths must retain their identity across the WSL boundary."""

import os
from pathlib import Path

import pytest

from dronedream_agent_app.runtime_manager import _wsl_path
from dronedream_agent_core.windows_paths import windows_drive_path_to_wsl


# 功能：
#   覆盖普通与扩展长度盘符、根目录、空格、中文及超过传统 MAX_PATH 的文件名组合。
# 输入：
#   path：Windows 路径；expected：期望的 WSL 路径。
# 输出：
#   None：转换丢失目录或字符时测试失败。
@pytest.mark.parametrize("path,expected", [
    ("C:\\", "/mnt/c/"),
    ("C:/Users/user/DroneDream/file.json", "/mnt/c/Users/user/DroneDream/file.json"),
    (r"\\?\C:\Users\user\DroneDream\file.json", "/mnt/c/Users/user/DroneDream/file.json"),
    (r"\\?\Q:\项目 数据\无人机\模型.onnx", "/mnt/q/项目 数据/无人机/模型.onnx"),
    ("//?/C:/Program Files/DroneDream/executor.py", "/mnt/c/Program Files/DroneDream/executor.py"),
    ("\\\\?\\Q:\\" + "long-directory\\" * 30 + "x", "/mnt/q/" + "long-directory/" * 30 + "x"),
])
def test_windows_drive_mapping(path, expected):
    assert windows_drive_path_to_wsl(path) == expected


# 功能：
#   不因增加长路径支持而错误放行网络共享、设备路径、相对位置、数据流或控制字符。
# 输入：
#   path：不可明确对应本地 WSL 盘符挂载的路径。
# 输出：
#   None：不支持的路径被拒绝，否则测试失败。
@pytest.mark.parametrize("path", [r"\\server\share\file", r"\\?\UNC\server\share\file",
    r"\\.\C:\file", r"\\?\Volume{abc}\file", r"C:relative", r"relative\file", "/tmp/file",
    r"C:\folder\..\file", r"C:\file:stream", "C:\\bad\x00file", r"1:\file"])
def test_unsupported_namespaces_are_not_guessed(path):
    with pytest.raises(ValueError, match="WINDOWS_PATH_CANNOT_MAP_TO_WSL"):
        windows_drive_path_to_wsl(path)


# 功能：
#   用 Windows 实际 Path.resolve 检查桌面传入的扩展长度资源路径，而非只测试字符串替身。
# 输入：
#   tmp_path：隔离资源目录。
# 输出：
#   None：相同文件的两种路径映射不一致时测试失败。
@pytest.mark.skipif(os.name != "nt", reason="Windows extended-length filesystem paths")
def test_desktop_extended_path_resolves_to_same_file(tmp_path):
    regular = tmp_path / "资源目录" / "executor.py"
    regular.parent.mkdir()
    regular.write_text("# test resource\n", encoding="utf-8")
    extended = Path("\\\\?\\" + str(regular.resolve()))
    assert extended.samefile(regular)
    assert _wsl_path(extended) == _wsl_path(regular)
