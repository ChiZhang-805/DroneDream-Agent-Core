"""Explicit Windows drive-path mapping for the managed WSL runtime."""

import re
from pathlib import PureWindowsPath


# 功能：
#   1. 将已经解析的普通盘符路径及 Windows 扩展长度盘符路径映射到 WSL /mnt。
#   2. 拒绝 UNC、设备命名空间、相对路径、残留父目录和数据流，不猜测其挂载位置。
# 输入：
#   path：已经解析的 Windows 绝对路径，允许 \\?\ 盘符前缀。
# 输出：
#   mapped：保留文件名、空格及 Unicode 的 WSL 绝对路径，命令调用方仍负责 shell 引用。
def windows_drive_path_to_wsl(path: str | PureWindowsPath) -> str:
    windows = PureWindowsPath(path)
    drive = windows.drive
    # Rust canonicalize 会保留扩展长度前缀；只识别盘符前缀，不将 UNC 或设备路径伪装为盘符。
    if drive.startswith("\\\\?\\"):
        drive = drive[4:]
    if not re.fullmatch(r"[A-Za-z]:", drive) or windows.root != "\\":
        raise ValueError("WINDOWS_PATH_CANNOT_MAP_TO_WSL")
    components = windows.parts[1:]
    if any(part == ".." or ":" in part or any(ord(char) < 32 for char in part) for part in components):
        raise ValueError("WINDOWS_PATH_CANNOT_MAP_TO_WSL")
    mapped = f"/mnt/{drive[0].lower()}/" + "/".join(components)
    return mapped
