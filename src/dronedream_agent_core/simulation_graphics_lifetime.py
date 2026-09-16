"""Process-local WSL D3D12 lifetime protection for Gazebo render processes.

Recorded SIGSEGV at shutdown called a thread-local destructor in already
unmapped libd3d12core.so. A startup loader reference prevents intermediate
dlclose from unmapping it before those threads exit. No driver installation,
global preload file, numerical setting or control-process environment changes.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path

from .plugin_files import read_plugin_file

WSL_D3D12_CORE = Path("/usr/lib/wsl/lib/libd3d12core.so")


# 功能：
#   1. 独立复制合法启动环境，仅在 Linux 显式选择 D3D12 时为渲染进程保留驱动引用。
#   2. 有界读取并记录驱动快照，拒绝读取中变化；保留已有预加载项，不修改父进程。
#   3. 返回环境不得用于 PX4 或模型推理；摘要不是实际加载证明、图像等价或飞行授权。
# 输入：
#   environment：渲染子进程准备继承的字符串环境映射。
# 输出：
#   prepared：独立子进程环境与驱动引用保护回执组成的二元组。
def prepare_render_process_environment(environment: Mapping[str, str]) -> tuple[dict, dict]:
    if not isinstance(environment, Mapping):
        raise ValueError("SIMULATION_RENDER_ENVIRONMENT_INVALID")
    result = dict(environment)
    if any(not isinstance(key, str) or not key or "=" in key or "\x00" in key
           or not isinstance(value, str) or "\x00" in value
           for key, value in result.items()):
        raise ValueError("SIMULATION_RENDER_ENVIRONMENT_INVALID")
    receipt = {"applied": False, "scope": "render-process-only",
               "flight_qualification_granted": False}
    if sys.platform != "linux" or result.get("GALLIUM_DRIVER") != "d3d12":
        prepared = result, {**receipt, "reason": "not-explicit-linux-d3d12"}
        return prepared
    if not WSL_D3D12_CORE.is_file():
        prepared = result, {**receipt, "reason": "wsl-d3d12-library-unavailable"}
        return prepared
    # 保留其他预加载项；只更新返回副本，不写系统预加载文件或全局环境。
    existing = result.get("LD_PRELOAD", "")
    if not isinstance(existing, str) or len(existing) > 2048:
        raise ValueError("SIMULATION_RENDER_PRELOAD_INVALID")
    try:
        # 系统驱动允许合法符号链接，先解析当前目标，再复用普通文件稳定读取检查。
        content = read_plugin_file(WSL_D3D12_CORE.resolve(strict=True), limit=32*1024*1024)
    except (OSError, ValueError, RuntimeError) as error:
        raise ValueError("SIMULATION_WSL_D3D12_LIBRARY_INVALID") from error
    if not content.startswith(b"\x7fELF"):
        raise ValueError("SIMULATION_WSL_D3D12_LIBRARY_INVALID")
    library = str(WSL_D3D12_CORE)
    inherited = library in existing.replace(":", " ").split()
    if not inherited:
        result["LD_PRELOAD"] = f"{library}:{existing}" if existing else library
    prepared = result, {**receipt, "applied": True, "library_path": library,
        "library_sha256": hashlib.sha256(content).hexdigest(), "library_bytes": len(content),
        "already_inherited": inherited, "reason": "retain-driver-until-process-exit"}
    return prepared
