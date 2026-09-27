"""Select a probed WSL renderer without changing global settings or control deadlines."""

import json
import subprocess
import sys
from pathlib import Path

from .simulation_graphics_lifetime import prepare_render_process_environment


# 功能：
#   在独立、有限时长的进程中建立 EGL 上下文，读取真实渲染器名称，不加载到控制进程。
# 输入：
#   environment：仅供图形探测子进程使用的环境副本。
# 输出：
#   renderer：有限渲染器名称；失败返回 None，不公开底层输出或环境中的凭证。
def _probe_renderer(environment: dict[str, str]) -> str | None:
    try:
        result = subprocess.run([sys.executable, "-m", "dronedream_agent_core.simulation_graphics_probe"],
            env={**environment, "EGL_PLATFORM": "surfaceless", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, timeout=8, check=False)
        if result.returncode or len(result.stdout) > 4096:
            return None
        renderer = json.loads(result.stdout).get("renderer")
        if not isinstance(renderer, str) or not 1 <= len(renderer) <= 256:
            return None
        return renderer
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


# 功能：
#   将实际探测的硬件渲染器与软件回退区分；名称不作为图像等价或飞行合格证明。
# 输入：
#   renderer：探测返回的有限名称。
# 输出：
#   hardware：非软件栅格器且具有有效名称时为真。
def _hardware(renderer: str | None) -> bool:
    return bool(renderer and not any(name in renderer.lower()
        for name in ("llvmpipe", "softpipe", "software", "basic render", "warp")))


# 功能：
#   1. 仅在 WSL 存在 GPU 接口且用户未指定渲染器时，探测并选择可用硬件 EGL 渲染。
#   2. 默认驱动已使用硬件则保持；软件回退时尝试 D3D12 的默认、NVIDIA、AMD、Intel Arc。
#   3. 优先实际匹配的独立 GPU，否则使用可用默认硬件；失败保留原环境，不绕过任何安全门控。
# 输入：
#   environment：运行环境；保持父环境及 PX4、模型进程不变。
# 输出：
#   selected、receipt：独立的渲染进程环境与不含凭据的探测回执。
def select_render_process_environment(environment: dict[str, str]) -> tuple[dict, dict]:
    selected = dict(environment)
    receipt = {"selection": "unchanged", "probes": [], "flight_qualification_granted": False}
    if (sys.platform != "linux" or not Path("/dev/dxg").exists()
            or any(selected.get(key) for key in ("GALLIUM_DRIVER", "LIBGL_ALWAYS_SOFTWARE",
                "MESA_D3D12_DEFAULT_ADAPTER_NAME", "EGL_PLATFORM"))):
        return selected, receipt
    original = _probe_renderer(selected)
    receipt["probes"].append({"candidate": "default", "renderer": original})
    if _hardware(original):
        return selected, receipt
    fallback = None
    for adapter in (None, "NVIDIA", "AMD", "Intel(R) Arc"):
        candidate = {**selected, "GALLIUM_DRIVER": "d3d12"}
        if adapter is not None:
            candidate["MESA_D3D12_DEFAULT_ADAPTER_NAME"] = adapter
        try:
            probe_environment, _ = prepare_render_process_environment(candidate)
        except (OSError, ValueError):
            continue
        renderer = _probe_renderer(probe_environment)
        receipt["probes"].append({"candidate": adapter or "d3d12-default", "renderer": renderer})
        if not _hardware(renderer) or not renderer.startswith("D3D12 ("):
            continue
        if fallback is None:
            fallback = candidate, renderer
        if adapter is not None and adapter.lower() in renderer.lower():
            fallback = candidate, renderer
            break
    if fallback is not None:
        selected, renderer = fallback
        receipt.update(selection="probed-wsl-hardware", renderer=renderer)
    return selected, receipt
