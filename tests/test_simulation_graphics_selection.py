"""Hardware selection boundaries; fixture names are not GPU or flight acceptance."""

import pytest

from dronedream_agent_core import simulation_graphics_selection as selection


# 功能：
#   只模拟平台与驱动探测，不触碰真实显卡或操作系统环境。
# 输入：
#   monkeypatch：自动恢复替换项的测试工具。
# 输出：
#   无。
@pytest.fixture
def wsl(monkeypatch):
    monkeypatch.setattr(selection.sys, "platform", "linux")
    monkeypatch.setattr(selection.Path, "exists", lambda _: True)
    monkeypatch.setattr(selection, "prepare_render_process_environment", lambda env: (dict(env), {}))


# 功能：
#   确保显式图形配置优先且不启动探测，无 GPU 的电脑不强制应用 WSL 驱动。
# 输入：
#   wsl：隔离平台；monkeypatch：探测替换工具；key：显式环境项。
# 输出：
#   无。
@pytest.mark.parametrize("key", ["GALLIUM_DRIVER", "LIBGL_ALWAYS_SOFTWARE", "EGL_PLATFORM", "MESA_D3D12_DEFAULT_ADAPTER_NAME"])
def test_explicit_configuration_is_not_overridden(wsl, monkeypatch, key):
    monkeypatch.setattr(selection, "_probe_renderer", lambda _: pytest.fail("unexpected probe"))
    original = {key: "explicit", "SECRET": "not-published"}
    result, receipt = selection.select_render_process_environment(original)
    assert result == original and result is not original
    assert "not-published" not in str(receipt)
    monkeypatch.setattr(selection.Path, "exists", lambda _: False)
    assert selection.select_render_process_environment({})[0] == {}


# 功能：
#   默认 WSL 仿真选择进程级软件渲染，避免长时 Gazebo 负载触发 dxg 内核故障。
# 输入：
#   wsl、monkeypatch：隔离平台与探测输出。
# 输出：
#   无。
def test_default_wsl_renderer_is_stable_process_local_software(wsl, monkeypatch):
    monkeypatch.setattr(selection, "_probe_renderer", lambda _: pytest.fail("unexpected probe"))
    original = {"SECRET": "private"}
    result, receipt = selection.select_render_process_environment(original)
    assert result["GALLIUM_DRIVER"] == "llvmpipe"
    assert result["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert result["LP_NUM_THREADS"] == "8"
    assert original == {"SECRET": "private"}
    assert receipt["selection"] == "stable-wsl-software"
    assert receipt["probes"] == []
    assert "private" not in str(receipt)
    assert not receipt["flight_qualification_granted"]


# 功能：默认策略不把短时成功的 D3D12 探测当成长程内存稳定性证明。
# 输入：隔离平台与可用的 NVIDIA 探测替身；输出：仍为限线程软件渲染。
def test_default_wsl_renderer_does_not_promote_short_nvidia_probe(wsl, monkeypatch):
    monkeypatch.setattr(selection, "_probe_renderer", lambda _: pytest.fail("unexpected probe"))
    result, receipt = selection.select_render_process_environment({})
    assert receipt["selection"] == "stable-wsl-software"
    assert result["GALLIUM_DRIVER"] == "llvmpipe"
    assert result["LP_NUM_THREADS"] == "8"
    assert "MESA_D3D12_DEFAULT_ADAPTER_NAME" not in result


# 功能：
#   显式启用实验性 WSL GPU 时仍以真实探测结果选择设备，不按某台电脑型号写死。
# 输入：
#   wsl、monkeypatch：隔离平台与探测输出。
# 输出：
#   无。
def test_opted_in_discrete_renderer_is_process_local(wsl, monkeypatch):
    names = iter(["llvmpipe (fixture)", "D3D12 (Intel UHD)", "D3D12 (NVIDIA fixture)"])
    monkeypatch.setattr(selection, "_probe_renderer", lambda _: next(names))
    original = {
        "SECRET": "private",
        "DRONEDREAM_ALLOW_EXPERIMENTAL_WSL_GPU_RENDERING": "1",
    }
    result, receipt = selection.select_render_process_environment(original)
    assert result["GALLIUM_DRIVER"] == "d3d12"
    assert result["MESA_D3D12_DEFAULT_ADAPTER_NAME"] == "NVIDIA"
    assert original == {
        "SECRET": "private",
        "DRONEDREAM_ALLOW_EXPERIMENTAL_WSL_GPU_RENDERING": "1",
    }
    assert "private" not in str(receipt)
    assert not receipt["flight_qualification_granted"]


# 功能：
#   探测失败不冒充硬件可用，只有集显时保持真实可用的默认 D3D12 设备。
# 输入：
#   wsl、monkeypatch：隔离平台；renderer：候选渲染器。
# 输出：
#   无。
@pytest.mark.parametrize("renderer", [None, "llvmpipe", "D3D12 (Intel UHD)"])
def test_unavailable_discrete_adapter_does_not_get_fabricated(wsl, monkeypatch, renderer):
    names = iter(["llvmpipe", renderer, renderer, renderer, renderer])
    monkeypatch.setattr(selection, "_probe_renderer", lambda _: next(names))
    result, receipt = selection.select_render_process_environment(
        {"DRONEDREAM_ALLOW_EXPERIMENTAL_WSL_GPU_RENDERING": "1"}
    )
    assert "MESA_D3D12_DEFAULT_ADAPTER_NAME" not in result
    assert (result.get("GALLIUM_DRIVER") == "d3d12") == (renderer == "D3D12 (Intel UHD)")


# 功能：
#   本机默认渲染器已经是硬件时保持其选择，避免无必要驱动切换。
# 输入：
#   wsl、monkeypatch：隔离平台与探测输出。
# 输出：
#   无。
def test_working_default_hardware_is_preserved(wsl, monkeypatch):
    monkeypatch.setattr(selection, "_probe_renderer", lambda _: "Mesa Intel hardware")
    environment = {"DRONEDREAM_ALLOW_EXPERIMENTAL_WSL_GPU_RENDERING": "1"}
    assert selection.select_render_process_environment(environment)[0] == environment
