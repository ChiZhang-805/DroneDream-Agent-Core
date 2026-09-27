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
#   默认 CPU 回退时优先选择真实探测成功的独立 GPU，不按某台电脑型号写死。
# 输入：
#   wsl、monkeypatch：隔离平台与探测输出。
# 输出：
#   无。
def test_probed_discrete_renderer_is_process_local(wsl, monkeypatch):
    names = iter(["llvmpipe (fixture)", "D3D12 (Intel UHD)", "D3D12 (NVIDIA fixture)"])
    monkeypatch.setattr(selection, "_probe_renderer", lambda _: next(names))
    original = {"SECRET": "private"}
    result, receipt = selection.select_render_process_environment(original)
    assert result["GALLIUM_DRIVER"] == "d3d12"
    assert result["MESA_D3D12_DEFAULT_ADAPTER_NAME"] == "NVIDIA"
    assert original == {"SECRET": "private"}
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
    result, receipt = selection.select_render_process_environment({})
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
    assert selection.select_render_process_environment({})[0] == {}
