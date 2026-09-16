from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from dronedream_agent_core import simulation_graphics_evidence as evidence
from dronedream_agent_core import simulation_graphics_lifetime as lifetime


# 功能：
#   请求证据不能把显式空值、不可执行的空字符或非映射配置记为合法环境。
# 输入：
#   environment：故意构造的错误图形环境。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("environment", [None, [], {"GALLIUM_DRIVER": None},
    {"EGL_PLATFORM": "surfaceless\x00ignored"}])
def test_graphics_evidence_rejects_unusable_environment(environment):
    with pytest.raises(ValueError, match="SIMULATION_GRAPHICS_REQUEST_INVALID"):
        evidence.graphics_request_evidence(environment)


# 功能：
#   即使不启用 WSL 保护，也不得输出 subprocess 无法使用的环境映射。
# 输入：
#   environment：带非法变量名、值或容器类型的启动配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("environment", [[("KEY", "value")], {"KEY": None},
    {"BAD=KEY": "x"}, {"": "x"}, {"KEY": "x\x00y"}, {1: "x"}])
def test_render_environment_rejects_invalid_entries(environment):
    with pytest.raises(ValueError, match="SIMULATION_RENDER_ENVIRONMENT_INVALID"):
        lifetime.prepare_render_process_environment(environment)


# 功能：
#   读取期间驱动文件变化必须报错，不能把不稳定内容记成已绑定驱动。
# 输入：
#   tmp_path：测试专属驱动文件目录。
#   monkeypatch：替换平台及底层文件读取的测试工具。
# 输出：
#   None：不返回业务数据。
def test_driver_snapshot_detects_file_growth(tmp_path, monkeypatch):
    path = tmp_path / "libd3d12core.so"
    path.write_bytes(b"\x7fELForiginal-test-library")
    original_open = Path.open

    # 功能：
    #   返回可在实际读取后追加内容的文件上下文，模拟驱动并发更新。
    # 输入：
    #   target：当前准备打开的文件。
    #   mode：调用方指定的打开模式。
    #   args：透传的位置参数。
    #   kwargs：透传的关键字参数。
    # 输出：
    #   reader：关联真实文件描述符的测试读取器。
    @contextmanager
    def changing_open(target, mode="r", *args, **kwargs):
        with original_open(target, mode, *args, **kwargs) as stream:
            # 功能：
            #   读取原内容后立刻改变文件，再让生产代码检查同一个描述符。
            # 输入：
            #   size：调用方实际要求的最大读取字节数。
            # 输出：
            #   content：修改前已读取的字节。
            def changing_read(size):
                content = stream.read(size)
                with original_open(target, "ab") as writer:
                    writer.write(b"changed")
                return content

            reader = (SimpleNamespace(fileno=stream.fileno, read=changing_read)
                      if target == path and mode == "rb" else stream)
            yield reader

    monkeypatch.setattr(lifetime, "WSL_D3D12_CORE", path)
    monkeypatch.setattr(lifetime.sys, "platform", "linux")
    monkeypatch.setattr(Path, "open", changing_open)
    with pytest.raises(ValueError, match="SIMULATION_WSL_D3D12_LIBRARY_INVALID"):
        lifetime.prepare_render_process_environment({"GALLIUM_DRIVER": "d3d12"})
