import ast
import hashlib
from pathlib import Path

import pytest

from dronedream_agent_core import simulation_graphics_lifetime as lifetime


class LinuxNamedTestLibrary:
    """Keep loader names POSIX even when the unit test host is Windows."""
    # 功能：
    #   保留真实临时文件，令测试加载器名称独立于宿主操作系统。
    # 输入：
    #   self：准备初始化的测试对象。
    #   path：实际读写的测试文件路径。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, path):
        self.path = path

    # 功能：
    #   提供测试专用的 POSIX 预加载名称，不指向真实系统驱动。
    # 输入：
    #   self：持有临时文件的测试对象。
    # 输出：
    #   name：测试加载器使用的库名。
    def __str__(self):
        name = "/test-wsl/libd3d12core.so"
        return name

    # 功能：
    #   按实际临时文件判断驱动是否存在。
    # 输入：
    #   self：持有临时文件的测试对象。
    # 输出：
    #   exists：实际路径是否为文件。
    def is_file(self):
        exists = self.path.is_file()
        return exists

    # 功能：
    #   返回可进行真实文件描述符与读取稳定性验证的路径。
    # 输入：
    #   self：持有临时文件的测试对象。
    #   strict：是否要求解析目标存在。
    # 输出：
    #   resolved：实际临时文件的绝对路径。
    def resolve(self, *, strict):
        resolved = self.path.resolve(strict=strict)
        return resolved

    # 功能：
    #   读取测试文件作为独立的摘要预期输入。
    # 输入：
    #   self：持有临时文件的测试对象。
    # 输出：
    #   content：当前测试文件的字节。
    def read_bytes(self):
        content = self.path.read_bytes()
        return content

    # 功能：
    #   修改测试专属文件以构造合法魔数或无效驱动反例。
    # 输入：
    #   self：持有临时文件的测试对象。
    #   value：本次测试需要的文件字节。
    # 输出：
    #   size：实际写入字节数。
    def write_bytes(self, value):
        size = self.path.write_bytes(value)
        return size


# 功能：
#   将驱动定位重定向到带 ELF 魔数的临时文件，不创建真实可执行库或操作系统驱动。
# 输入：
#   tmp_path：本次测试专属目录。
#   monkeypatch：自动恢复模块常量与平台的测试工具。
# 输出：
#   path：带 Linux 加载器名称的临时库对象。
@pytest.fixture
def library(tmp_path, monkeypatch):
    path = LinuxNamedTestLibrary(tmp_path / "libd3d12core.so")
    path.write_bytes(b"\x7fELFfake-unit-test-not-an-executable-library")
    monkeypatch.setattr(lifetime, "WSL_D3D12_CORE", path)
    monkeypatch.setattr(lifetime.sys, "platform", "linux")
    return path


# 功能：
#   验证预加载副本与实际字节绑定，父环境和凭证不写入回执。
# 输入：
#   library：临时驱动对象。
# 输出：
#   None：不返回业务数据。
def test_render_library_lifetime_is_explicit_bound_and_does_not_mutate_parent(library):
    original = {"GALLIUM_DRIVER": "d3d12", "API_KEY": "not-in-receipt"}
    child, report = lifetime.prepare_render_process_environment(original)
    assert "LD_PRELOAD" not in original
    assert child["LD_PRELOAD"] == str(library)
    assert report["library_sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
    assert report["applied"] and not report["already_inherited"]
    assert "not-in-receipt" not in str(report)
    assert report["flight_qualification_granted"] is False


# 功能：
#   验证保留其他预加载库且再次应用不会重复添加驱动。
# 输入：
#   library：临时驱动对象。
# 输出：
#   None：不返回业务数据。
def test_other_explicit_preloads_preserved_and_retention_is_idempotent(library):
    original = {"GALLIUM_DRIVER": "d3d12", "LD_PRELOAD": "/trusted/other.so"}
    first, _ = lifetime.prepare_render_process_environment(original)
    second, report = lifetime.prepare_render_process_environment(first)
    assert first == second and report["already_inherited"]
    assert second["LD_PRELOAD"] == f"{library}:/trusted/other.so"
    assert original["LD_PRELOAD"] == "/trusted/other.so"


# 功能：
#   未显式选择 D3D12 时不得自行切换渲染器或添加 WSL 驱动。
# 输入：
#   library：临时驱动对象。
#   environment：其他渲染器或未选择渲染器的环境。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("environment", [{}, {"GALLIUM_DRIVER": "llvmpipe"},
    {"GALLIUM_DRIVER": "zink"}])
def test_other_renderers_never_load_wsl_driver(library, environment):
    child, report = lifetime.prepare_render_process_environment(environment)
    assert child == environment and not report["applied"]


# 功能：
#   WSL 库不存在时只报告未应用，不下载驱动或猜测其他路径。
# 输入：
#   library：用于构造缺失路径的临时驱动对象。
#   monkeypatch：替换模块驱动路径的测试工具。
# 输出：
#   None：不返回业务数据。
def test_missing_wsl_library_is_not_downloaded_or_guessed(library, monkeypatch):
    monkeypatch.setattr(lifetime, "WSL_D3D12_CORE", library.path.with_name("missing.so"))
    child, report = lifetime.prepare_render_process_environment({"GALLIUM_DRIVER": "d3d12"})
    assert "LD_PRELOAD" not in child and not report["applied"]


# 功能：
#   ELF 魔数不匹配的普通文件不能被当成可预加载驱动。
# 输入：
#   library：可改写内容的临时驱动对象。
# 输出：
#   None：不返回业务数据。
def test_wrong_library_content_rejected(library):
    library.write_bytes(b"not a shared library")
    with pytest.raises(ValueError, match="LIBRARY_INVALID"):
        lifetime.prepare_render_process_environment({"GALLIUM_DRIVER": "d3d12"})


# 功能：
#   Windows 宿主即使配置了 D3D12，也不得注入 Linux 库。
# 输入：
#   library：临时驱动对象。
#   monkeypatch：替换平台名称的测试工具。
# 输出：
#   None：不返回业务数据。
def test_windows_never_injects_linux_driver(library, monkeypatch):
    monkeypatch.setattr(lifetime.sys, "platform", "win32")
    child, report = lifetime.prepare_render_process_environment({"GALLIUM_DRIVER": "d3d12"})
    assert "LD_PRELOAD" not in child and not report["applied"]


# 功能：
#   从真实适配器源码检查渲染环境只用于 Gazebo 与副本，不用于 PX4、推理或控制进程。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_only_gazebo_and_render_replica_use_render_environment():
    # 检查实际调用参数，而非仅匹配环境变量声明，防止全局替换 env 时误扩大影响。
    source = Path(__file__).parents[1] / "src/dronedream_agent_core/gazebo_adapter.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    uses = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        for keyword in call.keywords:
            if (keyword.arg == "env" and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "gazebo_env"):
                outputs = [kw.value.id for kw in call.keywords if kw.arg == "stdout"
                           and isinstance(kw.value, ast.Name)]
                uses.extend(outputs)
    assert sorted(uses) == ["gazebo_log", "render_replica_log"]
