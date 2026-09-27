"""Owned test-resource cleanup and opt-in OS doubles; no production fallback is installed."""

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest


# 功能：
#   1. 登记每个用例实际创建的会话池，保留真实构造、线程与协议行为。
#   2. 用例退出时回收全部登记实例，阻止未触发 API lifespan 的测试遗留后台线程。
# 输入：
#   monkeypatch：在当前用例结束后恢复构造器的测试工具。
# 输出：
#   instances：当前用例创建的会话池列表。
@pytest.fixture(autouse=True)
def owned_plugin_pools(monkeypatch):
    from dronedream_agent_core.plugin_process import McpSessionPool

    original_init = McpSessionPool.__init__
    original_close = McpSessionPool.close
    instances = []

    # 功能：
    #   仅在真实构造成功后登记实例，不替换参数校验或线程启动行为。
    # 输入：
    #   instance、args、kwargs：正在构造的会话池及原始参数。
    # 输出：
    #   None：无返回值。
    def register(instance, *args, **kwargs):
        original_init(instance, *args, **kwargs)
        instances.append(instance)

    monkeypatch.setattr(McpSessionPool, "__init__", register)
    yield instances
    failures = []
    for instance in reversed(instances):
        try:
            original_close(instance)
            if instance._heartbeat_thread.is_alive():
                raise RuntimeError("Test left a plugin heartbeat thread running")
        except Exception as error:
            # 一处回收失败不能跳过其他实例；最后统一让测试失败，不静默丢弃问题。
            failures.append(error)
    if failures:
        raise ExceptionGroup("Test plugin pool cleanup failed", failures)


@dataclass
class _MemoryOnlyVault:
    values: dict[str, str] = field(default_factory=dict)

    # 功能：
    #   在当前测试对象内保存测试凭证，不创建文件或读取用户凭证。
    # 输入：
    #   self、name、secret：当前替身、测试标识及测试秘密。
    # 输出：
    #   None：无返回值。
    def put(self, name: str, secret: str) -> None:
        self.values[name] = secret

    # 功能：
    #   读取当前测试对象已有的凭证，未保存的标识抛出 KeyError。
    # 输入：
    #   self、name：当前替身及测试标识。
    # 输出：
    #   secret：对应测试秘密。
    def get(self, name: str) -> str:
        secret = self.values[name]
        return secret

    # 功能：
    #   幂等移除测试对象中的一个凭证。
    # 输入：
    #   self、name：当前替身及测试标识。
    # 输出：
    #   None：无返回值。
    def delete(self, name: str) -> None:
        self.values.pop(name, None)


# 功能：
#   1. 为显式选择此夹具的非 Windows API 测试隔离操作系统凭证依赖。
#   2. 保持真实服务与路由逻辑；Windows 测试仍使用原生凭证库。
# 输入：
#   monkeypatch：测试结束后自动恢复服务构造器的测试工具。
# 输出：
#   None：无返回值。
@pytest.fixture
def isolated_server_credentials(monkeypatch):
    if os.name == "nt":
        return
    from dronedream_agent_app import server

    model_service = server.CustomModelService
    connector_service = server.ConnectorCredentialService

    # 功能：
    #   为真实自定义模型服务注入独立的内存测试凭证库。
    # 输入：
    #   store：当前测试的数据目录服务。
    # 输出：
    #   service：除凭证后端外保留真实逻辑的模型服务。
    def models(store):
        service = model_service(store, _MemoryOnlyVault())
        return service

    # 功能：
    #   保留调用者显式提供的凭证替身，仅为缺省值补入独立内存库。
    # 输入：
    #   store、vault：当前测试的数据服务及可选凭证替身。
    # 输出：
    #   service：保留真实授权、存取及撤销逻辑的连接器服务。
    def connectors(store, vault=None):
        selected = vault if vault is not None else _MemoryOnlyVault()
        service = connector_service(store, selected)
        return service

    monkeypatch.setattr(server, "CustomModelService", models)
    monkeypatch.setattr(server, "ConnectorCredentialService", connectors)


# 功能：
#   为显式选择此夹具的模拟 WSL 进程测试隔离 Windows 盘符转换。
# 输入：
#   monkeypatch：测试结束后恢复路径转换器的测试工具。
# 输出：
#   None：无返回值。
@pytest.fixture
def isolated_wsl_host_paths(monkeypatch):
    if os.name != "nt":
        from dronedream_agent_app import runtime_manager

        # 此组测试的进程已由用例替换，不会执行命令；Linux 临时目录不冒充 Windows 盘符。
        # Windows 上仍测试真实盘符映射，生产转换器没有接纳不受支持路径的回退。
        monkeypatch.setattr(runtime_manager, "_wsl_path", Path.as_posix)


# 功能：
#   复制当前 ROS 和执行器源码到独立 Git 夹具，避免借用用户工作树的跨系统子模块元数据。
# 输入：
#   tmp_path：当前测试拥有的临时目录。
# 输出：
#   repository：包含真实当前源码、独立提交和无凭证来源地址的测试仓库。
@pytest.fixture
def isolated_source_repository(tmp_path):
    from scripts.write_runtime_provenance import RUNTIME_SOURCES

    original = Path(__file__).resolve().parents[1]
    repository = tmp_path / "source-fixture"
    repository.mkdir()
    source_paths = [original / name for name, _ in RUNTIME_SOURCES.values()]
    source_paths.append(original / "scripts/run_pluginized_runtime_acceptance.py")
    source_paths.extend(path for path in (original / "ros_ws/src").rglob("*")
                        if path.is_file() and "__pycache__" not in path.parts)
    for source in source_paths:
        if source.is_symlink():
            raise ValueError("Source fixture cannot borrow linked files")
        target = repository / source.relative_to(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # 不读全局 Git 配置、不运行用户 hooks、不签名，也不复制原仓库的 .git 或子模块。
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    empty_hooks = tmp_path / "empty-test-hooks"
    empty_hooks.mkdir()
    arguments = ["git", "-c", f"core.hooksPath={empty_hooks}", "-c", "core.autocrlf=false",
                 "-c", "commit.gpgsign=false", "-c", "user.name=Source fixture",
                 "-c", "user.email=fixture@example.invalid"]
    for command in (["init", "--quiet"], ["add", "."],
                    ["commit", "--quiet", "-m", "Isolated source fixture"],
                    ["remote", "add", "origin", "https://github.com/example/source-fixture.git"]):
        subprocess.run([*arguments, *command], cwd=repository, env=environment,
                       check=True, capture_output=True, timeout=30)
    return repository
