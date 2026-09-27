"""Resource policy assertions are separate from inert protocol round trips."""

import sys
from types import SimpleNamespace

import pytest

from dronedream_agent_core import plugin_process as module
from dronedream_agent_core.plugin_contracts import PluginResourcePolicy


# 功能：
#   检查 POSIX 回调完整设置既定上限且系统拒绝时立即传播异常，不改当前解释器限额。
# 输入：
#   monkeypatch：仅替换模块平台标识和系统调用记录器的测试工具。
#   has_nproc、reject：模拟系统是否提供进程限额及是否拒绝设置。
# 输出：
#   None：无返回值。
@pytest.mark.parametrize("has_nproc", [False, True])
@pytest.mark.parametrize("reject", [False, True])
def test_posix_limits_remain_exact_and_fail_closed(monkeypatch, has_nproc, reject):
    calls = []

    # 功能：
    #   记录实际回调传入的资源上限，并按用例要求拒绝系统调用。
    # 输入：
    #   resource_id、limits：被设置的资源及软硬上限。
    # 输出：
    #   None：无返回值。
    def set_limit(resource_id, limits):
        calls.append((resource_id, limits))
        if reject:
            raise OSError("fixture resource denial")

    resource = SimpleNamespace(RLIMIT_AS=1, RLIMIT_CPU=2, setrlimit=set_limit)
    if has_nproc:
        resource.RLIMIT_NPROC = 3
    monkeypatch.setitem(sys.modules, "resource", resource)
    monkeypatch.setattr(module, "os", SimpleNamespace(name="posix"))
    policy = PluginResourcePolicy(memory_limit_mb=32, cpu_time_limit_seconds=5, process_limit=4)
    callback = module._posix_resource_limiter(policy)
    assert callable(callback)
    if reject:
        with pytest.raises(OSError, match="fixture resource denial"):
            callback()
        assert calls == [(1, (32 * 1024 * 1024, 32 * 1024 * 1024))]
    else:
        callback()
        expected = [(1, (32 * 1024 * 1024, 32 * 1024 * 1024)), (2, (5, 5))]
        if has_nproc:
            expected.append((3, (4, 4)))
        assert calls == expected


# 功能：
#   检查 POSIX 不会把普通进程限额冒充可用的系统级插件隔离。
# 输入：
#   tmp_path、monkeypatch：惰性路径及只替换平台和程序解析的测试工具。
# 输出：
#   None：无返回值。
def test_posix_strict_isolation_rejects_without_starting_process(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(module, "resolve_plugin_command", lambda *args: ["inert-fixture"])
    with pytest.raises(module.PluginProcessError, match="PLUGIN_OS_ISOLATOR_UNAVAILABLE"):
        module._isolated_command(plugin_root=tmp_path, command=["inert-fixture"],
                                 require_os_isolation=True, isolator_path=None)


# 功能：
#   检查测试资源夹具登记真实会话池；退出后夹具必须验证所有线程已停止。
# 输入：
#   owned_plugin_pools：本用例自动登记并回收的真实会话池列表。
# 输出：
#   None：无返回值。
def test_test_fixture_owns_real_pool_without_replacing_its_thread(owned_plugin_pools):
    pool = module.McpSessionPool(heartbeat_interval_seconds=3600)
    assert pool in owned_plugin_pools
    assert pool._heartbeat_thread.is_alive()
    # 故意交给统一退出夹具回收；夹具会用生产 close 并检查线程存活状态。
