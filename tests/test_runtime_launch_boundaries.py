"""Launch parameter, evidence isolation and host cleanup tests without WSL."""

import subprocess
from types import SimpleNamespace

import pytest

from dronedream_agent_app.runtime_launch_settings import (
    bounded_integer,
    packaged_executor,
    positive_seconds,
)
from dronedream_agent_app.runtime_manager import RuntimeBridgeError
from tests.test_runtime_manager import _manager, _runtime_resources
from tests.test_runtime_qualification_boundaries import _StubbornProcess

pytestmark = pytest.mark.usefixtures("isolated_wsl_host_paths")


# 功能：
#   验证整数配置只接受范围内的整数，不通过隐式转换接受错误类型。
# 输入：
#   value：边界或错误类型的整数配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, "20", 20.0, 19, 1001, None])
def test_integer_setting_is_strict(value):
    with pytest.raises(ValueError):
        bounded_integer(value, 20, 1000)
    assert bounded_integer(20, 20, 1000) == 20
    assert bounded_integer(1000, 20, 1000) == 1000


# 功能：
#   验证时间配置拒绝非数值、非有限值、非正数及过大整数。
# 输入：
#   value：不符合时间契约的输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "value",
    [True, "0.1", float("nan"), float("inf"), 0, -1, 10**1000],
    ids=["bool", "text", "nan", "infinite", "zero", "negative", "overflow"],
)
def test_time_setting_is_finite_and_positive(value):
    with pytest.raises(ValueError):
        positive_seconds(value)
    assert positive_seconds(0.25) == 0.25


# 功能：
#   验证插件执行器路径不能逃逸、别名化或指向不存在的文件。
# 输入：
#   tmp_path：隔离测试目录。
#   relative：待拒绝的执行器路径。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("relative", ["../outside.py", "./executor.py", "missing.py", "", None])
def test_executor_is_confined_to_packaged_resource(tmp_path, relative):
    (tmp_path / "executor.py").write_text("# fixture")
    with pytest.raises((OSError, ValueError)):
        packaged_executor(tmp_path, relative)
    assert packaged_executor(tmp_path, "executor.py") == tmp_path / "executor.py"


# 功能：
#   验证第二次 WSL 探测失败时报告未知部署状态，不令状态查询直接崩溃。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换 WSL 探测的测试工具。
# 输出：
#   None：不返回业务数据。
def test_provision_probe_failure_is_reported(tmp_path, monkeypatch):
    manager = _manager(tmp_path / "store", _runtime_resources(tmp_path / "resources"))
    probes = []

    # 功能：
    #   首次模拟基础环境可用，随后模拟依赖探测超时。
    # 输入：
    #   script：探测脚本。
    #   timeout：探测预算。
    # 输出：
    #   result：首次成功的进程结果。
    def probe(script, *, timeout):
        probes.append(script)
        if len(probes) > 1:
            raise RuntimeBridgeError("simulated probe timeout")
        result = subprocess.CompletedProcess([], 0, "", "")
        return result

    monkeypatch.setattr(manager, "_bash", probe)
    status = manager.status()
    assert status["runtime_available"] is True
    assert status["provisioned"] is False
    assert status["issue"] == "RUNTIME_PROVISION_PROBE_FAILED"


# 功能：
#   验证安装占用时拒绝另一个入口，异常结束后释放安装锁。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换实际安装步骤的测试工具。
# 输出：
#   None：不返回业务数据。
def test_provision_entries_share_an_exception_safe_lock(tmp_path, monkeypatch):
    manager = _manager(tmp_path / "store", tmp_path / "resources")
    manager._provision_lock.acquire()
    try:
        with pytest.raises(RuntimeBridgeError, match="ALREADY_ACTIVE"):
            manager.provision()
    finally:
        manager._provision_lock.release()

    # 功能：
    #   模拟安装步骤发生异常。
    # 输入：
    #   report：可选进度接收器。
    # 输出：
    #   None：不返回业务数据。
    def fail_install(report=None):
        raise RuntimeBridgeError("simulated install failure")

    monkeypatch.setattr(manager, "_perform_provision", fail_install)
    with pytest.raises(RuntimeBridgeError, match="simulated install failure"):
        manager.provision()
    assert manager._provision_lock.acquire(blocking=False)
    manager._provision_lock.release()


# 功能：
#   验证任务记录不能引用同一 missions 根目录下另一任务的验收材料。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_execution_evidence_cannot_read_another_task(tmp_path):
    manager = _manager(tmp_path / "store", tmp_path / "resources")
    thread_id = manager.store.create_thread("evidence boundary", "gpt-5.4")["thread_id"]
    execution_id = "execution-" + "a" * 32
    manager.store.append_message(
        thread_id,
        role="assistant",
        kind="status",
        content="fixture completion",
        metadata={
            "execution_id": execution_id,
            "run_dir": str(
                manager.store.missions_root / "another-task" / execution_id / "simulation"
            ),
        },
    )
    with pytest.raises(RuntimeBridgeError, match="EXECUTION_EVIDENCE_PATH_INVALID"):
        manager.execution_evidence(thread_id)


# 功能：
#   验证一个进程无法回收时，关闭流程仍继续清理其他进程并保留失败登记。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换关闭消息入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_shutdown_attempts_every_registered_process(tmp_path, monkeypatch):
    manager = _manager(tmp_path / "store", tmp_path / "resources")
    stuck = _StubbornProcess(tmp_path / "stuck", kill_fails=True)
    reaped = _StubbornProcess(tmp_path / "reaped")
    manager._active_runs["stuck"] = SimpleNamespace(process=stuck)
    manager._active_runs["reaped"] = SimpleNamespace(process=reaped)
    monkeypatch.setattr(manager, "_thread_locale", lambda identifier: "en-US")
    monkeypatch.setattr(manager, "submit_message", lambda *args: {"accepted": True})
    with pytest.raises(RuntimeBridgeError, match="RUNTIME_SHUTDOWN_CLEANUP_FAILED:stuck"):
        manager.shutdown()
    assert reaped.reaped
    assert list(manager._active_runs) == ["stuck"]
