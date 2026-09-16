"""Exercise local Runtime settings and temporary credentials with synthetic values."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_runtime_manager import _execution_fixture, _manager, _runtime_resources
from test_runtime_qualification_boundaries import _RecordingInput

import dronedream_agent_app.runtime_manager as bridge


# 功能：
#   验证实时来源列表形状错误时返回空目录，而不是让界面查询抛出异常。
# 输入：
#   tmp_path：隔离测试目录。
#   sources：待拒绝的来源列表值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("sources", [None, 1, True, "camera", {}])
def test_invalid_live_source_list_is_ignored(tmp_path, sources):
    manager = _manager(tmp_path / "store", _runtime_resources(tmp_path / "resources"))
    (manager.store.root / "live-sources.json").write_text(json.dumps({"sources": sources}))
    assert manager._configured_live_sources() == []


# 功能：
#   验证实时显示数据也拒绝重复键和非有限值，不将含糊的遥测传给前端。
# 输入：
#   tmp_path：隔离测试目录。
#   payload：非法遥测 JSON。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("payload", ['{"x":1,"x":2}', '{"x":NaN}'])
def test_live_telemetry_rejects_ambiguous_json(tmp_path, payload):
    manager = _manager(tmp_path / "store", _runtime_resources(tmp_path / "resources"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "live-telemetry.json").write_text(payload)
    manager._active_runs["test-thread"] = bridge.ActiveRun(
        execution_id="test-execution",
        run_dir=run_dir,
        process=SimpleNamespace(poll=lambda: None),
        started_at="2026-09-14T00:00:00Z",
    )
    assert manager.live_telemetry("test-thread") is None


# 功能：
#   验证安装线程无法启动时恢复为明确失败状态，不永久显示仍在安装。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：模拟线程启动失败的测试工具。
# 输出：
#   None：不返回业务数据。
def test_setup_thread_start_failure_is_not_left_active(tmp_path, monkeypatch):
    manager = _manager(tmp_path / "store", _runtime_resources(tmp_path / "resources"))

    # 功能：
    #   模拟系统拒绝启动线程，不执行安装任务。
    # 输入：
    #   self：待启动的线程对象。
    # 输出：
    #   None：不返回业务数据。
    def fail_start(self):
        raise RuntimeError("simulated thread start failure")

    monkeypatch.setattr(bridge.threading.Thread, "start", fail_start)
    with pytest.raises(bridge.RuntimeBridgeError, match="SETUP_WORKER_START_FAILED"):
        manager.start_setup()
    progress = manager.setup_progress()
    assert progress["active"] is False
    assert progress["phase"] == "failed"
    assert progress["failed_phase"] == "queued"


# 功能：
#   验证会话凭据文件只允许新建，不得覆盖碰撞目标；不使用任何真实密钥。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_secret_session_never_overwrites_existing_file(tmp_path):
    path = tmp_path / "session.env"
    path.write_text("existing-owner-content")
    with pytest.raises(FileExistsError):
        bridge.RuntimeManager._write_secret_session(
            path,
            provider="fixture",
            model_id="fixture",
            grant="not-a-secret",
            gateway="https://example.invalid",
        )
    assert path.read_text() == "existing-owner-content"


# 功能：
#   验证专家目录损坏等配置失败发生时，不提前写入会话凭据或消耗执行授权。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：构造宿主执行替身的测试工具。
# 输出：
#   None：不返回业务数据。
def test_prelaunch_failure_does_not_write_secret_or_consume_authority(tmp_path, monkeypatch):
    manager, arguments = _execution_fixture(tmp_path, monkeypatch)
    catalog = manager.resource_root / "runtime" / "local-policy" / "catalog.json"
    catalog.write_text("{}")
    consumed = []

    # 功能：
    #   记录授权消费请求，避免测试替身改变真实存储中的授权状态。
    # 输入：
    #   authority：待消费的执行授权。
    #   execution_id：新执行标识。
    # 输出：
    #   None：不返回业务数据。
    def capture_consume(authority, *, execution_id):
        consumed.append(execution_id)

    monkeypatch.setattr(manager.store, "consume_execution_authority", capture_consume)
    with pytest.raises(bridge.RuntimeBridgeError, match="LOCAL_POLICY_CATALOG_INVALID"):
        manager.execute(**arguments)
    assert consumed == []
    assert list(manager.store.missions_root.rglob(".model-session-*.env")) == []


class _LaunchProcess:
    # 功能：
    #   创建不会启动命令的进程替身，可模拟退出时留下损坏的结果文件。
    # 输入：
    #   self：进程替身。
    #   run_dir：模拟证据目录。
    #   malformed_result：是否写入损坏结果并以零退出。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, run_dir, malformed_result=False):
        self.stdin = _RecordingInput()
        self.run_dir = run_dir
        self.malformed_result = malformed_result
        self.terminated = False
        self.reaped = False

    # 功能：
    #   模拟退出和状态回收；必要时留下损坏 JSON 供宿主失败处理使用。
    # 输入：
    #   self：进程替身。
    #   timeout：可选等待预算。
    # 输出：
    #   return_code：模拟进程退出码。
    def wait(self, timeout=None):
        self.reaped = True
        if self.malformed_result:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "workflow-result.json").write_text("{invalid")
        return_code = 0 if self.malformed_result else 1
        return return_code

    # 功能：
    #   记录终止动作，不操作任何真实进程。
    # 输入：
    #   self：进程替身。
    # 输出：
    #   None：不返回业务数据。
    def terminate(self):
        self.terminated = True


# 功能：
#   验证进程零退出但结果文件损坏时，任务最终标为失败而非永久执行中。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换实际进程创建的测试工具。
# 输出：
#   None：不返回业务数据。
def test_monitor_records_corrupt_result_as_failed(tmp_path, monkeypatch):
    manager, arguments = _execution_fixture(tmp_path, monkeypatch)

    # 功能：
    #   从宿主日志位置确定证据目录，返回模拟失败结果的进程。
    # 输入：
    #   args：进程启动参数。
    #   kwargs：进程输入输出配置。
    # 输出：
    #   process：受控进程替身。
    def start_process(*args, **kwargs):
        run_dir = Path(kwargs["stdout"].name).parent / "simulation"
        process = _LaunchProcess(run_dir, malformed_result=True)
        return process

    monkeypatch.setattr(bridge.subprocess, "Popen", start_process)
    manager.execute(**arguments)
    thread_id = arguments["thread_id"]
    deadline = time.monotonic() + 2
    while (
        manager.store.get_thread(thread_id)["state"] == "executing" or manager._active_runs
    ) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.store.get_thread(thread_id)["state"] == "failed"
    assert not manager._active_runs
    assert list(manager.store.missions_root.rglob(".model-session-*.env")) == []


# 功能：
#   验证执行跟踪线程无法启动时停止已创建进程并清理临时连接文件。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换进程与线程启动的测试工具。
# 输出：
#   None：不返回业务数据。
def test_monitor_start_failure_cleans_launched_process(tmp_path, monkeypatch):
    manager, arguments = _execution_fixture(tmp_path, monkeypatch)
    process = _LaunchProcess(tmp_path / "unused")
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *args, **kwargs: process)

    # 功能：
    #   模拟线程启动失败，不执行监视代码。
    # 输入：
    #   self：待启动的线程对象。
    # 输出：
    #   None：不返回业务数据。
    def fail_start(self):
        raise RuntimeError("simulated monitor startup failure")

    monkeypatch.setattr(bridge.threading.Thread, "start", fail_start)
    with pytest.raises(bridge.RuntimeBridgeError, match="EXECUTION_TRACKING_FAILED"):
        manager.execute(**arguments)
    assert process.reaped
    assert not manager._active_runs
    assert manager.store.get_thread(arguments["thread_id"])["state"] == "failed"
    assert list(manager.store.missions_root.rglob(".model-session-*.env")) == []


# 功能：
#   验证监视器等待异常后回收进程，并把本次任务标为失败。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换进程等待的测试工具。
# 输出：
#   None：不返回业务数据。
def test_monitor_wait_failure_reaps_process(tmp_path, monkeypatch):
    manager, arguments = _execution_fixture(tmp_path, monkeypatch)
    process = _LaunchProcess(tmp_path / "unused")
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *args, **kwargs: process)

    # 功能：
    #   普通等待时模拟系统错误，终止后的有界等待允许完成回收。
    # 输入：
    #   timeout：可选的回收等待预算。
    # 输出：
    #   return_code：终止后的失败退出码。
    def wait(timeout=None):
        if timeout is None:
            raise OSError("simulated wait failure")
        process.reaped = True
        return_code = -1
        return return_code

    monkeypatch.setattr(process, "wait", wait)
    manager.execute(**arguments)
    deadline = time.monotonic() + 2
    while manager._active_runs and time.monotonic() < deadline:
        time.sleep(0.01)
    assert process.reaped and process.terminated
    assert not manager._active_runs
    assert manager.store.get_thread(arguments["thread_id"])["state"] == "failed"


# 功能：
#   验证完成记录写入数据库失败时保留宿主证据，并在查询中明确暴露跟踪失败。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：注入完成消息写入失败的测试工具。
# 输出：
#   None：不返回业务数据。
def test_monitor_persistence_failure_is_observable(tmp_path, monkeypatch):
    manager, arguments = _execution_fixture(tmp_path, monkeypatch)
    process = _LaunchProcess(tmp_path / "unused")
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *args, **kwargs: process)

    # 功能：
    #   模拟完成消息无法保存，不写入任何用户数据。
    # 输入：
    #   args：待保存消息的位置参数。
    #   kwargs：待保存消息的具名参数。
    # 输出：
    #   None：不返回业务数据。
    def fail_save(*args, **kwargs):
        raise OSError("simulated database write failure")

    monkeypatch.setattr(manager.store, "append_message", fail_save)
    launched = manager.execute(**arguments)
    thread_id = arguments["thread_id"]
    deadline = time.monotonic() + 2
    while manager._active_runs[thread_id].tracking_error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    summary = manager.execution_evidence(thread_id)
    assert summary["state"] == "failed"
    assert summary["issue"] == "EXECUTION_COMPLETION_PERSISTENCE_FAILED"
    assert (Path(launched["execution_root"]) / "host-completion.json").is_file()
    assert list(manager.store.missions_root.rglob(".model-session-*.env")) == []
