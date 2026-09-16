"""Fake process and local HTTP boundary tests; never starts WSL or flies an asset."""

import io
import json
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from test_runtime_manager import _manager, _runtime_resources

import dronedream_agent_app.runtime_manager as bridge
from dronedream_agent_app.asset_import_service import AssetImportService
from dronedream_agent_app.server import create_app
from dronedream_agent_app.storage import AppStore, AssetImportError


# 功能：
#   建立隔离运行桥接器，只声明测试资源就绪，不连接真实 WSL 环境。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：替换运行就绪查询的测试工具。
# 输出：
#   fixture：桥接器、准备目录和证据目录组成的元组。
def _bridge_fixture(tmp_path, monkeypatch):
    manager = _manager(tmp_path / "store", _runtime_resources(tmp_path / "resources"))
    monkeypatch.setattr(
        manager,
        "status",
        lambda: {"runtime_available": True, "resources_ready": True, "provisioned": True},
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "qualification-plan.json").write_text("{}")
    fixture = manager, prepared, tmp_path / "evidence"
    return fixture


class _RecordingInput(io.StringIO):
    # 功能：
    #   保留测试输入内容，便于检查取消前是否已经发送飞行脚本。
    # 输入：
    #   self：内存模拟输入流。
    # 输出：
    #   None：不返回业务数据。
    def close(self):
        pass


class _FakeProcess:
    # 功能：
    #   创建不会启动任何子进程的测试替身。
    # 输入：
    #   self：测试进程对象。
    #   run_dir：模拟运行写入证据的位置。
    #   payload：需要模拟的证据字节。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, run_dir, payload=b'{"status":"verified"}'):
        self.stdin = _RecordingInput()
        self.run_dir = run_dir
        self.payload = payload
        self.terminated = False

    # 功能：
    #   模拟进程退出；被终止时不生成证据，正常返回时写入指定测试内容。
    # 输入：
    #   self：测试进程对象。
    #   timeout：调用方传入的等待秒数。
    # 输出：
    #   return_code：模拟进程退出码。
    def wait(self, timeout):
        if not self.terminated:
            self.run_dir.mkdir()
            (self.run_dir / "mission_evidence.json").write_bytes(self.payload)
        return_code = 0
        return return_code

    # 功能：
    #   记录宿主发出的终止动作，便于验证未启动脚本的进程已清理。
    # 输入：
    #   self：测试进程对象。
    # 输出：
    #   None：不返回业务数据。
    def terminate(self):
        self.terminated = True


# 功能：
#   验证取消发生在进程创建前或创建过程中，都不会发送飞行脚本。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：替换进程创建的测试工具。
#   already_cancelled：是否在调用桥接器前就已取消。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("already_cancelled", [True, False])
def test_cancel_before_registration_never_sends_script(tmp_path, monkeypatch, already_cancelled):
    manager, prepared, evidence = _bridge_fixture(tmp_path, monkeypatch)
    cancelled = [already_cancelled]
    process = _FakeProcess(evidence)
    created = []

    # 功能：
    #   模拟创建过程中用户发来取消请求。
    # 输入：
    #   args：进程启动参数。
    #   kwargs：进程启动选项。
    # 输出：
    #   process：不会实际执行命令的进程替身。
    def start_process(*args, **kwargs):
        created.append(True)
        cancelled[0] = True
        return process

    monkeypatch.setattr(bridge.subprocess, "Popen", start_process)
    with pytest.raises(bridge.RuntimeBridgeError, match="QUALIFICATION_CANCELLED"):
        manager.run_asset_pair_qualification(
            job_id="asset-qualification-job-" + "a" * 24,
            work_root=prepared,
            run_dir=evidence,
            cancel_requested=lambda: cancelled[0],
        )
    assert bool(created) is not already_cancelled
    assert process.stdin.getvalue() == ""
    assert process.terminated is not already_cancelled
    assert not evidence.exists()
    assert not manager._active_asset_qualification_runs


# 功能：
#   验证运行超时必须是范围内有限数值，不接受 NaN、无穷、布尔或数字字符串。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：替换运行就绪查询的测试工具。
#   timeout：待拒绝的超时值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), True, "900", 10**1000])
def test_qualification_timeout_rejects_non_finite_or_coerced_values(tmp_path, monkeypatch, timeout):
    manager, prepared, evidence = _bridge_fixture(tmp_path, monkeypatch)
    with pytest.raises(bridge.RuntimeBridgeError, match="TIMEOUT_INVALID"):
        manager.run_asset_pair_qualification(
            job_id="asset-qualification-job-" + "a" * 24,
            work_root=prepared,
            run_dir=evidence,
            timeout_seconds=timeout,
        )


# 功能：
#   验证运行证据不能用重复键将失败状态覆盖成通过状态。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：替换进程创建的测试工具。
# 输出：
#   None：不返回业务数据。
def test_runtime_evidence_rejects_duplicate_status(tmp_path, monkeypatch):
    manager, prepared, evidence = _bridge_fixture(tmp_path, monkeypatch)
    process = _FakeProcess(evidence, b'{"status":"failed","status":"verified"}')
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(bridge.RuntimeBridgeError, match="EVIDENCE_INVALID"):
        manager.run_asset_pair_qualification(
            job_id="asset-qualification-job-" + "a" * 24,
            work_root=prepared,
            run_dir=evidence,
        )


# 功能：
#   验证原子 JSON 写入的临时名碰撞不会破坏已有文件，也不会更新最终目标。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：固定随机标识的测试工具。
# 输出：
#   None：不返回业务数据。
def test_runtime_atomic_json_preserves_colliding_file(tmp_path, monkeypatch):
    identifier = UUID("11111111-1111-1111-1111-111111111111")
    monkeypatch.setattr(bridge, "uuid4", lambda: identifier)
    target = tmp_path / "request.json"
    staging = tmp_path / f".request.json-{identifier.hex}.tmp"
    target.write_text('{"previous":true}')
    staging.write_text("another writer")
    with pytest.raises(FileExistsError):
        bridge.RuntimeManager._atomic_json(target, {"replacement": True})
    assert json.loads(target.read_bytes()) == {"previous": True}
    assert staging.read_text() == "another writer"


# 功能：
#   验证导出失败后的 HTTP 清理不会删除发生碰撞的外部文件。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：替换导出动作的测试工具。
# 输出：
#   None：不返回业务数据。
def test_failed_http_export_preserves_unowned_destination(tmp_path, monkeypatch):
    paths: list[Path] = []

    # 功能：
    #   模拟其他写入者抢先创建导出目标，随后导出服务拒绝覆盖。
    # 输入：
    #   self：当前导入服务。
    #   kwargs：导出资产身份及目标路径。
    # 输出：
    #   None：不返回业务数据。
    def collision_export(self, **kwargs):
        path = kwargs["destination"]
        paths.append(path)
        path.write_bytes(b"another writer")
        raise AssetImportError("ASSET_VERSION_EXPORT_DESTINATION_EXISTS")

    monkeypatch.setattr(AssetImportService, "export_version", collision_export)
    client = TestClient(create_app(store=AppStore(tmp_path / "store"), token="a" * 64))
    response = client.get(
        "/v1/asset-versions/example/" + "b" * 64 + "/package",
        headers={"Authorization": "Bearer " + "a" * 64},
    )
    assert response.status_code == 422
    assert paths[0].read_bytes() == b"another writer"


class _StubbornProcess(_FakeProcess):
    # 功能：
    #   模拟不响应普通终止、必须强制结束后才能回收的进程。
    # 输入：
    #   self：进程替身。
    #   run_dir：测试证据目录。
    #   kill_fails：是否模拟强制结束仍失败。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, run_dir, kill_fails=False):
        super().__init__(run_dir)
        self.killed = False
        self.reaped = False
        self.kill_fails = kill_fails

    # 功能：
    #   强制结束前持续模拟等待超时，结束后记录已经回收。
    # 输入：
    #   self：进程替身。
    #   timeout：调用方等待预算。
    # 输出：
    #   return_code：强制结束后的模拟退出码。
    def wait(self, timeout):
        if not self.killed:
            raise bridge.subprocess.TimeoutExpired("fake-wsl", timeout)
        self.reaped = True
        return_code = -9
        return return_code

    # 功能：
    #   记录强制结束操作，或模拟无法终止的进程以检查记录保留。
    # 输入：
    #   self：进程替身。
    # 输出：
    #   None：不返回业务数据。
    def kill(self):
        if self.kill_fails:
            raise OSError("simulated termination failure")
        self.killed = True


# 功能：
#   验证进程登记后的取消读取异常、脚本写入异常和运行超时都执行终止与回收。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：替换进程及取消回调的测试工具。
#   failure：要模拟的异常发生位置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failure", ["cancel_probe", "stdin_write", "timeout"])
def test_registered_process_is_reaped_on_failure(tmp_path, monkeypatch, failure):
    manager, prepared, evidence = _bridge_fixture(tmp_path, monkeypatch)
    process = _StubbornProcess(evidence)
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *args, **kwargs: process)
    probes = []

    # 功能：
    #   第一次允许启动，第二次按用例模拟取消状态读取失败。
    # 输入：
    #   无。
    # 输出：
    #   cancelled：正常情况下没有取消请求。
    def cancellation_probe():
        probes.append(True)
        if failure == "cancel_probe" and len(probes) == 2:
            raise RuntimeError("simulated cancellation query failure")
        cancelled = False
        return cancelled

    # 功能：
    #   模拟脚本管道写入中断，确保清理不依赖脚本已经启动。
    # 输入：
    #   script：待写入的运行脚本。
    # 输出：
    #   None：不返回业务数据。
    def interrupted_write(script):
        raise BrokenPipeError("simulated pipe failure")

    if failure == "stdin_write":
        monkeypatch.setattr(process.stdin, "write", interrupted_write)
    expected = {
        "cancel_probe": RuntimeError,
        "stdin_write": BrokenPipeError,
        "timeout": bridge.RuntimeBridgeError,
    }[failure]
    with pytest.raises(expected):
        manager.run_asset_pair_qualification(
            job_id="asset-qualification-job-" + "c" * 24,
            work_root=prepared,
            run_dir=evidence,
            cancel_requested=cancellation_probe,
        )
    assert process.terminated and process.killed and process.reaped
    assert not manager._active_asset_qualification_runs


# 功能：
#   验证无法确认进程结束时保留活动记录，防止同一任务再次启动。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：替换进程创建的测试工具。
# 输出：
#   None：不返回业务数据。
def test_failed_process_cleanup_keeps_active_registration(tmp_path, monkeypatch):
    manager, prepared, evidence = _bridge_fixture(tmp_path, monkeypatch)
    process = _StubbornProcess(evidence, kill_fails=True)
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *args, **kwargs: process)
    job_id = "asset-qualification-job-" + "d" * 24
    with pytest.raises(bridge.RuntimeBridgeError, match="CLEANUP_FAILED"):
        manager.run_asset_pair_qualification(job_id=job_id, work_root=prepared, run_dir=evidence)
    assert manager._active_asset_qualification_runs[job_id].process is process
    with pytest.raises(bridge.RuntimeBridgeError, match="ALREADY_RUNNING"):
        manager.run_asset_pair_qualification(job_id=job_id, work_root=prepared, run_dir=evidence)
