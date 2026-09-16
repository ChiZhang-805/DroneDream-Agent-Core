"""Read actual desktop stop files through both consumers without launching a vehicle."""

from types import SimpleNamespace

import pytest

from dronedream_agent_core.gazebo_adapter import _read_external_abort_evidence
from tests.test_px4_executor_integrity import executor_module
from tests.test_runtime_manager import _manager


# 功能：
#   核对桌面发出的停止文件被真实执行器识别为合作停止，证据读取器保留同一原因与来源。
# 输入：
#   tmp_path：隔离测试目录。
#   reason：桌面入口的内部停止原因。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("reason", [
    "qualification_cancelled_by_user", "execution_tracking_failed", "application_shutdown",
])
def test_stop_writer_matches_executor_and_evidence_reader(tmp_path, reason):
    manager = _manager(tmp_path / "store", tmp_path / "resources")
    run_dir = tmp_path / "run"
    source = "desktop:execution:fixture"
    manager._request_runtime_abort(run_dir, reason=reason, source=source)
    path = run_dir / "live_abort.request.json"
    assert _read_external_abort_evidence(path) == {
        "reason": reason, "world_paused": False, "source": source,
    }
    executor = executor_module()
    with pytest.raises(executor.ExternalSafetyAbort) as stopped:
        executor._raise_if_external_abort_requested(path)
    assert stopped.value.world_paused is False
    assert reason in str(stopped.value)


# 功能：
#   后发桌面停止不覆盖先发安全停止，包括已暂停世界及损坏但必须保留的请求。
# 输入：
#   tmp_path：隔离测试目录。
#   previous：先前安全模块已发布的原始字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("previous", [
    b'{"reason":"collision-stop","world_paused":true}',
    b'{"reason":"earlier-stop","world_paused":false}',
    b'{"damaged":',
])
def test_first_stop_request_is_not_overwritten(tmp_path, previous):
    manager = _manager(tmp_path / "store", tmp_path / "resources")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "live_abort.request.json"
    path.write_bytes(previous)
    manager._request_runtime_abort(run_dir, reason="application_shutdown", source="desktop")
    assert path.read_bytes() == previous
    assert list(run_dir.iterdir()) == [path]


# 功能：
#   非法原因在创建运行目录之前被拒绝，不能发布执行器不接受的请求。
# 输入：
#   tmp_path：隔离测试目录。
#   reason：空白、非字符串或超过长度上限的原因。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("reason", ["", " ", None, False, "x" * 241])
def test_bad_reason_never_creates_stop_file(tmp_path, reason):
    manager = _manager(tmp_path / "store", tmp_path / "resources")
    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="RUNTIME_ABORT_REASON_INVALID"):
        manager._request_runtime_abort(run_dir, reason=reason, source="desktop")
    assert not run_dir.exists()


# 功能：
#   关闭时消息通道失败仍发布有效停止文件，且进程回收前文件已经可被执行器读取。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换消息通道与进程等待的测试工具。
# 输出：
#   None：不返回业务数据。
def test_shutdown_fallback_delivers_stop_before_wait(tmp_path, monkeypatch):
    manager = _manager(tmp_path / "store", tmp_path / "resources")
    run_dir = tmp_path / "run"
    executor = executor_module()
    checked = []

    # 功能：
    #   模拟应用消息通道关闭，强制进入文件停止通道。
    # 输入：
    #   args：原消息入口的位置参数。
    # 输出：
    #   None：不返回业务数据。
    def unavailable(*args):
        raise RuntimeError("message channel closed")

    # 功能：
    #   不启动子进程，等待入口实际读取桌面停止文件后模拟退出。
    # 输入：
    #   timeout：管理器提供的等待预算。
    # 输出：
    #   status：模拟进程退出码。
    def wait(*, timeout):
        path = run_dir / "live_abort.request.json"
        assert executor._read_external_abort_request(path) == ("application_shutdown", False)
        checked.append(_read_external_abort_evidence(path))
        status = 0
        return status

    manager._active_runs["thread-fixture"] = SimpleNamespace(
        run_dir=run_dir, execution_id="fixture", process=SimpleNamespace(wait=wait),
    )
    monkeypatch.setattr(manager, "_thread_locale", lambda identifier: "en-US")
    monkeypatch.setattr(manager, "submit_message", unavailable)
    manager.shutdown()
    assert len(checked) == 1
    assert checked[0]["source"] == "desktop:execution:fixture"
    assert not manager._active_runs


# 功能：
#   已不存在的验收任务取消不创建目录或错误的停止文件。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_unknown_qualification_cancel_has_no_run_side_effect(tmp_path):
    manager = _manager(tmp_path / "store", tmp_path / "resources")
    assert manager.cancel_asset_pair_qualification("missing") is False
    assert not list(tmp_path.rglob("live_abort.request.json"))
