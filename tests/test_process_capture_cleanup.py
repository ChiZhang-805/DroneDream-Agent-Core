"""Exercise cleanup failures with owned process doubles, never terminate external processes."""

import io
from types import SimpleNamespace

import pytest

from dronedream_agent_core import process_capture as module


# 功能：
#   构造可记录清理顺序的进程、Job 和管道替身，不启动或查找任何系统进程。
# 输入：
#   failed_step：需要注入故障的清理步骤，为空表示所有步骤成功。
# 输出：
#   result：进程替身、Job 替身以及清理事件列表。
def _fixture(failed_step=None):
    events = []

    # 功能：
    #   记录一个清理步骤，并在指定步骤注入可恢复的系统错误。
    # 输入：
    #   name：当前执行的清理步骤名称。
    # 输出：
    #   None：不返回业务数据。
    def record(name):
        events.append(name)
        if name == failed_step:
            raise OSError(f"fixture {name}")

    class Pipe(io.BytesIO):
        # 功能：
        #   在首个管道关闭时模拟错误，其余管道使用真实内存流关闭语义。
        # 输入：
        #   self：内存管道替身。
        # 输出：
        #   None：不返回业务数据。
        def close(self):
            if self is process.stdin:
                record("stdin")
            super().close()

    process = SimpleNamespace(
        pid=123,
        poll=lambda: None,
        kill=lambda: record("kill"),
        wait=lambda timeout: record("wait"),
        stdin=Pipe(),
        stdout=Pipe(),
        stderr=Pipe(),
    )
    job = SimpleNamespace(close=lambda: record("job"))
    result = process, job, events
    return result


# 功能：
#   验证终止、Job 关闭、等待或首个管道关闭失败时，仍尝试其余资源清理并报告失败。
# 输入：
#   failed_step：要注入的故障点。
#   monkeypatch：隔离平台分支的测试工具。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failed_step", ["kill", "job", "wait", "stdin"])
def test_cleanup_continues_after_one_step_fails(failed_step, monkeypatch):
    process, job, events = _fixture(failed_step)
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    with pytest.raises(module.PluginProcessError, match="CLEANUP_FAILED") as caught:
        module._cleanup_capture(process, job, [])
    assert events == ["kill", "job", "wait", "stdin"]
    assert len(caught.value.__cause__.exceptions) == 1
    assert process.stdout.closed and process.stderr.closed
    # 释放故障夹具自己的内存流，不通过仍在故障注入状态的方法重复 close。
    io.BytesIO.close(process.stdin)


# 功能：
#   验证通信线程未退出时不从主线程关闭其可能持锁的管道，同时明确报告失败。
# 输入：
#   monkeypatch：隔离平台分支的测试工具。
# 输出：
#   None：不返回业务数据。
def test_live_channel_does_not_trigger_blocking_pipe_close(monkeypatch):
    process, job, events = _fixture()
    thread = SimpleNamespace(join=lambda timeout: events.append("join"), is_alive=lambda: True)
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    with pytest.raises(module.PluginProcessError, match="CLEANUP_FAILED") as caught:
        module._cleanup_capture(process, job, [thread])
    assert events == ["kill", "job", "wait", "join"]
    assert not process.stdin.closed
    assert any(isinstance(error, TimeoutError) for error in caught.value.__cause__.exceptions)
    for pipe in (process.stdin, process.stdout, process.stderr):
        io.BytesIO.close(pipe)


# 功能：
#   验证 POSIX 清理仅针对本次进程组，组已不存在时仍正常完成管道回收。
# 输入：
#   monkeypatch：替换进程组终止入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_missing_owned_posix_group_still_closes_pipes(monkeypatch):
    process, _job, events = _fixture()

    # 功能：
    #   模拟本次进程组已经退出，检查目标身份后抛出预期的不存在错误。
    # 输入：
    #   pid：准备清理的进程组编号。
    #   signal：用于清理的终止信号。
    # 输出：
    #   None：不返回业务数据。
    def absent(pid, signal):
        assert pid == process.pid
        assert signal == module.signal.SIGKILL
        events.append("group")
        raise ProcessLookupError("already exited")

    monkeypatch.setattr(module, "os", SimpleNamespace(name="posix", killpg=absent))
    monkeypatch.setattr(module, "signal", SimpleNamespace(SIGKILL=9))
    module._cleanup_capture(process, None, [])
    assert events == ["group", "wait", "stdin"]
    assert all(pipe.closed for pipe in (process.stdin, process.stdout, process.stderr))
