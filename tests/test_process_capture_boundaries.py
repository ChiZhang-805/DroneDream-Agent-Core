"""Real fixture subprocesses test bounded output/deadlines without invoking user executables."""

import os
import subprocess
import sys

import pytest

from dronedream_agent_core.plugin_contracts import PluginResourcePolicy
from dronedream_agent_core.process_capture import capture_process


# 功能：
#   仅以测试解释器执行固定夹具脚本，验证真实管道与进程行为，不加载用户配置。
# 输入：
#   script：测试中显式给定的 Python 脚本。
#   stdin：夹具进程的输入字节。
#   limit：输入及合计输出的字节预算。
#   timeout：执行与管道通信的总秒数预算。
# 输出：
#   result：夹具进程的退出码和分别捕获的输出。
def _capture(script, *, stdin=b"", limit=4096, timeout=5):
    result = capture_process(
        [sys.executable, "-c", script],
        stdin=stdin,
        maximum_bytes=limit,
        timeout=timeout,
        environment={"SYSTEMROOT": os.environ.get("SYSTEMROOT", ""), "PATH": ""},
        resource_policy=PluginResourcePolicy(),
    )
    return result


# 功能：
#   验证非零退出码及两个输出流分别保留，不把程序自行报告的失败改成成功。
# 输入：
#   无输入参数。
# 输出：
#   None：不返回业务数据。
def test_capture_preserves_exit_and_separate_streams():
    result = _capture("import sys; print('out'); print('err',file=sys.stderr); sys.exit(2)")
    assert result.returncode == 2
    assert result.stdout.strip() == b"out"
    assert result.stderr.strip() == b"err"


# 功能：
#   验证两个各未超限的输出流在合计超限时仍被拒绝。
# 输入：
#   无输入参数。
# 输出：
#   None：不返回业务数据。
def test_capture_limits_combined_stream_bytes():
    with pytest.raises(ValueError, match="OUTPUT_LIMIT"):
        _capture("import sys; sys.stdout.write('a'*3000); sys.stderr.write('b'*3000)")


# 功能：
#   验证子进程不读取大输入时，填满的管道不会阻止主线程执行超时清理。
# 输入：
#   无输入参数。
# 输出：
#   None：不返回业务数据。
def test_unread_stdin_does_not_block_deadline_owner():
    with pytest.raises(subprocess.TimeoutExpired):
        _capture("import time; time.sleep(10)", stdin=b"x" * 100_000, limit=100_000, timeout=0.2)


# 功能：
#   验证子进程主动忽略输入并正常退出时，输入断管不会覆盖真实退出结果。
# 输入：
#   无输入参数。
# 输出：
#   None：不返回业务数据。
def test_child_that_does_not_read_stdin_can_exit_cleanly():
    result = _capture("print('done')", stdin=b"x" * 100_000, limit=100_000)
    assert result.returncode == 0
    assert result.stdout.strip() == b"done"


# 功能：
#   验证非法字节预算在启动解释器前被拒绝。
# 输入：
#   limit：待拒绝的预算值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("limit", [True, -1, 0, "4096"])
def test_invalid_capture_budget_is_rejected_before_spawn(limit):
    with pytest.raises(ValueError, match="SIZE_INVALID"):
        _capture("raise AssertionError('must not run')", limit=limit)


# 功能：
#   验证超大整数、非有限值及布尔时限返回一致的配置错误，且不创建子进程。
# 输入：
#   timeout：待拒绝的总时限。
#   monkeypatch：拦截进程创建的测试工具。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("timeout", [10**400, float("inf"), float("nan"), True, 0, -1])
def test_invalid_timeout_is_rejected_before_spawn(timeout, monkeypatch):
    # 功能：
    #   使任何发生在参数校验完成前的进程创建立即失败。
    # 输入：
    #   args：进程创建的位置参数。
    #   kwargs：进程创建的配置。
    # 输出：
    #   None：不返回业务数据。
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid deadline must not spawn")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="TIMEOUT_INVALID"):
        _capture("pass", timeout=timeout)
