"""Lifecycle observation is neither actuation authority nor landing evidence."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dronedream_agent_core import runtime_phase_observer as module
from dronedream_agent_core.runtime_phase import ENDING_PHASES, phase_context


class ManualThread:
    # 功能：
    #   建立不会自行运行的线程替身，由测试显式触发采样。
    # 输入：
    #   self：线程替身。
    #   kwargs：与真实 Thread 构造器保持兼容的参数。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, **kwargs):
        self.alive = False

    # 功能：
    #   保持线程不启动，消除边界测试中不可控的后台调度。
    # 输入：
    #   self：线程替身。
    # 输出：
    #   None：不返回业务数据。
    def start(self):
        pass

    # 功能：
    #   保持关闭接口可调用，不对不存在的线程执行等待。
    # 输入：
    #   self：线程替身。
    #   timeout：接口要求的等待秒数。
    # 输出：
    #   None：不返回业务数据。
    def join(self, timeout):
        pass

    # 功能：
    #   报告测试显式维护的存活状态。
    # 输入：
    #   self：线程替身。
    # 输出：
    #   alive：当前测试设置的存活标志。
    def is_alive(self):
        alive = self.alive
        return alive


# 功能：
#   安装手动线程、可控时钟和阶段读取器，构造可重复的观察窗口。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：临时替换线程构造器。
# 输出：
#   fixture：观察器、读取器及可控时钟组成的元组。
def manual(tmp_path, monkeypatch):
    monkeypatch.setattr(module.threading, "Thread", ManualThread)
    clock = Mock(return_value=10.)
    reader = Mock(return_value=phase_context({"phase": "TRACK", "checkpoint_id": "pickup"}))
    observer = module.RuntimePhaseObserver(tmp_path / "phase.json", reader=reader, clock=clock)
    fixture = observer, reader, clock
    return fixture


# 功能：
#   latest 只读内存且返回独立对象，关闭后不再返回活跃阶段。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：安装手动线程。
# 输出：
#   None：不返回业务数据。
def test_reads_never_run_on_latest_call_and_return_owned_context(tmp_path, monkeypatch):
    observer, reader, clock = manual(tmp_path, monkeypatch)
    assert observer.latest() == phase_context(None)
    reader.assert_not_called()
    observer._sample()
    value = observer.latest()
    value["executor_phase"] = "COMPLETE"
    assert observer.latest()["executor_phase"] == "TRACK"
    reader.assert_called_once()
    assert observer.close()["confirms_physical_landing"] is False
    assert observer.latest() == phase_context(None)


# 功能：
#   过期或非法消费时钟只返回未知，不能触发补读或复用旧阶段。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：安装手动线程。
#   now：候选消费时刻。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("now", [10.051, 9.9, float("nan"), float("inf"), None, True, 10**400])
def test_stale_and_invalid_clocks_are_unknown_not_reused_context(tmp_path, monkeypatch, now):
    observer, reader, clock = manual(tmp_path, monkeypatch)
    observer._sample()
    clock.return_value = now
    assert observer.latest() == phase_context(None)
    reader.assert_called_once()
    observer.close()


# 功能：
#   读取耗时从开始计入年龄，八十毫秒读取不能通过五十毫秒使用期限。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：安装手动线程。
# 输出：
#   None：不返回业务数据。
def test_file_read_duration_is_included_in_age_not_reset_at_completion(tmp_path, monkeypatch):
    observer, reader, clock = manual(tmp_path, monkeypatch)
    clock.side_effect = [10., 10.08, 10.08]
    observer._sample()
    assert observer.latest() == phase_context(None)
    assert observer.close()["maximum_read_ms"] == pytest.approx(80.)


# 功能：
#   终止标签在同一回合不可逆，但新回合不继承它，且标签始终不能证明实际落地。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：安装手动线程。
#   phase：本次观测的终止阶段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("phase", sorted(ENDING_PHASES))
def test_ending_is_irreversible_in_one_episode_but_not_landing_proof(tmp_path, monkeypatch, phase):
    observer, reader, clock = manual(tmp_path, monkeypatch)
    reader.return_value = phase_context({"phase": phase})
    observer._sample()
    reader.return_value = phase_context({"phase": "TRACK"})
    observer._sample()
    clock.return_value = 30.
    assert observer.latest()["executor_phase"] == phase
    result = observer.close()
    assert result["ending_observed"] and not result["confirms_physical_landing"]
    replacement = module.RuntimePhaseObserver(
        tmp_path / "new-phase.json", reader=reader, clock=clock)
    assert replacement.latest() == phase_context(None)
    replacement._sample()
    assert replacement.latest()["executor_phase"] == "TRACK"
    replacement.close()


# 功能：
#   读取器失败清空活跃缓存，并在关闭时保留失败信息。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：安装手动线程。
# 输出：
#   None：不返回业务数据。
def test_reader_failure_clears_active_context_and_reports_failure(tmp_path, monkeypatch):
    observer, reader, clock = manual(tmp_path, monkeypatch)
    observer._sample()
    reader.side_effect = RuntimeError("injected reader failure")
    observer._run()
    assert observer.latest() == phase_context(None)
    assert observer.close()["reader_failed"]


# 功能：
#   两次采样之间时钟回退时，旧缓存不能重新标为新鲜。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：安装手动线程。
# 输出：
#   None：不返回业务数据。
def test_clock_regression_never_returns_a_fresh_phase(tmp_path, monkeypatch):
    observer, reader, clock = manual(tmp_path, monkeypatch)
    observer._sample()
    clock.return_value = 9.
    observer._run()
    assert observer.latest() == phase_context(None)
    assert observer.close()["reader_failed"]


# 功能：
#   阻塞实际读取线程时，控制侧内存查询保持快速，关闭超时必须如实报告。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_blocked_file_read_does_not_block_control_and_close_is_truthful(tmp_path):
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   阻塞后台文件读取替身，等待测试主动释放。
    # 输入：
    #   path：测试指定的阶段路径。
    # 输出：
    #   context：解阻后返回的活跃阶段标签。
    def read(path):
        entered.set()
        assert release.wait(2.)
        context = phase_context({"phase": "TRACK"})
        return context

    observer = module.RuntimePhaseObserver(tmp_path / "phase.json", reader=read)
    try:
        assert entered.wait(1.)
        started = time.monotonic()
        for _ in range(100):
            assert observer.latest() == phase_context(None)
        assert time.monotonic() - started < .1
        with pytest.raises(RuntimeError, match="DID_NOT_STOP"):
            observer.close(timeout_seconds=.01)
    finally:
        release.set()
        result = observer.close()
    assert result["thread_stopped"]
    assert observer.latest() == phase_context(None)


# 功能：
#   实际后台线程读取结束标签后能够停止并回收，回执明确不确认物理落地。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_live_worker_latches_end_and_joins(tmp_path):
    path = tmp_path / "phase.json"
    path.write_text('{"phase":"LANDED","landing_confirmed":true}')
    observer = module.RuntimePhaseObserver(path)
    try:
        deadline = time.monotonic() + 1.
        while observer.latest()["executor_phase"] != "LANDED" and time.monotonic() < deadline:
            time.sleep(.001)
        assert observer.latest() == phase_context({"phase": "LANDED"})
    finally:
        result = observer.close()
    assert result["ending_observed"] and not result["confirms_physical_landing"]


# 功能：
#   训练控制热路径只能读取内存观察器，结束时丢弃待执行交换消息，不恢复旧文件轮询。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：使直接文件打开立即失败。
# 输出：
#   None：不返回业务数据。
def test_training_hot_path_only_uses_memory_observer(tmp_path, monkeypatch):
    from test_px4_training_lifecycle import environment

    env = environment(tmp_path)
    env._phase_observer = SimpleNamespace(
        latest=Mock(return_value=phase_context({"phase": "TRACK"})))
    # If the old file-based check returned, this would fail immediately.
    monkeypatch.setattr(type(tmp_path), "open",
                        Mock(side_effect=AssertionError("hot-path file IO")))
    env._check_executor_ending()
    env._phase_observer.latest.assert_called_once()
    env._phase_observer.latest.return_value = phase_context({"phase": "LANDING"})
    env._exchange = SimpleNamespace(discard_pending=Mock())
    with pytest.raises(RuntimeError, match="PX4_TRAINING_EXECUTOR_ENDING:LANDING"):
        env._check_executor_ending()
    env._exchange.discard_pending.assert_called_once()


# 功能：
#   关闭预算超范围或类型不合法时，拒绝操作后仍允许正常关闭。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：安装手动线程。
#   timeout：非法等待秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("timeout", [0, -1, 3, True, float("nan"), None, 10**400])
def test_close_timeout_bound(tmp_path, monkeypatch, timeout):
    observer, _, _ = manual(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="CLOSE_TIMEOUT_INVALID"):
        observer.close(timeout_seconds=timeout)
    observer.close()
