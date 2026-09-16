import pytest

from dronedream_agent_core.pipeline_timing import PhaseTimings, PhaseTimingSummary


# 功能：
#   验证墙钟与线程 CPU 计数独立统计，不把等待时间解释为计算时间或飞行资格。
# 输入：
#   无。
# 输出：
#   None：不同计数被混合时测试失败。
def test_cpu_diagnostics_do_not_replace_wall_time():
    wall = iter((0, 100_000_000, 160_000_000))
    cpu = iter((0, 20_000_000, 30_000_000))
    timing = PhaseTimings(wall.__next__, cpu_clock_ns=cpu.__next__)
    timing.mark("fusion")
    timing.mark("publish")
    assert timing.snapshot() == {"fusion": 100., "publish": 60.}
    assert timing.cpu_snapshot() == {"fusion": 20., "publish": 10.}
    timing.cpu_snapshot()["fusion"] = 999.
    summary = PhaseTimingSummary()
    summary.record(timing, outcome="control")
    result = summary.snapshot()
    assert result["phase_ms"]["fusion"]["mean"] == 100.
    assert result["thread_cpu_phase_ms"]["fusion"]["mean"] == 20.
    assert result["qualification_granted"] is False
    summary.snapshot()["thread_cpu_phase_ms"]["fusion"]["mean"] = 999.
    assert summary.snapshot()["thread_cpu_phase_ms"]["fusion"]["mean"] == 20.


# 功能：
#   验证 CPU 时钟失败时不推进任一阶段基线，成功重试仍覆盖完整间隔。
# 输入：
#   invalid：回退、非整数或超预算 CPU 时刻。
# 输出：
#   None：失败阶段留下部分统计时测试失败。
@pytest.mark.parametrize("invalid", [-1, True, 1.5, 2**63])
def test_bad_cpu_clock_cannot_partially_advance_wall_statistics(invalid):
    wall = iter((0, 100_000_000, 160_000_000))
    cpu = iter((0, invalid, 30_000_000))
    timing = PhaseTimings(wall.__next__, cpu_clock_ns=cpu.__next__)
    with pytest.raises(ValueError):
        timing.mark("fusion")
    assert timing.snapshot() == timing.cpu_snapshot() == {}
    timing.mark("fusion")
    assert timing.snapshot() == {"fusion": 160.}
    assert timing.cpu_snapshot() == {"fusion": 30.}


# 功能：
#   验证未启用 CPU 计时的旧调用仍保持原输出，不把未测量值记为零。
# 输入：
#   无。
# 输出：
#   None：出现虚构 CPU 统计时测试失败。
def test_cpu_measurement_is_optional_and_absence_is_not_zero():
    wall = iter((0, 10_000_000))
    timing = PhaseTimings(wall.__next__)
    timing.mark("fusion")
    summary = PhaseTimingSummary()
    summary.record(timing, outcome="awaiting-target")
    assert timing.cpu_snapshot() == {}
    assert "thread_cpu_phase_ms" not in summary.snapshot()
    with pytest.raises(ValueError, match="CLOCK_INVALID"):
        PhaseTimings(cpu_clock_ns=7)
