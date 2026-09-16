"""Bounded diagnostic fixtures; no live profiling, signals, or flight controls."""

import struct
import threading
from pathlib import Path

import pytest

from dronedream_agent_core import native_process_diagnostics as diagnostics
from dronedream_agent_core.native_cpu_sampling import decode_samples, symbolize_mappings


# 功能：
#   软件采样记录解析只接受有限的实际字节，不能把任意容器或超预算记录当正常输入。
# 输入：
#   raw：错误类型或超预算的采样缓冲。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("raw", [None, [0] * 8, struct.pack("IHH", 3, 0, 8) * 131073],
                         ids=["none", "list", "over-budget"])
def test_perf_decoder_has_a_real_buffer_budget(raw):
    with pytest.raises(ValueError, match="NATIVE_PROFILE_"):
        decode_samples(raw)


# 功能：
#   指令地址及映射必须有有限的合理类型，非法区间不能计算出误导的镜像偏移。
# 输入：
#   ips：采样地址列表。
#   maps：可执行映射文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("ips,maps", [
    ([True], ""), ([-1], ""), ([2**64], ""), ([1] * 65537, ""),
    ([], "x" * (2 * 1024 * 1024 + 1)),
    ([0x1034], "2000-1000 r-xp 00000000 00:00 0 /bad.so"),
], ids=["bool", "negative", "address-overflow", "too-many", "maps-too-large", "reversed"])
def test_symbol_mapping_rejects_malformed_or_unbounded_inputs(ips, maps):
    with pytest.raises(ValueError, match="NATIVE_PROFILE_"):
        symbolize_mappings(ips, maps)


# 功能：
#   后台探针终止类异常必须出现在关闭回执中，不能线程已死却返回普通成功统计。
# 输入：
#   monkeypatch：用受控函数替代进程探测与墙钟。
# 输出：
#   None：不返回业务数据。
def test_background_failure_is_preserved_in_the_close_receipt(monkeypatch):
    probe = diagnostics.NativeSourceStallProbe()
    probe.close()
    reached = threading.Event()

    class OneIteration:
        # 功能：
        #   只触发一次后台迭代，后续立即结束。
        # 输入：
        #   self：停止信号替身。
        #   timeout：本测试不实际等待的超时时间。
        # 输出：
        #   stopped：是否已经进入过本次探测。
        def wait(self, timeout):
            stopped = reached.is_set()
            reached.set()
            return stopped

        # 功能：
        #   兼容探针关闭时设置停止信号。
        # 输入：
        #   self：停止信号替身。
        # 输出：
        #   None：不返回业务数据。
        def set(self):
            reached.set()

    # 功能：
    #   在受控探测位置触发终止异常，检验后台线程是否把故障带入回执。
    # 输入：
    #   pid：已经登记的测试进程标识。
    # 输出：
    #   None：不返回业务数据。
    def fail(pid):
        raise SystemExit("private diagnostic detail must not be copied")

    probe._stop = OneIteration()
    probe._last_source = 1.
    probe._processes = {"gazebo": 123}
    monkeypatch.setattr(diagnostics.time, "monotonic", lambda: 2.)
    monkeypatch.setattr(diagnostics, "sample_process_waits", fail)
    escaped = None
    try:
        probe._run()
    except BaseException as error:
        escaped = error
    assert escaped is None
    receipt = probe.close()
    assert receipt["read_error"] == "SystemExit"
    assert receipt["qualification_granted"] is False


# 功能：
#   非字符串角色不能导致集合查询异常；超大 PID 不能进入 proc 路径。
# 输入：
#   role：非法角色或合法角色配合非法 PID。
#   pid：被检验的进程标识。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("role,pid", [([], 1), ("gazebo", 2**40)])
def test_registration_rejects_invalid_values(role, pid):
    probe = diagnostics.NativeSourceStallProbe()
    try:
        with pytest.raises(ValueError, match="OWN_SIMULATOR_CHILD"):
            probe.register(role, pid)
    finally:
        probe.close()


# 功能：
#   线程目录枚举也必须有上限，不能只限制后续文件读取却先收集任意多目录项。
# 输入：
#   tmp_path：隔离的 proc 目录替身。
#   monkeypatch：提供可计数的大目录迭代器。
# 输出：
#   None：不返回业务数据。
def test_thread_directory_enumeration_is_bounded(tmp_path, monkeypatch):
    count = []

    # 功能：
    #   生成超过预算的目录项，并记录实际被消费的数量。
    # 输入：
    #   path：待枚举的线程目录。
    # 输出：
    #   item：逐次生成的线程路径。
    def entries(path):
        for index in range(10000):
            count.append(index)
            item = path / str(index + 1)
            yield item

    monkeypatch.setattr(Path, "iterdir", entries)
    row = diagnostics.sample_process_waits(101, proc_root=tmp_path)
    assert len(count) <= 4097
    assert row["thread_count"] is None
    assert row["thread_inventory_truncated"] is True


# 功能：
#   异常或回退的探测时钟应终止诊断并留下错误，不写出非有限时差或继续真实 proc 读取。
# 输入：
#   monkeypatch：替代时钟及进程读取函数。
#   clock：非法或回退的单调时钟值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("clock", [True, None, float("nan"), .5, 1e308])
def test_probe_clock_failures_are_explicit(monkeypatch, clock):
    probe = diagnostics.NativeSourceStallProbe()
    probe.close()
    probe._stop = threading.Event()
    probe._last_source = 1.
    monkeypatch.setattr(diagnostics.time, "monotonic", lambda: clock)
    probe._run()
    receipt = probe.close()
    assert receipt["read_error"] == "ValueError"
    assert receipt["sample_count"] == 0
