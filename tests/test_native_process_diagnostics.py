import pytest

from dronedream_agent_core.native_process_diagnostics import sample_process_waits


# 功能：
#   只允许登记标准模拟器角色；未收到来源样本时不应探测测试 PID 或授予飞行资格。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_render_replica_is_an_explicit_owned_child_role_not_an_arbitrary_process():
    from dronedream_agent_core.native_process_diagnostics import NativeSourceStallProbe
    probe = NativeSourceStallProbe()
    try:
        for role in ("gazebo", "px4", "render-replica"):
            probe.register(role, 123)
        for role, pid in (("browser", 123), ("render-replica", 0), ("gazebo", True)):
            with pytest.raises(ValueError, match="OWN_SIMULATOR_CHILD"):
                probe.register(role, pid)
    finally:
        receipt = probe.close()
    assert receipt["qualification_granted"] is False
    assert receipt["sample_count"] == 0  # No source gap: never reads fixture PIDs.


# 功能：
#   从指定 proc 替身读取线程状态，名称中的空格和括号不能错位后续状态字段。
# 输入：
#   tmp_path：隔离的 proc 目录替身。
# 输出：
#   None：不返回业务数据。
def test_probe_reads_only_registered_process_and_handles_names_with_parentheses(tmp_path):
    root = tmp_path / "101"
    task = root / "task" / "102"
    task.mkdir(parents=True)
    (root / "io").write_text("write_bytes: 4096\n")
    (task / "stat").write_text("102 (worker (physics)) D 1 2 3\n")
    (task / "wchan").write_text("filemap_fdatawait_range\n")
    row = sample_process_waits(101, proc_root=tmp_path)
    assert row["thread_count"] == 1
    assert row["active_threads"] == [
        {
            "tid": 102,
            "name": "worker (physics)",
            "state": "D",
            "wait_channel": "filemap_fdatawait_range",
        }
    ]
    assert row["wait_counts"][0]["count"] == 1
    assert not row["thread_inventory_truncated"]


# 功能：
#   子进程目录消失必须保留读取错误，而不是返回零活动或生成控制动作。
# 输入：
#   tmp_path：不存在所选 PID 的 proc 替身。
# 输出：
#   None：不返回业务数据。
def test_ended_child_is_a_diagnostic_error_not_a_control_action(tmp_path):
    row = sample_process_waits(101, proc_root=tmp_path)
    assert row == {"pid": 101, "read_error": "FileNotFoundError",
                   "io_read_error": "FileNotFoundError"}


# 功能：
#   可选 I/O 或等待通道文件缺失时仍保留可读取的运行状态，缺失值不能伪造为零。
# 输入：
#   tmp_path：只有部分诊断文件的 proc 替身。
# 输出：
#   None：不返回业务数据。
def test_missing_optional_io_or_wait_channel_does_not_hide_thread_state(tmp_path):
    task = tmp_path / "101" / "task" / "102"
    task.mkdir(parents=True)
    (task / "stat").write_text("102 (physics) R 1 2 3\n")
    row = sample_process_waits(101, proc_root=tmp_path)
    assert row["io_read_error"] == "FileNotFoundError"
    assert row["active_threads"][0]["state"] == "R"
    assert row["active_threads"][0]["wait_channel"] == "unavailable"


# 功能：
#   合法调度计数区分运行与排队时间；错误记录不生成伪计数，也不隐藏线程状态。
# 输入：
#   tmp_path：隔离 proc 替身。
#   scheduling：合法或损坏的调度计数文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("scheduling", ["1000000 2000000 30", "bad", "1 -2 3", "1 2"])
def test_scheduling_distinguishes_running_time_from_cpu_wait_without_inventing_data(
    tmp_path, scheduling,
):
    task = tmp_path / "101" / "task" / "102"
    task.mkdir(parents=True)
    (task / "stat").write_text("102 (physics) R 1 2 3\n")
    (task / "schedstat").write_text(scheduling)
    row = sample_process_waits(101, proc_root=tmp_path)
    assert row["thread_scheduling"] == ([{
        "tid": 102, "name": "physics", "runtime_ns": 1000000,
        "runqueue_wait_ns": 2000000, "timeslices": 30,
    }] if scheduling == "1000000 2000000 30" else [])
    assert row["active_threads"][0]["state"] == "R"


# 功能：
#   非法 PID 不能参与路径组合，防止读取错误目标或超出指定目录。
# 输入：
#   tmp_path：隔离 proc 根。
#   pid：错误类型、非正数或包含路径片段的输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("pid", [True, 0, -1, "../other", None])
def test_invalid_pid_does_not_read_other_paths(tmp_path, pid):
    with pytest.raises(ValueError, match="PROCESS_ID_INVALID"):
        sample_process_waits(pid, proc_root=tmp_path)


# 功能：
#   验证每次实际读取都有小的正数上限，而不是先无限读取再截断。
# 输入：
#   tmp_path：隔离线程目录。
#   monkeypatch：替换文件打开以记录 read 参数。
# 输出：
#   None：不返回业务数据。
def test_proc_reads_have_explicit_small_bounds(tmp_path, monkeypatch):
    import io
    from pathlib import Path

    task = tmp_path / "101/task/102"
    task.mkdir(parents=True)
    texts = {"io": "x" * 5000, "stat": "102 (worker) R 1 2 3",
             "wchan": "running", "schedstat": "1 2 3"}
    limits = []

    class BoundedStream(io.StringIO):
        # 功能：
        #   在文件替身读取入口验证大小，并返回相同边界下的实际文本。
        # 输入：
        #   self：内存文本流。
        #   size：调用者请求的字符数。
        # 输出：
        #   prefix：按指定限制取得的文本前缀。
        def read(self, size=-1):
            assert 0 < size <= 4096
            limits.append(size)
            prefix = super().read(size)
            return prefix

    monkeypatch.setattr(Path, "open", lambda path, **kw: BoundedStream(texts[path.name]))
    row = sample_process_waits(101, proc_root=tmp_path)
    assert len(row["io"]) == 2048
    assert limits == [2048, 4096, 128, 256]


# 功能：
#   回退或非法时钟不能覆盖最后有效来源，修改导出回执也不能改写内部历史。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_probe_receipts_are_owned_and_invalid_clocks_do_not_replace_last_source():
    from dronedream_agent_core.native_process_diagnostics import NativeSourceStallProbe

    probe = NativeSourceStallProbe()
    probe.close()  # No actual /proc reads for this state-ownership fixture.
    probe.source_received(10.)
    for value in (True, None, float("nan"), 10**400):
        with pytest.raises(ValueError, match="RECEIPT_TIME_INVALID"):
            probe.source_received(value)
    probe.source_received(9.)
    assert probe._last_source == 10.
    probe._rows.append({"processes": {"gazebo": {"pid": 123}}})
    receipt = probe.close()
    receipt["latest_stalls"][0]["processes"]["gazebo"]["pid"] = 999
    assert probe.close()["latest_stalls"][0]["processes"]["gazebo"]["pid"] == 123
