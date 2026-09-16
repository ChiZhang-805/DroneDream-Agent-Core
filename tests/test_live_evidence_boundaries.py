"""Live cursor ownership and independent-window scalar boundaries."""

from collections import deque
from pathlib import Path
from threading import Lock

import pytest
from test_training_runtime_evidence import pose

from dronedream_agent_core.training.runtime_evidence import IndependentPoseMonitor, JsonlCursor


# 功能：
#   验证游标创建时固定绝对路径，工作目录改变后不会读取另一份同名证据。
# 输入：
#   tmp_path：两个证据目录的共同父目录。
#   monkeypatch：隔离当前工作目录。
# 输出：
#   None：不返回业务数据。
def test_cursor_binds_original_directory(tmp_path, monkeypatch):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "evidence.jsonl").write_bytes(b'{"source":"original"}\n')
    (right / "evidence.jsonl").write_bytes(b'{"source":"other"}\n')
    monkeypatch.chdir(left)
    cursor = JsonlCursor(Path("evidence.jsonl"))
    monkeypatch.chdir(right)
    assert cursor.read() == [{"source": "original"}]


# 功能：
#   验证畸形记录导致失败后不能跳到下一段继续输出正常记录，防止错误被调用方重试掩盖。
# 输入：
#   tmp_path：当前测试证据路径。
# 输出：
#   None：不返回业务数据。
def test_cursor_latches_parse_failure(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"x":NaN}\n')
    cursor = JsonlCursor(path)
    with pytest.raises(ValueError):
        cursor.read()
    with path.open("ab") as stream:
        stream.write(b'{"x":2}\n')
    with pytest.raises(ValueError, match="CURSOR_FAILED"):
        cursor.read()


# 功能：
#   验证独立窗口的所有入口只接纳非负整数毫秒，不接受布尔、浮点或溢出时间。
# 输入：
#   stamp：非法源毫秒时间。
#   method：窗口、起点基线或流式初始帧入口。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("stamp", [True, 1050.5, -1, 2**63])
@pytest.mark.parametrize("method", ["window", "has_start_baseline", "initial_witness"])
def test_monitor_requires_integer_source_timestamps(stamp, method):
    monitor = object.__new__(IndependentPoseMonitor)
    monitor._lock, monitor._rows, monitor._error = Lock(), deque([pose(1, 0), pose(2, 1)]), None
    with pytest.raises(ValueError, match="TIMESTAMP_INVALID"):
        if method == "window":
            monitor.window(stamp, max(1100, stamp + 1))
        else:
            getattr(monitor, method)(stamp)
