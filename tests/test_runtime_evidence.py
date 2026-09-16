import json
import threading
import time
from pathlib import Path

import pytest

from dronedream_agent_core.runtime_evidence import BoundedRuntimeEvidenceWriter


# 功能：
#   非法队列容量或刷新周期在创建后台线程前拒绝。
# 输入：
#   tmp_path：测试私有目录。
#   field：被替换的配置名。
#   value：非法配置值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [
    ("maximum_pending_records", True), ("maximum_pending_records", float("inf")),
    ("maximum_pending_records", 1.5), ("flush_interval_seconds", True),
    ("flush_interval_seconds", 2**4096),
], ids=["bool-capacity", "infinite-capacity", "fractional-capacity",
        "bool-period", "overflow-period"])
def test_bad_queue_limits_fail_before_starting_a_writer(tmp_path, field, value):
    writer = None
    try:
        with pytest.raises(ValueError):
            writer = BoundedRuntimeEvidenceWriter(tmp_path / "summary.json",
                summary_publisher=lambda *_: None, serializer=json.dumps, **{field: value})
    finally:
        if writer is not None:
            writer.close()


# 功能：
#   非法关闭超时不应提前关闭队列，随后合法提交和关闭仍可执行。
# 输入：
#   tmp_path：测试私有目录。
#   timeout：非法等待秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("timeout", [True, float("inf"), float("nan"), -1, 2**4096],
                         ids=["bool", "infinite", "nan", "negative", "overflow"])
def test_invalid_drain_timeout_does_not_close_the_writer(tmp_path, timeout):
    writer = BoundedRuntimeEvidenceWriter(tmp_path / "summary.json",
        summary_publisher=lambda *_: None, serializer=json.dumps)
    try:
        with pytest.raises(ValueError):
            writer.close(timeout_seconds=timeout)
        assert not writer.summary()["closed"]
        assert writer.submit(tmp_path / "event.jsonl", {"valid": True})
    finally:
        writer.close()


# 功能：
#   序列化失败应锁定失败但不把异常中的测试敏感文字泄漏到诊断或磁盘汇总。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_serializer_exception_does_not_echo_payload_secrets_into_diagnostics(tmp_path):
    secret = "synthetic-sensitive-value"
    # 功能：
    #   抛出包含合成敏感文本的异常，检验错误类别与原文的隔离。
    # 输入：
    #   _：本测试不使用的载荷。
    # 输出：
    #   None：不返回业务数据。
    def fail(_):
        raise ValueError(secret)
    writer = BoundedRuntimeEvidenceWriter(tmp_path / "summary.json",
        summary_publisher=lambda path, value: path.write_text(json.dumps(value)), serializer=fail)
    writer.submit(tmp_path / "events.jsonl", {})
    summary = writer.close()
    assert not summary["complete"]
    assert secret not in json.dumps(summary)
    assert secret not in (tmp_path / "summary.json").read_text()


# 功能：
#   阻塞后台序列化后修改调用方嵌套数组，实际写入仍必须使用提交时的独立快照。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_queued_evidence_cannot_be_changed_by_caller(tmp_path):
    entered, release = threading.Event(), threading.Event()
    # 功能：
    #   通知测试已接收快照并等待释放，再输出真正要写盘的 JSON。
    # 输入：
    #   payload：后台持有的独立载荷。
    # 输出：
    #   text：单行 JSON 文本。
    def serialize(payload):
        entered.set()
        assert release.wait(2)
        text = json.dumps(payload)
        return text
    writer = BoundedRuntimeEvidenceWriter(tmp_path / "summary.json",
        summary_publisher=lambda path, value: path.write_text(json.dumps(value)),
        serializer=serialize)
    target = tmp_path / "commands.jsonl"
    payload = {"axes": [1, 0, 0, 0]}
    try:
        assert writer.submit(target, payload)
        assert entered.wait(2)
        payload["axes"][0] = -1
    finally:
        release.set()
        summary = writer.close()
    assert summary["complete"]
    assert json.loads(target.read_text())["axes"][0] == 1


# 功能：
#   最后刷新失败必须影响完成状态，不能因为写入计数相符就报告成功。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：注入最后刷新失败。
# 输出：
#   None：不返回业务数据。
def test_failed_final_flush_cannot_report_complete(tmp_path, monkeypatch):
    writer = BoundedRuntimeEvidenceWriter(tmp_path / "summary.json",
        summary_publisher=lambda path, value: path.write_text(json.dumps(value)),
        serializer=json.dumps, flush_interval_seconds=100)
    # 功能：
    #   模拟最后批量刷新时磁盘不可用。
    # 输入：
    #   _handles：本故障替身不操作的句柄映射。
    # 输出：
    #   None：不返回业务数据。
    def fail(_handles):
        raise OSError("disk unavailable")
    monkeypatch.setattr(writer, "_flush_handles", fail)
    writer.submit(tmp_path / "commands.jsonl", {"sent": True})
    summary = writer.close()
    assert not summary["complete"]
    assert summary["issue_code"].startswith("RUNTIME_EVIDENCE_FINAL_FLUSH_FAILED")


# 功能：
#   汇总发布失败后拒绝新记录，关闭返回失败码但不打断外层控制器清理。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_failed_summary_does_not_interrupt_controller_cleanup(tmp_path):
    # 功能：
    #   模拟汇总输出路径无法写入。
    # 输入：
    #   _path：被拒绝的目标路径。
    #   _value：未能发布的汇总对象。
    # 输出：
    #   None：不返回业务数据。
    def fail(_path, _value):
        raise OSError("disk unavailable")
    writer = BoundedRuntimeEvidenceWriter(tmp_path / "summary.json",
        summary_publisher=fail, serializer=json.dumps)
    assert not writer.submit(tmp_path / "commands.jsonl", {})
    summary = writer.close()
    assert not summary["complete"]
    assert summary["issue_code"].startswith("RUNTIME_EVIDENCE_SUMMARY_FAILED")


# 功能：
#   关键执行回执在关闭前逐条可见，普通流仍留在批量缓冲中直到刷新或关闭。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_receipt_stream_visible_before_bulk_flush_or_close(tmp_path):
    receipt, bulk = tmp_path / "receipts.jsonl", tmp_path / "bulk.jsonl"
    writer = BoundedRuntimeEvidenceWriter(
        tmp_path / "summary.json", summary_publisher=lambda *_: None,
        serializer=json.dumps, flush_interval_seconds=100,
        flush_on_record_paths=(receipt,))
    try:
        assert writer.submit(bulk, {"bulk": True})
        assert writer.submit(receipt, {"accepted": True})
        deadline = time.monotonic() + 2
        while writer.summary()["completed_count"] != 2 and time.monotonic() < deadline:
            time.sleep(.001)
        assert writer.summary()["completed_count"] == 2
        assert json.loads(receipt.read_text()) == {"accepted": True}
        assert bulk.read_text() == ""
        assert not writer.summary()["closed"]
    finally:
        summary = writer.close()
    assert summary["complete"]
    assert json.loads(bulk.read_text()) == {"bulk": True}


# 功能：
#   对真实句柄包装刷新故障，关键流刷新失败前不能增加完成计数，文件操作保持在后台。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：给目标流安装只在刷新处失败的包装器。
# 输出：
#   None：不返回业务数据。
def test_receipt_flush_failure_is_latched_before_completion(tmp_path, monkeypatch):
    receipt = tmp_path / "receipts.jsonl"
    flushed = threading.Event()
    original_open = Path.open
    producer_thread = threading.get_ident()

    class FailingReceipt:
        # 功能：
        #   保存实际打开的句柄，不伪造文件身份或写入返回值。
        # 输入：
        #   self：故障包装器。
        #   handle：实际普通文件句柄。
        # 输出：
        #   None：不返回业务数据。
        def __init__(self, handle):
            self.handle = handle

        # 功能：
        #   将实际文件描述符交给生产代码验证身份。
        # 输入：
        #   self：故障包装器。
        # 输出：
        #   descriptor：实际文件描述符。
        def fileno(self):
            descriptor = self.handle.fileno()
            return descriptor

        # 功能：
        #   确认写入来自后台线程并执行真实缓冲写入。
        # 输入：
        #   self：故障包装器。
        #   text：待写入的完整记录。
        # 输出：
        #   written：实际接受的字符数。
        def write(self, text):
            assert threading.get_ident() != producer_thread
            written = self.handle.write(text)
            return written

        # 功能：
        #   标记关键刷新已发生，并只在此处注入磁盘错误。
        # 输入：
        #   self：故障包装器。
        # 输出：
        #   None：不返回业务数据。
        def flush(self):
            flushed.set()
            raise OSError("receipt flush failed")

        # 功能：
        #   回收真实文件句柄，避免故障测试自己泄漏资源。
        # 输入：
        #   self：故障包装器。
        # 输出：
        #   None：不返回业务数据。
        def close(self):
            self.handle.close()

    # 功能：
    #   保持真实打开操作，仅给目标执行回执增加刷新故障包装。
    # 输入：
    #   path：被打开的路径。
    #   args、kwargs：实际文件打开参数。
    # 输出：
    #   handle：普通句柄或刷新失败包装后的句柄。
    def open_receipt(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path == receipt:
            handle = FailingReceipt(handle)
        return handle

    monkeypatch.setattr(Path, "open", open_receipt)
    writer = BoundedRuntimeEvidenceWriter(
        tmp_path / "summary.json", summary_publisher=lambda *_: None,
        serializer=json.dumps, flush_interval_seconds=100,
        flush_on_record_paths=(receipt,))
    try:
        assert writer.submit(receipt, {"accepted": True})
        assert flushed.wait(2)
    finally:
        summary = writer.close()
    assert not summary["complete"]
    assert summary["completed_count"] == 0
    assert summary["issue_code"].startswith("RUNTIME_EVIDENCE_WRITE_FAILED:OSError")
