"""Exercise the actual local FIFO and files without actuators or remote services."""

import json
import threading

import pytest

from dronedream_agent_core import runtime_evidence
from dronedream_agent_core.runtime_evidence import (
    BoundedRuntimeEvidenceWriter,
    navigation_evidence_inventory,
    reserve_new_evidence_files,
)


# 功能：
#   构造私有目录中的实际后台写入器，测试可替换序列化器。
# 输入：
#   directory：测试拥有的目录。
#   serializer：输出单行 JSON 的回调。
# 输出：
#   writer：已启动但没有提交数据的后台队列。
def writer_at(directory, serializer=json.dumps):
    writer = BoundedRuntimeEvidenceWriter(directory / "summary.json",
        summary_publisher=lambda *_: None, serializer=serializer)
    return writer


# 功能：
#   非法序列化结果不得写入流或计为成功记录。
# 输入：
#   tmp_path：测试私有目录。
#   text：序列化器返回的非法内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("text", ['{}\n{}', '{"a":1,"a":2}', '{"x":NaN}', '[]', 'bad', 1])
def test_bad_serialized_record_cannot_be_completed(tmp_path, text):
    writer = writer_at(tmp_path, lambda _: text)
    assert writer.submit(tmp_path / "event.jsonl", {})
    result = writer.close()
    assert not result["complete"] and result["completed_count"] == 0


# 功能：
#   不受支持的对象不能通过深复制执行自定义方法或进入记录队列。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_snapshot_rejects_custom_copy_without_executing_it(tmp_path):
    copied = []

    class Custom:
        # 功能：
        #   标记不应该被记录器调用的对象复制钩子。
        # 输入：
        #   self：测试对象。
        #   memo：复制缓存。
        # 输出：
        #   value：用于暴露错误执行的普通字典。
        def __deepcopy__(self, memo):
            copied.append(True)
            value = {}
            return value

    writer = writer_at(tmp_path)
    try:
        assert not writer.submit(tmp_path / "event.jsonl", Custom())
        assert copied == []
    finally:
        result = writer.close()
    assert not result["complete"]


# 功能：
#   新写入器不得追加到已有非空流中并声称只包含本次提交。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_existing_nonempty_stream_is_preserved(tmp_path):
    target = tmp_path / "event.jsonl"
    target.write_text('{"previous":true}\n', encoding="utf-8")
    writer = writer_at(tmp_path)
    writer.submit(target, {"new": True})
    result = writer.close()
    assert not result["complete"] and result["completed_count"] == 0
    assert target.read_text(encoding="utf-8") == '{"previous":true}\n'


# 功能：
#   写入目标必须位于当前运行目录且不能等于回执文件。
# 输入：
#   tmp_path：测试私有目录。
#   kind：越界路径或回执路径。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["outside", "summary"])
def test_destination_scope_is_enforced(tmp_path, kind):
    directory = tmp_path / "run"
    directory.mkdir()
    target = tmp_path / "foreign.jsonl" if kind == "outside" else directory / "summary.json"
    writer = writer_at(directory)
    try:
        assert not writer.submit(target, {})
    finally:
        result = writer.close()
    assert not result["complete"]


# 功能：
#   非法回调在创建线程和发布回执前拒绝。
# 输入：
#   tmp_path：测试私有目录。
#   field：被替换为不可调用对象的回调。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["serializer", "summary_publisher"])
def test_invalid_callbacks_fail_before_initialization(tmp_path, field):
    kwargs = {"serializer": json.dumps, "summary_publisher": lambda *_: None, field: None}
    writer = None
    try:
        with pytest.raises(ValueError):
            writer = BoundedRuntimeEvidenceWriter(tmp_path / "summary.json", **kwargs)
    finally:
        if writer is not None:
            writer.close()


# 功能：
#   复核流时，重复键或只有空行的文件不能计为正常完整记录。
# 输入：
#   tmp_path：测试私有目录。
#   contents：有歧义的 JSONL 字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("contents", [b'{"a":1,"a":2}\n', b'\n'])
def test_inventory_rejects_ambiguous_lines(tmp_path, contents):
    target = tmp_path / "model-navigation-timing.jsonl"
    target.write_bytes(contents)
    assert navigation_evidence_inventory(tmp_path)[str(target)] == -1


# 功能：
#   领取文件不能穿过父级符号链接进入另一个目录。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_claim_rejects_linked_parent(tmp_path):
    real, alias = tmp_path / "real", tmp_path / "alias"
    real.mkdir()
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit unprivileged symbolic links")
    with pytest.raises(ValueError):
        reserve_new_evidence_files(alias / "event.jsonl")
    assert not list(real.iterdir())


# 功能：
#   队列关闭时必须发现输出路径已指向另一个文件，不能对消失的原流报告完成。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_replaced_stream_is_not_a_complete_publication(tmp_path):
    target, held = tmp_path / "event.jsonl", tmp_path / "held.jsonl"
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   阻塞第二条序列化，为测试替换第一条已经打开的流留出可控窗口。
    # 输入：
    #   value：当前记录对象。
    # 输出：
    #   text：合法 JSON 行。
    def serialize(value):
        if value["n"] == 2:
            entered.set()
            assert release.wait(2)
        text = json.dumps(value)
        return text

    writer = writer_at(tmp_path, serialize)
    try:
        assert writer.submit(target, {"n": 1})
        assert writer.submit(target, {"n": 2})
        assert entered.wait(2)
        try:
            target.rename(held)
        except PermissionError:
            pytest.skip("host disallows renaming an open evidence file")
        target.write_text('{}\n', encoding="utf-8")
    finally:
        release.set()
        result = writer.close()
    assert not result["complete"]


# 功能：
#   同时排队数量尚未用完，但待处理字节预算用尽时仍拒绝，已接受记录继续排空。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：缩小字节预算以构造低内存的确定性测试。
# 输出：
#   None：不返回业务数据。
def test_pending_byte_budget_is_independent_of_record_slots(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   阻塞首个处理项，保持第二条记录在待处理队列中。
    # 输入：
    #   payload：当前待编码对象。
    # 输出：
    #   text：解阻后的单行 JSON。
    def serialize(payload):
        entered.set()
        assert release.wait(2)
        text = json.dumps(payload)
        return text

    monkeypatch.setattr(runtime_evidence, "MAXIMUM_PENDING_EVIDENCE_BYTES", 32)
    writer = writer_at(tmp_path, serialize)
    target = tmp_path / "events.jsonl"
    try:
        assert writer.submit(target, {"data": "1234567890"})
        assert entered.wait(2)
        assert writer.submit(target, {"data": "1234567890"})
        assert not writer.submit(target, {"data": "1234567890"})
        assert writer.summary()["pending_bytes"] <= 32
    finally:
        release.set()
        result = writer.close()
    assert not result["complete"] and result["completed_count"] == 2
    assert result["pending_bytes"] == 0


# 功能：
#   外部向已打开的流追加内容时，关闭复核检测实际字节不一致，不能报告完整。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_external_append_is_detected_even_without_rename(tmp_path):
    target = tmp_path / "events.jsonl"
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   第二次编码等待测试完成外部追加，确保第一次写入已在后台发生。
    # 输入：
    #   payload：当前记录。
    # 输出：
    #   text：解阻后的 JSON 文本。
    def serialize(payload):
        if payload["n"] == 2:
            entered.set()
            assert release.wait(2)
        text = json.dumps(payload)
        return text

    writer = writer_at(tmp_path, serialize)
    try:
        assert writer.submit(target, {"n": 1})
        assert writer.submit(target, {"n": 2})
        assert entered.wait(2)
        with target.open("ab") as stream:
            stream.write(b'{"external":true}\n')
    finally:
        release.set()
        result = writer.close()
    assert not result["complete"]
    assert result["issue_code"]


# 功能：
#   新传感器流不能绕过固定打开文件预算，无须真的打开大量句柄即可验证入口边界。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：将本测试允许的流数量缩小为一。
# 输出：
#   None：不返回业务数据。
def test_stream_budget_is_checked_before_enqueue(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_evidence, "MAXIMUM_EVIDENCE_STREAMS", 1)
    writer = writer_at(tmp_path)
    try:
        assert writer.submit(tmp_path / "first.jsonl", {})
        assert not writer.submit(tmp_path / "second.jsonl", {})
    finally:
        result = writer.close()
    assert result["completed_count"] == 1 and not result["complete"]


# 功能：
#   后台遇到线程终止类异常仍锁定失败并完成资源回收，不遗留活线程或虚假成功。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_fatal_serializer_failure_has_a_truthful_receipt(tmp_path):
    # 功能：
    #   模拟后台扩展抛出终止类异常。
    # 输入：
    #   payload：本测试不消费的记录。
    # 输出：
    #   None：不返回业务数据。
    def fail(payload):
        raise SystemExit("synthetic private diagnostic")

    writer = writer_at(tmp_path, fail)
    writer.submit(tmp_path / "events.jsonl", {})
    result = writer.close()
    assert not result["complete"] and result["thread_finished"]
    assert result["issue_code"] == "RUNTIME_EVIDENCE_WRITE_FAILED:SystemExit"
    assert "synthetic private diagnostic" not in json.dumps(result)
