import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_stream_collection import stream_fixture

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training import capture_archive, stream_capture
from dronedream_agent_core.training.px4_environment import _write_new


# 功能：
#   验证原生和连续流捕获在类型解析之前拒绝重复字段，即使最后一个字段合法。
# 输入：
#   tmp_path：合成回合目录。
#   native：是否检查原生奖励转移捕获的解码入口。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("native", [False, True])
def test_capture_unpack_rejects_duplicate_json_before_typed_validation(tmp_path, native):
    episode, _, _, packed = stream_fixture(tmp_path)
    if native:
        row = json.loads((episode / "fixture-source.json").read_bytes())
        content = json.dumps({
            "source_observation": row["source_observation"],
            "observation": row["step"]["observation"], "snapshot": row["source_snapshot"],
            "proposal": row["proposal"], "command": row["command"],
            "application": row["application"], "previous_application": None,
            "applied": row["step"]["applied_action"], "deadline_intervened": False,
            "witnesses": row["outcome_receipt"]["observations"], "truncated": False,
        }).encode()
        packed = capture_archive.PackedNativeTransition(content,
                                                       hashlib.sha256(content).hexdigest())
    # 先确认原始结构合法，再仅插入重复字段，避免把夹具缺字段当成产品漏洞。
    packed.unpack()
    content = b'{"observation":null,' + packed.content[1:]
    if native:
        candidate = capture_archive.PackedNativeTransition(content,
                                                           hashlib.sha256(content).hexdigest())
    else:
        candidate = replace(packed, content=content, sha256=hashlib.sha256(content).hexdigest())
    with pytest.raises(ValueError, match="DUPLICATE_KEY"):
        candidate.unpack()


# 功能：
#   验证归档单对象和逐行账本均拒绝重复键与非有限数字。
# 输入：
#   tmp_path：测试文件目录。
#   line：非法 JSON 内容。
#   ledger：是否使用逐行账本入口。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("line", [b'{"x":0,"x":1}\n', b'{"x":NaN}\n'])
@pytest.mark.parametrize("ledger", [False, True])
def test_record_readers_require_unambiguous_json(tmp_path, line, ledger):
    path = tmp_path / "capture.jsonl"
    path.write_bytes(line)
    with pytest.raises(ValueError, match="JSON"):
        if ledger:
            list(stream_capture._records(path))
        else:
            stream_capture.read_stream_record(path)


# 功能：
#   验证账本除了单行和行数限制，也有完整文件的字节预算。
# 输入：
#   tmp_path：账本目录。
#   monkeypatch：缩小总字节上限的工具。
# 输出：
#   None：不返回业务数据。
def test_ledger_total_bytes_are_bounded(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"x":1}\n' * 5)
    monkeypatch.setattr(stream_capture, "MAXIMUM_ARCHIVE_BYTES", 24)
    with pytest.raises(ValueError, match="TOO_LARGE"):
        list(stream_capture._records(path))


# 功能：
#   验证文件在打开前被同内容新文件替换时仍因身份改变被拒绝。
# 输入：
#   tmp_path：独立文件目录。
#   monkeypatch：文件打开替换器。
#   ledger：是否测试逐行账本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("ledger", [False, True])
def test_record_reader_rejects_replaced_file(tmp_path, monkeypatch, ledger):
    path = tmp_path / "record.jsonl"
    path.write_bytes(b'{"value":1}\n')
    saved = tmp_path / "original.jsonl"
    original_open = Path.open
    replaced = False

    # 功能：
    #   在目标首次打开时保留原文件并创建同内容的新文件，稳定复现身份竞态。
    # 输入：
    #   target：准备打开的路径。
    #   args：原始位置参数。
    #   kwargs：原始关键字参数。
    # 输出：
    #   stream：实际打开的文件流。
    def replace_before_open(target, *args, **kwargs):
        nonlocal replaced
        if target == path and not replaced:
            replaced = True
            path.rename(saved)
            with original_open(path, "wb") as replacement:
                replacement.write(b'{"value":1}\n')
        stream = original_open(target, *args, **kwargs)
        return stream

    monkeypatch.setattr(Path, "open", replace_before_open)
    with pytest.raises(ValueError, match="CHANGED"):
        if ledger:
            list(stream_capture._records(path))
        else:
            stream_capture.read_stream_record(path)
    assert saved.read_bytes() == path.read_bytes()


# 功能：
#   验证控制访问不能借用另一份捕获的摘要，实际命令正确也不能豁免来源检查。
# 输入：
#   tmp_path：合成回合目录。
# 输出：
#   None：不返回业务数据。
def test_stream_visit_rechecks_packed_capture_binding(tmp_path):
    episode, _, capture, packed = stream_fixture(tmp_path)
    call_id = "model-" + sha256_json(capture.proposal)[:24]
    command, application = stream_capture.grounded_control_index(
        episode / "flight/simulation", {call_id},
    )[call_id]
    with pytest.raises(ValueError, match="CAPTURE_CONTENT_CHANGED"):
        stream_capture.stream_visit(capture, replace(packed, sha256="0" * 64), command, application)


# 功能：
#   验证归档器在发布器回调修改外部捕获列表时，仍按入口固定的完整批次结算。
# 输入：
#   tmp_path：合成回合和输出目录。
# 输出：
#   None：不返回业务数据。
def test_finalizer_owns_capture_sequence_before_publication(tmp_path):
    episode, _, _, packed = stream_fixture(tmp_path)
    captures = [packed]

    # 功能：
    #   执行真实独占发布后清空外部列表，模拟调用方回收自己的存储。
    # 输入：
    #   path：需要保存的文件路径。
    #   record：发布对象。
    # 输出：
    #   None：不返回业务数据。
    def publish(path, record):
        _write_new(path, record)
        captures.clear()

    visits = stream_capture.finalize_stream_captures(episode, captures, write_new=publish)
    assert len(visits) == 1
    receipt = stream_capture.read_stream_record(episode / "stream-capture-receipt.json")
    assert receipt["submitted"] == receipt["actually_accepted_decisions"] == 1
