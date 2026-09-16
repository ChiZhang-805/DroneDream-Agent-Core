import json

import pytest

from dronedream_agent_core.runtime_evidence import (
    CONTROL_EVIDENCE_FILENAMES,
    NAVIGATION_EVIDENCE_FILENAMES,
    control_evidence_inventory,
    navigation_evidence_inventory,
    runtime_evidence_inventory_complete,
)


# 功能：
#   为独立计数构造自洽的合成关闭回执，不启动实际写入器。
# 输入：
#   counts：测试指定的文件到记录数映射。
# 输出：
#   value：用于逐字段破坏检查的正常回执夹具。
def receipt(counts):
    value = dict(
        schema_version="dronedream.runtime-evidence-writer.v1",
        complete=True,
        closed=True,
        issue_code=None,
        thread_finished=True,
        thread_alive=False,
        pending_count=0,
        rejected_count=0,
        submitted_count=sum(counts.values()),
        completed_count=sum(counts.values()),
        artifacts=[dict(path=k, submitted_count=v, completed_count=v) for k, v in counts.items()],
    )
    return value


# 功能：
#   只重数两个固定执行器流，不受未知文件干扰，任一固定流截断后整份验收失败。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_control_inventory_recounts_both_fixed_executor_streams(tmp_path):
    runtime = tmp_path / "runtime-state"
    runtime.mkdir()
    for count, name in enumerate(CONTROL_EVIDENCE_FILENAMES, start=1):
        (runtime / name).write_text('{}\n' * count, encoding="utf-8")
    # An unowned file is never opened because a writer receipt says it exists.
    (runtime / "unknown.jsonl").write_text("truncated", encoding="utf-8")
    counts = control_evidence_inventory(tmp_path)
    assert counts == {str(runtime / name): count for count, name in enumerate(
        CONTROL_EVIDENCE_FILENAMES, start=1)}
    assert runtime_evidence_inventory_complete(receipt(counts), artifact_counts=counts,
                                               record_count=3)
    (runtime / CONTROL_EVIDENCE_FILENAMES[1]).write_text("{}", encoding="utf-8")
    counts = control_evidence_inventory(tmp_path)
    assert counts[str(runtime / CONTROL_EVIDENCE_FILENAMES[1])] == -1
    assert not runtime_evidence_inventory_complete(receipt(counts), artifact_counts=counts,
                                                   record_count=sum(counts.values()))


# 功能：
#   固定清单包括时序证据，回执追加任意未知路径不能扩大读取范围或取得通过。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_independent_inventory_includes_timing_without_trusting_receipt_paths(tmp_path):
    expected = {}
    for i, name in enumerate(NAVIGATION_EVIDENCE_FILENAMES):
        path = tmp_path / name
        path.write_text(
            "".join(json.dumps({"record": j}) + "\n" for j in range(i + 1)), encoding="utf-8"
        )
        expected[str(path)] = i + 1
    actual = navigation_evidence_inventory(tmp_path)
    assert actual == expected
    assert actual[str(tmp_path / "model-navigation-timing.jsonl")] == 6
    summary = receipt(actual)
    assert runtime_evidence_inventory_complete(summary, artifact_counts=actual, record_count=21)
    summary["artifacts"].append(
        dict(path="../../unrelated.jsonl", submitted_count=0, completed_count=0)
    )
    assert not runtime_evidence_inventory_complete(summary, artifact_counts=actual, record_count=21)


# 功能：
#   有效前缀后出现截断、非对象、非有限数字或坏编码时，整条流计为失败而非前缀数量。
# 输入：
#   tmp_path：测试私有目录。
#   content：被破坏的 JSONL 内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "content",
    [b"{}", b"{}\n{", b"{}\n[]\n", b"{}\nnull\n", b'{}\n{"n":NaN}\n', b'{}\n{"x":"\xff"}\n'],
)
def test_bad_record_cannot_be_hidden_by_counting_valid_prefix(tmp_path, content):
    path = tmp_path / "model-navigation-timing.jsonl"
    path.write_bytes(content)
    counts = navigation_evidence_inventory(tmp_path)
    assert counts[str(path)] == -1
    assert not runtime_evidence_inventory_complete(
        receipt(counts), artifact_counts=counts, record_count=sum(counts.values())
    )


# 功能：
#   预期文件位置实际是目录时，不能当成一条未使用且缺失的流。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_unreadable_stream_is_not_treated_as_absent(tmp_path):
    (tmp_path / "model-navigation-timing.jsonl").mkdir()
    assert (
        navigation_evidence_inventory(tmp_path)[str(tmp_path / "model-navigation-timing.jsonl")]
        == -1
    )


# 功能：
#   未使用的固定流缺失时按零计数，正常关闭的空队列仍可以一致性验收。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_missing_unused_streams_are_zero_and_completed_empty_writer_is_checkable(tmp_path):
    counts = navigation_evidence_inventory(tmp_path)
    assert len(counts) == 6 and set(counts.values()) == {0}
    summary = receipt({})
    assert runtime_evidence_inventory_complete(summary, artifact_counts=counts, record_count=0)


# 功能：
#   逐项破坏文件分配、计数类型或关闭状态，验证只有总数量相等不足以通过。
# 输入：
#   change：对回执注入的异常类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "change",
    [
        "swapped",
        "duplicate",
        "unknown",
        "omitted",
        "row-float",
        "count-float",
        "count-bool",
        "not-closed",
        "rejected",
        "pending",
        "alive",
        "issue",
        "missing-list",
    ],
)
def test_totals_alone_cannot_admit_malformed_or_incomplete_inventory(change):
    counts = {"decisions.jsonl": 3, "timing.jsonl": 2}
    summary = receipt(counts)
    if change == "swapped":
        summary["artifacts"][0].update(submitted_count=2, completed_count=2)
        summary["artifacts"][1].update(submitted_count=3, completed_count=3)
    elif change == "duplicate":
        summary["artifacts"].append(summary["artifacts"][0].copy())
    elif change == "unknown":
        summary["artifacts"][0]["path"] = "../decisions.jsonl"
    elif change == "omitted":
        summary["artifacts"].pop()
    elif change == "row-float":
        summary["artifacts"][0]["completed_count"] = 3.0
    elif change == "count-float":
        summary["completed_count"] = 5.0
    elif change == "count-bool":
        summary["pending_count"] = False
    elif change == "not-closed":
        summary["closed"] = False
    elif change == "rejected":
        summary["rejected_count"] = 1
    elif change == "pending":
        summary["pending_count"] = 1
    elif change == "alive":
        summary["thread_alive"] = True
    elif change == "issue":
        summary["issue_code"] = "ERROR"
    else:
        summary.pop("artifacts")
    assert not runtime_evidence_inventory_complete(summary, artifact_counts=counts, record_count=5)


# 功能：
#   独立计数自身也必须是非负整数，不能信任负数、布尔值或浮点形式的数量。
# 输入：
#   counts：候选独立计数映射。
#   total：候选总记录数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "counts,total", [({"a": -1, "b": 2}, 1), ({"a": True}, 1), ({"a": 1.0}, 1), ({"a": 1}, True)]
)
def test_independent_counts_are_also_strict(counts, total):
    assert not runtime_evidence_inventory_complete(
        receipt(counts), artifact_counts=counts, record_count=total
    )
