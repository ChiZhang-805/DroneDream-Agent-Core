import threading
from functools import cached_property
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, PrivateAttr

from dronedream_agent_core.training import evidence_snapshot
from dronedream_agent_core.training.transition_writer import TransitionWriter


class ExtendedEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")
    values: list[int]
    _notes: list[str] = PrivateAttr(default_factory=list)

    # 功能：
    #   构造可变缓存字段，检验快照是否连同非声明字段一起独立持有。
    # 输入：
    #   self：用于证据所有权测试的模型。
    # 输出：
    #   result：当前值的独立列表。
    @cached_property
    def cached_values(self):
        result = self.values.copy()
        return result


# 功能：
#   验证模型额外字段、私有属性与已生成的缓存不会继续共享可变列表。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_model_metadata_and_cached_values_are_detached():
    source = ExtendedEvidence(values=[1], extra_values=[2])
    source._notes.append("original")
    assert source.cached_values == [1]
    detached = evidence_snapshot.detach_evidence(source)
    source.extra_values.append(3)
    source._notes.append("changed")
    source.cached_values.append(4)
    assert detached.extra_values == [2]
    assert detached._notes == ["original"]
    assert detached.cached_values == [1]


# 功能：
#   验证共享子树展开后的实际节点数受限，不能只用最大递归深度限制内存。
# 输入：
#   monkeypatch：把测试节点预算缩小的替换器。
# 输出：
#   None：不返回业务数据。
def test_expanded_node_count_is_bounded(monkeypatch):
    monkeypatch.setattr(evidence_snapshot, "MAX_EVIDENCE_NODES", 100, raising=False)
    shared = list(range(20))
    with pytest.raises(ValueError, match="NODE_BUDGET_EXCEEDED"):
        evidence_snapshot.detach_evidence([shared] * 6)
    assert evidence_snapshot.detach_evidence([shared] * 4) == [shared] * 4


# 功能：
#   验证提交时固定绝对路径，后台线程不会把旧相对路径解释为新的工作目录。
# 输入：
#   tmp_path：本例独立目录。
#   monkeypatch：工作目录替换器。
# 输出：
#   None：不返回业务数据。
def test_submit_binds_path_before_background_publication(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    published = []
    monkeypatch.chdir(tmp_path)
    other = tmp_path / "other"
    other.mkdir()

    # 功能：
    #   暂停实际记录路径，模拟提交完成后工作目录改变的调度次序。
    # 输入：
    #   path：队列交给发布器的路径。
    #   record：已经复制的记录。
    # 输出：
    #   None：不返回业务数据。
    def publish(path, record):
        entered.set()
        assert release.wait(3)
        published.append(path.absolute())

    writer = TransitionWriter(publish)
    try:
        writer.submit(Path("transition.json"), {})
        assert entered.wait(3)
        monkeypatch.chdir(other)
    finally:
        release.set()
        writer.close()
    assert published == [tmp_path / "transition.json"]


# 功能：
#   验证非法发布器在启动工作线程之前被拒绝。
# 输入：
#   publish：不可调用的测试值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("publish", [None, 1, {}, "writer"])
def test_invalid_publisher_is_rejected_before_thread_start(publish):
    with pytest.raises(ValueError, match="WRITER_PUBLISHER_INVALID"):
        writer = TransitionWriter(publish)
        writer.close()


# 功能：
#   验证记录必须是对象，拒绝后队列仍能处理下一条合法记录。
# 输入：
#   tmp_path：本例独立路径。
#   record：错误的顶层记录类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("record", [[], None, "text", 1])
def test_non_object_record_does_not_enter_queue(tmp_path, record):
    published = []
    writer = TransitionWriter(lambda path, value: published.append(value))
    try:
        with pytest.raises(ValueError, match="TRANSITION_RECORD_INVALID"):
            writer.submit(tmp_path / "invalid.json", record)
        writer.submit(tmp_path / "valid.json", {"valid": True})
    finally:
        result = writer.close()
    assert published == [{"valid": True}]
    assert result["submitted"] == result["persisted"] == 1


# 功能：
#   验证排空超时不宣告成功，后续真正写完后重试关闭只计数一次。
# 输入：
#   tmp_path：本例独立路径。
# 输出：
#   None：不返回业务数据。
def test_close_timeout_can_be_retried_without_duplicate_publication(tmp_path):
    entered, release = threading.Event(), threading.Event()
    published = []

    # 功能：
    #   用事件控制一次正在进行的写入，避免通过任意睡眠猜测线程次序。
    # 输入：
    #   path：写入路径。
    #   record：待保存记录。
    # 输出：
    #   None：不返回业务数据。
    def publish(path, record):
        entered.set()
        assert release.wait(3)
        published.append(record)

    writer = TransitionWriter(publish)
    try:
        writer.submit(tmp_path / "pending.json", {"value": 1})
        assert entered.wait(3)
        with pytest.raises(RuntimeError, match="DRAIN_TIMEOUT"):
            writer.close(timeout_seconds=0.001)
    finally:
        release.set()
        result = writer.close()
    assert published == [{"value": 1}]
    assert result["submitted"] == result["persisted"] == 1


# 功能：
#   验证队列溢出后发生的磁盘异常不会覆盖首次失败的原因。
# 输入：
#   tmp_path：本例独立目录。
# 输出：
#   None：不返回业务数据。
def test_first_persistence_failure_remains_available(tmp_path):
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   在队列已经满后注入发布错误，检验后台异常的保留顺序。
    # 输入：
    #   path：写入路径。
    #   record：待保存记录。
    # 输出：
    #   None：不返回业务数据。
    def publish(path, record):
        entered.set()
        assert release.wait(3)
        raise OSError("disk failed after saturation")

    writer = TransitionWriter(publish, capacity=1)
    try:
        writer.submit(tmp_path / "first.json", {})
        assert entered.wait(3)
        writer.submit(tmp_path / "second.json", {})
        with pytest.raises(RuntimeError, match="PERSISTENCE_FAILED") as initial:
            writer.submit(tmp_path / "third.json", {})
    finally:
        release.set()
        with pytest.raises(RuntimeError, match="PERSISTENCE_FAILED") as final:
            writer.close()
    assert final.value.__cause__ is initial.value.__cause__
