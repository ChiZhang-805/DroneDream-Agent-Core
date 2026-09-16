import threading

import pytest

from dronedream_agent_core.training.transition_writer import TransitionWriter


# 功能：
#   验证写入发生在独立线程、记录不随源对象改变，重复关闭不重复写入。
# 输入：
#   tmp_path：测试证据路径所在目录。
# 输出：
#   None：不返回业务数据。
def test_persistence_is_off_thread_complete_and_owns_input(tmp_path):
    entered, release = threading.Event(), threading.Event()
    records, threads = [], []

    # 功能：
    #   用事件暂停写入并记录线程身份，使所有权与线程断言不依赖随机时序。
    # 输入：
    #   path：发布路径。
    #   record：队列传来的独立记录。
    # 输出：
    #   None：不返回业务数据。
    def write(path, record):
        entered.set()
        assert release.wait(2)
        records.append((path, record))
        threads.append(threading.get_ident())

    writer = TransitionWriter(write)
    record = {"axes": [0, 1, 0, 0]}
    writer.submit(tmp_path / "transition.json", record)
    assert entered.wait(2)
    record["axes"][1] = -1
    release.set()
    result = writer.close()
    assert records[0][1]["axes"] == [0, 1, 0, 0]
    assert threads == [writer._thread.ident] and threads[0] != threading.get_ident()
    assert result == {"submitted": 1, "persisted": 1, "complete": True, "simulation_only": True}
    assert writer.close() == result
    with pytest.raises(RuntimeError, match="CLOSED"):
        writer.submit(tmp_path / "second.json", {})


# 功能：
#   验证磁盘失败在关闭与后续检查中持续可见，保留原始异常作为原因。
# 输入：
#   tmp_path：本例独立目录。
# 输出：
#   None：不返回业务数据。
def test_disk_failure_is_not_successful_training(tmp_path):
    # 功能：
    #   注入磁盘写满异常以验证后台错误传递。
    # 输入：
    #   args：发布器收到的路径与记录。
    # 输出：
    #   None：不返回业务数据。
    def fail(*args):
        raise OSError("disk full")

    writer = TransitionWriter(fail)
    writer.submit(tmp_path / "transition.json", {})
    with pytest.raises(RuntimeError, match="PERSISTENCE_FAILED") as error:
        writer.close()
    assert isinstance(error.value.__cause__, OSError)
    with pytest.raises(RuntimeError, match="PERSISTENCE_FAILED"):
        writer.check()


# 功能：
#   验证满队列拒绝新记录但排空已接纳记录，同时不能生成成功回执。
# 输入：
#   tmp_path：本例独立目录。
# 输出：
#   None：不返回业务数据。
def test_saturated_fifo_never_silently_discards_a_transition(tmp_path):
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   暂停正在写入的一条记录，为精确填满等待队列提供同步点。
    # 输入：
    #   args：路径与记录。
    # 输出：
    #   None：不返回业务数据。
    def write(*args):
        entered.set()
        assert release.wait(2)

    writer = TransitionWriter(write, capacity=1)
    writer.submit(tmp_path / "first.json", {})
    assert entered.wait(2)
    writer.submit(tmp_path / "second.json", {})
    try:
        with pytest.raises(RuntimeError, match="PERSISTENCE_FAILED"):
            writer.submit(tmp_path / "third.json", {})
    finally:
        release.set()
    with pytest.raises(RuntimeError, match="PERSISTENCE_FAILED"):
        writer.close()
    assert writer._submitted == writer._completed == 2


# 功能：
#   验证错误关闭参数不会提前关闭写入器，随后合法提交仍能保存。
# 输入：
#   tmp_path：本例独立目录。
#   timeout：非法等待参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("timeout", [None, True, -1., 0., float("nan"), float("inf"), 10**400])
def test_invalid_close_timeout_does_not_close_writer(tmp_path, timeout):
    records = []
    writer = TransitionWriter(lambda path, record: records.append(record))
    try:
        with pytest.raises(ValueError, match="WRITER_CLOSE_TIMEOUT_INVALID"):
            writer.close(timeout_seconds=timeout)
        writer.submit(tmp_path / "still-open.json", {"value": 1})
    finally:
        writer.close()
    assert records == [{"value": 1}]
