import threading

import pytest

from dronedream_agent_core.preview_worker import LatestPreviewWorker


# 功能：
#   用真实线程验证预览阻塞时只保留最后一帧，提交不等待界面处理。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_blocked_preview_coalesces_frames_without_blocking_submit():
    entered, release, latest = threading.Event(), threading.Event(), threading.Event()
    published = []

    # 功能：
    #   阻塞首帧发布并记录最终帧，以事件同步替代假定的线程执行顺序。
    # 输入：
    #   frame：本次预览编号。
    # 输出：
    #   None：不返回业务数据。
    def publish(frame):
        if frame == 0:
            entered.set()
            assert release.wait(2)
        published.append(frame)
        if frame == 20:
            latest.set()

    worker = LatestPreviewWorker(publish)
    try:
        assert worker.submit(0)
        assert entered.wait(2)
        for index in range(1, 21):
            assert worker.submit(index)
        release.set()
        assert latest.wait(2)
    finally:
        release.set()
        summary = worker.close()
    assert published == [0, 20]
    assert summary["replaced_pending"] == 19
    assert summary["received"] == 21 and summary["published"] == 2
    assert not worker.submit(21)


# 功能：
#   验证单帧发布异常被计为失败，后续帧仍可发布且不冒充传感器证据。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_preview_errors_do_not_kill_transport_or_silently_report_success():
    failed, success = threading.Event(), threading.Event()

    # 功能：
    #   令首帧失败并用后续帧触发完成事件。
    # 输入：
    #   frame：当前帧编号。
    # 输出：
    #   None：不返回业务数据。
    def publish(frame):
        if frame == 0:
            failed.set()
            raise ValueError("bad preview")
        success.set()

    worker = LatestPreviewWorker(publish)
    try:
        worker.submit(0)
        assert failed.wait(2)
        worker.submit(1)
        assert success.wait(2)
    finally:
        summary = worker.close()
    assert summary["failed"] == 1 and summary["published"] == 1
    assert summary["purpose"] == "ui-preview-only"


# 功能：
#   验证没有发布函数时立即拒绝，不启动无效后台线程。
# 输入：
#   publish：不可调用的发布配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("publish", [None, "publish", 3])
def test_invalid_publisher_rejected_before_thread_start(publish):
    with pytest.raises(TypeError, match="PREVIEW_PUBLISHER_NOT_CALLABLE"):
        LatestPreviewWorker(publish)


# 功能：
#   验证空帧不被计为已接收且无声消失。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_none_frame_rejected_without_incrementing_received_count():
    worker = LatestPreviewWorker(lambda frame: None)
    try:
        with pytest.raises(ValueError, match="PREVIEW_FRAME_MISSING"):
            worker.submit(None)
    finally:
        summary = worker.close()
    assert summary["received"] == 0


# 功能：
#   验证关闭超时不谎报线程退出；关闭丢弃的待发布帧进入明确统计。
# 输入：
#   monkeypatch：临时缩短真实线程 join 等待的测试工具。
# 输出：
#   None：不返回业务数据。
def test_blocked_close_reports_failure_then_can_finish_with_balanced_counts(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   保持一个真实发布回调在运行中，直到测试明确允许退出。
    # 输入：
    #   frame：发布帧，本夹具无需读取其内容。
    # 输出：
    #   None：不返回业务数据。
    def publish(frame):
        entered.set()
        assert release.wait(2)

    worker = LatestPreviewWorker(publish)
    try:
        worker.submit(0)
        assert entered.wait(2)
        worker.submit(1)
        join = worker._thread.join
        with monkeypatch.context() as patch:
            patch.setattr(worker._thread, "join", lambda timeout: join(timeout=0))
            with pytest.raises(RuntimeError, match="SIMULATION_PREVIEW_WORKER_DID_NOT_STOP"):
                worker.close()
        assert not worker.submit(2)
    finally:
        release.set()
        summary = worker.close()
    assert summary["discarded_pending"] == 1
    assert summary["received"] == sum(summary[key] for key in (
        "published", "failed", "replaced_pending", "discarded_pending"))
    assert worker.close() == summary
