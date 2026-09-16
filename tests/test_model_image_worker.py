import threading
import time

import pytest

from dronedream_agent_core.model_image_worker import LatestModelImageWorker


# 功能：
#   在短时有界等待内查找指定来源的编码结果，不把前一帧的结果当作新帧完成。
# 输入：
#   worker：被测试的异步编码器。
#   source_time：需要匹配的来源单调钟秒数。
# 输出：
#   sample：来源时刻精确匹配的已完成图像。
def wait_sample(worker, source_time):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        sample = worker.latest(now_monotonic_seconds=source_time, maximum_age_seconds=.2)
        if sample is not None and sample.image.received_monotonic_seconds == source_time:
            return sample
        time.sleep(.001)
    raise AssertionError("camera conversion did not complete")


# 功能：
#   验证编码确实在后台线程执行，来源身份及年龄保持不变，重复、改时钟和回退被正确处理。
# 输入：
#   tmp_path：图像多模态描述的独立目录。
# 输出：
#   None：不返回业务数据。
def test_background_conversion_keeps_original_source_and_age(tmp_path):
    decoder_threads = []

    # 功能：
    #   记录执行编码的线程并保留输入的三个像素字节。
    # 输入：
    #   message：当前一像素测试消息。
    #   output_size：请求的输出尺寸。
    # 输出：
    #   encoded：固定 PNG 和原 RGB 字节组成的元组。
    def decode(message, *, output_size):
        decoder_threads.append(threading.get_ident())
        encoded = b"png", message
        return encoded
    worker = LatestModelImageWorker(size=(1, 1), decoder=decode)
    try:
        message = b"rgb"
        assert worker.submit(message, received_monotonic_seconds=10., received_at_unix_ms=10000)
        sample = wait_sample(worker, 10.)
        assert sample.message is message
        assert sample.image.multimodal(tmp_path)["observed_at_unix_ms"] == 10000
        assert decoder_threads != [threading.get_ident()]
        assert worker.latest(now_monotonic_seconds=9.99, maximum_age_seconds=.2) is None
        assert worker.latest(now_monotonic_seconds=10.201, maximum_age_seconds=.2) is None
        assert not worker.submit(message, received_monotonic_seconds=10., received_at_unix_ms=10000)
        with pytest.raises(ValueError, match="REDATED"):
            worker.submit(message, received_monotonic_seconds=10., received_at_unix_ms=10100)
        with pytest.raises(ValueError, match="REGRESSED"):
            worker.submit(message, received_monotonic_seconds=9., received_at_unix_ms=9000)
    finally:
        summary = worker.close()
    assert summary == {"submitted": 1, "coalesced": 0, "prepared": 1, "failed": 0}
    assert worker.close() == summary
    assert not worker.submit(b"new", received_monotonic_seconds=11., received_at_unix_ms=11000)


# 功能：
#   验证正在编码时仍能提交新帧，待处理槽只保留最新帧而不形成增长队列。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_only_latest_pending_camera_is_retained_without_blocking_submit():
    started, release = threading.Event(), threading.Event()
    calls = []

    # 功能：
    #   暂停首帧编码，记录后续真正执行的来源，用事件而不是运行速度控制并发顺序。
    # 输入：
    #   message：当前一像素来源字节。
    #   output_size：请求的输出尺寸。
    # 输出：
    #   encoded：固定 PNG 和来源 RGB 字节对。
    def decode(message, *, output_size):
        calls.append(message)
        if message == b"one":
            started.set()
            assert release.wait(2)
        encoded = b"png", message
        return encoded
    worker = LatestModelImageWorker(size=(1, 1), decoder=decode)
    try:
        worker.submit(b"one", received_monotonic_seconds=1., received_at_unix_ms=1000)
        assert started.wait(2)
        worker.submit(b"two", received_monotonic_seconds=2., received_at_unix_ms=2000)
        worker.submit(b"end", received_monotonic_seconds=3., received_at_unix_ms=3000)
        assert calls == [b"one"]
        release.set()
        sample = wait_sample(worker, 3.)
        assert sample.message == b"end" and sample.image.rgb == b"end"
    finally:
        release.set()
        summary = worker.close()
    assert calls == [b"one", b"end"]
    assert summary == {"submitted": 3, "coalesced": 1, "prepared": 2, "failed": 0}


# 功能：
#   验证普通编码失败会隐藏旧图像，下一张合法来源完成后恢复服务并保留真实统计。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_decode_does_not_leave_old_picture_as_current_and_next_frame_recovers():
    # 功能：
    #   为指定来源产生可恢复失败，其余来源使用相同的合法编码格式。
    # 输入：
    #   message：用来区分失败与正常帧的测试字节。
    #   output_size：请求的输出尺寸。
    # 输出：
    #   encoded：正常来源的 PNG、RGB 字节对。
    def decode(message, *, output_size):
        if message == b"bad":
            raise ValueError("bad camera payload")
        encoded = b"png", message
        return encoded
    worker = LatestModelImageWorker(size=(1, 1), decoder=decode)
    try:
        worker.submit(b"one", received_monotonic_seconds=1., received_at_unix_ms=1000)
        wait_sample(worker, 1.)
        worker.submit(b"bad", received_monotonic_seconds=1.01, received_at_unix_ms=1010)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                worker.latest(now_monotonic_seconds=1.02, maximum_age_seconds=.2)
            except ValueError as error:
                assert "ASYNC_PREPARATION_FAILED" in str(error)
                break
            time.sleep(.001)
        else:
            raise AssertionError("decoder failure was not visible")
        worker.submit(b"new", received_monotonic_seconds=1.03, received_at_unix_ms=1030)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                sample = worker.latest(now_monotonic_seconds=1.03, maximum_age_seconds=.2)
                if sample is not None and sample.message == b"new":
                    break
            except ValueError:
                pass
            time.sleep(.001)
        else:
            raise AssertionError("new camera did not recover")
    finally:
        summary = worker.close()
    assert summary["failed"] == 1 and summary["prepared"] == 2


# 功能：
#   验证非法来源时间在提交前被拒绝，不增加工作线程的提交次数。
# 输入：
#   source：待校验的单调钟秒数和 UNIX 毫秒二元组。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("source", [(float("nan"), 1), (-1., 1), (1., True)])
def test_invalid_source_time_rejected_before_submission(source):
    worker = LatestModelImageWorker(size=(1, 1), decoder=lambda *a, **kw: (b"png", b"rgb"))
    try:
        with pytest.raises(ValueError, match="SOURCE_TIME_INVALID"):
            worker.submit(b"rgb", received_monotonic_seconds=source[0],
                          received_at_unix_ms=source[1])
    finally:
        assert worker.close()["submitted"] == 0


# 功能：
#   验证工作器关闭后，曾经合法且仍在年龄范围内的图像也不能继续向消费者提供。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_closed_worker_never_returns_a_retained_camera_frame():
    worker = LatestModelImageWorker(size=(1, 1), decoder=lambda *a, **kw: (b"png", b"rgb"))
    try:
        worker.submit(b"rgb", received_monotonic_seconds=1., received_at_unix_ms=1000)
        wait_sample(worker, 1.)
    finally:
        worker.close()
    assert worker.latest(now_monotonic_seconds=1., maximum_age_seconds=.2) is None


# 功能：
#   验证非法关闭预算先被拒绝，不会提前改变仍在服务的工作器状态。
# 输入：
#   timeout：待测试的关闭等待秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("timeout", [None, True, 0., -1., float("nan"), 10**400])
def test_bad_close_timeout_is_rejected_without_closing_worker(timeout):
    worker = LatestModelImageWorker(size=(1, 1), decoder=lambda *a, **kw: (b"png", b"rgb"))
    try:
        with pytest.raises(ValueError, match="CLOSE_TIMEOUT_INVALID"):
            worker.close(timeout_seconds=timeout)
        assert not worker._closed
    finally:
        worker.close()


# 功能：
#   验证消费者时钟与年龄预算具有稳定错误，而不是抛出未经处理的转换异常。
# 输入：
#   now：消费者当前单调钟秒数。
#   age：消费者允许的图像年龄。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("now,age", [(True, .2), (1., True), (10**400, .2), (1., None)])
def test_invalid_consumer_clock_has_a_stable_failure(now, age):
    worker = LatestModelImageWorker(size=(1, 1), decoder=lambda *a, **kw: (b"png", b"rgb"))
    try:
        with pytest.raises(ValueError, match="CONSUMER_TIME_INVALID"):
            worker.latest(now_monotonic_seconds=now, maximum_age_seconds=age)
    finally:
        worker.close()


# 功能：
#   验证构造工作器后修改原始尺寸列表不会影响实际编码尺寸。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_image_size_is_owned_after_validation():
    size, observed = [1, 1], []

    # 功能：
    #   记录后台编码器看到的固定输出尺寸。
    # 输入：
    #   message：本次测试来源。
    #   output_size：工作器传入的输出尺寸。
    # 输出：
    #   encoded：合法的一像素编码字节对。
    def decode(message, *, output_size):
        observed.append(output_size)
        encoded = b"png", b"rgb"
        return encoded
    worker = LatestModelImageWorker(size=size, decoder=decode)
    try:
        size[0] = 5000
        worker.submit(b"rgb", received_monotonic_seconds=1., received_at_unix_ms=1000)
        wait_sample(worker, 1.)
        assert observed == [(1, 1)]
    finally:
        worker.close()


# 功能：
#   验证无效编码器在启动线程前被拒绝，而不是等到第一帧才出现异步失败。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_invalid_decoder_is_rejected_before_thread_start():
    worker = None
    try:
        with pytest.raises(ValueError, match="MODEL_RGB_DECODER_INVALID"):
            worker = LatestModelImageWorker(size=(1, 1), decoder=None)
    finally:
        if worker is not None:
            worker.close()


# 功能：
#   验证关闭超时后迟到的编码失败不会重新存入错误、发布图像或增加完成统计。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_late_decoder_failure_cannot_repopulate_closed_worker():
    started, release = threading.Event(), threading.Event()

    # 功能：
    #   阻塞编码直至测试发起关闭，再产生受控异常。
    # 输入：
    #   message：本次来源消息。
    #   output_size：请求的图像尺寸。
    # 输出：
    #   None：不返回业务数据。
    def decode(message, *, output_size):
        started.set()
        assert release.wait(3)
        raise ValueError("late failure")

    worker = LatestModelImageWorker(size=(1, 1), decoder=decode)
    try:
        worker.submit(b"rgb", received_monotonic_seconds=1., received_at_unix_ms=1000)
        assert started.wait(2)
        with pytest.raises(RuntimeError, match="DID_NOT_STOP"):
            worker.close(timeout_seconds=.01)
        assert worker.latest(now_monotonic_seconds=1., maximum_age_seconds=.2) is None
    finally:
        release.set()
        summary = worker.close()
    assert worker._error is None
    assert summary["failed"] == 0 and summary["prepared"] == 0


# 功能：
#   验证持续保存的异步诊断不持有解码器回溯或错误消息中的大块像素数据。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_worker_error_does_not_retain_decoder_traceback_or_payload():
    failed = threading.Event()

    # 功能：
    #   从局部大缓冲构造异常，检查工作线程如何保留失败诊断。
    # 输入：
    #   message：本次来源消息。
    #   output_size：请求的图像尺寸。
    # 输出：
    #   None：不返回业务数据。
    def decode(message, *, output_size):
        payload = "pixel-data-" * 10000
        failed.set()
        raise ValueError(payload)

    worker = LatestModelImageWorker(size=(1, 1), decoder=decode)
    try:
        worker.submit(b"rgb", received_monotonic_seconds=1., received_at_unix_ms=1000)
        assert failed.wait(2)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with worker._condition:
                error = worker._error
            if error is not None:
                break
            time.sleep(.001)
        assert error is not None
        assert error.__traceback__ is None and error.__context__ is None
        assert "pixel-data" not in str(error)
        assert len(str(error)) < 160
    finally:
        worker.close()


# 功能：
#   验证编码线程遇到终止类异常后明确拒绝读取和新提交，不保留永远无人处理的任务。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_terminal_decoder_exit_is_visible_to_producers_and_consumers():
    entered = threading.Event()

    # 功能：
    #   模拟编码器主动结束其工作线程，不影响运行测试的主线程。
    # 输入：
    #   message：本次来源消息。
    #   output_size：请求的输出尺寸。
    # 输出：
    #   None：不返回业务数据。
    def decode(message, *, output_size):
        entered.set()
        raise SystemExit("decoder stopped")

    worker = LatestModelImageWorker(size=(1, 1), decoder=decode)
    try:
        worker.submit(b"rgb", received_monotonic_seconds=1., received_at_unix_ms=1000)
        assert entered.wait(2)
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._stopped, timeout=2)
        with pytest.raises(ValueError, match="MODEL_RGB_WORKER_STOPPED"):
            worker.latest(now_monotonic_seconds=1., maximum_age_seconds=.2)
        with pytest.raises(ValueError, match="MODEL_RGB_WORKER_STOPPED"):
            worker.submit(b"new", received_monotonic_seconds=2., received_at_unix_ms=2000)
        assert worker._pending is None
    finally:
        summary = worker.close()
    assert summary == {"submitted": 1, "coalesced": 0, "prepared": 0, "failed": 1}


# 功能：
#   验证关闭时撤销待处理槽，进行中的编码随后成功也不能重新发布图片。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_late_success_and_pending_frame_are_withdrawn_during_close():
    started, release = threading.Event(), threading.Event()
    calls = []

    # 功能：
    #   暂停首帧编码，允许测试在完成之前提交第二帧并关闭工作器。
    # 输入：
    #   message：当前处理的来源消息。
    #   output_size：请求的输出尺寸。
    # 输出：
    #   encoded：受控返回的编码字节对。
    def decode(message, *, output_size):
        calls.append(message)
        started.set()
        assert release.wait(3)
        encoded = b"png", message
        return encoded

    worker = LatestModelImageWorker(size=(1, 1), decoder=decode)
    try:
        worker.submit(b"one", received_monotonic_seconds=1., received_at_unix_ms=1000)
        assert started.wait(2)
        worker.submit(b"two", received_monotonic_seconds=2., received_at_unix_ms=2000)
        with pytest.raises(RuntimeError, match="DID_NOT_STOP"):
            worker.close(timeout_seconds=.01)
    finally:
        release.set()
        summary = worker.close()
    assert calls == [b"one"]
    assert summary == {"submitted": 2, "coalesced": 0, "prepared": 0, "failed": 0}
    assert worker.latest(now_monotonic_seconds=2., maximum_age_seconds=.2) is None
