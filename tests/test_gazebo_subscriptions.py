import threading
from types import SimpleNamespace

import pytest

from dronedream_agent_core.gazebo_subscriptions import (
    GazeboSubscriptions,
    subscription_shutdown_is_complete,
)


# 功能：
#   建立可观察订阅回调和退订顺序的原生节点替身，不启动真实 Gazebo 通信。
# 输入：
#   无。
# 输出：
#   fixture：节点替身、保存的回调字典和退订顺序列表组成的元组。
def node_fixture():
    callbacks, removed = {}, []

    # 功能：
    #   保存主题对应回调并返回原生成功标记，供测试显式触发消息。
    # 输入：
    #   _type：不参与本替身逻辑的消息类型。
    #   topic：订阅主题。
    #   callback：调用方传入的消息处理器。
    # 输出：
    #   accepted：订阅成功标记 True。
    def subscribe(_type, topic, callback):
        callbacks[topic] = callback
        accepted = True
        return accepted

    node = SimpleNamespace(subscribe=subscribe,
        unsubscribe=lambda topic: removed.append(topic) is None)
    fixture = node, callbacks, removed
    return fixture


# 功能：
#   验证逆序退订、关闭后丢弃迟到消息及重复关闭不重复退订，也不允许新增主题。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_close_unsubscribes_in_reverse_order_and_disables_late_callbacks():
    node, callbacks, removed = node_fixture()
    group, delivered = GazeboSubscriptions(node), []
    group.subscribe(object, "/depth", delivered.append)
    group.subscribe(object, "/rgb", delivered.append)
    callbacks["/depth"]("actual-frame")
    result = group.close()
    callbacks["/depth"]("late-frame")
    assert delivered == ["actual-frame"]
    assert removed == ["/rgb", "/depth"]
    assert result == {"complete": True, "subscribed_count": 2, "unsubscribed_count": 2,
                      "active_callbacks_at_close": 0, "errors": []}
    assert group.close() == result
    assert len(removed) == 2
    with pytest.raises(ValueError, match="LIFECYCLE_INVALID"):
        group.subscribe(object, "/new", delivered.append)


# 功能：
#   通过事件控制回调阻塞，确认关闭在原生退订之后等待回调完成再返回。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_shutdown_waits_for_inflight_callback_before_returning():
    node, callbacks, _ = node_fixture()
    entered, release, unsubscribe_started = (threading.Event() for _ in range(3))
    group = GazeboSubscriptions(node)

    # 功能：
    #   通知测试回调已进入，并等待显式释放以验证排空等待。
    # 输入：
    #   _：本测试不使用的消息。
    # 输出：
    #   None：不返回业务数据。
    def callback(_):
        entered.set()
        assert release.wait(2)

    node.unsubscribe = lambda _: unsubscribe_started.set() is None
    group.subscribe(object, "/depth", callback)
    worker = threading.Thread(target=lambda: callbacks["/depth"](None))
    worker.start()
    summaries = []
    closing = threading.Thread(target=lambda: summaries.append(group.close()))
    try:
        assert entered.wait(1)
        closing.start()
        assert unsubscribe_started.wait(1)
        assert not summaries
    finally:
        release.set()
        worker.join(2)
        if closing.ident is not None:
            closing.join(2)
        group.close()
    assert summaries[0]["complete"]
    assert not worker.is_alive() and not closing.is_alive()


# 功能：
#   验证消息处理器抛出异常后执行中计数仍归零，关闭不因泄漏计数而超时。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_callback_exception_cannot_leak_active_count():
    node, callbacks, _ = node_fixture()
    group = GazeboSubscriptions(node)

    # 功能：
    #   模拟消息转换失败，验证生命周期包装器的 finally 归还逻辑。
    # 输入：
    #   _：本测试不使用的消息。
    # 输出：
    #   None：不返回业务数据。
    def callback(_):
        raise ValueError("failed conversion")

    group.subscribe(object, "/depth", callback)
    with pytest.raises(ValueError, match="failed conversion"):
        callbacks["/depth"](None)
    assert group.close()["complete"]


# 功能：
#   验证一个主题拒绝或抛错不会中断其他主题退订，最终回执保留失败。
# 输入：
#   kind：注入的退订失败方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["rejected", "raised"])
def test_unsubscribe_failure_is_reported_and_other_topics_are_still_released(kind):
    node, _, removed = node_fixture()
    group = GazeboSubscriptions(node)
    group.subscribe(object, "/depth", lambda _: None)
    group.subscribe(object, "/rgb", lambda _: None)

    # 功能：
    #   记录退订顺序并只在 RGB 主题注入指定失败，其他主题返回成功。
    # 输入：
    #   topic：待退订主题。
    # 输出：
    #   accepted：非 RGB 主题退订时为 True，RGB 拒绝模式时为 False。
    def unsubscribe(topic):
        removed.append(topic)
        if topic == "/rgb" and kind == "raised":
            raise RuntimeError("native failure")
        accepted = topic != "/rgb"
        return accepted

    node.unsubscribe = unsubscribe
    result = group.close()
    assert not result["complete"] and result["unsubscribed_count"] == 1
    assert len(result["errors"]) == 1 and removed == ["/rgb", "/depth"]


# 功能：
#   确认原生订阅返回 False 时不把失败主题计为已接入的传感器。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_subscribe_does_not_claim_a_live_sensor():
    node, _, _ = node_fixture()
    node.subscribe = lambda *_: False
    group = GazeboSubscriptions(node)
    with pytest.raises(RuntimeError, match="SUBSCRIPTION_FAILED"):
        group.subscribe(object, "/depth", lambda _: None)
    assert group.close()["subscribed_count"] == 0


# 功能：
#   核对深度工作器源码中的资源关闭顺序与清理注册，不冒充原生通信验收。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_depth_worker_releases_subscriptions_before_image_worker_shutdown():
    from pathlib import Path
    source = (Path(__file__).parents[1] / "scripts/runtime_depth_safety_worker.py").read_text(
        encoding="utf-8")
    assert source.index("subscription_summary = subscriptions.close()") < source.index(
        "close_model_image_worker()")
    # 退出栈按逆序执行；异常路径也必须先停止生产回调，再释放图像消费线程。
    assert source.index("close_model_image_worker =") < source.index(
        "cleanup.callback(subscriptions.close)")


# 功能：
#   对深度工作器入口注入失败，确认其清理栈仍执行已注册回收动作。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_depth_worker_failure_still_releases_native_resources():
    from test_runtime_commands import _load_depth_worker

    worker = _load_depth_worker()
    released = []

    # 功能：
    #   注册可观察清理动作后抛出异常，用于检验入口清理栈。
    # 输入：
    #   cleanup：工作器入口持有的资源清理栈。
    # 输出：
    #   None：不返回业务数据。
    def failing(cleanup):
        cleanup.callback(lambda: released.append(True))
        raise ValueError("worker failure")

    worker._run_worker = failing
    with pytest.raises(ValueError, match="worker failure"):
        worker.main()
    assert released == [True]


# 功能：
#   逐项破坏关闭回执，验证验收不能忽略失败标记、残留回调、计数或错误列表。
# 输入：
#   changes：对正常回执注入的字段修改。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("changes", [
    {"complete": False}, {"unsubscribed_count": 1}, {"active_callbacks_at_close": 1},
    {"errors": ["unsubscribe-rejected"]}, {"subscribed_count": True},
    {"unsubscribed_count": 2.0},
])
def test_qualification_cannot_ignore_failed_or_missing_shutdown(changes):
    good = {"complete": True, "subscribed_count": 2, "unsubscribed_count": 2,
            "active_callbacks_at_close": 0, "errors": []}
    assert subscription_shutdown_is_complete(good)
    assert not subscription_shutdown_is_complete({**good, **changes})
    assert not subscription_shutdown_is_complete(None)


# 功能：
#   确认调用方修改返回错误列表不能修改订阅组内部保存的历史关闭结果。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_returned_shutdown_errors_do_not_mutate_retained_receipt():
    node, _, _ = node_fixture()
    group = GazeboSubscriptions(node)
    group.subscribe(object, "/depth", lambda _: None)
    summary = group.close()
    summary["errors"].append("caller-only-diagnostic")
    assert group.close()["errors"] == []


# 功能：
#   验证非法排空超时在退订之前拒绝，随后仍能用合法超时正常清理。
# 输入：
#   timeout：非法排空秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("timeout", [True, float("nan"), float("inf"), -1., 0.])
def test_invalid_drain_timeout_is_rejected_before_unsubscribe(timeout):
    node, _, removed = node_fixture()
    group = GazeboSubscriptions(node)
    group.subscribe(object, "/depth", lambda _: None)
    try:
        with pytest.raises(ValueError, match="DRAIN_TIMEOUT_INVALID"):
            group.close(timeout_seconds=timeout)
        assert removed == []
    finally:
        group.close()


# 功能：
#   验证非法主题或处理器在接触原生订阅前被拒绝，不留下已注册但无法正确处理的回调。
# 输入：
#   topic：非法主题或用于验证处理器的合法主题。
#   callback：候选消息处理器。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(("topic", "callback"), [
    (None, lambda _: None), (5, lambda _: None), ("", lambda _: None),
    (" ", lambda _: None), ("/" + "x" * 1024, lambda _: None),
    ("/depth\0hidden", lambda _: None), ("/depth", None),
])
def test_invalid_subscription_arguments_never_reach_native_node(topic, callback):
    node, callbacks, removed = node_fixture()
    group = GazeboSubscriptions(node)
    try:
        with pytest.raises(ValueError, match="LIFECYCLE_INVALID"):
            group.subscribe(object, topic, callback)
        assert callbacks == {} and removed == []
    finally:
        group.close()


# 功能：
#   验证订阅返回失败或异常后，底层残留的回调不能继续向消费者提交图像。
# 输入：
#   raised：原生订阅以异常还是 False 表示失败。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("raised", [False, True])
def test_failed_registration_cannot_deliver_unowned_frames(raised):
    node, callbacks, _ = node_fixture()
    group, delivered = GazeboSubscriptions(node), []

    # 功能：
    #   保存原生回调后报告订阅失败，模拟原生层还可能送达迟到消息的情况。
    # 输入：
    #   message_type：订阅的消息类型。
    #   topic：订阅主题。
    #   callback：生命周期包装后的处理器。
    # 输出：
    #   accepted：非异常路径返回 False。
    def subscribe(message_type, topic, callback):
        callbacks[topic] = callback
        if raised:
            raise RuntimeError("native-subscribe-error")
        accepted = False
        return accepted

    node.subscribe = subscribe
    try:
        with pytest.raises(RuntimeError):
            group.subscribe(object, "/depth", delivered.append)
        callbacks["/depth"]("unowned")
        assert delivered == []
        assert group.close()["subscribed_count"] == 0
    finally:
        group.close()


# 功能：
#   确认原生注册尚未返回成功时不开放回调，注册成功后的消息才交付消费者。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_callback_delivery_starts_only_after_registration_success():
    node, callbacks, _ = node_fixture()
    group, delivered = GazeboSubscriptions(node), []

    # 功能：
    #   在原生订阅返回前主动触发消息，用于检验注册确认边界。
    # 输入：
    #   message_type：订阅类型。
    #   topic：主题名称。
    #   callback：回调包装器。
    # 输出：
    #   accepted：原生注册成功标记 True。
    def subscribe(message_type, topic, callback):
        callbacks[topic] = callback
        callback("before-confirmation")
        accepted = True
        return accepted

    node.subscribe = subscribe
    try:
        group.subscribe(object, "/depth", delivered.append)
        callbacks["/depth"]("after-confirmation")
        assert delivered == ["after-confirmation"]
    finally:
        group.close()


# 功能：
#   验证回调未排空的关闭结果永久保留为失败，不被稍后的线程退出或重复关闭重写。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_drain_timeout_receipt_stays_failed_after_callback_finishes():
    node, callbacks, _ = node_fixture()
    group = GazeboSubscriptions(node)
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   阻塞当前回调直到测试明确释放，以建立可控的关闭超时情形。
    # 输入：
    #   message：本次回调消息。
    # 输出：
    #   None：不返回业务数据。
    def callback(message):
        entered.set()
        assert release.wait(3)

    group.subscribe(object, "/depth", callback)
    worker = threading.Thread(target=lambda: callbacks["/depth"](None))
    worker.start()
    try:
        assert entered.wait(1)
        result = group.close(timeout_seconds=.01)
        assert result["active_callbacks_at_close"] == 1
        assert result["errors"] == ["callback-drain-timeout"]
        assert not subscription_shutdown_is_complete(result)
    finally:
        release.set()
        worker.join(3)
        group.close()
    assert not worker.is_alive()
    assert group.close() == result


# 功能：
#   验证缺少原生订阅或退订接口的对象不能创建生命周期所有者。
# 输入：
#   node：接口缺失或接口不可调用的候选节点。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("node", [None, SimpleNamespace(subscribe=lambda *_: True),
                                SimpleNamespace(subscribe=True, unsubscribe=lambda _: True)])
def test_native_node_requires_subscription_and_cleanup_interfaces(node):
    with pytest.raises(ValueError, match="NODE_INVALID"):
        GazeboSubscriptions(node)


# 功能：
#   确认重复主题和第九项订阅在原生调用前拒绝，关闭仍回收原有全部主题。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_duplicate_and_over_capacity_subscriptions_preserve_owned_topics():
    node, callbacks, removed = node_fixture()
    group = GazeboSubscriptions(node)
    try:
        for index in range(8):
            group.subscribe(object, f"/sensor-{index}", lambda _: None)
        for topic in ("/sensor-0", "/excess"):
            with pytest.raises(ValueError, match="LIFECYCLE_INVALID"):
                group.subscribe(object, topic, lambda _: None)
        assert len(callbacks) == 8 and "/excess" not in callbacks
    finally:
        result = group.close()
    assert subscription_shutdown_is_complete(result)
    assert removed == [f"/sensor-{index}" for index in reversed(range(8))]
