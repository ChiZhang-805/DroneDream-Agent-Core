"""Boundary tests for the real transport implementation; no simulated flight claims."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


# 功能：
#   加载当前工作树中的执行器，测试其真实边界函数而非复制实现。
# 输入：
#   无。
# 输出：
#   module：本轮独立加载的执行器模块。
def executor_module():
    path = Path(__file__).resolve().parents[1] / "runtime/px4_offboard_track_executor.py"
    spec = importlib.util.spec_from_file_location("px4_executor_integrity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 功能：
#   验证结构化消息不会借用其他模型的位置，也不会接受歧义或无效旋转。
# 输入：
#   raw：缺失、重复或损坏的 Pose_V 消息。
# 输出：
#   None：每种输入都必须在使用位姿前拒绝。
@pytest.mark.parametrize(
    "raw",
    [
        'pose { name: "drone" } pose { name: "other" position { z: 2 } orientation { w: 1 } }',
        'pose { name: "drone" position {} position {z: 2} orientation { w: 1 } }',
        'pose { name: "drone" position {} orientation {} }',
        'pose { name: "drone" position {} orientation { x: 1e308 w: 1e308 } }',
        'pose { name: "drone" position { x: true } orientation { w: 1 } }',
        'pose { name: "drone" position { x: 1e999 } orientation { w: 1 } }',
        'pose { name: "drone" name: "drone" position {} orientation { w: 1 } }',
        'pose { name: "drone" position {} orientation {w: 1} } ' * 2,
    ],
)
def test_pose_parser_rejects_cross_entity_and_invalid_values(raw):
    base = executor_module()
    with pytest.raises((RuntimeError, ValueError)):
        base._parse_gazebo_model_pose(raw, "drone")


# 功能：
#   检查字符串内的花括号不会破坏消息边界，省略的数值分量按 protobuf 零值解释。
# 输入：
#   无。
# 输出：
#   None：断言准确选取目标且保留有效位置。
def test_pose_parser_handles_quoted_braces_without_crossing_records():
    base = executor_module()
    raw = (
        'header {data {key: "{pose}" value: "quoted }"}} '
        'pose {name: "drone" position {z: 3} orientation {w: 1}}'
    )
    pose = base._parse_gazebo_model_pose(raw, "drone")
    assert (pose.x, pose.y, pose.z) == (0, 0, 3)


# 功能：
#   拒绝两条相互冲突的状态或看起来像状态的尾随文本。
# 输入：
#   raw：具有歧义的关节文本。
# 输出：
#   None：不得解析出成功状态。
@pytest.mark.parametrize(
    "raw", ["data: true data: false", 'data: "attached" garbage', "data: detachedly"]
)
def test_payload_text_is_one_complete_state(raw):
    with pytest.raises(RuntimeError):
        executor_module()._parse_gazebo_payload_detached_state(raw)


# 功能：
#   确认传感器缓存拥有独立快照，原包修改和旧时刻不能覆盖当前值。
# 输入：
#   无。
# 输出：
#   None：断言缓存隔离、重复不续期与时刻倒退拒绝。
def test_native_packet_identity_and_ownership():
    base = executor_module()
    client = object.__new__(base.MavsdkOffboardClient)
    client._dynamics_samples, client._dynamics_errors = {}, {}
    payload = {"timestamp_us": 10, "values": [1.0]}
    client._store_dynamics_sample("imu", payload)
    payload["values"][0] = 99
    assert client._dynamics_samples["imu"][0]["values"] == [1.0]
    with pytest.raises(RuntimeError, match="regressed"):
        client._store_dynamics_sample("imu", {"timestamp_us": 9})
    with pytest.raises(ValueError, match="timestamp"):
        client._store_dynamics_sample("imu", {"timestamp_us": True})


# 功能：
#   确认快速来源报错或具有未来接收时刻时不会授权推断，返回数据也不共享缓存列表。
# 输入：
#   monkeypatch：固定本次消费的单调时刻。
# 输出：
#   None：断言阻断条件及独立快照。
def test_dynamics_errors_and_future_samples_revoke_readiness(monkeypatch):
    base = executor_module()
    monkeypatch.setattr(base.time, "monotonic", lambda: 100.0)
    client = object.__new__(base.MavsdkOffboardClient)
    client._dynamics_samples = {
        "imu": ({"timestamp_us": 1}, 100.0),
        "attitude": ({"timestamp_us": 1}, 100.0),
        "actuator_output": ({"actuator": [0.5], "normalization_ready": True}, 100.0),
    }
    client._dynamics_errors = {"imu": "read-failed"}
    evidence = client.latest_dynamics_telemetry(1.0)
    assert evidence["ready_for_payload_inference"] is False
    evidence["sources"]["actuator_output"]["actuator"][0] = 9
    assert client._dynamics_samples["actuator_output"][0]["actuator"] == [0.5]
    client._dynamics_errors.clear()
    client._dynamics_samples["imu"] = ({"timestamp_us": 1}, 101.0)
    assert client.latest_dynamics_telemetry(1.0)["ready_for_payload_inference"] is False


# 功能：
#   验证无效速度不能传入飞控插件，覆盖布尔、字符串与非有限数。
# 输入：
#   value：无效的前向速度值。
# 输出：
#   None：断言在获取 MAVSDK 传输接口前已经拒绝。
@pytest.mark.parametrize("value", [True, "0", float("nan"), float("inf")])
def test_bad_velocity_never_reaches_transport(value):
    base = executor_module()
    client = object.__new__(base.MavsdkOffboardClient)
    with pytest.raises(ValueError, match="finite native"):
        asyncio.run(client.set_velocity_ned(base.VelocitySetpoint(value, 0.0, 0.0, 0.0)))


# 功能：
#   验证重复 JSON 键及超深证据在读取或创建目录前被拒绝。
# 输入：
#   tmp_path：隔离的本次文件目录。
# 输出：
#   None：失败不生成证据目录。
def test_executor_json_rejects_ambiguous_and_unbounded_input(tmp_path):
    base = executor_module()
    path = tmp_path / "input.json"
    path.write_text('{"reason":"a","reason":"b"}', encoding="utf-8")
    with pytest.raises(ValueError):
        base._load_bounded_json(path, label="test")
    payload = {}
    for _ in range(70):
        payload = {"nested": payload}
    with pytest.raises(ValueError):
        base._write_json_atomic(tmp_path / "uncreated" / "evidence.json", payload)
    assert not (tmp_path / "uncreated").exists()


class EventNode:
    nodes = []

    # 功能：
    #   建立隔离的事件节点替身，不创建 Gazebo 连接。
    # 输入：
    #   self：本次节点。
    # 输出：
    #   None：保存显式测试回调容器。
    def __init__(self):
        self.callbacks = {}
        self.nodes.append(self)

    # 功能：
    #   登记测试回调，等待测试显式投递事件。
    # 输入：
    #   message_type：接口要求的消息类型。
    #   topic：精确状态主题。
    #   callback：被测代码的接收函数。
    # 输出：
    #   accepted：明确的注册确认。
    def subscribe(self, message_type, topic, callback):
        self.callbacks[topic] = callback
        accepted = True
        return accepted

    # 功能：
    #   删除本次测试主题回调，模拟明确退订回执。
    # 输入：
    #   topic：要关闭的主题。
    # 输出：
    #   accepted：明确退订成功。
    def unsubscribe(self, topic):
        self.callbacks.pop(topic, None)
        accepted = True
        return accepted

    # 功能：
    #   返回测试设定的存活发布者，只作为连接信息而不自动产生状态。
    # 输入：
    #   topic：查询主题。
    # 输出：
    #   endpoints：一个发布者及空订阅者列表。
    def topic_info(self, topic):
        endpoints = (["test-publisher"], [])
        return endpoints


# 功能：
#   绑定隔离事件节点到当前执行器，只测试真实订阅与状态更新代码。
# 输入：
#   base：当前执行器模块。
#   monkeypatch：测试替换器。
# 输出：
#   client：无连接的真实客户端实例。
def event_client(base, monkeypatch):
    EventNode.nodes = []
    monkeypatch.setattr(
        base,
        "_gazebo_transport_bindings",
        lambda: (EventNode, object, object, object, object, object),
    )
    client = object.__new__(base.MavsdkOffboardClient)
    return client


# 功能：
#   检查载荷事件按主题隔离，外部新事件会替换旧状态，关闭会退订。
# 输入：
#   monkeypatch：隔离原生传输。
# 输出：
#   None：断言未观察主题不被旧缓存授权，事件年龄没有在读取时归零。
def test_payload_observer_is_topic_bound_and_tracks_later_events(monkeypatch):
    base = executor_module()
    client = event_client(base, monkeypatch)

    # 功能：
    #   在同一事件循环投递连接及分离事件，并完成客户端清理。
    # 输入：
    #   无显式参数；使用本次隔离节点和客户端。
    # 输出：
    #   None：全部断言完成。
    async def scenario():
        observer = client._ensure_payload_observer("/payload/state")
        callback = observer["node"].callbacks["/payload/state"]
        callback(SimpleNamespace(data="attached"))
        await asyncio.sleep(0)
        first = await client.sample_payload_state("/payload/state", 0.02)
        assert first["detached"] is False
        callback(SimpleNamespace(data="detached"))
        await asyncio.sleep(0)
        assert (await client.sample_payload_state("/payload/state", 0.02))["detached"] is True
        with pytest.raises(RuntimeError, match="unknown"):
            await client.sample_payload_state("/another/state", 0.001)
        callback(SimpleNamespace(data="maybe"))
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="invalid"):
            await client.sample_payload_state("/payload/state", 0.001)
        await client.close()
        assert all(not node.callbacks for node in EventNode.nodes)

    asyncio.run(scenario())


# 功能：
#   已失败操作与外部停止同时到达时，等待器仍领取操作异常，且不改变安全停止身份。
# 输入：
#   keepalive：分别验证维持控制与纯停止轮询路径。
# 输出：
#   None：已完成失败被实际读取，世界暂停标志保持为真。
@pytest.mark.parametrize("keepalive", [False, True])
def test_completed_operation_failure_is_retrieved_on_external_abort(keepalive):
    base = executor_module()

    # 功能：
    #   在实际事件循环中构造已失败 Future，并记录等待器是否领取了失败。
    # 输入：
    #   无显式参数；使用外层 base 和 keepalive。
    # 输出：
    #   None：停止不被操作失败替换，且不会遗留未读取的失败。
    async def scenario():
        failures_read = []
        failure = RuntimeError("operation already failed")

        class RecordedFailure(asyncio.Future):
            # 功能：
            #   记录标准 Future 异常领取行为，不用日志时序或垃圾回收推断结果。
            # 输入：
            #   self：已完成的测试 Future。
            # 输出：
            #   error：Future 实际保存的失败对象。
            def exception(self):
                error = super().exception()
                failures_read.append(error)
                return error

        pending = RecordedFailure()
        pending.set_exception(failure)

        # 功能：
        #   使等待器在取得已完成操作结果前遇到真实的安全停止类型。
        # 输入：
        #   无。
        # 输出：
        #   None：通过异常报告世界已暂停。
        def abort():
            raise base.ExternalSafetyAbort("world already paused", world_paused=True)

        try:
            with pytest.raises(base.ExternalSafetyAbort) as caught:
                if keepalive:
                    await base._await_with_setpoint_keepalive(
                        base.FakeOffboardClient(),
                        pending,
                        hold_setpoint=base.Setpoint(0, 0, -1, 0),
                        rate_hz=10,
                        abort_check=abort,
                    )
                else:
                    await base._await_with_abort_polling(pending, abort_check=abort)
            assert caught.value.world_paused is True
            assert failures_read == [failure]
        finally:
            # 测试失败也领取夹具异常，不让反例自身污染后续用例的事件循环诊断。
            asyncio.Future.exception(pending)

    asyncio.run(scenario())


# 功能：
#   证明真实起飞前回执只能在精确运行、摘要及时间范围内初始化持续观察。
# 输入：
#   tmp_path：隔离回执目录。
#   monkeypatch：当前运行身份与节点替换器。
# 输出：
#   None：有效回执保留原时刻，变更后的回执被拒绝。
def test_preflight_payload_receipt_is_run_bound(tmp_path, monkeypatch):
    base = executor_module()
    client = event_client(base, monkeypatch)
    monkeypatch.setenv("GZ_PARTITION", "isolated-integrity-test")
    monkeypatch.setenv("PX4_GAZEBO_WORLD_NAME", "world-test")
    path = tmp_path / "preflight.json"
    observed_at = int(base.time.time() * 1000) - 1000
    base._write_json_atomic(
        path,
        {
            "schema_version": "dronedream.payload-preflight-observation",
            "world": "world-test",
            "partition": "isolated-integrity-test",
            "observed_at_unix_ms": observed_at,
            "observation": {"confirmed": True, "detached": True, "output_topic": "/payload/state"},
        },
    )
    monkeypatch.setenv("PX4_GAZEBO_PAYLOAD_PREFLIGHT_PATH", str(path))
    monkeypatch.setenv(
        "PX4_GAZEBO_PAYLOAD_PREFLIGHT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )

    # 功能：
    #   初始化、读回原生回执来源，再检验文件变化并回收观察资源。
    # 输入：
    #   无显式参数；使用外层生成的隔离回执。
    # 输出：
    #   None：断言原时刻被保留且坏摘要不能继续使用。
    async def scenario():
        await client._prime_payload_observer()
        state = await client.sample_payload_state("/payload/state", 0.01)
        assert state["observed_at_unix_ms"] == observed_at
        assert state["state_age_ms"] >= 1000
        assert state["state_source"] == "verified-preflight-detach-readback"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(RuntimeError, match="changed"):
            await client._prime_payload_observer()
        await client.close()

    asyncio.run(scenario())


# 功能：
#   检查已有任务异常不会阻断其他任务取消及服务退出。
# 输入：
#   无。
# 输出：
#   None：断言所有资源均尝试清理且失败如实上报。
def test_cleanup_continues_after_one_telemetry_task_fails():
    base = executor_module()
    stopped = []

    # 功能：
    #   提供已失败的任务，模拟独立遥测读取故障。
    # 输入：
    #   无。
    # 输出：
    #   None：立即抛出测试异常。
    async def failed():
        raise RuntimeError("sensor failure")

    # 功能：
    #   创建一坏一挂起任务，验证客户端关闭能处理二者。
    # 输入：
    #   无显式参数；使用当前被测客户端类型。
    # 输出：
    #   None：完成异常与清理断言。
    async def scenario():
        client = object.__new__(base.MavsdkOffboardClient)
        bad, waiting = asyncio.create_task(failed()), asyncio.create_task(asyncio.sleep(30))
        await asyncio.sleep(0)
        client._dynamics_tasks = [bad, waiting]
        client._system = SimpleNamespace(_stop_mavsdk_server=lambda: stopped.append(True))
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await client.close()
        assert waiting.cancelled() and stopped == [True]

    asyncio.run(scenario())


# 功能：
#   检查风场文本不会借用其他块的速度或接受冲突开关。
# 输入：
#   raw：非法或歧义的原生文本。
# 输出：
#   None：所有异常输入在形成读回证据前拒绝。
@pytest.mark.parametrize(
    "raw",
    [
        "header {linear_velocity {x: 3}} enable_wind: true",
        "linear_velocity {x: 1 x: 2} enable_wind: true",
        "linear_velocity {} enable_wind: true enable_wind: false",
        "linear_velocity {x: true} enable_wind: true",
        "linear_velocity {x: 1e999} enable_wind: true",
        'linear_velocity {} enable_wind: "true"',
    ],
)
def test_wind_info_requires_unambiguous_native_values(raw):
    with pytest.raises((RuntimeError, ValueError)):
        executor_module()._parse_gazebo_wind_info(raw)


# 功能：
#   验证缺失风速分量按 protobuf 零值解释，不误判为读取失败。
# 输入：
#   无。
# 输出：
#   None：返回精确三轴风速和开启状态。
def test_wind_info_accepts_proto_zero_axes():
    wind = executor_module()._parse_gazebo_wind_info("linear_velocity { y: 3 } enable_wind: true")
    assert wind == {"linear_velocity_mps": {"x": 0.0, "y": 3.0, "z": 0.0}, "enable_wind": True}


# 功能：
#   取消风场激活时等待线程收到停止信号，不让后台控制在拥有者退出后继续执行。
# 输入：
#   无。
# 输出：
#   None：取消结果返回前线程已停止。
def test_wind_activation_cancellation_drains_owner():
    base = executor_module()
    started, stopped = threading.Event(), threading.Event()

    # 功能：
    #   模拟可取消的阻塞工况操作，停止后不继续发布。
    # 输入：
    #   cancel_event：本次拥有者的停止信号。
    #   kwargs：本次世界、配置和激活时刻。
    # 输出：
    #   result：仅用于生命周期测试的空记录。
    def activator(*, cancel_event, **kwargs):
        started.set()
        assert cancel_event.wait(2.0)
        stopped.set()
        result = {}
        return result

    # 功能：
    #   启动并取消实际异步线程包装器，断言其等待线程终止。
    # 输入：
    #   无显式参数；使用外层同步事件。
    # 输出：
    #   None：取消和停止断言完成。
    async def scenario():
        task = asyncio.create_task(base._await_wind_activation(activator, "test", {}, 0))
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()

    asyncio.run(scenario())


# 功能：
#   检查真实进程收集器因取消而停止正在运行的子进程。
# 输入：
#   tmp_path：隔离的测试输出目录。
# 输出：
#   None：子进程不能在拥有者取消后写入迟到成功标志。
def test_gazebo_cli_capture_cancels_real_child(tmp_path):
    base = executor_module()
    cancel = threading.Event()
    marker = tmp_path / "late-result.txt"
    timer = threading.Timer(0.2, cancel.set)
    timer.start()
    try:
        with pytest.raises((RuntimeError, InterruptedError)):
            base._run_gazebo_cli(
                [
                    sys.executable,
                    "-c",
                    "import time,pathlib; time.sleep(10); pathlib.Path("
                    + repr(str(marker))
                    + ").write_text('late')",
                ],
                2.0,
                cancel,
            )
    finally:
        timer.cancel()
        timer.join()
    assert not marker.exists()


# 功能：
#   完成事件和停止请求同时出现时，不能返回旧的成功结果。
# 输入：
#   无。
# 输出：
#   None：优先抛出停止异常。
def test_abort_is_checked_even_for_completed_operation():
    base = executor_module()

    # 功能：
    #   提供已完成 Future，验证等待器仍检查当前停止状态。
    # 输入：
    #   无。
    # 输出：
    #   None：完成操作也不能绕过停止通道。
    async def scenario():
        done = asyncio.get_running_loop().create_future()
        done.set_result("must not return")

        # 功能：
        #   明确触发本次安全停止。
        # 输入：
        #   无。
        # 输出：
        #   None：抛出可辨识的停止错误。
        def abort():
            raise RuntimeError("stop requested")

        with pytest.raises(RuntimeError, match="stop requested"):
            await base._await_with_abort_polling(done, abort_check=abort)

    asyncio.run(scenario())


# 功能：
#   确认无效调度在建立飞控连接前拒绝，而不是在解锁之后才发现。
# 输入：
#   value：非数值或非有限的位置。
#   tmp_path：本次测试日志目录。
# 输出：
#   None：无效调度不会发出连接或解锁命令。
@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf")])
def test_invalid_schedule_never_connects(value, tmp_path):
    base = executor_module()
    client = base.FakeOffboardClient()
    with pytest.raises(ValueError):
        asyncio.run(
            base.run_executor(
                client,
                [base.Setpoint(value, 0, -1, 0)],
                connection="test",
                takeoff_timeout_seconds=1,
                track_timeout_seconds=1,
                rate_hz=10,
                land_after=True,
                log_path=tmp_path / "runtime.log",
            )
        )
    assert not client.connected and not client.armed


# 功能：
#   确认 armable 单项为真不能覆盖缺失本地定位的安全条件。
# 输入：
#   tmp_path：本次隔离日志目录。
# 输出：
#   None：客户端被关闭且从未解锁。
def test_incomplete_readiness_cannot_arm(tmp_path):
    base = executor_module()

    class IncompleteClient(base.FakeOffboardClient):
        # 功能：
        #   提供可解锁但本地定位不就绪的矛盾测试状态。
        # 输入：
        #   timeout_seconds：测试接口预算。
        # 输出：
        #   health：明确缺失本地定位的状态。
        async def wait_until_ready(self, timeout_seconds):
            health = base.TelemetryHealth(True, True, True, False, True)
            return health

    client = IncompleteClient()
    with pytest.raises(RuntimeError, match="incomplete readiness"):
        asyncio.run(
            base.run_executor(
                client,
                [base.Setpoint(0, 0, -1, 0)],
                connection="test",
                takeoff_timeout_seconds=1,
                track_timeout_seconds=1,
                rate_hz=10,
                land_after=True,
                log_path=tmp_path / "runtime.log",
            )
        )
    assert client.closed and not client.armed


# 功能：
#   操作回收失败时仍保留外部停止原因，防止丢失世界暂停标志后误触发降落清理。
# 输入：
#   keepalive：选择保持控制或纯停止轮询这两条实际等待路径。
# 输出：
#   None：原停止异常及暂停标志未被清理错误覆盖。
@pytest.mark.parametrize("keepalive", [False, True])
def test_operation_cleanup_cannot_replace_external_abort(keepalive):
    base = executor_module()

    # 功能：
    #   提供取消时自身回收失败的异步操作。
    # 输入：
    #   无。
    # 输出：
    #   None：取消过程中抛出清理异常。
    async def operation():
        try:
            await asyncio.sleep(30)
        finally:
            raise RuntimeError("cleanup broke")

    # 功能：
    #   对已经进入运行的操作发出明确暂停世界的停止请求。
    # 输入：
    #   无显式参数；使用外层 keepalive 选择实际等待器。
    # 输出：
    #   None：断言停止身份保持原值且失败清理被记录。
    async def scenario():
        pending = asyncio.create_task(operation())
        await asyncio.sleep(0)

        # 功能：
        #   明确表示安全系统已暂停世界，后续不得再假定正在正常飞行。
        # 输入：
        #   无。
        # 输出：
        #   None：抛出本次外部安全停止。
        def abort():
            raise base.ExternalSafetyAbort("paused by safety", world_paused=True)

        with pytest.raises(base.ExternalSafetyAbort) as caught:
            if keepalive:
                await base._await_with_setpoint_keepalive(
                    base.FakeOffboardClient(),
                    pending,
                    hold_setpoint=base.Setpoint(0, 0, -1, 0),
                    rate_hz=10,
                    abort_check=abort,
                )
            else:
                await base._await_with_abort_polling(pending, abort_check=abort)
        assert caught.value.world_paused is True
        assert any("cleanup failed" in note for note in caught.value.__notes__)
        assert pending.done()

    asyncio.run(scenario())
