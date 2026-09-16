import hmac
import json
import socket
import threading
import time
from collections import deque
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_training_runtime_evidence import pose

from dronedream_agent_core import gazebo_adapter
from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.training import outcome_channel as channel
from dronedream_agent_core.training.gazebo_witness import GazeboOutcomeWitness
from dronedream_agent_core.training.runtime_evidence import (
    IndependentPoseMonitor,
    OutcomeWindowError,
)


# 功能：
#   建立同一轮独立真值的真实本机收发通道，测试结束后依次关闭发布器和接收器。
# 输入：
#   tmp_path：当前测试的独立描述文件目录。
# 输出：
#   connection：包含接收器和发布器的二元组，交给测试使用一次。
@pytest.fixture
def endpoints(tmp_path):
    receiver = channel.OutcomeReceiver(tmp_path / "outcome.json")
    publisher = channel.OutcomePublisher(receiver.path)
    try:
        connection = receiver, publisher
        yield connection
    finally:
        publisher.close()
        receiver.close()


# 功能：
#   有界轮询具体就绪条件；条件报错立即传播，超时不能被当作已经收到证据。
# 输入：
#   check：返回布尔就绪状态的无参检查函数。
# 输出：
#   None：不返回业务数据。
def wait_for(check):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.002)
    raise AssertionError("observer did not receive evidence")


# 功能：
#   验证独立评分通道逐帧保留而非只取最新帧，读取后不重复交付旧记录。
# 输入：
#   endpoints：真实本机接收器与发布器。
# 输出：
#   None：不返回业务数据。
def test_channel_retains_all_original_witnesses_not_latest_only(endpoints):
    receiver, publisher = endpoints
    rows = [pose(n, n / 10) for n in range(1, 5)]
    for row in rows:
        assert publisher.send(row)
    assert receiver.read() == rows
    assert receiver.read() == []


# 功能：
#   验证重放帧或中间丢帧都使真值链失败，不能按剩余数据假造连续轨迹。
# 输入：
#   endpoints：本机真值收发通道。
#   sequence：第二次发送时故意重复或跳过的序号。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("sequence", [1, 3])
def test_channel_rejects_replay_or_missing_witness(endpoints, sequence):
    receiver, publisher = endpoints
    publisher.send(pose(1, 0))
    assert len(receiver.read()) == 1
    publisher.send(pose(sequence, 1))
    with pytest.raises(ValueError, match="LOST_OR_REPLAYED"):
        receiver.read()


# 功能：
#   验证其他轮次密钥签名的数据不能成为当前独立真值。
# 输入：
#   endpoints：当前轮次的收发通道。
# 输出：
#   None：不返回业务数据。
def test_different_episode_key_cannot_inject_witness(endpoints):
    receiver, publisher = endpoints
    publisher._secret = b"wrong-run" * 4
    publisher.send(pose(1, 0))
    with pytest.raises(ValueError, match="AUTHENTICATION"):
        receiver.read()


# 功能：
#   验证即使签名合法，机载估计也不能冒充独立仿真真值；发送和接收两端均拒绝。
# 输入：
#   endpoints：签名密钥相同的本机收发通道。
# 输出：
#   None：不返回业务数据。
def test_authenticated_sensor_packet_cannot_replace_independent_truth(endpoints):
    receiver, publisher = endpoints
    row = pose(1, 0).model_copy(update={"source": "onboard"})
    with pytest.raises(ValueError, match="INDEPENDENT"):
        publisher.send(row)
    body = row.model_dump_json().encode()
    packet = hmac.digest(publisher._secret, body, "sha256") + body
    publisher._socket.sendto(packet, channel._address(publisher._value))
    with pytest.raises(ValueError, match="INDEPENDENT"):
        receiver.read()


# 功能：
#   验证启动后真值收发路径不再读取描述文件，防止热路径磁盘阻塞或重新绑定。
# 输入：
#   endpoints：已完成初始化的通道。
#   monkeypatch：临时禁止所有 Path.open 调用。
# 输出：
#   None：不返回业务数据。
def test_no_descriptor_rebind_or_file_read_after_start(endpoints, monkeypatch):
    receiver, publisher = endpoints
    with monkeypatch.context() as patched:
        patched.setattr(Path, "open", lambda *a, **k: pytest.fail("live path read disk"))
        publisher.send(pose(1, 0))
        assert receiver.read() == [pose(1, 0)]


# 功能：
#   模拟非阻塞发送队列已满，核对丢包计数与下帧原始序号，接收端必须发现缺帧。
# 输入：
#   endpoints：待模拟队列满的本机通道。
#   monkeypatch：暂时替换发送 socket。
# 输出：
#   None：不返回业务数据。
def test_nonblocking_send_loss_does_not_renumber_next_witness(endpoints, monkeypatch):
    receiver, publisher = endpoints
    with monkeypatch.context() as patched:
        # 功能：
        #   模拟非阻塞发送缓冲已满，不实际发送数据。
        # 输入：
        #   a：发送器传入的数据包和目标地址。
        # 输出：
        #   None：不返回业务数据。
        def full(*a):
            raise BlockingIOError()
        patched.setattr(publisher, "_socket", SimpleNamespace(sendto=full))
        assert not publisher.send(pose(1, 0))
    assert publisher.dropped == 1
    publisher.send(pose(2, 0))
    with pytest.raises(ValueError, match="LOST_OR_REPLAYED"):
        receiver.read()


# 功能：
#   验证协议、地址或密钥损坏的描述不能初始化发布器，并保留被其他写入者改动的文件。
# 输入：
#   tmp_path：描述文件目录。
#   bad：选择损坏字段的方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", ["wrong protocol", "remote address", "invalid secret"])
def test_descriptor_validation_precedes_socket_use(tmp_path, bad):
    path = tmp_path / "outcome.json"
    receiver = channel.OutcomeReceiver(path)
    try:
        value = json.loads(path.read_text())
        if bad == "wrong protocol":
            value["protocol"] = "dronedream-native-state"
        elif bad == "remote address":
            value["address"] = "192.0.2.1"
        else:
            value["secret"] = "z" * 64
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="OUTCOME_"):
            channel.OutcomePublisher(path)
    finally:
        receiver.close()
    assert path.exists()  # Never remove another owner's replacement.


# 功能：
#   验证缺少显式通道描述时直接报错，不能悄悄创建备用真值来源。
# 输入：
#   tmp_path：不存在描述文件的测试目录。
# 输出：
#   None：不返回业务数据。
def test_missing_descriptor_does_not_start_a_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        channel.OutcomePublisher(tmp_path / "missing.json")


# 功能：
#   验证描述文件独占创建、关闭可重复调用，以及已关闭接收器不能继续读取。
# 输入：
#   tmp_path：两个接收器尝试共用的描述文件目录。
# 输出：
#   None：不返回业务数据。
def test_exclusive_descriptor_and_idempotent_close(tmp_path):
    path = tmp_path / "outcome.json"
    receiver = channel.OutcomeReceiver(path)
    try:
        with pytest.raises(FileExistsError):
            channel.OutcomeReceiver(path)
    finally:
        receiver.close()
        receiver.close()
    assert not path.exists()
    with pytest.raises(RuntimeError, match="CLOSED"):
        receiver.read()


# 功能：
#   分阶段发送两帧，核对基线与完整窗口的区别，再验证第三帧时钟倒退会使监视器失败。
# 输入：
#   tmp_path：实际线程和本机通道使用的描述文件目录。
# 输出：
#   None：不返回业务数据。
def test_monitor_retains_contiguous_clock_and_rejects_regression(tmp_path):
    monitor = IndependentPoseMonitor(tmp_path / "outcome.json")
    publisher = channel.OutcomePublisher(monitor.path)
    try:
        publisher.send(pose(1, 0))
        wait_for(lambda: monitor.has_start_baseline(1100))
        # 起点基线只需要第一帧；不能把它当成第二帧已经到达的同步信号。
        with pytest.raises(OutcomeWindowError, match="NOT_YET_OBSERVED"):
            monitor.window(1050, 1100)
        publisher.send(pose(2, .1))
        wait_for(lambda: monitor.initial_witness(1100).sequence == 2)
        assert len(monitor.window(1050, 1100)) == 2
        publisher.send(pose(3, .2).model_copy(update={"observed_at_unix_ms": 1099}))
        wait_for(lambda: monitor._error is not None)
        with pytest.raises(ValueError, match="CLOCK_REGRESSED"):
            monitor.window(1050, 1100)
    finally:
        publisher.close()
        monitor.close()


# 功能：
#   构造保持默认朝向的 Gazebo 风格实体，供模型位姿和规范链接组合测试使用。
# 输入：
#   name：实体名。
#   x：实体横向位置，米。
#   y：实体纵向位置，米。
#   z：实体高度，米。
# 输出：
#   result：带名称、位置及单位四元数的测试实体。
def entity(name, x=0., y=0., z=0.):
    result = SimpleNamespace(name=name, position=SimpleNamespace(x=x, y=y, z=z),
                             orientation=SimpleNamespace(w=1., x=0., y=0., z=0.))
    return result


# 功能：
#   构造测试机体和相对机体高 0.2 米的规范链接。
# 输入：
#   x：机体横向位置，米。
# 输出：
#   entities：机体与规范链接两项测试实体。
def model_entities(x=0.):
    entities = [entity("test_drone", x=x), entity("base_link", z=.2)]
    return entities


# 功能：
#   将真实位姿解析器与测试规范链接绑定，构造不依赖诊断磁盘快照的真值采集器。
# 输入：
#   publisher：本机独立真值通道发布器。
# 输出：
#   observer：使用明确机体、链接、碰撞中心及目标的真值采集器。
def witness(publisher):
    observer = GazeboOutcomeWitness(
        publisher=publisher,
        resolve_pose=partial(gazebo_adapter._resolve_controlled_vehicle_pose,
                             vehicle_name="test_drone",
                             collision_center_offset_model_m=(0., 0., .2),
                             frames={"vehicle_model_name": "test_drone",
                                 "canonical_link_name": "base_link",
                                 "canonical_at_rest": {"position_m": [0, 0, .2],
                                                       "orientation_wxyz": [1, 0, 0, 0]},
                                 "collision_center_model_m": [0, 0, .2]}),
        collision_offset=(0., 0., .2), target=Vector3(x=4., y=5., z=6.),
        dynamic_geometry={"person_worker": 1.8},
    )
    return observer


# 功能：
#   验证组合机体与规范链接得到真实控制点，保留源时间，并移除已经消失的动态物体。
# 输入：
#   endpoints：独立评分收发通道。
#   monkeypatch：禁止快速回调读磁盘。
# 输出：
#   None：不返回业务数据。
def test_fast_witness_uses_actual_canonical_link_and_original_clock(endpoints, monkeypatch):
    receiver, publisher = endpoints
    observer = witness(publisher)
    with monkeypatch.context() as patched:
        patched.setattr(Path, "open", lambda *a, **k: pytest.fail("witness read disk"))
        first = [entity("test_drone", 1., 2., 3.), entity("base_link", .5, .6, .7),
                 entity("person_worker", 0., 1., 0.)]
        observer.receive(first, time.monotonic() - .3, 1000, simulation_time_ns=0)
        second = [entity("test_drone", 1., 2., 3.), entity("base_link", .8, .6, .7)]
        observer.receive(second, time.monotonic(), 1300, simulation_time_ns=300_000_000)
        rows = receiver.read()
    assert len(rows) == 2
    assert rows[0].observed_at_unix_ms == 1000 and not rows[0].stream_healthy
    assert rows[0].current_position_m == Vector3(x=1.5, y=2.6, z=3.7)
    assert rows[1].current_position_m.x == 1.8
    assert rows[1].current_velocity_mps.x == pytest.approx(1., abs=.05)
    assert rows[0].dynamic_obstacles[0].obstacle_id == "person_worker"
    assert rows[1].dynamic_obstacles == []  # No stale, disappeared entity cache.
    assert observer.error is None


# 功能：
#   阻塞独立诊断线程，验证真值回调仍能发送，不被诊断工作等待拖住。
# 输入：
#   endpoints：实际本机真值通道。
# 输出：
#   None：不返回业务数据。
def test_fast_witness_continues_while_diagnostic_worker_is_blocked(endpoints):
    receiver, publisher = endpoints
    observer = witness(publisher)
    blocked, release = threading.Event(), threading.Event()

    # 功能：
    #   模拟正在等待磁盘工作的诊断线程，通过事件保证测试可控结束。
    # 输入：
    #   无；借用外层 blocked 和 release 事件。
    # 输出：
    #   None：不返回业务数据。
    def disk_worker():
        blocked.set()
        release.wait(2)
    thread = threading.Thread(target=disk_worker)
    thread.start()
    try:
        assert blocked.wait(1)
        observer.receive(model_entities(), time.monotonic(), 1000, simulation_time_ns=0)
        assert len(receiver.read()) == 1 and thread.is_alive()
    finally:
        release.set()
        thread.join(timeout=1)


# 功能：
#   验证找不到目标机体或位姿解析异常时不发送猜测位置，并记录真实解析错误。
# 输入：
#   endpoints：待检查的真值通道。
# 输出：
#   None：不返回业务数据。
def test_missing_vehicle_or_callback_failure_never_invents_a_pose(endpoints):
    receiver, publisher = endpoints
    observer = witness(publisher)
    observer.receive([entity("other_drone")], time.monotonic(), 1000, simulation_time_ns=0)
    assert receiver.read() == []
    observer.resolve_pose = lambda poses: (_ for _ in ()).throw(ValueError("wrong pose"))
    observer.receive([], time.monotonic(), 1050, simulation_time_ns=50_000_000)
    assert observer.error == "ValueError:wrong pose" and receiver.read() == []


# 功能：
#   验证截断或超大数据包在解析前被接收器拒绝。
# 输入：
#   endpoints：等待非法包的接收器与描述信息。
#   packet：过短或超出协议字节数的数据包。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("packet", [b"short", b"x" * (channel.MAX_PACKET_BYTES + 1)],
                         ids=["truncated", "oversized"])
def test_oversize_and_truncated_packets_are_rejected(endpoints, packet):
    receiver, publisher = endpoints
    with socket.socket(channel.FAMILY, socket.SOCK_DGRAM) as sender:
        sender.sendto(packet, channel._address(publisher._value))
    with pytest.raises(ValueError, match="PACKET_SIZE"):
        receiver.read()


# 功能：
#   验证非法单调时钟或毫秒时间不能产生健康真值帧。
# 输入：
#   endpoints：真值通道。
#   stamp：待拒绝的单调时钟秒值。
#   unix_ms：待检查的源时间毫秒值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("stamp,unix_ms", [(float("nan"), 1000), (float("inf"), 1000),
    (-1., 1000), (True, 1000), (1., True), (1., -1)])
def test_witness_rejects_invalid_source_clocks(endpoints, stamp, unix_ms):
    receiver, publisher = endpoints
    observer = witness(publisher)
    observer.receive(model_entities(), stamp, unix_ms, simulation_time_ns=0)
    assert observer.error == "ValueError:OUTCOME_SOURCE_CLOCK_INVALID"
    assert receiver.read() == []


# 功能：
#   验证节流间隔内也检查源时间倒退，不能因该帧不发布而忽略时间异常。
# 输入：
#   endpoints：实际本机真值通道。
# 输出：
#   None：不返回业务数据。
def test_throttled_callback_does_not_hide_source_clock_regression(endpoints):
    receiver, publisher = endpoints
    observer = witness(publisher)
    now = time.monotonic()
    observer.receive(model_entities(), now - .1, 1000, simulation_time_ns=0)
    observer.receive(model_entities(), now - .09, 1010, simulation_time_ns=10_000_000)
    observer.receive(model_entities(), now - .095, 1005, simulation_time_ns=20_000_000)
    assert observer.error == "ValueError:OUTCOME_SOURCE_CLOCK_REGRESSED"
    assert len(receiver.read()) == 1


# 功能：
#   验证未来时间不能经负年龄截零后伪装成新鲜健康帧。
# 输入：
#   endpoints：待检查的真值通道。
# 输出：
#   None：不返回业务数据。
def test_future_source_clock_does_not_become_zero_age_healthy_witness(endpoints):
    receiver, publisher = endpoints
    observer = witness(publisher)
    observer.receive(model_entities(), time.monotonic() + 10, 1000, simulation_time_ns=0)
    assert observer.error == "ValueError:OUTCOME_SOURCE_CLOCK_INVALID"
    assert receiver.read() == []


# 功能：
#   验证非均匀到达节拍保留必要边缘帧，缺失时仍严格拒绝窗口，不借未来观测补齐。
# 输入：
#   endpoints：采集器输出通道。
#   monkeypatch：固定主机单调时钟以控制源帧到达时刻。
# 输出：
#   None：不返回业务数据。
def test_witness_cadence_preserves_short_native_arrivals_at_window_edge(endpoints, monkeypatch):
    receiver, publisher = endpoints
    observer = witness(publisher)
    # Representative alternating 35/74 ms native arrivals: the previous
    # 40 ms throttle retained only 0/109/217, losing the 144 ms observation.
    # Do not use the later 217 ms sample to fill the 215 ms window boundary.
    clock = [9.9]
    monkeypatch.setattr("dronedream_agent_core.training.gazebo_witness.time.monotonic",
                        lambda: clock[0])
    # 先取得速度历史；首帧无速度证据不应参与健康窗口验收。
    observer.receive(model_entities(x=-.1), clock[0], 900, simulation_time_ns=900_000_000)
    assert receiver.read()[0].stream_healthy is False
    for offset in (0, 35, 109, 144, 217):
        clock[0] = 10 + offset / 1000
        observer.receive(model_entities(x=offset / 1000), clock[0], 1000 + offset,
                         simulation_time_ns=(1000 + offset) * 1_000_000)
    rows = receiver.read()
    assert [row.observed_at_unix_ms for row in rows] == [1000, 1035, 1109, 1144, 1217]
    assert observer.error is None
    monitor = object.__new__(IndependentPoseMonitor)
    monitor._lock, monitor._rows, monitor._error = threading.Lock(), deque(rows), None
    assert [row.observed_at_unix_ms for row in monitor.window(1000, 1215)] == [
        1000, 1035, 1109, 1144]
    # The original missing-evidence gate is still strict, even when a healthy
    # future observation is already buffered. No interpolated/relabelled truth.
    monitor._rows = deque([rows[0], rows[2], rows[4]])
    with pytest.raises(OutcomeWindowError, match="UNVERIFIED_GAP") as failure:
        monitor.window(1000, 1215)
    assert failure.value.diagnostics["end_gap_ms"] == 106


# 功能：
#   验证高频到达按最小发布间隔有界节流，并明确计数省略帧，不授予训练资格。
# 输入：
#   endpoints：采集器输出通道。
#   monkeypatch：可步进的主机时钟。
# 输出：
#   None：不返回业务数据。
def test_witness_throttle_remains_bounded_and_reports_omissions(endpoints, monkeypatch):
    receiver, publisher = endpoints
    observer = witness(publisher)
    clock = [20.]
    monkeypatch.setattr("dronedream_agent_core.training.gazebo_witness.time.monotonic",
                        lambda: clock[0])
    for offset in range(100):
        clock[0] = 20 + offset / 1000
        observer.receive(model_entities(), clock[0], 1000 + offset,
                         simulation_time_ns=offset * 1_000_000)
    rows = receiver.read()
    assert 4 <= len(rows) <= 5
    assert all(b.observed_at_unix_ms - a.observed_at_unix_ms >= 20
               for a, b in zip(rows, rows[1:], strict=False))
    receipt = observer.close()
    assert receipt["skipped_cadence"] == 100 - len(rows)
    assert receipt["minimum_publish_interval_seconds"] == .02
    assert receipt["qualification_granted"] is False


# 功能：
#   验证相同仿真位移和仿真时差得到相同速度，不能随主机运行快慢改变物理速度。
# 输入：
#   endpoints：真值收发通道。
#   monkeypatch：控制主机时间的测试替换器。
#   wall_dt：两次回调之间的主机时间差，秒。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("wall_dt", [.03, .06, .12, .5])
def test_physical_velocity_does_not_scale_with_runtime_speed(endpoints, monkeypatch, wall_dt):
    receiver, publisher = endpoints
    observer = witness(publisher)
    clock = [20.]
    monkeypatch.setattr("dronedream_agent_core.training.gazebo_witness.time.monotonic",
                        lambda: clock[0])
    observer.receive(model_entities(), clock[0], 1000, simulation_time_ns=1_000_000_000)
    clock[0] += wall_dt
    observer.receive(model_entities(x=.12), clock[0], 1000 + int(wall_dt * 1000),
                     simulation_time_ns=1_060_000_000)
    rows = receiver.read()
    assert len(rows) == 2 and observer.error is None
    assert rows[1].current_velocity_mps.x == pytest.approx(2.)
    assert rows[1].observed_at_unix_ms == 1000 + int(wall_dt * 1000)
    assert observer.close()["velocity_time_basis"] == "publisher-simulation-time"


# 功能：
#   验证重复仿真帧不刷新运动证据，仿真时钟倒退即使在节流区间也必须报错。
# 输入：
#   endpoints：真值通道。
#   monkeypatch：可步进的主机时钟。
# 输出：
#   None：不返回业务数据。
def test_repeated_simulation_frame_cannot_renew_motion_witness(endpoints, monkeypatch):
    receiver, publisher = endpoints
    observer = witness(publisher)
    clock = [20.]
    monkeypatch.setattr("dronedream_agent_core.training.gazebo_witness.time.monotonic",
                        lambda: clock[0])
    observer.receive(model_entities(), clock[0], 1000, simulation_time_ns=100)
    clock[0] += .03
    observer.receive(model_entities(x=5.), clock[0], 1030, simulation_time_ns=100)
    assert len(receiver.read()) == 1
    assert observer.skipped_repeated_scene == 1 and observer.error is None
    clock[0] += .001  # Regression must be found even within the cadence throttle.
    observer.receive(model_entities(), clock[0], 1031, simulation_time_ns=99)
    assert observer.error == "ValueError:OUTCOME_SIMULATION_CLOCK_REGRESSED"


# 功能：
#   验证缺失、错误类型或越界的仿真纳秒时间不能自动改用主机时间。
# 输入：
#   endpoints：采集器输出通道。
#   stamp：非法仿真纳秒时间。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("stamp", [None, True, -1, 1.5, 2**63, float("nan")])
def test_invalid_simulation_time_does_not_use_host_time(endpoints, stamp):
    receiver, publisher = endpoints
    observer = witness(publisher)
    observer.receive(model_entities(), time.monotonic(), 1000, simulation_time_ns=stamp)
    assert observer.error == "ValueError:OUTCOME_SIMULATION_CLOCK_INVALID"
    assert receiver.read() == []
