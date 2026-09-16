"""使用真实 ROS 节点逻辑和隔离消息类型验证真实性、时间、身份及中止发布。"""

import importlib
import json
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
from test_domain_action_logic import _logic_module, _targets


# 功能：
#   构建测试消息容器，ROS 传输层以外的节点读取、校验与发布分支仍执行产品代码。
# 输入：
#   无。
# 输出：
#   message：可接收观测字段的隔离消息。
def _observation_message():
    message = SimpleNamespace(
        header=SimpleNamespace(),
        simulator_time=SimpleNamespace(),
        pose_enu=SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace()),
    )
    return message


# 功能：
#   仅替换本机缺失的 ROS/Gazebo 导入，不启动模拟器，也不伪造产品判断结果。
# 输入：
#   monkeypatch：自动恢复的模块与路径替换工具。
# 输出：
#   modules：真实观测节点、动作服务、安全节点及 I/O 模块。
@pytest.fixture
def ros_modules(monkeypatch):
    _logic_module()
    for name in (
        "rclpy",
        "rclpy.executors",
        "rclpy.node",
        "rclpy.qos",
        "dronedream_agent_msgs",
        "dronedream_agent_msgs.msg",
        "dronedream_agent_msgs.srv",
        "gz",
        "gz.msgs10",
        "gz.msgs10.pose_v_pb2",
        "gz.transport13",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["rclpy.executors"].ExternalShutdownException = type(
        "ExternalShutdownException", (Exception,), {}
    )
    sys.modules["rclpy.node"].Node = object
    sys.modules["rclpy.qos"].qos_profile_sensor_data = object()
    sys.modules["rclpy.qos"].QoSProfile = object
    sys.modules["rclpy.qos"].ReliabilityPolicy = SimpleNamespace(RELIABLE=1)
    sys.modules["gz.msgs10.pose_v_pb2"].Pose_V = object
    sys.modules["gz.transport13"].Node = object
    sys.modules["dronedream_agent_msgs.msg"].MissionObservation = _observation_message
    sys.modules["dronedream_agent_msgs.msg"].SafetyEvent = object
    sys.modules["dronedream_agent_msgs.srv"].ExecuteDomainAction = object
    names = ("gazebo_pose_observer", "domain_action_server", "safety_event_guard")
    loaded = []
    for name in names:
        qualified = f"dronedream_agent_ros.{name}"
        # 导入本身会注册模块；用 monkeypatch 记录原值，避免假 ROS 绑定泄漏到后续测试。
        monkeypatch.delitem(sys.modules, qualified, raising=False)
        loaded.append(importlib.import_module(qualified))
    io = importlib.import_module("dronedream_agent_ros.runtime_io")
    yield (*loaded, io)
    for name in names:
        sys.modules.pop(f"dronedream_agent_ros.{name}", None)


# 功能：
#   创建不依赖 ROS 安装的真实观测节点实例，保留其传感器接收及发布方法。
# 输入：
#   module：产品观测模块。
# 输出：
#   fixture：节点和实际发布收集列表。
def _observer(module):
    node = object.__new__(module.GazeboPoseObserver)
    node._lock = threading.Lock()
    node._latest = None
    node._sequence = node._published_sequence = 0
    node._last_source_stamp = None
    node._maximum_pose_age_seconds = 0.25
    node._entity_name, node._gazebo_topic = "drone-current", "/world/current/pose"
    node._contract_id, node._segment_id = "current-mission", "segment-001"
    node._runtime_phase_path = None
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: SimpleNamespace(sec=900))
    )
    published = []
    node._publisher = SimpleNamespace(publish=published.append)
    fixture = node, published
    return fixture


# 功能：
#   提供带真实来源时间字段的原始姿态消息，便于区分采样时间与发布时间。
# 输入：
#   sec：仿真采样秒数。
#   x：世界东向坐标。
# 输出：
#   raw：Gazebo 消息结构的测试值。
def _raw_pose(sec=10, x=1.0):
    raw = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nsec=123)),
        pose=[
            SimpleNamespace(
                name="drone-current",
                position=SimpleNamespace(x=x, y=2.0, z=3.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=2.0),
            )
        ],
    )
    return raw


# 功能：
#   验证定时器重复触发不能把同一缓存姿态伪装成新观测，并保留原仿真时间。
# 输入：
#   ros_modules：真实节点及隔离传输模块。
# 输出：
#   None：单次新采样最多发布一条观测。
def test_pose_timer_does_not_refresh_old_measurements(ros_modules):
    module = ros_modules[0]
    node, published = _observer(module)
    node._on_gazebo_pose(_raw_pose())
    node._publish_latest()
    node._publish_latest()
    assert len(published) == 1
    assert published[0].sequence == 1
    assert published[0].simulator_time.sec == 10
    assert published[0].simulator_time.nanosec == 123
    assert published[0].pose_enu.orientation.w == 1.0
    node._on_gazebo_pose(_raw_pose(sec=11))
    node._publish_latest()
    assert len(published) == 2 and published[1].sequence == 2


# 功能：
#   验证过期、重复时间、倒退时间及非法姿态不能刷新有效观测。
# 输入：
#   ros_modules：真实桥接代码。
#   monkeypatch：可控的本机单调时间。
#   defect：观测失效方式。
# 输出：
#   None：被破坏的输入不产生第二条观测。
@pytest.mark.parametrize(
    "defect", ["stale", "duplicate", "backward", "nan", "zero-quaternion", "ambiguous"]
)
def test_pose_rejects_invalid_or_expired_samples(ros_modules, monkeypatch, defect):
    module = ros_modules[0]
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    node, published = _observer(module)
    node._on_gazebo_pose(_raw_pose())
    if defect == "stale":
        clock[0] += 0.3
        node._publish_latest()
        assert not published
        return
    node._publish_latest()
    raw = _raw_pose(sec=11)
    if defect == "duplicate":
        raw.header.stamp.sec = 10
    elif defect == "backward":
        raw.header.stamp.sec = 9
    elif defect == "nan":
        raw.pose[0].position.x = float("nan")
    elif defect == "zero-quaternion":
        raw.pose[0].orientation.w = 0.0
    else:
        raw.pose.append(raw.pose[0])
    node._on_gazebo_pose(raw)
    node._publish_latest()
    assert len(published) == 1


# 功能：
#   阶段文件损坏不能降级为起飞前状态，防止放宽飞行期间的超时限值。
# 输入：
#   ros_modules：真实桥接模块。
#   tmp_path：阶段记录目录。
#   raw：损坏或未知阶段记录。
# 输出：
#   None：返回显式未知阶段，而不是 PREFLIGHT。
@pytest.mark.parametrize("raw", ["not-json", "[]", '{"phase":"TRACK","phase":"PREFLIGHT"}', "{}"])
def test_invalid_phase_is_not_reported_as_preflight(ros_modules, tmp_path, raw):
    node, _ = _observer(ros_modules[0])
    node._runtime_phase_path = tmp_path / "phase.json"
    node._runtime_phase_path.write_text(raw)
    assert node._runtime_phase().startswith("UNRECOGNIZED:")


# 功能：
#   重复、乱序或其他任务的 ROS 观测不能重置动作服务的观测年龄。
# 输入：
#   ros_modules：真实动作服务模块。
# 输出：
#   None：服务只保留首次有效的新序号。
def test_action_observation_does_not_accept_replays(ros_modules):
    node = object.__new__(ros_modules[1].SimulationDomainActionServer)
    node._contract_id = "current"
    node._latest_observation = node._latest_observation_monotonic = None
    first = SimpleNamespace(contract_id="current", sequence=2, localization_ok=True, link_ok=True)
    node._on_observation(first)
    original_time = node._latest_observation_monotonic
    for contract, sequence in (("current", 2), ("current", 1), ("other", 3)):
        node._on_observation(
            SimpleNamespace(
                contract_id=contract, sequence=sequence, localization_ok=True, link_ok=True
            )
        )
    assert node._latest_observation is first
    assert node._latest_observation_monotonic == original_time


# 功能：
#   两个安全发布者不能互相覆盖中止原因，暂存文件在成功或目标存在时均被回收。
# 输入：
#   ros_modules：实际原子文件发布实现。
#   tmp_path：隔离目录。
# 输出：
#   None：第一条中止保留，第二次发布返回假。
def test_abort_publication_preserves_first_reason(ros_modules, tmp_path):
    io = ros_modules[3]
    path = tmp_path / "abort.json"
    assert io.publish_abort(path, {"reason": "operator"}) is True
    assert io.publish_abort(path, {"reason": "native safety"}) is False
    assert json.loads(path.read_text()) == {"reason": "operator"}
    assert not list(tmp_path.glob("*.tmp"))


# 功能：
#   检查纯动作判断对错误时间、阈值及 JSON 参数保持失败且输出标准 JSON。
# 输入：
#   tmp_path：有效目标夹具目录。
#   change：本例注入的无效输入。
# 输出：
#   None：任何无效输入均不能生成成功证据。
@pytest.mark.parametrize(
    "change",
    [
        {"observation_age_seconds": -1.0},
        {"maximum_target_distance_m": float("nan")},
        {"maximum_observation_age_seconds": float("inf")},
        {"observation_sequence": True},
        {"arguments_json": '{"x": 1, "x": 2}'},
        {"arguments_json": '{"x": NaN}'},
    ],
)
def test_action_invalid_input_cannot_issue_success(tmp_path, change):
    logic, targets = _targets(tmp_path)
    arguments = dict(
        expected_contract_id="mission",
        expected_action="native.payload.scan-code",
        request_contract_id="mission",
        task_id="scan",
        request_action="native.payload.scan-code",
        target_node="verified-001",
        arguments_json="{}",
        target_points=targets,
        observation_contract_id="mission",
        observation_sequence=1,
        observation_position_enu_m=(4.0, 5.0, 2.0),
        observation_age_seconds=0.1,
        maximum_observation_age_seconds=2.0,
        maximum_target_distance_m=0.75,
    )
    arguments.update(change)
    result = logic.evaluate_simulation_action(**arguments)
    assert result["success"] is False and result["evidence"] == []
    assert result["issue_code"] != "DOMAIN_ACTION_SENSOR_VERIFIER_UNAVAILABLE"
    assert not json.loads(result["details_json"])["proximity_verified"]
