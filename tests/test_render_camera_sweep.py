import json
import math
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
from test_geometry_fixture_transport import FakePipe, FakeProcess
from test_static_render_batching import world
from test_static_render_equivalence import script

import dronedream_agent_core.render_camera_sweep as sweep
from dronedream_agent_core.render_camera_sweep import prepare_camera_sweep, swept_pose


# 功能：
#   用现有静态渲染构造器添加一个已知姿态的诊断相机，不构造飞机。
# 输入：
#   无。
# 输出：
#   content：含诊断相机的世界 SDF 字节。
def camera_world():
    content = script().add_camera_rigs(world(), [[1., 2., 3., 0., 0., .3]])
    return content


# 功能：
#   确认添加扫描命令服务不会改动地图模型、相机定义或原始姿态。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_sweep_adds_service_but_changes_no_geometry_or_sensor_config():
    before = ET.fromstring(camera_world())
    result, name, cameras = prepare_camera_sweep(camera_world(), 1)
    after = ET.fromstring(result)
    assert cameras == [("render_probe_0", (1., 2., 3., 0., 0., .3))]
    assert name == before.find("world").get("name")
    assert [ET.tostring(m) for m in before.findall("world/model")] == [
        ET.tostring(m) for m in after.findall("world/model")]
    assert after.find("world/plugin[@name='gz::sim::systems::UserCommands']") is not None


# 功能：
#   检查不能将扫描请求指向飞机、未知实体或具有执行内容的伪诊断相机。
# 输入：
#   fault：身份、配置或姿态的破坏类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["missing", "duplicate", "dynamic", "controller",
    "collision", "include", "wrong-topic", "relative-pose", "nonfinite"])
def test_sweep_cannot_move_an_aircraft_or_unknown_camera(fault):
    root = ET.fromstring(camera_world())
    scene = root.find("world")
    model = scene.find("model[@name='render_probe_0']")
    if fault == "missing":
        model.set("name", "aircraft")
    elif fault == "duplicate":
        scene.append(ET.fromstring(ET.tostring(model)))
    elif fault == "dynamic":
        model.find("static").text = "false"
    elif fault in {"controller", "collision", "include"}:
        ET.SubElement(model, "plugin" if fault == "controller" else fault)
    elif fault == "wrong-topic":
        model.find("link/sensor/topic").text = "/aircraft/camera"
    elif fault == "relative-pose":
        model.find("pose").set("relative_to", "aircraft")
    else:
        model.find("pose").text = "1 2 3 0 0 nan"
    with pytest.raises(ValueError, match="RENDER_SWEEP_"):
        prepare_camera_sweep(ET.tostring(root), 1)


# 功能：
#   验证偏航扫描保持原始位置，复合旋转后四元数仍具有单位模长。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_sweep_preserves_position_and_unit_orientation():
    position, q = swept_pose((1., 2., 3., 0., 0., 0.), math.pi / 2)
    assert position == (1., 2., 3.)
    assert q == pytest.approx((math.sqrt(.5), 0., 0., math.sqrt(.5)))
    _, compound = swept_pose((1., 2., 3., .3, -.4, .8), 5.)
    assert sum(value * value for value in compound) == pytest.approx(1.)


# 功能：
#   为扫描客户端提供受控管道和子进程，兼容对修复前对象管道的失败复现。
# 输入：
#   monkeypatch：pytest 替换工具。
#   replies：服务结果字节序列。
#   polls：受控就绪序列。
#   process：可选假进程。
# 输出：
#   state：父管道、子管道及子进程。
def install_sweep_worker(monkeypatch, replies=(), polls=(), process=None):
    parent, child = FakePipe(replies, polls), FakePipe()
    process = process or FakeProcess()

    # 功能：
    #   模拟旧版对象管道发送，仅用于验证旧实现的回执错配问题。
    # 输入：
    #   value：旧客户端发送对象。
    # 输出：
    #   None：不返回业务数据。
    def send(value):
        parent.send_bytes(json.dumps(value).encode())

    # 功能：
    #   模拟旧版对象管道接收，不替生产代码执行回执合法性校验。
    # 输入：
    #   无。
    # 输出：
    #   value：收到的对象。
    def recv():
        value = json.loads(parent.recv_bytes(2048))
        return value

    # 功能：
    #   返回诊断扫描的双工管道对。
    # 输入：
    #   duplex：双工标记。
    # 输出：
    #   pair：父端与子端。
    def pipe(duplex=True):
        assert duplex is True
        pair = parent, child
        return pair

    # 功能：
    #   确认子进程只使用扫描工作函数，不启动仿真。
    # 输入：
    #   options：构造参数。
    # 输出：
    #   process：受控子进程。
    def factory(**options):
        assert options["target"] is sweep._request_worker
        assert options["args"][0] is child
        return process

    # 功能：
    #   提供 spawn 上下文，避免继承其他进程状态。
    # 输入：
    #   method：启动模式。
    # 输出：
    #   context：替换的管道与进程工厂。
    def context_factory(method):
        assert method == "spawn"
        context = SimpleNamespace(Pipe=pipe, Process=factory)
        return context

    parent.send, parent.recv = send, recv
    monkeypatch.setattr(sweep, "multiprocessing", SimpleNamespace(get_context=context_factory))
    state = parent, child, process
    return state


# 功能：
#   复现扫描请求超时后的迟到回执，要求关闭通道且下一请求不得消费旧结果。
# 输入：
#   monkeypatch：pytest 替换工具。
# 输出：
#   None：不返回业务数据。
def test_sweep_timeout_cannot_reuse_late_reply(monkeypatch):
    parent, _, _ = install_sweep_worker(monkeypatch, replies=(
        b'{"transport_accepted":true,"reply":true,"request_elapsed_ms":2000}',),
        polls=(False, True))
    client = sweep.CameraSweepClient(camera_world(), 1)
    with pytest.raises(TimeoutError):
        client.request(.1)
    count = len(parent.sent)
    with pytest.raises(RuntimeError):
        client.request(.2)
    assert parent.received == 0 and len(parent.sent) == count and parent.closed


# 功能：
#   拒绝非布尔接受标记、重复键及异常耗时，非法应答使通道失效。
# 输入：
#   monkeypatch：pytest 替换工具。
#   reply：待验证回执。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("reply", [b'[]',
    b'{"transport_accepted":1,"reply":true,"request_elapsed_ms":1}',
    b'{"transport_accepted":true,"reply":null,"request_elapsed_ms":1}',
    b'{"transport_accepted":true,"reply":true,"request_elapsed_ms":NaN}',
    b'{"transport_accepted":true,"reply":true,"request_elapsed_ms":-1}',
    b'{"transport_accepted":true,"reply":true,"request_elapsed_ms":true}',
    b'{"transport_accepted":true,"reply":false,"reply":true,"request_elapsed_ms":1}'])
def test_sweep_reply_requires_strict_bounded_evidence(monkeypatch, reply):
    parent, _, _ = install_sweep_worker(monkeypatch, replies=(reply,))
    client = sweep.CameraSweepClient(camera_world(), 1)
    with pytest.raises(RuntimeError):
        client.request(.1)
    assert parent.closed


# 功能：
#   合法的运输失败及有效服务拒绝可以继续下一请求，清理结果不共享可变字典。
# 输入：
#   monkeypatch：pytest 替换工具。
# 输出：
#   None：不返回业务数据。
def test_sweep_valid_negative_replies_and_owned_summary(monkeypatch):
    parent, _, _ = install_sweep_worker(monkeypatch, replies=(
        b'{"transport_accepted":false,"reply":null,"request_elapsed_ms":1}',
        b'{"transport_accepted":true,"reply":false,"request_elapsed_ms":2}'))
    client = sweep.CameraSweepClient(camera_world(), 1)
    assert client.request(.1)["transport_accepted"] is False
    assert client.request(.2)["reply"] is False
    summary = client.close()
    assert summary["complete"] is True
    summary["complete"] = False
    assert client.close()["complete"] is True and parent.closed
