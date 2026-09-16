import asyncio
import hashlib
import json
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
from test_preflight_render_warmup import fake_native
from test_simulation_camera_profile import source_model

from dronedream_agent_core import preflight_render_warmup as warmup


# 功能：
#   为原生替身构造一次独立相机预热请求，不启动 Gazebo 或飞机。
# 输入：
#   tmp_path：本测试独占的来源及输出目录。
# 输出：
#   arguments：带实际来源摘要的预热调用参数。
def arguments_for(tmp_path):
    source = tmp_path / "camera.sdf"
    source.write_bytes(source_model())
    arguments = dict(gz_binary="gz", world="school", flight_vehicle="my_drone", source=source,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        position=(3., 4., 5.), output=tmp_path / "output")
    return arguments


# 功能：
#   拒绝同一父节点中重复定义位姿，不能任选第一个位姿复制到诊断相机。
# 输入：
#   parent_path：分别覆盖模型、连接体和传感器的位姿层级。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("parent_path", ["model", "model/link", "model/link/sensor"])
def test_duplicate_pose_is_ambiguous(parent_path):
    root = ET.fromstring(source_model())
    parent = root.find(parent_path)
    while len(parent.findall("pose")) < 2:
        ET.SubElement(parent, "pose").text = "0 0 0 0 0 0"
    source = ET.tostring(root)
    with pytest.raises(ValueError, match="RENDER_WARMUP_.*POSE"):
        warmup.prepare_rig(source, hashlib.sha256(source).hexdigest(), "a" * 32)


# 功能：
#   来源 SDF 交换传感器排列后，完成回执仍使用统一的 RGB、深度顺序。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_sensor_order_does_not_change_prepared_stream_contract():
    root = ET.fromstring(source_model())
    link = root.find("model/link")
    sensors = link.findall("sensor")
    for sensor in sensors:
        link.remove(sensor)
    link.extend(reversed(sensors))
    source = ET.tostring(root)
    _, receipt = warmup.prepare_rig(source, hashlib.sha256(source).hexdigest(),
                                    "b" * 32, include_depth=True)
    assert receipt["prepared_streams"] == ["rgb", "depth"]


# 功能：
#   即使 complete 被错误保留为 True，未排空、退订不足或存在错误也不能通过预热门禁。
# 输入：
#   field：故意改写的退订回执字段。
#   value：该字段的不合格值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(("field", "value"), [("subscribed_count", 0),
    ("subscribed_count", 3), ("unsubscribed_count", 1), ("active_callbacks_at_close", 1),
    ("errors", ["unsubscribe-failed"]), ("subscribed_count", True)])
def test_gate_requires_actual_subscription_drain(field, value):
    shutdown = dict(complete=True, subscribed_count=2, unsubscribed_count=2,
                    active_callbacks_at_close=0, errors=[])
    shutdown[field] = value
    receipt = dict(complete=True, source_camera_sha256="a" * 64,
        depth_preparation_requested=False, depth_pipeline_prepared=False,
        retained_until_simulation_exit=False, removed_and_absence_observed=True,
        prepared_streams=["rgb"], native_subscriptions=shutdown, flight_vehicle_absent=True,
        no_pixels_sent_to_policy=True, flight_qualification_granted=False)
    assert not warmup.warmup_receipt_ready(receipt, source_sha256="a" * 64, include_depth=False)


# 功能：
#   退订接口异常仍保存失败回执，不把清理异常直接抛出而丢失本次预热状态。
# 输入：
#   tmp_path：本测试独占的来源与回执目录。
#   monkeypatch：安装原生替身及故障清理方法的测试工具。
# 输出：
#   None：不返回业务数据。
def test_shutdown_exception_preserves_failed_receipt(tmp_path, monkeypatch):
    fake_native(monkeypatch, None)

    # 功能：
    #   模拟订阅所有者在关闭过程中出现错误。
    # 输入：
    #   self：被关闭的订阅对象。
    # 输出：
    #   None：不返回业务数据。
    def failed_close(self):
        raise OSError("test shutdown failure")

    monkeypatch.setattr(warmup.GazeboSubscriptions, "close", failed_close)
    with pytest.raises(RuntimeError, match="RENDER_WARMUP_FAILED"):
        warmup.execute_warmup(**arguments_for(tmp_path))
    receipt = json.loads((tmp_path / "output/receipt.json").read_text(encoding="utf-8"))
    assert receipt["complete"] is False
    assert receipt["native_subscriptions"]["complete"] is False
    assert "shutdown" in receipt["failure"]


# 功能：
#   最后一次进度读取后才出现的图像错误，也必须阻止生成成功回执。
# 输入：
#   tmp_path：本测试独占的来源与回执目录。
#   monkeypatch：安装原生替身及延迟错误的测试工具。
# 输出：
#   None：不返回业务数据。
def test_late_frame_error_prevents_success(tmp_path, monkeypatch):
    fake_native(monkeypatch, None)
    original = warmup.WarmupFrames.require_progress
    calls = []

    # 功能：
    #   在第 24 个视角完成时标记异步流错误，再返回本次先前已读取的计数。
    # 输入：
    #   self：本次预热帧缓存。
    #   after：位姿确认后的接收水位。
    #   timeout：剩余等待预算。
    # 输出：
    #   counts：故障发生前已得到的各流计数。
    def progress_then_failure(self, after, timeout):
        counts = original(self, after, timeout)
        calls.append(after)
        if len(calls) == 24:
            self.error = "RENDER_WARMUP_STREAM_CLOCK_NOT_ADVANCING"
        return counts

    monkeypatch.setattr(warmup.WarmupFrames, "require_progress", progress_then_failure)
    with pytest.raises(RuntimeError, match="RENDER_WARMUP_FAILED"):
        warmup.execute_warmup(**arguments_for(tmp_path))
    receipt = json.loads((tmp_path / "output/receipt.json").read_text(encoding="utf-8"))
    assert receipt["complete"] is False


# 功能：
#   非法的自有实体位姿不能从原生回调抛出异常，或进入含 NaN 的诊断回执。
# 输入：
#   bad：错误的坐标或缺失字段的自有位姿。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", [SimpleNamespace(name="owned", id=12),
    SimpleNamespace(name="owned", id=12, position=SimpleNamespace(x=float("nan"), y=0, z=0),
        orientation=SimpleNamespace(w=1, x=0, y=0, z=0))])
def test_bad_owned_pose_latches_a_controlled_failure(bad):
    receiver = warmup.WarmupPose("owned")
    receiver.entity_id = 12
    receiver.observe(SimpleNamespace(pose=[bad]))
    with pytest.raises(ValueError, match="RENDER_WARMUP_LIVE_POSE_INVALID"):
        receiver.require_match((0, 0, 0), (1, 0, 0, 0), -1, timeout=.001)
    assert receiver.latest is None
    assert receiver.named_pose is None


# 功能：
#   执行中断仍删除自有 RGB 实体和退订，保存失败回执后重新抛出原中断对象。
# 输入：
#   tmp_path：本测试独占的来源与回执目录。
#   monkeypatch：安装原生替身及中断方法的测试工具。
#   error_type：键盘停止或异步取消的中断类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, asyncio.CancelledError])
def test_interruption_preserved_after_cleanup(tmp_path, monkeypatch, error_type):
    requests, entities, unsubscribed, _ = fake_native(monkeypatch, None)
    interruption = error_type("test interrupted")

    # 功能：
    #   在真实生命周期已创建自有实体后模拟外部中断。
    # 输入：
    #   self：本次帧缓存。
    #   after：本次位姿回读后的接收水位。
    #   timeout：剩余帧等待预算。
    # 输出：
    #   None：不返回业务数据。
    def interrupt_progress(self, after, timeout):
        raise interruption

    monkeypatch.setattr(warmup.WarmupFrames, "require_progress", interrupt_progress)
    with pytest.raises(error_type) as caught:
        warmup.execute_warmup(**arguments_for(tmp_path))
    receipt = json.loads((tmp_path / "output/receipt.json").read_text(encoding="utf-8"))
    assert caught.value is interruption
    assert not entities and len(unsubscribed) == 2
    assert any(service == "remove" for service, _ in requests)
    assert receipt["complete"] is False and receipt["native_subscriptions"]["complete"] is True


# 功能：
#   最后一个视角完成后出现的位姿回调错误不能遗漏到成功判定之外。
# 输入：
#   tmp_path：本测试独占的来源与回执目录。
#   monkeypatch：安装原生替身及延迟错误的测试工具。
# 输出：
#   None：不返回业务数据。
def test_late_pose_error_prevents_success(tmp_path, monkeypatch):
    fake_native(monkeypatch, None)
    original = warmup.WarmupPose.require_match
    calls = []

    # 功能：
    #   在最后一个位姿返回前锁存异步错误，模拟返回结果与退出之间的竞态。
    # 输入：
    #   self：本次位姿接收器。
    #   xyz：要求确认的位置。
    #   quaternion：要求确认的朝向。
    #   after：设置回执后的接收水位。
    #   timeout：剩余位姿等待预算。
    # 输出：
    #   observed：此前已经匹配的测试位姿。
    def match_then_failure(self, xyz, quaternion, after, timeout):
        observed = original(self, xyz, quaternion, after, timeout)
        calls.append(after)
        if len(calls) == 24:
            self.error = "RENDER_WARMUP_LIVE_POSE_INVALID"
        return observed

    monkeypatch.setattr(warmup.WarmupPose, "require_match", match_then_failure)
    with pytest.raises(RuntimeError, match="RENDER_WARMUP_FAILED"):
        warmup.execute_warmup(**arguments_for(tmp_path))
    receipt = json.loads((tmp_path / "output/receipt.json").read_text(encoding="utf-8"))
    assert receipt["complete"] is False
    assert "LIVE_POSE_INVALID" in receipt["failure"]


# 功能：
#   已锁存的位姿错误不被后续正常消息抹去，等待入口不会返回旧成功缓存。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_valid_pose_cannot_clear_latched_error():
    receiver = warmup.WarmupPose("owned")
    receiver.entity_id = 12
    receiver.observe(SimpleNamespace(pose=[SimpleNamespace(name="owned", id=12)]))
    receiver.observe(SimpleNamespace(pose=[SimpleNamespace(name="owned", id=12,
        position=SimpleNamespace(x=0, y=0, z=0),
        orientation=SimpleNamespace(w=1, x=0, y=0, z=0))]))
    with pytest.raises(ValueError, match="LIVE_POSE_INVALID"):
        receiver.require_match((0, 0, 0), (1, 0, 0, 0), -1)
    assert receiver.latest is None


# 功能：
#   同名诊断实体的原生 ID 改变后，不能继续移动或删除可能已被替换的实体。
# 输入：
#   tmp_path：本测试独占的来源与回执目录。
#   monkeypatch：替换原生服务应答的测试工具。
# 输出：
#   None：不返回业务数据。
def test_replaced_entity_is_not_moved_or_removed(tmp_path, monkeypatch):
    requests, entities, _, _ = fake_native(monkeypatch, None)
    original = warmup.subprocess.run

    # 功能：
    #   第一次位姿命令后更换实体 ID，模拟同名实体被其他操作者重新创建。
    # 输入：
    #   command：生产代码实际发出的原生命令。
    #   kwargs：超时及输出参数。
    # 输出：
    #   result：替换发生前原服务的执行应答。
    def replace_after_move(command, **kwargs):
        result = original(command, **kwargs)
        if requests[-1][0] == "set_pose":
            next(iter(entities.values())).id = 99
        return result

    monkeypatch.setattr(warmup.subprocess, "run", replace_after_move)
    with pytest.raises(RuntimeError, match="RENDER_WARMUP_FAILED"):
        warmup.execute_warmup(**arguments_for(tmp_path))
    assert sum(service == "set_pose" for service, _ in requests) == 1
    assert not any(service == "remove" for service, _ in requests)
    assert len(entities) == 1
