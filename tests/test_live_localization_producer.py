"""Independent source production retains clocks and cannot bypass sensor validation."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_depth_sensor_binding import frame

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.live_localization_producer import LiveLocalizationProducer
from dronedream_agent_core.localization_source_channel import decode_localization_source
from dronedream_agent_core.runtime_sensor_contracts import oakd_lite_depth_sensor_contract
from dronedream_agent_core.sensor_frame_clock import SensorFrameTime


# 功能：创建真实投影器与可观测的原生状态/发布端夹具；输入：无；输出：发布器及测试端口。
def fixture():
    native, publisher = Mock(), Mock()
    stamp = time.time_ns()
    mono = time.monotonic()
    clock = SensorFrameTime(stamp, stamp, mono, mono, "host-receipt", simulation_ns=1_000_000_000)
    image = frame()
    items = [(image, clock)]
    aligned = SimpleNamespace(
        pose=SimpleNamespace(position_world_enu_m=Vector3(x=1, y=2, z=3),
            orientation_world_from_body={"w": 1., "x": 0., "y": 0., "z": 0.},
            velocity_world_enu_mps=Vector3(x=0, y=0, z=0), binding_sha256="b" * 64),
        conservative_variance_m2=.01, native_odometry_snapshot={"source_alignment": {}})
    native.select_source_image_pose.return_value = (stamp // 1_000_000, clock.simulation_ns, aligned)
    worker = LiveLocalizationProducer(snapshot=lambda: items, native_buffer=native,
        publisher=publisher, mount=oakd_lite_depth_sensor_contract(), vehicle_id="quad",
        map_sha256="a" * 64, clock_domain="test-clock", maximum_acceleration_mps2=2.,
        maximum_alignment_variance_m2=.25)
    return worker, native, publisher, items


# 功能：验证真实编码传输保留原始时间和无控制权限，不重复发布同帧；输入：无；输出：字段断言。
def test_exact_source_once_without_route_or_truth():
    worker, native, publisher, items = fixture()
    worker._step()
    record = decode_localization_source(publisher.send.call_args.args[0])
    assert record["scan"]["observed_at_unix_ms"] == items[0][1].source_unix_ns // 1_000_000
    assert record["scan"]["body_position_world_enu_m"] == {"x": 1., "y": 2., "z": 3.}
    assert not record["truth_correction_applied"] and not record["motion_permission_granted"]
    assert record["map_sha256"] == "a" * 64
    assert native.select_source_image_pose.call_args.args[0][0][1] == 1_000_000_000
    native.align_source_image.assert_not_called()
    worker._step()
    publisher.send.assert_called_once()
    assert worker.close()["projected_frames"] == 1


# 功能：投影缓存精确匹配原帧且返回独立副本；输入：匹配/不匹配帧；输出：无跨线程可变共享。
def test_cached_projection_is_exact_and_owned():
    worker, _, _, items = fixture()
    worker._step()
    stamp = items[0][1].source_unix_ns
    assert worker.prepared(stamp + 1) is None
    projection, contract = worker.prepared(stamp)
    projection.samples[0].range_m = 999.
    contract.vehicle_id = "other"
    again, contract_again = worker.prepared(stamp)
    assert again.samples[0].range_m < 999.
    assert contract_again.vehicle_id == "quad"
    worker.close()


# 功能：结构错误不可伪装成等待；输入：真实缓冲层传播的错误；输出：不得发布观测。
def test_selection_identity_failure():
    worker, native, publisher, _ = fixture()
    native.select_source_image_pose.side_effect = ValueError("IMAGE_SOURCE_MAP_BINDING_CHANGED")
    with pytest.raises(ValueError, match="IMAGE_SOURCE_MAP_BINDING_CHANGED"):
        worker._step()
    publisher.send.assert_not_called()
    worker.close()


# 功能：拒绝未配对、超方差、超容量和坏像素；输入：缺失类型；输出：不能发送伪造有效观测。
@pytest.mark.parametrize("kind", ["no-pair", "variance", "capacity", "pixels"])
def test_invalid_or_unavailable_source(kind):
    worker, native, publisher, items = fixture()
    if kind == "no-pair":
        native.select_source_image_pose.return_value = None
    elif kind == "variance":
        native.select_source_image_pose.return_value[2].conservative_variance_m2 = 1.
    elif kind == "capacity":
        items.extend(items * 16)
    else:
        items[0][0].data = b""
    if kind in {"capacity", "pixels"}:
        with pytest.raises(ValueError):
            worker._step()
    else:
        worker._step()
    publisher.send.assert_not_called()
    worker.close()


# 功能：真实后台线程在主线程未处理避障时仍发布，异常传播且关闭后不再访问端口。
# 输入：事件同步的发布端，不使用固定等待猜测线程完成。
# 输出：独立进度、错误传播及资源所有权断言。
def test_independent_thread_and_failure_lifecycle():
    worker, _, publisher, _ = fixture()
    published = threading.Event()
    publisher.send.side_effect = lambda payload: published.set() or True
    worker.start()
    assert published.wait(2)
    assert worker.close()["sent_frames"] == 1
    publisher.close.assert_not_called()
    with pytest.raises(RuntimeError, match="CLOSED"):
        worker.start()
    worker, _, publisher, _ = fixture()
    failed = threading.Event()

    # 功能：注入明确通道异常；输入：已编码帧；输出：异常，不能误报已发送。
    def fail(payload):
        failed.set()
        raise ValueError("CHANNEL_BROKEN")

    publisher.send.side_effect = fail
    worker.start()
    assert failed.wait(2)
    summary = worker.close()
    assert summary["issue"] == "CHANNEL_BROKEN"
    with pytest.raises(RuntimeError, match="PRODUCER_FAILED") as caught:
        worker.raise_if_failed()
    assert str(caught.value.__cause__) == "CHANNEL_BROKEN"
