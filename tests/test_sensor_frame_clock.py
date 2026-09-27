from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from dronedream_agent_core.sensor_frame_clock import (
    SensorFrameClock,
    SensorImageIngress,
    require_model_frame_time,
)

EPOCH = "a" * 64


# 功能：原生预更新时钟必须显式选择身份，并保留实际原始帧龄穿过模型边界。
# 输入：临时图像目录。输出：验证后的元数据仍在原 250 毫秒截止内，而不是接收后续期。
def test_native_preupdate_clock_is_pinned_and_reaches_model_deadline(tmp_path):
    from dronedream_agent_core.model_image_cache import PreparedModelImage
    from dronedream_agent_core.observation_validity import image_control_deadline
    message = frame(basis="native-preupdate")
    value = admit(SensorFrameClock(expected_native_epoch=EPOCH), message)
    assert value.clock_kind == "native-simulation"
    assert value.age_at_receipt_ns == 100_000_000
    require_model_frame_time(message, value, value.sample_monotonic_seconds, 1000)
    prepared = PreparedModelImage(value.sample_monotonic_seconds, 1000, (1, 1),
                                  b"png", b"rgb", "c"*64, "d"*64, value)
    payload = prepared.multimodal(tmp_path)
    assert payload["timestamp_basis"] == "native-simulation-preupdate"
    assert image_control_deadline(payload, now_unix_ms=1100) == 1250
    assert image_control_deadline(payload, now_unix_ms=1250) == 0
    for clock in (SensorFrameClock(), SensorFrameClock(expected_scene_epoch=EPOCH)):
        with pytest.raises(ValueError):
            admit(clock, message)
    with pytest.raises(ValueError):
        admit(SensorFrameClock(expected_native_epoch=EPOCH), frame())
    with pytest.raises(ValueError):
        SensorFrameClock(expected_native_epoch=EPOCH, expected_scene_epoch=EPOCH)


# 功能：验证原生预更新时钟不可被删掉依据、改身份或以后来的到达重新计时。
# 输入：元数据破坏方式。输出：拒绝数据，不授予图像控制权限。
@pytest.mark.parametrize("changes", [{"basis":"host-receipt"}, {"epoch":"c"*64},
                                    {"source_unix_ns":"1"}, {"basis":"scene-snapshot"}])
def test_native_clock_rejects_wrong_source_and_expiry(changes):
    fields = {"basis":"native-preupdate", **changes}
    with pytest.raises(ValueError):
        admit(SensorFrameClock(expected_native_epoch=EPOCH), frame(**fields))


# 功能：
#   生成保留原始采集时间和场景摘要的图像报头，允许单独覆盖字段构造非法输入。
# 输入：
#   overrides：待覆盖或追加的场景报头字段。
# 输出：
#   message：带仿真 stamp 和场景元数据的测试消息。
def frame(**overrides):
    fields = dict(epoch=EPOCH, sha256="b" * 64, sequence="1",
                  source_unix_ns="1000000000", simulation_ns="500000000")
    fields.update(overrides)
    message = NS(header=NS(stamp=NS(sec=0, nsec=500_000_000), data=[
        NS(key="dronedream_scene_" + key, value=[value]) for key, value in fields.items()]))
    return message


# 功能：
#   用显式受控的接收时钟调用真实图像准入逻辑，不依赖测试运行速度。
# 输入：
#   clock：待测试的单相机时钟。
#   message：待接入的图像消息。
#   now：测试指定的接收 UNIX 纳秒。
#   mono：测试指定的接收单调钟秒数。
# 输出：
#   value：准入函数返回的来源时钟记录。
def admit(clock, message, now=1_100_000_000, mono=10.):
    value = clock.admit(message, received_unix_ns=now, received_monotonic_seconds=mono)
    return value


# 功能：
#   验证转发图像保留原始一百毫秒年龄，并正确映射到接收侧的单调时钟。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_original_exposure_age_survives_relay_and_receipt():
    value = admit(SensorFrameClock(expected_scene_epoch=EPOCH), frame())
    assert value.sample_monotonic_seconds == 9.9
    assert value.received_monotonic_seconds == 10
    assert value.age_at_receipt_ns == 100_000_000
    assert value.source_unix_ns == 1_000_000_000
    assert value.clock_kind == "scene-source"


# 功能：
#   验证直连流明确使用较弱的接收钟，不能靠传入场景报头擅自切换信任身份。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_direct_stream_is_explicitly_receipt_only_and_rejects_unselected_scene():
    clock = SensorFrameClock()
    value = admit(clock, NS(header=NS(data=[])))
    assert value.clock_kind == "host-receipt"
    assert value.source_unix_ns == value.received_unix_ns
    with pytest.raises(ValueError, match="NOT_CONFIGURED"):
        admit(clock, frame())


# 功能：
#   验证直连发布者时间戳重放、丢失及接收钟回退都会拒绝，且拒绝不推进顺序基线。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_direct_gazebo_stamp_replay_cannot_be_renewed_by_later_arrival():
    clock = SensorFrameClock()
    message = NS(header=NS(data=[], stamp=NS(sec=1, nsec=0)))
    admit(clock, message)
    with pytest.raises(ValueError, match="DIRECT_FRAME_REPLAY"):
        admit(clock, message, now=1_200_000_000, mono=10.1)
    with pytest.raises(ValueError, match="STAMP_MISSING"):
        admit(clock, NS(header=NS(data=[])), now=1_200_000_000, mono=10.1)
    message.header.stamp.nsec = 100_000_000
    with pytest.raises(ValueError, match="RECEIPT_REGRESSED"):
        admit(clock, message, now=1_000_000_000, mono=10.1)
    assert admit(clock, message, now=1_200_000_000, mono=10.1).simulation_ns == 1_100_000_000


# 功能：
#   验证超过接入时效上限一纳秒或来自未来的场景图像均被拒绝。
# 输入：
#   age：接收时图像的原始年龄，单位纳秒。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("age", [250_000_001, -1])
def test_receive_side_rejects_expired_or_future(age):
    with pytest.raises(ValueError, match="EXPIRED_OR_FUTURE"):
        admit(SensorFrameClock(expected_scene_epoch=EPOCH), frame(), now=1_000_000_000 + age)


# 功能：
#   验证接入时效上限是包含边界的二百五十毫秒，不因时间换算被改变。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_exact_original_deadline_is_not_extended():
    clock = SensorFrameClock(expected_scene_epoch=EPOCH)
    assert admit(clock, frame(), now=1_250_000_000).age_at_receipt_ns == 250_000_000


# 功能：
#   验证错误身份、非法整数、未知字段和矛盾仿真时间不能构成新鲜场景帧。
# 输入：
#   changes：对默认场景报头的非法字段覆盖。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("changes", [
    {"epoch": "c" * 64}, {"sha256": "not-a-hash"}, {"sequence": "0"},
    {"sequence": "1e3"}, {"sequence": "-1"}, {"sequence": "9223372036854775808"},
    {"simulation_ns": "1"}, {"unexpected": "value"}, {"source_unix_ns": "0"},
])
def test_malformed_metadata_is_not_a_fresh_frame(changes):
    with pytest.raises(ValueError):
        admit(SensorFrameClock(expected_scene_epoch=EPOCH), frame(**changes))


# 功能：
#   验证缺字段、重复字段、多值及超长场景元数据均在准入阶段拒绝。
# 输入：
#   mutation：本次对报头实施的结构破坏。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "multivalue", "oversized"])
def test_partial_ambiguous_and_oversized_headers(mutation):
    message = frame()
    rows = message.header.data
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "multivalue":
        rows[0].value.append(EPOCH)
    else:
        rows[0].value[0] = "a" * 2048
    with pytest.raises(ValueError):
        admit(SensorFrameClock(expected_scene_epoch=EPOCH), message)


# 功能：
#   验证重放或未来帧不会污染顺序基线，同时允许跳过中间序号后的完整合法快照。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_rejected_packet_cannot_advance_state_but_valid_full_snapshot_can_skip():
    clock = SensorFrameClock(expected_scene_epoch=EPOCH)
    admit(clock, frame())
    with pytest.raises(ValueError, match="REPLAY"):
        admit(clock, frame(), now=1_150_000_000, mono=10.05)
    future = frame(sequence="99", source_unix_ns="2000000000")
    with pytest.raises(ValueError, match="FUTURE"):
        admit(clock, future)
    valid = frame(sequence="5", source_unix_ns="1100000000", simulation_ns="600000000")
    valid.header.stamp.nsec = 600_000_000
    assert admit(clock, valid, now=1_200_000_000, mono=10.1).sequence == 5


# 功能：
#   验证布尔 UNIX 时间、NaN、负值及布尔单调钟不会成为合法接收时钟。
# 输入：
#   now：待验证的接收 UNIX 纳秒。
#   mono：待验证的接收单调钟秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("now,mono", [(True, 10.), (1, float("nan")), (1, -1.), (1, True)])
def test_invalid_host_clock(now, mono):
    with pytest.raises(ValueError, match="RECEIPT_CLOCK_INVALID"):
        admit(SensorFrameClock(), NS(header=NS(data=[])), now, mono)


# 功能：
#   验证坏来源不刷新最后合法帧，并确认入口最多保留四十八条时钟记录。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_bounded_ingress_rejects_bad_source_without_renewing_last_frame():
    ingress = SensorImageIngress(expected_scene_epoch=EPOCH)
    first = ingress.admit("rgb", frame(), received_unix_ns=1_100_000_000,
                          received_monotonic_seconds=10.)
    assert ingress.admit("rgb", frame(epoch="c" * 64), received_unix_ns=1_150_000_000,
                          received_monotonic_seconds=10.05) is None
    assert ingress.lookup("rgb", 9.9) == first
    assert ingress.lookup("rgb", 9.95) is None
    assert ingress.summary()["rejected"] == {"rgb:SENSOR_SCENE_IDENTITY_MISMATCH": 1}
    direct = SensorImageIngress()
    for index in range(100):
        direct.admit("depth", NS(header=NS(data=[])), received_unix_ns=1_000_000_000 + index,
                     received_monotonic_seconds=1. + index / 1e9)
    assert direct.summary()["history_entries"] == 48
    assert direct.lookup("depth", 1.) is None


# 功能：
#   验证频率诊断分别累计来源、接收及仿真间隔，不将重复图像计入有效采样。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_camera_cadence_compares_actual_clocks_and_never_counts_rejected_frames():
    ingress = SensorImageIngress(expected_scene_epoch=EPOCH)
    first = ingress.admit("depth", frame(), received_unix_ns=1_100_000_000,
                          received_monotonic_seconds=10.)
    # Replay arrives later; it is rejected and not a new sensor interval.
    assert ingress.admit("depth", frame(), received_unix_ns=1_150_000_000,
                         received_monotonic_seconds=10.05) is None
    next_frame = frame(sequence="3", source_unix_ns="1100000000", simulation_ns="520000000")
    next_frame.header.stamp.nsec = 520_000_000
    second = ingress.admit("depth", next_frame, received_unix_ns=1_230_000_000,
                           received_monotonic_seconds=10.13)
    assert first.source_unix_ns == 1_000_000_000 and second.source_unix_ns == 1_100_000_000
    summary = ingress.summary()
    stats = summary["accepted_frame_cadence"]["depth"]
    assert stats == {"interval_count": 1, "source_span_ns": 100_000_000,
        "receive_span_seconds": pytest.approx(.13), "simulation_interval_count": 1,
        "simulation_span_ns": 20_000_000}
    assert summary["accepted"] == {"depth": 2}
    assert summary["qualification_granted"] is False
    stats["interval_count"] = 999
    assert ingress.summary()["accepted_frame_cadence"]["depth"]["interval_count"] == 1


# 功能：
#   验证图像缓存、异步编码及多模态证据完整保留原始时钟，不以处理完成时间延长寿命。
# 输入：
#   tmp_path：多模态证据使用的独立路径上下文。
# 输出：
#   None：不返回业务数据。
def test_encoding_and_async_worker_preserve_original_scene_time(tmp_path):
    from dronedream_agent_core.model_image_cache import ModelImageCache
    from dronedream_agent_core.model_image_worker import LatestModelImageWorker
    from tests.test_model_image_worker import wait_sample

    message = frame()
    clock = admit(SensorFrameClock(expected_scene_epoch=EPOCH), message)
    arguments = dict(received_monotonic_seconds=clock.sample_monotonic_seconds,
                     received_at_unix_ms=clock.source_unix_ns // 1_000_000, frame_time=clock)

    # 功能：
    #   返回固定的一像素编码结果，将本用例聚焦到跨线程来源时钟绑定。
    # 输入：
    #   message：本次编码的测试图像。
    #   output_size：测试固定的一像素尺寸。
    # 输出：
    #   encoded：PNG 占位字节与三通道 RGB 字节组成的元组。
    def decode(message, *, output_size):
        encoded = b"png", b"rgb"
        return encoded

    cache = ModelImageCache()
    prepared, new = cache.prepare(message, size=(1, 1), decoder=decode, **arguments)
    assert new and prepared.frame_time == clock
    payload = prepared.multimodal(tmp_path)
    assert payload["observed_at_unix_ms"] == 1000  # NOT the 1100 ms arrival time.
    assert payload["timestamp_basis"] == "simulation-scene-capture"
    assert payload["host_received_unix_ns"] == 1_100_000_000
    assert payload["scene_epoch"] == EPOCH
    with pytest.raises(ValueError, match="CLOCK_REQUIRED"):
        cache.prepare(message, size=(1, 1), decoder=decode,
                      received_monotonic_seconds=10., received_at_unix_ms=1100)
    with pytest.raises(ValueError, match="CLOCK_REDATED"):
        cache.prepare(message, size=(1, 1), decoder=decode, **{
            **arguments, "received_at_unix_ms": 1100})
    worker = LatestModelImageWorker(size=(1, 1), decoder=decode)
    try:
        assert worker.submit(message, **arguments)
        sample = wait_sample(worker, 9.9, now=10.)
        # 场景采集在 9.9，但消息直到 10.0 才收到；9.99 的控制快照不能借用未来消息。
        assert worker.latest(now_monotonic_seconds=9.99, maximum_age_seconds=.2) is None
        assert sample.image.frame_time == clock
        assert worker.latest(now_monotonic_seconds=10.11, maximum_age_seconds=.2) is None
        assert sample.image.multimodal(tmp_path)["scene_source_unix_ns"] == 1_000_000_000
    finally:
        worker.close()


# 功能：
#   验证学习视觉证据记录真实 RGB 像素时仍保留场景时间、序号及独立接收时间。
# 输入：
#   tmp_path：学习证据关联的独立目录。
# 输出：
#   None：不返回业务数据。
def test_learning_visual_evidence_cannot_relabel_scene_time_as_receipt(tmp_path):
    from dronedream_agent_core.model_image_cache import ModelImageCache
    from scripts.runtime_depth_safety_worker import _prepare_learning_visual

    message = frame()
    message.width, message.height, message.step = 32, 32, 96
    message.pixel_format_type = 3
    message.data = b"rgb" * 32 * 32
    clock = admit(SensorFrameClock(expected_scene_epoch=EPOCH), message)
    row, = _prepare_learning_visual(cache=ModelImageCache(), image=message,
        received_monotonic=clock.sample_monotonic_seconds,
        received_utc_ms=1000, size=(32, 32), directory=tmp_path, frame_time=clock)
    assert row["observed_at_unix_ms"] == 1000
    assert row["timestamp_basis"] == "simulation-scene-capture"
    assert row["scene_sequence"] == 1
    assert row["host_received_unix_ns"] == 1_100_000_000


# 功能：
#   验证原始数据集复制来源身份而不修改调用方状态，并拒绝丢钟或重新标注采集时刻。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_raw_dataset_keeps_clock_identity_and_rejects_dropped_or_redated_clock():
    from scripts.runtime_depth_safety_worker import _dataset_clock_state
    message = frame()
    clock = admit(SensorFrameClock(expected_scene_epoch=EPOCH), message)
    state = {"task": "fixture"}
    actual = _dataset_clock_state(message, 9.9, clock, state)
    assert "rgb_source_clock" not in state
    assert actual["rgb_source_clock"]["source_unix_ns"] == 1_000_000_000
    assert actual["rgb_source_clock"]["received_unix_ns"] == 1_100_000_000
    assert actual["rgb_source_clock"]["scene_sha256"] == "b" * 64
    with pytest.raises(ValueError, match="CLOCK_REQUIRED"):
        _dataset_clock_state(message, 9.9, None, state)
    with pytest.raises(ValueError, match="CLOCK_REDATED"):
        _dataset_clock_state(message, 10., clock, state)


# 功能：
#   验证外部传入的纪元类型不会触发未归类的正则异常。
# 输入：
#   epoch：非字符串的场景纪元。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("epoch", [True, 42, b"a" * 64, []])
def test_epoch_requires_a_text_digest(epoch):
    with pytest.raises(ValueError, match="EPOCH_INVALID"):
        SensorFrameClock(expected_scene_epoch=epoch)


# 功能：
#   验证非法单调钟值被入口作为拒绝记录，不打断后续合法图像接入。
# 输入：
#   mono：非数字或浮点转换会溢出的单调钟值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mono", [None, "10", 1 << 5000], ids=["none", "text", "overflow"])
def test_ingress_invalid_receipt_types_are_contained(mono):
    ingress = SensorImageIngress(expected_scene_epoch=EPOCH)
    assert ingress.admit("rgb", frame(), received_unix_ns=1_100_000_000,
                         received_monotonic_seconds=mono) is None
    assert ingress.summary()["accepted"] == {}
    assert ingress.admit("rgb", frame(), received_unix_ns=1_100_000_000,
                         received_monotonic_seconds=10.).sequence == 1


# 功能：
#   验证畸形报头在固定入口被拒绝，不能成为未捕获异常或字符串字符数组。
# 输入：
#   mutation：破坏报头结构的方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["rows-none", "key-number", "value-none", "value-number",
                                     "value-text", "empty-row"])
def test_ingress_malformed_header_types_are_contained(mutation):
    message = frame()
    if mutation == "rows-none":
        message.header.data = None
    elif mutation == "key-number":
        message.header.data[0].key = 1
    elif mutation == "value-none":
        message.header.data[0].value = None
    elif mutation == "value-number":
        message.header.data[0].value = [123]
    elif mutation == "value-text":
        message.header.data[2].value = "1"
    else:
        message.header.data[0] = NS()
    ingress = SensorImageIngress(expected_scene_epoch=EPOCH)
    assert ingress.admit("rgb", message, received_unix_ns=1_100_000_000,
                         received_monotonic_seconds=10.) is None
    assert not ingress.summary()["accepted"]
    assert ingress.admit("rgb", frame(), received_unix_ns=1_100_000_000,
                         received_monotonic_seconds=10.).sequence == 1


# 功能：
#   验证编码阶段不能重新标注直连图像的来源时间或丢失已接入的发布者时间戳。
# 输入：
#   changes：对合法直连时钟做的单字段篡改。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("changes", [
    {"source_unix_ns": 1_000_000_000}, {"sample_monotonic_seconds": 9.9},
    {"received_monotonic_seconds": 11.}, {"simulation_ns": None},
    {"epoch": EPOCH}, {"sequence": 1},
])
def test_direct_encoding_requires_the_exact_admitted_clock(changes):
    message = NS(header=NS(data=[], stamp=NS(sec=0, nsec=500_000_000)))
    clock = replace(admit(SensorFrameClock(), message), **changes)
    with pytest.raises(ValueError, match="MODEL_RGB_"):
        require_model_frame_time(message, clock, clock.sample_monotonic_seconds,
                                 clock.source_unix_ns // 1_000_000)


# 功能：
#   验证布尔或浮点字段不能利用 Python 数值相等绕过整数时钟契约。
# 输入：
#   changes：与原数值相等但类型错误的字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("changes", [{"sequence": True}, {"simulation_ns": 500_000_000.},
                                     {"source_unix_ns": 1_000_000_000.}])
def test_scene_clock_revalidation_rejects_equal_values_with_wrong_types(changes):
    message = frame()
    clock = replace(admit(SensorFrameClock(expected_scene_epoch=EPOCH), message), **changes)
    with pytest.raises(ValueError, match="MODEL_RGB_"):
        require_model_frame_time(message, clock, 9.9, 1000)


# 功能：
#   验证 protobuf 未填充的默认时间戳不被误认为仿真零时刻，不阻碍随后真实零时刻。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_absent_protobuf_stamp_is_not_a_real_zero_simulation_stamp():
    clock = SensorFrameClock()
    message = NS(HasField=lambda name: True, header=NS(
        data=[], HasField=lambda name: False, stamp=NS(sec=0, nsec=0)))
    assert admit(clock, message).simulation_ns is None
    message.header.HasField = lambda name: True
    assert admit(clock, message, now=1_200_000_000, mono=10.1).simulation_ns == 0


# 功能：
#   验证错误流类型统一计数且不创建无限量诊断键，不影响合法 RGB 接入。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_unknown_streams_do_not_expand_ingress_diagnostics():
    ingress = SensorImageIngress(expected_scene_epoch=EPOCH)
    for kind in [None, [], *(f"unknown-{index}" for index in range(100))]:
        assert ingress.admit(kind, frame(), received_unix_ns=1_100_000_000,
                             received_monotonic_seconds=10.) is None
    summary = ingress.summary()
    assert summary["rejected"] == {"unknown:SENSOR_STREAM_KIND_INVALID": 102}
    assert summary["history_entries"] == 0
    assert ingress.admit("rgb", frame(), received_unix_ns=1_100_000_000,
                         received_monotonic_seconds=10.) is not None
