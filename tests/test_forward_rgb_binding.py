import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from dronedream_agent_core.forward_rgb_binding import ForwardRgbBinding
from dronedream_agent_core.runtime_sensor_contracts import RuntimeSensorRegistry
from dronedream_agent_core.sensor_frame_clock import SensorFrameTime


# 功能：
#   建立来源与接收时间相同的直连 RGB 样本，默认像素包含足够的空间对比。
# 输入：
#   source：当前帧来源单调钟秒数，同时确定测试 UNIX 时间。
#   payload：待登记的像素缓冲。
# 输出：
#   result：图像消息与原始时钟、当前登记时间参数组成的元组。
def sample(source=10., *, payload=(b"\x20\x20\x20" + b"\xc0\xc0\xc0") * 512):
    frame = SimpleNamespace(width=32, height=32, step=96, pixel_format_type=3, data=payload)
    clock = SensorFrameTime(round(source * 1e9), round(source * 1e9),
                            source, source, "host-receipt")
    result = frame, dict(frame_time=clock, source_monotonic=source,
                         source_unix_ms=round(source * 1000), now_monotonic=source + .01)
    return result


# 功能：
#   验证不同质量的 RGB 帧同步更新原始时间和摘要，迟来的重复登记不增加序号。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_late_completed_rgb_updates_quality_and_source_together_without_redating():
    registry = RuntimeSensorRegistry()
    binding = ForwardRgbBinding(registry, vehicle_id="quad")
    first, args = sample(payload=b"\x10" * 3072)
    assert binding.register(first, **args)
    previous = registry.latest("oakd-lite-forward-rgb")
    assert previous.quality == 0. and previous.sample_monotonic_seconds == 10.
    newer, new_args = sample(10.05)
    assert binding.register(newer, **new_args)
    current = registry.latest("oakd-lite-forward-rgb")
    assert current.sequence == 2 and current.sample_monotonic_seconds == 10.05
    assert current.received_monotonic_seconds == 10.05 and current.quality == 1.
    assert current.payload_sha256 == hashlib.sha256(newer.data).hexdigest()
    assert not registry.contract("oakd-lite-forward-rgb").required_for_motion
    assert previous.sequence == 1 and previous.quality == 0.
    assert not binding.register(newer, **{**new_args, "now_monotonic": 10.08})
    assert registry.latest("oakd-lite-forward-rgb") == current


# 功能：
#   验证同钟换图、时钟变化、错误布局、过期和回退输入不能替换已接受的注册表观测。
# 输入：
#   kind：本次对后续输入实施的非法变更。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", [
    "pixels", "clock", "dimensions", "stride", "expired", "regressed",
])
def test_rejected_source_cannot_replace_accepted_binding(kind):
    registry = RuntimeSensorRegistry()
    binding = ForwardRgbBinding(registry, vehicle_id="quad")
    frame, args = sample()
    binding.register(frame, **args)
    previous = registry.latest("oakd-lite-forward-rgb")
    if kind == "pixels":
        frame.data = b"\x01" * 3072
    elif kind == "clock":
        args["frame_time"] = replace(args["frame_time"], received_monotonic_seconds=10.02)
    elif kind == "dimensions":
        frame.width = 33
    elif kind == "stride":
        frame.step = 93
    elif kind == "expired":
        frame, args = sample(10.01)
        args["now_monotonic"] = 12.
    else:
        frame, args = sample(9.99)
    with pytest.raises(ValueError):
        binding.register(frame, **args)
    assert registry.latest("oakd-lite-forward-rgb") == previous
    good, good_args = sample(10.05)
    assert binding.register(good, **good_args)
    assert registry.latest("oakd-lite-forward-rgb").sequence == 2


# 功能：
#   验证首次登记必须具有完整原始时钟和像素，失败后不会出现虚假的最新观测。
# 输入：
#   kind：对首帧来源或像素进行的破坏。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["missing-clock", "redated", "incomplete"])
def test_original_source_and_complete_pixels_are_required(kind):
    registry = RuntimeSensorRegistry()
    binding = ForwardRgbBinding(registry, vehicle_id="quad")
    frame, args = sample()
    if kind == "missing-clock":
        args["frame_time"] = None
    elif kind == "redated":
        args["source_monotonic"] = 10.01
    else:
        frame.data = b"\x01"
    with pytest.raises(ValueError):
        binding.register(frame, **args)
    assert registry.latest("oakd-lite-forward-rgb") is None


# 功能：
#   验证颜色通道之间的差异不被误认为相邻像素之间的空间纹理。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_uniform_color_is_not_visual_texture():
    binding = ForwardRgbBinding(RuntimeSensorRegistry(), vehicle_id="quad")
    frame, args = sample(payload=b"\x00\x40\x80" * 1024)
    binding.register(frame, **args)
    assert binding.last_quality.score == 0.
    assert "RGB_LOW_SPATIAL_INFORMATION" in binding.last_quality.issue_codes


# 功能：
#   复现解码期间调用方复用输入缓冲，要求质量和摘要仍来自同一个冻结像素快照。
# 输入：
#   monkeypatch：插入模拟缓冲复用的解码探针。
# 输出：
#   None：不返回业务数据。
def test_quality_and_digest_share_one_snapshot_when_caller_reuses_pixels(monkeypatch):
    import dronedream_agent_core.forward_rgb_binding as module

    registry = RuntimeSensorRegistry()
    binding = ForwardRgbBinding(registry, vehicle_id="quad")
    message, args = sample(payload=bytearray(b"\x40" * 3072))
    original = bytes(message.data)
    decode = module.decode_gazebo_rgb

    # 功能：
    #   解码后改写原消息的像素，模拟下一帧复用缓冲而非新建消息。
    # 输入：
    #   snapshot：绑定器传入的图像快照。
    # 输出：
    #   rgb：原快照对应的解码图像。
    def decode_and_reuse(snapshot):
        rgb = decode(snapshot)
        message.data[:] = b"\xff" * 3072
        return rgb

    monkeypatch.setattr(module, "decode_gazebo_rgb", decode_and_reuse)
    assert binding.register(message, **args)
    current = registry.latest("oakd-lite-forward-rgb")
    assert current.payload_sha256 == hashlib.sha256(original).hexdigest()
    assert binding.last_quality.exposure_clipped_fraction == 0.


# 功能：
#   验证相同来源与像素的重复调用不再次执行解码和质量计算，也不增加传感器序号。
# 输入：
#   monkeypatch：记录实际解码次数的测试替换工具。
# 输出：
#   None：不返回业务数据。
def test_identical_source_does_not_repeat_pixel_decoding(monkeypatch):
    import dronedream_agent_core.forward_rgb_binding as module

    calls = []
    decode = module.decode_gazebo_rgb

    # 功能：
    #   记录绑定器实际执行的像素解码，仍调用生产解码函数。
    # 输入：
    #   message：本次准备解码的快照。
    # 输出：
    #   rgb：生产解码器返回的图像。
    def counted_decode(message):
        calls.append(message)
        rgb = decode(message)
        return rgb

    monkeypatch.setattr(module, "decode_gazebo_rgb", counted_decode)
    registry = RuntimeSensorRegistry()
    binding = ForwardRgbBinding(registry, vehicle_id="quad")
    message, args = sample()
    assert binding.register(message, **args)
    assert not binding.register(message, **args)
    assert len(calls) == 1
    assert registry.latest("oakd-lite-forward-rgb").sequence == 1
