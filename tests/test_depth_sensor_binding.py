import struct
from types import SimpleNamespace

import pytest

from dronedream_agent_core.depth_sensor_binding import DepthSensorBinding
from dronedream_agent_core.runtime_sensor_contracts import RuntimeSensorRegistry


# 功能：
#   生成明确尺寸、行跨度和轴向浮点编码的合成深度消息。
# 输入：
#   width、height：图像像素尺寸。
# 输出：
#   message：所有像素轴向距离为两米的测试帧。
def frame(width=160, height=120):
    message = SimpleNamespace(width=width, height=height, step=width*4, pixel_format_type=13,
                           data=struct.pack("<f", 2.)*width*height)
    return message


# 功能：
#   每种实际流尺寸形成不同校准身份，不沿用历史原生尺寸，也不把绑定本身视为运动许可。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_real_stream_dimensions_bind_contract_not_old_native_dimensions():
    hashes = []
    for width, height in [(160, 120), (320, 240), (640, 480)]:
        registry = RuntimeSensorRegistry()
        assert not registry.snapshot(now_monotonic_seconds=1.).ready_for_motion
        binding = DepthSensorBinding(registry, vehicle_id="quad")
        projected = binding.project(frame(width, height))
        hashes.append(registry.contract("oakd-lite-depth").calibration_sha256)
        assert len(projected.samples) == 300
        assert projected.source_coverage == 1.
        # Binding alone is not a measurement, never a motion permit.
        assert not registry.snapshot(now_monotonic_seconds=1.).ready_for_motion
        with pytest.raises(ValueError, match="LAYOUT_CHANGED"):
            binding.project(frame(width//2, height//2))
    assert len(set(hashes)) == 3


# 功能：
#   首帧解码失败不登记校准，随后合法帧仍能建立真实布局绑定。
# 输入：
#   failure：首帧损坏方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failure", ["format", "stride", "payload", "invalid"])
def test_invalid_first_frame_does_not_establish_a_contract(failure):
    registry = RuntimeSensorRegistry()
    binding = DepthSensorBinding(registry, vehicle_id="quad")
    message = frame()
    if failure == "format":
        message.pixel_format_type = 6  # RGB_INT16, NOT R_FLOAT32.
    elif failure == "stride":
        message.step = 100
    elif failure == "payload":
        message.data = b""
    else:
        message.data = struct.pack("<f", float("nan"))*160*120
    with pytest.raises(ValueError):
        binding.project(message)
    with pytest.raises(ValueError, match="NOT_REGISTERED"):
        registry.contract("oakd-lite-depth")
    assert binding.project(frame()).source_coverage == 1.


# 功能：
#   任意列表和自定义字节转换对象不能伪装成传感器像素缓冲。
# 输入：
#   kind：非法缓冲类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["list", "convertible", "missing-layout"])
def test_binding_rejects_implicit_pixel_coercion(kind):
    message = frame()
    pixels = message.data
    class Convertible:
        # 功能：
        #   模拟不应被传感器入口调用的隐式字节生成器。
        # 输入：
        #   self：测试转换对象。
        # 输出：
        #   pixels：原始合成像素。
        def __bytes__(self):
            return pixels
    if kind == "missing-layout":
        message = object()
    else:
        message.data = list(pixels) if kind == "list" else Convertible()
    binding = DepthSensorBinding(RuntimeSensorRegistry(), vehicle_id="quad")
    with pytest.raises(ValueError):
        binding.project(message)
    assert binding._calibration is None


# 功能：
#   浮点 memoryview 按实际字节数验证，不将元素数量误判成像素缓冲长度。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_typed_memoryview_preserves_depth_pixels():
    message = frame()
    message.data = memoryview(message.data).cast("f")
    projection = DepthSensorBinding(RuntimeSensorRegistry(), vehicle_id="quad").project(message)
    assert projection.source_coverage == 1.
