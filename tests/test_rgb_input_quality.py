from array import array
from types import SimpleNamespace

import pytest
from PIL import Image

from dronedream_agent_core.gazebo_adapter import _gazebo_image_model_payload
from dronedream_agent_core.rgb_input_quality import (
    decode_gazebo_rgb,
    freeze_gazebo_rgb,
    measure_rgb_input_quality,
)


# 功能：
#   验证四种 RGB 排列的模型输入与质量统计使用相同真实像素，不混入行填充或透明度。
# 输入：
#   pixel_format：仿真相机的像素格式编号。
#   raw：该格式下一个固定颜色像素的原始字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("pixel_format,raw", [
    (3, bytes([20, 80, 140])), (8, bytes([140, 80, 20])),
    (4, bytes([20, 80, 140, 250])), (5, bytes([140, 80, 20, 250])),
])
def test_quality_and_model_read_identical_real_pixels_not_padding_or_alpha(pixel_format, raw):
    message = SimpleNamespace(width=32, height=32, pixel_format_type=pixel_format,
                              step=len(raw)*32+8, data=(raw*32 + b"\xff\x00"*4)*32)
    with decode_gazebo_rgb(message) as rgb:
        assert rgb.getpixel((4, 5)) == (20, 80, 140)
        quality = measure_rgb_input_quality(rgb)
        assert quality.score == 0.
        assert quality.exposure_clipped_fraction == 0.
        _, model_bytes = _gazebo_image_model_payload(message, output_size=(32, 32))
        assert model_bytes == rgb.tobytes()


# 功能：
#   验证全黑与全白图像的曝光问题显式出现，不能得到非零质量分数。
# 输入：
#   color：每个颜色通道的测试亮度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("color", [0, 255])
def test_exposure_failure_is_explicit(color):
    with Image.new("RGB", (32, 32), (color,)*3) as image:
        quality = measure_rgb_input_quality(image)
    assert quality.score == 0.
    assert quality.exposure_clipped_fraction == 1.
    assert "RGB_EXPOSURE_CLIPPED" in quality.issue_codes


# 功能：
#   验证单个亮点不能让平坦图像通过稳健纹理统计。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_isolated_pixel_does_not_create_robust_texture():
    with Image.new("RGB", (64, 64), (64,)*3) as image:
        image.putpixel((20, 20), (230,)*3)
        assert measure_rgb_input_quality(image).score == 0.


# 功能：
#   验证零跨度、缺通道字节、布尔跨度和超预算行填充不会被静默修补。
# 输入：
#   step：注入图像消息的每行字节跨度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("step", [0, 94, True, 100_000])
def test_corrupt_stride_is_not_silently_repaired(step):
    message = SimpleNamespace(width=32, height=32, step=step, pixel_format_type=3,
                              data=b"\x40" * 4096)
    with pytest.raises(ValueError):
        decode_gazebo_rgb(message)


# 功能：
#   验证 RGB 解码只接受真实字节缓冲，不把整数当作分配长度或把列表转换成像素。
# 输入：
#   payload：足够长但类型不合法的图像数据。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("payload", [3072, [64] * 3072, None], ids=["integer", "list", "none"])
def test_rgb_rejects_non_buffer_payload(payload):
    message = SimpleNamespace(width=32, height=32, step=96, pixel_format_type=3, data=payload)
    with pytest.raises(ValueError, match="FORWARD_RGB_"):
        decode_gazebo_rgb(message)


# 功能：
#   验证空图及错误颜色模式不能产生伪造质量分数或除零异常。
# 输入：
#   mode：待测试的图像颜色模式。
#   size：待测试的图像尺寸。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mode,size", [("RGB", (0, 0)), ("L", (32, 32))])
def test_quality_rejects_invalid_image_domain(mode, size):
    with Image.new(mode, size) as image, pytest.raises(ValueError, match="FORWARD_RGB_"):
        measure_rgb_input_quality(image)


# 功能：
#   验证多字节 memoryview 使用实际字节数，快照不受原缓冲及消息布局后续修改影响。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_snapshot_owns_typed_memoryview_bytes_and_layout():
    raw = array("I", [0x40404040] * 768)
    with memoryview(raw) as view:
        message = SimpleNamespace(width=32, height=32, step=96, pixel_format_type=3, data=view)
        snapshot = freeze_gazebo_rgb(message)
    raw[0] = 0
    message.width = 1
    assert snapshot.width == 32
    assert snapshot.data == b"\x40" * 3072
    with decode_gazebo_rgb(snapshot) as rgb:
        assert rgb.getpixel((0, 0)) == (64, 64, 64)


# 功能：
#   验证已释放的内存视图以稳定诊断拒绝，不继续读取或分配像素。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_released_buffer_is_rejected_with_pixel_diagnostic():
    view = memoryview(b"\x40" * 3072)
    view.release()
    message = SimpleNamespace(width=32, height=32, step=96, pixel_format_type=3, data=view)
    with pytest.raises(ValueError, match="FORWARD_RGB_PAYLOAD_TYPE_INVALID"):
        freeze_gazebo_rgb(message)


# 功能：
#   验证质量统计只关闭内部临时图像，一像素输入仍保持可读取且质量值有限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_quality_keeps_callers_single_pixel_image_open():
    with Image.new("RGB", (1, 1), (128, 128, 128)) as rgb:
        quality = measure_rgb_input_quality(rgb)
        assert quality.score == 0.
        assert quality.mean_spatial_difference == 0.
        assert rgb.getpixel((0, 0)) == (128, 128, 128)
