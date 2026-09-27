import math
import struct

import pytest

from dronedream_agent_core.depth_projection import (
    DepthProjectionCalibration,
    metric_depth_sample_stride,
    project_float32_depth_image,
    project_metric_depth_frame,
)


# 功能：
#   构造小型解析测试的针孔校准，仅用显式覆盖修改待测条件。
# 输入：
#   updates：测试需要改变的校准参数。
# 输出：
#   calibration：通过真实校准构造器验证的测试配置。
def _calibration(**updates: object) -> DepthProjectionCalibration:
    values = {
        "width": 4,
        "height": 4,
        "horizontal_fov_rad": math.pi / 2,
        "minimum_depth_m": 0.2,
        "maximum_depth_m": 10.0,
        "sample_stride_pixels": 2,
        "no_return_mode": "gazebo-far-clip",
    }
    values.update(updates)
    calibration = DepthProjectionCalibration(**values)  # type: ignore[arg-type]
    return calibration


# 功能：
#   不同实际帧尺寸采用相应采样区块，保留相近射线预算而不沿用旧尺寸的固定步长。
# 输入：
#   width、height、stride：实际图像尺寸及预期区块边长。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("width,height,stride", [(640, 480, 32), (320, 240, 16), (160, 120, 8)])
def test_explicit_depth_profiles_preserve_spatial_ray_budget(width, height, stride):
    selected = metric_depth_sample_stride(width=width, height=height)
    assert selected == stride
    samples = project_float32_depth_image(
        data=struct.pack("<f", 2.) * width * height, row_step_bytes=width * 4,
        calibration=DepthProjectionCalibration(width=width, height=height,
            horizontal_fov_rad=1.274, minimum_depth_m=.2, maximum_depth_m=19.1,
            sample_stride_pixels=selected),
    )
    assert len(samples) == 300
    assert all(sample.hit for sample in samples)


# 功能：
#   非整数、零或过大像素尺寸不能进入采样预算计算。
# 输入：
#   width、height：非法尺寸组合。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("width,height", [(True, 480), (320, 0), (8193, 240), (640., 480)])
def test_depth_sample_density_rejects_invalid_dimensions(width, height):
    with pytest.raises(ValueError, match="sampling dimensions"):
        metric_depth_sample_stride(width=width, height=height)


# 功能：
#   按小端 float32 编码四行像素，并加入明确的行填充字节。
# 输入：
#   values：十六个轴向深度值。
#   row_padding_bytes：每行附加的填充字节数。
# 输出：
#   data、step：编码缓冲及每行实际字节跨度。
def _image(values: list[float], *, row_padding_bytes: int = 0) -> tuple[bytes, int]:
    rows: list[bytes] = []
    for offset in range(0, 16, 4):
        rows.append(
            b"".join(struct.pack("<f", value) for value in values[offset : offset + 4])
            + b"\0" * row_padding_bytes
        )
    data, step = b"".join(rows), 16 + row_padding_bytes
    return data, step


# 功能：
#   正确跳过行填充，并按射线方向把径向距离还原为原始轴向深度。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_depth_projection_preserves_axial_geometry_and_padding() -> None:
    data, step = _image([2.0] * 16, row_padding_bytes=8)

    samples = project_float32_depth_image(
        data=data,
        row_step_bytes=step,
        calibration=_calibration(),
    )

    assert len(samples) == 4
    sample = samples[0]
    direction = sample.direction_sensor
    direction_norm = math.sqrt(direction.x**2 + direction.y**2 + direction.z**2)
    assert sample.range_m / direction_norm == pytest.approx(2.0)
    assert sample.hit is True


# 功能：
#   非法像素不产生射线，合法命中与显式模拟无回波分别保留。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_depth_projection_omits_invalid_pixels_and_preserves_no_return_rays() -> None:
    values = [math.nan] * 16
    values[5] = 10.0
    values[7] = 3.0
    values[13] = 0.1
    values[15] = math.inf
    data, step = _image(values)

    samples = project_float32_depth_image(
        data=data,
        row_step_bytes=step,
        calibration=_calibration(),
    )

    assert len(samples) == 3
    assert [sample.hit for sample in samples] == [False, True, False]
    assert samples[0].range_m == pytest.approx(10.0)
    assert samples[2].range_m == pytest.approx(10.0)


# 功能：
#   在明确 Gazebo 裁剪语义下，无穷深度可产生无命中射线，不宣称整个区块为空。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_depth_projection_treats_open_gazebo_frame_as_calibrated_free_space() -> None:
    data, step = _image([math.inf] * 16)

    samples = project_float32_depth_image(
        data=data,
        row_step_bytes=step,
        calibration=_calibration(),
    )

    assert len(samples) == 4
    assert all(sample.hit is False for sample in samples)
    assert all(sample.range_m == pytest.approx(10.0) for sample in samples)


# 功能：
#   轴向深度合法但径向距离越出量程时，不生成超量程障碍命中。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_depth_projection_never_emits_off_axis_hits_beyond_radial_calibration() -> None:
    data, step = _image([9.5] * 16)

    samples = project_float32_depth_image(
        data=data,
        row_step_bytes=step,
        calibration=_calibration(),
    )

    assert any(sample.hit is False for sample in samples)
    assert all(sample.range_m <= 10.0 for sample in samples)


# 功能：
#   缓冲不足以覆盖完整行布局时，在像素计算前拒绝截断帧。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_depth_projection_rejects_truncated_frame() -> None:
    with pytest.raises(ValueError, match="truncated"):
        project_float32_depth_image(
            data=b"\0" * 8,
            row_step_bytes=16,
            calibration=_calibration(),
        )


# 功能：
#   枚举区块内每种像素相位，确认细障碍保留其真实位置而非旧规则采样中心。
# 输入：
#   row、col：细障碍在区块内的位置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("row", range(8))
@pytest.mark.parametrize("col", range(8))
def test_thin_obstacle_at_every_subsample_phase_keeps_actual_geometry(row, col):
    values = [8.] * 256
    values[row * 16 + col] = 1.25
    calibration = _calibration(width=16, height=16, sample_stride_pixels=8)
    result = project_metric_depth_frame(
        data=struct.pack("<256f", *values), row_step_bytes=64, calibration=calibration,
    )
    near = [s for s in result.samples if s.hit and s.range_m < 3.]
    assert len(near) == 1
    # Independent pinhole forward projection, not the implementation's inverse.
    s = near[0]
    norm = math.sqrt(sum(v*v for v in s.direction_sensor.model_dump().values()))
    point = [s.range_m * v/norm for v in s.direction_sensor.model_dump().values()]
    assert point == pytest.approx([1.25, (7.5-col)*1.25/8, (7.5-row)*1.25/8])
    assert result.source_coverage == 1.
    assert len(result.samples) == 4


# 功能：
#   无效像素不被补成自由空间，覆盖率严格等于有效原始像素比例。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_invalid_pixels_do_not_become_free_and_source_coverage_is_measured():
    values = [math.nan, -math.inf, -1., .1] * 4
    values[0] = 1.
    data, step = _image(values)
    result = project_metric_depth_frame(data=data, row_step_bytes=step, calibration=_calibration())
    assert result.source_coverage == 1/16
    assert len(result.samples) == 1 and result.samples[0].hit


# 功能：
#   未声明裁剪语义的硬件无回波不能继承模拟器的自由射线解释。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_unknown_hardware_no_return_never_becomes_free():
    for value in (math.inf, 10., 11., math.nan, -.1):
        data, step = _image([value] * 16)
        with pytest.raises(ValueError, match="no calibrated"):
            project_metric_depth_frame(data=data, row_step_bytes=step,
                                       calibration=_calibration(no_return_mode="unknown"))


# 功能：
#   保留右下边缘不完整区块，使用非对称显式内参计算真实方向并绑定不同校准摘要。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_partial_edge_tiles_and_explicit_intrinsics_are_not_lost():
    calibration = _calibration(width=5, height=3, sample_stride_pixels=4,
                               fx_pixels=7., fy_pixels=9., cx_pixels=1., cy_pixels=.5)
    values = [math.nan] * 15
    values[-1] = 2.
    result = project_metric_depth_frame(data=struct.pack("<15f", *values),
                                       row_step_bytes=20, calibration=calibration)
    sample = result.samples[0]
    assert sample.direction_sensor.model_dump() == pytest.approx(
        {"x": 1., "y": -3/7, "z": -1.5/9})
    assert result.valid_pixel_count == 1
    assert result.calibration_sha256 != _calibration().sha256


# 功能：
#   非法光学、量程、置信度或输出预算在读取任何像素前被拒绝。
# 输入：
#   updates：非法校准参数组合。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("updates", [
    {"width": True}, {"height": 4.5}, {"minimum_depth_m": math.nan},
    {"maximum_depth_m": math.inf}, {"confidence": True},
    {"fx_pixels": 5.}, {"sample_stride_pixels": True},
    {"horizontal_fov_rad": math.nan}, {"no_return_mode": "guess"},
    {"width": 8192, "height": 8192},
    {"fx_pixels": 1e-300, "fy_pixels": 1., "cx_pixels": 1., "cy_pixels": 1.},
])
def test_bad_calibration_fails_before_reading_pixels(updates):
    with pytest.raises(ValueError):
        _calibration(**updates)


# 功能：
#   区块选择按离传感器最近的径向距离，而不是仅比较轴向深度。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_nearest_radial_not_nearest_axial_sample_is_retained():
    values = [math.nan] * 16
    values[0], values[5] = 1., 1.1
    data, step = _image(values)
    result = project_metric_depth_frame(data=data, row_step_bytes=step,
                                       calibration=_calibration(sample_stride_pixels=4))
    assert result.samples[0].direction_sensor.y == pytest.approx(.25)
    assert result.samples[0].range_m == pytest.approx(1.1 * math.sqrt(1.125))


# 功能：
#   按 memoryview 实际字节数接入合法浮点缓冲，而不是按元素个数误判截断。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_projection_accepts_typed_contiguous_buffer():
    data, step = _image([2.] * 16)
    expected = project_metric_depth_frame(data=data, row_step_bytes=step,
                                         calibration=_calibration())
    actual = project_metric_depth_frame(data=memoryview(data).cast("f"), row_step_bytes=step,
                                        calibration=_calibration())
    assert actual == expected


# 功能：
#   错误缓冲类型、不连续视图及额外尾部不能绕过布局绑定。
# 输入：
#   kind：缓冲损坏方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["integer", "list", "strided", "trailing"])
def test_projection_rejects_ambiguous_pixel_buffers(kind):
    data, step = _image([2.] * 16)
    damaged = {"integer": 64, "list": list(data), "strided": memoryview(data)[::2],
               "trailing": data + b"extra"}[kind]
    with pytest.raises(ValueError):
        project_metric_depth_frame(data=damaged, row_step_bytes=step, calibration=_calibration())


# 功能：
#   无回波模式不是字符串时应明确拒绝，而不是触发集合哈希类型错误。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_calibration_rejects_nonstring_no_return_mode():
    with pytest.raises(ValueError):
        _calibration(no_return_mode=[])


# 功能：以独立逐像素参考验证批量归约、边缘区块、并列选择和未知区域完全一致。
# 输入：多种尺寸、超图幅步长及带行填充的随机坏像素/命中/无回波混合图。
# 输出：每条射线方向、距离、命中及有效像素计数等价，不以提速改变几何语义。
@pytest.mark.parametrize("width,height,stride", [(31, 19, 8), (8, 5, 8192), (2, 2, 1),
                                               (160, 120, 8), (320, 240, 16)])
@pytest.mark.parametrize("mode", ["unknown", "gazebo-far-clip"])
def test_batched_tiles_match_independent_pixel_reference(width, height, stride, mode):
    import random

    rng = random.Random(width * height + stride)
    choices = [math.nan, math.inf, -math.inf, -.1, .1, .2, 1., 2., 9., 10., 11.]
    values = [rng.choice(choices) for _ in range(width * height)]
    values[0] = 1.
    data = b"".join(struct.pack(f"<{width}f", *values[row*width:(row+1)*width]) + b"pad!"
                    for row in range(height))
    calibration = _calibration(width=width, height=height, sample_stride_pixels=stride,
                               no_return_mode=mode)
    fx, fy, cx, cy = calibration.intrinsics
    far = calibration.maximum_depth_m
    expected, valid_count = [], 0
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            hits, free = [], []
            for row in range(top, min(height, top+stride)):
                for col in range(left, min(width, left+stride)):
                    value = struct.unpack("<f", struct.pack("<f", values[row*width+col]))[0]
                    finite = math.isfinite(value)
                    valid = finite and calibration.minimum_depth_m <= value <= far + 1e-5
                    valid = (valid or value == math.inf) if mode == "gazebo-far-clip" else (
                        valid and value < far-1e-5)
                    if not valid:
                        continue
                    valid_count += 1
                    y, z = (cx-col)/fx, (cy-row)/fy
                    radial = min(value, far)**2 * (1.+y*y+z*z) if finite else math.inf
                    if finite and radial < far*far and value < far-1e-5:
                        hits.append((radial, row, col, y, z))
                    free.append((abs(row % stride-stride//2)+abs(col % stride-stride//2),
                                 row, col, y, z))
            if hits or free:
                cost, _, _, y, z = min(hits or free)
                expected.append((y, z, math.sqrt(cost) if hits else far, bool(hits)))
    actual = project_metric_depth_frame(data=data, row_step_bytes=width*4+4,
                                       calibration=calibration)
    assert actual.valid_pixel_count == valid_count
    assert len(actual.samples) == len(expected)
    for sample, (y, z, distance, hit) in zip(actual.samples, expected, strict=True):
        assert (sample.direction_sensor.y, sample.direction_sensor.z, sample.range_m) == (
            pytest.approx((y, z, distance), rel=1e-14, abs=1e-14))
        assert sample.hit is hit
