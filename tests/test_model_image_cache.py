from array import array
from types import SimpleNamespace

import pytest

from dronedream_agent_core.contracts import QuaternionWxyz, Vector3
from dronedream_agent_core.gazebo_adapter import _gazebo_image_model_payload
from dronedream_agent_core.model_image_cache import ModelImageCache, require_model_image_runtime
from dronedream_agent_core.runtime_sensor_contracts import (
    forward_camera_motion_alignment,
    forward_rgb_can_inform_control,
)


# 功能：
#   验证重复模型周期复用同一编码并保留原始来源时间，改写同一来源时间会被拒绝。
# 输入：
#   tmp_path：多模态图像描述使用的独立目录。
# 输出：
#   None：不返回业务数据。
def test_repeated_model_ticks_reuse_conversion_and_original_image_time(tmp_path):
    calls = []

    # 功能：
    #   记录实际编码调用，并返回符合目标尺寸的固定字节结果。
    # 输入：
    #   message：当前来源标识。
    #   output_size：请求的模型图像尺寸。
    # 输出：
    #   encoded：测试 PNG 与完整 RGB 字节对。
    def decode(message, *, output_size):
        calls.append(message)
        encoded = b"png", bytes(output_size[0] * output_size[1] * 3)
        return encoded
    cache = ModelImageCache()
    for index in range(12):
        image, new = cache.prepare("source", received_monotonic_seconds=1.,
                                   received_at_unix_ms=1000,
                                   size=(32, 32), decoder=decode)
        assert new is (index == 0)
        payload = image.multimodal(tmp_path)
        assert payload["observed_at_unix_ms"] == 1000
        assert payload["timestamp_basis"] == "host-receive-not-hardware-exposure"
    assert calls == ["source"]
    with pytest.raises(ValueError, match="REDATED"):
        cache.prepare("source", received_monotonic_seconds=1., received_at_unix_ms=1050,
                       size=(32, 32), decoder=decode)
    image, new = cache.prepare("new", received_monotonic_seconds=1.05, received_at_unix_ms=1050,
                               size=(32, 32), decoder=decode)
    assert new and image.received_at_unix_ms == 1050
    assert calls == ["source", "new"]


# 功能：
#   使用真实 Pillow 和生产 RGB 编码器，验证缓存返回 PNG 内容并保持声明的 RGB 像素。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_real_rgb_encoder_matches_declared_cached_payload():
    require_model_image_runtime()
    message = SimpleNamespace(width=32, height=32, step=96, pixel_format_type=3,
                              data=bytes([0, 128, 255] * 32 * 32))
    image, _ = ModelImageCache().prepare(message, received_monotonic_seconds=1.,
        received_at_unix_ms=1000, size=(32, 32), decoder=_gazebo_image_model_payload)
    assert image.png.startswith(b"\x89PNG")
    assert image.rgb == message.data


# 功能：
#   验证图像依赖缺失会在运行自检时产生明确错误，不留到飞行处理首帧时才发现。
# 输入：
#   monkeypatch：临时阻断 Pillow 导入的测试工具。
# 输出：
#   None：不返回业务数据。
def test_missing_image_runtime_is_detected_before_a_flight(monkeypatch):
    import builtins

    original = builtins.__import__

    # 功能：
    #   仅模拟 Pillow 缺失，其余导入仍交给原始导入函数。
    # 输入：
    #   name：请求导入的模块名。
    #   args：导入函数的其余位置参数。
    #   kwargs：导入函数的其余关键字参数。
    # 输出：
    #   module：原导入函数返回的非 Pillow 模块。
    def missing(name, *args, **kwargs):
        if name == "PIL":
            raise ModuleNotFoundError("PIL unavailable")
        module = original(name, *args, **kwargs)
        return module

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(RuntimeError, match="MODEL_RGB_RUNTIME_UNAVAILABLE"):
        require_model_image_runtime()


# 功能：
#   验证侧移或后退时前视相机不能单独证明前方路线，但连续控制模式仍可使用其辅助信息。
# 输入：
#   velocity：当前世界坐标系速度向量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("velocity", [(-.5, 0, 0), (0, .5, 0), (0, -.5, 0)])
def test_side_or_reverse_motion_does_not_blind_continuous_pilot(velocity):
    alignment = forward_camera_motion_alignment(
        body_orientation_world_from_body=QuaternionWxyz(w=1, x=0, y=0, z=0),
        body_velocity_world_enu_mps=Vector3(x=velocity[0], y=velocity[1], z=velocity[2]),
    )
    assert not alignment.aligned_for_forward_navigation
    assert forward_rgb_can_inform_control(alignment, continuous_control=True)
    assert not forward_rgb_can_inform_control(alignment, continuous_control=False)


# 功能：
#   验证缓存固定尺寸副本，调用方或解码器修改原尺寸列表不会篡改已生成图像的布局。
# 输入：
#   tmp_path：多模态描述使用的独立目录。
# 输出：
#   None：不返回业务数据。
def test_cache_owns_dimensions_before_calling_decoder(tmp_path):
    size, seen = [1, 1], []

    # 功能：
    #   在编码时修改调用方尺寸列表，观察编码器实际收到的尺寸快照。
    # 输入：
    #   message：当前测试来源。
    #   output_size：缓存传给编码器的尺寸。
    # 输出：
    #   encoded：一像素编码结果。
    def decode(message, *, output_size):
        seen.append(output_size)
        size[0] = 2
        encoded = b"png", b"rgb"
        return encoded

    image, new = ModelImageCache().prepare("source", received_monotonic_seconds=1.,
        received_at_unix_ms=1000, size=size, decoder=decode)
    assert new and image.size == (1, 1)
    assert seen == [(1, 1)]
    assert image.multimodal(tmp_path)["model_rgb_width"] == 1


# 功能：
#   验证非法尺寸容器以稳定错误拒绝，且完全不调用编码器。
# 输入：
#   size：类型或长度不符合二维尺寸契约的值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("size", [None, 1, {1: 1, 2: 2}, [1], [True, 1]])
def test_invalid_size_container_does_not_reach_decoder(size):
    with pytest.raises(ValueError, match="MODEL_RGB_SIZE_INVALID"):
        ModelImageCache().prepare("source", received_monotonic_seconds=1.,
            received_at_unix_ms=1000, size=size,
            decoder=lambda *a, **k: pytest.fail("invalid size reached decoder"))


# 功能：
#   验证编码器返回值必须为真实字节缓冲，不以整数分配内容，也不将列表隐式转成像素。
# 输入：
#   encoded：待拒绝的 PNG 与 RGB 返回值组合。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("encoded", [(3, b"rgb"), ([1, 2], b"rgb"), (b"png", [1, 2, 3]),
                                    (b"png", None), None, b"ab"])
def test_invalid_decoder_result_never_replaces_valid_cache(encoded):
    cache = ModelImageCache()
    previous, _ = cache.prepare("first", received_monotonic_seconds=1.,
        received_at_unix_ms=1000, size=(1, 1), decoder=lambda *a, **k: (b"png", b"rgb"))
    with pytest.raises(ValueError, match="MODEL_RGB_DECODED_PAYLOAD_INVALID"):
        cache.prepare("bad", received_monotonic_seconds=2., received_at_unix_ms=2000,
                      size=(1, 1), decoder=lambda *a, **k: encoded)
    assert cache._latest is previous


# 功能：
#   验证多字节 memoryview 按实际字节数校验，保存后不再引用可变输出缓冲。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_decoder_buffers_are_owned_and_measured_in_bytes():
    raw = array("I", [0x40404040] * 3)
    png = bytearray(b"png")
    with memoryview(raw) as view:
        image, _ = ModelImageCache().prepare("source", received_monotonic_seconds=1.,
            received_at_unix_ms=1000, size=(2, 2), decoder=lambda *a, **k: (png, view))
    raw[0] = 0
    png[0] = 0
    assert image.png == b"png"
    assert image.rgb == b"\x40" * 12


# 功能：
#   验证编码输出预算在复制前生效，使用长度探针而不分配真实超大图像。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_png_output_budget_is_checked_before_copying():
    class OversizedPng(bytearray):
        # 功能：
        #   报告超过输出预算的逻辑字节长度，不创建对应的大缓冲。
        # 输入：
        #   self：测试字节缓冲。
        # 输出：
        #   size：超过六十四 MiB 的长度。
        def __len__(self):
            size = 64 * 1024 * 1024 + 1
            return size

        # 功能：
        #   检测预算验证之前发生的不允许的字节复制。
        # 输入：
        #   self：测试字节缓冲。
        # 输出：
        #   None：不返回业务数据。
        def __bytes__(self):
            pytest.fail("oversized PNG copied before checking its budget")

    with pytest.raises(ValueError, match="MODEL_RGB_DECODED_PAYLOAD_INVALID"):
        ModelImageCache().prepare("source", received_monotonic_seconds=1.,
            received_at_unix_ms=1000, size=(1, 1),
            decoder=lambda *a, **k: (OversizedPng(b"png"), b"rgb"))
