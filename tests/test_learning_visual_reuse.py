from dataclasses import replace
from pathlib import Path

import pytest
from test_sensor_frame_clock import EPOCH, admit, frame

from dronedream_agent_core.model_image_cache import ModelImageCache
from dronedream_agent_core.model_image_worker import PreparedCameraSample
from dronedream_agent_core.sensor_frame_clock import SensorFrameClock
from scripts import runtime_depth_safety_worker as worker


# 功能：
#   用真实 Pillow 编码合成像素，构造保留场景来源钟的同源编码样本。
# 输入：
#   无。
# 输出：
#   result：消息、来源钟及其真实 PNG/RGB 编码样本组成的元组。
def _prepared_sample():
    message = frame()
    message.width, message.height, message.step = 32, 32, 96
    message.pixel_format_type = 3
    message.data = b"rgb" * 32 * 32
    clock = admit(SensorFrameClock(expected_scene_epoch=EPOCH), message)
    prepared, _ = ModelImageCache().prepare(
        message, received_monotonic_seconds=clock.sample_monotonic_seconds,
        received_at_unix_ms=1000, size=(32, 32), frame_time=clock,
        decoder=worker._gazebo_image_model_payload)
    result = message, clock, PreparedCameraSample(message, prepared)
    return result


# 功能：
#   验证学习记录复用实际编码、只保存一次，并保留来源时钟和真实文件摘要。
# 输入：
#   tmp_path：本测试独占的图像目录。
#   monkeypatch：使不必要的再次编码或写入显式失败的工具。
# 输出：
#   None：重复转换、重复写入或来源变化时测试失败。
def test_learning_reuses_same_prepared_pixels_without_reencoding(tmp_path, monkeypatch):
    message, clock, sample = _prepared_sample()

    # 功能：
    #   拒绝本用例中不应发生的再次编码或重复保存。
    # 输入：
    #   args、kwargs：被禁止操作原本会收到的参数。
    # 输出：
    #   无：被调用时使测试失败。
    def unexpected_work(*args, **kwargs):
        pytest.fail("identical prepared image should not be encoded or persisted twice")

    monkeypatch.setattr(worker, "_gazebo_image_model_payload", unexpected_work)
    arguments = dict(cache=ModelImageCache(), image=message,
        received_monotonic=clock.sample_monotonic_seconds, received_utc_ms=1000,
        size=(32, 32), directory=tmp_path, frame_time=clock, prepared_sample=sample)
    first, = worker._prepare_learning_visual(**arguments)
    assert Path(first["path"]).read_bytes() == sample.image.png
    assert first["sha256"] == sample.image.png_sha256
    assert first["model_rgb_sha256"] == sample.image.rgb_sha256
    assert first["observed_at_unix_ms"] == 1000
    assert first["scene_sequence"] == clock.sequence
    assert first["host_received_unix_ns"] == clock.received_unix_ns
    monkeypatch.setattr(worker, "_atomic_bytes", unexpected_work)
    assert worker._prepare_learning_visual(**arguments) == [first]


# 功能：
#   验证尺寸不同只能按需求重新编码，不能把现有低分辨率编码伪装成另一尺寸。
# 输入：
#   tmp_path：保存当前重新编码像素的独占目录。
# 输出：
#   None：像素尺寸或摘要被不当复用时测试失败。
def test_learning_different_dimensions_reencode_original_pixels(tmp_path):
    from PIL import Image

    message, clock, sample = _prepared_sample()
    record, = worker._prepare_learning_visual(cache=ModelImageCache(), image=message,
        received_monotonic=clock.sample_monotonic_seconds, received_utc_ms=1000,
        size=(16, 16), directory=tmp_path, frame_time=clock, prepared_sample=sample)
    with Image.open(record["path"]) as png:
        assert png.size == (16, 16)
    assert record["sha256"] != sample.image.png_sha256
    assert record["model_rgb_sha256"] != sample.image.rgb_sha256


# 功能：
#   验证不可用其他消息、改写来源时钟或丢失场景绑定来复用已编码图像。
# 输入：
#   tmp_path：不应产生任何图像的独占目录。
#   mismatch：待构造的来源不一致种类。
# 输出：
#   None：不一致被接受或提前写入文件时测试失败。
@pytest.mark.parametrize("mismatch", ["message", "wall", "monotonic", "clock", "type"])
def test_learning_rejects_prepared_source_mismatch(tmp_path, mismatch):
    message, clock, sample = _prepared_sample()
    if mismatch == "message":
        sample = replace(sample, message=frame())
    elif mismatch == "wall":
        sample = replace(sample, image=replace(sample.image, received_at_unix_ms=1001))
    elif mismatch == "monotonic":
        sample = replace(sample, image=replace(sample.image, received_monotonic_seconds=99.))
    elif mismatch == "clock":
        sample = replace(sample, image=replace(sample.image, frame_time=None))
    else:
        sample = {"image": sample.image}
    with pytest.raises(ValueError, match="SOURCE_MISMATCH"):
        worker._prepare_learning_visual(cache=ModelImageCache(), image=message,
            received_monotonic=clock.sample_monotonic_seconds, received_utc_ms=1000,
            size=(32, 32), directory=tmp_path, frame_time=clock, prepared_sample=sample)
    assert not list(tmp_path.iterdir())
