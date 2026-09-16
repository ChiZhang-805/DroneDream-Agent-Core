import threading
from dataclasses import replace

import pytest
from test_forward_rgb_binding import sample
from test_model_image_worker import wait_sample

from dronedream_agent_core.forward_rgb_binding import ForwardRgbBinding
from dronedream_agent_core.model_image_worker import LatestModelImageWorker
from dronedream_agent_core.rgb_input_quality import prepare_rgb_measurement
from dronedream_agent_core.runtime_sensor_contracts import RuntimeSensorRegistry


# 功能：
#   验证后台准备的原始质量与同步测量完全一致，控制线程不再次进行全图质量计算。
# 输入：
#   monkeypatch：把控制线程的解码路径替换成明确失败，以捕获重复计算。
# 输出：
#   None：不返回业务数据。
def test_prepared_quality_matches_original_without_control_thread_pixel_analysis(monkeypatch):
    frame, args = sample()
    direct = ForwardRgbBinding(RuntimeSensorRegistry(), vehicle_id="quad")
    direct.register(frame, **args)
    prepared = prepare_rgb_measurement(frame)
    monkeypatch.setattr("dronedream_agent_core.forward_rgb_binding.decode_gazebo_rgb",
                        lambda _: pytest.fail("pixel decoding ran in control thread"))
    registry = RuntimeSensorRegistry()
    binding = ForwardRgbBinding(registry, vehicle_id="quad")
    assert binding.register(frame, **args, prepared_measurement=prepared)
    assert binding.last_quality == direct.last_quality
    assert registry.latest("oakd-lite-forward-rgb").sample_monotonic_seconds == 10.
    assert not binding.register(frame, **args, prepared_measurement=prepared)


# 功能：
#   拒绝其他像素、布局或被篡改摘要的预处理结果，不能因后台计算而失去同源绑定。
# 输入：
#   kind：本次伪造的不匹配字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["pixels", "layout", "digest"])
def test_prepared_quality_rejects_wrong_source(kind):
    frame, args = sample()
    prepared = prepare_rgb_measurement(frame)
    if kind == "pixels":
        frame.data = b"\xff" * len(frame.data)
    elif kind == "layout":
        frame.width, frame.height, frame.step = 16, 64, 48
    else:
        prepared = replace(prepared, sha256="0" * 64)
    binding = ForwardRgbBinding(RuntimeSensorRegistry(), vehicle_id="quad")
    with pytest.raises(ValueError, match="PREPARATION_SOURCE_MISMATCH"):
        binding.register(frame, **args, prepared_measurement=prepared)
    assert binding.last_quality is None


# 功能：
#   真实图像工作器在后台测量原像素，结果保留来源时间，不使用缩小图像冒充原始质量。
# 输入：
#   monkeypatch：记录质量计算发生的线程身份。
# 输出：
#   None：不返回业务数据。
def test_image_worker_measures_full_source_in_background(monkeypatch):
    import dronedream_agent_core.model_image_worker as module

    threads = []
    original = module.prepare_rgb_measurement

    # 功能：
    #   记录测量线程并执行真实原始像素测量。
    # 输入：
    #   frame：当前工作器独占的源图像。
    # 输出：
    #   measured：真实全图质量结果。
    def measured(frame):
        threads.append(threading.get_ident())
        measured = original(frame)
        return measured

    monkeypatch.setattr(module, "prepare_rgb_measurement", measured)
    worker = LatestModelImageWorker(size=(1, 1), prepare_sensor_quality=True,
                                   decoder=lambda *a, **kw: (b"png", b"rgb"))
    frame, args = sample()
    try:
        worker.submit(frame, received_monotonic_seconds=10., received_at_unix_ms=10000,
                      frame_time=args["frame_time"])
        prepared = wait_sample(worker, 10.)
        assert prepared.measurement.snapshot.width == 32
        assert prepared.image.size == (1, 1)
        assert threads and threads[0] != threading.get_ident()
        assert prepared.image.received_at_unix_ms == 10000
    finally:
        assert worker.close()["failed"] == 0
