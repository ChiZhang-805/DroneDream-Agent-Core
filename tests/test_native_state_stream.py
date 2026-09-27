import json
import time

import pytest
from test_localization_evidence import covariance
from test_native_flight_state import _identity

from dronedream_agent_core.native_flight_state import native_flight_state_sample
from dronedream_agent_core.native_state_stream import NativeStateBuffer, NativeStateSampler
from dronedream_agent_core.source_clock import source_received_at_unix_ms


# 功能：
#   创建带独立位置、姿态、IMU 和里程计接收时间的原生状态夹具。
# 输入：
#   timestamp：统一初始接收时刻，单位为 UNIX 毫秒。
# 输出：
#   value：包含有效协方差及各来源身份的独立遥测字典。
def packet(timestamp=1000):
    value = _identity(timestamp)
    value["dynamics"]["sources"]["odometry"] = {
        "frame_id": "ESTIM_NED", "timestamp_us": timestamp * 1000,
        "sample_age_seconds": 0., "pose_covariance_upper_m2": covariance(),
    }
    for source in value["dynamics"]["sources"].values():
        source["received_at_unix_ms"] = timestamp
        source["timestamp_us"] = timestamp * 1000
        source["sample_age_seconds"] = 0.
    return value


# 功能：验证对齐图像时保留的是匹配历史包的协方差及原时钟，而非最新包或图像时间。
# 输入：无。
# 输出：无；检查快照隔离、坐标系和不同来源期限。
def test_alignment_retains_separate_odometry_source_without_retimestamping():
    buffer = NativeStateBuffer()
    first = packet(1000)
    first["dynamics"]["sources"]["odometry"].update(
        child_frame_id="BODY_FRD", twist_covariance_upper=covariance(),
        velocity_child_frame_m_s=[1., 2., 3.])
    buffer.ingest(first, now_unix_ms=1000)
    buffer.ingest(packet(1080), now_unix_ms=1080)
    aligned = buffer.align_image(image_received_at_unix_ms=1010, now_unix_ms=1090,
        maximum_range_m=10., maximum_acceleration_mps2=1.)
    snapshot = aligned.native_odometry_snapshot
    assert snapshot["image_synchronized"] is False
    assert snapshot["observation_available_at_unix_ms"] == 1000
    assert snapshot["odometry"]["received_at_unix_ms"] == 1000
    assert snapshot["odometry"]["child_frame_id"] == "BODY_FRD"
    assert snapshot["odometry"]["velocity_child_frame_m_s"] == [1., 2., 3.]
    snapshot["odometry"]["twist_covariance_upper"][0] = 999
    again = buffer.align_image(image_received_at_unix_ms=1010, now_unix_ms=1090,
        maximum_range_m=10., maximum_acceleration_mps2=1.)
    assert again.native_odometry_snapshot["odometry"]["twist_covariance_upper"][0] != 999


# 功能：估计器重置不一定重置 IMU 钟；必须丢弃旧图像匹配和控制历史，并重新预热。
# 输入：无。
# 输出：无；检查历史长度、重置次数及重置前图像拒绝。
def test_odometry_reset_rewarms_control_and_rejects_pre_reset_images():
    buffer = NativeStateBuffer()
    for timestamp, reset in ((1000, 255), (1040, 255), (1080, 0)):
        current = packet(timestamp)
        current["dynamics"]["sources"]["odometry"].update(
            reset_counter=reset, estimator_type=8, child_frame_id="LOCAL_NED")
        buffer.ingest(current, now_unix_ms=timestamp)
    assert buffer.history_summary()["odometry_reset_count"] == 1
    assert len(buffer._encoder._samples) == 1
    assert len(buffer._history) == 1
    assert buffer.select_image_time((1060,), now_unix_ms=1090) is None
    with pytest.raises(ValueError, match="RESET_BOUNDARY"):
        buffer.align_image(image_received_at_unix_ms=1060, now_unix_ms=1090,
                           maximum_range_m=10., maximum_acceleration_mps2=1.)
    with pytest.raises(ValueError, match="RESET_EVIDENCE_LOST"):
        buffer.ingest(packet(1120), now_unix_ms=1120)


# 功能：不允许布尔或越界重置次数被解释为新的定位段。
# 输入：counter：畸形重置计数。
# 输出：无；不接入该状态。
@pytest.mark.parametrize("counter", [True, -1, 256, 1.5])
def test_odometry_reset_counter_requires_wire_integer(counter):
    value = packet()
    value["dynamics"]["sources"]["odometry"]["reset_counter"] = counter
    with pytest.raises(ValueError, match="RESET_COUNTER_INVALID"):
        NativeStateBuffer().ingest(value, now_unix_ms=1000)


# 功能：重置消息先到而位置/姿态仍旧时不能生成混合状态；恢复不会重复计算一次重置。
# 输入：无；各遥测来源的独立接收时刻按实际异步形态设置。
# 输出：无；等待所有原生来源越过边界后重新预热。
def test_reset_waits_for_coherent_post_reset_state():
    state = NativeStateBuffer()
    first = packet(1000)
    first["dynamics"]["sources"]["odometry"]["reset_counter"] = 1
    state.ingest(first, now_unix_ms=1000)
    mixed = packet(1080)
    mixed["dynamics"]["sources"]["odometry"]["reset_counter"] = 2
    mixed["position_received_at_unix_ms"] = 1060
    mixed["dynamics"]["sources"]["attitude"]["received_at_unix_ms"] = 1060
    with pytest.raises(ValueError, match="WAITING_FOR_COHERENT_STATE"):
        state.ingest(mixed, now_unix_ms=1080)
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        state.latest(now_unix_ms=1080)
    assert state.history_summary()["odometry_reset_count"] == 1
    assert not state._history
    current = packet(1100)
    current["dynamics"]["sources"]["odometry"]["reset_counter"] = 2
    state.ingest(current, now_unix_ms=1100)
    assert state.history_summary()["odometry_reset_count"] == 1
    assert len(state._encoder._samples) == 1


# 功能：等待异步来源追上重置时若再次重置，必须推进边界，不允许混入中间段状态。
# 输入：无；连续两个重置发生在旧来源尚未追上的时间段。
# 输出：无；中间段仍拒绝，最终新段仅生成一条编码历史。
def test_second_reset_while_waiting_advances_coherence_boundary():
    state = NativeStateBuffer()
    first = packet(1000)
    first["dynamics"]["sources"]["odometry"]["reset_counter"] = 1
    state.ingest(first, now_unix_ms=1000)
    for timestamp, counter, position_time in ((1080, 2, 1060), (1120, 3, 1100)):
        mixed = packet(timestamp)
        mixed["dynamics"]["sources"]["odometry"]["reset_counter"] = counter
        mixed["position_received_at_unix_ms"] = position_time
        with pytest.raises(ValueError, match="WAITING_FOR_COHERENT_STATE"):
            state.ingest(mixed, now_unix_ms=timestamp)
    assert state.history_summary()["odometry_reset_count"] == 2
    assert state._pending_odometry_reset_receive_ms == 1120
    current = packet(1140)
    current["dynamics"]["sources"]["odometry"]["reset_counter"] = 3
    state.ingest(current, now_unix_ms=1140)
    assert len(state._encoder._samples) == 1


# 功能：验证实际缓冲入口保留源时钟配对，不把原生控制编码或独立协方差改成理想值。
# 输入：monkeypatch：仅固定诊断消费单调时刻，不替换原生测量或任何校验。
# 输出：无；同段可选图像成功，错误时钟域不可回退。
def test_source_clock_buffer_integration_and_selection(monkeypatch):
    from test_native_image_pose import payload as source_payload

    state = NativeStateBuffer(source_clock_domain="same-run")
    for timestamp in (1000, 1020):
        current = packet(timestamp)
        wire = source_payload(timestamp * 1000, timestamp / 100., timestamp)
        current["dynamics"]["sources"]["odometry"].update(
            wire["dynamics"]["sources"]["odometry"])
        monkeypatch.setattr(time, "monotonic", lambda stamp=timestamp: stamp / 100. + .01)
        state.ingest(current, now_unix_ms=timestamp)
    limits = dict(clock_domain="same-run", now_unix_ms=1160, now_monotonic=10.23,
                  maximum_range_m=10., maximum_acceleration_mps2=1.)
    assert state.select_source_image_time(((1150, 1_001_000_000),),
        maximum_alignment_variance_m2=10., **limits) == 1150
    result = state.align_source_image(image_timestamp_ns=1_001_000_000,
        image_received_at_unix_ms=1150, **limits)
    assert result.pose.observed_at_unix_ms == 1000
    assert result.native_odometry_snapshot["source_alignment"]["source_skew_us"] == -1000
    assert result.native_odometry_snapshot["image_synchronized"] is False
    selected = state.select_source_image_pose(((1150, 1_001_000_000),),
        maximum_alignment_variance_m2=10., **limits)
    assert selected[:2] == (1150, 1_001_000_000)
    assert selected[2].native_odometry_snapshot == result.native_odometry_snapshot
    with pytest.raises(ValueError, match="CONTRACT"):
        state.select_source_image_pose(((1150, 1_001_000_000),),
            maximum_alignment_variance_m2=10., **{**limits, "clock_domain": "wrong"})
    assert state.select_source_image_time(((1150, 1_001_000_000),),
        maximum_alignment_variance_m2=1e-9, **limits) is None
    with pytest.raises(ValueError, match="CONTRACT"):
        state.align_source_image(image_timestamp_ns=1_001_000_000,
            image_received_at_unix_ms=1150, **{**limits, "clock_domain": "wrong"})


# 功能：
#   时间上无关的历史不重复解析编码，仍对每个可能匹配的候选执行原完整性校验。
# 输入：
#   monkeypatch：只记录真实编码校验入口的调用，不替换校验结果。
# 输出：
#   无。
def test_alignment_filters_unrelated_history_before_encoding_validation(monkeypatch):
    from dronedream_agent_core.realtime_feature_encoders import RealtimeFeatureEncoding

    buffer = NativeStateBuffer()
    for stamp in range(1000, 1640, 10):
        buffer.ingest(packet(stamp), now_unix_ms=stamp)
    original = RealtimeFeatureEncoding.fresh_at
    seen = []

    # 功能：
    #   记录被验证编码的真实来源，继续调用原时效及结构检查。
    # 输入：
    #   self：编码；now：消费时刻。
    # 输出：
    #   fresh：原校验结果。
    def recorded(self, now):
        seen.append(self.observed_at_unix_ms)
        fresh = original(self, now)
        return fresh

    monkeypatch.setattr(RealtimeFeatureEncoding, "fresh_at", recorded)
    assert buffer.select_image_time((1630,), now_unix_ms=1640) == 1630
    assert seen == list(range(1580, 1640, 10))
    seen.clear()
    buffer.align_image(image_received_at_unix_ms=1630, now_unix_ms=1640,
                       maximum_range_m=10., maximum_acceleration_mps2=1.)
    assert seen == list(range(1580, 1640, 10))
    # 即便时间匹配，篡改的特征仍不能被快速路径接受。
    for observation in buffer._history:
        observation.encoding.features[0] = float("nan")
    assert buffer.select_image_time((1630,), now_unix_ms=1640) is None


# 功能：
#   控制参考只可使用真实新状态；重复获取不续龄，未来状态不能进入过去参考时刻。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_control_refresh_uses_real_arrivals_without_renewing_old_samples():
    buffer = NativeStateBuffer()
    buffer.ingest(packet(1000), now_unix_ms=1000)
    previous = buffer.latest(now_unix_ms=1010)
    same = buffer.latest_after(previous, now_unix_ms=1100)
    assert same.sample.observed_at_unix_ms == 1000
    assert same.encoding.observed_at_unix_ms == 1000
    buffer.ingest(packet(1150), now_unix_ms=1160)
    before_arrival = buffer.latest_after(previous, now_unix_ms=1140)
    assert before_arrival.sample.observed_at_unix_ms == 1000
    current = buffer.latest_after(previous, now_unix_ms=1170)
    assert current.sample.observed_at_unix_ms == 1150
    assert previous.sample.observed_at_unix_ms == 1000
    assert current.pose.binding_sha256 == previous.pose.binding_sha256
    with pytest.raises(ValueError, match='EXPIRED'):
        buffer.latest_after(current, now_unix_ms=1500)


# 功能：
#   拒绝另一个部署绑定或后来参考的回退，不允许控制刷新跨任务借状态。
# 输入：
#   change：需要模拟的参考损坏。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize('change', ['binding', 'clock', 'type'])
def test_control_refresh_rejects_changed_reference(change):
    from dataclasses import replace

    buffer = NativeStateBuffer()
    buffer.ingest(packet(1000), now_unix_ms=1000)
    previous = buffer.latest(now_unix_ms=1010)
    if change == 'binding':
        previous = replace(previous, pose=replace(previous.pose, binding_sha256='f'*64))
    elif change == 'clock':
        previous = replace(previous, sample=previous.sample.model_copy(
            update={'observed_at_unix_ms': 1100}))
    else:
        previous = None
    with pytest.raises(ValueError, match='NATIVE_CONTROL_'):
        buffer.latest_after(previous, now_unix_ms=1050)


# 功能：
#   缓存发布时间和仿真真值变化不能制造新的原生传感器身份或延长年龄。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_republication_metadata_is_not_a_new_sensor_identity():
    data = packet()
    first = native_flight_state_sample(data, now_unix_ms=1000)
    data["updated_at_unix_ms"] = 1010
    data["dynamics"]["collected_at_unix_ms"] = 1010
    for source in data["dynamics"]["sources"].values():
        source["sample_age_seconds"] = .01
    data["observed_world_collision_center_m"] = {"x": 100, "y": -8, "z": 6}
    republished = native_flight_state_sample(data, now_unix_ms=1010)
    assert first.source_sample_sha256 == republished.source_sample_sha256
    assert republished.observed_at_unix_ms == 1000


# 功能：
#   确认候选计量和深复制在生产者锁外进行，且筛选阶段不复制完整状态。
# 输入：
#   monkeypatch：在原测量和复制函数外包裹断言的测试工具。
# 输出：
#   None：不返回业务数据。
def test_pose_selection_and_detached_copy_do_not_hold_the_producer_lock(monkeypatch):
    from dronedream_agent_core import native_state_stream as module

    buffer = NativeStateBuffer()
    for now in range(1000, 1101, 20):
        buffer.ingest(packet(now), now_unix_ms=now)
    measured, copied = [], []
    original_measure, original_copy = module._alignment_measurement, module.copy.deepcopy
    # 功能：
    #   记录对齐测量次数，并断言测量没有持有生产者锁。
    # 输入：
    #   args：传给原测量函数的位置参数。
    #   kwargs：传给原测量函数的关键字参数。
    # 输出：
    #   result：原测量函数返回的时差及方差。
    def measure(*args, **kwargs):
        assert not buffer._lock.locked()
        measured.append(1)
        result = original_measure(*args, **kwargs)
        return result
    # 功能：
    #   记录深复制次数，并断言复制没有阻塞生产者锁。
    # 输入：
    #   value：准备复制的状态对象。
    #   args：附加位置参数。
    #   kwargs：附加关键字参数。
    # 输出：
    #   result：原深复制函数生成的独立对象。
    def detached(value, *args, **kwargs):
        assert not buffer._lock.locked()
        copied.append(1)
        result = original_copy(value, *args, **kwargs)
        return result
    monkeypatch.setattr(module, "_alignment_measurement", measure)
    monkeypatch.setattr(module.copy, "deepcopy", detached)
    assert buffer.select_image_time((1095,), now_unix_ms=1100, maximum_range_m=10.,
        maximum_acceleration_mps2=1., maximum_alignment_variance_m2=10.) == 1095
    assert len(measured) == 1 and not copied
    aligned = buffer.align_image(image_received_at_unix_ms=1095, now_unix_ms=1100,
                                 maximum_range_m=10., maximum_acceleration_mps2=1.)
    assert aligned.receive_skew_ms == 5 and len(copied) == 1
    buffer.latest(now_unix_ms=1100)
    assert len(copied) == 2


# 功能：
#   报告年龄被写成零也不能让显式旧接收时刻重新有效。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cached_receive_time_does_not_become_fresh_when_reported_age_is_zero():
    with pytest.raises(ValueError, match="EXPIRED"):
        source_received_at_unix_ms(
            {"sample_age_seconds": 0, "received_at_unix_ms": 1000},
            collected_at_unix_ms=1400, now_unix_ms=1400, maximum_age_ms=250,
        )


# 功能：
#   状态历史独立于图像和模型调用推进；重复包不增加样本，返回副本不污染缓冲。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_history_advances_without_any_image_or_model_call():
    buffer = NativeStateBuffer()
    for index in range(16):
        now = 1000 + index * 20
        buffer.ingest(packet(now), now_unix_ms=now)
    state = buffer.latest(now_unix_ms=1300)
    assert "FLIGHT_STATE_HISTORY_WARMING" not in state.encoding.issue_codes
    before = len(buffer._history)
    for _ in range(20):
        buffer.ingest(packet(1300), now_unix_ms=1300)
    assert len(buffer._history) == before
    assert buffer.latest(now_unix_ms=1300).encoding == state.encoding
    state.payload["position_received_at_unix_ms"] = 9999
    assert buffer.latest(now_unix_ms=1300).payload["position_received_at_unix_ms"] == 1300


# 功能：
#   图像配对使用接收时间相近的历史位姿，不总是使用最新位姿。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_image_gets_matching_native_pose_not_the_newest_pose():
    buffer = NativeStateBuffer()
    first, second = packet(1000), packet(1100)
    second["observed_position_ned_m"]["east_m"] = 1.
    buffer.ingest(first, now_unix_ms=1000)
    buffer.ingest(second, now_unix_ms=1100)
    aligned = buffer.align_image(image_received_at_unix_ms=1005, now_unix_ms=1100,
                                 maximum_range_m=10, maximum_acceleration_mps2=1)
    assert aligned.pose.position_world_enu_m.x == 0
    assert aligned.receive_skew_ms == 5
    assert aligned.conservative_variance_m2 > .03
    with pytest.raises(ValueError, match="NOT_SYNCHRONIZED"):
        buffer.align_image(image_received_at_unix_ms=1160, now_unix_ms=1160,
                           maximum_range_m=10, maximum_acceleration_mps2=1)


# 功能：
#   控制周期不能消费捕获周期时间之后才收到的状态；来源失效后不回退历史。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_consumer_tick_cannot_observe_a_concurrently_newer_native_packet():
    buffer = NativeStateBuffer()
    buffer.ingest(packet(1000), now_unix_ms=1000)
    buffer.ingest(packet(1020), now_unix_ms=1020)
    assert buffer.latest(now_unix_ms=1010).pose.observed_at_unix_ms == 1000
    assert buffer.latest(now_unix_ms=1020).pose.observed_at_unix_ms == 1020
    with pytest.raises(ValueError, match="NOT_YET_AVAILABLE"):
        buffer.latest(now_unix_ms=999)
    buffer.invalidate("explicit failure")
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        buffer.latest(now_unix_ms=1010)


# 功能：
#   图像配对也必须遵守控制周期可见性，不借用后来到达的位姿。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_alignment_does_not_include_a_pose_arriving_after_the_tick_clock():
    buffer = NativeStateBuffer()
    buffer.ingest(packet(1000), now_unix_ms=1000)
    buffer.ingest(packet(1100), now_unix_ms=1100)
    assert buffer.select_image_time((1090,), now_unix_ms=1095) is None
    with pytest.raises(ValueError, match="NOT_SYNCHRONIZED"):
        buffer.align_image(image_received_at_unix_ms=1090, now_unix_ms=1095,
                           maximum_range_m=10, maximum_acceleration_mps2=1)


# 功能：
#   未知协方差或明确失效状态不能借用旧好样本，新的有效观测允许恢复。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_missing_covariance_and_invalid_stream_do_not_reuse_good_history():
    buffer = NativeStateBuffer()
    with pytest.raises(ValueError, match="COVARIANCE_UNAVAILABLE"):
        buffer.ingest(_identity(), now_unix_ms=1000)
    buffer.ingest(packet(), now_unix_ms=1000)
    buffer.invalidate("telemetry missing")
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        buffer.latest(now_unix_ms=1005)
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        buffer.align_image(image_received_at_unix_ms=1000, now_unix_ms=1005,
                           maximum_range_m=10, maximum_acceleration_mps2=1)
    buffer.ingest(packet(1010), now_unix_ms=1010)
    assert buffer.latest(now_unix_ms=1010)


# 功能：
#   最新帧尚未同步时选择更早的有效配对，未来或过期图像不参与。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_newest_pairable_image_avoids_asynchronous_latest_frame_starvation():
    buffer = NativeStateBuffer()
    buffer.ingest(packet(1000), now_unix_ms=1000)
    buffer.ingest(packet(1100), now_unix_ms=1100)
    assert buffer.select_image_time((1005, 1105, 1180), now_unix_ms=1190) == 1105
    assert buffer.select_image_time((1180,), now_unix_ms=1190) is None
    assert buffer.select_image_time((900, 950, 1000), now_unix_ms=1251) is None
    assert buffer.select_image_time((1100,), now_unix_ms=1099) is None
    buffer.invalidate("source expired")
    assert buffer.select_image_time((1105,), now_unix_ms=1110) is None


# 功能：
#   图像必须同时满足位置与姿态时差，候选数量及时间类型也受约束。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_image_pair_needs_both_position_and_attitude_not_only_nearest_one():
    buffer = NativeStateBuffer()
    payload = packet(1100)
    payload["position_received_at_unix_ms"] = 1050
    buffer.ingest(payload, now_unix_ms=1100)
    assert buffer.select_image_time((1110, 1075), now_unix_ms=1120) == 1075
    for invalid in ((True,), (1.0,), tuple(range(33))):
        with pytest.raises(ValueError, match="CANDIDATE_TIMES_INVALID"):
            buffer.select_image_time(invalid, now_unix_ms=1100)


# 功能：
#   高角速度时按不确定性筛选图像，不能为了选到新帧而放宽预算。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_turning_selects_fresh_pair_inside_unchanged_uncertainty_budget():
    buffer = NativeStateBuffer()
    for timestamp in (1000, 1100):
        payload = packet(timestamp)
        payload["dynamics"]["sources"]["imu"]["angular_velocity_down_rad_s"] = 2.
        buffer.ingest(payload, now_unix_ms=timestamp)
    limits = {"maximum_range_m": 20., "maximum_acceleration_mps2": 1.,
              "maximum_alignment_variance_m2": .25}
    assert buffer.select_image_time((1005, 1070), now_unix_ms=1110, **limits) == 1005
    assert buffer.select_image_time((1070,), now_unix_ms=1110, **limits) is None
    assert buffer.select_image_time((1005,), now_unix_ms=1300, **limits) is None
    with pytest.raises(ValueError, match="LIMIT_INVALID"):
        buffer.select_image_time((1005,), now_unix_ms=1110, maximum_range_m=20.)


# 功能：
#   用真实临时文件验证独立采样线程读取来源，关闭后线程确实结束。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_sampler_is_bounded_and_stops_after_reading_actual_file(tmp_path):
    path = tmp_path / "native-state.json"
    now = int(time.time() * 1000)
    path.write_text(json.dumps(packet(now)), encoding="utf-8")
    sampler = NativeStateSampler(path)
    sampler.start()
    try:
        for _ in range(40):
            try:
                state = sampler.buffer.latest(now_unix_ms=int(time.time() * 1000))
                break
            except ValueError:
                time.sleep(.005)
        else:
            raise AssertionError("independent sampler did not read native input")
        assert state.pose.observed_at_unix_ms == now
        assert len(sampler.buffer._history) == 1
    finally:
        sampler.close()
    assert not sampler._thread.is_alive()


# 功能：
#   即使图像与位置时间接近，已过期的 IMU 或协方差也不能参与配对。
# 输入：
#   source：被保持为旧观测的传感器来源名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("source", ["imu", "odometry"])
def test_image_pair_requires_all_native_sources_to_remain_fresh(source):
    buffer = NativeStateBuffer()
    payload = packet(1200)
    payload["dynamics"]["sources"][source]["received_at_unix_ms"] = 1180
    payload["dynamics"]["sources"][source]["sample_age_seconds"] = .02
    buffer.ingest(payload, now_unix_ms=1200)
    assert buffer.select_image_time((1230,), now_unix_ms=1450) is None
    with pytest.raises(ValueError, match="NOT_SYNCHRONIZED"):
        buffer.align_image(image_received_at_unix_ms=1230, now_unix_ms=1450,
                           maximum_range_m=10, maximum_acceleration_mps2=1)


# 功能：
#   缓冲与采样器配置只接受契约类型，非法值以可识别错误拒绝。
# 输入：
#   value：错误容量或频率。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [None, "50", 10**400])
def test_sampler_and_buffer_invalid_configuration_is_a_domain_error(value):
    with pytest.raises(ValueError, match="CAPACITY_INVALID"):
        NativeStateBuffer(capacity=value)
    with pytest.raises(ValueError, match="RATE_INVALID"):
        NativeStateSampler(None, rate_hz=value)


# 功能：
#   图像配对不接受布尔、小数或空时刻，即使缓冲为空也先验证调用参数。
# 输入：
#   now：非法消费时刻。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("now", [None, True, 1000.0, 2**63])
def test_empty_buffer_does_not_hide_invalid_consumer_clock(now):
    buffer = NativeStateBuffer()
    with pytest.raises(ValueError, match="TIME_INVALID"):
        buffer.latest(now_unix_ms=now)
    with pytest.raises(ValueError, match="TIME_INVALID"):
        buffer.select_image_time((1000,), now_unix_ms=now)
    with pytest.raises(ValueError, match="TIME_INVALID"):
        buffer.align_image(image_received_at_unix_ms=1000, now_unix_ms=now,
                           maximum_range_m=10, maximum_acceleration_mps2=1)


# 功能：
#   候选图像列表和不确定性预算必须有界且类型明确。
# 输入：
#   value：不能充当有效候选列表或数值预算的输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [None, True, "1000", 10**400])
def test_alignment_candidates_and_limits_fail_with_domain_errors(value):
    buffer = NativeStateBuffer()
    with pytest.raises(ValueError, match="CANDIDATE_TIMES_INVALID"):
        buffer.select_image_time(value, now_unix_ms=1000)
    with pytest.raises(ValueError, match="LIMIT_INVALID"):
        buffer.align_image(image_received_at_unix_ms=1000, now_unix_ms=1000,
                           maximum_range_m=value, maximum_acceleration_mps2=1)


# 功能：
#   验证文件采样入口拒绝重复 JSON 键，不能按解析器最后一个值继续发布状态。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：将循环等待替换为单次结束的测试工具。
# 输出：
#   None：不返回业务数据。
def test_sampler_rejects_duplicate_json_keys(tmp_path, monkeypatch):
    path = tmp_path / "native-duplicate.json"
    now = int(time.time() * 1000)
    raw = json.dumps(packet(now))
    path.write_text(raw[:-1] + ', "updated_at_unix_ms": ' + str(now) + '}', encoding="utf-8")
    sampler = NativeStateSampler(path)

    # 功能：
    #   在第一次轮询结束后退出同步测试循环，不启动后台线程。
    # 输入：
    #   timeout：生产循环请求的等待秒数。
    # 输出：
    #   stopped：确认停止标记已设置的布尔值。
    def finish_wait(timeout):
        assert sampler.buffer._issue == "ValueError"
        assert sampler.buffer._latest is None
        sampler._stop.set()
        stopped = True
        return stopped

    monkeypatch.setattr(sampler._stop, "wait", finish_wait)
    sampler._run()
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        sampler.buffer.latest(now_unix_ms=now)
    sampler.close()


# 功能：
#   极大但有限的运动与量程导致不可表示的对齐方差时明确拒绝，不返回 Infinity。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_alignment_never_returns_nonfinite_uncertainty():
    buffer = NativeStateBuffer()
    buffer.ingest(packet(), now_unix_ms=1000)
    with pytest.raises(ValueError, match="ALIGNMENT_UNCERTAINTY_INVALID"):
        buffer.align_image(image_received_at_unix_ms=1050, now_unix_ms=1050,
                           maximum_range_m=1e308, maximum_acceleration_mps2=1e308)


# 功能：
#   非普通异常结束循环时撤销可用状态，保留原异常，不将其冒充可恢复输入错误。
# 输入：
#   tmp_path：隔离输入目录。
#   monkeypatch：注入不可恢复读取错误的测试工具。
# 输出：
#   None：不返回业务数据。
def test_terminal_sampler_error_invalidates_cached_state(tmp_path, monkeypatch):
    from dronedream_agent_core import native_state_stream as module

    sampler = NativeStateSampler(tmp_path / "native.json")
    sampler.buffer.ingest(packet(), now_unix_ms=1000)

    # 功能：
    #   模拟底层读取器发生不可恢复运行错误。
    # 输入：
    #   path：生产代码准备读取的路径。
    #   limit：生产代码设置的字节预算。
    # 输出：
    #   None：不返回业务数据。
    def fail_read(path, *, limit):
        raise RuntimeError("terminal fixture")

    monkeypatch.setattr(module, "read_plugin_file", fail_read)
    with pytest.raises(RuntimeError, match="terminal fixture"):
        sampler._run()
    with pytest.raises(ValueError, match="UNAVAILABLE:SAMPLER_STOPPED"):
        sampler.buffer.latest(now_unix_ms=1000)
    sampler.close()


# 功能：
#   关闭尚未启动的采样器后不可再次启动，也不能继续消费预置状态。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_close_before_start_is_terminal_and_repeatable(tmp_path):
    sampler = NativeStateSampler(tmp_path / "native.json")
    sampler.buffer.ingest(packet(), now_unix_ms=1000)
    sampler.close()
    sampler.close()
    with pytest.raises(ValueError, match="UNAVAILABLE:SAMPLER_STOPPED"):
        sampler.buffer.latest(now_unix_ms=1000)
    with pytest.raises(RuntimeError, match="SAMPLER_CLOSED"):
        sampler.start()


# 功能：
#   直接接入也必须执行 JSON 大小和类型校验，拒绝损坏包时不修改已存状态。
# 输入：
#   value：超预算内容或不属于 JSON 的对象。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", ["x" * 262_144, object()], ids=["oversized", "non-json"])
def test_direct_ingest_has_the_same_bounded_json_contract(value):
    buffer = NativeStateBuffer()
    buffer.ingest(packet(), now_unix_ms=1000)
    payload = packet(1010)
    payload["unrelated"] = value
    with pytest.raises(ValueError, match="JSON"):
        buffer.ingest(payload, now_unix_ms=1010)
    assert buffer.latest(now_unix_ms=1010).pose.observed_at_unix_ms == 1000
