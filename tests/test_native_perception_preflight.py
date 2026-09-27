import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from dronedream_agent_core.native_preflight import (
    NativePerceptionReadiness,
    assert_native_preflight_current,
    wait_for_native_perception,
)


# 功能：
#   生成用于来源新鲜度测试的健康快照，不代表原生飞行已经通过检查。
# 输入：
#   sequence：传感器来源序列号。
#   timestamp：来源 UNIX 毫秒时刻。
# 输出：
#   payload：包含显式布尔状态和固定来源类型的合成健康记录。
def health(sequence=1, timestamp=1000):
    payload = {"schema_version": "dronedream.perception-fusion-health.v1",
            "stream_healthy": True, "identity_accepted": True,
            "truth_correction_applied": False,
            "pose_source": "native-estimator-fixed-deployment-binding",
            "realtime_features_ready": True, "latest_sequence": sequence,
            "updated_at_unix_ms": timestamp}
    return payload


# 功能：
#   重复轮询不增加独立帧数，来源间隔满足才就绪，超过原始期限后清空窗口。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_preflight_requires_distinct_progressing_frames_not_polls():
    gate = NativePerceptionReadiness()
    for _ in range(10):
        assert not gate.observe(health(), now_unix_ms=1000)
    assert not gate.observe(health(2, 1050), now_unix_ms=1050)
    assert gate.observe(health(3, 1100), now_unix_ms=1100)
    assert not gate.observe(health(3, 1100), now_unix_ms=1351)
    assert gate.independent_frames == 0


# 功能：连续窗口成立但来源只剩3毫秒时不发放回执，等待真正更鲜的下一帧。
# 输入：前三帧247毫秒延迟、第四帧100毫秒延迟；输出：原始期限与至少70毫秒交接余量。
def test_preflight_waits_for_real_dispatch_margin(tmp_path, monkeypatch):
    from dronedream_agent_core import native_preflight
    clock, sequence = [1000], [0]
    monkeypatch.setattr(native_preflight, 'time', SimpleNamespace(
        time=lambda: clock[0]/1000, monotonic=lambda: clock[0]/1000))
    def read_latest():
        clock[0] += 70
        sequence[0] += 1
        age = 247 if sequence[0] <= 3 else 100
        return {**health(sequence[0], clock[0]), 'perception_observed_at_unix_ms': clock[0]-age}
    evidence = {}
    result = asyncio.run(wait_for_native_perception(tmp_path/'unused', timeout_seconds=1,
        receiver=SimpleNamespace(read_latest=read_latest), require_source_timestamps=True,
        evidence=evidence))
    assert sequence[0] == 4
    assert result['source_observed_at_unix_ms'] == clock[0] - 100
    assert result['valid_until_unix_ms'] == clock[0] + 150
    assert evidence['issue_counts']['NATIVE_PREFLIGHT_WAITING_FOR_DISPATCH_MARGIN'] == 1


# 功能：真实来源每100毫秒推进，180毫秒处理延迟不能被重复加到帧间隔中。
# 输入：连续独立帧及不变250毫秒时效；输出：完整一秒窗口通过，回执保持原始来源钟。
def test_delayed_but_fresh_frames_keep_the_source_continuity_window():
    gate = NativePerceptionReadiness(stable_window_ms=1000, require_source_timestamps=True)
    for index in range(11):
        source = 1000 + index * 100
        now = source + 180
        packet = {**health(index + 1, now), 'perception_observed_at_unix_ms': source}
        assert gate.observe(packet, now_unix_ms=now) is (index == 10)
    assert gate.independent_frames == 11
    assert gate.source_observed_at_unix_ms == 2000
    assert not gate.observe(packet, now_unix_ms=2251)
    assert gate.independent_frames == 0


# 功能：即使新帧本身新鲜，真实来源断流仍清空稳定窗口。
# 输入：间隔251毫秒的两个来源；输出：拒绝沿用旧计数。
def test_fresh_new_frame_does_not_hide_a_real_source_gap():
    gate = NativePerceptionReadiness(require_source_timestamps=True)
    assert not gate.observe({**health(1, 1180), 'perception_observed_at_unix_ms': 1000},
                            now_unix_ms=1180)
    assert not gate.observe({**health(2, 1300), 'perception_observed_at_unix_ms': 1251},
                            now_unix_ms=1300)
    assert gate.last_issue == 'NATIVE_PREFLIGHT_SOURCE_INTERRUPTED'
    assert gate.independent_frames == 0


# 功能：消费者失联和时钟回退不得凭一张新鲜图片继承之前的就绪窗口。
# 输入：过长消费间隔或回退时钟；输出：清空窗口并保留具体原因。
@pytest.mark.parametrize('now,source,issue', [
    (1351, 1200, 'NATIVE_PREFLIGHT_SOURCE_INTERRUPTED'),
    (1099, 1050, 'NATIVE_PREFLIGHT_SOURCE_REGRESSED'),
])
def test_consumer_gap_or_clock_regression_resets_window(now, source, issue):
    gate = NativePerceptionReadiness(require_source_timestamps=True)
    assert not gate.observe({**health(1, 1050), 'perception_observed_at_unix_ms': 1000},
                            now_unix_ms=1100)
    assert not gate.observe({**health(2, now), 'perception_observed_at_unix_ms': source},
                            now_unix_ms=now)
    assert gate.last_issue == issue
    assert gate.independent_frames == 0


# 功能：
#   同一序列改写发布时间不能扩展独立帧跨度，必须等待新的来源序列。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_republished_health_cannot_extend_the_distinct_frame_span():
    gate = NativePerceptionReadiness()
    for sequence, timestamp in ((1, 1000), (2, 1020), (3, 1040)):
        assert not gate.observe(health(sequence, timestamp), now_unix_ms=timestamp)
    assert not gate.observe(health(3, 1200), now_unix_ms=1200)
    assert gate.observe(health(4, 1220), now_unix_ms=1220)


# 功能：
#   错误健康标志、真值控制、错误序列类型或未来来源时间均不能取得就绪资格。
# 输入：
#   field：被替换的健康字段。
#   value：对应非法或不健康值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [
    ("stream_healthy", False), ("identity_accepted", False), ("truth_correction_applied", True),
    ("pose_source", "simulation-ground-truth"), ("realtime_features_ready", False),
    ("latest_sequence", True), ("updated_at_unix_ms", 1001),
])
def test_unusable_perception_does_not_authorize_arming(field, value):
    gate = NativePerceptionReadiness()
    payload = health()
    payload[field] = value
    assert not gate.observe(payload, now_unix_ms=1000)


# 功能：
#   健康文件缺失时在短期限内失败，不发出任何解锁或运动命令。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_preflight_missing_file_is_bounded_and_never_an_arm_command(tmp_path):
    with pytest.raises(RuntimeError, match="NOT_READY_BEFORE_ARM"):
        asyncio.run(wait_for_native_perception(tmp_path / "missing.json", timeout_seconds=.01))


# 功能：
#   通过真实本地文件读写验证连续进展可产生就绪回执，不使用预设成功返回值。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_preflight_reads_actual_progressing_health_file(tmp_path):
    # 功能：
    #   同时运行有限发布器和检查器，并确保发布器结束。
    # 输入：
    #   无，使用外层测试目录。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        path = tmp_path / "health.json"

        # 功能：
        #   定期发布五份独立来源健康记录供真实读取链路消费。
        # 输入：
        #   无，使用当前场景路径。
        # 输出：
        #   None：不返回业务数据。
        async def publisher():
            for index in range(1, 6):
                path.write_text(json.dumps(health(index, int(time.time() * 1000))))
                await asyncio.sleep(.07)

        writer = asyncio.create_task(publisher())
        try:
            result = await wait_for_native_perception(path, timeout_seconds=1)
            assert result["ready"] and result["independent_frames"] >= 3
        finally:
            await writer

    asyncio.run(scenario())


# 功能：
#   大实测方差使窄通道余量不足；估计改善后仍需重新积累独立帧才能恢复。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_route_budget_uses_measured_covariance_not_three_centimetre_assumption():
    gate = NativePerceptionReadiness(minimum_route_clearance_m=.435)
    # Replay the variance scale observed in the failed native simulation.
    for sequence in range(1, 4):
        timestamp = 1000 + sequence * 60
        packet = {**health(sequence, timestamp), "localization_covariance_m2": .0535,
                  "localization_observed_at_unix_ms": timestamp - 10}
        assert not gate.observe(packet, now_unix_ms=timestamp)
        assert gate.last_issue.startswith("NATIVE_ROUTE_UNCERTAINTY_BUDGET_UNFUNDED")
    assert gate.independent_frames == 0
    # A genuinely better estimator can recover without restarting the process,
    # but still requires distinct, fresh, progressing frames.
    for sequence in range(4, 7):
        timestamp = 1000 + sequence * 60
        packet = {**health(sequence, timestamp), "localization_covariance_m2": .001,
                  "localization_observed_at_unix_ms": timestamp - 10}
        assert gate.observe(packet, now_unix_ms=timestamp) is (sequence == 6)
    assert gate.tracking_budget["reserved_uncertainty_m"] == pytest.approx(3 * .001**.5)
    assert gate.tracking_budget["tracking_lag_limit_m"] < .2


# 功能：
#   缺失、非法或过期协方差不能进入路线余量计算，也不能残留上一次预算。
# 输入：
#   variance：候选实测位置方差。
#   age：定位来源相对消费时刻的毫秒年龄。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("variance,age", [
    (None, 0), (True, 0), ("0.001", 0), (float("nan"), 0), (-1, 0),
    (.001, 251), (.001, -1),
])
def test_route_budget_rejects_unavailable_uncertainty_and_original_stale_time(variance, age):
    gate = NativePerceptionReadiness(minimum_route_clearance_m=.435)
    packet = {**health(), "localization_covariance_m2": variance,
              "localization_observed_at_unix_ms": 1000 - age}
    assert not gate.observe(packet, now_unix_ms=1000)
    assert gate.last_issue is not None
    assert gate.tracking_budget is None


# 功能：
#   用受控来源时钟和真实文件验证方差进入就绪回执，避免 CI 调度延迟伪装为传感器故障。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：只替换本测试的时钟和后台读任务调度，不修改实际读取及就绪判定。
# 输出：
#   None：不返回业务数据。
def test_live_route_budget_is_included_in_actual_file_readiness_receipt(tmp_path, monkeypatch):
    from dronedream_agent_core import native_preflight

    clock = [1000]
    sequence = [0]
    monkeypatch.setattr(native_preflight, "time", SimpleNamespace(
        time=lambda: clock[0] / 1000, monotonic=lambda: clock[0] / 1000))

    # 功能：
    #   在读任务执行时发布下一份来源独立的合成记录，然后调用真实的有界文件读取函数。
    # 输入：
    #   reader：生产代码传入的健康文件读取函数。
    #   path：实际测试健康文件路径。
    # 输出：
    #   payload：实际读取并解析的当前记录。
    async def read_next(reader, path):
        clock[0] += 70
        sequence[0] += 1
        path.write_text(json.dumps({**health(sequence[0], clock[0]),
            "localization_covariance_m2": .001,
            "localization_observed_at_unix_ms": clock[0] - 10}))
        payload = reader(path)
        return payload

    monkeypatch.setattr(native_preflight.asyncio, "to_thread", read_next)
    result = asyncio.run(wait_for_native_perception(tmp_path / "health.json", timeout_seconds=1,
                                                   minimum_route_clearance_m=.435))
    assert result["uncertainty_basis"] == "live-native-localization"
    assert result["tracking_budget"]["reserved_uncertainty_m"] > .09
    assert result["valid_until_unix_ms"] == result["source_observed_at_unix_ms"] + 250
    assert result["independent_frames"] == sequence[0] == 3
    assert_native_preflight_current(result, now_unix_ms=clock[0])


# 功能：
#   健康序列推进但定位来源不变时，不能冒充多次独立定位测量。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_repeated_localization_cannot_become_three_independent_native_samples():
    gate = NativePerceptionReadiness(minimum_route_clearance_m=.435)
    for sequence, timestamp in ((1, 1000), (2, 1060), (3, 1120), (4, 1180)):
        packet = {**health(sequence, timestamp), "localization_covariance_m2": .001,
                  "localization_observed_at_unix_ms": 1000}
        assert not gate.observe(packet, now_unix_ms=timestamp)
    assert gate.independent_frames == 1
    assert gate.source_observed_at_unix_ms == 1000


# 功能：
#   就绪窗口采用其中最差的实测方差，不被最后一帧较乐观的数据覆盖。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_preflight_uses_worst_measured_variance_in_readiness_window():
    gate = NativePerceptionReadiness(minimum_route_clearance_m=.435)
    for sequence, variance in enumerate((.0025, .001, .0001), start=1):
        now = 1000 + sequence * 60
        packet = {**health(sequence, now), "localization_covariance_m2": variance,
                  "localization_observed_at_unix_ms": now}
        assert gate.observe(packet, now_unix_ms=now) is (sequence == 3)
    assert gate.tracking_budget["reserved_uncertainty_m"] == pytest.approx(.15)


# 功能：
#   已就绪包重新发布不能续期，超时中断之后必须重新累计完整窗口。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_republished_ready_packet_cannot_renew_lease_and_gap_requires_requalification():
    gate = NativePerceptionReadiness()
    for sequence, now in ((1, 1000), (2, 1060), (3, 1120)):
        gate.observe(health(sequence, now), now_unix_ms=now)
    assert gate.observe(health(3, 1300), now_unix_ms=1300)
    assert gate.source_observed_at_unix_ms == 1120
    assert not gate.observe(health(3, 1371), now_unix_ms=1371)
    assert gate.independent_frames == 0
    assert not gate.observe(health(4, 1400), now_unix_ms=1400)


# 功能：
#   在解锁前拒绝过期、未来、错误类型或被延长的回执期限。
# 输入：
#   changes：注入原始回执的修改。
#   now：解锁前复查的消费时刻。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("changes,now", [
    ({}, 1351), ({}, 1099), ({"ready": False}, 1200),
    ({"source_observed_at_unix_ms": True}, 1200),
    ({"valid_until_unix_ms": 2000}, 1200), ({"valid_until_unix_ms": None}, 1200),
])
def test_preflight_receipt_cannot_authorize_after_priming_delay_or_tampering(changes, now):
    receipt = {"ready": True, "source_observed_at_unix_ms": 1100,
               "valid_until_unix_ms": 1350, **changes}
    with pytest.raises(ValueError, match="EVIDENCE_EXPIRED_OR_INVALID"):
        assert_native_preflight_current(receipt, now_unix_ms=now)


# 功能：
#   非法消费时钟应使观察窗口失效，而不是抛出不可控类型错误。
# 输入：
#   clock：候选消费时钟。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("clock", [-10, True, 1000.0, "1000", None])
def test_observer_rejects_invalid_consumer_clock_without_raising(clock):
    gate = NativePerceptionReadiness()
    assert not gate.observe(health(), now_unix_ms=clock)
    assert gate.independent_frames == 0


# 功能：
#   非法总等待预算在读取文件或创建后台任务前拒绝。
# 输入：
#   tmp_path：测试私有目录。
#   timeout：非法秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("timeout", [True, "1", None, float("nan"), 2**4096])
def test_invalid_timeout_is_rejected_before_filesystem_work(tmp_path, timeout):
    with pytest.raises(ValueError, match="TIMEOUT_INVALID"):
        asyncio.run(wait_for_native_perception(tmp_path / "health", timeout_seconds=timeout))


# 功能：
#   健康记录过深时保持有界失败，不把解析异常当作就绪。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_deeply_nested_health_is_a_bounded_failed_preflight(tmp_path):
    path = tmp_path / "health.json"
    path.write_text("[" * 2000 + "]" * 2000)
    with pytest.raises(RuntimeError, match="NOT_READY_BEFORE_ARM"):
        asyncio.run(wait_for_native_perception(path, timeout_seconds=.01))


# 功能：
#   用可控时钟跳跃验证磁盘结果晚于总期限时，即使观察器返回就绪也不得放行。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：替换本模块时钟与后台读取。
# 输出：
#   None：不返回业务数据。
def test_completed_read_after_overall_deadline_cannot_authorize(tmp_path, monkeypatch):
    from dronedream_agent_core import native_preflight

    clock = [10.]
    monkeypatch.setattr(native_preflight, "time", SimpleNamespace(
        monotonic=lambda: clock[0], time=lambda: 1.))

    # 功能：
    #   在读取完成前推进测试时钟，复现磁盘结果越过总期限。
    # 输入：
    #   args：本测试不使用的读取函数及路径参数。
    # 输出：
    #   payload：合成健康记录。
    async def delayed_read(*args):
        clock[0] += .2
        payload = health()
        return payload

    monkeypatch.setattr(native_preflight.asyncio, "to_thread", delayed_read)
    monkeypatch.setattr(NativePerceptionReadiness, "observe", lambda *args, **kwargs: True)
    with pytest.raises(RuntimeError, match="NOT_READY_BEFORE_ARM"):
        asyncio.run(wait_for_native_perception(tmp_path / "health", timeout_seconds=.1))


# 功能：
#   健康文件超预算、包含非有限值或不是对象时必须拒绝。
# 输入：
#   tmp_path：测试私有目录。
#   contents：非法文件原始字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("contents", [b" " * 262145, b'{"value":NaN}', b'[]'],
                         ids=["oversized", "nonfinite", "not-object"])
def test_native_health_file_rejects_invalid_wire_payload(tmp_path, contents):
    from dronedream_agent_core.native_preflight import _read_native_health

    path = tmp_path / "health"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="NATIVE_PERCEPTION_HEALTH"):
        _read_native_health(path)
