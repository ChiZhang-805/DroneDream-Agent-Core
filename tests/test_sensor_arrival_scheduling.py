import math

import pytest

from dronedream_agent_core.runtime_scheduling import (
    SensorArrivalScheduler,
    local_input_cadence_enabled,
    sensor_input_maximum_rate_hz,
)


# 功能：
#   教师连续操纵与本地策略复用同一采样调度，云端、旧坐标输出和仅记录模式不混入。
# 输入：
#   provider、mode、teacher：被验证的运行组合。
#   expected：应采用的采样调度种类。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("provider,mode,teacher,expected", [
    (None, "normalized-body-velocity", True, True),
    ("local-policy", "normalized-body-velocity", False, True),
    ("simulation-training", "normalized-body-velocity", False, True),
    (None, "normalized-body-velocity", False, False),
    ("kimi", "normalized-body-velocity", False, False),
    ("kimi", "normalized-body-velocity", True, False),
    (None, "legacy-candidate-selection", True, False),
    ("local-policy", "legacy-candidate-selection", False, False),
])
def test_teacher_and_deployed_control_share_input_cadence(provider, mode, teacher, expected):
    assert local_input_cadence_enabled(
        provider, mode, simulation_teacher_control=teacher) is expected


# 功能：
#   建立来源尚未过期且未处理的新传感器提示。
# 输入：
#   changes：本测试要提供的补充字段。
# 输出：
#   flags：采样唤醒条件。
def hint(**changes):
    flags = dict(now_monotonic=1.02, sample_monotonic=.95,
                processed_sample_monotonic=.8, fault_active=False, **changes)
    return flags


# 功能：
#   独立新帧在输入预算允许时立即唤醒，不必等待无关维护定时边界。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_new_frame_does_not_wait_for_unrelated_maintenance_timer():
    scheduler = SensorArrivalScheduler(rate_hz=20.)
    scheduler.record_attempt(now_monotonic=.90)
    # A maintenance pass at 1.00 does not consume the input budget: a frame
    # arriving at 1.02 can start then, without waiting for the 1.05 timer.
    assert scheduler.eligible(**hint())
    scheduler.record_attempt(now_monotonic=1.02)
    assert not scheduler.eligible(**{**hint(), "processed_sample_monotonic": .95})


# 功能：
#   处理尝试限速且不因延迟累积追赶突发，也不重复处理已消费来源。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_newest_value_is_coalesced_without_catchup_or_duplicate_processing():
    scheduler = SensorArrivalScheduler(rate_hz=20.)
    scheduler.record_attempt(now_monotonic=1.)
    assert not scheduler.eligible(**hint())
    assert scheduler.eligible(**{**hint(), "now_monotonic": 1.051})
    scheduler.record_attempt(now_monotonic=1.051)
    assert not scheduler.eligible(**{**hint(), "now_monotonic": 1.06,
                                    "sample_monotonic": 1.01})
    assert scheduler.eligible(**{**hint(), "now_monotonic": 3., "sample_monotonic": 2.95})
    scheduler.record_attempt(now_monotonic=3.)
    assert not scheduler.eligible(**{**hint(), "now_monotonic": 3.001,
                                    "sample_monotonic": 2.99})


# 功能：
#   缺失、已处理、未来、过期或损坏的来源不能触发处理。
# 输入：
#   changes：非法提示字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("changes", [
    {"sample_monotonic": None}, {"sample_monotonic": .77},
    {"sample_monotonic": .9, "now_monotonic": 1.151},
    {"sample_monotonic": 1.021}, {"sample_monotonic": math.nan},
    {"processed_sample_monotonic": .95}, {"fault_active": True},
    {"now_monotonic": math.inf}, {"now_monotonic": True},
])
def test_invalid_missing_processed_or_expired_samples_cannot_wake(changes):
    scheduler = SensorArrivalScheduler(rate_hz=20.)
    assert not scheduler.eligible(**{**hint(), **changes})


# 功能：
#   失败处理在限速后可重试，但仍使用原时间且不能延长样本寿命。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_alignment_attempt_is_bounded_but_can_retry_without_redating():
    scheduler = SensorArrivalScheduler(rate_hz=20.)
    original = hint()
    assert scheduler.eligible(**original)
    scheduler.record_attempt(now_monotonic=original["now_monotonic"])
    assert not scheduler.eligible(**{**original, "now_monotonic": 1.03})
    assert scheduler.eligible(**{**original, "now_monotonic": 1.071})
    # Waking never extends the original 250 ms input lease.
    assert not scheduler.eligible(**{**original, "now_monotonic": 1.201})


# 功能：
#   零、负数、布尔或非有限频率不能建立输入调度器。
# 输入：
#   rate：非法采样频率。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("rate", [0., -1., math.nan, math.inf, True])
def test_bad_rates_fail_at_construction(rate):
    with pytest.raises(ValueError, match="SENSOR_ARRIVAL_RATE_INVALID"):
        SensorArrivalScheduler(rate_hz=rate)


# 功能：
#   单调时间倒退不能重新增加输入处理机会。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_clock_regression_does_not_restore_input_budget():
    scheduler = SensorArrivalScheduler(rate_hz=20.)
    scheduler.record_attempt(now_monotonic=1.02)
    with pytest.raises(ValueError, match="SENSOR_ARRIVAL_CLOCK_INVALID"):
        scheduler.record_attempt(now_monotonic=1.01)
    assert not scheduler.eligible(**hint())


# 功能：
#   半周期唤醒可以减少相机相位等待，仍不允许旧帧重放、追赶突发或来源续期。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_local_input_can_remove_camera_phase_lag_without_changing_control_period():
    maintenance_rate = 20.
    local_rate = sensor_input_maximum_rate_hz(maintenance_rate_hz=maintenance_rate,
                                             continuous_local_control=True)
    assert local_rate == 40.
    coarse = SensorArrivalScheduler(rate_hz=maintenance_rate)
    local = SensorArrivalScheduler(rate_hz=local_rate)
    for scheduler in (coarse, local):
        scheduler.record_attempt(now_monotonic=1.)
    # Original scene at .956, delivered at 1.036; previous work ended at 1.035.
    # Periodic-only input would add another 14 ms despite an idle worker.
    incoming = dict(now_monotonic=1.036, sample_monotonic=.956,
                    processed_sample_monotonic=.90, fault_active=False)
    assert local.eligible(**incoming)
    assert not coarse.eligible(**incoming)
    local.record_attempt(now_monotonic=1.036)
    assert not local.eligible(**{**incoming, "now_monotonic": 1.037,
                                 "sample_monotonic": .957})  # No catch-up burst.
    assert not local.eligible(**{**incoming, "now_monotonic": 1.062,
                                 "processed_sample_monotonic": .956})  # No replay.
    assert not local.eligible(**{**incoming, "now_monotonic": 1.207})  # Still expires.
    assert not local.eligible(**{**incoming, "now_monotonic": 1.062, "fault_active": True})
    assert sensor_input_maximum_rate_hz(maintenance_rate_hz=maintenance_rate,
                                        continuous_local_control=False) == 20.


# 功能：
#   动态唤醒频率计算也拒绝非法维护频率，不产生无界处理速率。
# 输入：
#   rate：非法维护频率。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("rate", [0., -1., math.nan, math.inf, True])
def test_bad_phase_handoff_rates_cannot_create_unbounded_processing(rate):
    with pytest.raises(ValueError, match="SENSOR_ARRIVAL_RATE_INVALID"):
        sensor_input_maximum_rate_hz(maintenance_rate_hz=rate, continuous_local_control=True)
