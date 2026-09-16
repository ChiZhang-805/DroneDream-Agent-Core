"""Workload-order tests; no synthetic flight success or deadline relaxation."""

import itertools

import pytest

from dronedream_agent_core.runtime_scheduling import model_input_work_allowed


# 功能：
#   结果交接周期在所有新帧组合下都不启动下一份输入准备。
# 输入：
#   fresh、waiting：新深度接入与更新帧等待状态。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fresh,waiting", tuple(itertools.product((False, True), repeat=2)))
def test_handoff_only_completes_prior_request(fresh, waiting):
    assert not model_input_work_allowed(continuous_control=True, handoff_tick=True,
                                        new_depth_frame=fresh, newer_depth_waiting=waiting)


# 功能：
#   等待的新相机优先于旧几何重用，但已完成新帧处理不因追逐后续帧而饿死推理。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_waiting_camera_takes_precedence_over_reusing_previous_geometry():
    assert not model_input_work_allowed(continuous_control=True, handoff_tick=False,
                                        new_depth_frame=False, newer_depth_waiting=True)
    # After actual ingestion, a request can start even when the next camera
    # frame arrived during processing. No unbounded pursuit of latest input.
    assert model_input_work_allowed(continuous_control=True, handoff_tick=False,
                                    new_depth_frame=True, newer_depth_waiting=True)


# 功能：
#   没有等待图像时保留已有新鲜几何的使用机会，下游仍独立核对时效和动作权限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_no_waiting_image_does_not_disable_existing_fresh_geometry():
    assert model_input_work_allowed(continuous_control=True, handoff_tick=False,
                                    new_depth_frame=False, newer_depth_waiting=False)
    # This says nothing about freshness or actuation. The caller must retain
    # source clocks, independent state, periodic gating and runtime validation.


# 功能：
#   非连续、无直接运动授权的建议调度保持已有工作顺序。
# 输入：
#   handoff、fresh、waiting：交接、新帧和等待状态。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("handoff,fresh,waiting",
                         tuple(itertools.product((False, True), repeat=3)))
def test_coarse_non_actuating_advisory_schedule_is_unchanged(handoff, fresh, waiting):
    assert model_input_work_allowed(continuous_control=False, handoff_tick=handoff,
                                    new_depth_frame=fresh, newer_depth_waiting=waiting)


# 功能：
#   验证交接、接入和再请求的跨周期顺序，不启动旧输入重试循环。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_handoff_then_perception_then_request_without_stale_retry_loop():
    ticks = ((True, False, True), (False, True, False), (True, False, True),
             (False, True, True), (False, False, True))
    starts = [model_input_work_allowed(continuous_control=True, handoff_tick=handoff,
                                      new_depth_frame=fresh, newer_depth_waiting=waiting)
              for handoff, fresh, waiting in ticks]
    assert starts == [False, True, False, True, False]
