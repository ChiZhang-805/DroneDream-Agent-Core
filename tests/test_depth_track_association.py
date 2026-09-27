"""Synthetic regressions for depth fragments and physically feasible association."""

from dataclasses import replace

import pytest
from test_depth_obstacle_tracker import _frame

from dronedream_agent_core.depth_obstacle_tracker import DepthMotionTracker, _Cluster, _Track


# 功能：
#   生成明确标为合成的表面簇，隔离关联选择与像素聚类过程。
# 输入：
#   center：实测簇中心；low、high：全部支持点的包围范围。
# 输出：
#   cluster：只供测试的观测簇。
def cluster(center, low, high):
    result = _Cluster(center, low, high, 10, 0.95)
    return result


# 功能：
#   建立具有已知年龄、速度和确认状态的测试轨迹，不参与产品传感器输入。
# 输入：
#   identity、point、velocity、confirmed：测试身份、位置、速度及确认状态。
# 输出：
#   track：不可变的合成历史。
def track(identity, point, velocity=(0.0, 0.0, 0.0), confirmed=False):
    support = cluster(point, tuple(p-0.1 for p in point), tuple(p+0.1 for p in point))
    result = _Track(identity, point, velocity, 0.1, 100, 10, support, point, 0.0001, confirmed)
    return result


# 功能：
#   验证实测合并簇覆盖已确认身份时，略近的新碎片不会夺走唯一已发布轨迹。
# 输入：
#   monkeypatch：测试中注入固定观测簇的工具。
# 输出：
#   None：断言身份延续且未修改旧观测年龄。
def test_merged_surface_keeps_confirmed_identity(monkeypatch):
    tracker = DepthMotionTracker()
    tracker._tracks = {1: track(1, (1., 0., .8), confirmed=True),
                       2: track(2, (1., 0., 1.1))}
    tracker._next_track_id = 3
    merged = cluster((1., 0., 1.), (.9, -.1, .7), (1.1, .1, 1.2))
    monkeypatch.setattr(tracker, '_clusters', lambda frame: [merged])
    observations = tracker.update(_frame(2, 1), observed_at_monotonic_seconds=.2)
    assert len(observations) == 1
    assert observations[0].obstacle_id == 'depth-track-1'
    assert observations[0].age_seconds == 0
    assert tracker._tracks[2].observed_at_unix_ms == 100


# 功能：
#   验证位于新簇外的已确认目标不能凭身份优先抢走另一个目标的观测。
# 输入：
#   monkeypatch：固定观测簇的测试工具。
# 输出：
#   None：断言匹配最近目标，原目标保留原始时间。
def test_confirmed_identity_outside_measured_box_does_not_override_distance(monkeypatch):
    tracker = DepthMotionTracker()
    tracker._tracks = {1: track(1, (1., 0., .8), confirmed=True),
                       2: track(2, (1., 0., 1.))}
    tracker._next_track_id = 3
    measured = cluster((1., 0., 1.), (.9, -.1, .95), (1.1, .1, 1.05))
    monkeypatch.setattr(tracker, '_clusters', lambda frame: [measured])
    observations = tracker.update(_frame(2, 1), observed_at_monotonic_seconds=.2)
    assert tracker._tracks[2].observed_at_unix_ms == 200
    assert observations[0].obstacle_id == 'depth-track-1'
    assert observations[0].age_seconds == pytest.approx(.1)


# 功能：
#   验证最近候选加速度不合理时，仍检查次近但物理可行的候选。
# 输入：
#   monkeypatch：固定观测簇的测试工具。
# 输出：
#   None：断言可行身份被更新、坏候选不删除也不续龄。
def test_invalid_nearest_candidate_does_not_hide_valid_match(monkeypatch):
    tracker = DepthMotionTracker()
    tracker._tracks = {1: track(1, (.5, 0., 1.), (5., 0., 0.), True),
                       2: track(2, (.98, 0., 1.), confirmed=True)}
    tracker._next_track_id = 3
    measured = cluster((1.02, 0., 1.), (1.01, -.1, .9), (1.03, .1, 1.1))
    # 不改变预测位置，却使第一条的最后实测时间更近，产生不可能的速度/加速度。
    tracker._tracks[1] = replace(tracker._tracks[1], observed_at_seconds=.199,
                                 observed_at_unix_ms=199, center=(1.015, 0., 1.),
                                 velocity=(-5., 0., 0.))
    monkeypatch.setattr(tracker, '_clusters', lambda frame: [measured])
    tracker.update(_frame(2, 1.02), observed_at_monotonic_seconds=.2)
    assert tracker._tracks[1].observed_at_unix_ms == 199
    assert tracker._tracks[2].observed_at_unix_ms == 200


# 功能：
#   验证精细单调钟通过但整数毫秒加速度越界的关联被拒绝，保留旧身份和原始时间。
# 输入：
#   monkeypatch：固定簇观测以隔离两种时钟分辨率的测试工具。
# 输出：
#   None：关联被错误接收、身份重置或观测续龄时测试失败。
def test_association_respects_published_millisecond_acceleration(monkeypatch):
    tracker = DepthMotionTracker()
    prior = track(1, (1., 0., 1.), confirmed=True)
    tracker._tracks = {1: prior}
    tracker._next_track_id = 2
    # 平滑速度 1.195 m/s；100 ms 为 11.95 m/s²，99 ms 为 12.07 m/s²。
    center = (1. + 1.195 * .1 / .35, 0., 1.)
    measured = cluster(center, (center[0]-.1, -.1, .9), (center[0]+.1, .1, 1.1))
    monkeypatch.setattr(tracker, '_clusters', lambda frame: [measured])
    frame = _frame(2, center[0]).model_copy(update={"observed_at_unix_ms": 199})
    observations = tracker.update(frame, observed_at_monotonic_seconds=.2)
    assert tracker._tracks == {1: prior}
    assert observations[0].age_seconds == pytest.approx(.099)
    assert tracker._next_track_id == 2
