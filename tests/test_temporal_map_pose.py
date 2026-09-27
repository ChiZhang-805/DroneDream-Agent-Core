"""Temporal correction continuity, not single-frame or route-based localization."""

from dataclasses import replace

import numpy as np
import pytest
from test_local_pose_alignment import scene

from dronedream_agent_core.local_pose_alignment import fit_map_pose, rotation_exp_and_left_jacobian
from dronedream_agent_core.temporal_map_pose import TemporalMapPoseTracker

IDENTITY = dict(map_sha256="a" * 64, binding_sha256="b" * 64, clock_domain="unit-clock")
BIAS = np.array([.04, -.03, .02])


# 功能：构造带固定坐标误差的解析几何输入，真值只属于测试夹具而非定位器输入。
# 输入：tracker、source stamp、可见平面数、当前位置及可选姿态误差。
# 输出：连续配准结果及同帧独立配准结果。
def observe(tracker, stamp, planes=3, position=(0., 0., 0.), rotation=None, **overrides):
    cloud, _ = scene(planes)
    rotation = np.eye(3) if rotation is None else rotation
    raw_position = np.asarray(position) @ rotation.T + BIAS
    points = cloud @ rotation.T + BIAS
    args = dict(sensor_origins_world_m=raw_position,
                reference_position_world_m=raw_position)
    result = tracker.update(points, INDEX, **args, **{
        **IDENTITY, "source_timestamp_ns": stamp, "reset_counter": 0, **overrides})
    cold = fit_map_pose(points, INDEX, **args)
    return result, cold, raw_position


_, INDEX = scene()


# 功能：验证从完整定位进入缺少纵向特征的走廊时，保留历史校正而不是每帧重置。
# 输入：多个可重复姿态误差；原生位移经地图旋转后传播。
# 输出：位置随实测运动推进，缺失方向仍明确缺失，不凭空授予协方差。
@pytest.mark.parametrize("angle", [0., .005, -.03, .06])
def test_anchor_then_partial_geometry_keeps_measured_motion(angle):
    tracker = TemporalMapPoseTracker(**IDENTITY)
    rotation = rotation_exp_and_left_jacobian([0., 0., angle])[0]
    first, _, _ = observe(tracker, 1_000_000_000, rotation=rotation)
    assert first.fit.usable_candidate and not first.history_used
    for step in range(1, 12):
        position = [0., .02 * step, 0.]
        result, cold, native = observe(tracker, 1_000_000_000 + step * 50_000_000,
                                      planes=2, position=position, rotation=rotation)
        assert result.history_used and result.fit.usable_candidate
        assert result.fit.observed_translation_rank == 2
        np.testing.assert_allclose(native + result.fit.correction_world_m, position, atol=1e-5)
        assert abs((native + cold.correction_world_m - position)[1]) > .015
        assert not result.covariance_qualified and not result.motion_permission_granted
        assert not result.fit.covariance_qualified


# 功能：无完整历史锚时不得从走廊或路线猜出纵向位置。
# 输入：首次观测仅有两个独立平面。
# 输出：未观测方向的偏差仍在，且没有紧协方差。
def test_partial_initialization_does_not_invent_an_anchor():
    result, _, native = observe(TemporalMapPoseTracker(**IDENTITY), 100, planes=2)
    assert result.fit.observed_translation_rank == 2
    assert (native + result.fit.correction_world_m)[1] == pytest.approx(BIAS[1])
    assert not result.history_used and not result.covariance_qualified


# 功能：估计器重置和源时间空洞必须淘汰旧修正，不能把跨段位移当连续运动。
# 输入：重置、回绕重置计数、超时三种边界。
# 输出：重新初始化后保留缺失方向的不确定性。
@pytest.mark.parametrize("case", ["reset", "wrap", "gap"])
def test_reset_and_gap_retire_history(case):
    tracker = TemporalMapPoseTracker(**IDENTITY)
    observe(tracker, 1_000_000_000, reset_counter=255 if case == "wrap" else 0)
    stamp = 1_300_000_000 if case == "gap" else 1_050_000_000
    result, _, native = observe(tracker, stamp, planes=2,
                               reset_counter=1 if case == "reset" else 0)
    assert not result.history_used
    assert result.history_retired_reason == ("SOURCE_GAP" if case == "gap" else "ESTIMATOR_RESET")
    assert (native + result.fit.correction_world_m)[1] == pytest.approx(BIAS[1])


# 功能：重复图像、错序、身份错配、损坏数值均禁止继续借用历史。
# 输入：字段变更矩阵。
# 输出：明确拒绝且后续调用不能悄悄恢复旧缓存。
@pytest.mark.parametrize("changes", [
    {"source_timestamp_ns": 100}, {"source_timestamp_ns": 99},
    {"source_timestamp_ns": True}, {"source_timestamp_ns": 2**63},
    {"reset_counter": True}, {"reset_counter": -1}, {"reset_counter": 256},
    {"map_sha256": "c" * 64}, {"binding_sha256": "c" * 64}, {"clock_domain": "other"},
])
def test_invalid_source_invalidates_state(changes):
    tracker = TemporalMapPoseTracker(**IDENTITY)
    observe(tracker, 100)
    stamp = changes.get("source_timestamp_ns", 200)
    rest = {k: v for k, v in changes.items() if k != "source_timestamp_ns"}
    with pytest.raises(ValueError):
        observe(tracker, stamp, **rest)
    with pytest.raises(ValueError, match="TRACKER_INVALIDATED"):
        observe(tracker, 300)


# 功能：配准失败帧不能续租旧校正的有效期。
# 输入：一次完整锚、两次无有效对应点观测和超过历史时限的新帧。
# 输出：失败保留，后续重置历史而不是无限延续旧锚。
def test_failed_fits_do_not_refresh_history():
    tracker = TemporalMapPoseTracker(**IDENTITY)
    observe(tracker, 1_000_000_000)
    for stamp in [1_100_000_000, 1_200_000_000]:
        bad = tracker.update([[100., 100., 100.]] * 20, INDEX,
            sensor_origins_world_m=BIAS, reference_position_world_m=BIAS,
            source_timestamp_ns=stamp, reset_counter=0, **IDENTITY)
        assert not bad.fit.usable_candidate
        assert bad.previous_correction_timestamp_ns == 1_000_000_000
    result, _, _ = observe(tracker, 1_260_000_000, planes=2)
    assert not result.history_used and result.history_retired_reason == "SOURCE_GAP"


# 功能：已知方向缺少新约束时，原生运动漂移不能被误报为已消除。
# 输入：两个平面只能看到横向和高度，纵向原生输入额外漂移两厘米。
# 输出：历史校正可保留，但新漂移仍存在；这是能力边界，不伪造定位证据。
def test_unobserved_new_drift_remains_explicit():
    tracker = TemporalMapPoseTracker(**IDENTITY)
    observe(tracker, 1_000_000_000)
    result, _, native = observe(tracker, 1_050_000_000, planes=2, position=[0., .02, 0.])
    assert (native + result.fit.correction_world_m)[1] == pytest.approx(.02, abs=1e-6)
    assert result.fit.observed_translation_rank == 2
    assert result.fit.translation_information_shape[1][1] == pytest.approx(0., abs=1e-12)


# 功能：拒绝无效初始化配置，防止布尔值和无限缓存窗口进入源时钟处理。
# 输入：配置变更。
# 输出：无；所有非法配置均抛出明确异常。
@pytest.mark.parametrize("changes", [
    {"map_sha256": "wrong"}, {"binding_sha256": None}, {"clock_domain": ""},
    {"maximum_history_gap_ns": True}, {"maximum_history_gap_ns": 250_000_001},
    {"maximum_history_gap_ns": 0}, {"limits": {}},
])
def test_constructor_rejects_invalid_configuration(changes):
    with pytest.raises(ValueError):
        TemporalMapPoseTracker(**{**IDENTITY, **changes})


# 功能：历史初值失败时仅允许一致的全姿态重定位；不能用部分解覆盖缺失方向。
# 输入：重定位解的秩或与历史预测的偏差，故障注入仅属于单元测试。
# 输出：最多两次有界求解，保留原始失败及重定位尝试的完整结果。
@pytest.mark.parametrize("case", ["consistent", "partial", "position-jump", "rotation-jump"])
def test_bounded_reinitialization_retains_failure_and_requires_consistency(monkeypatch, case):
    import dronedream_agent_core.temporal_map_pose as module
    tracker = TemporalMapPoseTracker(**IDENTITY)
    first, _, _ = observe(tracker, 100)
    calls = []

    # 功能：注入一次初值不收敛及一个重定位解，检查选择规则而不伪造实测资料。
    # 输入：正常求解调用参数。
    # 输出：明确标记的单元测试候选。
    def solve(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return replace(first.fit, usable_candidate=False, issue="MAP_POSE_DID_NOT_CONVERGE")
        if case == "partial":
            return replace(first.fit, observed_pose_rank=5, observed_translation_rank=2)
        if case == "position-jump":
            return replace(first.fit, correction_world_m=(.1, .1, .1))
        if case == "rotation-jump":
            return replace(first.fit, rotation_world_from_input=tuple(map(tuple,
                rotation_exp_and_left_jacobian([0., 0., .02])[0])))
        return first.fit

    monkeypatch.setattr(module, "fit_map_pose", solve)
    result, _, _ = observe(tracker, 200)
    assert len(calls) == 2
    assert result.failed_temporal_fit.issue == "MAP_POSE_DID_NOT_CONVERGE"
    assert result.reinitialization_attempt is not None
    assert result.fit.usable_candidate == (case == "consistent")
    assert not result.covariance_qualified
