"""Causal measurement boundary; no route, truth or flight qualification inputs."""

import copy
import json
import math
from dataclasses import replace

import numpy as np
import pytest
from test_native_odometry_source import fields
from test_temporal_native_map_pose import inputs

from dronedream_agent_core.live_map_measurement import (
    MAXIMUM_MEASUREMENT_DISPATCH_AGE_NS,
    LiveMapMeasurementSource,
    MapMeasurementPending,
    MapNoiseBounds,
)


# 功能：建立完整来源链夹具；输入：无；输出：实际求解器及同帧记录，不提供定位真值。
def fixture():
    _, args = inputs()
    record = dict(map_sha256=args['map_sha256'], truth_correction_applied=False,
        motion_permission_granted=False, native_pose_binding_sha256=args['binding_sha256'],
        scan=args['scan'].model_dump(mode='json'), mount=args['mount'].model_dump(mode='json'),
        native_odometry_snapshot={'source_alignment': dict(clock_domain=args['clock_domain'],
            image_timestamp_ns=args['image_timestamp_ns'], system_id=1, component_id=1,
            received_monotonic_seconds=10., fields_json=json.dumps(fields(
                x=0., y=0., z=0., q=[math.sqrt(.5), 0., 0., math.sqrt(.5)])))})
    source = LiveMapMeasurementSource(index=args['index'], map_sha256=args['map_sha256'],
        binding=args['binding'], binding_sha256=args['binding_sha256'],
        clock_domain=args['clock_domain'], noise=MapNoiseBounds(.001, .02, .01, .6, 2., 1., 2.))
    return source, record


# 功能：验证真实求解串联的坐标、源钟、噪声下限及无授权属性。
# 输入：三面几何夹具和固定当前钟。
# 输出：预期 NED 修正，不改输入，不用像素数消除系统误差。
def test_bound_record_produces_measured_pose_without_authority():
    source, record = fixture()
    original = copy.deepcopy(record)
    result = source.measure(record, source_now_ns=lambda: 1_050_000_000,
                            monotonic_now=lambda: 10.05)
    np.testing.assert_allclose(result['position_ned_m'], [.03, -.04, .02], atol=1e-6)
    assert result['source_timestamp_ns'] == 1_000_000_000
    assert result['pose_covariance_upper'][0] >= .02**2 - 1e-12
    assert not result['motion_permission_granted'] and not result['covariance_qualified']
    assert record == original
    with pytest.raises(MapMeasurementPending, match='DID_NOT_ADVANCE'):
        source.measure(record, source_now_ns=lambda: 1_050_000_000, monotonic_now=lambda: 10.05)


# 功能：身份、真值污染和来源钟错误不能被解释为可重试迟到帧。
# 输入：绑定、真值和时钟错误。
# 输出：明确拒绝且不是普通 Pending。
@pytest.mark.parametrize('fault', ['map', 'binding', 'truth', 'clock', 'future'])
def test_invalid_identity_and_clock_are_not_waiting_states(fault):
    source, record = fixture()
    clock = 1_050_000_000
    if fault == 'map':
        record['map_sha256'] = 'b'*64
    elif fault == 'binding':
        record['native_pose_binding_sha256'] = 'c'*64
    elif fault == 'truth':
        record['truth_correction_applied'] = True
    elif fault == 'clock':
        record['native_odometry_snapshot']['source_alignment']['clock_domain'] = 'wrong-run'
    else:
        clock = 999_000_000
    with pytest.raises(ValueError) as captured:
        source.measure(record, source_now_ns=lambda: clock, monotonic_now=lambda: 10.05)
    assert not isinstance(captured.value, MapMeasurementPending)


# 功能：算完才过期也不能发送，不能把计算结束时间伪装成拍摄时间。
# 输入：开始新鲜、计算结束时源年龄过界的时钟序列。
# 输出：拒绝发送；该旧帧不能重新求解刷新有效期。
def test_computation_expiry_does_not_restamp_source():
    source, record = fixture()
    expired = 1_000_000_001 + MAXIMUM_MEASUREMENT_DISPATCH_AGE_NS
    clock = iter((1_050_000_000, expired))
    with pytest.raises(MapMeasurementPending, match='SOURCE_EXPIRED'):
        source.measure(record, source_now_ns=lambda: next(clock), monotonic_now=lambda: 10.05)
    with pytest.raises(MapMeasurementPending, match='DID_NOT_ADVANCE'):
        source.measure(record, source_now_ns=lambda: 1_050_000_000, monotonic_now=lambda: 10.05)


# 功能：缺失、非法与零误差配置不能得到看似确定的位置；输入：非法界；输出：配置拒绝。
@pytest.mark.parametrize('value', [True, 0., -1., float('nan'), float('inf'), '0.01'])
def test_noise_bounds_require_explicit_finite_positive_values(value):
    with pytest.raises(ValueError, match='NOISE_BOUND_INVALID'):
        MapNoiseBounds(.001, value, .01, .6, 2., 1., 2.)


# 功能：允许有限同步误差时必须同时增加位置和姿态不确定性；不改真实采样时刻。
def test_transport_uncertainty_is_paid_in_covariance():
    tight, record = fixture()
    tolerant, second = fixture()
    tolerant.noise = replace(tolerant.noise, transport_clock_uncertainty_seconds=.008)
    a = tight.measure(record, source_now_ns=lambda: 1_050_000_000, monotonic_now=lambda: 10.05)
    b = tolerant.measure(second, source_now_ns=lambda: 1_050_000_000, monotonic_now=lambda: 10.05)
    assert a['source_timestamp_ns'] == b['source_timestamp_ns']
    assert b['transport_clock_uncertainty_us'] == 8000
    for slot in (0, 6, 11, 15, 18, 20):
        assert b['pose_covariance_upper'][slot] > a['pose_covariance_upper'][slot]


@pytest.mark.parametrize('value', [True, 0, -.1, .010001, float('nan'), float('inf')])
def test_transport_uncertainty_bound(value):
    with pytest.raises(ValueError, match='NOISE_BOUND_INVALID'):
        MapNoiseBounds(.001, .02, .01, .6, 2., 1., 2., value)
