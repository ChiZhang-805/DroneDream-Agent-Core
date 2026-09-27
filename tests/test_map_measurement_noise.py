"""No false precision from point counts, missing axes, time skew, or frame changes."""
from dataclasses import replace

import numpy as np
import pytest
from test_local_pose_alignment import scene

from dronedream_agent_core.local_pose_alignment import fit_map_pose
from dronedream_agent_core.map_measurement_noise import (
    map_attitude_variance_candidate,
    map_position_noise_candidate,
    map_position_noise_to_ned,
)


# 功能：创建具有三个独立平面的真实解析配准和独立噪声参数。
# 输入：无。
# 输出：满秩候选及测试用噪声入参，不表示产品标定。
def inputs():
    points, index = scene()
    fit = fit_map_pose(points, index, sensor_origins_world_m=[[0., 0., 0.]] * len(points),
                       reference_position_world_m=[0., 0., 0.])
    return dict(fit=fit, point_to_plane_sigma_m=.01, position_floor_m=.02,
                source_skew_seconds=0., speed_bound_mps=1., acceleration_bound_mps2=2.)


# 功能：验证像素数不增加虚构信息，时间误差增加而不缩小协方差。
# 输入：满秩解析配准及延迟变体。
# 输出：正定性、单调性和权限断言。
def test_no_pixel_multiplication_and_skew_is_conservative():
    args = inputs()
    baseline = map_position_noise_candidate(**args)
    duplicated = map_position_noise_candidate(**{**args,
        "fit": replace(args["fit"], matched_count=args["fit"].matched_count * 100)})
    assert duplicated == baseline
    delayed = map_position_noise_candidate(**{**args, "source_skew_seconds": .02})
    p, q = np.array(baseline.covariance_world_m2), np.array(delayed.covariance_world_m2)
    assert np.linalg.eigvalsh(q-p).min() > 0
    assert np.linalg.eigvalsh(p).min() >= .02**2 - 1e-12
    assert delayed.variance_bound_m2 >= np.linalg.eigvalsh(q).max() - 1e-12
    assert not delayed.covariance_qualified
    assert not delayed.motion_permission_granted


@pytest.mark.parametrize("changes", [dict(observed_translation_rank=2), dict(observed_pose_rank=5),
                                   dict(usable_candidate=False), dict(issue="bad"),
                                   dict(translation_information_shape=None)])
# 功能：拒绝用当前看不到的方向生成高精度三维观测。
# 输入：缺失约束或拟合失败的变体。
# 输出：明确拒绝断言。
def test_partial_or_failed_measurement_rejected(changes):
    args = inputs()
    args['fit'] = replace(args['fit'], **changes)
    with pytest.raises(ValueError):
        map_position_noise_candidate(**args)


@pytest.mark.parametrize("key,value", [
    ("point_to_plane_sigma_m", 0.), ("position_floor_m", True),
    ("source_skew_seconds", .020001), ("source_skew_seconds", -1.),
    ("speed_bound_mps", float('nan')), ("acceleration_bound_mps2", float('inf'))])
# 功能：拒绝非法噪声和时间预算，不能通过转换或 NaN 比较绕过限制。
# 输入：一个损坏的参数。
# 输出：拒绝断言。
def test_noise_parameters_validated(key, value):
    with pytest.raises(ValueError):
        map_position_noise_candidate(**{**inputs(), key: value})


# 功能：核对 ENU→NED 旋转不丢交叉项或改变特征值。
# 输入：带非零相关项的严格正定测试矩阵。
# 输出：符号、轴序和谱保持断言。
def test_covariance_frame_conversion():
    candidate = map_position_noise_candidate(**inputs())
    matrix = ((.02, .001, .002), (.001, .03, .003), (.002, .003, .04))
    candidate = replace(candidate, covariance_world_m2=matrix)
    result = map_position_noise_to_ned(candidate)
    np.testing.assert_allclose(result, [[.03, .001, -.003], [.001, .02, -.002], [-.003,-.002,.04]])
    np.testing.assert_allclose(np.linalg.eigvalsh(result), np.linalg.eigvalsh(matrix))


# 功能：构建联合配准的姿态噪声测试参数，独立于任何起飞资格。
# 输入：无。
# 输出：真实满秩几何结果及明确的实验参数。
def attitude_inputs():
    return dict(fit=inputs()['fit'], point_to_plane_sigma_m=.01, attitude_floor_rad=.02,
                source_skew_seconds=0., angular_speed_bound_radps=1.,
                angular_acceleration_bound_radps2=2., pitch_rad=0.)


# 功能：检验边缘化与重复像素、时间偏差、欧拉奇异性下的保守界。
# 输入：联合配准及各类姿态变体。
# 输出：单调性、合法性和不虚增精度断言。
def test_attitude_marginal_noise_bounds():
    args = attitude_inputs()
    assert args['fit'].pose_information_shape is not None
    baseline = map_attitude_variance_candidate(**args)
    duplicated = map_attitude_variance_candidate(**{**args,
        'fit': replace(args['fit'], matched_count=100000)})
    assert duplicated == baseline
    assert map_attitude_variance_candidate(**{**args, 'source_skew_seconds': .02}) > baseline
    assert map_attitude_variance_candidate(**{**args, 'pitch_rad': 1.}) > baseline
    information = np.eye(6)
    information[0, 3] = information[3, 0] = .999
    coupled = map_attitude_variance_candidate(**{**args,
        'fit': replace(args['fit'], pose_information_shape=information.tolist())})
    independent = map_attitude_variance_candidate(**{**args,
        'fit': replace(args['fit'], pose_information_shape=np.eye(6).tolist())})
    assert coupled > independent * 100


@pytest.mark.parametrize('key,value', [('attitude_floor_rad', 0.),
    ('pitch_rad', 1.5), ('pitch_rad', True), ('source_skew_seconds', .021),
    ('angular_speed_bound_radps', float('nan')), ('angular_acceleration_bound_radps2', -1.)])
# 功能：拒绝不可信姿态噪声参数，不让非法或奇异输入降级成高精度数据。
# 输入：损坏参数。
# 输出：明确错误。
def test_attitude_invalid_parameters(key, value):
    with pytest.raises(ValueError):
        map_attitude_variance_candidate(**{**attitude_inputs(), key: value})


@pytest.mark.parametrize('information', [None, np.zeros((6, 6)).tolist(),
                                       np.eye(3).tolist()])
# 功能：检查缺失或退化信息矩阵不会被当作满秩姿态测量。
# 输入：缺失或错误维数、秩的矩阵。
# 输出：拒绝断言。
def test_attitude_invalid_information(information):
    args = attitude_inputs()
    args['fit'] = replace(args['fit'], pose_information_shape=information)
    with pytest.raises(ValueError):
        map_attitude_variance_candidate(**args)
