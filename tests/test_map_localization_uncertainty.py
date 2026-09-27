"""Unknown correlation and absent directions must survive map fusion."""

from dataclasses import replace

import numpy as np
import pytest
from test_local_pose_alignment import scene

from dronedream_agent_core.local_pose_alignment import fit_map_pose
from dronedream_agent_core.map_localization_uncertainty import fuse_map_localization_candidate


# 功能：生成真实求解的分析几何测试夹具，不是仿真飞行或定位资格证据。
# 输入：planes：可见独立平面个数；repeats：同一观测点重复次数。
# 输出：先验位置和联合拟合结果。
def inputs(planes=3, repeats=1):
    points, index = scene(planes)
    bias = np.array([0.04, -0.03, 0.02])
    points = np.tile(points, (repeats, 1))
    fit = fit_map_pose(
        points + bias, index, sensor_origins_world_m=bias, reference_position_world_m=bias
    )
    assert fit.usable_candidate
    return bias.tolist(), fit


# 功能：核对不确定性缺失的方向仍保留先验，不能用伪逆得到零方差。
# 输入：planes：一到三平面。
# 输出：无。
@pytest.mark.parametrize("planes", [1, 2, 3])
def test_missing_directions_keep_uncertainty(planes):
    position, fit = inputs(planes)
    candidate = fuse_map_localization_candidate(
        position_world_m=position,
        covariance_world_m2=(np.eye(3) * 0.0537).tolist(),
        fit=fit,
        point_to_plane_sigma_m=0.005,
    )
    covariance = np.array(candidate.covariance_world_m2)
    for direction in fit.unobserved_directions_world:
        direction = np.array(direction)
        assert direction @ covariance @ direction >= 0.0537 - 1e-12
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert not candidate.covariance_qualified and not candidate.motion_permission_granted
    if planes == 3:
        np.testing.assert_allclose(candidate.position_world_m, [0, 0, 0], atol=1e-6)


# 功能：同一图像像素重复不能增加独立信息、让方差凭空缩小。
# 输入：无。
# 输出：无。
def test_duplicate_pixels_do_not_shrink_information():
    position, fit = inputs()
    _, duplicated = inputs(repeats=2)
    np.testing.assert_allclose(
        fit.translation_information_shape, duplicated.translation_information_shape, atol=1e-12
    )


# 功能：相同先验与测量的重复融合不能按独立观测再次把方差减半。
# 输入：无。
# 输出：无。
def test_perfectly_correlated_estimates_do_not_halve_covariance():
    position, fit = inputs()
    precision = np.array(fit.translation_information_shape) / 0.005**2
    prior = np.linalg.inv(precision)
    fit = replace(fit, correction_world_m=(0.0, 0.0, 0.0))
    candidate = fuse_map_localization_candidate(
        position_world_m=position,
        covariance_world_m2=prior.tolist(),
        fit=fit,
        point_to_plane_sigma_m=0.005,
    )
    np.testing.assert_allclose(candidate.covariance_world_m2, prior, atol=1e-12)


# 功能：无效先验、噪声模型和错配位姿明确拒绝，不能靠宽松转换获得授权。
# 输入：case：破坏的证据字段。
# 输出：无。
@pytest.mark.parametrize(
    "case",
    ["boolean", "nonsymmetric", "negative", "zero", "noise", "reference", "rank", "failed-fit"],
)
def test_invalid_evidence_rejected(case):
    position, fit = inputs()
    covariance = (np.eye(3) * 0.0537).tolist()
    noise = 0.005
    if case == "boolean":
        covariance[0][1] = True
    elif case == "nonsymmetric":
        covariance[0][1] = 0.01
    elif case == "negative":
        covariance[0][0] = -0.1
    elif case == "zero":
        covariance = np.zeros((3, 3)).tolist()
    elif case == "noise":
        noise = None
    elif case == "reference":
        position[0] += 0.01
    elif case == "rank":
        fit = replace(fit, observed_translation_rank=2)
    elif case == "failed-fit":
        fit = replace(fit, usable_candidate=False)
    with pytest.raises(ValueError):
        fuse_map_localization_candidate(
            position_world_m=position,
            covariance_world_m2=covariance,
            fit=fit,
            point_to_plane_sigma_m=noise,
        )
