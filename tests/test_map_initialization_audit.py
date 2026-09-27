"""Initialization propagation must preserve missing directions and failures."""

import numpy as np
import pytest
from test_local_pose_alignment import scene

from dronedream_agent_core.map_initialization_audit import audit_map_initialization


# 功能：以解析平面产生有偏观测，验证初始化审计，不将测试夹具当飞行证据。
# 输入：planes：可见平面数；covariance：完整六维地图系先验。
# 输出：审计报告。
def run(planes=3, covariance=None):
    points, index = scene(planes)
    bias = np.array([0.04, -0.03, 0.02])
    if covariance is None:
        covariance = np.diag([0.0001] * 3 + [0.000001] * 3).tolist()
    return audit_map_initialization(
        points + bias,
        index,
        sensor_origins_world_m=bias,
        reference_position_world_m=bias,
        prior_covariance_world=covariance,
    )


# 功能：完整几何应消除小初始平移偏差；输出仍不能自称标定完毕。
# 输入：无。
# 输出：无。
def test_three_planes_correct_initialization_without_granting_authority():
    result = run()
    assert result["failed_samples"] == 0
    assert len(result["samples"]) == 13
    assert result["propagation"]["minimum_translation_rank"] == 3
    np.testing.assert_allclose(
        result["propagation"]["mean_delta_world_m"], [-0.04, 0.03, -0.02], atol=1e-6
    )
    assert result["propagation"]["maximum_position_deviation_from_center_m"] < 1e-6
    assert not result["covariance_qualified"] and not result["motion_permission_granted"]


# 功能：缺失方向必须传播初始位置不确定性，不能由配准残差小推断为零。
# 输入：无。
# 输出：无。
def test_two_planes_keep_unobserved_prior_and_cross_correlation():
    result = run(2)
    assert result["failed_samples"] == 0
    spread = result["propagation"]
    assert spread["minimum_translation_rank"] == 2
    assert spread["position_covariance_from_initialization_m2"][1][1] == pytest.approx(0.0001)
    assert spread["input_pose_output_position_cross_covariance"][1][1] == pytest.approx(0.0001)


# 功能：固定等权采样必须重建完整输入协方差，包括位置姿态交叉相关。
# 输入：无。
# 输出：无。
def test_cubature_keeps_full_cross_terms_and_center_is_not_extra_information():
    prior = np.diag([0.0001] * 3 + [0.000001] * 3)
    prior[0, 3] = prior[3, 0] = 0.000004
    result = run(covariance=prior.tolist())
    offsets = np.array([s["initial_offset_world"] for s in result["samples"][1:]])
    np.testing.assert_allclose(offsets.T @ offsets / 12, prior, atol=1e-14)


# 功能：任何采样越出求解范围都必须保存失败，不对剩余成功样本单独宣称低方差。
# 输入：无。
# 输出：无。
def test_broad_prior_failures_are_not_discarded():
    result = run(covariance=np.diag([0.25] * 3 + [0.000001] * 3).tolist())
    assert result["failed_samples"] > 0
    assert result["propagation"] is None
    assert any(row["fit"]["issue"] for row in result["samples"])


# 功能：非法协方差与未知值不能被数值库静默转成有效证据。
# 输入：case：待破坏的协方差内容。
# 输出：无。
@pytest.mark.parametrize(
    "case", ["bool", "string", "unknown", "negative", "asymmetric", "nan", "huge", "shape"]
)
def test_invalid_prior(case):
    prior = np.diag([0.0001] * 6).tolist()
    if case == "shape":
        prior.pop()
    elif case == "asymmetric":
        prior[0][1] = 0.00001
    else:
        prior[0][0] = {
            "bool": True,
            "string": "0.001",
            "unknown": None,
            "negative": -0.1,
            "nan": float("nan"),
            "huge": 10**1000,
        }[case]
    with pytest.raises(ValueError, match="MAP_INITIALIZATION_PRIOR_INVALID"):
        run(covariance=prior)


# 功能：零先验只说明当前诊断没有注入扰动，不代表传感器误差或地图误差为零。
# 输入：无。
# 输出：无。
def test_zero_prior_is_not_measurement_covariance():
    result = run(covariance=np.zeros((6, 6)).tolist())
    np.testing.assert_allclose(
        result["propagation"]["position_covariance_from_initialization_m2"], 0, atol=1e-20
    )
    assert not result["covariance_qualified"]


# 功能：同一次多起点诊断保留所有尝试，不把失败分支变成中心零误差样本。
# 输入：无。
# 输出：无。
def test_multistart_audit_records_all_attempts():
    points, index = scene()
    result = audit_map_initialization(
        points,
        index,
        sensor_origins_world_m=[0, 0, 0],
        reference_position_world_m=[0, 0, 0],
        prior_covariance_world=np.zeros((6, 6)).tolist(),
        registration_mode="multistart",
    )
    assert result["failed_samples"] == 0
    assert all(len(row["search"]["attempts"]) == 13 for row in result["samples"])
    assert not result["covariance_qualified"]


# 功能：诊断计算不就地修改输入观测、射线起点及先验，防止后续求解读到被篡改的数据。
# 输入：无。
# 输出：无。
def test_input_arrays_not_modified():
    points, index = scene()
    origins = np.zeros_like(points)
    original = points.copy()
    prior = np.diag([0.0001] * 6).tolist()
    prior_copy = [row[:] for row in prior]
    audit_map_initialization(
        points,
        index,
        sensor_origins_world_m=origins,
        reference_position_world_m=[0, 0, 0],
        prior_covariance_world=prior,
    )
    np.testing.assert_array_equal(points, original)
    np.testing.assert_array_equal(origins, 0)
    assert prior == prior_copy
