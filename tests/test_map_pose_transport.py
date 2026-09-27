"""Coordinate and source-time boundaries for unqualified map-fit transport."""

import math
from dataclasses import replace

import numpy as np
import pytest
from test_native_odometry_source import packet

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_pose_alignment import MapPoseFit, rotation_exp_and_left_jacobian
from dronedream_agent_core.map_pose_transport import map_pose_transport_candidate
from dronedream_agent_core.native_odometry_source import source_pose_in_map


# 功能：在不同地图原点和机体姿态下建立一致配准输入，无任何定位资格标记。
# 输入：origin：地图原点；q：原生 NED/FRD 姿态；rotation：地图系旋转增量。
# 输出：转换函数参数。
def inputs(origin=(10.0, -20.0, 30.0), q=(1.0, 0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)):
    native = packet(q=list(q))
    binding = {
        "orientation": "NED-to-ENU-fixed",
        "source": "deployment-coordinate-contract",
        "collision_center_origin_world_enu_m": dict(zip("xyz", origin, strict=True)),
    }
    digest = sha256_json(binding)
    position, _, _ = source_pose_in_map(native, binding=binding, binding_sha256=digest)
    matrix, _ = rotation_exp_and_left_jacobian(rotation)
    fit = MapPoseFit(
        True,
        (0.02, -0.03, 0.04),
        tuple(rotation),
        tuple(tuple(row) for row in matrix.tolist()),
        (position.x, position.y, position.z),
        6,
        (),
        3,
        (),
        2.0,
        100,
        0.001,
        4,
        None,
        0,
    )
    return dict(
        packet=native,
        fit=fit,
        binding=binding,
        binding_sha256=digest,
        image_timestamp_ns=1_004_000_000,
    )


# 功能：均值修正不依赖本机地图原点，也不以原生包时刻替换图像的采样时刻。
# 输入：origin：平移后的地图原点。
# 输出：无；核对 NED 修正、图像时刻和未授予权限。
@pytest.mark.parametrize(
    "origin", [(0.0, 0.0, 0.0), (-42.25, 15.3, 7.715), (900.0, -1200.0, 400.0)]
)
def test_map_origin_independent_ned_conversion(origin):
    candidate = map_pose_transport_candidate(**inputs(origin=origin))
    assert candidate.position_ned_m == pytest.approx((0.97, 2.02, -3.04))
    assert candidate.quaternion_ned_from_frd_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert candidate.source_timestamp_us == 1_004_000
    assert candidate.native_timestamp_us == 1_000_000
    assert candidate.source_skew_us == -4000
    assert candidate.covariance_qualified is candidate.motion_permission_granted is False
    assert not hasattr(candidate, "velocity_ned_m_s")


# 功能：地图旋转左乘并正确转换基底，不能把 ENU 旋转向量直接加到 NED 欧拉角。
# 输入：axis：地图旋转轴；yaw：任意原生初始偏航。
# 输出：无；把结果独立变回地图系并核对实际几何旋转。
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.0])
def test_attitude_rotation_roundtrip(axis, yaw):
    vector = [0.0, 0.0, 0.0]
    vector[axis] = 0.04
    args = inputs(q=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), rotation=vector)
    result = map_pose_transport_candidate(**args)
    _, before, _ = source_pose_in_map(
        args["packet"], binding=args["binding"], binding_sha256=args["binding_sha256"]
    )
    _, after, _ = source_pose_in_map(
        replace(
            args["packet"], orientation_frame_from_body_wxyz=result.quaternion_ned_from_frd_wxyz
        ),
        binding=args["binding"],
        binding_sha256=args["binding_sha256"],
    )
    from dronedream_agent_core.native_pose import _multiply

    for unit in np.eye(3):
        before_v = _multiply(
            _multiply((before.w, before.x, before.y, before.z), (0, *unit)),
            (before.w, -before.x, -before.y, -before.z),
        )[1:]
        after_v = _multiply(
            _multiply((after.w, after.x, after.y, after.z), (0, *unit)),
            (after.w, -after.x, -after.y, -after.z),
        )[1:]
        expected = np.asarray(args["fit"].rotation_world_from_input) @ before_v
        assert after_v == pytest.approx(expected, abs=1e-8)


# 功能：拒绝不一致拟合与非法采样时刻，避免转换器绕过来源和坐标检查。
# 输入：change：故障变体。
# 输出：无；所有变体明确拒绝。
@pytest.mark.parametrize(
    "change",
    [
        "unusable",
        "reference",
        "rotation",
        "huge",
        "bool",
        "image_bool",
        "image_skew",
        "image_fraction",
        "binding",
    ],
)
def test_invalid_candidate_is_not_transported(change):
    args = inputs()
    fit_changes = {
        "unusable": {"usable_candidate": False},
        "reference": {"reference_position_world_m": (0.0, 0.0, 0.0)},
        "rotation": {"rotation_world_from_input": ((0.0, 0.0, 0.0),) * 3},
        "huge": {"correction_world_m": (1.0, 0.0, 0.0)},
        "bool": {"correction_world_m": (True, 0.0, 0.0)},
    }
    if change in fit_changes:
        args["fit"] = replace(args["fit"], **fit_changes[change])
    elif change == "binding":
        args["binding_sha256"] = "0" * 64
    else:
        args["image_timestamp_ns"] = {
            "image_bool": True,
            "image_skew": 2_000_000_000,
            "image_fraction": 1_004_000_001,
        }[change]
    with pytest.raises(ValueError):
        map_pose_transport_candidate(**args)
