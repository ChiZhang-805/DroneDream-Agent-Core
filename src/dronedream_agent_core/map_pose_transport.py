"""Convert a map-fit mean back to native NED/FRD without inventing confidence.

The result is explicitly an unqualified candidate: it is not an estimator
measurement until a separately validated uncertainty model and clock transport
have been supplied. No native velocity is recycled as an independent visual
velocity measurement.
"""

import math
from dataclasses import dataclass

import numpy as np

from .local_pose_alignment import MapPoseFit, rotation_exp_and_left_jacobian
from .native_odometry_source import SourceOdometry, source_pose_in_map
from .native_pose import _multiply


@dataclass(frozen=True)
class MapPoseTransportCandidate:
    source_timestamp_us: int
    native_timestamp_us: int
    source_skew_us: int
    position_ned_m: tuple[float, float, float]
    quaternion_ned_from_frd_wxyz: tuple[float, float, float, float]
    observed_translation_rank: int
    covariance_qualified: bool = False
    motion_permission_granted: bool = False


# 功能：把图像地图配准的位姿均值转换到原飞控参考系，不混用 ENU/NED 或 FLU/FRD。
# 输入：packet：图像配对的原始定位；fit：该位姿的配准；binding 及摘要：固定地图变换；
#       image_timestamp_ns：真实图像源时刻，不允许拿当前接收时刻替代。
# 输出：仅含均值与时间偏差的未验收候选；不输出伪造速度、协方差或融合权限。
def map_pose_transport_candidate(
    *,
    packet,
    fit,
    binding,
    binding_sha256,
    image_timestamp_ns,
):
    if type(packet) is not SourceOdometry or type(fit) is not MapPoseFit:
        raise ValueError("MAP_POSE_TRANSPORT_SOURCE_INVALID")
    if (
        fit.usable_candidate is not True
        or fit.issue is not None
        or type(fit.observed_translation_rank) is not int
        or not 1 <= fit.observed_translation_rank <= 3
    ):
        raise ValueError("MAP_POSE_TRANSPORT_FIT_UNUSABLE")
    if (
        type(image_timestamp_ns) is not int
        or not 0 < image_timestamp_ns < 2**63
        or image_timestamp_ns % 1000
    ):
        raise ValueError("MAP_POSE_TRANSPORT_IMAGE_TIMESTAMP_INVALID")
    image_us = image_timestamp_ns // 1000
    skew_us = packet.timestamp_us - image_us
    if abs(skew_us) > 20_000:
        raise ValueError("MAP_POSE_TRANSPORT_SOURCE_SKEW_TOO_LARGE")
    position, _, _ = source_pose_in_map(packet, binding=binding, binding_sha256=binding_sha256)
    expected = (position.x, position.y, position.z)
    vectors = (
        fit.reference_position_world_m,
        fit.correction_world_m,
        fit.rotation_vector_world_rad,
    )
    for vector in vectors:
        if (
            not isinstance(vector, (list, tuple))
            or len(vector) != 3
            or any(type(v) not in (int, float) or not -1e6 <= v <= 1e6 for v in vector)
        ):
            raise ValueError("MAP_POSE_TRANSPORT_VECTOR_INVALID")
    if not np.allclose(expected, fit.reference_position_world_m, rtol=0, atol=1e-8):
        raise ValueError("MAP_POSE_TRANSPORT_REFERENCE_MISMATCH")
    correction = fit.correction_world_m
    vector = fit.rotation_vector_world_rad
    angle = math.hypot(*vector)
    if math.hypot(*correction) > 0.5 or angle > math.radians(10):
        raise ValueError("MAP_POSE_TRANSPORT_CORRECTION_OUT_OF_RANGE")
    rotation, _ = rotation_exp_and_left_jacobian(vector)
    reported_rotation = fit.rotation_world_from_input
    if (
        not isinstance(reported_rotation, (list, tuple))
        or len(reported_rotation) != 3
        or any(
            not isinstance(row, (list, tuple))
            or len(row) != 3
            or any(type(v) not in (int, float) or not -1.0 <= v <= 1.0 for v in row)
            for row in reported_rotation
        )
        or not np.allclose(rotation, reported_rotation, atol=1e-8, rtol=0)
    ):
        raise ValueError("MAP_POSE_TRANSPORT_ROTATION_MISMATCH")
    factor = math.sin(angle / 2) / angle if angle else 0.5
    delta_enu = (math.cos(angle / 2), *(v * factor for v in vector))
    enu_from_ned = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)
    ned_from_enu = (0.0, -math.sqrt(0.5), -math.sqrt(0.5), 0.0)
    delta_ned = _multiply(_multiply(ned_from_enu, delta_enu), enu_from_ned)
    quaternion = _multiply(delta_ned, packet.orientation_frame_from_body_wxyz)
    norm = math.hypot(*quaternion)
    quaternion = tuple(v / norm for v in quaternion)
    if quaternion[0] < 0:
        quaternion = tuple(-v for v in quaternion)
    north, east, down = packet.position_frame_m
    return MapPoseTransportCandidate(
        image_us,
        packet.timestamp_us,
        skew_us,
        (north + correction[1], east + correction[0], down - correction[2]),
        quaternion,
        fit.observed_translation_rank,
    )
