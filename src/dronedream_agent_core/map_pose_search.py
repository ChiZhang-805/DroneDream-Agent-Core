"""Bounded multi-start map registration candidate, with explicit ambiguity.

This is not an uncertainty calibration or a flight-enabling fallback. All
starts use the same raw rays, map, support threshold and total pose envelope.
Unsuccessful starts remain in the report; competing valid poses reject it.
"""

import math
from dataclasses import dataclass

import numpy as np

from .local_pose_alignment import MapPoseAlignmentLimits, MapPoseFit, fit_map_pose


@dataclass(frozen=True)
class MapPoseSearch:
    candidate: MapPoseFit | None
    attempts: tuple[MapPoseFit, ...]
    issue: str | None
    covariance_qualified: bool = False
    motion_permission_granted: bool = False


# 功能：以固定十三个位置/姿态起点修复初始关联缺失，不放宽对应点门槛或最终修正范围。
# 输入：points、index、sensor_origins_world_m、reference_position_world_m：同次观测与地图；
#       limits：调用方明确的总修正包络；默认保留原局部求解限制。
# 输出：所有尝试和一致的候选；不同可行解相差超过两厘米或半度则报告歧义，不选低残差掩盖。
def search_map_pose(
    points, index, *, sensor_origins_world_m, reference_position_world_m, limits=None
):
    limits = MapPoseAlignmentLimits() if limits is None else limits
    if not isinstance(limits, MapPoseAlignmentLimits):
        raise ValueError("MAP_POSE_SEARCH_LIMITS_INVALID")
    step = min(math.radians(3), limits.maximum_rotation_rad * 0.6)
    distance_step = min(0.1, limits.geometry.maximum_translation_m * 0.5)
    seeds = np.vstack(
        (
            np.zeros((1, 6)),
            np.column_stack((np.zeros((6, 3)), np.vstack((np.eye(3), -np.eye(3))) * step)),
            np.column_stack((np.vstack((np.eye(3), -np.eye(3))) * distance_step, np.zeros((6, 3)))),
        )
    )
    attempts = tuple(
        fit_map_pose(
            points,
            index,
            sensor_origins_world_m=sensor_origins_world_m,
            reference_position_world_m=reference_position_world_m,
            limits=limits,
            initial_correction_world_m=tuple(seed[:3]),
            initial_rotation_vector_world_rad=tuple(seed[3:]),
        )
        for seed in seeds
    )
    valid = [fit for fit in attempts if fit.usable_candidate]
    if not valid:
        return MapPoseSearch(None, attempts, "MAP_POSE_SEARCH_NO_VALID_START")
    # Check every pair, not only distances to the winning solution: two modes
    # on opposite sides of a central estimate must not pass a doubled bound.
    for i, left in enumerate(valid):
        for right in valid[i + 1 :]:
            distance = np.linalg.norm(np.array(left.correction_world_m) - right.correction_world_m)
            relative = (
                np.array(left.rotation_world_from_input)
                @ np.array(right.rotation_world_from_input).T
            )
            angle = math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
            if distance > 0.02 or angle > math.radians(0.5):
                return MapPoseSearch(None, attempts, "MAP_POSE_SEARCH_AMBIGUOUS")
    # Rank changes indicate a different constraint set, not better confidence.
    # Do not cherry-pick the start claiming the strongest observability.
    if len({(fit.observed_pose_rank, fit.observed_translation_rank) for fit in valid}) != 1:
        return MapPoseSearch(None, attempts, "MAP_POSE_SEARCH_OBSERVABILITY_UNSTABLE")
    chosen = min(valid, key=lambda fit: (fit.residual_p95_m, -fit.matched_count))
    return MapPoseSearch(chosen, attempts, None)
