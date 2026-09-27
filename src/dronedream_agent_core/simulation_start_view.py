"""Plan an informative initial sensor view, never manufacture an observation.

Only the simulated entity's initial heading is selected, before it is spawned.
The physical sensor, deployment origin and runtime localization gates stay intact.
Predicted map rays must never enter the live measurement or training channels.
"""

import math

import numpy as np

from .local_map_alignment import MapAlignmentLimits, MapSurfaceIndex
from .local_pose_alignment import fit_map_pose
from .simulation_sensor_frames import _rotation, _vector


# 功能：在固定起点比较真实相机可见的地图表面，避免默认朝向只有单墙导致定位无法初始化。
# 输入：光学地图索引、已校验机型安装/光学参数和模型起点；不读取仿真真值或用户电脑路径。
# 输出：仅用于生成前朝向的预测回执；不授予定位资格，运行时仍必须取得真实相机与飞控回读。
def select_simulation_start_view(*, index, frames, model_root):
    if not isinstance(index, MapSurfaceIndex) or frames.get('depth_mount_verified') is not True:
        raise ValueError('START_VIEW_BOUND_DEPTH_CAMERA_REQUIRED')
    root = _vector(model_root, 3)
    center = _vector(frames['collision_center_model_m'], 3)
    receipt = dict(schema_version='dronedream.simulation-start-view.v1',
        purpose='pre-spawn-heading-only', predicted_only=True,
        localization_qualified=False, motion_permission_granted=False,
        selected_yaw_rad=None, candidates=[])
    # 当前位置契约采用固定碰撞中心；有水平偏心时不能暗中转动后仍沿用旧地图原点。
    if np.linalg.norm(center[:2]) > 1e-8:
        return dict(receipt, issue='START_VIEW_ECCENTRIC_ORIGIN_REQUIRES_EXPLICIT_CONTRACT')
    optics = frames['depth_optics']
    width, height = optics['image_width_px'], optics['image_height_px']
    if any(type(v) is not int or not 32 <= v <= 8192 for v in (width, height)):
        raise ValueError('START_VIEW_CAMERA_DIMENSIONS_INVALID')
    fov, near, far = (optics[k] for k in ('horizontal_fov_rad', 'near_m', 'far_m'))
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (fov, near, far))
            or not 0 < fov < math.pi or not 0 < near < far <= 1000):
        raise ValueError('START_VIEW_CAMERA_OPTICS_INVALID')
    camera = frames['optical_at_rest']
    offset = _vector(camera['position_m'], 3)
    rotation = _rotation(camera['orientation_wxyz'])
    # 像素中心网格保持真实宽高比与针孔参数，视线范围有界；不足的信息不补造为命中。
    xs, ys = np.meshgrid(np.linspace(.025, .975, 20), np.linspace(.025, .975, 15))
    tangent = math.tan(fov / 2)
    rays = np.column_stack((np.ones(xs.size), (1 - 2 * xs.ravel()) * tangent,
                            (1 - 2 * ys.ravel()) * tangent * height / width)) @ rotation.T
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    distance = min(far, 6.)
    limits = MapAlignmentLimits()
    reference = root + center
    for degrees in (0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 135, -135, 180):
        yaw = math.radians(degrees)
        turn = np.array([[math.cos(yaw), -math.sin(yaw), 0.],
                         [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
        origin = root + turn @ offset
        directions = rays @ turn.T
        entry = dict(yaw_rad=yaw, predicted_pose_rank=0, predicted_translation_rank=0)
        try:
            matches = index.match(origin + directions * distance, limits,
                                  sensor_origins_world_m=origin.tolist())
            if matches.surface_ranges_m is None:
                entry['issue'] = 'NO_PREDICTED_SURFACE'
            else:
                ranges = matches.surface_ranges_m
                valid = np.isfinite(ranges) & (ranges > near) & (ranges < distance)
                if int(valid.sum()) < limits.minimum_correspondences:
                    entry['issue'] = 'INSUFFICIENT_PREDICTED_SURFACES'
                else:
                    fit = fit_map_pose(origin + directions[valid] * ranges[valid, None], index,
                        sensor_origins_world_m=origin.tolist(),
                        reference_position_world_m=reference.tolist())
                    entry.update(predicted_pose_rank=fit.observed_pose_rank,
                                 predicted_translation_rank=fit.observed_translation_rank,
                                 matched_count=fit.matched_count, issue=fit.issue)
                    if (fit.usable_candidate and fit.observed_pose_rank == 6
                            and fit.observed_translation_rank == 3):
                        receipt['selected_yaw_rad'] = yaw
        except ValueError as error:
            # 地图预算/表面不支持只淘汰该朝向，不放宽求解阈值或偷偷裁剪地图。
            entry['issue'] = str(error)
        receipt['candidates'].append(entry)
        if receipt['selected_yaw_rad'] is not None:
            return receipt
    return dict(receipt, issue='START_VIEW_NO_COMPLETE_PREDICTED_CONSTRAINT')
