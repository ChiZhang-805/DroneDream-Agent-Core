"""Nominal camera coverage for demonstration design, never evidence of free space."""

import itertools
import math

from ..contracts import QuaternionWxyz, Vector3
from ..quaternion_geometry import rotate_vector


# 功能：
#   1. 检查完整米制包围盒是否位于标定相机视锥内，以最小有符号平面距离表示覆盖余量。
#   2. 只评估给定姿态的几何覆盖，不判断遮挡、目标检测质量、预测姿态或飞行安全。
# 输入：
#   camera_position、orientation_world_from_sensor：相机世界位置和前向为 x 的标定姿态。
#   box_minimum、box_maximum：世界 ENU 包围盒上下界。
#   horizontal_fov_rad、vertical_fov_rad：水平及竖直视场弧度。
#   minimum_range_m、maximum_range_m：近远轴向平面米距离；远面不是径向量程。
# 输出：
#   margin_m：完整包围盒到最近视锥平面的有符号米距离，负值表示部分或全部超出。
def box_frustum_margin(camera_position: Vector3, orientation_world_from_sensor: QuaternionWxyz,
                       box_minimum: Vector3, box_maximum: Vector3, *, horizontal_fov_rad: float,
                       vertical_fov_rad: float, minimum_range_m: float,
                       maximum_range_m: float) -> float:
    vectors = [Vector3.model_validate(v.model_dump(), strict=True)
               for v in (camera_position, box_minimum, box_maximum)]
    origin, low, high = [tuple(v.model_dump().values()) for v in vectors]
    orientation = QuaternionWxyz.model_validate(
        orientation_world_from_sensor.model_dump(), strict=True)
    values = (horizontal_fov_rad, vertical_fov_rad, minimum_range_m, maximum_range_m)
    if (any(type(v) not in (int, float) or not 0 < v < 100_000 for v in values)
            or not 0 < horizontal_fov_rad < math.pi
            or not 0 < vertical_fov_rad < math.pi
            or minimum_range_m >= maximum_range_m
            or any(low[i] > high[i] for i in range(3))):
        raise ValueError('OBSERVATION_FRUSTUM_BOUND_INVALID')
    inverse = QuaternionWxyz(w=orientation.w, x=-orientation.x, y=-orientation.y, z=-orientation.z)
    horizontal, vertical = horizontal_fov_rad/2, vertical_fov_rad/2
    margins = []
    # 凸包与平面距离为线性关系，八个角均在六个平面内才能覆盖整个盒；中心可见并不充分。
    for corner in itertools.product(*zip(low, high, strict=True)):
        x, y, z = rotate_vector(inverse, tuple(corner[i]-origin[i] for i in range(3)))
        margins.extend((x-minimum_range_m, maximum_range_m-x,
            x*math.sin(horizontal)-abs(y)*math.cos(horizontal),
            x*math.sin(vertical)-abs(z)*math.cos(vertical)))
    margin_m = min(margins)
    if not math.isfinite(margin_m):
        raise ValueError('OBSERVATION_FRUSTUM_MARGIN_NONFINITE')
    return margin_m
