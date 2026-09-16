"""Calibrated LiDAR/depth bridge into the deterministic world-ENU map boundary.

Neither the language model nor the occupancy integrator is allowed to guess
camera geometry.  This adapter consumes pose-synchronized metric samples and an
explicit, immutable sensor mount, then emits the same world-frame rays in
simulation and on real hardware.
"""

from __future__ import annotations

import math

from .contracts import (
    CalibratedRangeSensorMount,
    OnboardPerceptionFrame,
    RangeRayObservation,
    RawMetricRangeScan,
    Vector3,
)
from .quaternion_geometry import rotate_vector as _rotate

Point = tuple[float, float, float]


# 功能：
#   将具名三维分量转为固定轴顺序，供几何运算使用，不交换坐标轴。
# 输入：
#   vector：已验证的三维向量。
# 输出：
#   result：按 x、y、z 排列的元组。
def _tuple(vector: Vector3) -> Point:
    result = vector.x, vector.y, vector.z
    return result


class MetricRangeSensorBridge:
    """Convert calibrated sensor-frame samples to a fusion-ready perception frame."""

    # 功能：
    #   校验采样下限与安装偏移，并冻结校准副本，避免外部修改影响后续投影。
    # 输入：
    #   self：当前传感器桥接实例。
    #   mount：传感器到机体的安装校准与合法测距区间。
    #   minimum_samples：一帧允许接收的最少射线数。
    #   maximum_sensor_offset_m：安装原点相对机体中心的最大距离，单位米。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, mount: CalibratedRangeSensorMount, *, minimum_samples: int = 8,
                 maximum_sensor_offset_m: float = 2.0) -> None:
        if type(minimum_samples) is not int or not 1 <= minimum_samples <= 250_000:
            raise ValueError("minimum samples must be positive")
        try:
            if (type(maximum_sensor_offset_m) not in (int, float)
                    or not math.isfinite(maximum_sensor_offset_m) or maximum_sensor_offset_m <= 0):
                raise ValueError("maximum sensor offset must be positive")
        except OverflowError as error:
            raise ValueError("maximum sensor offset must be positive") from error
        mount = CalibratedRangeSensorMount.model_validate(mount.model_dump(mode="python"))
        if math.dist((0.0, 0.0, 0.0), _tuple(mount.translation_body_m)) > maximum_sensor_offset_m:
            raise ValueError("SENSOR_MOUNT_OUTSIDE_CALIBRATED_BODY")
        self._mount = mount
        self.minimum_samples = minimum_samples

    # 功能：
    #   返回安装校准副本，调用方不能经此属性改写内部校准。
    # 输入：
    #   self：持有冻结校准的桥接实例。
    # 输出：
    #   result：与内部校准无可变引用共享的副本。
    @property
    def mount(self) -> CalibratedRangeSensorMount:
        result = self._mount.model_copy(deep=True)
        return result

    # 功能：
    #   1. 复核扫描身份、射线密度、有限方向和校准量程，将传感器射线转换为世界 ENU 米制射线。
    #   2. 保留来源时刻、质量与定位不确定性，不将缺失覆盖补成已知自由空间。
    # 输入：
    #   self：持有校准和采样下限的桥接实例。
    #   scan：与机体姿态同步的原始米制扫描。
    # 输出：
    #   frame：包含世界坐标射线、定位状态与动态观测的融合输入帧。
    def assemble(self, scan: RawMetricRangeScan) -> OnboardPerceptionFrame:
        scan = RawMetricRangeScan.model_validate(scan.model_dump(mode="python"))
        mount = self._mount
        if scan.sensor_id != mount.sensor_id:
            raise ValueError("PERCEPTION_SENSOR_CALIBRATION_MISMATCH")
        if len(scan.samples) < self.minimum_samples:
            raise ValueError("PERCEPTION_RAY_DENSITY_INSUFFICIENT")
        # 先旋转安装偏移得到真正的发射原点，不能把全部射线误设在机体中心。
        sensor_offset_world = _rotate(
            scan.body_orientation_world_from_body,
            _tuple(mount.translation_body_m),
        )
        body_position = _tuple(scan.body_position_world_enu_m)
        origin = tuple(body_position[index] + sensor_offset_world[index] for index in range(3))
        rays: list[RangeRayObservation] = []
        for sample in scan.samples:
            if not mount.minimum_range_m <= sample.range_m <= mount.maximum_range_m:
                raise ValueError("PERCEPTION_RANGE_OUTSIDE_CALIBRATED_LIMIT")
            raw_direction = _tuple(sample.direction_sensor)
            norm = math.dist((0.0, 0.0, 0.0), raw_direction)
            if not math.isfinite(norm) or norm <= 1e-9:
                raise ValueError("PERCEPTION_RAY_DIRECTION_ZERO")
            sensor_direction = tuple(component / norm for component in raw_direction)
            # 方向先经过安装旋转，再经过机体旋转；距离只在最后乘一次。
            body_direction = _rotate(
                mount.orientation_body_from_sensor,
                sensor_direction,  # type: ignore[arg-type]
            )
            world_direction = _rotate(scan.body_orientation_world_from_body, body_direction)
            endpoint = tuple(
                origin[index] + world_direction[index] * sample.range_m for index in range(3)
            )
            rays.append(
                RangeRayObservation(
                    origin_m=Vector3(x=origin[0], y=origin[1], z=origin[2]),
                    endpoint_m=Vector3(x=endpoint[0], y=endpoint[1], z=endpoint[2]),
                    hit=sample.hit,
                    confidence=sample.confidence,
                    observed_at_monotonic_seconds=scan.observed_at_monotonic_seconds,
                )
            )
        frame = OnboardPerceptionFrame(
            sensor_id=scan.sensor_id,
            sequence=scan.sequence,
            observed_at_unix_ms=scan.observed_at_unix_ms,
            localization_position_m=scan.body_position_world_enu_m,
            localization_velocity_mps=scan.body_velocity_world_enu_mps,
            localization_covariance_m2=scan.localization_covariance_m2,
            source_coverage=scan.source_coverage,
            range_rays=rays,
            dynamic_obstacles=scan.dynamic_obstacles,
        )
        return frame
