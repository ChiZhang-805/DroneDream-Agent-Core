import math

import pytest

from dronedream_agent_core.contracts import (
    CalibratedRangeSensorMount,
    QuaternionWxyz,
    RawMetricRangeScan,
    RawRangeSample,
    Vector3,
)
from dronedream_agent_core.sensor_bridge import MetricRangeSensorBridge


# 功能：
#   创建具有前向及上向安装偏移的米制传感器校准，供坐标转换测试使用。
# 输入：
#   无。
# 输出：
#   mount：测距范围为 0.1 至 20 米的测试校准。
def _mount() -> CalibratedRangeSensorMount:
    mount = CalibratedRangeSensorMount(
        sensor_id="lidar-3d",
        translation_body_m=Vector3(x=0.2, y=0.0, z=0.1),
        orientation_body_from_sensor=QuaternionWxyz(w=1, x=0, y=0, z=0),
        minimum_range_m=0.1,
        maximum_range_m=20,
    )
    return mount


# 功能：
#   创建机体偏航九十度、包含八条同向射线的扫描，明确区分方向幅度与物理距离。
# 输入：
#   direction：传感器坐标系中的射线方向，可尚未归一化。
#   range_m：每条射线测得的米制距离。
# 输出：
#   scan：带固定来源时刻、机体姿态和定位状态的测试扫描。
def _scan(direction: Vector3, *, range_m: float = 2.0) -> RawMetricRangeScan:
    half = math.sqrt(0.5)
    scan = RawMetricRangeScan(
        sensor_id="lidar-3d",
        sequence=1,
        observed_at_unix_ms=1_000,
        observed_at_monotonic_seconds=1.0,
        body_position_world_enu_m=Vector3(x=10, y=20, z=2),
        body_orientation_world_from_body=QuaternionWxyz(
            w=half, x=0, y=0, z=half
        ),
        body_velocity_world_enu_mps=Vector3(x=0, y=0, z=0),
        localization_covariance_m2=0.01,
        samples=[
            RawRangeSample(
                direction_sensor=direction,
                range_m=range_m,
                hit=True,
                confidence=0.9,
            )
            for _ in range(8)
        ],
    )
    return scan


# 功能：
#   核对安装偏移与机体旋转先后正确，且非单位射线方向不改变测得的距离。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_bridge_applies_mount_and_vehicle_pose_to_world_enu() -> None:
    frame = MetricRangeSensorBridge(_mount()).assemble(
        _scan(Vector3(x=2, y=0, z=0))
    )

    ray = frame.range_rays[0]
    assert ray.origin_m.x == pytest.approx(10.0)
    assert ray.origin_m.y == pytest.approx(20.2)
    assert ray.origin_m.z == pytest.approx(2.1)
    assert ray.endpoint_m.x == pytest.approx(10.0)
    assert ray.endpoint_m.y == pytest.approx(22.2)
    assert ray.endpoint_m.z == pytest.approx(2.1)
    assert frame.coordinate_frame == "world-enu"


# 功能：
#   检查零方向及超过校准范围的测距不能形成融合射线。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_bridge_fails_closed_on_zero_direction_and_uncalibrated_range() -> None:
    bridge = MetricRangeSensorBridge(_mount())

    with pytest.raises(ValueError, match="DIRECTION_ZERO"):
        bridge.assemble(_scan(Vector3(x=0, y=0, z=0)))
    with pytest.raises(ValueError, match="OUTSIDE_CALIBRATED_LIMIT"):
        bridge.assemble(_scan(Vector3(x=1, y=0, z=0), range_m=25))


# 功能：
#   检查未知传感器扫描不能套用另一传感器的校准。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_bridge_rejects_sensor_identity_mismatch() -> None:
    scan = _scan(Vector3(x=1, y=0, z=0)).model_copy(
        update={"sensor_id": "unknown-lidar"}
    )

    with pytest.raises(ValueError, match="CALIBRATION_MISMATCH"):
        MetricRangeSensorBridge(_mount()).assemble(scan)
