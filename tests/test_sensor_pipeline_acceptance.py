"""Analytic integration fixtures, deliberately not flight qualification."""

import math
import struct
from types import SimpleNamespace

import pytest

from dronedream_agent_core.contracts import QuaternionWxyz, RawMetricRangeScan, Vector3
from dronedream_agent_core.depth_sensor_binding import DepthSensorBinding
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.perception_runtime import RuntimePerceptionFusion
from dronedream_agent_core.realtime_feature_encoders import encode_metric_geometry
from dronedream_agent_core.runtime_sensor_contracts import (
    RuntimeSensorRegistry,
    oakd_lite_depth_sensor_contract,
)
from dronedream_agent_core.sensor_bridge import MetricRangeSensorBridge


# 功能：
#   通过真实投影、射线组装、融合和几何编码验证源像素有效率逐层传播，不伪造高质量。
# 输入：
#   coverage：合成图像的已知有效像素比例。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("coverage", [1., 1/64])
def test_raw_depth_validity_reaches_fusion_and_model_features(coverage):
    width, height = 160, 120
    values = [2. if coverage == 1. else math.nan] * (width*height)
    if coverage != 1.:
        for row in range(0, height, 8):
            for col in range(0, width, 8):
                values[row*width+col] = 2.
    message = SimpleNamespace(width=width, height=height, step=width*4, pixel_format_type=13,
                              data=struct.pack(f"<{len(values)}f", *values))
    registry = RuntimeSensorRegistry()
    result = DepthSensorBinding(registry, vehicle_id="quad").project(message)
    zero = Vector3(x=0, y=0, z=0)
    mount = oakd_lite_depth_sensor_contract()
    scan = RawMetricRangeScan(sensor_id=mount.sensor_id, sequence=1,
        observed_at_unix_ms=1000, observed_at_monotonic_seconds=1.,
        body_position_world_enu_m=zero, body_velocity_world_enu_mps=zero,
        body_orientation_world_from_body=QuaternionWxyz(w=1, x=0, y=0, z=0),
        localization_covariance_m2=.01, source_coverage=result.source_coverage,
        samples=list(result.samples))
    frame = MetricRangeSensorBridge(mount).assemble(scan)
    world = MetricVoxelMap(resolution_m=.25,
        minimum_bound_m=Vector3(x=-25, y=-25, z=-25),
        maximum_bound_m=Vector3(x=25, y=25, z=25))
    fusion = RuntimePerceptionFusion(world=world, accepted_sensor_ids={mount.sensor_id},
                                     sensor_registry=registry)
    health = fusion.ingest(frame, now_unix_ms=1020, now_monotonic_seconds=1.02)
    assert registry.latest(mount.sensor_id).coverage == coverage
    assert fusion.frozen_frame.source_coverage == coverage
    assert health.stream_healthy == (coverage == 1.)
    encoded = encode_metric_geometry(scan, sensor_mount=mount,
        expected_horizontal_fov_rad=1.274,
        expected_vertical_fov_rad=2*math.atan(math.tan(1.274/2)*height/width),
        encoded_at_unix_ms=1020)
    assert encoded.quality == pytest.approx(.92*coverage)
    assert (encoded.quality >= .35) == (coverage == 1.)


# 功能：
#   单像素细障碍经过完整感知链后仍为占据单元，射线顺序不能把它覆盖成自由空间。
# 输入：
#   reverse_order：是否反转射线处理顺序。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("reverse_order", [False, True])
def test_thin_depth_hit_survives_full_pixel_to_map_pipeline(reverse_order):
    width, height = 160, 120
    values = [8.] * (width * height)
    # Deliberately away from the historical lattice's tile-center pixel.
    values[41 * width + 65] = 1.25
    registry = RuntimeSensorRegistry()
    projection = DepthSensorBinding(registry, vehicle_id="quad").project(
        SimpleNamespace(width=width, height=height, step=width * 4, pixel_format_type=13,
                        data=struct.pack(f"<{len(values)}f", *values)))
    mount = oakd_lite_depth_sensor_contract()
    zero = Vector3(x=0, y=0, z=0)
    scan = RawMetricRangeScan(sensor_id=mount.sensor_id, sequence=1,
        observed_at_unix_ms=1000, observed_at_monotonic_seconds=1.,
        body_position_world_enu_m=zero, body_velocity_world_enu_mps=zero,
        body_orientation_world_from_body=QuaternionWxyz(w=1, x=0, y=0, z=0),
        localization_covariance_m2=.01, source_coverage=projection.source_coverage,
        samples=list(projection.samples))
    frame = MetricRangeSensorBridge(mount).assemble(scan)
    near = [ray.endpoint_m for ray in frame.range_rays if ray.hit and ray.endpoint_m.x < 2.]
    assert len(near) == 1
    if reverse_order:
        frame.range_rays.reverse()
    world = MetricVoxelMap(resolution_m=.25,
        minimum_bound_m=Vector3(x=-25, y=-25, z=-25),
        maximum_bound_m=Vector3(x=25, y=25, z=25))
    fusion = RuntimePerceptionFusion(world=world, accepted_sensor_ids={mount.sensor_id},
                                     sensor_registry=registry)
    health = fusion.ingest(frame, now_unix_ms=1020, now_monotonic_seconds=1.02)
    assert health.stream_healthy
    key = world.key_for(near[0])
    assert world.is_occupied(key)
    assert not world.is_observed_free(key)
