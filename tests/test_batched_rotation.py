import math
import numpy as np
import pytest

from dronedream_agent_core.contracts import QuaternionWxyz, Vector3
from dronedream_agent_core.quaternion_geometry import rotate_vector, rotate_vectors
from dronedream_agent_core.sensor_bridge import MetricRangeSensorBridge
from test_sensor_bridge import _mount, _scan


# 功能：
#   对随机、极大、极小和零向量，逐位核对批量几何与原标量实现相同。
# 输入：
#   scale：契约允许的四元数范数偏差。
# 输出：
#   None：断言数组、长度和输入未变。
@pytest.mark.parametrize('scale', [1., .9995, 1.0005])
def test_batch_matches_scalar_without_geometry_drift(scale):
    rng = np.random.default_rng(805)
    for _ in range(10):
        raw = rng.normal(size=4)
        raw *= scale / np.linalg.norm(raw)
        q = QuaternionWxyz(**dict(zip(('w', 'x', 'y', 'z'), raw)))
        values = rng.normal(size=(80, 3))
        values[:4] = [[0, 0, 0], [1e200, -1e200, 1e200], [1e-300, 0, 0], [1, 0, 0]]
        before = values.copy()
        expected = np.asarray([rotate_vector(q, tuple(row.tolist())) for row in values])
        result = rotate_vectors(q, values)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(values, before)
        assert not np.shares_memory(result, values)


# 功能：
#   拒绝坏形状、过大批次、非有限输入及绕过赋值校验的坏姿态。
# 输入：
#   value：无效数组。
# 输出：
#   None：断言保守拒绝。
@pytest.mark.parametrize('value', [np.zeros((0, 3)), np.zeros((1025, 3)), np.zeros((2, 4)),
                                  np.zeros((3,)), np.array([[math.nan, 0, 0]]),
                                  np.array([[math.inf, 0, 0]]), np.zeros((1, 3), dtype=np.int64)])
def test_invalid_batch_is_rejected(value):
    q = QuaternionWxyz(w=1, x=0, y=0, z=0)
    with pytest.raises(ValueError, match='VECTOR_INVALID'):
        rotate_vectors(q, value)
    with pytest.raises(ValueError, match='QUATERNION_INVALID'):
        rotate_vectors(q.model_copy(update={'w': 2.}), np.zeros((1, 3)))


# 功能：
#   大帧跨越批次边界后，逐条射线核对校准偏移、端点、原始时刻与质量。
# 输入：
#   无。
# 输出：
#   None：断言桥接与逐射线公式等价。
def test_bridge_large_frame_preserves_all_ordered_endpoints():
    scan, mount = _scan(Vector3(x=1, y=0, z=0)), _mount()
    rng = np.random.default_rng(99)
    scan.samples = [scan.samples[0].model_copy(update={
        'direction_sensor': Vector3(x=float(x), y=float(y), z=float(z)),
        'range_m': 1. + n % 17, 'hit': bool(n % 2)})
        for n, (x, y, z) in enumerate(rng.normal(size=(1100, 3)))]
    frame = MetricRangeSensorBridge(mount).assemble(scan)
    offset = rotate_vector(scan.body_orientation_world_from_body, tuple(mount.translation_body_m.model_dump().values()))
    origin = tuple(scan.body_position_world_enu_m.model_dump().values())
    origin = tuple(origin[i] + offset[i] for i in range(3))
    for sample, ray in zip(scan.samples, frame.range_rays, strict=True):
        direction = tuple(sample.direction_sensor.model_dump().values())
        norm = math.dist((0, 0, 0), direction)
        direction = rotate_vector(mount.orientation_body_from_sensor, tuple(v / norm for v in direction))
        direction = rotate_vector(scan.body_orientation_world_from_body, direction)
        assert tuple(ray.endpoint_m.model_dump().values()) == tuple(origin[i] + direction[i] * sample.range_m for i in range(3))
        assert ray.hit == sample.hit and ray.confidence == sample.confidence
        assert ray.observed_at_monotonic_seconds == scan.observed_at_monotonic_seconds
