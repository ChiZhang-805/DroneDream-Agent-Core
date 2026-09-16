"""Three-dimensional, action-relevant encoding fixtures, not training evidence."""

import math

import pytest
from test_realtime_feature_encoders import IDENTITY, _mount, _scan

from dronedream_agent_core.contracts import DynamicObstacleObservation, Vector3
from dronedream_agent_core.control_feature_contract import (
    DYNAMIC_TARGET_FEATURE_COUNT,
    GEOMETRY_FEATURE_COUNT,
)
from dronedream_agent_core.realtime_feature_encoders import (
    encode_dynamic_targets,
    encode_metric_geometry,
)
from dronedream_agent_core.relative_motion import cylinder_contact_seconds


# 功能：
#   验证正上方与正下方深度射线进入独立空间槽位，不混入前向水平信息。
# 输入：
#   z：竖直方向符号。
#   cell：对应的特征槽位索引。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("z,cell", [(1.0, 25), (-1.0, 24)])
def test_vertical_rays_have_separate_cells_from_forward(z, cell):
    scan = _scan()
    for sample in scan.samples:
        sample.direction_sensor = Vector3(x=0, y=0, z=z)
    result = encode_metric_geometry(scan, sensor_mount=_mount(), encoded_at_unix_ms=1000)
    assert len(result.features) == GEOMETRY_FEATURE_COUNT
    assert result.valid_mask[cell * 5 : cell * 5 + 5] == [1.0] * 5
    assert result.valid_mask[:40] == [0.0] * 40
    assert result.features[cell * 5] == pytest.approx(0.4)


# 功能：
#   验证传感器安装平移参与从机体原点量取的距离，不被重复远射线掩盖。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_geometry_mount_translation_and_near_obstacle_not_hidden_by_far_ray():
    scan = _scan()
    scan.samples = scan.samples[:4] * 8
    mount = _mount()
    mount.translation_body_m = Vector3(x=0.5, y=0, z=0)
    result = encode_metric_geometry(scan, sensor_mount=mount, encoded_at_unix_ms=1000)
    assert result.features[:2] == pytest.approx([0.5, 0.5])


# 功能：
#   验证声明竖直视场后，只有平面射线的观测不能再报告完整空间覆盖。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_declared_vertical_fov_requires_upper_and_lower_coverage():
    scan = _scan()
    planar = encode_metric_geometry(scan, sensor_mount=_mount(), encoded_at_unix_ms=1000)
    spatial = encode_metric_geometry(
        scan, sensor_mount=_mount(), expected_vertical_fov_rad=math.pi, encoded_at_unix_ms=1000
    )
    assert planar.coverage == 1.0
    assert spatial.coverage < 0.5
    assert "GEOMETRY_ANGULAR_COVERAGE_LOW" in spatial.issue_codes


# 功能：
#   构造具有明确位置、速度、尺寸及原始年龄的合成动态目标。
# 输入：
#   name：目标身份。
#   changes：本用例需要覆盖的目标字段。
# 输出：
#   target：通过合同验证的动态障碍对象。
def _target(name="target", **changes):
    data = dict(
        obstacle_id=name,
        position_m=Vector3(x=4, y=0, z=1),
        velocity_mps=Vector3(x=-2, y=0, z=0),
        radius_m=0.4,
        height_m=1.0,
        confidence=0.9,
        age_seconds=0.0,
    )
    target = DynamicObstacleObservation(**(data | changes))
    return target


# 功能：
#   在静止、单位姿态机体的合成状态下调用真实动态编码器。
# 输入：
#   targets：本次观测到的动态目标。
#   now：编码时刻的 Unix 毫秒数。
# 输出：
#   encoding：包含特征、有效掩码和原始期限的编码结果。
def _encode(targets, *, now=1000):
    encoding = encode_dynamic_targets(
        targets,
        body_position_world_enu_m=Vector3(x=0, y=0, z=1),
        body_orientation_world_from_body=IDENTITY,
        body_velocity_world_enu_mps=Vector3(x=0, y=0, z=0),
        observed_at_unix_ms=1000,
        encoded_at_unix_ms=now,
    )
    return encoding


# 功能：
#   验证动态特征保留相对速度方向、目标尺寸与预计接触时间，区分接近与远离。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_dynamic_features_preserve_signed_vector_size_and_contact_time():
    result = _encode([_target()])
    assert len(result.features) == DYNAMIC_TARGET_FEATURE_COUNT
    assert result.features[:11] == pytest.approx(
        [
            4 / 30,
            0.2,
            1.8 / 30,
            0.9,
            0,
            -0.2,
            0,
            0,
            0.4 / 30,
            1 / 30,
            0,
        ]
    )
    receding = _encode([_target(velocity_mps=Vector3(x=2, y=0, z=0))])
    assert receding.features[1] == pytest.approx(-0.2)
    assert receding.features[2] == 1.0


# 功能：
#   验证同一方向优先编码较早接触的来向目标，而非较近但正在远离的目标。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_same_cell_prioritizes_earliest_contact_not_nearest_receding_target():
    receding = _target(
        "near-receding", position_m=Vector3(x=2, y=0, z=1), velocity_mps=Vector3(x=2, y=0, z=0)
    )
    incoming = _target("far-incoming")
    assert _encode([receding, incoming]).features == _encode([incoming, receding]).features
    assert _encode([receding, incoming]).features[0] == pytest.approx(4 / 30)


# 功能：
#   验证从上方接近的目标与已经重叠的目标都保留有效特征。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_vertical_target_and_direct_overlap_are_not_dropped():
    above = _encode(
        [_target(position_m=Vector3(x=0, y=0, z=3), velocity_mps=Vector3(x=0, y=0, z=-1))]
    )
    assert above.valid_mask[25 * 11 :] == [1.0] * 11
    assert above.features[25 * 11 + 2] == pytest.approx(1.5 / 30)
    overlap = _encode([_target(position_m=Vector3(x=0, y=0, z=1))])
    assert overlap.features[2] == 0
    assert overlap.valid_mask[:11] == [1.0] * 11


# 功能：
#   验证位置外推不会续租目标观测，编码时刻推进后仍按原始时间失效。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_dynamic_track_prediction_never_refreshes_original_expiry():
    result = _encode([_target(age_seconds=0.4)], now=1050)
    assert result.observed_at_unix_ms == 600
    assert result.features[4] == pytest.approx(0.9)
    assert result.features[0] == pytest.approx(3.1 / 30)
    assert result.fresh_at(1100)
    assert not result.fresh_at(1101)
    with pytest.raises(ValueError, match="future"):
        _encode([_target()], now=999)


# 功能：
#   验证不同高度的水平擦过不算圆柱接触，同高侧向穿越保留正确接触时间。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_swept_contact_rejects_horizontal_pass_at_different_altitude():
    assert (
        cylinder_contact_seconds(
            Vector3(x=4, y=0, z=4), Vector3(x=-2, y=0, z=0), radius_m=0.4, height_m=1
        )
        == 30
    )
    assert cylinder_contact_seconds(
        Vector3(x=0, y=4, z=0), Vector3(x=0, y=-2, z=0), radius_m=0.4, height_m=1
    ) == pytest.approx(1.8)
