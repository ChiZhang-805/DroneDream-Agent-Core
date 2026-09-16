import math

import pytest
from test_native_flight_state import _identity

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.native_pose import (
    finite_number,
    native_attitude_enu_flu,
    native_map_pose,
    required_mapping,
    source_timestamp,
)
from dronedream_agent_core.quaternion_geometry import rotate_vector
from dronedream_agent_core.realtime_feature_encoders import world_enu_to_body


# 功能：
#   对照四个水平航向，验证世界方向转为机体前向且上方向保持正确。
# 输入：
#   yaw：原生偏航角。
#   world：该航向对应的世界前向向量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "yaw,world", [(0, (0, 1, 0)), (90, (1, 0, 0)), (180, (0, -1, 0)), (-90, (-1, 0, 0))]
)
def test_native_heading_and_body_handedness(yaw, world):
    orientation = native_attitude_enu_flu(0, 0, yaw)
    body = world_enu_to_body(orientation, Vector3(x=world[0], y=world[1], z=world[2]))
    assert list(body.model_dump().values()) == pytest.approx([1, 0, 0], abs=1e-10)
    up = world_enu_to_body(orientation, Vector3(x=0, y=0, z=1))
    assert up.z == pytest.approx(1)


# 功能：
#   注入与估计状态矛盾的仿真真值，确认地图位姿只消费原生估计数据。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_pose_does_not_consume_truth_position_or_orientation():
    payload = _identity()
    payload["observed_world_collision_center_m"] = {"x": 1e10, "y": -9, "z": 6}
    payload["orientation"] = "untrusted unrelated simulation truth"
    payload["observed_position_ned_m"] = {"north_m": 2.0, "east_m": 3.0, "down_m": -4.0}
    pose = native_map_pose(payload, now_unix_ms=1020)
    assert pose.position_world_enu_m == Vector3(x=3, y=2, z=4)
    assert abs(pose.orientation_world_from_body.w) == pytest.approx(1)


# 功能：
#   删除任一必要来源字段，确认不会回退到推测的原点或当前时间。
# 输入：
#   field：删除的来源字段名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "field",
    [
        "position_received_at_unix_ms",
        "map_frame_binding_sha256",
        "observed_position_ned_m",
        "map_frame_binding",
    ],
)
def test_missing_pose_provenance_never_falls_back(field):
    payload = _identity()
    del payload[field]
    with pytest.raises(ValueError):
        native_map_pose(payload, now_unix_ms=1020)


# 功能：
#   更新外层发布时间后，旧位置或旧姿态仍因源年龄过期而拒绝。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_republished_cached_pose_and_attitude_do_not_refresh():
    payload = _identity()
    payload["updated_at_unix_ms"] = 1400
    with pytest.raises(ValueError, match="(?i)expired"):
        native_map_pose(payload, now_unix_ms=1400)
    payload["position_received_at_unix_ms"] = 1400
    payload["dynamics"]["collected_at_unix_ms"] = 1400
    payload["dynamics"]["sources"]["attitude"]["sample_age_seconds"] = 0.4
    with pytest.raises(ValueError, match="EXPIRED"):
        native_map_pose(payload, now_unix_ms=1400)


# 功能：
#   即使摘要正确，也不允许任意局部航向坐标系冒充固定北向坐标系。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_arbitrary_yaw_frame_is_not_treated_as_north_aligned():
    payload = _identity()
    payload["map_frame_binding"]["orientation"] = "LOCAL_FRD"
    payload["map_frame_binding_sha256"] = sha256_json(payload["map_frame_binding"])
    with pytest.raises(ValueError, match="UNSUPPORTED"):
        native_map_pose(payload, now_unix_ms=1020)


# 功能：
#   拒绝超出欧拉角范围、非有限数及不合法类型的姿态输入。
# 输入：
#   angles：候选横滚、俯仰及偏航角。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("angles", [(0, 91, 0), (181, 0, 0), (0, 0, math.nan),
                                    (True, 0, 0), (0, None, 0), (0, 0, 10**400)])
def test_bad_attitude_rejected(angles):
    with pytest.raises(ValueError):
        native_attitude_enu_flu(*angles)


# 功能：
#   确认超大整数数值以领域错误拒绝，而不是抛出浮点转换溢出。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_oversized_sensor_number_is_invalid_not_an_overflow():
    with pytest.raises(ValueError, match="finite east_m"):
        finite_number({"east_m": 10**400}, "east_m")


# 功能：
#   验证调用方时间与年龄上限的严格类型检查。
# 输入：
#   now：候选消费时刻。
#   age：候选年龄上限。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("now,age", [(1020, True), (1020, None), (None, 250),
                                    (float("nan"), 250), (1020., 250)])
def test_pose_consumer_clock_contract_is_strict(now, age):
    with pytest.raises(ValueError, match="NATIVE_POSE_.*INVALID"):
        native_map_pose(_identity(), now_unix_ms=now, maximum_age_ms=age)


# 功能：
#   即使重新计算摘要，也不能把字符串或布尔坐标当成已验证的部署原点。
# 输入：
#   value：非法的原点坐标。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, "1", 10**400, float("inf")])
def test_map_origin_requires_native_finite_numbers(value):
    payload = _identity()
    payload["map_frame_binding"]["collision_center_origin_world_enu_m"]["x"] = value
    if value != float("inf"):
        payload["map_frame_binding_sha256"] = sha256_json(payload["map_frame_binding"])
    with pytest.raises(ValueError):
        native_map_pose(payload, now_unix_ms=1020)


# 功能：
#   非对象遥测在公共解码入口统一拒绝，不泄漏属性访问错误。
# 输入：
#   reader：数字、时间或子对象读取函数。
#   payload：不符合对象契约的输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("reader", [finite_number, source_timestamp, required_mapping])
@pytest.mark.parametrize("payload", [None, [], 0])
def test_native_field_readers_reject_non_mapping_payloads(reader, payload):
    with pytest.raises(ValueError):
        reader(payload, "field")


# 功能：
#   限制来源整数时刻的存储范围，禁止异常大设备时间进入身份绑定。
# 输入：
#   value：越界或类型错误的源时刻。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [2**63, 10**400, True, 1.0, -1])
def test_source_timestamp_has_a_storage_bound(value):
    with pytest.raises(ValueError):
        source_timestamp({"timestamp_us": value}, "timestamp_us")


# 功能：
#   独立用旋转矩阵对照混合横滚、俯仰和偏航，覆盖仅测水平航向不能发现的轴符号错误。
# 输入：
#   angles：一组原生角度，包含极限俯仰和混合倾斜情况。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("angles", [(30, -20, 70), (-45, 50, -130), (0, 90, 180), (180, -90, 0)])
def test_attitude_matches_independent_world_and_body_basis_changes(angles):
    import numpy as np

    r, p, y = map(math.radians, angles)
    cr, sr, cp, sp = math.cos(r), math.sin(r), math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    enu_from_ned = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    frd_from_flu = np.diag([1, -1, -1])
    world_from_body = enu_from_ned @ rz @ ry @ rx @ frd_from_flu
    orientation = native_attitude_enu_flu(*angles)
    for axis in np.eye(3):
        world = world_from_body @ axis
        rotated = rotate_vector(orientation, tuple(map(float, axis)))
        assert rotated == pytest.approx(world, abs=1e-10)
        body = world_enu_to_body(orientation, Vector3(x=world[0], y=world[1], z=world[2]))
        # 姿态四元数使用右手 FLU，而特征编码器的公共向量明确使用前右上。
        expected = [axis[0], -axis[1], axis[2]]
        assert list(body.model_dump().values()) == pytest.approx(expected, abs=1e-10)


# 功能：
#   地图绑定必须是有界标准 JSON，不能通过摘要字符串化或超量附加字段绕过输入约束。
# 输入：
#   extra：非字符串键或超预算的附加字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("extra", [{1: "ambiguous key"}, {"extra": "x" * 16_384}])
def test_frame_binding_is_bounded_strict_json(extra):
    payload = _identity()
    payload["map_frame_binding"].update(extra)
    payload["map_frame_binding_sha256"] = sha256_json(payload["map_frame_binding"])
    with pytest.raises(ValueError, match="JSON"):
        native_map_pose(payload, now_unix_ms=1020)
