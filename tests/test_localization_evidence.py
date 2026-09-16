"""Synthetic boundary tests; these do not qualify physical sensor hardware."""

import pytest

from dronedream_agent_core.localization_evidence import (
    native_localization_evidence,
    normalize_native_pose_covariance,
    position_variance_bound_m2,
)


# 功能：
#   构造三个位置轴方差不同、交叉项为零的独立协方差夹具。
# 输入：
#   无。
# 输出：
#   packed：按 6×6 上三角顺序排列的 21 项数组。
def covariance():
    packed = [0.0] * 21
    packed[0], packed[6], packed[11] = 0.01, 0.02, 0.03
    return packed


# 功能：
#   构造已知年龄及原生估计坐标标签的定位包，不含仿真位置真值。
# 输入：
#   无。
# 输出：
#   payload：每次调用独立创建的定位遥测对象。
def identity():
    payload = {"dynamics": {"collected_at_unix_ms": 1000, "sources": {"odometry": {
        "frame_id": "ESTIM_NED", "timestamp_us": 100_000,
        "sample_age_seconds": 0.05, "pose_covariance_upper_m2": covariance(),
    }}}}
    return payload


# 功能：
#   非零相关项必须增加方差上界，源年龄必须从汇集时间中扣除。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_covariance_bound_includes_correlations():
    assert position_variance_bound_m2(covariance()) == 0.03
    packed = covariance()
    packed[2] = 0.01
    assert position_variance_bound_m2(packed) == 0.04
    assert native_localization_evidence(
        identity(), now_unix_ms=1020, maximum_age_ms=250,
    ) == (0.03, 950)


# 功能：
#   度量坐标标签不改变标量不确定性上界，也不授权使用包内任意位置和姿态。
# 输入：
#   frame：允许的度量估计坐标标签。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("frame", ["BODY_NED", "VISION_NED", "ESTIM_NED", "LOCAL_FRD", "LOCAL_NED"])
def test_metric_frame_changes_do_not_change_scalar_uncertainty_bound(frame):
    payload = identity()
    packet = payload["dynamics"]["sources"]["odometry"]
    packet["frame_id"] = frame
    packet["position_body"] = [999, 999, 999]  # Never used as deployed position.
    packet["q"] = [0, 0, 0, 0]  # Never used as deployed attitude.
    assert native_localization_evidence(payload, now_unix_ms=1020, maximum_age_ms=250) == (
        .03, 950,
    )


# 功能：
#   用独立特征值计算和随机正交旋转对照，确认完整及缺失相关项的上界都保守。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_position_uncertainty_bounds_are_conservative_under_frame_rotation():
    import numpy as np

    matrix = np.array([[.01, .002, -.001], [.002, .02, .003], [-.001, .003, .03]])
    rotation, _ = np.linalg.qr(np.random.default_rng(71).normal(size=(3, 3)))
    rotated = rotation @ matrix @ rotation.T
    for covariance_matrix in (matrix, rotated):
        packed = [0.0] * 21
        for index, (row, column) in zip((0, 1, 2, 6, 7, 11),
                                      ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)),
                                      strict=True):
            packed[index] = float(covariance_matrix[row, column])
        assert position_variance_bound_m2(packed) >= np.linalg.eigvalsh(matrix).max()
        packed[1] = packed[2] = packed[7] = None
        assert position_variance_bound_m2(packed) == pytest.approx(np.trace(matrix))


# 功能：
#   验证原生 NaN 相关项保留为未知并使用迹上界；已知错误相关项仍要拒绝。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_px4_diagonal_covariance_preserves_unknown_correlations_conservatively():
    native = [float("nan")] * 21
    native[0], native[6], native[11] = .01, .02, .03
    # This is the layout emitted by the installed PX4 ODOMETRY stream.
    packet = normalize_native_pose_covariance(native)
    assert packet[1] is None and packet[15] is None
    assert position_variance_bound_m2(packet) == pytest.approx(.06)
    payload = identity()
    payload["dynamics"]["sources"]["odometry"]["pose_covariance_upper_m2"] = packet
    assert native_localization_evidence(payload, now_unix_ms=1020, maximum_age_ms=250) == (
        pytest.approx(.06), 950,
    )
    packet[1] = .9  # Partial reporting is not permission to violate PSD.
    with pytest.raises(ValueError, match="POSITIVE_SEMIDEFINITE"):
        position_variance_bound_m2(packet)
    assert normalize_native_pose_covariance([float("nan")] * 21) is None


# 功能：
#   区分协议允许的 NaN 未知值与 Infinity、布尔值或文本等损坏数据。
# 输入：
#   value：非法协方差元素。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), True, "unknown"])
def test_native_covariance_malformed_values_are_not_unknown_correlations(value):
    native = covariance()
    native[1] = value
    with pytest.raises(ValueError, match="VALUE_INVALID"):
        normalize_native_pose_covariance(native)


# 功能：
#   拒绝负方差、非半正定相关项、非有限数及过大位置不确定性。
# 输入：
#   index：被修改的协方差索引。
#   value：非法测量值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("index,value", [(0, -1), (0, float("nan")), (0, True),
                                         (1, 1), (11, float("inf")), (11, 1e308)])
def test_invalid_covariance_never_becomes_good_localization(index, value):
    packed = covariance()
    packed[index] = value
    with pytest.raises(ValueError):
        position_variance_bound_m2(packed)


# 功能：
#   零方差和错误长度被拒绝；未报告协方差保持未知并保留已有源时间。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_zero_unknown_and_truncated_covariance_are_distinct_from_measurement():
    for packed in ([0.0] * 21, covariance()[:20]):
        with pytest.raises(ValueError):
            position_variance_bound_m2(packed)
    assert native_localization_evidence({}, now_unix_ms=1000, maximum_age_ms=250) == (None, None)
    payload = identity()
    payload["dynamics"]["sources"]["odometry"]["pose_covariance_upper_m2"] = None
    assert native_localization_evidence(
        payload, now_unix_ms=1000, maximum_age_ms=250,
    ) == (None, 950)


# 功能：
#   过期年龄、非法坐标标签和错误设备时间不能成为可用定位。
# 输入：
#   field：被修改的字段名。
#   value：待拒绝的字段内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [("sample_age_seconds", 0.251),
    ("sample_age_seconds", 1e308), ("sample_age_seconds", -0.1),
    ("sample_age_seconds", float("nan")), ("frame_id", "UNDEF"), ("timestamp_us", True)])
def test_present_invalid_localization_fails_closed(field, value):
    payload = identity()
    payload["dynamics"]["sources"]["odometry"][field] = value
    with pytest.raises(ValueError):
        native_localization_evidence(payload, now_unix_ms=1020, maximum_age_ms=250)


# 功能：
#   重新发布外层身份不延长定位年龄，未来汇集时刻也不能被使用。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_original_age_cannot_be_renewed_by_republishing_identity():
    payload = identity()
    payload["updated_at_unix_ms"] = 1400
    with pytest.raises(ValueError, match="EXPIRED"):
        native_localization_evidence(payload, now_unix_ms=1400, maximum_age_ms=250)
    with pytest.raises(ValueError, match="FUTURE"):
        native_localization_evidence(identity(), now_unix_ms=999, maximum_age_ms=250)


# 功能：
#   超大数值与错误形状必须由协方差入口统一拒绝，不能泄漏类型或浮点溢出异常。
# 输入：
#   operation：原始协方差规范化或位置方差上界函数。
#   values：畸形协方差。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "operation", [normalize_native_pose_covariance, position_variance_bound_m2]
)
@pytest.mark.parametrize("values", [None, 1, {i: 0 for i in range(21)}, [10**400] * 21])
def test_covariance_invalid_input_is_a_domain_error(operation, values):
    with pytest.raises(ValueError, match="LOCALIZATION_COVARIANCE"):
        operation(values)


# 功能：
#   在定位包解析前拒绝非法消费时间或年龄上限，即使尚未收到定位也不忽略错误参数。
# 输入：
#   now：候选消费 UNIX 毫秒时刻。
#   age：候选最大年龄毫秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("now,age", [(True, 250), (1000.0, 250), (None, 250),
                                    (2**63, 250), (1000, True), (1000, 250.0), (1000, None)])
def test_localization_consumer_clock_is_strict_even_without_odometry(now, age):
    with pytest.raises(ValueError, match="LOCALIZATION_.*INVALID"):
        native_localization_evidence({}, now_unix_ms=now, maximum_age_ms=age)


# 功能：
#   定位字段存在但损坏时明确拒绝，不得作为从未收到的可选信息静默忽略。
# 输入：
#   payload：顶层、动力学或来源结构损坏的包。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("payload", [None, [], {"dynamics": None}, {"dynamics": []},
    {"dynamics": {"sources": None}}, {"dynamics": {"sources": []}}])
def test_present_corrupt_localization_structure_is_not_missing(payload):
    with pytest.raises(ValueError, match="LOCALIZATION_.*INVALID"):
        native_localization_evidence(payload, now_unix_ms=1020, maximum_age_ms=250)


# 功能：
#   包内非法帧标识、超大年龄与溢出设备时间均以领域错误拒绝。
# 输入：
#   field：被篡改的里程计字段。
#   value：非法字段值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [("frame_id", []), ("sample_age_seconds", 10**400),
                                         ("timestamp_us", 2**63)])
def test_odometry_packet_rejects_unbounded_or_unhashable_fields(field, value):
    payload = identity()
    payload["dynamics"]["sources"]["odometry"][field] = value
    with pytest.raises(ValueError):
        native_localization_evidence(payload, now_unix_ms=1020, maximum_age_ms=250)
