"""Synthetic malformed inputs must fail before creating plausible model features."""

import pytest
from test_native_flight_state import _identity

from dronedream_agent_core.control_feature_contract import (
    ENCODER_ROLES,
    encoder_contract_sha256,
    policy_feature_contract_sha256,
)
from dronedream_agent_core.depth_projection import DepthProjectionCalibration
from dronedream_agent_core.native_flight_state import native_flight_state_sample


# 功能：
#   深度标定中的超大整数必须被受控拒绝，不能在转换中溢出或生成可信特征。
# 输入：
#   field：被超大整数替换的标定字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["minimum_depth_m", "maximum_depth_m", "confidence", "fx_pixels"])
def test_oversized_calibration_numbers_have_a_controlled_validation_error(field):
    values = dict(width=4, height=4, horizontal_fov_rad=1.2,
                  minimum_depth_m=.2, maximum_depth_m=10.,
                  fx_pixels=2., fy_pixels=2., cx_pixels=1., cy_pixels=1.)
    values[field] = 2**4096
    with pytest.raises(ValueError):
        DepthProjectionCalibration(**values)


# 功能：
#   大于浮点精确整数范围的 IMU 微秒时间戳仍保持整数身份，不丢掉相邻样本差异。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_imu_integer_timestamp_is_not_rounded_through_float():
    payload = _identity()
    original = 2**53 + 1
    payload["dynamics"]["sources"]["imu"]["timestamp_us"] = original
    sample = native_flight_state_sample(payload, now_unix_ms=1020)
    assert sample.imu_timestamp_us == original


# 功能：
#   过大 IMU 加速度在原生状态转换处产生校验错误，不能泄漏未捕获数值异常。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_imu_overflow_is_a_validation_error_not_an_uncaught_arithmetic_error():
    payload = _identity()
    payload["dynamics"]["sources"]["imu"]["acceleration_forward_m_s2"] = 2**4096
    with pytest.raises(ValueError):
        native_flight_state_sample(payload, now_unix_ms=1020)


# 功能：
#   策略语义组合只接收真实格式的小写来源摘要，拒绝空值、布尔值和错误长度。
# 输入：
#   value：替换其中一个编码器摘要的非法值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [None, True, "not-a-digest", "A" * 64, "a" * 63])
def test_policy_contract_requires_real_lowercase_source_digests(value):
    hashes = {role: encoder_contract_sha256(role) for role in ENCODER_ROLES}
    hashes[ENCODER_ROLES[0]] = value
    assert policy_feature_contract_sha256(hashes) is None


# 功能：
#   非字典不能被当成编码器身份集合，也不应在检查中抛出无关异常。
# 输入：
#   hashes：错误的摘要容器。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("hashes", [None, [], False])
def test_policy_contract_rejects_nonmapping_inputs_without_minting_an_identity(hashes):
    assert policy_feature_contract_sha256(hashes) is None


# 功能：
#   不可能的历史长度或来源间隔不能获得看似有效的飞行状态语义摘要。
# 输入：
#   field：被修改的时序参数名。
#   value：超界或类型不明确的配置值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [
    ("history_length", True), ("history_length", 4.5), ("history_length", 0),
    ("maximum_gap_ms", True), ("maximum_gap_ms", float("inf")), ("maximum_gap_ms", -1),
])
def test_flight_state_identity_cannot_describe_an_impossible_history_configuration(field, value):
    with pytest.raises(ValueError):
        encoder_contract_sha256("flight-state-encoder", **{field: value})


# 功能：
#   实际飞行状态编码器与语义摘要采用相同的整数配置门槛，在消费样本前拒绝歧义。
# 输入：
#   field：历史长度或最大来源间隔参数名。
#   value：不能作为整数配置的输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [
    ("history_length", 4.5), ("history_length", True), ("history_length", "4"),
    ("maximum_gap_ms", 250.5), ("maximum_gap_ms", True), ("maximum_gap_ms", "250"),
])
def test_temporal_encoder_configuration_matches_integer_feature_contract(field, value):
    from dronedream_agent_core.realtime_feature_encoders import FlightStateEncoder

    with pytest.raises(ValueError, match="flight state"):
        FlightStateEncoder(**{field: value})
