"""Offline sensor/control boundary tests, not model or flight qualifications."""

import pytest
from control_fixtures import complete_feature_snapshot
from test_realtime_feature_encoders import IDENTITY, _mount, _scan

import dronedream_agent_core.realtime_feature_encoders as encoders
from dronedream_agent_core.contracts import NormalizedPilotControl, Vector3


# 功能：
#   提供不含外部账户或执行器的控制参数，用于检查物理量边界。
# 输入：
#   无。
# 输出：
#   arguments：明确的测试控制请求参数。
def _control_arguments():
    arguments = dict(
        source_expert="local-navigation-policy",
        model_call_id="model-" + "a" * 24,
        navigation_snapshot_sha256="b" * 64,
        task_reference_sha256="c" * 64,
        generated_at_unix_ms=1000,
        maximum_acceleration_mps2=1.0,
        maximum_jerk_mps3=5.0,
    )
    return arguments


# 功能：
#   核对候选路径转换拒绝负数、布尔、非有限及超界幅度，不会把非法上限当反向速度。
# 输入：
#   limit：错误的最大速度。
# 输出：
#   无：错误在生成控制请求前被拒绝。
@pytest.mark.parametrize("limit", [-1.0, 0.0, True, "1", float("nan"), float("inf"), 21.0])
def test_candidate_control_rejects_invalid_speed(limit):
    with pytest.raises(ValueError):
        encoders.body_control_intent_for_target(
            **_control_arguments(),
            current_position_world_enu_m=Vector3(x=0, y=0, z=0),
            target_position_world_enu_m=Vector3(x=1, y=0, z=0),
            body_orientation_world_from_body=IDENTITY,
            maximum_speed_mps=limit,
        )


# 功能：
#   验证连续杆量入口不接受布尔幅度、浮点毫秒或失效窗口。
# 输入：
#   field：被替换的物理或时间参数。
#   value：该参数的非法值。
# 输出：
#   无：严格错误类型不能变成有效控制。
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximum_horizontal_speed_mps", True),
        ("maximum_yaw_rate_dps", -20),
        ("harness_control_scale", True),
        ("generated_at_unix_ms", True),
        ("validity_milliseconds", 500.0),
        ("validity_milliseconds", True),
        ("maximum_acceleration_mps2", True),
    ],
)
def test_continuous_control_rejects_coerced_parameters(field, value):
    arguments = _control_arguments() | dict(
        maximum_horizontal_speed_mps=1.0,
        maximum_vertical_speed_mps=0.5,
        maximum_yaw_rate_dps=20.0,
        pilot_control=NormalizedPilotControl(forward_axis=0.5, right_axis=0, up_axis=0, yaw_axis=0),
    )
    arguments[field] = value
    with pytest.raises(ValueError):
        encoders.body_control_intent_for_pilot_control(**arguments)


# 功能：
#   修改已签成快照的数据，确认 fresh_at 不是单纯读取可篡改的就绪标记。
# 输入：
#   mutation：破坏特征、掩码、必需角色或内容散列的方式。
# 输出：
#   无：被破坏的快照不能得到可控制时限。
@pytest.mark.parametrize("mutation", ["features", "mask", "roles", "hash"])
def test_snapshot_freshness_rejects_mutated_content(mutation):
    snapshot = complete_feature_snapshot()
    if mutation == "features":
        snapshot.encodings[0].features[0] = 0.75
    elif mutation == "mask":
        snapshot.fused_valid_mask[0] = 0.0
    elif mutation == "roles":
        snapshot.required_roles.clear()
    else:
        snapshot.__dict__["snapshot_sha256"] = "0" * 64
    assert snapshot.fresh_at(1000) is False
    assert snapshot.control_deadline_unix_ms(now_unix_ms=1000) == 0


# 功能：截止计算必须只读刚刚验证的副本，避免校验后重新读取调用方可变容器。
# 输入：monkeypatch：在副本生成后改变原对象；输出：保留原截止，不采用被改写的时间。
def test_deadline_uses_the_verified_detached_snapshot(monkeypatch):
    snapshot = complete_feature_snapshot(1000)
    original_copy = encoders._strict_copy

    # 功能：模拟验证后的原容器变动；输入：原对象和契约；输出：已验证且未修改的副本。
    def copy_then_mutate(value, contract):
        copied = original_copy(value, contract)
        if value is snapshot:
            for encoding in snapshot.encodings:
                encoding.__dict__["observed_at_unix_ms"] = 999999
        return copied

    monkeypatch.setattr(encoders, "_strict_copy", copy_then_mutate)
    assert snapshot.control_deadline_unix_ms(now_unix_ms=1100) == 1250
    assert snapshot.control_deadline_unix_ms(now_unix_ms=1100) == 0


# 功能：
#   更新飞行状态不能给已损坏的历史几何快照重新签出有效散列。
# 输入：
#   无：函数创建标准契约夹具后故意破坏融合内容。
# 输出：
#   无：刷新入口应拒绝原快照的散列不一致。
def test_native_refresh_cannot_launder_a_mutated_snapshot():
    snapshot = complete_feature_snapshot()
    snapshot.encodings[0].features[0] = 0.75
    with pytest.raises(ValueError):
        encoders.refresh_flight_state_features(
            snapshot, snapshot.encodings[-1], captured_at_unix_ms=1000
        )


# 功能：
#   证明融合最多消费容量加一项，外部超长生成器不能占满实时循环。
# 输入：
#   role_stream：选择测试角色流或编码流。
# 输出：
#   无：应在第四项即拒绝，而非遍历完整流。
@pytest.mark.parametrize("role_stream", [False, True])
def test_fusion_rejects_overlong_iterators_with_bounded_consumption(role_stream):
    snapshot = complete_feature_snapshot()
    consumed = []

    # 功能：
    #   模拟永远还可能有下一项的来源，十项守卫防止错误实现令测试卡死。
    # 输入：
    #   无。
    # 输出：
    #   item：重复的角色或编码，每次产出均被记录。
    def stream():
        for index in range(10):
            consumed.append(index)
            item = "metric-geometry-encoder" if role_stream else snapshot.encodings[0]
            yield item
        raise AssertionError("encoder consumed beyond its input budget")

    with pytest.raises(ValueError, match="COLLECTION_LIMIT_EXCEEDED"):
        if role_stream:
            encoders.fuse_realtime_features(
                snapshot.encodings, captured_at_unix_ms=1000, required_roles=stream()
            )
        else:
            encoders.fuse_realtime_features(stream(), captured_at_unix_ms=1000)
    assert consumed == [0, 1, 2, 3]


# 功能：
#   防止错误时钟类型通过 Python 数值比较被当成真实毫秒。
# 输入：
#   clock：布尔、浮点、负值或空时钟。
# 输出：
#   无：单编码和融合的新鲜度均返回假，重新融合抛出明确错误。
@pytest.mark.parametrize("clock", [True, 1000.0, -1, None])
def test_freshness_requires_integer_clock(clock):
    snapshot = complete_feature_snapshot()
    assert snapshot.fresh_at(clock) is False
    assert snapshot.encodings[0].fresh_at(clock) is False
    with pytest.raises(ValueError, match="FUSION_CLOCK_INVALID"):
        encoders.fuse_realtime_features(snapshot.encodings, captured_at_unix_ms=clock)


# 功能：
#   验证直接调用几何编码器也严格检查 FOV 和采样字段，不依赖上游已经检查过。
# 输入：
#   无：函数使用独立内存中的距离扫描。
# 输出：
#   无：布尔视场和字符串置信度均应拒绝。
def test_geometry_rejects_coerced_measurements():
    with pytest.raises(ValueError):
        encoders.encode_metric_geometry(
            _scan(),
            sensor_mount=_mount(),
            expected_horizontal_fov_rad=True,
            encoded_at_unix_ms=1001,
        )
    scan = _scan()
    scan.samples[0].__dict__["confidence"] = "0.9"
    with pytest.raises(ValueError):
        encoders.encode_metric_geometry(scan, sensor_mount=_mount(), encoded_at_unix_ms=1001)


# 功能：
#   极端有限分量的范数仍可能溢出，不能归一化为零方向进入空间编码。
# 输入：
#   无：明确超出可表示范数的数值夹具。
# 输出：
#   无：范数计算抛出边界错误。
def test_overflowing_finite_direction_is_not_a_zero_direction():
    with pytest.raises(ValueError, match="VECTOR_MAGNITUDE_INVALID"):
        encoders._magnitude(Vector3(x=1.7e308, y=1.7e308, z=1.7e308))
