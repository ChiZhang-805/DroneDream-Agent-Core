from __future__ import annotations

import pytest

from dronedream_agent_core.contracts import QuaternionWxyz, Vector3
from dronedream_agent_core.runtime_sensor_contracts import (
    RuntimeSensorContract,
    RuntimeSensorEnvelope,
    RuntimeSensorRegistry,
    forward_camera_motion_alignment,
    forward_rgb_can_inform_control,
    oakd_lite_depth_sensor_contract,
    oakd_lite_forward_rgb_runtime_contract,
)


# 功能：
#   创建具有明确校准、传输时限和质量门限的传感器契约夹具。
# 输入：
#   sensor_id：测试传感器标识。
#   required：是否为必需来源。
#   minimum_quality：最低质量门限。
# 输出：
#   contract：本例独立的来源契约。
def _contract(
    sensor_id: str = "forward-rgb",
    *,
    required: bool = True,
    minimum_quality: float = 0.5,
) -> RuntimeSensorContract:
    contract = RuntimeSensorContract(
        sensor_id=sensor_id,
        vehicle_id="dronedream.x500-depth",
        modality="rgb-camera" if "rgb" in sensor_id else "depth-camera",
        coordinate_frame="camera-optical",
        unit="uint8-rgb" if "rgb" in sensor_id else "meter",
        calibration_sha256="a" * 64,
        intrinsics_sha256="b" * 64,
        extrinsics_sha256="c" * 64,
        maximum_sample_age_seconds=0.2,
        maximum_transport_latency_seconds=0.05,
        minimum_quality=minimum_quality,
        minimum_coverage=0.4,
        required_for_motion=required,
    )
    return contract


# 功能：
#   根据指定契约构造内容与校准身份相同的样本元数据，时钟及质量由用例控制。
# 输入：
#   contract：样本所属契约。
#   sequence：来源序号。
#   sample_time：来源采样单调时刻。
#   receive_time：主机接收单调时刻。
#   quality：样本质量。
#   coverage：观测覆盖率。
# 输出：
#   envelope：用于注册表边界测试的元数据。
def _envelope(
    contract: RuntimeSensorContract,
    *,
    sequence: int = 1,
    sample_time: float = 10.0,
    receive_time: float = 10.01,
    quality: float = 0.9,
    coverage: float = 0.8,
) -> RuntimeSensorEnvelope:
    envelope = RuntimeSensorEnvelope(
        sensor_id=contract.sensor_id,
        vehicle_id=contract.vehicle_id,
        modality=contract.modality,
        sequence=sequence,
        sample_monotonic_seconds=sample_time,
        received_monotonic_seconds=receive_time,
        coordinate_frame=contract.coordinate_frame,
        unit=contract.unit,
        calibration_sha256=contract.calibration_sha256,
        intrinsics_sha256=contract.intrinsics_sha256,
        extrinsics_sha256=contract.extrinsics_sha256,
        covariance_diagonal=[0.01, 0.01, 0.02],
        quality=quality,
        coverage=coverage,
        payload_sha256="d" * 64,
    )
    return envelope


# 功能：
#   核对内置深度来源标识与校准量程仍对应当前固定安装契约。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_metric_mount_contract_remains_stable() -> None:
    mount = oakd_lite_depth_sensor_contract()

    assert mount.sensor_id == "oakd-lite-depth"
    assert mount.minimum_range_m == 0.2
    assert mount.maximum_range_m == 19.1


# 功能：
#   检查空契约集合或只有可选来源时，不能把没有检查对象当作运动信息已就绪。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_missing_required_sensor_set_cannot_be_ready():
    for contracts in ([], [_contract(required=False)]):
        registry = RuntimeSensorRegistry(contracts)
        assert not registry.snapshot(now_monotonic_seconds=10.).ready_for_motion


# 功能：
#   核对原对象和返回副本都不能改写注册表校准，并拒绝绕过模型构造的非法质量。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_registry_owns_calibration_and_revalidates_samples():
    original = _contract()
    registry = RuntimeSensorRegistry([original])
    digest = registry.contract_set_sha256
    original.minimum_quality = 0.
    registry.contract(original.sensor_id).minimum_quality = 0.
    assert registry.contract_set_sha256 == digest
    assert registry.contract(original.sensor_id).minimum_quality == .5
    forged = _envelope(registry.contract(original.sensor_id)).model_copy(update={"quality": 9.})
    with pytest.raises(ValueError):
        registry.ingest(forged, now_monotonic_seconds=10.02)


# 功能：
#   检查缺少质量或覆盖字段时保留未知的低质量状态，不补成完美观测。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_unknown_quality_and_coverage_do_not_default_to_perfect():
    contract = _contract()
    payload = _envelope(contract).model_dump()
    del payload["quality"], payload["coverage"]
    registry = RuntimeSensorRegistry([contract])
    registry.ingest(RuntimeSensorEnvelope.model_validate(payload), now_monotonic_seconds=10.02)
    status = registry.status(contract.sensor_id, now_monotonic_seconds=10.03)
    assert status.health == "degraded"
    assert status.quality == status.coverage == 0.


# 功能：
#   检查查询时钟回退后，原本合法的样本会标记为未来数据而非继续健康。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_read_clock_rollback_cannot_make_future_data_healthy():
    contract = _contract()
    registry = RuntimeSensorRegistry([contract])
    registry.ingest(_envelope(contract), now_monotonic_seconds=10.02)
    assert "SENSOR_TIMESTAMP_IN_FUTURE" in registry.status(
        contract.sensor_id, now_monotonic_seconds=9.).issue_codes


# 功能：
#   对照前进、后退与悬停状态，核对前视方向要求只约束具有水平运动的路径观察模式。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_forward_camera_alignment_rejects_backward_motion_but_not_hover() -> None:
    orientation = QuaternionWxyz(w=1.0, x=0.0, y=0.0, z=0.0)

    forward = forward_camera_motion_alignment(
        body_orientation_world_from_body=orientation,
        body_velocity_world_enu_mps=Vector3(x=1.0, y=0.0, z=0.0),
    )
    backward = forward_camera_motion_alignment(
        body_orientation_world_from_body=orientation,
        body_velocity_world_enu_mps=Vector3(x=-0.04, y=0.0, z=0.0),
    )
    hover = forward_camera_motion_alignment(
        body_orientation_world_from_body=orientation,
        body_velocity_world_enu_mps=Vector3(x=0.0, y=0.0, z=0.2),
    )

    assert forward.alignment_cosine == pytest.approx(1.0)
    assert forward.aligned_for_forward_navigation is True
    assert backward.alignment_cosine == pytest.approx(-1.0)
    assert backward.minimum_alignment_speed_mps == pytest.approx(0.03)
    assert backward.aligned_for_forward_navigation is False
    assert backward.issue_codes == ["FORWARD_CAMERA_NOT_MOTION_ALIGNED"]
    assert hover.alignment_required is False
    assert hover.aligned_for_forward_navigation is True


# 功能：
#   检查新鲜、身份相符且质量足够的必需来源会出现在内容绑定快照中。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_registry_accepts_latest_content_bound_sample_and_reports_ready() -> None:
    contract = _contract()
    registry = RuntimeSensorRegistry([contract])

    registry.ingest(_envelope(contract), now_monotonic_seconds=10.02)
    snapshot = registry.snapshot(now_monotonic_seconds=10.03)

    assert snapshot.ready_for_motion is True
    assert snapshot.active_sensor_ids == ["forward-rgb"]
    assert snapshot.statuses[0].health == "healthy"
    assert snapshot.statuses[0].payload_sha256 == "d" * 64
    assert registry.latest("forward-rgb") is not None


# 功能：
#   检查重复序号被拒绝，新序号替换最新元数据而不积累旧帧队列。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_registry_is_depth_one_and_rejects_nonmonotonic_sequence() -> None:
    contract = _contract()
    registry = RuntimeSensorRegistry([contract])
    registry.ingest(_envelope(contract), now_monotonic_seconds=10.02)

    with pytest.raises(ValueError, match="SENSOR_SEQUENCE_NOT_MONOTONIC"):
        registry.ingest(_envelope(contract), now_monotonic_seconds=10.02)

    registry.ingest(
        _envelope(contract, sequence=2, sample_time=10.03, receive_time=10.04),
        now_monotonic_seconds=10.05,
    )
    assert registry.latest(contract.sensor_id).sequence == 2  # type: ignore[union-attr]


# 功能：
#   检查序号递增不能掩盖来源或接收时钟回退，失败后保留上一条有效样本。
# 输入：
#   sample_time：本例采样时刻。
#   receive_time：本例接收时刻。
#   issue：预期的时钟回退问题代码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    ("sample_time", "receive_time", "issue"),
    [
        (9.98, 10.02, "SENSOR_SAMPLE_TIMESTAMP_REGRESSION"),
        (9.995, 9.999, "SENSOR_RECEIVE_TIMESTAMP_REGRESSION"),
    ],
)
def test_registry_rejects_timestamp_rollback_even_when_sequence_increases(
    sample_time: float,
    receive_time: float,
    issue: str,
) -> None:
    contract = _contract()
    registry = RuntimeSensorRegistry([contract])
    registry.ingest(_envelope(contract), now_monotonic_seconds=10.02)

    with pytest.raises(ValueError, match=issue):
        registry.ingest(
            _envelope(
                contract,
                sequence=2,
                sample_time=sample_time,
                receive_time=receive_time,
            ),
            now_monotonic_seconds=10.03,
        )

    assert registry.latest(contract.sensor_id).sequence == 1  # type: ignore[union-attr]


# 功能：
#   核对契约允许范围内的小幅时钟抖动不会被误当作回退，仍要求序号前进。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_registry_allows_bounded_timestamp_jitter_with_increasing_sequence() -> None:
    contract = _contract()
    registry = RuntimeSensorRegistry([contract], maximum_clock_skew_seconds=0.01)
    registry.ingest(_envelope(contract), now_monotonic_seconds=10.02)

    registry.ingest(
        _envelope(
            contract,
            sequence=2,
            sample_time=9.995,
            receive_time=10.005,
        ),
        now_monotonic_seconds=10.02,
    )

    assert registry.latest(contract.sensor_id).sequence == 2  # type: ignore[union-attr]


# 功能：
#   检查飞行器、坐标系或校准摘要变化时不能沿用已登记来源的身份。
# 输入：
#   mutation：对样本身份字段的修改。
#   issue：预期拒绝代码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    ("mutation", "issue"),
    [
        ({"vehicle_id": "other-vehicle"}, "SENSOR_SAMPLE_CONTRACT_MISMATCH"),
        ({"coordinate_frame": "world-enu"}, "SENSOR_SAMPLE_CONTRACT_MISMATCH"),
        ({"calibration_sha256": "e" * 64}, "SENSOR_SAMPLE_CONTRACT_MISMATCH"),
    ],
)
def test_registry_rejects_identity_or_calibration_mismatch(
    mutation: dict[str, object], issue: str
) -> None:
    contract = _contract()
    registry = RuntimeSensorRegistry([contract])
    envelope = _envelope(contract).model_copy(update=mutation)

    with pytest.raises(ValueError, match=issue):
        registry.ingest(envelope, now_monotonic_seconds=10.02)


# 功能：
#   分别验证传输超时与总体观测过期，避免只检查其中一个时限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_registry_rejects_stale_and_transport_delayed_samples() -> None:
    contract = _contract()
    registry = RuntimeSensorRegistry([contract])

    with pytest.raises(ValueError, match="SENSOR_TRANSPORT_LATENCY_EXCEEDED"):
        registry.ingest(
            _envelope(contract, receive_time=10.08),
            now_monotonic_seconds=10.09,
        )
    with pytest.raises(ValueError, match="SENSOR_SAMPLE_STALE_AT_INGEST"):
        registry.ingest(
            _envelope(contract, receive_time=10.01),
            now_monotonic_seconds=10.25,
        )


# 功能：
#   检查必需来源未收到、质量低或随后过期时都不能报告运动信息已就绪。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_snapshot_fails_closed_for_missing_stale_or_low_quality_required_sensor() -> None:
    required = _contract()
    optional = _contract("rear-depth", required=False, minimum_quality=0.0)
    registry = RuntimeSensorRegistry([required, optional])

    missing = registry.snapshot(now_monotonic_seconds=10.0)
    assert missing.ready_for_motion is False
    assert "forward-rgb:SENSOR_SAMPLE_NOT_RECEIVED" in missing.issue_codes

    registry.ingest(
        _envelope(required, quality=0.4),
        now_monotonic_seconds=10.02,
    )
    degraded = registry.snapshot(now_monotonic_seconds=10.03)
    assert degraded.ready_for_motion is False
    assert degraded.statuses[0].health == "degraded"

    stale = registry.snapshot(now_monotonic_seconds=10.25)
    assert stale.ready_for_motion is False
    assert "forward-rgb:SENSOR_SAMPLE_STALE" in stale.issue_codes


# 功能：
#   检查可选引导来源退化仍显示诊断，但不覆盖健康必需控制来源的状态。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_degraded_guidance_sensor_does_not_mask_healthy_control_sensor() -> None:
    depth = _contract("forward-depth", required=True, minimum_quality=0.5)
    rgb = _contract("forward-rgb", required=False, minimum_quality=0.5)
    registry = RuntimeSensorRegistry([depth, rgb])

    registry.ingest(_envelope(depth, quality=0.9), now_monotonic_seconds=10.02)
    registry.ingest(_envelope(rgb, quality=0.0), now_monotonic_seconds=10.02)
    snapshot = registry.snapshot(now_monotonic_seconds=10.03)

    assert snapshot.ready_for_motion is True
    assert [status.health for status in snapshot.statuses] == ["healthy", "degraded"]
    assert "forward-rgb:SENSOR_QUALITY_BELOW_CONTRACT" in snapshot.issue_codes


# 功能：
#   检查相同来源标识不能重新登记为不同的单位或校准契约。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_conflicting_contract_registration_is_rejected() -> None:
    contract = _contract()
    registry = RuntimeSensorRegistry([contract])
    changed = contract.model_copy(update={"unit": "bgr8"})

    with pytest.raises(ValueError, match="SENSOR_CONTRACT_IDENTITY_CHANGED"):
        registry.register_contract(changed)


# 功能：
#   检查超过快照支持数量的新传感器在登记时就被拒绝，而非留下无法读取的内部状态。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_registry_capacity_failure_preserves_readable_state():
    registry = RuntimeSensorRegistry([_contract(f"sensor-{i}") for i in range(64)])
    before = registry.contract_set_sha256
    with pytest.raises(ValueError, match="LIMIT"):
        registry.register_contract(_contract("overflow"))
    assert registry.contract_set_sha256 == before
    assert len(registry.snapshot(now_monotonic_seconds=10.).statuses) == 64


# 功能：
#   检查大量传感器同时退化时仍可读取快照，摘要截断须显式标记且保留各传感器完整诊断。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_registry_many_degraded_sensors_keep_complete_statuses():
    contracts = [_contract(f"sensor-{i}") for i in range(64)]
    registry = RuntimeSensorRegistry(contracts)
    for contract in contracts:
        registry.ingest(_envelope(contract, quality=0., coverage=0.), now_monotonic_seconds=10.02)
    snapshot = registry.snapshot(now_monotonic_seconds=10.3)
    assert not snapshot.ready_for_motion
    assert len(snapshot.statuses) == 64
    assert len(snapshot.issue_codes) <= 64
    assert "SENSOR_ISSUE_CODES_TRUNCATED" in snapshot.issue_codes
    assert all(len(status.issue_codes) == 3 for status in snapshot.statuses)


# 功能：
#   检查空注册表也校验查询时钟，不能将布尔值或字符串自动转成合法时间。
# 输入：
#   value：非法查询时钟。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, "10", float("nan"), float("inf"), -1.])
def test_empty_registry_rejects_invalid_clock(value):
    with pytest.raises(ValueError):
        RuntimeSensorRegistry().snapshot(now_monotonic_seconds=value)


# 功能：
#   检查相机契约的像素尺寸、视场和必需来源标记拒绝类型混淆。
# 输入：
#   field：被替换的契约参数。
#   value：非法参数值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [("width", 32.5), ("height", 64.),
    ("horizontal_fov_rad", True), ("required_for_motion", "false")])
def test_rgb_contract_rejects_coerced_dimensions_and_flags(field, value):
    values = {"vehicle_id": "test-aircraft", "width": 640, "height": 480}
    values[field] = value
    with pytest.raises(ValueError):
        oakd_lite_forward_rgb_runtime_contract(**values)


# 功能：
#   检查连续控制模式必须是布尔值，字符串 false 不能作为真值跳过相机方向要求。
# 输入：
#   mode：非法的模式值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mode", ["false", 1, None])
def test_camera_information_mode_cannot_use_truthiness(mode):
    alignment = forward_camera_motion_alignment(
        body_orientation_world_from_body=QuaternionWxyz(w=1., x=0., y=0., z=0.),
        body_velocity_world_enu_mps=Vector3(x=-1., y=0., z=0.),
    )
    with pytest.raises(ValueError):
        forward_rgb_can_inform_control(alignment, continuous_control=mode)
