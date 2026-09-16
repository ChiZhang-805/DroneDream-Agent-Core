"""Native ray replay is data evidence, never actuation or flight qualification."""

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from dronedream_agent_core.contracts import QuaternionWxyz, RawMetricRangeScan, Vector3
from dronedream_agent_core.depth_sensor_binding import DepthSensorBinding
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.localization_observations import (
    GeometryObservationCapture,
    encode_geometry_record,
)
from dronedream_agent_core.runtime_sensor_contracts import (
    RuntimeSensorRegistry,
    oakd_lite_depth_sensor_contract,
)


# 功能：
#   将采集回执写入测试私有目录，不发布真实运行状态。
# 输入：
#   path：测试回执路径。
#   value：待保存的回执对象。
# 输出：
#   None：不返回业务数据。
def _publish(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


# 功能：
#   通过真实深度校准及区块投影构造带窄障碍的合成扫描，不启动仿真传感器。
# 输入：
#   无。
# 输出：
#   fixture：扫描以及安装、校准和来源绑定参数组成的元组。
def _scan():
    values = [2.] * (160 * 120)
    values[41 * 160 + 65] = 1.25
    projection = DepthSensorBinding(RuntimeSensorRegistry(), vehicle_id="quad").project(
        SimpleNamespace(width=160, height=120, step=640, pixel_format_type=13,
                        data=struct.pack(f"<{len(values)}f", *values)))
    mount = oakd_lite_depth_sensor_contract()
    zero = Vector3(x=0, y=0, z=0)
    scan = RawMetricRangeScan(sensor_id=mount.sensor_id, sequence=1,
        observed_at_unix_ms=1000, observed_at_monotonic_seconds=1.,
        body_position_world_enu_m=zero, body_velocity_world_enu_mps=zero,
        body_orientation_world_from_body=QuaternionWxyz(w=1, x=0, y=0, z=0),
        localization_covariance_m2=.0535, source_coverage=projection.source_coverage,
        samples=list(projection.samples))
    fixture = scan, {"mount": mount, "calibration_sha256": projection.calibration_sha256,
                     "native_pose_binding_sha256": "a" * 64, "source_clock": None}
    return fixture


# 功能：
#   核对入队快照、摘要、射线覆盖及显式无权限字段，输入后续变化不能改写记录。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_measured_depth_rays_replay_without_truth_or_motion_authority(tmp_path):
    capture = GeometryObservationCapture(tmp_path, map_sha256="b" * 64,
                                          summary_publisher=_publish)
    scan, kwargs = _scan()
    expected = scan.model_dump(mode="json")
    assert capture.record(scan, **kwargs)
    scan.samples[0].range_m = 123  # Must not rewrite the queued snapshot.
    summary = capture.close()
    assert summary["complete"] and summary["completed_count"] == 1
    record = json.loads(capture.path.read_text(encoding="utf-8"))
    digest = record.pop("record_sha256")
    assert digest == sha256_json(record)
    assert record["scan"] == expected
    assert record["scan"]["localization_covariance_m2"] == .0535
    assert record["source_clock"] is None  # No invented scene-clock proof.
    assert record["truth_correction_applied"] is False
    assert record["motion_permission_granted"] is False
    assert record["model_control_qualification_granted"] is False
    assert len(record["scan"]["samples"]) == 300
    assert min(s["range_m"] for s in record["scan"]["samples"]) < 1.5
    assert len(capture.path.read_bytes()) < 128 * 1024
    assert capture.close()["complete"]  # Cleanup callback is idempotent.


# 功能：
#   验证重复来源、过快采样与配额用完分别计数，不把跳过记录计为已接收。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_capture_bounds_rate_and_total_storage_without_labeling_skips_as_success(tmp_path):
    capture = GeometryObservationCapture(tmp_path, map_sha256="b" * 64,
        summary_publisher=_publish, maximum_records=2)
    scan, kwargs = _scan()
    assert capture.record(scan, **kwargs)
    assert not capture.record(scan, **kwargs)
    scan.sequence, scan.observed_at_monotonic_seconds = 2, 1.1
    assert not capture.record(scan, **kwargs)
    scan.sequence, scan.observed_at_monotonic_seconds = 3, 1.3
    assert capture.record(scan, **kwargs)
    scan.sequence, scan.observed_at_monotonic_seconds = 4, 1.6
    assert not capture.record(scan, **kwargs)
    summary = capture.close()
    assert summary["complete"] and summary["quota_reached"]
    assert summary["accepted_count"] == 2
    assert summary["sampling_skipped_count"] == 2
    assert summary["quota_skipped_count"] == 1
    assert summary["maximum_jsonl_bytes"] == 256 * 1024


# 功能：
#   注入时钟回退、错误绑定或超量射线，确认采集失败锁定为不完整。
# 输入：
#   tmp_path：测试私有目录。
#   fault：注入的错误类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["clock", "binding", "mount", "rays"])
def test_malformed_capture_cannot_silently_claim_complete(tmp_path, fault):
    capture = GeometryObservationCapture(tmp_path, map_sha256="b" * 64,
                                          summary_publisher=_publish)
    scan, kwargs = _scan()
    if fault == "clock":
        assert capture.record(scan, **kwargs)
        scan.sequence, scan.observed_at_monotonic_seconds = 2, .9
    elif fault == "binding":
        kwargs["native_pose_binding_sha256"] = "not-a-digest"
    elif fault == "mount":
        scan.sensor_id = "different-sensor"
    else:
        scan.samples *= 2
    assert not capture.record(scan, **kwargs)
    assert not capture.close()["complete"]


# 功能：
#   已存在的观测文件应阻止新采集器初始化，原始内容不得改变。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_existing_capture_is_never_overwritten(tmp_path):
    path = tmp_path / "native-geometry-observations.jsonl"
    path.write_text("preserved", encoding="utf-8")
    with pytest.raises(FileExistsError):
        GeometryObservationCapture(tmp_path, map_sha256="b" * 64, summary_publisher=_publish)
    assert path.read_text(encoding="utf-8") == "preserved"


# 功能：
#   验证后台序列化仍独立限制单条记录大小，不能依赖生产者采样总量代替字节预算。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_background_serializer_enforces_individual_record_byte_cap():
    with pytest.raises(ValueError, match="RECORD_TOO_LARGE"):
        encode_geometry_record({"unbounded": "x" * (128 * 1024)})


# 功能：
#   模拟存在性预检过期，验证独占创建仍阻止两个所有者领取同一组文件。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：临时覆盖存在性预检。
# 输出：
#   None：不返回业务数据。
def test_stale_exists_check_cannot_create_two_capture_owners(tmp_path, monkeypatch):
    first = GeometryObservationCapture(tmp_path, map_sha256="b" * 64,
                                       summary_publisher=_publish)
    second = None
    try:
        monkeypatch.setattr(Path, "exists", lambda self: False)
        with pytest.raises(FileExistsError):
            second = GeometryObservationCapture(tmp_path, map_sha256="b" * 64,
                                                summary_publisher=_publish)
    finally:
        if second is not None:
            second.close()
        first.close()


# 功能：
#   绕过模型赋值校验后注入坏值，确认采集入口重新校验且不产生成功计数。
# 输入：
#   tmp_path：测试私有目录。
#   mutation：副本中替换的非法字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", [
    {"sequence": True}, {"observed_at_monotonic_seconds": float("nan")},
    {"observed_at_monotonic_seconds": 2**4096},
    {"localization_covariance_m2": -1},
])
def test_model_copy_bypass_is_rejected_before_recording(tmp_path, mutation):
    scan, kwargs = _scan()
    capture = GeometryObservationCapture(tmp_path, map_sha256="b" * 64,
                                         summary_publisher=_publish)
    try:
        assert not capture.record(scan.model_copy(update=mutation), **kwargs)
    finally:
        summary = capture.close()
    assert not summary["complete"]
    assert summary["accepted_count"] == 0
