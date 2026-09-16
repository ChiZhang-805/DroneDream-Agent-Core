import json

import numpy as np
import pytest

from dronedream_agent_core.depth_projection import DepthProjectionCalibration
from dronedream_agent_core.geometry_motion_audit import compare_moving_fixture, nearest_source_pose
from dronedream_agent_core.geometry_motion_fixture import bytes_digest


# 功能：
#   写入可被故意破坏的单元测试 JSON，不用于生产证据发布。
# 输入：
#   path：测试临时目录中的目标文件。
#   value：待序列化的测试对象。
# 输出：
#   None：不返回业务数据。
def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


# 功能：
#   构造单平面深度、已知姿态及配套摘要的合成单元资料，不作为 Gazebo 验收证据。
# 输入：
#   tmp_path：pytest 为当前用例隔离的临时目录。
# 输出：
#   result：合成采集目录和对应地图语义路径。
@pytest.fixture
def synthetic_fixture(tmp_path):
    from dataclasses import asdict

    (tmp_path / "raw").mkdir()
    semantic = tmp_path / "semantic.json"
    write_json(semantic, {"collision_primitives": [{"center_x": 2.05, "center_y": 0.,
        "center_z": 0., "size_x": .1, "size_y": 10., "size_z": 10.}]})
    poses = [{"simulation_time_ns": stamp, "position_m": [0., i*.4, 0.],
              "orientation_wxyz": [1., 0., 0., 0.]}
             for i, stamp in enumerate([0, 500_000_000, 1_000_000_000])]
    frames = []
    raw = np.full((48, 64), 2.-.13233, dtype="<f4").tobytes()
    for i, pose in enumerate(poses):
        path = f"raw/{i:04d}.depth.f32"
        (tmp_path / path).write_bytes(raw)
        frames.append({"path": path, "bytes": len(raw), "sha256": bytes_digest(raw),
            "simulation_time_ns": pose["simulation_time_ns"], "width": 64, "height": 48,
            "step": 64*4, "pixel_format": "R_FLOAT32"})
    for name, value in (("poses.json", poses), ("frames.json", frames), ("commands.json", [])):
        write_json(tmp_path / name, value)
    calibration = DepthProjectionCalibration(64, 48, 1.274, .2, 19.1, sample_stride_pixels=3,
                                             no_return_mode="gazebo-far-clip")
    report = {"schema": "dronedream.camera-motion-calibration", "complete": True,
        "measured_motion": True, "native_estimator_qualification": False,
        "covariance_qualified": False, "model_control_qualification": False,
        "actuator_commands_sent": False, "pose_count": len(poses), "frame_count": len(frames),
        "measured_position_span_m": [0., .8, 0.], "sources": {"semantic": {
            "sha256": bytes_digest(semantic.read_bytes())}}, "calibration": asdict(calibration),
        "calibration_sha256": calibration.sha256, "files": {name:
            bytes_digest((tmp_path / name).read_bytes())
            for name in ("poses.json", "frames.json", "commands.json")}}
    write_json(tmp_path / "capture.json", report)
    result = tmp_path, semantic
    return result


# 功能：
#   核对源时间配对拒绝外推及超差姿态，只接受窗口内最近观测。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_nearest_source_pair_never_extrapolates_or_uses_distant_pose():
    poses, times = [{"id": 1}, {"id": 2}], np.array([20_000_000, 40_000_000])
    assert nearest_source_pose(poses, times, 19_000_000) is None
    assert nearest_source_pose(poses, times, 41_000_000) is None
    assert nearest_source_pose(poses, times, 30_000_000) is None
    assert nearest_source_pose(poses, times, 24_000_000) == poses[0]


# 功能：
#   单平面只约束法向平移，核对切向误差保留且几何比较不授予协方差资质。
# 输入：
#   synthetic_fixture：合成单平面采集资料。
# 输出：
#   None：不返回业务数据。
def test_audit_fits_observed_plane_but_does_not_qualify_covariance(synthetic_fixture):
    root, semantic = synthetic_fixture
    result = compare_moving_fixture(root, semantic)
    exact = result["summary"]["exact_attitude"]
    assert exact["usable_candidates"] == 3
    assert exact["observed_translation_ranks"] == {1: 3}
    assert exact["observed_subspace_error_norm_m"]["max"] < 1e-6
    assert exact["corrected_absolute_error_xyz_p95_m"][1:] == pytest.approx([.03, .02])
    assert result["covariance_qualified"] is False
    assert result["truth_used_to_construct_corrupted_fixture_input"] is True
    assert result["summary"]["delayed_pose_100ms"]["usable_candidates"] == 0


# 功能：
#   验证替换原始深度字节而不更新摘要时，比较过程拒绝继续。
# 输入：
#   synthetic_fixture：合成采集资料。
# 输出：
#   None：不返回业务数据。
def test_rejects_raw_pixel_digest_mismatch(synthetic_fixture):
    root, semantic = synthetic_fixture
    (root / "raw/0000.depth.f32").write_bytes(bytes(64*48*4))
    with pytest.raises(ValueError, match="FRAME_IDENTITY"):
        compare_moving_fixture(root, semantic)


# 功能：
#   核对联合姿态求解显式开启，保留退化秩及非飞行验收边界，拒绝字符串模式。
# 输入：
#   synthetic_fixture：单平面几何测试资料。
# 输出：
#   None：不返回业务数据。
def test_joint_pose_mode_is_explicit_and_retains_degeneracy_and_scope(synthetic_fixture):
    root, semantic = synthetic_fixture
    result = compare_moving_fixture(root, semantic, joint_pose=True)
    assert result["solver"] == "joint-position-attitude"
    assert result["covariance_qualified"] is False
    assert result["model_control_qualification"] is False
    for condition in ("exact_attitude", "biased_attitude"):
        assert result["summary"][condition]["usable_candidates"] == 3
        assert result["summary"][condition]["observed_pose_ranks"] == {3: 3}
        assert result["summary"][condition]["observed_translation_ranks"] == {1: 3}
    with pytest.raises(ValueError, match="MODE_INVALID"):
        compare_moving_fixture(root, semantic, joint_pose="true")


# 功能：
#   改写测试文件并同步摘要，用来确认摘要吻合不能代替内容合法性校验。
# 输入：
#   root：隔离测试目录。
#   filename：拟修改的采集文件名。
#   change：对解析对象进行原地修改的测试函数。
# 输出：
#   None：不返回业务数据。
def mutate_file(root, filename, change):
    value = json.loads((root / filename).read_bytes())
    change(value)
    write_json(root / filename, value)
    report = json.loads((root / "capture.json").read_bytes())
    report["files"][filename] = bytes_digest((root / filename).read_bytes())
    write_json(root / "capture.json", report)


# 功能：
#   核对文件摘要合法时仍拒绝越出采集目录的原始图像路径。
# 输入：
#   synthetic_fixture：合成采集资料。
# 输出：
#   None：不返回业务数据。
def test_rejects_path_escape_even_with_valid_manifest_hash(synthetic_fixture):
    root, semantic = synthetic_fixture
    mutate_file(root, "frames.json", lambda v: v[0].update(path="../secret"))
    with pytest.raises(ValueError, match="FRAME_PATH"):
        compare_moving_fixture(root, semantic)


# 功能：
#   核对姿态文件即使重新签入摘要，也必须保持严格递增的源时间。
# 输入：
#   synthetic_fixture：合成采集资料。
# 输出：
#   None：不返回业务数据。
def test_rejects_reordered_source_time_even_with_valid_manifest_hash(synthetic_fixture):
    root, semantic = synthetic_fixture
    mutate_file(root, "poses.json", lambda v: v.reverse())
    with pytest.raises(ValueError, match="TIME_NOT_PROGRESSING"):
        compare_moving_fixture(root, semantic)


# 功能：
#   拒绝未完成采集及越界宣称原生定位、协方差或执行机构资质的资料。
# 输入：
#   synthetic_fixture：合成采集资料。
#   field：拟篡改的资质或完成字段。
#   value：替换后的字段值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [("complete", False), ("covariance_qualified", True),
    ("native_estimator_qualification", True), ("actuator_commands_sent", True)])
def test_rejects_incomplete_or_wrong_scope_capture(synthetic_fixture, field, value):
    root, semantic = synthetic_fixture
    report = json.loads((root / "capture.json").read_bytes())
    report[field] = value
    write_json(root / "capture.json", report)
    with pytest.raises(ValueError, match="COMPLETE_CAMERA_ONLY"):
        compare_moving_fixture(root, semantic)


# 功能：
#   显式允许分析失败采集时，输出仍保留失败状态及清理错误，不升级为飞行证明。
# 输入：
#   synthetic_fixture：合成采集资料。
# 输出：
#   None：不返回业务数据。
def test_partial_capture_analysis_preserves_failed_status(synthetic_fixture):
    root, semantic = synthetic_fixture
    report = json.loads((root / "capture.json").read_bytes())
    report.update(complete=False, close_errors=["TEST_SHUTDOWN_FAILURE"])
    write_json(root / "capture.json", report)
    result = compare_moving_fixture(root, semantic, allow_incomplete_capture=True)
    assert result["capture_complete"] is False
    assert result["capture_close_errors"] == ["TEST_SHUTDOWN_FAILURE"]
    assert result["model_control_qualification"] is False


# 功能：
#   即使摘要与文件吻合，JSON 重复字段和非有限值也不能进入离线审计。
# 输入：
#   synthetic_fixture：独立生成的单元测试资料。
#   case：重复字段或非有限字段场景。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("case", ["duplicate", "nan", "infinity", "overflow"])
def test_audit_rejects_ambiguous_or_nonfinite_json(synthetic_fixture, case):
    root, semantic = synthetic_fixture
    path = root / "capture.json"
    raw = path.read_bytes()
    field = {"duplicate": b'"complete":false,', "nan": b'"extra":NaN,',
             "infinity": b'"extra":Infinity,', "overflow": b'"extra":1e999,'}[case]
    path.write_bytes(b"{" + field + raw[1:])
    with pytest.raises(ValueError):
        compare_moving_fixture(root, semantic)


# 功能：
#   检查输入姿态不能通过浮点转换把布尔值或数字字符串伪装成坐标。
# 输入：
#   synthetic_fixture：单元测试采集资料。
#   value：非法坐标分量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, "0.0"])
def test_audit_rejects_pose_numeric_coercion(synthetic_fixture, value):
    root, semantic = synthetic_fixture
    mutate_file(root, "poses.json", lambda poses: poses[0].update(position_m=[value, 0., 0.]))
    with pytest.raises(ValueError):
        compare_moving_fixture(root, semantic)


# 功能：
#   未配对图像同样必须通过完整字节布局校验，不能借时间不匹配绕过检查。
# 输入：
#   synthetic_fixture：单元测试采集资料。
#   mutation：错误的行跨度、分辨率类型或文件字节数声明。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", [{"step": 999}, {"width": 64.}, {"bytes": 12288.}])
def test_unmatched_frame_still_requires_valid_layout(synthetic_fixture, mutation):
    root, semantic = synthetic_fixture
    mutate_file(root, "frames.json", lambda frames: frames[0].update(
        simulation_time_ns=20_000_000, **mutation))
    with pytest.raises(ValueError):
        compare_moving_fixture(root, semantic)


# 功能：
#   全部射线未命中时保留失败帧及原因，不让一张空场图像中止整份比较报告。
# 输入：
#   synthetic_fixture：单元测试采集资料。
#   depth：全图填充的深度值。
#   expected_issue：应当保留的无命中或无有效像素原因。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("depth,expected_issue", [(np.inf, "NO_METRIC_HITS"),
    (np.nan, "NO_VALID_METRIC_PIXELS"), (0., "NO_VALID_METRIC_PIXELS")])
def test_no_hit_frame_remains_visible_in_report(synthetic_fixture, depth, expected_issue):
    root, semantic = synthetic_fixture
    raw = np.full((48, 64), depth, dtype="<f4").tobytes()
    (root / "raw/0000.depth.f32").write_bytes(raw)
    mutate_file(root, "frames.json", lambda frames: frames[0].update(sha256=bytes_digest(raw)))
    result = compare_moving_fixture(root, semantic)
    assert result["frames"][0]["issue"] == expected_issue
    assert result["invalid_geometry_frames"] == 1 and result["unmatched_frames"] == 0
    assert result["summary"]["exact_attitude"]["usable_candidates"] == 2


# 功能：
#   允许分析未完成采集不等于允许非布尔完成状态。
# 输入：
#   synthetic_fixture：单元测试采集资料。
# 输出：
#   None：不返回业务数据。
def test_partial_mode_still_rejects_nonboolean_completion(synthetic_fixture):
    root, semantic = synthetic_fixture
    report = json.loads((root / "capture.json").read_bytes())
    report["complete"] = "false"
    write_json(root / "capture.json", report)
    with pytest.raises(ValueError):
        compare_moving_fixture(root, semantic, allow_incomplete_capture=True)


# 功能：
#   即使内参摘要合法，审计量程也必须与当前米制安装契约相同，避免两种边界并行使用。
# 输入：
#   synthetic_fixture：单元测试采集资料。
# 输出：
#   None：不返回业务数据。
def test_audit_requires_current_mount_range(synthetic_fixture):
    root, semantic = synthetic_fixture
    report = json.loads((root / "capture.json").read_bytes())
    report["calibration"]["maximum_depth_m"] = 20.
    report["calibration_sha256"] = DepthProjectionCalibration(**report["calibration"]).sha256
    write_json(root / "capture.json", report)
    with pytest.raises(ValueError, match="RANGE_MISMATCH"):
        compare_moving_fixture(root, semantic)
