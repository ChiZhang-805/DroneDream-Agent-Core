"""Offline witness comparisons use synthetic captures, never product flight acceptance."""

import hashlib
import json

import numpy as np
import pytest
from test_localization_observations import _scan
from test_localization_truth_capture import contract, message, publish

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.localization_observations import GeometryObservationCapture
from dronedream_agent_core.localization_truth_audit import _witness_motion, compare_geometry_truth
from dronedream_agent_core.localization_truth_capture import LocalizationTruthCapture


def recordings(tmp_path, scene):
    """Create both streams through production writers with a synthetic, declared correction."""
    frames = contract()
    frames_path = tmp_path / "simulation-sensor-frames.json"
    publish(frames_path, frames)
    truth = LocalizationTruthCapture(tmp_path, frames=frames, summary_publisher=publish)
    for sec, mono in ((1, .99), (2, 1.03)):
        m = message(sec)
        m.pose[0].position.x = m.pose[0].position.y = 0.
        m.pose[0].position.z = -.228
        assert truth.record(m, received_monotonic=mono, received_unix_ms=int(mono*1000))
    assert truth.close()["complete"]
    capture = GeometryObservationCapture(tmp_path, map_sha256="a"*64, summary_publisher=publish)
    scan, kwargs = _scan()
    scan.body_position_world_enu_m = Vector3(x=.06, y=-.03, z=.02)
    if scene:
        kwargs["source_clock"] = {"publisher_simulation_time_ns": 10**9}
    assert capture.record(scan, **kwargs)
    assert capture.close()["complete"]
    # Test comparison arithmetic and provenance, not a second implementation of
    # registration. Production CLI obtains this candidate from the real solver.
    report = {"capture_sha256": hashlib.sha256(capture.path.read_bytes()).hexdigest(),
              "frames": [{"sequence": 1, "translation_fit": {"usable_candidate": True,
                          "correction_world_m": [-.06, .03, -.02]}}]}
    return capture.path, report, truth.path, frames_path


@pytest.mark.parametrize("scene", [False, True])
def test_comparison_preserves_clock_strength_and_uses_exact_declared_center(tmp_path, scene):
    inputs = recordings(tmp_path, scene)
    result = compare_geometry_truth(*inputs)
    assert result["compared_count"] == 1 and result["truth_frame_count"] == 2
    assert result["clock_comparison"] == (
        "publisher-simulation-time" if scene else "host-receipt-time")
    np.testing.assert_allclose(result["baseline_absolute_error_xyz_m"]["p50"], [.06, .03, .02])
    np.testing.assert_allclose(result["fitted_absolute_error_xyz_m"]["p95"], [0, 0, 0], atol=1e-12)
    assert not result["qualification_granted"] and not result["truth_used_as_control_input"]
    assert result["motion_strata"]["unclassified"]["compared_count"] == 1
    assert result["motion_strata"]["moving"]["fitted_absolute_error_xyz_m"] is None


def test_tampered_center_is_rejected_even_if_record_hash_was_recomputed(tmp_path):
    inputs = recordings(tmp_path, False)
    path = inputs[2]
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["collision_center_world_enu_m"][2] += .012
    rows[0].pop("record_sha256")
    rows[0]["record_sha256"] = sha256_json(rows[0])
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="CENTER_REFERENCE_MISMATCH"):
        compare_geometry_truth(*inputs)


def test_candidate_sequence_mismatch_is_not_silently_zipped(tmp_path):
    inputs = recordings(tmp_path, False)
    inputs[1]["frames"][0]["sequence"] = 2
    with pytest.raises(ValueError, match="SEQUENCE_MISMATCH"):
        compare_geometry_truth(*inputs)


def test_different_run_cannot_reuse_a_reset_simulation_clock(tmp_path):
    inputs = list(recordings(tmp_path, True))
    inputs[3] = tmp_path / "different-run" / inputs[3].name
    with pytest.raises(ValueError, match="SAME_RUN"):
        compare_geometry_truth(*inputs)


def test_joint_candidate_scores_attitude_without_refitting_to_witness(tmp_path):
    inputs = recordings(tmp_path, True)
    report = inputs[1]
    report["solver"] = "joint-position-attitude"
    fit = report["frames"][0].pop("translation_fit")
    fit["rotation_world_from_input"] = np.eye(3).tolist()
    report["frames"][0]["pose_fit"] = fit
    result = compare_geometry_truth(*inputs)
    assert result["compared_count"] == 1
    assert result["fitted_attitude_error_rad"]["maximum"] == 0.
    assert not result["truth_used_as_control_input"]
    fit["rotation_world_from_input"] = np.diag([1., 1., -1.]).tolist()
    with pytest.raises(ValueError, match="ROTATION_INVALID"):
        compare_geometry_truth(*inputs)


def test_motion_classification_uses_source_time_and_keeps_endpoints_unknown():
    centers = np.array([[0., 0., 0.], [.002, 0., 0.], [.004, 0., 0.]])
    rotations = [np.eye(3)]*3
    result = _witness_motion(centers, rotations, np.array([0., .02, .04]))
    assert [r["state"] for r in result] == ["unclassified", "moving", "unclassified"]
    assert result[1]["linear_speed_mps"] == pytest.approx(.1)
    assert _witness_motion(centers, rotations, [0., .2, .4])[1]["state"] == "unclassified"
    assert _witness_motion(centers*0, rotations, [0., .02, .04])[1]["state"] == "stationary"
    assert _witness_motion(centers*.3, rotations, [0., .02, .04])[1]["state"] == "slow"


def test_rotating_in_place_is_not_misclassified_as_stationary():
    from dronedream_agent_core.local_pose_alignment import rotation_exp_and_left_jacobian

    rotations = [rotation_exp_and_left_jacobian(np.array([0., 0., a]))[0]
                 for a in [0., .02, .04]]
    result = _witness_motion(np.zeros((3,3)), rotations, [0., .02, .04])
    assert result[1]["state"] == "moving"
    assert result[1]["linear_speed_mps"] == 0.
    assert result[1]["angular_speed_radps"] == pytest.approx(1.)


@pytest.mark.parametrize("field,value", [
    ("sequence", 1.5), ("publisher_simulation_time_ns", 1e9),
    ("received_at_unix_ms", 990.5), ("received_monotonic_seconds", 2**4096),
], ids=["fractional-sequence", "floating-nanoseconds", "fractional-unix", "overflowing-host"])
def test_rehashed_witness_requires_exact_clock_types(tmp_path, field, value):
    inputs = recordings(tmp_path, True)
    rows = [json.loads(line) for line in inputs[2].read_text().splitlines()]
    rows[0][field] = value
    rows[0].pop("record_sha256")
    rows[0]["record_sha256"] = sha256_json(rows[0])
    inputs[2].write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="CLOCK_INVALID"):
        compare_geometry_truth(*inputs)


@pytest.mark.parametrize("correction", [[False, 0, 0], ["0", 0, 0], [2**4096, 0, 0]])
def test_candidate_corrections_cannot_coerce_non_numeric_values(tmp_path, correction):
    inputs = recordings(tmp_path, True)
    inputs[1]["frames"][0]["translation_fit"]["correction_world_m"] = correction
    with pytest.raises(ValueError, match="CANDIDATE_INVALID"):
        compare_geometry_truth(*inputs)


def test_non_object_frame_contract_has_a_domain_error(tmp_path):
    inputs = recordings(tmp_path, True)
    inputs[3].write_text("[]")
    with pytest.raises(ValueError, match="FRAME_CONTRACT_INVALID"):
        compare_geometry_truth(*inputs)


def test_large_source_clock_preserves_small_matching_intervals(tmp_path):
    inputs = recordings(tmp_path, True)
    origin = 2**63

    def rewrite(path, transform):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            transform(row)
            row.pop("record_sha256")
            row["record_sha256"] = sha256_json(row)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    rewrite(inputs[2], lambda row: row.update(
        publisher_simulation_time_ns=origin + row["publisher_simulation_time_ns"] - 10**9))
    rewrite(inputs[0], lambda row: row.update(
        source_clock={"publisher_simulation_time_ns": origin + 20_000_001}))
    inputs[1]["capture_sha256"] = hashlib.sha256(inputs[0].read_bytes()).hexdigest()
    result = compare_geometry_truth(*inputs)
    assert result["frames"][0]["nearest_witness_time_gap_ms"] == pytest.approx(20.000001, abs=1e-9)
    assert result["compared_count"] == 0  # One nanosecond past the 20 ms matching budget.
    assert result["comparison_rejection_counts"] == {"source_time_gap_exceeded": 1}


# 功能：复现图像源时刻正确但到达延迟过大的情况，保留拒绝并明确诊断原因。
# 输入：tmp_path 中通过正式写入器生成的独立见证与图像样本。
# 输出：断言不能把接收延迟误报为拟合误差或成功；原始阈值保持不变。
def test_source_time_match_does_not_hide_receive_delay(tmp_path):
    inputs = recordings(tmp_path, True)
    path = inputs[0]
    row = json.loads(path.read_text())
    row["scan"]["observed_at_monotonic_seconds"] = 1.2
    row.pop("record_sha256")
    row["record_sha256"] = sha256_json(row)
    path.write_text(json.dumps(row) + "\n")
    inputs[1]["capture_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = compare_geometry_truth(*inputs)
    assert result["compared_count"] == 0
    assert result["comparison_rejection_counts"] == {"receive_time_gap_exceeded": 1}
    assert result["frames"][0]["nearest_witness_time_gap_ms"] == 0
    assert result["frames"][0]["nearest_witness_receive_gap_ms"] == pytest.approx(210.)
    assert result["baseline_absolute_error_xyz_m"] is None
    assert not result["qualification_granted"]


# 功能：源钟中点采用确定的较早见证，不能由浮点秒舍入改变接收时延判定。
# 输入：大源钟下相距四十毫秒的两个真值、恰在中点的图像。
# 输出：无；仍使用二十毫秒边界和原始接收时间，不放宽任何验收预算。
@pytest.mark.parametrize("origin", [0, 10**12, 2**63])
def test_exact_source_midpoint_uses_earlier_witness(tmp_path, origin):
    inputs = recordings(tmp_path, True)
    truth_path = inputs[2]
    rows = [json.loads(line) for line in truth_path.read_text().splitlines()]
    for index, row in enumerate(rows):
        row["publisher_simulation_time_ns"] = origin + 1_001_000_000 + index*40_000_000
        row.pop("record_sha256")
        row["record_sha256"] = sha256_json(row)
    truth_path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    capture_path = inputs[0]
    row = json.loads(capture_path.read_text())
    row["source_clock"]["publisher_simulation_time_ns"] = origin + 1_021_000_000
    row.pop("record_sha256")
    row["record_sha256"] = sha256_json(row)
    capture_path.write_text(json.dumps(row)+"\n")
    inputs[1]["capture_sha256"] = hashlib.sha256(capture_path.read_bytes()).hexdigest()
    result = compare_geometry_truth(*inputs)
    assert result["frames"][0]["nearest_witness_time_gap_ms"] == 20.
    expected_gap = abs(row["scan"]["observed_at_monotonic_seconds"]-.99)*1000
    assert result["frames"][0]["nearest_witness_receive_gap_ms"] == pytest.approx(expected_gap)
