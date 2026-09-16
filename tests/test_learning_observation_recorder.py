"""Algorithm fixtures, not physical demonstration evidence."""

import threading
from dataclasses import replace

import pytest
from control_fixtures import complete_feature_snapshot
from test_executed_control_training import teacher_evidence

from dronedream_agent_core.contracts import (
    OnboardPerceptionFrame,
    PerceptionFusionHealth,
    QuaternionWxyz,
    RangeRayObservation,
    Vector3,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.learning_observation_recorder import LearningObservationRecorder
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.navigation_snapshot import (
    NavigationSnapshotRequest,
    compile_navigation_snapshot,
)
from dronedream_agent_core.realtime_feature_encoders import fuse_realtime_features
from dronedream_agent_core.simulation_teacher import teacher_heading_rate, teacher_input_deadline


# 功能：
#   构造独立小地图、当前特征契约与教师指令的观测请求，不执行策略或飞行。
# 输入：
#   无。
# 输出：
#   request、command：用于观测编译的冻结输入及待绑定指令。
def request_fixture():
    original, command, _, _ = teacher_evidence()
    point = Vector3(x=0.5, y=0.5, z=0.5)
    request = NavigationSnapshotRequest(
        world=MetricVoxelMap(
            resolution_m=0.5,
            minimum_bound_m=Vector3(x=0, y=0, z=0),
            maximum_bound_m=Vector3(x=4, y=2, z=2),
        ),
        frame=OnboardPerceptionFrame(
            sensor_id="synthetic",
            sequence=1,
            observed_at_unix_ms=1000,
            localization_position_m=point,
            localization_velocity_mps=Vector3(x=0, y=0, z=0),
            localization_covariance_m2=0.01,
            range_rays=[
                RangeRayObservation(
                    origin_m=point,
                    endpoint_m=Vector3(x=3, y=0.5, z=0.5),
                    hit=False,
                    confidence=1.0,
                    observed_at_monotonic_seconds=1.0,
                )
            ],
        ),
        health=PerceptionFusionHealth(
            sensor_id="synthetic",
            latest_sequence=1,
            stream_age_seconds=0.0,
            localization_covariance_m2=0.01,
            accepted_ray_count=8,
            map_observation_count=8,
            stream_healthy=True,
        ),
        goal_position_m=Vector3(x=3, y=1, z=1),
        required_clearance_m=0.3,
        candidate_speed_mps=0.8,
        vehicle_radius_m=0.1,
        vehicle_height_m=0.2,
        visual_evidence=[],
        multimodal_sensor_snapshot=None,
        realtime_feature_snapshot=original["realtime_feature_snapshot"],
        strategic_context=original["strategic_context"],
        maximum_snapshot_planning_seconds=0.05,
        include_candidate_paths=False,
        control_reference_observed_at_unix_ms=1000,
    )
    return request, command


# 功能：
#   记录必须复用部署快照编译器并保持源时间，不虚构模型调用或控制授权。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_observer_uses_deployment_compiler_without_model_or_control_authority():
    request, command = request_fixture()
    records = []
    observer = LearningObservationRecorder(lambda record: records.append(record) is None)
    expected = compile_navigation_snapshot(request)
    assert observer.submit(request, command)
    assert observer.close()["complete"]
    assert len(records) == 1
    record = records[0]
    assert record["snapshot"] == expected
    assert record["model_invoked"] is record["control_authority_granted"] is False
    assert record["evaluated_command_sha256"] == sha256_json(command)
    assert record["recorded_at_unix_ms"] == 1000


# 功能：
#   编译忙碌时不积压任务，调用方后续修改消息也不能改变已接受的观测。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_slow_visual_job_has_no_backlog_and_freezes_mutable_inputs():
    request, command = request_fixture()
    release = threading.Event()
    records = []
    observer = LearningObservationRecorder(lambda record: records.append(record) is None)

    # 功能：
    #   用受控事件延迟视觉准备，制造提交后调用者修改输入的窗口。
    # 输入：
    #   无。
    # 输出：
    #   evidence：本测试不附加图像证据的空列表。
    def visual():
        assert release.wait(2.0)
        evidence = []
        return evidence

    try:
        assert observer.submit(request, command, prepare_visual=visual)
        assert not observer.submit(request, command)
        request.strategic_context["task"]["control_profile"] = "changed"
        request.frame.localization_position_m.x = 99
    finally:
        release.set()
        summary = observer.close()
    assert summary["busy_skipped"] == 1
    assert summary["complete"]
    snapshot = records[0]["snapshot"]
    assert snapshot["strategic_context"]["task"]["control_profile"] == "cruise"
    assert snapshot["current_position_m"]["x"] == 0.5


# 功能：
#   关闭超时后迟到的编译结果不得再写入接收端，必须保留不完整状态。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_timed_out_recording_can_never_write_later_into_a_closed_sink():
    request, command = request_fixture()
    release, started, finished = threading.Event(), threading.Event(), threading.Event()
    records = []
    observer = LearningObservationRecorder(lambda record: records.append(record) is None)

    # 功能：
    #   显式记录视觉任务的开始和结束，验证关闭与后台完成的先后关系。
    # 输入：
    #   无。
    # 输出：
    #   evidence：不附加图像证据的空列表。
    def visual():
        started.set()
        release.wait(2.0)
        finished.set()
        evidence = []
        return evidence

    try:
        assert observer.submit(request, command, prepare_visual=visual)
        assert started.wait(1.0)
        summary = observer.close(timeout_seconds=0.0)
        assert summary["issue_code"] == "LEARNING_OBSERVATION_DRAIN_TIMEOUT"
        assert not summary["complete"]
    finally:
        release.set()
        assert finished.wait(1.0)
    observer.poll()
    assert records == []
    assert not observer.submit(request, command)


# 功能：
#   拒绝坐标候选、重复／改期的来源及写入端拒收，不将这些情况升级为记录成功。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_no_candidate_inputs_duplicates_redating_or_failed_sink_promotion():
    request, command = request_fixture()
    observer = LearningObservationRecorder(lambda record: True)
    try:
        with pytest.raises(ValueError, match="COORDINATE_CANDIDATES"):
            observer.submit(replace(request, include_candidate_paths=True), command)
        assert observer.submit(request, command)
        observer._pending.result(timeout=2.0)
        assert not observer.submit(request, command)
        with pytest.raises(ValueError, match="SOURCE_CLOCK_CONFLICT"):
            observer.submit(
                replace(
                    request,
                    realtime_feature_snapshot=complete_feature_snapshot(1050).model_dump(
                        mode="json"
                    ),
                    control_reference_observed_at_unix_ms=1050,
                ),
                command,
            )
    finally:
        assert observer.close()["complete"]
    failed = LearningObservationRecorder(lambda record: False)
    assert failed.submit(request, command)
    assert not failed.close()["complete"]
    assert failed.issue == "LEARNING_OBSERVATION_SINK_REJECTED"


# 功能：
#   教师航向速率保持连续幅度、上限与机体系左右方向约定。
# 输入：
#   goal：合成目标的水平相对位置。
#   expected：独立给定的预期角速度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "goal,expected", [((1, 0), 0), ((1, -1), 20), ((1, 1), -20), ((1, -0.1), 5.710593), ((0, 0), 0)]
)
def test_teacher_yaw_has_continuous_magnitude_and_correct_handedness(goal, expected):
    assert teacher_heading_rate(
        orientation=QuaternionWxyz(w=1, x=0, y=0, z=0),
        position=Vector3(x=0, y=0, z=1),
        goal=Vector3(x=goal[0], y=goal[1], z=1),
        maximum_rate_dps=20.0,
    ) == pytest.approx(expected, abs=1e-6)


# 功能：
#   教师输入期限不能延长原生观测期限，预热中的编码器也不能授予可用输入。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_teacher_cannot_extend_native_deadline_or_use_warming_encoders():
    features = complete_feature_snapshot()
    assert teacher_input_deadline(features, now_ms=1050) == 1250
    assert teacher_input_deadline(features, now_ms=1300) == 0
    assert teacher_input_deadline(None, now_ms=1050) == 0
    long = fuse_realtime_features(
        [
            encoding.model_copy(update={"maximum_age_milliseconds": 1000})
            for encoding in features.encodings
        ],
        captured_at_unix_ms=1000,
    )
    assert teacher_input_deadline(long, now_ms=1100) == 1250
    warming = fuse_realtime_features(
        [
            encoding.model_copy(update={"issue_codes": ["FLIGHT_STATE_HISTORY_WARMING"]})
            if encoding.encoder_role == "flight-state-encoder"
            else encoding
            for encoding in features.encodings
        ],
        captured_at_unix_ms=1000,
    )
    assert teacher_input_deadline(warming, now_ms=1050) == 0
