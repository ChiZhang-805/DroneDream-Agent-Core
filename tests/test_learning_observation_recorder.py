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
from dronedream_agent_core.learning_observation_recorder import LearningObservationRecorder, learning_recovery_task_context
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.navigation_snapshot import (
    NavigationSnapshotRequest,
    compile_navigation_snapshot,
    freeze_navigation_frame,
)
from dronedream_agent_core.realtime_feature_encoders import fuse_realtime_features
from dronedream_agent_core.simulation_teacher import teacher_heading_rate, teacher_input_deadline


# 功能：
#   无指令时保存真实观测并显式禁止监督标签；同步与后台模式采用同一标记。
# 输入：
#   synchronous：记录编译是否在调用线程完成。
# 输出：
#   None：断言无伪造指令、无模型权限，来源与原始快照一致。
@pytest.mark.parametrize("synchronous", [True, False])
def test_no_command_observation_retains_source_without_action(synchronous):
    request, _ = request_fixture()
    records = []
    recorder = LearningObservationRecorder(lambda row: records.append(row) is None,
                                          synchronous=synchronous)
    try:
        assert recorder.submit(request, None)
    finally:
        assert recorder.close()["complete"]
    assert len(records) == 1
    assert records[0]["evaluated_command_sha256"] is None
    assert records[0]["control_evaluation_status"] == "no-command"
    assert records[0]["model_invoked"] is False
    assert records[0]["control_authority_granted"] is False
    assert records[0]["snapshot"] == compile_navigation_snapshot(request)


# 功能：
#   无动作历史仍受原有感知时效限制，不能以保留历史为由接纳过期输入。
# 输入：
#   无。
# 输出：
#   None：断言记录被拒绝且来源提交计数不增加。
def test_no_command_history_does_not_bypass_feature_freshness():
    request, _ = request_fixture()
    recorder = LearningObservationRecorder(lambda row: pytest.fail("expired source"), synchronous=True)
    request = replace(request, control_reference_observed_at_unix_ms=2000)
    assert recorder.submit(request, None) is False
    assert recorder.expired_skipped == 1
    assert recorder.close()["submitted"] == 0


# 功能：
#   核对教师恢复标签来自真实威胁和同目标事件，排除远处目标、旧事件和普通跟踪状态。
# 输入：
#   无：显式构造目标与事件身份，不产生正式训练样本。
# 输出：
#   无：断言恢复标注和普通标注边界。
def test_learning_recovery_context_requires_bound_incident():
    event = {'navigation_goal_id': 'goal-a', 'episode_id': 'incident-a'}
    ordinary = {'decision_trigger': 'periodic', 'recovery_episode_id': None}
    assert learning_recovery_task_context({}, 'goal-a', event, True) == {
        'decision_trigger': 'dynamic-obstacle', 'recovery_episode_id': 'incident-a'}
    assert learning_recovery_task_context({}, 'goal-a', event, False) == ordinary
    assert learning_recovery_task_context({}, 'goal-b', event, True) == ordinary
    assert learning_recovery_task_context({'executor_phase': 'TRACKING_RECOVERY'}, 'goal-a', None, False) == ordinary
    assert learning_recovery_task_context({'decision_trigger': 'progress-stalled'}, 'goal-a', None, False) == ordinary
    stalled = {'decision_trigger': 'progress-stalled', 'recovery_episode_id': 'stall-a'}
    assert learning_recovery_task_context(stalled, 'goal-a', event, True) == stalled
    assert learning_recovery_task_context({}, 'goal-a', {'navigation_goal_id': 'goal-a'}, True) == ordinary
    with pytest.raises(ValueError, match='CONTEXT_INVALID'):
        learning_recovery_task_context({}, 'goal-a', event, 1)


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
#   同线程采集复用完整部署编译器，不创建后台线程、不让活地图引用进入已生成证据。
# 输入：
#   monkeypatch：禁止创建后台编译池，以验证实际路径而非只检查模式字段。
# 输出：
#   None：断言内容、来源、控制权限与默认编译一致，后续地图变化不影响旧记录。
def test_same_thread_compilation_owns_live_map_until_return(monkeypatch):
    from unittest.mock import Mock
    import dronedream_agent_core.learning_observation_recorder as module

    request, command = request_fixture()
    request.world.integrate_scan(request.frame.range_rays)
    expected = compile_navigation_snapshot(request)
    owner = request.world._evidence_owner
    monkeypatch.setattr(module, 'ThreadPoolExecutor', Mock(side_effect=AssertionError('unexpected background reader')))
    records = []
    observer = LearningObservationRecorder(lambda record: records.append(record) is None, synchronous=True)
    assert observer.submit(request, command)
    assert observer.completed == 1 and len(records) == 1
    assert not observer.submit(request, command)
    assert request.world._evidence_owner is owner
    request.world.integrate_scan([request.frame.range_rays[0].model_copy(update={
        'hit': True, 'observed_at_monotonic_seconds': 2.})])
    request.frame.localization_position_m.x = 99
    assert records[0]['snapshot'] == expected
    assert records[0]['model_invoked'] is records[0]['control_authority_granted'] is False
    summary = observer.close()
    assert summary['complete'] and summary['compiler_mode'] == 'same-thread'
    assert summary['duplicate_skipped'] == 1


# 功能：
#   同线程模式拒绝其他线程借用活地图；错误调用不能推进计数或来源基线。
# 输入：
#   无：另建仅用于测试调用边界的线程，不运行飞行或模型。
# 输出：
#   None：断言跨线程请求失败而所有者仍能提交原始请求。
def test_same_thread_compilation_rejects_foreign_thread():
    from concurrent.futures import ThreadPoolExecutor

    request, command = request_fixture()
    observer = LearningObservationRecorder(lambda record: True, synchronous=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        attempted = pool.submit(observer.submit, request, command)
        with pytest.raises(ValueError, match='WRONG_OWNER_THREAD'):
            attempted.result()
    assert observer.submitted == 0
    assert observer.submit(request, command)
    assert observer.close()['complete']


# 功能：
#   同线程编译或落盘接收失败必须留下失败状态，不能把提交计数误报为完整记录。
# 输入：
#   failure：模拟编译异常或接收端拒绝。
#   monkeypatch：仅替换被测试的失败边界。
# 输出：
#   None：断言失败后不再接收新工作且没有伪成功回执。
@pytest.mark.parametrize('failure', ['compile', 'sink'])
def test_same_thread_compilation_failure_is_not_complete(monkeypatch, failure):
    from unittest.mock import Mock

    request, command = request_fixture()
    observer = LearningObservationRecorder(lambda record: False, synchronous=True)
    if failure == 'compile':
        monkeypatch.setattr(observer, '_compile', Mock(side_effect=ValueError('synthetic failure')))
    assert observer.submit(request, command)
    assert not observer.submit(request, command)
    summary = observer.close()
    assert summary['submitted'] == 1 and summary['completed'] == 0
    assert not summary['complete']
    assert summary['issue_code'] == ('LEARNING_OBSERVATION_FAILED:ValueError' if failure == 'compile'
                                      else 'LEARNING_OBSERVATION_SINK_REJECTED')


# 功能：
#   拒绝隐式真值作为同步模式，避免配置字符串意外改变地图所有权约定。
# 输入：
#   value：非法模式值。
# 输出：
#   None：断言初始化在创建线程前拒绝配置。
@pytest.mark.parametrize('value', [1, 'true', None])
def test_compiler_mode_requires_boolean(value):
    with pytest.raises(ValueError, match='COMPILER_MODE_INVALID'):
        LearningObservationRecorder(lambda record: True, synchronous=value)


# 功能：
#   验证轻量状态与原感知帧编译结果完全一致，且异步提交不复制未被导航编译使用的射线。
# 输入：
#   monkeypatch：隔离替换射线复制操作，误复制时立即使测试失败。
# 输出：
#   None：内容摘要、源时间和权限保持一致，源向量修改不能污染冻结状态。
def test_navigation_state_omits_raw_ray_copy_without_changing_snapshot(monkeypatch):
    request, command = request_fixture()
    expected = compile_navigation_snapshot(request)

    # 功能：
    #   阻止射线进入导航状态复制路径，检查并非只优化了最终序列化。
    # 输入：
    #   self、memo：深拷贝协议参数。
    # 输出：
    #   无：调用意味着回归，立即抛出断言错误。
    def reject_copy(self, memo=None):
        raise AssertionError('navigation state copied raw depth rays')

    monkeypatch.setattr(RangeRayObservation, '__deepcopy__', reject_copy)
    frozen = freeze_navigation_frame(request.frame)
    assert not hasattr(frozen, 'range_rays')
    assert compile_navigation_snapshot(replace(request, frame=frozen)) == expected
    records = []
    observer = LearningObservationRecorder(lambda row: records.append(row) is None)
    assert observer.submit(request, command)
    request.frame.localization_position_m.x = 99
    assert observer.close()['complete']
    assert records[0]['snapshot'] == expected
    assert frozen.localization_position_m.x == .5


# 功能：
#   验证导航裁剪保留动态目标的真实年龄并隔离可变向量，拒绝超过目标预算的输入。
# 输入：
#   无：使用明确的动态目标和测试感知帧。
# 输出：
#   无：断言冻结结果不随原目标改变，异常输入被拒绝。
def test_navigation_state_freezes_dynamic_targets_and_enforces_budget():
    from dronedream_agent_core.contracts import DynamicObstacleObservation

    request, _ = request_fixture()
    target = DynamicObstacleObservation(obstacle_id='cart',
        position_m=Vector3(x=2, y=1, z=1), velocity_mps=Vector3(x=0, y=.1, z=0),
        radius_m=.2, height_m=.5, confidence=.9, age_seconds=.12)
    request.frame.dynamic_obstacles = [target]
    frozen = freeze_navigation_frame(request.frame)
    target.position_m.x = 99
    target.age_seconds = 4
    assert frozen.dynamic_obstacles[0].position_m.x == 2
    assert frozen.dynamic_obstacles[0].age_seconds == .12
    # 模拟已验证容器随后被原地扩展；赋值验证无法拦截列表内部修改。
    request.frame.dynamic_obstacles.extend([target] * 512)
    with pytest.raises(ValueError, match='budget'):
        freeze_navigation_frame(request.frame)
    with pytest.raises(ValueError, match='typed'):
        freeze_navigation_frame({})


# 功能：
#   验证过期输入有明确计数，且不推进来源基线或制造训练记录。
# 输入：
#   无：使用合成来源时间和超过期限的消费时间。
# 输出：
#   None：过期输入被拒绝，原先尚未提交的合法输入仍能被接受。
def test_expired_observation_is_counted_without_advancing_source():
    request, command = request_fixture()
    records = []
    observer = LearningObservationRecorder(lambda row: records.append(row) is None)
    assert not observer.submit(replace(request, control_reference_observed_at_unix_ms=2000), command)
    assert observer.submit(request, command)
    summary = observer.close()
    assert summary['expired_skipped'] == 1 and summary['submitted'] == 1
    assert summary['complete'] and len(records) == 1


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
