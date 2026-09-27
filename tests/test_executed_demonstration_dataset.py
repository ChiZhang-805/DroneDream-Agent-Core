"""Synthetic receipts exercise integrity, not a physical flight qualification."""

import copy
import hashlib
import json
from dataclasses import replace

import pytest
from control_fixtures import complete_feature_snapshot
from test_executed_control_training import teacher_evidence

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.simulation_teacher_contract import SIMULATION_TEACHER_CONTRACT_SHA256
from dronedream_agent_core.realtime_feature_encoders import fuse_realtime_features
from dronedream_agent_core.training.demonstrations import (
    collect_demonstrations,
    file_sha256,
    validate_demonstration_splits,
)


# 功能：
#   创建可校验的合成运行记录，可选加入真实编码的 PNG；不伪装为已完成飞行验收。
# 输入：
#   root：尚不存在的测试运行目录。
#   visual：是否加入带像素及文件摘要的相机图像。
# 输出：
#   record：保留原模型输入及命令身份的学习观测信封。
def source_fixture(root, *, visual=False):
    root.mkdir()
    (root / "runtime-state").mkdir()
    snapshot, command, application, _ = teacher_evidence()
    task = snapshot["strategic_context"]["task"]
    task.update(
        control_session_id="synthetic-session", navigation_goal_id=command.navigation_goal_id,
        simulation_teacher_contract_sha256=SIMULATION_TEACHER_CONTRACT_SHA256,
    )
    if visual:
        from PIL import Image

        directory = root / "learning-observation-frames"
        directory.mkdir()
        path = directory / "fixture.png"
        pixels = Image.new("RGB", (64, 64), color=(4, 5, 6))
        pixels.save(path)
        snapshot["visual_evidence"] = [
            {
                "kind": "forward-rgb-camera",
                "relative_path": "learning-observation-frames/fixture.png",
                "sha256": file_sha256(path),
                "model_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "width": 64,
                "height": 64,
                "observed_at_unix_ms": 950,
            }
        ]
    snapshot.pop("snapshot_sha256")
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    record = {
        "evidence_kind": "simulation-observation-only",
        "policy_feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "model_invoked": False,
        "control_authority_granted": False,
        "recorded_at_unix_ms": 1000,
        "evaluated_command_sha256": sha256_json(command),
        "snapshot": snapshot,
    }
    values = {
        "learning-observations.jsonl": record,
        "learning-observation-summary.json": {"complete": True, "submitted": 1, "completed": 1},
        "depth-local-safety-history.jsonl": {"command": command.model_dump(mode="json")},
        "runtime-state/control-applications.jsonl": application.model_dump(mode="json"),
        "runtime-evidence-writer-summary.json": {"complete": True},
    }
    for relative, payload in values.items():
        (root / relative).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    learning = {
        "observations_recorded": True,
        "deterministic_teacher_control": True,
        "teacher_contract_sha256": SIMULATION_TEACHER_CONTRACT_SHA256,
        "model_control_qualification_granted": False,
    }
    evidence = {
        "schema_version": "dronedream.generic-px4-gazebo-run.v1",
        "status": "verified",
        "gates": {"synthetic-fixture-only": True},
        "measurements": {"simulation_learning": learning},
        "artifacts": {
            key: "e" * 64
            for key in ("world_sha256", "semantic_sha256", "vehicle_sha256", "route_sha256")
        },
    }
    (root / "mission-route.json").write_text(json.dumps({"positions_m": [
        {"x": 0., "y": 0., "z": 1.}, {"x": 1., "y": 0., "z": 1.}]}))
    evidence["artifacts"]["route_sha256"] = file_sha256(root / "mission-route.json")
    (root / "mission_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    rebind(root)
    return record


# 功能：
#   在测试主动修改文件后重新记录各证据文件摘要，以单独检验内容语义边界。
# 输入：
#   root：合成运行目录。
# 输出：
#   None：不返回业务数据。
def rebind(root):
    path = root / "mission_evidence.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    for relative, key in (
        ("learning-observations.jsonl", "learning_observations_sha256"),
        ("learning-observation-summary.json", "learning_observation_summary_sha256"),
        ("depth-local-safety-history.jsonl", "depth_safety_history_sha256"),
        ("runtime-state/control-applications.jsonl", "control_applications_sha256"),
        ("runtime-evidence-writer-summary.json", "runtime_evidence_writer_summary_sha256"),
    ):
        evidence["artifacts"][key] = file_sha256(root / relative)
    path.write_text(json.dumps(evidence), encoding="utf-8")


# 功能：
#   拒绝缺失或过期的教师行为契约，防止同形状输入承载不一致的转向监督。
# 输入：
#   tmp_path：隔离证据目录；where：运行回执或单条观测；value：缺失或旧身份。
# 输出：
#   None：重新绑定文件摘要后仍因行为契约不一致被明确拒绝。
@pytest.mark.parametrize('where', ['run', 'observation'])
@pytest.mark.parametrize('value', [None, '0' * 64])
def test_teacher_contract_required_at_run_and_observation(tmp_path, where, value):
    root = tmp_path / 'teacher'
    source_fixture(root)
    if where == 'run':
        path = root / 'mission_evidence.json'
        payload = json.loads(path.read_text())
        payload['measurements']['simulation_learning']['teacher_contract_sha256'] = value
    else:
        path = root / 'learning-observations.jsonl'
        payload = json.loads(path.read_text())
        snapshot = payload['snapshot']
        snapshot['strategic_context']['task']['simulation_teacher_contract_sha256'] = value
        snapshot.pop('snapshot_sha256')
        snapshot['snapshot_sha256'] = sha256_json(snapshot)
    path.write_text(json.dumps(payload) + '\n')
    rebind(root)
    with pytest.raises(ValueError, match='TEACHER_CONTRACT_MISMATCH'):
        collect_demonstrations([root], require_visual=False)


# 功能：
#   将有真实传输回执的合成教师恢复交给完整读取器，核对来源姿态四轴和恢复角色。
# 输入：
#   tmp_path：仅用于契约回归的临时目录，不计入正式数据。
# 输出：
#   None：恢复标签必须保持原始快照与实际速度回执的对应关系。
def test_pure_velocity_teacher_recovery_is_admitted_end_to_end(tmp_path):
    from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand
    from dronedream_agent_core.control_execution_evidence import control_application_record

    root = tmp_path / 'run'
    row = source_fixture(root)
    snapshot = row['snapshot']
    snapshot['strategic_context']['task'].update(decision_trigger='dynamic-obstacle', recovery_episode_id='synthetic-recovery')
    snapshot['dynamic_obstacles'] = [{'obstacle_id': 'moving-1', 'confidence': .9, 'observation_age_seconds': .01}]
    snapshot.pop('snapshot_sha256')
    snapshot['snapshot_sha256'] = sha256_json(snapshot)
    path = root / 'depth-local-safety-history.jsonl'
    command = RuntimeLocalSafetyCommand.model_validate(json.loads(path.read_text())['command'])
    command.decision.action = 'replan'
    command.decision.control_source = 'deterministic-brake'
    command.decision.selected_yaw_rate_dps = 0.
    command.decision.avoidance_obstacle_id = 'moving-1'
    application = control_application_record(command, sequence=1, accepted_at_unix_ms=1050,
        transport='velocity-ned', velocity_ned_mps=(-.2, .4, -.375), yaw_heading_deg=20.,
        yaw_rate_application={'previous_heading_deg': 20., 'clockwise_rate_dps': 0., 'integration_seconds': .05})
    row['evaluated_command_sha256'] = sha256_json(command)
    path.write_text(json.dumps({'command': command.model_dump(mode='json')}) + '\n')
    (root / 'learning-observations.jsonl').write_text(json.dumps(row) + '\n')
    (root / 'runtime-state/control-applications.jsonl').write_text(application.model_dump_json() + '\n')
    rebind(root)
    corpus = collect_demonstrations([root], require_visual=False)
    assert corpus.counts['accepted'] == 1
    assert corpus.samples[0].navigation_expert_role == 'recovery-policy'
    assert corpus.samples[0].target_pilot_control == pytest.approx([.5, .25, .5, 0.])


# 功能：
#   拒绝旧的仅凭最近物体赋予的动态恢复标签，不静默改写原始观测。
# 输入：
#   tmp_path：独立合成证据目录。
# 输出：
#   None：断言离线准入在生成任何标签之前失败。
def test_dynamic_recovery_without_causal_evidence_is_rejected(tmp_path):
    root = tmp_path / "run"
    row = source_fixture(root)
    snapshot = row["snapshot"]
    snapshot["strategic_context"]["task"].update(decision_trigger="dynamic-obstacle", recovery_episode_id="event-1")
    snapshot.pop("snapshot_sha256")
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    (root / "learning-observations.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    rebind(root)
    with pytest.raises(ValueError, match="DYNAMIC_RECOVERY_CAUSE_MISSING"):
        collect_demonstrations([root], require_visual=False)


# 功能：
#   动态恢复起因必须对应输入中唯一、新鲜且有置信度的目标，而非悬空身份。
# 输入：
#   tmp_path：合成运行目录。
#   fault：关联缺失、低置信度、过期、重复目标或非法数值。
# 输出：
#   None：断言读取器拒绝错误的监督来源绑定。
@pytest.mark.parametrize("fault", ["missing", "confidence", "age", "duplicate", "bool"])
def test_dynamic_recovery_cause_must_bind_current_obstacle(tmp_path, fault):
    from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand

    root = tmp_path / "run"
    row = source_fixture(root)
    snapshot = row["snapshot"]
    snapshot["strategic_context"]["task"].update(decision_trigger="dynamic-obstacle", recovery_episode_id="event-1")
    obstacle = {"obstacle_id": "moving-1", "confidence": .9, "observation_age_seconds": .01}
    snapshot["dynamic_obstacles"] = [obstacle]
    if fault == "missing":
        snapshot["dynamic_obstacles"] = []
    elif fault == "confidence":
        obstacle["confidence"] = .2
    elif fault == "age":
        obstacle["observation_age_seconds"] = 1.
    elif fault == "bool":
        obstacle["confidence"] = True
    else:
        snapshot["dynamic_obstacles"].append(dict(obstacle))
    snapshot.pop("snapshot_sha256")
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    command_path = root / "depth-local-safety-history.jsonl"
    command = RuntimeLocalSafetyCommand.model_validate(json.loads(command_path.read_text())["command"])
    command.decision.avoidance_obstacle_id = "moving-1"
    row["evaluated_command_sha256"] = sha256_json(command)
    command_path.write_text(json.dumps({"command": command.model_dump(mode="json")}) + "\n")
    (root / "learning-observations.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    rebind(root)
    with pytest.raises(ValueError, match="DYNAMIC_RECOVERY_CAUSE_UNBOUND"):
        collect_demonstrations([root], require_visual=False)


# 功能：
#   证明缺图的真实数值历史被保留，但不凭空制造视觉动作标签。
# 输入：
#   tmp_path：独立合成证据目录。
# 输出：
#   None：两条历史只产生一条带真实图片的监督标签。
def test_nonvisual_history_does_not_create_visual_labels(tmp_path):
    root = tmp_path / 'run'
    original = source_fixture(root, visual=True)
    later = copy.deepcopy(original)
    snapshot = later['snapshot']
    snapshot['visual_evidence'] = []
    snapshot['control_reference_observed_at_unix_ms'] = 1100
    encodings = [encoding.model_copy(update={'source_sha256': 'f' * 64})
                 for encoding in complete_feature_snapshot(1100).encodings]
    snapshot['realtime_feature_snapshot'] = fuse_realtime_features(
        encodings, captured_at_unix_ms=1100).model_dump(mode='json')
    later['recorded_at_unix_ms'] = 1100
    snapshot.pop('snapshot_sha256')
    snapshot['snapshot_sha256'] = sha256_json(snapshot)
    # 第二帧不绑定旧已执行动作，采用另一个没有执行回执的教师提案。
    commands_path = root / 'depth-local-safety-history.jsonl'
    first_command = json.loads(commands_path.read_text())
    next_command = copy.deepcopy(first_command)
    next_command['command']['generated_at_unix_ms'] = 1120
    next_command['command']['valid_until_unix_ms'] = 1300
    later['evaluated_command_sha256'] = sha256_json(next_command['command'])
    commands_path.write_text(json.dumps(first_command) + '\n' + json.dumps(next_command) + '\n')
    (root / 'learning-observations.jsonl').write_text(json.dumps(original) + '\n' + json.dumps(later) + '\n')
    (root / 'learning-observation-summary.json').write_text(json.dumps({'complete': True, 'submitted': 2, 'completed': 2}))
    rebind(root)
    corpus = collect_demonstrations([root], require_visual=True, allow_nonvisual_history=True)
    assert len(corpus.observations) == 2 and len(corpus.samples) == 1
    assert corpus.observations[-1].source_visual_sha256 is None
    assert corpus.counts['not_executed'] == 1
    with pytest.raises(ValueError, match='FORWARD_CAMERA_REQUIRED'):
        collect_demonstrations([root], require_visual=True)


# 功能：
#   构造一条真实执行标签与后续无指令观测的隔离契约样例，不算正式训练数据。
# 输入：
#   root：测试独占目录。
# 输出：
#   original、later、witness：执行观测、无动作观测及其安全日志绑定。
def no_command_history_fixture(root):
    original = source_fixture(root, visual=True)
    later = copy.deepcopy(original)
    snapshot = later['snapshot']
    snapshot['visual_evidence'] = []
    snapshot['control_reference_observed_at_unix_ms'] = 1100
    encodings = [encoding.model_copy(update={'source_sha256': 'f' * 64})
                 for encoding in complete_feature_snapshot(1100).encodings]
    snapshot['realtime_feature_snapshot'] = fuse_realtime_features(
        encodings, captured_at_unix_ms=1100).model_dump(mode='json')
    later.update(recorded_at_unix_ms=1100, evaluated_command_sha256=None,
                 control_evaluation_status='no-command')
    snapshot.pop('snapshot_sha256')
    snapshot['snapshot_sha256'] = sha256_json(snapshot)
    witness = {'recorded_at_unix_ms': 1100, 'command': None, 'identity_accepted': True,
               'navigation_goal_id': snapshot['strategic_context']['task']['navigation_goal_id'],
               'realtime_feature_snapshot': copy.deepcopy(snapshot['realtime_feature_snapshot'])}
    return original, later, witness


# 功能：
#   验证无获准指令的有效观测可以保留历史，但必须精确关联原始安全日志，不能制造标签。
# 输入：
#   tmp_path：独立合成证据目录。
#   fault：可选篡改类型，包含缺失见证、身份失败、时刻、目标、编码及伪造指令摘要。
# 输出：
#   None：合法输入保留两条历史一个标签，错误绑定全部拒绝。
@pytest.mark.parametrize('fault', [None, 'missing', 'identity', 'time', 'goal', 'features', 'digest', 'status'])
def test_no_command_history_requires_exact_safety_witness(tmp_path, fault):
    root = tmp_path / 'run'
    original, later, witness = no_command_history_fixture(root)
    if fault == 'identity':
        witness['identity_accepted'] = False
    elif fault == 'time':
        witness['recorded_at_unix_ms'] += 1
    elif fault == 'goal':
        witness['navigation_goal_id'] = 'wrong-goal'
    elif fault == 'features':
        witness['realtime_feature_snapshot']['captured_at_unix_ms'] += 1
    elif fault == 'digest':
        later['evaluated_command_sha256'] = original['evaluated_command_sha256']
    elif fault == 'status':
        later['control_evaluation_status'] = 'unknown'
    path = root / 'depth-local-safety-history.jsonl'
    if fault != 'missing':
        path.write_text(path.read_text() + json.dumps(witness) + '\n')
    (root / 'learning-observations.jsonl').write_text(json.dumps(original) + '\n' + json.dumps(later) + '\n')
    (root / 'learning-observation-summary.json').write_text(json.dumps({'complete': True, 'submitted': 2, 'completed': 2}))
    rebind(root)
    if fault is not None:
        with pytest.raises(ValueError, match='DEMONSTRATION_(NO_COMMAND_HISTORY_UNBOUND|CONTROL_EVALUATION_STATUS_INVALID)'):
            collect_demonstrations([root], require_visual=True, allow_nonvisual_history=True)
    else:
        corpus = collect_demonstrations([root], require_visual=True, allow_nonvisual_history=True)
        assert len(corpus.observations) == 2 and len(corpus.samples) == 1
        assert corpus.counts['no_command_history'] == 1
        assert corpus.observations[-1].source_visual_sha256 is None


# 功能：
#   验证允许无图历史不等于允许无图监督，已执行动作也不能绕过图像要求。
# 输入：
#   tmp_path：独立合成运行目录。
# 输出：
#   None：没有带图动作时拒绝生成视觉训练语料。
def test_executed_action_without_image_is_not_visual_supervision(tmp_path):
    root = tmp_path / 'run'
    source_fixture(root, visual=False)
    with pytest.raises(ValueError, match='NO_EXECUTED_SAMPLES'):
        collect_demonstrations([root], require_visual=True, allow_nonvisual_history=True)


# 功能：
#   验证宽松历史选项不能掩盖已经存在但损坏的相机文件。
# 输入：
#   tmp_path：独立合成运行目录。
# 输出：
#   None：损坏的图像仍被拒绝，而不是作为无图历史保留。
def test_optional_history_still_rejects_corrupt_image(tmp_path):
    root = tmp_path / 'run'
    source_fixture(root, visual=True)
    (root / 'learning-observation-frames/fixture.png').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        collect_demonstrations([root], require_visual=True, allow_nonvisual_history=True)


# 功能：
#   改写指定合成证据，重绑快照及文件摘要，让测试区分摘要损坏与语义损坏。
# 输入：
#   root：本测试的运行目录。
#   relative：目录内待修改的证据路径。
#   mutate：对解析对象施加测试变更的回调。
# 输出：
#   None：不返回业务数据。
def update_file(root, relative, mutate):
    path = root / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    if relative == "learning-observations.jsonl":
        value["snapshot"].pop("snapshot_sha256")
        value["snapshot"]["snapshot_sha256"] = sha256_json(value["snapshot"])
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    rebind(root)


# 功能：
#   验证数据集使用实际四轴控制标签，保留原始粗粒度输入且禁用旧坐标候选。
# 输入：
#   tmp_path：测试独占临时目录。
# 输出：
#   None：不返回业务数据。
def test_dataset_uses_executed_four_axes_and_preserves_coarse_input(tmp_path):
    root = tmp_path / "run"
    record = source_fixture(root)
    corpus = collect_demonstrations([root], require_visual=False)
    assert corpus.counts == {"accepted": 1, "recorded": 1}
    assert len(corpus.observations) == 1
    assert not hasattr(corpus.observations[0], "target_action_index")
    sample = corpus.samples[0]
    assert sample.target_pilot_control == pytest.approx([0.5, 0.25, 0.5, 0.5])
    assert sample.source_snapshot_sha256 == record["snapshot"]["snapshot_sha256"]
    assert sample.pilot_control_limits.horizontal_speed_mps == 0.8
    assert sample.temporal_evidence.stream_id in corpus.stream_groups
    assert not any(sample.candidate_mask)
    assert sample.risk_proposed_control == []  # Not a counterfactual risk corpus.


# 功能：
#   验证未完成运行、错误权限、旧特征契约、错绑任务及过期回执不能成为教师数据。
# 输入：
#   tmp_path：合成运行目录的父目录。
#   relative：注入问题的证据文件。
#   mutate：具体变更回调。
#   issue：应出现的拒绝代码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "relative,mutate,issue",
    [
        ("mission_evidence.json", lambda v: v.update(status="failed"), "NOT_VERIFIED"),
        ("learning-observation-summary.json", lambda v: v.update(complete=False), "INCOMPLETE"),
        (
            "learning-observation-summary.json",
            lambda v: v.update(completed=2, submitted=2),
            "COUNT",
        ),
        ("learning-observations.jsonl", lambda v: v.update(model_invoked=True), "AUTHORITY"),
        (
            "learning-observations.jsonl",
            lambda v: v.update(policy_feature_contract_sha256="0" * 64),
            "AUTHORITY",
        ),
        (
            "learning-observations.jsonl",
            lambda v: v.update(evaluated_command_sha256="b" * 64),
            "COMMAND_MISSING",
        ),
        (
            "learning-observations.jsonl",
            lambda v: v["snapshot"]["strategic_context"]["task"].update(
                navigation_goal_id="some-other-goal"
            ),
            "TASK_REFERENCE",
        ),
        (
            "runtime-state/control-applications.jsonl",
            lambda v: v.update(transport="position-velocity-ned"),
            "POSITION_CONTROL",
        ),
        (
            "runtime-state/control-applications.jsonl",
            lambda v: v.update(yaw_rate_application=None),
            "YAW_RATE_MISSING",
        ),
        (
            "runtime-state/control-applications.jsonl",
            lambda v: v.update(accepted_at_unix_ms=1300),
            "TIME_ALIGNMENT",
        ),
    ],
)
def test_dataset_rejects_incomplete_stale_misattributed_or_legacy_inputs(
    tmp_path, relative, mutate, issue
):
    root = tmp_path / "run"
    source_fixture(root)
    update_file(root, relative, mutate)
    with pytest.raises(ValueError, match=issue):
        collect_demonstrations([root], require_visual=False)


# 功能：
#   验证文件变化必须与摘要一致；即使重绑摘要也不能靠重复帧填充训练样本数。
# 输入：
#   tmp_path：合成运行目录的父目录。
# 输出：
#   None：不返回业务数据。
def test_changes_without_receipt_update_rejected_and_duplicate_frames_not_padded(tmp_path):
    root = tmp_path / "run"
    source_fixture(root)
    path = root / "learning-observations.jsonl"
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="FILE_HASH"):
        collect_demonstrations([root], require_visual=False)
    rebind(root)
    with pytest.raises(ValueError, match="REPEATED_OBSERVATION"):
        collect_demonstrations([root], require_visual=False)


# 功能：
#   验证图像来源、原始时间和目录边界，不用空特征冒充已经运行过视觉编码器。
# 输入：
#   tmp_path：包含合成 PNG 的运行目录的父目录。
# 输出：
#   None：不返回业务数据。
def test_image_pixels_original_time_and_confined_path_are_checked(tmp_path):
    root = tmp_path / "run"
    source_fixture(root, visual=True)
    corpus = collect_demonstrations([root])
    assert corpus.samples[0].source_visual_sha256
    assert corpus.samples[0].visual_features == []  # Real encoder must still be run.
    update_file(
        root,
        "learning-observations.jsonl",
        lambda v: v["snapshot"]["visual_evidence"][0].update(observed_at_unix_ms=500),
    )
    with pytest.raises(ValueError, match="VISUAL_TIME"):
        collect_demonstrations([root])
    update_file(
        root,
        "learning-observations.jsonl",
        lambda v: v["snapshot"]["visual_evidence"][0].update(relative_path="../outside.png"),
    )
    with pytest.raises(ValueError, match="VISUAL_PATH"):
        collect_demonstrations([root])


# 功能：
#   分别验证路线组、输入观测及物理传感器来源不能同时出现在训练集和留出集。
# 输入：
#   tmp_path：合成运行目录的父目录。
# 输出：
#   None：不返回业务数据。
def test_route_groups_and_physical_sources_cannot_cross_splits(tmp_path):
    root = tmp_path / "run"
    source_fixture(root)
    corpus = collect_demonstrations([root], require_visual=False)
    with pytest.raises(ValueError, match="MISSION_GROUP_LEAKAGE"):
        validate_demonstration_splits(corpus, corpus)
    other = replace(corpus, stream_groups={"different-stream": "different-group"})
    with pytest.raises(ValueError, match="OBSERVATION_LEAKAGE"):
        validate_demonstration_splits(corpus, other)
    different = copy.deepcopy(other.samples[0])
    different.source_snapshot_sha256 = "f" * 64
    with pytest.raises(ValueError, match="PHYSICAL_SOURCE_LEAKAGE"):
        validate_demonstration_splits(corpus, replace(other, samples=[different], observations=[]))
