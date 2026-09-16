"""Synthetic receipts exercise integrity, not a physical flight qualification."""

import copy
import hashlib
import json
from dataclasses import replace

import pytest
from test_executed_control_training import teacher_evidence

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.hashing import sha256_json
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
        control_session_id="synthetic-session", navigation_goal_id=command.navigation_goal_id
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
