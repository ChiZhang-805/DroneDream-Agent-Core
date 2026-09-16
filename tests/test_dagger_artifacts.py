"""Synthetic artifact/learner integration; these are not physical receipts."""

import hashlib
import json

import pytest
from test_deferred_student_collection import Lifecycle
from test_mission_groups import split_fixture

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.causal_policy import causal_examples
from dronedream_agent_core.training.dagger import ProposedActionRisk, TeacherCorrection
from dronedream_agent_core.training.dagger_artifacts import (
    load_dagger_training_artifacts,
    write_rows,
    write_training_artifacts,
)
from dronedream_agent_core.training.flight_environment import PilotAction
from dronedream_agent_core.training.student_collection import (
    collect_student_visits,
    label_student_visits,
)


# 功能：
#   通过合成学生采集和独立教师标签生成完整 DAgger 数据；不当作物理飞行证明。
# 输入：
#   root：已存在的独占输出目录。
# 输出：
#   visits：学生访问记录。
#   result：教师纠正和风险标注结果。
def dataset(root):
    env = Lifecycle()
    visits = collect_student_visits(env,
        lambda _: PilotAction(mode="pilot-control", axes=[.5,0,0,0]),
        seed=805, steps=4, held_out_missions=set(), student_policy_sha256="d"*64)
    assert env.grounded
    result = label_student_visits(visits,
        lambda obs: TeacherCorrection(observation_sha256=sha256_json(obs),
            action=PilotAction(mode="pilot-control", axes=[-.1,0,0,0]),
            verified_action_risk=.1, verifier_receipt_sha256="a"*64),
        lambda obs, action: ProposedActionRisk(observation_sha256=sha256_json(obs),
            proposed_action_sha256=sha256_json(action), risk=.9, source="swept-geometry",
            verifier_receipt_sha256="b"*64),
        held_out_missions=set(), evidence_kind="px4-gazebo")  # synthetic format fixture only
    episode_groups = {v.observation.episode_id: split_fixture(semantic=v.observation.map_sha256)
                      for v in visits}
    hashes = write_training_artifacts(root, visits, result, episode_groups=episode_groups)
    write_rows(root / "collection-receipt.jsonl", [{"purpose": "native-student-visited-dagger",
                                                   "file_sha256": hashes}])
    return visits, result


# 功能：
#   验证标签和历史完整读回，因果窗口保持原传感器身份，且不能覆盖既有数据集。
# 输入：
#   tmp_path：测试独占数据目录。
# 输出：
#   None：不返回业务数据。
def test_corrected_native_format_history_compiles_without_rewriting_sensor_identity(tmp_path):
    visits, result = dataset(tmp_path)
    labels, history, groups, digest = load_dagger_training_artifacts(tmp_path)
    assert len(digest) == 64
    assert labels == result.behavior_samples
    assert labels[-1].source_snapshot_sha256 == visits[-1].observation.sample.source_snapshot_sha256
    assert labels[-1].source_snapshot_sha256 != sha256_json(visits[-1].observation)
    examples = causal_examples(labels, stream_groups=groups.groups, history_length=4,
                               history_observations=history)
    assert len(examples) == 1  # startup observations are history, not fabricated full windows
    assert examples[0].sample.target_pilot_control == [-.1,0,0,0]
    assert examples[0].source_sha256 == tuple(
        visit.observation.sample.temporal_evidence.sample_sha256 for visit in visits)
    with pytest.raises(FileExistsError):
        write_training_artifacts(tmp_path, visits, result, episode_groups={
            v.observation.episode_id: split_fixture(semantic=v.observation.map_sha256)
            for v in visits})


# 功能：
#   验证五类训练产物的原始字节变化均被摘要检查拒绝。
# 输入：
#   tmp_path：合成数据目录。
#   name：要追加空白字节的文件类别。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("name", ["behavior-corrections", "proposed-action-risk",
                                 "correction-records", "training-observations", "stream-groups"])
def test_changed_content_is_rejected(tmp_path, name):
    dataset(tmp_path)
    path = tmp_path / (name + (".json" if name == "stream-groups" else ".jsonl"))
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="CONTENT_CHANGED"):
        load_dagger_training_artifacts(tmp_path)


# 功能：
#   验证重绑摘要后，错误动作、风险或教师选择标记仍不能通过类型及标签对应检查。
# 输入：
#   tmp_path：合成数据目录。
#   name：被修改的文件类别。
#   field：故意改写的记录字段。
#   value：注入的不一致字段值。
#   error：应出现的语义拒绝代码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("name,field,value,error", [
    ("behavior-corrections", "target_pilot_control", [.1,0,0,0], "CORRECTION_LABEL"),
    ("proposed-action-risk", "risk_target", .2, "ACTION_RISK_LABEL"),
    ("correction-records", "teacher_selected", True, "CORRECTION_LABEL"),
])
def test_rehashed_wrong_labels_still_fail_typed_correspondence(tmp_path, name, field, value, error):
    dataset(tmp_path)
    path = tmp_path / (name + ".jsonl")
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    rows[0][field] = value
    content = ("\n".join(json.dumps(row) for row in rows)+"\n").encode()
    path.write_bytes(content)
    receipt_path = tmp_path / "collection-receipt.jsonl"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["file_sha256"][name] = hashlib.sha256(content).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=error):
        load_dagger_training_artifacts(tmp_path)
