"""Full sensory histories and immutable training lineage, using synthetic sources."""

import hashlib
import json
import shutil
import sys

import pytest
from test_admission_input_freezing import spatial_receipt
from test_causal_packaging import base_package as base_package
from test_causal_packaging import complete_current_base
from test_causal_policy import samples
from test_mission_groups import split_fixture

from dronedream_agent_core.local_policy_packages import load_local_policy_package
from dronedream_agent_core.local_policy_training import LocalPolicyObservation
from dronedream_agent_core.training.admission_inputs import (
    bind_admission_history,
    read_admission_observations,
)
from dronedream_agent_core.training.ensemble_lineage import preserve_complete_expert_lineage
from dronedream_agent_core.training.mission_groups import MissionGroupManifest
from scripts import encode_local_policy_visual_features as encoding
from scripts import evaluate_local_policy_offline as evaluation


# 功能：
#   将合成标签剥离成原观测，视觉编码仍只留在带标签的当前帧，避免伪造监督。
# 输入：
#   rows：合成因果控制样本。
# 输出：
#   observations：没有动作与风险标签的传感器历史。
def history_rows(rows):
    observations = [
        LocalPolicyObservation(
            **{name: getattr(row, name) for name in LocalPolicyObservation.model_fields}
        ).model_copy(update={"visual_features": []})
        for row in rows
    ]
    return observations


# 功能：
#   无标签行保留历史，只有真实匹配标签参与计分；倒序文件按同一流时序重新排列。
# 输入：
#   无：使用十帧合成观测和末尾两个动作标签。
# 输出：
#   None：不返回业务数据。
def test_history_binding_does_not_invent_labels():
    rows = samples()[:10]
    replay = bind_admission_history(rows[-2:], list(reversed(history_rows(rows))))
    assert len(replay) == 10
    assert [label for _, label in replay if label is not None] == rows[-2:]
    assert [row.temporal_evidence for row, _ in replay] == [row.temporal_evidence for row in rows]


# 功能：
#   观测丢失、重复、特征漂移、旧契约和坐标候选不能混入当前因果控制评估。
# 输入：
#   failure：本次要破坏的观测或标签边界。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failure", ["missing", "duplicate", "changed", "old", "candidate"])
def test_history_binding_rejects_unfaithful_sources(failure):
    rows = samples()[:4]
    observations = history_rows(rows)
    if failure == "missing":
        observations.pop()
    elif failure == "duplicate":
        observations.append(observations[-1])
    elif failure == "changed":
        observations[-1].state_features[0] += 0.1
    elif failure == "old":
        observations[0].control_feature_contract_sha256 = None
    else:
        observations[0].candidate_mask[0] = 1.0
    with pytest.raises(ValueError, match="CAUSAL_ADMISSION"):
        bind_admission_history([rows[-1]], observations)


# 功能：
#   写入真实测试历史与可重算空间分组，并用实际写入摘要绑定回执。
# 输入：
#   tmp_path：仅属于本测试的文件目录。
# 输出：
#   case：历史路径、分组路径及原始数据回执组成的元组。
@pytest.fixture
def history_case(tmp_path):
    rows = history_rows(samples()[:4])
    path, groups_path = tmp_path / "history.jsonl", tmp_path / "groups.json"
    path.write_bytes(b"".join((row.model_dump_json() + "\n").encode() for row in rows))
    split = split_fixture(y=2.0)
    manifest = MissionGroupManifest(groups={"train": split.group_sha256}, evidence=[split])
    groups_path.write_bytes(manifest.model_dump_json().encode())
    receipt = spatial_receipt()
    receipt["outputs"]["validation"].update(
        observation_count=len(rows),
        observation_history_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    receipt["stream_groups_sha256"] = hashlib.sha256(groups_path.read_bytes()).hexdigest()
    case = path, groups_path, receipt
    return case


# 功能：
#   当前生产者回执绑定的完整历史能读回，并拒绝字节、行数或路线声明漂移。
# 输入：
#   history_case：实际历史文件和空间证据。
#   failure：正常来源或要破坏的绑定字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failure", [None, "history", "groups", "count", "route"])
def test_history_reader_checks_complete_dataset_identity(history_case, failure):
    path, groups_path, receipt = history_case
    if failure == "history":
        path.write_bytes(path.read_bytes() + b"\n")
    elif failure == "groups":
        receipt["stream_groups_sha256"] = "e" * 64
    elif failure == "count":
        receipt["outputs"]["validation"]["observation_count"] = True
    elif failure == "route":
        receipt["outputs"]["validation"]["sources"][0]["mission_group"] = (
            split_fixture().group_sha256
        )
    if failure is None:
        assert len(read_admission_observations(path, groups_path, receipt)) == 4
    else:
        with pytest.raises(ValueError, match="causal admission"):
            read_admission_observations(path, groups_path, receipt)


# 功能：
#   用实际十角色 ONNX 后端把无标签历史接入评估，仅对暖机后的两个真实标签计分。
# 输入：
#   base_package：合成十角色来源包。
#   tmp_path：完整当前模型及验证数据目录。
# 输出：
#   None：不返回业务数据。
def test_actual_onnx_admission_consumes_unlabelled_sensor_history(base_package, tmp_path):
    from test_risk_admission_evidence import synthetic_risk_evidence

    from dronedream_agent_core.local_policy_packages import load_local_policy_package

    root = complete_current_base(base_package, tmp_path)
    manifest = load_local_policy_package(root).manifest
    risk_sha = next(a.sha256 for a in manifest.artifacts if a.role == 'risk-critic')
    rows = [row.model_copy(update={"visual_features": [0.1]}) for row in samples()[:10]]
    path = tmp_path / "labels.jsonl"
    path.write_bytes(b"".join((row.model_dump_json() + "\n").encode() for row in rows[-2:]))
    metrics, latencies = evaluation._evaluate_runtime_package(
        root, path, observations=history_rows(rows),
        action_risk_evidence=synthetic_risk_evidence(risk_sha),
    )
    assert metrics.sample_count == 2
    assert len(latencies) == 2


# 功能：
#   换载具时保留同一权重的全部训练来源并绑定新包；不携带旧准入或飞行资格。
# 输入：
#   base_package：合成十角色来源包。
#   tmp_path：本测试候选目录。
# 输出：
#   None：不返回业务数据。
def test_vehicle_rebinding_preserves_complete_training_lineage(base_package, tmp_path):
    base = load_local_policy_package(complete_current_base(base_package, tmp_path))
    staged = tmp_path / "rebound"
    staged.mkdir()
    manifest = base.manifest.model_copy(
        update={"vehicle_sha256": "f" * 64, "package_id": "unit.rebound"}
    )
    for artifact in manifest.artifacts:
        shutil.copyfile(base.artifact_paths[artifact.role], staged / artifact.relative_path)
    (staged / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    candidate = load_local_policy_package(staged)
    preserve_complete_expert_lineage(base, staged, candidate, {}, rebind_vehicle=True)
    evidence = json.loads((staged / "assembly-receipt.json").read_bytes())
    assert evidence["package_sha256"] == candidate.package_sha256 != base.package_sha256
    assert evidence["inherited_admission"] is False
    assert evidence["qualified_for_flight"] is False
    assert len(evidence["expert_evidence"]) == 10
    for role in evidence["expert_evidence"]:
        name = f"training-evidence/{role}.json"
        assert (staged / name).read_bytes() == (base.root / name).read_bytes()
    with pytest.raises(FileExistsError):
        preserve_complete_expert_lineage(base, staged, candidate, {}, rebind_vehicle=True)
    bad = candidate.manifest.model_copy(update={"risk_hold_threshold": 0.9})
    # 包载入对象为冻结数据类，重载清单验证不能借重绑定改变其他控制条件。
    (staged / "manifest.json").write_text(bad.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="MORE_THAN_VEHICLE"):
        preserve_complete_expert_lineage(
            base, staged, load_local_policy_package(staged), {}, rebind_vehicle=True
        )


# 功能：
#   串联实际 PNG 编码、输入冻结、十角色 ONNX 评估和拒绝回执发布，缺少质量证据不能冒称准入。
# 输入：
#   base_package：轻量合成模型来源。
#   tmp_path：本例所有测试文件目录。
#   monkeypatch：设置命令行和独立风险证据夹具；视觉与行为 ONNX 仍实际执行。
# 输出：
#   None：不返回业务数据。
def test_current_admission_cli_connects_visual_history_and_frozen_inputs(
    base_package, tmp_path, monkeypatch
):
    from PIL import Image

    root = complete_current_base(base_package, tmp_path)
    package = load_local_policy_package(root)
    frames = tmp_path / "frames"
    frames.mkdir()
    frame = frames / "front.png"
    Image.new("RGB", (32, 32), color=(80, 120, 160)).save(frame)
    frame_digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    rows = [row.model_copy(update={"source_visual_sha256": frame_digest}) for row in samples()[:10]]
    raw, encoded = tmp_path / "raw.jsonl", tmp_path / "encoded.jsonl"
    history, groups_path = tmp_path / "observations.jsonl", tmp_path / "stream-groups.json"
    dataset_path, visual_path = tmp_path / "dataset.json", tmp_path / "visual.json"
    output = tmp_path / "decision.json"
    raw.write_bytes(b"".join((row.model_dump_json() + "\n").encode() for row in rows[-2:]))
    history.write_bytes(
        b"".join((row.model_dump_json() + "\n").encode() for row in history_rows(rows))
    )
    split = split_fixture(y=2.0)
    groups = MissionGroupManifest(groups={"train": split.group_sha256}, evidence=[split])
    groups_path.write_bytes(groups.model_dump_json().encode())
    dataset = spatial_receipt()
    dataset["outputs"]["validation"].update(
        sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        observation_count=10,
        observation_history_sha256=hashlib.sha256(history.read_bytes()).hexdigest(),
    )
    dataset["stream_groups_sha256"] = hashlib.sha256(groups_path.read_bytes()).hexdigest()
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "encode",
            "--policy-data",
            str(raw),
            "--frame-root",
            str(frames),
            "--perception-encoder",
            str(package.artifact_paths["perception-encoder"]),
            "--width",
            "32",
            "--height",
            "32",
            "--visual-feature-count",
            "1",
            "--normalization",
            package.manifest.visual_normalization,
            "--output",
            str(encoded),
            "--receipt",
            str(visual_path),
        ],
    )
    assert encoding.main() == 0
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--package",
            str(root),
            "--validation-data",
            str(encoded),
            "--dataset-receipt",
            str(dataset_path),
            "--validation-observations",
            str(history),
            "--stream-groups",
            str(groups_path),
            "--visual-encoding-receipt",
            str(visual_path),
            "--output",
            str(output),
        ],
    )
    # 先证明真实入口拒绝缺失的动作风险证据；行为成功标签不能代替该证据。
    with pytest.raises(ValueError, match="CAUSAL_ADMISSION_ACTION_RISK_EVIDENCE_REQUIRED"):
        evaluation.main()
    assert not output.exists()
    assert not list(tmp_path.glob(".admission-inputs-*"))
    from test_risk_admission_evidence import synthetic_risk_evidence

    risk_sha = next(a.sha256 for a in package.manifest.artifacts if a.role == "risk-critic")
    # 本例只隔离风险证据来源，专测真实视觉/历史/行为入口；不算实际风险验收。
    monkeypatch.setattr(evaluation, "_evaluate_frozen_action_risk",
                        lambda *_args: synthetic_risk_evidence(risk_sha))
    assert evaluation.main() == 1
    result = json.loads(output.read_bytes())
    assert result["sample_count"] == 2
    assert result["admitted_to_simulation"] is False
    assert result["issue_codes"]
    assert result["dataset_receipt_sha256"] == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert result["policy_package_sha256"] == package.package_sha256
    assert not list(tmp_path.glob(".admission-inputs-*"))
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        evaluation.main()
    assert output.read_bytes() == original
