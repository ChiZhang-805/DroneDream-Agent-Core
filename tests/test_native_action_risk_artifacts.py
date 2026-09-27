"""Synthetic native-format admission tests, not physical safety evidence."""

import hashlib
import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_native_corrections import fixture

from dronedream_agent_core.contracts import VehicleAsset
from dronedream_agent_core.local_policy_training import (
    LocalPolicyTrainingConfig,
    train_local_risk_critic,
)
from dronedream_agent_core.training.action_risk_artifacts import (
    load_action_risk_dataset,
    validate_action_risk_splits,
)
from dronedream_agent_core.training.action_risk_training import (
    action_discrimination,
    risk_class_coverage,
    train_action_risk_expert,
    verify_risk_export,
)
from dronedream_agent_core.training.counterfactual_teacher import CounterfactualConfig
from dronedream_agent_core.training.native_episode import load_grounded_episode
from dronedream_agent_core.training.px4_environment import ASSET_FIELDS, Px4TrainingConfig


# 功能：
#   创建带车辆、地图、路线及逐文件摘要的合成原生回合，运行器路径明确不可执行。
# 输入：
#   root：测试独占根目录。
# 输出：
#   episode：合成回合目录。
#   teacher_config：离线几何教师配置。
def native_episode(root):
    oracle, observation, transition, terminal = fixture(root)
    vehicle = VehicleAsset(
        asset_id="unit",
        name="unit",
        dry_mass_kg=1.0,
        max_takeoff_mass_kg=2.0,
        body_radius_m=0.2,
        body_height_m=0.3,
        max_speed_mps=1.0,
        max_acceleration_mps2=2.0,
        qualified_range_m=100.0,
        reserve_battery_percent=20.0,
        max_pickup_payload_kg=0.1,
        sensors=["depth"],
    )
    values = {
        "vehicle": vehicle.model_dump(),
        "semantic": {"collision_primitives": oracle.teacher.geometry.primitives},
        "route": {"positions_m": [{"x": 0., "y": 0., "z": 1.},
                                   {"x": 2., "y": 0., "z": 1.}]},
    }
    paths, hashes = {}, {}
    for name in ASSET_FIELDS:
        paths[name] = root / (name + ".json")
        content = json.dumps(values.get(name, {"unit": name})).encode()
        paths[name].write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    config = Px4TrainingConfig(
        runner=root / "not-an-executable.py",
        output_root=root,
        mission_id="unit",
        expert_role="local-navigation-policy",
        **paths,
        asset_sha256=hashes,
        minimum_enu_m=(-10.0, -10.0, -5.0),
        maximum_enu_m=(10.0, 10.0, 8.0),
    )
    reset = transition.parent / "reset.json"
    data = json.loads(reset.read_bytes())
    data["config"] = config.model_dump(mode="json")
    reset.write_text(json.dumps(data))
    row = json.loads(transition.read_bytes())
    row["source_observation"]["map_sha256"] = hashes["semantic"]
    row["step"]["observation"]["map_sha256"] = hashes["semantic"]
    transition.write_text(json.dumps(row))
    episode, teacher_config = transition.parent, oracle.teacher.config
    return episode, teacher_config


# 功能：
#   使用真实离线构建入口为单个合成观测生成动作条件风险数据，不启动原生仿真。
# 输入：
#   root：源回合与输出目录的父目录。
#   monkeypatch：替换命令行参数的测试工具。
# 输出：
#   output：已生成的风险专用数据集目录。
def dataset(root, monkeypatch, *, legacy=False):
    episode, config = native_episode(root)
    if not legacy:
        config.risk_label_semantics = "observation-clearance-v2"
    config_path = root / "teacher.json"
    config_path.write_text(config.model_dump_json())
    output = root / "risk-only"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build",
            "--episode",
            str(episode),
            "--teacher-config",
            str(config_path),
            "--output",
            str(output),
            "--maximum-observations-per-episode",
            "1",
        ],
    )
    script = Path(__file__).resolve().parents[1] / "scripts/build_native_action_risk_dataset.py"
    if legacy:
        monkeypatch.setattr(sys, 'argv', [*sys.argv, '--reproduce-legacy-labels'])
    assert runpy.run_path(str(script))["main"]() == 0
    return output


# 功能：
#   为每个测试独立生成有来源绑定的风险数据集，避免共享故障注入状态。
# 输入：
#   tmp_path：本测试独占目录。
#   monkeypatch：命令行参数替换工具。
# 输出：
#   output：该测试的风险数据集目录。
@pytest.fixture
def risk_data(tmp_path, monkeypatch):
    output = dataset(tmp_path, monkeypatch)
    return output


# 功能：
#   修改合成数据后重算文件摘要，以验证语义绑定不能仅凭摘要自洽就被绕过。
# 输入：
#   root：数据集目录。
#   name：清单中的数据类别。
#   mutation：对解析记录列表施加改动的回调。
# 输出：
#   None：不返回业务数据。
def rehash(root, name, mutation):
    from dronedream_agent_core.training.action_risk_artifacts import FILES

    path = root / FILES[name]
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    mutation(rows)
    content = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    path.write_bytes(content)
    receipt_path = root / "dataset-receipt.jsonl"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["file_sha256"][name] = hashlib.sha256(content).hexdigest()
    receipt_path.write_text(json.dumps(receipt))


# 功能：
#   验证一个独立观测对应多种动作探针，但不能被统计成多个独立观测；复查源文件变化。
# 输入：
#   risk_data：已构建的合成风险数据集。
# 输出：
#   None：不返回业务数据。
def test_complete_native_format_roundtrip_and_source_change_guard(risk_data):
    loaded = load_action_risk_dataset(risk_data)
    assert len(loaded.samples) == 161 and len(loaded.observations) == 1
    assert risk_class_coverage([loaded])["independent_observation_count"] == 1
    assert loaded.receipt["qualified_for_flight"] is False
    episode = risk_data.parent / "episode-unit"
    native = load_grounded_episode(
        episode,
        CounterfactualConfig.model_validate(loaded.receipt["teacher_config"]),
    )
    native.verify_sources(episode)
    (episode / "transition-000001.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        native.verify_sources(episode)


# 功能：
#   验证各类数据文件的原始字节被修改后均拒绝，连空白变化也必须符合摘要。
# 输入：
#   risk_data：待注入故障的数据集。
#   name：清单中的文件类别。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "name",
    [
        "action-risk",
        "action-records",
        "source-observations",
        "counterfactual-receipts",
        "stream-groups",
    ],
)
def test_modified_bytes_rejected(risk_data, name):
    from dronedream_agent_core.training.action_risk_artifacts import FILES

    path = risk_data / FILES[name]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="CONTENT_CHANGED"):
        load_action_risk_dataset(risk_data)


# 功能：
#   验证重算摘要后仍不能给风险标签借用错误物理控制或其他观测的特征。
# 输入：
#   risk_data：待改动的合成数据集。
#   mutation：施加错误标签或特征的回调。
#   error：预期拒绝原因。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda rows: rows[0].update(risk_proposed_control=[0.0, 0.0, 0.0, 0.0]),
            "PHYSICAL_CONTROL",
        ),
        (lambda rows: rows[0].update(risk_target=0.99), "PHYSICAL_CONTROL"),
        (lambda rows: rows[0]["state_features"].__setitem__(0, 0.777), "PHYSICAL_CONTROL"),
    ],
)
def test_rehashed_labels_cannot_borrow_wrong_physics_or_observations(risk_data, mutation, error):
    rehash(risk_data, "action-risk", mutation)
    with pytest.raises(ValueError, match=error):
        load_action_risk_dataset(risk_data)


# 功能：
#   验证重复同一个动作探针不能冒充覆盖了另一个尚未评估的动作。
# 输入：
#   risk_data：合成风险数据集。
# 输出：
#   None：不返回业务数据。
def test_same_source_probes_cannot_be_repeated_to_replace_missing_probes(risk_data):
    rehash(risk_data, "action-records", lambda rows: rows.__setitem__(1, rows[0]))
    with pytest.raises(ValueError, match="PROPOSAL_SOURCE"):
        load_action_risk_dataset(risk_data)


# 功能：
#   验证只用于评价的反事实动作记录不能被改成行为模仿标签并用于训练控制策略。
# 输入：
#   risk_data：合成反事实数据集。
# 输出：
#   None：不返回业务数据。
def test_rehashed_counterfactual_does_not_authorize_actor_training(risk_data):
    rehash(risk_data, "action-records", lambda rows: rows[0].update(behavior_cloning_label=True))
    with pytest.raises(ValueError, match="PROPOSAL_SOURCE"):
        load_action_risk_dataset(risk_data)


# 功能：
#   验证留出按整条空间路线隔离，改文件、回合名或动力学配置均不能掩盖不合法分割。
# 输入：
#   risk_data：包含路线和教师配置身份的数据集。
# 输出：
#   None：不返回业务数据。
def test_holdout_is_entire_route_not_a_new_file_or_episode_name(risk_data):
    first = load_action_risk_dataset(risk_data)
    second = replace(
        first,
        receipt_sha256="c" * 64,
        observations=tuple(
            row.model_copy(update={"episode_id": "another-flight"}) for row in first.observations
        ),
    )
    with pytest.raises(ValueError, match="HOLDOUT_OVERLAP"):
        validate_action_risk_splits([first], [second])
    with pytest.raises(ValueError, match="DATASET_REPEATED"):
        validate_action_risk_splits([first], [first])
    third = replace(second, groups=frozenset({"d" * 64}), teacher_config_sha256="b" * 64)
    with pytest.raises(ValueError, match="PHYSICS_SCOPE_MISMATCH"):
        validate_action_risk_splits([first], [third])


# 功能：
#   验证缺少危险观测覆盖时保存拒绝回执，不优化或导出看似合格的风险模型。
# 输入：
#   risk_data：只含安全观测的合成数据集。
# 输出：
#   None：不返回业务数据。
def test_safe_only_dataset_is_recorded_but_never_optimized(risk_data):
    first = load_action_risk_dataset(risk_data)
    second = replace(
        first,
        receipt_sha256="c" * 64,
        groups=frozenset({"d" * 64}),
        observations=tuple(
            row.model_copy(update={"episode_id": "validation-flight"}) for row in first.observations
        ),
    )
    output = risk_data.parent / "rejected-training"
    receipt = train_action_risk_expert(
        [first], [second], LocalPolicyTrainingConfig(epoch_count=1), output
    )
    assert not receipt["optimized"] and not receipt["offline_validation_passed"]
    assert not receipt["qualified_for_flight"] and not (output / "risk-critic.onnx").exists()
    assert any("UNSAFE_OBSERVATION_COUNT" in code for code in receipt["issue_codes"])
    assert (output / "training-receipt.jsonl").is_file()


# 功能：
#   验证恒定环境评分不能冒充动作区分能力，且无可区分标签时指标应明确为未知。
# 输入：
#   risk_data：提供原始观测分组的合成数据集。
# 输出：
#   None：不返回业务数据。
def test_constant_environment_commentary_is_not_action_discrimination(risk_data):
    data = load_action_risk_dataset(risk_data)
    samples = tuple(
        row.model_copy(update={"risk_target": float(i % 2)}) for i, row in enumerate(data.samples)
    )
    data = replace(data, samples=samples)  # pure metric fixture, not admitted training data
    assert action_discrimination([data], np.zeros(len(samples)))["correct_fraction"] == 0.0
    assert (
        action_discrimination([data], np.array([s.risk_target for s in samples]))[
            "correct_fraction"
        ]
        == 1.0
    )
    assert (
        action_discrimination([load_action_risk_dataset(risk_data)], np.zeros(len(samples)))[
            "correct_fraction"
        ]
        is None
    )


# 功能：
#   实际训练小型风险网络并导出 ONNX，比较原网络和运行时推理的数值一致性。
# 输入：
#   tmp_path：独占模型导出目录。
# 输出：
#   None：不返回业务数据。
def test_export_preserves_physical_action_conditioned_network(tmp_path):
    from test_action_conditioned_risk import _samples

    from dronedream_agent_core.local_policy_training import export_local_risk_critic_onnx

    samples = _samples()
    model, _ = train_local_risk_critic(
        samples, LocalPolicyTrainingConfig(hidden_feature_count=16, epoch_count=20)
    )
    path = tmp_path / "critic.onnx"
    export_local_risk_critic_onnx(model, path)
    assert verify_risk_export(model, path.read_bytes(), samples) < 1e-5


# 功能：
#   验证落地未确认时即拒绝加载，不先读取资产或启动离线教师标注。
# 输入：
#   tmp_path：合成原生回合根目录。
# 输出：
#   None：不返回业务数据。
def test_unconfirmed_landing_is_rejected_before_asset_reads(tmp_path):
    episode, config = native_episode(tmp_path)
    terminal = episode / "flight/simulation/native-terminal-lifecycle.json"
    terminal.write_text('{"terminal_state":"IN_AIR"}')
    with pytest.raises(ValueError, match="LANDING_NOT_CONFIRMED"):
        load_grounded_episode(episode, config)


# 功能：
#   验证风险探针在优化器和因果样本编译入口均被拒绝，而非只依赖命令行类别说明。
# 输入：
#   risk_data：只供风险训练的合成探针数据。
# 输出：
#   None：不返回业务数据。
def test_risk_probes_are_rejected_by_actor_optimizers_not_just_cli_labels(risk_data):
    from dronedream_agent_core.local_policy_training import train_local_policy
    from dronedream_agent_core.training.causal_policy import causal_examples

    samples = list(load_action_risk_dataset(risk_data).samples)
    with pytest.raises(ValueError, match="CANNOT_TRAIN_OR_EVALUATE_BEHAVIOR"):
        train_local_policy(samples, LocalPolicyTrainingConfig(epoch_count=1))
    with pytest.raises(ValueError, match="CANNOT_TRAIN_OR_EVALUATE_BEHAVIOR"):
        causal_examples(samples, stream_groups={})


# 功能：
#   验证原网络参考输出为 NaN 时不能把有限的 ONNX 输出判作零误差且通过验收。
# 输入：
#   monkeypatch：替换原网络和 ONNX 会话的测试工具。
# 输出：
#   None：不返回业务数据。
def test_nonfinite_reference_cannot_pass_export_equivalence(monkeypatch):
    import onnxruntime

    from dronedream_agent_core.training import action_risk_training

    class SyntheticSession:
        # 功能：
        #   模拟一个返回有限概率值的 ONNX 会话，用于对照非法原网络输出。
        # 输入：
        #   self：测试会话。
        #   names：本例不使用的输出节点名。
        #   feeds：本例不使用的输入张量映射。
        # 输出：
        #   outputs：包含有限单值张量的输出列表。
        def run(self, names, feeds):
            outputs = [np.array([0.5], dtype=np.float32)]
            return outputs

    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **kw: SyntheticSession())
    monkeypatch.setattr(action_risk_training, "_predict",
                        lambda model, samples: (np.array([np.nan]), {}))
    with pytest.raises(ValueError, match="EXPORT_OUTPUT_INVALID"):
        verify_risk_export(None, b"synthetic-session", [object()])
