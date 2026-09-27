"""Complete ten-role synthetic package identity/atomic assembly tests."""

import hashlib
import json
from pathlib import Path

import onnx
import pytest
import torch
from test_causal_policy import samples
from test_local_policy_admission_dataset import _module
from test_local_policy_packages import (
    _write_brightness_perception_encoder,
    _write_high_anomaly_detector,
    _write_low_cross_modal_consistency_critic,
    _write_low_perception_health_critic,
    _write_low_settle_stability_critic,
    _write_payload_adapter,
)

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.local_expert_harness import NAVIGATION_EXPERT_ROLES
from dronedream_agent_core.local_policy_packages import load_local_policy_package
from dronedream_agent_core.local_policy_training import (
    LocalPolicyTrainingConfig,
    _new_model,
    write_local_policy_package,
)
from dronedream_agent_core.training.causal_policy import (
    CausalPilotPolicy,
    CausalPolicyConfig,
    export_causal_policy,
)
from dronedream_agent_core.training.causal_replay import REPLAY_FILES
from dronedream_agent_core.training.mission_groups import SPATIAL_SPLIT_CONTRACT
from dronedream_agent_core.training.policy_packaging import (
    assemble_causal_package,
    validate_causal_base,
)
from dronedream_agent_core.training.visual_lineage import VisualInputContract


# 功能：
#   为合成操纵模型生成与权重、视觉输入和独立分组绑定的测试回执，不充当真实训练证据。
# 输入：
#   paths：三个操纵角色与合成 ONNX 文件的映射。
#   base_root：提供传感器和视觉契约的测试基座。
# 输出：
#   receipts：按操纵角色索引的测试回执字典。
def training_receipts(paths, base_root):
    manifest = load_local_policy_package(base_root).manifest
    encoder = next(a.sha256 for a in manifest.artifacts if a.role == "perception-encoder")
    receipts = {
        role: {
            "architecture": "causal-gru-control",
            "split_contract": SPATIAL_SPLIT_CONTRACT,
            "replay_artifact_sha256": {name: "c" * 64 for name in REPLAY_FILES},
            "expert_role": role,
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "feature_contract_sha256": manifest.control_feature_contract_sha256,
            "config": {"history_length": 4, "visual_feature_count": manifest.visual_feature_count},
            "metrics": {
                "training_window_count": 40,
                "validation_window_count": 20,
                "training_groups": ["a" * 64],
                "validation_groups": ["b" * 64],
            },
            "input_sha256": {"train": "c" * 64, "validation": "d" * 64},
            "visual_input_contract": VisualInputContract.from_manifest(
                manifest, encoder
            ).model_dump(),
            "qualified_for_flight": False,
        }
        for role, path in paths.items()
    }
    return receipts


# 功能：
#   构造包含十角色接口的轻量 ONNX 基座，仅供打包与后端加载测试。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   root：合成基座包目录。
@pytest.fixture
def base_package(tmp_path):
    paths = {}
    for name, writer in (
        ("state_anomaly_detector", _write_high_anomaly_detector),
        ("perception_health_critic", _write_low_perception_health_critic),
        ("settle_stability_critic", _write_low_settle_stability_critic),
        ("payload_dynamics_adapter", _write_payload_adapter),
        ("cross_modal_consistency_critic", _write_low_cross_modal_consistency_critic),
        ("perception_encoder", _write_brightness_perception_encoder),
    ):
        path = tmp_path / f"{name}.onnx"
        writer(path)
        paths[name + "_path"] = path
    config = LocalPolicyTrainingConfig(hidden_feature_count=8)
    policy = _new_model(
        config, visual_feature_count=1, realtime_feature_count=446, include_pilot_control=True
    )
    critic = _new_model(config, realtime_feature_count=446, action_conditioned=True)
    root = tmp_path / "base"
    write_local_policy_package(
        output_root=root,
        model=policy,
        risk_model=critic,
        precision_model=policy,
        recovery_model=policy,
        package_id="unit.complete-control",
        display_name="Unit fixture",
        scope="general",
        vehicle_sha256="a" * 64,
        sensor_contract_sha256="b" * 64,
        maximum_inference_latency_ms=100,
        visual_width=32,
        visual_height=32,
        visual_feature_count=1,
        control_feature_contract_sha256=CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        **paths,
    )
    return root


# 功能：
#   导出三个同契约的合成因果操纵模型，并在结束或失败后恢复 Torch 线程设置。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   paths：三个操纵角色与新 ONNX 文件的映射。
def replacements(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        policy = CausalPilotPolicy(
            CausalPolicyConfig(
                history_length=4,
                visual_feature_count=1,
                encoder_width=32,
                recurrent_width=32,
                head_width=32,
                epochs=1,
            )
        )
        paths = {}
        for role in NAVIGATION_EXPERT_ROLES:
            paths[role] = tmp_path / f"causal-{role}.onnx"
            export_causal_policy(policy, paths[role])
        return paths
    finally:
        torch.set_num_threads(previous)


# 功能：
#   1. 验证三个操纵模型整体替换，其余专家权重、基座与完整来源证据保持一致。
#   2. 通过实际下游评价器检查可继承的顾问证据，拒绝特征语义不符的旧准入记录。
# 输入：
#   base_package：合成十角色基座。
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_atomic_package_replaces_manoeuvres_preserves_advisors(base_package, tmp_path):
    base_package = complete_current_base(base_package, tmp_path)
    original = load_local_policy_package(base_package)
    output = tmp_path / "assembled"
    paths = replacements(tmp_path)
    package = assemble_causal_package(
        base_root=base_package,
        navigation_models=paths,
        navigation_receipts=training_receipts(paths, base_package),
        output_root=output,
        package_id="unit.causal-complete",
        history_length=4,
    )
    assert len(package.artifact_paths) == 10
    assert package.manifest.navigation_architecture == "causal-gru-control"
    assert package.package_sha256 != original.package_sha256
    assert load_local_policy_package(base_package).package_sha256 == original.package_sha256
    for role, path in package.artifact_paths.items():
        if role not in NAVIGATION_EXPERT_ROLES:
            assert (
                hashlib.sha256(path.read_bytes()).digest()
                == hashlib.sha256(original.artifact_paths[role].read_bytes()).digest()
            )
    lineage = json.loads((output / "training-lineage.json").read_text())
    assert lineage["qualified_for_flight"] is False
    assert lineage["qualification_granted"] is False
    assert lineage["source_package_sha256"] == original.package_sha256
    assert lineage["package_sha256"] == package.package_sha256
    assert lineage["control_feature_contract_sha256"] == CURRENT_POLICY_FEATURE_CONTRACT_SHA256
    assert set(lineage["preserved_artifact_sha256"]) == (
        set(package.artifact_paths) - set(NAVIGATION_EXPERT_ROLES)
    )
    assert set(lineage["replacement_artifact_sha256"]) == set(NAVIGATION_EXPERT_ROLES)
    assert lineage["fresh_simulation_admission_required"] is True
    complete = json.loads((output / "assembly-receipt.json").read_bytes())
    assert len(complete["expert_evidence"]) == 10
    for evidence in complete["expert_evidence"].values():
        content = (output / evidence["training_receipt_path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == evidence["training_receipt_sha256"]

    # Exercise the actual downstream evaluator, not just the JSON shape.
    inherited_path = tmp_path / "base-admission.json"
    inherited = {
        "receipt_id": "policy-simulation-admission-" + "8" * 32,
        "policy_package_sha256": original.package_sha256,
        "control_feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "dataset_receipt_sha256": "9" * 64,
        "evaluation_suite_sha256": "a" * 64,
        "map_sha256": original.manifest.map_sha256,
        "vehicle_sha256": original.manifest.vehicle_sha256,
        "sensor_contract_sha256": original.manifest.sensor_contract_sha256,
        "sample_count": 40,
        "motion_sample_count": 20,
        "non_motion_sample_count": 20,
        "motion_authorization_accuracy": 1.0,
        "candidate_selection_accuracy": 1.0,
        "risk_hold_recall": 1.0,
        "safe_motion_recall": 1.0,
        "risk_mean_absolute_error": 0.0,
        "p50_inference_latency_ms": 1.0,
        "p95_inference_latency_ms": 2.0,
        "p99_inference_latency_ms": 3.0,
        "combined_p99_inference_latency_ms": 5.0,
        "admitted_to_simulation": True,
        "issue_codes": [],
        "advisor_metrics": {
            "state-anomaly-detector": {
                "sample_count": 40,
                "risky_sample_count": 20,
                "safe_sample_count": 20,
                "risk_hold_recall": 1.0,
                "safe_motion_recall": 1.0,
                "risk_mean_absolute_error": 0.0,
                "p99_inference_latency_ms": 1.0,
            }
        },
    }
    inherited_path.write_text(json.dumps(inherited), encoding="utf-8")
    evaluator = _module()
    metrics, evidence = evaluator._load_inherited_advisor_evidence(
        package_path=output,
        preservation_receipt_path=output / "training-lineage.json",
        inherited_admission_receipt_path=inherited_path,
        trainable_advisor_roles={"state-anomaly-detector"},
    )
    assert set(metrics) == {"state-anomaly-detector"}
    assert (
        evidence[0]["artifact_sha256"]
        == lineage["preserved_artifact_sha256"]["state-anomaly-detector"]
    )
    inherited["control_feature_contract_sha256"] = "0" * 64
    inherited_path.write_text(json.dumps(inherited), encoding="utf-8")
    with pytest.raises(ValueError, match="feature semantics differ"):
        evaluator._load_inherited_advisor_evidence(
            package_path=output,
            preservation_receipt_path=output / "training-lineage.json",
            inherited_admission_receipt_path=inherited_path,
            trainable_advisor_roles={"state-anomaly-detector"},
        )
    assert not list(tmp_path.glob(".causal-policy-staging-*"))


# 功能：
#   验证缺少操纵角色或视觉维数不匹配时，不会创建可加载的候选目录。
# 输入：
#   base_package：合成十角色基座。
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_partial_replacement_and_visual_mismatch_fail_before_output(base_package, tmp_path):
    with pytest.raises(ValueError, match="ALL_THREE"):
        assemble_causal_package(
            base_root=base_package,
            navigation_models={"local-navigation-policy": Path("not-read.onnx")},
            output_root=tmp_path / "partial",
            package_id="unit.partial",
            history_length=4,
        )
    assert not (tmp_path / "partial").exists()
    with pytest.raises(ValueError, match="VISUAL_WIDTH"):
        validate_causal_base(base_package, visual_feature_count=2)


# 功能：
#   验证外部权重不能绕过 ONNX 文件摘要进入模型包，拒绝后清理本次暂存目录。
# 输入：
#   base_package：合成十角色基座。
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_unbound_external_weights_never_enter_assembled_package(base_package, tmp_path):
    paths = replacements(tmp_path)
    path = paths["local-navigation-policy"]
    graph = onnx.load(str(path))
    onnx.save_model(
        graph,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="unbound-data.bin",
        size_threshold=0,
    )
    with pytest.raises(ValueError, match="EXTERNAL_WEIGHTS"):
        assemble_causal_package(
            base_root=base_package,
            navigation_models=paths,
            navigation_receipts=training_receipts(paths, base_package),
            output_root=tmp_path / "external",
            package_id="unit.external",
            history_length=4,
        )
    assert not (tmp_path / "external").exists()
    assert not list(tmp_path.glob(".causal-policy-staging-*"))


# 功能：
#   用实际准入后端验证跨专家观测可共同预热历史，只对历史填满后的目标角色样本计数。
# 输入：
#   base_package：合成十角色基座。
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_real_admission_backend_consumes_cross_role_history(base_package, tmp_path):
    from test_risk_admission_evidence import synthetic_risk_evidence

    base_package = complete_current_base(base_package, tmp_path)
    paths = replacements(tmp_path)
    package = assemble_causal_package(
        base_root=base_package,
        navigation_models=paths,
        navigation_receipts=training_receipts(paths, base_package),
        output_root=tmp_path / "temporal-package",
        package_id="unit.temporal-admission",
        history_length=4,
    )
    rows = samples()[:10]
    rows = [
        row.model_copy(
            update={
                "visual_features": [0.1],
                "navigation_expert_role": "precision-maneuver-policy"
                if i % 2
                else "recovery-policy",
            }
        )
        for i, row in enumerate(rows)
    ]
    dataset = tmp_path / "admission.jsonl"
    dataset.write_text("\n".join(row.model_dump_json() for row in rows), encoding="utf-8")
    metrics, latencies = _module()._evaluate_runtime_package(
        package.manifest_path.parent, dataset, navigation_role="precision-maneuver-policy",
        action_risk_evidence=synthetic_risk_evidence(
            next(a.sha256 for a in package.manifest.artifacts if a.role == 'risk-critic')),
    )
    assert metrics.sample_count == 4  # 3,5,7,9; first three observations only warm history
    assert len(latencies) == 4


# 功能：
#   将原始合成基座组装为带当前完整来源回执的测试包。
# 输入：
#   raw_base：尚未附带完整来源链的合成基座。
#   tmp_path：测试独立目录。
# 输出：
#   root：完整合成候选包目录。
def complete_current_base(raw_base, tmp_path):
    from test_complete_artifact_assembly import build_recipe

    from dronedream_agent_core.training.artifact_assembly import assemble_complete_ensemble

    source_root = tmp_path / "base-training-sources"
    source_root.mkdir()
    root = tmp_path / "current-base"
    assemble_complete_ensemble(
        recipe=build_recipe(raw_base, source_root),
        source_root=tmp_path,
        output_root=root,
    )
    return root


# 功能：
#   验证替换不能掩盖基座来源缺失、保留专家使用旧划分或跨角色留出污染。
# 输入：
#   base_package：合成十角色基座。
#   tmp_path：测试独立目录。
#   failure：要破坏的来源边界。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failure", ["unproven-base", "old-retained-role", "cross-role-leakage"])
def test_policy_replacement_cannot_hide_unproven_or_contaminated_advisors(
    base_package, tmp_path, failure
):
    if failure != "unproven-base":
        base_package = complete_current_base(base_package, tmp_path)
    paths = replacements(tmp_path)
    receipts = training_receipts(paths, base_package)
    if failure == "cross-role-leakage":
        receipts["local-navigation-policy"]["metrics"]["training_groups"] = ["b" * 64]
        receipts["local-navigation-policy"]["metrics"]["validation_groups"] = ["c" * 64]
    if failure == "old-retained-role":
        role = "payload-dynamics-adapter"
        path = base_package / "training-evidence" / f"{role}.json"
        data = json.loads(path.read_bytes())
        data.pop("dataset_split_method")
        path.write_text(json.dumps(data))
        lineage_path = base_package / "assembly-receipt.json"
        lineage = json.loads(lineage_path.read_bytes())
        lineage["expert_evidence"][role]["training_receipt_sha256"] = hashlib.sha256(
            path.read_bytes()).hexdigest()
        lineage_path.write_text(json.dumps(lineage))
    with pytest.raises(ValueError, match="LINEAGE|SPATIAL_SPLIT|HOLDOUT_LEAKAGE"):
        assemble_causal_package(base_root=base_package, navigation_models=paths,
            navigation_receipts=receipts, output_root=tmp_path / "rejected-package",
            package_id="unit.rejected", history_length=4)
    assert not (tmp_path / "rejected-package").exists()
