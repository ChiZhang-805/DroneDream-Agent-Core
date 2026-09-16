from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


# 功能：
#   显式加载仓库当前离线评估入口，不搜索历史构建或同名脚本作为替代。
# 输入：
#   无：路径由当前测试所在仓库确定。
# 输出：
#   module：本次加载的评估模块。
def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "evaluate_local_policy_offline.py"
    spec = importlib.util.spec_from_file_location(
        "test_local_policy_admission_dataset_script", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 功能：
#   构造训练与调参任务不重叠的合成来源回执，只用于数据隔离边界测试。
# 输入：
#   无：使用固定的测试摘要。
# 输出：
#   receipt：包含两种历史数据及任务来源的回执对象。
def _training_receipt() -> dict[str, object]:
    receipt = {
        "schema_version": "dronedream.local-advisor-training-receipt.v1",
        "training_data_sha256": "a" * 64,
        "validation_data_sha256": "b" * 64,
        "training_source_evidence_sha256": ["c" * 64],
        "validation_source_evidence_sha256": ["d" * 64],
        "fresh_admission_validation_required": True,
    }
    return receipt


# 功能：
#   根据模型明确声明区分旧状态历史和专用负载历史，不能仅凭专家名称选用错误维数。
# 输入：
#   input_names：模拟会话真实声明的输入名。
#   expected_history_name：应消费的历史字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    ("input_names", "expected_history_name"),
    [
        (
            ("payload_features", "state_history", "history_mask"),
            "state_history",
        ),
        (
            ("payload_features", "payload_history", "history_mask"),
            "payload_history",
        ),
    ],
)
def test_payload_admission_uses_packaged_history_contract(
    input_names: tuple[str, ...], expected_history_name: str
) -> None:
    module = _module()
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name=name) for name in input_names]
    )
    sample = SimpleNamespace(
        payload_features=[0.0] * 8,
        state_history=[[1.0] * 16 for _ in range(8)],
        payload_history=[[2.0] * 24 for _ in range(8)],
        history_mask=[1.0] * 8,
    )

    feeds = module._advisor_feeds(
        session,
        sample,
        role="payload-dynamics-adapter",
    )

    assert set(feeds) == set(input_names)
    assert feeds[expected_history_name].shape == (
        1,
        8,
        24 if expected_history_name == "payload_history" else 16,
    )


# 功能：
#   声明额外未知输入时明确拒绝，不用零向量或未声明字段拼凑通过。
# 输入：
#   无：构造显式未知输入的会话替身。
# 输出：
#   None：不返回业务数据。
def test_payload_admission_rejects_unknown_artifact_input_contract() -> None:
    module = _module()
    session = SimpleNamespace(
        get_inputs=lambda: [
            SimpleNamespace(name="payload_features"),
            SimpleNamespace(name="payload_history"),
            SimpleNamespace(name="history_mask"),
            SimpleNamespace(name="undeclared_extra"),
        ]
    )
    sample = SimpleNamespace(
        payload_features=[0.0] * 8,
        state_history=[[1.0] * 16 for _ in range(8)],
        payload_history=[[2.0] * 24 for _ in range(8)],
        history_mask=[1.0] * 8,
    )

    with pytest.raises(ValueError, match="input contract mismatch"):
        module._advisor_feeds(
            session,
            sample,
            role="payload-dynamics-adapter",
        )


# 功能：
#   写入合成顾问数据的划分与任务摘要，供独立性检查读取真实文件。
# 输入：
#   path：本测试回执路径。
#   data_sha256：验证数据的实际摘要。
#   source_sha256：验证任务身份。
# 输出：
#   None：不返回业务数据。
def _write_dataset_receipt(path: Path, *, data_sha256: str, source_sha256: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-advisor-dataset-receipt.v1",
                "split_method": "held-out-complete-mission",
                "verified_sources_required": True,
                "training_output_sha256": "e" * 64,
                "validation_output_sha256": data_sha256,
                "training_sources": [{"mission_evidence_sha256": "f" * 64}],
                "validation_sources": [{"mission_evidence_sha256": source_sha256}],
            }
        ),
        encoding="utf-8",
    )


# 功能：
#   恢复冷启动仅在强留出指标和类别数量达标时允许采集，降低危险样本支持必须被拒绝。
# 输入：
#   tmp_path：合成引导回执路径。
# 输出：
#   None：不返回业务数据。
def test_recovery_bootstrap_scope_requires_strong_held_out_training_evidence(
    tmp_path: Path,
) -> None:
    module = _module()
    receipt = tmp_path / "recovery-bootstrap.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-augmentation-receipt.v1",
                "expert_role": "recovery-policy",
                "bootstrap_only": True,
                "deployment_scope": "simulation-only",
                "fresh_admission_validation_required": True,
                "qualification_granted": False,
                "validation_class_counts": {"safe": 40, "risky": 10},
                "validation_metrics": {
                    "motion_authorization_accuracy": 1.0,
                    "candidate_selection_accuracy": 0.95,
                    "risk_hold_recall": 1.0,
                    "safe_motion_recall": 0.96,
                    "risk_mean_absolute_error": 0.05,
                },
            }
        ),
        encoding="utf-8",
    )

    assert module._navigation_admission_scope(receipt) == (
        "recovery-bootstrap-simulation",
        20,
        3,
    )

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["validation_class_counts"]["risky"] = 9
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="held-out validation is insufficient"):
        module._navigation_admission_scope(receipt)


# 功能：
#   新顾问准入数据必须有与训练和调参不同的任务身份，解析后保留真实分区。
# 输入：
#   tmp_path：新数据和对应回执目录。
# 输出：
#   None：不返回业务数据。
def test_advisor_admission_requires_new_complete_source_mission(tmp_path: Path) -> None:
    module = _module()
    data = tmp_path / "admission.jsonl"
    data.write_text("fresh admission rows\n", encoding="utf-8")
    data_sha256 = module._sha256(data)
    receipt = tmp_path / "dataset-receipt.json"
    _write_dataset_receipt(
        receipt,
        data_sha256=data_sha256,
        source_sha256="1" * 64,
    )

    split, sources = module._validate_fresh_advisor_admission_data(
        data_path=data,
        dataset_receipt_path=receipt,
        training_receipt=_training_receipt(),
    )

    assert split == "validation"
    assert sources == ["1" * 64]


# 功能：
#   即使数据文件字节不同，只要仍来自同一个历史任务，就不能视为独立准入数据。
# 输入：
#   tmp_path：重复任务的合成数据和回执。
# 输出：
#   None：不返回业务数据。
def test_advisor_admission_rejects_reused_source_mission(tmp_path: Path) -> None:
    module = _module()
    data = tmp_path / "admission.jsonl"
    data.write_text("different rows from an old mission\n", encoding="utf-8")
    receipt = tmp_path / "dataset-receipt.json"
    _write_dataset_receipt(
        receipt,
        data_sha256=module._sha256(data),
        source_sha256="c" * 64,
    )

    with pytest.raises(ValueError, match="reuses a training or tuning source"):
        module._validate_fresh_advisor_admission_data(
            data_path=data,
            dataset_receipt_path=receipt,
            training_receipt=_training_receipt(),
        )


# 功能：
#   写入候选时代导航数据的两分区来源结构，用于明确测试该历史接口而非当前因果空间划分。
# 输入：
#   path：本测试回执路径。
#   training_sha256：训练数据身份。
#   validation_sha256：验证数据身份。
#   training_source：训练任务身份。
#   validation_source：验证任务身份。
# 输出：
#   None：不返回业务数据。
def _write_policy_dataset_receipt(
    path: Path,
    *,
    training_sha256: str,
    validation_sha256: str,
    training_source: str,
    validation_source: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-dataset-receipt.v1",
                "split_method": "held-out-source-campaign",
                "verified_sources_required": True,
                "training_output_sha256": training_sha256,
                "validation_output_sha256": validation_sha256,
                "training_sources": [{"mission_evidence_sha256": training_source}],
                "validation_sources": [{"mission_evidence_sha256": validation_source}],
            }
        ),
        encoding="utf-8",
    )


# 功能：
#   导航增补回执绑定原训练划分后，只接受历史来源之外的新任务进行准入。
# 输入：
#   tmp_path：历史、增补和新准入回执目录。
# 输出：
#   None：不返回业务数据。
def test_navigation_admission_requires_fresh_source_mission(tmp_path: Path) -> None:
    module = _module()
    historical = tmp_path / "historical-receipt.json"
    _write_policy_dataset_receipt(
        historical,
        training_sha256="a" * 64,
        validation_sha256="b" * 64,
        training_source="c" * 64,
        validation_source="d" * 64,
    )
    admission = tmp_path / "admission-receipt.json"
    _write_policy_dataset_receipt(
        admission,
        training_sha256="e" * 64,
        validation_sha256="f" * 64,
        training_source="0" * 64,
        validation_source="1" * 64,
    )
    augmentation = tmp_path / "augmentation-receipt.json"
    augmentation.write_text(
        json.dumps(
            {
                "schema_version": ("dronedream.local-policy-augmentation-receipt.v1"),
                "dataset_receipt_sha256": module._sha256(historical),
                "fresh_admission_validation_required": True,
                "qualification_granted": False,
            }
        ),
        encoding="utf-8",
    )

    split, sources = module._validate_fresh_navigation_admission_data(
        source_data_sha256="f" * 64,
        admission_dataset_receipt_path=admission,
        augmentation_receipt_path=augmentation,
        training_dataset_receipt_path=historical,
    )

    assert split == "validation"
    assert sources == ["1" * 64]


# 功能：
#   训练任务再次出现于准入验证侧时拒绝，不能因文件和其他任务摘要不同而放行。
# 输入：
#   tmp_path：有来源交集的合成回执目录。
# 输出：
#   None：不返回业务数据。
def test_navigation_admission_rejects_training_source_reuse(tmp_path: Path) -> None:
    module = _module()
    historical = tmp_path / "historical-receipt.json"
    _write_policy_dataset_receipt(
        historical,
        training_sha256="a" * 64,
        validation_sha256="b" * 64,
        training_source="c" * 64,
        validation_source="d" * 64,
    )
    admission = tmp_path / "admission-receipt.json"
    _write_policy_dataset_receipt(
        admission,
        training_sha256="e" * 64,
        validation_sha256="f" * 64,
        training_source="0" * 64,
        validation_source="c" * 64,
    )
    augmentation = tmp_path / "augmentation-receipt.json"
    augmentation.write_text(
        json.dumps(
            {
                "schema_version": ("dronedream.local-policy-augmentation-receipt.v1"),
                "dataset_receipt_sha256": module._sha256(historical),
                "fresh_admission_validation_required": True,
                "qualification_granted": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reuses a training or tuning source"):
        module._validate_fresh_navigation_admission_data(
            source_data_sha256="f" * 64,
            admission_dataset_receipt_path=admission,
            augmentation_receipt_path=augmentation,
            training_dataset_receipt_path=historical,
        )


# 功能：
#   已替换的负载专家不能继承旧指标，未改写异常专家可保留；控制特征语义变化后也须拒绝。
# 输入：
#   tmp_path：两种角色及继承回执的测试目录。
#   monkeypatch：提供显式模型包替身的工具，不加载或运行产品模型。
# 输出：
#   None：不返回业务数据。
def test_composition_replacement_is_not_treated_as_preserved_advisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    package_root = tmp_path / "package"
    package_root.mkdir()
    replacement_hash = "1" * 64
    preserved_hash = "2" * 64
    package = type(
        "Package",
        (),
        {
            "package_sha256": "3" * 64,
            "manifest": type(
                "Manifest",
                (),
                {
                    "vehicle_sha256": "4" * 64,
                    "sensor_contract_sha256": "5" * 64,
                    "map_sha256": "6" * 64,
                    "pilot_control_mode": None,
                    "control_feature_contract_sha256": None,
                    "artifacts": (
                        type(
                            "Artifact",
                            (),
                            {
                                "role": "payload-dynamics-adapter",
                                "sha256": replacement_hash,
                            },
                        )(),
                        type(
                            "Artifact",
                            (),
                            {
                                "role": "state-anomaly-detector",
                                "sha256": preserved_hash,
                            },
                        )(),
                    ),
                },
            )(),
        },
    )()
    monkeypatch.setattr(module, "load_local_policy_package", lambda _path: package)
    composition = tmp_path / "composition.json"
    composition.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-composition-receipt.v1",
                "source_package_sha256": "7" * 64,
                "package_sha256": package.package_sha256,
                "preserved_artifact_sha256": {
                    "state-anomaly-detector": preserved_hash,
                },
                "advisor_evidence": [
                    {
                        "role": "payload-dynamics-adapter",
                        "artifact_sha256": replacement_hash,
                    }
                ],
                "qualification_granted": False,
            }
        ),
        encoding="utf-8",
    )
    inherited = tmp_path / "inherited.json"
    inherited.write_text(
        json.dumps(
            {
                "receipt_id": "policy-simulation-admission-" + "8" * 32,
                "policy_package_sha256": "7" * 64,
                "dataset_receipt_sha256": "9" * 64,
                "evaluation_suite_sha256": "a" * 64,
                "map_sha256": package.manifest.map_sha256,
                "vehicle_sha256": package.manifest.vehicle_sha256,
                "sensor_contract_sha256": package.manifest.sensor_contract_sha256,
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
                "advisor_metrics": {
                    role: {
                        "sample_count": 40,
                        "risky_sample_count": 20,
                        "safe_sample_count": 20,
                        "risk_hold_recall": 1.0,
                        "safe_motion_recall": 1.0,
                        "risk_mean_absolute_error": 0.0,
                        "controller_step_scale_mean_absolute_error": (
                            0.0 if role == "payload-dynamics-adapter" else None
                        ),
                        "p99_inference_latency_ms": 1.0,
                    }
                    for role in (
                        "payload-dynamics-adapter",
                        "state-anomaly-detector",
                    )
                },
                "combined_p99_inference_latency_ms": 5.0,
                "admitted_to_simulation": True,
                "issue_codes": [],
            }
        ),
        encoding="utf-8",
    )

    metrics, evidence = module._load_inherited_advisor_evidence(
        package_path=package_root,
        preservation_receipt_path=composition,
        inherited_admission_receipt_path=inherited,
        trainable_advisor_roles={
            "payload-dynamics-adapter",
            "state-anomaly-detector",
        },
    )

    assert set(metrics) == {"state-anomaly-detector"}
    assert [item["role"] for item in evidence] == ["state-anomaly-detector"]

    # Identical artifact bytes are not permission to reuse old input semantics.
    package.manifest.pilot_control_mode = "normalized-body-velocity"
    package.manifest.control_feature_contract_sha256 = "b" * 64
    with pytest.raises(ValueError, match="semantics differ"):
        module._load_inherited_advisor_evidence(
            package_path=package_root,
            preservation_receipt_path=composition,
            inherited_admission_receipt_path=inherited,
            trainable_advisor_roles={"payload-dynamics-adapter", "state-anomaly-detector"},
        )


# 功能：
#   地图专用训练可以保留同载具、同传感器和同输入契约的未改写顾问，不继承导航或飞行资格。
# 输入：
#   tmp_path：原准入和地图训练回执目录。
#   monkeypatch：构造保留顾问字节的模型包替身。
# 输出：
#   None：不返回业务数据。
def test_training_receipt_preserves_advisor_across_map_specialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    package_root = tmp_path / "package"
    package_root.mkdir()
    artifact_hash = "2" * 64
    package = type(
        "Package",
        (),
        {
            "package_sha256": "3" * 64,
            "manifest": type(
                "Manifest",
                (),
                {
                    "vehicle_sha256": "4" * 64,
                    "sensor_contract_sha256": "5" * 64,
                    "map_sha256": "6" * 64,
                    "pilot_control_mode": None,
                    "control_feature_contract_sha256": None,
                    "artifacts": (
                        type(
                            "Artifact",
                            (),
                            {
                                "role": "state-anomaly-detector",
                                "sha256": artifact_hash,
                            },
                        )(),
                    ),
                },
            )(),
        },
    )()
    monkeypatch.setattr(module, "load_local_policy_package", lambda _path: package)
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-training-receipt.v1",
                "source_package_sha256": "7" * 64,
                "source_vehicle_sha256": "4" * 64,
                "package_sha256": package.package_sha256,
                "vehicle_sha256": "4" * 64,
                "sensor_contract_sha256": "5" * 64,
                "map_sha256": "6" * 64,
                "preserved_artifact_sha256": {
                    "state-anomaly-detector": artifact_hash,
                },
                "preserved_advisor_input_contract_unchanged": True,
                "fresh_simulation_admission_required": True,
                "qualification_granted": False,
            }
        ),
        encoding="utf-8",
    )
    inherited = tmp_path / "inherited.json"
    inherited.write_text(
        json.dumps(
            {
                "receipt_id": "policy-simulation-admission-" + "8" * 32,
                "policy_package_sha256": "7" * 64,
                "dataset_receipt_sha256": "9" * 64,
                "evaluation_suite_sha256": "a" * 64,
                "map_sha256": None,
                "vehicle_sha256": "4" * 64,
                "sensor_contract_sha256": "5" * 64,
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
                "advisor_metrics": {
                    "state-anomaly-detector": {
                        "sample_count": 40,
                        "risky_sample_count": 20,
                        "safe_sample_count": 20,
                        "risk_hold_recall": 1.0,
                        "safe_motion_recall": 1.0,
                        "risk_mean_absolute_error": 0.0,
                        "controller_step_scale_mean_absolute_error": None,
                        "p99_inference_latency_ms": 1.0,
                    }
                },
                "combined_p99_inference_latency_ms": 5.0,
                "admitted_to_simulation": True,
                "issue_codes": [],
            }
        ),
        encoding="utf-8",
    )

    metrics, evidence = module._load_inherited_advisor_evidence(
        package_path=package_root,
        preservation_receipt_path=training,
        inherited_admission_receipt_path=inherited,
        trainable_advisor_roles={"state-anomaly-detector"},
    )

    assert set(metrics) == {"state-anomaly-detector"}
    assert evidence[0]["evidence_kind"] == "byte-identical-inherited-admission"
