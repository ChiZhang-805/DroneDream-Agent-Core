from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from test_local_policy_quality import _metrics

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.local_policy_packages import (
    LoadedLocalPolicyPackage,
    LocalPolicySimulationAdmissionReceipt,
    load_local_policy_package,
)
from dronedream_agent_core.realtime_feature_encoders import POLICY_REALTIME_FEATURE_COUNT

ROLES = (
    "local-navigation-policy",
    "precision-maneuver-policy",
    "recovery-policy",
    "risk-critic",
    "perception-encoder",
    "perception-health-critic",
    "settle-stability-critic",
    "payload-dynamics-adapter",
    "state-anomaly-detector",
    "cross-modal-consistency-critic",
)
VEHICLE_SHA256 = "a" * 64
SENSOR_SHA256 = "b" * 64
NAVIGATION_INPUTS = [
    "state_features",
    "candidate_features",
    "candidate_mask",
    "visual_features",
]
NAVIGATION_OUTPUTS = ["candidate_scores", "action_scores", "risk_score"]
ARTIFACT_CONTRACTS = {
    "local-navigation-policy": (NAVIGATION_INPUTS, NAVIGATION_OUTPUTS),
    "precision-maneuver-policy": (NAVIGATION_INPUTS, NAVIGATION_OUTPUTS),
    "recovery-policy": (NAVIGATION_INPUTS, NAVIGATION_OUTPUTS),
    "risk-critic": (NAVIGATION_INPUTS, ["risk_score"]),
    "perception-encoder": (
        ["forward_rgb"],
        [
            "quality_logits",
            "scene_logits",
            "semantic_logits",
            "traversability_logits",
            "visual_features",
        ],
    ),
    "perception-health-critic": (["state_features"], ["risk_score"]),
    "settle-stability-critic": (
        ["history_mask", "maneuver_features", "state_history"],
        ["risk_score"],
    ),
    "payload-dynamics-adapter": (
        ["history_mask", "payload_features", "payload_history"],
        ["controller_step_scale", "risk_score"],
    ),
    "state-anomaly-detector": (
        ["state_history", "history_mask"],
        ["anomaly_score"],
    ),
    "cross-modal-consistency-critic": (["sensor_features"], ["risk_score"]),
}


# 功能：
#   构造具有真实摘要及角色契约的文件测试包；权重为不可执行占位字节，不代表推理或飞行能力。
# 输入：
#   root：尚未创建的包目录。
#   roles：需要写入的专家角色。
#   continuous_control：是否声明连续控制输入输出，以检查旧回执不能跨契约复用。
# 输出：
#   package：通过包结构和内容身份校验的测试包。
def _package(
    root: Path, *, roles: tuple[str, ...] = ROLES, continuous_control: bool = False
) -> LoadedLocalPolicyPackage:
    root.mkdir()
    artifacts = []
    for role in roles:
        content = f"model:{role}".encode()
        relative_path = f"models/{role}.onnx"
        path = root / "models" / f"{role}.onnx"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        input_names, output_names = ARTIFACT_CONTRACTS[role]
        if continuous_control and role in {
            "local-navigation-policy",
            "precision-maneuver-policy",
            "recovery-policy",
            "risk-critic",
        }:
            input_names = [*input_names, "realtime_features", "realtime_valid_mask"]
        if continuous_control and role in {
            "local-navigation-policy",
            "precision-maneuver-policy",
            "recovery-policy",
        }:
            output_names = [*output_names, "pilot_control"]
        if continuous_control and role == "risk-critic":
            input_names = [
                name for name in input_names if name not in {"candidate_features", "candidate_mask"}
            ]
            input_names.append("proposed_control")
        artifacts.append(
            {
                "role": role,
                "relative_path": relative_path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "format": "onnx",
                "input_names": input_names,
                "output_names": output_names,
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "package_id": "dronedream.test.expert-ensemble",
                "display_name": "Test expert ensemble",
                "scope": "general",
                "vehicle_sha256": VEHICLE_SHA256,
                "sensor_contract_sha256": SENSOR_SHA256,
                "state_feature_count": 46,
                "candidate_feature_count": 15,
                "maximum_candidates": 8,
                "maximum_inference_latency_ms": 250,
                "visual_width": 224,
                "visual_height": 128,
                "visual_feature_count": 139,
                "realtime_feature_count": (
                    POLICY_REALTIME_FEATURE_COUNT if continuous_control else None
                ),
                "control_feature_contract_sha256": (
                    CURRENT_POLICY_FEATURE_CONTRACT_SHA256 if continuous_control else None
                ),
                "pilot_control_mode": ("normalized-body-velocity" if continuous_control else None),
                "visual_normalization": "imagenet",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    package = load_local_policy_package(root)
    return package


# 功能：
#   写入固定测试指标及来源绑定，只用于准入解析测试，不作为实际实验成果。
# 输入：
#   path：本测试回执文件。
#   package：回执绑定的模型包。
#   continuous_evidence：是否附加连续控制整体指标；不包含各导航专家的完整质量证据。
#   expert_evidence：是否附加三种导航专家的合成双向控制质量指标。
# 输出：
#   None：不返回业务数据。
def _admission(
    path: Path, package: LoadedLocalPolicyPackage, *, continuous_evidence: bool = False,
    expert_evidence: bool = False,
) -> None:
    receipt = LocalPolicySimulationAdmissionReceipt(
        control_feature_contract_sha256=package.manifest.control_feature_contract_sha256,
        receipt_id="policy-simulation-admission-" + "c" * 32,
        policy_package_sha256=package.package_sha256,
        dataset_receipt_sha256="d" * 64,
        evaluation_suite_sha256="e" * 64,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
        sample_count=100,
        motion_sample_count=50,
        non_motion_sample_count=50,
        motion_authorization_accuracy=1.0,
        candidate_selection_accuracy=1.0,
        risk_hold_recall=1.0,
        safe_motion_recall=1.0,
        risk_mean_absolute_error=0.0,
        realtime_feature_ready_sample_count=(100 if continuous_evidence else None),
        pilot_control_target_sample_count=(100 if continuous_evidence else None),
        pilot_control_mean_absolute_error=(0.05 if continuous_evidence else None),
        p50_inference_latency_ms=1.0,
        p95_inference_latency_ms=2.0,
        p99_inference_latency_ms=3.0,
        admitted_to_simulation=True,
        navigation_expert_metrics=(
            {role: _metrics() for role in ROLES[:3]} if expert_evidence else {}
        ),
    )
    path.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")


# 功能：
#   经真实命令行暂存完整专家包，再读回目录和模型摘要，确认只获得仿真范围。
# 输入：
#   tmp_path：测试源包、回执和暂存目录。
# 输出：
#   None：不返回业务数据。
def test_stages_only_the_complete_admitted_ensemble(tmp_path: Path) -> None:
    package_path = tmp_path / "package"
    package = _package(package_path, continuous_control=True)
    admission = tmp_path / "admission.json"
    _admission(admission, package, continuous_evidence=True, expert_evidence=True)
    output = tmp_path / "runtime-local-policy"
    repository = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "stage_local_policy_runtime.py"),
            "--package",
            str(package_path),
            "--simulation-admission",
            str(admission),
            "--output",
            str(output),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["schema_version"] == "dronedream.local-policy-runtime-catalog.v2"
    assert catalog["deployment_scope"] == "simulation-only"
    assert catalog["entries"][0]["package_sha256"] == package.package_sha256
    copied = load_local_policy_package(output / "packages" / "general-navigation")
    assert copied.package_sha256 == package.package_sha256
    assert set(copied.artifact_paths) == set(ROLES)


# 功能：
#   缺少一个专家时必须在输出创建前失败，不能靠其他角色或旧文件补齐缺失能力。
# 输入：
#   tmp_path：缺少跨模态专家的测试包及输出目录。
# 输出：
#   None：不返回业务数据。
def test_rejects_an_incomplete_ensemble_before_copy(tmp_path: Path) -> None:
    package_path = tmp_path / "package"
    package = _package(package_path, roles=ROLES[:-1])
    admission = tmp_path / "admission.json"
    _admission(admission, package)
    output = tmp_path / "runtime-local-policy"
    repository = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "stage_local_policy_runtime.py"),
            "--package",
            str(package_path),
            "--simulation-admission",
            str(admission),
            "--output",
            str(output),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "LOCAL_POLICY_RUNTIME_ENSEMBLE_INCOMPLETE" in result.stderr
    assert not output.exists()


# 功能：
#   连续控制包不能沿用只有候选选择指标的回执，原准入标记也不能绕过控制契约。
# 输入：
#   tmp_path：新控制接口包与不完整的旧类证据目录。
# 输出：
#   None：不返回业务数据。
def test_rejects_continuous_ensemble_with_legacy_admission_evidence(tmp_path: Path) -> None:
    package_path = tmp_path / "continuous-package"
    package = _package(package_path, continuous_control=True)
    admission = tmp_path / "legacy-admission.json"
    _admission(admission, package)
    output = tmp_path / "runtime-local-policy"
    repository = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "stage_local_policy_runtime.py"),
            "--package",
            str(package_path),
            "--simulation-admission",
            str(admission),
            "--output",
            str(output),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "LOCAL_POLICY_RUNTIME_SIMULATION_ADMISSION_INVALID" in result.stderr
    assert not output.exists()


# 功能：
#   用仓库合格载具执行真实重绑定，逐角色核对字节不变、新身份变化且不继承资格。
# 输入：
#   tmp_path：测试原包、回执及新候选目录；仓库载具仅只读使用。
# 输出：
#   None：不返回业务数据。
def test_rebinding_preserves_every_model_and_requires_fresh_admission(tmp_path: Path) -> None:
    source_path = tmp_path / "source-package"
    source = _package(source_path)
    source_admission = tmp_path / "source-admission.json"
    _admission(source_admission, source)
    output = tmp_path / "rebound-package"
    rebinding_receipt = tmp_path / "rebinding-receipt.json"
    repository = Path(__file__).resolve().parents[1]
    vehicle_package = (
        repository
        / "app"
        / "desktop"
        / "src-tauri"
        / "resources"
        / "default-assets"
        / "my-drone.ddpkg"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "rebind_local_policy_package.py"),
            "--source-package",
            str(source_path),
            "--source-simulation-admission",
            str(source_admission),
            "--qualified-vehicle-package",
            str(vehicle_package),
            "--output-package",
            str(output),
            "--rebinding-receipt",
            str(rebinding_receipt),
            "--package-id",
            "dronedream.test.current-expert-ensemble",
            "--display-name",
            "Current expert ensemble",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    rebound = load_local_policy_package(output)
    assert rebound.package_sha256 != source.package_sha256
    assert rebound.manifest.vehicle_sha256 != VEHICLE_SHA256
    assert {
        role: artifact.sha256
        for role, artifact in ((item.role, item) for item in rebound.manifest.artifacts)
    } == {
        role: artifact.sha256
        for role, artifact in ((item.role, item) for item in source.manifest.artifacts)
    }
    receipt = json.loads(rebinding_receipt.read_text(encoding="utf-8"))
    assert receipt["runtime_input_contract_unchanged"] is True
    assert receipt["fresh_simulation_admission_required"] is True
    assert receipt["current_asset_closed_loop_required"] is True
    assert receipt["qualification_granted"] is False
