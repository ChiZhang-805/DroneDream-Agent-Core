"""Numerical and ownership boundaries for the actual local control port."""

from dataclasses import replace

import numpy as np
import pytest
from test_local_policy_packages import _SelectingBackend, _snapshot, _write_package

from dronedream_agent_core import local_policy_port as policy
from dronedream_agent_core.contracts import TextNavigationDecision
from dronedream_agent_core.hashing import sha256_json


# 功能：
#   顾问输出不能通过强制类型转换或任意展平伪装成合法 float32 标量。
# 输入：
#   output：错误类型或错误秩的单值张量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "output",
    [
        np.array([True]),
        np.array(["0.2"]),
        np.array([0.2], dtype=np.float64),
        np.zeros((1, 1, 1), dtype=np.float32),
    ],
)
def test_scalar_output_preserves_runtime_tensor_contract(output):
    with pytest.raises(RuntimeError):
        policy._bounded_scalar(output, "TEST")


# 功能：
#   顾问分数逐项限制在概率范围，聚合为最大值也不能掩盖负数或超范围分量。
# 输入：
#   score：违反风险概率范围的顾问分数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_raw_advisor_scores_are_bounded(score):
    with pytest.raises(ValueError):
        policy.LocalPolicyRawInference(
            candidate_scores=[0.0] * 8,
            action_scores=[0.0] * 3,
            risk_score=1.0,
            invoked_advisory_roles=["risk-critic"],
            advisory_risk_scores={"risk-critic": score},
        )


# 功能：
#   声明 healthy 但从未收到样本的必需传感器不能提高健康比例。
# 输入：
#   无：一个健康标签但序号为零的测试传感器。
# 输出：
#   None：不返回业务数据。
def test_sensor_health_requires_actual_sample_presence():
    features = policy.compile_multimodal_sensor_features(
        {
            "multimodal_sensor_snapshot": {
                "statuses": [
                    {
                        "modality": "imu",
                        "required_for_motion": True,
                        "health": "healthy",
                        "latest_sequence": 0,
                    }
                ],
            }
        }
    )
    assert features[-1] == 0.0


# 功能：
#   描述字段中的字符串不能变成健康或就绪真值，防止错误遥测提高模型信心。
# 输入：
#   无：包含字符串 false 的测试快照。
# 输出：
#   None：不返回业务数据。
def test_sensor_health_does_not_coerce_string_flags():
    features = policy.compile_multimodal_sensor_features(
        {
            "multimodal_sensor_snapshot": {
                "ready_for_motion": "false",
                "statuses": [],
            }
        }
    )
    assert features[-4] == 0.0


# 功能：
#   控制速度必须是真实数值，不能把布尔开关解释为每秒一米。
# 输入：
#   无：使用正常快照和错误布尔速度。
# 输出：
#   None：不返回业务数据。
def test_control_scale_rejects_boolean_speed():
    with pytest.raises(ValueError):
        policy.compile_local_policy_features(_snapshot(), continuous_control_speed_mps=True)


# 功能：
#   开发采集和延迟宽限必须显式使用正确类型，不能因字符串或布尔转换误开权限。
# 输入：
#   tmp_path：测试包目录。
#   option：错误的端口设置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "option", [{"development_payload_collection": "false"}, {"scheduling_jitter_grace_ms": True}]
)
def test_port_options_are_explicitly_typed(tmp_path, option):
    package = _write_package(tmp_path / "package", package_id="unit.options")
    with pytest.raises(ValueError):
        policy.LocalPolicyPort(package, backend=_SelectingBackend(), **option)


# 功能：
#   推理期间调用方修改嵌套原快照后，本次输入记录仍绑定最初校验的完整内容。
# 输入：
#   tmp_path：合成模型包目录。
# 输出：
#   None：不返回业务数据。
def test_local_call_owns_input_snapshot_before_backend_runs(tmp_path):
    package = _write_package(tmp_path / "package", package_id="unit.owned-input")
    snapshot = _snapshot()
    original_hash = sha256_json({"text_navigation_snapshot": snapshot})

    class MutatingBackend(_SelectingBackend):
        # 功能：
        #   模拟调用方在推理期间改变自己持有的原始快照，不直接修改端口私有输入。
        # 输入：
        #   self：测试替身后端。
        #   batch：端口已构建的模型输入。
        #   multimodal：本测试没有图像。
        # 输出：
        #   result：父类提供的固定合法测试结果。
        def infer(self, batch, *, multimodal):
            snapshot["goal_position_m"]["x"] = 999.0
            result = super().infer(batch, multimodal=multimodal)
            return result

    port = policy.LocalPolicyPort(package, backend=MutatingBackend())
    result = port.call(
        role="local_navigation_advisor",
        output_type=TextNavigationDecision,
        instructions="",
        input_artifact={"text_navigation_snapshot": snapshot},
    )
    assert result.record.input_sha256 == original_hash
    assert result.record.input_sha256 != sha256_json({"text_navigation_snapshot": snapshot})


# 功能：
#   端口建立后原包对象被修改，也不能改变本次端口的风险阈值、角色目录和调用身份。
# 输入：
#   tmp_path：测试包目录。
# 输出：
#   None：不返回业务数据。
def test_port_owns_selected_manifest_and_artifact_mapping(tmp_path):
    package = _write_package(tmp_path / "package", package_id="unit.selected")
    port = policy.LocalPolicyPort(package, backend=_SelectingBackend())
    threshold = package.manifest.risk_hold_threshold
    package.manifest.risk_hold_threshold = 1.0
    package.artifact_paths.clear()
    assert port.package.manifest.risk_hold_threshold == threshold
    assert "risk-critic" in port.package.artifact_paths
    assert port.package.manifest is not package.manifest


# 功能：
#   稀疏候选槽位按实际掩码对应身份，屏蔽槽位的高分不能覆盖有效候选或导致错误选项。
# 输入：
#   tmp_path：仅提供仲裁阈值的合成包目录。
# 输出：
#   None：不返回业务数据。
def test_sparse_candidate_mask_matches_compact_identities(tmp_path):
    package = _write_package(tmp_path / "package", package_id="unit.sparse-mask")
    port = policy.LocalPolicyPort(package, backend=_SelectingBackend())
    snapshot = _snapshot()
    batch = replace(
        policy.compile_local_policy_features(snapshot),
        candidate_mask=(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        candidate_ids=("first-valid", "second-valid"),
    )
    raw = policy.LocalPolicyRawInference(
        candidate_scores=[100.0, -3.0, 100.0, 100.0, 100.0, 3.0, 100.0, 100.0],
        action_scores=[0.0, -1.0, -2.0],
        risk_score=0.1,
    )
    decision = port._decision(snapshot=snapshot, batch=batch, raw=raw)
    assert decision.action == "select-candidate"
    assert decision.selected_candidate_id == "second-valid"


# 功能：
#   非法重试配置不能被布尔与整数的相等关系接受，也不能实际调用后端。
# 输入：
#   tmp_path：测试包目录。
# 输出：
#   None：不返回业务数据。
def test_physical_attempt_budget_rejects_boolean(tmp_path):
    package = _write_package(tmp_path / "package", package_id="unit.attempts")
    backend = _SelectingBackend()
    port = policy.LocalPolicyPort(package, backend=backend)
    with pytest.raises(ValueError, match="one physical attempt"):
        port.call(
            role="local_navigation_advisor",
            output_type=TextNavigationDecision,
            instructions="",
            input_artifact={"text_navigation_snapshot": _snapshot()},
            maximum_physical_attempts=True,
        )
    assert backend.last_navigation_expert_role is None


# 功能：
#   错误离线开关或 CPU 配置必须在读取和加载模型之前拒绝，避免默默开启其他执行模式。
# 输入：
#   tmp_path：仅含占位模型字节的测试包目录。
#   option：非法后端选项。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "option",
    [
        {"allow_precomputed_visual_features": "false"},
        {"visual_worker_cpu_ids": [True]},
        {"visual_worker_cpu_ids": [0, 0]},
        {"visual_worker_cpu_ids": [-1]},
    ],
)
def test_backend_configuration_rejected_before_model_loading(tmp_path, option):
    package = _write_package(tmp_path / "package", package_id="unit.backend-options")
    with pytest.raises(ValueError):
        policy.OnnxLocalPolicyBackend(package, **option)
