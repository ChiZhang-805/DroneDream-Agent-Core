"""Fault injection for real routing/port code; no flight qualification."""

from types import SimpleNamespace

import pytest
from test_local_policy_packages import (
    _PayloadSelectingBackend,
    _SelectingBackend,
    _snapshot,
    _write_package,
)

import dronedream_agent_core.local_policy_port as policy
from dronedream_agent_core.contracts import TextNavigationDecision
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_expert_harness import (
    requested_navigation_expert,
    route_local_experts,
)
from dronedream_agent_core.model_harness.model_port import ModelInvocationError


# 功能：
#   用真实端口调用协议提交测试快照，省略时使用合成基准快照。
# 输入：
#   port：待测本地模型端口。
#   snapshot：可选的本测试导航快照。
# 输出：
#   result：实际端口产生的结构化决策及调用记录。
def call(port, snapshot=None):
    result = port.call(
        role="local_navigation_advisor",
        output_type=TextNavigationDecision,
        instructions="",
        input_artifact={
            "text_navigation_snapshot": snapshot if snapshot is not None else _snapshot()
        },
    )
    return result


# 功能：
#   恢复专家接管时仍根据真实飞行阶段启用精细稳定顾问，不能仅由 profile 名称决定。
# 输入：
#   phase：需要精细操纵的阶段。
#   profile：可能缺失或不匹配的描述配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "phase",
    [
        "TAKEOFF",
        "CHECKPOINT",
        "ACTION",
        "PICKUP",
        "HOVER",
        "WAYPOINT_SETTLE",
        "LOCAL_SLOW",
        "LAND",
        "LANDING",
    ],
)
@pytest.mark.parametrize("profile", [None, "transit", "precision"])
def test_recovery_preserves_precision_regime_without_relying_on_profile(phase, profile):
    decision = route_local_experts(
        {
            "strategic_context": {
                "task": {
                    "phase": phase,
                    "control_profile": profile,
                    "decision_trigger": "progress-stalled",
                }
            }
        },
        available_roles={
            "local-navigation-policy",
            "recovery-policy",
            "risk-critic",
            "settle-stability-critic",
        },
    )
    assert decision.selected_navigation_role == "recovery-policy"
    assert "settle-stability-critic" in decision.advisory_roles
    assert decision.motion_permitted


# 功能：
#   动作检查点即使处于 TRACK 阶段且正在恢复，也必须保留稳定检查顾问。
# 输入：
#   无：使用固定测试阶段与障碍触发条件。
# 输出：
#   None：不返回业务数据。
def test_recovery_of_action_checkpoint_still_uses_settle_critic():
    decision = route_local_experts(
        {
            "strategic_context": {
                "task": {
                    "phase": "TRACK",
                    "action_checkpoint_goal": True,
                    "decision_trigger": "dynamic-obstacle",
                }
            }
        },
        available_roles={
            "local-navigation-policy",
            "recovery-policy",
            "risk-critic",
            "settle-stability-critic",
        },
    )
    assert decision.advisory_roles == ["risk-critic", "settle-stability-critic"]


# 功能：
#   检查点标记只接受明确真值，不能把字符串或整数转换成检查点权限。
# 输入：
#   value：不属于合法真值的测试标记。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", ["false", "true", 1, None, [], {}])
def test_checkpoint_flag_does_not_coerce_untyped_truthy_values(value):
    assert (
        requested_navigation_expert(
            {
                "strategic_context": {
                    "task": {
                        "phase": "TRACK",
                        "action_checkpoint_goal": value,
                    }
                }
            }
        )
        == "local-navigation-policy"
    )


# 功能：
#   无论是否允许兼容回退，缺少独立风险专家都不能获得操纵许可。
# 输入：
#   allow_fallback：是否允许普通导航角色回退。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("allow_fallback", [False, True])
def test_missing_risk_expert_cannot_authorize_motion_even_in_compatibility_mode(allow_fallback):
    decision = route_local_experts(
        {}, available_roles={"local-navigation-policy"}, allow_general_fallback=allow_fallback
    )
    assert not decision.motion_permitted
    assert "LOCAL_EXPERT_RISK_CRITIC_UNAVAILABLE" in decision.reason_codes


# 功能：
#   直接调用端口同样受必需风险专家约束，显式测试后端不具有绕过能力。
# 输入：
#   tmp_path：合成模型包目录。
# 输出：
#   None：不返回业务数据。
def test_direct_port_does_not_bypass_risk_expert_gate(tmp_path):
    package = _write_package(tmp_path / "package", package_id="test.direct")
    package.artifact_paths.pop("risk-critic")

    class UnreviewedBackend(_SelectingBackend):
        # 功能：
        #   模拟没有调用任何顾问的测试后端，保留其导航提案用于验证外层拒绝。
        # 输入：
        #   self：测试后端。
        #   args：原推理位置参数。
        #   kwargs：原推理关键字参数。
        # 输出：
        #   result：明确删去顾问执行记录的测试结果。
        def infer(self, *args, **kwargs):
            result = (
                super()
                .infer(*args, **kwargs)
                .model_copy(
                    update={
                        "invoked_advisory_roles": [],
                        "advisory_risk_scores": {},
                    }
                )
            )
            return result

    result = call(policy.LocalPolicyPort(package, backend=UnreviewedBackend()))
    assert result.artifact.action == "hold"
    assert "LOCAL_EXPERT_RISK_CRITIC_UNAVAILABLE" in result.artifact.risk_notes


# 功能：
#   即使有负载专家且显式处于开发采集，也不能覆盖缺失或非法的载荷上限确认。
# 输入：
#   tmp_path：合成模型包目录。
#   verification：未确认为真的载荷限制标记。
#   development：开发采集开关。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("verification", [None, False, "true", "false", 1, 0])
@pytest.mark.parametrize("development", [False, True])
def test_missing_or_malformed_payload_limit_cannot_be_overridden_by_adapter(
    tmp_path,
    verification,
    development,
):
    package = _write_package(tmp_path / "package", package_id="test.payload")
    package.artifact_paths["payload-dynamics-adapter"] = tmp_path / "unused.onnx"
    snapshot = _snapshot()
    snapshot["strategic_context"] = {
        "payload": {
            "state": "loaded-stable",
            "within_declared_payload_limit": verification,
            "dynamics": {"available": True, "ready": True},
        }
    }
    # Keep the envelope valid so this test reaches the payload gate it targets.
    snapshot["snapshot_sha256"] = sha256_json(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )
    result = call(
        policy.LocalPolicyPort(
            package, backend=_PayloadSelectingBackend(), development_payload_collection=development
        ),
        snapshot,
    )
    assert result.artifact.action == "hold"
    assert "LOCAL_EXPERT_PAYLOAD_LIMIT_NOT_VERIFIED" in result.artifact.risk_notes


# 功能：
#   对输入、历史、推理、决策和记账分别注入延迟，所有步骤都必须计入调用预算。
# 输入：
#   tmp_path：合成包目录。
#   monkeypatch：替换单调测试时钟及阶段函数的工具。
#   stage：当前注入延迟的阶段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("stage", ["input", "history", "inference", "decision", "record"])
def test_complete_call_budget_includes_all_work(tmp_path, monkeypatch, stage):
    package = _write_package(tmp_path / "package", package_id="test.budget")
    clock = [100.0]
    monkeypatch.setattr(policy, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    invoked = []

    class Backend(_SelectingBackend):
        # 功能：
        #   只在历史场景推进测试时钟，不制造或执行实际传感器历史。
        # 输入：
        #   self：测试后端。
        #   batch：待准备的特征批次。
        # 输出：
        #   None：不返回业务数据。
        def prepare_temporal_context(self, batch):
            if stage == "history":
                clock[0] += 0.5004

        # 功能：
        #   记录实际到达替身推理边界的次数，在推理场景注入超预算时长。
        # 输入：
        #   self：测试后端。
        #   args：透传推理位置参数。
        #   kwargs：透传关键字参数。
        # 输出：
        #   result：父类的固定合法测试输出。
        def infer(self, *args, **kwargs):
            invoked.append(True)
            if stage == "inference":
                clock[0] += 0.5004
            result = super().infer(*args, **kwargs)
            return result

    target = {"input": "compile_local_policy_features", "record": "ModelCallRecord"}.get(stage)
    if target:
        original = getattr(policy, target)

        # 功能：
        #   执行真实阶段后模拟其消耗时间，以验证端口不会漏计输入或记账开销。
        # 输入：
        #   args：真实函数位置参数。
        #   kwargs：真实函数关键字参数。
        # 输出：
        #   value：真实阶段的原始结果。
        def delayed(*args, **kwargs):
            value = original(*args, **kwargs)
            clock[0] += 0.5004
            return value

        monkeypatch.setattr(policy, target, delayed)
    port = policy.LocalPolicyPort(package, backend=Backend())
    if stage == "decision":
        original_decision = port._decision

        # 功能：
        #   对每次真实仲裁增加半段延迟，两次归因与最终仲裁合计超过预算。
        # 输入：
        #   kwargs：仲裁函数的真实关键字参数。
        # 输出：
        #   value：真实仲裁产生的决策。
        def delayed_decision(**kwargs):
            value = original_decision(**kwargs)
            clock[0] += 0.2502  # Two decisions: proposal attribution and final arbitration.
            return value

        monkeypatch.setattr(port, "_decision", delayed_decision)
    if stage in {"input", "history"}:
        with pytest.raises(TimeoutError, match="INPUT_PREPARATION_EXCEEDED"):
            call(port)
        assert not invoked
    else:
        with pytest.raises(ModelInvocationError) as error:
            call(port)
        assert invoked == [True]
        assert error.value.attempts_used == 1
        assert error.value.reason_code == "LOCAL_POLICY_QUALIFIED_LATENCY_EXCEEDED"
        assert error.value.diagnostic_metrics["total-latency-ms"] == pytest.approx(500.4)


# 功能：
#   验证记录保留准备、推理和输出构造的嵌套耗时，整个调用总计四十二毫秒。
# 输入：
#   tmp_path：合成包目录。
#   monkeypatch：提供确定性时钟的工具，不声称实际机器具有该性能。
# 输出：
#   None：不返回业务数据。
def test_record_contains_nested_preparation_and_complete_call_intervals(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(policy, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    package = _write_package(tmp_path / "package", package_id="test.timing")

    class Backend(_SelectingBackend):
        # 功能：
        #   模拟准备历史耗时十二毫秒，不触发真实传感器。
        # 输入：
        #   self：测试后端。
        #   batch：当前特征批次。
        # 输出：
        #   None：不返回业务数据。
        def prepare_temporal_context(self, batch):
            clock[0] += 0.012

        # 功能：
        #   模拟推理耗时二十三毫秒，返回既有合法测试结果。
        # 输入：
        #   self：测试后端。
        #   args：透传位置参数。
        #   kwargs：透传关键字参数。
        # 输出：
        #   result：父类的固定合法输出。
        def infer(self, *args, **kwargs):
            clock[0] += 0.023
            result = super().infer(*args, **kwargs)
            return result

    original_record = policy.ModelCallRecord

    # 功能：
    #   模拟构造调用记录耗时七毫秒，仍执行真实记录模型校验。
    # 输入：
    #   kwargs：真实调用记录字段。
    # 输出：
    #   record：完成真实校验的调用记录。
    def delayed_record(**kwargs):
        clock[0] += 0.007
        record = original_record(**kwargs)
        return record

    monkeypatch.setattr(policy, "ModelCallRecord", delayed_record)
    result = call(policy.LocalPolicyPort(package, backend=Backend()))
    assert result.record.latency_ms == 42
    timings = result.record.local_expert_trace.pipeline_latency_ms
    assert timings["port-input-preparation"] == pytest.approx(12.0)
    assert timings["port-decision-record"] == pytest.approx(7.0)
    assert timings["port-wall"] == pytest.approx(42.0)


# 功能：
#   后端重复报告同一顾问调用时拒绝，不能把重复记录当作多位独立专家证据。
# 输入：
#   tmp_path：合成包目录。
# 输出：
#   None：不返回业务数据。
def test_duplicate_advisor_receipt_is_rejected(tmp_path):
    package = _write_package(tmp_path / "package", package_id="test.duplicate")

    class Backend(_SelectingBackend):
        # 功能：
        #   绕过赋值校验注入重复顾问列表，验证真实端口会重新检查。
        # 输入：
        #   self：测试后端。
        #   args：透传推理位置参数。
        #   kwargs：透传推理关键字参数。
        # 输出：
        #   result：含重复顾问记录的非法测试结果。
        def infer(self, *args, **kwargs):
            result = (
                super()
                .infer(*args, **kwargs)
                .model_copy(update={"invoked_advisory_roles": ["risk-critic", "risk-critic"]})
            )
            return result

    with pytest.raises(ModelInvocationError, match="ADVISORY_EXECUTION_MISMATCH"):
        call(policy.LocalPolicyPort(package, backend=Backend()))
