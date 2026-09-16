"""结构化控制边界的离线反例，不签发真实用户任务或调用外部模型。"""

import math
from datetime import UTC, datetime

import pytest

from dronedream_agent_core.model_harness.boundary import (
    HarnessControlPlaneReceipt,
    HarnessInputEnvelope,
    HarnessOutputEnvelope,
    PluginSelection,
    _sha256,
    autonomy_runtime_envelope,
    compile_autonomy_control_plane_receipt,
    compile_execution_authority,
    harness_input_sha256,
    selections_from_plugin_snapshot,
    validate_output_against_boundaries,
)
from dronedream_agent_core.model_harness.memory import MemoryOwnerScope
from dronedream_agent_core.plugin_contracts import PluginSnapshot


# 功能：
#   创建不含真实账户和插件的匹配控制回执、运行边界及结构化输入。
# 输入：
#   无。
# 输出：
#   boundaries：测试回执、运行边界、输入和冻结插件快照组成的元组。
def _boundaries():
    receipt = compile_autonomy_control_plane_receipt((), effective_maximum_model_calls=16)
    snapshot = PluginSnapshot(
        snapshot_id="plugin-snapshot-" + "a" * 24,
        catalog_sha256=_sha256([]),
        plugins=[],
        created_at=datetime.now(UTC),
    )
    scope = MemoryOwnerScope(owner_account_id="fixture-owner", source_edition="autonomy")
    runtime = autonomy_runtime_envelope(
        scope,
        receipt,
        snapshot,
        selected_maximum_model_calls=16,
        maximum_intent_rounds=2,
        maximum_planning_rounds=4,
    )
    envelope = HarnessInputEnvelope(
        request_id="request-fixture",
        task_id="task-fixture",
        thread_id="thread-fixture",
        owner_binding_sha256=runtime.owner_binding_sha256,
        tenant_binding_sha256=runtime.tenant_binding_sha256,
        source_edition="autonomy",
        control_plane_selection_sha256=receipt.selection_sha256,
        current_request={"message": "inspect the map"},
    )
    boundaries = receipt, runtime, envelope, snapshot
    return boundaries


# 功能：
#   为匹配的结构化输入构造仅用于验证的提案，不宣称已执行动作。
# 输入：
#   envelope：测试输入封装。
# 输出：
#   output：包含验证回执标识的测试提案。
def _output(envelope):
    output = HarnessOutputEnvelope(
        request_id=envelope.request_id,
        task_id=envelope.task_id,
        control_plane_selection_sha256=envelope.control_plane_selection_sha256,
        input_envelope_sha256=harness_input_sha256(envelope),
        status="validated_proposal",
        structured_result={"plan": ["inspect"]},
        model_call_count=2,
        repair_cycle_count=1,
        validation_receipt_ids=("validation-fixture",),
    )
    return output


# 功能：
#   拒绝输入中的非有限数、非字符串键和非 JSON 容器，不让编码时的转换改变数据语义。
# 输入：
#   value：不允许进入模型输入的嵌套值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, {1: "value"}, (1, 2)],
    ids=["nan", "infinity", "numeric-key", "tuple"],
)
def test_input_requires_bounded_standard_json(value):
    _, _, envelope, _ = _boundaries()
    raw = envelope.model_dump(mode="python")
    raw["current_request"] = {"nested": value}
    with pytest.raises(ValueError):
        HarnessInputEnvelope.model_validate(raw)


# 功能：
#   结构化输入复制调用方提供的嵌套数据，之后的界面修改不应改变已构建输入。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_input_detaches_nested_caller_data():
    _, _, envelope, _ = _boundaries()
    raw = envelope.model_dump(mode="python")
    raw["current_request"] = {"items": ["original"]}
    accepted = HarnessInputEnvelope.model_validate(raw)
    raw["current_request"]["items"].clear()
    assert accepted.current_request == {"items": ["original"]}


# 功能：
#   结构化输出同样拒绝非有限数、非 JSON 结构和超出 2 MiB 的结果。
# 输入：
#   value：非法结果中的嵌套内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "value",
    [math.nan, {1: "value"}, "x" * (2 * 1024 * 1024)],
    ids=["nan", "numeric-key", "oversized"],
)
def test_output_requires_bounded_standard_json(value):
    _, _, envelope, _ = _boundaries()
    raw = _output(envelope).model_dump(mode="python")
    raw["structured_result"] = {"value": value}
    with pytest.raises(ValueError):
        HarnessOutputEnvelope.model_validate(raw)


# 功能：
#   验收入口必须重新验证被原地改写的输出，不因对象曾构造成功而跳过计数和权限检查。
# 输入：
#   field：被改写的输出字段。
#   value：违反契约的改写值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_call_count", -1),
        ("repair_cycle_count", -1),
        ("grants_execution_authority", True),
        ("physical_action_performed", True),
    ],
)
def test_acceptance_revalidates_modified_output(field, value):
    receipt, runtime, envelope, _ = _boundaries()
    output = _output(envelope)
    setattr(output, field, value)
    with pytest.raises(ValueError):
        validate_output_against_boundaries(receipt, runtime, output, input_envelope=envelope)


# 功能：
#   文本、布尔和浮点不能被静默转成模型调用计数。
# 输入：
#   value：伪装为调用次数的错误类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, "2", 2.0])
def test_output_count_is_a_strict_integer(value):
    _, _, envelope, _ = _boundaries()
    raw = _output(envelope).model_dump(mode="python")
    raw["model_call_count"] = value
    with pytest.raises(ValueError):
        HarnessOutputEnvelope.model_validate(raw)


# 功能：
#   数字零不是显式的无执行权限声明，权限字段只能接受布尔值。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_execution_flag_cannot_be_an_integer():
    _, _, envelope, _ = _boundaries()
    raw = _output(envelope).model_dump(mode="python")
    raw["grants_execution_authority"] = 0
    with pytest.raises(ValueError):
        HarnessOutputEnvelope.model_validate(raw)


# 功能：
#   即使重算摘要，也不能移除固定内核职责；摘要不能替代语义约束。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_rehash_cannot_remove_fixed_kernel_responsibilities():
    receipt, _, _, _ = _boundaries()
    receipt.fixed_kernel_responsibilities = ()
    raw = receipt.model_dump(mode="python")
    raw["selection_sha256"] = _sha256(receipt.selection_payload())
    with pytest.raises(ValueError):
        HarnessControlPlaneReceipt.model_validate(raw)


# 功能：
#   签发运行边界时要求选中预算与公共回执一致，不能拼接较宽松的运行预算。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_runtime_cannot_select_budget_different_from_receipt():
    receipt, _, _, snapshot = _boundaries()
    with pytest.raises(ValueError):
        autonomy_runtime_envelope(
            MemoryOwnerScope(owner_account_id="fixture-owner", source_edition="autonomy"),
            receipt,
            snapshot,
            selected_maximum_model_calls=32,
            maximum_intent_rounds=2,
            maximum_planning_rounds=4,
        )


# 功能：
#   无插件快照不能与声称选择了插件的回执拼接成一次任务边界。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_runtime_requires_receipt_selections_from_same_snapshot():
    _, _, _, snapshot = _boundaries()
    selection = PluginSelection(
        slot="planner",
        plugin_id="fixture.planner",
        version="1.0.0",
        content_sha256="c" * 64,
        trust="managed",
    )
    receipt = compile_autonomy_control_plane_receipt((selection,), effective_maximum_model_calls=16)
    with pytest.raises(ValueError):
        autonomy_runtime_envelope(
            MemoryOwnerScope(owner_account_id="fixture-owner", source_edition="autonomy"),
            receipt,
            snapshot,
            selected_maximum_model_calls=16,
            maximum_intent_rounds=2,
            maximum_planning_rounds=4,
        )


# 功能：
#   拒绝重复插件身份，避免同一回执把同一插件重复算作不同选择。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_duplicate_plugin_selections_are_rejected():
    selection = PluginSelection(
        slot="planner",
        plugin_id="fixture.planner",
        version="1.0.0",
        content_sha256="c" * 64,
        trust="managed",
    )
    with pytest.raises(ValueError):
        compile_autonomy_control_plane_receipt(
            (selection, selection), effective_maximum_model_calls=16
        )


# 功能：
#   签发执行绑定前重新验证运行预算和身份，不接受被绕过构造校验的模型副本。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_authority_revalidates_runtime_before_issuing():
    _, runtime, _, _ = _boundaries()
    runtime.effective_maximum_model_calls = 48
    with pytest.raises(ValueError):
        compile_execution_authority(
            thread_id="thread-fixture",
            plan_revision_id="plan-" + "a" * 32,
            contract_id="contract-fixture",
            prepared_mission_sha256="d" * 64,
            runtime=runtime,
        )


# 功能：
#   执行绑定签发时间必须包含时区，避免同一时刻被不同机器解释成不同日期。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_authority_rejects_naive_issue_time():
    _, runtime, _, _ = _boundaries()
    with pytest.raises(ValueError):
        compile_execution_authority(
            thread_id="thread-fixture",
            plan_revision_id="plan-" + "a" * 32,
            contract_id="contract-fixture",
            prepared_mission_sha256="d" * 64,
            runtime=runtime,
            issued_at=datetime(2026, 9, 15),
        )


# 功能：
#   公共 JSON 字段和摘要算法保持兼容，经过 JSON 序列化回读后仍可验收匹配提案。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_valid_public_json_round_trip_preserves_binding():
    receipt, runtime, envelope, _ = _boundaries()
    copied_receipt = HarnessControlPlaneReceipt.model_validate_json(receipt.model_dump_json())
    copied_input = HarnessInputEnvelope.model_validate_json(envelope.model_dump_json())
    output = HarnessOutputEnvelope.model_validate_json(_output(copied_input).model_dump_json())
    assert copied_receipt.selection_sha256 == receipt.selection_sha256
    assert harness_input_sha256(copied_input) == harness_input_sha256(envelope)
    validate_output_against_boundaries(copied_receipt, runtime, output, input_envelope=copied_input)


# 功能：
#   公共投影入口必须调用实际快照校验，不能把伪造目录摘要继续带入运行资料。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_selection_projection_rejects_catalog_drift():
    _, _, _, snapshot = _boundaries()
    snapshot.catalog_sha256 = "f" * 64
    with pytest.raises(ValueError):
        selections_from_plugin_snapshot(snapshot)


# 功能：
#   单独自洽但比公共回执更宽的运行预算，也必须在最终输出验收时被拒绝。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_acceptance_rejects_independently_rewritten_runtime_budget():
    receipt, runtime, envelope, _ = _boundaries()
    runtime.selected_maximum_model_calls = 32
    runtime.effective_maximum_model_calls = 32
    with pytest.raises(ValueError):
        validate_output_against_boundaries(
            receipt, runtime, _output(envelope), input_envelope=envelope
        )
