from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from dronedream_agent_core.contracts import ModelCallRecord, Vector3
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_contracts import PluginHookReceipt
from dronedream_agent_core.runtime_plugins import (
    append_hook_receipts,
    require_plugin_acceptance,
    validate_runtime_model_output,
)
from dronedream_agent_plugins.native_runtime_plugins import _watchdog


# 功能：
#   验证桌面原生进程的默认存活检测参数与既定调度包络一致，不作为感知实时性验收。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_watchdog_default_matches_desktop_runtime_envelope() -> None:
    watchdog = _watchdog(configuration={})

    assert watchdog["deadline_ms"] == 1_000
    assert watchdog["startup_deadline_ms"] == 30_000
    assert watchdog["heartbeat_ms"] == 25
    assert watchdog["scheduling_jitter_grace_ms"] == 250
    assert watchdog["persistent_miss_ms"] == 250
    assert watchdog["on_miss"] == "safe-hold-then-land"
    assert watchdog["fail_closed"] is True


# 功能：
#   验证可配置更紧的时间包络且保留其值；配置被接受不等于真机已经通过资格测量。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_watchdog_accepts_tighter_hardware_qualification() -> None:
    watchdog = _watchdog(
        configuration={
            "deadline_ms": 250,
            "startup_deadline_ms": 5_000,
            "heartbeat_ms": 10,
            "scheduling_jitter_grace_ms": 25,
            "persistent_miss_ms": 100,
        }
    )

    assert watchdog["deadline_ms"] == 250
    assert watchdog["startup_deadline_ms"] == 5_000
    assert watchdog["heartbeat_ms"] == 10
    assert watchdog["scheduling_jitter_grace_ms"] == 25
    assert watchdog["persistent_miss_ms"] == 100


# 功能：
#   验证具名检测器和验证器生成稳定门控键，并保留原来的通过或否决结论。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_named_plugin_acceptance_builds_stable_gate_keys() -> None:
    gates, normalized = require_plugin_acceptance(
        [
            {"detector": "Runtime collision detector", "accepted": True},
            {"validator": "Flight/Envelope", "accepted": False},
        ],
        gate_prefix="runtime",
    )

    assert gates == {
        "runtime_runtime_collision_detector": True,
        "runtime_flight_envelope": False,
    }
    assert normalized[0]["accepted"] is True


# 功能：
#   验证未命名结果按序编号，非对象输出明确产生拒绝门控及可定位的问题码。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_unnamed_and_malformed_plugin_outputs_fail_closed() -> None:
    gates, normalized = require_plugin_acceptance(
        [{"accepted": True}, "not-an-object"],
        gate_prefix="checkpoint",
    )

    assert gates == {
        "checkpoint_01": True,
        "checkpoint_02": False,
    }
    assert normalized[1]["issue_codes"] == ["PLUGIN_VERDICT_NOT_OBJECT"]


# 功能：
#   验证重名插件结论获得确定的后缀，后来的结果不能覆盖先前门控。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_duplicate_plugin_identities_get_deterministic_suffixes() -> None:
    gates, _ = require_plugin_acceptance(
        [
            {"evaluation": "clearance", "accepted": True},
            {"evaluation": "clearance", "accepted": True},
        ],
        gate_prefix="route",
    )

    assert list(gates) == ["route_clearance", "route_clearance_2"]


# 功能：
#   验证归一化评估证据不共享插件原输出的深层对象，调用方修改不能污染原始结果。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_plugin_verdict_normalization_detaches_nested_evidence():
    original = {"accepted": True, "details": {"samples": [1]}}
    _, normalized = require_plugin_acceptance([original], gate_prefix="runtime")
    normalized[0]["details"]["samples"].append(2)
    assert original["details"]["samples"] == [1]


# 功能：
#   验证检查声称通过但分项失败／缺失，以及无法无损编码的证据均不能生成通过门控。
# 输入：
#   output：损坏或自相矛盾的插件输出。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "output",
    [
        {"accepted": True, "gates": {"collision_free": False}},
        {"accepted": True, "gates": {}},
        {"accepted": True, "gates": {"collision_free": "true"}},
        {"accepted": True, "details": {1: "first", "1": "second"}},
        {"accepted": True, "details": {"clearance": float("nan")}},
    ],
)
def test_contradictory_or_lossy_plugin_verdict_cannot_be_normalized_as_success(output):
    with pytest.raises(ValueError):
        require_plugin_acceptance([output], gate_prefix="runtime")


# 功能：
#   构造带真实调用记录契约的最小结构化输出，隔离运行期公共检查与模型供应商。
# 输入：
#   无。
# 输出：
#   output：向量制品与绑定该制品的调用记录组成的元组。
def _model_output():
    artifact = Vector3(x=1, y=0, z=0)
    record = ModelCallRecord(
        call_id="model-" + "a" * 24,
        role="execution_monitor",
        attempt=1,
        input_sha256="a" * 64,
        output_sha256=sha256_json(artifact),
        output_schema="Vector3",
        provider="fixture",
        model="fixture",
        latency_ms=1,
        created_at=datetime.now(UTC),
    )
    output = (artifact, record)
    return output


# 功能：
#   验证 Python 模型内部的 NaN 在 JSON 序列化变成 null 前被拒绝，不能交给输出钩子。
# 输入：
#   field：本例损坏的是制品还是调用记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["artifact", "record"])
def test_runtime_output_checks_original_finite_values_before_json_conversion(field):
    artifact, record = _model_output()
    if field == "artifact":
        artifact = artifact.model_copy(update={"x": float("nan")})
    else:
        record = record.model_copy(update={"latency_ms": float("nan")})
    calls = []

    # 功能：
    #   记录公共检查是否过早调用了插件，并原样返回信封作为通过对照。
    # 输入：
    #   slot：目标插件槽。
    #   hook：钩子名称。
    #   value：待校验信封。
    #   kwargs：调用角色与结构名称。
    # 输出：
    #   result：原信封及空测试回执列表。
    def passthrough(slot, hook, value, **kwargs):
        calls.append(slot)
        result = (value, [])
        return result

    with pytest.raises(ValueError, match="JSON_NUMBER_INVALID"):
        validate_runtime_model_output(
            SimpleNamespace(invoke_pipeline=passthrough),
            role="execution_monitor",
            expected_schema="Vector3",
            artifact=artifact,
            record=record,
        )
    assert not calls


# 功能：
#   验证输出检查不能利用 Python 的 True == 1 或原地修改来绕过不可改写约束。
# 输入：
#   mutation：等值类型替换或原地字段改写。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["equal_type", "in_place"])
def test_runtime_guard_cannot_change_envelope_without_detection(mutation):
    artifact, record = _model_output()

    # 功能：
    #   模拟错误守卫篡改调用次数或供应商字段，只测试公共输出不变式。
    # 输入：
    #   slot：插件槽名。
    #   hook：钩子名。
    #   value：实际传入的信封。
    #   kwargs：额外调用约束。
    # 输出：
    #   result：损坏的信封及空回执列表。
    def mutate(slot, hook, value, **kwargs):
        changed = deepcopy(value) if mutation == "equal_type" else value
        if mutation == "equal_type":
            changed["record"]["attempt"] = True
        else:
            changed["record"]["provider"] = "changed"
        result = (changed, [])
        return result

    with pytest.raises(RuntimeError, match="MUTATION_FORBIDDEN"):
        validate_runtime_model_output(
            SimpleNamespace(invoke_pipeline=mutate),
            role="execution_monitor",
            expected_schema="Vector3",
            artifact=artifact,
            record=record,
        )


# 功能：
#   验证已构造后被破坏的钩子回执不能直接序列化进持久记录，也不能留下空日志文件。
# 输入：
#   tmp_path：隔离的证据目录。
# 输出：
#   None：不返回业务数据。
def test_invalid_hook_receipt_is_rejected_before_creating_log(tmp_path):
    receipt = PluginHookReceipt(
        invocation_id="hook-" + "a" * 24,
        plugin_id="fixture.plugin",
        plugin_version="1.0.0",
        plugin_package_sha256="a" * 64,
        capability_id="fixture.check",
        slot_id="runtime.checks",
        hook="validate_output",
        outcome="accepted",
        input_sha256="a" * 64,
        output_sha256="b" * 64,
        created_at=datetime.now(UTC),
    ).model_copy(update={"input_sha256": "broken"})
    path = tmp_path / "new" / "hooks.jsonl"
    with pytest.raises(ValueError):
        append_hook_receipts(path, [receipt])
    assert not path.exists()


# 功能：
#   验证合法输出仍保留既有模型 JSON 表示（包括 UTC 时间），严格检查不制造摘要漂移。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_valid_output_keeps_legacy_json_representation():
    artifact, record = _model_output()
    expected = {
        "artifact": artifact.model_dump(mode="json"),
        "record": record.model_dump(mode="json"),
    }

    # 功能：
    #   核对运行输出信封未改变正常编码，并模拟合法守卫返回独立副本。
    # 输入：
    #   slot：守卫插槽。
    #   hook：守卫钩子名。
    #   value：传入信封。
    #   kwargs：角色与结构约束。
    # 输出：
    #   result：原样复制的信封及空测试回执列表。
    def guard(slot, hook, value, **kwargs):
        assert value == expected
        assert kwargs == {"role": "execution_monitor", "expected_schema": "Vector3"}
        result = (deepcopy(value), [])
        return result

    receipts = validate_runtime_model_output(
        SimpleNamespace(invoke_pipeline=guard),
        role="execution_monitor",
        expected_schema="Vector3",
        artifact=artifact,
        record=record,
    )
    assert receipts == []


# 功能：
#   验证合法回执按行追加且可恢复为相同契约，空批次既不清空日志也不创建新文件。
# 输入：
#   tmp_path：隔离的日志目录。
# 输出：
#   None：不返回业务数据。
def test_valid_receipts_append_without_replacing_existing_evidence(tmp_path):
    receipt = PluginHookReceipt(
        invocation_id="hook-" + "a" * 24,
        plugin_id="fixture.plugin",
        plugin_version="1.0.0",
        plugin_package_sha256="a" * 64,
        capability_id="fixture.check",
        slot_id="runtime.checks",
        hook="validate_output",
        outcome="accepted",
        input_sha256="a" * 64,
        output_sha256="b" * 64,
        created_at=datetime.now(UTC),
    )
    path = tmp_path / "new" / "hooks.jsonl"
    append_hook_receipts(path, [])
    assert not path.exists()
    append_hook_receipts(path, [receipt])
    append_hook_receipts(path, [receipt])
    append_hook_receipts(path, [])
    rows = path.read_text(encoding="utf-8").splitlines()
    assert [PluginHookReceipt.model_validate(json.loads(row)) for row in rows] == [receipt, receipt]
