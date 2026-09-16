from __future__ import annotations

import pytest

from dronedream_agent_core.extensions import (
    ExtensionExecutionError,
    ExtensionPlugin,
    ExtensionRegistry,
)
from dronedream_agent_core.hashing import sha256_json


# 功能：
#   构造独立扩展能力夹具，允许精确设置激活方式、失败策略和顺序约束。
# 输入：
#   plugin_id：测试插件标识。
#   mode：激活方式。
#   failure：失败隔离或失败关闭策略。
#   order：没有依赖关系时使用的排序权重。
#   runs_after：必须先执行的插件或能力标识。
#   runs_before：必须后执行的插件或能力标识。
#   handler：本例钩子处理器，默认原样返回 value。
# 输出：
#   plugin：供测试注册的扩展能力定义。
def _plugin(
    plugin_id: str,
    *,
    mode: str = "pipeline",
    failure: str = "isolate",
    order: int = 500,
    runs_after: tuple[str, ...] = (),
    runs_before: tuple[str, ...] = (),
    handler=lambda **kwargs: kwargs.get("value"),
) -> ExtensionPlugin:
    plugin = ExtensionPlugin(
        plugin_id=plugin_id,
        version="1.0.0",
        package_sha256="a" * 64,
        capability_id=f"{plugin_id}.capability",
        slot_id="harness.test-pipeline",
        activation_mode=mode,  # type: ignore[arg-type]
        failure_mode=failure,  # type: ignore[arg-type]
        swap_policy="next-mission",
        pipeline_order=order,
        runs_after=runs_after,
        runs_before=runs_before,
        hooks={"transform": handler},
    )
    return plugin


# 功能：
#   验证显式依赖关系优先于数字排序，并按真实执行顺序产生含输入摘要的回执。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_pipeline_is_ordered_and_hash_bound() -> None:
    registry = ExtensionRegistry()
    registry.register(
        _plugin(
            "pipeline.second",
            order=10,
            runs_after=("pipeline.first",),
            handler=lambda *, value: value + ["second"],
        )
    )
    registry.register(
        _plugin(
            "pipeline.first",
            order=900,
            handler=lambda *, value: value + ["first"],
        )
    )

    output, receipts = registry.invoke_pipeline("harness.test-pipeline", "transform", [])

    assert output == ["first", "second"]
    assert [receipt.plugin_id for receipt in receipts] == [
        "pipeline.first",
        "pipeline.second",
    ]
    assert all(receipt.outcome == "accepted" for receipt in receipts)
    assert all(len(receipt.input_sha256) == 64 for receipt in receipts)


# 功能：
#   验证隔离失败保留前一步值，后续正常钩子继续处理，并分别记录失败与成功。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_isolated_pipeline_failure_does_not_corrupt_value() -> None:
    registry = ExtensionRegistry()

    # 功能：
    #   用本地异常模拟一个允许隔离的钩子失败，不进行任何外部副作用。
    # 输入：
    #   value：到达本阶段的业务列表。
    # 输出：
    #   None：不返回业务数据。
    def broken(*, value: list[str]) -> list[str]:
        raise RuntimeError(value)

    registry.register(_plugin("pipeline.broken", order=10, handler=broken))
    registry.register(
        _plugin(
            "pipeline.healthy",
            order=20,
            handler=lambda *, value: value + ["healthy"],
        )
    )

    output, receipts = registry.invoke_pipeline("harness.test-pipeline", "transform", ["original"])

    assert output == ["original", "healthy"]
    assert [receipt.outcome for receipt in receipts] == ["failed", "accepted"]


# 功能：
#   验证失败关闭的钩子中止流水线，异常携带发生失败的确切插件回执。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_fail_closed_extension_stops_dispatch() -> None:
    registry = ExtensionRegistry()

    # 功能：
    #   模拟必须阻断后续执行的检查器失败。
    # 输入：
    #   value：到达检查器的输入。
    # 输出：
    #   None：不返回业务数据。
    def broken(*, value: object) -> object:
        raise RuntimeError(value)

    registry.register(_plugin("pipeline.guard", failure="fail-closed", handler=broken))

    try:
        registry.invoke_pipeline("harness.test-pipeline", "transform", {})
    except ExtensionExecutionError as error:
        assert error.receipt.plugin_id == "pipeline.guard"
        assert error.receipt.outcome == "failed"
    else:
        raise AssertionError("fail-closed extension did not stop the pipeline")


# 功能：
#   验证互相依赖的两个能力在注册阶段即被拒绝，不留到执行时任意排序。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_pipeline_cycle_is_rejected() -> None:
    registry = ExtensionRegistry()
    registry.register(_plugin("pipeline.a", runs_after=("pipeline.b",)))
    try:
        registry.register(_plugin("pipeline.b", runs_after=("pipeline.a",)))
    except ValueError as error:
        assert "cycle" in str(error)
    else:
        raise AssertionError("pipeline cycle was accepted")


# 功能：
#   验证同一插件包内多个能力可以独立注册、按钩子筛选并产生各自能力身份的回执。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_plugin_suite_can_register_multiple_hook_capabilities() -> None:
    registry = ExtensionRegistry()
    for capability_id, hook_name in (
        ("suite.risk", "risk"),
        ("suite.energy", "energy"),
    ):
        registry.register(
            ExtensionPlugin(
                plugin_id="suite.inspection",
                version="1.0.0",
                package_sha256="b" * 64,
                capability_id=capability_id,
                slot_id="harness.suite-advisors",
                activation_mode="multiple",
                failure_mode="isolate",
                swap_policy="next-mission",
                pipeline_order=100,
                runs_after=(),
                runs_before=(),
                hooks={hook_name: lambda **kwargs: kwargs["mission"]},
            )
        )

    risk, risk_receipts = registry.invoke_multiple(
        "harness.suite-advisors", "risk", mission="inspect"
    )
    energy, energy_receipts = registry.invoke_multiple(
        "harness.suite-advisors", "energy", mission="inspect"
    )

    assert risk == ["inspect"]
    assert energy == ["inspect"]
    assert risk_receipts[0].capability_id == "suite.risk"
    assert energy_receipts[0].capability_id == "suite.energy"


# 功能：
#   验证钩子修改输入副本后失败，不污染后续阶段或调用方原值，失败回执仍绑定执行前输入。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_pipeline_mutation_does_not_leak_into_next_hook():
    registry = ExtensionRegistry()

    # 功能：
    #   先改写嵌套字段再故意失败，区分数据隔离与仅在成功后决定是否接收返回值。
    # 输入：
    #   value：宿主交付的业务数据副本。
    # 输出：
    #   None：不返回业务数据。
    def broken(*, value):
        value["nested"]["value"] = 999
        raise RuntimeError("failed after mutation")

    registry.register(_plugin("pipeline.broken", handler=broken))
    original = {"nested": {"value": 1}}
    output, receipts = registry.invoke_pipeline("harness.test-pipeline", "transform", original)
    assert output == original == {"nested": {"value": 1}}
    assert receipts[0].input_sha256 == sha256_json({"value": original})


# 功能：
#   验证成功钩子的输入回执仍描述执行前数据，输出修改不回写调用方对象。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_successful_mutation_receipt_binds_original_input():
    registry = ExtensionRegistry()

    # 功能：
    #   修改并返回本次业务副本，用于校验输入和输出分别绑定正确阶段。
    # 输入：
    #   value：独立业务输入副本。
    # 输出：
    #   value：修改后的副本。
    def mutate(*, value):
        value["nested"]["value"] = 2
        return value

    registry.register(_plugin("pipeline.mutate", handler=mutate))
    original = {"nested": {"value": 1}}
    output, receipts = registry.invoke_pipeline("harness.test-pipeline", "transform", original)
    assert output == {"nested": {"value": 2}}
    assert original == {"nested": {"value": 1}}
    assert receipts[0].input_sha256 == sha256_json({"value": original})


# 功能：
#   验证新能力形成依赖环而被拒绝后，原目录和原流水线仍可正常使用。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_rejected_registration_does_not_poison_existing_pipeline():
    registry = ExtensionRegistry()
    registry.register(_plugin("pipeline.a", runs_after=("pipeline.b",)))
    with pytest.raises(ValueError, match="cycle"):
        registry.register(_plugin("pipeline.b", runs_after=("pipeline.a",)))
    assert [item["plugin_id"] for item in registry.catalog()] == ["pipeline.a"]
    assert registry.invoke_pipeline("harness.test-pipeline", "transform", ["original"])[0] == [
        "original",
    ]


# 功能：
#   验证处理器返回无法生成有限 JSON 回执的值时，也遵守隔离或失败关闭策略。
# 输入：
#   failure：被测插件的失败策略。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failure", ["isolate", "fail-closed"])
def test_invalid_output_evidence_uses_declared_failure_policy(failure):
    registry = ExtensionRegistry()
    registry.register(
        _plugin(
            "pipeline.invalid",
            failure=failure,
            handler=lambda **_: {"value": float("nan")},
        )
    )
    if failure == "fail-closed":
        with pytest.raises(ExtensionExecutionError) as raised:
            registry.invoke_pipeline("harness.test-pipeline", "transform", {"value": 1})
        assert raised.value.receipt.outcome == "failed"
    else:
        output, receipts = registry.invoke_pipeline(
            "harness.test-pipeline",
            "transform",
            {"value": 1},
        )
        assert output == {"value": 1}
        assert receipts[0].outcome == "failed"


# 功能：
#   验证注册后改动原定义或查询返回值中的钩子映射，不会替换内部实际处理器。
# 输入：
#   source：原定义或插槽查询返回值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("source", ["definition", "query"])
def test_extension_registry_owns_hook_selection(source):
    registry = ExtensionRegistry()
    plugin = _plugin("pipeline.owned")
    registry.register(plugin)
    selected = (
        plugin if source == "definition" else registry.plugins_for_slot("harness.test-pipeline")[0]
    )
    selected.hooks["transform"] = lambda **_: ["unexpected replacement"]
    output, _ = registry.invoke_pipeline("harness.test-pipeline", "transform", ["original"])
    assert output == ["original"]
