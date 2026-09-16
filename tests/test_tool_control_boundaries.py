"""Regressions for guarded tool caching, retries and input ownership."""

import threading
import time

import pytest
from pydantic import BaseModel

from dronedream_agent_core.tools import ToolExecutionError, ToolPlugin, ToolRegistry


# 功能：
#   创建短时限测试注册表，明确打开缓存和重试，以检验宿主是否仍保留必要边界。
# 输入：
#   handler：本例的本地处理器。
#   authority：测试工具和注册表共同允许的权限。
#   input_type：可选输入模型类型。
#   output_type：可选输出模型类型。
# 输出：
#   registry：独立的工具注册表。
def registry_for(handler, *, authority="read", input_type=None, output_type=None):
    registry = ToolRegistry(allowed_authorities={authority})
    registry.register(
        ToolPlugin(
            tool_id="test.boundary",
            version="1.0.0",
            authority=authority,
            input_type=input_type,
            output_type=output_type,
            handler=handler,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
    )
    registry.configure_runtime_limits(maximum_calls=20, timeout_seconds=0.1)
    registry._execution_policy = lambda _: {"cache": True, "maximum_attempts": 3}
    return registry


# 功能：
#   验证输出被中间件拒绝后不能进入缓存，重复同一输入也不能取回被拒绝结果。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_rejected_output_is_never_cached():
    registry = registry_for(lambda _: {"_private": "must-not-escape"})

    # 功能：
    #   只拒绝输出阶段，允许输入进入处理器，以定位缓存插入是否早于输出校验。
    # 输入：
    #   hook：调用阶段。
    #   value：当前阶段的业务值。
    #   _：未使用的工具元数据。
    # 输出：
    #   result：输入原值和空失败回执组成的二元组。
    def guard(hook, value, **_):
        if hook == "after_tool_call":
            raise ValueError("rejected output")
        result = (value, None)
        return result

    registry._middleware = guard
    for _ in range(2):
        with pytest.raises(ToolExecutionError):
            registry.call("test.boundary", {})


# 功能：
#   验证调用方修改正常返回、回执或命中缓存的返回值，都不能改写内部缓存。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cache_owns_nested_output_independently_of_returned_results():
    registry = registry_for(lambda _: {"nested": {"value": 1}})
    first, receipt = registry.call("test.boundary", {})
    first["nested"]["value"] = 999
    receipt.output["nested"]["value"] = 888
    second, _ = registry.call("test.boundary", {})
    assert second == {"nested": {"value": 1}}
    second["nested"]["value"] = 777
    assert registry.call("test.boundary", {})[0] == {"nested": {"value": 1}}


# 功能：
#   验证类型化处理器使用中间件修改后的输入，不错误地继续传入原模型实例。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_typed_handler_receives_middleware_transformed_input():
    class Payload(BaseModel):
        value: int

    registry = registry_for(lambda value: value, input_type=Payload, output_type=Payload)
    registry._middleware = lambda hook, value, **_: (
        ({"value": 2} if hook == "before_tool_call" else value),
        None,
    )
    output, _ = registry.call("test.boundary", Payload(value=1))
    assert output.value == 2


# 功能：
#   验证执行器确认丢失时，即使策略要求三次尝试也只发一次动作，避免重复副作用。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_actuation_is_not_retried_by_permissive_policy():
    attempts = []

    # 功能：
    #   记录一次合成发送后报告确认丢失，不连接真实飞控或执行器。
    # 输入：
    #   _：未使用的测试命令内容。
    # 输出：
    #   None：不返回业务数据。
    def command(_):
        attempts.append("sent")
        raise RuntimeError("acknowledgement lost")

    registry = registry_for(command, authority="actuate")
    with pytest.raises(ToolExecutionError):
        registry.call("test.boundary", {})
    assert attempts == ["sent"]


# 功能：
#   验证处理器超时但仍在运行时不启动重叠重试，并在测试结束释放线程。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_timeout_does_not_start_overlapping_retry():
    release = threading.Event()
    finished = threading.Event()
    attempts = []

    # 功能：
    #   等待测试明确释放后返回，模拟不能被宿主强制取消的本地 Python 处理器。
    # 输入：
    #   value：准备原样返回的输入。
    # 输出：
    #   value：测试释放后的原输入。
    def slow(value):
        attempts.append("started")
        try:
            assert release.wait(2)
            return value
        finally:
            finished.set()

    registry = registry_for(slow)
    try:
        with pytest.raises(ToolExecutionError):
            registry.call("test.boundary", {})
        assert attempts == ["started"]
    finally:
        release.set()
        assert finished.wait(2)


# 功能：
#   验证处理器获得输入副本，修改嵌套数据不会污染调用方的原对象。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_handler_mutation_cannot_modify_callers_input():
    # 功能：
    #   故意改写收到的嵌套字段，用于区分深复制与仅复制外层字典。
    # 输入：
    #   value：宿主交付的输入副本。
    # 输出：
    #   value：修改后的副本。
    def mutate(value):
        value["nested"]["value"] = 999
        return value

    registry = registry_for(mutate)
    original = {"nested": {"value": 1}}
    registry.call("test.boundary", original)
    assert original == {"nested": {"value": 1}}


# 功能：
#   验证多个尝试共享一个等待截止时间，重试不能各自获得完整的新时限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_retries_share_one_deadline():
    attempts = []

    # 功能：
    #   消耗部分等待预算后失败，模拟可重试但不能无限延长的本地工具。
    # 输入：
    #   _：未使用的业务输入。
    # 输出：
    #   None：不返回业务数据。
    def slow_failure(_):
        attempts.append("started")
        time.sleep(0.07)
        raise RuntimeError("retryable offline failure")

    registry = registry_for(slow_failure)
    try:
        with pytest.raises(ToolExecutionError):
            registry.call("test.boundary", {})
        assert len(attempts) <= 2
    finally:
        # 等待可能仍在运行的离线夹具退出，不把本次测试当作真实飞行任务。
        time.sleep(0.08)


# 功能：
#   验证输出中间件修改结果后再次按类型规范化，返回模型和持久回执的值保持一致。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_guard_transformed_typed_output_matches_receipt_payload():
    class Payload(BaseModel):
        value: int

    registry = registry_for(lambda _: Payload(value=1), output_type=Payload)
    registry._middleware = lambda hook, value, **_: (
        ({"value": "2"} if hook == "after_tool_call" else value),
        None,
    )
    output, receipt = registry.call("test.boundary", {})
    assert output.value == 2
    assert receipt.output == {"value": 2}


# 功能：
#   验证先构造合法模型再修改字段，也不能规避输出的最终类型校验。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_mutated_pydantic_output_is_revalidated():
    class Payload(BaseModel):
        value: int

    # 功能：
    #   返回创建后被改坏字段类型的模型，检验宿主是否重新校验而非信任实例类型。
    # 输入：
    #   _：未使用的业务输入。
    # 输出：
    #   result：故意无效的模型实例。
    def invalid(_):
        result = Payload(value=1)
        result.value = "not-an-integer"
        return result

    registry = registry_for(invalid, output_type=Payload)
    with pytest.raises(ToolExecutionError):
        registry.call("test.boundary", {})


# 功能：
#   验证允许空值的类型也不能把 NaN 序列化为 null 后通过输入或输出校验。
# 输入：
#   boundary：需要测试的输入边界或输出边界。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("boundary", ["input", "output"])
def test_typed_nonfinite_value_cannot_disappear_as_null(boundary):
    class Payload(BaseModel):
        value: float | None

    returned = Payload(value=float("nan"))
    registry = registry_for(lambda _: returned, input_type=Payload, output_type=Payload)
    if boundary == "input":
        with pytest.raises((ValueError, ToolExecutionError)):
            registry.call("test.boundary", returned)
    else:
        with pytest.raises(ToolExecutionError):
            registry.call("test.boundary", Payload(value=1.0))


# 功能：
#   验证注册后的约束不受原插件对象或对外目录的嵌套字典修改影响。
# 输入：
#   source：尝试修改最初的插件定义或导出的公开目录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("source", ["definition", "catalog"])
def test_registered_tool_owns_schema_and_routing_metadata(source):
    registry = ToolRegistry(allowed_authorities={"read"})
    plugin = ToolPlugin(
        tool_id="test.owned",
        version="1.0.0",
        authority="read",
        input_type=None,
        output_type=None,
        input_schema={"type": "object", "properties": {"value": {"type": "integer"}}},
        output_schema={"type": "object", "required": ["value"]},
        routing_metadata={"hints": {"lane": "original"}},
        handler=lambda value: value,
    )
    registry.register(plugin)
    if source == "definition":
        plugin.input_schema["properties"]["value"]["type"] = "string"
        plugin.output_schema["required"].clear()
        plugin.routing_metadata["hints"]["lane"] = "changed"
    else:
        entry = registry.catalog()[0]
        entry["input_schema"]["properties"]["value"]["type"] = "string"
        entry["output_schema"]["required"].clear()
        entry["routing_metadata"]["hints"]["lane"] = "changed"
    entry = registry.catalog()[0]
    assert entry["input_schema"]["properties"]["value"]["type"] == "integer"
    assert entry["output_schema"]["required"] == ["value"]
    assert entry["routing_metadata"]["hints"]["lane"] == "original"
    with pytest.raises(ToolExecutionError):
        registry.call("test.owned", {"value": "not a number"})
