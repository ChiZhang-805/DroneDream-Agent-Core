"""Offline receipt normalization checks; handlers never contact a model or a flight controller."""

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel
from test_extensions import _plugin

from dronedream_agent_core.extensions import ExtensionRegistry, _jsonable
from dronedream_agent_core.hashing import sha256_json
from dronedream_plugin_sdk.protocol import MAX_JSON_DEPTH, MAX_JSON_NODES


class _Reading(BaseModel):
    value: float
    captured_at: datetime = datetime(2026, 9, 14, tzinfo=UTC)


class _KeyedReading(BaseModel):
    values: dict[object, str]


# 功能：
#   验证模型序列化不能把非有限数变成 null，再为实际非法输出生成接受回执。
# 输入：
#   number：待测试的非有限浮点值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_model_output_is_not_accepted_as_null(number):
    registry = ExtensionRegistry()
    registry.register(_plugin("pipeline.model", handler=lambda **_: _Reading(value=number)))
    output, receipts = registry.invoke_pipeline("harness.test-pipeline", "transform", {"safe": 1})
    assert output == {"safe": 1}
    assert receipts[0].outcome == "failed"


# 功能：
#   验证输入无法形成有效有限数据回执时，在处理器产生副作用之前拒绝。
# 输入：
#   value：普通容器或模型包装的非法输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [{"value": float("nan")}, _Reading(value=float("inf"))])
def test_invalid_input_is_rejected_before_handler_runs(value):
    calls = []
    registry = ExtensionRegistry()
    registry.register(_plugin("pipeline.input", handler=lambda **_: calls.append("called")))
    with pytest.raises(ValueError):
        registry.invoke_pipeline("harness.test-pipeline", "transform", value)
    assert calls == []


# 功能：
#   验证非字符串键不会经字符串化丢掉同名字段而仍产生接受回执。
# 输入：
#   value：普通字典或类型化模型中的冲突键字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "value", [{1: "lost", "1": "kept"}, _KeyedReading(values={1: "lost", "1": "kept"})]
)
def test_output_key_collision_is_rejected_without_silent_field_loss(value):
    registry = ExtensionRegistry()
    registry.register(_plugin("pipeline.keys", handler=lambda **_: value))
    output, receipts = registry.invoke_pipeline("harness.test-pipeline", "transform", {})
    assert output == {}
    assert receipts[0].outcome == "failed"


# 功能：
#   验证集合回执采用确定的内容顺序，列表顺序则保持业务原意。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_set_representation_is_canonical_without_reordering_lists():
    values = {"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"}
    assert _jsonable(values) == sorted(values)
    assert _jsonable(frozenset(values)) == sorted(values)
    assert _jsonable(["bravo", "alpha"]) == ["bravo", "alpha"]


# 功能：
#   验证已合法模型的日期序列化与既有回执摘要格式保持一致。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_valid_model_preserves_existing_json_receipt_bytes():
    reading = _Reading(value=1.5)
    assert _jsonable(reading) == reading.model_dump(mode="json")
    assert sha256_json(_jsonable(reading)) == sha256_json(reading)


# 功能：
#   验证借用的宿主服务不被遍历、复制或调用，回执只记录其类型标记。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_borrowed_service_is_preserved_without_exposing_its_fields():
    service = object()
    received = []
    registry = ExtensionRegistry()
    registry.register(
        _plugin(
            "pipeline.service",
            handler=lambda *, value, service: received.append(service) or value,
        )
    )
    output, receipts = registry.invoke_pipeline(
        "harness.test-pipeline", "transform", {"safe": 1}, service=service
    )
    assert output == {"safe": 1}
    assert received[0] is service
    assert receipts[0].input_sha256 == sha256_json(
        {"value": {"safe": 1}, "service": {"python_type": "object"}}
    )


# 功能：
#   验证自引用输入被有界拒绝，不无限递归且不进入业务处理器。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cyclic_input_has_a_bounded_error_before_dispatch():
    value = []
    value.append(value)
    calls = []
    registry = ExtensionRegistry()
    registry.register(_plugin("pipeline.cycle", handler=lambda **_: calls.append("called")))
    with pytest.raises(ValueError, match="TOO_COMPLEX"):
        registry.invoke_pipeline("harness.test-pipeline", "transform", value)
    assert not calls


# 功能：
#   验证输入表示有明确的深度及访问量上限，过大数据在业务处理之前拒绝。
# 输入：
#   shape：超深嵌套或超宽列表的测试形状。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("shape", ["depth", "breadth"])
def test_receipt_value_budget_precedes_handler_execution(shape):
    value = [0] * (MAX_JSON_NODES + 1) if shape == "breadth" else 0
    if shape == "depth":
        for _ in range(MAX_JSON_DEPTH + 1):
            value = [value]
    calls = []
    registry = ExtensionRegistry()
    registry.register(_plugin("pipeline.budget", handler=lambda **_: calls.append("called")))
    with pytest.raises(ValueError, match="TOO_COMPLEX"):
        registry.invoke_pipeline("harness.test-pipeline", "transform", value)
    assert not calls
