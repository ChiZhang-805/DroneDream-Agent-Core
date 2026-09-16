"""Schema admission caching, not a substitute for validating tool calls."""

from __future__ import annotations

import jsonschema
import pytest

from dronedream_plugin_sdk import protocol


# 功能：
#   隔离每个用例的缓存状态，避免测试替换的校验器或成功记录影响其他用例。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
@pytest.fixture(autouse=True)
def isolated_schema_cache():
    protocol._check_cached_schema.cache_clear()
    yield
    protocol._check_cached_schema.cache_clear()


# 功能：
#   验证相同内容只执行一次完整检查，而输入或返回副本的变化都会重新校验。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cache_reuses_content_without_sharing_mutable_schema():
    source = {"properties": {"value": {"type": "string"}}}
    first = protocol.validate_local_schema(source)
    second = protocol.validate_local_schema(source)
    assert protocol._check_cached_schema.cache_info().hits == 1
    first["properties"]["value"]["type"] = "number"
    assert second == source
    assert second["properties"] is not source["properties"]
    assert protocol.validate_local_schema(source) == second
    source["properties"]["value"] = {"$ref": "https://example.test/schema"}
    with pytest.raises(ValueError, match="EXTERNAL_REFERENCE"):
        protocol.validate_local_schema(source)
    first["properties"]["value"]["type"] = "invalid-type"
    with pytest.raises(jsonschema.SchemaError):
        protocol.validate_local_schema(first)


# 功能：
#   验证语法错误和嵌套外部引用反复调用仍被拒绝，失败不会占据成功缓存。
# 输入：
#   schema：非法声明或带外部引用的 Schema。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("schema", [
    {"type": "invalid"},
    {"allOf": [{"$ref": "https://example.test/schema"}]},
    {"properties": {"child": {"$dynamicRef": "https://example.test/schema"}}},
])
def test_invalid_schemas_never_enter_cache(schema):
    for _ in range(2):
        with pytest.raises((ValueError, jsonschema.SchemaError)):
            protocol.validate_local_schema(schema)
    info = protocol._check_cached_schema.cache_info()
    assert info.currsize == 0
    assert info.hits == 0


# 功能：
#   验证缓存同时约束条目数和 UTF-8 字节大小，较大合法声明走未缓存检查。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cache_has_entry_and_utf8_byte_bounds():
    for index in range(protocol.MAX_CACHED_SCHEMAS + 2):
        protocol.validate_local_schema({"title": str(index), "type": "string"})
    assert protocol._check_cached_schema.cache_info().currsize == protocol.MAX_CACHED_SCHEMAS
    before = protocol._check_cached_schema.cache_info()
    large = {"description": "中" * (protocol.MAX_CACHED_SCHEMA_BYTES // 2)}
    assert protocol.validate_local_schema(large) == large
    assert protocol._check_cached_schema.cache_info() == before
    with pytest.raises(ValueError, match="SIZE_LIMIT"):
        protocol.validate_local_schema({"description": "x" * protocol.MAX_MESSAGE_BYTES})
    assert protocol._check_cached_schema.cache_info() == before


# 功能：
#   验证 Schema 草案或注册校验器变化时不复用旧校验器的成功结果。
# 输入：
#   monkeypatch：替换校验器选择函数的独立测试工具。
# 输出：
#   None：不返回业务数据。
def test_cache_key_includes_draft_and_selected_validator(monkeypatch):
    draft4 = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "minimum": 0,
        "exclusiveMinimum": True,
    }
    assert protocol.validate_local_schema(draft4) == draft4
    draft2020 = {**draft4, "$schema": "https://json-schema.org/draft/2020-12/schema"}
    with pytest.raises(jsonschema.SchemaError):
        protocol.validate_local_schema(draft2020)
    schema = {"type": "string"}
    protocol.validate_local_schema(schema)
    # 自定义草案可被宿主注册；同一文本不能绕过新校验器的入场检查。
    monkeypatch.setattr(jsonschema.validators, "validator_for", lambda _: RejectingValidator)
    with pytest.raises(ValueError, match="TEST_VALIDATOR_REJECTED"):
        protocol.validate_local_schema(schema)


class RejectingValidator:
    # 功能：
    #   拒绝测试声明，用于证明缓存没有跳过替换后的校验器。
    # 输入：
    #   schema：候选 Schema。
    # 输出：
    #   None：不返回业务数据。
    @staticmethod
    def check_schema(schema):
        raise ValueError("TEST_VALIDATOR_REJECTED")
