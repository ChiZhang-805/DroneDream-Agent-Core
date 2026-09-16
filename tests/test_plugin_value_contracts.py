"""插件值转换、依赖版本与共享 XML 解析的纯本地契约测试。"""

from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from xml.etree.ElementTree import ParseError

import pytest
from pydantic import BaseModel

from dronedream_agent_core.plugin_values import plugin_json_value
from dronedream_agent_core.plugin_versions import version_matches, version_precedence
from dronedream_agent_core.xml_values import parse_xml
from dronedream_plugin_sdk.protocol import MAX_JSON_BYTES, decode_json, encode_json


# 功能：
#   验证依赖约束拒绝非字符串和超长输入，错误表现不依赖对象是否具有 startswith 方法。
# 输入：
#   requirement：非法依赖约束。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("requirement", [None, True, 1, [], {}, "x" * 163])
def test_version_requirement_has_explicit_input_boundary(requirement):
    with pytest.raises(ValueError, match="PLUGIN_VERSION_REQUIREMENT_INVALID"):
        version_matches("1.0.0", requirement)


# 功能：
#   验证预发布按数值与文本标识排序，且正式版本的构建元数据不改变优先级。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_version_order_preserves_prerelease_semantics():
    versions = [
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-alpha.2",
        "1.0.0-alpha.10",
        "1.0.0-alpha.beta",
        "1.0.0-beta",
        "1.0.0-rc.1",
        "1.0.0",
    ]
    ordered = [version_precedence(version) for version in versions]
    assert ordered == sorted(ordered)
    assert version_precedence("1.0.0+build.a") == version_precedence("1.0.0+build.b")
    assert version_matches("0.0.1", "^0.0.1")
    assert not version_matches("0.0.2", "^0.0.1")
    assert not version_matches("1.1.0-alpha", ">=1.0.0-alpha")


# 功能：
#   验证支持的非 JSON 类型按明确规则转换，并确保嵌套列表与原值不再共享可变引用。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_plugin_supported_values_have_explicit_conversion():
    class Mode(Enum):
        HOLD = "hold"

    class Measurement(BaseModel):
        value: float

    original = {
        "path": Path("asset.ddpkg"),
        "mode": Mode.HOLD,
        "date": date(2026, 1, 1),
        "time": datetime(2026, 1, 1, tzinfo=UTC),
        "measurement": Measurement(value=1.5),
        "tuple": (1, 2),
        "set": frozenset({"b", "a"}),
        "nested": [[1]],
    }
    detached = plugin_json_value(original)
    assert detached == {
        "path": "asset.ddpkg",
        "mode": "hold",
        "date": "2026-01-01",
        "time": "2026-01-01T00:00:00+00:00",
        "measurement": {"value": 1.5},
        "tuple": [1, 2],
        "set": ["a", "b"],
        "nested": [[1]],
    }
    detached["nested"][0].append(2)
    assert original["nested"] == [[1]]


# 功能：
#   验证未知对象被拒绝时不会调用它的 repr，避免错误兜底泄露对象内部信息。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_unknown_plugin_value_never_uses_repr():
    class UntrustedObject:
        # 功能：
        #   捕捉不允许发生的表示转换，触发时使测试直接失败。
        # 输入：
        #   无。
        # 输出：
        #   None：不返回业务数据。
        def __repr__(self):
            pytest.fail("unknown plugin object must not be rendered")

    with pytest.raises(ValueError, match="PLUGIN_EXTENSION_INPUT_NOT_JSON:UntrustedObject"):
        plugin_json_value(UntrustedObject())


# 功能：
#   验证 XML 的节点及深度预算在边界上仍允许合法文档，不同解析之间不继承计数。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_xml_limits_are_exact_and_reset_per_document():
    content = b"<root><child/></root>"
    assert parse_xml(content, maximum_bytes=len(content), maximum_elements=2).tag == "root"
    with pytest.raises(ValueError, match="XML_STRUCTURE_LIMIT"):
        parse_xml(content, maximum_bytes=len(content), maximum_elements=1)
    assert parse_xml(b"<ok/>", maximum_bytes=5, maximum_elements=1).tag == "ok"
    nested = b"<n>" * 128 + b"</n>" * 128
    assert parse_xml(nested, maximum_bytes=len(nested)).tag == "n"
    with pytest.raises(ValueError, match="XML_STRUCTURE_LIMIT"):
        parse_xml(b"<n>" + nested + b"</n>", maximum_bytes=len(nested) + 7)


# 功能：
#   验证 DTD 拒绝逻辑基于 XML 解析事件，不因 UTF-16 编码而漏过外部或内部实体声明。
# 输入：
#   encoding：测试文档的字符编码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_xml_rejects_doctype_across_encodings(encoding):
    content = '<!DOCTYPE root [<!ENTITY x "expanded">]><root>&x;</root>'.encode(encoding)
    with pytest.raises(ValueError, match="XML_DOCTYPE_FORBIDDEN"):
        parse_xml(content, maximum_bytes=1024)
    with pytest.raises(ParseError):
        parse_xml(b"<root><x></root>", maximum_bytes=1024)


# 功能：
#   验证 XML 两种预算都拒绝布尔值、非正数、非整数与超大上限。
# 输入：
#   budget：非法预算值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("budget", [True, 0, -1, 1.5, float("inf"), None, 64 * 1024 * 1024 + 1])
def test_xml_budget_types_are_not_coerced(budget):
    with pytest.raises(ValueError, match="XML_BUDGET_INVALID"):
        parse_xml(b"<a/>", maximum_bytes=budget)
    with pytest.raises(ValueError, match="XML_BUDGET_INVALID"):
        parse_xml(b"<a/>", maximum_bytes=4, maximum_elements=budget)


# 功能：
#   验证 JSON 解析拒绝未声明的输入类型，同时允许宿主在硬上限内使用显式较大预算。
# 输入：
#   value：既非字符串也非字节串的输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [None, [], {}, 1, True, bytearray(b"0")])
def test_json_input_type_and_larger_host_budget(value):
    with pytest.raises(ValueError, match="PLUGIN_JSON_INPUT_TYPE_INVALID"):
        decode_json(value)
    assert encode_json([0], limit=MAX_JSON_BYTES) == "[0]"
    assert decode_json("[0]", limit=MAX_JSON_BYTES) == [0]
