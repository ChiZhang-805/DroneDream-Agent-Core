"""Exercise actual normalization/guard hooks without cloud models or flight I/O."""

from types import SimpleNamespace

import pytest

from dronedream_agent_plugins.input_channel_plugins import _ingest
from dronedream_agent_plugins.language_entity_plugins import _map_entity_resolver
from dronedream_agent_plugins.structured_output_plugins import _finite_value_guard
from dronedream_agent_plugins.tool_middleware import _finite_number_guard, _secret_guard


# 功能：
#   验证定时输入拒绝空白或非文本的调度标识，不能将其当作可追溯的定时任务接收。
# 输入：
#   identifier：故意构造的非法调度标识。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("identifier", ["", "   ", None, 123])
def test_scheduled_channel_requires_nonempty_identifier(identifier):
    request = SimpleNamespace(
        input_channel="scheduled",
        input_metadata={"schedule_id": identifier},
        message="inspect the corridor",
        attachments=[],
    )
    assert _ingest("scheduled", request=request)["accepted"] is False


# 功能：
#   验证地图实体别名可嵌在没有空格的中文句子中，不要求中文也按英文词边界分隔。
# 输入：
#   name：地图目录中的实体别名。
#   message：包含别名的自然语言指令。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "name,message",
    [
        ("一号楼", "从办公室飞到一号楼门口"),
        ("north gate", "飞到north gate再返回"),
    ],
)
def test_catalog_aliases_match_inside_unspaced_chinese_sentences(name, message):
    catalog = SimpleNamespace(
        entities=[
            SimpleNamespace(entity_id="test.destination", aliases=[name]),
        ]
    )
    result = _map_entity_resolver(
        value={},
        request=SimpleNamespace(message=message),
        map_catalog=catalog,
    )
    assert [item["entity_id"] for item in result["resolved_entities"]] == ["test.destination"]


# 功能：
#   验证英文别名不会误命中更长单词中的子串，避免把普通动作词识别为目的地。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_english_alias_does_not_match_inside_longer_identifier():
    catalog = SimpleNamespace(entities=[SimpleNamespace(entity_id="test.gate", aliases=["gate"])])
    result = _map_entity_resolver(
        value={},
        request=SimpleNamespace(message="investigate the corridor"),
        map_catalog=catalog,
    )
    assert result["resolved_entities"] == []


# 功能：
#   验证密钥检查会遍历元组中的字典，不允许用容器类型规避敏感字段过滤。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_tuple_cannot_hide_secret_key_from_tool_guard():
    with pytest.raises(ValueError, match="CONTAINS_SECRET"):
        _secret_guard(value={"items": ({"api_key": "synthetic-test-value"},)})


# 功能：
#   验证工具和结构化输出的数值检查都能发现嵌套元组中的非有限数。
# 输入：
#   guard：本例调用的真实数值校验函数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("guard", [_finite_number_guard, _finite_value_guard])
def test_tuple_cannot_hide_nonfinite_value_from_guard(guard):
    with pytest.raises(ValueError, match="NON_FINITE"):
        guard(value={"artifact": {"items": (float("nan"),)}})
